from __future__ import annotations

from enum import StrEnum
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest
from pydantic import ValidationError

import saliencegate.runtime.model_token_counting as counting
from saliencegate.domain import canonical_json
from saliencegate.memory import MEMORY_EDIT_OUTPUT_SCHEMA_VERSION
from saliencegate.ports.model_calls import (
    STRUCTURED_CALL_REQUEST_SCHEMA_VERSION,
    STRUCTURED_CALL_USAGE_SCHEMA_VERSION,
    CanonicalUsageProvenance,
    ProviderUsageProvenance,
    StructuredCallPhase,
    StructuredCallRequest,
    StructuredCallUsage,
)
from saliencegate.prompts import PAPER_TWO_PHASE_V1
from saliencegate.prompts.contracts import (
    StructuredPromptPayload,
    SystemPromptMessage,
    UntrustedPromptDataMessage,
)
from saliencegate.runtime.model_token_counting import (
    DeterministicModelTokenCounter,
    HarmonyTokenCounter,
    ModelTokenAccountingUnavailableError,
    ModelTokenCount,
    ModelTokenCounter,
    ModelTokenCounterIdentity,
    ModelTokenCounterInputError,
    ModelTokenCounterPairingError,
    ModelTokenCounterUnavailableError,
    ModelTokenDirection,
    validated_live_model_token_usage,
)
from saliencegate.runtime.token_counting import DeterministicTokenCounter

RUN_ID = UUID("00000000-0000-4000-8000-00000000e001")
MAX_SIGNED_64 = (1 << 63) - 1


def _request(
    *,
    model_id: str = "gpt-oss:20b",
    system_content: str = "Follow the schema.",
    user_content: str = "Return an empty operation list.",
) -> StructuredCallRequest:
    payload = StructuredPromptPayload(
        messages=(
            SystemPromptMessage(role="system", content=system_content),
            UntrustedPromptDataMessage(role="user", content=user_content),
        ),
        response_format=PAPER_TWO_PHASE_V1.memory_edit_template.response_format,
    )
    return StructuredCallRequest(
        schema_version=STRUCTURED_CALL_REQUEST_SCHEMA_VERSION,
        run_id=RUN_ID,
        cycle_id="a" * 64,
        model_call_index=0,
        phase=StructuredCallPhase.MEMORY_EDIT,
        attempt=0,
        model_id=model_id,
        prompt_template_id="paper-two-phase/memory-edit-v1",
        prompt_template_digest="b" * 64,
        response_schema_version=MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
        payload=payload.as_json_object(),
    )


def test_identity_and_count_are_strict_frozen_and_content_free() -> None:
    counter = DeterministicModelTokenCounter(
        model_id="fixture-model/v1",
        input_token_count=17,
        output_token_count=5,
    )
    count = counter.count_input(_request(model_id="fixture-model/v1"))

    assert count == ModelTokenCount(
        direction=ModelTokenDirection.INPUT,
        token_count=17,
        provenance=CanonicalUsageProvenance.LOCAL_COUNTER,
        counter_identity=counter.identity,
    )
    assert count.counter_id == counter.identity.counter_id
    assert count.counter_version == counter.identity.counter_version
    assert count.configuration_digest == counter.identity.configuration_digest
    assert count.model_id == "fixture-model/v1"
    assert "Follow the schema" not in repr(count)
    assert "Follow the schema" not in count.model_dump_json()

    with pytest.raises(ValidationError, match="frozen"):
        count.__setattr__("token_count", 18)
    with pytest.raises(ValidationError):
        ModelTokenCounterIdentity.model_validate(
            {
                **counter.identity.model_dump(mode="python"),
                "unexpected": True,
            }
        )
    with pytest.raises(ValidationError):
        ModelTokenCount.model_validate(
            {
                **count.model_dump(mode="python"),
                "token_count": True,
            }
        )
    with pytest.raises(ValidationError):
        ModelTokenCount(
            direction=ModelTokenDirection.INPUT,
            token_count=17,
            provenance=CanonicalUsageProvenance.UNAVAILABLE,
            counter_identity=counter.identity,
        )


def test_deterministic_counter_is_stable_directional_and_protocol_compatible() -> None:
    counter = DeterministicModelTokenCounter(
        model_id="fixture-model/v1",
        input_token_count=23,
        output_token_count=7,
    )
    request = _request(model_id="fixture-model/v1")

    assert isinstance(counter, ModelTokenCounter)
    assert counter.count_input(request) == counter.count_input(request)
    assert counter.count_input(request).direction is ModelTokenDirection.INPUT
    assert (
        counter.count_output(model_id="fixture-model/v1", completion="{}").direction
        is ModelTokenDirection.OUTPUT
    )
    assert counter.count_output(model_id="fixture-model/v1", completion="{}").token_count == 7
    assert (
        counter.identity.configuration_digest
        == DeterministicModelTokenCounter(
            model_id="fixture-model/v1",
            input_token_count=23,
            output_token_count=7,
        ).identity.configuration_digest
    )
    assert (
        counter.identity.configuration_digest
        != DeterministicModelTokenCounter(
            model_id="fixture-model/v1",
            input_token_count=24,
            output_token_count=7,
        ).identity.configuration_digest
    )


def test_counter_identity_requires_exact_model_pairing() -> None:
    counter = DeterministicModelTokenCounter(
        model_id="gpt-oss:20b",
        input_token_count=1,
        output_token_count=1,
    )

    with pytest.raises(ModelTokenCounterPairingError):
        counter.count_input(_request(model_id="gpt-oss:120b"))
    with pytest.raises(ModelTokenCounterPairingError):
        counter.count_output(model_id="gpt-oss:120b", completion="secret")


def test_provider_and_local_counts_remain_distinct_when_they_disagree() -> None:
    identity = DeterministicModelTokenCounter(
        model_id="fixture-model/v1",
        input_token_count=23,
        output_token_count=7,
    ).identity
    usage = StructuredCallUsage(
        schema_version=STRUCTURED_CALL_USAGE_SCHEMA_VERSION,
        provider_input_tokens=19,
        provider_output_tokens=5,
        provider_usage_provenance=ProviderUsageProvenance.PROVIDER_REPORTED,
        latency_us=10,
        canonical_input_tokens=23,
        canonical_output_tokens=7,
        canonical_usage_provenance=CanonicalUsageProvenance.LOCAL_COUNTER,
        local_counter_id=identity.counter_id,
        local_counter_version=identity.counter_version,
        local_counter_configuration_digest=identity.configuration_digest,
        local_counter_model_id=identity.model_id,
    )

    assert usage.provider_tokens == 24
    assert usage.canonical_tokens == 30
    assert usage.provider_tokens != usage.canonical_tokens


def test_live_usage_accepts_provider_or_matching_local_evidence() -> None:
    identity = DeterministicModelTokenCounter(
        model_id="fixture-model/v1",
        input_token_count=23,
        output_token_count=7,
    ).identity
    provider_only = StructuredCallUsage(
        schema_version=STRUCTURED_CALL_USAGE_SCHEMA_VERSION,
        provider_input_tokens=20,
        provider_output_tokens=6,
        provider_usage_provenance=ProviderUsageProvenance.PROVIDER_REPORTED,
        latency_us=1,
    )
    local_only = StructuredCallUsage(
        schema_version=STRUCTURED_CALL_USAGE_SCHEMA_VERSION,
        provider_input_tokens=None,
        provider_output_tokens=None,
        provider_usage_provenance=ProviderUsageProvenance.UNAVAILABLE,
        latency_us=1,
        canonical_input_tokens=23,
        canonical_output_tokens=7,
        canonical_usage_provenance=CanonicalUsageProvenance.LOCAL_COUNTER,
        local_counter_id=identity.counter_id,
        local_counter_version=identity.counter_version,
        local_counter_configuration_digest=identity.configuration_digest,
        local_counter_model_id=identity.model_id,
    )

    assert (
        validated_live_model_token_usage(
            provider_only,
            configured_counter=None,
        ).provider_tokens
        == 26
    )
    assert (
        validated_live_model_token_usage(
            local_only,
            configured_counter=identity,
        ).canonical_tokens
        == 30
    )


def test_live_usage_refuses_missing_partial_or_mismatched_evidence() -> None:
    identity = DeterministicModelTokenCounter(
        model_id="fixture-model/v1",
        input_token_count=23,
        output_token_count=7,
    ).identity
    unavailable = StructuredCallUsage(
        schema_version=STRUCTURED_CALL_USAGE_SCHEMA_VERSION,
        provider_input_tokens=None,
        provider_output_tokens=None,
        provider_usage_provenance=ProviderUsageProvenance.UNAVAILABLE,
        latency_us=1,
    )
    partial = StructuredCallUsage(
        schema_version=STRUCTURED_CALL_USAGE_SCHEMA_VERSION,
        provider_input_tokens=None,
        provider_output_tokens=None,
        provider_usage_provenance=ProviderUsageProvenance.UNAVAILABLE,
        latency_us=1,
        canonical_input_tokens=23,
        canonical_output_tokens=None,
        canonical_usage_provenance=CanonicalUsageProvenance.LOCAL_COUNTER,
        local_counter_id=identity.counter_id,
        local_counter_version=identity.counter_version,
        local_counter_configuration_digest=identity.configuration_digest,
        local_counter_model_id=identity.model_id,
    )
    wrong_counter = DeterministicModelTokenCounter(
        model_id="fixture-model/v1",
        input_token_count=24,
        output_token_count=7,
    ).identity
    replay_claim = partial.model_copy(
        update={"canonical_usage_provenance": CanonicalUsageProvenance.REPLAY_ATTESTED}
    )

    for usage, configured in (
        (unavailable, None),
        (partial, identity),
        (partial, wrong_counter),
        (replay_claim, identity),
        (cast(StructuredCallUsage, object()), None),
    ):
        with pytest.raises(ModelTokenAccountingUnavailableError) as error:
            validated_live_model_token_usage(usage, configured_counter=configured)
        assert "fixture-model" not in str(error.value)


@pytest.mark.parametrize("value", (-1, MAX_SIGNED_64 + 1, True))
def test_deterministic_counter_rejects_counts_outside_signed_64(value: object) -> None:
    with pytest.raises(ModelTokenCounterInputError):
        DeterministicModelTokenCounter(
            model_id="fixture-model/v1",
            input_token_count=cast(int, value),
            output_token_count=1,
        )


def test_output_validation_is_value_free() -> None:
    counter = DeterministicModelTokenCounter(
        model_id="fixture-model/v1",
        input_token_count=1,
        output_token_count=1,
    )
    secret = "fixture-secret\ud800"

    with pytest.raises(ModelTokenCounterInputError) as error:
        counter.count_output(model_id="fixture-model/v1", completion=secret)

    assert "fixture-secret" not in str(error.value)
    with pytest.raises(ModelTokenCounterInputError):
        counter.count_output(
            model_id="fixture-model/v1",
            completion=cast(str, b"not-text"),
        )


def test_deterministic_counter_sanitizes_invalid_identity_and_request() -> None:
    with pytest.raises(ModelTokenCounterInputError):
        DeterministicModelTokenCounter(
            model_id="",
            input_token_count=1,
            output_token_count=1,
        )

    counter = DeterministicModelTokenCounter(
        model_id="fixture-model/v1",
        input_token_count=1,
        output_token_count=1,
    )
    with pytest.raises(ModelTokenCounterInputError):
        counter.count_input(cast(StructuredCallRequest, object()))


class _Role(StrEnum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class _Message:
    def __init__(self, role: _Role, content: str) -> None:
        self.role = role
        self.content = content
        self.channel: str | None = None

    @classmethod
    def from_role_and_content(cls, role: _Role, content: str) -> _Message:
        return cls(role, content)

    def with_channel(self, channel: str) -> _Message:
        self.channel = channel
        return self


class _Conversation:
    def __init__(self, messages: list[_Message]) -> None:
        self.messages = messages

    @classmethod
    def from_messages(cls, messages: list[_Message]) -> _Conversation:
        return cls(messages)


class _Encoding:
    def __init__(self) -> None:
        self.input_calls: list[tuple[_Conversation, _Role]] = []
        self.output_calls: list[str] = []

    def render_conversation_for_completion(
        self,
        conversation: _Conversation,
        next_turn_role: _Role,
    ) -> list[int]:
        self.input_calls.append((conversation, next_turn_role))
        # The fixture makes framing visible: three tokens per message plus content.
        return list(range(sum(3 + len(item.content) for item in conversation.messages) + 2))

    def encode(self, completion: str) -> list[int]:
        self.output_calls.append(completion)
        return list(range(2 + len(completion)))


def _harmony_module(encoding: _Encoding) -> SimpleNamespace:
    class _EncodingName(StrEnum):
        HARMONY_GPT_OSS = "HarmonyGptOss"

    loaded_names: list[_EncodingName] = []

    def load_harmony_encoding(name: _EncodingName) -> _Encoding:
        loaded_names.append(name)
        return encoding

    return SimpleNamespace(
        Conversation=_Conversation,
        HarmonyEncodingName=_EncodingName,
        Message=_Message,
        Role=_Role,
        load_harmony_encoding=load_harmony_encoding,
        loaded_names=loaded_names,
    )


def test_harmony_counter_uses_official_request_and_message_framing_apis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoding = _Encoding()
    module = _harmony_module(encoding)
    monkeypatch.setattr(counting, "_import_harmony", lambda: (module, "0.0.fixture"))
    counter = HarmonyTokenCounter(model_id="gpt-oss:20b")
    request = _request()

    input_count = counter.count_input(request)
    output_count = counter.count_output(model_id="gpt-oss:20b", completion="{}")

    assert module.loaded_names == [module.HarmonyEncodingName.HARMONY_GPT_OSS]
    assert input_count.token_count == (3 + 18) + (3 + 31) + 2
    assert output_count.token_count == 4
    assert input_count.provenance is CanonicalUsageProvenance.LOCAL_COUNTER
    assert output_count.counter_identity == input_count.counter_identity
    conversation, next_role = encoding.input_calls[0]
    assert [(item.role, item.content) for item in conversation.messages] == [
        (_Role.SYSTEM, "Follow the schema."),
        (_Role.USER, "Return an empty operation list."),
    ]
    assert next_role is _Role.ASSISTANT
    assert encoding.output_calls == ["{}"]
    assert counter.identity.counter_id == "openai-harmony"
    assert counter.identity.counter_version == "0.0.fixture"


def test_harmony_counter_is_lazy_and_sanitizes_optional_dependency_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> tuple[object, str]:
        raise ModuleNotFoundError("fixture-secret-package-path")

    monkeypatch.setattr(counting, "_import_harmony", unavailable)

    with pytest.raises(ModelTokenCounterUnavailableError) as error:
        HarmonyTokenCounter(model_id="gpt-oss:20b")

    assert "fixture-secret" not in str(error.value)


def test_harmony_loader_uses_the_optional_package_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _harmony_module(_Encoding())
    imported: list[str] = []
    versioned: list[str] = []
    monkeypatch.setattr(
        counting,
        "import_module",
        lambda name: imported.append(name) or module,
    )
    monkeypatch.setattr(
        counting.metadata,
        "version",
        lambda name: versioned.append(name) or "0.0.fixture",
    )

    loaded, package_version = counting._import_harmony()

    assert loaded is module
    assert package_version == "0.0.fixture"
    assert imported == ["openai_harmony"]
    assert versioned == ["openai-harmony"]


def test_harmony_counter_rejects_invalid_package_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _harmony_module(_Encoding())
    monkeypatch.setattr(counting, "_import_harmony", lambda: (module, ""))

    with pytest.raises(ModelTokenCounterUnavailableError):
        HarmonyTokenCounter(model_id="gpt-oss:20b")


def test_harmony_counter_rejects_non_gpt_oss_and_wrong_request_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _harmony_module(_Encoding())
    monkeypatch.setattr(counting, "_import_harmony", lambda: (module, "0.0.fixture"))

    with pytest.raises(ModelTokenCounterPairingError):
        HarmonyTokenCounter(model_id="other-model/v1")

    counter = HarmonyTokenCounter(model_id="gpt-oss:20b")
    with pytest.raises(ModelTokenCounterPairingError):
        counter.count_input(_request(model_id="gpt-oss:120b"))


def test_harmony_failures_and_malformed_prompt_payload_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoding = _Encoding()
    module = _harmony_module(encoding)
    monkeypatch.setattr(counting, "_import_harmony", lambda: (module, "0.0.fixture"))
    counter = HarmonyTokenCounter(model_id="gpt-oss:20b")
    invalid = _request().model_copy(update={"payload": {"secret": "fixture-secret"}})

    with pytest.raises(ModelTokenCounterInputError) as error:
        counter.count_input(invalid)

    assert "fixture-secret" not in str(error.value)

    def fail_encode(_completion: str) -> list[int]:
        raise RuntimeError("fixture-secret-rendering-state")

    encoding.encode = fail_encode  # type: ignore[method-assign]
    with pytest.raises(ModelTokenCounterInputError) as render_error:
        counter.count_output(model_id="gpt-oss:20b", completion="fixture-secret")

    assert "fixture-secret" not in str(render_error.value)


def test_harmony_rejects_a_non_list_input_token_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoding = _Encoding()
    module = _harmony_module(encoding)
    monkeypatch.setattr(counting, "_import_harmony", lambda: (module, "0.0.fixture"))
    counter = HarmonyTokenCounter(model_id="gpt-oss:20b")
    encoding.render_conversation_for_completion = (  # type: ignore[method-assign]
        lambda _conversation, _role: (1, 2, 3)
    )

    with pytest.raises(ModelTokenCounterInputError):
        counter.count_input(_request())


def test_harmony_count_rejects_signed_64_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    encoding = _Encoding()
    module = _harmony_module(encoding)
    monkeypatch.setattr(counting, "_import_harmony", lambda: (module, "0.0.fixture"))
    counter = HarmonyTokenCounter(model_id="gpt-oss:20b")

    class _HugeTokens(list[int]):
        def __len__(self) -> int:
            return MAX_SIGNED_64 + 1

    encoding.encode = lambda _completion: _HugeTokens()  # type: ignore[method-assign]

    with pytest.raises(ModelTokenCounterInputError):
        counter.count_output(model_id="gpt-oss:20b", completion="{}")


def test_stage_one_heuristic_is_neither_reused_nor_reinterpreted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = DeterministicTokenCounter().measure("abcdefgh")

    def stage_one_must_not_run(_self: object, _text: str) -> object:
        raise AssertionError("byte-based batching heuristic was reused")

    monkeypatch.setattr(DeterministicTokenCounter, "measure", stage_one_must_not_run)
    counter = DeterministicModelTokenCounter(
        model_id="fixture-model/v1",
        input_token_count=19,
        output_token_count=11,
    )

    assert counter.count_input(_request(model_id="fixture-model/v1")).token_count == 19
    assert (
        counter.count_output(model_id="fixture-model/v1", completion="abcdefgh").token_count == 11
    )
    assert original.approximate_tokens == 2
    assert counter.identity.counter_id != original.algorithm_version
    assert canonical_json(counter.identity) == canonical_json(counter.identity)
