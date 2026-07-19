from __future__ import annotations

import asyncio
import gzip
import json
import subprocess
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from pydantic import ValidationError

from saliencegate.domain import (
    ClaimKind,
    EvidenceReference,
    EvidenceSource,
    InterventionAction,
    PayloadDigestAlgorithm,
    canonical_json,
)
from saliencegate.intervention import ProposedClaim
from saliencegate.memory.proposals import (
    INTERVENTION_OUTPUT_SCHEMA_VERSION,
    MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
    BankOperationsProposal,
    InterventionSelectionOutput,
)
from saliencegate.models.openai_compatible import (
    OpenAICompatibleClient,
    OpenAICompatibleConfig,
    OpenAICompatibleError,
    OpenAICompatibleErrorCode,
)
from saliencegate.ports.model_calls import (
    CanonicalUsageProvenance,
    ProviderUsageProvenance,
    StructuredCallParseStatus,
    StructuredCallPhase,
    StructuredCallRequest,
    StructuredCallStatus,
    validated_result_for_request,
)
from saliencegate.prompts.contracts import (
    StructuredPromptPayload,
    SystemPromptMessage,
    UntrustedPromptDataMessage,
)
from saliencegate.prompts.paper_two_phase_v1 import (
    PAPER_TWO_PHASE_FORCED_REMINDER_V1,
    PAPER_TWO_PHASE_V1,
)
from saliencegate.runtime.model_token_counting import (
    DeterministicModelTokenCounter,
    ModelTokenCounterIdentity,
)
from saliencegate.security.keys import InstallationKey

RUN_ID = UUID("00000000-0000-4000-8000-00000000e101")
MEMORY_ID = UUID("00000000-0000-4000-8000-00000000e102")
KEY = InstallationKey(b"k" * 32)


def _payload(
    phase: StructuredCallPhase,
    *,
    forced_reminder: bool = False,
) -> StructuredPromptPayload:
    template = (
        PAPER_TWO_PHASE_V1.memory_edit_template
        if phase is StructuredCallPhase.MEMORY_EDIT
        else (
            PAPER_TWO_PHASE_FORCED_REMINDER_V1.intervention_template
            if forced_reminder
            else PAPER_TWO_PHASE_V1.intervention_template
        )
    )
    return StructuredPromptPayload(
        messages=(
            SystemPromptMessage(role="system", content="Return only strict JSON."),
            UntrustedPromptDataMessage(role="user", content='{"authority":"none"}'),
        ),
        response_format=template.response_format,
    )


def _request(
    phase: StructuredCallPhase = StructuredCallPhase.MEMORY_EDIT,
    *,
    model_id: str = "gpt-oss:20b",
    payload: dict[str, object] | None = None,
    forced_reminder: bool = False,
) -> StructuredCallRequest:
    intervention = phase is StructuredCallPhase.INTERVENTION
    return StructuredCallRequest(
        schema_version="structured-call-request/v1",
        run_id=RUN_ID,
        cycle_id="a" * 64,
        model_call_index=1 if intervention else 0,
        phase=phase,
        attempt=0,
        model_id=model_id,
        prompt_template_id=(
            (
                "paper-two-phase/intervention-forced-reminder-v1"
                if forced_reminder
                else "paper-two-phase/intervention-v1"
            )
            if intervention
            else "paper-two-phase/memory-edit-v1"
        ),
        prompt_template_digest="b" * 64,
        response_schema_version=(
            INTERVENTION_OUTPUT_SCHEMA_VERSION
            if intervention
            else MEMORY_EDIT_OUTPUT_SCHEMA_VERSION
        ),
        payload=(
            _payload(phase, forced_reminder=forced_reminder).as_json_object()
            if payload is None
            else payload
        ),
    )


def _output(phase: StructuredCallPhase) -> str:
    if phase is StructuredCallPhase.INTERVENTION:
        return InterventionSelectionOutput(
            schema_version=INTERVENTION_OUTPUT_SCHEMA_VERSION,
            action=InterventionAction.SILENCE,
            claims=(),
            confidence=1.0,
        ).model_dump_json()
    return BankOperationsProposal(
        schema_version=MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
        operations=(),
    ).model_dump_json()


def _forced_reminder_output() -> str:
    return InterventionSelectionOutput(
        schema_version=INTERVENTION_OUTPUT_SCHEMA_VERSION,
        action=InterventionAction.REMIND,
        claims=(
            ProposedClaim(
                kind=ClaimKind.REQUIREMENT,
                evidence=EvidenceReference(
                    source=EvidenceSource.MEMORY,
                    source_id=MEMORY_ID,
                    revision=1,
                    field_path="/content",
                ),
            ),
        ),
        confidence=1.0,
    ).model_dump_json()


def _envelope(
    phase: StructuredCallPhase = StructuredCallPhase.MEMORY_EDIT,
    *,
    content: str | None = None,
    usage: object = None,
    reasoning: object | None = None,
    model: str = "gpt-oss:20b",
) -> dict[str, object]:
    message: dict[str, object] = {
        "role": "assistant",
        "content": _output(phase) if content is None else content,
    }
    if reasoning is not None:
        message["reasoning"] = reasoning
        message["reasoning_content"] = reasoning
    return {
        "id": "local-completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": (
            {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}
            if usage is None
            else usage
        ),
        "provider_reasoning": reasoning,
    }


def _transport(
    handler: Callable[[httpx.Request], httpx.Response | asyncio.Future[httpx.Response]],
) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _client(
    handler: Callable[..., object],
    *,
    config: OpenAICompatibleConfig | None = None,
    counter: DeterministicModelTokenCounter | None = None,
    credential_lookup: Callable[[str], str | None] | None = None,
    monotonic_ns: Callable[[], int] | None = None,
) -> OpenAICompatibleClient:
    values: dict[str, object] = {
        "installation_key": KEY,
        "model_token_counter": counter,
        "transport_factory": lambda: httpx.MockTransport(handler),
    }
    if credential_lookup is not None:
        values["credential_lookup"] = credential_lookup
    if monotonic_ns is not None:
        values["monotonic_ns"] = monotonic_ns
    return OpenAICompatibleClient(
        OpenAICompatibleConfig() if config is None else config,
        **values,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("base_url", "endpoint"),
    (
        ("http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/v1/chat/completions"),
        ("http://127.0.0.2/v1/", "http://127.0.0.2/v1/chat/completions"),
        ("http://[::1]:11434/v1", "http://[::1]:11434/v1/chat/completions"),
    ),
)
def test_config_accepts_only_canonical_numeric_loopback_by_default(
    base_url: str,
    endpoint: str,
) -> None:
    config = OpenAICompatibleConfig(base_url=base_url)

    assert config.endpoint_url == endpoint
    assert config.model == "gpt-oss:20b"
    assert config.reasoning_effort == "medium"
    assert config.credential_env is None
    assert config.allow_remote is False


@pytest.mark.parametrize(
    "base_url",
    (
        "http://localhost:11434/v1",
        "http://localhost.:11434/v1",
        "http://0.0.0.0/v1",
        "http://10.0.0.1/v1",
        "http://169.254.1.1/v1",
        "http://127.1/v1",
        "http://2130706433/v1",
        "http://0x7f000001/v1",
        "http://127.00.0.1/v1",
        "http://[::]/v1",
        "http://[::ffff:127.0.0.1]/v1",
        "http://[::1%25lo0]/v1",
        "ftp://127.0.0.1/v1",
        "http://user:secret@127.0.0.1/v1",
        "http://127.0.0.1/v1?target=remote",
        "http://127.0.0.1/v1#fragment",
        "http://127.0.0.1/%2e%2e/v1",
        "http://127.0.0.1//v1",
        "http://127.0.0.1/./v1",
        "http://127.0.0.1:/v1",
        "http://:80/v1",
        "http://[0:0:0:0:0:0:0:1]/v1",
        "http://127.0.0.1:99999/v1",
        "http://127.0.0.1\\@example.test/v1",
    ),
)
def test_config_rejects_ambiguous_or_non_loopback_endpoints(base_url: str) -> None:
    with pytest.raises(ValidationError, match="base URL failed validation"):
        OpenAICompatibleConfig(base_url=base_url)


def test_remote_destination_requires_explicit_opt_in_without_weakening_url_checks() -> None:
    config = OpenAICompatibleConfig(
        base_url="https://api.example.test/v1/",
        allow_remote=True,
    )

    assert config.endpoint_url == "https://api.example.test/v1/chat/completions"
    with pytest.raises(ValidationError):
        OpenAICompatibleConfig(
            base_url="https://user:secret@api.example.test/v1",
            allow_remote=True,
        )
    with pytest.raises(ValidationError):
        OpenAICompatibleConfig(
            base_url="https://api.example.test/v1#remote",
            allow_remote=True,
        )
    with pytest.raises(ValidationError):
        OpenAICompatibleConfig(
            base_url="http://api.example.test/v1",
            allow_remote=True,
        )
    numeric = OpenAICompatibleConfig(
        base_url="https://192.0.2.1/v1",
        allow_remote=True,
    )
    assert numeric.endpoint_url == "https://192.0.2.1/v1/chat/completions"
    with pytest.raises(ValidationError):
        OpenAICompatibleConfig(
            base_url=f"https://{'a' * 250}.test/v1",
            allow_remote=True,
        )
    with pytest.raises(ValidationError):
        OpenAICompatibleConfig(
            base_url="https://api.example.test./v1",
            allow_remote=True,
        )


def test_config_is_strict_frozen_and_cannot_contain_a_credential_value() -> None:
    with pytest.raises(ValidationError):
        OpenAICompatibleConfig(api_key="do-not-store")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        OpenAICompatibleConfig(timeout_seconds=5)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        OpenAICompatibleConfig(credential_env="BAD-NAME")
    with pytest.raises(ValidationError):
        OpenAICompatibleConfig(base_url=1)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        OpenAICompatibleConfig(allow_remote=1)  # type: ignore[arg-type]
    config = OpenAICompatibleConfig(credential_env="LOCAL_MODEL_TOKEN")
    with pytest.raises(ValidationError):
        config.model = "changed"  # type: ignore[misc]
    assert "do-not-store" not in repr(config)


class _InvalidIdentityCounter:
    @property
    def identity(self) -> object:
        return object()


@pytest.mark.parametrize("case", ("config", "key", "callable", "counter", "transport"))
def test_constructor_rejects_invalid_injected_boundaries_value_free(case: str) -> None:
    values: dict[str, object] = {
        "config": OpenAICompatibleConfig(),
        "installation_key": KEY,
        "transport_factory": lambda: httpx.MockTransport(lambda _request: httpx.Response(500)),
    }
    if case == "config":
        values["config"] = object()
    elif case == "key":
        values["installation_key"] = object()
    elif case == "callable":
        values["credential_lookup"] = 1
    elif case == "counter":
        values["model_token_counter"] = _InvalidIdentityCounter()
    else:
        values["transport_factory"] = lambda: object()

    with pytest.raises(OpenAICompatibleError) as captured:
        OpenAICompatibleClient(  # type: ignore[arg-type]
            values.pop("config"),
            **values,
        )

    assert captured.value.code is OpenAICompatibleErrorCode.INVALID_REQUEST


def test_constructor_rejects_counter_for_another_model() -> None:
    counter = DeterministicModelTokenCounter(
        model_id="gpt-oss:120b",
        input_token_count=1,
        output_token_count=1,
    )
    with pytest.raises(OpenAICompatibleError) as captured:
        OpenAICompatibleClient(
            OpenAICompatibleConfig(),
            installation_key=KEY,
            model_token_counter=counter,
            transport_factory=lambda: httpx.MockTransport(lambda _request: httpx.Response(500)),
        )
    assert captured.value.code is OpenAICompatibleErrorCode.INVALID_REQUEST


@pytest.mark.parametrize(
    "phase",
    (StructuredCallPhase.MEMORY_EDIT, StructuredCallPhase.INTERVENTION),
)
async def test_generate_posts_the_exact_canonical_phase_request(
    phase: StructuredCallPhase,
) -> None:
    observed: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(
            200,
            json=_envelope(phase),
            headers={"Content-Type": "application/json; charset=utf-8"},
        )

    config = OpenAICompatibleConfig(
        seed=73,
        reasoning_effort="high",
        timeout_seconds=15.0,
    )
    client = _client(handler, config=config)
    structured_request = _request(phase)

    result = await client.generate(structured_request)

    assert len(observed) == 1
    request = observed[0]
    assert request.method == "POST"
    assert str(request.url) == "http://127.0.0.1:11434/v1/chat/completions"
    assert request.headers["accept"] == "application/json"
    assert request.headers["accept-encoding"] == "identity"
    assert request.headers["content-type"] == "application/json"
    payload = _payload(phase).as_json_object()
    assert request.content == canonical_json(
        {
            "messages": payload["messages"],
            "model": "gpt-oss:20b",
            "reasoning_effort": "high",
            "response_format": payload["response_format"],
            "seed": 73,
            "stream": False,
            "temperature": 0,
        }
    )
    assert json.loads(request.content)["response_format"] == payload["response_format"]
    assert validated_result_for_request(structured_request, result) == result
    assert result.status is StructuredCallStatus.COMPLETED
    assert result.parse_status is StructuredCallParseStatus.VALID
    await client.aclose()


async def test_forced_reminder_contract_is_accepted_and_preserved_on_wire() -> None:
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(
            200,
            json=_envelope(
                StructuredCallPhase.INTERVENTION,
                content=_forced_reminder_output(),
            ),
            headers={"Content-Type": "application/json"},
        )

    client = _client(handler)
    request = _request(
        StructuredCallPhase.INTERVENTION,
        forced_reminder=True,
    )

    result = await client.generate(request)

    assert result.status is StructuredCallStatus.COMPLETED
    assert result.parse_status is StructuredCallParseStatus.VALID
    assert result.output is not None
    assert result.output.action is InterventionAction.REMIND
    assert len(observed) == 1
    wire = json.loads(observed[0].content)
    assert (
        wire["response_format"]
        == _payload(
            StructuredCallPhase.INTERVENTION,
            forced_reminder=True,
        ).as_json_object()["response_format"]
    )
    assert wire["response_format"]["json_schema"]["name"] == (
        "saliencegate_forced_reminder_output_v1"
    )
    await client.aclose()


@pytest.mark.parametrize(
    ("phase", "fixture_index"),
    (
        (StructuredCallPhase.MEMORY_EDIT, 0),
        (StructuredCallPhase.INTERVENTION, 1),
    ),
)
async def test_published_paper_prompt_is_accepted_and_preserved_on_wire(
    phase: StructuredCallPhase,
    fixture_index: int,
) -> None:
    fixture_path = Path(__file__).parents[1] / "fixtures/prompts/paper_two_phase_v1.json"
    fixture = json.loads(fixture_path.read_bytes())
    request_payload = fixture["phase_prompts"][fixture_index]["request_payload"]
    payload = StructuredPromptPayload.model_validate_json(canonical_json(request_payload))
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(
            200,
            json=_envelope(phase),
            headers={"Content-Type": "application/json"},
        )

    client = _client(handler)
    structured_request = _request(phase, payload=payload.as_json_object())

    result = await client.generate(structured_request)

    assert result.status is StructuredCallStatus.COMPLETED
    assert len(observed) == 1
    wire = json.loads(observed[0].content)
    assert wire["messages"] == request_payload["messages"]
    assert wire["response_format"] == request_payload["response_format"]
    await client.aclose()


async def test_authorization_comes_only_from_the_named_environment_variable() -> None:
    authorizations: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        authorizations.append(request.headers.get("authorization"))
        return httpx.Response(
            200,
            json=_envelope(),
            headers={"Content-Type": "application/json"},
        )

    def forbidden_lookup(_name: str) -> str | None:
        raise AssertionError("ambient OPENAI_API_KEY must not be read")

    without_name = _client(handler, credential_lookup=forbidden_lookup)
    await without_name.generate(_request())
    await without_name.aclose()

    environment: dict[str, str] = {}
    named = _client(
        handler,
        config=OpenAICompatibleConfig(credential_env="LOCAL_MODEL_TOKEN"),
        credential_lookup=environment.get,
    )
    await named.generate(_request())
    environment["LOCAL_MODEL_TOKEN"] = "local-secret"
    result = await named.generate(_request())

    assert authorizations == [None, None, "Bearer local-secret"]
    assert "local-secret" not in repr(named)
    assert "local-secret" not in result.model_dump_json()
    await named.aclose()


async def test_header_injection_is_rejected_before_transport_without_secret_leakage() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError

    client = _client(
        handler,
        config=OpenAICompatibleConfig(credential_env="LOCAL_MODEL_TOKEN"),
        credential_lookup=lambda _name: "secret\r\nX-Leak: yes",
    )

    with pytest.raises(OpenAICompatibleError) as captured:
        await client.generate(_request())

    assert captured.value.code is OpenAICompatibleErrorCode.INVALID_REQUEST
    assert "secret" not in str(captured.value)
    assert calls == 0
    await client.aclose()


async def test_non_text_credential_lookup_is_rejected_before_transport() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError

    client = _client(
        handler,
        config=OpenAICompatibleConfig(credential_env="LOCAL_MODEL_TOKEN"),
        credential_lookup=lambda _name: 123,  # type: ignore[return-value]
    )

    with pytest.raises(OpenAICompatibleError) as captured:
        await client.generate(_request())

    assert captured.value.code is OpenAICompatibleErrorCode.INVALID_REQUEST
    assert calls == 0
    await client.aclose()


async def test_wrong_phase_response_schema_is_rejected_before_transport() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError

    client = _client(handler)
    request = _request(
        StructuredCallPhase.MEMORY_EDIT,
        payload=_payload(StructuredCallPhase.INTERVENTION).as_json_object(),
    )

    with pytest.raises(OpenAICompatibleError) as captured:
        await client.generate(request)

    assert captured.value.code is OpenAICompatibleErrorCode.INVALID_REQUEST
    assert calls == 0
    await client.aclose()


@pytest.mark.parametrize(
    "mismatch",
    ("forced_name_optional_schema", "optional_name_forced_schema", "unknown_name"),
)
async def test_only_exact_allowlisted_response_format_pairs_reach_transport(
    mismatch: str,
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError

    optional = _payload(StructuredCallPhase.INTERVENTION).as_json_object()
    forced = _payload(
        StructuredCallPhase.INTERVENTION,
        forced_reminder=True,
    ).as_json_object()
    payload = json.loads(canonical_json(optional))
    response_schema = payload["response_format"]["json_schema"]
    if mismatch == "forced_name_optional_schema":
        response_schema["name"] = "saliencegate_forced_reminder_output_v1"
    elif mismatch == "optional_name_forced_schema":
        response_schema["schema"] = forced["response_format"]["json_schema"]["schema"]
    else:
        response_schema["name"] = "saliencegate_unreviewed_output_v1"

    client = _client(handler)
    request = _request(StructuredCallPhase.INTERVENTION, payload=payload)

    with pytest.raises(OpenAICompatibleError) as captured:
        await client.generate(request)

    assert captured.value.code is OpenAICompatibleErrorCode.INVALID_REQUEST
    assert calls == 0
    await client.aclose()


async def test_completed_result_keeps_provider_and_local_usage_distinct() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_envelope(),
            headers={"Content-Type": "application/json"},
        )

    counter = DeterministicModelTokenCounter(
        model_id="gpt-oss:20b",
        input_token_count=13,
        output_token_count=7,
    )
    client = _client(handler, counter=counter)

    result = await client.generate(_request())

    assert result.usage.provider_usage_provenance is ProviderUsageProvenance.PROVIDER_REPORTED
    assert result.usage.provider_tokens == 25
    assert result.usage.canonical_usage_provenance is CanonicalUsageProvenance.LOCAL_COUNTER
    assert result.usage.canonical_tokens == 20
    assert result.usage.local_counter_id == counter.identity.counter_id
    assert result.usage.local_counter_configuration_digest == counter.identity.configuration_digest
    await client.aclose()


class _InvalidCountCounter:
    def __init__(self, *, fail_direction: str) -> None:
        self.delegate = DeterministicModelTokenCounter(
            model_id="gpt-oss:20b",
            input_token_count=2,
            output_token_count=3,
        )
        self.fail_direction = fail_direction

    @property
    def identity(self) -> ModelTokenCounterIdentity:
        return self.delegate.identity

    def count_input(self, request: StructuredCallRequest) -> object:
        if self.fail_direction == "input_value":
            return object()
        if self.fail_direction == "input_direction":
            return self.delegate.count_output(model_id=request.model_id, completion="x")
        return self.delegate.count_input(request)

    def count_output(self, *, model_id: str, completion: str) -> object:
        if self.fail_direction == "output":
            return object()
        return self.delegate.count_output(model_id=model_id, completion=completion)


@pytest.mark.parametrize("failure", ("input_value", "input_direction"))
async def test_invalid_input_counter_attestation_fails_before_transport(failure: str) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError

    client = OpenAICompatibleClient(
        OpenAICompatibleConfig(),
        installation_key=KEY,
        model_token_counter=_InvalidCountCounter(fail_direction=failure),  # type: ignore[arg-type]
        transport_factory=lambda: httpx.MockTransport(handler),
    )

    with pytest.raises(OpenAICompatibleError) as captured:
        await client.generate(_request())

    assert captured.value.code is OpenAICompatibleErrorCode.TOKEN_ACCOUNTING
    assert calls == 0
    await client.aclose()


async def test_invalid_output_counter_attestation_fails_the_attempt_closed() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_envelope(),
            headers={"Content-Type": "application/json"},
        )

    client = OpenAICompatibleClient(
        OpenAICompatibleConfig(),
        installation_key=KEY,
        model_token_counter=_InvalidCountCounter(fail_direction="output"),  # type: ignore[arg-type]
        transport_factory=lambda: httpx.MockTransport(handler),
    )

    result = await client.generate(_request())

    assert result.status is StructuredCallStatus.MODEL_ERROR
    assert result.usage.canonical_input_tokens == 2
    assert result.usage.canonical_output_tokens is None
    await client.aclose()


async def test_complete_local_counter_allows_missing_provider_usage() -> None:
    envelope = _envelope()
    envelope.pop("usage")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=envelope,
            headers={"Content-Type": "application/json"},
        )

    counter = DeterministicModelTokenCounter(
        model_id="gpt-oss:20b",
        input_token_count=11,
        output_token_count=3,
    )
    client = _client(handler, counter=counter)

    result = await client.generate(_request())

    assert result.status is StructuredCallStatus.COMPLETED
    assert result.usage.provider_tokens is None
    assert result.usage.canonical_tokens == 14
    await client.aclose()


async def test_visible_completion_is_hmac_attested_and_reasoning_is_discarded() -> None:
    completion = _output(StructuredCallPhase.MEMORY_EDIT)
    clock_values = iter((1_000, 6_000, 1_000, 6_000))

    def clock() -> int:
        return next(clock_values)

    def first(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_envelope(reasoning="private-chain-a"),
            headers={"Content-Type": "application/json"},
        )

    def second(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_envelope(reasoning={"private": "chain-b"}),
            headers={"Content-Type": "application/json"},
        )

    first_client = _client(first, monotonic_ns=clock)
    second_client = _client(second, monotonic_ns=clock)
    request = _request()

    first_result = await first_client.generate(request)
    second_result = await second_client.generate(request)

    assert first_result == second_result
    assert first_result.completion_digest is not None
    assert first_result.completion_digest.algorithm is PayloadDigestAlgorithm.HMAC_SHA256
    assert first_result.completion_digest.value == KEY._hmac_sha256(
        completion.encode(),
        domain=b"saliencegate:model:assistant-message-content-utf8/v1",
    )
    serialized = first_result.model_dump_json()
    assert "private-chain" not in serialized
    assert "provider_reasoning" not in serialized
    await first_client.aclose()
    await second_client.aclose()


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_model",
        "wrong_model",
        "wrong_role",
        "missing_role",
        "wrong_index",
        "missing_index",
        "unfinished",
        "missing_finish_reason",
    ),
)
async def test_provider_envelope_identity_must_be_exact(mutation: str) -> None:
    envelope = _envelope()
    choice = envelope["choices"][0]  # type: ignore[index]
    message = choice["message"]  # type: ignore[index]
    if mutation == "missing_model":
        envelope.pop("model")
    elif mutation == "wrong_model":
        envelope["model"] = "gpt-oss:120b"
    elif mutation == "wrong_role":
        message["role"] = "tool"  # type: ignore[index]
    elif mutation == "missing_role":
        message.pop("role")  # type: ignore[union-attr]
    elif mutation == "wrong_index":
        choice["index"] = 99  # type: ignore[index]
    elif mutation == "missing_index":
        choice.pop("index")  # type: ignore[union-attr]
    elif mutation == "unfinished":
        choice["finish_reason"] = "length"  # type: ignore[index]
    else:
        choice.pop("finish_reason")  # type: ignore[union-attr]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=envelope,
            headers={"Content-Type": "application/json"},
        )

    client = _client(handler)
    result = await client.generate(_request())

    assert result.status is StructuredCallStatus.MODEL_ERROR
    assert result.completion_digest is None
    await client.aclose()


@pytest.mark.parametrize(
    "content_type",
    (
        "application/json; charset=utf-16",
        "application/json; charset=utf-8; charset=utf-16",
        "application/json; secret=value",
        "application/json, text/html",
        "application/json;",
        "",
    ),
)
async def test_only_bare_or_utf8_json_mime_is_accepted(content_type: str) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        if not content_type:
            return httpx.Response(200, content=canonical_json(_envelope()))
        return httpx.Response(
            200,
            json=_envelope(),
            headers={"Content-Type": content_type},
        )

    client = _client(handler)
    result = await client.generate(_request())

    assert result.status is StructuredCallStatus.MODEL_ERROR
    await client.aclose()


@pytest.mark.parametrize(
    ("phase", "content", "expected"),
    (
        (
            StructuredCallPhase.MEMORY_EDIT,
            "not-json",
            StructuredCallParseStatus.SCHEMA_INVALID,
        ),
        (
            StructuredCallPhase.MEMORY_EDIT,
            '{"schema_version":"intervention-output/v1","action":"silence",'
            '"claims":[],"confidence":1.0}',
            StructuredCallParseStatus.SCHEMA_INVALID,
        ),
        (
            StructuredCallPhase.INTERVENTION,
            '{"schema_version":"intervention-output/v1","action":"remind",'
            '"claims":[],"confidence":1.0}',
            StructuredCallParseStatus.EMPTY_REMINDER,
        ),
        (
            StructuredCallPhase.INTERVENTION,
            '{"schema_version":"intervention-output/v1","action":"remind",'
            '"claims":[{},{},{}],"confidence":1.0}',
            StructuredCallParseStatus.CLAIM_OVER_LIMIT,
        ),
    ),
)
async def test_invalid_visible_content_is_a_repairable_attested_observation(
    phase: StructuredCallPhase,
    content: str,
    expected: StructuredCallParseStatus,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_envelope(phase, content=content),
            headers={"Content-Type": "application/json"},
        )

    client = _client(handler)
    request = _request(phase)

    result = await client.generate(request)

    assert result.status is StructuredCallStatus.COMPLETED
    assert result.parse_status is expected
    assert result.output is None
    assert result.completion_byte_count == len(content.encode())
    assert result.completion_digest is not None
    await client.aclose()


@pytest.mark.parametrize(
    ("content", "expected"),
    (
        (
            '{"schema_version":"intervention-output/v1","action":"silence",'
            '"claims":[],"confidence":1.0}',
            StructuredCallParseStatus.SCHEMA_INVALID,
        ),
        (
            '{"schema_version":"intervention-output/v1","action":"remind",'
            '"claims":[],"confidence":1.0}',
            StructuredCallParseStatus.EMPTY_REMINDER,
        ),
    ),
)
async def test_forced_reminder_completion_is_validated_against_its_request_contract(
    content: str,
    expected: StructuredCallParseStatus,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_envelope(StructuredCallPhase.INTERVENTION, content=content),
            headers={"Content-Type": "application/json"},
        )

    client = _client(handler)
    request = _request(
        StructuredCallPhase.INTERVENTION,
        forced_reminder=True,
    )

    result = await client.generate(request)

    assert result.status is StructuredCallStatus.COMPLETED
    assert result.parse_status is expected
    assert result.output is None
    assert result.completion_digest is not None
    await client.aclose()


@pytest.mark.parametrize(
    ("response_factory", "expected_status"),
    (
        (
            lambda request: (_ for _ in ()).throw(
                httpx.ReadTimeout("provider secret", request=request)
            ),
            StructuredCallStatus.MODEL_TIMEOUT,
        ),
        (
            lambda request: (_ for _ in ()).throw(
                httpx.ConnectError("provider secret", request=request)
            ),
            StructuredCallStatus.MODEL_ERROR,
        ),
        (
            lambda _request: httpx.Response(
                503,
                content=b"provider secret",
                headers={"Content-Type": "application/json"},
            ),
            StructuredCallStatus.MODEL_ERROR,
        ),
        (
            lambda _request: httpx.Response(
                200,
                content=b"provider secret",
                headers={"Content-Type": "text/html"},
            ),
            StructuredCallStatus.MODEL_ERROR,
        ),
        (
            lambda _request: httpx.Response(
                200,
                content=b"{malformed",
                headers={"Content-Type": "application/json"},
            ),
            StructuredCallStatus.MODEL_ERROR,
        ),
        (
            lambda _request: httpx.Response(
                200,
                json={"choices": []},
                headers={"Content-Type": "application/json"},
            ),
            StructuredCallStatus.MODEL_ERROR,
        ),
        (
            lambda _request: httpx.Response(
                200,
                json={key: value for key, value in _envelope().items() if key != "usage"},
                headers={"Content-Type": "application/json"},
            ),
            StructuredCallStatus.MODEL_ERROR,
        ),
    ),
)
async def test_attempted_failures_return_sanitized_bound_results(
    response_factory: Callable[[httpx.Request], httpx.Response],
    expected_status: StructuredCallStatus,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return response_factory(request)

    client = _client(handler)
    request = _request()

    result = await client.generate(request)

    assert calls == 1
    assert validated_result_for_request(request, result) == result
    assert result.status is expected_status
    assert result.parse_status is StructuredCallParseStatus.NOT_ATTEMPTED
    assert result.output is None
    assert result.completion_digest is None
    assert result.completion_byte_count is None
    assert "provider secret" not in result.model_dump_json()
    await client.aclose()


@pytest.mark.parametrize(
    "usage",
    (
        {},
        {"prompt_tokens": 1},
        {"prompt_tokens": True, "completion_tokens": 1},
        {"prompt_tokens": -1, "completion_tokens": 1},
        {"prompt_tokens": 1.0, "completion_tokens": 1},
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 3},
        {"prompt_tokens": (1 << 63) - 1, "completion_tokens": 1},
    ),
)
async def test_malformed_provider_usage_is_never_coerced_or_reconciled(usage: object) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_envelope(usage=usage),
            headers={"Content-Type": "application/json"},
        )

    client = _client(handler)
    result = await client.generate(_request())

    assert result.status is StructuredCallStatus.MODEL_ERROR
    assert result.usage.provider_tokens is None
    await client.aclose()


async def test_provider_total_is_optional_when_component_counts_are_exact() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_envelope(usage={"prompt_tokens": 4, "completion_tokens": 2}),
            headers={"Content-Type": "application/json"},
        )

    client = _client(handler)
    result = await client.generate(_request())

    assert result.status is StructuredCallStatus.COMPLETED
    assert result.usage.provider_tokens == 6
    await client.aclose()


async def test_non_utf8_visible_unicode_is_rejected_as_an_invalid_envelope() -> None:
    envelope = _envelope(content="\ud800")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=json.dumps(envelope).encode("ascii"),
            headers={"Content-Type": "application/json"},
        )

    client = _client(handler)
    result = await client.generate(_request())

    assert result.status is StructuredCallStatus.MODEL_ERROR
    await client.aclose()


class _ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self.chunks = chunks
        self.closed = False
        self.yielded = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            self.yielded += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


async def test_response_limit_is_enforced_during_streaming_and_closes_the_body() -> None:
    stream = _ChunkStream((b"12345678", b"901234567", b"must-not-be-read"))

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=stream,
            headers={"Content-Type": "application/json"},
        )

    client = _client(handler, config=OpenAICompatibleConfig(max_response_bytes=16))

    result = await client.generate(_request())

    assert result.status is StructuredCallStatus.MODEL_ERROR
    assert stream.yielded == 2
    assert stream.closed is True
    await client.aclose()


async def test_compressed_response_is_rejected_before_decompression() -> None:
    compressed = gzip.compress(canonical_json(_envelope()) * 1_000)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(
            200,
            content=compressed,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
        )

    client = _client(handler, config=OpenAICompatibleConfig(max_response_bytes=1_024))
    result = await client.generate(_request())

    assert result.status is StructuredCallStatus.MODEL_ERROR
    assert result.completion_digest is None
    await client.aclose()


async def test_declared_oversized_body_is_rejected_without_streaming() -> None:
    stream = _ChunkStream((b"must-not-be-read",))

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=stream,
            headers={
                "Content-Type": "application/json",
                "Content-Length": "17",
            },
        )

    client = _client(handler, config=OpenAICompatibleConfig(max_response_bytes=16))
    result = await client.generate(_request())

    assert result.status is StructuredCallStatus.MODEL_ERROR
    assert stream.yielded == 0
    assert stream.closed is True
    await client.aclose()


async def test_split_utf8_response_is_counted_as_bytes_and_accepted() -> None:
    envelope = _envelope()
    envelope["note"] = "café"
    body = canonical_json(envelope)
    split = body.index("é".encode()) + 1
    stream = _ChunkStream((body[:split], body[split:]))

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=stream,
            headers={"Content-Type": "application/json"},
        )

    client = _client(handler, config=OpenAICompatibleConfig(max_response_bytes=len(body)))
    result = await client.generate(_request())

    assert result.status is StructuredCallStatus.COMPLETED
    assert stream.closed is True
    await client.aclose()


@pytest.mark.parametrize("invalid_json", ("duplicate", "non_finite", "top_level_array"))
async def test_outer_json_must_be_unique_finite_and_an_object(invalid_json: str) -> None:
    body = canonical_json(_envelope())
    if invalid_json == "duplicate":
        body = body.replace(
            b'"model":"gpt-oss:20b"',
            b'"model":"gpt-oss:20b","model":"gpt-oss:20b"',
            1,
        )
    elif invalid_json == "non_finite":
        body = body[:-1] + b',"temperature":NaN}'
    else:
        body = b"[]"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body,
            headers={"Content-Type": "application/json"},
        )

    client = _client(handler)
    result = await client.generate(_request())

    assert result.status is StructuredCallStatus.MODEL_ERROR
    await client.aclose()


@pytest.mark.parametrize("shape", ("deep", "wide_list", "wide_object"))
async def test_outer_json_structural_limits_precede_recursive_validation(shape: str) -> None:
    envelope = _envelope()
    if shape == "deep":
        nested: object = None
        for _ in range(70):
            nested = [nested]
        envelope["extra"] = nested
    elif shape == "wide_list":
        envelope["extra"] = list(range(100_001))
    else:
        envelope["extra"] = {f"k{index}": index for index in range(100_001)}
    body = canonical_json(envelope)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body,
            headers={"Content-Type": "application/json"},
        )

    client = _client(handler)
    result = await client.generate(_request())

    assert result.status is StructuredCallStatus.MODEL_ERROR
    await client.aclose()


async def test_preconsumed_body_still_enforces_the_byte_ceiling() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"x" * 17,
            headers={
                "Content-Type": "application/json",
                "Content-Length": "1",
            },
        )

    client = _client(handler, config=OpenAICompatibleConfig(max_response_bytes=16))
    result = await client.generate(_request())

    assert result.status is StructuredCallStatus.MODEL_ERROR
    await client.aclose()


async def test_body_at_the_exact_response_limit_is_accepted() -> None:
    body = canonical_json(_envelope())

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body,
            headers={"Content-Type": "application/json"},
        )

    client = _client(handler, config=OpenAICompatibleConfig(max_response_bytes=len(body)))

    result = await client.generate(_request())

    assert result.status is StructuredCallStatus.COMPLETED
    await client.aclose()


async def test_redirects_statuses_and_transport_failures_are_never_retried() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            307,
            headers={
                "Location": "https://remote.example.test/v1/chat/completions",
                "Content-Type": "application/json",
            },
        )

    client = _client(handler)
    result = await client.generate(_request())

    assert result.status is StructuredCallStatus.MODEL_ERROR
    assert calls == 1
    await client.aclose()


async def test_configured_model_mismatch_fails_before_transport() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError

    client = _client(handler)

    with pytest.raises(OpenAICompatibleError) as captured:
        await client.generate(_request(model_id="gpt-oss:120b"))

    assert captured.value.code is OpenAICompatibleErrorCode.INVALID_REQUEST
    assert calls == 0
    await client.aclose()


@pytest.mark.parametrize("clock_value", (-1, "invalid"))
async def test_invalid_start_clock_fails_before_transport(clock_value: object) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError

    client = _client(handler, monotonic_ns=lambda: clock_value)  # type: ignore[arg-type]

    with pytest.raises(OpenAICompatibleError) as captured:
        await client.generate(_request())

    assert captured.value.code is OpenAICompatibleErrorCode.INVALID_REQUEST
    assert calls == 0
    await client.aclose()


async def test_invalid_finish_clock_is_safely_recorded_as_zero_latency() -> None:
    values = iter((1, "invalid"))

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_envelope(),
            headers={"Content-Type": "application/json"},
        )

    client = _client(handler, monotonic_ns=lambda: next(values))  # type: ignore[arg-type]
    result = await client.generate(_request())

    assert result.status is StructuredCallStatus.COMPLETED
    assert result.usage.latency_us == 0
    await client.aclose()


async def test_cancellation_propagates_value_free_and_does_not_retry() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        raise AssertionError

    client = _client(handler)
    task = asyncio.create_task(client.generate(_request()))
    await entered.wait()
    task.cancel("caller secret")

    with pytest.raises(asyncio.CancelledError) as captured:
        await task

    assert captured.value.args == ()
    assert captured.value.__cause__ is None
    assert calls == 1
    await client.aclose()


async def test_queued_generation_cancellation_is_also_value_free() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return httpx.Response(
            200,
            json=_envelope(),
            headers={"Content-Type": "application/json"},
        )

    client = _client(handler)
    first = asyncio.create_task(client.generate(_request()))
    await entered.wait()
    queued = asyncio.create_task(client.generate(_request()))
    await asyncio.sleep(0)
    queued.cancel("queue-secret")

    with pytest.raises(asyncio.CancelledError) as captured:
        await queued

    release.set()
    assert (await first).status is StructuredCallStatus.COMPLETED
    assert captured.value.args == ()
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert calls == 1
    await client.aclose()


async def test_queued_close_cancellation_is_value_free_and_retryable() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(_request: httpx.Request) -> httpx.Response:
        entered.set()
        await release.wait()
        return httpx.Response(
            200,
            json=_envelope(),
            headers={"Content-Type": "application/json"},
        )

    client = _client(handler)
    active = asyncio.create_task(client.generate(_request()))
    await entered.wait()
    closing = asyncio.create_task(client.aclose())
    await asyncio.sleep(0)
    closing.cancel("close-secret")

    with pytest.raises(asyncio.CancelledError) as captured:
        await closing

    release.set()
    assert (await active).status is StructuredCallStatus.COMPLETED
    assert captured.value.args == ()
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    await client.aclose()


async def test_close_is_idempotent_and_post_close_generation_is_local() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json=_envelope(),
            headers={"Content-Type": "application/json"},
        )

    client = _client(handler)
    await client.aclose()
    await client.aclose()

    with pytest.raises(OpenAICompatibleError) as captured:
        await client.generate(_request())

    assert client.is_closed is True
    assert captured.value.code is OpenAICompatibleErrorCode.CLOSED
    assert calls == 0


class _RetryCloseTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.close_calls = 0

    async def handle_async_request(self, _request: httpx.Request) -> httpx.Response:
        raise AssertionError

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            raise RuntimeError("transport secret")


async def test_failed_close_is_sanitized_and_can_be_retried() -> None:
    transport = _RetryCloseTransport()
    client = OpenAICompatibleClient(
        OpenAICompatibleConfig(),
        installation_key=KEY,
        transport_factory=lambda: transport,
    )

    with pytest.raises(OpenAICompatibleError) as captured:
        await client.aclose()

    assert captured.value.code is OpenAICompatibleErrorCode.TRANSPORT
    assert "secret" not in str(captured.value)
    assert client.is_closed is True
    await client.aclose()
    await client.aclose()
    assert transport.close_calls == 2


class _CancelCloseTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.close_calls = 0

    async def handle_async_request(self, _request: httpx.Request) -> httpx.Response:
        raise AssertionError

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            raise asyncio.CancelledError("transport secret")


async def test_cancelled_transport_close_is_value_free_and_can_be_retried() -> None:
    transport = _CancelCloseTransport()
    client = OpenAICompatibleClient(
        OpenAICompatibleConfig(),
        installation_key=KEY,
        transport_factory=lambda: transport,
    )

    with pytest.raises(asyncio.CancelledError) as captured:
        await client.aclose()

    assert captured.value.args == ()
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    await client.aclose()
    assert transport.close_calls == 2


async def test_async_context_manager_closes_the_owned_client() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_envelope(),
            headers={"Content-Type": "application/json"},
        )

    client = _client(handler)
    async with client as entered:
        assert entered is client
        assert (await entered.generate(_request())).status is StructuredCallStatus.COMPLETED

    assert client.is_closed is True
    with pytest.raises(OpenAICompatibleError) as captured:
        async with client:
            raise AssertionError
    assert captured.value.code is OpenAICompatibleErrorCode.CLOSED


async def test_default_transport_ignores_ambient_tls_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SSL_CERT_FILE", "/definitely/missing/ambient-ca.pem")
    client = OpenAICompatibleClient(OpenAICompatibleConfig(), installation_key=KEY)

    await client.aclose()


def test_core_models_namespace_does_not_eagerly_import_optional_runtime() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; import saliencegate; import saliencegate.models; "
                "assert 'httpx' not in sys.modules; "
                "assert 'openai_harmony' not in sys.modules; "
                "assert 'saliencegate.models.openai_compatible' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
