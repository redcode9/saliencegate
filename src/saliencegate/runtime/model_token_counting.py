from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from importlib import import_module, metadata
from typing import Annotated, Literal, Protocol, Self, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.domain import canonical_json, length_prefixed_sha256
from saliencegate.domain.records import ComponentIdentifier, Sha256Digest
from saliencegate.ports.model_calls import (
    CanonicalUsageProvenance,
    ProviderUsageProvenance,
    StructuredCallRequest,
    StructuredCallUsage,
    validated_structured_call_request,
)
from saliencegate.prompts.contracts import StructuredPromptPayload

MODEL_TOKEN_COUNTER_IDENTITY_SCHEMA_VERSION: Literal["model-token-counter-identity/v1"] = (
    "model-token-counter-identity/v1"
)
MODEL_TOKEN_COUNT_SCHEMA_VERSION: Literal["model-token-count/v1"] = "model-token-count/v1"

_COUNTER_CONFIGURATION_DIGEST_DOMAIN = "saliencegate:runtime:model-token-counter:v1"
_MAX_SIGNED_64 = (1 << 63) - 1
_HARMONY_PACKAGE_NAME = "openai-harmony"
_HARMONY_COUNTER_ID = "openai-harmony"
_HARMONY_ENCODING_NAME = "HarmonyGptOss"
_HARMONY_MODEL_IDS = frozenset(
    (
        "gpt-oss:20b",
        "gpt-oss:120b",
        "gpt-oss-20b",
        "gpt-oss-120b",
    )
)

NonNegativeSigned64 = Annotated[int, Field(ge=0, le=_MAX_SIGNED_64)]


class ModelTokenCounterInputError(ValueError):
    """A value-free rejection at the model-token counting boundary."""

    def __init__(self) -> None:
        super().__init__("model token counting input failed validation")


class ModelTokenCounterPairingError(ValueError):
    """Raised when a counter is asked to measure a different model identity."""

    def __init__(self) -> None:
        super().__init__("model token counter does not match the requested model")


class ModelTokenCounterUnavailableError(RuntimeError):
    """Raised when an explicitly selected optional counter cannot be loaded."""

    def __init__(self) -> None:
        super().__init__("optional model token counter is unavailable")


class ModelTokenAccountingUnavailableError(RuntimeError):
    """Raised when a completed live call has no trustworthy token evidence."""

    def __init__(self) -> None:
        super().__init__("live model token accounting is unavailable")


class ModelTokenDirection(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


class _ModelTokenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class ModelTokenCounterIdentity(_ModelTokenModel):
    """Opaque, immutable identity of one exact local counting configuration."""

    schema_version: Literal["model-token-counter-identity/v1"] = (
        MODEL_TOKEN_COUNTER_IDENTITY_SCHEMA_VERSION
    )
    counter_id: ComponentIdentifier
    counter_version: ComponentIdentifier
    configuration_digest: Sha256Digest
    model_id: ComponentIdentifier


class ModelTokenCount(_ModelTokenModel):
    """One local canonical count, kept separate from provider-reported usage."""

    schema_version: Literal["model-token-count/v1"] = MODEL_TOKEN_COUNT_SCHEMA_VERSION
    direction: ModelTokenDirection
    token_count: NonNegativeSigned64
    provenance: CanonicalUsageProvenance
    counter_identity: ModelTokenCounterIdentity

    @model_validator(mode="after")
    def provenance_is_a_live_local_count(self) -> Self:
        if self.provenance is not CanonicalUsageProvenance.LOCAL_COUNTER:
            raise ValueError("model token count must be produced by a local counter")
        return self

    @property
    def counter_id(self) -> str:
        return self.counter_identity.counter_id

    @property
    def counter_version(self) -> str:
        return self.counter_identity.counter_version

    @property
    def configuration_digest(self) -> str:
        return self.counter_identity.configuration_digest

    @property
    def model_id(self) -> str:
        return self.counter_identity.model_id


@runtime_checkable
class ModelTokenCounter(Protocol):
    @property
    def identity(self) -> ModelTokenCounterIdentity: ...

    def count_input(self, request: StructuredCallRequest) -> ModelTokenCount: ...

    def count_output(self, *, model_id: str, completion: str) -> ModelTokenCount: ...


def validated_live_model_token_usage(
    usage: StructuredCallUsage,
    *,
    configured_counter: ModelTokenCounterIdentity | None,
) -> StructuredCallUsage:
    """Accept a completed live call only with provider or complete local evidence."""

    try:
        if type(usage) is not StructuredCallUsage:
            raise ValueError
        checked = StructuredCallUsage.model_validate_json(usage.model_dump_json(warnings=False))
        provider_complete = (
            checked.provider_usage_provenance is ProviderUsageProvenance.PROVIDER_REPORTED
            and checked.provider_input_tokens is not None
            and checked.provider_output_tokens is not None
        )
        canonical_claimed = (
            checked.canonical_usage_provenance is not CanonicalUsageProvenance.UNAVAILABLE
        )
        canonical_complete = (
            checked.canonical_usage_provenance is CanonicalUsageProvenance.LOCAL_COUNTER
            and checked.canonical_input_tokens is not None
            and checked.canonical_output_tokens is not None
        )
        if canonical_claimed:
            if checked.canonical_usage_provenance is not CanonicalUsageProvenance.LOCAL_COUNTER:
                raise ValueError
            identity = ModelTokenCounterIdentity(
                counter_id=cast(str, checked.local_counter_id),
                counter_version=cast(str, checked.local_counter_version),
                configuration_digest=cast(
                    str,
                    checked.local_counter_configuration_digest,
                ),
                model_id=cast(str, checked.local_counter_model_id),
            )
            if configured_counter is None or identity != configured_counter:
                raise ValueError
        if not provider_complete and not canonical_complete:
            raise ValueError
        return checked
    except Exception:
        raise ModelTokenAccountingUnavailableError() from None


def _configuration_digest(
    *,
    counter_id: str,
    counter_version: str,
    model_id: str,
    configuration: Mapping[str, object],
) -> str:
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": "model-token-counter-configuration/v1",
                "counter_id": counter_id,
                "counter_version": counter_version,
                "model_id": model_id,
                "configuration": configuration,
            }
        ),
        domain=_COUNTER_CONFIGURATION_DIGEST_DOMAIN,
    )


def _identity(
    *,
    counter_id: str,
    counter_version: str,
    model_id: str,
    configuration: Mapping[str, object],
) -> ModelTokenCounterIdentity:
    return ModelTokenCounterIdentity(
        counter_id=counter_id,
        counter_version=counter_version,
        configuration_digest=_configuration_digest(
            counter_id=counter_id,
            counter_version=counter_version,
            model_id=model_id,
            configuration=configuration,
        ),
        model_id=model_id,
    )


def _exact_signed_64(value: object) -> int:
    if type(value) is not int or value < 0 or value > _MAX_SIGNED_64:
        raise ModelTokenCounterInputError()
    return value


def _exact_utf8(value: object) -> str:
    if type(value) is not str:
        raise ModelTokenCounterInputError()
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise ModelTokenCounterInputError() from None
    return value


def _paired_model(identity: ModelTokenCounterIdentity, model_id: object) -> None:
    if type(model_id) is not str or model_id != identity.model_id:
        raise ModelTokenCounterPairingError()


def _count(
    *,
    identity: ModelTokenCounterIdentity,
    direction: ModelTokenDirection,
    token_count: object,
) -> ModelTokenCount:
    return ModelTokenCount(
        direction=direction,
        token_count=_exact_signed_64(token_count),
        provenance=CanonicalUsageProvenance.LOCAL_COUNTER,
        counter_identity=identity,
    )


class DeterministicModelTokenCounter:
    """A fixed-count test double with an explicit non-tokenizer identity.

    The values are supplied by the fixture. No text-size heuristic or model tokenizer is
    consulted, so this counter cannot be mistaken for the byte-based batching estimate.
    """

    __slots__ = ("_identity", "_input_token_count", "_output_token_count")

    def __init__(
        self,
        *,
        model_id: str,
        input_token_count: int,
        output_token_count: int,
    ) -> None:
        try:
            checked_model_id = _exact_utf8(model_id)
            checked_input = _exact_signed_64(input_token_count)
            checked_output = _exact_signed_64(output_token_count)
            identity = _identity(
                counter_id="deterministic-model-token-counter",
                counter_version="fixed-count-fixture/v1",
                model_id=checked_model_id,
                configuration={
                    "input_token_count": checked_input,
                    "output_token_count": checked_output,
                    "counting_rule": "fixture-supplied-fixed-counts/v1",
                },
            )
        except ModelTokenCounterInputError:
            raise
        except Exception:
            raise ModelTokenCounterInputError() from None
        self._identity = identity
        self._input_token_count = checked_input
        self._output_token_count = checked_output

    @property
    def identity(self) -> ModelTokenCounterIdentity:
        return self._identity

    def count_input(self, request: StructuredCallRequest) -> ModelTokenCount:
        try:
            checked = validated_structured_call_request(request)
        except Exception:
            raise ModelTokenCounterInputError() from None
        _paired_model(self._identity, checked.model_id)
        return _count(
            identity=self._identity,
            direction=ModelTokenDirection.INPUT,
            token_count=self._input_token_count,
        )

    def count_output(self, *, model_id: str, completion: str) -> ModelTokenCount:
        _paired_model(self._identity, model_id)
        _exact_utf8(completion)
        return _count(
            identity=self._identity,
            direction=ModelTokenDirection.OUTPUT,
            token_count=self._output_token_count,
        )


DeterministicFakeModelTokenCounter = DeterministicModelTokenCounter


class _HarmonyMessage(Protocol):
    pass


class _HarmonyMessageFactory(Protocol):
    def from_role_and_content(self, role: object, content: str) -> _HarmonyMessage: ...


class _HarmonyConversationFactory(Protocol):
    def from_messages(self, messages: list[_HarmonyMessage]) -> object: ...


class _HarmonyEncoding(Protocol):
    def render_conversation_for_completion(
        self,
        conversation: object,
        next_turn_role: object,
    ) -> list[int]: ...

    def encode(self, text: str) -> list[int]: ...


class _HarmonyEncodingName(Protocol):
    HARMONY_GPT_OSS: object


class _HarmonyRole(Protocol):
    SYSTEM: object
    USER: object
    ASSISTANT: object


class _HarmonyModule(Protocol):
    Conversation: _HarmonyConversationFactory
    HarmonyEncodingName: _HarmonyEncodingName
    Message: _HarmonyMessageFactory
    Role: _HarmonyRole

    def load_harmony_encoding(self, name: object) -> _HarmonyEncoding: ...


def _import_harmony() -> tuple[_HarmonyModule, str]:
    """Import the optional runtime only when a Harmony counter is selected."""

    module = cast(_HarmonyModule, import_module("openai_harmony"))
    package_version = metadata.version(_HARMONY_PACKAGE_NAME)
    return module, package_version


class HarmonyTokenCounter:
    """Count canonical gpt-oss prompt and visible completion tokens with Harmony."""

    __slots__ = ("_encoding", "_harmony", "_identity")

    def __init__(self, *, model_id: str = "gpt-oss:20b") -> None:
        if type(model_id) is not str or model_id not in _HARMONY_MODEL_IDS:
            raise ModelTokenCounterPairingError()
        try:
            harmony, package_version = _import_harmony()
            if type(package_version) is not str or not package_version:
                raise ValueError
            encoding = harmony.load_harmony_encoding(harmony.HarmonyEncodingName.HARMONY_GPT_OSS)
            identity = _identity(
                counter_id=_HARMONY_COUNTER_ID,
                counter_version=package_version,
                model_id=model_id,
                configuration={
                    "package_name": _HARMONY_PACKAGE_NAME,
                    "encoding_name": _HARMONY_ENCODING_NAME,
                    "input_rule": "render-conversation-for-assistant-completion/v1",
                    "input_payload_schema": "structured-prompt-payload/v1",
                    "output_rule": "encode-assistant-message-content/v1",
                    "reasoning_and_provider_envelope": "excluded",
                },
            )
        except Exception:
            raise ModelTokenCounterUnavailableError() from None
        self._harmony = harmony
        self._encoding = encoding
        self._identity = identity

    @property
    def identity(self) -> ModelTokenCounterIdentity:
        return self._identity

    def count_input(self, request: StructuredCallRequest) -> ModelTokenCount:
        try:
            checked = validated_structured_call_request(request)
        except Exception:
            raise ModelTokenCounterInputError() from None
        _paired_model(self._identity, checked.model_id)
        try:
            payload = StructuredPromptPayload.model_validate_json(canonical_json(checked.payload))
            messages = [
                self._harmony.Message.from_role_and_content(
                    self._harmony.Role.SYSTEM,
                    payload.messages[0].content,
                ),
                self._harmony.Message.from_role_and_content(
                    self._harmony.Role.USER,
                    payload.messages[1].content,
                ),
            ]
            conversation = self._harmony.Conversation.from_messages(messages)
            tokens = self._encoding.render_conversation_for_completion(
                conversation,
                self._harmony.Role.ASSISTANT,
            )
            if type(tokens) is not list:
                raise ValueError
            return _count(
                identity=self._identity,
                direction=ModelTokenDirection.INPUT,
                token_count=len(tokens),
            )
        except Exception:
            raise ModelTokenCounterInputError() from None

    def count_output(self, *, model_id: str, completion: str) -> ModelTokenCount:
        _paired_model(self._identity, model_id)
        checked_completion = _exact_utf8(completion)
        try:
            tokens = self._encoding.encode(checked_completion)
            if type(tokens) is not list:
                raise ValueError
            return _count(
                identity=self._identity,
                direction=ModelTokenDirection.OUTPUT,
                token_count=len(tokens),
            )
        except Exception:
            raise ModelTokenCounterInputError() from None


__all__ = [
    "MODEL_TOKEN_COUNTER_IDENTITY_SCHEMA_VERSION",
    "MODEL_TOKEN_COUNT_SCHEMA_VERSION",
    "DeterministicFakeModelTokenCounter",
    "DeterministicModelTokenCounter",
    "HarmonyTokenCounter",
    "ModelTokenAccountingUnavailableError",
    "ModelTokenCount",
    "ModelTokenCounter",
    "ModelTokenCounterIdentity",
    "ModelTokenCounterInputError",
    "ModelTokenCounterPairingError",
    "ModelTokenCounterUnavailableError",
    "ModelTokenDirection",
    "validated_live_model_token_usage",
]
