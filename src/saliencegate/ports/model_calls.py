from __future__ import annotations

import json
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Literal, Protocol, Self, TypeAlias, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from saliencegate.domain import JsonObject, PayloadDigest, canonical_json, length_prefixed_sha256
from saliencegate.domain.records import UUID4, ComponentIdentifier, Sha256Digest
from saliencegate.memory.proposals import (
    INTERVENTION_OUTPUT_SCHEMA_VERSION,
    MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
    BankOperationsProposal,
    InterventionSelectionOutput,
)

STRUCTURED_CALL_REQUEST_SCHEMA_VERSION: Literal["structured-call-request/v1"] = (
    "structured-call-request/v1"
)
STRUCTURED_CALL_RESULT_SCHEMA_VERSION: Literal["structured-call-result/v1"] = (
    "structured-call-result/v1"
)
STRUCTURED_CALL_USAGE_SCHEMA_VERSION: Literal["structured-call-usage/v1"] = (
    "structured-call-usage/v1"
)

MAX_STRUCTURED_CALL_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_STRUCTURED_CALL_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_STRUCTURED_CALL_PAYLOAD_NODES = 100_000
MAX_STRUCTURED_CALL_PAYLOAD_DEPTH = 64
COMPLETION_DIGEST_SCOPE: Literal["assistant-message-content-utf8/v1"] = (
    "assistant-message-content-utf8/v1"
)

_REQUEST_DIGEST_DOMAIN = "saliencegate:model:structured-call-request:v1"
_CALL_DIGEST_DOMAIN = "saliencegate:model:structured-call-result:v1"
_MAX_SIGNED_64 = (1 << 63) - 1

NonNegativeSigned64: TypeAlias = Annotated[int, Field(ge=0, le=_MAX_SIGNED_64)]
StructuredResponseSchemaVersion: TypeAlias = Literal[
    "memory-edit-output/v1",
    "intervention-output/v1",
]


class StructuredCallBoundaryError(ValueError):
    """A value-free failure at the structured-call boundary."""

    def __init__(self, operation: str) -> None:
        super().__init__(f"{operation} failed structured-call boundary validation")


class StructuredCallPhase(StrEnum):
    MEMORY_EDIT = "memory_edit"
    INTERVENTION = "intervention"


class StructuredCallStatus(StrEnum):
    COMPLETED = "completed"
    MODEL_ERROR = "model_error"
    MODEL_TIMEOUT = "model_timeout"


class StructuredCallParseStatus(StrEnum):
    VALID = "valid"
    SCHEMA_INVALID = "schema_invalid"
    EMPTY_REMINDER = "empty_reminder"
    CLAIM_OVER_LIMIT = "claim_over_limit"
    NOT_ATTEMPTED = "not_attempted"


class ProviderUsageProvenance(StrEnum):
    PROVIDER_REPORTED = "provider_reported"
    REPLAY_ATTESTED = "replay_attested"
    UNAVAILABLE = "unavailable"


class CanonicalUsageProvenance(StrEnum):
    LOCAL_COUNTER = "local_counter"
    REPLAY_ATTESTED = "replay_attested"
    UNAVAILABLE = "unavailable"


def _exclude_none(value: object) -> bool:
    return value is None


def _exclude_unavailable_canonical_usage(value: object) -> bool:
    return value is CanonicalUsageProvenance.UNAVAILABLE


class _StructuredCallModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def _phase_schema_matches(
    phase: StructuredCallPhase,
    response_schema_version: StructuredResponseSchemaVersion,
) -> bool:
    return (
        phase is StructuredCallPhase.MEMORY_EDIT
        and response_schema_version == MEMORY_EDIT_OUTPUT_SCHEMA_VERSION
    ) or (
        phase is StructuredCallPhase.INTERVENTION
        and response_schema_version == INTERVENTION_OUTPUT_SCHEMA_VERSION
    )


def _json_scalar_size(value: object) -> int | None:
    if value is not None and type(value) not in (str, bool, int, float):
        return None
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, UnicodeError, ValueError):
        return None
    return len(encoded)


def _json_input_is_bounded(value: object, *, max_bytes: int) -> bool:
    """Bound untrusted JSON iteratively before recursive Pydantic freezing."""

    try:
        if type(value) not in (dict, MappingProxyType):
            return False
        stack: list[tuple[object, int]] = [(value, 0)]
        nodes = 0
        encoded_bytes = 0
        while stack:
            item, depth = stack.pop()
            nodes += 1
            if (
                nodes > MAX_STRUCTURED_CALL_PAYLOAD_NODES
                or depth > MAX_STRUCTURED_CALL_PAYLOAD_DEPTH
            ):
                return False
            if type(item) in (dict, MappingProxyType):
                assert isinstance(item, (dict, MappingProxyType))
                declared_length = len(item)
                if declared_length > MAX_STRUCTURED_CALL_PAYLOAD_NODES - nodes - len(stack):
                    return False
                encoded_bytes += 2 + max(0, declared_length - 1)
                observed = 0
                for key, nested in item.items():
                    if observed >= declared_length or type(key) is not str:
                        return False
                    observed += 1
                    key_size = _json_scalar_size(key)
                    if key_size is None:
                        return False
                    encoded_bytes += key_size + 1
                    stack.append((nested, depth + 1))
                if observed != declared_length:
                    return False
            elif type(item) in (list, tuple):
                assert isinstance(item, (list, tuple))
                declared_length = len(item)
                if declared_length > MAX_STRUCTURED_CALL_PAYLOAD_NODES - nodes - len(stack):
                    return False
                encoded_bytes += 2 + max(0, declared_length - 1)
                stack.extend((nested, depth + 1) for nested in item)
            else:
                scalar_size = _json_scalar_size(item)
                if scalar_size is None:
                    return False
                encoded_bytes += scalar_size
            if encoded_bytes > max_bytes:
                return False
        return True
    except Exception:
        return False


def _request_digest_from_values(values: dict[str, object]) -> str:
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": values["schema_version"],
                "run_id": str(values["run_id"]),
                "cycle_id": values["cycle_id"],
                "model_call_index": values["model_call_index"],
                "phase": values["phase"],
                "attempt": values["attempt"],
                "model_id": values["model_id"],
                "prompt_template_id": values["prompt_template_id"],
                "prompt_template_digest": values["prompt_template_digest"],
                "response_schema_version": values["response_schema_version"],
                "payload": values["payload"],
            }
        ),
        domain=_REQUEST_DIGEST_DOMAIN,
    )


class StructuredCallRequest(_StructuredCallModel):
    """One content-bound request for one visible phase attempt.

    Attempt zero is the primary call; positive attempts are explicit schema repairs.
    """

    schema_version: Literal["structured-call-request/v1"]
    run_id: UUID4
    cycle_id: Sha256Digest
    model_call_index: NonNegativeSigned64
    phase: StructuredCallPhase
    attempt: NonNegativeSigned64
    model_id: ComponentIdentifier
    prompt_template_id: ComponentIdentifier
    prompt_template_digest: Sha256Digest
    response_schema_version: StructuredResponseSchemaVersion
    payload: JsonObject = Field(repr=False)
    request_digest: Sha256Digest = Field(default_factory=_request_digest_from_values)

    @field_validator("payload", mode="before")
    @classmethod
    def structurally_bound_payload(cls, value: object) -> object:
        if not _json_input_is_bounded(value, max_bytes=MAX_STRUCTURED_CALL_PAYLOAD_BYTES):
            raise ValueError("structured call payload exceeds its structural bound") from None
        return value

    @model_validator(mode="after")
    def phase_schema_and_digest_match(self) -> Self:
        if self.attempt > self.model_call_index:
            raise ValueError("structured call attempt cannot exceed call index")
        if self.phase is StructuredCallPhase.INTERVENTION and self.model_call_index == 0:
            raise ValueError("structured call intervention call index must be positive")
        if not _phase_schema_matches(self.phase, self.response_schema_version):
            raise ValueError("structured call phase response schema does not match")
        expected = _request_digest_from_values(
            cast(dict[str, object], self.model_dump(mode="python", exclude={"request_digest"}))
        )
        if self.request_digest != expected:
            raise ValueError("structured call request digest does not match")
        return self


class StructuredCallUsage(_StructuredCallModel):
    """Distinct provider and canonical token evidence for one call."""

    schema_version: Literal["structured-call-usage/v1"]
    provider_input_tokens: NonNegativeSigned64 | None
    provider_output_tokens: NonNegativeSigned64 | None
    provider_usage_provenance: ProviderUsageProvenance
    latency_us: NonNegativeSigned64
    canonical_input_tokens: NonNegativeSigned64 | None = Field(
        default=None,
        exclude_if=_exclude_none,
    )
    canonical_output_tokens: NonNegativeSigned64 | None = Field(
        default=None,
        exclude_if=_exclude_none,
    )
    canonical_usage_provenance: CanonicalUsageProvenance = Field(
        default=CanonicalUsageProvenance.UNAVAILABLE,
        exclude_if=_exclude_unavailable_canonical_usage,
    )
    local_counter_id: ComponentIdentifier | None = Field(
        default=None,
        exclude_if=_exclude_none,
    )
    local_counter_version: ComponentIdentifier | None = Field(
        default=None,
        exclude_if=_exclude_none,
    )
    local_counter_configuration_digest: Sha256Digest | None = Field(
        default=None,
        exclude_if=_exclude_none,
    )
    local_counter_model_id: ComponentIdentifier | None = Field(
        default=None,
        exclude_if=_exclude_none,
    )

    @model_validator(mode="after")
    def provenance_matches_provider_counts(self) -> Self:
        has_input = self.provider_input_tokens is not None
        has_output = self.provider_output_tokens is not None
        if has_input != has_output:
            raise ValueError("provider token counts must be both present or both unavailable")
        has_counts = has_input and has_output
        if self.provider_usage_provenance is ProviderUsageProvenance.UNAVAILABLE:
            if has_counts:
                raise ValueError("unavailable provider usage cannot carry token counts")
        elif not has_counts:
            raise ValueError("attested provider usage requires token counts")
        if (
            self.provider_input_tokens is not None
            and self.provider_output_tokens is not None
            and self.provider_input_tokens + self.provider_output_tokens > _MAX_SIGNED_64
        ):
            raise ValueError("provider token total exceeds its signed-64 limit")
        canonical_counts = (
            self.canonical_input_tokens,
            self.canonical_output_tokens,
        )
        counter_identity = (
            self.local_counter_id,
            self.local_counter_version,
            self.local_counter_configuration_digest,
            self.local_counter_model_id,
        )
        has_canonical_count = any(value is not None for value in canonical_counts)
        has_complete_counter_identity = all(value is not None for value in counter_identity)
        if self.canonical_usage_provenance is CanonicalUsageProvenance.UNAVAILABLE:
            if has_canonical_count or any(value is not None for value in counter_identity):
                raise ValueError("unavailable canonical usage cannot carry local counts")
        elif not has_canonical_count or not has_complete_counter_identity:
            raise ValueError("local canonical usage requires a complete counter identity")
        if (
            self.canonical_input_tokens is not None
            and self.canonical_output_tokens is not None
            and self.canonical_input_tokens + self.canonical_output_tokens > _MAX_SIGNED_64
        ):
            raise ValueError("canonical token total exceeds its signed-64 limit")
        return self

    @property
    def provider_tokens(self) -> int | None:
        if self.provider_input_tokens is None or self.provider_output_tokens is None:
            return None
        return self.provider_input_tokens + self.provider_output_tokens

    @property
    def canonical_tokens(self) -> int | None:
        if self.canonical_input_tokens is None or self.canonical_output_tokens is None:
            return None
        return self.canonical_input_tokens + self.canonical_output_tokens


StructuredPhaseOutput: TypeAlias = Annotated[
    BankOperationsProposal | InterventionSelectionOutput,
    Field(discriminator="schema_version"),
]

_OUTPUT_SCHEMA_TYPES: dict[
    str,
    type[BankOperationsProposal] | type[InterventionSelectionOutput],
] = {
    MEMORY_EDIT_OUTPUT_SCHEMA_VERSION: BankOperationsProposal,
    INTERVENTION_OUTPUT_SCHEMA_VERSION: InterventionSelectionOutput,
}


def _model_json(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", warnings=False)
    return value


def _call_digest_from_values(values: dict[str, object]) -> str:
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": values["schema_version"],
                "request_digest": values["request_digest"],
                "model_call_index": values["model_call_index"],
                "phase": values["phase"],
                "attempt": values["attempt"],
                "response_schema_version": values["response_schema_version"],
                "status": values["status"],
                "parse_status": values["parse_status"],
                "output": _model_json(values.get("output")),
                "completion_digest": values.get("completion_digest"),
                "completion_byte_count": values.get("completion_byte_count"),
                "usage": _model_json(values["usage"]),
            }
        ),
        domain=_CALL_DIGEST_DOMAIN,
    )


class StructuredCallResult(_StructuredCallModel):
    """A replay-safe call result with no raw completion or provider error text.

    ``completion_digest`` attests only the exact UTF-8 bytes of assistant
    ``message.content`` under ``COMPLETION_DIGEST_SCOPE``. It never covers the provider
    envelope or reasoning. Live clients use HMAC-SHA256; explicit synthetic replay may
    use the tagged synthetic algorithm.
    """

    schema_version: Literal["structured-call-result/v1"]
    request_digest: Sha256Digest
    model_call_index: NonNegativeSigned64
    phase: StructuredCallPhase
    attempt: NonNegativeSigned64
    response_schema_version: StructuredResponseSchemaVersion
    status: StructuredCallStatus
    parse_status: StructuredCallParseStatus
    output: StructuredPhaseOutput | None = Field(repr=False)
    completion_digest: PayloadDigest | None = Field(repr=False)
    completion_byte_count: Annotated[
        int | None,
        Field(ge=0, le=MAX_STRUCTURED_CALL_OUTPUT_BYTES),
    ]
    usage: StructuredCallUsage
    call_digest: Sha256Digest = Field(default_factory=_call_digest_from_values)

    @field_validator("output", mode="before")
    @classmethod
    def prevalidate_output_schema(cls, value: object) -> object:
        if value is None or type(value) in (BankOperationsProposal, InterventionSelectionOutput):
            return value
        if type(value) is not dict or not _json_input_is_bounded(
            value,
            max_bytes=MAX_STRUCTURED_CALL_OUTPUT_BYTES,
        ):
            raise ValueError("structured call output failed validation") from None
        tag = value.get("schema_version")
        if type(tag) is not str or tag not in _OUTPUT_SCHEMA_TYPES:
            raise ValueError("structured call output schema tag failed validation") from None
        try:
            output_type = _OUTPUT_SCHEMA_TYPES[tag]
            return output_type.model_validate_json(canonical_json(value))
        except Exception:
            raise ValueError("structured call output failed validation") from None

    @field_validator("output")
    @classmethod
    def bound_canonical_output(
        cls,
        value: StructuredPhaseOutput | None,
    ) -> StructuredPhaseOutput | None:
        if value is not None and len(canonical_json(value)) > MAX_STRUCTURED_CALL_OUTPUT_BYTES:
            raise ValueError("structured call output exceeds its canonical byte limit")
        return value

    @field_validator("completion_digest")
    @classmethod
    def revalidate_completion_digest(
        cls,
        value: PayloadDigest | None,
    ) -> PayloadDigest | None:
        if value is None:
            return None
        if type(value) is not PayloadDigest:
            raise ValueError("structured call completion digest failed validation") from None
        try:
            return PayloadDigest.model_validate_json(value.model_dump_json(warnings=False))
        except Exception:
            raise ValueError("structured call completion digest failed validation") from None

    @model_validator(mode="after")
    def state_phase_and_digest_match(self) -> Self:
        if self.attempt > self.model_call_index:
            raise ValueError("structured call attempt cannot exceed call index")
        if self.phase is StructuredCallPhase.INTERVENTION and self.model_call_index == 0:
            raise ValueError("structured call intervention call index must be positive")
        if not _phase_schema_matches(self.phase, self.response_schema_version):
            raise ValueError("structured call phase response schema does not match")

        intervention_rejection = self.parse_status in (
            StructuredCallParseStatus.EMPTY_REMINDER,
            StructuredCallParseStatus.CLAIM_OVER_LIMIT,
        )
        if intervention_rejection and self.phase is not StructuredCallPhase.INTERVENTION:
            raise ValueError("structured call parse status does not match phase")

        has_output = self.output is not None
        has_completion_digest = self.completion_digest is not None
        has_completion_count = self.completion_byte_count is not None
        if self.status is StructuredCallStatus.COMPLETED:
            valid_state = (
                self.parse_status is StructuredCallParseStatus.VALID
                and has_output
                and has_completion_digest
                and has_completion_count
                and self.completion_byte_count is not None
                and self.completion_byte_count > 0
            ) or (
                self.parse_status
                in (
                    StructuredCallParseStatus.SCHEMA_INVALID,
                    StructuredCallParseStatus.EMPTY_REMINDER,
                    StructuredCallParseStatus.CLAIM_OVER_LIMIT,
                )
                and not has_output
                and has_completion_digest
                and has_completion_count
            )
        else:
            valid_state = (
                self.parse_status is StructuredCallParseStatus.NOT_ATTEMPTED
                and not has_output
                and not has_completion_digest
                and not has_completion_count
                and self.usage.canonical_output_tokens is None
            )
        if not valid_state:
            raise ValueError("structured call result state is inconsistent")

        if self.output is not None:
            output_matches = (
                self.phase is StructuredCallPhase.MEMORY_EDIT
                and type(self.output) is BankOperationsProposal
            ) or (
                self.phase is StructuredCallPhase.INTERVENTION
                and type(self.output) is InterventionSelectionOutput
            )
            if not output_matches:
                raise ValueError("structured call output does not match phase")

        expected = _call_digest_from_values(
            cast(dict[str, object], self.model_dump(mode="json", exclude={"call_digest"}))
        )
        if self.call_digest != expected:
            raise ValueError("structured call call digest does not match")
        return self


@runtime_checkable
class StructuredCallClient(Protocol):
    async def generate(self, request: StructuredCallRequest) -> StructuredCallResult: ...


def validated_structured_call_request(value: object) -> StructuredCallRequest:
    """Revalidate an exact request without retaining its content in an error."""

    if type(value) is not StructuredCallRequest:
        raise StructuredCallBoundaryError("structured call request")
    try:
        return StructuredCallRequest.model_validate_json(value.model_dump_json(warnings=False))
    except Exception:
        raise StructuredCallBoundaryError("structured call request") from None


def validated_structured_call_result(value: object) -> StructuredCallResult:
    """Revalidate an exact result and every recursively typed value."""

    if type(value) is not StructuredCallResult:
        raise StructuredCallBoundaryError("structured call result")
    try:
        return StructuredCallResult.model_validate_json(value.model_dump_json(warnings=False))
    except Exception:
        raise StructuredCallBoundaryError("structured call result") from None


def validated_result_for_request(
    request: object,
    result: object,
) -> StructuredCallResult:
    """Return a revalidated result only when it belongs to the exact request."""

    try:
        checked_request = validated_structured_call_request(request)
        checked_result = validated_structured_call_result(result)
        if (
            checked_result.request_digest != checked_request.request_digest
            or checked_result.model_call_index != checked_request.model_call_index
            or checked_result.phase is not checked_request.phase
            or checked_result.attempt != checked_request.attempt
            or checked_result.response_schema_version != checked_request.response_schema_version
            or (
                checked_result.usage.local_counter_model_id is not None
                and checked_result.usage.local_counter_model_id != checked_request.model_id
            )
        ):
            raise ValueError
        return checked_result
    except Exception:
        raise StructuredCallBoundaryError("structured call result for request") from None


__all__ = [
    "COMPLETION_DIGEST_SCOPE",
    "INTERVENTION_OUTPUT_SCHEMA_VERSION",
    "MAX_STRUCTURED_CALL_OUTPUT_BYTES",
    "MAX_STRUCTURED_CALL_PAYLOAD_BYTES",
    "MAX_STRUCTURED_CALL_PAYLOAD_DEPTH",
    "MAX_STRUCTURED_CALL_PAYLOAD_NODES",
    "MEMORY_EDIT_OUTPUT_SCHEMA_VERSION",
    "STRUCTURED_CALL_REQUEST_SCHEMA_VERSION",
    "STRUCTURED_CALL_RESULT_SCHEMA_VERSION",
    "STRUCTURED_CALL_USAGE_SCHEMA_VERSION",
    "CanonicalUsageProvenance",
    "ProviderUsageProvenance",
    "StructuredCallBoundaryError",
    "StructuredCallClient",
    "StructuredCallParseStatus",
    "StructuredCallPhase",
    "StructuredCallRequest",
    "StructuredCallResult",
    "StructuredCallStatus",
    "StructuredCallUsage",
    "StructuredPhaseOutput",
    "StructuredResponseSchemaVersion",
    "validated_result_for_request",
    "validated_structured_call_request",
    "validated_structured_call_result",
]
