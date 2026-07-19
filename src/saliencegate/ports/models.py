from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Protocol, Self, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from saliencegate.domain import JsonObject, canonical_json, length_prefixed_sha256
from saliencegate.domain.records import (
    UUID4,
    ComponentIdentifier,
    NonNegativeInt,
    Sha256Digest,
)
from saliencegate.ports.memory import MemoryCycleOutput

MODEL_REQUEST_SCHEMA_VERSION: Literal["model-request/v1"] = "model-request/v1"
MODEL_RESULT_SCHEMA_VERSION: Literal["model-result/v1"] = "model-result/v1"
MEMORY_CYCLE_OUTPUT_SCHEMA_VERSION: Literal["memory-cycle-output/v1"] = "memory-cycle-output/v1"

_REQUEST_DIGEST_DOMAIN = "saliencegate:model:request:v1"
_CALL_DIGEST_DOMAIN = "saliencegate:model:call:v1"
MAX_MODEL_REQUEST_PAYLOAD_BYTES = 16 * 1024 * 1024


class ModelBoundaryError(ValueError):
    """A value-free model-boundary validation failure."""

    def __init__(self, operation: str) -> None:
        super().__init__(f"{operation} failed model-boundary validation")


class ModelCallStatus(StrEnum):
    COMPLETED = "completed"
    MODEL_ERROR = "model_error"
    MODEL_TIMEOUT = "model_timeout"


class _Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def _request_digest_from_values(values: dict[str, object]) -> str:
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": values["schema_version"],
                "run_id": str(values["run_id"]),
                "cycle_id": values["cycle_id"],
                "model_call_index": values["model_call_index"],
                "model_id": values["model_id"],
                "prompt_template_digest": values["prompt_template_digest"],
                "response_schema_version": values["response_schema_version"],
                "payload": values["payload"],
            }
        ),
        domain=_REQUEST_DIGEST_DOMAIN,
    )


class ModelRequest(_Model):
    """A content-bound structured-model request.

    ``payload`` is expected to contain only repository-redacted, normalized data. It is
    intentionally omitted from representations. The digest is computed when omitted and
    verified whenever a caller supplies or deserializes one.
    """

    schema_version: Literal["model-request/v1"] = MODEL_REQUEST_SCHEMA_VERSION
    run_id: UUID4
    cycle_id: Sha256Digest
    model_call_index: Annotated[int, Field(ge=0, le=(1 << 63) - 1)]
    model_id: ComponentIdentifier
    prompt_template_digest: Sha256Digest
    response_schema_version: Literal["memory-cycle-output/v1"] = MEMORY_CYCLE_OUTPUT_SCHEMA_VERSION
    payload: JsonObject = Field(repr=False)
    request_digest: Sha256Digest = Field(default_factory=_request_digest_from_values)

    @field_validator("payload")
    @classmethod
    def bound_canonical_payload(cls, value: JsonObject) -> JsonObject:
        if len(canonical_json(value)) > MAX_MODEL_REQUEST_PAYLOAD_BYTES:
            raise ValueError("model request payload exceeds its canonical byte limit")
        return value

    @model_validator(mode="after")
    def verify_request_digest(self) -> Self:
        expected = _request_digest_from_values(
            cast(dict[str, object], self.model_dump(mode="python", exclude={"request_digest"}))
        )
        if self.request_digest != expected:
            raise ValueError("model request digest does not match")
        return self


class ModelUsage(_Model):
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    canonical_token_equivalents: NonNegativeInt = 0
    latency_us: NonNegativeInt = 0
    schema_repairs: NonNegativeInt = 0


def _call_digest_from_values(values: dict[str, object]) -> str:
    output = values.get("output")
    if isinstance(output, BaseModel):
        output = output.model_dump(mode="json", warnings=False)
    usage = values["usage"]
    if isinstance(usage, BaseModel):
        usage = usage.model_dump(mode="json", warnings=False)
    status = values["status"]
    if isinstance(status, ModelCallStatus):
        status = status.value
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": values["schema_version"],
                "status": status,
                "request_digest": values["request_digest"],
                "output": output,
                "usage": usage,
            }
        ),
        domain=_CALL_DIGEST_DOMAIN,
    )


class ModelResult(_Model):
    """A replay-safe model result without provider error text or raw responses."""

    schema_version: Literal["model-result/v1"] = MODEL_RESULT_SCHEMA_VERSION
    status: ModelCallStatus
    request_digest: Sha256Digest
    output: MemoryCycleOutput | None = Field(default=None, repr=False)
    usage: ModelUsage
    call_digest: Sha256Digest = Field(default_factory=_call_digest_from_values)

    @model_validator(mode="after")
    def status_and_digest_match(self) -> Self:
        if (self.status is ModelCallStatus.COMPLETED) != (self.output is not None):
            raise ValueError("only a completed model call can carry structured output")
        expected = _call_digest_from_values(
            cast(dict[str, object], self.model_dump(mode="json", exclude={"call_digest"}))
        )
        if self.call_digest != expected:
            raise ValueError("model call digest does not match")
        return self


@runtime_checkable
class StructuredModel(Protocol):
    async def generate(self, request: ModelRequest) -> ModelResult: ...


def validated_model_request(value: object) -> ModelRequest:
    """Revalidate a caller-created request without retaining its input in an error."""

    if type(value) is not ModelRequest:
        raise ModelBoundaryError("model request")
    try:
        candidate = ModelRequest.model_validate_json(value.model_dump_json(warnings=False))
    except Exception:
        raise ModelBoundaryError("model request") from None
    return candidate


def validated_model_result(value: object) -> ModelResult:
    """Revalidate a model result and all of its recursively typed output."""

    if type(value) is not ModelResult:
        raise ModelBoundaryError("model result")
    try:
        candidate = ModelResult.model_validate_json(value.model_dump_json(warnings=False))
    except Exception:
        raise ModelBoundaryError("model result") from None
    return candidate


__all__ = [
    "MAX_MODEL_REQUEST_PAYLOAD_BYTES",
    "MEMORY_CYCLE_OUTPUT_SCHEMA_VERSION",
    "MODEL_REQUEST_SCHEMA_VERSION",
    "MODEL_RESULT_SCHEMA_VERSION",
    "ModelBoundaryError",
    "ModelCallStatus",
    "ModelRequest",
    "ModelResult",
    "ModelUsage",
    "StructuredModel",
    "validated_model_request",
    "validated_model_result",
]
