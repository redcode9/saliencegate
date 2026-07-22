"""Strict, bounded contracts for pseudonymized capture records."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Annotated, Literal, Never, Self, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from saliencegate.domain import canonical_json
from saliencegate.domain.primitives import ComponentIdentifier, Sha256Digest, UtcDatetime

MAX_CAPTURE_EVENT_BYTES = 64 * 1_024
MAX_CAPTURE_NATIVE_BYTES = 2 * 1_024 * 1_024
MAX_CAPTURE_JSON_DEPTH = 32
MAX_CAPTURE_JSON_ITEMS = 10_000
MAX_CAPTURE_JSON_STRING_BYTES = 1 * 1_024 * 1_024


class CaptureSchemaError(ValueError):
    """A content-free failure at the capture schema boundary."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture schema is invalid")


@dataclass(frozen=True, slots=True)
class CaptureJSONLimits:
    """Bounds applied before native JSON reaches a provider adapter."""

    max_bytes: int
    max_depth: int
    max_items: int
    max_string_bytes: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (
                self.max_bytes,
                self.max_depth,
                self.max_items,
                self.max_string_bytes,
            )
        ):
            raise CaptureSchemaError()


_CAPTURE_EVENT_LIMITS = CaptureJSONLimits(
    max_bytes=MAX_CAPTURE_EVENT_BYTES,
    max_depth=MAX_CAPTURE_JSON_DEPTH,
    max_items=MAX_CAPTURE_JSON_ITEMS,
    max_string_bytes=MAX_CAPTURE_JSON_STRING_BYTES,
)
CAPTURE_NATIVE_JSON_LIMITS = CaptureJSONLimits(
    max_bytes=MAX_CAPTURE_NATIVE_BYTES,
    max_depth=MAX_CAPTURE_JSON_DEPTH,
    max_items=MAX_CAPTURE_JSON_ITEMS,
    max_string_bytes=MAX_CAPTURE_JSON_STRING_BYTES,
)


@lru_cache(maxsize=16)
def _validate_profile_binding(adapter_profile: str, manifest_digest: str) -> None:
    try:
        from saliencegate.capture.capabilities import (
            CaptureProfile,
            validate_capture_capability_binding,
        )

        validate_capture_capability_binding(
            CaptureProfile(adapter_profile),
            manifest_digest,
        )
    except Exception:
        raise ValueError("capture capability binding is invalid") from None


class _CaptureModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<redacted>)"

    __str__ = __repr__


class _CaptureIntakeBase(_CaptureModel):
    schema_version: Literal["capture-intake/v1"] = "capture-intake/v1"
    adapter_profile: Annotated[
        Literal[
            "codex-hooks/v1",
            "claude-code-hooks/v1",
            "opencode-plugin/v1",
            "pi-extension/v1",
        ],
        Field(repr=False),
    ]
    capability_manifest_digest: Annotated[Sha256Digest, Field(repr=False)]
    connection_id: Annotated[ComponentIdentifier, Field(repr=False)]
    session_id: Annotated[Sha256Digest, Field(repr=False)]
    producer_event_digest: Annotated[Sha256Digest, Field(repr=False)]
    intake_tag: Annotated[Sha256Digest, Field(repr=False)]
    occurred_at: Annotated[UtcDatetime | None, Field(repr=False)] = None
    timestamp_authority: Literal["local_observation", "unavailable"] = "unavailable"
    producer_sequence: Annotated[int | None, Field(ge=0, le=(1 << 63) - 1)] = None
    sequence_authority: Literal["producer_exact", "unavailable"] = "unavailable"
    capture_disposition: Literal[
        "captured",
        "coverage_boundary",
        "degraded",
    ] = "captured"

    @field_validator("connection_id", mode="before")
    @classmethod
    def require_exact_identifiers(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("capture identifier is invalid")
        return value

    @model_validator(mode="after")
    def authority_is_consistent(self) -> Self:
        if (self.occurred_at is None) != (self.timestamp_authority == "unavailable"):
            raise ValueError("capture timestamp authority is inconsistent")
        if (self.producer_sequence is None) != (self.sequence_authority == "unavailable"):
            raise ValueError("capture sequence authority is inconsistent")
        _validate_profile_binding(
            self.adapter_profile,
            self.capability_manifest_digest,
        )
        return self


class CaptureSessionStartedIntake(_CaptureIntakeBase):
    kind: Literal["session_started"] = "session_started"


class CaptureActionStartedIntake(_CaptureIntakeBase):
    kind: Literal["action_started"] = "action_started"
    call_ref: Annotated[Sha256Digest, Field(repr=False)]
    action_digest: Annotated[Sha256Digest, Field(repr=False)]
    workspace_digest: Annotated[Sha256Digest, Field(repr=False)]
    environment_digest: Annotated[Sha256Digest, Field(repr=False)]
    tool_class: Literal[
        "shell",
        "file_read",
        "file_write",
        "search",
        "network",
        "subagent",
        "other",
    ]
    identity_authority: Literal["exact", "coarse", "unavailable"]


class CaptureActionFinishedIntake(_CaptureIntakeBase):
    kind: Literal["action_finished"] = "action_finished"
    call_ref: Annotated[Sha256Digest, Field(repr=False)]
    outcome_status: Literal["succeeded", "failed"] | None
    outcome_authority: Literal["producer_claimed_structured", "unavailable"]
    exit_status: Annotated[int | None, Field(ge=-(1 << 31), le=(1 << 31) - 1)] = None
    error_code: (
        Literal[
            "tool_error",
            "permission_denied",
            "interrupted",
            "timeout",
            "provider_error",
        ]
        | None
    ) = None
    failure_signature: Annotated[Sha256Digest | None, Field(repr=False)] = None

    @model_validator(mode="after")
    def outcome_is_consistent(self) -> Self:
        detail_present = any(
            value is not None
            for value in (self.exit_status, self.error_code, self.failure_signature)
        )
        if self.outcome_authority == "unavailable":
            if self.outcome_status is not None or detail_present:
                raise ValueError("unavailable capture outcome carries evidence")
        elif self.outcome_status is None:
            raise ValueError("structured capture outcome has no status")
        if self.outcome_status == "succeeded" and (
            self.exit_status not in (None, 0)
            or self.error_code is not None
            or self.failure_signature is not None
        ):
            raise ValueError("successful capture outcome carries failure evidence")
        return self


class CapturePermissionDeniedIntake(_CaptureIntakeBase):
    kind: Literal["permission_denied"] = "permission_denied"
    call_ref: Annotated[Sha256Digest, Field(repr=False)]


class CaptureSubagentStartedIntake(_CaptureIntakeBase):
    kind: Literal["subagent_started"] = "subagent_started"
    subagent_id: Annotated[Sha256Digest, Field(repr=False)]


class CaptureSubagentFinishedIntake(_CaptureIntakeBase):
    kind: Literal["subagent_finished"] = "subagent_finished"
    subagent_id: Annotated[Sha256Digest, Field(repr=False)]


class CaptureTurnFinishedIntake(_CaptureIntakeBase):
    kind: Literal["turn_finished"] = "turn_finished"
    turn_id: Annotated[Sha256Digest, Field(repr=False)]


class CaptureControllerFailedIntake(_CaptureIntakeBase):
    kind: Literal["controller_failed"] = "controller_failed"
    error_code: Literal[
        "provider_callback_failed",
        "invalid_transition",
        "spawn_failed",
        "timeout",
        "overflow",
        "gap_detected",
    ]
    failure_signature: Annotated[Sha256Digest | None, Field(repr=False)] = None


class CaptureSessionFinishedIntake(_CaptureIntakeBase):
    kind: Literal["session_finished"] = "session_finished"


CaptureIntake: TypeAlias = (
    CaptureSessionStartedIntake
    | CaptureActionStartedIntake
    | CaptureActionFinishedIntake
    | CapturePermissionDeniedIntake
    | CaptureSubagentStartedIntake
    | CaptureSubagentFinishedIntake
    | CaptureTurnFinishedIntake
    | CaptureControllerFailedIntake
    | CaptureSessionFinishedIntake
)
_DiscriminatedCaptureIntake: TypeAlias = Annotated[
    CaptureIntake,
    Field(discriminator="kind"),
]

_CAPTURE_INTAKE_ADAPTER: TypeAdapter[CaptureIntake] = TypeAdapter(_DiscriminatedCaptureIntake)


class CaptureEvent(_CaptureModel):
    schema_version: Literal["capture-event/v1"] = "capture-event/v1"
    receipt_ordinal: Annotated[int, Field(ge=1, le=(1 << 63) - 1)]
    previous_event_tag: Annotated[Sha256Digest | None, Field(repr=False)] = None
    event_tag: Annotated[Sha256Digest, Field(repr=False)]
    intake: Annotated[_DiscriminatedCaptureIntake, Field(repr=False)]

    @model_validator(mode="after")
    def chain_position_is_consistent(self) -> Self:
        if (self.receipt_ordinal == 1) != (self.previous_event_tag is None):
            raise ValueError("capture event chain position is inconsistent")
        return self


def _reject_constant(_value: str) -> Never:
    raise ValueError("non-finite JSON number")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _bounded_json(value: object, *, limits: CaptureJSONLimits) -> bool:
    try:
        items = 0
        string_bytes = 0
        stack: list[tuple[object, int]] = [(value, 0)]
        while stack:
            item, depth = stack.pop()
            items += 1
            if items > limits.max_items or depth > limits.max_depth:
                return False
            if type(item) is dict:
                assert isinstance(item, dict)
                if len(item) > limits.max_items - items - len(stack):
                    return False
                for key, nested in item.items():
                    if type(key) is not str:
                        return False
                    string_bytes += len(key.encode("utf-8", errors="strict"))
                    stack.append((nested, depth + 1))
            elif type(item) is list:
                assert isinstance(item, list)
                if len(item) > limits.max_items - items - len(stack):
                    return False
                stack.extend((nested, depth + 1) for nested in item)
            elif type(item) is str:
                string_bytes += len(item.encode("utf-8", errors="strict"))
            elif item is None or type(item) in (bool, int):
                pass
            elif type(item) is float:
                if not math.isfinite(item):
                    return False
            else:
                return False
            if string_bytes > limits.max_string_bytes:
                return False
        return True
    except Exception:
        return False


def _freeze_json(value: object) -> object:
    if type(value) is dict:
        assert isinstance(value, dict)
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if type(value) is list:
        assert isinstance(value, list)
        return tuple(_freeze_json(item) for item in value)
    return value


def read_bounded_json(value: object, *, limits: CaptureJSONLimits) -> Mapping[str, object]:
    """Decode duplicate-safe JSON into recursively immutable values."""

    if type(value) is not bytes or not 1 <= len(value) <= limits.max_bytes:
        raise CaptureSchemaError()
    try:
        decoded = json.loads(
            value.decode("utf-8", errors="strict"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
        if type(decoded) is not dict or not _bounded_json(decoded, limits=limits):
            raise ValueError("capture JSON is invalid")
        frozen = _freeze_json(decoded)
        assert isinstance(frozen, Mapping)
        return cast(Mapping[str, object], frozen)
    except CaptureSchemaError:
        raise
    except Exception:
        raise CaptureSchemaError() from None


def validate_capture_intake(value: object) -> CaptureIntake:
    try:
        return _CAPTURE_INTAKE_ADAPTER.validate_python(value)
    except Exception:
        raise CaptureSchemaError() from None


def canonical_capture_intake(value: CaptureIntake) -> bytes:
    try:
        validated = validate_capture_intake(value)
        encoded = canonical_json(validated)
        read_bounded_json(encoded, limits=_CAPTURE_EVENT_LIMITS)
        return encoded
    except CaptureSchemaError:
        raise
    except Exception:
        raise CaptureSchemaError() from None


def load_capture_intake(value: bytes) -> CaptureIntake:
    read_bounded_json(value, limits=_CAPTURE_EVENT_LIMITS)
    try:
        intake = _CAPTURE_INTAKE_ADAPTER.validate_json(value)
        if canonical_capture_intake(intake) != value:
            raise CaptureSchemaError()
        return intake
    except CaptureSchemaError:
        raise
    except Exception:
        raise CaptureSchemaError() from None


def validate_capture_event(value: object) -> CaptureEvent:
    try:
        return CaptureEvent.model_validate(value)
    except Exception:
        raise CaptureSchemaError() from None


def canonical_capture_event(value: CaptureEvent) -> bytes:
    try:
        validated = validate_capture_event(value)
        encoded = canonical_json(validated)
        read_bounded_json(encoded, limits=_CAPTURE_EVENT_LIMITS)
        return encoded
    except CaptureSchemaError:
        raise
    except Exception:
        raise CaptureSchemaError() from None


def _read_canonical_capture_event_document(value: bytes) -> Mapping[str, object]:
    """Return one bounded event document only when its stored bytes are canonical."""

    document = read_bounded_json(value, limits=_CAPTURE_EVENT_LIMITS)
    try:
        if canonical_json(document) != value:
            raise CaptureSchemaError()
        return document
    except CaptureSchemaError:
        raise
    except Exception:
        raise CaptureSchemaError() from None


def load_capture_event(value: bytes) -> CaptureEvent:
    read_bounded_json(value, limits=_CAPTURE_EVENT_LIMITS)
    try:
        event = CaptureEvent.model_validate_json(value)
        # The raw document was already bounded above; this comparison rejects both
        # non-canonical bytes and values that strict schema validation normalizes.
        if canonical_json(event) != value:
            raise CaptureSchemaError()
        return event
    except CaptureSchemaError:
        raise
    except Exception:
        raise CaptureSchemaError() from None


__all__ = [
    "CAPTURE_NATIVE_JSON_LIMITS",
    "MAX_CAPTURE_EVENT_BYTES",
    "MAX_CAPTURE_JSON_DEPTH",
    "MAX_CAPTURE_JSON_ITEMS",
    "MAX_CAPTURE_JSON_STRING_BYTES",
    "MAX_CAPTURE_NATIVE_BYTES",
    "CaptureActionFinishedIntake",
    "CaptureActionStartedIntake",
    "CaptureControllerFailedIntake",
    "CaptureEvent",
    "CaptureIntake",
    "CaptureJSONLimits",
    "CapturePermissionDeniedIntake",
    "CaptureSchemaError",
    "CaptureSessionFinishedIntake",
    "CaptureSessionStartedIntake",
    "CaptureSubagentFinishedIntake",
    "CaptureSubagentStartedIntake",
    "CaptureTurnFinishedIntake",
    "canonical_capture_event",
    "canonical_capture_intake",
    "load_capture_event",
    "load_capture_intake",
    "read_bounded_json",
    "validate_capture_event",
    "validate_capture_intake",
]
