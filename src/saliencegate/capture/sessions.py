"""Authenticated, point-in-time capture session snapshots."""

from __future__ import annotations

import hmac
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from saliencegate.capture.capabilities import CaptureProfile, CompatibilityStatus
from saliencegate.capture.health import CaptureHealthCode
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.capture.schema import CaptureEvent
from saliencegate.capture.store import (
    MAX_CAPTURE_EVENTS_PER_SESSION,
    CaptureAdmissionSource,
    CaptureConnectionState,
    CaptureSessionState,
)
from saliencegate.domain import canonical_json
from saliencegate.domain.records import (
    ComponentIdentifier,
    Sha256Digest,
    UtcDatetime,
)
from saliencegate.security import InstallationKey


class CaptureSessionSnapshotError(ValueError):
    """A snapshot failed validation or key-bound authentication."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture session snapshot is invalid")


def _require_exact_human_id(value: str) -> str:
    if type(value) is not str:
        raise ValueError("capture human identifier is invalid")
    return value


CaptureHumanSessionId = Annotated[
    str,
    StringConstraints(min_length=12, max_length=52, pattern=r"^[a-z2-7]+$"),
    AfterValidator(_require_exact_human_id),
]


class _CaptureSnapshotModel(BaseModel):
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


class CaptureSnapshotEvent(_CaptureSnapshotModel):
    """One authenticated capture event plus its store admission metadata."""

    receipt_ordinal: Annotated[
        int,
        Field(ge=1, le=MAX_CAPTURE_EVENTS_PER_SESSION),
    ]
    admission_source: CaptureAdmissionSource
    admitted_at: Annotated[UtcDatetime, Field(repr=False)]
    event: Annotated[CaptureEvent, Field(repr=False)]

    @model_validator(mode="after")
    def wrapper_matches_event(self) -> Self:
        if self.receipt_ordinal != self.event.receipt_ordinal:
            raise ValueError("capture snapshot event position is inconsistent")
        return self


class CaptureSnapshotHealth(_CaptureSnapshotModel):
    """One authenticated, content-free session health counter."""

    code: CaptureHealthCode
    count: Annotated[int, Field(ge=1)]
    lower_bound: Annotated[int, Field(ge=0, le=1)]
    created_at: Annotated[UtcDatetime, Field(repr=False)]
    updated_at: Annotated[UtcDatetime, Field(repr=False)]


class CaptureSessionSnapshot(_CaptureSnapshotModel):
    """A fully verified MVCC view of one capture session."""

    schema_version: Literal["capture-session-snapshot/v1"] = "capture-session-snapshot/v1"
    connection_id: Annotated[ComponentIdentifier, Field(repr=False)]
    project_digest: Annotated[Sha256Digest, Field(repr=False)]
    profile_id: CaptureProfile
    capability_manifest_digest: Annotated[Sha256Digest, Field(repr=False)]
    host_version: Annotated[ComponentIdentifier, Field(repr=False)]
    compatibility_status: CompatibilityStatus
    connection_state: CaptureConnectionState
    session_id: Annotated[Sha256Digest, Field(repr=False)]
    human_id: Annotated[CaptureHumanSessionId, Field(repr=False)]
    state: CaptureSessionState
    event_count: Annotated[
        int,
        Field(ge=0, le=MAX_CAPTURE_EVENTS_PER_SESSION),
    ]
    coverage_degraded: bool
    unattributed_drop: bool
    opened_at: Annotated[UtcDatetime, Field(repr=False)]
    updated_at: Annotated[UtcDatetime, Field(repr=False)]
    closed_at: Annotated[UtcDatetime | None, Field(repr=False)]
    events: Annotated[
        tuple[CaptureSnapshotEvent, ...],
        Field(max_length=MAX_CAPTURE_EVENTS_PER_SESSION, repr=False),
    ]
    health: Annotated[
        tuple[CaptureSnapshotHealth, ...],
        Field(max_length=len(CaptureHealthCode), repr=False),
    ]
    spool_observation: Literal["not_observed_by_store_snapshot"] = "not_observed_by_store_snapshot"
    at_rest_integrity: Literal["hmac_sha256_local_mutation_detection"] = (
        "hmac_sha256_local_mutation_detection"
    )
    rollback_detection: Literal["none"] = "none"
    spool_boundary_digest: Annotated[Sha256Digest | None, Field(repr=False)]
    snapshot_digest: Annotated[Sha256Digest, Field(repr=False)]

    @model_validator(mode="after")
    def commitments_are_consistent(self) -> Self:
        final_kind = None if not self.events else self.events[-1].event.intake.kind
        has_session_finished = any(
            item.event.intake.kind == "session_finished" for item in self.events
        )
        if (
            self.connection_state is CaptureConnectionState.DELETING
            or self.state is CaptureSessionState.DELETING
            or self.compatibility_status is CompatibilityStatus.INCOMPATIBLE
            or self.event_count != len(self.events)
            or ((self.state is CaptureSessionState.CLOSED) != (self.closed_at is not None))
            or (self.closed_at is not None and self.closed_at != self.updated_at)
            or (self.state is CaptureSessionState.CLOSED and final_kind != "session_finished")
            or (self.state is CaptureSessionState.OPEN and has_session_finished)
            or ((bool(self.health) or self.unattributed_drop) and not self.coverage_degraded)
        ):
            raise ValueError("capture session snapshot commitments are inconsistent")
        previous_tag: str | None = None
        for ordinal, item in enumerate(self.events, start=1):
            intake = item.event.intake
            if (
                item.receipt_ordinal != ordinal
                or item.event.previous_event_tag != previous_tag
                or intake.connection_id != self.connection_id
                or intake.session_id != self.session_id
                or intake.adapter_profile != self.profile_id.value
                or intake.capability_manifest_digest != self.capability_manifest_digest
            ):
                raise ValueError("capture session snapshot event chain is inconsistent")
            previous_tag = item.event.event_tag
        health_codes = tuple(item.code.value for item in self.health)
        if health_codes != tuple(sorted(set(health_codes))):
            raise ValueError("capture session snapshot health is not canonical")
        return self


def _snapshot_preimage(snapshot: CaptureSessionSnapshot) -> bytes:
    return canonical_json(
        {
            "schema_version": "capture-session-snapshot-integrity/v1",
            "snapshot": snapshot.model_dump(
                mode="json",
                exclude={"snapshot_digest"},
                warnings="error",
            ),
        }
    )


def _validated_snapshot(value: object) -> CaptureSessionSnapshot:
    # A Python-mode dump forces a recursive defensive copy even when the caller
    # supplied an existing (nominally frozen) Pydantic instance.
    if type(value) is CaptureSessionSnapshot:
        value = value.model_dump(mode="python", warnings="error")
    return CaptureSessionSnapshot.model_validate(value)


def _authenticate_capture_session_snapshot(
    snapshot: object,
    *,
    context: CaptureDigestContext,
) -> CaptureSessionSnapshot:
    """Internal signer used only after store rows have been authenticated."""

    try:
        if type(context) is not CaptureDigestContext:
            raise CaptureSessionSnapshotError()
        validated = _validated_snapshot(snapshot)
        digest = context.integrity_tag(_snapshot_preimage(validated))
        return _validated_snapshot(validated.model_copy(update={"snapshot_digest": digest}))
    except CaptureSessionSnapshotError:
        raise
    except Exception:
        raise CaptureSessionSnapshotError() from None


def verify_capture_session_snapshot(
    snapshot: object,
    *,
    installation_key: InstallationKey,
) -> CaptureSessionSnapshot:
    """Defensively copy and verify one key-bound capture snapshot."""

    try:
        if type(installation_key) is not InstallationKey:
            raise CaptureSessionSnapshotError()
        validated = _validated_snapshot(snapshot)
        expected = CaptureDigestContext(installation_key).integrity_tag(
            _snapshot_preimage(validated)
        )
        if not hmac.compare_digest(validated.snapshot_digest, expected):
            raise CaptureSessionSnapshotError()
        return validated
    except CaptureSessionSnapshotError:
        raise
    except Exception:
        raise CaptureSessionSnapshotError() from None


__all__ = [
    "CaptureHumanSessionId",
    "CaptureSessionSnapshot",
    "CaptureSessionSnapshotError",
    "CaptureSnapshotEvent",
    "CaptureSnapshotHealth",
    "verify_capture_session_snapshot",
]
