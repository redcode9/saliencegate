"""Content-free summaries for authenticated capture-store queries."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.capture.capabilities import CaptureProfile, CompatibilityStatus
from saliencegate.capture.sessions import CaptureHumanSessionId
from saliencegate.capture.store import (
    MAX_CAPTURE_EVENTS_PER_SESSION,
    CaptureConnectionState,
    CaptureSessionState,
)
from saliencegate.domain.records import ComponentIdentifier, Sha256Digest, UtcDatetime


class _CaptureQueryModel(BaseModel):
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


class CaptureConnectionSummary(_CaptureQueryModel):
    """One authenticated connection row without provider-native content."""

    schema_version: Literal["capture-connection-summary/v1"] = "capture-connection-summary/v1"
    connection_id: Annotated[ComponentIdentifier, Field(repr=False)]
    project_digest: Annotated[Sha256Digest, Field(repr=False)]
    profile_id: CaptureProfile
    capability_manifest_digest: Annotated[Sha256Digest, Field(repr=False)]
    host_version: Annotated[ComponentIdentifier, Field(repr=False)]
    compatibility_status: CompatibilityStatus
    state: CaptureConnectionState
    created_at: Annotated[UtcDatetime, Field(repr=False)]
    updated_at: Annotated[UtcDatetime, Field(repr=False)]

    @model_validator(mode="after")
    def timestamps_are_consistent(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError("capture connection timestamps are inconsistent")
        return self


class CaptureSessionSummary(_CaptureQueryModel):
    """One fully verified session summary without captured event content."""

    schema_version: Literal["capture-session-summary/v1"] = "capture-session-summary/v1"
    connection_id: Annotated[ComponentIdentifier, Field(repr=False)]
    project_digest: Annotated[Sha256Digest, Field(repr=False)]
    profile_id: CaptureProfile
    session_id: Annotated[Sha256Digest, Field(repr=False)]
    human_id: Annotated[CaptureHumanSessionId, Field(repr=False)]
    state: CaptureSessionState
    event_count: Annotated[int, Field(ge=0, le=MAX_CAPTURE_EVENTS_PER_SESSION)]
    coverage_degraded: bool
    unattributed_drop: bool
    opened_at: Annotated[UtcDatetime, Field(repr=False)]
    updated_at: Annotated[UtcDatetime, Field(repr=False)]
    closed_at: Annotated[UtcDatetime | None, Field(repr=False)]

    @model_validator(mode="after")
    def commitments_are_consistent(self) -> Self:
        if ((self.state is CaptureSessionState.CLOSED) != (self.closed_at is not None)) or (
            self.unattributed_drop and not self.coverage_degraded
        ):
            raise ValueError("capture session summary commitments are inconsistent")
        return self


class CaptureSessionInventory(_CaptureQueryModel):
    """Verified project/profile totals without truncating the status view."""

    schema_version: Literal["capture-session-inventory/v1"] = "capture-session-inventory/v1"
    session_count: Annotated[int, Field(ge=0)]
    quarantined_sessions: Annotated[int, Field(ge=0)]
    degraded_sessions: Annotated[int, Field(ge=0)]
    oldest_session: Annotated[CaptureHumanSessionId | None, Field(repr=False)]

    @model_validator(mode="after")
    def totals_are_consistent(self) -> Self:
        if (
            self.quarantined_sessions > self.session_count
            or self.degraded_sessions > self.session_count
            or ((self.oldest_session is None) != (self.session_count == 0))
        ):
            raise ValueError("capture session inventory is inconsistent")
        return self


__all__ = [
    "CaptureConnectionSummary",
    "CaptureSessionInventory",
    "CaptureSessionSummary",
]
