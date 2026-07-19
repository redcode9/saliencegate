"""Content-free health marker contracts for capture coverage degradation."""

from __future__ import annotations

from enum import StrEnum
from typing import TypedDict


class CaptureHealthCode(StrEnum):
    """Closed vocabulary persisted by the capture store."""

    PRODUCER_COLLISION = "producer_collision"
    SESSION_OVERFLOW = "session_overflow"
    SPOOL_QUOTA = "spool_quota"
    SPOOL_UNAVAILABLE = "spool_unavailable"
    UNATTRIBUTED_DROP = "unattributed_drop"
    INTEGRITY_FAILURE = "integrity_failure"
    GAP_DETECTED = "gap_detected"
    COVERAGE_DEGRADED = "coverage_degraded"


class CaptureHealthIdentityMaterial(TypedDict):
    schema_version: str
    connection_id: str | None
    session_id: str | None
    code: str


class CaptureHealthIntegrityMaterial(CaptureHealthIdentityMaterial):
    marker_id: str
    count: int
    lower_bound: int
    created_at: str
    updated_at: str


def capture_health_identity_material(
    *,
    connection_id: str | None,
    session_id: str | None,
    code: CaptureHealthCode,
) -> CaptureHealthIdentityMaterial:
    """Build the versioned, value-free marker identity preimage."""

    return {
        "schema_version": "capture-health-id/v1",
        "connection_id": connection_id,
        "session_id": session_id,
        "code": code.value,
    }


def capture_health_integrity_material(
    *,
    marker_id: str,
    connection_id: str | None,
    session_id: str | None,
    code: CaptureHealthCode,
    count: int,
    lower_bound: int,
    created_at: str,
    updated_at: str,
) -> CaptureHealthIntegrityMaterial:
    """Build the authenticated health counter preimage."""

    return {
        "schema_version": "capture-health-integrity/v1",
        "marker_id": marker_id,
        "connection_id": connection_id,
        "session_id": session_id,
        "code": code.value,
        "count": count,
        "lower_bound": lower_bound,
        "created_at": created_at,
        "updated_at": updated_at,
    }


__all__ = [
    "CaptureHealthCode",
    "CaptureHealthIdentityMaterial",
    "CaptureHealthIntegrityMaterial",
    "capture_health_identity_material",
    "capture_health_integrity_material",
]
