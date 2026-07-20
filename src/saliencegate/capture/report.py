"""Deterministic, authority-explicit reports over verified capture snapshots."""

from __future__ import annotations

import hmac
from collections import Counter
from contextlib import suppress
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Literal, Self, TypeAlias

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from saliencegate.capture.capabilities import (
    CapabilitySupport,
    CaptureProfile,
    CompatibilityStatus,
    capture_capability_digest,
    capture_profile,
)
from saliencegate.capture.health import CaptureHealthCode
from saliencegate.capture.schema import (
    CaptureControllerFailedIntake,
    CaptureJSONLimits,
    read_bounded_json,
)
from saliencegate.capture.sessions import verify_capture_session_snapshot
from saliencegate.capture.spool import (
    CaptureSpool,
    CaptureSpoolObservation,
    verify_capture_spool_observation,
)
from saliencegate.capture.store import CaptureSessionState
from saliencegate.domain import SignalType, canonical_json, length_prefixed_sha256
from saliencegate.domain.records import Sha256Digest, UtcDatetime
from saliencegate.security import InstallationKey
from saliencegate.shadow.evaluation import ShadowHeuristicDisposition
from saliencegate.signals import DetectionStatus

if TYPE_CHECKING:
    from saliencegate.capture.normalization import CaptureNormalization
    from saliencegate.capture.sessions import CaptureSessionSnapshot

CAPTURE_SESSION_REPORT_SCHEMA_VERSION: Literal["capture-session-report/v1"] = (
    "capture-session-report/v1"
)


class CaptureReportError(ValueError):
    """A content-free failure at the capture report boundary."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture session report is invalid")


class CaptureReportHeadline(StrEnum):
    """The complete human-facing conclusion vocabulary for capture v1."""

    MEMORY_REVIEW_SUGGESTED = "memory_review_suggested"
    NO_CURRENT_EVIDENCE = "no_current_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class CaptureSpoolReportStatus(StrEnum):
    """Whether authenticated spool state permits a negative conclusion."""

    VERIFIED_CLEAN_DRAINED = "verified_clean_drained"
    VERIFIED_PENDING = "verified_pending"
    VERIFIED_DEGRADED = "verified_degraded"
    UNAVAILABLE = "unavailable"


class CaptureReportLimit(StrEnum):
    """Closed, content-free reasons that bound a capture conclusion."""

    SESSION_OPEN = "session_open"
    SESSION_QUARANTINED = "session_quarantined"
    SESSION_DELETING = "session_deleting"
    SEMANTIC_INCOHERENCE = "semantic_incoherence"
    CAPTURE_DEGRADED = "capture_degraded"
    GAP_DETECTED = "gap_detected"
    SESSION_OVERFLOW = "session_overflow"
    UNATTRIBUTED_DROP = "unattributed_drop"
    INTEGRITY_FAILURE = "integrity_failure"
    PRODUCER_COLLISION = "producer_collision"
    SPOOL_QUOTA = "spool_quota"
    SPOOL_UNAVAILABLE = "spool_unavailable"
    SPOOL_PENDING = "spool_pending"
    SPOOL_DROP = "spool_drop"
    COMPATIBILITY_UNVERIFIED = "compatibility_unverified"
    NO_APPLICABLE_DETECTOR = "no_applicable_detector"
    DETECTOR_MINIMUM_NOT_MET = "detector_minimum_not_met"


_NonNegativeCount = Annotated[int, Field(ge=0, le=(1 << 63) - 1)]
_CaptureWindowCount = Annotated[int, Field(ge=0, le=1_000)]
_CaptureSignalCount = Annotated[int, Field(ge=0, le=1_000)]


def _require_exact_report_text(value: str) -> str:
    if type(value) is not str:
        raise ValueError("capture report text is invalid")
    return value


_HumanSessionId: TypeAlias = Annotated[
    str,
    StringConstraints(min_length=12, max_length=52, pattern=r"^[a-z2-7]+$"),
    AfterValidator(_require_exact_report_text),
]
_HostVersion: TypeAlias = Annotated[
    str,
    StringConstraints(
        min_length=3,
        max_length=64,
        pattern=r"^(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){1,3}$",
    ),
    AfterValidator(_require_exact_report_text),
]
_CoverageExclusion: TypeAlias = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9_@][A-Za-z0-9._@:/+\-\[\]]*$",
    ),
    AfterValidator(_require_exact_report_text),
]


class _CaptureReportModel(BaseModel):
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


class CaptureReportCounts(_CaptureReportModel):
    """Content-free denominators for the reported capture window."""

    captured_events: _CaptureWindowCount
    projected_events: _CaptureWindowCount
    action_identities: _CaptureWindowCount
    structured_results: _CaptureWindowCount
    detected_signals: Annotated[int, Field(ge=0, le=9_000)]
    ignored_records: _CaptureWindowCount

    @model_validator(mode="after")
    def counts_are_bounded_by_the_capture_window(self) -> Self:
        if (
            self.captured_events != self.projected_events + self.ignored_records
            or self.action_identities > self.projected_events
            or self.structured_results > self.projected_events
            or self.action_identities + self.structured_results > self.projected_events
        ):
            raise ValueError("capture report counts disagree")
        return self


class CaptureReportHealthCount(_CaptureReportModel):
    """One authenticated, content-free store health counter."""

    code: CaptureHealthCode
    count: _NonNegativeCount
    lower_bound: _NonNegativeCount

    @model_validator(mode="after")
    def lower_bound_does_not_exceed_count(self) -> Self:
        if self.lower_bound > self.count:
            raise ValueError("capture report health count is invalid")
        return self


class CaptureReportDetector(_CaptureReportModel):
    """Capability support and evidence sufficiency for one detector."""

    signal_type: SignalType
    support: CapabilitySupport
    omissions: Annotated[tuple[_CoverageExclusion, ...], Field(max_length=32)]
    disposition: ShadowHeuristicDisposition
    minimum_authorized_observations: Annotated[int, Field(ge=0, le=2)]
    authorized_observation_count: _CaptureWindowCount
    unresolved_observation_count: _CaptureWindowCount
    detected_count: _CaptureSignalCount
    sufficient_for_absence: bool

    @model_validator(mode="after")
    def capability_and_evidence_agree(self) -> Self:
        from saliencegate.capture.normalization import _capture_detector_minimum

        if self.omissions != tuple(sorted(set(self.omissions))):
            raise ValueError("capture report detector omissions are not canonical")
        if self.support is CapabilitySupport.SUPPORTED and self.omissions:
            raise ValueError("supported detector cannot declare omissions")
        if self.support is not CapabilitySupport.SUPPORTED and not self.omissions:
            raise ValueError("bounded detector must declare omissions")
        if self.support is CapabilitySupport.UNSUPPORTED:
            valid = (
                self.minimum_authorized_observations == 0
                and self.authorized_observation_count == 0
                and self.unresolved_observation_count == 0
                and self.detected_count == 0
                and not self.sufficient_for_absence
                and self.disposition is ShadowHeuristicDisposition.NOT_APPLICABLE
            )
        else:
            if self.minimum_authorized_observations != _capture_detector_minimum(self.signal_type):
                raise ValueError("capture report detector minimum is invalid")
            minimum_met = (
                self.authorized_observation_count >= self.minimum_authorized_observations >= 1
                and self.unresolved_observation_count == 0
            )
            valid = self.sufficient_for_absence is minimum_met
            if self.detected_count:
                valid = valid and self.disposition is ShadowHeuristicDisposition.FLAGGED
            elif self.sufficient_for_absence:
                valid = valid and self.disposition is ShadowHeuristicDisposition.NOT_FLAGGED
            else:
                valid = valid and self.disposition is ShadowHeuristicDisposition.INDETERMINATE
        if not valid:
            raise ValueError("capture report detector evidence is inconsistent")
        return self


class CaptureReportCoverage(_CaptureReportModel):
    """Authenticated coverage state and the declared limits on the report."""

    capture_scope: Literal["bounded_window"] = "bounded_window"
    snapshot_integrity: Literal["hmac_verified"] = "hmac_verified"
    health_integrity: Literal["hmac_verified"] = "hmac_verified"
    spool_integrity: Literal["hmac_verified_snapshot_bound", "unavailable"]
    spool_observation_tag: Sha256Digest | None
    spool_status: CaptureSpoolReportStatus
    coverage_degraded: bool
    gap_count: _NonNegativeCount
    drop_count: _NonNegativeCount
    overflow_count: _NonNegativeCount
    queued_spool_events: _NonNegativeCount
    dropped_spool_events: _NonNegativeCount
    health: Annotated[
        tuple[CaptureReportHealthCount, ...],
        Field(min_length=len(CaptureHealthCode), max_length=len(CaptureHealthCode)),
    ]
    capability_exclusions: Annotated[tuple[_CoverageExclusion, ...], Field(max_length=64)]
    limits: Annotated[tuple[CaptureReportLimit, ...], Field(max_length=len(CaptureReportLimit))]

    @model_validator(mode="after")
    def coverage_is_closed_and_canonical(self) -> Self:
        if tuple(item.code for item in self.health) != tuple(CaptureHealthCode):
            raise ValueError("capture report health matrix is not closed")
        if self.capability_exclusions != tuple(sorted(set(self.capability_exclusions))):
            raise ValueError("capture report exclusions are not canonical")
        if self.limits != tuple(sorted(set(self.limits), key=lambda item: item.value)):
            raise ValueError("capture report limits are not canonical")
        spool_available = self.spool_status is not CaptureSpoolReportStatus.UNAVAILABLE
        if spool_available is not (
            self.spool_integrity == "hmac_verified_snapshot_bound"
            and self.spool_observation_tag is not None
        ):
            raise ValueError("capture report spool integrity is inconsistent")
        clean_spool = (
            self.queued_spool_events == 0
            and self.dropped_spool_events == 0
            and self.spool_status is CaptureSpoolReportStatus.VERIFIED_CLEAN_DRAINED
        )
        if clean_spool is not (
            self.spool_status is CaptureSpoolReportStatus.VERIFIED_CLEAN_DRAINED
        ):
            raise ValueError("capture report spool state is inconsistent")
        limits = set(self.limits)
        if (
            (self.gap_count > 0) is not (CaptureReportLimit.GAP_DETECTED in limits)
            or (self.drop_count > 0) is not (CaptureReportLimit.UNATTRIBUTED_DROP in limits)
            or (self.overflow_count > 0) is not (CaptureReportLimit.SESSION_OVERFLOW in limits)
            or (self.queued_spool_events > 0) is not (CaptureReportLimit.SPOOL_PENDING in limits)
            or (self.dropped_spool_events > 0) is not (CaptureReportLimit.SPOOL_DROP in limits)
        ):
            raise ValueError("capture report coverage counters disagree with limits")
        if self.spool_status is CaptureSpoolReportStatus.UNAVAILABLE:
            valid_spool = CaptureReportLimit.SPOOL_UNAVAILABLE in limits
        elif self.spool_status is CaptureSpoolReportStatus.VERIFIED_PENDING:
            valid_spool = self.queued_spool_events > 0 and self.dropped_spool_events == 0
        elif self.spool_status is CaptureSpoolReportStatus.VERIFIED_DEGRADED:
            valid_spool = self.dropped_spool_events > 0
        else:
            valid_spool = not (
                limits
                & {
                    CaptureReportLimit.SPOOL_DROP,
                    CaptureReportLimit.SPOOL_PENDING,
                }
            )
        if not valid_spool:
            raise ValueError("capture report spool status disagrees with limits")
        health_counts = {item.code: item.count for item in self.health}
        health_limit_pairs = (
            (CaptureHealthCode.COVERAGE_DEGRADED, CaptureReportLimit.CAPTURE_DEGRADED),
            (CaptureHealthCode.GAP_DETECTED, CaptureReportLimit.GAP_DETECTED),
            (CaptureHealthCode.INTEGRITY_FAILURE, CaptureReportLimit.INTEGRITY_FAILURE),
            (CaptureHealthCode.PRODUCER_COLLISION, CaptureReportLimit.PRODUCER_COLLISION),
            (CaptureHealthCode.SESSION_OVERFLOW, CaptureReportLimit.SESSION_OVERFLOW),
            (CaptureHealthCode.SPOOL_QUOTA, CaptureReportLimit.SPOOL_QUOTA),
            (CaptureHealthCode.SPOOL_UNAVAILABLE, CaptureReportLimit.SPOOL_UNAVAILABLE),
            (CaptureHealthCode.UNATTRIBUTED_DROP, CaptureReportLimit.UNATTRIBUTED_DROP),
        )
        if any(
            health_counts[code] > 0 and limit not in limits for code, limit in health_limit_pairs
        ):
            raise ValueError("capture report health counters disagree with limits")
        evidence_only = {
            CaptureReportLimit.NO_APPLICABLE_DETECTOR,
            CaptureReportLimit.DETECTOR_MINIMUM_NOT_MET,
        }
        if self.coverage_degraded is not bool(set(self.limits) - evidence_only):
            raise ValueError("capture report coverage status is inconsistent")
        return self


class CaptureReportInterval(_CaptureReportModel):
    """Authenticated wall-clock observations without a monotonic-clock claim."""

    opened_at: UtcDatetime
    updated_at: UtcDatetime
    closed_at: UtcDatetime | None


class CaptureSessionReport(_CaptureReportModel):
    """One deterministic, content-free report over a verified capture snapshot."""

    schema_version: Literal["capture-session-report/v1"] = CAPTURE_SESSION_REPORT_SCHEMA_VERSION
    session_id: _HumanSessionId
    session_state: CaptureSessionState
    interval: CaptureReportInterval
    profile_id: CaptureProfile
    host_version: _HostVersion
    compatibility_status: CompatibilityStatus
    headline: CaptureReportHeadline
    shadow_disposition: ShadowHeuristicDisposition
    counts: CaptureReportCounts
    coverage: CaptureReportCoverage
    detectors: Annotated[
        tuple[CaptureReportDetector, ...],
        Field(min_length=len(SignalType), max_length=len(SignalType)),
    ]
    source_authentication: Literal["none_same_user_untrusted"] = "none_same_user_untrusted"
    raw_content_persisted: Literal[False] = False
    transcript_read: Literal[False] = False
    complete_execution_session_coverage: Literal[False] = False
    at_rest_integrity: Literal["hmac_sha256_local_mutation_detection"] = (
        "hmac_sha256_local_mutation_detection"
    )
    report_integrity: Literal["sha256_canonical_body"] = "sha256_canonical_body"
    rollback_detection: Literal["none"] = "none"
    timestamp_authority: Literal["local_observation"] = "local_observation"
    sequence_authority: Literal["local_receipt_order"] = "local_receipt_order"
    evidence_level: Literal["descriptive_observational"] = "descriptive_observational"
    decision_authority: Literal[False] = False
    model_calls: Literal[0] = 0
    confirmatory: Literal[False] = False
    capability_manifest_digest: Sha256Digest
    snapshot_digest: Sha256Digest
    normalization_digest: Sha256Digest
    report_digest: Sha256Digest

    @model_validator(mode="after")
    def conclusion_and_bindings_are_consistent(self) -> Self:
        manifest = capture_profile(self.profile_id)
        if (
            not hmac.compare_digest(
                self.capability_manifest_digest,
                capture_capability_digest(manifest),
            )
            or self.coverage.capability_exclusions != manifest.coverage_exclusions
            or tuple((item.signal_type, item.support, item.omissions) for item in self.detectors)
            != tuple(
                (item.signal_type, item.support, item.omissions) for item in manifest.detectors
            )
            or self.compatibility_status is CompatibilityStatus.INCOMPATIBLE
            or (self.compatibility_status is CompatibilityStatus.VERIFIED)
            is not (self.host_version == manifest.host_version)
        ):
            raise ValueError("capture report capability binding is inconsistent")
        if tuple(item.signal_type for item in self.detectors) != tuple(SignalType):
            raise ValueError("capture report detector matrix is not closed")
        rows = {item.signal_type: item for item in self.detectors}
        repeated_action_observations = (
            rows[SignalType.REPEATED_ACTION].authorized_observation_count
            + rows[SignalType.REPEATED_ACTION].unresolved_observation_count
        )
        repeated_failure_observations = (
            rows[SignalType.REPEATED_FAILURE].authorized_observation_count
            + rows[SignalType.REPEATED_FAILURE].unresolved_observation_count
        )
        tool_error_observations = (
            rows[SignalType.TOOL_ERROR].authorized_observation_count
            + rows[SignalType.TOOL_ERROR].unresolved_observation_count
        )
        if (
            repeated_action_observations > self.counts.action_identities
            or repeated_failure_observations > self.counts.structured_results
            or tool_error_observations > self.counts.structured_results
            or rows[SignalType.REPEATED_ACTION].detected_count > self.counts.action_identities
            or rows[SignalType.REPEATED_FAILURE].detected_count > self.counts.structured_results
            or any(item.detected_count > self.counts.projected_events for item in self.detectors)
        ):
            raise ValueError("capture report detector denominators are inconsistent")
        detected = sum(item.detected_count for item in self.detectors)
        if self.counts.detected_signals != detected:
            raise ValueError("capture report signal count disagrees")
        if self.session_state is CaptureSessionState.CLOSED:
            if self.interval.closed_at is None:
                raise ValueError("closed capture report has no closing time")
        elif self.interval.closed_at is not None:
            raise ValueError("non-closed capture report has a closing time")

        limits = set(self.coverage.limits)
        state_limits = {
            CaptureReportLimit.SESSION_OPEN,
            CaptureReportLimit.SESSION_QUARANTINED,
            CaptureReportLimit.SESSION_DELETING,
        }
        expected_state_limit = {
            CaptureSessionState.OPEN: CaptureReportLimit.SESSION_OPEN,
            CaptureSessionState.CLOSED: None,
            CaptureSessionState.QUARANTINED: CaptureReportLimit.SESSION_QUARANTINED,
            CaptureSessionState.DELETING: CaptureReportLimit.SESSION_DELETING,
        }[self.session_state]
        expected_state_limits = set() if expected_state_limit is None else {expected_state_limit}
        if limits & state_limits != expected_state_limits or (
            CaptureReportLimit.COMPATIBILITY_UNVERIFIED in limits
        ) is not (self.compatibility_status is not CompatibilityStatus.VERIFIED):
            raise ValueError("capture report boundary limits are inconsistent")

        health_counts = {item.code: item.count for item in self.coverage.health}
        positive_precedence_blocked = (
            self.session_state is CaptureSessionState.QUARANTINED
            or health_counts[CaptureHealthCode.INTEGRITY_FAILURE] > 0
        )
        if self.headline is CaptureReportHeadline.MEMORY_REVIEW_SUGGESTED:
            valid = (
                self.shadow_disposition is ShadowHeuristicDisposition.FLAGGED
                and detected > 0
                and not positive_precedence_blocked
            )
        elif self.headline is CaptureReportHeadline.NO_CURRENT_EVIDENCE:
            valid = (
                self.shadow_disposition is ShadowHeuristicDisposition.NOT_FLAGGED
                and self.session_state is CaptureSessionState.CLOSED
                and self.coverage.spool_status is CaptureSpoolReportStatus.VERIFIED_CLEAN_DRAINED
                and not self.coverage.limits
                and any(item.sufficient_for_absence for item in self.detectors)
                and not detected
            )
        else:
            valid = (
                bool(self.coverage.limits)
                and (not detected or positive_precedence_blocked)
                and self.shadow_disposition
                in (
                    ShadowHeuristicDisposition.INDETERMINATE,
                    ShadowHeuristicDisposition.NOT_APPLICABLE,
                )
            )
        if not valid:
            raise ValueError("capture report headline is inconsistent")
        if not hmac.compare_digest(_capture_report_body_digest(self), self.report_digest):
            raise ValueError("capture report digest is invalid")
        return self


_REPORT_DIGEST_DOMAIN = "saliencegate:capture:session-report:v1"
_CAPTURE_REPORT_LIMITS = CaptureJSONLimits(
    max_bytes=512 * 1_024,
    max_depth=16,
    max_items=10_000,
    max_string_bytes=256 * 1_024,
)


def _capture_report_body(report: CaptureSessionReport) -> dict[str, object]:
    return {
        key: value
        for key, value in report.model_dump(mode="json", warnings=False).items()
        if key != "report_digest"
    }


def _capture_report_body_digest(report: CaptureSessionReport) -> str:
    return _capture_report_mapping_digest(_capture_report_body(report))


def _capture_report_mapping_digest(body: dict[str, object]) -> str:
    return length_prefixed_sha256(canonical_json(body), domain=_REPORT_DIGEST_DOMAIN)


def _seal_capture_session_report(body: dict[str, object]) -> CaptureSessionReport:
    if "report_digest" in body:
        raise CaptureReportError()
    try:
        complete_body = {
            "schema_version": CAPTURE_SESSION_REPORT_SCHEMA_VERSION,
            "source_authentication": "none_same_user_untrusted",
            "raw_content_persisted": False,
            "transcript_read": False,
            "complete_execution_session_coverage": False,
            "at_rest_integrity": "hmac_sha256_local_mutation_detection",
            "report_integrity": "sha256_canonical_body",
            "rollback_detection": "none",
            "timestamp_authority": "local_observation",
            "sequence_authority": "local_receipt_order",
            "evidence_level": "descriptive_observational",
            "decision_authority": False,
            "model_calls": 0,
            "confirmatory": False,
            **body,
        }
        report = CaptureSessionReport.model_validate(
            {
                **complete_body,
                "report_digest": _capture_report_mapping_digest(complete_body),
            }
        )
        if not _model_state_is_exact(CaptureSessionReport, report):
            raise ValueError("capture report model is invalid")
        return report
    except Exception:
        raise CaptureReportError() from None


def _model_state_is_exact(model_type: type[BaseModel], value: object) -> bool:
    try:
        return (
            type(value) is model_type
            and type(value.__dict__) is dict
            and set(value.__dict__) == set(model_type.model_fields)
            and value.__pydantic_extra__ is None
            and value.__pydantic_private__ is None
        )
    except Exception:
        return False


def _observe_spool(
    spool: CaptureSpool | None,
    *,
    snapshot_digest: str,
    spool_boundary_digest: str | None,
    installation_key: InstallationKey,
) -> CaptureSpoolObservation | None:
    if spool is None:
        return None
    if type(spool) is not CaptureSpool or type(spool_boundary_digest) is not str:
        raise CaptureReportError()
    try:
        observed = spool.observe_health(snapshot_digest)
        return verify_capture_spool_observation(
            observed,
            expected_snapshot_digest=snapshot_digest,
            expected_spool_boundary_digest=spool_boundary_digest,
            installation_key=installation_key,
        )
    except Exception:
        raise CaptureReportError() from None


def _health_matrix(snapshot: CaptureSessionSnapshot) -> tuple[CaptureReportHealthCount, ...]:
    by_code = {item.code: item for item in snapshot.health}
    return tuple(
        CaptureReportHealthCount(
            code=code,
            count=0 if (item := by_code.get(code)) is None else item.count,
            lower_bound=0 if item is None else item.lower_bound,
        )
        for code in CaptureHealthCode
    )


def _spool_report_state(
    observation: CaptureSpoolObservation | None,
) -> tuple[CaptureSpoolReportStatus, int, int, tuple[CaptureReportLimit, ...]]:
    if observation is None:
        return (
            CaptureSpoolReportStatus.UNAVAILABLE,
            0,
            0,
            (CaptureReportLimit.SPOOL_UNAVAILABLE,),
        )
    limits: list[CaptureReportLimit] = []
    if observation.dropped_events:
        status = CaptureSpoolReportStatus.VERIFIED_DEGRADED
        limits.append(CaptureReportLimit.SPOOL_DROP)
        if observation.queued_events:
            limits.append(CaptureReportLimit.SPOOL_PENDING)
        if observation.last_drop_reason == "spool_quota":
            limits.append(CaptureReportLimit.SPOOL_QUOTA)
    elif observation.queued_events:
        status = CaptureSpoolReportStatus.VERIFIED_PENDING
        limits.append(CaptureReportLimit.SPOOL_PENDING)
    else:
        status = CaptureSpoolReportStatus.VERIFIED_CLEAN_DRAINED
    return (
        status,
        observation.queued_events,
        observation.dropped_events,
        tuple(limits),
    )


def _detector_matrix(
    snapshot: CaptureSessionSnapshot,
    normalization: CaptureNormalization,
) -> tuple[CaptureReportDetector, ...]:
    manifest = capture_profile(snapshot.profile_id)
    if not hmac.compare_digest(
        capture_capability_digest(manifest),
        snapshot.capability_manifest_digest,
    ):
        raise CaptureReportError()
    evidence_by_type = {item.signal_type: item for item in normalization.detector_evidence}
    detected: Counter[SignalType] = Counter()
    for extraction in normalization.extraction_reports:
        for evaluation in extraction.evaluations:
            if evaluation.outcome.status is DetectionStatus.DETECTED:
                detected[evaluation.signal_type] += 1

    rows: list[CaptureReportDetector] = []
    for capability in manifest.detectors:
        evidence = evidence_by_type.get(capability.signal_type)
        if capability.support is CapabilitySupport.UNSUPPORTED:
            if evidence is not None or detected[capability.signal_type]:
                raise CaptureReportError()
            rows.append(
                CaptureReportDetector(
                    signal_type=capability.signal_type,
                    support=capability.support,
                    omissions=capability.omissions,
                    disposition=ShadowHeuristicDisposition.NOT_APPLICABLE,
                    minimum_authorized_observations=0,
                    authorized_observation_count=0,
                    unresolved_observation_count=0,
                    detected_count=0,
                    sufficient_for_absence=False,
                )
            )
            continue
        if (
            evidence is None
            or evidence.support is not capability.support
            or evidence.omissions != capability.omissions
        ):
            raise CaptureReportError()
        detected_count = detected[capability.signal_type]
        disposition = (
            ShadowHeuristicDisposition.FLAGGED
            if detected_count
            else ShadowHeuristicDisposition.NOT_FLAGGED
            if evidence.minimum_observation_met
            else ShadowHeuristicDisposition.INDETERMINATE
        )
        rows.append(
            CaptureReportDetector(
                signal_type=capability.signal_type,
                support=capability.support,
                omissions=capability.omissions,
                disposition=disposition,
                minimum_authorized_observations=(evidence.minimum_authorized_observations),
                authorized_observation_count=evidence.authorized_observation_count,
                unresolved_observation_count=evidence.unresolved_observation_count,
                detected_count=detected_count,
                sufficient_for_absence=evidence.minimum_observation_met,
            )
        )
    return tuple(rows)


def build_capture_session_report(
    snapshot: CaptureSessionSnapshot,
    normalization: CaptureNormalization,
    *,
    installation_key: InstallationKey,
    spool: CaptureSpool | None,
) -> CaptureSessionReport:
    """Build a deterministic report from authenticated snapshot and normalization state."""

    result: CaptureSessionReport | None = None
    try:
        from saliencegate.capture.normalization import verify_capture_normalization

        verified_snapshot = verify_capture_session_snapshot(
            snapshot,
            installation_key=installation_key,
        )
        verified_normalization = verify_capture_normalization(
            normalization,
            snapshot=verified_snapshot,
            installation_key=installation_key,
        )
        if not hmac.compare_digest(
            verified_snapshot.snapshot_digest,
            verified_normalization.snapshot_digest,
        ):
            raise CaptureReportError()
        checked_spool = _observe_spool(
            spool,
            snapshot_digest=verified_snapshot.snapshot_digest,
            spool_boundary_digest=verified_snapshot.spool_boundary_digest,
            installation_key=installation_key,
        )
        detectors = _detector_matrix(verified_snapshot, verified_normalization)
        health = _health_matrix(verified_snapshot)
        health_counts = {item.code: item.count for item in health}
        controller_failures = Counter(
            item.event.intake.error_code
            for item in verified_snapshot.events
            if type(item.event.intake) is CaptureControllerFailedIntake
        )
        gap_count = (
            health_counts[CaptureHealthCode.GAP_DETECTED]
            + controller_failures["gap_detected"]
            + verified_snapshot.incomplete_transport_batch_count
        )
        overflow_count = (
            health_counts[CaptureHealthCode.SESSION_OVERFLOW] + controller_failures["overflow"]
        )
        local_controller_degradation = sum(controller_failures.values()) - (
            controller_failures["gap_detected"] + controller_failures["overflow"]
        )
        drop_count = max(
            health_counts[CaptureHealthCode.UNATTRIBUTED_DROP],
            int(verified_snapshot.unattributed_drop),
        )

        limits: list[CaptureReportLimit] = []
        if verified_snapshot.state is CaptureSessionState.OPEN:
            limits.append(CaptureReportLimit.SESSION_OPEN)
        elif verified_snapshot.state is CaptureSessionState.QUARANTINED:
            limits.append(CaptureReportLimit.SESSION_QUARANTINED)
        elif verified_snapshot.state is CaptureSessionState.DELETING:
            limits.append(CaptureReportLimit.SESSION_DELETING)
        if not verified_normalization.semantic_coherence:
            limits.append(CaptureReportLimit.SEMANTIC_INCOHERENCE)
        if verified_snapshot.coverage_degraded or local_controller_degradation:
            limits.append(CaptureReportLimit.CAPTURE_DEGRADED)
        if health_counts[CaptureHealthCode.COVERAGE_DEGRADED]:
            limits.append(CaptureReportLimit.CAPTURE_DEGRADED)
        if gap_count:
            limits.append(CaptureReportLimit.GAP_DETECTED)
        if overflow_count:
            limits.append(CaptureReportLimit.SESSION_OVERFLOW)
        if drop_count:
            limits.append(CaptureReportLimit.UNATTRIBUTED_DROP)
        if health_counts[CaptureHealthCode.INTEGRITY_FAILURE]:
            limits.append(CaptureReportLimit.INTEGRITY_FAILURE)
        if health_counts[CaptureHealthCode.PRODUCER_COLLISION]:
            limits.append(CaptureReportLimit.PRODUCER_COLLISION)
        if health_counts[CaptureHealthCode.SPOOL_QUOTA]:
            limits.append(CaptureReportLimit.SPOOL_QUOTA)
        if health_counts[CaptureHealthCode.SPOOL_UNAVAILABLE]:
            limits.append(CaptureReportLimit.SPOOL_UNAVAILABLE)
        if verified_snapshot.compatibility_status is not CompatibilityStatus.VERIFIED:
            limits.append(CaptureReportLimit.COMPATIBILITY_UNVERIFIED)

        spool_status, queued, dropped, spool_limits = _spool_report_state(checked_spool)
        limits.extend(spool_limits)
        detected_count = sum(item.detected_count for item in detectors)
        sufficient = any(item.sufficient_for_absence for item in detectors)
        applicable = any(
            item.support is not CapabilitySupport.UNSUPPORTED
            and (
                item.authorized_observation_count > 0
                or item.unresolved_observation_count > 0
                or item.detected_count > 0
            )
            for item in detectors
        )
        if not applicable:
            limits.append(CaptureReportLimit.NO_APPLICABLE_DETECTOR)
        elif not detected_count and not sufficient:
            limits.append(CaptureReportLimit.DETECTOR_MINIMUM_NOT_MET)

        ordered_limits = tuple(sorted(set(limits), key=lambda item: item.value))
        precedence_block = (
            verified_snapshot.state is CaptureSessionState.QUARANTINED
            or health_counts[CaptureHealthCode.INTEGRITY_FAILURE] > 0
        )
        if detected_count and not precedence_block:
            headline = CaptureReportHeadline.MEMORY_REVIEW_SUGGESTED
            disposition = ShadowHeuristicDisposition.FLAGGED
        elif (
            not detected_count
            and applicable
            and sufficient
            and verified_snapshot.state is CaptureSessionState.CLOSED
            and spool_status is CaptureSpoolReportStatus.VERIFIED_CLEAN_DRAINED
            and not ordered_limits
        ):
            headline = CaptureReportHeadline.NO_CURRENT_EVIDENCE
            disposition = ShadowHeuristicDisposition.NOT_FLAGGED
        else:
            headline = CaptureReportHeadline.INSUFFICIENT_EVIDENCE
            disposition = (
                ShadowHeuristicDisposition.NOT_APPLICABLE
                if not applicable
                else ShadowHeuristicDisposition.INDETERMINATE
            )

        evidence_only = {
            CaptureReportLimit.NO_APPLICABLE_DETECTOR,
            CaptureReportLimit.DETECTOR_MINIMUM_NOT_MET,
        }
        counts = verified_normalization.counts
        manifest = capture_profile(verified_snapshot.profile_id)
        result = _seal_capture_session_report(
            {
                "session_id": verified_snapshot.human_id,
                "session_state": verified_snapshot.state,
                "interval": CaptureReportInterval(
                    opened_at=verified_snapshot.opened_at,
                    updated_at=verified_snapshot.updated_at,
                    closed_at=verified_snapshot.closed_at,
                ),
                "profile_id": verified_snapshot.profile_id,
                "host_version": verified_snapshot.host_version,
                "compatibility_status": verified_snapshot.compatibility_status,
                "headline": headline,
                "shadow_disposition": disposition,
                "counts": CaptureReportCounts(
                    captured_events=counts.source_event_count,
                    projected_events=counts.mapped_event_count,
                    action_identities=counts.action_identity_count,
                    structured_results=counts.authorized_tool_result_count,
                    detected_signals=detected_count,
                    ignored_records=counts.ignored_event_count,
                ),
                "coverage": CaptureReportCoverage(
                    spool_integrity=(
                        "unavailable" if checked_spool is None else "hmac_verified_snapshot_bound"
                    ),
                    spool_observation_tag=(
                        None if checked_spool is None else checked_spool.observation_tag
                    ),
                    spool_status=spool_status,
                    coverage_degraded=bool(set(ordered_limits) - evidence_only),
                    gap_count=gap_count,
                    drop_count=drop_count,
                    overflow_count=overflow_count,
                    queued_spool_events=queued,
                    dropped_spool_events=dropped,
                    health=health,
                    capability_exclusions=manifest.coverage_exclusions,
                    limits=ordered_limits,
                ),
                "detectors": detectors,
                "capability_manifest_digest": (verified_snapshot.capability_manifest_digest),
                "snapshot_digest": verified_snapshot.snapshot_digest,
                "normalization_digest": verified_normalization.normalization_digest,
            }
        )
        if checked_spool is not None:
            rechecked_spool = _observe_spool(
                spool,
                snapshot_digest=verified_snapshot.snapshot_digest,
                spool_boundary_digest=verified_snapshot.spool_boundary_digest,
                installation_key=installation_key,
            )
            if rechecked_spool is None or not hmac.compare_digest(
                checked_spool.observation_tag,
                rechecked_spool.observation_tag,
            ):
                raise CaptureReportError()
    except CaptureReportError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        result = None
    if result is None:
        raise CaptureReportError()
    return result


def _decode_capture_session_report(data: bytes) -> CaptureSessionReport:
    if type(data) is not bytes:
        raise ValueError("capture report bytes are invalid")
    parsed = read_bounded_json(data, limits=_CAPTURE_REPORT_LIMITS)
    if not hmac.compare_digest(canonical_json(parsed), data):
        raise ValueError("capture report JSON is not canonical")
    report = CaptureSessionReport.model_validate_json(data)
    if not _model_state_is_exact(CaptureSessionReport, report):
        raise ValueError("capture report model is invalid")
    if not hmac.compare_digest(_capture_report_body_digest(report), report.report_digest):
        raise ValueError("capture report digest is invalid")
    return report


def encode_capture_session_report(report: CaptureSessionReport) -> bytes:
    """Return the one bounded canonical representation of a capture report."""

    result: bytes | None = None
    try:
        if not _model_state_is_exact(CaptureSessionReport, report):
            raise ValueError("capture report type is invalid")
        candidate = canonical_json(report.model_dump(mode="json", warnings=False))
        if _decode_capture_session_report(candidate) != report:
            raise ValueError("capture report defensive copy differs")
        result = candidate
    except Exception:
        result = None
    if result is None:
        raise CaptureReportError()
    return result


def decode_capture_session_report(data: bytes) -> CaptureSessionReport:
    """Decode only bounded, canonical, duplicate-safe capture report bytes."""

    result: CaptureSessionReport | None = None
    with suppress(Exception):
        result = _decode_capture_session_report(data)
    if result is None:
        raise CaptureReportError()
    return result


def render_capture_session_report_json(report: CaptureSessionReport) -> str:
    """Render a canonical JSON line for terminals and command adapters."""

    return encode_capture_session_report(report).decode("utf-8") + "\n"


def render_capture_session_report_human(report: CaptureSessionReport) -> str:
    """Render the bounded conclusion without paths, opaque IDs, or digests."""

    checked = decode_capture_session_report(encode_capture_session_report(report))
    conclusion = {
        CaptureReportHeadline.MEMORY_REVIEW_SUGGESTED: "Memory review suggested",
        CaptureReportHeadline.NO_CURRENT_EVIDENCE: "No current evidence",
        CaptureReportHeadline.INSUFFICIENT_EVIDENCE: "Insufficient evidence",
    }[checked.headline]
    detector_summary = ", ".join(
        f"{item.signal_type.value}={item.support.value}/{item.disposition.value}"
        for item in checked.detectors
    )
    limit_summary = (
        ", ".join(item.value for item in checked.coverage.limits)
        if checked.coverage.limits
        else "none"
    )
    exclusions = (
        ", ".join(checked.coverage.capability_exclusions)
        if checked.coverage.capability_exclusions
        else "none"
    )
    closed_at = checked.interval.closed_at.isoformat() if checked.interval.closed_at else "open"
    return (
        f"{conclusion}\n"
        "SalienceGate capture report\n"
        f"session: {checked.session_id}\n"
        f"state: {checked.session_state.value}; window: {checked.coverage.capture_scope}\n"
        f"profile: {checked.profile_id.value}; compatibility: "
        f"{checked.compatibility_status.value}; host version: {checked.host_version}\n"
        f"interval: opened={checked.interval.opened_at.isoformat()}, "
        f"updated={checked.interval.updated_at.isoformat()}, "
        f"closed={closed_at}\n"
        f"events: captured={checked.counts.captured_events}, "
        f"actions={checked.counts.action_identities}, "
        f"results={checked.counts.structured_results}, "
        f"signals={checked.counts.detected_signals}, "
        f"ignored={checked.counts.ignored_records}\n"
        f"detectors: {detector_summary}\n"
        f"spool: {checked.coverage.spool_status.value}; "
        f"integrity: {checked.coverage.spool_integrity}; limits: {limit_summary}\n"
        f"capability exclusions: {exclusions}\n"
        "authority: source=none_same_user_untrusted; "
        "integrity=hmac_sha256_local_mutation_detection; rollback=none; "
        "timestamps=local_observation; sequence=local_receipt_order\n"
        "evidence: descriptive_observational; complete coverage: false; "
        "raw content persisted: false; transcript read: false\n"
        "report integrity: sha256_canonical_body; decision authority: false; "
        "model calls: 0; confirmatory: false\n"
    )


__all__ = [
    "CAPTURE_SESSION_REPORT_SCHEMA_VERSION",
    "CaptureReportCounts",
    "CaptureReportCoverage",
    "CaptureReportDetector",
    "CaptureReportError",
    "CaptureReportHeadline",
    "CaptureReportHealthCount",
    "CaptureReportInterval",
    "CaptureReportLimit",
    "CaptureSessionReport",
    "CaptureSpoolReportStatus",
    "build_capture_session_report",
    "decode_capture_session_report",
    "encode_capture_session_report",
    "render_capture_session_report_human",
    "render_capture_session_report_json",
]
