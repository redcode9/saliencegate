"""Deterministic, authority-bounded projection of capture snapshots into Shadow."""

from __future__ import annotations

import hmac
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.capture.capabilities import (
    CapabilitySupport,
    CaptureCapabilityManifest,
    CaptureDetectorCapability,
    capture_capability_digest,
    capture_profile,
)
from saliencegate.capture.schema import (
    CaptureActionFinishedIntake,
    CaptureActionStartedIntake,
    CaptureControllerFailedIntake,
    CaptureEvent,
    CaptureIntake,
    CapturePermissionDeniedIntake,
    CaptureSessionFinishedIntake,
    CaptureSessionStartedIntake,
    CaptureSubagentFinishedIntake,
    CaptureSubagentStartedIntake,
    CaptureTurnFinishedIntake,
)
from saliencegate.capture.sessions import (
    CaptureSessionSnapshot,
    CaptureSnapshotEvent,
    verify_capture_session_snapshot,
)
from saliencegate.capture.store import CaptureSessionState
from saliencegate.domain import (
    SignalType,
    TraceEvent,
    canonical_json,
)
from saliencegate.domain.records import UUID4, Sha256Digest
from saliencegate.security import InstallationKey, Redactor
from saliencegate.shadow.inputs import (
    ShadowActionIdentityInput,
    ShadowControllerErrorInput,
    ShadowEventRef,
    ShadowFinishInput,
    ShadowInputRecord,
    ShadowStartInput,
    ShadowToolResultInput,
    derive_shadow_event_id,
    project_shadow_input,
)
from saliencegate.shadow.trace import ShadowTrace
from saliencegate.signals import (
    DetectionContext,
    DetectionStatus,
    DeterministicSignalExtractor,
    ExtractionReport,
    FingerprintUnavailableError,
    RepeatedActionDetector,
    RepeatedFailureDetector,
    RepetitionConfig,
    SignalDetector,
    TestFailureDetector,
    ToolErrorDetector,
    failure_fingerprint,
)

_LOGICAL_EPOCH = datetime(2000, 1, 1, tzinfo=UTC)
_RUN_ID_DOMAIN = b"saliencegate:capture:shadow-run-id:v1"
_NORMALIZATION_DOMAIN = b"saliencegate:capture:normalization:v1"
_REPETITION_WINDOW_EVENTS = 8
_SHARED_TOOL_OUTCOME_AUTHORITIES = frozenset(
    {
        "provider_claimed_tool_outcome",
        "tool_state_discriminator",
    }
)
_SUCCESS_ONLY_TOOL_OUTCOME_AUTHORITIES = frozenset({"confirmed_success_or_ambiguous_error"})
_CONTROLLER_OUTCOME_AUTHORITIES = frozenset(
    {
        "controller_failure_when_session_correlated",
        "provider_claimed_controller_failure",
    }
)


class CaptureNormalizationError(ValueError):
    """A content-free failure at the capture-to-Shadow boundary."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture normalization failed")


class CaptureNormalizationDiagnosticCode(StrEnum):
    """Content-free reasons why a capture record did not become Shadow evidence."""

    EVENT_NOT_PROJECTABLE = "event_not_projectable"
    CAPTURE_DISPOSITION_NOT_CAPTURED = "capture_disposition_not_captured"
    OUTCOME_UNAVAILABLE = "outcome_unavailable"
    OUTCOME_NOT_AUTHORIZED = "outcome_not_authorized"
    PERMISSION_NOT_AUTHORIZED = "permission_not_authorized"
    CONTROLLER_FAILURE_NOT_AUTHORIZED = "controller_failure_not_authorized"
    MISSING_CALL_PARENT = "missing_call_parent"
    FUTURE_CALL_PARENT = "future_call_parent"
    DUPLICATE_CALL_PARENT = "duplicate_call_parent"
    CONFLICTING_CALL_PARENT = "conflicting_call_parent"
    DUPLICATE_CALL_RESULT = "duplicate_call_result"
    INCOHERENT_SUBAGENT_LIFECYCLE = "incoherent_subagent_lifecycle"
    DUPLICATE_TURN_ID = "duplicate_turn_id"
    INVALID_SESSION_MARKER = "invalid_session_marker"
    SESSION_QUARANTINED = "session_quarantined"


CaptureEventKind = Literal[
    "session_started",
    "action_started",
    "action_finished",
    "permission_denied",
    "subagent_started",
    "subagent_finished",
    "turn_finished",
    "controller_failed",
    "session_finished",
]


class _NormalizationModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
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


class CaptureNormalizationDiagnostic(_NormalizationModel):
    """One ordinal-only normalization disposition without provider identifiers."""

    receipt_ordinal: Annotated[int | None, Field(ge=1, le=1_000)] = None
    event_kind: CaptureEventKind | None = None
    code: CaptureNormalizationDiagnosticCode

    @model_validator(mode="after")
    def event_coordinate_is_complete(self) -> Self:
        if (self.receipt_ordinal is None) != (self.event_kind is None):
            raise ValueError("capture normalization diagnostic coordinate is incomplete")
        return self


class CaptureNormalizationCounts(_NormalizationModel):
    """Exhaustive source and evidence counts used by report sufficiency checks."""

    source_event_count: Annotated[int, Field(ge=0, le=1_000)]
    mapped_event_count: Annotated[int, Field(ge=0, le=1_000)]
    ignored_event_count: Annotated[int, Field(ge=0, le=1_000)]
    action_identity_count: Annotated[int, Field(ge=0, le=1_000)]
    exact_action_identity_count: Annotated[int, Field(ge=0, le=1_000)]
    authorized_tool_result_count: Annotated[int, Field(ge=0, le=1_000)]
    classifiable_failed_result_count: Annotated[int, Field(ge=0, le=1_000)]
    exact_parent_classifiable_failed_result_count: Annotated[
        int,
        Field(ge=0, le=1_000),
    ]
    authorized_controller_error_count: Annotated[int, Field(ge=0, le=1_000)]

    @model_validator(mode="after")
    def count_equations_hold(self) -> Self:
        if (
            self.source_event_count != self.mapped_event_count + self.ignored_event_count
            or self.exact_action_identity_count > self.action_identity_count
            or self.classifiable_failed_result_count > self.authorized_tool_result_count
            or self.exact_parent_classifiable_failed_result_count
            > self.classifiable_failed_result_count
            or self.action_identity_count
            + self.authorized_tool_result_count
            + self.authorized_controller_error_count
            > self.mapped_event_count
        ):
            raise ValueError("capture normalization counts are inconsistent")
        return self


class CaptureDetectorEvidence(_NormalizationModel):
    """Manifest-selected detector support and its explicit absence threshold."""

    signal_type: SignalType
    support: CapabilitySupport
    omissions: Annotated[tuple[str, ...], Field(max_length=32)]
    minimum_authorized_observations: Annotated[int, Field(ge=1, le=2)]
    authorized_observation_count: Annotated[int, Field(ge=0, le=1_000)]
    unresolved_observation_count: Annotated[int, Field(ge=0, le=1_000)]
    minimum_observation_met: bool

    @model_validator(mode="after")
    def threshold_is_exact(self) -> Self:
        if self.support is CapabilitySupport.UNSUPPORTED:
            raise ValueError("unsupported capture detector cannot carry evidence")
        if (
            self.authorized_observation_count + self.unresolved_observation_count > 1_000
            or self.minimum_observation_met
            != (
                self.authorized_observation_count >= self.minimum_authorized_observations
                and self.unresolved_observation_count == 0
            )
        ):
            raise ValueError("capture detector evidence threshold is inconsistent")
        if self.omissions != tuple(sorted(set(self.omissions))):
            raise ValueError("capture detector omissions are not canonical")
        return self


class CaptureNormalization(_NormalizationModel):
    """A replay-stable, key-bound Shadow projection of one verified snapshot."""

    schema_version: Literal["capture-normalization/v1"] = "capture-normalization/v1"
    snapshot_digest: Annotated[Sha256Digest, Field(repr=False)]
    run_id: Annotated[UUID4, Field(repr=False)]
    shadow_trace: Annotated[ShadowTrace | None, Field(exclude=True, repr=False)]
    events: Annotated[tuple[TraceEvent, ...], Field(max_length=1_000, repr=False)]
    extraction_reports: Annotated[
        tuple[ExtractionReport, ...],
        Field(max_length=1_000, repr=False),
    ]
    detector_evidence: Annotated[
        tuple[CaptureDetectorEvidence, ...],
        Field(max_length=len(SignalType)),
    ]
    diagnostics: Annotated[
        tuple[CaptureNormalizationDiagnostic, ...],
        Field(max_length=1_002),
    ]
    counts: CaptureNormalizationCounts
    semantic_coherence: bool
    normalization_digest: Annotated[Sha256Digest, Field(repr=False)]

    @model_validator(mode="after")
    def projection_is_consistent(self) -> Self:
        if (
            len(self.events) != self.counts.mapped_event_count
            or len(self.extraction_reports) != len(self.events)
            or len(self.diagnostics) > self.counts.source_event_count + 2
            or (self.shadow_trace is None) != (not self.events)
            or tuple(item.signal_type for item in self.detector_evidence)
            != tuple(
                sorted(
                    {item.signal_type for item in self.detector_evidence},
                    key=lambda item: item.value,
                )
            )
        ):
            raise ValueError("capture normalization projection is inconsistent")
        for sequence, (event, report) in enumerate(
            zip(self.events, self.extraction_reports, strict=True),
            start=1,
        ):
            if (
                event.run_id != self.run_id
                or event.sequence != sequence
                or report.run_id != self.run_id
                or report.current_event_id != event.event_id
            ):
                raise ValueError("capture normalization event sequence is inconsistent")
            if tuple(item.signal_type for item in report.evaluations) != tuple(
                item.signal_type for item in self.detector_evidence
            ):
                raise ValueError("capture normalization detector selection is inconsistent")
        return self


@dataclass(frozen=True, slots=True, repr=False)
class _ProjectedRecord:
    receipt_ordinal: int
    event_kind: CaptureEventKind
    value: ShadowInputRecord
    wire: dict[str, object]
    marker: Literal["start", "finish"] | None = None
    exact_action: bool = False
    authorized_tool_result: bool = False
    authorized_controller_error: bool = False


def _run_id(snapshot_digest: str, key: InstallationKey) -> UUID:
    digest = key._hmac_sha256(snapshot_digest.encode("ascii"), domain=_RUN_ID_DOMAIN)
    raw = bytearray(bytes.fromhex(digest)[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(raw))


def _logical_time(receipt_ordinal: int) -> datetime:
    return _LOGICAL_EPOCH + timedelta(microseconds=receipt_ordinal)


def _wire_time(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _source_id(receipt_ordinal: int, suffix: str) -> str:
    return f"capture-r{receipt_ordinal:04d}-{suffix}"


def _diagnostic(
    item: CaptureSnapshotEvent,
    code: CaptureNormalizationDiagnosticCode,
) -> CaptureNormalizationDiagnostic:
    return CaptureNormalizationDiagnostic(
        receipt_ordinal=item.receipt_ordinal,
        event_kind=item.event.intake.kind,
        code=code,
    )


def _action_identity(
    intake: CaptureActionStartedIntake,
) -> tuple[str, str, str, str]:
    return (
        intake.action_digest,
        intake.workspace_digest,
        intake.environment_digest,
        intake.identity_authority,
    )


def _result_call_ref(intake: CaptureIntake) -> str | None:
    if type(intake) is CaptureActionFinishedIntake:
        return intake.call_ref
    if type(intake) is CapturePermissionDeniedIntake:
        return intake.call_ref
    return None


def _manifest_authorities(manifest: CaptureCapabilityManifest) -> frozenset[str]:
    return frozenset(event.outcome_authority for event in manifest.events)


def _structured_status_is_authorized(
    status: Literal["succeeded", "failed"] | None,
    authorities: frozenset[str],
) -> bool:
    if status == "succeeded":
        return bool(
            authorities
            & (
                _SHARED_TOOL_OUTCOME_AUTHORITIES
                | _SUCCESS_ONLY_TOOL_OUTCOME_AUTHORITIES
                | {"provider_claimed_success"}
            )
        )
    if status == "failed":
        return bool(authorities & (_SHARED_TOOL_OUTCOME_AUTHORITIES | {"provider_claimed_failure"}))
    return False


def _authorized_structured_error_code(
    status: Literal["succeeded", "failed"] | None,
    authorities: frozenset[str],
) -> Literal["provider_error", "tool_error"] | None:
    """Reduce an audited structured failure to its profile-authorized generic class."""

    if status != "failed":
        return None
    if "provider_claimed_failure" in authorities:
        return "provider_error"
    if authorities & _SHARED_TOOL_OUTCOME_AUTHORITIES:
        return "tool_error"
    raise CaptureNormalizationError()


def _correlation_preflight(
    snapshot: CaptureSessionSnapshot,
) -> tuple[
    dict[str, tuple[CaptureSnapshotEvent, ...]],
    dict[int, CaptureNormalizationDiagnosticCode],
]:
    action_groups: dict[str, list[CaptureSnapshotEvent]] = defaultdict(list)
    result_groups: dict[str, list[CaptureSnapshotEvent]] = defaultdict(list)
    subagent_starts: dict[str, list[CaptureSnapshotEvent]] = defaultdict(list)
    subagent_finishes: dict[str, list[CaptureSnapshotEvent]] = defaultdict(list)
    turn_groups: dict[str, list[CaptureSnapshotEvent]] = defaultdict(list)
    for item in snapshot.events:
        intake = item.event.intake
        result_call_ref = _result_call_ref(intake)
        if type(intake) is CaptureActionStartedIntake:
            action_groups[intake.call_ref].append(item)
        elif result_call_ref is not None:
            result_groups[result_call_ref].append(item)
        elif type(intake) is CaptureSubagentStartedIntake:
            subagent_starts[intake.subagent_id].append(item)
        elif type(intake) is CaptureSubagentFinishedIntake:
            subagent_finishes[intake.subagent_id].append(item)
        elif type(intake) is CaptureTurnFinishedIntake:
            turn_groups[intake.turn_id].append(item)

    diagnostics: dict[int, CaptureNormalizationDiagnosticCode] = {}
    for candidates in action_groups.values():
        if len(candidates) < 2:
            continue
        identities = {
            _action_identity(candidate.event.intake)
            for candidate in candidates
            if type(candidate.event.intake) is CaptureActionStartedIntake
        }
        code = (
            CaptureNormalizationDiagnosticCode.DUPLICATE_CALL_PARENT
            if len(identities) == 1
            else CaptureNormalizationDiagnosticCode.CONFLICTING_CALL_PARENT
        )
        diagnostics.update((item.receipt_ordinal, code) for item in candidates)

    for results in result_groups.values():
        if len(results) > 1:
            diagnostics.update(
                (
                    item.receipt_ordinal,
                    CaptureNormalizationDiagnosticCode.DUPLICATE_CALL_RESULT,
                )
                for item in results
            )

    for subagent_id in subagent_starts.keys() | subagent_finishes.keys():
        starts = subagent_starts.get(subagent_id, ())
        finishes = subagent_finishes.get(subagent_id, ())
        invalid = (
            len(starts) > 1
            or len(finishes) > 1
            or (bool(finishes) and not starts)
            or (
                len(starts) == len(finishes) == 1
                and starts[0].receipt_ordinal >= finishes[0].receipt_ordinal
            )
            or (snapshot.state is CaptureSessionState.CLOSED and bool(starts) and not finishes)
        )
        if invalid:
            diagnostics.update(
                (
                    item.receipt_ordinal,
                    CaptureNormalizationDiagnosticCode.INCOHERENT_SUBAGENT_LIFECYCLE,
                )
                for item in (*starts, *finishes)
            )

    for turns in turn_groups.values():
        if len(turns) > 1:
            diagnostics.update(
                (
                    item.receipt_ordinal,
                    CaptureNormalizationDiagnosticCode.DUPLICATE_TURN_ID,
                )
                for item in turns
            )
    return (
        {key: tuple(value) for key, value in action_groups.items()},
        diagnostics,
    )


def _selected_detector_capabilities(
    manifest: CaptureCapabilityManifest,
) -> tuple[CaptureDetectorCapability, ...]:
    selected = tuple(
        detector
        for detector in manifest.detectors
        if detector.support is not CapabilitySupport.UNSUPPORTED
    )
    return tuple(sorted(selected, key=lambda item: item.signal_type.value))


def _extractor(
    selected: tuple[CaptureDetectorCapability, ...],
) -> DeterministicSignalExtractor:
    repetition = RepetitionConfig(window_events=_REPETITION_WINDOW_EVENTS)
    installed: dict[SignalType, SignalDetector] = {
        SignalType.REPEATED_ACTION: RepeatedActionDetector(repetition),
        SignalType.REPEATED_FAILURE: RepeatedFailureDetector(repetition),
        SignalType.TEST_FAILURE: TestFailureDetector(),
        SignalType.TOOL_ERROR: ToolErrorDetector(),
    }
    detectors: list[SignalDetector] = []
    for capability in selected:
        detector = installed.get(capability.signal_type)
        if detector is None:
            raise CaptureNormalizationError()
        detectors.append(detector)
    return DeterministicSignalExtractor(tuple(detectors))


def _parent_code(
    item: CaptureSnapshotEvent,
    candidates: tuple[CaptureSnapshotEvent, ...],
) -> CaptureNormalizationDiagnosticCode | None:
    if not candidates:
        return CaptureNormalizationDiagnosticCode.MISSING_CALL_PARENT
    if len(candidates) > 1:
        identities = {
            _action_identity(candidate.event.intake)
            for candidate in candidates
            if type(candidate.event.intake) is CaptureActionStartedIntake
        }
        return (
            CaptureNormalizationDiagnosticCode.DUPLICATE_CALL_PARENT
            if len(identities) == 1
            else CaptureNormalizationDiagnosticCode.CONFLICTING_CALL_PARENT
        )
    if candidates[0].receipt_ordinal >= item.receipt_ordinal:
        return CaptureNormalizationDiagnosticCode.FUTURE_CALL_PARENT
    return None


def _wire_common(
    *,
    source_event_id: str,
    occurred_at: datetime,
    kind: str,
) -> dict[str, object]:
    return {
        "schema_version": "shadow-input/v1",
        "kind": kind,
        "source_event_id": source_event_id,
        "occurred_at": _wire_time(occurred_at),
    }


def _project_capture_events(
    snapshot: CaptureSessionSnapshot,
    *,
    manifest: CaptureCapabilityManifest,
    run_id: UUID,
) -> tuple[
    tuple[_ProjectedRecord, ...],
    tuple[CaptureNormalizationDiagnostic, ...],
    bool,
]:
    frozen_actions, preflight_diagnostics = _correlation_preflight(snapshot)
    authorities = _manifest_authorities(manifest)
    permission_authorized = "provider_claimed_denial" in authorities
    controller_authorized = bool(authorities & _CONTROLLER_OUTCOME_AUTHORITIES)

    projected: list[_ProjectedRecord] = []
    diagnostics: list[CaptureNormalizationDiagnostic] = []
    action_refs: dict[int, ShadowEventRef] = {}
    coherent = snapshot.state is not CaptureSessionState.QUARANTINED and not preflight_diagnostics
    if snapshot.state is CaptureSessionState.QUARANTINED:
        diagnostics.append(
            CaptureNormalizationDiagnostic(
                code=CaptureNormalizationDiagnosticCode.SESSION_QUARANTINED,
            )
        )

    value: ShadowInputRecord
    for item in snapshot.events:
        event: CaptureEvent = item.event
        intake = event.intake
        preflight_code = preflight_diagnostics.get(item.receipt_ordinal)
        if preflight_code is not None:
            diagnostics.append(_diagnostic(item, preflight_code))
            continue
        if intake.capture_disposition != "captured":
            if preflight_code is None:
                diagnostics.append(
                    _diagnostic(
                        item,
                        CaptureNormalizationDiagnosticCode.CAPTURE_DISPOSITION_NOT_CAPTURED,
                    )
                )
            continue
        timestamp = _logical_time(item.receipt_ordinal)

        if type(intake) is CaptureSessionStartedIntake:
            source_id = _source_id(item.receipt_ordinal, "start")
            value = ShadowStartInput(source_event_id=source_id, occurred_at=timestamp)
            projected.append(
                _ProjectedRecord(
                    item.receipt_ordinal,
                    intake.kind,
                    value,
                    _wire_common(
                        source_event_id=source_id,
                        occurred_at=timestamp,
                        kind="run_start",
                    ),
                    marker="start",
                )
            )
            continue

        if type(intake) is CaptureActionStartedIntake:
            source_id = _source_id(item.receipt_ordinal, "action")
            value = ShadowActionIdentityInput(
                source_event_id=source_id,
                occurred_at=timestamp,
                action_digest=intake.action_digest,
                workspace_digest=intake.workspace_digest,
                environment_digest=intake.environment_digest,
                identity_authority=intake.identity_authority,
            )
            projected.append(
                _ProjectedRecord(
                    item.receipt_ordinal,
                    intake.kind,
                    value,
                    {
                        **_wire_common(
                            source_event_id=source_id,
                            occurred_at=timestamp,
                            kind="action_identity",
                        ),
                        "action_digest": intake.action_digest,
                        "workspace_digest": intake.workspace_digest,
                        "environment_digest": intake.environment_digest,
                        "identity_authority": intake.identity_authority,
                    },
                    exact_action=intake.identity_authority == "exact",
                )
            )
            action_refs[item.receipt_ordinal] = ShadowEventRef(
                run_id=run_id,
                event_id=derive_shadow_event_id(run_id, source_id),
                sequence=len(projected),
            )
            continue

        call_ref = _result_call_ref(intake)
        if call_ref is not None:
            candidates = frozen_actions.get(call_ref, ())
            parent_problem = _parent_code(item, candidates)
            if parent_problem is not None:
                diagnostics.append(_diagnostic(item, parent_problem))
                coherent = False
                continue
            parent_item = candidates[0]
            parent_ref = action_refs.get(parent_item.receipt_ordinal)
            if parent_ref is None:
                diagnostics.append(
                    _diagnostic(
                        item,
                        CaptureNormalizationDiagnosticCode.MISSING_CALL_PARENT,
                    )
                )
                coherent = False
                continue
            status: Literal["succeeded", "failed"] | None
            error_code: (
                Literal[
                    "permission_denied",
                    "provider_error",
                    "tool_error",
                ]
                | None
            )
            if type(intake) is CaptureActionFinishedIntake:
                if intake.outcome_authority == "unavailable":
                    diagnostics.append(
                        _diagnostic(
                            item,
                            CaptureNormalizationDiagnosticCode.OUTCOME_UNAVAILABLE,
                        )
                    )
                    continue
                if not _structured_status_is_authorized(
                    intake.outcome_status,
                    authorities,
                ):
                    diagnostics.append(
                        _diagnostic(
                            item,
                            CaptureNormalizationDiagnosticCode.OUTCOME_NOT_AUTHORIZED,
                        )
                    )
                    coherent = False
                    continue
                status = intake.outcome_status
                error_code = _authorized_structured_error_code(status, authorities)
            elif type(intake) is CapturePermissionDeniedIntake:
                if not permission_authorized:
                    diagnostics.append(
                        _diagnostic(
                            item,
                            CaptureNormalizationDiagnosticCode.PERMISSION_NOT_AUTHORIZED,
                        )
                    )
                    continue
                status = "failed"
                error_code = "permission_denied"
            else:  # pragma: no cover - guarded by the exact call_ref branches
                raise CaptureNormalizationError()

            source_id = _source_id(item.receipt_ordinal, "result")
            value = ShadowToolResultInput(
                source_event_id=source_id,
                occurred_at=timestamp,
                action=parent_ref,
                status=status,
                error_code=error_code,
            )
            wire = {
                **_wire_common(
                    source_event_id=source_id,
                    occurred_at=timestamp,
                    kind="tool_result",
                ),
                "action_source_event_id": projected[parent_ref.sequence - 1].value.source_event_id,
            }
            if status is not None:
                wire["status"] = status
            if error_code is not None:
                wire["error_code"] = error_code
            projected.append(
                _ProjectedRecord(
                    item.receipt_ordinal,
                    intake.kind,
                    value,
                    wire,
                    authorized_tool_result=True,
                )
            )
            continue

        if type(intake) is CaptureControllerFailedIntake:
            if intake.error_code != "provider_callback_failed" or not controller_authorized:
                diagnostics.append(
                    _diagnostic(
                        item,
                        CaptureNormalizationDiagnosticCode.CONTROLLER_FAILURE_NOT_AUTHORIZED,
                    )
                )
                continue
            source_id = _source_id(item.receipt_ordinal, "controller")
            value = ShadowControllerErrorInput(
                source_event_id=source_id,
                occurred_at=timestamp,
                error_code="provider_callback_failed",
            )
            projected.append(
                _ProjectedRecord(
                    item.receipt_ordinal,
                    intake.kind,
                    value,
                    {
                        **_wire_common(
                            source_event_id=source_id,
                            occurred_at=timestamp,
                            kind="controller_error",
                        ),
                        "error_code": "provider_callback_failed",
                    },
                    authorized_controller_error=True,
                )
            )
            continue

        if type(intake) is CaptureSessionFinishedIntake:
            source_id = _source_id(item.receipt_ordinal, "finish")
            value = ShadowFinishInput(source_event_id=source_id, occurred_at=timestamp)
            projected.append(
                _ProjectedRecord(
                    item.receipt_ordinal,
                    intake.kind,
                    value,
                    _wire_common(
                        source_event_id=source_id,
                        occurred_at=timestamp,
                        kind="run_end",
                    ),
                    marker="finish",
                )
            )
            continue

        diagnostics.append(
            _diagnostic(
                item,
                CaptureNormalizationDiagnosticCode.EVENT_NOT_PROJECTABLE,
            )
        )

    valid_markers = bool(projected) and projected[0].marker == "start"
    valid_markers = valid_markers and all(item.marker != "start" for item in projected[1:])
    finish_positions = tuple(
        index for index, item in enumerate(projected) if item.marker == "finish"
    )
    valid_markers = valid_markers and (
        not finish_positions or finish_positions == (len(projected) - 1,)
    )
    if projected and not valid_markers:
        diagnostics.append(
            CaptureNormalizationDiagnostic(
                code=CaptureNormalizationDiagnosticCode.INVALID_SESSION_MARKER,
            )
        )
        projected = []
        coherent = False
    return tuple(projected), tuple(diagnostics), coherent


def _trace(
    projected: tuple[_ProjectedRecord, ...],
    *,
    snapshot: CaptureSessionSnapshot,
    run_id: UUID,
) -> ShadowTrace | None:
    if not projected:
        return None
    descriptor: dict[str, object] = {
        "schema_version": "capture-shadow-adapter/v1",
        "profile_id": snapshot.profile_id.value,
        "projection": "opaque-action-identity/v1",
    }
    return ShadowTrace.from_records(
        (dict(item.wire) for item in projected),
        run_id=run_id,
        adapter_profile_id=f"capture-{snapshot.profile_id.value}",
        adapter_descriptor=descriptor,
        timestamp_mode="logical_order",
        capture_scope="bounded_window",
        capture_manifest_digest=snapshot.capability_manifest_digest,
    )


def _trace_events(
    projected: tuple[_ProjectedRecord, ...],
    *,
    trace: ShadowTrace | None,
    run_id: UUID,
    installation_key: InstallationKey,
) -> tuple[TraceEvent, ...]:
    if trace is None:
        return ()
    redactor = Redactor()
    materialized: list[TraceEvent] = []
    marker_payload = {
        "schema_version": "capture-shadow-marker/v1",
        "capture_scope": "bounded_window",
    }
    for sequence, item in enumerate(projected, start=1):
        draft = project_shadow_input(
            item.value,
            run_id=run_id,
            source_adapter=trace.binding.source_adapter,
            start_payload=marker_payload if item.marker == "start" else None,
            finish_payload=marker_payload if item.marker == "finish" else None,
        )
        redacted = redactor.redact_event(draft, key=installation_key)
        if redacted.findings:
            raise CaptureNormalizationError()
        values = redacted.event.model_dump(mode="python", warnings="error")
        values.update(
            record_type="trace_event",
            event_id=derive_shadow_event_id(run_id, item.value.source_event_id),
            sequence=sequence,
        )
        materialized.append(TraceEvent.model_validate(values))
    return tuple(materialized)


def _extraction_reports(
    events: tuple[TraceEvent, ...],
    *,
    extractor: DeterministicSignalExtractor,
    run_id: UUID,
) -> tuple[ExtractionReport, ...]:
    return tuple(
        extractor.extract_report(DetectionContext(run_id=run_id, events=events[:end_ordinal]))
        for end_ordinal in range(1, len(events) + 1)
    )


def _failed_result_counts(
    projected: tuple[_ProjectedRecord, ...],
    events: tuple[TraceEvent, ...],
) -> tuple[int, int]:
    classifiable = 0
    exact_parent = 0
    exact_actions = frozenset(
        event.event_id for item, event in zip(projected, events, strict=True) if item.exact_action
    )
    for item, event in zip(projected, events, strict=True):
        if not _is_classifiable_failed_result(item, event):
            continue
        classifiable += 1
        if _has_exact_action_parent(event, exact_actions):
            exact_parent += 1
    return classifiable, exact_parent


def _is_classifiable_failed_result(
    item: _ProjectedRecord,
    event: TraceEvent,
) -> bool:
    if not item.authorized_tool_result:
        return False
    try:
        failure_fingerprint(event)
    except FingerprintUnavailableError:
        return False
    return True


def _has_exact_action_parent(
    event: TraceEvent,
    exact_actions: frozenset[UUID],
) -> bool:
    return len(event.parent_ids) == 1 and event.parent_ids[0] in exact_actions


def _counts(
    snapshot: CaptureSessionSnapshot,
    projected: tuple[_ProjectedRecord, ...],
    events: tuple[TraceEvent, ...],
) -> CaptureNormalizationCounts:
    failed, exact_failed = _failed_result_counts(projected, events)
    projected_counts = Counter(item.event_kind for item in projected)
    return CaptureNormalizationCounts(
        source_event_count=len(snapshot.events),
        mapped_event_count=len(projected),
        ignored_event_count=len(snapshot.events) - len(projected),
        action_identity_count=projected_counts["action_started"],
        exact_action_identity_count=sum(item.exact_action for item in projected),
        authorized_tool_result_count=sum(item.authorized_tool_result for item in projected),
        classifiable_failed_result_count=failed,
        exact_parent_classifiable_failed_result_count=exact_failed,
        authorized_controller_error_count=sum(
            item.authorized_controller_error for item in projected
        ),
    )


def _capture_detector_minimum(signal_type: SignalType) -> int:
    """Return the closed v1 absence threshold for one installed capture detector."""

    if type(signal_type) is not SignalType:
        raise CaptureNormalizationError()
    if signal_type in (SignalType.REPEATED_ACTION, SignalType.REPEATED_FAILURE):
        return 2
    if signal_type in (SignalType.TOOL_ERROR, SignalType.TEST_FAILURE):
        return 1
    raise CaptureNormalizationError()


def _detector_record_is_eligible(
    signal_type: SignalType,
    item: _ProjectedRecord,
    event: TraceEvent,
    *,
    exact_actions: frozenset[UUID],
) -> bool:
    if signal_type is SignalType.REPEATED_ACTION:
        return item.exact_action
    if signal_type is SignalType.REPEATED_FAILURE:
        return _is_classifiable_failed_result(item, event) and _has_exact_action_parent(
            event,
            exact_actions,
        )
    if signal_type is SignalType.TOOL_ERROR:
        return item.authorized_tool_result
    if signal_type is SignalType.TEST_FAILURE:
        return False
    raise CaptureNormalizationError()


def _detector_observation_counts(
    signal_type: SignalType,
    projected: tuple[_ProjectedRecord, ...],
    events: tuple[TraceEvent, ...],
    extraction_reports: tuple[ExtractionReport, ...],
) -> tuple[int, int]:
    exact_actions = frozenset(
        event.event_id for item, event in zip(projected, events, strict=True) if item.exact_action
    )
    observed = 0
    unresolved = 0
    for item, event, report in zip(
        projected,
        events,
        extraction_reports,
        strict=True,
    ):
        if not _detector_record_is_eligible(
            signal_type,
            item,
            event,
            exact_actions=exact_actions,
        ):
            continue
        evaluations = tuple(
            evaluation for evaluation in report.evaluations if evaluation.signal_type is signal_type
        )
        if len(evaluations) != 1:
            raise CaptureNormalizationError()
        if evaluations[0].outcome.status in (
            DetectionStatus.NO_MATCH,
            DetectionStatus.DETECTED,
        ):
            observed += 1
        elif evaluations[0].outcome.status is DetectionStatus.ABSTAINED:
            unresolved += 1
        else:  # pragma: no cover - DetectionStatus is a closed enum
            raise CaptureNormalizationError()
    return observed, unresolved


def _detector_evidence(
    selected: tuple[CaptureDetectorCapability, ...],
    projected: tuple[_ProjectedRecord, ...],
    events: tuple[TraceEvent, ...],
    extraction_reports: tuple[ExtractionReport, ...],
) -> tuple[CaptureDetectorEvidence, ...]:
    result: list[CaptureDetectorEvidence] = []
    for capability in selected:
        minimum = _capture_detector_minimum(capability.signal_type)
        observed, unresolved = _detector_observation_counts(
            capability.signal_type,
            projected,
            events,
            extraction_reports,
        )
        result.append(
            CaptureDetectorEvidence(
                signal_type=capability.signal_type,
                support=capability.support,
                omissions=capability.omissions,
                minimum_authorized_observations=minimum,
                authorized_observation_count=observed,
                unresolved_observation_count=unresolved,
                minimum_observation_met=observed >= minimum and unresolved == 0,
            )
        )
    return tuple(result)


def _normalization_preimage(
    *,
    snapshot_digest: str,
    run_id: UUID,
    trace: ShadowTrace | None,
    events: tuple[TraceEvent, ...],
    extraction_reports: tuple[ExtractionReport, ...],
    detector_evidence: tuple[CaptureDetectorEvidence, ...],
    diagnostics: tuple[CaptureNormalizationDiagnostic, ...],
    counts: CaptureNormalizationCounts,
    semantic_coherence: bool,
) -> bytes:
    return canonical_json(
        {
            "schema_version": "capture-normalization-integrity/v1",
            "snapshot_digest": snapshot_digest,
            "run_id": str(run_id),
            "trace_binding_digest": (None if trace is None else trace.binding.binding_digest),
            "trace_record_digest": (None if trace is None else trace.mapped_record_digest),
            "events": tuple(event.model_dump(mode="json", warnings="error") for event in events),
            "extraction_reports": tuple(
                report.model_dump(mode="json", warnings="error") for report in extraction_reports
            ),
            "detector_evidence": tuple(
                item.model_dump(mode="json", warnings="error") for item in detector_evidence
            ),
            "diagnostics": tuple(
                item.model_dump(mode="json", warnings="error") for item in diagnostics
            ),
            "counts": counts.model_dump(mode="json", warnings="error"),
            "semantic_coherence": semantic_coherence,
        }
    )


def normalize_capture_session_snapshot(
    snapshot: object,
    *,
    installation_key: InstallationKey,
) -> CaptureNormalization:
    """Verify and deterministically project one session snapshot into Shadow evidence."""

    result: CaptureNormalization | None = None
    try:
        verified = verify_capture_session_snapshot(
            snapshot,
            installation_key=installation_key,
        )
        manifest = capture_profile(verified.profile_id)
        if not hmac.compare_digest(
            capture_capability_digest(manifest),
            verified.capability_manifest_digest,
        ):
            raise CaptureNormalizationError()
        run_id = _run_id(verified.snapshot_digest, installation_key)
        projected, diagnostics, coherent = _project_capture_events(
            verified,
            manifest=manifest,
            run_id=run_id,
        )
        trace = _trace(projected, snapshot=verified, run_id=run_id)
        events = _trace_events(
            projected,
            trace=trace,
            run_id=run_id,
            installation_key=installation_key,
        )
        selected = _selected_detector_capabilities(manifest)
        extractor = _extractor(selected)
        extraction_reports = _extraction_reports(
            events,
            extractor=extractor,
            run_id=run_id,
        )
        counts = _counts(verified, projected, events)
        detector_evidence = _detector_evidence(
            selected,
            projected,
            events,
            extraction_reports,
        )
        preimage = _normalization_preimage(
            snapshot_digest=verified.snapshot_digest,
            run_id=run_id,
            trace=trace,
            events=events,
            extraction_reports=extraction_reports,
            detector_evidence=detector_evidence,
            diagnostics=diagnostics,
            counts=counts,
            semantic_coherence=coherent,
        )
        digest = installation_key._hmac_sha256(
            preimage,
            domain=_NORMALIZATION_DOMAIN,
        )
        result = CaptureNormalization(
            snapshot_digest=verified.snapshot_digest,
            run_id=run_id,
            shadow_trace=trace,
            events=events,
            extraction_reports=extraction_reports,
            detector_evidence=detector_evidence,
            diagnostics=diagnostics,
            counts=counts,
            semantic_coherence=coherent,
            normalization_digest=digest,
        )
    except CaptureNormalizationError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        result = None
    if result is None:
        raise CaptureNormalizationError()
    return result


def _normalization_identity(value: CaptureNormalization) -> bytes:
    trace = value.shadow_trace
    return canonical_json(
        {
            "normalization": value.model_dump(mode="json", warnings="error"),
            "trace_binding_digest": (None if trace is None else trace.binding.binding_digest),
            "trace_record_digest": (None if trace is None else trace.mapped_record_digest),
        }
    )


def verify_capture_normalization(
    normalization: object,
    *,
    snapshot: object,
    installation_key: InstallationKey,
) -> CaptureNormalization:
    """Recompute and defensively copy a snapshot-bound normalization."""

    expected: CaptureNormalization | None = None
    try:
        if type(normalization) is not CaptureNormalization:
            raise CaptureNormalizationError()
        expected = normalize_capture_session_snapshot(
            snapshot,
            installation_key=installation_key,
        )
        if not hmac.compare_digest(
            normalization.normalization_digest,
            expected.normalization_digest,
        ) or not hmac.compare_digest(
            _normalization_identity(normalization),
            _normalization_identity(expected),
        ):
            raise CaptureNormalizationError()
    except CaptureNormalizationError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        expected = None
    if expected is None:
        raise CaptureNormalizationError()
    return expected


__all__ = [
    "CaptureDetectorEvidence",
    "CaptureNormalization",
    "CaptureNormalizationCounts",
    "CaptureNormalizationDiagnostic",
    "CaptureNormalizationDiagnosticCode",
    "CaptureNormalizationError",
    "normalize_capture_session_snapshot",
    "verify_capture_normalization",
]
