from __future__ import annotations

import hmac
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Annotated, Literal, Self, TypeAlias, TypeVar
from uuid import UUID
from weakref import WeakKeyDictionary

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from saliencegate.domain import (
    EventPhase,
    EventType,
    PayloadDigest,
    PayloadDigestAlgorithm,
    SignalType,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.domain.records import UUID4, Sha256Digest
from saliencegate.shadow.errors import ShadowInvariantError
from saliencegate.shadow.evaluation import ShadowHeuristicDisposition
from saliencegate.shadow.inputs import (
    SHADOW_PROJECTION_MATRIX,
    ShadowInputKind,
)
from saliencegate.shadow.observation import ShadowObservation
from saliencegate.signals import AbstentionReason, DetectionStatus

SHADOW_REPORT_ROW_SCHEMA_VERSION: Literal["shadow-report-row/v1"] = "shadow-report-row/v1"
SHADOW_RUN_REPORT_SCHEMA_VERSION: Literal["shadow-run-report/v1"] = "shadow-run-report/v1"

_RUN_REPORT_DIGEST_DOMAIN = "saliencegate:shadow:run-report:v1"
_TRUSTED_REPORT_TOKEN = object()


class _TrustedReportSeal:
    __slots__ = ("__weakref__",)


_TRUSTED_REPORT_SEALS: WeakKeyDictionary[_TrustedReportSeal, tuple[object, ...]] = (
    WeakKeyDictionary()
)
_MAX_INPUT_ROWS = 10_000
_MAX_SIGNED_64 = (1 << 63) - 1

_SUPPORTED_SIGNAL_TYPES = (
    SignalType.REPEATED_ACTION,
    SignalType.REPEATED_FAILURE,
    SignalType.TEST_FAILURE,
    SignalType.TOOL_ERROR,
)
_UNSUPPORTED_SIGNAL_TYPES = (
    SignalType.CONFLICT,
    SignalType.CONTEXT_SHIFT,
    SignalType.IRREVERSIBLE_ACTION,
    SignalType.STAGNATION,
    SignalType.STALE_CONSTRAINT,
)
_DETECTION_STATUSES = tuple(sorted(DetectionStatus, key=lambda item: item.value))
_ABSTENTION_REASONS = tuple(sorted(AbstentionReason, key=lambda item: item.value))
_HEURISTIC_DISPOSITIONS = tuple(sorted(ShadowHeuristicDisposition, key=lambda item: item.value))
_SHADOW_EVENT_TYPES = tuple(
    sorted(
        {projection.event_type for projection in SHADOW_PROJECTION_MATRIX.values()},
        key=lambda item: item.value,
    )
)
_SHADOW_PHASES = tuple(
    sorted(
        {projection.phase for projection in SHADOW_PROJECTION_MATRIX.values()},
        key=lambda item: item.value,
    )
)
_SIGNAL_PAIRS = tuple(combinations(_SUPPORTED_SIGNAL_TYPES, 2))

_ModelT = TypeVar("_ModelT", bound=BaseModel)
_PositiveOrdinal = Annotated[int, Field(ge=1, le=_MAX_SIGNED_64)]
_NonNegativeCount = Annotated[int, Field(ge=0, le=_MAX_SIGNED_64)]
_CaptureScope: TypeAlias = Literal[
    "unknown",
    "selected_events",
    "bounded_window",
    "complete_run_declared",
]
_PersistenceDisposition: TypeAlias = Literal["appended", "preexisting"]
_DetectorOutcomeCount: TypeAlias = tuple[SignalType, DetectionStatus, _NonNegativeCount]
_AbstentionReasonCount: TypeAlias = tuple[SignalType, AbstentionReason, _NonNegativeCount]
_HeuristicDispositionCount: TypeAlias = tuple[
    ShadowHeuristicDisposition,
    _NonNegativeCount,
]
_SignalCooccurrenceCount: TypeAlias = tuple[SignalType, SignalType, _NonNegativeCount]
_EventTypeCount: TypeAlias = tuple[EventType, _NonNegativeCount]
_PhaseCount: TypeAlias = tuple[EventPhase, _NonNegativeCount]


class _FrozenReportModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


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


def _copy_exact_model(model_type: type[_ModelT], value: object) -> _ModelT:
    if not _model_state_is_exact(model_type, value):
        raise ValueError("report record failed preflight validation")
    serialized = model_type.__pydantic_serializer__.to_json(value, warnings=False)
    copied = model_type.model_validate_json(serialized)
    if copied != value:
        raise ValueError("report record failed defensive validation")
    return copied


def _copy_hmac_tag(value: object) -> PayloadDigest:
    copied = _copy_exact_model(PayloadDigest, value)
    if copied.algorithm is not PayloadDigestAlgorithm.HMAC_SHA256:
        raise ValueError("report identity must use HMAC-SHA-256")
    return copied


def _is_uuid4(value: object) -> bool:
    return type(value) is UUID and value.version == 4


def _is_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact_digest(value: object) -> object:
    if not _is_digest(value):
        raise ValueError("report digest is invalid")
    return value


def _require_optional_exact_digest(value: object) -> object:
    if value is not None and not _is_digest(value):
        raise ValueError("optional report digest is invalid")
    return value


class ShadowReportRow(_FrozenReportModel):
    """One private, sanitized command-input row used to prove report denominators."""

    schema_version: Literal["shadow-report-row/v1"] = SHADOW_REPORT_ROW_SCHEMA_VERSION
    input_ordinal: _PositiveOrdinal
    source_event_digest: Sha256Digest = Field(repr=False)
    first_occurrence_ordinal: _PositiveOrdinal | None
    retry_target_ordinal: _PositiveOrdinal | None
    event_type: EventType
    phase: EventPhase
    input_kind: ShadowInputKind
    persistence_disposition: _PersistenceDisposition
    observation_digest: Sha256Digest

    @field_validator("schema_version", "persistence_disposition", mode="before")
    @classmethod
    def require_exact_fixed_strings(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("report row string is invalid")
        return value

    @field_validator("source_event_digest", "observation_digest", mode="before")
    @classmethod
    def require_exact_digest_strings(cls, value: object) -> object:
        return _require_exact_digest(value)

    @model_validator(mode="after")
    def identifies_one_first_occurrence_or_prior_retry_target(self) -> Self:
        is_first = (
            self.first_occurrence_ordinal == self.input_ordinal
            and self.retry_target_ordinal is None
        )
        is_retry = (
            self.first_occurrence_ordinal is None
            and self.retry_target_ordinal is not None
            and self.retry_target_ordinal < self.input_ordinal
            and self.persistence_disposition == "preexisting"
        )
        if not (is_first or is_retry):
            raise ValueError("report row occurrence identity is invalid")
        projection = SHADOW_PROJECTION_MATRIX[self.input_kind]
        if self.event_type is not projection.event_type or self.phase is not projection.phase:
            raise ValueError("report row projection is invalid")
        return self


def _copy_rows(value: object) -> tuple[ShadowReportRow, ...]:
    if type(value) is not tuple or not 1 <= len(value) <= _MAX_INPUT_ROWS:
        raise ValueError("report rows are invalid")
    return tuple(_copy_exact_model(ShadowReportRow, item) for item in value)


def _copy_observations(value: object) -> tuple[ShadowObservation, ...]:
    if type(value) is not tuple or not 1 <= len(value) <= _MAX_INPUT_ROWS:
        raise ValueError("report observations are invalid")
    return tuple(_copy_exact_model(ShadowObservation, item) for item in value)


def _copy_signal_types(value: object) -> tuple[SignalType, ...]:
    if type(value) is not tuple or not all(type(item) is SignalType for item in value):
        raise ValueError("report signal type declaration is invalid")
    return tuple(value)


def _canonical_detector_outcome_counts(
    counter: Counter[tuple[SignalType, DetectionStatus]],
) -> tuple[_DetectorOutcomeCount, ...]:
    return tuple(
        (signal_type, status, counter[(signal_type, status)])
        for signal_type in _SUPPORTED_SIGNAL_TYPES
        for status in _DETECTION_STATUSES
    )


def _canonical_abstention_reason_counts(
    counter: Counter[tuple[SignalType, AbstentionReason]],
) -> tuple[_AbstentionReasonCount, ...]:
    return tuple(
        (signal_type, reason, counter[(signal_type, reason)])
        for signal_type in _SUPPORTED_SIGNAL_TYPES
        for reason in _ABSTENTION_REASONS
    )


def _canonical_heuristic_disposition_counts(
    counter: Counter[ShadowHeuristicDisposition],
) -> tuple[_HeuristicDispositionCount, ...]:
    return tuple((disposition, counter[disposition]) for disposition in _HEURISTIC_DISPOSITIONS)


def _canonical_signal_cooccurrence_counts(
    counter: Counter[tuple[SignalType, SignalType]],
) -> tuple[_SignalCooccurrenceCount, ...]:
    return tuple((left, right, counter[(left, right)]) for left, right in _SIGNAL_PAIRS)


def _canonical_event_type_counts(
    counter: Counter[EventType],
) -> tuple[_EventTypeCount, ...]:
    return tuple((event_type, counter[event_type]) for event_type in _SHADOW_EVENT_TYPES)


def _canonical_phase_counts(counter: Counter[EventPhase]) -> tuple[_PhaseCount, ...]:
    return tuple((phase, counter[phase]) for phase in _SHADOW_PHASES)


class _DerivedAggregates(_FrozenReportModel):
    input_row_count: _NonNegativeCount
    unique_input_event_count: _NonNegativeCount
    retry_row_count: _NonNegativeCount
    appended_event_count: _NonNegativeCount
    preexisting_event_count: _NonNegativeCount
    rejected_row_count: Literal[0] = 0
    evaluated_unique_event_count: _NonNegativeCount
    observation_count: _NonNegativeCount
    split_metadata_complete: bool
    detector_outcome_counts: tuple[_DetectorOutcomeCount, ...]
    abstention_reason_counts: tuple[_AbstentionReasonCount, ...]
    heuristic_disposition_counts: tuple[_HeuristicDispositionCount, ...]
    applicable_detector_evaluation_count: _NonNegativeCount
    evidence_sufficient_applicable_detector_evaluation_count: _NonNegativeCount
    signal_cooccurrence_counts: tuple[_SignalCooccurrenceCount, ...]
    event_type_counts: tuple[_EventTypeCount, ...]
    phase_counts: tuple[_PhaseCount, ...]
    first_flagged_event_sequence: _PositiveOrdinal | None


def _row_matches_retry_target(row: ShadowReportRow, target: ShadowReportRow) -> bool:
    return (
        row.source_event_digest == target.source_event_digest
        and row.event_type is target.event_type
        and row.phase is target.phase
        and row.input_kind is target.input_kind
        and row.observation_digest == target.observation_digest
    )


def _expected_disposition(
    *,
    applicable_count: int,
    incomplete_count: int,
    detected: bool,
) -> ShadowHeuristicDisposition:
    if detected:
        return ShadowHeuristicDisposition.FLAGGED
    if applicable_count == 0:
        return ShadowHeuristicDisposition.NOT_APPLICABLE
    if incomplete_count:
        return ShadowHeuristicDisposition.INDETERMINATE
    return ShadowHeuristicDisposition.NOT_FLAGGED


def _derive_aggregates(
    *,
    run_id: UUID,
    initial_ledger_entry_count: int,
    initial_ledger_chain_tag: PayloadDigest | None,
    initial_ledger_projection_tag: PayloadDigest | None,
    initial_ledger_head_tag: PayloadDigest | None,
    redaction_policy_tag: PayloadDigest,
    detector_profile_digest: str,
    capture_scope: _CaptureScope,
    task_scope_digest: str | None,
    lineage_scope_digest: str | None,
    capture_manifest_digest: str | None,
    rows: tuple[ShadowReportRow, ...],
    observations: tuple[ShadowObservation, ...],
) -> _DerivedAggregates:
    if not _is_uuid4(run_id):
        raise ValueError("report run identity is invalid")
    if type(initial_ledger_entry_count) is not int or not (
        0 <= initial_ledger_entry_count <= _MAX_SIGNED_64
    ):
        raise ValueError("initial ledger count is invalid")
    tags = (
        initial_ledger_chain_tag,
        initial_ledger_projection_tag,
        initial_ledger_head_tag,
    )
    if initial_ledger_entry_count == 0:
        if any(tag is not None for tag in tags):
            raise ValueError("absent initial ledger has an identity")
    elif any(tag is None for tag in tags):
        raise ValueError("present initial ledger is missing an identity")
    if any(
        tag is not None and tag.algorithm is not PayloadDigestAlgorithm.HMAC_SHA256 for tag in tags
    ):
        raise ValueError("initial ledger identity algorithm is invalid")
    if type(capture_scope) is not str or capture_scope not in (
        "unknown",
        "selected_events",
        "bounded_window",
        "complete_run_declared",
    ):
        raise ValueError("report capture scope is invalid")
    if not _is_digest(detector_profile_digest):
        raise ValueError("report detector profile identity is invalid")
    if any(
        value is not None and not _is_digest(value)
        for value in (task_scope_digest, lineage_scope_digest, capture_manifest_digest)
    ):
        raise ValueError("report capture identity is invalid")
    if redaction_policy_tag.algorithm is not PayloadDigestAlgorithm.HMAC_SHA256:
        raise ValueError("report redaction identity is invalid")

    expected_ordinals = tuple(range(1, len(rows) + 1))
    if tuple(row.input_ordinal for row in rows) != expected_ordinals:
        raise ValueError("report rows are not in canonical input order")
    rows_by_ordinal: dict[int, ShadowReportRow] = {}
    first_rows: list[ShadowReportRow] = []
    first_source_digests: set[str] = set()
    retry_count = 0
    for row in rows:
        if row.first_occurrence_ordinal is not None:
            if row.source_event_digest in first_source_digests:
                raise ValueError("report source identities are not unique")
            first_rows.append(row)
            first_source_digests.add(row.source_event_digest)
        else:
            retry_count += 1
            target = rows_by_ordinal.get(row.retry_target_ordinal or 0)
            if target is None or target.first_occurrence_ordinal is None:
                raise ValueError("report retry target is missing")
            if not _row_matches_retry_target(row, target):
                raise ValueError("report retry row disagrees with its target")
        rows_by_ordinal[row.input_ordinal] = row

    if not first_rows or first_rows[0].input_kind is not ShadowInputKind.START:
        raise ValueError("report does not start with one run marker")
    if sum(row.input_kind is ShadowInputKind.START for row in rows) != 1:
        raise ValueError("report contains multiple run starts")
    finish_rows = tuple(row for row in rows if row.input_kind is ShadowInputKind.FINISH)
    if len(finish_rows) > 1 or (finish_rows and finish_rows[0] is not rows[-1]):
        raise ValueError("report run end is not unique and final")
    if capture_scope == "complete_run_declared" and len(finish_rows) != 1:
        raise ValueError("complete capture is missing its terminal run marker")

    if len(observations) != len(first_rows):
        raise ValueError("report observation count does not match unique rows")
    if len({observation.event_id for observation in observations}) != len(observations):
        raise ValueError("report observations contain duplicate events")
    if len({observation.observation_digest for observation in observations}) != len(observations):
        raise ValueError("report observations contain duplicate identities")

    appended_count = 0
    preexisting_count = 0
    saw_appended = False
    detector_counter: Counter[tuple[SignalType, DetectionStatus]] = Counter()
    abstention_counter: Counter[tuple[SignalType, AbstentionReason]] = Counter()
    disposition_counter: Counter[ShadowHeuristicDisposition] = Counter()
    cooccurrence_counter: Counter[tuple[SignalType, SignalType]] = Counter()
    event_type_counter: Counter[EventType] = Counter()
    phase_counter: Counter[EventPhase] = Counter()
    applicable_total = 0
    evidence_sufficient_total = 0
    first_flagged_sequence: int | None = None

    for unique_ordinal, (row, observation) in enumerate(
        zip(first_rows, observations, strict=True),
        start=1,
    ):
        if (
            observation.run_id != run_id
            or observation.sequence != unique_ordinal
            or observation.cli_input_ordinal != row.input_ordinal
            or observation.source_event_digest != row.source_event_digest
            or observation.observation_digest != row.observation_digest
            or observation.redaction_policy_tag != redaction_policy_tag
            or observation.detector_profile_digest != detector_profile_digest
            or observation.supported_signal_types != _SUPPORTED_SIGNAL_TYPES
            or observation.unsupported_signal_types != _UNSUPPORTED_SIGNAL_TYPES
        ):
            raise ValueError("report observation does not bind its first input occurrence")

        if row.persistence_disposition == "appended":
            saw_appended = True
            appended_count += 1
        else:
            if saw_appended:
                raise ValueError("pre-existing events do not form a canonical prefix")
            preexisting_count += 1

        projection = SHADOW_PROJECTION_MATRIX[row.input_kind]
        applicable = frozenset(projection.applicable_detectors)
        incomplete_types: list[SignalType] = []
        incomplete_reasons: set[AbstentionReason] = set()
        detected_types: list[SignalType] = []
        sufficient_count = 0
        for evaluation in observation.detector_evaluations:
            outcome = evaluation.outcome
            detector_counter[(evaluation.signal_type, outcome.status)] += 1
            if outcome.status is DetectionStatus.ABSTAINED:
                if outcome.abstention_reason is None:
                    raise ValueError("abstention has no exact reason")
                abstention_counter[(evaluation.signal_type, outcome.abstention_reason)] += 1
            if evaluation.signal_type in applicable:
                if (
                    outcome.status is DetectionStatus.ABSTAINED
                    and outcome.abstention_reason is AbstentionReason.EVENT_NOT_APPLICABLE
                ):
                    raise ValueError("applicable detector claims event-not-applicable")
                if outcome.status is DetectionStatus.ABSTAINED:
                    incomplete_types.append(evaluation.signal_type)
                    assert outcome.abstention_reason is not None
                    incomplete_reasons.add(outcome.abstention_reason)
                else:
                    sufficient_count += 1
                if outcome.status is DetectionStatus.DETECTED:
                    detected_types.append(evaluation.signal_type)
            elif outcome.status is DetectionStatus.DETECTED:
                raise ValueError("non-applicable detector emitted a signal")

        heuristic = observation.heuristic_evaluations[0]
        canonical_incomplete = tuple(sorted(incomplete_types, key=lambda item: item.value))
        expected_disposition = _expected_disposition(
            applicable_count=len(applicable),
            incomplete_count=len(canonical_incomplete),
            detected=bool(detected_types),
        )
        expected_reasons = (
            tuple(sorted(incomplete_reasons, key=lambda item: item.value))
            if expected_disposition is ShadowHeuristicDisposition.INDETERMINATE
            else ()
        )
        if (
            heuristic.applicable_detector_count != len(applicable)
            or heuristic.evidence_sufficient_detector_count != sufficient_count
            or heuristic.incomplete_detector_types != canonical_incomplete
            or heuristic.disposition is not expected_disposition
            or heuristic.reason_codes != expected_reasons
        ):
            raise ValueError("report heuristic does not match row applicability")

        disposition_counter[heuristic.disposition] += 1
        applicable_total += len(applicable)
        evidence_sufficient_total += sufficient_count
        for pair in combinations(detected_types, 2):
            cooccurrence_counter[pair] += 1
        event_type_counter[row.event_type] += 1
        phase_counter[row.phase] += 1
        if (
            first_flagged_sequence is None
            and heuristic.disposition is ShadowHeuristicDisposition.FLAGGED
        ):
            first_flagged_sequence = observation.sequence

    if initial_ledger_entry_count == 0 and preexisting_count:
        raise ValueError("absent initial ledger cannot contain pre-existing events")
    if initial_ledger_entry_count > 0 and preexisting_count == 0:
        raise ValueError("present initial ledger has no pre-existing marker")
    if preexisting_count > initial_ledger_entry_count:
        raise ValueError("pre-existing events exceed the initial ledger")

    split_complete = capture_manifest_digest is not None or (
        task_scope_digest is not None and lineage_scope_digest is not None
    )
    return _DerivedAggregates(
        input_row_count=len(rows),
        unique_input_event_count=len(first_rows),
        retry_row_count=retry_count,
        appended_event_count=appended_count,
        preexisting_event_count=preexisting_count,
        evaluated_unique_event_count=len(observations),
        observation_count=len(observations),
        split_metadata_complete=split_complete,
        detector_outcome_counts=_canonical_detector_outcome_counts(detector_counter),
        abstention_reason_counts=_canonical_abstention_reason_counts(abstention_counter),
        heuristic_disposition_counts=_canonical_heuristic_disposition_counts(disposition_counter),
        applicable_detector_evaluation_count=applicable_total,
        evidence_sufficient_applicable_detector_evaluation_count=(evidence_sufficient_total),
        signal_cooccurrence_counts=_canonical_signal_cooccurrence_counts(cooccurrence_counter),
        event_type_counts=_canonical_event_type_counts(event_type_counter),
        phase_counts=_canonical_phase_counts(phase_counter),
        first_flagged_event_sequence=first_flagged_sequence,
    )


class _ShadowRunReportBody(_FrozenReportModel):
    schema_version: Literal["shadow-run-report/v1"] = SHADOW_RUN_REPORT_SCHEMA_VERSION
    run_id: UUID4 = Field(repr=False)
    initial_ledger_entry_count: _NonNegativeCount
    initial_ledger_chain_tag: PayloadDigest | None
    initial_ledger_projection_tag: PayloadDigest | None
    initial_ledger_head_tag: PayloadDigest | None
    input_byte_digest: Sha256Digest
    normalized_input_digest: Sha256Digest
    redaction_policy_tag: PayloadDigest
    detector_profile_digest: Sha256Digest
    capture_scope: _CaptureScope
    task_scope_digest: Sha256Digest | None = None
    lineage_scope_digest: Sha256Digest | None = None
    capture_manifest_digest: Sha256Digest | None = None
    split_metadata_complete: bool
    input_row_count: _NonNegativeCount
    unique_input_event_count: _NonNegativeCount
    retry_row_count: _NonNegativeCount
    appended_event_count: _NonNegativeCount
    preexisting_event_count: _NonNegativeCount
    rejected_row_count: Literal[0] = 0
    evaluated_unique_event_count: _NonNegativeCount
    observation_count: _NonNegativeCount
    rows: Annotated[tuple[ShadowReportRow, ...], Field(min_length=1, max_length=_MAX_INPUT_ROWS)]
    observations: Annotated[
        tuple[ShadowObservation, ...],
        Field(min_length=1, max_length=_MAX_INPUT_ROWS),
    ]
    supported_signal_types: tuple[SignalType, ...]
    unsupported_signal_types: tuple[SignalType, ...]
    detector_outcome_counts: tuple[_DetectorOutcomeCount, ...]
    abstention_reason_counts: tuple[_AbstentionReasonCount, ...]
    heuristic_disposition_counts: tuple[_HeuristicDispositionCount, ...]
    applicable_detector_evaluation_count: _NonNegativeCount
    evidence_sufficient_applicable_detector_evaluation_count: _NonNegativeCount
    signal_cooccurrence_counts: tuple[_SignalCooccurrenceCount, ...]
    event_type_counts: tuple[_EventTypeCount, ...]
    phase_counts: tuple[_PhaseCount, ...]
    first_flagged_event_sequence: _PositiveOrdinal | None
    execution_mode: Literal["shadow"] = "shadow"
    evidence_level: Literal["descriptive_observational"] = "descriptive_observational"
    task_outcome_evidence: Literal["none"] = "none"
    intervention_outcome_evidence: Literal["none"] = "none"
    confirmatory: Literal[False] = False
    calibrated: Literal[False] = False
    calibration_eligible: Literal[False] = False
    decision_authority: Literal[False] = False
    representativeness_supported: Literal[False] = False
    task_efficacy_supported: Literal[False] = False
    counterfactual_effect_supported: Literal[False] = False
    model_calls: Literal[0] = 0
    budget_reservations: Literal[0] = 0
    cycles_created: Literal[0] = 0
    memory_revisions: Literal[0] = 0
    interventions: Literal[0] = 0
    delivery_authorizations: Literal[0] = 0
    deliveries: Literal[0] = 0
    intervention_outcomes: Literal[0] = 0

    @field_validator("schema_version", "capture_scope", mode="before")
    @classmethod
    def require_exact_fixed_strings(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("report fixed string is invalid")
        return value

    @field_validator(
        "input_byte_digest",
        "normalized_input_digest",
        "detector_profile_digest",
        mode="before",
    )
    @classmethod
    def require_exact_digest_strings(cls, value: object) -> object:
        return _require_exact_digest(value)

    @field_validator(
        "task_scope_digest",
        "lineage_scope_digest",
        "capture_manifest_digest",
        mode="before",
    )
    @classmethod
    def require_optional_exact_digest_strings(cls, value: object) -> object:
        return _require_optional_exact_digest(value)

    @field_validator(
        "initial_ledger_chain_tag",
        "initial_ledger_projection_tag",
        "initial_ledger_head_tag",
    )
    @classmethod
    def copy_optional_head_tag(cls, value: object) -> PayloadDigest | None:
        return None if value is None else _copy_hmac_tag(value)

    @field_validator("redaction_policy_tag")
    @classmethod
    def copy_redaction_tag(cls, value: object) -> PayloadDigest:
        return _copy_hmac_tag(value)

    @field_validator("rows")
    @classmethod
    def copy_report_rows(cls, value: object) -> tuple[ShadowReportRow, ...]:
        return _copy_rows(value)

    @field_validator("observations")
    @classmethod
    def copy_report_observations(cls, value: object) -> tuple[ShadowObservation, ...]:
        return _copy_observations(value)

    @field_validator("supported_signal_types", "unsupported_signal_types")
    @classmethod
    def copy_report_signal_types(cls, value: object) -> tuple[SignalType, ...]:
        return _copy_signal_types(value)

    @model_validator(mode="after")
    def aggregates_match_the_unique_ordered_evidence(self) -> Self:
        if self.supported_signal_types != _SUPPORTED_SIGNAL_TYPES:
            raise ValueError("report supported detector set is invalid")
        if self.unsupported_signal_types != _UNSUPPORTED_SIGNAL_TYPES:
            raise ValueError("report unsupported detector set is invalid")
        derived = _derive_aggregates(
            run_id=self.run_id,
            initial_ledger_entry_count=self.initial_ledger_entry_count,
            initial_ledger_chain_tag=self.initial_ledger_chain_tag,
            initial_ledger_projection_tag=self.initial_ledger_projection_tag,
            initial_ledger_head_tag=self.initial_ledger_head_tag,
            redaction_policy_tag=self.redaction_policy_tag,
            detector_profile_digest=self.detector_profile_digest,
            capture_scope=self.capture_scope,
            task_scope_digest=self.task_scope_digest,
            lineage_scope_digest=self.lineage_scope_digest,
            capture_manifest_digest=self.capture_manifest_digest,
            rows=self.rows,
            observations=self.observations,
        )
        for field_name in _DerivedAggregates.model_fields:
            if getattr(self, field_name) != getattr(derived, field_name):
                raise ValueError("report aggregate does not match its ordered evidence")
        if self.input_row_count != (
            self.unique_input_event_count + self.retry_row_count + self.rejected_row_count
        ):
            raise ValueError("report input count equation is invalid")
        if self.unique_input_event_count != (
            self.appended_event_count + self.preexisting_event_count
        ):
            raise ValueError("report persistence count equation is invalid")
        if self.evaluated_unique_event_count != self.unique_input_event_count:
            raise ValueError("report evaluation count equation is invalid")
        if self.observation_count != self.unique_input_event_count:
            raise ValueError("report observation count equation is invalid")
        return self


def _report_body_digest(value: _ShadowRunReportBody) -> str:
    serializer = (
        ShadowRunReport.__pydantic_serializer__
        if type(value) is ShadowRunReport
        else _ShadowRunReportBody.__pydantic_serializer__
    )
    fields = serializer.to_python(
        value,
        mode="json",
        exclude={"report_digest"},
        warnings=False,
    )
    return length_prefixed_sha256(
        canonical_json(fields),
        domain=_RUN_REPORT_DIGEST_DOMAIN,
    )


class ShadowRunReport(_ShadowRunReportBody):
    """One canonical, content-addressed descriptive Shadow run report."""

    report_digest: Sha256Digest

    @field_validator("report_digest", mode="before")
    @classmethod
    def require_exact_report_digest(cls, value: object) -> object:
        return _require_exact_digest(value)

    @model_validator(mode="after")
    def report_digest_matches_every_other_field(self) -> Self:
        if not hmac.compare_digest(self.report_digest, _report_body_digest(self)):
            raise ValueError("shadow run report digest does not match its canonical fields")
        return self


@dataclass(frozen=True, slots=True, repr=False)
class _TrustedShadowRunReport:
    report: ShadowRunReport = field(repr=False)
    _token: object = field(repr=False, compare=False)
    _seal: _TrustedReportSeal = field(repr=False, compare=False)


def _require_trusted_shadow_run_report(value: object) -> ShadowRunReport:
    sealed: tuple[object, ...] | None = None
    if type(value) is _TrustedShadowRunReport and type(value._seal) is _TrustedReportSeal:
        sealed = _TRUSTED_REPORT_SEALS.get(value._seal)
    if (
        type(value) is not _TrustedShadowRunReport
        or sealed is None
        or len(sealed) != 1
        or value.report is not sealed[0]
        or value._token is not _TRUSTED_REPORT_TOKEN
        or not _model_state_is_exact(ShadowRunReport, value.report)
    ):
        raise ShadowInvariantError()
    return value.report


def build_shadow_run_report(
    *,
    run_id: UUID,
    initial_ledger_entry_count: int,
    initial_ledger_chain_tag: PayloadDigest | None,
    initial_ledger_projection_tag: PayloadDigest | None,
    initial_ledger_head_tag: PayloadDigest | None,
    input_byte_digest: str,
    normalized_input_digest: str,
    redaction_policy_tag: PayloadDigest,
    detector_profile_digest: str,
    capture_scope: _CaptureScope,
    task_scope_digest: str | None = None,
    lineage_scope_digest: str | None = None,
    capture_manifest_digest: str | None = None,
    rows: tuple[ShadowReportRow, ...],
    observations: tuple[ShadowObservation, ...],
) -> ShadowRunReport:
    """Build a report solely from explicit provenance and sanitized ordered evidence."""

    result: ShadowRunReport | None = None
    try:
        copied_rows = _copy_rows(rows)
        copied_observations = _copy_observations(observations)
        copied_redaction_policy_tag = _copy_hmac_tag(redaction_policy_tag)
        copied_head_tags = tuple(
            None if tag is None else _copy_hmac_tag(tag)
            for tag in (
                initial_ledger_chain_tag,
                initial_ledger_projection_tag,
                initial_ledger_head_tag,
            )
        )
        derived = _derive_aggregates(
            run_id=run_id,
            initial_ledger_entry_count=initial_ledger_entry_count,
            initial_ledger_chain_tag=copied_head_tags[0],
            initial_ledger_projection_tag=copied_head_tags[1],
            initial_ledger_head_tag=copied_head_tags[2],
            redaction_policy_tag=copied_redaction_policy_tag,
            detector_profile_digest=detector_profile_digest,
            capture_scope=capture_scope,
            task_scope_digest=task_scope_digest,
            lineage_scope_digest=lineage_scope_digest,
            capture_manifest_digest=capture_manifest_digest,
            rows=copied_rows,
            observations=copied_observations,
        )
        body = _ShadowRunReportBody(
            run_id=run_id,
            initial_ledger_entry_count=initial_ledger_entry_count,
            initial_ledger_chain_tag=copied_head_tags[0],
            initial_ledger_projection_tag=copied_head_tags[1],
            initial_ledger_head_tag=copied_head_tags[2],
            input_byte_digest=input_byte_digest,
            normalized_input_digest=normalized_input_digest,
            redaction_policy_tag=copied_redaction_policy_tag,
            detector_profile_digest=detector_profile_digest,
            capture_scope=capture_scope,
            task_scope_digest=task_scope_digest,
            lineage_scope_digest=lineage_scope_digest,
            capture_manifest_digest=capture_manifest_digest,
            rows=copied_rows,
            observations=copied_observations,
            supported_signal_types=_SUPPORTED_SIGNAL_TYPES,
            unsupported_signal_types=_UNSUPPORTED_SIGNAL_TYPES,
            **derived.model_dump(mode="python", warnings=False),
        )
        report = ShadowRunReport(
            **body.model_dump(mode="python", warnings=False),
            report_digest=_report_body_digest(body),
        )
        result = _copy_exact_model(ShadowRunReport, report)
    except Exception:
        result = None
    if result is None:
        raise ShadowInvariantError()
    return result


def _build_shadow_run_report_trusted(
    *,
    run_id: UUID,
    initial_ledger_entry_count: int,
    initial_ledger_chain_tag: PayloadDigest | None,
    initial_ledger_projection_tag: PayloadDigest | None,
    initial_ledger_head_tag: PayloadDigest | None,
    input_byte_digest: str,
    normalized_input_digest: str,
    redaction_policy_tag: PayloadDigest,
    detector_profile_digest: str,
    capture_scope: _CaptureScope,
    task_scope_digest: str | None = None,
    lineage_scope_digest: str | None = None,
    capture_manifest_digest: str | None = None,
    rows: tuple[ShadowReportRow, ...],
    observations: tuple[ShadowObservation, ...],
) -> _TrustedShadowRunReport:
    """Build once-validated evidence without recursively recopying it at every layer."""

    result: _TrustedShadowRunReport | None = None
    try:
        copied_rows = _copy_rows(rows)
        copied_observations = _copy_observations(observations)
        copied_redaction_policy_tag = _copy_hmac_tag(redaction_policy_tag)
        copied_head_tags = tuple(
            None if tag is None else _copy_hmac_tag(tag)
            for tag in (
                initial_ledger_chain_tag,
                initial_ledger_projection_tag,
                initial_ledger_head_tag,
            )
        )
        derived = _derive_aggregates(
            run_id=run_id,
            initial_ledger_entry_count=initial_ledger_entry_count,
            initial_ledger_chain_tag=copied_head_tags[0],
            initial_ledger_projection_tag=copied_head_tags[1],
            initial_ledger_head_tag=copied_head_tags[2],
            redaction_policy_tag=copied_redaction_policy_tag,
            detector_profile_digest=detector_profile_digest,
            capture_scope=capture_scope,
            task_scope_digest=task_scope_digest,
            lineage_scope_digest=lineage_scope_digest,
            capture_manifest_digest=capture_manifest_digest,
            rows=copied_rows,
            observations=copied_observations,
        )
        body = _ShadowRunReportBody.model_construct(
            run_id=run_id,
            initial_ledger_entry_count=initial_ledger_entry_count,
            initial_ledger_chain_tag=copied_head_tags[0],
            initial_ledger_projection_tag=copied_head_tags[1],
            initial_ledger_head_tag=copied_head_tags[2],
            input_byte_digest=input_byte_digest,
            normalized_input_digest=normalized_input_digest,
            redaction_policy_tag=copied_redaction_policy_tag,
            detector_profile_digest=detector_profile_digest,
            capture_scope=capture_scope,
            task_scope_digest=task_scope_digest,
            lineage_scope_digest=lineage_scope_digest,
            capture_manifest_digest=capture_manifest_digest,
            rows=copied_rows,
            observations=copied_observations,
            supported_signal_types=_SUPPORTED_SIGNAL_TYPES,
            unsupported_signal_types=_UNSUPPORTED_SIGNAL_TYPES,
            **derived.model_dump(mode="python", warnings=False),
        )
        report = ShadowRunReport.model_construct(
            **body.__dict__,
            report_digest=_report_body_digest(body),
        )
        if not _model_state_is_exact(_ShadowRunReportBody, body) or not _model_state_is_exact(
            ShadowRunReport,
            report,
        ):
            raise ValueError("trusted report construction failed")
        seal = _TrustedReportSeal()
        result = _TrustedShadowRunReport(
            report=report,
            _token=_TRUSTED_REPORT_TOKEN,
            _seal=seal,
        )
        _TRUSTED_REPORT_SEALS[seal] = (result.report,)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        result = None
    if result is None:
        raise ShadowInvariantError()
    return result


__all__ = [
    "ShadowRunReport",
    "build_shadow_run_report",
]
