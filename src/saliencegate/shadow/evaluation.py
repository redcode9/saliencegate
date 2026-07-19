from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.domain import (
    CURRENT_SCHEMA_VERSION,
    MAX_SIGNAL_EVIDENCE_EVENTS,
    ReasonCode,
    Signal,
    SignalType,
)
from saliencegate.domain.records import Sha256Digest
from saliencegate.shadow.config import ShadowConfig, validate_shadow_config
from saliencegate.shadow.errors import ShadowConfigurationError, ShadowInvariantError
from saliencegate.shadow.inputs import ShadowInputKind
from saliencegate.signals import (
    AbstentionReason,
    DetectionOutcome,
    DetectionStatus,
    DetectorEvaluation,
    ExtractionReport,
)

_MAX_DETECTION_EVENT_IDS = 10_000
_MAX_DETECTOR_VERSION_LENGTH = 256
_DETECTOR_VERSION = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._:/+\-]{0,255}$")


class ShadowHeuristicDisposition(StrEnum):
    FLAGGED = "flagged"
    NOT_FLAGGED = "not_flagged"
    INDETERMINATE = "indeterminate"
    NOT_APPLICABLE = "not_applicable"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class ShadowHeuristicEvaluation(_FrozenModel):
    """One non-authoritative baseline result over the supported detector scope."""

    schema_version: Literal["shadow-heuristic-evaluation/v1"] = "shadow-heuristic-evaluation/v1"
    evaluator_id: Literal["any-detected-signal-baseline/v1"]
    configuration_digest: Sha256Digest
    scope: Literal["supported_detectors_only"] = "supported_detectors_only"
    disposition: ShadowHeuristicDisposition
    reason_codes: Annotated[tuple[AbstentionReason, ...], Field(max_length=7)]
    feature_snapshot_digest: Sha256Digest
    applicable_detector_count: Annotated[int, Field(ge=0, le=4)]
    evidence_sufficient_detector_count: Annotated[int, Field(ge=0, le=4)]
    incomplete_detector_types: Annotated[tuple[SignalType, ...], Field(max_length=4)]
    calibrated: Literal[False] = False
    decision_authority: Literal[False] = False

    @model_validator(mode="after")
    def fields_match_disposition(self) -> Self:
        if self.reason_codes != tuple(sorted(set(self.reason_codes), key=lambda item: item.value)):
            raise ValueError("heuristic reasons are not canonical")
        if AbstentionReason.EVENT_NOT_APPLICABLE in self.reason_codes:
            raise ValueError("event-not-applicable is not an incompleteness reason")
        if self.incomplete_detector_types != tuple(
            sorted(set(self.incomplete_detector_types), key=lambda item: item.value)
        ):
            raise ValueError("incomplete detector types are not canonical")
        if any(
            item
            not in (
                SignalType.REPEATED_ACTION,
                SignalType.REPEATED_FAILURE,
                SignalType.TEST_FAILURE,
                SignalType.TOOL_ERROR,
            )
            for item in self.incomplete_detector_types
        ):
            raise ValueError("incomplete detector type is unsupported")
        if (
            self.evidence_sufficient_detector_count + len(self.incomplete_detector_types)
            != self.applicable_detector_count
        ):
            raise ValueError("heuristic evidence counts disagree")

        if self.disposition is ShadowHeuristicDisposition.NOT_APPLICABLE:
            valid = (
                self.applicable_detector_count == 0
                and self.evidence_sufficient_detector_count == 0
                and not self.incomplete_detector_types
                and not self.reason_codes
            )
        elif self.disposition is ShadowHeuristicDisposition.INDETERMINATE:
            valid = (
                self.applicable_detector_count > 0
                and bool(self.incomplete_detector_types)
                and bool(self.reason_codes)
            )
        elif self.disposition is ShadowHeuristicDisposition.NOT_FLAGGED:
            valid = (
                self.applicable_detector_count > 0
                and self.evidence_sufficient_detector_count == self.applicable_detector_count
                and not self.incomplete_detector_types
                and not self.reason_codes
            )
        else:
            valid = (
                self.applicable_detector_count > 0
                and self.evidence_sufficient_detector_count > 0
                and not self.reason_codes
            )
        if not valid:
            raise ValueError("heuristic fields do not match disposition")
        return self


def _uuid4_is_safe(value: object) -> bool:
    return type(value) is UUID and value.version == 4


def _uuid4_tuple_is_safe(
    value: object,
    *,
    min_length: int = 0,
    max_length: int,
) -> bool:
    return (
        type(value) is tuple
        and min_length <= len(value) <= max_length
        and all(_uuid4_is_safe(item) for item in value)
    )


def _utc_datetime_is_safe(value: object) -> bool:
    return type(value) is datetime and value.tzinfo is UTC


def _outcome_is_safe(value: object) -> bool:
    return (
        type(value) is DetectionOutcome
        and type(value.signal_type) is SignalType
        and type(value.status) is DetectionStatus
        and (
            value.strength is None
            or (
                type(value.strength) is float
                and math.isfinite(value.strength)
                and 0.0 <= value.strength <= 1.0
            )
        )
        and (value.reason_code is None or type(value.reason_code) is ReasonCode)
        and _uuid4_tuple_is_safe(
            value.evidence_event_ids,
            max_length=_MAX_DETECTION_EVENT_IDS,
        )
        and _uuid4_tuple_is_safe(
            value.related_event_ids,
            max_length=_MAX_DETECTION_EVENT_IDS,
        )
        and (value.abstention_reason is None or type(value.abstention_reason) is AbstentionReason)
    )


def _evaluation_is_safe(value: object) -> bool:
    return (
        type(value) is DetectorEvaluation
        and type(value.signal_type) is SignalType
        and type(value.detector_version) is str
        and len(value.detector_version) <= _MAX_DETECTOR_VERSION_LENGTH
        and _DETECTOR_VERSION.fullmatch(value.detector_version) is not None
        and _outcome_is_safe(value.outcome)
    )


def _signal_is_safe(value: object) -> bool:
    return (
        type(value) is Signal
        and type(value.schema_version) is str
        and value.schema_version == CURRENT_SCHEMA_VERSION
        and type(value.record_type) is str
        and value.record_type == "signal"
        and _uuid4_is_safe(value.signal_id)
        and _uuid4_is_safe(value.run_id)
        and _utc_datetime_is_safe(value.created_at)
        and type(value.signal_type) is SignalType
        and type(value.strength) is float
        and math.isfinite(value.strength)
        and 0.0 <= value.strength <= 1.0
        and _uuid4_tuple_is_safe(
            value.evidence_event_ids,
            min_length=1,
            max_length=MAX_SIGNAL_EVIDENCE_EVENTS,
        )
        and type(value.detector_version) is str
        and len(value.detector_version) <= _MAX_DETECTOR_VERSION_LENGTH
        and _DETECTOR_VERSION.fullmatch(value.detector_version) is not None
        and type(value.reason_code) is ReasonCode
    )


def _report_is_safe(value: object) -> bool:
    try:
        return (
            type(value) is ExtractionReport
            and _uuid4_is_safe(value.run_id)
            and _uuid4_is_safe(value.current_event_id)
            and _utc_datetime_is_safe(value.current_event_timestamp)
            and type(value.evaluations) is tuple
            and len(value.evaluations) == 4
            and all(_evaluation_is_safe(evaluation) for evaluation in value.evaluations)
            and type(value.signals) is tuple
            and len(value.signals) <= 4
            and all(_signal_is_safe(signal) for signal in value.signals)
        )
    except Exception:
        return False


def _validated_report(value: object) -> ExtractionReport | None:
    if not _report_is_safe(value):
        return None
    assert type(value) is ExtractionReport
    validated: ExtractionReport | None = None
    try:
        candidate = ExtractionReport.model_validate_json(
            ExtractionReport.__pydantic_serializer__.to_json(
                value,
                warnings=False,
            )
        )
        if candidate == value:
            validated = candidate
    except Exception:
        validated = None
    return validated


def _evaluate(
    report: ExtractionReport,
    *,
    input_kind: ShadowInputKind,
    config: ShadowConfig,
    feature_snapshot_digest: str,
) -> ShadowHeuristicEvaluation | None:
    expected_evaluations = tuple(
        (spec.signal_type, spec.detector_version) for spec in config.detectors
    )
    observed_evaluations = tuple(
        (evaluation.signal_type, evaluation.detector_version) for evaluation in report.evaluations
    )
    if observed_evaluations != expected_evaluations:
        return None

    applicability_kind = (
        ShadowInputKind.ACTION if input_kind is ShadowInputKind.ACTION_IDENTITY else input_kind
    )
    rows = tuple(row for row in config.applicability if row.input_kind is applicability_kind)
    if len(rows) != 1:
        return None
    applicable_types = rows[0].applicable_signal_types
    applicable_set = set(applicable_types)

    if any(
        evaluation.outcome.status is DetectionStatus.DETECTED
        and evaluation.signal_type not in applicable_set
        for evaluation in report.evaluations
    ):
        return None
    applicable = tuple(
        evaluation for evaluation in report.evaluations if evaluation.signal_type in applicable_set
    )
    if len(applicable) != len(applicable_set):
        return None
    if any(
        evaluation.outcome.status is DetectionStatus.ABSTAINED
        and evaluation.outcome.abstention_reason is AbstentionReason.EVENT_NOT_APPLICABLE
        for evaluation in applicable
    ):
        return None

    incomplete = tuple(
        sorted(
            (
                evaluation.signal_type
                for evaluation in applicable
                if evaluation.outcome.status is DetectionStatus.ABSTAINED
            ),
            key=lambda item: item.value,
        )
    )
    incomplete_reasons = tuple(
        sorted(
            {
                evaluation.outcome.abstention_reason
                for evaluation in applicable
                if evaluation.outcome.status is DetectionStatus.ABSTAINED
                and evaluation.outcome.abstention_reason is not None
            },
            key=lambda item: item.value,
        )
    )
    if any(reason not in config.indeterminate_reasons for reason in incomplete_reasons):
        return None

    applicable_count = len(applicable)
    sufficient_count = sum(
        evaluation.outcome.status is not DetectionStatus.ABSTAINED for evaluation in applicable
    )
    detected = any(
        evaluation.outcome.status is DetectionStatus.DETECTED for evaluation in applicable
    )
    if detected:
        disposition = ShadowHeuristicDisposition.FLAGGED
    elif applicable_count == 0:
        disposition = ShadowHeuristicDisposition.NOT_APPLICABLE
    elif incomplete:
        disposition = ShadowHeuristicDisposition.INDETERMINATE
    else:
        disposition = ShadowHeuristicDisposition.NOT_FLAGGED

    reason_codes = (
        incomplete_reasons if disposition is ShadowHeuristicDisposition.INDETERMINATE else ()
    )
    return ShadowHeuristicEvaluation(
        evaluator_id=config.evaluator_id,
        configuration_digest=config.evaluator_configuration_digest,
        disposition=disposition,
        reason_codes=reason_codes,
        feature_snapshot_digest=feature_snapshot_digest,
        applicable_detector_count=applicable_count,
        evidence_sufficient_detector_count=sufficient_count,
        incomplete_detector_types=incomplete,
    )


def evaluate_shadow_heuristic(
    report: ExtractionReport,
    *,
    input_kind: ShadowInputKind,
    config: ShadowConfig,
    feature_snapshot_digest: str,
) -> ShadowHeuristicEvaluation:
    """Evaluate the frozen non-authoritative baseline without repository state."""

    validated_config: ShadowConfig | None = None
    validated_report = _validated_report(report)
    try:
        validated_config = validate_shadow_config(config)
    except ShadowConfigurationError:
        validated_config = None
    result: ShadowHeuristicEvaluation | None = None
    if (
        validated_report is not None
        and validated_config is not None
        and type(input_kind) is ShadowInputKind
        and type(feature_snapshot_digest) is str
        and len(feature_snapshot_digest) == 64
        and all(character in "0123456789abcdef" for character in feature_snapshot_digest)
    ):
        try:
            result = _evaluate(
                validated_report,
                input_kind=input_kind,
                config=validated_config,
                feature_snapshot_digest=feature_snapshot_digest,
            )
        except Exception:
            result = None
    if result is None:
        raise ShadowInvariantError()
    return result


__all__ = [
    "ShadowHeuristicDisposition",
    "ShadowHeuristicEvaluation",
    "evaluate_shadow_heuristic",
]
