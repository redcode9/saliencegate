from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from saliencegate.domain import ReasonCode, Signal, SignalType
from saliencegate.shadow.config import ShadowConfig
from saliencegate.shadow.errors import ShadowInvariantError
from saliencegate.shadow.evaluation import (
    ShadowHeuristicDisposition,
    ShadowHeuristicEvaluation,
    evaluate_shadow_heuristic,
)
from saliencegate.shadow.inputs import ShadowInputKind
from saliencegate.signals import (
    AbstentionReason,
    DetectionOutcome,
    DetectionStatus,
    DetectorEvaluation,
    ExtractionReport,
)
from saliencegate.signals.base import _deterministic_signal_id

RUN_ID = UUID("11111111-1111-4111-8111-111111111111")
EVENT_ID = UUID("22222222-2222-4222-8222-222222222222")
NOW = datetime(2026, 7, 16, 10, 0, tzinfo=UTC)
FEATURE_DIGEST = "f" * 64


def extraction_report(
    outcomes: dict[SignalType, tuple[DetectionStatus, AbstentionReason | None]] | None = None,
) -> ExtractionReport:
    config = ShadowConfig.reference()
    selected = {} if outcomes is None else outcomes
    evaluations: list[DetectorEvaluation] = []
    signals: list[Signal] = []
    for spec in config.detectors:
        status, abstention = selected.get(
            spec.signal_type,
            (DetectionStatus.ABSTAINED, AbstentionReason.EVENT_NOT_APPLICABLE),
        )
        if status is DetectionStatus.DETECTED:
            outcome = DetectionOutcome.detected(spec.signal_type, (EVENT_ID,))
        elif status is DetectionStatus.NO_MATCH:
            outcome = DetectionOutcome.no_match(spec.signal_type, (EVENT_ID,))
        else:
            assert abstention is not None
            outcome = DetectionOutcome.abstained(spec.signal_type, abstention, (EVENT_ID,))
        evaluation = DetectorEvaluation(
            signal_type=spec.signal_type,
            detector_version=spec.detector_version,
            outcome=outcome,
        )
        evaluations.append(evaluation)
        if status is DetectionStatus.DETECTED:
            assert outcome.strength is not None
            reason_code = ReasonCode(spec.signal_type.value)
            signals.append(
                Signal(
                    signal_id=_deterministic_signal_id(
                        RUN_ID,
                        outcome,
                        spec.detector_version,
                    ),
                    run_id=RUN_ID,
                    created_at=NOW,
                    signal_type=spec.signal_type,
                    strength=outcome.strength,
                    evidence_event_ids=outcome.evidence_event_ids,
                    detector_version=spec.detector_version,
                    reason_code=reason_code,
                )
            )
    return ExtractionReport(
        run_id=RUN_ID,
        current_event_id=EVENT_ID,
        current_event_timestamp=NOW,
        evaluations=tuple(evaluations),
        signals=tuple(signals),
    )


def evaluate(
    report: ExtractionReport,
    kind: ShadowInputKind,
) -> ShadowHeuristicEvaluation:
    return evaluate_shadow_heuristic(
        report,
        input_kind=kind,
        config=ShadowConfig.reference(),
        feature_snapshot_digest=FEATURE_DIGEST,
    )


def test_detected_applicable_signal_is_flagged_and_retains_incompleteness() -> None:
    result = evaluate(
        extraction_report(
            {
                SignalType.TOOL_ERROR: (DetectionStatus.DETECTED, None),
                SignalType.REPEATED_FAILURE: (
                    DetectionStatus.ABSTAINED,
                    AbstentionReason.STRUCTURED_EVIDENCE_MISSING,
                ),
            }
        ),
        ShadowInputKind.TOOL_RESULT,
    )

    assert result.schema_version == "shadow-heuristic-evaluation/v1"
    assert result.evaluator_id == "any-detected-signal-baseline/v1"
    assert result.configuration_digest == ShadowConfig.reference().evaluator_configuration_digest
    assert result.scope == "supported_detectors_only"
    assert result.disposition is ShadowHeuristicDisposition.FLAGGED
    assert result.reason_codes == ()
    assert result.feature_snapshot_digest == FEATURE_DIGEST
    assert result.applicable_detector_count == 2
    assert result.evidence_sufficient_detector_count == 1
    assert result.incomplete_detector_types == (SignalType.REPEATED_FAILURE,)
    assert result.calibrated is False
    assert result.decision_authority is False


def test_zero_applicable_detectors_is_not_applicable() -> None:
    result = evaluate(extraction_report(), ShadowInputKind.START)

    assert result.disposition is ShadowHeuristicDisposition.NOT_APPLICABLE
    assert result.applicable_detector_count == 0
    assert result.evidence_sufficient_detector_count == 0
    assert result.incomplete_detector_types == ()
    assert result.reason_codes == ()


def test_applicable_abstentions_are_indeterminate_and_canonically_sorted() -> None:
    result = evaluate(
        extraction_report(
            {
                SignalType.REPEATED_FAILURE: (
                    DetectionStatus.ABSTAINED,
                    AbstentionReason.STRUCTURED_EVIDENCE_MISSING,
                ),
                SignalType.TEST_FAILURE: (
                    DetectionStatus.ABSTAINED,
                    AbstentionReason.INSUFFICIENT_HISTORY,
                ),
            }
        ),
        ShadowInputKind.TEST_RESULT,
    )

    assert result.disposition is ShadowHeuristicDisposition.INDETERMINATE
    assert result.applicable_detector_count == 2
    assert result.evidence_sufficient_detector_count == 0
    assert result.incomplete_detector_types == (
        SignalType.REPEATED_FAILURE,
        SignalType.TEST_FAILURE,
    )
    assert result.reason_codes == (
        AbstentionReason.INSUFFICIENT_HISTORY,
        AbstentionReason.STRUCTURED_EVIDENCE_MISSING,
    )


def test_complete_applicable_no_match_is_not_flagged() -> None:
    result = evaluate(
        extraction_report({SignalType.REPEATED_ACTION: (DetectionStatus.NO_MATCH, None)}),
        ShadowInputKind.ACTION,
    )

    assert result.disposition is ShadowHeuristicDisposition.NOT_FLAGGED
    assert result.applicable_detector_count == 1
    assert result.evidence_sufficient_detector_count == 1
    assert result.incomplete_detector_types == ()
    assert result.reason_codes == ()


def test_action_identity_uses_action_applicability_without_extending_reference_config() -> None:
    report = extraction_report({SignalType.REPEATED_ACTION: (DetectionStatus.NO_MATCH, None)})

    identity = evaluate(report, ShadowInputKind.ACTION_IDENTITY)
    legacy = evaluate(report, ShadowInputKind.ACTION)

    assert identity == legacy
    assert tuple(row.input_kind for row in ShadowConfig.reference().applicability) == (
        ShadowInputKind.START,
        ShadowInputKind.ACTION,
        ShadowInputKind.TOOL_RESULT,
        ShadowInputKind.TEST_RESULT,
        ShadowInputKind.OBSERVATION,
        ShadowInputKind.CONTROLLER_ERROR,
        ShadowInputKind.FINISH,
    )


def test_non_applicable_detection_and_applicable_not_applicable_abstention_fail_closed() -> None:
    outside_mask = extraction_report({SignalType.TOOL_ERROR: (DetectionStatus.DETECTED, None)})
    invalid_abstention = extraction_report(
        {
            SignalType.REPEATED_ACTION: (
                DetectionStatus.ABSTAINED,
                AbstentionReason.EVENT_NOT_APPLICABLE,
            )
        }
    )

    with pytest.raises(ShadowInvariantError):
        evaluate(outside_mask, ShadowInputKind.ACTION)
    with pytest.raises(ShadowInvariantError):
        evaluate(invalid_abstention, ShadowInputKind.ACTION)


def test_evaluator_revalidates_order_versions_signal_consistency_and_digest() -> None:
    config = ShadowConfig.reference()
    report = extraction_report()
    reordered = report.model_copy(update={"evaluations": tuple(reversed(report.evaluations))})
    wrong_version = report.model_copy(
        update={
            "evaluations": (
                report.evaluations[0].model_copy(update={"detector_version": "wrong/v1"}),
                *report.evaluations[1:],
            )
        }
    )

    for candidate, kind, digest, candidate_config in (
        (reordered, ShadowInputKind.START, FEATURE_DIGEST, config),
        (wrong_version, ShadowInputKind.START, FEATURE_DIGEST, config),
        (report, cast(ShadowInputKind, "start"), FEATURE_DIGEST, config),
        (report, ShadowInputKind.START, "F" * 64, config),
        (
            report,
            ShadowInputKind.START,
            FEATURE_DIGEST,
            config.model_copy(update={"detector_profile_digest": "0" * 64}),
        ),
    ):
        with pytest.raises(ShadowInvariantError) as error:
            evaluate_shadow_heuristic(
                candidate,
                input_kind=kind,
                config=candidate_config,
                feature_snapshot_digest=digest,
            )
        assert error.value.__cause__ is None
        assert error.value.__context__ is None


def test_heuristic_evaluation_model_is_strict_frozen_and_internally_consistent() -> None:
    result = evaluate(
        extraction_report({SignalType.REPEATED_ACTION: (DetectionStatus.NO_MATCH, None)}),
        ShadowInputKind.ACTION,
    )
    restored = ShadowHeuristicEvaluation.model_validate_json(result.model_dump_json())

    assert restored == result
    assert restored is not result
    with pytest.raises(ValidationError):
        result.calibrated = True  # type: ignore[assignment,misc]
    with pytest.raises(ValidationError):
        ShadowHeuristicEvaluation.model_validate(
            {**result.model_dump(), "decision_authority": True}
        )
    with pytest.raises(ValidationError):
        ShadowHeuristicEvaluation.model_validate(
            {
                **result.model_dump(),
                "disposition": ShadowHeuristicDisposition.INDETERMINATE,
                "reason_codes": (),
                "incomplete_detector_types": (),
            }
        )


def test_evaluator_rejects_subclassed_reports_and_forged_nested_outcomes() -> None:
    report = extraction_report()
    forged = report.model_copy(
        update={
            "evaluations": (
                report.evaluations[0].model_copy(update={"outcome": object()}),
                *report.evaluations[1:],
            )
        }
    )

    for candidate in (cast(ExtractionReport, object()), forged):
        with pytest.raises(ShadowInvariantError):
            evaluate(candidate, ShadowInputKind.START)


def test_report_copy_never_dispatches_through_the_caller_instance() -> None:
    report = extraction_report()
    serializer_called = False

    def poisoned_model_dump_json(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("caller-controlled-secret")

    class PoisonedSerializer:
        def to_json(self, *args: object, **kwargs: object) -> object:
            nonlocal serializer_called
            del args, kwargs
            serializer_called = True
            raise AssertionError("caller-controlled-secret")

    object.__setattr__(report, "model_dump_json", poisoned_model_dump_json)
    object.__setattr__(report, "__pydantic_serializer__", PoisonedSerializer())

    result = evaluate(report, ShadowInputKind.START)

    assert result.disposition is ShadowHeuristicDisposition.NOT_APPLICABLE
    assert serializer_called is False


def test_evaluator_preflights_forged_nested_bounds_before_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = extraction_report()
    oversized_version = report.evaluations[0].model_copy(update={"detector_version": "x" * 257})
    oversized_outcome = report.evaluations[0].outcome.model_copy(
        update={"related_event_ids": (EVENT_ID,) * 10_001}
    )
    evaluation_with_oversized_outcome = report.evaluations[0].model_copy(
        update={"outcome": oversized_outcome}
    )
    detected_report = extraction_report(
        {SignalType.REPEATED_ACTION: (DetectionStatus.DETECTED, None)}
    )
    oversized_signal = detected_report.signals[0].model_copy(
        update={"evidence_event_ids": (EVENT_ID,) * 65}
    )
    candidates = (
        report.model_copy(update={"evaluations": (oversized_version, *report.evaluations[1:])}),
        report.model_copy(
            update={
                "evaluations": (
                    evaluation_with_oversized_outcome,
                    *report.evaluations[1:],
                )
            }
        ),
        detected_report.model_copy(update={"signals": (oversized_signal,)}),
    )
    serialization_calls = 0

    def counting_model_dump_json(
        model: ExtractionReport,
        *args: object,
        **kwargs: object,
    ) -> str:
        nonlocal serialization_calls
        del model, args, kwargs
        serialization_calls += 1
        return ""

    monkeypatch.setattr(ExtractionReport, "model_dump_json", counting_model_dump_json)

    for candidate in candidates:
        with pytest.raises(ShadowInvariantError):
            evaluate(candidate, ShadowInputKind.START)

    assert serialization_calls == 0


def test_public_disposition_values_are_exact() -> None:
    assert tuple(item.value for item in ShadowHeuristicDisposition) == (
        "flagged",
        "not_flagged",
        "indeterminate",
        "not_applicable",
    )
