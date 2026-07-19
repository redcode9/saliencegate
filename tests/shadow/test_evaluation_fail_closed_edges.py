from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError
from tests.shadow.test_evaluation import FEATURE_DIGEST, evaluate, extraction_report

import saliencegate.shadow.evaluation as evaluation_module
from saliencegate.domain import SignalType
from saliencegate.shadow import ShadowConfig, ShadowInvariantError
from saliencegate.shadow.config import ShadowApplicability
from saliencegate.shadow.evaluation import ShadowHeuristicEvaluation
from saliencegate.shadow.inputs import ShadowInputKind
from saliencegate.signals import AbstentionReason, DetectionStatus, ExtractionReport


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            {
                "reason_codes": (
                    AbstentionReason.STRUCTURED_EVIDENCE_MISSING,
                    AbstentionReason.INSUFFICIENT_HISTORY,
                )
            },
            "not canonical",
        ),
        (
            {"reason_codes": (AbstentionReason.EVENT_NOT_APPLICABLE,)},
            "event-not-applicable",
        ),
        (
            {
                "incomplete_detector_types": (
                    SignalType.TEST_FAILURE,
                    SignalType.REPEATED_FAILURE,
                )
            },
            "not canonical",
        ),
        (
            {"incomplete_detector_types": (SignalType.CONFLICT,)},
            "unsupported",
        ),
        (
            {
                "applicable_detector_count": 1,
                "evidence_sufficient_detector_count": 0,
                "incomplete_detector_types": (),
            },
            "counts disagree",
        ),
    ),
)
def test_heuristic_model_rejects_noncanonical_reasons_types_and_counts(
    mutation: dict[str, object],
    message: str,
) -> None:
    values = evaluate(extraction_report(), ShadowInputKind.START).model_dump(mode="python")
    values.update(mutation)

    with pytest.raises(ValidationError, match=message):
        ShadowHeuristicEvaluation.model_validate(values)


def test_extraction_report_safety_probe_fails_closed_on_missing_state() -> None:
    report = extraction_report()
    del report.__dict__["run_id"]

    assert not evaluation_module._report_is_safe(report)


def test_report_copy_rejects_validation_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = extraction_report()
    drifted = report.model_copy(update={"current_event_id": uuid4()})

    def return_drifted(_cls: type[ExtractionReport], _data: bytes) -> ExtractionReport:
        return drifted

    monkeypatch.setattr(
        ExtractionReport,
        "model_validate_json",
        classmethod(return_drifted),
    )
    assert evaluation_module._validated_report(report) is None


def test_evaluator_rejects_missing_applicability_row() -> None:
    report = extraction_report()
    config = ShadowConfig.reference()
    without_start = config.model_copy(
        update={
            "applicability": tuple(
                row for row in config.applicability if row.input_kind is not ShadowInputKind.START
            )
        }
    )

    assert (
        evaluation_module._evaluate(
            report,
            input_kind=ShadowInputKind.START,
            config=without_start,
            feature_snapshot_digest=FEATURE_DIGEST,
        )
        is None
    )


def test_evaluator_rejects_applicability_without_a_matching_detector() -> None:
    report = extraction_report()
    config = ShadowConfig.reference()
    forged_start = ShadowApplicability(
        input_kind=ShadowInputKind.START,
        applicable_signal_types=(SignalType.CONFLICT,),
    )
    forged = config.model_copy(
        update={
            "applicability": (forged_start, *config.applicability[1:]),
        }
    )

    assert (
        evaluation_module._evaluate(
            report,
            input_kind=ShadowInputKind.START,
            config=forged,
            feature_snapshot_digest=FEATURE_DIGEST,
        )
        is None
    )


def test_evaluator_rejects_unconfigured_incompleteness_reason() -> None:
    report = extraction_report(
        {
            SignalType.REPEATED_ACTION: (
                DetectionStatus.ABSTAINED,
                AbstentionReason.STRUCTURED_EVIDENCE_MISSING,
            )
        }
    )
    config = ShadowConfig.reference()
    forged = config.model_copy(
        update={
            "indeterminate_reasons": tuple(
                reason
                for reason in config.indeterminate_reasons
                if reason is not AbstentionReason.STRUCTURED_EVIDENCE_MISSING
            )
        }
    )

    assert (
        evaluation_module._evaluate(
            report,
            input_kind=ShadowInputKind.ACTION,
            config=forged,
            feature_snapshot_digest=FEATURE_DIGEST,
        )
        is None
    )


def test_public_evaluator_sanitizes_unexpected_core_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_evaluation(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("fixture-secret-evaluator-detail")

    monkeypatch.setattr(evaluation_module, "_evaluate", fail_evaluation)
    with pytest.raises(ShadowInvariantError) as captured:
        evaluation_module.evaluate_shadow_heuristic(
            extraction_report(),
            input_kind=ShadowInputKind.START,
            config=ShadowConfig.reference(),
            feature_snapshot_digest=FEATURE_DIGEST,
        )
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert "fixture-secret" not in repr(captured.value)
