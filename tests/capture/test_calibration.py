from __future__ import annotations

import ast
import json
from fractions import Fraction
from pathlib import Path

import pytest

import saliencegate.capture.feedback as feedback_module
from saliencegate.capture.capabilities import CaptureProfile
from saliencegate.capture.feedback import (
    CAPTURE_CALIBRATION_BOOTSTRAP_REPLICATES,
    CAPTURE_CALIBRATION_BOOTSTRAP_SEED,
    CAPTURE_CALIBRATION_BOOTSTRAP_SEED_COMMITMENT,
    CAPTURE_CALIBRATION_MIN_INTERVAL_DENOMINATOR,
    CAPTURE_CALIBRATION_MIN_PROVIDER_DISCLOSURE_SUPPORT,
    CAPTURE_E01_MIN_FINAL_TEST_SESSIONS,
    CAPTURE_E01_MIN_MEMORY_NEEDED,
    CAPTURE_E01_MIN_NOT_MEMORY_NEEDED,
    CAPTURE_E01_MIN_PROJECTS,
    CAPTURE_E01_MIN_PROVIDERS,
    CaptureCalibrationEvidenceStatus,
    CaptureCalibrationInsufficiency,
    CaptureFeedbackError,
    CaptureFeedbackEvidenceSource,
    CaptureFeedbackExportRecord,
    CaptureFeedbackLabel,
    CaptureFeedbackPartition,
    CaptureFeedbackPrediction,
    build_capture_feedback_dataset,
    build_synthetic_capture_feedback_export_record,
    decode_capture_calibration_report,
    encode_capture_calibration_report,
)
from saliencegate.capture.feedback import (
    evaluate_capture_feedback_dataset as _evaluate_capture_feedback_dataset,
)
from saliencegate.domain import canonical_json, length_prefixed_sha256
from saliencegate.security import InstallationKey

KEY = InstallationKey(b"k" * 32)


def evaluate_capture_feedback_dataset(dataset):
    return _evaluate_capture_feedback_dataset(dataset, installation_key=KEY)


def _digest(index: int) -> str:
    return f"{index:064x}"


def _record(
    index: int,
    *,
    project: int = 1,
    profile: CaptureProfile = CaptureProfile.CODEX_HOOKS_V1,
    label: CaptureFeedbackLabel = CaptureFeedbackLabel.MEMORY_NEEDED,
    prediction: CaptureFeedbackPrediction = CaptureFeedbackPrediction.MEMORY_NEEDED,
    partition: CaptureFeedbackPartition = CaptureFeedbackPartition.FINAL_TEST,
) -> CaptureFeedbackExportRecord:
    return build_synthetic_capture_feedback_export_record(
        project_digest=_digest(10_000 + project),
        profile_id=profile,
        session_id=_digest(index),
        label=label,
        prediction=prediction,
        partition=partition,
        installation_key=KEY,
    )


def _dataset(
    records: tuple[CaptureFeedbackExportRecord, ...],
    *,
    source: CaptureFeedbackEvidenceSource = CaptureFeedbackEvidenceSource.SYNTHETIC,
):
    return build_capture_feedback_dataset(
        records,
        installation_key=KEY,
        export_nonce=b"e" * 32,
        evidence_source=source,
        opt_in=True,
    )


def _metric_values(report) -> dict[str, int | None]:
    return {
        name: None if estimate is None else estimate.value_ppm
        for name in type(report.pooled.metrics).model_fields
        if (estimate := getattr(report.pooled.metrics, name)) is not None or True
    }


def _e01_records() -> tuple[CaptureFeedbackExportRecord, ...]:
    profiles = (CaptureProfile.CODEX_HOOKS_V1, CaptureProfile.CLAUDE_CODE_HOOKS_V1)
    rows: list[CaptureFeedbackExportRecord] = []
    for index in range(240):
        positive = index % 2 == 0
        label = (
            CaptureFeedbackLabel.MEMORY_NEEDED
            if positive
            else CaptureFeedbackLabel.NOT_MEMORY_NEEDED
        )
        if positive:
            prediction = (
                CaptureFeedbackPrediction.ABSTAIN
                if index % 20 == 0
                else CaptureFeedbackPrediction.NOT_MEMORY_NEEDED
                if index % 10 == 2
                else CaptureFeedbackPrediction.MEMORY_NEEDED
            )
        else:
            prediction = (
                CaptureFeedbackPrediction.MEMORY_NEEDED
                if index % 10 == 1
                else CaptureFeedbackPrediction.ABSTAIN
                if index % 20 == 3
                else CaptureFeedbackPrediction.NOT_MEMORY_NEEDED
            )
        rows.append(
            _record(
                index + 1,
                project=(index % 3) + 1,
                profile=profiles[(index // 2) % 2],
                label=label,
                prediction=prediction,
            )
        )
    return tuple(rows)


def _development_records() -> tuple[CaptureFeedbackExportRecord, ...]:
    return tuple(
        _record(
            10_000 + index,
            project=(index % 3) + 1,
            profile=(
                CaptureProfile.CODEX_HOOKS_V1 if index % 2 else CaptureProfile.CLAUDE_CODE_HOOKS_V1
            ),
            label=(
                CaptureFeedbackLabel.MEMORY_NEEDED
                if index % 2
                else CaptureFeedbackLabel.NOT_MEMORY_NEEDED
            ),
            partition=CaptureFeedbackPartition.CALIBRATION,
        )
        for index in range(1, 31)
    )


def _replace_record(
    record: CaptureFeedbackExportRecord,
    **changes: object,
) -> CaptureFeedbackExportRecord:
    values: dict[str, object] = {
        "project_digest": record.project_digest,
        "profile_id": record.profile_id,
        "session_id": record.session_id,
        "label": record.label,
        "prediction": record.prediction,
        "partition": record.partition,
    }
    values.update(changes)
    return build_synthetic_capture_feedback_export_record(
        installation_key=KEY,
        **values,  # type: ignore[arg-type]
    )


def _recomputed_report_bytes(payload: dict[str, object]) -> bytes:
    body = {
        key: value for key, value in payload.items() if key not in {"report_digest", "report_tag"}
    }
    report_digest = length_prefixed_sha256(
        canonical_json(body),
        domain="saliencegate:capture:calibration-report:v1",
    )
    payload["report_digest"] = report_digest
    payload["report_tag"] = KEY._hmac_sha256(
        canonical_json({**body, "report_digest": report_digest}),
        domain=b"saliencegate:capture:calibration-report-tag:v1",
    )
    return canonical_json(payload)


def test_e01_constants_and_fixed_bootstrap_contract_are_explicit() -> None:
    assert CAPTURE_E01_MIN_FINAL_TEST_SESSIONS == 200
    assert CAPTURE_E01_MIN_PROJECTS == 3
    assert CAPTURE_E01_MIN_PROVIDERS == 2
    assert CAPTURE_E01_MIN_MEMORY_NEEDED == 30
    assert CAPTURE_E01_MIN_NOT_MEMORY_NEEDED == 30
    assert CAPTURE_CALIBRATION_BOOTSTRAP_REPLICATES == 2_000
    assert CAPTURE_CALIBRATION_MIN_INTERVAL_DENOMINATOR == 30
    assert CAPTURE_CALIBRATION_MIN_PROVIDER_DISCLOSURE_SUPPORT == 30
    assert len(CAPTURE_CALIBRATION_BOOTSTRAP_SEED) == 64
    assert set(CAPTURE_CALIBRATION_BOOTSTRAP_SEED) <= set("0123456789abcdef")
    assert len(CAPTURE_CALIBRATION_BOOTSTRAP_SEED_COMMITMENT) == 64
    assert (
        length_prefixed_sha256(
            CAPTURE_CALIBRATION_BOOTSTRAP_SEED,
            domain="saliencegate:capture:feedback-bootstrap-seed-commitment:v1",
        )
        == CAPTURE_CALIBRATION_BOOTSTRAP_SEED_COMMITMENT
    )


def test_unassigned_partition_is_rejected_at_factory_and_dataset_boundaries() -> None:
    with pytest.raises(CaptureFeedbackError):
        _record(1, partition=CaptureFeedbackPartition.UNASSIGNED)

    record = _record(2)
    body = record.model_dump(mode="json", warnings="error")
    body.pop("record_tag")
    body["partition"] = CaptureFeedbackPartition.UNASSIGNED.value
    forged = record.model_copy(
        update={
            "partition": CaptureFeedbackPartition.UNASSIGNED,
            "record_tag": KEY._hmac_sha256(
                canonical_json(body),
                domain=b"saliencegate:capture:feedback-export-record:v1",
            ),
        }
    )

    with pytest.raises(CaptureFeedbackError):
        _dataset((forged,))


def test_zero_examples_are_insufficient_and_never_impute_undefined_metrics() -> None:
    report = evaluate_capture_feedback_dataset(_dataset(()))

    assert (
        report.evidence_status is CaptureCalibrationEvidenceStatus.INSUFFICIENT_REAL_WORLD_EVIDENCE
    )
    assert report.dataset_support == 0
    assert report.final_test_support == 0
    assert report.pooled.total_support == 0
    assert report.providers == ()
    assert all(value is None for value in _metric_values(report).values())
    assert CaptureCalibrationInsufficiency.FINAL_TEST_SUPPORT in report.insufficiency_reasons
    assert CaptureCalibrationInsufficiency.NON_E01_EVIDENCE in report.insufficiency_reasons
    assert report.confirmatory is False
    assert report.decision_authority is False


def test_few_examples_report_exact_selective_metrics_but_no_intervals() -> None:
    records = (
        _record(1),
        _record(2),
        _record(
            3,
            prediction=CaptureFeedbackPrediction.NOT_MEMORY_NEEDED,
        ),
        _record(4, prediction=CaptureFeedbackPrediction.ABSTAIN),
        _record(
            5,
            label=CaptureFeedbackLabel.NOT_MEMORY_NEEDED,
            prediction=CaptureFeedbackPrediction.MEMORY_NEEDED,
        ),
        _record(
            6,
            label=CaptureFeedbackLabel.NOT_MEMORY_NEEDED,
            prediction=CaptureFeedbackPrediction.NOT_MEMORY_NEEDED,
        ),
        _record(
            7,
            label=CaptureFeedbackLabel.NOT_MEMORY_NEEDED,
            prediction=CaptureFeedbackPrediction.NOT_MEMORY_NEEDED,
        ),
        _record(
            8,
            label=CaptureFeedbackLabel.NOT_MEMORY_NEEDED,
            prediction=CaptureFeedbackPrediction.ABSTAIN,
        ),
        _record(
            9,
            label=CaptureFeedbackLabel.UNCERTAIN,
            prediction=CaptureFeedbackPrediction.MEMORY_NEEDED,
        ),
        _record(
            10,
            label=CaptureFeedbackLabel.UNCERTAIN,
            prediction=CaptureFeedbackPrediction.ABSTAIN,
        ),
    )

    report = evaluate_capture_feedback_dataset(_dataset(records))
    pooled = report.pooled

    assert report.evidence_status.value == "insufficient_real_world_evidence"
    assert pooled.total_support == 10
    assert pooled.memory_needed_support == 4
    assert pooled.not_memory_needed_support == 4
    assert pooled.uncertain_support == 2
    assert pooled.confusion.true_positive == 2
    assert pooled.confusion.false_negative == 1
    assert pooled.confusion.false_positive == 1
    assert pooled.confusion.true_negative == 2
    assert pooled.confusion.abstained_memory_needed == 1
    assert pooled.confusion.abstained_not_memory_needed == 1
    assert pooled.metrics.prevalence.value_ppm == 500_000
    assert pooled.metrics.precision.value_ppm == 666_666
    assert pooled.metrics.recall.value_ppm == 500_000
    assert pooled.metrics.false_positive_rate.value_ppm == 250_000
    assert pooled.metrics.reference_abstention_rate.value_ppm == 200_000
    assert pooled.metrics.prediction_abstention_rate.value_ppm == 300_000
    assert pooled.metrics.joint_abstention_rate.value_ppm == 400_000
    assert all(
        estimate is None or estimate.interval is None
        for estimate in (
            pooled.metrics.prevalence,
            pooled.metrics.precision,
            pooled.metrics.recall,
            pooled.metrics.false_positive_rate,
            pooled.metrics.reference_abstention_rate,
            pooled.metrics.prediction_abstention_rate,
            pooled.metrics.joint_abstention_rate,
        )
    )


def test_zero_denominators_remain_null_not_zero() -> None:
    report = evaluate_capture_feedback_dataset(
        _dataset(
            tuple(
                _record(
                    index,
                    label=CaptureFeedbackLabel.UNCERTAIN,
                    prediction=CaptureFeedbackPrediction.ABSTAIN,
                )
                for index in range(1, 6)
            )
        )
    )

    metrics = report.pooled.metrics
    assert metrics.prevalence is None
    assert metrics.precision is None
    assert metrics.recall is None
    assert metrics.false_positive_rate is None
    assert metrics.reference_abstention_rate.value_ppm == 1_000_000
    assert metrics.prediction_abstention_rate.value_ppm == 1_000_000
    assert metrics.joint_abstention_rate.value_ppm == 1_000_000


def test_predicted_positives_only_for_uncertain_labels_do_not_define_precision() -> None:
    report = evaluate_capture_feedback_dataset(
        _dataset(
            tuple(
                _record(
                    index,
                    label=CaptureFeedbackLabel.UNCERTAIN,
                    prediction=CaptureFeedbackPrediction.MEMORY_NEEDED,
                )
                for index in range(1, 4)
            )
        )
    )

    assert report.pooled.confusion.uncertain_predicted_memory_needed == 3
    assert report.pooled.metrics.precision is None
    assert report.pooled.metrics.reference_abstention_rate.value_ppm == 1_000_000
    assert report.pooled.metrics.prediction_abstention_rate.value_ppm == 0


def test_all_system_predictions_abstaining_preserves_reference_denominators() -> None:
    report = evaluate_capture_feedback_dataset(
        _dataset(
            (
                _record(1, prediction=CaptureFeedbackPrediction.ABSTAIN),
                _record(
                    2,
                    label=CaptureFeedbackLabel.NOT_MEMORY_NEEDED,
                    prediction=CaptureFeedbackPrediction.ABSTAIN,
                ),
                _record(
                    3,
                    label=CaptureFeedbackLabel.UNCERTAIN,
                    prediction=CaptureFeedbackPrediction.ABSTAIN,
                ),
            )
        )
    )

    metrics = report.pooled.metrics
    assert report.pooled.confusion.abstained_memory_needed == 1
    assert report.pooled.confusion.abstained_not_memory_needed == 1
    assert report.pooled.confusion.uncertain_prediction_abstained == 1
    assert metrics.prevalence.value_ppm == 500_000
    assert metrics.precision is None
    assert metrics.recall.value_ppm == 0
    assert metrics.false_positive_rate.value_ppm == 0
    assert metrics.reference_abstention_rate.value_ppm == 333_333
    assert metrics.prediction_abstention_rate.value_ppm == 1_000_000
    assert metrics.joint_abstention_rate.value_ppm == 1_000_000


def test_bootstrap_draws_and_percentile_ranks_match_golden_contract() -> None:
    cases = ((0, 0), (0, 1), (1, 0), (49, 6), (1_999, 3))

    assert [
        feedback_module._bootstrap_draw_index(
            size=7,
            replicate=replicate,
            profile=CaptureProfile.CODEX_HOOKS_V1.value,
            project_id="a" * 64,
            draw=draw,
        )
        for replicate, draw in cases
    ] == [5, 1, 2, 3, 6]

    interval = feedback_module._interval(
        [Fraction(index, 1_999) for index in range(2_000)],
        undefined=0,
    )
    assert interval is not None
    assert (interval.lower_ppm, interval.upper_ppm) == (24_512, 974_988)
    assert interval.percentile_rule == "nearest-rank-50-1950-of-2000/v1"


def test_only_the_locked_final_test_partition_enters_metrics_and_support_gates() -> None:
    development = tuple(
        _record(
            index,
            project=(index % 3) + 1,
            profile=(
                CaptureProfile.CODEX_HOOKS_V1 if index % 2 else CaptureProfile.CLAUDE_CODE_HOOKS_V1
            ),
            label=(
                CaptureFeedbackLabel.MEMORY_NEEDED
                if index % 2
                else CaptureFeedbackLabel.NOT_MEMORY_NEEDED
            ),
            partition=CaptureFeedbackPartition.DEVELOPMENT,
        )
        for index in range(1, 201)
    )
    final_test = tuple(
        _record(
            1_000 + index,
            label=CaptureFeedbackLabel.NOT_MEMORY_NEEDED,
            prediction=CaptureFeedbackPrediction.NOT_MEMORY_NEEDED,
        )
        for index in range(1, 11)
    )

    report = evaluate_capture_feedback_dataset(
        _dataset(
            development + final_test,
            source=CaptureFeedbackEvidenceSource.SYNTHETIC,
        )
    )

    assert report.dataset_support == 210
    assert report.final_test_support == 10
    assert report.pooled.total_support == 10
    assert report.pooled.memory_needed_support == 0
    assert report.pooled.not_memory_needed_support == 10
    assert report.evidence_status.value == "insufficient_real_world_evidence"


def test_metric_interval_requires_thirty_raw_observations_after_global_floors() -> None:
    def report_with_precision_denominator(denominator: int):
        records = tuple(
            _record(
                index,
                project=1 if index <= denominator else 2 + (index % 2),
                profile=(
                    CaptureProfile.CODEX_HOOKS_V1
                    if index % 2
                    else CaptureProfile.CLAUDE_CODE_HOOKS_V1
                ),
                label=(
                    CaptureFeedbackLabel.MEMORY_NEEDED
                    if index % 2
                    else CaptureFeedbackLabel.NOT_MEMORY_NEEDED
                ),
                prediction=(
                    CaptureFeedbackPrediction.MEMORY_NEEDED
                    if index <= denominator
                    else CaptureFeedbackPrediction.NOT_MEMORY_NEEDED
                ),
            )
            for index in range(1, 201)
        )
        return evaluate_capture_feedback_dataset(_dataset(records))

    below = report_with_precision_denominator(CAPTURE_CALIBRATION_MIN_INTERVAL_DENOMINATOR - 1)
    at_floor = report_with_precision_denominator(CAPTURE_CALIBRATION_MIN_INTERVAL_DENOMINATOR)
    global_floor_reasons = {
        CaptureCalibrationInsufficiency.FINAL_TEST_SUPPORT,
        CaptureCalibrationInsufficiency.PROJECT_SUPPORT,
        CaptureCalibrationInsufficiency.PROVIDER_SUPPORT,
        CaptureCalibrationInsufficiency.MEMORY_NEEDED_SUPPORT,
        CaptureCalibrationInsufficiency.NOT_MEMORY_NEEDED_SUPPORT,
    }

    assert below.final_test_support == at_floor.final_test_support == 200
    assert below.pooled.project_count == at_floor.pooled.project_count == 3
    assert below.provider_count == at_floor.provider_count == 2
    assert global_floor_reasons.isdisjoint(below.insufficiency_reasons)
    assert global_floor_reasons.isdisjoint(at_floor.insufficiency_reasons)

    below_precision = below.pooled.metrics.precision
    at_floor_precision = at_floor.pooled.metrics.precision
    assert below_precision.denominator == 29
    assert below_precision.interval is None
    assert below_precision.interval_status.value == "insufficient_support"
    assert below_precision.undefined_bootstrap_replicates == 0
    assert at_floor_precision.denominator == 30
    assert at_floor_precision.interval is not None
    assert at_floor_precision.interval_status.value == "estimated"
    assert at_floor_precision.undefined_bootstrap_replicates == 0

    forged_below = json.loads(encode_capture_calibration_report(below))
    forged_below_precision = forged_below["pooled"]["metrics"]["precision"]
    forged_below_precision["interval"] = {
        "lower_ppm": 0,
        "upper_ppm": 1_000_000,
        "confidence_ppm": 950_000,
        "bootstrap_replicates": 2_000,
        "percentile_rule": "nearest-rank-50-1950-of-2000/v1",
        "boundary_status": "interior",
        "finite_sample_safety_bound": False,
    }
    forged_below_precision["interval_status"] = "estimated"

    forged_at_floor = json.loads(encode_capture_calibration_report(at_floor))
    forged_at_floor_precision = forged_at_floor["pooled"]["metrics"]["precision"]
    forged_at_floor_precision["interval"] = None
    forged_at_floor_precision["interval_status"] = "insufficient_support"

    for forged in (forged_below, forged_at_floor):
        with pytest.raises(CaptureFeedbackError):
            decode_capture_calibration_report(
                _recomputed_report_bytes(forged),
                installation_key=KEY,
            )


def test_report_decoder_rejects_an_interval_below_global_bootstrap_support() -> None:
    report = evaluate_capture_feedback_dataset(
        _dataset(
            tuple(
                _record(
                    index,
                    label=(
                        CaptureFeedbackLabel.MEMORY_NEEDED
                        if index % 2
                        else CaptureFeedbackLabel.NOT_MEMORY_NEEDED
                    ),
                    prediction=CaptureFeedbackPrediction.MEMORY_NEEDED,
                )
                for index in range(1, 31)
            )
        )
    )
    forged = json.loads(encode_capture_calibration_report(report))
    precision = forged["pooled"]["metrics"]["precision"]
    assert precision["denominator"] == 30
    assert precision["interval_status"] == "insufficient_support"
    precision["interval"] = {
        "lower_ppm": 0,
        "upper_ppm": 1_000_000,
        "confidence_ppm": 950_000,
        "bootstrap_replicates": 2_000,
        "percentile_rule": "nearest-rank-50-1950-of-2000/v1",
        "boundary_status": "interior",
        "finite_sample_safety_bound": False,
    }
    precision["interval_status"] = "estimated"

    with pytest.raises(CaptureFeedbackError):
        decode_capture_calibration_report(
            _recomputed_report_bytes(forged),
            installation_key=KEY,
        )


def test_large_synthetic_report_is_deterministic_stratified_and_content_free() -> None:
    records = _development_records() + _e01_records()
    dataset = _dataset(
        records,
        source=CaptureFeedbackEvidenceSource.SYNTHETIC,
    )

    first = evaluate_capture_feedback_dataset(dataset)
    second = evaluate_capture_feedback_dataset(dataset)
    unlinked_dataset = build_capture_feedback_dataset(
        records,
        installation_key=KEY,
        export_nonce=b"u" * 32,
        evidence_source=CaptureFeedbackEvidenceSource.SYNTHETIC,
        opt_in=True,
    )
    unlinked = evaluate_capture_feedback_dataset(unlinked_dataset)
    encoded = encode_capture_calibration_report(first)

    assert (
        first
        == second
        == decode_capture_calibration_report(
            encoded,
            installation_key=KEY,
        )
    )
    assert first.report_digest == second.report_digest
    assert first.pooled == unlinked.pooled
    assert first.providers == unlinked.providers
    assert first.dataset_digest != unlinked.dataset_digest
    assert (
        first.evidence_status is CaptureCalibrationEvidenceStatus.INSUFFICIENT_REAL_WORLD_EVIDENCE
    )
    assert first.insufficiency_reasons == (
        CaptureCalibrationInsufficiency.EXTERNAL_REVIEW,
        CaptureCalibrationInsufficiency.NON_E01_EVIDENCE,
    )
    assert first.dataset_support == 270
    assert first.calibration_support == 30
    assert first.final_test_support == 240
    assert first.pooled.project_count == 3
    assert tuple(item.profile_id for item in first.providers) == (
        CaptureProfile.CLAUDE_CODE_HOOKS_V1,
        CaptureProfile.CODEX_HOOKS_V1,
    )
    assert all(item.project_count == 3 for item in first.providers)
    for stratum in (first.pooled, *first.providers):
        estimates = tuple(
            getattr(stratum.metrics, name) for name in type(stratum.metrics).model_fields
        )
        assert all(estimate is not None for estimate in estimates)
        assert all(estimate.interval is not None for estimate in estimates)
        assert all(
            0 <= estimate.interval.lower_ppm <= estimate.interval.upper_ppm <= 1_000_000
            for estimate in estimates
        )
        assert all(
            estimate.interval.bootstrap_replicates == CAPTURE_CALIBRATION_BOOTSTRAP_REPLICATES
            for estimate in estimates
        )
    assert first.bootstrap_seed_commitment == CAPTURE_CALIBRATION_BOOTSTRAP_SEED_COMMITMENT
    assert first.bootstrap_stratification == "provider_project"
    assert first.project_identifiers_disclosed is False
    assert first.raw_content_used is False
    for example in dataset.examples:
        assert example.project_id.encode() not in encoded
        assert example.example_id.encode() not in encoded


def test_synthetic_evidence_never_becomes_real_world_evidence_at_large_support() -> None:
    report = evaluate_capture_feedback_dataset(_dataset(_development_records() + _e01_records()))

    assert report.evidence_status.value == "insufficient_real_world_evidence"
    assert CaptureCalibrationInsufficiency.NON_E01_EVIDENCE in report.insufficiency_reasons
    assert CaptureCalibrationInsufficiency.EXTERNAL_REVIEW in report.insufficiency_reasons
    assert report.pooled.metrics.precision.interval is not None


def test_small_provider_strata_are_suppressed_from_the_report() -> None:
    records = tuple(
        _record(
            index,
            project=(index % 3) + 1,
            profile=(
                CaptureProfile.CLAUDE_CODE_HOOKS_V1
                if index > 198
                else CaptureProfile.CODEX_HOOKS_V1
            ),
            label=(
                CaptureFeedbackLabel.MEMORY_NEEDED
                if index % 2
                else CaptureFeedbackLabel.NOT_MEMORY_NEEDED
            ),
            prediction=(
                CaptureFeedbackPrediction.MEMORY_NEEDED
                if index % 2
                else CaptureFeedbackPrediction.NOT_MEMORY_NEEDED
            ),
        )
        for index in range(1, 201)
    )

    report = evaluate_capture_feedback_dataset(_dataset(records))
    encoded = encode_capture_calibration_report(report)

    assert report.provider_count == 2
    assert report.suppressed_provider_count == 1
    assert report.suppressed_provider_support == 2
    assert tuple(item.profile_id for item in report.providers) == (CaptureProfile.CODEX_HOOKS_V1,)
    assert CaptureCalibrationInsufficiency.PROVIDER_DISCLOSURE in report.insufficiency_reasons
    assert CaptureProfile.CLAUDE_CODE_HOOKS_V1.value.encode() not in encoded


def test_zero_false_positive_percentile_interval_is_not_a_safety_bound() -> None:
    report = evaluate_capture_feedback_dataset(
        _dataset(
            tuple(
                _record(
                    index,
                    project=(index % 3) + 1,
                    label=CaptureFeedbackLabel.NOT_MEMORY_NEEDED,
                    prediction=CaptureFeedbackPrediction.NOT_MEMORY_NEEDED,
                )
                for index in range(1, 201)
            )
        )
    )

    interval = report.pooled.metrics.false_positive_rate.interval
    assert interval is not None
    assert (interval.lower_ppm, interval.upper_ppm) == (0, 0)
    assert interval.boundary_status.value == "degenerate_zero_width"
    assert interval.finite_sample_safety_bound is False


@pytest.mark.parametrize(
    ("mutate", "expected"),
    (
        (
            lambda rows: rows[:199],
            CaptureCalibrationInsufficiency.FINAL_TEST_SUPPORT,
        ),
        (
            lambda rows: tuple(
                _replace_record(row, project_digest=_digest(10_001)) for row in rows
            ),
            CaptureCalibrationInsufficiency.PROJECT_SUPPORT,
        ),
        (
            lambda rows: tuple(
                _replace_record(row, profile_id=CaptureProfile.CODEX_HOOKS_V1) for row in rows
            ),
            CaptureCalibrationInsufficiency.PROVIDER_SUPPORT,
        ),
        (
            lambda rows: tuple(
                _replace_record(row, label=CaptureFeedbackLabel.NOT_MEMORY_NEEDED)
                if index >= 29 and row.label is CaptureFeedbackLabel.MEMORY_NEEDED
                else row
                for index, row in enumerate(rows)
            ),
            CaptureCalibrationInsufficiency.MEMORY_NEEDED_SUPPORT,
        ),
        (
            lambda rows: tuple(
                _replace_record(row, label=CaptureFeedbackLabel.MEMORY_NEEDED)
                if index >= 29 and row.label is CaptureFeedbackLabel.NOT_MEMORY_NEEDED
                else row
                for index, row in enumerate(rows)
            ),
            CaptureCalibrationInsufficiency.NOT_MEMORY_NEEDED_SUPPORT,
        ),
    ),
)
def test_each_preregistered_support_floor_fails_closed(mutate, expected) -> None:
    report = evaluate_capture_feedback_dataset(
        _dataset(
            mutate(_e01_records()),
            source=CaptureFeedbackEvidenceSource.SYNTHETIC,
        )
    )

    assert report.evidence_status.value == "insufficient_real_world_evidence"
    assert expected in report.insufficiency_reasons


def test_calibration_report_decoder_rejects_noncanonical_or_tampered_bytes() -> None:
    report = evaluate_capture_feedback_dataset(_dataset(()))
    encoded = encode_capture_calibration_report(report)
    payload = json.loads(encoded)
    payload["final_test_support"] = 1
    tampered = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    duplicate = encoded[:-1] + b',"schema_version":"capture-calibration-report/v1"}'

    for invalid in (encoded + b"\n", tampered, duplicate, b"[]", b"{", b"\xff"):
        with pytest.raises(CaptureFeedbackError):
            decode_capture_calibration_report(invalid, installation_key=KEY)
    with pytest.raises(CaptureFeedbackError):
        decode_capture_calibration_report(
            encoded,
            installation_key=InstallationKey(b"z" * 32),
        )


def test_report_decoder_rejects_digest_recomputed_structural_forgeries() -> None:
    report = evaluate_capture_feedback_dataset(
        _dataset(
            tuple(
                _record(
                    index,
                    label=(
                        CaptureFeedbackLabel.MEMORY_NEEDED
                        if index <= 15
                        else CaptureFeedbackLabel.NOT_MEMORY_NEEDED
                    ),
                    prediction=(
                        CaptureFeedbackPrediction.MEMORY_NEEDED
                        if index <= 15
                        else CaptureFeedbackPrediction.NOT_MEMORY_NEEDED
                    ),
                )
                for index in range(1, 31)
            )
        )
    )

    inconsistent_metric = json.loads(encode_capture_calibration_report(report))
    precision = inconsistent_metric["pooled"]["metrics"]["precision"]
    precision["numerator"] = 0
    precision["value_ppm"] = 0

    inconsistent_provider = json.loads(encode_capture_calibration_report(report))
    provider = inconsistent_provider["providers"][0]
    provider["confusion"]["true_positive"] = 0
    provider["confusion"]["false_negative"] = 1
    provider["metrics"]["precision"] = None
    provider["metrics"]["recall"]["numerator"] = 0
    provider["metrics"]["recall"]["value_ppm"] = 0

    for forged in (inconsistent_metric, inconsistent_provider):
        with pytest.raises(CaptureFeedbackError):
            decode_capture_calibration_report(
                _recomputed_report_bytes(forged),
                installation_key=KEY,
            )


def test_calibration_module_has_no_runtime_or_intervention_activation_path() -> None:
    source_path = Path(feedback_module.__file__ or "")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    report = evaluate_capture_feedback_dataset(_dataset(()))
    keys = set(report.model_dump(mode="json"))

    assert not any(name.startswith("saliencegate.runtime") for name in imported)
    assert not any(name.startswith("saliencegate.intervention") for name in imported)
    assert {
        "activate",
        "activation",
        "enabled",
        "reminder",
        "inject",
        "injection",
    }.isdisjoint(keys)
    assert report.confirmatory is False
    assert report.decision_authority is False
