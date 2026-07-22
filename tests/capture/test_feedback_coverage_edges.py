from __future__ import annotations

from datetime import UTC, datetime
from fractions import Fraction

import pytest
from pydantic import ValidationError

import saliencegate.capture.feedback as feedback_module
from saliencegate.capture.capabilities import CaptureProfile
from saliencegate.capture.feedback import (
    CAPTURE_CALIBRATION_BOOTSTRAP_REPLICATES,
    CaptureCalibrationBootstrapInterval,
    CaptureCalibrationBoundaryStatus,
    CaptureCalibrationEstimate,
    CaptureCalibrationIntervalStatus,
    CaptureFeedbackBlindingStatus,
    CaptureFeedbackDataset,
    CaptureFeedbackError,
    CaptureFeedbackEvidenceSource,
    CaptureFeedbackLabel,
    CaptureFeedbackPartition,
    CaptureFeedbackPrediction,
    CaptureFeedbackReceipt,
    CaptureFeedbackRecordOrigin,
    CaptureFeedbackStudyAttestation,
    CaptureFeedbackWriteDisposition,
    _attestation_digest,
    _bootstrap_draw_index,
    _decode_calibration_report,
    _decode_dataset,
    _estimate,
    _interval,
    _seal_export_record,
    _verified_calibration_report,
    _verified_dataset,
    _verified_export_record,
    _verified_feedback_record,
    build_capture_feedback_dataset,
    build_synthetic_capture_feedback_export_record,
    decode_capture_calibration_report,
    decode_capture_feedback_dataset,
    encode_capture_calibration_report,
    encode_capture_feedback_dataset,
    evaluate_capture_feedback_dataset,
)
from saliencegate.security import InstallationKey

_KEY = InstallationKey(b"f" * 32)


def _dataset() -> CaptureFeedbackDataset:
    record = build_synthetic_capture_feedback_export_record(
        project_digest="a" * 64,
        profile_id=CaptureProfile.CODEX_HOOKS_V1,
        session_id="b" * 64,
        label=CaptureFeedbackLabel.MEMORY_NEEDED,
        prediction=CaptureFeedbackPrediction.MEMORY_NEEDED,
        partition=CaptureFeedbackPartition.FINAL_TEST,
        installation_key=_KEY,
    )
    return build_capture_feedback_dataset(
        (record,),
        installation_key=_KEY,
        export_nonce=b"n" * 32,
        evidence_source=CaptureFeedbackEvidenceSource.SYNTHETIC,
        opt_in=True,
    )


@pytest.mark.parametrize(
    ("disposition", "revision"),
    (
        (CaptureFeedbackWriteDisposition.RECORDED, 2),
        (CaptureFeedbackWriteDisposition.CHANGED, 1),
    ),
)
def test_feedback_receipt_rejects_disposition_revision_contradictions(
    disposition: CaptureFeedbackWriteDisposition,
    revision: int,
) -> None:
    with pytest.raises(ValidationError):
        CaptureFeedbackReceipt(
            session_id="abcdefghijkl",
            label=CaptureFeedbackLabel.MEMORY_NEEDED,
            disposition=disposition,
            revision_count=revision,
            labeled_at="2026-07-19T12:00:00Z",
        )


@pytest.mark.parametrize(
    "change",
    ("duplicate", "unassigned", "attestation_mismatch", "digest"),
)
def test_feedback_dataset_validator_rejects_each_binding_contradiction(change: str) -> None:
    dataset = _dataset()
    body = dataset.model_dump(mode="python")
    if change == "duplicate":
        body["examples"] = (dataset.examples[0], dataset.examples[0])
    elif change == "unassigned":
        body["examples"] = (
            dataset.examples[0].model_copy(
                update={"partition": CaptureFeedbackPartition.UNASSIGNED}
            ),
        )
    elif change == "attestation_mismatch":
        body["evidence_source"] = CaptureFeedbackEvidenceSource.DECLARED_E01
    else:
        body["dataset_digest"] = "0" * 64

    with pytest.raises(ValidationError):
        CaptureFeedbackDataset.model_validate(body)


def test_feedback_dataset_verification_rejects_wrong_types_and_tags() -> None:
    dataset = _dataset()
    with pytest.raises(CaptureFeedbackError):
        _verified_dataset(dataset, installation_key=object())  # type: ignore[arg-type]
    with pytest.raises(CaptureFeedbackError):
        _verified_dataset(
            dataset.model_copy(update={"dataset_tag": "0" * 64}),
            installation_key=_KEY,
        )


def test_export_record_sealing_rejects_owned_tags_and_invalid_bodies() -> None:
    with pytest.raises(CaptureFeedbackError):
        _seal_export_record({"record_tag": "0" * 64}, installation_key=_KEY)
    with pytest.raises(CaptureFeedbackError):
        _seal_export_record({}, installation_key=_KEY)
    with pytest.raises(CaptureFeedbackError):
        _seal_export_record({}, installation_key=object())  # type: ignore[arg-type]


def test_dataset_codec_fail_closed_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _dataset()
    encoded = encode_capture_feedback_dataset(dataset)
    assert decode_capture_feedback_dataset(encoded, installation_key=_KEY) == dataset
    with pytest.raises(ValueError):
        _decode_dataset("not-bytes")  # type: ignore[arg-type]
    with pytest.raises(CaptureFeedbackError):
        decode_capture_feedback_dataset(encoded, installation_key=object())  # type: ignore[arg-type]

    monkeypatch.setattr(feedback_module, "_model_is_exact", lambda *_args: False)
    with pytest.raises(ValueError):
        _decode_dataset(encoded)
    with pytest.raises(CaptureFeedbackError):
        encode_capture_feedback_dataset(dataset)


@pytest.mark.parametrize(
    "change",
    ("reversed", "zero_width_marked_interior", "range_marked_degenerate"),
)
def test_bootstrap_interval_rejects_order_and_boundary_mismatches(change: str) -> None:
    body = {
        "lower_ppm": 100_000,
        "upper_ppm": 200_000,
        "boundary_status": CaptureCalibrationBoundaryStatus.INTERIOR,
    }
    if change == "reversed":
        body.update(lower_ppm=300_000, upper_ppm=200_000)
    elif change == "zero_width_marked_interior":
        body.update(lower_ppm=200_000, upper_ppm=200_000)
    else:
        body["boundary_status"] = CaptureCalibrationBoundaryStatus.DEGENERATE_ZERO_WIDTH

    with pytest.raises(ValidationError):
        CaptureCalibrationBootstrapInterval.model_validate(body)


@pytest.mark.parametrize(
    "change",
    (
        "ratio",
        "estimated_without_interval",
        "insufficient_with_interval",
        "undefined_without_count",
        "small_estimated_sample",
    ),
)
def test_calibration_estimate_rejects_ratio_interval_and_support_conflicts(change: str) -> None:
    interval = CaptureCalibrationBootstrapInterval(
        lower_ppm=400_000,
        upper_ppm=600_000,
        boundary_status=CaptureCalibrationBoundaryStatus.INTERIOR,
    )
    body: dict[str, object] = {
        "numerator": 15,
        "denominator": 30,
        "value_ppm": 500_000,
        "interval": interval,
        "interval_status": CaptureCalibrationIntervalStatus.ESTIMATED,
        "undefined_bootstrap_replicates": 0,
    }
    if change == "ratio":
        body["value_ppm"] = 500_001
    elif change == "estimated_without_interval":
        body["interval"] = None
    elif change == "insufficient_with_interval":
        body["interval_status"] = CaptureCalibrationIntervalStatus.INSUFFICIENT_SUPPORT
    elif change == "undefined_without_count":
        body.update(
            interval=None,
            interval_status=CaptureCalibrationIntervalStatus.UNDEFINED_BOOTSTRAP_REPLICATE,
        )
    else:
        body.update(numerator=1, denominator=2, value_ppm=500_000)

    with pytest.raises(ValidationError):
        CaptureCalibrationEstimate.model_validate(body)


def test_bootstrap_helpers_cover_invalid_rejection_and_interval_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(CaptureFeedbackError):
        _bootstrap_draw_index(size=0, replicate=0, profile="p", project_id="q", draw=0)

    monkeypatch.setattr(
        feedback_module, "length_prefixed_sha256", lambda *_args, **_kwargs: "f" * 64
    )
    with pytest.raises(CaptureFeedbackError):
        _bootstrap_draw_index(size=3, replicate=0, profile="p", project_id="q", draw=0)

    assert _interval([], undefined=1) is None
    degenerate = _interval(
        [Fraction(1, 2)] * CAPTURE_CALIBRATION_BOOTSTRAP_REPLICATES,
        undefined=0,
    )
    assert degenerate is not None
    assert degenerate.boundary_status is CaptureCalibrationBoundaryStatus.DEGENERATE_ZERO_WIDTH
    interior = _interval(
        [Fraction(0)] * 50 + [Fraction(1)] * (CAPTURE_CALIBRATION_BOOTSTRAP_REPLICATES - 50),
        undefined=0,
    )
    assert interior is not None
    assert interior.boundary_status is CaptureCalibrationBoundaryStatus.INTERIOR


def test_estimate_helper_covers_undefined_disabled_and_estimated_paths() -> None:
    counts = {
        "true_positive": 15,
        "false_negative": 15,
        "abstained_memory_needed": 0,
        "false_positive": 0,
        "true_negative": 0,
        "abstained_not_memory_needed": 0,
        "uncertain_predicted_memory_needed": 0,
        "uncertain_predicted_not_memory_needed": 0,
        "uncertain_prediction_abstained": 0,
    }
    zeros = {name: 0 for name in counts}
    assert (
        _estimate(
            zeros,
            "prevalence",
            values=[],
            undefined=0,
            bootstrap_enabled=False,
        )
        is None
    )
    disabled = _estimate(
        counts,
        "prevalence",
        values=[],
        undefined=0,
        bootstrap_enabled=False,
    )
    assert disabled is not None
    assert disabled.interval_status is CaptureCalibrationIntervalStatus.INSUFFICIENT_SUPPORT
    undefined = _estimate(
        counts,
        "prevalence",
        values=[],
        undefined=1,
        bootstrap_enabled=True,
    )
    assert undefined is not None
    assert (
        undefined.interval_status is CaptureCalibrationIntervalStatus.UNDEFINED_BOOTSTRAP_REPLICATE
    )
    estimated = _estimate(
        counts,
        "prevalence",
        values=[Fraction(1, 2)] * CAPTURE_CALIBRATION_BOOTSTRAP_REPLICATES,
        undefined=0,
        bootstrap_enabled=True,
    )
    assert estimated is not None
    assert estimated.interval_status is CaptureCalibrationIntervalStatus.ESTIMATED


def test_calibration_report_verification_and_codec_fail_closed_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = evaluate_capture_feedback_dataset(_dataset(), installation_key=_KEY)
    encoded = encode_capture_calibration_report(report)
    assert decode_capture_calibration_report(encoded, installation_key=_KEY) == report
    with pytest.raises(CaptureFeedbackError):
        _verified_calibration_report(report, installation_key=object())  # type: ignore[arg-type]
    with pytest.raises(CaptureFeedbackError):
        _verified_calibration_report(
            report.model_copy(update={"report_tag": "0" * 64}),
            installation_key=_KEY,
        )
    with pytest.raises(ValueError):
        _decode_calibration_report("not-bytes")  # type: ignore[arg-type]
    with pytest.raises(CaptureFeedbackError):
        decode_capture_calibration_report(encoded, installation_key=object())  # type: ignore[arg-type]

    monkeypatch.setattr(feedback_module, "_model_is_exact", lambda *_args: False)
    with pytest.raises(ValueError):
        _decode_calibration_report(encoded)
    with pytest.raises(CaptureFeedbackError):
        encode_capture_calibration_report(report)


def _attestation(
    status: CaptureFeedbackBlindingStatus = CaptureFeedbackBlindingStatus.EXTERNALLY_ATTESTED,
) -> CaptureFeedbackStudyAttestation:
    return CaptureFeedbackStudyAttestation(
        consent_protocol_digest="1" * 64,
        preregistration_digest="2" * 64,
        temporal_split_digest="3" * 64,
        evaluator_freeze_digest="4" * 64,
        label_freeze_digest="5" * 64,
        report_selection_policy_digest="6" * 64,
        no_test_tuning_digest="7" * 64,
        blinding_status=status,
    )


def _export_record_with_origin(origin: CaptureFeedbackRecordOrigin) -> object:
    record = build_synthetic_capture_feedback_export_record(
        project_digest="a" * 64,
        profile_id=CaptureProfile.CODEX_HOOKS_V1,
        session_id="b" * 64,
        label=CaptureFeedbackLabel.MEMORY_NEEDED,
        prediction=CaptureFeedbackPrediction.MEMORY_NEEDED,
        partition=CaptureFeedbackPartition.FINAL_TEST,
        installation_key=_KEY,
    )
    body = record.model_dump(mode="python")
    body.pop("record_tag")
    body["origin"] = origin
    return _seal_export_record(body, installation_key=_KEY)


def test_feedback_direct_defensive_type_and_receipt_branches() -> None:
    with pytest.raises(ValueError, match="text"):
        feedback_module._require_exact_text(1)  # type: ignore[arg-type]

    receipt = CaptureFeedbackReceipt(
        session_id="abcdefghijkl",
        label=CaptureFeedbackLabel.MEMORY_NEEDED,
        disposition=CaptureFeedbackWriteDisposition.UNCHANGED,
        revision_count=2,
        labeled_at=datetime(2026, 7, 19, 12, tzinfo=UTC),
    )
    forged = receipt.model_copy(
        update={
            "disposition": CaptureFeedbackWriteDisposition.RECORDED,
            "revision_count": 2,
        }
    )
    with pytest.raises(ValueError, match="disposition"):
        forged.disposition_matches_revision()

    with pytest.raises(CaptureFeedbackError):
        _verified_feedback_record(object(), installation_key=_KEY)
    with pytest.raises(CaptureFeedbackError):
        _verified_export_record(object(), installation_key=_KEY)


def test_feedback_dataset_source_contract_rejects_cross_origin_records() -> None:
    local = _export_record_with_origin(CaptureFeedbackRecordOrigin.LOCAL_CAPTURE_REPORT)
    synthetic = _export_record_with_origin(CaptureFeedbackRecordOrigin.SYNTHETIC_FIXTURE)

    with pytest.raises(CaptureFeedbackError):
        build_capture_feedback_dataset(
            (local,),  # type: ignore[arg-type]
            installation_key=_KEY,
            export_nonce=b"l" * 32,
            evidence_source=CaptureFeedbackEvidenceSource.SYNTHETIC,
            opt_in=True,
        )
    with pytest.raises(CaptureFeedbackError):
        build_capture_feedback_dataset(
            (synthetic,),  # type: ignore[arg-type]
            installation_key=_KEY,
            export_nonce=b"m" * 32,
            evidence_source=CaptureFeedbackEvidenceSource.LOCAL_FEEDBACK,
            opt_in=True,
        )
    with pytest.raises(CaptureFeedbackError):
        build_capture_feedback_dataset(
            (local,),  # type: ignore[arg-type]
            installation_key=_KEY,
            export_nonce=b"n" * 32,
            evidence_source=CaptureFeedbackEvidenceSource.DECLARED_E01,
            opt_in=True,
        )


def test_feedback_dataset_final_exactness_and_attestation_digest_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feedback_module, "_model_is_exact", lambda *_args: False)
    with pytest.raises(CaptureFeedbackError):
        build_capture_feedback_dataset(
            (),
            installation_key=_KEY,
            export_nonce=b"z" * 32,
            evidence_source=CaptureFeedbackEvidenceSource.SYNTHETIC,
            opt_in=True,
        )
    monkeypatch.undo()
    assert _attestation_digest(_attestation()) is not None


def test_e01_process_attestation_and_evaluator_exception_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _export_record_with_origin(CaptureFeedbackRecordOrigin.LOCAL_CAPTURE_REPORT)
    dataset = build_capture_feedback_dataset(
        (local,),  # type: ignore[arg-type]
        installation_key=_KEY,
        export_nonce=b"e" * 32,
        evidence_source=CaptureFeedbackEvidenceSource.DECLARED_E01,
        opt_in=True,
        study_attestation=_attestation(CaptureFeedbackBlindingStatus.NOT_BLINDED),
    )
    report = evaluate_capture_feedback_dataset(dataset, installation_key=_KEY)
    assert feedback_module.CaptureCalibrationInsufficiency.PROCESS_ATTESTATION in (
        report.insufficiency_reasons
    )

    monkeypatch.setattr(
        feedback_module,
        "encode_capture_feedback_dataset",
        lambda _dataset: (_ for _ in ()).throw(CaptureFeedbackError()),
    )
    with pytest.raises(CaptureFeedbackError):
        evaluate_capture_feedback_dataset(dataset, installation_key=_KEY)
    monkeypatch.setattr(
        feedback_module,
        "encode_capture_feedback_dataset",
        lambda _dataset: (_ for _ in ()).throw(RuntimeError()),
    )
    with pytest.raises(CaptureFeedbackError):
        evaluate_capture_feedback_dataset(dataset, installation_key=_KEY)


def test_calibration_encoder_rejects_a_nonidentical_defensive_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = evaluate_capture_feedback_dataset(_dataset(), installation_key=_KEY)
    monkeypatch.setattr(
        feedback_module,
        "_decode_calibration_report",
        lambda _data: object(),
    )
    with pytest.raises(CaptureFeedbackError):
        encode_capture_calibration_report(report)
