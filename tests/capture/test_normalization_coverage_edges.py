from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import ValidationError

import saliencegate.capture.normalization as normalization_module
from saliencegate.capture.capabilities import (
    CapabilitySupport,
    CaptureProfile,
    capture_profile,
)
from saliencegate.capture.normalization import (
    CaptureDetectorEvidence,
    CaptureNormalization,
    CaptureNormalizationCounts,
    CaptureNormalizationDiagnostic,
    CaptureNormalizationDiagnosticCode,
    CaptureNormalizationError,
    _authorized_structured_error_code,
    _capture_detector_minimum,
    _detector_observation_counts,
    _detector_record_is_eligible,
    _extractor,
    _ProjectedRecord,
    _structured_status_is_authorized,
    _trace_events,
    normalize_capture_session_snapshot,
    verify_capture_normalization,
)
from saliencegate.domain import SignalType
from saliencegate.security import InstallationKey

_KEY = InstallationKey(b"n" * 32)
_RUN_ID = UUID("11111111-1111-4111-8111-111111111111")


def _counts(**changes: int) -> CaptureNormalizationCounts:
    body = {
        "source_event_count": 0,
        "mapped_event_count": 0,
        "ignored_event_count": 0,
        "action_identity_count": 0,
        "exact_action_identity_count": 0,
        "authorized_tool_result_count": 0,
        "classifiable_failed_result_count": 0,
        "exact_parent_classifiable_failed_result_count": 0,
        "authorized_controller_error_count": 0,
    }
    body.update(changes)
    return CaptureNormalizationCounts.model_validate(body)


def _empty_normalization() -> CaptureNormalization:
    return CaptureNormalization(
        snapshot_digest="a" * 64,
        run_id=_RUN_ID,
        shadow_trace=None,
        events=(),
        extraction_reports=(),
        detector_evidence=(),
        diagnostics=(),
        counts=_counts(),
        semantic_coherence=True,
        normalization_digest="b" * 64,
    )


def _evidence(signal_type: SignalType) -> CaptureDetectorEvidence:
    return CaptureDetectorEvidence(
        signal_type=signal_type,
        support=CapabilitySupport.CONDITIONAL,
        omissions=("bounded_fixture",),
        minimum_authorized_observations=(
            2 if signal_type in (SignalType.REPEATED_ACTION, SignalType.REPEATED_FAILURE) else 1
        ),
        authorized_observation_count=0,
        unresolved_observation_count=0,
        minimum_observation_met=False,
    )


def test_normalization_diagnostic_requires_both_event_coordinates() -> None:
    with pytest.raises(ValidationError):
        CaptureNormalizationDiagnostic(
            receipt_ordinal=1,
            event_kind=None,
            code=CaptureNormalizationDiagnosticCode.EVENT_NOT_PROJECTABLE,
        )


@pytest.mark.parametrize(
    "change",
    (
        "source_equation",
        "exact_actions",
        "failed_results",
        "exact_failed_results",
        "mapped_components",
    ),
)
def test_normalization_counts_reject_each_exhaustiveness_contradiction(change: str) -> None:
    body = {
        "source_event_count": 1,
        "mapped_event_count": 1,
        "ignored_event_count": 0,
        "action_identity_count": 0,
        "exact_action_identity_count": 0,
        "authorized_tool_result_count": 0,
        "classifiable_failed_result_count": 0,
        "exact_parent_classifiable_failed_result_count": 0,
        "authorized_controller_error_count": 0,
    }
    if change == "source_equation":
        body["source_event_count"] = 2
    elif change == "exact_actions":
        body["exact_action_identity_count"] = 1
    elif change == "failed_results":
        body["classifiable_failed_result_count"] = 1
    elif change == "exact_failed_results":
        body["exact_parent_classifiable_failed_result_count"] = 1
    else:
        body.update(
            action_identity_count=1,
            authorized_tool_result_count=1,
        )

    with pytest.raises(ValidationError):
        CaptureNormalizationCounts.model_validate(body)


@pytest.mark.parametrize(
    "change",
    ("unsupported", "total", "threshold", "duplicate_omission"),
)
def test_detector_evidence_rejects_support_threshold_and_canonicality_edges(
    change: str,
) -> None:
    body = _evidence(SignalType.REPEATED_ACTION).model_dump(mode="python")
    if change == "unsupported":
        body["support"] = CapabilitySupport.UNSUPPORTED
    elif change == "total":
        body.update(authorized_observation_count=1_000, unresolved_observation_count=1)
    elif change == "threshold":
        body["minimum_observation_met"] = True
    else:
        body["omissions"] = ("bounded_fixture", "bounded_fixture")

    with pytest.raises(ValidationError):
        CaptureDetectorEvidence.model_validate(body)


@pytest.mark.parametrize(
    "change",
    ("mapped_count", "too_many_diagnostics", "detector_order"),
)
def test_normalization_projection_rejects_topology_contradictions(change: str) -> None:
    normalization = _empty_normalization()
    body = normalization.model_dump(mode="python")
    body["shadow_trace"] = None
    if change == "mapped_count":
        body["counts"] = _counts(source_event_count=1, mapped_event_count=1)
    elif change == "too_many_diagnostics":
        diagnostic = CaptureNormalizationDiagnostic(
            code=CaptureNormalizationDiagnosticCode.EVENT_NOT_PROJECTABLE
        )
        body["diagnostics"] = (diagnostic, diagnostic, diagnostic)
    else:
        body["detector_evidence"] = (
            _evidence(SignalType.TOOL_ERROR),
            _evidence(SignalType.REPEATED_ACTION),
        )

    with pytest.raises(ValidationError):
        CaptureNormalization.model_validate(body)


@pytest.mark.parametrize(
    ("status", "authorities", "expected"),
    (
        ("succeeded", frozenset({"provider_claimed_success"}), True),
        ("succeeded", frozenset(), False),
        ("failed", frozenset({"provider_claimed_failure"}), True),
        ("failed", frozenset(), False),
        (None, frozenset({"provider_claimed_success"}), False),
    ),
)
def test_structured_status_authority_is_closed(
    status: str | None,
    authorities: frozenset[str],
    expected: bool,
) -> None:
    assert (
        _structured_status_is_authorized(status, authorities)  # type: ignore[arg-type]
        is expected
    )


def test_structured_error_classification_is_closed_and_fail_closed() -> None:
    assert _authorized_structured_error_code(None, frozenset()) is None
    assert (
        _authorized_structured_error_code(
            "failed",
            frozenset({"provider_claimed_failure"}),
        )
        == "provider_error"
    )
    assert (
        _authorized_structured_error_code(
            "failed",
            frozenset({"provider_claimed_tool_outcome"}),
        )
        == "tool_error"
    )
    with pytest.raises(CaptureNormalizationError):
        _authorized_structured_error_code("failed", frozenset())


def test_detector_minima_and_eligibility_are_closed() -> None:
    assert _capture_detector_minimum(SignalType.REPEATED_ACTION) == 2
    assert _capture_detector_minimum(SignalType.TOOL_ERROR) == 1
    with pytest.raises(CaptureNormalizationError):
        _capture_detector_minimum(SignalType.CONTEXT_SHIFT)
    with pytest.raises(CaptureNormalizationError):
        _capture_detector_minimum("repeated_action")  # type: ignore[arg-type]

    projected = _ProjectedRecord(
        receipt_ordinal=1,
        event_kind="action_started",
        value=object(),  # type: ignore[arg-type]
        wire={},
        exact_action=True,
    )
    assert _detector_record_is_eligible(
        SignalType.REPEATED_ACTION,
        projected,
        object(),  # type: ignore[arg-type]
        exact_actions=frozenset(),
    )
    assert not _detector_record_is_eligible(
        SignalType.TEST_FAILURE,
        projected,
        object(),  # type: ignore[arg-type]
        exact_actions=frozenset(),
    )
    with pytest.raises(CaptureNormalizationError):
        _detector_record_is_eligible(
            SignalType.CONTEXT_SHIFT,
            projected,
            object(),  # type: ignore[arg-type]
            exact_actions=frozenset(),
        )


def test_extractor_rejects_a_manifest_detector_without_a_capture_implementation() -> None:
    unsupported = next(
        item
        for item in capture_profile(CaptureProfile.CODEX_HOOKS_V1).detectors
        if item.signal_type
        not in {
            SignalType.REPEATED_ACTION,
            SignalType.REPEATED_FAILURE,
            SignalType.TEST_FAILURE,
            SignalType.TOOL_ERROR,
        }
    )
    with pytest.raises(CaptureNormalizationError):
        _extractor((unsupported,))


@pytest.mark.parametrize("failure", (CaptureNormalizationError, RuntimeError))
def test_normalization_entrypoint_preserves_contract_errors_and_sanitizes_other_failures(
    monkeypatch: pytest.MonkeyPatch,
    failure: type[Exception],
) -> None:
    def fail_verification(*_args: object, **_kwargs: object) -> object:
        raise failure("fixture-sensitive-detail")

    monkeypatch.setattr(normalization_module, "verify_capture_session_snapshot", fail_verification)
    with pytest.raises(CaptureNormalizationError) as captured:
        normalize_capture_session_snapshot(object(), installation_key=_KEY)
    assert captured.value.__cause__ is None


def test_normalization_entrypoints_preserve_process_interrupts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt(*_args: object, **_kwargs: object) -> object:
        raise KeyboardInterrupt()

    monkeypatch.setattr(normalization_module, "verify_capture_session_snapshot", interrupt)
    with pytest.raises(KeyboardInterrupt):
        normalize_capture_session_snapshot(object(), installation_key=_KEY)

    monkeypatch.setattr(normalization_module, "normalize_capture_session_snapshot", interrupt)
    with pytest.raises(KeyboardInterrupt):
        verify_capture_normalization(
            _empty_normalization(),
            snapshot=object(),
            installation_key=_KEY,
        )


def test_normalization_verifier_rejects_wrong_type_and_sanitizes_unexpected_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(CaptureNormalizationError):
        verify_capture_normalization(object(), snapshot=object(), installation_key=_KEY)

    def fail(*_args: object, **_kwargs: object) -> CaptureNormalization:
        raise RuntimeError("fixture-sensitive-detail")

    monkeypatch.setattr(normalization_module, "normalize_capture_session_snapshot", fail)
    with pytest.raises(CaptureNormalizationError) as captured:
        verify_capture_normalization(
            _empty_normalization(),
            snapshot=object(),
            installation_key=_KEY,
        )
    assert captured.value.__cause__ is None


@pytest.mark.parametrize("change", ("event_identity", "detector_selection"))
def test_normalization_direct_projection_revalidation_covers_inner_sequence_edges(
    change: str,
) -> None:
    normalization = _empty_normalization()
    event_id = UUID("22222222-2222-4222-8222-222222222222")
    event = SimpleNamespace(run_id=_RUN_ID, sequence=1, event_id=event_id)
    report = SimpleNamespace(run_id=_RUN_ID, current_event_id=event_id, evaluations=())
    if change == "event_identity":
        event.run_id = UUID("33333333-3333-4333-8333-333333333333")
    else:
        report.evaluations = (SimpleNamespace(signal_type=SignalType.TOOL_ERROR),)
    counts = normalization.counts.model_copy(
        update={
            "source_event_count": 1,
            "mapped_event_count": 1,
            "action_identity_count": 1,
            "exact_action_identity_count": 1,
        }
    )
    forged = normalization.model_copy(
        update={
            "shadow_trace": object(),
            "events": (event,),
            "extraction_reports": (report,),
            "counts": counts,
        }
    )
    with pytest.raises(ValueError):
        forged.projection_is_consistent()


def test_trace_materialization_rejects_any_redaction_finding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projected = (
        _ProjectedRecord(
            receipt_ordinal=1,
            event_kind="session_started",  # type: ignore[arg-type]
            value=object(),  # type: ignore[arg-type]
            wire={},
            marker="start",
        ),
    )
    monkeypatch.setattr(normalization_module, "project_shadow_input", lambda *_a, **_k: object())

    class _Redactor:
        def redact_event(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(findings=(object(),), event=object())

    monkeypatch.setattr(normalization_module, "Redactor", _Redactor)
    with pytest.raises(CaptureNormalizationError):
        _trace_events(
            projected,
            trace=SimpleNamespace(binding=SimpleNamespace(source_adapter="capture/test")),
            run_id=_RUN_ID,
            installation_key=_KEY,
        )


def test_repeated_failure_eligibility_and_evaluation_cardinality_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projected = _ProjectedRecord(
        receipt_ordinal=1,
        event_kind="action_finished",  # type: ignore[arg-type]
        value=object(),  # type: ignore[arg-type]
        wire={},
        authorized_tool_result=True,
    )
    monkeypatch.setattr(normalization_module, "_is_classifiable_failed_result", lambda *_a: True)
    monkeypatch.setattr(normalization_module, "_has_exact_action_parent", lambda *_a: True)
    assert _detector_record_is_eligible(
        SignalType.REPEATED_FAILURE,
        projected,
        object(),  # type: ignore[arg-type]
        exact_actions=frozenset(),
    )

    with pytest.raises(CaptureNormalizationError):
        _detector_observation_counts(
            SignalType.TOOL_ERROR,
            (projected,),
            (object(),),  # type: ignore[arg-type]
            (SimpleNamespace(evaluations=()),),
        )


def test_normalization_capability_mismatch_and_contract_rethrow_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = SimpleNamespace(
        profile_id=CaptureProfile.CODEX_HOOKS_V1,
        capability_manifest_digest="0" * 64,
    )
    monkeypatch.setattr(
        normalization_module,
        "verify_capture_session_snapshot",
        lambda *_args, **_kwargs: verified,
    )
    with pytest.raises(CaptureNormalizationError):
        normalize_capture_session_snapshot(object(), installation_key=_KEY)

    monkeypatch.setattr(
        normalization_module,
        "verify_capture_session_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(CaptureNormalizationError()),
    )
    with pytest.raises(CaptureNormalizationError):
        normalize_capture_session_snapshot(object(), installation_key=_KEY)
