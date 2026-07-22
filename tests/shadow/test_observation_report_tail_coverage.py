"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from tests.shadow.conftest import OTHER_RUN_ID, TraceEventFactory
from tests.shadow.test_analyzer import _memory_session
from tests.shadow.test_report import (
    REDACTION_POLICY_TAG,
    _builder_kwargs,
    _event,
    _make_case,
    _observation,
)
from tests.shadow.test_trusted_observation_path import _mixed_trace

import saliencegate.shadow.analyzer as analyzer_module
import saliencegate.shadow.observation as observation_module
import saliencegate.shadow.report as report_module
from saliencegate.domain import SignalType, TrustLabel
from saliencegate.shadow.config import ShadowConfig
from saliencegate.shadow.errors import ShadowInvariantError
from saliencegate.shadow.inputs import (
    SHADOW_PROJECTION_MATRIX,
    ShadowInputKind,
)
from saliencegate.signals import (
    AbstentionReason,
    DetectionContext,
    DetectionOutcome,
    ExtractionReport,
)
from saliencegate.signals.base import (
    _admit_detection_sequence,
    _extract_trusted_report,
    _longest_trusted_detection_context,
)


def _feature_case(trace_event_factory: TraceEventFactory) -> dict[str, Any]:
    event = _event(trace_event_factory, 1, ShadowInputKind.START)
    observation = _observation(
        prefix=(event,),
        kind=ShadowInputKind.START,
        cli_input_ordinal=1,
    )
    context = DetectionContext(run_id=event.run_id, events=(event,))
    report = ExtractionReport(
        run_id=event.run_id,
        current_event_id=event.event_id,
        current_event_timestamp=event.timestamp,
        evaluations=observation.detector_evaluations,
        signals=observation.detected_signals,
    )
    return {
        "prefix": (event,),
        "context": context,
        "report": report,
        "config": ShadowConfig.reference(),
        "input_kind": ShadowInputKind.START,
        "heuristic": observation.heuristic_evaluations[0],
        "source_event_digest": observation.source_event_digest,
        "redaction_policy_tag": REDACTION_POLICY_TAG,
        "cli_input_ordinal": 1,
    }


def _replace_evaluation_outcome(
    observation: Any,
    signal_type: SignalType,
    outcome: DetectionOutcome,
) -> Any:
    evaluations = tuple(
        evaluation.model_copy(update={"outcome": outcome})
        if evaluation.signal_type is signal_type
        else evaluation
        for evaluation in observation.detector_evaluations
    )
    return observation.model_copy(update={"detector_evaluations": evaluations})


def test_detection_context_copy_rejects_shape_exact_invalid_fields(
    trace_event_factory: TraceEventFactory,
) -> None:
    context = _feature_case(trace_event_factory)["context"]
    context.__dict__["run_id"] = UUID(int=0)

    with pytest.raises(ValueError, match="preflight validation"):
        observation_module._copy_detection_context(context)


def test_detection_context_copy_rejects_bounded_equality_drift(
    trace_event_factory: TraceEventFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _feature_case(trace_event_factory)["context"]
    monkeypatch.setattr(DetectionContext, "__eq__", lambda _self, _other: False)

    with pytest.raises(ValueError, match="bounded validation"):
        observation_module._copy_detection_context(context)


def test_detection_context_copy_rejects_defensive_round_trip_drift(
    trace_event_factory: TraceEventFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _feature_case(trace_event_factory)["context"]
    drifted = context.model_copy(update={"run_id": OTHER_RUN_ID})
    monkeypatch.setattr(
        DetectionContext,
        "model_validate_json",
        classmethod(lambda _cls, _value: drifted),
    )

    with pytest.raises(ValueError, match="defensive validation"):
        observation_module._copy_detection_context(context)


@pytest.mark.parametrize(
    "update",
    (
        {"supported_signal_types": ()},
        {"unsupported_signal_types": ()},
    ),
)
def test_feature_validation_rejects_forged_detector_declarations(
    trace_event_factory: TraceEventFactory,
    monkeypatch: pytest.MonkeyPatch,
    update: dict[str, object],
) -> None:
    values = _feature_case(trace_event_factory)
    values["config"] = values["config"].model_copy(update=update)
    monkeypatch.setattr(observation_module, "_copy_shadow_config", lambda value: value)

    with pytest.raises(ValueError, match="detector set is invalid"):
        observation_module._validate_feature_inputs(
            prefix=values["prefix"],
            context=values["context"],
            report=values["report"],
            config=values["config"],
        )


def test_extraction_digest_public_boundary_fails_closed() -> None:
    with pytest.raises(ShadowInvariantError):
        observation_module.derive_shadow_extraction_report_digest(object())  # type: ignore[arg-type]


def test_observation_validator_rejects_incomplete_detector_profile(
    trace_event_factory: TraceEventFactory,
) -> None:
    observation = _make_case(trace_event_factory).observations[0]
    damaged = observation.model_copy(
        update={"detector_evaluations": tuple(reversed(observation.detector_evaluations))}
    )

    with pytest.raises(ValueError, match="complete detector profile"):
        damaged.fields_form_one_non_authoritative_observation()


def test_observation_validator_rejects_signal_attribution_drift(
    trace_event_factory: TraceEventFactory,
) -> None:
    observation = _make_case(trace_event_factory).observations[2]
    first, *remaining = observation.detected_signals
    damaged_signal = first.model_copy(update={"run_id": OTHER_RUN_ID})
    damaged = observation.model_copy(update={"detected_signals": (damaged_signal, *remaining)})

    with pytest.raises(ValueError, match="signal attribution is inconsistent"):
        damaged.fields_form_one_non_authoritative_observation()


def test_trusted_sequence_exactness_rejects_untrusted_objects() -> None:
    assert observation_module._trusted_shadow_observation_sequence_is_exact(object()) is False


def test_observation_admission_rechecks_sealed_event_semantics(
    trace_event_factory: TraceEventFactory,
) -> None:
    event = _event(trace_event_factory, 1, ShadowInputKind.START)
    sequence = _admit_detection_sequence((event,))
    sequence.events[0].__dict__["sequence"] = 2

    with pytest.raises(ShadowInvariantError):
        observation_module._admit_shadow_observation_sequence(
            sequence,
            config=ShadowConfig.reference(),
            redaction_policy_tag=REDACTION_POLICY_TAG,
        )


def _trusted_builder_case() -> tuple[Any, Any, Any]:
    trace = _mixed_trace()
    session = _memory_session(trace)
    prepared = analyzer_module._prepare_analysis(session, trace)
    sequence = _admit_detection_sequence(prepared.expected_events)
    admission = observation_module._admit_shadow_observation_sequence(
        sequence,
        config=session._config,
        redaction_policy_tag=session._redaction_policy_tag,
    )
    trusted_context = _longest_trusted_detection_context(sequence, len(sequence.events))
    extraction = _extract_trusted_report(session._extractor, trusted_context)
    return admission, extraction, prepared.events[-1]


def test_trusted_observation_builder_rejects_evidence_identity_drift() -> None:
    admission, extraction, item = _trusted_builder_case()

    with pytest.raises(ShadowInvariantError):
        observation_module._build_shadow_observation_trusted(
            admission,
            extraction,
            input_kind=item.row.input_kind,
            source_event_digest="0" * 64,
            cli_input_ordinal=item.row.input_ordinal,
        )


def test_trusted_observation_builder_rechecks_detector_versions() -> None:
    admission, extraction, item = _trusted_builder_case()
    admission.config.detectors[0].__dict__["detector_version"] = "observation-report-drift"

    with pytest.raises(ShadowInvariantError):
        observation_module._build_shadow_observation_trusted(
            admission,
            extraction,
            input_kind=item.row.input_kind,
            source_event_digest=item.row.source_event_digest,
            cli_input_ordinal=item.row.input_ordinal,
        )


def test_event_kind_match_rejects_invalid_observation_trust(
    trace_event_factory: TraceEventFactory,
) -> None:
    spec = SHADOW_PROJECTION_MATRIX[ShadowInputKind.OBSERVATION]
    event = trace_event_factory(
        1,
        event_type=spec.event_type,
        phase=spec.phase,
        payload={spec.payload_namespace: {"bounded": True}},
        trust_label=TrustLabel.SYNTHETIC_FIXTURE,
    )

    assert observation_module._event_matches_input_kind(event, ShadowInputKind.OBSERVATION) is False


def test_event_kind_match_rejects_invalid_fixed_trust(
    trace_event_factory: TraceEventFactory,
) -> None:
    event = _event(trace_event_factory, 1, ShadowInputKind.START).model_copy(
        update={"trust_label": TrustLabel.SYNTHETIC_FIXTURE}
    )

    assert observation_module._event_matches_input_kind(event, ShadowInputKind.START) is False


def test_context_selection_rejects_invalid_singleton(
    trace_event_factory: TraceEventFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = (_event(trace_event_factory, 1, ShadowInputKind.START),)
    monkeypatch.setattr(observation_module, "_try_detection_context", lambda *_args: None)

    with pytest.raises(ShadowInvariantError):
        observation_module._select_detection_context(prefix)


def test_context_selection_rejects_failed_final_validation(
    trace_event_factory: TraceEventFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = (_event(trace_event_factory, 1, ShadowInputKind.START),)
    calls = iter((True, False))

    def selected_or_none(run_id: UUID, events: tuple[Any, ...]) -> DetectionContext | None:
        return DetectionContext(run_id=run_id, events=events) if next(calls) else None

    monkeypatch.setattr(observation_module, "_try_detection_context", selected_or_none)

    with pytest.raises(ShadowInvariantError):
        observation_module._select_detection_context(prefix)


def test_context_selection_rejects_nonmaximal_suffix(
    trace_event_factory: TraceEventFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = (
        _event(trace_event_factory, 1, ShadowInputKind.START),
        _event(trace_event_factory, 2, ShadowInputKind.ACTION),
    )
    calls = iter((True, False, True, True))

    def selected_or_none(run_id: UUID, events: tuple[Any, ...]) -> DetectionContext | None:
        return DetectionContext(run_id=run_id, events=events) if next(calls) else None

    monkeypatch.setattr(observation_module, "_try_detection_context", selected_or_none)

    with pytest.raises(ShadowInvariantError):
        observation_module._select_detection_context(prefix)


def test_observation_builder_rejects_different_public_selection(
    trace_event_factory: TraceEventFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _feature_case(trace_event_factory)
    different = values["context"].model_copy(update={"run_id": OTHER_RUN_ID})
    monkeypatch.setattr(observation_module, "select_detection_context", lambda _prefix: different)

    with pytest.raises(ShadowInvariantError):
        observation_module._build_shadow_observation(selection=None, **values)


def test_observation_builder_rejects_forged_selection_proof(
    trace_event_factory: TraceEventFactory,
) -> None:
    values = _feature_case(trace_event_factory)

    with pytest.raises(ShadowInvariantError):
        observation_module._build_shadow_observation(selection=object(), **values)  # type: ignore[arg-type]


def test_observation_builder_rejects_invalid_cli_ordinal(
    trace_event_factory: TraceEventFactory,
) -> None:
    values = _feature_case(trace_event_factory)
    values["cli_input_ordinal"] = 0

    with pytest.raises(ShadowInvariantError):
        observation_module._build_shadow_observation(selection=None, **values)


def test_observation_builder_rejects_heuristic_binding_drift(
    trace_event_factory: TraceEventFactory,
) -> None:
    values = _feature_case(trace_event_factory)
    values["heuristic"] = values["heuristic"].model_copy(update={"configuration_digest": "f" * 64})

    with pytest.raises(ShadowInvariantError):
        observation_module._build_shadow_observation(selection=None, **values)


def test_selected_observation_builder_rejects_untrusted_selection() -> None:
    with pytest.raises(ShadowInvariantError):
        observation_module._build_shadow_observation_from_selection(
            selection=object(),  # type: ignore[arg-type]
            report=object(),  # type: ignore[arg-type]
            config=object(),  # type: ignore[arg-type]
            input_kind=ShadowInputKind.START,
            heuristic=object(),  # type: ignore[arg-type]
            source_event_digest="0" * 64,
            redaction_policy_tag=REDACTION_POLICY_TAG,
        )


def test_report_copy_rejects_round_trip_drift(
    trace_event_factory: TraceEventFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _make_case(trace_event_factory).rows[0]
    drifted = row.model_copy(update={"input_ordinal": row.input_ordinal + 1})
    monkeypatch.setattr(
        report_module.ShadowReportRow,
        "model_validate_json",
        classmethod(lambda _cls, _value: drifted),
    )

    with pytest.raises(ValueError, match="defensive validation"):
        report_module._copy_exact_model(report_module.ShadowReportRow, row)


@pytest.mark.parametrize(
    ("scenario", "message"),
    (
        ("missing_reason", "abstention has no exact reason"),
        ("applicable_not_applicable", "applicable detector claims event-not-applicable"),
        ("nonapplicable_detected", "non-applicable detector emitted a signal"),
        ("heuristic", "heuristic does not match row applicability"),
    ),
)
def test_report_aggregate_rechecks_nested_observation_semantics(
    trace_event_factory: TraceEventFactory,
    scenario: str,
    message: str,
) -> None:
    case = _make_case(trace_event_factory)
    observations = list(case.observations)
    if scenario == "missing_reason":
        observation = observations[0]
        evaluation = observation.detector_evaluations[0]
        outcome = evaluation.outcome.model_copy(update={"abstention_reason": None})
        observations[0] = _replace_evaluation_outcome(
            observation,
            evaluation.signal_type,
            outcome,
        )
    elif scenario == "applicable_not_applicable":
        observation = observations[1]
        observations[1] = _replace_evaluation_outcome(
            observation,
            SignalType.REPEATED_ACTION,
            DetectionOutcome.abstained(
                SignalType.REPEATED_ACTION,
                AbstentionReason.EVENT_NOT_APPLICABLE,
                (observation.event_id,),
            ),
        )
    elif scenario == "nonapplicable_detected":
        observation = observations[0]
        signal_type = observation.detector_evaluations[0].signal_type
        observations[0] = _replace_evaluation_outcome(
            observation,
            signal_type,
            DetectionOutcome.detected(signal_type, (observation.event_id,)),
        )
    else:
        observation = observations[0]
        heuristic = observation.heuristic_evaluations[0].model_copy(
            update={"applicable_detector_count": 99}
        )
        observations[0] = observation.model_copy(update={"heuristic_evaluations": (heuristic,)})

    kwargs = _builder_kwargs(case.rows, tuple(observations))
    del kwargs["input_byte_digest"]
    del kwargs["normalized_input_digest"]
    with pytest.raises(ValueError, match=message):
        report_module._derive_aggregates(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("initial_count", "preexisting_indexes", "message"),
    (
        (0, (0,), "absent initial ledger"),
        (1, (), "present initial ledger"),
        (1, (0, 1), "exceed the initial ledger"),
    ),
)
def test_report_aggregate_reconciles_preexisting_rows_with_initial_ledger(
    trace_event_factory: TraceEventFactory,
    initial_count: int,
    preexisting_indexes: tuple[int, ...],
    message: str,
) -> None:
    case = _make_case(trace_event_factory)
    rows = list(case.rows)
    for index in preexisting_indexes:
        rows[index] = rows[index].model_copy(update={"persistence_disposition": "preexisting"})
    kwargs = _builder_kwargs(tuple(rows), case.observations)
    del kwargs["input_byte_digest"]
    del kwargs["normalized_input_digest"]
    kwargs["initial_ledger_entry_count"] = initial_count
    if initial_count:
        kwargs.update(
            initial_ledger_chain_tag=REDACTION_POLICY_TAG,
            initial_ledger_projection_tag=REDACTION_POLICY_TAG,
            initial_ledger_head_tag=REDACTION_POLICY_TAG,
        )

    with pytest.raises(ValueError, match=message):
        report_module._derive_aggregates(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"input_row_count": 7}, "input count equation"),
        (
            {"input_row_count": 7, "unique_input_event_count": 6},
            "persistence count equation",
        ),
        ({"evaluated_unique_event_count": 6}, "evaluation count equation"),
        ({"observation_count": 6}, "observation count equation"),
    ),
)
def test_report_body_rechecks_count_equations_after_aggregate_match(
    trace_event_factory: TraceEventFactory,
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, int],
    message: str,
) -> None:
    report = _make_case(trace_event_factory).report.model_copy(update=updates)
    derived = report_module._DerivedAggregates.model_construct(
        **{
            field_name: getattr(report, field_name)
            for field_name in report_module._DerivedAggregates.model_fields
        }
    )
    monkeypatch.setattr(report_module, "_derive_aggregates", lambda **_kwargs: derived)

    with pytest.raises(ValueError, match=message):
        report.aggregates_match_the_unique_ordered_evidence()


def test_trusted_report_boundary_rejects_unsealed_objects() -> None:
    with pytest.raises(ShadowInvariantError):
        report_module._require_trusted_shadow_run_report(object())


def test_trusted_report_builder_fails_closed_on_constructed_state_drift(
    trace_event_factory: TraceEventFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(trace_event_factory)
    original = report_module._model_state_is_exact

    def reject_constructed_body(model_type: type[Any], value: object) -> bool:
        if model_type in (report_module._ShadowRunReportBody, report_module.ShadowRunReport):
            return False
        return original(model_type, value)

    monkeypatch.setattr(report_module, "_model_state_is_exact", reject_constructed_body)

    with pytest.raises(ShadowInvariantError):
        report_module._build_shadow_run_report_trusted(
            **_builder_kwargs(case.rows, case.observations)  # type: ignore[arg-type]
        )
