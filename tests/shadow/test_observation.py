from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

import pytest
from pydantic import ValidationError
from tests.shadow.conftest import NOW, OTHER_RUN_ID, RUN_ID, TraceEventFactory, identifier

from saliencegate.domain import (
    EventPhase,
    EventType,
    PayloadDigest,
    PayloadDigestAlgorithm,
    ReasonCode,
    Signal,
    SignalType,
    TraceEvent,
    TrustLabel,
)
from saliencegate.shadow.config import ShadowConfig
from saliencegate.shadow.errors import ShadowInvariantError
from saliencegate.shadow.evaluation import (
    ShadowHeuristicDisposition,
    ShadowHeuristicEvaluation,
    evaluate_shadow_heuristic,
)
from saliencegate.shadow.inputs import (
    ShadowEventRef,
    ShadowInputKind,
    derive_shadow_event_id,
    derive_shadow_source_event_digest,
)
from saliencegate.shadow.observation import (
    ShadowEventResult,
    ShadowObservation,
    _build_shadow_observation_from_selection,
    _observation_body_digest,
    _select_detection_context,
    build_shadow_observation,
    derive_shadow_detection_context_digest,
    derive_shadow_event_prefix_digest,
    derive_shadow_extraction_report_digest,
    derive_shadow_feature_snapshot_digest,
    derive_shadow_observation_digest,
    derive_shadow_redacted_event_digest,
    select_detection_context,
)
from saliencegate.signals import (
    AbstentionReason,
    DetectionContext,
    DetectionOutcome,
    DetectorEvaluation,
    ExtractionReport,
)

SOURCE_DIGEST = "3da4d80a2c68457803a83924c148a26d9d58f00e8f7d0b0569cd96bf46253a18"
REDACTION_POLICY_TAG = PayloadDigest(
    algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
    value="b" * 64,
)


@dataclass(frozen=True)
class ObservationCase:
    config: ShadowConfig
    prefix: tuple[TraceEvent, ...]
    context: DetectionContext
    report: ExtractionReport
    heuristic: ShadowHeuristicEvaluation
    observation: ShadowObservation


def no_match_report(
    prefix: tuple[TraceEvent, ...],
    config: ShadowConfig,
    *,
    evaluations: tuple[DetectorEvaluation, ...] | None = None,
) -> ExtractionReport:
    current = prefix[-1]
    return ExtractionReport(
        run_id=current.run_id,
        current_event_id=current.event_id,
        current_event_timestamp=current.timestamp,
        evaluations=(
            tuple(
                DetectorEvaluation(
                    signal_type=spec.signal_type,
                    detector_version=spec.detector_version,
                    outcome=DetectionOutcome.no_match(
                        spec.signal_type,
                        (current.event_id,),
                    ),
                )
                for spec in config.detectors
            )
            if evaluations is None
            else evaluations
        ),
        signals=(),
    )


def replace_event(event: TraceEvent, **updates: object) -> TraceEvent:
    fields = event.model_dump(mode="python", warnings=False)
    fields.update(updates)
    return TraceEvent.model_validate(fields)


def make_case(
    trace_event_factory: TraceEventFactory,
    *,
    prefix: tuple[TraceEvent, ...] | None = None,
    input_kind: ShadowInputKind = ShadowInputKind.OBSERVATION,
    cli_input_ordinal: int | None = None,
) -> ObservationCase:
    config = ShadowConfig.reference()
    selected_prefix = (
        (
            trace_event_factory(
                1,
                payload={"observation": {"secret": "caller-secret"}},
            ),
            trace_event_factory(2, payload={"observation": {"bounded": True}}),
        )
        if prefix is None
        else prefix
    )
    context = select_detection_context(selected_prefix)
    report = no_match_report(selected_prefix, config)
    feature_digest = derive_shadow_feature_snapshot_digest(
        prefix=selected_prefix,
        context=context,
        report=report,
        config=config,
    )
    heuristic = evaluate_shadow_heuristic(
        report,
        input_kind=input_kind,
        config=config,
        feature_snapshot_digest=feature_digest,
    )
    current = selected_prefix[-1]
    observation = build_shadow_observation(
        prefix=selected_prefix,
        context=context,
        report=report,
        config=config,
        input_kind=input_kind,
        heuristic=heuristic,
        source_event_digest=derive_shadow_source_event_digest(
            current.run_id,
            current.source_event_id,
        ),
        redaction_policy_tag=REDACTION_POLICY_TAG,
        cli_input_ordinal=cli_input_ordinal,
    )
    return ObservationCase(
        config=config,
        prefix=selected_prefix,
        context=context,
        report=report,
        heuristic=heuristic,
        observation=observation,
    )


def test_observation_is_strict_frozen_payload_free_and_evidence_bounded(
    trace_event_factory: TraceEventFactory,
) -> None:
    case = make_case(trace_event_factory)
    observation = case.observation
    serialized = observation.model_dump_json()

    assert observation.schema_version == "shadow-observation/v1"
    assert observation.run_id == RUN_ID
    assert observation.event_id == case.prefix[-1].event_id
    assert observation.sequence == 2
    assert observation.context_first_sequence == 1
    assert observation.context_last_sequence == 2
    assert observation.context_event_count == 2
    assert observation.context_truncated is False
    assert observation.detector_evaluations == case.report.evaluations
    assert observation.detected_signals == ()
    assert observation.heuristic_evaluations == (case.heuristic,)
    assert observation.supported_signal_types == (
        SignalType.REPEATED_ACTION,
        SignalType.REPEATED_FAILURE,
        SignalType.TEST_FAILURE,
        SignalType.TOOL_ERROR,
    )
    assert observation.unsupported_signal_types == (
        SignalType.CONFLICT,
        SignalType.CONTEXT_SHIFT,
        SignalType.IRREVERSIBLE_ACTION,
        SignalType.STAGNATION,
        SignalType.STALE_CONSTRAINT,
    )
    assert {
        "execution_mode": observation.execution_mode,
        "evidence_level": observation.evidence_level,
        "task_outcome_evidence": observation.task_outcome_evidence,
        "intervention_outcome_evidence": observation.intervention_outcome_evidence,
        "confirmatory": observation.confirmatory,
        "calibrated": observation.calibrated,
        "calibration_eligible": observation.calibration_eligible,
        "decision_authority": observation.decision_authority,
        "representativeness_supported": observation.representativeness_supported,
        "task_efficacy_supported": observation.task_efficacy_supported,
        "counterfactual_effect_supported": observation.counterfactual_effect_supported,
        "model_calls": observation.model_calls,
        "budget_reservations": observation.budget_reservations,
        "cycles_created": observation.cycles_created,
        "memory_revisions": observation.memory_revisions,
        "interventions": observation.interventions,
        "delivery_authorizations": observation.delivery_authorizations,
        "deliveries": observation.deliveries,
        "intervention_outcomes": observation.intervention_outcomes,
    } == {
        "execution_mode": "shadow",
        "evidence_level": "descriptive_observational",
        "task_outcome_evidence": "none",
        "intervention_outcome_evidence": "none",
        "confirmatory": False,
        "calibrated": False,
        "calibration_eligible": False,
        "decision_authority": False,
        "representativeness_supported": False,
        "task_efficacy_supported": False,
        "counterfactual_effect_supported": False,
        "model_calls": 0,
        "budget_reservations": 0,
        "cycles_created": 0,
        "memory_revisions": 0,
        "interventions": 0,
        "delivery_authorizations": 0,
        "deliveries": 0,
        "intervention_outcomes": 0,
    }
    assert "caller-secret" not in repr(observation)
    assert "caller-secret" not in serialized
    assert "shadow-source-1" not in serialized
    assert "shadow-source-2" not in serialized
    assert "payload" not in observation.model_dump()
    assert "source_event_id" not in observation.model_dump()
    assert ShadowObservation.model_validate_json(serialized) == observation
    with pytest.raises(ValidationError):
        observation.model_calls = 1  # type: ignore[misc]


def test_prevalidated_context_path_is_byte_identical_to_the_public_builder(
    trace_event_factory: TraceEventFactory,
) -> None:
    case = make_case(trace_event_factory)
    selection = _select_detection_context(case.prefix)

    optimized = _build_shadow_observation_from_selection(
        selection=selection,
        report=case.report,
        config=case.config,
        input_kind=ShadowInputKind.OBSERVATION,
        heuristic=case.heuristic,
        source_event_digest=derive_shadow_source_event_digest(
            case.prefix[-1].run_id,
            case.prefix[-1].source_event_id,
        ),
        redaction_policy_tag=REDACTION_POLICY_TAG,
    )

    assert optimized.model_dump_json() == case.observation.model_dump_json()


def test_observation_defensively_copies_every_nested_public_model(
    trace_event_factory: TraceEventFactory,
) -> None:
    case = make_case(trace_event_factory)
    observation = case.observation

    assert observation.redaction_policy_tag is not REDACTION_POLICY_TAG
    assert observation.detector_evaluations[0] is not case.report.evaluations[0]
    assert observation.heuristic_evaluations[0] is not case.heuristic


def test_event_result_copies_and_requires_exact_same_event_identity(
    trace_event_factory: TraceEventFactory,
) -> None:
    case = make_case(trace_event_factory)
    current = case.prefix[-1]
    reference = ShadowEventRef(
        run_id=current.run_id,
        event_id=current.event_id,
        sequence=current.sequence,
    )
    result = ShadowEventResult(ref=reference, observation=case.observation)

    assert result.schema_version == "shadow-event-result/v1"
    assert result.ref == reference
    assert result.ref is not reference
    assert result.observation == case.observation
    assert result.observation is not case.observation
    assert ShadowEventResult.model_validate_json(result.model_dump_json()) == result

    mismatches = (
        reference.model_copy(update={"run_id": OTHER_RUN_ID}),
        reference.model_copy(update={"event_id": identifier(0x7777)}),
        reference.model_copy(update={"sequence": 1}),
    )
    for mismatch in mismatches:
        with pytest.raises(ValidationError):
            ShadowEventResult(ref=mismatch, observation=case.observation)


def test_identity_recipes_have_frozen_domains_and_length_prefixed_goldens(
    trace_event_factory: TraceEventFactory,
) -> None:
    event = trace_event_factory(
        1,
        payload={"observation": {"secret": "hidden"}},
    )
    config = ShadowConfig.reference()
    prefix = (event,)
    context = select_detection_context(prefix)
    report = no_match_report(prefix, config)
    feature_digest = derive_shadow_feature_snapshot_digest(
        prefix=prefix,
        context=context,
        report=report,
        config=config,
    )
    heuristic = evaluate_shadow_heuristic(
        report,
        input_kind=ShadowInputKind.OBSERVATION,
        config=config,
        feature_snapshot_digest=feature_digest,
    )
    observation = build_shadow_observation(
        prefix=prefix,
        context=context,
        report=report,
        config=config,
        input_kind=ShadowInputKind.OBSERVATION,
        heuristic=heuristic,
        source_event_digest=derive_shadow_source_event_digest(RUN_ID, event.source_event_id),
        redaction_policy_tag=REDACTION_POLICY_TAG,
    )

    assert derive_shadow_redacted_event_digest(event) == (
        "3d5f4e0f14d5fdff12582b47b726ffd428363c5297e00992ad8eee4e3b2457e7"
    )
    assert derive_shadow_event_prefix_digest(prefix) == (
        "a6c54198973db7a33aa97b371ac2bdcd03656413d6b3a40b8c2b9895e8f15039"
    )
    assert derive_shadow_detection_context_digest(context) == (
        "a7a98ba9b9ecb7f12956f739e76d8b9876c7f016babfb766c20b710a785a1bd4"
    )
    assert derive_shadow_extraction_report_digest(report) == (
        "a033e514aa6a58891cfbf3e8c7c02e7f8db704f08af3a23f4e924754f8736c4a"
    )
    assert feature_digest == "54cf37482f17bd12d76b548c1d6e857792408f0dd1d0421177bb0899d1b21103"
    assert observation.observation_digest == (
        "3ebab584a2c0a08a8c7a4a41e5295fefd6e5d38e344a7d6368039e0cdcc636a9"
    )
    assert derive_shadow_observation_digest(observation) == observation.observation_digest


def test_context_selector_finds_and_verifies_the_longest_suffix_above_ten_megabytes(
    trace_event_factory: TraceEventFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blob = "x" * 950_000
    prefix = tuple(
        trace_event_factory(
            sequence,
            payload={"observation": {"blob": blob}},
        )
        for sequence in range(1, 13)
    )

    assert len(blob.encode()) * len(prefix) > 10 * 1024 * 1024
    context = select_detection_context(prefix)

    assert context.events == prefix[-1:]
    assert DetectionContext(run_id=RUN_ID, events=prefix[-1:]) == context
    with pytest.raises(ValidationError):
        DetectionContext(run_id=RUN_ID, events=prefix[-2:])

    forged = context.model_copy(update={"events": prefix})
    serializer = DetectionContext.__pydantic_serializer__
    serialization_calls = 0

    class CountingSerializer:
        def to_json(self, *args: object, **kwargs: object) -> bytes:
            nonlocal serialization_calls
            serialization_calls += 1
            return serializer.to_json(*args, **kwargs)

    monkeypatch.setattr(
        DetectionContext,
        "__pydantic_serializer__",
        CountingSerializer(),
    )

    with pytest.raises(ShadowInvariantError):
        derive_shadow_detection_context_digest(forged)
    assert serialization_calls == 0


@pytest.mark.parametrize(
    ("event_type", "phase", "payload", "parent_ids", "trust_label"),
    (
        (
            EventType.RUN_START,
            EventPhase.INITIALIZATION,
            {"shadow_run": {"schema_version": "shadow-run/v1"}},
            (),
            TrustLabel.TRUSTED_CONTROLLER,
        ),
        (
            EventType.ACTION_PROPOSAL,
            EventPhase.PRE_ACTION,
            {"action": {"schema_version": "1.0"}},
            (),
            TrustLabel.UNTRUSTED_MODEL_OUTPUT,
        ),
        (
            EventType.ACTION_PROPOSAL,
            EventPhase.PRE_ACTION,
            {"action_identity": {"schema_version": "1.0"}},
            (),
            TrustLabel.UNTRUSTED_MODEL_OUTPUT,
        ),
        (
            EventType.TOOL_COMPLETION,
            EventPhase.POST_ACTION,
            {"tool_outcome": {"schema_version": "1.0"}},
            (identifier(0x7101),),
            TrustLabel.UNTRUSTED_TOOL_OUTPUT,
        ),
        (
            EventType.OBSERVATION,
            EventPhase.POST_ACTION,
            {"test_report": {"schema_version": "1.0"}},
            (identifier(0x7102),),
            TrustLabel.UNTRUSTED_TOOL_OUTPUT,
        ),
        *(
            (
                EventType.OBSERVATION,
                EventPhase.POST_ACTION,
                {"observation": {"bounded": True}},
                (),
                trust_label,
            )
            for trust_label in (
                TrustLabel.UNTRUSTED_TASK_INPUT,
                TrustLabel.UNTRUSTED_TOOL_OUTPUT,
                TrustLabel.UNTRUSTED_MODEL_OUTPUT,
                TrustLabel.UNTRUSTED_EXTERNAL_MEMORY,
            )
        ),
        (
            EventType.CONTROLLER_ERROR,
            EventPhase.INTERNAL,
            {"controller_error": {"schema_version": "controller_error/v1"}},
            (),
            TrustLabel.TRUSTED_CONTROLLER,
        ),
        (
            EventType.RUN_END,
            EventPhase.TERMINAL,
            {"shadow_run_end": {"schema_version": "shadow-run-end/v1"}},
            (),
            TrustLabel.TRUSTED_CONTROLLER,
        ),
    ),
)
def test_every_accepted_shadow_projection_is_singleton_context_valid(
    trace_event_factory: TraceEventFactory,
    event_type: EventType,
    phase: EventPhase,
    payload: dict[str, object],
    parent_ids: tuple[UUID, ...],
    trust_label: TrustLabel,
) -> None:
    event = trace_event_factory(
        1,
        event_type=event_type,
        phase=phase,
        payload=payload,
        parent_ids=parent_ids,
        trust_label=trust_label,
    )

    selected = select_detection_context((event,))

    assert selected == DetectionContext(run_id=RUN_ID, events=(event,))


def test_context_selector_rejects_non_prefix_state_without_leaking_values(
    trace_event_factory: TraceEventFactory,
) -> None:
    secret_event = trace_event_factory(
        1,
        payload={"observation": {"secret": "context-secret"}},
    )
    invalid_prefixes = (
        (trace_event_factory(2),),
        (trace_event_factory(1), trace_event_factory(3)),
        (trace_event_factory(2), trace_event_factory(1)),
        (trace_event_factory(1), trace_event_factory(2, run_id=OTHER_RUN_ID)),
        (replace_event(trace_event_factory(1), event_id=identifier(0x7FFE)),),
        (secret_event, secret_event),
    )

    for prefix in invalid_prefixes:
        with pytest.raises(ShadowInvariantError) as caught:
            select_detection_context(prefix)
        assert str(caught.value) == "shadow invariant is invalid"
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
        assert "context-secret" not in str(caught.value)


def test_feature_identity_binds_every_complete_historical_event_field(
    trace_event_factory: TraceEventFactory,
) -> None:
    case = make_case(trace_event_factory)
    first, current = case.prefix
    baseline = case.observation.feature_snapshot_digest
    mutations: tuple[dict[str, object], ...] = (
        {
            "event_id": derive_shadow_event_id(RUN_ID, "changed-source-a"),
            "source_event_id": "changed-source-a",
        },
        {
            "event_id": derive_shadow_event_id(RUN_ID, "changed-source-b"),
            "source_event_id": "changed-source-b",
        },
        {"timestamp": NOW},
        {"event_type": EventType.RUN_START},
        {"phase": EventPhase.INITIALIZATION},
        {"payload": {"observation": {"changed": True}}},
        {
            "payload_digest": PayloadDigest(
                algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
                value="d" * 64,
            )
        },
        {"parent_ids": (identifier(0x7002),)},
        {"source_adapter": "changed-adapter/v1"},
        {"trust_label": TrustLabel.UNTRUSTED_EXTERNAL_MEMORY},
    )

    for update in mutations:
        changed_prefix = (replace_event(first, **update), current)
        changed_context = select_detection_context(changed_prefix)
        changed_report = no_match_report(changed_prefix, case.config)
        assert (
            derive_shadow_feature_snapshot_digest(
                prefix=changed_prefix,
                context=changed_context,
                report=changed_report,
                config=case.config,
            )
            != baseline
        )

    changed_run_prefix = tuple(
        replace_event(
            event,
            run_id=OTHER_RUN_ID,
            event_id=derive_shadow_event_id(OTHER_RUN_ID, event.source_event_id),
        )
        for event in case.prefix
    )
    changed_run_context = select_detection_context(changed_run_prefix)
    changed_run_report = no_match_report(changed_run_prefix, case.config)
    assert (
        derive_shadow_feature_snapshot_digest(
            prefix=changed_run_prefix,
            context=changed_run_context,
            report=changed_run_report,
            config=case.config,
        )
        != baseline
    )


def test_feature_identity_binds_every_detector_outcome_and_excludes_future_events(
    trace_event_factory: TraceEventFactory,
) -> None:
    case = make_case(trace_event_factory)
    baseline = case.observation.feature_snapshot_digest

    for index, evaluation in enumerate(case.report.evaluations):
        changed_evaluations = list(case.report.evaluations)
        changed_evaluations[index] = DetectorEvaluation(
            signal_type=evaluation.signal_type,
            detector_version=evaluation.detector_version,
            outcome=DetectionOutcome.abstained(
                evaluation.signal_type,
                AbstentionReason.INSUFFICIENT_HISTORY,
                (case.prefix[-1].event_id,),
            ),
        )
        changed_report = no_match_report(
            case.prefix,
            case.config,
            evaluations=tuple(changed_evaluations),
        )
        assert (
            derive_shadow_feature_snapshot_digest(
                prefix=case.prefix,
                context=case.context,
                report=changed_report,
                config=case.config,
            )
            != baseline
        )

    complete_run = (*case.prefix, trace_event_factory(3, payload={"future": "ignored"}))
    observed_prefix = tuple(event for event in complete_run if event.sequence <= 2)
    assert observed_prefix == case.prefix
    assert (
        derive_shadow_feature_snapshot_digest(
            prefix=observed_prefix,
            context=case.context,
            report=case.report,
            config=case.config,
        )
        == baseline
    )


def test_cli_ordinal_changes_only_the_observation_self_identity(
    trace_event_factory: TraceEventFactory,
) -> None:
    sdk = make_case(trace_event_factory)
    cli = make_case(trace_event_factory, cli_input_ordinal=7)
    sdk_fields = sdk.observation.model_dump(
        mode="json",
        exclude={"cli_input_ordinal", "observation_digest"},
    )
    cli_fields = cli.observation.model_dump(
        mode="json",
        exclude={"cli_input_ordinal", "observation_digest"},
    )

    assert sdk_fields == cli_fields
    assert sdk.observation.cli_input_ordinal is None
    assert cli.observation.cli_input_ordinal == 7
    assert sdk.observation.feature_snapshot_digest == cli.observation.feature_snapshot_digest
    assert sdk.observation.observation_digest != cli.observation.observation_digest


def test_observation_self_digest_binds_every_body_field_and_rejects_mutation(
    trace_event_factory: TraceEventFactory,
) -> None:
    case = make_case(trace_event_factory)
    observation = case.observation
    changed_digest = "e" * 64
    first_evaluation = observation.detector_evaluations[0]
    changed_evaluation = DetectorEvaluation(
        signal_type=first_evaluation.signal_type,
        detector_version=first_evaluation.detector_version,
        outcome=DetectionOutcome.abstained(
            first_evaluation.signal_type,
            AbstentionReason.INSUFFICIENT_HISTORY,
            (observation.event_id,),
        ),
    )
    tool_evaluation = observation.detector_evaluations[-1]
    changed_signal = Signal(
        signal_id=identifier(0x7A01),
        run_id=observation.run_id,
        created_at=case.prefix[-1].timestamp,
        signal_type=tool_evaluation.signal_type,
        strength=1.0,
        evidence_event_ids=(observation.event_id,),
        detector_version=tool_evaluation.detector_version,
        reason_code=ReasonCode.TOOL_ERROR,
    )
    heuristic_fields = case.heuristic.model_dump(mode="python", warnings=False)
    heuristic_fields["feature_snapshot_digest"] = changed_digest
    changed_heuristic = ShadowHeuristicEvaluation.model_validate(heuristic_fields)
    mutations: dict[str, object] = {
        "schema_version": "shadow-observation/v2",
        "run_id": OTHER_RUN_ID,
        "event_id": identifier(0x7A02),
        "source_event_digest": changed_digest,
        "sequence": observation.sequence + 1,
        "event_prefix_digest": changed_digest,
        "context_first_sequence": observation.context_first_sequence + 1,
        "context_last_sequence": observation.context_last_sequence + 1,
        "context_event_count": observation.context_event_count + 1,
        "context_truncated": not observation.context_truncated,
        "detection_context_digest": changed_digest,
        "redacted_event_digest": changed_digest,
        "redaction_policy_tag": PayloadDigest(
            algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
            value="c" * 64,
        ),
        "detector_profile_digest": changed_digest,
        "evaluator_configuration_digest": changed_digest,
        "extraction_report_digest": changed_digest,
        "feature_snapshot_digest": changed_digest,
        "supported_signal_types": tuple(reversed(observation.supported_signal_types)),
        "unsupported_signal_types": tuple(reversed(observation.unsupported_signal_types)),
        "detector_evaluations": (changed_evaluation, *observation.detector_evaluations[1:]),
        "detected_signals": (changed_signal,),
        "heuristic_evaluations": (changed_heuristic,),
        "cli_input_ordinal": 7,
        "execution_mode": "active",
        "evidence_level": "confirmatory",
        "task_outcome_evidence": "present",
        "intervention_outcome_evidence": "present",
        "confirmatory": True,
        "calibrated": True,
        "calibration_eligible": True,
        "decision_authority": True,
        "representativeness_supported": True,
        "task_efficacy_supported": True,
        "counterfactual_effect_supported": True,
        "model_calls": 1,
        "budget_reservations": 1,
        "cycles_created": 1,
        "memory_revisions": 1,
        "interventions": 1,
        "delivery_authorizations": 1,
        "deliveries": 1,
        "intervention_outcomes": 1,
    }
    body_fields = set(ShadowObservation.model_fields) - {"observation_digest"}
    assert set(mutations) == body_fields

    for field_name, replacement in mutations.items():
        forged = observation.model_copy(update={field_name: replacement})
        assert _observation_body_digest(forged) != observation.observation_digest
        with pytest.raises(ValidationError):
            ShadowObservation.model_validate(forged)

    forged_digest = observation.model_copy(update={"observation_digest": changed_digest})
    with pytest.raises(ValidationError):
        ShadowObservation.model_validate(forged_digest)


def test_builder_binds_source_identity_and_recomputes_the_exact_heuristic(
    trace_event_factory: TraceEventFactory,
) -> None:
    action = trace_event_factory(
        1,
        event_type=EventType.ACTION_PROPOSAL,
        phase=EventPhase.PRE_ACTION,
        payload={"action": {"schema_version": "1.0"}},
        trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
    )
    config = ShadowConfig.reference()
    prefix = (action,)
    context = select_detection_context(prefix)
    report = no_match_report(prefix, config)
    feature_digest = derive_shadow_feature_snapshot_digest(
        prefix=prefix,
        context=context,
        report=report,
        config=config,
    )
    wrong_heuristic = evaluate_shadow_heuristic(
        report,
        input_kind=ShadowInputKind.OBSERVATION,
        config=config,
        feature_snapshot_digest=feature_digest,
    )
    correct_heuristic = evaluate_shadow_heuristic(
        report,
        input_kind=ShadowInputKind.ACTION,
        config=config,
        feature_snapshot_digest=feature_digest,
    )

    assert wrong_heuristic.disposition is ShadowHeuristicDisposition.NOT_APPLICABLE
    assert correct_heuristic.disposition is ShadowHeuristicDisposition.NOT_FLAGGED
    for source_digest, heuristic in (
        ("f" * 64, correct_heuristic),
        (
            derive_shadow_source_event_digest(RUN_ID, action.source_event_id),
            wrong_heuristic,
        ),
    ):
        with pytest.raises(ShadowInvariantError) as caught:
            build_shadow_observation(
                prefix=prefix,
                context=context,
                report=report,
                config=config,
                input_kind=ShadowInputKind.ACTION,
                heuristic=heuristic,
                source_event_digest=source_digest,
                redaction_policy_tag=REDACTION_POLICY_TAG,
            )
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None


def test_action_and_action_identity_cannot_be_cross_labeled(
    trace_event_factory: TraceEventFactory,
) -> None:
    config = ShadowConfig.reference()
    cases = (
        (
            trace_event_factory(
                1,
                event_type=EventType.ACTION_PROPOSAL,
                phase=EventPhase.PRE_ACTION,
                payload={"action": {"schema_version": "1.0"}},
                trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
            ),
            ShadowInputKind.ACTION_IDENTITY,
        ),
        (
            trace_event_factory(
                1,
                event_type=EventType.ACTION_PROPOSAL,
                phase=EventPhase.PRE_ACTION,
                payload={"action_identity": {"schema_version": "1.0"}},
                trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
            ),
            ShadowInputKind.ACTION,
        ),
    )

    for event, wrong_kind in cases:
        prefix = (event,)
        context = select_detection_context(prefix)
        report = no_match_report(prefix, config)
        feature_digest = derive_shadow_feature_snapshot_digest(
            prefix=prefix,
            context=context,
            report=report,
            config=config,
        )
        heuristic = evaluate_shadow_heuristic(
            report,
            input_kind=wrong_kind,
            config=config,
            feature_snapshot_digest=feature_digest,
        )

        with pytest.raises(ShadowInvariantError):
            build_shadow_observation(
                prefix=prefix,
                context=context,
                report=report,
                config=config,
                input_kind=wrong_kind,
                heuristic=heuristic,
                source_event_digest=derive_shadow_source_event_digest(
                    RUN_ID,
                    event.source_event_id,
                ),
                redaction_policy_tag=REDACTION_POLICY_TAG,
            )


def test_report_references_must_be_ordered_members_of_the_selected_context(
    trace_event_factory: TraceEventFactory,
) -> None:
    case = make_case(trace_event_factory)
    evaluations = list(case.report.evaluations)
    first = evaluations[0]
    evaluations[0] = DetectorEvaluation(
        signal_type=first.signal_type,
        detector_version=first.detector_version,
        outcome=DetectionOutcome.no_match(
            first.signal_type,
            (identifier(0x7FFF), case.prefix[-1].event_id),
        ),
    )
    alien_report = no_match_report(
        case.prefix,
        case.config,
        evaluations=tuple(evaluations),
    )

    with pytest.raises(ShadowInvariantError):
        derive_shadow_feature_snapshot_digest(
            prefix=case.prefix,
            context=case.context,
            report=alien_report,
            config=case.config,
        )
    with pytest.raises(ShadowInvariantError):
        build_shadow_observation(
            prefix=case.prefix,
            context=case.context,
            report=alien_report,
            config=case.config,
            input_kind=ShadowInputKind.OBSERVATION,
            heuristic=case.heuristic,
            source_event_digest=derive_shadow_source_event_digest(
                RUN_ID,
                case.prefix[-1].source_event_id,
            ),
            redaction_policy_tag=REDACTION_POLICY_TAG,
        )


def test_public_boundaries_reject_poisoned_instances_without_calling_instance_methods(
    trace_event_factory: TraceEventFactory,
) -> None:
    event = trace_event_factory(1, payload={"observation": {"secret": "poison-secret"}})
    called = False

    def poisoned_serializer(*args: object, **kwargs: object) -> str:
        nonlocal called
        called = True
        return "poison-secret"

    object.__setattr__(event, "model_dump_json", poisoned_serializer)

    with pytest.raises(ShadowInvariantError) as caught:
        derive_shadow_redacted_event_digest(event)
    assert called is False
    assert "poison-secret" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_all_public_builder_failures_are_value_free_and_have_no_exception_chain(
    trace_event_factory: TraceEventFactory,
) -> None:
    case = make_case(trace_event_factory)
    forged_observation = case.observation.model_copy(update={"observation_digest": "f" * 64})
    calls: tuple[Callable[[], object], ...] = (
        lambda: derive_shadow_event_prefix_digest((trace_event_factory(2),)),
        lambda: derive_shadow_observation_digest(forged_observation),
        lambda: build_shadow_observation(
            prefix=case.prefix,
            context=case.context,
            report=case.report,
            config=case.config,
            input_kind=ShadowInputKind.OBSERVATION,
            heuristic=case.heuristic,
            source_event_digest="caller-secret",
            redaction_policy_tag=REDACTION_POLICY_TAG,
        ),
        lambda: build_shadow_observation(
            prefix=case.prefix,
            context=case.context,
            report=case.report,
            config=case.config,
            input_kind=ShadowInputKind.OBSERVATION,
            heuristic=case.heuristic,
            source_event_digest=derive_shadow_source_event_digest(
                RUN_ID,
                case.prefix[-1].source_event_id,
            ),
            redaction_policy_tag=PayloadDigest(
                algorithm=PayloadDigestAlgorithm.SYNTHETIC_SHA256,
                value="f" * 64,
            ),
        ),
    )

    for call in calls:
        with pytest.raises(ShadowInvariantError) as caught:
            call()
        assert str(caught.value) == "shadow invariant is invalid"
        assert "caller-secret" not in str(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None


def test_observation_module_exports_only_the_deliberate_contract() -> None:
    from saliencegate.shadow import observation

    assert observation.__all__ == [
        "ShadowEventResult",
        "ShadowObservation",
        "build_shadow_observation",
        "derive_shadow_detection_context_digest",
        "derive_shadow_event_prefix_digest",
        "derive_shadow_extraction_report_digest",
        "derive_shadow_feature_snapshot_digest",
        "derive_shadow_observation_digest",
        "derive_shadow_redacted_event_digest",
        "select_detection_context",
    ]
