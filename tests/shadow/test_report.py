from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError
from tests.shadow.conftest import NOW, OTHER_RUN_ID, RUN_ID, TraceEventFactory

import saliencegate.shadow.report as report_module
from saliencegate.domain import (
    EventPhase,
    EventType,
    PayloadDigest,
    PayloadDigestAlgorithm,
    Signal,
    SignalType,
    TraceEvent,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.security import InstallationKey
from saliencegate.shadow.config import ShadowConfig
from saliencegate.shadow.errors import ShadowInvariantError
from saliencegate.shadow.evaluation import (
    ShadowHeuristicDisposition,
    evaluate_shadow_heuristic,
)
from saliencegate.shadow.inputs import (
    SHADOW_PROJECTION_MATRIX,
    ShadowActionInput,
    ShadowFinishInput,
    ShadowInputKind,
    ShadowStartInput,
    ShadowToolResultInput,
    derive_shadow_source_event_digest,
)
from saliencegate.shadow.observation import (
    ShadowObservation,
    build_shadow_observation,
    derive_shadow_feature_snapshot_digest,
    select_detection_context,
)
from saliencegate.shadow.report import (
    ShadowReportRow,
    ShadowRunReport,
    _report_body_digest,
    build_shadow_run_report,
)
from saliencegate.shadow.session import ShadowSession
from saliencegate.signals import (
    AbstentionReason,
    DetectionOutcome,
    DetectionStatus,
    DetectorEvaluation,
    ExtractionReport,
)

INPUT_BYTE_DIGEST = "1" * 64
NORMALIZED_INPUT_DIGEST = "2" * 64
TASK_SCOPE_DIGEST = "3" * 64
LINEAGE_SCOPE_DIGEST = "4" * 64
CAPTURE_MANIFEST_DIGEST = "5" * 64
REDACTION_POLICY_TAG = PayloadDigest(
    algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
    value="a" * 64,
)


@dataclass(frozen=True)
class ReportCase:
    rows: tuple[ShadowReportRow, ...]
    observations: tuple[ShadowObservation, ...]
    report: ShadowRunReport


def _event(
    trace_event_factory: TraceEventFactory,
    sequence: int,
    kind: ShadowInputKind,
    *,
    run_id: UUID = RUN_ID,
    parent_ids: tuple[UUID, ...] = (),
) -> TraceEvent:
    projection = SHADOW_PROJECTION_MATRIX[kind]
    payloads: dict[ShadowInputKind, dict[str, object]] = {
        ShadowInputKind.START: {
            "shadow_run": {"schema_version": "shadow-run/v1"},
        },
        ShadowInputKind.ACTION: {
            "action": {"schema_version": "1.0", "command": "private-command"},
        },
        ShadowInputKind.TOOL_RESULT: {
            "tool_outcome": {"schema_version": "1.0", "message": "private-tool-text"},
        },
        ShadowInputKind.CONTROLLER_ERROR: {
            "controller_error": {
                "schema_version": "controller_error/v1",
                "error_code": "private-controller-code",
            },
        },
        ShadowInputKind.FINISH: {
            "shadow_run_end": {"schema_version": "shadow-run-end/v1"},
        },
    }
    return trace_event_factory(
        sequence,
        event_type=projection.event_type,
        phase=projection.phase,
        payload=payloads[kind],
        parent_ids=parent_ids,
        trust_label=projection.trust_label,
        run_id=run_id,
    )


def _signal_id(run_id: UUID, outcome: DetectionOutcome, detector_version: str) -> UUID:
    identity = canonical_json(
        {
            "detector_version": detector_version,
            "evidence_event_ids": tuple(str(value) for value in outcome.evidence_event_ids),
            "reason_code": outcome.reason_code.value if outcome.reason_code is not None else None,
            "run_id": str(run_id),
            "signal_type": outcome.signal_type.value,
            "strength": outcome.strength,
        }
    )
    raw = bytearray(
        bytes.fromhex(length_prefixed_sha256(identity, domain="saliencegate:signal:identity:v1"))[
            :16
        ]
    )
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(raw))


def _observation(
    *,
    prefix: tuple[TraceEvent, ...],
    kind: ShadowInputKind,
    cli_input_ordinal: int,
    applicable_outcomes: dict[SignalType, DetectionOutcome] | None = None,
) -> ShadowObservation:
    config = ShadowConfig.reference()
    current = prefix[-1]
    applicable = set(SHADOW_PROJECTION_MATRIX[kind].applicable_detectors)
    selected_outcomes = {} if applicable_outcomes is None else applicable_outcomes
    evaluations: list[DetectorEvaluation] = []
    signals: list[Signal] = []
    for detector in config.detectors:
        signal_type = detector.signal_type
        if signal_type in selected_outcomes:
            outcome = selected_outcomes[signal_type]
        elif signal_type in applicable:
            outcome = DetectionOutcome.no_match(signal_type, (current.event_id,))
        else:
            outcome = DetectionOutcome.abstained(
                signal_type,
                AbstentionReason.EVENT_NOT_APPLICABLE,
                (current.event_id,),
            )
        evaluation = DetectorEvaluation(
            signal_type=signal_type,
            detector_version=detector.detector_version,
            outcome=outcome,
        )
        evaluations.append(evaluation)
        if outcome.status is DetectionStatus.DETECTED:
            assert outcome.strength is not None
            assert outcome.reason_code is not None
            signals.append(
                Signal(
                    signal_id=_signal_id(
                        current.run_id,
                        outcome,
                        detector.detector_version,
                    ),
                    run_id=current.run_id,
                    created_at=current.timestamp,
                    signal_type=signal_type,
                    strength=outcome.strength,
                    evidence_event_ids=outcome.evidence_event_ids,
                    detector_version=detector.detector_version,
                    reason_code=outcome.reason_code,
                )
            )
    extraction = ExtractionReport(
        run_id=current.run_id,
        current_event_id=current.event_id,
        current_event_timestamp=current.timestamp,
        evaluations=tuple(evaluations),
        signals=tuple(signals),
    )
    context = select_detection_context(prefix)
    feature_digest = derive_shadow_feature_snapshot_digest(
        prefix=prefix,
        context=context,
        report=extraction,
        config=config,
    )
    heuristic = evaluate_shadow_heuristic(
        extraction,
        input_kind=kind,
        config=config,
        feature_snapshot_digest=feature_digest,
    )
    return build_shadow_observation(
        prefix=prefix,
        context=context,
        report=extraction,
        config=config,
        input_kind=kind,
        heuristic=heuristic,
        source_event_digest=derive_shadow_source_event_digest(
            current.run_id,
            current.source_event_id,
        ),
        redaction_policy_tag=REDACTION_POLICY_TAG,
        cli_input_ordinal=cli_input_ordinal,
    )


def _row(
    observation: ShadowObservation,
    *,
    input_ordinal: int,
    kind: ShadowInputKind,
    first_occurrence_ordinal: int | None,
    retry_target_ordinal: int | None,
    persistence_disposition: str = "appended",
) -> ShadowReportRow:
    projection = SHADOW_PROJECTION_MATRIX[kind]
    return ShadowReportRow(
        input_ordinal=input_ordinal,
        source_event_digest=observation.source_event_digest,
        first_occurrence_ordinal=first_occurrence_ordinal,
        retry_target_ordinal=retry_target_ordinal,
        event_type=projection.event_type,
        phase=projection.phase,
        input_kind=kind,
        persistence_disposition=persistence_disposition,
        observation_digest=observation.observation_digest,
    )


def _builder_kwargs(
    rows: tuple[ShadowReportRow, ...],
    observations: tuple[ShadowObservation, ...],
    *,
    run_id: UUID = RUN_ID,
    capture_scope: str = "complete_run_declared",
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "initial_ledger_entry_count": 0,
        "initial_ledger_chain_tag": None,
        "initial_ledger_projection_tag": None,
        "initial_ledger_head_tag": None,
        "input_byte_digest": INPUT_BYTE_DIGEST,
        "normalized_input_digest": NORMALIZED_INPUT_DIGEST,
        "redaction_policy_tag": REDACTION_POLICY_TAG,
        "detector_profile_digest": ShadowConfig.reference().detector_profile_digest,
        "capture_scope": capture_scope,
        "task_scope_digest": TASK_SCOPE_DIGEST,
        "lineage_scope_digest": LINEAGE_SCOPE_DIGEST,
        "capture_manifest_digest": CAPTURE_MANIFEST_DIGEST,
        "rows": rows,
        "observations": observations,
    }


def _make_case(
    trace_event_factory: TraceEventFactory,
    *,
    run_id: UUID = RUN_ID,
) -> ReportCase:
    start = _event(trace_event_factory, 1, ShadowInputKind.START, run_id=run_id)
    action = _event(trace_event_factory, 2, ShadowInputKind.ACTION, run_id=run_id)
    tool = _event(
        trace_event_factory,
        3,
        ShadowInputKind.TOOL_RESULT,
        run_id=run_id,
        parent_ids=(action.event_id,),
    )
    error = _event(trace_event_factory, 4, ShadowInputKind.CONTROLLER_ERROR, run_id=run_id)
    finish = _event(trace_event_factory, 5, ShadowInputKind.FINISH, run_id=run_id)
    events = (start, action, tool, error, finish)
    tool_outcomes = {
        SignalType.REPEATED_ACTION: DetectionOutcome.no_match(
            SignalType.REPEATED_ACTION,
            (tool.event_id,),
        ),
        SignalType.REPEATED_FAILURE: DetectionOutcome.detected(
            SignalType.REPEATED_FAILURE,
            (tool.event_id,),
        ),
        SignalType.TEST_FAILURE: DetectionOutcome.abstained(
            SignalType.TEST_FAILURE,
            AbstentionReason.STRUCTURED_EVIDENCE_MISSING,
            (tool.event_id,),
        ),
        SignalType.TOOL_ERROR: DetectionOutcome.detected(
            SignalType.TOOL_ERROR,
            (tool.event_id,),
        ),
    }
    error_outcomes = {
        SignalType.TOOL_ERROR: DetectionOutcome.abstained(
            SignalType.TOOL_ERROR,
            AbstentionReason.STRUCTURED_EVIDENCE_MISSING,
            (error.event_id,),
        ),
    }
    kinds = (
        ShadowInputKind.START,
        ShadowInputKind.ACTION,
        ShadowInputKind.TOOL_RESULT,
        ShadowInputKind.CONTROLLER_ERROR,
        ShadowInputKind.FINISH,
    )
    input_ordinals = (1, 2, 3, 5, 6)
    observations = tuple(
        _observation(
            prefix=events[:index],
            kind=kind,
            cli_input_ordinal=input_ordinal,
            applicable_outcomes=(
                tool_outcomes
                if kind is ShadowInputKind.TOOL_RESULT
                else error_outcomes
                if kind is ShadowInputKind.CONTROLLER_ERROR
                else None
            ),
        )
        for index, (kind, input_ordinal) in enumerate(
            zip(kinds, input_ordinals, strict=True),
            start=1,
        )
    )
    unique_rows = tuple(
        _row(
            observation,
            input_ordinal=input_ordinal,
            kind=kind,
            first_occurrence_ordinal=input_ordinal,
            retry_target_ordinal=None,
        )
        for observation, input_ordinal, kind in zip(
            observations,
            input_ordinals,
            kinds,
            strict=True,
        )
    )
    retry = _row(
        observations[2],
        input_ordinal=4,
        kind=ShadowInputKind.TOOL_RESULT,
        first_occurrence_ordinal=None,
        retry_target_ordinal=3,
        persistence_disposition="preexisting",
    )
    rows = (*unique_rows[:3], retry, *unique_rows[3:])
    report = build_shadow_run_report(  # type: ignore[arg-type]
        **_builder_kwargs(rows, observations, run_id=run_id)
    )
    return ReportCase(rows=rows, observations=observations, report=report)


def _counts_by_pair(values: tuple[tuple[Any, ...], ...]) -> dict[tuple[Any, ...], int]:
    return {tuple(item[:-1]): item[-1] for item in values}


def test_report_is_canonical_payload_free_and_aggregates_only_unique_events(
    trace_event_factory: TraceEventFactory,
) -> None:
    case = _make_case(trace_event_factory)
    report = case.report
    serialized = report.model_dump_json()

    assert report.schema_version == "shadow-run-report/v1"
    assert report.run_id == RUN_ID
    assert report.input_row_count == 6
    assert report.unique_input_event_count == 5
    assert report.retry_row_count == 1
    assert report.appended_event_count == 5
    assert report.preexisting_event_count == 0
    assert report.rejected_row_count == 0
    assert report.evaluated_unique_event_count == 5
    assert report.observation_count == 5
    assert report.first_flagged_event_sequence == 3
    assert report.capture_scope == "complete_run_declared"
    assert report.split_metadata_complete is True
    assert report.rows == case.rows
    assert report.observations == case.observations
    assert report.rows[0] is not case.rows[0]
    assert report.observations[0] is not case.observations[0]
    assert report.redaction_policy_tag is not REDACTION_POLICY_TAG
    assert report.rows[3].retry_target_ordinal == 3
    assert report.rows[3].observation_digest == report.rows[2].observation_digest
    assert report.report_digest == (
        "a807b8b31ff57f5c7c15a9b02239b0007196ec84b2ad29cc95092a8b11b1ae03"
    )
    assert _report_body_digest(report) == report.report_digest

    detector_counts = _counts_by_pair(report.detector_outcome_counts)
    assert detector_counts[(SignalType.REPEATED_ACTION, DetectionStatus.NO_MATCH)] == 2
    assert detector_counts[(SignalType.REPEATED_FAILURE, DetectionStatus.DETECTED)] == 1
    assert detector_counts[(SignalType.TOOL_ERROR, DetectionStatus.DETECTED)] == 1
    assert detector_counts[(SignalType.TEST_FAILURE, DetectionStatus.NO_MATCH)] == 0
    assert sum(detector_counts.values()) == 5 * 4

    abstention_counts = _counts_by_pair(report.abstention_reason_counts)
    assert (
        abstention_counts[(SignalType.TOOL_ERROR, AbstentionReason.STRUCTURED_EVIDENCE_MISSING)]
        == 1
    )
    assert (
        abstention_counts[(SignalType.TEST_FAILURE, AbstentionReason.STRUCTURED_EVIDENCE_MISSING)]
        == 1
    )
    assert abstention_counts[(SignalType.TOOL_ERROR, AbstentionReason.EVENT_NOT_APPLICABLE)] == 3
    disposition_counts = _counts_by_pair(report.heuristic_disposition_counts)
    assert disposition_counts == {
        (ShadowHeuristicDisposition.FLAGGED,): 1,
        (ShadowHeuristicDisposition.INDETERMINATE,): 1,
        (ShadowHeuristicDisposition.NOT_APPLICABLE,): 2,
        (ShadowHeuristicDisposition.NOT_FLAGGED,): 1,
    }
    assert report.applicable_detector_evaluation_count == 4
    assert report.evidence_sufficient_applicable_detector_evaluation_count == 3

    cooccurrences = _counts_by_pair(report.signal_cooccurrence_counts)
    assert cooccurrences[(SignalType.REPEATED_FAILURE, SignalType.TOOL_ERROR)] == 1
    assert sum(cooccurrences.values()) == 1
    event_counts = _counts_by_pair(report.event_type_counts)
    assert event_counts[(EventType.RUN_START,)] == 1
    assert event_counts[(EventType.RUN_END,)] == 1
    assert event_counts[(EventType.TOOL_COMPLETION,)] == 1
    assert sum(event_counts.values()) == report.unique_input_event_count
    phase_counts = _counts_by_pair(report.phase_counts)
    assert phase_counts[(EventPhase.POST_ACTION,)] == 1
    assert sum(phase_counts.values()) == report.unique_input_event_count

    assert report.execution_mode == "shadow"
    assert report.evidence_level == "descriptive_observational"
    assert report.task_outcome_evidence == "none"
    assert report.intervention_outcome_evidence == "none"
    assert report.confirmatory is False
    assert report.calibrated is False
    assert report.calibration_eligible is False
    assert report.decision_authority is False
    assert report.representativeness_supported is False
    assert report.task_efficacy_supported is False
    assert report.counterfactual_effect_supported is False
    assert report.model_calls == 0
    assert report.budget_reservations == 0
    assert report.cycles_created == 0
    assert report.memory_revisions == 0
    assert report.interventions == 0
    assert report.delivery_authorizations == 0
    assert report.deliveries == 0
    assert report.intervention_outcomes == 0

    for forbidden in (
        "private-command",
        "private-tool-text",
        "private-controller-code",
        "shadow-source-1",
        "source_event_id",
        "payload",
        "working_directory",
        "input_path",
        "output_path",
    ):
        assert forbidden not in serialized
    with pytest.raises(ValidationError):
        ShadowRunReport.model_validate(
            {
                **report.model_dump(mode="python"),
                "input_path": "/private/report-input.ndjson",
            }
        )
    assert ShadowRunReport.model_validate_json(serialized) == report
    assert canonical_json(report) == canonical_json(
        build_shadow_run_report(**_builder_kwargs(case.rows, case.observations))  # type: ignore[arg-type]
    )


def test_report_count_equations_and_canonical_complete_zero_cells(
    trace_event_factory: TraceEventFactory,
) -> None:
    report = _make_case(trace_event_factory).report

    assert report.input_row_count == (
        report.unique_input_event_count + report.retry_row_count + report.rejected_row_count
    )
    assert report.unique_input_event_count == (
        report.appended_event_count + report.preexisting_event_count
    )
    assert report.evaluated_unique_event_count == report.unique_input_event_count
    assert report.observation_count == report.unique_input_event_count
    assert len(report.detector_outcome_counts) == 4 * 3
    assert len(report.abstention_reason_counts) == 4 * 8
    assert len(report.heuristic_disposition_counts) == 4
    assert len(report.signal_cooccurrence_counts) == 6
    assert report.detector_outcome_counts == tuple(
        sorted(report.detector_outcome_counts, key=lambda item: (item[0].value, item[1].value))
    )
    assert report.abstention_reason_counts == tuple(
        sorted(report.abstention_reason_counts, key=lambda item: (item[0].value, item[1].value))
    )
    assert report.heuristic_disposition_counts == tuple(
        sorted(report.heuristic_disposition_counts, key=lambda item: item[0].value)
    )


@pytest.mark.asyncio
async def test_report_accepts_real_session_outcomes_outside_the_applicability_mask() -> None:
    async with ShadowSession.in_memory(
        run_id=RUN_ID,
        installation_key=InstallationKey(b"r" * 32),
        capture_scope="complete_run_declared",
    ) as session:
        start = await session._submit(
            ShadowStartInput(source_event_id="start", occurred_at=NOW),
            cli_input_ordinal=1,
        )
        action = await session._submit(
            ShadowActionInput(
                source_event_id="action",
                occurred_at=NOW + timedelta(seconds=1),
                command="private-real-command",
                working_directory="/private/real-directory",
                environment_digest="9" * 64,
            ),
            cli_input_ordinal=2,
        )
        tool = await session._submit(
            ShadowToolResultInput(
                source_event_id="tool",
                occurred_at=NOW + timedelta(seconds=2),
                action=action.ref,
                status="failed",
                exit_status=1,
            ),
            cli_input_ordinal=3,
        )
        finish = await session._submit(
            ShadowFinishInput(
                source_event_id="finish",
                occurred_at=NOW + timedelta(seconds=3),
            ),
            cli_input_ordinal=4,
        )

    observations = tuple(result.observation for result in (start, action, tool, finish))
    kinds = (
        ShadowInputKind.START,
        ShadowInputKind.ACTION,
        ShadowInputKind.TOOL_RESULT,
        ShadowInputKind.FINISH,
    )
    rows = tuple(
        _row(
            observation,
            input_ordinal=ordinal,
            kind=kind,
            first_occurrence_ordinal=ordinal,
            retry_target_ordinal=None,
        )
        for ordinal, (observation, kind) in enumerate(
            zip(observations, kinds, strict=True),
            start=1,
        )
    )
    kwargs = _builder_kwargs(rows, observations)
    kwargs["redaction_policy_tag"] = observations[0].redaction_policy_tag

    report = build_shadow_run_report(**kwargs)  # type: ignore[arg-type]

    tool_outcomes = {
        evaluation.signal_type: evaluation.outcome
        for evaluation in tool.observation.detector_evaluations
    }
    assert tool_outcomes[SignalType.TEST_FAILURE].status is DetectionStatus.ABSTAINED
    assert (
        tool_outcomes[SignalType.TEST_FAILURE].abstention_reason
        is AbstentionReason.STRUCTURED_EVIDENCE_INVALID
    )
    assert report.unique_input_event_count == 4
    assert "private-real-command" not in report.model_dump_json()
    assert "/private/real-directory" not in report.model_dump_json()


def test_private_row_is_strict_frozen_sanitized_and_projection_bound(
    trace_event_factory: TraceEventFactory,
) -> None:
    case = _make_case(trace_event_factory)
    row = case.rows[0]
    fields = row.model_dump(mode="python")

    assert report_module.__all__ == ["ShadowRunReport", "build_shadow_run_report"]
    assert "ShadowReportRow" not in report_module.__all__
    assert set(fields) == {
        "schema_version",
        "input_ordinal",
        "source_event_digest",
        "first_occurrence_ordinal",
        "retry_target_ordinal",
        "event_type",
        "phase",
        "input_kind",
        "persistence_disposition",
        "observation_digest",
    }
    assert not any(
        name in fields for name in ("source_event_id", "payload", "path", "command", "tool_text")
    )
    with pytest.raises(ValidationError):
        row.input_ordinal = 9  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ShadowReportRow.model_validate(
            {
                **row.model_dump(mode="python"),
                "source_event_id": "must-not-enter-a-report",
            }
        )
    with pytest.raises(ValidationError):
        ShadowReportRow(
            **{
                **row.model_dump(mode="python"),
                "event_type": EventType.OBSERVATION,
            }
        )
    with pytest.raises(ValidationError):
        ShadowReportRow.model_validate(
            case.rows[3].model_copy(update={"persistence_disposition": "appended"})
        )


def test_report_builder_rejects_duplicate_mixed_or_unbound_evidence(
    trace_event_factory: TraceEventFactory,
) -> None:
    case = _make_case(trace_event_factory)
    retry = case.rows[3]
    wrong_retry = retry.model_copy(update={"retry_target_ordinal": 99})
    noncanonical_rows = (case.rows[1], case.rows[0], *case.rows[2:])
    mixed = _make_case(trace_event_factory, run_id=OTHER_RUN_ID).observations[-1]
    failures = (
        (case.rows, (case.observations[0], *case.observations)),
        (case.rows, (*case.observations[:-1], mixed)),
        ((*case.rows[:3], wrong_retry, *case.rows[4:]), case.observations),
        (noncanonical_rows, case.observations),
    )

    for rows, observations in failures:
        with pytest.raises(ShadowInvariantError, match=r"^shadow invariant is invalid$") as caught:
            build_shadow_run_report(
                **_builder_kwargs(rows, observations),  # type: ignore[arg-type]
            )
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None


def test_complete_capture_requires_one_final_run_end(
    trace_event_factory: TraceEventFactory,
) -> None:
    case = _make_case(trace_event_factory)
    open_rows = case.rows[:-1]
    open_observations = case.observations[:-1]

    with pytest.raises(ShadowInvariantError):
        build_shadow_run_report(
            **_builder_kwargs(open_rows, open_observations),  # type: ignore[arg-type]
        )

    open_report = build_shadow_run_report(
        **_builder_kwargs(  # type: ignore[arg-type]
            open_rows,
            open_observations,
            capture_scope="bounded_window",
        )
    )
    assert open_report.capture_scope == "bounded_window"
    assert open_report.first_flagged_event_sequence == 3


@pytest.mark.parametrize(
    ("task_digest", "lineage_digest", "manifest_digest", "expected"),
    (
        (TASK_SCOPE_DIGEST, LINEAGE_SCOPE_DIGEST, None, True),
        (None, None, CAPTURE_MANIFEST_DIGEST, True),
        (TASK_SCOPE_DIGEST, None, None, False),
        (None, LINEAGE_SCOPE_DIGEST, None, False),
        (None, None, None, False),
    ),
)
def test_split_metadata_completeness_uses_only_the_declared_capture_provenance(
    trace_event_factory: TraceEventFactory,
    task_digest: str | None,
    lineage_digest: str | None,
    manifest_digest: str | None,
    expected: bool,
) -> None:
    case = _make_case(trace_event_factory)
    kwargs = _builder_kwargs(case.rows, case.observations)
    kwargs.update(
        task_scope_digest=task_digest,
        lineage_scope_digest=lineage_digest,
        capture_manifest_digest=manifest_digest,
    )

    report = build_shadow_run_report(**kwargs)  # type: ignore[arg-type]

    assert report.split_metadata_complete is expected
    assert report.calibration_eligible is False


def test_initial_head_and_append_dispositions_are_cross_checked(
    trace_event_factory: TraceEventFactory,
) -> None:
    case = _make_case(trace_event_factory)
    tag = PayloadDigest(
        algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
        value="c" * 64,
    )
    first_two_preexisting = tuple(
        row.model_copy(update={"persistence_disposition": "preexisting"})
        if row.first_occurrence_ordinal in (1, 2)
        else row
        for row in case.rows
    )
    kwargs = _builder_kwargs(first_two_preexisting, case.observations)
    kwargs.update(
        initial_ledger_entry_count=3,
        initial_ledger_chain_tag=tag,
        initial_ledger_projection_tag=tag,
        initial_ledger_head_tag=tag,
    )

    report = build_shadow_run_report(**kwargs)  # type: ignore[arg-type]
    assert report.preexisting_event_count == 2
    assert report.appended_event_count == 3

    invalid_dispositions = list(first_two_preexisting)
    invalid_dispositions[-1] = invalid_dispositions[-1].model_copy(
        update={"persistence_disposition": "preexisting"}
    )
    with pytest.raises(ShadowInvariantError):
        build_shadow_run_report(
            **{
                **kwargs,
                "rows": tuple(invalid_dispositions),
            }
        )


def test_report_model_rejects_incomplete_counts_and_every_field_mutation(
    trace_event_factory: TraceEventFactory,
) -> None:
    report = _make_case(trace_event_factory).report
    changed_digest = "f" * 64
    tag = PayloadDigest(
        algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
        value="d" * 64,
    )
    mutations: dict[str, object] = {
        "schema_version": "shadow-run-report/v2",
        "run_id": OTHER_RUN_ID,
        "initial_ledger_entry_count": 1,
        "initial_ledger_chain_tag": tag,
        "initial_ledger_projection_tag": tag,
        "initial_ledger_head_tag": tag,
        "input_byte_digest": changed_digest,
        "normalized_input_digest": changed_digest,
        "redaction_policy_tag": tag,
        "detector_profile_digest": changed_digest,
        "capture_scope": "selected_events",
        "task_scope_digest": changed_digest,
        "lineage_scope_digest": changed_digest,
        "capture_manifest_digest": changed_digest,
        "split_metadata_complete": False,
        "input_row_count": report.input_row_count + 1,
        "unique_input_event_count": report.unique_input_event_count + 1,
        "retry_row_count": report.retry_row_count + 1,
        "appended_event_count": report.appended_event_count + 1,
        "preexisting_event_count": report.preexisting_event_count + 1,
        "rejected_row_count": 1,
        "evaluated_unique_event_count": report.evaluated_unique_event_count + 1,
        "observation_count": report.observation_count + 1,
        "rows": tuple(reversed(report.rows)),
        "observations": tuple(reversed(report.observations)),
        "supported_signal_types": tuple(reversed(report.supported_signal_types)),
        "unsupported_signal_types": tuple(reversed(report.unsupported_signal_types)),
        "detector_outcome_counts": tuple(reversed(report.detector_outcome_counts)),
        "abstention_reason_counts": tuple(reversed(report.abstention_reason_counts)),
        "heuristic_disposition_counts": tuple(reversed(report.heuristic_disposition_counts)),
        "applicable_detector_evaluation_count": (report.applicable_detector_evaluation_count + 1),
        "evidence_sufficient_applicable_detector_evaluation_count": (
            report.evidence_sufficient_applicable_detector_evaluation_count + 1
        ),
        "signal_cooccurrence_counts": tuple(reversed(report.signal_cooccurrence_counts)),
        "event_type_counts": tuple(reversed(report.event_type_counts)),
        "phase_counts": tuple(reversed(report.phase_counts)),
        "first_flagged_event_sequence": None,
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
    body_fields = set(ShadowRunReport.model_fields) - {"report_digest"}
    assert set(mutations) == body_fields

    for field_name, replacement in mutations.items():
        forged = report.model_copy(update={field_name: replacement})
        assert _report_body_digest(forged) != report.report_digest
        with pytest.raises(ValidationError):
            ShadowRunReport.model_validate(forged)

    forged_digest = report.model_copy(update={"report_digest": changed_digest})
    with pytest.raises(ValidationError):
        ShadowRunReport.model_validate(forged_digest)
