from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from saliencegate.artifacts import algorithm_manifest as _algorithm
from saliencegate.artifacts.manifest import (
    ArtifactClassification,
    DeliveryAttestation,
    delivery_binding_digest,
)
from saliencegate.artifacts.tree import ArtifactExportError
from saliencegate.domain import InterventionOutcome, canonical_digest, canonical_json
from saliencegate.experiments.evidence import Stage2BoundaryEvidence
from saliencegate.experiments.runner import Stage2ExperimentRunResult
from saliencegate.intervention import GroundingReceipt
from saliencegate.models.replay_two_phase import (
    two_phase_replay_fixture_digest_from_receipts,
)
from saliencegate.ports.two_phase import CallReceipt


def _validate_source_execution_binding(
    result: Stage2ExperimentRunResult,
    execution: _algorithm.AlgorithmExecutionAttestation,
) -> None:
    calls = result.call_receipts
    fixture = execution.response_fixture
    source_fixture = result.response_fixture
    if execution.checkpoint.model_id != result.model_profile.model_id:
        raise ArtifactExportError("algorithm execution model binding failed validation")
    if not _algorithm.algorithm_call_evidence_matches_execution_mode(
        execution.execution_mode,
        calls,
    ):
        raise ArtifactExportError("algorithm execution call evidence failed validation")
    if execution.execution_mode is _algorithm.AlgorithmExecutionMode.FROZEN_REPLAY:
        if calls:
            try:
                digest = (
                    None
                    if fixture is None
                    else two_phase_replay_fixture_digest_from_receipts(
                        calls,
                        replay_id=fixture.replay_id,
                    )
                )
            except Exception:
                raise ArtifactExportError(
                    "algorithm execution fixture binding failed validation"
                ) from None
            if (
                fixture is None
                or source_fixture is None
                or fixture.replay_id != source_fixture.replay_id
                or fixture.fixture_digest != source_fixture.fixture_digest
                or fixture.fixture_digest != digest
                or fixture.response_count != len(calls)
                or fixture.consumed_count != len(calls)
                or source_fixture.response_count != len(calls)
            ):
                raise ArtifactExportError("algorithm execution fixture binding failed validation")
        elif fixture is not None or source_fixture is not None:
            raise ArtifactExportError("algorithm execution fixture binding failed validation")
    elif source_fixture is not None:
        raise ArtifactExportError("algorithm execution provenance failed validation")
    if not _algorithm.algorithm_call_evidence_matches_tokenizer(execution.tokenizer, calls):
        raise ArtifactExportError("algorithm execution tokenizer binding failed validation")


def _run_component(
    result: Stage2ExperimentRunResult,
    execution: _algorithm.AlgorithmExecutionAttestation,
) -> _algorithm.AlgorithmRunComponent:
    return _algorithm._seal_algorithm_model(
        _algorithm.AlgorithmRunComponent,
        {
            "run_id": result.run_id,
            "condition": result.condition,
            "policy_version": result.policy_version,
            "cycle_mode": _algorithm.algorithm_cycle_mode_for_condition(
                result.condition.condition_id
            ),
            "prompt_bundle": result.prompt_bundle,
            "model_profile": result.model_profile,
            "call_policy": result.call_policy,
            "cycle_reservation": result.cycle_reservation,
            "budget_limits": result.budget_limits,
            "execution": execution,
            "source_result_digest": result.result_digest,
        },
    )


def _trajectory_component(
    result: Stage2ExperimentRunResult,
) -> _algorithm.AlgorithmTrajectoryComponent:
    records = tuple(
        _algorithm.AlgorithmTrajectoryRecordAttestation(
            ordinal=fixture.ordinal,
            fixture_record_digest=fixture.record_digest,
            input_digest=fixture.input_digest,
            event_id=item.event.event_id,
            event_sequence=item.event.sequence,
            event_timestamp=item.event.timestamp,
            event_type=item.event.event_type,
            event_phase=item.event.phase,
            parent_ids=item.event.parent_ids,
            trust_label=item.event.trust_label,
            payload_digest=item.event.payload_digest,
            normalized_draft_digest=normalized,
            persisted_event_draft_digest=persisted,
            event_digest=canonical_digest(item.event),
            binding=item.binding,
            action_step_ordinal=scheduled.action_step_ordinal,
        )
        for fixture, item, scheduled, normalized, persisted in zip(
            result.trajectory.records,
            result.trajectory_prefix.items,
            result.schedule.decisions,
            result.normalized_draft_digests,
            result.persisted_event_draft_digests,
            strict=True,
        )
    )
    windows = tuple(
        _algorithm.AlgorithmWindowAttestation(
            invocation_ordinal=ordinal,
            version=window.version,
            boundary_event_id=window.boundary_event_id,
            boundary_event_sequence=window.boundary_event_sequence,
            boundary_ledger_position=window.boundary_ledger_position,
            boundary_chain_tag=window.boundary_chain_tag,
            trajectory_prefix_digest=window.trajectory_prefix_digest,
            task_digest=window.task_description.task_digest,
            message_count=len(window.payload.messages),
            payload_canonical_utf8_bytes=window.payload_canonical_utf8_bytes,
            source_attestation_digests=tuple(
                canonical_digest(source) for source in window.source_attestations
            ),
            window_digest=window.window_digest,
        )
        for ordinal, window in enumerate(result.windows, start=1)
    )
    return _algorithm._seal_algorithm_model(
        _algorithm.AlgorithmTrajectoryComponent,
        {
            "run_id": result.run_id,
            "fixture_id": result.trajectory.fixture_id,
            "fixture_digest": result.trajectory.fixture_digest,
            "trace_digest": result.trace_digest,
            "trajectory_prefix_request_digest": result.trajectory_prefix.request_digest,
            "trajectory_prefix_digest": result.trajectory_prefix.prefix_digest,
            "records": records,
            "schedule": result.schedule,
            "windows": windows,
            "window_set_digest": _algorithm.algorithm_window_set_digest(
                tuple(window.window_digest for window in windows)
            ),
        },
    )


def _active_boundaries(
    result: Stage2ExperimentRunResult,
) -> tuple[Stage2BoundaryEvidence, ...]:
    return tuple(boundary for boundary in result.boundaries if boundary.cycle is not None)


def _calls_component(
    result: Stage2ExperimentRunResult,
) -> _algorithm.AlgorithmCallsComponent:
    groups: list[_algorithm.AlgorithmCallGroup] = []
    flattened: list[CallReceipt] = []
    for boundary in _active_boundaries(result):
        if boundary.cycle is None or boundary.request is None or not boundary.call_receipts:
            raise ArtifactExportError("algorithm call group failed artifact-boundary validation")
        grounded = tuple(
            call for call in boundary.call_receipts if call.grounding_state_digest is not None
        )
        if len(grounded) > 1:
            raise ArtifactExportError(
                "algorithm call grounding failed artifact-boundary validation"
            )
        groups.append(
            _algorithm.AlgorithmCallGroup(
                invocation_ordinal=boundary.observation.observed.invocation_ordinal,
                boundary_event_sequence=boundary.boundary_event.sequence,
                cycle_id=boundary.cycle.cycle_id,
                cycle_request_digest=boundary.request.request_digest,
                call_receipt_digests=tuple(call.receipt_digest for call in boundary.call_receipts),
                grounding_call_index=(None if not grounded else grounded[0].model_call_index),
                grounding_state_digest=(
                    None if not grounded else grounded[0].grounding_state_digest
                ),
            )
        )
        flattened.extend(boundary.call_receipts)
    calls = tuple(flattened)
    if calls != result.call_receipts:
        raise ArtifactExportError("algorithm call ordering failed artifact-boundary validation")
    return _algorithm._seal_algorithm_model(
        _algorithm.AlgorithmCallsComponent,
        {
            "run_id": result.run_id,
            "ordered_request_digests": tuple(call.request_digest for call in calls),
            "groups": tuple(groups),
            "calls": calls,
        },
    )


def _decisions_component(
    result: Stage2ExperimentRunResult,
) -> _algorithm.AlgorithmDecisionsComponent:
    return _algorithm._seal_algorithm_model(
        _algorithm.AlgorithmDecisionsComponent,
        {"run_id": result.run_id, "decisions": result.decisions},
    )


def _cycle_attestation(
    boundary: Stage2BoundaryEvidence,
) -> _algorithm.AlgorithmCycleAttestation:
    cycle = boundary.cycle
    request = boundary.request
    execution_result = boundary.two_phase_result or boundary.phase_one_result
    if (
        cycle is None
        or request is None
        or execution_result is None
        or (boundary.two_phase_result is None) == (boundary.phase_one_result is None)
        or cycle.validated_delta is None
        or cycle.intervention is None
        or cycle.budget_reservation is None
        or cycle.budget_settlement is None
        or cycle.batch_digest is None
    ):
        raise ArtifactExportError("algorithm cycle failed artifact-boundary validation")
    intervention = cycle.intervention
    try:
        grounding = GroundingReceipt.model_validate_json(
            canonical_json(intervention.grounding_receipt)
        )
    except Exception:
        raise ArtifactExportError(
            "algorithm grounding failed artifact-boundary validation"
        ) from None
    rendered_text_digest = (
        None if intervention.rendered_text is None else canonical_digest(intervention.rendered_text)
    )
    delivery_source_digest = (
        None if boundary.delivery_record is None else canonical_digest(boundary.delivery_record)
    )
    delta = cycle.validated_delta
    return _algorithm._seal_algorithm_model(
        _algorithm.AlgorithmCycleAttestation,
        {
            "invocation_ordinal": boundary.observation.observed.invocation_ordinal,
            "cycle_id": cycle.cycle_id,
            "run_id": cycle.run_id,
            "revision": cycle.revision,
            "invocation_decision_id": cycle.invocation_decision_id,
            "boundary_event_sequence": boundary.boundary_event.sequence,
            "window_digest": boundary.window.window_digest,
            "policy_version": cycle.policy_version,
            "configuration_digest": cycle.configuration_digest,
            "grounding_version": cycle.grounding_version,
            "grounding_configuration_digest": cycle.grounding_configuration_digest,
            "state": cycle.state,
            "source_cycle_digest": canonical_digest(cycle),
            "cycle_request_digest": request.request_digest,
            "execution_result_digest": execution_result.result_digest,
            "observation_digest": boundary.observation.observation_digest,
            "budget_reservation": cycle.budget_reservation,
            "budget_settlement": cycle.budget_settlement,
            "batch_digest": cycle.batch_digest,
            "model_call_digests": cycle.model_call_digests,
            "model_call_latencies_us": cycle.model_call_latencies_us,
            "call_receipt_digests": tuple(call.receipt_digest for call in boundary.call_receipts),
            "memory_create_count": len(delta.creates),
            "memory_update_count": len(delta.updates),
            "memory_invalidation_count": len(delta.invalidations),
            "private_status_replaced": delta.private_status_replacement is not None,
            "intervention_id": intervention.intervention_id,
            "intervention_action": intervention.action,
            "intervention_digest": canonical_digest(intervention),
            "grounding_receipt_digest": canonical_digest(intervention.grounding_receipt),
            "grounding_model_call_index": grounding.model_call_index,
            "grounding_model_call_digest": grounding.model_call_digest,
            "selector_provenance": grounding.selector_provenance,
            "rendered_text_digest": rendered_text_digest,
            "reason_code": intervention.reason_code,
            "delivery_source_digest": delivery_source_digest,
            "boundary_evidence_digest": boundary.evidence_digest,
        },
    )


def _cycles_component(
    result: Stage2ExperimentRunResult,
) -> _algorithm.AlgorithmCyclesComponent:
    cycles = tuple(_cycle_attestation(boundary) for boundary in _active_boundaries(result))
    return _algorithm._seal_algorithm_model(
        _algorithm.AlgorithmCyclesComponent,
        {"run_id": result.run_id, "cycles": cycles},
    )


def _delivery_attestation(boundary: Stage2BoundaryEvidence) -> DeliveryAttestation:
    delivery = boundary.delivery_record
    if delivery is None:
        raise ArtifactExportError("algorithm delivery failed artifact-boundary validation")
    values: dict[str, object] = {
        "delivery_id": delivery.delivery_id,
        "event_sequence": boundary.boundary_event.sequence,
        "cycle_id": delivery.cycle_id,
        "intervention_id": delivery.intervention_id,
        "rendered_text_digest": delivery.rendered_text_digest,
        "target_request_id_digest": canonical_digest(delivery.target_request_id),
        "target": delivery.target,
        "state": delivery.state,
        "attempt_count": delivery.attempt_count,
        "adapter_id_digest": canonical_digest(delivery.adapter_id),
        "adapter_deduplicates": delivery.adapter_deduplicates,
        "adapter_deduplication_guarantee": delivery.adapter_deduplication_guarantee,
        "adapter_supports_pre_action": delivery.adapter_supports_pre_action,
        "adapter_contract_version": delivery.adapter_contract_version,
        "adapter_capabilities_digest": delivery.adapter_capabilities_digest,
        "claim_id": delivery.claim_id,
        "attempt_id": delivery.attempt_id,
        "receipt_digest": (
            None if delivery.receipt is None else canonical_digest(delivery.receipt)
        ),
        "outcome": delivery.outcome,
        "reason_code": delivery.reason_code,
        "created_at": delivery.created_at,
        "updated_at": delivery.updated_at,
        "source_delivery_digest": canonical_digest(delivery),
    }
    values["binding_digest"] = delivery_binding_digest(values)
    try:
        return DeliveryAttestation.model_validate(values)
    except Exception:
        raise ArtifactExportError(
            "algorithm delivery failed artifact-boundary validation"
        ) from None


def _deliveries_component(
    result: Stage2ExperimentRunResult,
) -> _algorithm.AlgorithmDeliveriesComponent:
    deliveries = tuple(
        _delivery_attestation(boundary)
        for boundary in result.boundaries
        if boundary.delivery_record is not None
    )
    return _algorithm._seal_algorithm_model(
        _algorithm.AlgorithmDeliveriesComponent,
        {"run_id": result.run_id, "deliveries": deliveries},
    )


def _outcomes(result: Stage2ExperimentRunResult) -> tuple[InterventionOutcome, ...]:
    return tuple(
        entry.record for entry in result.ledger if type(entry.record) is InterventionOutcome
    )


def _outcomes_component(
    result: Stage2ExperimentRunResult,
    outcomes: tuple[InterventionOutcome, ...],
) -> _algorithm.AlgorithmOutcomesComponent:
    return _algorithm._seal_algorithm_model(
        _algorithm.AlgorithmOutcomesComponent,
        {"run_id": result.run_id, "outcomes": outcomes},
    )


def _metrics_component(
    result: Stage2ExperimentRunResult,
) -> _algorithm.AlgorithmMetricsComponent:
    snapshot = result.final_memory_snapshot
    final_memory = _algorithm.AlgorithmFinalMemoryAttestation(
        run_id=snapshot.run_id,
        ledger_position=snapshot.ledger_position,
        ingestion_cursor=snapshot.ingestion_cursor,
        memory_cursor=snapshot.memory_cursor,
        record_count=len(snapshot.records),
        record_digests=tuple(canonical_digest(record) for record in snapshot.records),
        projection_digest=snapshot.projection_digest,
        source_snapshot_digest=canonical_digest(snapshot),
    )
    return _algorithm._seal_algorithm_model(
        _algorithm.AlgorithmMetricsComponent,
        {
            "run_id": result.run_id,
            "metrics": result.metrics,
            "final_budget_snapshot": result.final_budget_snapshot,
            "final_memory": final_memory,
        },
    )


def _boundary_attestation(
    boundary: Stage2BoundaryEvidence,
) -> _algorithm.AlgorithmBoundaryAttestation:
    return _algorithm._seal_algorithm_model(
        _algorithm.AlgorithmBoundaryAttestation,
        {
            "invocation_ordinal": boundary.observation.observed.invocation_ordinal,
            "boundary_event_id": boundary.boundary_event.event_id,
            "boundary_event_sequence": boundary.boundary_event.sequence,
            "window_digest": boundary.window.window_digest,
            "invocation_decision_id": boundary.invocation_decision.decision_id,
            "cycle_id": None if boundary.cycle is None else boundary.cycle.cycle_id,
            "source_evidence_digest": boundary.evidence_digest,
            "observation": boundary.observation,
        },
    )


def _ledger_attestation(entry: object) -> _algorithm.AlgorithmLedgerEntryAttestation:
    from saliencegate.ports.repository import LedgerEntry

    if type(entry) is not LedgerEntry:
        raise ArtifactExportError("algorithm ledger failed artifact-boundary validation")
    record = entry.record
    revision = getattr(record, "revision", None)
    state = getattr(record, "state", None)
    record_type = getattr(record, "record_type", None)
    if type(record_type) is not str:
        raise ArtifactExportError("algorithm ledger failed artifact-boundary validation")
    return _algorithm.AlgorithmLedgerEntryAttestation(
        position=entry.position,
        record_key=entry.record_key,
        record_type=record_type,
        record_revision=revision if type(revision) is int else None,
        record_state=state.value if isinstance(state, StrEnum) else None,
        source_record_digest=canonical_digest(record),
        record_tag=entry.record_tag,
        previous_chain_tag=entry.previous_chain_tag,
        chain_tag=entry.chain_tag,
    )


def _attestations_component(
    result: Stage2ExperimentRunResult,
    classification: _algorithm.AlgorithmArtifactClassification,
) -> _algorithm.AlgorithmAttestationsComponent:
    return _algorithm._seal_algorithm_model(
        _algorithm.AlgorithmAttestationsComponent,
        {
            "run_id": result.run_id,
            "boundaries": tuple(_boundary_attestation(boundary) for boundary in result.boundaries),
            "semantic_projection_digests": result.semantic_projection_digests,
            "repository_projection_digests": result.repository_projection_digests,
            "ledger_entries": tuple(_ledger_attestation(entry) for entry in result.ledger),
            "ledger_entry_count": result.ledger_entry_count,
            "ledger_head": result.ledger_head,
            "rebuild_equivalent": result.rebuild_equivalent,
            "source_result_digest": result.result_digest,
            "raw_synthetic_result": (
                result if classification is ArtifactClassification.SYNTHETIC_RAW else None
            ),
        },
    )


def _project_algorithm_components(
    result: Stage2ExperimentRunResult,
    execution: _algorithm.AlgorithmExecutionAttestation,
    classification: _algorithm.AlgorithmArtifactClassification,
) -> tuple[
    dict[_algorithm.AlgorithmArtifactComponentName, BaseModel],
    tuple[InterventionOutcome, ...],
]:
    outcomes = _outcomes(result)
    return (
        {
            _algorithm.AlgorithmArtifactComponentName.RUN: _run_component(result, execution),
            _algorithm.AlgorithmArtifactComponentName.TRAJECTORY: _trajectory_component(result),
            _algorithm.AlgorithmArtifactComponentName.CALLS: _calls_component(result),
            _algorithm.AlgorithmArtifactComponentName.DECISIONS: _decisions_component(result),
            _algorithm.AlgorithmArtifactComponentName.CYCLES: _cycles_component(result),
            _algorithm.AlgorithmArtifactComponentName.DELIVERIES: _deliveries_component(result),
            _algorithm.AlgorithmArtifactComponentName.OUTCOMES: _outcomes_component(
                result, outcomes
            ),
            _algorithm.AlgorithmArtifactComponentName.METRICS: _metrics_component(result),
            _algorithm.AlgorithmArtifactComponentName.ATTESTATIONS: (
                _attestations_component(result, classification)
            ),
        },
        outcomes,
    )
