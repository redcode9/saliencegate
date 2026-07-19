from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, Self, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.domain import (
    CycleRecord,
    CycleState,
    DeliveryRecord,
    DeliveryState,
    InterventionAction,
    InterventionDecision,
    InvocationDecision,
    TraceEvent,
    canonical_digest,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.domain.records import Sha256Digest
from saliencegate.intervention import (
    DeterministicSelectorProvenance,
    GroundingConfig,
    GroundingContext,
    GroundingReceipt,
    GroundingState,
    verify_grounded_intervention,
)
from saliencegate.memory.materialize import (
    OperationMaterializationRequest,
    validated_materialized_bank_operations_for_request,
)
from saliencegate.ports.two_phase import (
    CallReceipt,
    PhaseOneCycleResult,
    TwoPhaseCycleRequest,
    TwoPhaseCycleResult,
)
from saliencegate.prompts import PAPER_TWO_PHASE_FORCED_REMINDER_V1, PAPER_TWO_PHASE_V1
from saliencegate.prompts.paper_two_phase_v1 import PaperTwoPhasePromptBundle
from saliencegate.runtime.message_window import MessageWindow
from saliencegate.runtime.scheduling import FixedStepSchedule

from .conditions import (
    BankMaintenanceMode,
    CandidateBankMode,
    ResolvedStage2Condition,
    SelectionMode,
    Stage2ConditionId,
    Stage2ConditionObservation,
    Stage2ObservedBehavior,
    resolve_stage2_condition,
)
from .retrieval import (
    RetrievalRequest,
    RetrievalResult,
    build_retrieval_request,
    retrieval_selector_provenance,
    validated_retrieval_result,
)

STAGE2_BOUNDARY_EVIDENCE_SCHEMA_VERSION: Literal["stage2-boundary-evidence/v1"] = (
    "stage2-boundary-evidence/v1"
)

_EVIDENCE_DIGEST_DOMAIN = "saliencegate:experiments:stage2-boundary-evidence:v1"
_T = TypeVar("_T", bound=BaseModel)


class Stage2EvidenceError(ValueError):
    """A value-free failure at the offline experiment evidence boundary."""

    def __init__(self) -> None:
        super().__init__("offline experiment boundary evidence failed validation")


class _EvidenceModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def _exact(value: _T, expected_type: type[_T]) -> _T:
    if type(value) is not expected_type:
        raise ValueError("offline experiment evidence requires an exact source type")
    checked = expected_type.model_validate_json(value.model_dump_json(warnings=False))
    if checked != value or canonical_json(checked) != canonical_json(value):  # pragma: no cover
        raise ValueError("offline experiment evidence source failed exact validation")
    return checked


def _evidence_digest(values: Mapping[str, object]) -> str:
    material = {key: value for key, value in values.items() if key != "evidence_digest"}
    return length_prefixed_sha256(canonical_json(material), domain=_EVIDENCE_DIGEST_DOMAIN)


def _memory_mutation_count(result: TwoPhaseCycleResult | PhaseOneCycleResult) -> int:
    delta = result.validated_delta
    return (
        len(delta.creates)
        + len(delta.updates)
        + len(delta.invalidations)
        + int(delta.private_status_replacement is not None)
    )


def _phase_two_schema_digest(condition: ResolvedStage2Condition) -> str | None:
    if condition.expected.selection_mode is SelectionMode.MODEL_OPTIONAL:
        return condition.shared_controls.optional_phase_two_schema_digest
    if condition.expected.selection_mode is SelectionMode.MODEL_REQUIRED:
        return condition.shared_controls.forced_phase_two_schema_digest
    return None


def _prompt_bundle(condition: ResolvedStage2Condition) -> PaperTwoPhasePromptBundle:
    if condition.condition_id is Stage2ConditionId.ALWAYS_INJECT:
        return PAPER_TWO_PHASE_FORCED_REMINDER_V1
    return PAPER_TWO_PHASE_V1


def _validate_common_sources(
    *,
    condition: ResolvedStage2Condition,
    schedule: FixedStepSchedule,
    invocation_decision: InvocationDecision,
    boundary_event: TraceEvent,
    window: MessageWindow,
) -> tuple[ResolvedStage2Condition, int]:
    checked_condition = _exact(condition, ResolvedStage2Condition)
    resolve_stage2_condition(checked_condition.condition_id)
    checked_schedule = _exact(schedule, FixedStepSchedule)
    checked_decision = _exact(invocation_decision, InvocationDecision)
    checked_event = _exact(boundary_event, TraceEvent)
    checked_window = _exact(window, MessageWindow)
    scheduled = checked_schedule.decisions[-1]
    active = checked_condition.condition_id is not Stage2ConditionId.NO_MEMORY
    if (
        checked_schedule.run_id != checked_event.run_id
        or checked_schedule.boundary_event_sequence != checked_event.sequence
        or checked_schedule.trajectory_prefix_digest != checked_window.trajectory_prefix_digest
        or scheduled.event_id != checked_event.event_id
        or scheduled.event_sequence != checked_event.sequence
        or not scheduled.invoke
        or scheduled.invocation_ordinal is None
        or checked_decision.run_id != checked_event.run_id
        or checked_decision.event_sequence != checked_event.sequence
        or checked_decision.created_at != checked_event.timestamp
        or checked_decision.configuration_digest != checked_condition.condition_digest
        or checked_decision.invoke is not active
        or checked_window.run_id != checked_event.run_id
        or checked_window.boundary_event_id != checked_event.event_id
        or checked_window.boundary_event_sequence != checked_event.sequence
    ):
        raise ValueError("offline experiment schedule, decision, event, and window do not match")
    return checked_condition, scheduled.invocation_ordinal


def _validate_cycle_request(
    *,
    condition: ResolvedStage2Condition,
    invocation_decision: InvocationDecision,
    boundary_event: TraceEvent,
    window: MessageWindow,
    cycle: CycleRecord,
    request: TwoPhaseCycleRequest,
) -> tuple[CycleRecord, TwoPhaseCycleRequest]:
    checked_cycle = _exact(cycle, CycleRecord)
    checked_request = _exact(request, TwoPhaseCycleRequest)
    running = checked_request.cycle_receipt.cycle
    grounding = condition.shared_controls.grounding
    boundary_events = tuple(
        item
        for item in checked_request.grounding_state.events
        if item.sequence == boundary_event.sequence
    )
    identity_fields = (
        "cycle_id",
        "run_id",
        "invocation_decision_id",
        "policy_version",
        "configuration_digest",
        "grounding_version",
        "grounding_configuration",
        "grounding_configuration_digest",
        "requested_delivery_target",
        "first_event_sequence",
        "last_event_sequence",
        "created_at",
        "batch_digest",
        "budget_reservation",
    )
    if (
        checked_cycle.state is not CycleState.COMMITTED
        or running.state is not CycleState.RUNNING
        or checked_cycle.revision != running.revision + 1
        or any(getattr(checked_cycle, name) != getattr(running, name) for name in identity_fields)
        or checked_cycle.run_id != boundary_event.run_id
        or checked_cycle.invocation_decision_id != invocation_decision.decision_id
        or checked_cycle.policy_version != invocation_decision.policy_version
        or checked_cycle.configuration_digest != condition.condition_digest
        or checked_cycle.last_event_sequence != boundary_event.sequence
        or checked_cycle.batch_digest != window.window_digest
        or checked_cycle.grounding_version != grounding.pipeline_version
        or checked_cycle.grounding_configuration != grounding.configuration
        or checked_cycle.grounding_configuration_digest != grounding.configuration_digest
        or checked_cycle.requested_delivery_target
        is not condition.shared_controls.requested_delivery_target
        or checked_request.window != window
        or checked_request.created_at != boundary_event.timestamp
        or checked_request.current_bank.run_id != boundary_event.run_id
        or checked_request.current_bank.as_of != checked_request.created_at
        or len(boundary_events) != 1
        or boundary_events[0] != boundary_event
    ):
        raise ValueError("experiment cycle and request do not match their boundary")
    return checked_cycle, checked_request


def _validate_prompt_receipts(
    *,
    condition: ResolvedStage2Condition,
    prompt_bundle_digest: str,
    receipts: tuple[CallReceipt, ...],
) -> None:
    bundle = _prompt_bundle(condition)
    identities = {item.phase: item for item in bundle.identity.templates}
    if prompt_bundle_digest != bundle.identity.bundle_digest:
        raise ValueError("offline experiment execution uses the wrong prompt bundle")
    for receipt in receipts:
        identity = identities.get(receipt.phase)
        if identity is None or (
            receipt.prompt_template_id != identity.template_id
            or receipt.prompt_template_digest != identity.template_digest
        ):
            raise ValueError("experiment call receipt uses the wrong prompt template")


def _validate_execution_links(
    *,
    condition: ResolvedStage2Condition,
    boundary_event: TraceEvent,
    window: MessageWindow,
    cycle: CycleRecord,
    request: TwoPhaseCycleRequest,
    result: TwoPhaseCycleResult | PhaseOneCycleResult,
    call_receipts: tuple[CallReceipt, ...],
) -> None:
    checked_receipts = tuple(_exact(item, CallReceipt) for item in call_receipts)
    materialization = result.materialization
    settlement = cycle.budget_settlement
    assignments = result.memory_id_assignments
    provider_known = all(
        item.usage.provider_input_tokens is not None
        and item.usage.provider_output_tokens is not None
        for item in checked_receipts
    )
    canonical_known = all(
        item.usage.canonical_input_tokens is not None
        and item.usage.canonical_output_tokens is not None
        for item in checked_receipts
    )
    if settlement is None:  # pragma: no cover - committed-cycle invariant
        raise ValueError("committed experiment cycle lacks a settlement")
    materialization_request = OperationMaterializationRequest(
        schema_version="operation-materialization-request/v1",
        cycle_receipt=request.cycle_receipt,
        proposal=result.memory_edit_output,
        delta_id=request.delta_id,
        created_at=request.created_at,
        assigned_memory_ids=request.assigned_memory_ids[: len(assignments)],
    )
    exact_materialization = validated_materialized_bank_operations_for_request(
        materialization_request,
        materialization,
    )
    if (
        result.request_digest != request.request_digest
        or result.run_id != cycle.run_id
        or result.cycle_id != cycle.cycle_id
        or result.window_digest != window.window_digest
        or result.current_bank_view_digest != request.current_bank.view_digest
        or result.call_receipts != checked_receipts
        or tuple(item.phase for item in checked_receipts) != condition.expected.call_phases
        or tuple(item.model_call_index for item in checked_receipts)
        != tuple(range(len(checked_receipts)))
        or any(item.attempt != 0 for item in checked_receipts)
        or any(
            item.run_id != cycle.run_id
            or item.cycle_id != cycle.cycle_id
            or item.window_digest != window.window_digest
            for item in checked_receipts
        )
        or materialization.run_id != cycle.run_id
        or materialization.source_cycle_id != cycle.cycle_id
        or materialization.source_last_event_sequence != boundary_event.sequence
        or materialization.source_ingestion_cursor != boundary_event.sequence
        or materialization.delta.created_at != request.created_at
        or materialization.source_projection_digest != request.current_bank.source_projection_digest
        or exact_materialization != materialization
        or cycle.validated_delta != result.validated_delta
        or cycle.memory_id_assignments != assignments
        or {item.memory_id for item in assignments}
        != set(request.assigned_memory_ids[: len(assignments)])
        or cycle.model_call_digests != tuple(item.call_digest for item in checked_receipts)
        or cycle.model_call_latencies_us
        != tuple(item.usage.latency_us for item in checked_receipts)
        or settlement.model_calls != len(checked_receipts)
        or settlement.latency_us != sum(item.usage.latency_us for item in checked_receipts)
        or settlement.schema_repairs != 0
        or settlement.interventions
        != int(
            cycle.intervention is not None
            and cycle.intervention.action is InterventionAction.REMIND
        )
        or (
            provider_known
            and settlement.input_tokens
            != sum(cast(int, item.usage.provider_input_tokens) for item in checked_receipts)
        )
        or (
            provider_known
            and settlement.output_tokens
            != sum(cast(int, item.usage.provider_output_tokens) for item in checked_receipts)
        )
        or (
            canonical_known
            and settlement.canonical_token_equivalents
            != sum(
                cast(int, item.usage.canonical_input_tokens)
                + cast(int, item.usage.canonical_output_tokens)
                for item in checked_receipts
            )
        )
    ):
        raise ValueError("offline experiment execution does not match its committed cycle")
    _validate_prompt_receipts(
        condition=condition,
        prompt_bundle_digest=result.prompt_bundle_digest,
        receipts=checked_receipts,
    )


def _grounding_receipt(intervention: InterventionDecision) -> GroundingReceipt:
    return GroundingReceipt.model_validate_json(canonical_json(intervention.grounding_receipt))


def _cycle_selector_provenance(
    cycle: CycleRecord,
) -> DeterministicSelectorProvenance | None:
    if cycle.selector_provenance is None:
        return None
    return DeterministicSelectorProvenance.model_validate_json(
        canonical_json(cycle.selector_provenance)
    )


def _validate_model_intervention(
    *,
    condition: ResolvedStage2Condition,
    cycle: CycleRecord,
    request: TwoPhaseCycleRequest,
    result: TwoPhaseCycleResult,
    receipts: tuple[CallReceipt, ...],
) -> None:
    intervention = cycle.intervention
    if (
        intervention is None
        or intervention != result.intervention
        or _cycle_selector_provenance(cycle) is not None
    ):
        raise ValueError("experiment model intervention differs from the committed decision")
    receipt = _grounding_receipt(intervention)
    final_call = receipts[-1]
    if (
        intervention.intervention_id != request.intervention_id
        or receipt.receipt_version != "grounding-receipt/v1"
        or receipt.model_call_index != final_call.model_call_index
        or receipt.model_call_digest != final_call.call_digest
        or receipt.selector_provenance is not None
    ):
        raise ValueError("experiment model intervention lacks exact final-call provenance")
    if (
        condition.condition_id is Stage2ConditionId.ALWAYS_INJECT
        and result.intervention_output is not None
        and (
            result.intervention_output.action is not InterventionAction.REMIND
            or not result.intervention_output.claims
        )
    ):
        raise ValueError("forced-reminder output does not satisfy its provider schema")


def _candidate_grounding_state(
    request: TwoPhaseCycleRequest,
    result: PhaseOneCycleResult,
) -> GroundingState:
    return GroundingState(
        schema_version="1.0",
        events=request.grounding_state.events,
        memories=result.materialization.active_bank,
        reminder_history=request.grounding_state.reminder_history,
    )


def _validate_retrieval_intervention(
    *,
    condition: ResolvedStage2Condition,
    boundary_event: TraceEvent,
    window: MessageWindow,
    cycle: CycleRecord,
    request: TwoPhaseCycleRequest,
    result: PhaseOneCycleResult,
    retrieval_request: RetrievalRequest,
    retrieval_result: RetrievalResult,
) -> None:
    checked_retrieval_request = _exact(retrieval_request, RetrievalRequest)
    checked_retrieval_result = _exact(retrieval_result, RetrievalResult)
    expected_request = build_retrieval_request(
        condition=condition,
        window=window,
        materialization=result.materialization,
    )
    exact_result = validated_retrieval_result(
        checked_retrieval_request,
        checked_retrieval_result,
    )
    provenance = retrieval_selector_provenance(
        checked_retrieval_request,
        exact_result,
    )
    cycle_provenance = _cycle_selector_provenance(cycle)
    intervention = cycle.intervention
    if intervention is None:
        raise ValueError("retrieval cycle lacks its grounded intervention")
    receipt = _grounding_receipt(intervention)
    if (
        checked_retrieval_request != expected_request
        or checked_retrieval_result != exact_result
        or receipt.receipt_version != "grounding-receipt/v2"
        or receipt.model_call_index is not None
        or receipt.model_call_digest is not None
        or receipt.selector_provenance != provenance
        or cycle_provenance != provenance
        or receipt.proposal_action is not checked_retrieval_result.selection.action
        or receipt.claims != checked_retrieval_result.selection.claims
        or receipt.confidence != checked_retrieval_result.selection.confidence
        or intervention.intervention_id != request.intervention_id
    ):
        raise ValueError("retrieval intervention lacks exact selector provenance")
    state = _candidate_grounding_state(request, result)
    context = GroundingContext(
        schema_version="2.0",
        intervention_id=intervention.intervention_id,
        run_id=cycle.run_id,
        cycle_id=cycle.cycle_id,
        current_event_sequence=boundary_event.sequence,
        created_at=result.materialization.delta.created_at,
        requested_delivery_target=condition.shared_controls.requested_delivery_target,
        selector_provenance=provenance,
    )
    GroundingConfig.model_validate_json(
        canonical_json(condition.shared_controls.grounding.configuration)
    )
    verify_grounded_intervention(
        intervention,
        context=context,
        state=state,
        expected_configuration=condition.shared_controls.grounding,
    )


def _validate_delivery(cycle: CycleRecord, delivery_record: DeliveryRecord | None) -> None:
    intervention = cycle.intervention
    if intervention is None:  # pragma: no cover - committed-cycle invariant
        raise ValueError("committed experiment cycle lacks an intervention")
    if intervention.action is InterventionAction.SILENCE:
        if delivery_record is not None:
            raise ValueError("safe silence cannot carry a delivery")
        return
    if delivery_record is None or intervention.rendered_text is None:
        raise ValueError("a reminder requires its final delivery")
    delivery = _exact(delivery_record, DeliveryRecord)
    if (
        delivery.run_id != cycle.run_id
        or delivery.cycle_id != cycle.cycle_id
        or delivery.intervention_id != intervention.intervention_id
        or delivery.target is not cycle.requested_delivery_target
        or delivery.rendered_text_digest != canonical_digest(intervention.rendered_text)
        or delivery.state
        in (DeliveryState.PENDING, DeliveryState.CLAIMED, DeliveryState.ATTEMPTING)
    ):
        raise ValueError("final delivery does not match the grounded reminder")


def _derive_stage2_condition_observation(
    *,
    condition: ResolvedStage2Condition,
    schedule: FixedStepSchedule,
    invocation_decision: InvocationDecision,
    boundary_event: TraceEvent,
    window: MessageWindow,
    cycle: CycleRecord | None,
    request: TwoPhaseCycleRequest | None,
    two_phase_result: TwoPhaseCycleResult | None,
    phase_one_result: PhaseOneCycleResult | None,
    call_receipts: tuple[CallReceipt, ...],
    retrieval_request: RetrievalRequest | None,
    retrieval_result: RetrievalResult | None,
    delivery_record: DeliveryRecord | None,
) -> Stage2ConditionObservation:
    checked_condition, invocation_ordinal = _validate_common_sources(
        condition=condition,
        schedule=schedule,
        invocation_decision=invocation_decision,
        boundary_event=boundary_event,
        window=window,
    )
    if type(call_receipts) is not tuple or any(
        type(item) is not CallReceipt for item in call_receipts
    ):
        raise ValueError("experiment call receipts are not an exact tuple")
    active = checked_condition.condition_id is not Stage2ConditionId.NO_MEMORY
    if not active:
        if (
            any(
                item is not None
                for item in (
                    cycle,
                    request,
                    two_phase_result,
                    phase_one_result,
                    retrieval_request,
                    retrieval_result,
                    delivery_record,
                )
            )
            or call_receipts
        ):
            raise ValueError("no-memory evidence cannot contain cycle effects")
        observed = Stage2ObservedBehavior(
            run_id=boundary_event.run_id,
            invocation_decision_id=invocation_decision.decision_id,
            invocation_decision_digest=canonical_digest(invocation_decision),
            boundary_event_id=boundary_event.event_id,
            boundary_event_sequence=boundary_event.sequence,
            invocation_ordinal=invocation_ordinal,
            schedule_digest=schedule.schedule_digest,
            window_digest=window.window_digest,
            cycle_id=None,
            call_phases=(),
            call_receipt_digests=(),
            candidate_bank_mode=CandidateBankMode.DISABLED,
            current_bank_view_digest=None,
            candidate_bank_view_digest=None,
            materialization_digest=None,
            bank_maintenance_mode=BankMaintenanceMode.DISABLED,
            selection_mode=SelectionMode.DISABLED,
            phase_two_schema_digest=None,
            retrieval_request_digest=None,
            retrieval_result_digest=None,
            memory_mutation_count=0,
            intervention_action=None,
            intervention_digest=None,
            delivery_record_count=0,
            delivery_record_digests=(),
        )
        return Stage2ConditionObservation(
            condition_id=checked_condition.condition_id,
            condition_digest=checked_condition.condition_digest,
            expected=checked_condition.expected,
            observed=observed,
            condition_violation=False,
        )

    if cycle is None or request is None:
        raise ValueError("an offline memory condition requires cycle evidence")
    checked_cycle, checked_request = _validate_cycle_request(
        condition=checked_condition,
        invocation_decision=invocation_decision,
        boundary_event=boundary_event,
        window=window,
        cycle=cycle,
        request=request,
    )
    if checked_condition.condition_id is Stage2ConditionId.RETRIEVAL_ALWAYS:
        if (
            type(phase_one_result) is not PhaseOneCycleResult
            or two_phase_result is not None
            or type(retrieval_request) is not RetrievalRequest
            or type(retrieval_result) is not RetrievalResult
            or len(call_receipts) != 1
        ):
            raise ValueError("retrieval condition requires one exact Phase 1 execution")
        phase_execution = _exact(phase_one_result, PhaseOneCycleResult)
        execution: TwoPhaseCycleResult | PhaseOneCycleResult = phase_execution
        _validate_execution_links(
            condition=checked_condition,
            boundary_event=boundary_event,
            window=window,
            cycle=checked_cycle,
            request=checked_request,
            result=phase_execution,
            call_receipts=call_receipts,
        )
        _validate_retrieval_intervention(
            condition=checked_condition,
            boundary_event=boundary_event,
            window=window,
            cycle=checked_cycle,
            request=checked_request,
            result=phase_execution,
            retrieval_request=retrieval_request,
            retrieval_result=retrieval_result,
        )
    else:
        if (
            type(two_phase_result) is not TwoPhaseCycleResult
            or phase_one_result is not None
            or retrieval_request is not None
            or retrieval_result is not None
            or len(call_receipts) != 2
        ):
            raise ValueError("model condition requires one exact two-call execution")
        model_execution = _exact(two_phase_result, TwoPhaseCycleResult)
        execution = model_execution
        _validate_execution_links(
            condition=checked_condition,
            boundary_event=boundary_event,
            window=window,
            cycle=checked_cycle,
            request=checked_request,
            result=model_execution,
            call_receipts=call_receipts,
        )
        _validate_model_intervention(
            condition=checked_condition,
            cycle=checked_cycle,
            request=checked_request,
            result=model_execution,
            receipts=call_receipts,
        )

    _validate_delivery(checked_cycle, delivery_record)
    intervention = checked_cycle.intervention
    if intervention is None:  # pragma: no cover - checked above
        raise ValueError("committed experiment cycle lacks an intervention")
    observed = Stage2ObservedBehavior(
        run_id=boundary_event.run_id,
        invocation_decision_id=invocation_decision.decision_id,
        invocation_decision_digest=canonical_digest(invocation_decision),
        boundary_event_id=boundary_event.event_id,
        boundary_event_sequence=boundary_event.sequence,
        invocation_ordinal=invocation_ordinal,
        schedule_digest=schedule.schedule_digest,
        window_digest=window.window_digest,
        cycle_id=checked_cycle.cycle_id,
        call_phases=tuple(item.phase for item in call_receipts),
        call_receipt_digests=tuple(item.receipt_digest for item in call_receipts),
        candidate_bank_mode=checked_condition.expected.candidate_bank_mode,
        current_bank_view_digest=execution.current_bank_view_digest,
        candidate_bank_view_digest=execution.candidate_bank_view_digest,
        materialization_digest=execution.materialization.materialization_digest,
        bank_maintenance_mode=checked_condition.expected.bank_maintenance_mode,
        selection_mode=checked_condition.expected.selection_mode,
        phase_two_schema_digest=_phase_two_schema_digest(checked_condition),
        retrieval_request_digest=(
            retrieval_request.request_digest if retrieval_request is not None else None
        ),
        retrieval_result_digest=(
            retrieval_result.result_digest if retrieval_result is not None else None
        ),
        memory_mutation_count=_memory_mutation_count(execution),
        intervention_action=intervention.action,
        intervention_digest=canonical_digest(intervention),
        delivery_record_count=int(delivery_record is not None),
        delivery_record_digests=(
            (canonical_digest(delivery_record),) if delivery_record is not None else ()
        ),
    )
    violation = (
        checked_condition.condition_id is Stage2ConditionId.ALWAYS_INJECT
        and intervention.action is InterventionAction.SILENCE
    )
    return Stage2ConditionObservation(
        condition_id=checked_condition.condition_id,
        condition_digest=checked_condition.condition_digest,
        expected=checked_condition.expected,
        observed=observed,
        condition_violation=violation,
    )


def derive_stage2_condition_observation(
    *,
    condition: ResolvedStage2Condition,
    schedule: FixedStepSchedule,
    invocation_decision: InvocationDecision,
    boundary_event: TraceEvent,
    window: MessageWindow,
    cycle: CycleRecord | None = None,
    request: TwoPhaseCycleRequest | None = None,
    two_phase_result: TwoPhaseCycleResult | None = None,
    phase_one_result: PhaseOneCycleResult | None = None,
    call_receipts: tuple[CallReceipt, ...] = (),
    retrieval_request: RetrievalRequest | None = None,
    retrieval_result: RetrievalResult | None = None,
    delivery_record: DeliveryRecord | None = None,
) -> Stage2ConditionObservation:
    """Derive the non-authoritative observation from complete boundary sources."""

    try:
        return _derive_stage2_condition_observation(
            condition=condition,
            schedule=schedule,
            invocation_decision=invocation_decision,
            boundary_event=boundary_event,
            window=window,
            cycle=cycle,
            request=request,
            two_phase_result=two_phase_result,
            phase_one_result=phase_one_result,
            call_receipts=call_receipts,
            retrieval_request=retrieval_request,
            retrieval_result=retrieval_result,
            delivery_record=delivery_record,
        )
    except Exception:
        raise Stage2EvidenceError() from None


class Stage2BoundaryEvidence(_EvidenceModel):
    """Complete authoritative sources for one scheduled offline experiment boundary."""

    schema_version: Literal["stage2-boundary-evidence/v1"] = STAGE2_BOUNDARY_EVIDENCE_SCHEMA_VERSION
    condition: ResolvedStage2Condition
    schedule: FixedStepSchedule = Field(repr=False)
    invocation_decision: InvocationDecision
    boundary_event: TraceEvent = Field(repr=False)
    window: MessageWindow = Field(repr=False)
    cycle: CycleRecord | None = Field(default=None, repr=False)
    request: TwoPhaseCycleRequest | None = Field(default=None, repr=False)
    two_phase_result: TwoPhaseCycleResult | None = Field(default=None, repr=False)
    phase_one_result: PhaseOneCycleResult | None = Field(default=None, repr=False)
    call_receipts: Annotated[tuple[CallReceipt, ...], Field(max_length=2, repr=False)] = ()
    retrieval_request: RetrievalRequest | None = Field(default=None, repr=False)
    retrieval_result: RetrievalResult | None = Field(default=None, repr=False)
    delivery_record: DeliveryRecord | None = Field(default=None, repr=False)
    observation: Stage2ConditionObservation
    evidence_digest: Sha256Digest = Field(default_factory=_evidence_digest)

    @model_validator(mode="after")
    def sources_and_projection_match_exactly(self) -> Self:
        expected = _derive_stage2_condition_observation(
            condition=self.condition,
            schedule=self.schedule,
            invocation_decision=self.invocation_decision,
            boundary_event=self.boundary_event,
            window=self.window,
            cycle=self.cycle,
            request=self.request,
            two_phase_result=self.two_phase_result,
            phase_one_result=self.phase_one_result,
            call_receipts=self.call_receipts,
            retrieval_request=self.retrieval_request,
            retrieval_result=self.retrieval_result,
            delivery_record=self.delivery_record,
        )
        if self.observation != expected or canonical_json(self.observation) != canonical_json(
            expected
        ):
            raise ValueError("experiment observation is not derived from its sources")
        values = self.model_dump(mode="json", exclude={"evidence_digest"}, warnings=False)
        if self.evidence_digest != _evidence_digest(values):
            raise ValueError("experiment boundary evidence digest does not match")
        return self


__all__ = [
    "STAGE2_BOUNDARY_EVIDENCE_SCHEMA_VERSION",
    "Stage2BoundaryEvidence",
    "Stage2EvidenceError",
    "derive_stage2_condition_observation",
]
