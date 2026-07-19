from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeVar, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.domain import (
    MAX_MEMORY_DELTA_ITEMS,
    BudgetAmounts,
    BudgetLimits,
    BudgetSnapshot,
    ConstraintStatus,
    CycleRecord,
    CycleState,
    DeduplicationGuarantee,
    DeliveryRecord,
    DeliveryState,
    InterventionAction,
    InterventionDecision,
    InterventionOutcome,
    InvocationDecision,
    MemoryRecord,
    NormalizedTraceEventDraft,
    OutcomeEvidenceMode,
    ReasonCode,
    RepeatedErrorStatus,
    TraceEvent,
    ValidityState,
    canonical_digest,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.domain import (
    cycle_id as derive_cycle_id,
)
from saliencegate.domain import (
    delivery_id as derive_delivery_id,
)
from saliencegate.domain.records import ComponentIdentifier, Sha256Digest
from saliencegate.intervention import (
    DeterministicSelectorProvenance,
    GroundingConfig,
    GroundingContext,
    GroundingPipeline,
    GroundingState,
    ReminderHistory,
    claim_fingerprint,
    verify_grounded_intervention,
)
from saliencegate.memory.two_phase import (
    PaperTwoPhaseCycleExecutor,
    RepositoryOperationMaterializer,
)
from saliencegate.models.replay_two_phase import (
    TwoPhaseReplayClient,
    two_phase_receipts_are_replay_native,
    two_phase_replay_fixture_digest_from_receipts,
)
from saliencegate.ports.adapters import (
    ADAPTER_CONTRACT_VERSION,
    AdapterCapabilities,
    DeliveryAdapter,
    DeliveryChannel,
    DeliveryEnvelope,
    DeliveryReceipt,
    DeliveryRole,
    InjectionMapping,
    adapter_capabilities_digest,
    enqueue_delivery_binding,
)
from saliencegate.ports.model_calls import StructuredCallClient, StructuredCallPhase
from saliencegate.ports.repository import (
    CycleReceipt,
    EnqueueDelivery,
    GroundingPin,
    LedgerEntry,
    LedgerHead,
    MemorySnapshot,
    ProjectionDigests,
    RunRepository,
)
from saliencegate.ports.trajectory import (
    AttestedTrajectoryEvent,
    AttestedTrajectoryPrefix,
    TrajectoryPrefixRequest,
    bind_persisted_trajectory_event,
)
from saliencegate.ports.two_phase import (
    CallReceipt,
    PhaseOneCycleResult,
    TwoPhaseCallPolicy,
    TwoPhaseCycleFailure,
    TwoPhaseCycleRequest,
    TwoPhaseCycleResult,
    TwoPhaseFailureReason,
    TwoPhaseModelProfile,
    TwoPhaseUsage,
)
from saliencegate.prompts import (
    PAPER_TWO_PHASE_FORCED_REMINDER_V1,
    PAPER_TWO_PHASE_V1,
    BankViewKind,
    build_active_bank_prompt_view,
)
from saliencegate.prompts.contracts import PromptBundleIdentity
from saliencegate.prompts.paper_two_phase_v1 import PaperTwoPhasePromptBundle
from saliencegate.repository.integrity import IntegrityContext
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.repository.projector import (
    apply_entry,
    empty_projection,
    projection_digests,
)
from saliencegate.repository.projector import (
    budget_snapshot as projected_budget_snapshot,
)
from saliencegate.repository.projector import snapshot as projected_memory_snapshot
from saliencegate.runtime.algorithm_result import (
    algorithm_runtime_uuid,
    algorithm_trace_digest,
    semantic_projection_digests,
)
from saliencegate.runtime.budget import BudgetGovernor
from saliencegate.runtime.cycles import CycleCoordinator
from saliencegate.runtime.delivery import DeliveryWorker
from saliencegate.runtime.fixed_step_core import (
    FixedStepTraceBoundary,
    FixedStepTraceDriver,
    FixedStepTraceInput,
    record_reconciled_invocation_decision,
)
from saliencegate.runtime.message_window import (
    MessageWindow,
    _project_verified_message_window,
)
from saliencegate.runtime.scheduling import (
    FixedStepReason,
    FixedStepSchedule,
    _project_verified_fixed_step_schedule,
)

from .conditions import (
    ResolvedStage2Condition,
    Stage2ConditionId,
    resolve_stage2_condition,
)
from .evidence import Stage2BoundaryEvidence, derive_stage2_condition_observation
from .retrieval import (
    RetrievalRequest,
    RetrievalResult,
    build_retrieval_request,
    retrieval_selector_provenance,
    retrieve_candidate_bank,
)
from .trajectory import Stage2Trajectory, load_stage2_trajectory

STAGE2_EXPERIMENT_METRICS_SCHEMA_VERSION: Literal["stage2-experiment-metrics/v1"] = (
    "stage2-experiment-metrics/v1"
)
STAGE2_EXPERIMENT_RUN_RESULT_SCHEMA_VERSION: Literal["stage2-experiment-run-result/v2"] = (
    "stage2-experiment-run-result/v2"
)
STAGE2_RESPONSE_FIXTURE_IDENTITY_SCHEMA_VERSION: Literal["stage2-response-fixture-identity/v1"] = (
    "stage2-response-fixture-identity/v1"
)
STAGE2_EXPERIMENT_POLICY_VERSION: Literal["stage2-fixed-step-comparison/v1"] = (
    "stage2-fixed-step-comparison/v1"
)

_RESULT_DIGEST_DOMAIN = "saliencegate:experiments:stage2-run-result:v2"
_MAX_EVENTS = 10_000
_MAX_CALLS = _MAX_EVENTS * 2
_MODEL_ID = "gpt-oss:20b"
_MODEL_PROFILE_ID = "stage2-openai-compatible-replay/v1"
_DELIVERY_ADAPTER_ID = "stage2-offline-delivery/v1"
_MAX_PROVIDER_INPUT_TOKENS = 262_144
_MAX_PROVIDER_OUTPUT_TOKENS = 65_536
_MAX_CALL_LATENCY_US = 600_000_000

_BoundaryT = TypeVar("_BoundaryT")


class Stage2ExperimentError(RuntimeError):
    """A value-free failure at the closed offline experiment boundary."""

    def __init__(self) -> None:
        super().__init__("offline experiment failed validation")


class _ExperimentModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def _result_digest(values: Mapping[str, object]) -> str:
    material = {key: value for key, value in values.items() if key != "result_digest"}
    material["run_id"] = str(values["run_id"])
    return length_prefixed_sha256(canonical_json(material), domain=_RESULT_DIGEST_DOMAIN)


class Stage2ExperimentMetrics(_ExperimentModel):
    """Only measurements supported directly by the offline execution evidence."""

    schema_version: Literal["stage2-experiment-metrics/v1"] = (
        STAGE2_EXPERIMENT_METRICS_SCHEMA_VERSION
    )
    model_call_count: Annotated[int, Field(ge=0, le=_MAX_CALLS)]
    provider_input_tokens: Annotated[int, Field(ge=0)] | None
    provider_output_tokens: Annotated[int, Field(ge=0)] | None
    canonical_input_tokens: Annotated[int, Field(ge=0)] | None
    canonical_output_tokens: Annotated[int, Field(ge=0)] | None
    canonical_token_equivalents: Annotated[int, Field(ge=0)] | None
    memory_call_latency_us: Annotated[int, Field(ge=0)]
    intervention_count: Annotated[int, Field(ge=0, le=_MAX_EVENTS)]
    grounding_rejection_count: Annotated[int, Field(ge=0, le=_MAX_EVENTS)]
    provenance_validated_boundary_count: Annotated[int, Field(ge=0, le=_MAX_EVENTS)]
    memory_mutation_count: Annotated[int, Field(ge=0)]
    condition_violation_count: Annotated[int, Field(ge=0, le=_MAX_EVENTS)]

    @model_validator(mode="after")
    def token_totals_are_honest(self) -> Self:
        if (
            (self.provider_input_tokens is None) != (self.provider_output_tokens is None)
            or (self.canonical_input_tokens is None) != (self.canonical_output_tokens is None)
            or (self.canonical_input_tokens is None) != (self.canonical_token_equivalents is None)
            or (
                self.canonical_input_tokens is not None
                and self.canonical_output_tokens is not None
                and self.canonical_token_equivalents
                != self.canonical_input_tokens + self.canonical_output_tokens
            )
        ):
            raise ValueError("offline experiment metric token totals are inconsistent")
        return self


class Stage2ResponseFixtureIdentity(_ExperimentModel):
    """Replay fixture identity asserted by the consuming replay client."""

    schema_version: Literal["stage2-response-fixture-identity/v1"] = (
        STAGE2_RESPONSE_FIXTURE_IDENTITY_SCHEMA_VERSION
    )
    replay_id: ComponentIdentifier
    fixture_digest: Sha256Digest
    response_count: Annotated[int, Field(ge=1, le=_MAX_CALLS)]


def _metrics(
    boundaries: tuple[Stage2BoundaryEvidence, ...],
    calls: tuple[CallReceipt, ...],
) -> Stage2ExperimentMetrics:
    provider_known = all(
        item.usage.provider_input_tokens is not None
        and item.usage.provider_output_tokens is not None
        for item in calls
    )
    canonical_known = all(
        item.usage.canonical_input_tokens is not None
        and item.usage.canonical_output_tokens is not None
        for item in calls
    )
    provider_input = (
        sum(cast(int, item.usage.provider_input_tokens) for item in calls)
        if provider_known
        else None
    )
    provider_output = (
        sum(cast(int, item.usage.provider_output_tokens) for item in calls)
        if provider_known
        else None
    )
    canonical_input = (
        sum(cast(int, item.usage.canonical_input_tokens) for item in calls)
        if canonical_known
        else None
    )
    canonical_output = (
        sum(cast(int, item.usage.canonical_output_tokens) for item in calls)
        if canonical_known
        else None
    )
    active = tuple(item for item in boundaries if item.cycle is not None)
    interventions = tuple(
        item.cycle.intervention
        for item in active
        if item.cycle is not None and item.cycle.intervention is not None
    )
    return Stage2ExperimentMetrics(
        model_call_count=len(calls),
        provider_input_tokens=provider_input,
        provider_output_tokens=provider_output,
        canonical_input_tokens=canonical_input,
        canonical_output_tokens=canonical_output,
        canonical_token_equivalents=(
            canonical_input + canonical_output
            if canonical_input is not None and canonical_output is not None
            else None
        ),
        memory_call_latency_us=sum(
            item.usage.latency_us for item in calls if item.phase is StructuredCallPhase.MEMORY_EDIT
        ),
        intervention_count=sum(item.action is InterventionAction.REMIND for item in interventions),
        grounding_rejection_count=sum(
            item.action is InterventionAction.SILENCE
            and item.reason_code is not ReasonCode.SILENCE_SELECTED
            for item in interventions
        ),
        provenance_validated_boundary_count=len(active),
        memory_mutation_count=sum(
            item.observation.observed.memory_mutation_count for item in boundaries
        ),
        condition_violation_count=sum(item.observation.condition_violation for item in boundaries),
    )


@dataclass(frozen=True, slots=True)
class _BoundaryProjection:
    decision: InvocationDecision
    evidence: Stage2BoundaryEvidence | None


@dataclass(slots=True)
class _RunningCycle:
    receipt: CycleReceipt | None = None


@dataclass(frozen=True, slots=True)
class _CycleExecution:
    cycle: CycleRecord
    request: TwoPhaseCycleRequest
    two_phase_result: TwoPhaseCycleResult | None
    phase_one_result: PhaseOneCycleResult | None
    calls: tuple[CallReceipt, ...]
    retrieval_request: RetrievalRequest | None
    retrieval_result: RetrievalResult | None
    delivery: DeliveryRecord | None


@dataclass(frozen=True, slots=True)
class _OfflineDeliveryAdapter:
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            schema_version="1.0",
            adapter_id=_DELIVERY_ADAPTER_ID,
            pre_action_interception=False,
            deduplicates_delivery_id=True,
            deduplication_guarantee=DeduplicationGuarantee.DURABLE_DELIVERY_ID,
            injection_mappings=(
                InjectionMapping(
                    channel=DeliveryChannel.PROVIDER_DATA,
                    role=DeliveryRole.DATA,
                    provider_channel="context",
                ),
            ),
        )

    async def deliver(self, delivery: DeliveryEnvelope) -> DeliveryReceipt:
        return DeliveryReceipt(
            schema_version="1.0",
            delivery_id=delivery.delivery_id,
            attempt_id=delivery.attempt_id,
            attempt_number=delivery.attempt_number,
            adapter_id=_DELIVERY_ADAPTER_ID,
            target_request_id=delivery.target_request_id,
            delivered_at=delivery.created_at + timedelta(microseconds=1),
            provider_receipt_id=f"stage2-offline-{delivery.attempt_number}",
        )


async def _complete_boundary(
    operation: Coroutine[Any, Any, _BoundaryT],
) -> tuple[_BoundaryT, bool]:
    task = asyncio.create_task(operation)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.cancelled():
                cancelled = True
        except Exception:
            break
    try:
        result = task.result()
    except asyncio.CancelledError:
        raise
    except Exception:
        if cancelled:
            raise asyncio.CancelledError() from None
        raise
    return result, cancelled


def _prompt_bundle(
    condition: ResolvedStage2Condition,
) -> PaperTwoPhasePromptBundle:
    if condition.condition_id is Stage2ConditionId.ALWAYS_INJECT:
        return PAPER_TWO_PHASE_FORCED_REMINDER_V1
    return PAPER_TWO_PHASE_V1


def _model_profile(condition: ResolvedStage2Condition) -> TwoPhaseModelProfile:
    bundle = _prompt_bundle(condition).identity
    return TwoPhaseModelProfile(
        schema_version="two-phase-model-profile/v1",
        profile_id=_MODEL_PROFILE_ID,
        model_id=_MODEL_ID,
        prompt_bundle_id=bundle.bundle_id,
        prompt_bundle_digest=bundle.bundle_digest,
    )


def _call_policy() -> TwoPhaseCallPolicy:
    return TwoPhaseCallPolicy(
        schema_version="two-phase-call-policy/v1",
        max_model_calls=2,
        max_schema_repairs=0,
        client_retries=0,
        max_provider_input_tokens=_MAX_PROVIDER_INPUT_TOKENS,
        max_provider_output_tokens=_MAX_PROVIDER_OUTPUT_TOKENS,
        max_total_latency_us=_MAX_CALL_LATENCY_US * 2,
        max_call_latency_us=_MAX_CALL_LATENCY_US,
    )


def _reservation(policy: TwoPhaseCallPolicy) -> BudgetAmounts:
    return BudgetAmounts(
        model_calls=policy.max_model_calls,
        input_tokens=policy.max_provider_input_tokens,
        output_tokens=policy.max_provider_output_tokens,
        canonical_token_equivalents=(
            policy.max_provider_input_tokens + policy.max_provider_output_tokens
        ),
        latency_us=policy.max_total_latency_us,
        interventions=1,
        schema_repairs=policy.max_schema_repairs,
    )


def _budget_limits(reservation: BudgetAmounts, event_count: int) -> BudgetLimits:
    return BudgetLimits(
        model_calls=reservation.model_calls * event_count,
        input_tokens=reservation.input_tokens * event_count,
        output_tokens=reservation.output_tokens * event_count,
        canonical_token_equivalents=reservation.canonical_token_equivalents * event_count,
        latency_us=reservation.latency_us * event_count,
        interventions=reservation.interventions * event_count,
        schema_repairs=reservation.schema_repairs * event_count,
        max_call_latency_us=_MAX_CALL_LATENCY_US,
    )


def _usage_settlement(
    usage: TwoPhaseUsage,
    reservation: BudgetAmounts,
    *,
    interventions: int,
) -> BudgetAmounts:
    provider_known = (
        usage.provider_input_tokens is not None and usage.provider_output_tokens is not None
    )
    return BudgetAmounts(
        model_calls=usage.model_calls,
        input_tokens=(
            cast(int, usage.provider_input_tokens) if provider_known else reservation.input_tokens
        ),
        output_tokens=(
            cast(int, usage.provider_output_tokens) if provider_known else reservation.output_tokens
        ),
        canonical_token_equivalents=(
            usage.canonical_token_equivalents
            if usage.canonical_token_equivalents is not None
            else reservation.canonical_token_equivalents
        ),
        latency_us=usage.latency_us,
        interventions=interventions,
        schema_repairs=usage.schema_repairs,
    )


def _failure_reason(value: TwoPhaseFailureReason) -> ReasonCode:
    if value is TwoPhaseFailureReason.MODEL_TIMEOUT:
        return ReasonCode.MODEL_TIMEOUT
    if value is TwoPhaseFailureReason.MODEL_ERROR:
        return ReasonCode.MODEL_ERROR
    return ReasonCode.INVALID_STRUCTURED_OUTPUT


def _delivery_id_factory(trace_digest: str, delivery_id: UUID) -> Callable[[], UUID]:
    ordinal = 0

    def next_identifier() -> UUID:
        nonlocal ordinal
        ordinal += 1
        return algorithm_runtime_uuid(
            trace_digest,
            "stage2-delivery-worker",
            delivery_id,
            ordinal,
        )

    return next_identifier


def _repository_id_factory(trace_digest: str) -> Callable[[], UUID]:
    ordinal = 0

    def next_identifier() -> UUID:
        nonlocal ordinal
        ordinal += 1
        return algorithm_runtime_uuid(trace_digest, "stage2-repository", ordinal)

    return next_identifier


def _persisted_draft_digest(event: TraceEvent) -> str:
    return canonical_digest(
        NormalizedTraceEventDraft(
            run_id=event.run_id,
            source_event_id=event.source_event_id,
            timestamp=event.timestamp,
            event_type=event.event_type,
            phase=event.phase,
            payload=event.payload,
            parent_ids=event.parent_ids,
            source_adapter=event.source_adapter,
            trust_label=event.trust_label,
        )
    )


def _prefix_at(
    *,
    run_id: UUID,
    items: tuple[AttestedTrajectoryEvent, ...],
    sequence: int,
) -> AttestedTrajectoryPrefix:
    selected = items[:sequence]
    request = TrajectoryPrefixRequest(
        schema_version="trajectory-prefix-request/v1",
        run_id=run_id,
        boundary_event_sequence=sequence,
        bindings=tuple(item.binding for item in selected),
    )
    return AttestedTrajectoryPrefix(
        schema_version="attested-trajectory-prefix/v1",
        run_id=run_id,
        boundary_event_sequence=sequence,
        request_digest=request.request_digest,
        items=selected,
    )


def _ledger_record_key(record: object) -> str:
    if type(record) is TraceEvent:
        return f"trace_event:{record.event_id}"
    if type(record) is InvocationDecision:
        return f"invocation_decision:{record.decision_id}"
    if type(record) is CycleRecord:
        return f"cycle:{record.cycle_id}:{record.revision}"
    if type(record) is InterventionOutcome:
        return f"intervention_outcome:{record.outcome_id}"
    if type(record) is DeliveryRecord:
        return f"delivery:{record.delivery_id}:{record.revision}"
    raise ValueError


def _transitioned_cycle(source: CycleRecord, **updates: object) -> CycleRecord:
    values = source.model_dump(mode="python", warnings=False)
    values.update(updates)
    return CycleRecord.model_validate(values)


def _pending_delivery(
    cycle: CycleRecord,
    intervention: InterventionDecision,
    binding: EnqueueDelivery | None,
    *,
    updated_at: datetime,
) -> DeliveryRecord | None:
    if binding is None:
        return None
    target = intervention.delivery_target
    rendered_text = intervention.rendered_text
    if target is None or rendered_text is None:
        raise Stage2ExperimentError()
    rendered_text_digest = canonical_digest(rendered_text)
    return DeliveryRecord(
        delivery_id=derive_delivery_id(
            cycle.run_id,
            cycle.cycle_id,
            intervention.intervention_id,
            binding.target_request_id,
            target,
            binding.adapter_id,
            binding.adapter_capabilities_digest,
            rendered_text_digest,
        ),
        run_id=cycle.run_id,
        revision=1,
        cycle_id=cycle.cycle_id,
        intervention_id=intervention.intervention_id,
        rendered_text_digest=rendered_text_digest,
        target_request_id=binding.target_request_id,
        target=target,
        state=DeliveryState.PENDING,
        attempt_count=0,
        adapter_id=binding.adapter_id,
        adapter_deduplicates=binding.adapter_deduplicates,
        adapter_deduplication_guarantee=binding.adapter_deduplication_guarantee,
        adapter_supports_pre_action=binding.adapter_supports_pre_action,
        adapter_contract_version=binding.adapter_contract_version,
        adapter_capabilities_digest=binding.adapter_capabilities_digest,
        created_at=updated_at,
        updated_at=updated_at,
    )


class Stage2ExperimentRunResult(_ExperimentModel):
    """Closed, replay-safe evidence for one complete offline condition run."""

    schema_version: Literal["stage2-experiment-run-result/v2"] = (
        STAGE2_EXPERIMENT_RUN_RESULT_SCHEMA_VERSION
    )
    trajectory: Stage2Trajectory = Field(repr=False)
    condition: ResolvedStage2Condition
    policy_version: Literal["stage2-fixed-step-comparison/v1"] = STAGE2_EXPERIMENT_POLICY_VERSION
    prompt_bundle: PromptBundleIdentity
    model_profile: TwoPhaseModelProfile
    call_policy: TwoPhaseCallPolicy
    cycle_reservation: BudgetAmounts
    budget_limits: BudgetLimits
    run_id: UUID
    trace_digest: Sha256Digest
    normalized_draft_digests: Annotated[
        tuple[Sha256Digest, ...],
        Field(min_length=1, max_length=_MAX_EVENTS),
    ]
    persisted_event_draft_digests: Annotated[
        tuple[Sha256Digest, ...],
        Field(min_length=1, max_length=_MAX_EVENTS),
    ]
    trajectory_prefix: AttestedTrajectoryPrefix = Field(repr=False)
    schedule: FixedStepSchedule
    windows: Annotated[
        tuple[MessageWindow, ...],
        Field(min_length=1, max_length=_MAX_EVENTS, repr=False),
    ]
    decisions: Annotated[
        tuple[InvocationDecision, ...],
        Field(min_length=1, max_length=_MAX_EVENTS),
    ]
    boundaries: Annotated[
        tuple[Stage2BoundaryEvidence, ...],
        Field(min_length=1, max_length=_MAX_EVENTS, repr=False),
    ]
    call_receipts: Annotated[
        tuple[CallReceipt, ...],
        Field(max_length=_MAX_CALLS, repr=False),
    ]
    response_fixture: Stage2ResponseFixtureIdentity | None = None
    metrics: Stage2ExperimentMetrics
    final_budget_snapshot: BudgetSnapshot
    final_memory_snapshot: MemorySnapshot = Field(repr=False)
    semantic_projection_digests: ProjectionDigests
    repository_projection_digests: ProjectionDigests
    ledger: Annotated[
        tuple[LedgerEntry, ...],
        Field(min_length=1, max_length=_MAX_EVENTS * 16, repr=False),
    ]
    ledger_entry_count: Annotated[int, Field(ge=1, le=_MAX_EVENTS * 16)]
    ledger_head: LedgerHead
    rebuild_equivalent: Literal[True]
    result_digest: Sha256Digest = Field(default_factory=_result_digest)

    @model_validator(mode="after")
    def authoritative_sources_rebuild_the_result(self) -> Self:
        try:
            trajectory = Stage2Trajectory.model_validate_json(
                self.trajectory.model_dump_json(warnings=False)
            )
            condition = ResolvedStage2Condition.model_validate_json(
                self.condition.model_dump_json(warnings=False)
            )
            expected_condition = resolve_stage2_condition(condition.condition_id)
            expected_bundle = _prompt_bundle(condition).identity
            expected_profile = _model_profile(condition)
            expected_policy = _call_policy()
            expected_reservation = _reservation(expected_policy)
            expected_limits = _budget_limits(expected_reservation, len(trajectory.inputs))
            normalized = tuple(canonical_digest(item.draft) for item in trajectory.inputs)
            expected_trace_digest = algorithm_trace_digest(normalized)

            integrity = IntegrityContext(key=None, synthetic_benchmark=True)
            projection = empty_projection(self.run_id)
            previous_chain_tag = None
            previous_projection_tag = None
            for position, entry in enumerate(self.ledger, start=1):
                record_key = _ledger_record_key(entry.record)
                record_tag = integrity.tag(
                    entry.record,
                    domain="saliencegate:ledger-record:v1",
                )
                chain_tag = integrity.tag(
                    {
                        "run_id": str(self.run_id),
                        "position": position,
                        "record_key": record_key,
                        "record_tag": record_tag,
                        "previous_chain_tag": previous_chain_tag,
                    },
                    domain="saliencegate:ledger-chain:v1",
                )
                if (
                    entry.run_id != self.run_id
                    or entry.position != position
                    or entry.record_key != record_key
                    or entry.record_tag != record_tag
                    or entry.previous_chain_tag != previous_chain_tag
                    or entry.chain_tag != chain_tag
                ):
                    raise ValueError
                projection = apply_entry(projection, entry)
                previous_projection_tag = integrity.tag(
                    {
                        "previous_projection_tag": previous_projection_tag,
                        "entry_chain_tag": entry.chain_tag,
                        "ledger_position": entry.position,
                        "ingestion_cursor": projection.ingestion_cursor,
                        "memory_cursor": projection.memory_cursor,
                        "counts": {
                            "events": len(projection.events_by_id),
                            "signals": len(projection.signals),
                            "decisions": len(projection.decisions),
                            "cycle_revisions": len(projection.cycle_history),
                            "memories": len(projection.memories),
                            "memory_revisions": len(projection.memory_history),
                            "interventions": len(projection.interventions),
                            "outcomes": len(projection.outcomes),
                            "delivery_revisions": len(projection.delivery_history),
                        },
                    },
                    domain="saliencegate:projection-checkpoint:v1",
                )
                previous_chain_tag = entry.chain_tag
            if previous_chain_tag is None or previous_projection_tag is None:
                raise ValueError
            expected_head = LedgerHead(
                run_id=self.run_id,
                entry_count=len(self.ledger),
                chain_tag=previous_chain_tag,
                projection_tag=previous_projection_tag,
                head_tag=integrity.tag(
                    {
                        "run_id": str(self.run_id),
                        "entry_count": len(self.ledger),
                        "chain_tag": previous_chain_tag,
                        "projection_tag": previous_projection_tag,
                    },
                    domain="saliencegate:ledger-head:v1",
                ),
            )

            event_entries = tuple(
                entry for entry in self.ledger if type(entry.record) is TraceEvent
            )
            if len(event_entries) != len(trajectory.inputs):
                raise ValueError
            expected_items: list[AttestedTrajectoryEvent] = []
            for item, entry in zip(trajectory.inputs, event_entries, strict=True):
                event = entry.record
                if type(event) is not TraceEvent:
                    raise ValueError
                binding = bind_persisted_trajectory_event(
                    entry,
                    task_description=item.task_description,
                    logical_messages=item.logical_messages,
                    action_step=item.action_step,
                )
                expected_items.append(AttestedTrajectoryEvent(event=event, binding=binding))
            exact_items = tuple(expected_items)
            expected_prefix = _prefix_at(
                run_id=self.run_id,
                items=exact_items,
                sequence=len(exact_items),
            )
            expected_schedule = _project_verified_fixed_step_schedule(expected_prefix)
            expected_boundary_sources: dict[
                int,
                tuple[FixedStepSchedule, MessageWindow, TraceEvent],
            ] = {}
            expected_windows_list: list[MessageWindow] = []
            for scheduled in expected_schedule.decisions:
                if not scheduled.invoke:
                    continue
                prefix = _prefix_at(
                    run_id=self.run_id,
                    items=exact_items,
                    sequence=scheduled.event_sequence,
                )
                schedule = _project_verified_fixed_step_schedule(prefix)
                window = _project_verified_message_window(prefix)
                event = exact_items[scheduled.event_sequence - 1].event
                expected_boundary_sources[scheduled.event_sequence] = (
                    schedule,
                    window,
                    event,
                )
                expected_windows_list.append(window)
            expected_windows = tuple(expected_windows_list)
            prefix_events = tuple(item.event for item in exact_items)
            persisted = tuple(_persisted_draft_digest(item) for item in prefix_events)
            exact_boundaries = tuple(
                Stage2BoundaryEvidence.model_validate_json(item.model_dump_json(warnings=False))
                for item in self.boundaries
            )
            expected_calls = tuple(
                call for boundary in exact_boundaries for call in boundary.call_receipts
            )
            if not expected_calls:
                if self.response_fixture is not None:
                    raise ValueError
            elif self.response_fixture is not None and (
                not two_phase_receipts_are_replay_native(expected_calls)
                or self.response_fixture.response_count != len(expected_calls)
                or self.response_fixture.fixture_digest
                != two_phase_replay_fixture_digest_from_receipts(
                    expected_calls,
                    replay_id=self.response_fixture.replay_id,
                )
            ):
                raise ValueError
            expected_metrics = _metrics(exact_boundaries, expected_calls)
            expected_repository_digests = projection_digests(
                projection,
                integrity,
                ledger_position=len(self.ledger),
            )
            expected_semantic_digests = semantic_projection_digests(
                self.run_id,
                self.ledger,
            )
            expected_budget = projected_budget_snapshot(projection)
            expected_memory = projected_memory_snapshot(
                projection,
                integrity,
                ledger_position=len(self.ledger),
            )
        except Exception:
            raise ValueError("offline experiment result sources failed exact validation") from None

        ledger_decisions = tuple(
            entry.record for entry in self.ledger if type(entry.record) is InvocationDecision
        )
        ledger_events = tuple(entry.record for entry in event_entries)
        active = condition.condition_id is not Stage2ConditionId.NO_MEMORY
        decisions_match = True
        for decision, scheduled, event in zip(
            self.decisions,
            expected_schedule.decisions,
            prefix_events,
            strict=True,
        ):
            invoke = scheduled.invoke and active
            reason = (
                ReasonCode.BOOTSTRAP
                if invoke and scheduled.reason is FixedStepReason.BOOTSTRAP
                else ReasonCode.SCRIPTED_INVOKE
                if invoke
                else ReasonCode.SCRIPTED_SILENCE
            )
            if (
                scheduled.event_id != event.event_id
                or decision.decision_id
                != algorithm_runtime_uuid(
                    self.trace_digest,
                    "stage2-decision",
                    event.sequence,
                )
                or decision.run_id != self.run_id
                or decision.event_sequence != event.sequence
                or decision.invoke is not invoke
                or decision.risk_score is not None
                or decision.reason_codes != (reason,)
                or decision.policy_version != self.policy_version
                or decision.configuration_digest != condition.condition_digest
                or decision.budget_snapshot.limits != self.budget_limits
                or decision.cooldown_active
                or decision.created_at != event.timestamp
            ):
                decisions_match = False
                break

        boundaries_match = True
        for boundary in exact_boundaries:
            sequence = boundary.boundary_event.sequence
            source = expected_boundary_sources.get(sequence)
            if source is None:
                boundaries_match = False
                break
            schedule, window, event = source
            decision = self.decisions[sequence - 1]
            trace_input = trajectory.inputs[sequence - 1]
            execution = (
                boundary.phase_one_result
                if boundary.phase_one_result is not None
                else boundary.two_phase_result
            )
            if (
                boundary.condition != condition
                or boundary.invocation_decision != decision
                or boundary.boundary_event != event
                or boundary.schedule != schedule
                or boundary.window != window
            ):
                boundaries_match = False
                break
            if boundary.cycle is None:
                if execution is not None or boundary.delivery_record is not None:
                    boundaries_match = False
                    break
                continue
            intervention = boundary.cycle.intervention
            if (
                boundary.request is None
                or execution is None
                or intervention is None
                or boundary.cycle.budget_reservation != self.cycle_reservation
                or boundary.request.cycle_receipt.cycle.budget_reservation != self.cycle_reservation
                or execution.prompt_bundle_digest != self.prompt_bundle.bundle_digest
                or execution.model_id != self.model_profile.model_id
                or execution.model_profile_digest != self.model_profile.profile_digest
                or execution.call_policy != self.call_policy
                or execution.call_policy_digest != self.call_policy.policy_digest
                or (
                    type(execution) is PhaseOneCycleResult
                    and execution.model_profile != self.model_profile
                )
                or boundary.cycle.budget_settlement
                != _usage_settlement(
                    execution.usage,
                    self.cycle_reservation,
                    interventions=int(intervention.action is InterventionAction.REMIND),
                )
            ):
                boundaries_match = False
                break
            delivery = boundary.delivery_record
            if intervention.action is InterventionAction.SILENCE:
                if delivery is not None:
                    boundaries_match = False
                    break
            else:
                capabilities = _OfflineDeliveryAdapter().capabilities()
                if delivery is None:
                    boundaries_match = False
                    break
                identifier_factory = _delivery_id_factory(
                    self.trace_digest,
                    delivery.delivery_id,
                )
                expected_claim_id = identifier_factory()
                expected_attempt_id = identifier_factory()
                if (
                    trace_input.target_request_id is None
                    or delivery.target_request_id != trace_input.target_request_id
                    or delivery.adapter_id != _DELIVERY_ADAPTER_ID
                    or delivery.adapter_deduplicates is not capabilities.deduplicates_delivery_id
                    or delivery.adapter_deduplication_guarantee
                    is not capabilities.deduplication_guarantee
                    or delivery.adapter_supports_pre_action
                    is not capabilities.pre_action_interception
                    or delivery.adapter_contract_version != ADAPTER_CONTRACT_VERSION
                    or delivery.adapter_capabilities_digest
                    != adapter_capabilities_digest(capabilities)
                    or delivery.state is not DeliveryState.DELIVERED
                    or delivery.attempt_count != 1
                    or delivery.claim_id != expected_claim_id
                    or delivery.attempt_id != expected_attempt_id
                    or delivery.receipt != {"provider_receipt_id": "stage2-offline-1"}
                    or delivery.reason_code is not ReasonCode.DELIVERY_SUCCEEDED
                    or delivery.created_at != event.timestamp
                    or delivery.updated_at != event.timestamp
                ):
                    boundaries_match = False
                    break

        latest_cycles: dict[str, CycleRecord] = {}
        latest_deliveries: dict[UUID, DeliveryRecord] = {}
        ledger_outcomes: list[InterventionOutcome] = []
        for entry in self.ledger:
            if type(entry.record) is CycleRecord:
                latest_cycles[entry.record.cycle_id] = entry.record
            elif type(entry.record) is DeliveryRecord:
                latest_deliveries[entry.record.delivery_id] = entry.record
            elif type(entry.record) is InterventionOutcome:
                ledger_outcomes.append(entry.record)
        boundary_cycles = tuple(
            boundary.cycle for boundary in exact_boundaries if boundary.cycle is not None
        )
        projected_cycles = tuple(
            sorted(
                latest_cycles.values(),
                key=lambda item: (item.last_event_sequence, item.cycle_id),
            )
        )
        boundary_deliveries = {
            boundary.delivery_record.cycle_id: boundary.delivery_record
            for boundary in exact_boundaries
            if boundary.delivery_record is not None
        }
        projected_deliveries = {
            delivery.cycle_id: delivery for delivery in latest_deliveries.values()
        }
        interventions = tuple(
            cycle.intervention for cycle in projected_cycles if cycle.intervention is not None
        )
        expected_outcomes = tuple(
            InterventionOutcome(
                outcome_id=algorithm_runtime_uuid(
                    self.trace_digest,
                    "stage2-outcome",
                    intervention.intervention_id,
                ),
                run_id=intervention.run_id,
                intervention_id=intervention.intervention_id,
                repeated_error_status=RepeatedErrorStatus.UNKNOWN,
                constraint_status=ConstraintStatus.UNKNOWN,
                evidence_mode=OutcomeEvidenceMode.POLICY_REPLAY,
                created_at=intervention.created_at,
            )
            for intervention in interventions
        )
        expected_boundary_sequences = tuple(
            item.event_sequence for item in expected_schedule.decisions if item.invoke
        )
        if (
            trajectory != self.trajectory
            or condition != self.condition
            or expected_condition != condition
            or self.prompt_bundle != expected_bundle
            or self.model_profile != expected_profile
            or self.call_policy != expected_policy
            or self.cycle_reservation != expected_reservation
            or self.budget_limits != expected_limits
            or self.run_id != trajectory.run_id
            or self.trace_digest != expected_trace_digest
            or self.normalized_draft_digests != normalized
            or self.persisted_event_draft_digests != persisted
            or persisted != normalized
            or tuple(item.expected_event_id for item in trajectory.inputs)
            != tuple(item.event_id for item in prefix_events)
            or self.trajectory_prefix != expected_prefix
            or self.schedule != expected_schedule
            or self.windows != expected_windows
            or len(exact_boundaries) != expected_schedule.invocation_count
            or tuple(item.boundary_event.sequence for item in exact_boundaries)
            != expected_boundary_sequences
            or tuple(item.window for item in exact_boundaries) != expected_windows
            or len(self.decisions) != len(trajectory.inputs)
            or not decisions_match
            or ledger_decisions != self.decisions
            or ledger_events != prefix_events
            or self.boundaries != exact_boundaries
            or not boundaries_match
            or self.call_receipts != expected_calls
            or self.metrics != expected_metrics
            or self.final_budget_snapshot != expected_budget
            or self.final_memory_snapshot != expected_memory
            or self.semantic_projection_digests != expected_semantic_digests
            or self.repository_projection_digests != expected_repository_digests
            or self.semantic_projection_digests != self.repository_projection_digests
            or self.ledger_entry_count != len(self.ledger)
            or self.ledger_head != expected_head
            or boundary_cycles != projected_cycles
            or any(cycle.state is not CycleState.COMMITTED for cycle in projected_cycles)
            or boundary_deliveries != projected_deliveries
            or len(projected_deliveries) != len(latest_deliveries)
            or tuple(ledger_outcomes) != expected_outcomes
        ):
            raise ValueError("offline experiment result sources do not reconcile")

        values = self.model_dump(mode="json", exclude={"result_digest"}, warnings=False)
        if self.result_digest != _result_digest(values):
            raise ValueError("offline experiment result digest does not match")
        return self


class Stage2ExperimentRunner:
    """Run one closed condition through a shared offline fixed-step path."""

    __slots__ = (
        "_call_policy",
        "_client",
        "_condition",
        "_delivery_adapter",
        "_executor",
        "_governor",
        "_grounding",
        "_model_profile",
        "_prompt_bundle",
        "_repository",
        "_reservation",
    )

    def __init__(
        self,
        *,
        repository: RunRepository,
        condition: Stage2ConditionId | str,
        client: StructuredCallClient | None,
    ) -> None:
        try:
            resolved = resolve_stage2_condition(condition)
            active = resolved.condition_id is not Stage2ConditionId.NO_MEMORY
            if active != (client is not None):
                raise ValueError
            if client is not None and not isinstance(client, StructuredCallClient):
                raise TypeError
            configuration = GroundingConfig.model_validate_json(
                canonical_json(resolved.shared_controls.grounding.configuration)
            )
            grounding = GroundingPipeline(configuration)
            if grounding.resolved_configuration != resolved.shared_controls.grounding:
                raise ValueError
            prompt_bundle = _prompt_bundle(resolved)
            model_profile = _model_profile(resolved)
            call_policy = _call_policy()
            reservation = _reservation(call_policy)
            executor = (
                None
                if client is None
                else PaperTwoPhaseCycleExecutor(
                    materializer=RepositoryOperationMaterializer(repository),
                    client=client,
                    prompt_bundle=prompt_bundle,
                    grounding_pipeline=grounding,
                    model_profile=model_profile,
                    call_policy=call_policy,
                )
            )
        except Exception:
            raise Stage2ExperimentError() from None

        self._repository = repository
        self._condition = resolved
        self._client = client
        self._grounding = grounding
        self._prompt_bundle = prompt_bundle
        self._model_profile = model_profile
        self._call_policy = call_policy
        self._reservation = reservation
        self._executor = executor
        self._governor = BudgetGovernor()
        self._delivery_adapter: DeliveryAdapter = _OfflineDeliveryAdapter()

    async def _budget(
        self,
        run_id: UUID,
        *,
        limits: BudgetLimits,
        first_decision: bool,
    ) -> BudgetSnapshot:
        if first_decision:
            return BudgetSnapshot(
                limits=limits,
                reserved=BudgetAmounts(),
                consumed=BudgetAmounts(),
            )
        snapshot = await self._repository.budget_snapshot(run_id)
        if snapshot.limits != limits:
            raise Stage2ExperimentError()
        return snapshot

    async def _grounding_state(
        self,
        run_id: UUID,
        *,
        current_sequence: int,
        memories: tuple[MemoryRecord, ...],
    ) -> GroundingState:
        ledger = await self._repository.ledger(run_id)
        events = tuple(
            entry.record
            for entry in ledger
            if type(entry.record) is TraceEvent and entry.record.sequence <= current_sequence
        )
        latest_cycles: dict[str, CycleRecord] = {}
        for entry in ledger:
            if type(entry.record) is CycleRecord:
                latest_cycles[entry.record.cycle_id] = entry.record
        config = self._grounding.configuration
        history_window = max(config.duplicate_window_events, config.cooldown_events)
        first_sequence = max(1, current_sequence - history_window)
        history: list[ReminderHistory] = []
        for cycle in sorted(
            latest_cycles.values(),
            key=lambda value: (value.last_event_sequence, value.cycle_id),
        ):
            intervention = cycle.intervention
            if (
                cycle.state is CycleState.COMMITTED
                and intervention is not None
                and intervention.action is InterventionAction.REMIND
                and first_sequence <= cycle.last_event_sequence < current_sequence
            ):
                history.append(
                    ReminderHistory(
                        schema_version="1.0",
                        intervention_id=intervention.intervention_id,
                        run_id=run_id,
                        event_sequence=cycle.last_event_sequence,
                        claim_digests=tuple(
                            claim_fingerprint(claim) for claim in intervention.claims
                        ),
                    )
                )
        return GroundingState(
            schema_version="1.0",
            events=events,
            memories=memories,
            reminder_history=tuple(history),
        )

    async def _complete_reconciled_transition(
        self,
        operation: Coroutine[Any, Any, CycleReceipt],
        reconcile: Callable[[], Coroutine[Any, Any, CycleReceipt | None]],
    ) -> tuple[CycleReceipt, bool]:
        try:
            return await _complete_boundary(operation)
        except asyncio.CancelledError:
            receipt, _ = await _complete_boundary(reconcile())
            if receipt is None:
                raise
            return receipt, True
        except Exception:
            receipt, reconcile_cancelled = await _complete_boundary(reconcile())
            if receipt is None:
                if reconcile_cancelled:
                    raise asyncio.CancelledError() from None
                raise
            return receipt, reconcile_cancelled

    async def _reconcile_cycle_transition(
        self,
        *,
        expected: CycleRecord,
        previous: CycleReceipt | None,
        expected_budget: BudgetSnapshot,
        expected_delivery: DeliveryRecord | None = None,
        decision: InvocationDecision | None = None,
    ) -> CycleReceipt | None:
        try:
            ledger = await self._repository.ledger(expected.run_id)
            if decision is not None:
                decision_records = tuple(
                    record
                    for record in (entry.record for entry in ledger)
                    if type(record) is InvocationDecision
                    and record.decision_id == decision.decision_id
                )
                related_cycles = tuple(
                    record
                    for record in (entry.record for entry in ledger)
                    if type(record) is CycleRecord
                    and record.invocation_decision_id == decision.decision_id
                )
                if decision_records != (decision,) or any(
                    cycle.cycle_id != expected.cycle_id for cycle in related_cycles
                ):
                    raise ValueError

            cycle_entries = tuple(
                entry
                for entry in ledger
                if type(entry.record) is CycleRecord and entry.record.cycle_id == expected.cycle_id
            )
            if previous is not None:
                source_index = previous.ledger_position - 1
                if not 0 <= source_index < len(ledger):
                    raise ValueError
                source_entry = ledger[source_index]
                if (
                    source_entry.record != previous.cycle
                    or source_entry.record_tag != previous.record_tag
                    or source_entry.chain_tag != previous.chain_tag
                ):
                    raise ValueError
            if not cycle_entries:
                if previous is None:
                    return None
                raise ValueError

            latest_entry = cycle_entries[-1]
            latest = cast(CycleRecord, latest_entry.record)
            if previous is not None and latest == previous.cycle:
                return None
            if latest != expected or tuple(
                cast(CycleRecord, entry.record).revision for entry in cycle_entries
            ) != tuple(range(1, expected.revision + 1)):
                raise ValueError

            budget = await self._repository.budget_snapshot(expected.run_id)
            if budget != expected_budget:
                raise ValueError
            deliveries = tuple(
                record
                for record in (entry.record for entry in ledger)
                if type(record) is DeliveryRecord and record.cycle_id == expected.cycle_id
            )
            expected_deliveries = () if expected_delivery is None else (expected_delivery,)
            if deliveries != expected_deliveries:
                raise ValueError
            return CycleReceipt(
                appended=False,
                cycle=expected,
                record_tag=latest_entry.record_tag,
                ledger_position=latest_entry.position,
                chain_tag=latest_entry.chain_tag,
                budget_snapshot=budget,
                delivery=expected_delivery,
            )
        except Stage2ExperimentError:
            raise
        except Exception:
            raise Stage2ExperimentError() from None

    async def _reconcile_begin(
        self,
        *,
        decision: InvocationDecision,
        grounding: GroundingPin,
        created_at: datetime,
    ) -> CycleReceipt | None:
        try:
            ledger = await self._repository.ledger(decision.run_id)
            beginnings = tuple(
                record
                for record in (entry.record for entry in ledger)
                if type(record) is CycleRecord
                and record.invocation_decision_id == decision.decision_id
            )
            if not beginnings:
                return None
            snapshot = await self._repository.snapshot(decision.run_id)
            first_event_sequence = snapshot.memory_cursor + 1
            expected = CycleRecord(
                cycle_id=derive_cycle_id(
                    decision.run_id,
                    first_event_sequence,
                    decision.event_sequence,
                    decision.policy_version,
                    decision.configuration_digest,
                    grounding.grounding_version,
                    grounding.grounding_configuration_digest,
                    grounding.requested_delivery_target,
                ),
                run_id=decision.run_id,
                revision=1,
                invocation_decision_id=decision.decision_id,
                policy_version=decision.policy_version,
                configuration_digest=decision.configuration_digest,
                grounding_version=grounding.grounding_version,
                grounding_configuration=grounding.grounding_configuration,
                grounding_configuration_digest=grounding.grounding_configuration_digest,
                requested_delivery_target=grounding.requested_delivery_target,
                first_event_sequence=first_event_sequence,
                last_event_sequence=decision.event_sequence,
                state=CycleState.PENDING,
                created_at=created_at,
                updated_at=created_at,
            )
            return await self._reconcile_cycle_transition(
                expected=expected,
                previous=None,
                expected_budget=decision.budget_snapshot,
                decision=decision,
            )
        except Stage2ExperimentError:
            raise
        except Exception:
            raise Stage2ExperimentError() from None

    async def _terminalize_unknown(
        self,
        coordinator: CycleCoordinator,
        running: _RunningCycle,
        *,
        updated_at: datetime,
    ) -> None:
        receipt = running.receipt
        if receipt is None:
            return
        try:
            state = receipt.cycle.state
            if state is CycleState.PENDING:
                reason = ReasonCode.MODEL_ERROR
                settlement: BudgetAmounts | None = None
                expected_budget = receipt.budget_snapshot
                operation = coordinator.fail(
                    receipt,
                    reason=reason,
                    updated_at=updated_at,
                )
            elif state is CycleState.RESERVED:
                if receipt.cycle.budget_reservation != self._reservation:
                    raise ValueError
                reason = ReasonCode.MODEL_ERROR
                settlement = BudgetAmounts()
                expected_budget = self._governor.settle(
                    receipt.budget_snapshot,
                    self._reservation,
                    settlement,
                    model_call_latencies_us=(),
                )
                operation = coordinator.fail(
                    receipt,
                    reason=reason,
                    settlement=settlement,
                    updated_at=updated_at,
                )
            elif state is CycleState.RUNNING:
                if receipt.cycle.budget_reservation != self._reservation:
                    raise ValueError
                reason = ReasonCode.FAILED_UNKNOWN_COST
                settlement = self._reservation
                expected_budget = self._governor.consume_unknown(
                    receipt.budget_snapshot,
                    self._reservation,
                )
                operation = coordinator.fail(
                    receipt,
                    reason=reason,
                    settlement=settlement,
                    updated_at=updated_at,
                )
            else:
                raise ValueError
            expected = _transitioned_cycle(
                receipt.cycle,
                revision=receipt.cycle.revision + 1,
                state=CycleState.FAILED,
                budget_settlement=settlement,
                failure_reason=reason,
                updated_at=updated_at,
            )
            _, cancelled = await self._complete_reconciled_transition(
                operation,
                lambda: self._reconcile_cycle_transition(
                    expected=expected,
                    previous=receipt,
                    expected_budget=expected_budget,
                ),
            )
            running.receipt = None
            if cancelled:
                raise asyncio.CancelledError()
        except asyncio.CancelledError:
            raise
        except Exception:
            raise Stage2ExperimentError() from None

    async def _terminalize_known(
        self,
        coordinator: CycleCoordinator,
        running: _RunningCycle,
        failure: TwoPhaseCycleFailure,
        request: TwoPhaseCycleRequest,
        *,
        updated_at: datetime,
    ) -> None:
        receipt = running.receipt
        if receipt is None or receipt.cycle.state is not CycleState.RUNNING:
            raise Stage2ExperimentError()
        try:
            checked = TwoPhaseCycleFailure.model_validate_json(
                failure.model_dump_json(warnings=False)
            )
            if (
                checked.run_id != request.cycle_receipt.cycle.run_id
                or checked.cycle_id != request.cycle_receipt.cycle.cycle_id
                or checked.request_digest != request.request_digest
                or checked.window_digest != request.window.window_digest
                or checked.prompt_bundle_digest != self._prompt_bundle.identity.bundle_digest
                or checked.model_id != self._model_profile.model_id
                or checked.model_profile_digest != self._model_profile.profile_digest
                or checked.call_policy_digest != self._call_policy.policy_digest
                or checked.call_policy != self._call_policy
            ):
                raise ValueError
            await self._settle_known_failure(
                coordinator,
                running,
                usage=checked.usage,
                calls=checked.call_receipts,
                reason=_failure_reason(checked.reason),
                updated_at=updated_at,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            raise Stage2ExperimentError() from None

    async def _settle_known_failure(
        self,
        coordinator: CycleCoordinator,
        running: _RunningCycle,
        *,
        usage: TwoPhaseUsage,
        calls: tuple[CallReceipt, ...],
        reason: ReasonCode,
        updated_at: datetime,
    ) -> None:
        receipt = running.receipt
        if receipt is None or receipt.cycle.state is not CycleState.RUNNING or not calls:
            raise Stage2ExperimentError()
        settlement = _usage_settlement(
            usage,
            self._reservation,
            interventions=0,
        )
        latencies = tuple(item.usage.latency_us for item in calls)
        held = await self._repository.budget_snapshot(receipt.cycle.run_id)
        if held != receipt.budget_snapshot:
            raise Stage2ExperimentError()
        expected_budget = self._governor.settle(
            held,
            self._reservation,
            settlement,
            model_call_latencies_us=latencies,
        )
        call_digests = tuple(item.call_digest for item in calls)
        expected = _transitioned_cycle(
            receipt.cycle,
            revision=receipt.cycle.revision + 1,
            state=CycleState.FAILED,
            budget_settlement=settlement,
            model_call_digests=call_digests,
            model_call_latencies_us=latencies,
            failure_reason=reason,
            updated_at=updated_at,
        )
        _, cancelled = await self._complete_reconciled_transition(
            coordinator.fail(
                receipt,
                reason=reason,
                settlement=settlement,
                model_call_digests=call_digests,
                model_call_latencies_us=latencies,
                updated_at=updated_at,
            ),
            lambda: self._reconcile_cycle_transition(
                expected=expected,
                previous=receipt,
                expected_budget=expected_budget,
            ),
        )
        running.receipt = None
        if cancelled:
            raise asyncio.CancelledError()

    @staticmethod
    def _outcome(
        *,
        trace_digest: str,
        intervention: InterventionDecision,
    ) -> InterventionOutcome:
        return InterventionOutcome(
            outcome_id=algorithm_runtime_uuid(
                trace_digest,
                "stage2-outcome",
                intervention.intervention_id,
            ),
            run_id=intervention.run_id,
            intervention_id=intervention.intervention_id,
            repeated_error_status=RepeatedErrorStatus.UNKNOWN,
            constraint_status=ConstraintStatus.UNKNOWN,
            evidence_mode=OutcomeEvidenceMode.POLICY_REPLAY,
            created_at=intervention.created_at,
        )

    async def _outcome_is_durable(self, outcome: InterventionOutcome) -> bool:
        ledger = await self._repository.ledger(outcome.run_id)
        matches = tuple(
            record
            for record in (entry.record for entry in ledger)
            if type(record) is InterventionOutcome and record.outcome_id == outcome.outcome_id
        )
        if len(matches) > 1:
            raise Stage2ExperimentError()
        return matches == (outcome,)

    async def _record_outcome_idempotently(self, outcome: InterventionOutcome) -> None:
        try:
            receipt = await self._repository.record_outcome(outcome)
        except asyncio.CancelledError:
            if not await self._outcome_is_durable(outcome):
                raise
            raise
        except Exception:
            if not await self._outcome_is_durable(outcome):
                raise Stage2ExperimentError() from None
            return
        if not receipt.appended and not await self._outcome_is_durable(outcome):
            raise Stage2ExperimentError()

    async def _finalize_committed_cycle(
        self,
        *,
        trace_digest: str,
        committed: CycleReceipt,
        intervention: InterventionDecision,
        timestamp: datetime,
    ) -> DeliveryRecord | None:
        final_delivery: DeliveryRecord | None = None
        delivery_succeeded = True
        delivery_failure: BaseException | None = None
        try:
            if committed.delivery is not None:
                worker = DeliveryWorker(
                    repository=self._repository,
                    adapter=self._delivery_adapter,
                    id_factory=_delivery_id_factory(
                        trace_digest,
                        committed.delivery.delivery_id,
                    ),
                )
                delivered = await worker.deliver(
                    intervention.run_id,
                    committed.delivery.delivery_id,
                    now=timestamp,
                )
                delivery_succeeded = delivered.delivered
                final_delivery = delivered.delivery
        except BaseException as error:
            delivery_failure = error
        await self._record_outcome_idempotently(
            self._outcome(
                trace_digest=trace_digest,
                intervention=intervention,
            )
        )
        if delivery_failure is not None:
            raise delivery_failure
        if not delivery_succeeded:
            raise Stage2ExperimentError()
        return final_delivery

    async def _execute_cycle(
        self,
        *,
        boundary: FixedStepTraceBoundary,
        decision: InvocationDecision,
    ) -> _CycleExecution:
        executor = self._executor
        window = boundary.window
        if executor is None or window is None:
            raise Stage2ExperimentError()
        timestamp = decision.created_at.astimezone(UTC)
        coordinator = CycleCoordinator(self._repository)
        running_guard = _RunningCycle()
        try:
            grounding_pin = self._grounding.pin(
                self._condition.shared_controls.requested_delivery_target
            )
            pending, cancelled = await self._complete_reconciled_transition(
                coordinator.begin(
                    decision,
                    grounding=grounding_pin,
                    created_at=timestamp,
                ),
                lambda: self._reconcile_begin(
                    decision=decision,
                    grounding=grounding_pin,
                    created_at=timestamp,
                ),
            )
            running_guard.receipt = pending
            if cancelled:
                raise asyncio.CancelledError
            expected_reserved_budget = self._governor.reserve(
                pending.budget_snapshot,
                self._reservation,
            )
            expected_reserved = _transitioned_cycle(
                pending.cycle,
                revision=pending.cycle.revision + 1,
                state=CycleState.RESERVED,
                budget_reservation=self._reservation,
                updated_at=timestamp,
            )
            reserved, cancelled = await self._complete_reconciled_transition(
                coordinator.reserve(
                    pending,
                    reservation=self._reservation,
                    updated_at=timestamp,
                ),
                lambda: self._reconcile_cycle_transition(
                    expected=expected_reserved,
                    previous=pending,
                    expected_budget=expected_reserved_budget,
                ),
            )
            running_guard.receipt = reserved
            if cancelled:
                raise asyncio.CancelledError
            expected_running = _transitioned_cycle(
                reserved.cycle,
                revision=reserved.cycle.revision + 1,
                state=CycleState.RUNNING,
                batch_digest=window.window_digest,
                updated_at=timestamp,
            )
            running, cancelled = await self._complete_reconciled_transition(
                coordinator.start(
                    reserved,
                    batch_digest=window.window_digest,
                    updated_at=timestamp,
                ),
                lambda: self._reconcile_cycle_transition(
                    expected=expected_running,
                    previous=reserved,
                    expected_budget=reserved.budget_snapshot,
                ),
            )
            running_guard.receipt = running
            if cancelled:
                raise asyncio.CancelledError

            snapshot = await self._repository.snapshot(decision.run_id)
            cycle = running.cycle
            if (
                snapshot.ledger_position != running.ledger_position
                or snapshot.ingestion_cursor != cycle.last_event_sequence
                or snapshot.memory_cursor != cycle.first_event_sequence - 1
            ):
                raise ValueError
            records = tuple(
                sorted(
                    (
                        record
                        for record in snapshot.records
                        if record.validity is ValidityState.ACTIVE
                        and (record.expires_at is None or record.expires_at > timestamp)
                    ),
                    key=lambda record: (record.kind.value, str(record.memory_id)),
                )
            )
            current_bank = build_active_bank_prompt_view(
                kind=BankViewKind.CURRENT,
                run_id=decision.run_id,
                as_of=timestamp,
                source_projection_digest=snapshot.projection_digest,
                records=records,
            )
            grounding_state = await self._grounding_state(
                decision.run_id,
                current_sequence=cycle.last_event_sequence,
                memories=records,
            )
            request = TwoPhaseCycleRequest(
                schema_version="two-phase-cycle-request/v1",
                cycle_receipt=running,
                window=window,
                current_bank=current_bank,
                grounding_state=grounding_state,
                delta_id=algorithm_runtime_uuid(
                    boundary.trace_digest,
                    "stage2-delta",
                    cycle.cycle_id,
                ),
                assigned_memory_ids=tuple(
                    algorithm_runtime_uuid(
                        boundary.trace_digest,
                        "stage2-memory",
                        cycle.cycle_id,
                        ordinal,
                    )
                    for ordinal in range(1, MAX_MEMORY_DELTA_ITEMS + 1)
                ),
                intervention_id=algorithm_runtime_uuid(
                    boundary.trace_digest,
                    "stage2-intervention",
                    cycle.cycle_id,
                ),
                created_at=timestamp,
            )

            two_phase_result: TwoPhaseCycleResult | None = None
            phase_one_result: PhaseOneCycleResult | None = None
            retrieval_request: RetrievalRequest | None = None
            retrieval_result: RetrievalResult | None = None
            selector_provenance: DeterministicSelectorProvenance | None = None
            if self._condition.condition_id is Stage2ConditionId.RETRIEVAL_ALWAYS:
                raw_phase_one = await executor.execute_phase_one(request)
                if type(raw_phase_one) is TwoPhaseCycleFailure:
                    await self._terminalize_known(
                        coordinator,
                        running_guard,
                        raw_phase_one,
                        request,
                        updated_at=timestamp,
                    )
                    raise Stage2ExperimentError()
                if type(raw_phase_one) is not PhaseOneCycleResult:
                    raise ValueError
                phase_one_result = PhaseOneCycleResult.model_validate_json(
                    raw_phase_one.model_dump_json(warnings=False)
                )
                retrieval_request = build_retrieval_request(
                    condition=self._condition,
                    window=window,
                    materialization=phase_one_result.materialization,
                )
                retrieval_result = retrieve_candidate_bank(retrieval_request)
                provenance = retrieval_selector_provenance(
                    retrieval_request,
                    retrieval_result,
                )
                selector_provenance = provenance
                candidate_state = GroundingState(
                    schema_version="1.0",
                    events=request.grounding_state.events,
                    memories=phase_one_result.materialization.active_bank,
                    reminder_history=request.grounding_state.reminder_history,
                )
                context = GroundingContext(
                    schema_version="2.0",
                    intervention_id=request.intervention_id,
                    run_id=decision.run_id,
                    cycle_id=cycle.cycle_id,
                    current_event_sequence=cycle.last_event_sequence,
                    created_at=timestamp,
                    requested_delivery_target=(
                        self._condition.shared_controls.requested_delivery_target
                    ),
                    selector_provenance=provenance,
                )
                intervention = self._grounding.ground(
                    retrieval_result.selection.to_grounding_proposal(),
                    context=context,
                    state=candidate_state,
                )
                verify_grounded_intervention(
                    intervention,
                    context=context,
                    state=candidate_state,
                    expected_configuration=self._grounding.resolved_configuration,
                )
                execution_usage = phase_one_result.usage
                validated_delta = phase_one_result.validated_delta
                assignments = phase_one_result.memory_id_assignments
                calls = phase_one_result.call_receipts
            else:
                raw_two_phase = await executor.execute(request)
                if type(raw_two_phase) is TwoPhaseCycleFailure:
                    await self._terminalize_known(
                        coordinator,
                        running_guard,
                        raw_two_phase,
                        request,
                        updated_at=timestamp,
                    )
                    raise Stage2ExperimentError()
                if type(raw_two_phase) is not TwoPhaseCycleResult:
                    raise ValueError
                two_phase_result = TwoPhaseCycleResult.model_validate_json(
                    raw_two_phase.model_dump_json(warnings=False)
                )
                intervention = two_phase_result.intervention
                execution_usage = two_phase_result.usage
                validated_delta = two_phase_result.validated_delta
                assignments = two_phase_result.memory_id_assignments
                calls = two_phase_result.call_receipts

            intervention_count = int(intervention.action is InterventionAction.REMIND)
            target_request_id = boundary.trace_input.target_request_id
            if intervention_count and target_request_id is None:
                await self._settle_known_failure(
                    coordinator,
                    running_guard,
                    usage=execution_usage,
                    calls=calls,
                    reason=ReasonCode.TARGET_UNAVAILABLE,
                    updated_at=timestamp,
                )
                raise Stage2ExperimentError()
            settlement = _usage_settlement(
                execution_usage,
                self._reservation,
                interventions=intervention_count,
            )
            held = await self._repository.budget_snapshot(decision.run_id)
            if held != running.budget_snapshot:
                raise ValueError
            expected_committed_budget = self._governor.settle(
                held,
                self._reservation,
                settlement,
                model_call_latencies_us=tuple(item.usage.latency_us for item in calls),
            )

            delivery_binding = None
            if intervention_count:
                assert target_request_id is not None
                delivery_binding = enqueue_delivery_binding(
                    target_request_id=target_request_id,
                    capabilities=self._delivery_adapter.capabilities(),
                )
            selector_provenance_payload = (
                None
                if selector_provenance is None
                else selector_provenance.model_dump(mode="json", warnings=False)
            )
            call_digests = tuple(item.call_digest for item in calls)
            call_latencies = tuple(item.usage.latency_us for item in calls)
            expected_committed = _transitioned_cycle(
                running.cycle,
                revision=running.cycle.revision + 1,
                state=CycleState.COMMITTED,
                budget_settlement=settlement,
                model_call_digests=call_digests,
                model_call_latencies_us=call_latencies,
                validated_delta=validated_delta,
                memory_id_assignments=assignments,
                intervention=intervention,
                selector_provenance=selector_provenance_payload,
                updated_at=timestamp,
            )
            expected_delivery = _pending_delivery(
                expected_committed,
                intervention,
                delivery_binding,
                updated_at=timestamp,
            )
            committed, commit_cancelled = await self._complete_reconciled_transition(
                coordinator.commit(
                    running,
                    settlement=settlement,
                    validated_delta=validated_delta,
                    memory_id_assignments=assignments,
                    intervention=intervention,
                    selector_provenance=selector_provenance_payload,
                    delivery=delivery_binding,
                    updated_at=timestamp,
                    model_call_digests=call_digests,
                    model_call_latencies_us=call_latencies,
                ),
                lambda: self._reconcile_cycle_transition(
                    expected=expected_committed,
                    previous=running,
                    expected_budget=expected_committed_budget,
                    expected_delivery=expected_delivery,
                ),
            )
            running_guard.receipt = None
            final_delivery, finalization_cancelled = await _complete_boundary(
                self._finalize_committed_cycle(
                    trace_digest=boundary.trace_digest,
                    committed=committed,
                    intervention=intervention,
                    timestamp=timestamp,
                )
            )
            if commit_cancelled or finalization_cancelled:
                raise asyncio.CancelledError
            return _CycleExecution(
                cycle=committed.cycle,
                request=request,
                two_phase_result=two_phase_result,
                phase_one_result=phase_one_result,
                calls=calls,
                retrieval_request=retrieval_request,
                retrieval_result=retrieval_result,
                delivery=final_delivery,
            )
        except asyncio.CancelledError:
            await self._terminalize_unknown(
                coordinator,
                running_guard,
                updated_at=timestamp,
            )
            raise
        except Exception:
            await self._terminalize_unknown(
                coordinator,
                running_guard,
                updated_at=timestamp,
            )
            raise Stage2ExperimentError() from None

    async def run(self, trajectory: Stage2Trajectory) -> Stage2ExperimentRunResult:
        try:
            checked_trajectory = Stage2Trajectory.model_validate_json(
                trajectory.model_dump_json(warnings=False)
            )
            if checked_trajectory != trajectory:
                raise ValueError
        except Exception:
            raise Stage2ExperimentError() from None
        limits = _budget_limits(self._reservation, len(checked_trajectory.inputs))
        active = self._condition.condition_id is not Stage2ConditionId.NO_MEMORY

        async def on_boundary(boundary: FixedStepTraceBoundary) -> _BoundaryProjection:
            budget = await self._budget(
                boundary.event.run_id,
                limits=limits,
                first_decision=boundary.ordinal == 1,
            )
            if boundary.scheduled.invoke and active:
                try:
                    self._governor.reserve(budget, self._reservation)
                except Exception:
                    raise Stage2ExperimentError() from None
            invoke = boundary.scheduled.invoke and active
            reason = (
                ReasonCode.BOOTSTRAP
                if invoke and boundary.scheduled.reason is FixedStepReason.BOOTSTRAP
                else ReasonCode.SCRIPTED_INVOKE
                if invoke
                else ReasonCode.SCRIPTED_SILENCE
            )
            decision = InvocationDecision(
                decision_id=algorithm_runtime_uuid(
                    boundary.trace_digest,
                    "stage2-decision",
                    boundary.event.sequence,
                ),
                run_id=boundary.event.run_id,
                event_sequence=boundary.event.sequence,
                invoke=invoke,
                risk_score=None,
                reason_codes=(reason,),
                policy_version=STAGE2_EXPERIMENT_POLICY_VERSION,
                configuration_digest=self._condition.condition_digest,
                budget_snapshot=budget,
                cooldown_active=False,
                created_at=boundary.event.timestamp,
            )
            _, decision_cancelled = await record_reconciled_invocation_decision(
                self._repository,
                decision,
            )

            async def finish_boundary() -> _BoundaryProjection:
                if not boundary.scheduled.invoke:
                    return _BoundaryProjection(decision=decision, evidence=None)
                if boundary.window is None:
                    raise Stage2ExperimentError()
                execution = (
                    await self._execute_cycle(boundary=boundary, decision=decision)
                    if active
                    else None
                )
                observation = derive_stage2_condition_observation(
                    condition=self._condition,
                    schedule=boundary.schedule,
                    invocation_decision=decision,
                    boundary_event=boundary.event,
                    window=boundary.window,
                    cycle=None if execution is None else execution.cycle,
                    request=None if execution is None else execution.request,
                    two_phase_result=(None if execution is None else execution.two_phase_result),
                    phase_one_result=(None if execution is None else execution.phase_one_result),
                    call_receipts=() if execution is None else execution.calls,
                    retrieval_request=(None if execution is None else execution.retrieval_request),
                    retrieval_result=(None if execution is None else execution.retrieval_result),
                    delivery_record=None if execution is None else execution.delivery,
                )
                evidence = Stage2BoundaryEvidence(
                    condition=self._condition,
                    schedule=boundary.schedule,
                    invocation_decision=decision,
                    boundary_event=boundary.event,
                    window=boundary.window,
                    cycle=None if execution is None else execution.cycle,
                    request=None if execution is None else execution.request,
                    two_phase_result=(None if execution is None else execution.two_phase_result),
                    phase_one_result=(None if execution is None else execution.phase_one_result),
                    call_receipts=() if execution is None else execution.calls,
                    retrieval_request=(None if execution is None else execution.retrieval_request),
                    retrieval_result=(None if execution is None else execution.retrieval_result),
                    delivery_record=None if execution is None else execution.delivery,
                    observation=observation,
                )
                return _BoundaryProjection(decision=decision, evidence=evidence)

            if decision_cancelled:
                await finish_boundary()
                raise asyncio.CancelledError
            return await finish_boundary()

        inputs = tuple(
            FixedStepTraceInput(
                draft=item.draft,
                expected_event_id=item.expected_event_id,
                task_description=item.task_description,
                logical_messages=item.logical_messages,
                action_step=item.action_step,
                target_request_id=item.target_request_id,
            )
            for item in checked_trajectory.inputs
        )
        try:
            trace = await FixedStepTraceDriver(self._repository).run(inputs, on_boundary)
            projections = trace.boundary_projections
            decisions = tuple(item.decision for item in projections)
            boundaries = tuple(item.evidence for item in projections if item.evidence is not None)
            calls = tuple(call for item in boundaries for call in item.call_receipts)
            expected_calls = trace.spine.schedule.invocation_count * len(
                self._condition.expected.call_phases
            )
            if len(calls) != expected_calls:
                raise ValueError
            response_fixture = None
            if isinstance(self._client, TwoPhaseReplayClient):
                if (
                    self._client.total_responses != expected_calls
                    or self._client.remaining_responses != 0
                ):
                    raise ValueError
                response_fixture = Stage2ResponseFixtureIdentity(
                    replay_id=self._client.replay_id,
                    fixture_digest=self._client.fixture_digest,
                    response_count=self._client.total_responses,
                )
            budget = await self._repository.budget_snapshot(checked_trajectory.run_id)
            memory = await self._repository.snapshot(checked_trajectory.run_id)
            semantic = semantic_projection_digests(
                checked_trajectory.run_id,
                trace.ledger,
            )
            return Stage2ExperimentRunResult(
                trajectory=checked_trajectory,
                condition=self._condition,
                prompt_bundle=self._prompt_bundle.identity,
                model_profile=self._model_profile,
                call_policy=self._call_policy,
                cycle_reservation=self._reservation,
                budget_limits=limits,
                run_id=trace.spine.run_id,
                trace_digest=trace.spine.trace_digest,
                normalized_draft_digests=trace.spine.normalized_draft_digests,
                persisted_event_draft_digests=(trace.spine.persisted_event_draft_digests),
                trajectory_prefix=trace.spine.trajectory_prefix,
                schedule=trace.spine.schedule,
                windows=trace.spine.windows,
                decisions=decisions,
                boundaries=boundaries,
                call_receipts=calls,
                response_fixture=response_fixture,
                metrics=_metrics(boundaries, calls),
                final_budget_snapshot=budget,
                final_memory_snapshot=memory,
                semantic_projection_digests=semantic,
                repository_projection_digests=trace.projection_digests,
                ledger=trace.ledger,
                ledger_entry_count=len(trace.ledger),
                ledger_head=trace.ledger_head,
                rebuild_equivalent=True,
            )
        except asyncio.CancelledError:
            raise
        except Stage2ExperimentError:
            raise
        except Exception:
            raise Stage2ExperimentError() from None


async def _replay_stage2_fixture_once(
    trajectory_path: str | Path,
    *,
    condition: Stage2ConditionId | str,
    responses_path: str | Path | None,
    expected_trajectory_fixture_digest: str | None,
    expected_response_fixture_digest: str | None,
) -> Stage2ExperimentRunResult:
    try:
        resolved = resolve_stage2_condition(condition)
        trajectory = load_stage2_trajectory(
            trajectory_path,
            expected_fixture_digest=expected_trajectory_fixture_digest,
        )
        normalized = tuple(canonical_digest(item.draft) for item in trajectory.inputs)
        trace_digest = algorithm_trace_digest(normalized)
        if resolved.condition_id is Stage2ConditionId.NO_MEMORY:
            if responses_path is not None or expected_response_fixture_digest is not None:
                raise ValueError
            client: StructuredCallClient | None = None
        else:
            if responses_path is None:
                raise ValueError
            client = TwoPhaseReplayClient.from_path(
                responses_path,
                expected_fixture_digest=expected_response_fixture_digest,
            )
        repository = MemoryRunRepository(
            synthetic_benchmark=True,
            id_factory=_repository_id_factory(trace_digest),
        )
        runner = Stage2ExperimentRunner(
            repository=repository,
            condition=resolved.condition_id,
            client=client,
        )
        return await runner.run(trajectory)
    except asyncio.CancelledError:
        raise
    except Stage2ExperimentError:
        raise
    except Exception:
        raise Stage2ExperimentError() from None


async def replay_stage2_fixture_twice(
    trajectory_path: str | Path,
    *,
    condition: Stage2ConditionId | str,
    responses_path: str | Path | None = None,
    expected_trajectory_fixture_digest: str | None = None,
    expected_response_fixture_digest: str | None = None,
) -> Stage2ExperimentRunResult:
    """Replay two fresh offline runs and return their byte-identical result."""

    first = await _replay_stage2_fixture_once(
        trajectory_path,
        condition=condition,
        responses_path=responses_path,
        expected_trajectory_fixture_digest=expected_trajectory_fixture_digest,
        expected_response_fixture_digest=expected_response_fixture_digest,
    )
    second = await _replay_stage2_fixture_once(
        trajectory_path,
        condition=condition,
        responses_path=responses_path,
        expected_trajectory_fixture_digest=expected_trajectory_fixture_digest,
        expected_response_fixture_digest=expected_response_fixture_digest,
    )
    if canonical_json(first) != canonical_json(second):
        raise Stage2ExperimentError()
    return first


__all__ = [
    "STAGE2_EXPERIMENT_METRICS_SCHEMA_VERSION",
    "STAGE2_EXPERIMENT_POLICY_VERSION",
    "STAGE2_EXPERIMENT_RUN_RESULT_SCHEMA_VERSION",
    "STAGE2_RESPONSE_FIXTURE_IDENTITY_SCHEMA_VERSION",
    "Stage2ExperimentError",
    "Stage2ExperimentMetrics",
    "Stage2ExperimentRunResult",
    "Stage2ExperimentRunner",
    "Stage2ResponseFixtureIdentity",
    "replay_stage2_fixture_twice",
]
