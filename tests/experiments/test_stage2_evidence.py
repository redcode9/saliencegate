from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

import saliencegate.experiments.evidence as evidence_module
from saliencegate.domain import (
    BudgetAmounts,
    BudgetLimits,
    BudgetSnapshot,
    ClaimKind,
    CycleRecord,
    DeduplicationGuarantee,
    DeliveryRecord,
    DeliveryTarget,
    EventPhase,
    EventType,
    EvidenceReference,
    EvidenceSource,
    InterventionAction,
    InvocationDecision,
    NormalizedTraceEventDraft,
    PayloadDigest,
    PayloadDigestAlgorithm,
    ReasonCode,
    TrustLabel,
    ValidityState,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.experiments.conditions import (
    Stage2ConditionId,
    resolve_stage2_condition,
)
from saliencegate.experiments.evidence import (
    Stage2BoundaryEvidence,
    Stage2EvidenceError,
    derive_stage2_condition_observation,
)
from saliencegate.experiments.retrieval import (
    RetrievalRequest,
    RetrievalResult,
    build_retrieval_request,
    retrieval_selector_provenance,
    retrieve_candidate_bank,
)
from saliencegate.intervention import (
    GroundingConfig,
    GroundingContext,
    GroundingPipeline,
    GroundingState,
    InterventionProposal,
    ProposedClaim,
    RenderingConfig,
)
from saliencegate.memory import (
    BankOperationsProposal,
    InterventionSelectionOutput,
    SaveKnowledge,
)
from saliencegate.memory.two_phase import (
    PaperTwoPhaseCycleExecutor,
    RepositoryOperationMaterializer,
)
from saliencegate.ports.adapters import (
    AdapterCapabilities,
    DeliveryChannel,
    DeliveryEnvelope,
    DeliveryReceipt,
    DeliveryRole,
    InjectionMapping,
    enqueue_delivery_binding,
)
from saliencegate.ports.model_calls import (
    ProviderUsageProvenance,
    StructuredCallClient,
    StructuredCallParseStatus,
    StructuredCallPhase,
    StructuredCallRequest,
    StructuredCallResult,
    StructuredCallStatus,
    StructuredCallUsage,
)
from saliencegate.ports.trajectory import ActionStepBinding, EventTextSelector
from saliencegate.ports.two_phase import (
    CallReceipt,
    PhaseOneCycleResult,
    TwoPhaseCallPolicy,
    TwoPhaseCycleRequest,
    TwoPhaseCycleResult,
    TwoPhaseModelProfile,
)
from saliencegate.prompts import PAPER_TWO_PHASE_FORCED_REMINDER_V1, PAPER_TWO_PHASE_V1
from saliencegate.prompts.contracts import BankViewKind, build_active_bank_prompt_view
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.runtime.algorithm_result import derive_cycle_reservation
from saliencegate.runtime.cycles import CycleCoordinator
from saliencegate.runtime.delivery import DeliveryWorker
from saliencegate.runtime.fixed_step_core import (
    FixedStepTraceBoundary,
    FixedStepTraceDriver,
    FixedStepTraceInput,
)
from saliencegate.runtime.message_window import MessageWindow
from saliencegate.security import InstallationKey

RUN_ID = UUID("00000000-0000-4000-8000-00000000e001")
EVENT_ID = UUID("00000000-0000-4000-8000-00000000e011")
MEMORY_ID = UUID("00000000-0000-4000-8000-00000000e204")
NOW = datetime(2026, 7, 13, 9, 0, tzinfo=UTC)
_COMPLETION_DIGEST_DOMAIN = "saliencegate:test:stage2-evidence-completion:v1"


@dataclass(frozen=True, slots=True)
class _BoundarySources:
    boundary: FixedStepTraceBoundary
    invocation_decision: InvocationDecision
    cycle: CycleRecord | None = None
    request: TwoPhaseCycleRequest | None = None
    two_phase_result: TwoPhaseCycleResult | None = None
    phase_one_result: PhaseOneCycleResult | None = None
    call_receipts: tuple[CallReceipt, ...] = ()
    retrieval_request: RetrievalRequest | None = None
    retrieval_result: RetrievalResult | None = None
    delivery_record: DeliveryRecord | None = None


def _grounding_config() -> GroundingConfig:
    return GroundingConfig(
        schema_version="1.0",
        pipeline_version="grounding-pipeline/v1",
        claim_schema_version="citation-only-claims/v1",
        max_claims=2,
        max_evidence_per_claim=1,
        max_pointer_segments=32,
        max_pointer_utf8_bytes=1_024,
        duplicate_window_events=0,
        cooldown_events=0,
        ttl_steps=1,
        allowed_delivery_targets=(
            DeliveryTarget.NEXT_MODEL_CALL,
            DeliveryTarget.PRE_ACTION_REPLAN,
        ),
        rendering=RenderingConfig(
            schema_version="1.0",
            renderer_version="fixed-ascii/v1",
            token_counter_version="utf8-bytes-ceil-div-4-v1",
            max_claims=2,
            max_evidence_bytes=1_024,
            max_output_bytes=4_096,
            max_token_equivalents=1_024,
            include_provenance=False,
        ),
    )


def _completed_result(
    request: StructuredCallRequest,
    output: BankOperationsProposal | InterventionSelectionOutput,
) -> StructuredCallResult:
    completion = canonical_json(output)
    return StructuredCallResult(
        schema_version="structured-call-result/v1",
        request_digest=request.request_digest,
        model_call_index=request.model_call_index,
        phase=request.phase,
        attempt=request.attempt,
        response_schema_version=request.response_schema_version,
        status=StructuredCallStatus.COMPLETED,
        parse_status=StructuredCallParseStatus.VALID,
        output=output,
        completion_digest=PayloadDigest(
            algorithm=PayloadDigestAlgorithm.SYNTHETIC_SHA256,
            value=length_prefixed_sha256(
                completion,
                domain=_COMPLETION_DIGEST_DOMAIN,
            ),
        ),
        completion_byte_count=len(completion),
        usage=StructuredCallUsage(
            schema_version="structured-call-usage/v1",
            provider_input_tokens=11,
            provider_output_tokens=7,
            provider_usage_provenance=ProviderUsageProvenance.REPLAY_ATTESTED,
            latency_us=100 + request.model_call_index,
        ),
    )


class _FixtureClient:
    def __init__(self, *, reminder: bool) -> None:
        self.reminder = reminder

    async def generate(self, request: StructuredCallRequest) -> StructuredCallResult:
        if request.phase is StructuredCallPhase.MEMORY_EDIT:
            operations = (
                (
                    SaveKnowledge(
                        operation="save_knowledge",
                        content="Run the verified suite before release.",
                        evidence=(
                            EvidenceReference(
                                source=EvidenceSource.EVENT,
                                source_id=EVENT_ID,
                                field_path="/payload/task",
                            ),
                        ),
                        confidence=1.0,
                    ),
                )
                if self.reminder
                else ()
            )
            return _completed_result(
                request,
                BankOperationsProposal(
                    schema_version="memory-edit-output/v1",
                    operations=operations,
                ),
            )
        return _completed_result(
            request,
            InterventionSelectionOutput(
                schema_version="intervention-output/v1",
                action=(InterventionAction.REMIND if self.reminder else InterventionAction.SILENCE),
                claims=(
                    (
                        ProposedClaim(
                            kind=ClaimKind.REQUIREMENT,
                            evidence=EvidenceReference(
                                source=EvidenceSource.MEMORY,
                                source_id=MEMORY_ID,
                                revision=1,
                                field_path="/content",
                            ),
                        ),
                    )
                    if self.reminder
                    else ()
                ),
                confidence=1.0,
            ),
        )


class _ForcedSchemaInvalidClient:
    def __init__(self, delegate: _FixtureClient) -> None:
        self._delegate = delegate

    async def generate(self, request: StructuredCallRequest) -> StructuredCallResult:
        if request.phase is StructuredCallPhase.MEMORY_EDIT:
            return await self._delegate.generate(request)
        return StructuredCallResult(
            schema_version="structured-call-result/v1",
            request_digest=request.request_digest,
            model_call_index=request.model_call_index,
            phase=request.phase,
            attempt=request.attempt,
            response_schema_version=request.response_schema_version,
            status=StructuredCallStatus.COMPLETED,
            parse_status=StructuredCallParseStatus.SCHEMA_INVALID,
            output=None,
            completion_digest=PayloadDigest(
                algorithm=PayloadDigestAlgorithm.SYNTHETIC_SHA256,
                value="a" * 64,
            ),
            completion_byte_count=2,
            usage=StructuredCallUsage(
                schema_version="structured-call-usage/v1",
                provider_input_tokens=11,
                provider_output_tokens=7,
                provider_usage_provenance=ProviderUsageProvenance.REPLAY_ATTESTED,
                latency_us=101,
            ),
        )


class _DeliveryAdapter:
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            schema_version="1.0",
            adapter_id="stage2-evidence-delivery/v1",
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
            adapter_id=delivery.adapter_id,
            target_request_id=delivery.target_request_id,
            delivered_at=delivery.created_at + timedelta(microseconds=1),
            provider_receipt_id="stage2-evidence-receipt/v1",
        )


def _trace_input(*, reminder: bool) -> FixedStepTraceInput:
    return FixedStepTraceInput(
        draft=NormalizedTraceEventDraft(
            run_id=RUN_ID,
            source_event_id="stage2-evidence-run-start",
            timestamp=NOW,
            event_type=EventType.RUN_START,
            phase=EventPhase.INITIALIZATION,
            payload={"task": "Keep verified release constraints current.", "step": 1},
            source_adapter="stage2-evidence-fixture/v1",
            trust_label=TrustLabel.SYNTHETIC_FIXTURE,
        ),
        expected_event_id=EVENT_ID,
        task_description=EventTextSelector(field_path="/payload/task"),
        action_step=ActionStepBinding(field_path="/payload/step"),
        target_request_id="stage2-evidence-target/v1" if reminder else None,
    )


def _repository() -> MemoryRunRepository:
    identifiers = itertools.count(0xE100)
    return MemoryRunRepository(
        installation_key=InstallationKey(b"e" * 32),
        id_factory=lambda: UUID(f"00000000-0000-4000-8000-{next(identifiers):012x}"),
    )


def _call_policy() -> TwoPhaseCallPolicy:
    return TwoPhaseCallPolicy(
        schema_version="two-phase-call-policy/v1",
        max_model_calls=2,
        max_schema_repairs=0,
        client_retries=0,
        max_provider_input_tokens=200,
        max_provider_output_tokens=100,
        max_total_latency_us=10_000,
        max_call_latency_us=5_000,
    )


def _profile(condition_id: Stage2ConditionId) -> TwoPhaseModelProfile:
    bundle = (
        PAPER_TWO_PHASE_FORCED_REMINDER_V1
        if condition_id is Stage2ConditionId.ALWAYS_INJECT
        else PAPER_TWO_PHASE_V1
    )
    identity = bundle.identity
    return TwoPhaseModelProfile(
        schema_version="two-phase-model-profile/v1",
        profile_id=f"stage2-evidence-{condition_id.value}/v1",
        model_id="openai-compatible-stage2-evidence/v1",
        prompt_bundle_id=identity.bundle_id,
        prompt_bundle_digest=identity.bundle_digest,
    )


def _budget(policy: TwoPhaseCallPolicy) -> tuple[BudgetSnapshot, BudgetAmounts]:
    reservation = derive_cycle_reservation(policy)
    limits = BudgetLimits(
        model_calls=reservation.model_calls,
        input_tokens=reservation.input_tokens,
        output_tokens=reservation.output_tokens,
        canonical_token_equivalents=reservation.canonical_token_equivalents,
        latency_us=reservation.latency_us,
        interventions=reservation.interventions,
        schema_repairs=reservation.schema_repairs,
        max_call_latency_us=policy.max_call_latency_us,
    )
    return (
        BudgetSnapshot(
            limits=limits,
            reserved=BudgetAmounts(),
            consumed=BudgetAmounts(),
        ),
        reservation,
    )


def _settlement(
    result: TwoPhaseCycleResult | PhaseOneCycleResult,
    reservation: BudgetAmounts,
) -> BudgetAmounts:
    usage = result.usage
    assert usage.provider_input_tokens is not None
    assert usage.provider_output_tokens is not None
    interventions = (
        int(result.intervention.action is InterventionAction.REMIND)
        if type(result) is TwoPhaseCycleResult
        else 0
    )
    return BudgetAmounts(
        model_calls=usage.model_calls,
        input_tokens=usage.provider_input_tokens,
        output_tokens=usage.provider_output_tokens,
        canonical_token_equivalents=reservation.canonical_token_equivalents,
        latency_us=usage.latency_us,
        interventions=interventions,
        schema_repairs=usage.schema_repairs,
    )


async def _sources(
    condition_id: Stage2ConditionId,
    *,
    reminder: bool = False,
) -> _BoundarySources:
    condition = resolve_stage2_condition(condition_id)
    repository = _repository()
    trace_input = _trace_input(reminder=reminder)
    policy = _call_policy()
    snapshot, reservation = _budget(policy)
    grounding = GroundingPipeline(_grounding_config())
    coordinator = CycleCoordinator(repository)

    async def project(boundary: FixedStepTraceBoundary) -> _BoundarySources:
        assert boundary.window is not None
        active = condition_id is not Stage2ConditionId.NO_MEMORY
        decision = InvocationDecision(
            decision_id=UUID(
                "00000000-0000-4000-8000-00000000e201"
                if active
                else "00000000-0000-4000-8000-00000000e202"
            ),
            run_id=boundary.event.run_id,
            event_sequence=boundary.event.sequence,
            invoke=active,
            risk_score=None,
            reason_codes=(ReasonCode.BOOTSTRAP if active else ReasonCode.POLICY_NEVER,),
            policy_version="paper-fixed-step/v1",
            configuration_digest=condition.condition_digest,
            budget_snapshot=snapshot,
            cooldown_active=False,
            created_at=boundary.event.timestamp,
        )
        recorded = await repository.record_invocation_decision(decision)
        assert recorded.appended
        if not active:
            return _BoundarySources(boundary=boundary, invocation_decision=decision)

        pending = await coordinator.begin(
            decision,
            grounding=grounding.pin(condition.shared_controls.requested_delivery_target),
            created_at=boundary.event.timestamp,
        )
        reserved = await coordinator.reserve(
            pending,
            reservation=reservation,
            updated_at=boundary.event.timestamp,
        )
        running = await coordinator.start(
            reserved,
            batch_digest=boundary.window.window_digest,
            updated_at=boundary.event.timestamp,
        )
        repository_snapshot = await repository.snapshot(boundary.event.run_id)
        records = tuple(
            sorted(
                (
                    record
                    for record in repository_snapshot.records
                    if record.validity is ValidityState.ACTIVE
                ),
                key=lambda item: (item.kind.value, str(item.memory_id)),
            )
        )
        current_bank = build_active_bank_prompt_view(
            kind=BankViewKind.CURRENT,
            run_id=boundary.event.run_id,
            as_of=boundary.event.timestamp,
            source_projection_digest=repository_snapshot.projection_digest,
            records=records,
        )
        state = GroundingState(
            schema_version="1.0",
            events=tuple(item.event for item in boundary.prefix.items),
            memories=records,
            reminder_history=(),
        )
        request = TwoPhaseCycleRequest(
            schema_version="two-phase-cycle-request/v1",
            cycle_receipt=running,
            window=boundary.window,
            current_bank=current_bank,
            grounding_state=state,
            delta_id=UUID("00000000-0000-4000-8000-00000000e203"),
            assigned_memory_ids=(
                (UUID("00000000-0000-4000-8000-00000000e204"),) if reminder else ()
            ),
            intervention_id=UUID("00000000-0000-4000-8000-00000000e205"),
            created_at=boundary.event.timestamp,
        )
        delegate = _FixtureClient(reminder=reminder)
        client: StructuredCallClient = (
            _ForcedSchemaInvalidClient(delegate)
            if condition_id is Stage2ConditionId.ALWAYS_INJECT
            else delegate
        )
        bundle = (
            PAPER_TWO_PHASE_FORCED_REMINDER_V1
            if condition_id is Stage2ConditionId.ALWAYS_INJECT
            else PAPER_TWO_PHASE_V1
        )
        executor = PaperTwoPhaseCycleExecutor(
            materializer=RepositoryOperationMaterializer(repository),
            client=client,
            prompt_bundle=bundle,
            grounding_pipeline=grounding,
            model_profile=_profile(condition_id),
            call_policy=policy,
        )

        retrieval_request = None
        retrieval_result = None
        selector = None
        if condition_id is Stage2ConditionId.RETRIEVAL_ALWAYS:
            phase_one = await executor.execute_phase_one(request)
            assert type(phase_one) is PhaseOneCycleResult
            retrieval_request = build_retrieval_request(
                condition=condition,
                window=boundary.window,
                materialization=phase_one.materialization,
            )
            retrieval_result = retrieve_candidate_bank(retrieval_request)
            selector = retrieval_selector_provenance(retrieval_request, retrieval_result)
            selection = retrieval_result.selection
            intervention = grounding.ground(
                InterventionProposal(
                    action=selection.action,
                    claims=selection.claims,
                    confidence=selection.confidence,
                    model_free_text=None,
                ),
                context=GroundingContext(
                    schema_version="2.0",
                    intervention_id=request.intervention_id,
                    run_id=boundary.event.run_id,
                    cycle_id=running.cycle.cycle_id,
                    current_event_sequence=boundary.event.sequence,
                    created_at=boundary.event.timestamp,
                    requested_delivery_target=(condition.shared_controls.requested_delivery_target),
                    selector_provenance=selector,
                ),
                state=GroundingState(
                    schema_version="1.0",
                    events=state.events,
                    memories=phase_one.materialization.active_bank,
                    reminder_history=state.reminder_history,
                ),
            )
            result: TwoPhaseCycleResult | PhaseOneCycleResult = phase_one
            two_phase = None
        else:
            completed = await executor.execute(request)
            assert type(completed) is TwoPhaseCycleResult
            result = completed
            two_phase = completed
            phase_one = None
            intervention = completed.intervention

        enqueue = None
        adapter = None
        if intervention.action is InterventionAction.REMIND:
            adapter = _DeliveryAdapter()
            assert trace_input.target_request_id is not None
            enqueue = enqueue_delivery_binding(
                target_request_id=trace_input.target_request_id,
                capabilities=adapter.capabilities(),
            )
        committed = await coordinator.commit(
            running,
            settlement=_settlement(result, reservation),
            validated_delta=result.validated_delta,
            memory_id_assignments=result.memory_id_assignments,
            intervention=intervention,
            selector_provenance=(
                None if selector is None else selector.model_dump(mode="json", warnings=False)
            ),
            delivery=enqueue,
            updated_at=boundary.event.timestamp,
            model_call_digests=tuple(item.call_digest for item in result.call_receipts),
            model_call_latencies_us=tuple(item.usage.latency_us for item in result.call_receipts),
        )
        delivery_record = None
        if committed.delivery is not None:
            assert adapter is not None
            worker_ids = itertools.count(0xE300)
            delivery_record = (
                await DeliveryWorker(
                    repository=repository,
                    adapter=adapter,
                    id_factory=lambda: UUID(f"00000000-0000-4000-8000-{next(worker_ids):012x}"),
                ).deliver(
                    boundary.event.run_id,
                    committed.delivery.delivery_id,
                    now=boundary.event.timestamp,
                )
            ).delivery
        return _BoundarySources(
            boundary=boundary,
            invocation_decision=decision,
            cycle=committed.cycle,
            request=request,
            two_phase_result=two_phase,
            phase_one_result=phase_one,
            call_receipts=result.call_receipts,
            retrieval_request=retrieval_request,
            retrieval_result=retrieval_result,
            delivery_record=delivery_record,
        )

    trace = await FixedStepTraceDriver(repository).run((trace_input,), project)
    return trace.boundary_projections[0]


def _evidence(
    condition_id: Stage2ConditionId,
    sources: _BoundarySources,
) -> Stage2BoundaryEvidence:
    condition = resolve_stage2_condition(condition_id)
    boundary = sources.boundary
    window = boundary.window
    assert type(window) is MessageWindow
    observation = derive_stage2_condition_observation(
        condition=condition,
        schedule=boundary.schedule,
        invocation_decision=sources.invocation_decision,
        boundary_event=boundary.event,
        window=window,
        cycle=sources.cycle,
        request=sources.request,
        two_phase_result=sources.two_phase_result,
        phase_one_result=sources.phase_one_result,
        call_receipts=sources.call_receipts,
        retrieval_request=sources.retrieval_request,
        retrieval_result=sources.retrieval_result,
        delivery_record=sources.delivery_record,
    )
    return Stage2BoundaryEvidence(
        condition=condition,
        schedule=boundary.schedule,
        invocation_decision=sources.invocation_decision,
        boundary_event=boundary.event,
        window=window,
        cycle=sources.cycle,
        request=sources.request,
        two_phase_result=sources.two_phase_result,
        phase_one_result=sources.phase_one_result,
        call_receipts=sources.call_receipts,
        retrieval_request=sources.retrieval_request,
        retrieval_result=sources.retrieval_result,
        delivery_record=sources.delivery_record,
        observation=observation,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("condition_id", "expected_calls", "expected_violation"),
    (
        (Stage2ConditionId.NO_MEMORY, 0, False),
        (Stage2ConditionId.FIXED_STEP, 2, False),
        (Stage2ConditionId.RETRIEVAL_ALWAYS, 1, False),
        (Stage2ConditionId.ALWAYS_INJECT, 2, True),
    ),
)
async def test_boundary_evidence_derives_exact_closed_condition_cardinality(
    condition_id: Stage2ConditionId,
    expected_calls: int,
    expected_violation: bool,
) -> None:
    evidence = _evidence(condition_id, await _sources(condition_id))

    assert len(evidence.call_receipts) == expected_calls
    assert evidence.observation.condition_violation is expected_violation
    assert evidence.observation.observed.call_receipt_digests == tuple(
        item.receipt_digest for item in evidence.call_receipts
    )
    assert (
        Stage2BoundaryEvidence.model_validate_json(evidence.model_dump_json(warnings=False))
        == evidence
    )
    with pytest.raises(ValidationError):
        evidence.observation = evidence.observation  # type: ignore[misc]


@pytest.mark.asyncio
async def test_observation_is_recomputed_and_cannot_replace_authoritative_sources() -> None:
    fixed = _evidence(Stage2ConditionId.FIXED_STEP, await _sources(Stage2ConditionId.FIXED_STEP))
    control = _evidence(Stage2ConditionId.NO_MEMORY, await _sources(Stage2ConditionId.NO_MEMORY))
    payload = fixed.model_dump(mode="python", exclude={"evidence_digest"})
    payload["observation"] = control.observation

    with pytest.raises(ValidationError, match="not derived"):
        Stage2BoundaryEvidence.model_validate(payload)
    with pytest.raises(Stage2EvidenceError):
        derive_stage2_condition_observation(
            condition=fixed.condition,
            schedule=fixed.schedule,
            invocation_decision=fixed.invocation_decision,
            boundary_event=fixed.boundary_event,
            window=fixed.window,
            cycle=fixed.cycle,
            request=fixed.request,
            two_phase_result=fixed.two_phase_result,
            call_receipts=fixed.call_receipts[:-1],
        )


@pytest.mark.asyncio
async def test_evidence_rejects_tampered_boundary_cycle_request_calls_and_materialization() -> None:
    evidence = _evidence(
        Stage2ConditionId.FIXED_STEP,
        await _sources(Stage2ConditionId.FIXED_STEP),
    )
    assert evidence.cycle is not None
    assert evidence.cycle.intervention is not None
    assert evidence.request is not None
    assert evidence.two_phase_result is not None
    changed_intervention = evidence.cycle.intervention.model_copy(
        update={"intervention_id": UUID("00000000-0000-4000-8000-00000000eff1")}
    )
    changed_materialization = evidence.two_phase_result.materialization.model_copy(
        update={"materialization_digest": "0" * 64}
    )
    cases = (
        {"condition": resolve_stage2_condition(Stage2ConditionId.NO_MEMORY)},
        {"schedule": evidence.schedule.model_copy(update={"schedule_digest": "0" * 64})},
        {
            "invocation_decision": evidence.invocation_decision.model_copy(
                update={"event_sequence": 2}
            )
        },
        {
            "boundary_event": evidence.boundary_event.model_copy(
                update={"payload": {"task": "tampered", "step": 1}}
            )
        },
        {"window": evidence.window.model_copy(update={"window_digest": "0" * 64})},
        {
            "cycle": evidence.cycle.model_copy(
                update={"model_call_digests": tuple(reversed(evidence.cycle.model_call_digests))}
            )
        },
        {"cycle": evidence.cycle.model_copy(update={"intervention": changed_intervention})},
        {"call_receipts": evidence.call_receipts[:-1]},
        {
            "two_phase_result": evidence.two_phase_result.model_copy(
                update={"candidate_bank_view_digest": "0" * 64}
            )
        },
        {
            "two_phase_result": evidence.two_phase_result.model_copy(
                update={"materialization": changed_materialization}
            )
        },
        {"request": evidence.request.model_copy(update={"request_digest": "0" * 64})},
    )
    for change in cases:
        payload = evidence.model_dump(mode="python", exclude={"evidence_digest"}) | change
        with pytest.raises(ValidationError):
            Stage2BoundaryEvidence.model_validate(payload)


@pytest.mark.asyncio
async def test_retrieval_grounding_requires_the_recomputed_v2_selector_provenance() -> None:
    evidence = _evidence(
        Stage2ConditionId.RETRIEVAL_ALWAYS,
        await _sources(Stage2ConditionId.RETRIEVAL_ALWAYS),
    )
    assert evidence.cycle is not None
    assert evidence.cycle.intervention is not None
    assert evidence.retrieval_request is not None
    assert evidence.retrieval_result is not None
    for change in (
        {
            "retrieval_request": evidence.retrieval_request.model_copy(
                update={"request_digest": "0" * 64}
            )
        },
        {
            "retrieval_result": evidence.retrieval_result.model_copy(
                update={"result_digest": "0" * 64}
            )
        },
    ):
        payload = evidence.model_dump(mode="python", exclude={"evidence_digest"}) | change
        with pytest.raises(ValidationError):
            Stage2BoundaryEvidence.model_validate(payload)

    receipt = dict(evidence.cycle.intervention.grounding_receipt)
    raw_selector = receipt["selector_provenance"]
    assert isinstance(raw_selector, Mapping)
    selector = dict(raw_selector)
    selector["result_digest"] = "0" * 64
    receipt["selector_provenance"] = selector
    intervention = evidence.cycle.intervention.model_copy(update={"grounding_receipt": receipt})
    cycle = evidence.cycle.model_copy(update={"intervention": intervention})
    payload = evidence.model_dump(mode="python", exclude={"evidence_digest"})
    payload["cycle"] = cycle

    with pytest.raises(ValidationError):
        Stage2BoundaryEvidence.model_validate(payload)


@pytest.mark.asyncio
async def test_forced_condition_binds_the_forced_prompt_and_safe_silence_schema() -> None:
    evidence = _evidence(
        Stage2ConditionId.ALWAYS_INJECT,
        await _sources(Stage2ConditionId.ALWAYS_INJECT),
    )
    assert evidence.two_phase_result is not None
    intervention_call = evidence.call_receipts[-1]
    forced_identity = PAPER_TWO_PHASE_FORCED_REMINDER_V1.identity
    forced_template = next(
        item for item in forced_identity.templates if item.phase is StructuredCallPhase.INTERVENTION
    )

    assert evidence.two_phase_result.prompt_bundle_digest == forced_identity.bundle_digest
    assert intervention_call.prompt_template_id == forced_template.template_id
    assert intervention_call.prompt_template_digest == forced_template.template_digest
    assert evidence.observation.observed.phase_two_schema_digest == (
        evidence.condition.shared_controls.forced_phase_two_schema_digest
    )
    assert evidence.observation.observed.intervention_action is InterventionAction.SILENCE
    assert evidence.observation.condition_violation is True


@pytest.mark.asyncio
async def test_final_delivery_is_exactly_bound_to_the_grounded_reminder() -> None:
    evidence = _evidence(
        Stage2ConditionId.FIXED_STEP,
        await _sources(Stage2ConditionId.FIXED_STEP, reminder=True),
    )
    assert evidence.delivery_record is not None
    assert evidence.observation.observed.delivery_record_count == 1
    assert evidence.delivery_record.target is DeliveryTarget.NEXT_MODEL_CALL
    changed = evidence.delivery_record.model_copy(update={"rendered_text_digest": "0" * 64})
    payload = evidence.model_dump(mode="python", exclude={"evidence_digest"})
    payload["delivery_record"] = changed

    with pytest.raises(ValidationError):
        Stage2BoundaryEvidence.model_validate(payload)


@pytest.mark.asyncio
async def test_evidence_digest_rejects_recalculated_source_free_tampering() -> None:
    evidence = _evidence(Stage2ConditionId.NO_MEMORY, await _sources(Stage2ConditionId.NO_MEMORY))
    payload = evidence.model_dump(mode="python")
    payload["evidence_digest"] = "0" * 64

    with pytest.raises(ValidationError, match="digest does not match"):
        Stage2BoundaryEvidence.model_validate(payload)
    assert len(canonical_json(evidence)) > 0


@pytest.mark.asyncio
async def test_every_closed_failure_branch_rejects_incomplete_or_cross_bound_sources() -> None:
    fixed = _evidence(Stage2ConditionId.FIXED_STEP, await _sources(Stage2ConditionId.FIXED_STEP))
    retrieval = _evidence(
        Stage2ConditionId.RETRIEVAL_ALWAYS,
        await _sources(Stage2ConditionId.RETRIEVAL_ALWAYS),
    )
    control = _evidence(Stage2ConditionId.NO_MEMORY, await _sources(Stage2ConditionId.NO_MEMORY))
    reminder = _evidence(
        Stage2ConditionId.FIXED_STEP,
        await _sources(Stage2ConditionId.FIXED_STEP, reminder=True),
    )
    assert fixed.cycle is not None and fixed.request is not None
    assert fixed.two_phase_result is not None
    assert retrieval.cycle is not None and retrieval.request is not None
    assert retrieval.phase_one_result is not None
    assert retrieval.retrieval_request is not None
    assert retrieval.retrieval_result is not None
    assert reminder.cycle is not None
    assert reminder.delivery_record is not None

    with pytest.raises(ValueError, match="exact source type"):
        evidence_module._exact(object(), MessageWindow)  # type: ignore[type-var]
    with pytest.raises(ValueError, match="wrong prompt bundle"):
        evidence_module._validate_prompt_receipts(
            condition=fixed.condition,
            prompt_bundle_digest="0" * 64,
            receipts=fixed.call_receipts,
        )
    wrong_template = fixed.call_receipts[0].model_copy(
        update={"prompt_template_id": "wrong-template/v1"}
    )
    with pytest.raises(ValueError, match="wrong prompt template"):
        evidence_module._validate_prompt_receipts(
            condition=fixed.condition,
            prompt_bundle_digest=fixed.two_phase_result.prompt_bundle_digest,
            receipts=(wrong_template, *fixed.call_receipts[1:]),
        )

    wrong_model_request = fixed.request.model_copy(
        update={"intervention_id": UUID("00000000-0000-4000-8000-00000000eff2")}
    )
    with pytest.raises(ValueError, match="final-call provenance"):
        evidence_module._validate_model_intervention(
            condition=fixed.condition,
            cycle=fixed.cycle,
            request=wrong_model_request,
            result=fixed.two_phase_result,
            receipts=fixed.call_receipts,
        )
    with pytest.raises(ValueError, match="forced-reminder output"):
        evidence_module._validate_model_intervention(
            condition=resolve_stage2_condition(Stage2ConditionId.ALWAYS_INJECT),
            cycle=fixed.cycle,
            request=fixed.request,
            result=fixed.two_phase_result,
            receipts=fixed.call_receipts,
        )

    without_intervention = retrieval.cycle.model_copy(update={"intervention": None})
    with pytest.raises(ValueError, match="lacks its grounded intervention"):
        evidence_module._validate_retrieval_intervention(
            condition=retrieval.condition,
            boundary_event=retrieval.boundary_event,
            window=retrieval.window,
            cycle=without_intervention,
            request=retrieval.request,
            result=retrieval.phase_one_result,
            retrieval_request=retrieval.retrieval_request,
            retrieval_result=retrieval.retrieval_result,
        )
    wrong_retrieval_request = retrieval.request.model_copy(
        update={"intervention_id": UUID("00000000-0000-4000-8000-00000000eff3")}
    )
    with pytest.raises(ValueError, match="selector provenance"):
        evidence_module._validate_retrieval_intervention(
            condition=retrieval.condition,
            boundary_event=retrieval.boundary_event,
            window=retrieval.window,
            cycle=retrieval.cycle,
            request=wrong_retrieval_request,
            result=retrieval.phase_one_result,
            retrieval_request=retrieval.retrieval_request,
            retrieval_result=retrieval.retrieval_result,
        )

    with pytest.raises(ValueError, match="safe silence"):
        evidence_module._validate_delivery(fixed.cycle, reminder.delivery_record)
    with pytest.raises(ValueError, match="requires its final delivery"):
        evidence_module._validate_delivery(
            reminder.cycle,
            None,
        )

    with pytest.raises(Stage2EvidenceError):
        derive_stage2_condition_observation(
            condition=control.condition,
            schedule=control.schedule,
            invocation_decision=control.invocation_decision,
            boundary_event=control.boundary_event,
            window=control.window,
            cycle=fixed.cycle,
        )
    with pytest.raises(Stage2EvidenceError):
        derive_stage2_condition_observation(
            condition=fixed.condition,
            schedule=fixed.schedule,
            invocation_decision=fixed.invocation_decision,
            boundary_event=fixed.boundary_event,
            window=fixed.window,
        )
    with pytest.raises(Stage2EvidenceError):
        derive_stage2_condition_observation(
            condition=retrieval.condition,
            schedule=retrieval.schedule,
            invocation_decision=retrieval.invocation_decision,
            boundary_event=retrieval.boundary_event,
            window=retrieval.window,
            cycle=retrieval.cycle,
            request=retrieval.request,
            call_receipts=retrieval.call_receipts,
        )
    with pytest.raises(Stage2EvidenceError):
        derive_stage2_condition_observation(
            condition=fixed.condition,
            schedule=fixed.schedule,
            invocation_decision=fixed.invocation_decision,
            boundary_event=fixed.boundary_event,
            window=fixed.window,
            cycle=fixed.cycle,
            request=fixed.request,
            two_phase_result=fixed.two_phase_result,
            call_receipts=list(fixed.call_receipts),  # type: ignore[arg-type]
        )
