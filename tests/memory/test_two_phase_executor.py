from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import cast
from uuid import UUID

import pytest
from pydantic import ValidationError
from tests.repository.conformance import (
    CYCLE_NOW,
    CYCLE_RUN_ID,
    CycleContext,
    begin_cycle_context,
    cycle_commit_command,
    cycle_grounding_config,
)

import saliencegate.memory.two_phase as two_phase_module
from saliencegate.domain import (
    BudgetAmounts,
    ClaimKind,
    EventPhase,
    EventType,
    EvidenceReference,
    EvidenceSource,
    InterventionAction,
    MemoryCreate,
    MemoryDelta,
    MemoryIdAssignment,
    MemoryKind,
    NormalizedTraceEventDraft,
    PayloadDigest,
    PayloadDigestAlgorithm,
    PrivateStatusReplacement,
    ReasonCode,
    TextSpan,
    TraceEvent,
    TrustLabel,
    ValidityState,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.intervention import (
    GroundingPipeline,
    GroundingReceipt,
    GroundingState,
    ProposalParseStatus,
    ProposedClaim,
)
from saliencegate.memory.materialize import (
    MaterializedBankOperations,
    OperationMaterializationRequest,
    materialize_bank_operations,
)
from saliencegate.memory.proposals import (
    INTERVENTION_OUTPUT_SCHEMA_VERSION,
    MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
    BankOperationsProposal,
    DeleteMemory,
    InterventionSelectionOutput,
    SaveKnowledge,
    UpdatePrivateStatus,
)
from saliencegate.memory.two_phase import (
    PaperTwoPhaseCycleExecutor,
    RepositoryOperationMaterializer,
    TwoPhaseExecutionError,
)
from saliencegate.models import TwoPhaseReplayClient, TwoPhaseReplayRecord
from saliencegate.ports.model_calls import (
    CanonicalUsageProvenance,
    ProviderUsageProvenance,
    StructuredCallClient,
    StructuredCallParseStatus,
    StructuredCallPhase,
    StructuredCallRequest,
    StructuredCallResult,
    StructuredCallStatus,
    StructuredCallUsage,
    StructuredPhaseOutput,
)
from saliencegate.ports.repository import CycleReceipt, RunRepository
from saliencegate.ports.trajectory import (
    EventTextSelector,
    LogicalMessageBinding,
    LogicalMessageRole,
    TrajectoryPrefixRequest,
    bind_persisted_trajectory_event,
    resolve_trajectory_prefix,
)
from saliencegate.ports.two_phase import (
    CallReceipt,
    OperationMaterializer,
    TwoPhaseBoundaryError,
    TwoPhaseCallPolicy,
    TwoPhaseCycleExecutor,
    TwoPhaseCycleFailure,
    TwoPhaseCycleRequest,
    TwoPhaseCycleResult,
    TwoPhaseFailureReason,
    TwoPhaseModelProfile,
    TwoPhaseUsage,
    validated_two_phase_cycle_request,
)
from saliencegate.prompts import PAPER_TWO_PHASE_V1
from saliencegate.prompts.contracts import (
    ActiveBankPromptView,
    BankViewKind,
    BuiltPrompt,
    build_active_bank_prompt_view,
)
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.runtime.message_window import MessageWindow, project_message_window
from saliencegate.security import InstallationKey

SEED_KNOWLEDGE_ID = UUID("00000000-0000-4000-8000-000000005101")
SEED_STATUS_ID = UUID("00000000-0000-4000-8000-000000005102")
CREATED_KNOWLEDGE_ID = UUID("00000000-0000-4000-8000-000000005103")
REPLACEMENT_STATUS_ID = UUID("00000000-0000-4000-8000-000000005104")
UNUSED_ASSIGNMENT_ID = UUID("00000000-0000-4000-8000-000000005105")
DELTA_ID = UUID("00000000-0000-4000-8000-000000005106")
INTERVENTION_ID = UUID("00000000-0000-4000-8000-000000005107")
REPLAY_ID = "two-phase-success/v1"

_FIXTURE_DIGEST_DOMAIN = "saliencegate:model:two-phase-replay-fixture:v1"
_COMPLETION_DIGEST_DOMAIN = "saliencegate:test:two-phase-executor-completion:v1"


def _id_factory():
    identifiers = itertools.count(0x5200)
    return lambda: UUID(f"00000000-0000-4000-8000-{next(identifiers):012x}")


def _event_evidence(event: TraceEvent) -> EvidenceReference:
    return EvidenceReference(
        source=EvidenceSource.EVENT,
        source_id=event.event_id,
        field_path="/payload/message",
    )


def _memory_evidence(memory_id: UUID) -> EvidenceReference:
    return EvidenceReference(
        source=EvidenceSource.MEMORY,
        source_id=memory_id,
        revision=1,
        field_path="/content",
    )


@dataclass(frozen=True, slots=True)
class _Case:
    repository: MemoryRunRepository
    context: CycleContext
    running: CycleReceipt
    window: MessageWindow
    current_bank: ActiveBankPromptView
    grounding_state: GroundingState


class _RepositoryMaterializer:
    def __init__(self, repository: RunRepository) -> None:
        self._repository = repository
        self.requests: list[OperationMaterializationRequest] = []

    async def materialize(
        self,
        request: OperationMaterializationRequest,
    ) -> MaterializedBankOperations:
        self.requests.append(request)
        return await materialize_bank_operations(request, repository=self._repository)


@dataclass(slots=True)
class _CapturingReplayClient:
    replay: TwoPhaseReplayClient
    requests: list[StructuredCallRequest] = field(default_factory=list)

    async def generate(self, request: StructuredCallRequest) -> StructuredCallResult:
        self.requests.append(request)
        return await self.replay.generate(request)


@dataclass(slots=True)
class _StaticClient:
    result: object
    requests: list[StructuredCallRequest] = field(default_factory=list)

    async def generate(self, request: StructuredCallRequest) -> StructuredCallResult:
        self.requests.append(request)
        return cast(StructuredCallResult, self.result)


@dataclass(frozen=True, slots=True)
class _PreparedExecution:
    executor: PaperTwoPhaseCycleExecutor
    request: TwoPhaseCycleRequest
    materializer: _RepositoryMaterializer
    client: _CapturingReplayClient
    expected_materialization: MaterializedBankOperations
    expected_calls: tuple[StructuredCallRequest, StructuredCallRequest]
    expected_results: tuple[StructuredCallResult, StructuredCallResult]
    expected_intervention_prompt: BuiltPrompt


async def _running_case() -> _Case:
    repository = MemoryRunRepository(
        installation_key=InstallationKey(b"x" * 32),
        id_factory=_id_factory(),
    )
    await repository.append(
        NormalizedTraceEventDraft(
            run_id=CYCLE_RUN_ID,
            source_event_id="two-phase-run-start",
            timestamp=CYCLE_NOW,
            event_type=EventType.RUN_START,
            phase=EventPhase.INITIALIZATION,
            payload={"task": "Keep verified release constraints current."},
            source_adapter="two-phase-fixture/v1",
            trust_label=TrustLabel.UNTRUSTED_TASK_INPUT,
        )
    )
    first = await begin_cycle_context(repository, ordinal=1)
    await repository.reserve_cycle(first.reserve)
    await repository.mark_cycle_running(first.start)
    evidence = (_event_evidence(first.event),)
    seed_delta = MemoryDelta(
        delta_id=UUID("00000000-0000-4000-8000-000000005100"),
        run_id=CYCLE_RUN_ID,
        creates=(
            MemoryCreate(
                handle="seed-knowledge",
                kind=MemoryKind.KNOWLEDGE,
                content="The obsolete deployment constraint must be removed.",
                provenance=evidence,
                confidence=1.0,
                trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
            ),
        ),
        private_status_replacement=PrivateStatusReplacement(
            replacement=MemoryCreate(
                handle="seed-status",
                kind=MemoryKind.PRIVATE_STATUS,
                content="The previous private subgoal is still open.",
                provenance=evidence,
                confidence=1.0,
                trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
            ),
        ),
        created_at=first.commit_time,
    )
    await repository.commit_cycle(
        cycle_commit_command(
            first,
            delta=seed_delta,
            assignments=(
                MemoryIdAssignment(
                    handle="seed-knowledge",
                    memory_id=SEED_KNOWLEDGE_ID,
                ),
                MemoryIdAssignment(handle="seed-status", memory_id=SEED_STATUS_ID),
            ),
        )
    )

    context = await begin_cycle_context(repository, ordinal=2)
    reservation = BudgetAmounts(
        model_calls=2,
        input_tokens=500,
        output_tokens=200,
        canonical_token_equivalents=700,
        latency_us=500_000,
        interventions=1,
        schema_repairs=0,
    )
    await repository.reserve_cycle(context.reserve.model_copy(update={"reservation": reservation}))
    running = await repository.mark_cycle_running(context.start)

    entries = tuple(
        entry for entry in await repository.ledger(CYCLE_RUN_ID) if type(entry.record) is TraceEvent
    )
    assert len(entries) == 3
    bindings = (
        bind_persisted_trajectory_event(
            entries[0],
            task_description=EventTextSelector(field_path="/payload/task"),
        ),
        bind_persisted_trajectory_event(entries[1]),
        bind_persisted_trajectory_event(
            entries[2],
            logical_messages=(
                LogicalMessageBinding(
                    role=LogicalMessageRole.USER,
                    selector=EventTextSelector(field_path="/payload/message"),
                ),
            ),
        ),
    )
    prefix = await resolve_trajectory_prefix(
        repository,
        TrajectoryPrefixRequest(
            schema_version="trajectory-prefix-request/v1",
            run_id=CYCLE_RUN_ID,
            boundary_event_sequence=context.event.sequence,
            bindings=bindings,
        ),
    )
    window = await project_message_window(repository, prefix)
    snapshot = await repository.snapshot(CYCLE_RUN_ID)
    records = tuple(
        sorted(
            (
                record
                for record in snapshot.records
                if record.validity is ValidityState.ACTIVE
                and (record.expires_at is None or record.expires_at > context.commit_time)
            ),
            key=lambda record: (record.kind.value, str(record.memory_id)),
        )
    )
    current_bank = build_active_bank_prompt_view(
        kind=BankViewKind.CURRENT,
        run_id=CYCLE_RUN_ID,
        as_of=context.commit_time,
        source_projection_digest=snapshot.projection_digest,
        records=records,
    )
    grounding_state = GroundingState(
        schema_version="1.0",
        events=tuple(cast(TraceEvent, entry.record) for entry in entries),
        memories=records,
        reminder_history=(),
    )
    return _Case(
        repository=repository,
        context=context,
        running=running,
        window=window,
        current_bank=current_bank,
        grounding_state=grounding_state,
    )


def _model_profile() -> TwoPhaseModelProfile:
    identity = PAPER_TWO_PHASE_V1.identity
    return TwoPhaseModelProfile(
        schema_version="two-phase-model-profile/v1",
        profile_id="openai-offline-two-phase/v1",
        model_id="openai-compatible-replay/v1",
        prompt_bundle_id=identity.bundle_id,
        prompt_bundle_digest=identity.bundle_digest,
    )


def _call_policy() -> TwoPhaseCallPolicy:
    return TwoPhaseCallPolicy(
        schema_version="two-phase-call-policy/v1",
        max_model_calls=2,
        max_schema_repairs=0,
        client_retries=0,
        max_provider_input_tokens=1_000,
        max_provider_output_tokens=1_000,
        max_total_latency_us=1_000_000,
        max_call_latency_us=500_000,
    )


def _cycle_request(
    case: _Case,
    *,
    assigned_memory_ids: tuple[UUID, ...],
) -> TwoPhaseCycleRequest:
    return TwoPhaseCycleRequest(
        schema_version="two-phase-cycle-request/v1",
        cycle_receipt=case.running,
        window=case.window,
        current_bank=case.current_bank,
        grounding_state=case.grounding_state,
        delta_id=DELTA_ID,
        assigned_memory_ids=assigned_memory_ids,
        intervention_id=INTERVENTION_ID,
        created_at=case.context.commit_time,
    )


def _structured_request(
    cycle_request: TwoPhaseCycleRequest,
    profile: TwoPhaseModelProfile,
    prompt: BuiltPrompt,
    *,
    model_call_index: int,
) -> StructuredCallRequest:
    return StructuredCallRequest(
        schema_version="structured-call-request/v1",
        run_id=CYCLE_RUN_ID,
        cycle_id=cycle_request.cycle_receipt.cycle.cycle_id,
        model_call_index=model_call_index,
        phase=prompt.template.phase,
        attempt=0,
        model_id=profile.model_id,
        prompt_template_id=prompt.template.template_id,
        prompt_template_digest=prompt.template.template_digest,
        response_schema_version=prompt.template.response_schema_version,
        payload=prompt.request_payload.as_json_object(),
    )


def _completion_digest(output: StructuredPhaseOutput) -> PayloadDigest:
    return PayloadDigest(
        algorithm=PayloadDigestAlgorithm.SYNTHETIC_SHA256,
        value=length_prefixed_sha256(
            canonical_json(output),
            domain=_COMPLETION_DIGEST_DOMAIN,
        ),
    )


def _completed_result(
    request: StructuredCallRequest,
    output: StructuredPhaseOutput,
    *,
    provider_input_tokens: int | None,
    provider_output_tokens: int | None,
    latency_us: int,
    canonical_input_tokens: int | None = None,
    canonical_output_tokens: int | None = None,
) -> StructuredCallResult:
    completion = canonical_json(output)
    canonical_usage: dict[str, object] = {}
    if canonical_input_tokens is not None or canonical_output_tokens is not None:
        canonical_usage = {
            "canonical_input_tokens": canonical_input_tokens,
            "canonical_output_tokens": canonical_output_tokens,
            "canonical_usage_provenance": CanonicalUsageProvenance.REPLAY_ATTESTED,
            "local_counter_id": "deterministic-model-token-counter",
            "local_counter_version": "fixed-count-fixture/v1",
            "local_counter_configuration_digest": "d" * 64,
            "local_counter_model_id": request.model_id,
        }
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
        completion_digest=_completion_digest(output),
        completion_byte_count=len(completion),
        usage=StructuredCallUsage(
            schema_version="structured-call-usage/v1",
            provider_input_tokens=provider_input_tokens,
            provider_output_tokens=provider_output_tokens,
            provider_usage_provenance=(
                ProviderUsageProvenance.REPLAY_ATTESTED
                if provider_input_tokens is not None
                else ProviderUsageProvenance.UNAVAILABLE
            ),
            latency_us=latency_us,
            **canonical_usage,
        ),
    )


def _fixture_digest(results: tuple[StructuredCallResult, ...]) -> str:
    material = tuple(
        {
            "schema_version": "two-phase-replay-record/v1",
            "record_type": "two_phase_replay_response",
            "replay_version": "two-phase-replay/v1",
            "ordinal": ordinal,
            "request_digest": result.request_digest,
            "model_call_index": result.model_call_index,
            "phase": result.phase.value,
            "attempt": result.attempt,
            "response_schema_version": result.response_schema_version,
            "call_digest": result.call_digest,
        }
        for ordinal, result in enumerate(results, start=1)
    )
    return length_prefixed_sha256(
        REPLAY_ID,
        canonical_json(material),
        domain=_FIXTURE_DIGEST_DOMAIN,
    )


def _replay_client(
    results: tuple[StructuredCallResult, StructuredCallResult],
) -> _CapturingReplayClient:
    fixture_digest = _fixture_digest(results)
    records = tuple(
        TwoPhaseReplayRecord(
            replay_id=REPLAY_ID,
            fixture_digest=fixture_digest,
            ordinal=ordinal,
            request_digest=result.request_digest,
            model_call_index=result.model_call_index,
            phase=result.phase,
            attempt=result.attempt,
            response_schema_version=result.response_schema_version,
            result=result,
        )
        for ordinal, result in enumerate(results, start=1)
    )
    return _CapturingReplayClient(TwoPhaseReplayClient(records, replay_id=REPLAY_ID))


async def _prepare(
    case: _Case,
    *,
    operations: BankOperationsProposal,
    selection: InterventionSelectionOutput,
    assignment_pool: tuple[UUID, ...],
) -> _PreparedExecution:
    profile = _model_profile()
    policy = _call_policy()
    cycle_request = _cycle_request(case, assigned_memory_ids=assignment_pool)
    memory_prompt = PAPER_TWO_PHASE_V1.build_memory_edit(
        window=case.window,
        bank=case.current_bank,
    )
    memory_request = _structured_request(
        cycle_request,
        profile,
        memory_prompt,
        model_call_index=0,
    )
    memory_result = _completed_result(
        memory_request,
        operations,
        provider_input_tokens=21,
        provider_output_tokens=5,
        latency_us=101,
    )
    write_count = sum(
        type(operation) in (SaveKnowledge, UpdatePrivateStatus)
        for operation in operations.operations
    )
    materialization_request = OperationMaterializationRequest(
        schema_version="operation-materialization-request/v1",
        cycle_receipt=case.running,
        proposal=operations,
        delta_id=DELTA_ID,
        created_at=case.context.commit_time,
        assigned_memory_ids=assignment_pool[:write_count],
    )
    expected_materialization = await materialize_bank_operations(
        materialization_request,
        repository=case.repository,
    )
    candidate_bank = build_active_bank_prompt_view(
        kind=BankViewKind.CANDIDATE_POST_DELTA,
        run_id=CYCLE_RUN_ID,
        as_of=case.context.commit_time,
        source_projection_digest=expected_materialization.preview_projection_digest,
        records=expected_materialization.active_bank,
    )
    intervention_prompt = PAPER_TWO_PHASE_V1.build_intervention(
        window=case.window,
        bank=candidate_bank,
    )
    intervention_request = _structured_request(
        cycle_request,
        profile,
        intervention_prompt,
        model_call_index=1,
    )
    intervention_result = _completed_result(
        intervention_request,
        selection,
        provider_input_tokens=34,
        provider_output_tokens=8,
        latency_us=202,
    )
    results = (memory_result, intervention_result)
    client = _replay_client(results)
    materializer = _RepositoryMaterializer(case.repository)
    executor = PaperTwoPhaseCycleExecutor(
        materializer=materializer,
        client=client,
        prompt_bundle=PAPER_TWO_PHASE_V1,
        grounding_pipeline=GroundingPipeline(cycle_grounding_config()),
        model_profile=profile,
        call_policy=policy,
    )
    return _PreparedExecution(
        executor=executor,
        request=cycle_request,
        materializer=materializer,
        client=client,
        expected_materialization=expected_materialization,
        expected_calls=(memory_request, intervention_request),
        expected_results=results,
        expected_intervention_prompt=intervention_prompt,
    )


def _executor_with(
    prepared: _PreparedExecution,
    *,
    policy: TwoPhaseCallPolicy | None = None,
    client: StructuredCallClient | None = None,
    materializer: OperationMaterializer | None = None,
) -> PaperTwoPhaseCycleExecutor:
    return PaperTwoPhaseCycleExecutor(
        materializer=prepared.materializer if materializer is None else materializer,
        client=prepared.client if client is None else client,
        prompt_bundle=PAPER_TWO_PHASE_V1,
        grounding_pipeline=GroundingPipeline(cycle_grounding_config()),
        model_profile=_model_profile(),
        call_policy=_call_policy() if policy is None else policy,
    )


def _operations(case: _Case, scenario: str) -> BankOperationsProposal:
    evidence = (_event_evidence(case.context.event),)
    if scenario == "noop":
        operations = ()
    elif scenario == "create":
        operations = (
            SaveKnowledge(
                operation="save_knowledge",
                content="Keep the newly verified deployment requirement.",
                evidence=evidence,
                confidence=1.0,
            ),
        )
    elif scenario == "status":
        operations = (
            UpdatePrivateStatus(
                operation="update_private_status",
                content="Verify the release before publishing it.",
                evidence=evidence,
                confidence=0.9,
            ),
        )
    else:
        assert scenario == "delete"
        operations = (
            DeleteMemory(
                operation="delete_memory",
                memory_id=SEED_KNOWLEDGE_ID,
                expected_revision=1,
            ),
        )
    return BankOperationsProposal(
        schema_version=MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
        operations=operations,
    )


def _selection(*, remind_created_memory: bool) -> InterventionSelectionOutput:
    return InterventionSelectionOutput(
        schema_version=INTERVENTION_OUTPUT_SCHEMA_VERSION,
        action=(InterventionAction.REMIND if remind_created_memory else InterventionAction.SILENCE),
        claims=(
            (
                ProposedClaim(
                    kind=ClaimKind.REQUIREMENT,
                    evidence=_memory_evidence(CREATED_KNOWLEDGE_ID),
                ),
            )
            if remind_created_memory
            else ()
        ),
        confidence=1.0,
    )


def _invalid_candidate_selection(case: _Case, violation: str) -> InterventionSelectionOutput:
    if violation == "event":
        evidence = _event_evidence(case.context.event)
        kind = ClaimKind.REQUIREMENT
    else:
        reference = _memory_evidence(CREATED_KNOWLEDGE_ID)
        if violation == "span":
            evidence = reference.model_copy(update={"span": TextSpan(start_byte=0, end_byte=1)})
            kind = ClaimKind.REQUIREMENT
        elif violation == "stale_revision":
            evidence = reference.model_copy(update={"revision": 2})
            kind = ClaimKind.REQUIREMENT
        else:
            assert violation == "wrong_kind"
            evidence = reference
            kind = ClaimKind.DIAGNOSIS
    return InterventionSelectionOutput(
        schema_version=INTERVENTION_OUTPUT_SCHEMA_VERSION,
        action=InterventionAction.REMIND,
        claims=(ProposedClaim(kind=kind, evidence=evidence),),
        confidence=1.0,
    )


async def _repository_state(repository: RunRepository) -> tuple[object, ...]:
    return (
        await repository.ledger(CYCLE_RUN_ID),
        await repository.snapshot(CYCLE_RUN_ID),
        await repository.budget_snapshot(CYCLE_RUN_ID),
    )


@pytest.mark.asyncio
async def test_noop_and_silence_preserve_the_authoritative_repository() -> None:
    case = await _running_case()
    prepared = await _prepare(
        case,
        operations=_operations(case, "noop"),
        selection=_selection(remind_created_memory=False),
        assignment_pool=(UNUSED_ASSIGNMENT_ID,),
    )
    before = await _repository_state(case.repository)

    outcome = await prepared.executor.execute(prepared.request)

    assert type(outcome) is TwoPhaseCycleResult
    assert outcome.outcome == "completed"
    assert not outcome.validated_delta.creates
    assert not outcome.validated_delta.invalidations
    assert outcome.validated_delta.private_status_replacement is None
    assert outcome.intervention.action is InterventionAction.SILENCE
    assert outcome.materialization.active_bank == tuple(
        record.to_memory_record() for record in case.current_bank.records
    )
    assert await _repository_state(case.repository) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ("create", "status", "delete"))
async def test_materializes_each_edit_and_sends_the_exact_candidate_bank_to_phase_two(
    scenario: str,
) -> None:
    case = await _running_case()
    first_assignment = CREATED_KNOWLEDGE_ID if scenario == "create" else REPLACEMENT_STATUS_ID
    prepared = await _prepare(
        case,
        operations=_operations(case, scenario),
        selection=_selection(remind_created_memory=scenario == "create"),
        assignment_pool=(first_assignment, UNUSED_ASSIGNMENT_ID),
    )
    before = await _repository_state(case.repository)

    outcome = await prepared.executor.execute(prepared.request)

    assert type(outcome) is TwoPhaseCycleResult
    assert outcome.materialization == prepared.expected_materialization
    assert prepared.client.requests == list(prepared.expected_calls)
    phase_two = prepared.client.requests[1]
    assert canonical_json(phase_two.payload) == canonical_json(
        prepared.expected_intervention_prompt.request_payload.as_json_object()
    )
    assert phase_two.request_digest == prepared.expected_calls[1].request_digest
    assert outcome.call_receipts[1].request_digest == phase_two.request_digest
    assert outcome.call_receipts[1].bank_view_digest == (
        prepared.expected_intervention_prompt.bank_view_digest
    )
    assert await _repository_state(case.repository) == before

    if scenario == "create":
        assert outcome.memory_id_assignments == (
            MemoryIdAssignment(
                handle="memory-edit-operation/v1/0001",
                memory_id=CREATED_KNOWLEDGE_ID,
            ),
        )
        assert outcome.validated_delta.creates[0].kind is MemoryKind.KNOWLEDGE
        assert outcome.intervention.action is InterventionAction.REMIND
    elif scenario == "status":
        replacement = outcome.validated_delta.private_status_replacement
        assert replacement is not None
        assert replacement.expected_memory_id == SEED_STATUS_ID
        assert replacement.expected_revision == 1
        assert outcome.memory_id_assignments[0].memory_id == REPLACEMENT_STATUS_ID
        assert tuple(
            record.memory_id
            for record in outcome.materialization.active_bank
            if record.kind is MemoryKind.PRIVATE_STATUS
        ) == (REPLACEMENT_STATUS_ID,)
    else:
        assert tuple(item.memory_id for item in outcome.validated_delta.invalidations) == (
            SEED_KNOWLEDGE_ID,
        )
        assert SEED_KNOWLEDGE_ID not in {
            record.memory_id for record in outcome.materialization.active_bank
        }


@pytest.mark.asyncio
async def test_same_cycle_memory_citation_and_call_accounting_are_content_bound() -> None:
    case = await _running_case()
    prepared = await _prepare(
        case,
        operations=_operations(case, "create"),
        selection=_selection(remind_created_memory=True),
        assignment_pool=(CREATED_KNOWLEDGE_ID, UNUSED_ASSIGNMENT_ID),
    )

    outcome = await prepared.executor.execute(prepared.request)

    assert type(outcome) is TwoPhaseCycleResult
    assert outcome.intervention.action is InterventionAction.REMIND
    assert outcome.intervention.cited_memory_ids == (CREATED_KNOWLEDGE_ID,)
    assert outcome.intervention.claims[0].evidence[0] == _memory_evidence(CREATED_KNOWLEDGE_ID)
    assert outcome.grounding_receipt.model_call_index == 1
    assert outcome.grounding_receipt.model_call_digest == prepared.expected_results[1].call_digest
    assert tuple(receipt.model_call_index for receipt in outcome.call_receipts) == (0, 1)
    assert tuple(receipt.phase for receipt in outcome.call_receipts) == (
        StructuredCallPhase.MEMORY_EDIT,
        StructuredCallPhase.INTERVENTION,
    )
    assert tuple(receipt.call_digest for receipt in outcome.call_receipts) == tuple(
        result.call_digest for result in prepared.expected_results
    )
    assert tuple(receipt.usage for receipt in outcome.call_receipts) == tuple(
        result.usage for result in prepared.expected_results
    )
    assert outcome.usage.model_calls == 2
    assert outcome.usage.provider_input_tokens == 55
    assert outcome.usage.provider_output_tokens == 13
    assert outcome.usage.latency_us == 303
    assert outcome.usage.schema_repairs == 0
    assert outcome.usage.canonical_token_equivalents is None
    assert prepared.client.replay.remaining_responses == 0
    assert len(prepared.materializer.requests) == 1


def test_profile_and_policy_are_strict_frozen_and_digest_bound() -> None:
    profile = _model_profile()
    policy = _call_policy()

    with pytest.raises(ValidationError):
        profile.model_id = "changed-model/v1"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        policy.max_model_calls = 3  # type: ignore[misc]
    with pytest.raises(ValidationError):
        TwoPhaseModelProfile.model_validate(
            profile.model_dump(mode="python") | {"profile_digest": "0" * 64}
        )
    with pytest.raises(ValidationError):
        TwoPhaseCallPolicy.model_validate(
            policy.model_dump(mode="python") | {"policy_digest": "0" * 64}
        )
    with pytest.raises(ValidationError):
        TwoPhaseModelProfile.model_validate(profile.model_dump(mode="python") | {"extra": True})
    with pytest.raises(ValidationError):
        TwoPhaseCallPolicy(
            **policy.model_dump(
                mode="python",
                exclude={"policy_digest", "max_model_calls"},
            ),
            max_model_calls=3,
        )


@pytest.mark.asyncio
async def test_cycle_request_rejects_cross_run_boundary_bank_state_and_identity_mismatches() -> (
    None
):
    case = await _running_case()
    valid = _cycle_request(case, assigned_memory_ids=(CREATED_KNOWLEDGE_ID,))
    window_values = case.window.model_dump(mode="python", exclude={"window_digest"})
    window_values["boundary_event_sequence"] = case.window.boundary_event_sequence + 1
    wrong_boundary = MessageWindow(**window_values)
    wrong_kind_bank = build_active_bank_prompt_view(
        kind=BankViewKind.CANDIDATE_POST_DELTA,
        run_id=CYCLE_RUN_ID,
        as_of=case.context.commit_time,
        source_projection_digest=case.current_bank.source_projection_digest,
        records=tuple(item.to_memory_record() for item in case.current_bank.records),
    )
    other_run = UUID("00000000-0000-4000-8000-000000005199")
    cross_run_bank = build_active_bank_prompt_view(
        kind=BankViewKind.CURRENT,
        run_id=other_run,
        as_of=case.context.commit_time,
        source_projection_digest=case.current_bank.source_projection_digest,
        records=(),
    )
    missing_bank_state = GroundingState(
        schema_version="1.0",
        events=case.grounding_state.events,
        memories=(),
        reminder_history=case.grounding_state.reminder_history,
    )
    base = valid.model_dump(mode="python", exclude={"request_digest"})
    variants = (
        {"window": wrong_boundary},
        {"current_bank": wrong_kind_bank},
        {"current_bank": cross_run_bank},
        {"grounding_state": missing_bank_state},
        {"assigned_memory_ids": (DELTA_ID,)},
        {"intervention_id": DELTA_ID},
    )

    for change in variants:
        with pytest.raises(ValidationError, match="views do not match"):
            TwoPhaseCycleRequest.model_validate(base | change)

    forged = valid.model_copy(update={"request_digest": "0" * 64})
    with pytest.raises(TwoPhaseBoundaryError):
        validated_two_phase_cycle_request(forged)


@pytest.mark.asyncio
async def test_receipt_result_protocols_and_repr_are_strict_content_boundaries() -> None:
    case = await _running_case()
    prepared = await _prepare(
        case,
        operations=_operations(case, "create"),
        selection=_selection(remind_created_memory=True),
        assignment_pool=(CREATED_KNOWLEDGE_ID,),
    )

    outcome = await prepared.executor.execute(prepared.request)

    assert type(outcome) is TwoPhaseCycleResult
    receipt = outcome.call_receipts[0]
    with pytest.raises(ValidationError):
        CallReceipt.model_validate(receipt.model_dump(mode="python") | {"receipt_digest": "0" * 64})
    with pytest.raises(ValidationError):
        TwoPhaseCycleResult.model_validate(
            outcome.model_dump(mode="python") | {"result_digest": "0" * 64}
        )
    with pytest.raises(ValidationError):
        receipt.model_call_index = 7  # type: ignore[misc]
    with pytest.raises(ValidationError):
        outcome.model_id = "changed/v1"  # type: ignore[misc]

    assert isinstance(prepared.executor, TwoPhaseCycleExecutor)
    assert isinstance(prepared.materializer, OperationMaterializer)
    assert isinstance(RepositoryOperationMaterializer(case.repository), OperationMaterializer)
    assert isinstance(prepared.client, StructuredCallClient)
    assert not isinstance(object(), TwoPhaseCycleExecutor)
    for value in (prepared.request, receipt, outcome):
        rendered = repr(value)
        assert "Keep verified release constraints current." not in rendered
        assert "Keep the newly verified deployment requirement." not in rendered


@pytest.mark.asyncio
async def test_unavailable_provider_usage_remains_unknown_instead_of_becoming_zero() -> None:
    case = await _running_case()
    cycle_request = _cycle_request(case, assigned_memory_ids=())
    prompt = PAPER_TWO_PHASE_V1.build_memory_edit(
        window=case.window,
        bank=case.current_bank,
    )
    call_request = _structured_request(
        cycle_request,
        _model_profile(),
        prompt,
        model_call_index=0,
    )
    result = _completed_result(
        call_request,
        _operations(case, "noop"),
        provider_input_tokens=None,
        provider_output_tokens=None,
        latency_us=77,
    )
    receipt = CallReceipt.from_call(prompt, call_request, result)

    usage = TwoPhaseUsage.from_receipts((receipt,))

    assert usage.model_calls == 1
    assert usage.provider_input_tokens is None
    assert usage.provider_output_tokens is None
    assert usage.canonical_token_equivalents is None
    assert usage.latency_us == 77

    counted_result = _completed_result(
        call_request,
        _operations(case, "noop"),
        provider_input_tokens=None,
        provider_output_tokens=None,
        latency_us=77,
        canonical_input_tokens=9,
        canonical_output_tokens=4,
    )
    counted_receipt = CallReceipt.from_call(prompt, call_request, counted_result)
    counted = TwoPhaseUsage.from_receipts((counted_receipt,))
    assert counted.canonical_input_tokens == 9
    assert counted.canonical_output_tokens == 4
    assert counted.canonical_token_equivalents == 13

    partial_result = _completed_result(
        call_request,
        _operations(case, "noop"),
        provider_input_tokens=None,
        provider_output_tokens=None,
        latency_us=77,
        canonical_input_tokens=9,
        canonical_output_tokens=None,
    )
    partial_receipt = CallReceipt.from_call(prompt, call_request, partial_result)
    partial = TwoPhaseUsage.from_receipts((partial_receipt,))
    assert partial.canonical_input_tokens == 9
    assert partial.canonical_output_tokens is None
    assert partial.canonical_token_equivalents is None
    with pytest.raises(ValidationError):
        TwoPhaseUsage(
            model_calls=1,
            provider_input_tokens=None,
            provider_output_tokens=0,
            canonical_token_equivalents=None,
            latency_us=77,
            schema_repairs=0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("violation", ("event", "span", "stale_revision", "wrong_kind"))
async def test_phase_two_silences_every_non_exact_candidate_memory_claim(violation: str) -> None:
    case = await _running_case()
    prepared = await _prepare(
        case,
        operations=_operations(case, "create"),
        selection=_invalid_candidate_selection(case, violation),
        assignment_pool=(CREATED_KNOWLEDGE_ID,),
    )
    before = await _repository_state(case.repository)

    outcome = await prepared.executor.execute(prepared.request)

    assert type(outcome) is TwoPhaseCycleResult
    assert outcome.materialization == prepared.expected_materialization
    assert outcome.intervention.action is InterventionAction.SILENCE
    assert outcome.intervention.reason_code is (
        ReasonCode.SCHEMA_INVALID
        if violation in ("event", "span")
        else ReasonCode.INVALID_PROVENANCE
    )
    assert outcome.grounding_receipt.parse_status is (
        ProposalParseStatus.SCHEMA_INVALID
        if violation in ("event", "span")
        else ProposalParseStatus.VALID
    )
    assert len(outcome.call_receipts) == 2
    assert await _repository_state(case.repository) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy_change", "expected_calls"),
    (
        ({"max_provider_input_tokens": 20}, 1),
        ({"max_call_latency_us": 100}, 1),
        ({"max_total_latency_us": 250, "max_call_latency_us": 250}, 2),
    ),
)
async def test_call_policy_stops_token_and_latency_overflow_at_the_observed_phase(
    policy_change: dict[str, int],
    expected_calls: int,
) -> None:
    case = await _running_case()
    prepared = await _prepare(
        case,
        operations=_operations(case, "create"),
        selection=_selection(remind_created_memory=True),
        assignment_pool=(CREATED_KNOWLEDGE_ID,),
    )
    values = _call_policy().model_dump(mode="python", exclude={"policy_digest"})
    values.update(policy_change)
    executor = _executor_with(prepared, policy=TwoPhaseCallPolicy(**values))

    outcome = await executor.execute(prepared.request)

    assert type(outcome) is TwoPhaseCycleFailure
    assert outcome.reason is TwoPhaseFailureReason.CALL_POLICY_EXCEEDED
    assert len(outcome.call_receipts) == expected_calls
    assert len(prepared.client.requests) == expected_calls
    assert len(prepared.materializer.requests) == int(expected_calls == 2)


@pytest.mark.asyncio
async def test_materialization_projection_mismatch_fails_before_phase_two() -> None:
    case = await _running_case()
    wrong_projection = PayloadDigest(
        algorithm=case.current_bank.source_projection_digest.algorithm,
        value="0" * 64,
    )
    mismatched_bank = build_active_bank_prompt_view(
        kind=BankViewKind.CURRENT,
        run_id=CYCLE_RUN_ID,
        as_of=case.context.commit_time,
        source_projection_digest=wrong_projection,
        records=tuple(item.to_memory_record() for item in case.current_bank.records),
    )
    mismatched = replace(case, current_bank=mismatched_bank)
    prepared = await _prepare(
        mismatched,
        operations=_operations(case, "noop"),
        selection=_selection(remind_created_memory=False),
        assignment_pool=(),
    )

    outcome = await prepared.executor.execute(prepared.request)

    assert type(outcome) is TwoPhaseCycleFailure
    assert outcome.failed_phase is StructuredCallPhase.MEMORY_EDIT
    assert outcome.reason is TwoPhaseFailureReason.MATERIALIZATION_REJECTED
    assert len(prepared.client.requests) == 1


@pytest.mark.asyncio
async def test_malformed_client_result_is_sanitized_at_the_executor_boundary() -> None:
    case = await _running_case()
    prepared = await _prepare(
        case,
        operations=_operations(case, "noop"),
        selection=_selection(remind_created_memory=False),
        assignment_pool=(),
    )
    malformed = _StaticClient(object())
    executor = _executor_with(prepared, client=malformed)

    with pytest.raises(TwoPhaseExecutionError) as raised:
        await executor.execute(prepared.request)

    assert str(raised.value) == "two-phase cycle execution failed validation"
    assert len(malformed.requests) == 1
    assert not prepared.materializer.requests


@pytest.mark.asyncio
async def test_call_receipt_rejects_cross_request_or_forged_result_value_free() -> None:
    case = await _running_case()
    prepared = await _prepare(
        case,
        operations=_operations(case, "noop"),
        selection=_selection(remind_created_memory=False),
        assignment_pool=(),
    )
    memory_prompt = PAPER_TWO_PHASE_V1.build_memory_edit(
        window=case.window,
        bank=case.current_bank,
    )
    forged = prepared.expected_results[0].model_copy(update={"call_digest": "0" * 64})

    for result in (prepared.expected_results[1], forged):
        with pytest.raises(TwoPhaseBoundaryError) as raised:
            CallReceipt.from_call(memory_prompt, prepared.expected_calls[0], result)
        assert prepared.expected_calls[0].request_digest not in str(raised.value)
        assert prepared.expected_results[0].call_digest not in str(raised.value)


@pytest.mark.asyncio
async def test_result_rebinds_grounded_claims_to_the_exact_candidate_bank() -> None:
    case = await _running_case()
    prepared = await _prepare(
        case,
        operations=_operations(case, "create"),
        selection=_selection(remind_created_memory=True),
        assignment_pool=(CREATED_KNOWLEDGE_ID,),
    )
    outcome = await prepared.executor.execute(prepared.request)
    assert type(outcome) is TwoPhaseCycleResult

    changed_receipt = GroundingReceipt(
        **outcome.grounding_receipt.model_dump(
            mode="python",
            exclude={"claims"},
        ),
        claims=(
            ProposedClaim(
                kind=ClaimKind.REQUIREMENT,
                evidence=_memory_evidence(SEED_KNOWLEDGE_ID),
            ),
        ),
    )
    changed_intervention = outcome.intervention.model_copy(
        update={"grounding_receipt": changed_receipt.model_dump(mode="json")}
    )
    stale_time = outcome.intervention.model_copy(
        update={"created_at": outcome.intervention.created_at + timedelta(microseconds=1)}
    )
    base = outcome.model_dump(mode="python", exclude={"result_digest"})

    for change in (
        {
            "grounding_receipt": changed_receipt,
            "intervention": changed_intervention,
        },
        {"intervention": stale_time},
    ):
        with pytest.raises(ValidationError, match="components do not match"):
            TwoPhaseCycleResult.model_validate(base | change)


@pytest.mark.asyncio
async def test_post_call_grounding_failure_preserves_known_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = await _running_case()
    prepared = await _prepare(
        case,
        operations=_operations(case, "create"),
        selection=_selection(remind_created_memory=True),
        assignment_pool=(CREATED_KNOWLEDGE_ID,),
    )

    def reject_grounding(*_args: object, **_kwargs: object) -> None:
        raise ValueError("must remain value-free")

    monkeypatch.setattr(two_phase_module, "verify_grounded_intervention", reject_grounding)
    outcome = await prepared.executor.execute(prepared.request)

    assert type(outcome) is TwoPhaseCycleFailure
    assert outcome.failed_phase is StructuredCallPhase.INTERVENTION
    assert outcome.reason is TwoPhaseFailureReason.CALL_CONTRACT_INVALID
    assert outcome.cost_certainty == "known"
    assert len(outcome.call_receipts) == 2
    assert outcome.usage.model_calls == 2
    assert outcome.usage.provider_input_tokens == 55


@pytest.mark.asyncio
async def test_public_repository_materializer_runs_the_authoritative_preview() -> None:
    case = await _running_case()
    proposal = _operations(case, "noop")
    request = OperationMaterializationRequest(
        schema_version="operation-materialization-request/v1",
        cycle_receipt=case.running,
        proposal=proposal,
        delta_id=DELTA_ID,
        created_at=case.context.commit_time,
        assigned_memory_ids=(),
    )

    actual = await RepositoryOperationMaterializer(case.repository).materialize(request)

    assert actual.source_cycle_id == case.running.cycle.cycle_id
    assert actual.source_projection_digest == case.current_bank.source_projection_digest
    assert actual.active_bank == tuple(
        item.to_memory_record() for item in case.current_bank.records
    )


@pytest.mark.asyncio
async def test_executor_constructor_rejects_unreviewed_capabilities_and_profiles() -> None:
    case = await _running_case()
    prepared = await _prepare(
        case,
        operations=_operations(case, "noop"),
        selection=_selection(remind_created_memory=False),
        assignment_pool=(),
    )
    identity = PAPER_TWO_PHASE_V1.identity
    unreviewed_profile = TwoPhaseModelProfile(
        profile_id="unreviewed/v1",
        model_id="openai-compatible-replay/v1",
        prompt_bundle_id=identity.bundle_id,
        prompt_bundle_digest="0" * 64,
    )
    repair_policy = TwoPhaseCallPolicy(
        max_model_calls=3,
        max_schema_repairs=1,
        client_retries=0,
        max_provider_input_tokens=1_000,
        max_provider_output_tokens=1_000,
        max_total_latency_us=1_000_000,
        max_call_latency_us=500_000,
    )
    valid = {
        "materializer": prepared.materializer,
        "client": prepared.client,
        "prompt_bundle": PAPER_TWO_PHASE_V1,
        "grounding_pipeline": GroundingPipeline(cycle_grounding_config()),
        "model_profile": _model_profile(),
        "call_policy": _call_policy(),
    }

    for change in (
        {"materializer": object()},
        {"client": object()},
        {"model_profile": unreviewed_profile},
    ):
        with pytest.raises(TwoPhaseExecutionError):
            PaperTwoPhaseCycleExecutor(**(valid | change))  # type: ignore[arg-type]

    repaired = PaperTwoPhaseCycleExecutor(**(valid | {"call_policy": repair_policy}))
    assert isinstance(repaired, TwoPhaseCycleExecutor)
