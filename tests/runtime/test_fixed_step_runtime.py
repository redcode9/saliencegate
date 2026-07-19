from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast
from uuid import UUID

import pytest
from pydantic import ValidationError

import saliencegate.ports.two_phase as two_phase_module
import saliencegate.runtime as runtime_module
import saliencegate.runtime.algorithm_result as algorithm_result_module
from saliencegate.domain import (
    BudgetAmounts,
    BudgetLimits,
    BudgetSnapshot,
    ClaimKind,
    CycleRecord,
    CycleState,
    DeduplicationGuarantee,
    DeliveryState,
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
    TextSpan,
    TraceEvent,
    TrustLabel,
    canonical_digest,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.intervention import GroundingPipeline, GroundingState, ProposedClaim
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
    AdapterDeliveryRefusedError,
    DeliveryAdapter,
    DeliveryChannel,
    DeliveryEnvelope,
    DeliveryReceipt,
    DeliveryRole,
    InjectionMapping,
)
from saliencegate.ports.model_calls import (
    CanonicalUsageProvenance,
    ProviderUsageProvenance,
    StructuredCallParseStatus,
    StructuredCallPhase,
    StructuredCallRequest,
    StructuredCallResult,
    StructuredCallStatus,
    StructuredCallUsage,
    StructuredPhaseOutput,
)
from saliencegate.ports.repository import LedgerReceipt, RunRepository
from saliencegate.ports.trajectory import (
    ActionStepBinding,
    EventTextSelector,
    LogicalMessageBinding,
    LogicalMessageRole,
)
from saliencegate.ports.two_phase import (
    TwoPhaseCallPolicy,
    TwoPhaseCycleExecutor,
    TwoPhaseCycleOutcome,
    TwoPhaseCycleRequest,
    TwoPhaseCycleResult,
    TwoPhaseModelProfile,
)
from saliencegate.prompts import PAPER_TWO_PHASE_V1
from saliencegate.prompts.contracts import (
    PromptDataEnvelope,
    PromptDataSectionName,
    StructuredPromptPayload,
    parse_untrusted_prompt_data,
)
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.repository.sqlite import SQLiteRunRepository
from saliencegate.runtime.algorithm_result import (
    AlgorithmConfigurationAttestation,
    AlgorithmRunResult,
    FixedStepRecoveryResult,
    ModelTokenUsageAttestation,
    ModelTokenUsageSource,
    algorithm_result_digest,
    algorithm_runtime_uuid,
    algorithm_trace_digest,
    derive_cycle_reservation,
    fixed_step_recovery_digest,
    model_token_usage_attestation,
    semantic_projection_digests,
)
from saliencegate.runtime.fixed_step import (
    FixedStepEventInput,
    FixedStepExecutionError,
    FixedStepInputError,
    FixedStepRunner,
)
from saliencegate.runtime.fixed_step_core import (
    FixedStepTraceBoundary,
    FixedStepTraceDriver,
    FixedStepTraceInput,
    FixedStepTraceInputError,
    FixedStepTraceInvariantError,
    FixedStepTraceResult,
    FixedStepTraceSpine,
)
from saliencegate.runtime.model_token_counting import (
    DeterministicModelTokenCounter,
    ModelTokenCounterIdentity,
)
from saliencegate.security import InstallationKey, RedactionPolicy

RUN_ID = UUID("00000000-0000-4000-8000-00000000a001")
EVENT_IDS = tuple(UUID(f"00000000-0000-4000-8000-{value:012x}") for value in range(0xA011, 0xA015))
NOW = datetime(2026, 7, 12, 9, 0, tzinfo=UTC)
INSTALLATION_KEY = InstallationKey(b"f" * 32)
ADAPTER_ID = "fixed-step-refusal-fixture/v1"
_COMPLETION_DIGEST_DOMAIN = "saliencegate:test:fixed-step-completion:v1"
_REDACTED_FIXTURE_SECRET = "fixture-secret-token"


def _repository_ids() -> Callable[[], UUID]:
    values = itertools.count(0xA100)
    return lambda: UUID(f"00000000-0000-4000-8000-{next(values):012x}")


def _make_repository(kind: str, path: Path) -> RunRepository:
    redaction_policy = RedactionPolicy(literal_secrets=(_REDACTED_FIXTURE_SECRET,))
    if kind == "memory":
        return MemoryRunRepository(
            redaction_policy=redaction_policy,
            installation_key=INSTALLATION_KEY,
            id_factory=_repository_ids(),
        )
    assert kind == "sqlite"
    return SQLiteRunRepository(
        path,
        redaction_policy=redaction_policy,
        installation_key=INSTALLATION_KEY,
        id_factory=_repository_ids(),
    )


@pytest.fixture(params=("memory", "sqlite"))
def repository(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Iterator[RunRepository]:
    current = _make_repository(cast(str, request.param), tmp_path / "fixed-step.sqlite3")
    try:
        yield current
    finally:
        if isinstance(current, SQLiteRunRepository):
            current.close()


def _profile() -> TwoPhaseModelProfile:
    prompt = PAPER_TWO_PHASE_V1.identity
    return TwoPhaseModelProfile(
        schema_version="two-phase-model-profile/v1",
        profile_id="fixed-step-offline-fixture/v1",
        model_id="openai-compatible-scripted/v1",
        prompt_bundle_id=prompt.bundle_id,
        prompt_bundle_digest=prompt.bundle_digest,
    )


def _policy() -> TwoPhaseCallPolicy:
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


def _configuration(
    grounding: GroundingPipeline,
    *,
    cycle_capacity: int,
    requested_delivery_target: DeliveryTarget | None = None,
    policy_version: str = "paper-fixed-step/v1",
    model_token_counter: ModelTokenCounterIdentity | None = None,
) -> AlgorithmConfigurationAttestation:
    policy = _policy()
    reservation = derive_cycle_reservation(policy)
    return AlgorithmConfigurationAttestation(
        schema_version="algorithm-configuration-attestation/v1",
        policy_version=policy_version,
        budget_limits=BudgetLimits(
            model_calls=reservation.model_calls * cycle_capacity,
            input_tokens=reservation.input_tokens * cycle_capacity,
            output_tokens=reservation.output_tokens * cycle_capacity,
            canonical_token_equivalents=(reservation.canonical_token_equivalents * cycle_capacity),
            latency_us=reservation.latency_us * cycle_capacity,
            interventions=reservation.interventions * cycle_capacity,
            schema_repairs=reservation.schema_repairs * cycle_capacity,
            max_call_latency_us=policy.max_call_latency_us,
        ),
        cycle_reservation=reservation,
        prompt_bundle=PAPER_TWO_PHASE_V1.identity,
        model_profile=_profile(),
        call_policy=policy,
        grounding_configuration=grounding.resolved_configuration,
        model_token_counter=model_token_counter,
        requested_delivery_target=requested_delivery_target,
    )


def _prompt_envelope(request: StructuredCallRequest) -> PromptDataEnvelope:
    payload = StructuredPromptPayload.model_validate_json(canonical_json(request.payload))
    return parse_untrusted_prompt_data(payload.messages[1].content)


def _completed_result(
    request: StructuredCallRequest,
    output: StructuredPhaseOutput,
    *,
    model_token_counter: DeterministicModelTokenCounter | None = None,
) -> StructuredCallResult:
    completion = canonical_json(output)
    canonical_usage: dict[str, object] = {}
    if model_token_counter is not None:
        input_count = model_token_counter.count_input(request)
        output_count = model_token_counter.count_output(
            model_id=request.model_id,
            completion=completion.decode("utf-8", errors="strict"),
        )
        identity = input_count.counter_identity
        assert output_count.counter_identity == identity
        canonical_usage = {
            "canonical_input_tokens": input_count.token_count,
            "canonical_output_tokens": output_count.token_count,
            "canonical_usage_provenance": CanonicalUsageProvenance.LOCAL_COUNTER,
            "local_counter_id": identity.counter_id,
            "local_counter_version": identity.counter_version,
            "local_counter_configuration_digest": identity.configuration_digest,
            "local_counter_model_id": identity.model_id,
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
            provider_usage_provenance=(
                ProviderUsageProvenance.PROVIDER_REPORTED
                if model_token_counter is not None
                else ProviderUsageProvenance.REPLAY_ATTESTED
            ),
            latency_us=100 + request.model_call_index,
            **canonical_usage,
        ),
    )


class _RequestAwareClient:
    """Script valid outputs from the exact provider-visible request, not hidden IDs."""

    def __init__(
        self,
        repository: RunRepository,
        mode: Literal["reminder", "silence"],
        model_token_counter: DeterministicModelTokenCounter | None = None,
    ) -> None:
        self.repository = repository
        self.mode = mode
        self.model_token_counter = model_token_counter
        self.requests: list[StructuredCallRequest] = []
        self.dispatch_repository_states: list[tuple[BudgetSnapshot, CycleRecord]] = []
        self.phase_two_repository_states: list[tuple[int, int, int]] = []

    async def generate(self, request: StructuredCallRequest) -> StructuredCallResult:
        self.requests.append(request)
        budget = await self.repository.budget_snapshot(request.run_id)
        ledger = await self.repository.ledger(request.run_id)
        running = next(
            entry.record for entry in reversed(ledger) if type(entry.record) is CycleRecord
        )
        self.dispatch_repository_states.append((budget, running))
        envelope = _prompt_envelope(request)
        if request.phase is StructuredCallPhase.MEMORY_EDIT:
            operations: tuple[SaveKnowledge, ...] = ()
            if self.mode == "reminder":
                task = envelope.section(PromptDataSectionName.TASK)
                evidence = EvidenceReference.model_validate_json(canonical_json(task["evidence"]))
                operations = (
                    SaveKnowledge(
                        operation="save_knowledge",
                        content="Run the complete verified test suite before release.",
                        evidence=(evidence,),
                        confidence=1.0,
                    ),
                )
            return _completed_result(
                request,
                BankOperationsProposal(
                    schema_version="memory-edit-output/v1",
                    operations=operations,
                ),
                model_token_counter=self.model_token_counter,
            )

        assert request.phase is StructuredCallPhase.INTERVENTION
        snapshot = await self.repository.snapshot(request.run_id)
        self.phase_two_repository_states.append(
            (
                snapshot.ingestion_cursor,
                snapshot.memory_cursor,
                len(snapshot.records),
            )
        )
        claims: tuple[ProposedClaim, ...] = ()
        action = InterventionAction.SILENCE
        if self.mode == "reminder":
            bank = envelope.section(PromptDataSectionName.MEMORY_BANK)
            records = cast(list[Mapping[str, object]], bank["records"])
            knowledge = next(record for record in records if record["kind"] == "knowledge")
            claims = (
                ProposedClaim(
                    kind=ClaimKind.REQUIREMENT,
                    evidence=EvidenceReference(
                        source=EvidenceSource.MEMORY,
                        source_id=UUID(cast(str, knowledge["memory_id"])),
                        revision=cast(int, knowledge["revision"]),
                        field_path="/content",
                    ),
                ),
            )
            action = InterventionAction.REMIND
        return _completed_result(
            request,
            InterventionSelectionOutput(
                schema_version="intervention-output/v1",
                action=action,
                claims=claims,
                confidence=1.0,
            ),
            model_token_counter=self.model_token_counter,
        )


class _MismatchingExecutor:
    def __init__(self, inner: TwoPhaseCycleExecutor) -> None:
        self.inner = inner

    async def execute(self, request: TwoPhaseCycleRequest) -> TwoPhaseCycleOutcome:
        outcome = await self.inner.execute(request)
        assert type(outcome) is TwoPhaseCycleResult
        payload = outcome.model_dump(mode="json", exclude={"result_digest"}, warnings=False)
        payload["request_digest"] = "0" * 64
        payload["result_digest"] = two_phase_module._result_digest(payload)
        return TwoPhaseCycleResult.model_validate_json(canonical_json(payload))


class _LostDecisionAcknowledgementRepository(MemoryRunRepository):
    def __init__(self) -> None:
        super().__init__(
            redaction_policy=RedactionPolicy(literal_secrets=(_REDACTED_FIXTURE_SECRET,)),
            installation_key=INSTALLATION_KEY,
            id_factory=_repository_ids(),
        )
        self.acknowledgement_lost = False

    async def record_invocation_decision(
        self,
        decision: InvocationDecision,
    ) -> LedgerReceipt:
        receipt = await super().record_invocation_decision(decision)
        if not self.acknowledgement_lost:
            self.acknowledgement_lost = True
            raise RuntimeError("simulated lost decision acknowledgement")
        return receipt


class _DeliveryAdapter:
    def __init__(self, mode: Literal["deliver", "pre_dispatch_refusal", "unknown"]) -> None:
        self.mode = mode
        self.calls: list[DeliveryEnvelope] = []

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            schema_version="1.0",
            adapter_id=ADAPTER_ID,
            pre_action_interception=False,
            deduplicates_delivery_id=True,
            deduplication_guarantee=DeduplicationGuarantee.DURABLE_DELIVERY_ID,
            injection_mappings=(
                ()
                if self.mode == "pre_dispatch_refusal"
                else (
                    InjectionMapping(
                        channel=DeliveryChannel.PROVIDER_DATA,
                        role=DeliveryRole.DATA,
                        provider_channel="context",
                    ),
                )
            ),
        )

    async def deliver(self, delivery: DeliveryEnvelope) -> DeliveryReceipt:
        self.calls.append(delivery)
        if self.mode == "unknown":
            raise AdapterDeliveryRefusedError(ReasonCode.TARGET_UNAVAILABLE)
        return DeliveryReceipt(
            schema_version="1.0",
            delivery_id=delivery.delivery_id,
            attempt_id=delivery.attempt_id,
            attempt_number=delivery.attempt_number,
            adapter_id=delivery.adapter_id,
            target_request_id=delivery.target_request_id,
            delivered_at=delivery.created_at + timedelta(microseconds=1),
            provider_receipt_id="fixed-step-provider-receipt/v1",
        )


def _run_start(*, target_request_id: str | None = None) -> FixedStepEventInput:
    return FixedStepEventInput(
        draft=NormalizedTraceEventDraft(
            run_id=RUN_ID,
            source_event_id="fixed-step-run-start",
            timestamp=NOW,
            event_type=EventType.RUN_START,
            phase=EventPhase.INITIALIZATION,
            payload={
                "task": (
                    "Keep verified release constraints available when they matter; "
                    f"never persist {_REDACTED_FIXTURE_SECRET}."
                ),
                "step": 1,
            },
            source_adapter="fixed-step-fixture/v1",
            trust_label=TrustLabel.SYNTHETIC_FIXTURE,
        ),
        expected_event_id=EVENT_IDS[0],
        task_description=EventTextSelector(field_path="/payload/task"),
        action_step=ActionStepBinding(field_path="/payload/step"),
        target_request_id=target_request_id,
    )


def _message_event(
    ordinal: int,
    *,
    step: int,
    role: LogicalMessageRole,
) -> FixedStepEventInput:
    event_types = (
        EventType.MODEL_OUTPUT,
        EventType.TOOL_COMPLETION,
        EventType.ACTION_PROPOSAL,
    )
    phases = (
        EventPhase.POST_ACTION,
        EventPhase.ACTION_EXECUTION,
        EventPhase.PRE_ACTION,
    )
    return FixedStepEventInput(
        draft=NormalizedTraceEventDraft(
            run_id=RUN_ID,
            source_event_id=f"fixed-step-event-{ordinal}",
            timestamp=NOW + timedelta(seconds=ordinal),
            event_type=event_types[ordinal - 2],
            phase=phases[ordinal - 2],
            payload={"message": f"Logical trajectory message {ordinal}.", "step": step},
            source_adapter="fixed-step-fixture/v1",
            trust_label=TrustLabel.SYNTHETIC_FIXTURE,
        ),
        expected_event_id=EVENT_IDS[ordinal - 1],
        logical_messages=(
            LogicalMessageBinding(
                role=role,
                selector=EventTextSelector(field_path="/payload/message"),
            ),
        ),
        action_step=ActionStepBinding(field_path="/payload/step"),
    )


def _multi_step_events() -> tuple[FixedStepEventInput, ...]:
    return (
        _run_start(),
        _message_event(2, step=1, role=LogicalMessageRole.ASSISTANT),
        _message_event(3, step=1, role=LogicalMessageRole.TOOL),
        _message_event(4, step=2, role=LogicalMessageRole.ASSISTANT),
    )


def _runner(
    repository: RunRepository,
    *,
    mode: Literal["reminder", "silence"],
    cycle_capacity: int,
    delivery_adapter: DeliveryAdapter | None = None,
    requested_delivery_target: DeliveryTarget | None = None,
    policy_version: str = "paper-fixed-step/v1",
    mismatching_executor: bool = False,
    model_token_counter: DeterministicModelTokenCounter | None = None,
) -> tuple[FixedStepRunner, _RequestAwareClient, AlgorithmConfigurationAttestation]:
    grounding = GroundingPipeline(_grounding_config())
    configuration = _configuration(
        grounding,
        cycle_capacity=cycle_capacity,
        requested_delivery_target=requested_delivery_target,
        policy_version=policy_version,
        model_token_counter=(
            model_token_counter.identity if model_token_counter is not None else None
        ),
    )
    client = _RequestAwareClient(repository, mode, model_token_counter)
    paper_executor = PaperTwoPhaseCycleExecutor(
        materializer=RepositoryOperationMaterializer(repository),
        client=client,
        prompt_bundle=PAPER_TWO_PHASE_V1,
        grounding_pipeline=grounding,
        model_profile=_profile(),
        call_policy=_policy(),
    )
    executor: TwoPhaseCycleExecutor = (
        _MismatchingExecutor(paper_executor) if mismatching_executor else paper_executor
    )
    runner = FixedStepRunner(
        repository=repository,
        executor=executor,
        grounding_pipeline=grounding,
        configuration=configuration,
        delivery_adapter=delivery_adapter,
    )
    return runner, client, configuration


async def _run(
    repository: RunRepository,
    events: tuple[FixedStepEventInput, ...],
    *,
    mode: Literal["reminder", "silence"],
    cycle_capacity: int,
    delivery_adapter: DeliveryAdapter | None = None,
    requested_delivery_target: DeliveryTarget | None = None,
) -> tuple[AlgorithmRunResult, _RequestAwareClient]:
    runner, client, _configuration_attestation = _runner(
        repository,
        mode=mode,
        cycle_capacity=cycle_capacity,
        delivery_adapter=delivery_adapter,
        requested_delivery_target=requested_delivery_target,
    )
    return await runner.run(events), client


def _grounding_config():
    # Keep this fixture colocated with the runtime test: its complete, reviewed
    # configuration is part of the algorithm attestation under test.
    from tests.repository.conformance import cycle_grounding_config

    return cycle_grounding_config()


def _assert_recalculated_tamper_rejected(
    result: AlgorithmRunResult,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    payload = result.model_dump(mode="json", warnings=False)
    mutate(payload)
    payload["result_digest"] = algorithm_result_digest(payload)
    with pytest.raises(ValidationError):
        AlgorithmRunResult.model_validate_json(canonical_json(payload))


def test_runtime_exports_the_additive_fixed_step_surface() -> None:
    assert runtime_module.AlgorithmConfigurationAttestation is AlgorithmConfigurationAttestation
    assert runtime_module.AlgorithmRunResult is AlgorithmRunResult
    assert runtime_module.FixedStepRecoveryResult is FixedStepRecoveryResult
    assert runtime_module.FixedStepEventInput is FixedStepEventInput
    assert runtime_module.FixedStepRunner is FixedStepRunner
    assert runtime_module.FixedStepTraceBoundary is FixedStepTraceBoundary
    assert runtime_module.FixedStepTraceDriver is FixedStepTraceDriver
    assert runtime_module.FixedStepTraceInput is FixedStepTraceInput
    assert runtime_module.FixedStepTraceInputError is FixedStepTraceInputError
    assert runtime_module.FixedStepTraceInvariantError is FixedStepTraceInvariantError
    assert runtime_module.FixedStepTraceResult is FixedStepTraceResult
    assert runtime_module.FixedStepTraceSpine is FixedStepTraceSpine
    assert runtime_module.ModelTokenUsageAttestation is ModelTokenUsageAttestation
    assert runtime_module.ModelTokenUsageSource is ModelTokenUsageSource
    assert runtime_module.algorithm_runtime_uuid is algorithm_runtime_uuid
    assert runtime_module.derive_cycle_reservation is derive_cycle_reservation
    assert runtime_module.fixed_step_recovery_digest is fixed_step_recovery_digest
    assert runtime_module.model_token_usage_attestation is model_token_usage_attestation
    assert runtime_module.semantic_projection_digests is semantic_projection_digests
    assert {
        "algorithm_runtime_uuid",
        "model_token_usage_attestation",
        "semantic_projection_digests",
    } <= set(runtime_module.__all__)


@pytest.mark.asyncio
async def test_trace_driver_streams_authoritative_boundaries_before_rebuild(
    repository: RunRepository,
) -> None:
    events = _multi_step_events()
    inputs = tuple(
        FixedStepTraceInput(
            draft=item.draft,
            expected_event_id=item.expected_event_id,
            task_description=item.task_description,
            logical_messages=item.logical_messages,
            action_step=item.action_step,
            target_request_id=item.target_request_id,
        )
        for item in events
    )
    observed_event_counts: list[int] = []
    observed_prefix_digests: list[str] = []

    async def project(boundary: FixedStepTraceBoundary) -> str:
        ledger = await repository.ledger(RUN_ID)
        observed_event_counts.append(sum(type(entry.record) is TraceEvent for entry in ledger))
        assert boundary.ordinal == boundary.event.sequence
        assert boundary.prefix.boundary_event_sequence == boundary.ordinal
        assert boundary.schedule.decisions[-1] == boundary.scheduled
        assert (boundary.window is not None) is boundary.scheduled.invoke
        observed_prefix_digests.append(boundary.prefix.prefix_digest)
        return boundary.prefix.prefix_digest

    result = await FixedStepTraceDriver(repository).run(inputs, project)

    expected_normalized = tuple(canonical_digest(item.draft) for item in inputs)
    assert observed_event_counts == [1, 2, 3, 4]
    assert result.spine.run_id == RUN_ID
    assert result.spine.trace_digest == algorithm_trace_digest(expected_normalized)
    assert result.spine.normalized_draft_digests == expected_normalized
    assert result.spine.persisted_event_draft_digests != expected_normalized
    assert result.spine.persisted_events == tuple(
        item.event for item in result.spine.trajectory_prefix.items
    )
    assert result.spine.bindings == tuple(
        item.binding for item in result.spine.trajectory_prefix.items
    )
    assert result.boundary_projections == tuple(observed_prefix_digests)
    assert result.ledger_head.entry_count == len(result.ledger) == len(events)
    assert result.rebuild_equivalent is True
    rebuilt = await repository.rebuild(RUN_ID)
    assert rebuilt.equivalent is True
    assert result.projection_digests == rebuilt.after


@pytest.mark.asyncio
async def test_fixed_step_runner_reconciles_lost_decision_acknowledgement() -> None:
    events = _multi_step_events()
    baseline_repository = _make_repository("memory", Path("unused-baseline.sqlite3"))
    baseline, _ = await _run(
        baseline_repository,
        events,
        mode="reminder",
        cycle_capacity=2,
    )
    repository = _LostDecisionAcknowledgementRepository()

    recovered, _ = await _run(
        repository,
        events,
        mode="reminder",
        cycle_capacity=2,
    )

    assert repository.acknowledgement_lost is True
    assert recovered == baseline


def test_configuration_and_reservation_reject_inconsistent_envelopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="call policy"):
        derive_cycle_reservation(cast(TwoPhaseCallPolicy, object()))

    policy = _policy()

    def fail_serialization(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError

    monkeypatch.setattr(TwoPhaseCallPolicy, "model_dump_json", fail_serialization)
    with pytest.raises(ValueError, match="call policy"):
        derive_cycle_reservation(policy)
    monkeypatch.undo()

    grounding = GroundingPipeline(_grounding_config())
    valid = _configuration(grounding, cycle_capacity=1)

    def payload() -> dict[str, object]:
        value = valid.model_dump(mode="json", warnings=False)
        value.pop("configuration_digest")
        return value

    wrong_reservation = payload()
    reservation = cast(dict[str, object], wrong_reservation["cycle_reservation"])
    reservation["model_calls"] = 1
    with pytest.raises(ValidationError, match="reservation"):
        AlgorithmConfigurationAttestation.model_validate_json(canonical_json(wrong_reservation))

    wrong_latency = payload()
    limits = cast(dict[str, object], wrong_latency["budget_limits"])
    limits["max_call_latency_us"] = policy.max_call_latency_us - 1
    with pytest.raises(ValidationError, match="latency"):
        AlgorithmConfigurationAttestation.model_validate_json(canonical_json(wrong_latency))

    foreign_profile = TwoPhaseModelProfile(
        schema_version="two-phase-model-profile/v1",
        profile_id="foreign-prompt-profile/v1",
        model_id=valid.model_profile.model_id,
        prompt_bundle_id=valid.prompt_bundle.bundle_id,
        prompt_bundle_digest="a" * 64,
    )
    wrong_prompt = payload()
    wrong_prompt["model_profile"] = foreign_profile.model_dump(mode="json", warnings=False)
    with pytest.raises(ValidationError, match="prompt bundle"):
        AlgorithmConfigurationAttestation.model_validate_json(canonical_json(wrong_prompt))

    wrong_digest = valid.model_dump(mode="json", warnings=False)
    wrong_digest["configuration_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="digest"):
        AlgorithmConfigurationAttestation.model_validate_json(canonical_json(wrong_digest))

    wrong_counter = payload()
    wrong_counter["model_token_counter"] = DeterministicModelTokenCounter(
        model_id="different-model/v1",
        input_token_count=1,
        output_token_count=1,
    ).identity.model_dump(mode="json", warnings=False)
    with pytest.raises(ValidationError, match="token counter"):
        AlgorithmConfigurationAttestation.model_validate_json(canonical_json(wrong_counter))

    restricted_config = _grounding_config().model_copy(
        update={"allowed_delivery_targets": (DeliveryTarget.NEXT_MODEL_CALL,)}
    )
    restricted = GroundingPipeline(restricted_config)
    with pytest.raises(ValidationError, match="delivery target"):
        AlgorithmConfigurationAttestation(
            policy_version="paper-fixed-step/v1",
            budget_limits=valid.budget_limits,
            cycle_reservation=valid.cycle_reservation,
            prompt_bundle=valid.prompt_bundle,
            model_profile=valid.model_profile,
            call_policy=valid.call_policy,
            grounding_configuration=restricted.resolved_configuration,
            requested_delivery_target=DeliveryTarget.PRE_ACTION_REPLAN,
        )

    monkeypatch.setattr(
        algorithm_result_module,
        "resolve_grounding_configuration",
        lambda _configuration: (_ for _ in ()).throw(RuntimeError()),
    )
    invalid_grounding = payload()
    with pytest.raises(ValidationError, match="grounding configuration"):
        AlgorithmConfigurationAttestation.model_validate_json(canonical_json(invalid_grounding))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("delivery_mode", "expected_state", "expected_reason", "expected_adapter_calls"),
    (
        ("deliver", DeliveryState.DELIVERED, ReasonCode.DELIVERY_SUCCEEDED, 1),
        (
            "pre_dispatch_refusal",
            DeliveryState.REJECTED,
            ReasonCode.UNSUPPORTED_DELIVERY_CHANNEL,
            0,
        ),
        ("unknown", DeliveryState.UNKNOWN, ReasonCode.DELIVERY_UNKNOWN, 1),
    ),
)
async def test_bootstrap_reminder_commits_memory_before_delivery_terminalizes(
    repository: RunRepository,
    delivery_mode: Literal["deliver", "pre_dispatch_refusal", "unknown"],
    expected_state: DeliveryState,
    expected_reason: ReasonCode,
    expected_adapter_calls: int,
) -> None:
    adapter = _DeliveryAdapter(delivery_mode)

    result, client = await _run(
        repository,
        (_run_start(target_request_id="request-after-bootstrap"),),
        mode="reminder",
        cycle_capacity=1,
        delivery_adapter=adapter,
        requested_delivery_target=DeliveryTarget.NEXT_MODEL_CALL,
    )

    assert len(result.decisions) == 1
    assert _REDACTED_FIXTURE_SECRET.encode() not in canonical_json(result)
    assert result.normalized_draft_digests != result.persisted_event_draft_digests
    assert _REDACTED_FIXTURE_SECRET not in cast(
        str,
        result.trajectory_prefix.items[0].event.payload["task"],
    )
    assert all(
        _REDACTED_FIXTURE_SECRET.encode() not in canonical_json(request)
        for request in client.requests
    )
    assert result.decisions[0].invoke is True
    assert result.decisions[0].reason_codes == (ReasonCode.BOOTSTRAP,)
    assert len(result.cycles) == 1
    cycle = result.cycles[0]
    assert cycle.state is CycleState.COMMITTED
    assert cycle.intervention is not None
    assert cycle.intervention.action is InterventionAction.REMIND
    assert len(result.call_receipts) == 2
    assert cycle.model_call_digests == tuple(
        receipt.call_digest for receipt in result.call_receipts
    )
    assert cycle.model_call_latencies_us == tuple(
        receipt.usage.latency_us for receipt in result.call_receipts
    )
    assert cycle.budget_settlement is not None
    assert cycle.budget_settlement.model_calls == 2
    assert cycle.budget_settlement.input_tokens == 22
    assert cycle.budget_settlement.output_tokens == 14
    assert cycle.budget_settlement.latency_us == 201
    assert cycle.budget_settlement.interventions == 1
    assert len(result.deliveries) == 1
    assert result.deliveries[0].state is expected_state
    assert result.deliveries[0].reason_code is expected_reason
    assert len(adapter.calls) == expected_adapter_calls
    assert len(client.requests) == 2
    assert len(client.dispatch_repository_states) == 2
    for held, running in client.dispatch_repository_states:
        assert held.reserved == result.configuration.cycle_reservation
        assert held.consumed == BudgetAmounts()
        assert running.state is CycleState.RUNNING
        assert running.revision == 3
        assert running.cycle_id == result.call_receipts[0].cycle_id
        assert running.budget_reservation == result.configuration.cycle_reservation
        assert running.batch_digest == result.windows[0].window_digest
    assert client.phase_two_repository_states == [(1, 0, 0)]
    snapshot = await repository.snapshot(RUN_ID)
    assert snapshot.ingestion_cursor == snapshot.memory_cursor == 1
    assert len(snapshot.records) == 1
    assert snapshot.records[0].content == ("Run the complete verified test suite before release.")
    budget = await repository.budget_snapshot(RUN_ID)
    assert budget.reserved.model_calls == 0
    assert budget.consumed.model_calls == 2
    assert budget.consumed.interventions == 1
    assert (
        budget.consumed.canonical_token_equivalents
        == result.configuration.cycle_reservation.canonical_token_equivalents
    )
    assert len(result.outcomes) == 1
    assert result.rebuild_equivalent is True
    if delivery_mode == "deliver":
        _assert_recalculated_tamper_rejected(
            result,
            lambda payload: payload.update(deliveries=[]),
        )
        _assert_recalculated_tamper_rejected(
            result,
            lambda payload: payload.update(outcomes=[]),
        )

        def delivery_target(payload: dict[str, object]) -> None:
            deliveries = cast(list[dict[str, object]], payload["deliveries"])
            deliveries[0]["cycle_id"] = "0" * 64

        _assert_recalculated_tamper_rejected(result, delivery_target)


@pytest.mark.asyncio
async def test_same_step_is_silent_and_a_new_step_runs_exactly_two_calls(
    repository: RunRepository,
) -> None:
    result, client = await _run(
        repository,
        _multi_step_events(),
        mode="silence",
        cycle_capacity=2,
    )

    assert tuple(decision.invoke for decision in result.decisions) == (
        True,
        False,
        False,
        True,
    )
    assert tuple(decision.reason_codes for decision in result.decisions) == (
        (ReasonCode.BOOTSTRAP,),
        (ReasonCode.SCRIPTED_SILENCE,),
        (ReasonCode.SCRIPTED_SILENCE,),
        (ReasonCode.SCRIPTED_INVOKE,),
    )
    assert result.schedule.invocation_count == 2
    assert len(result.windows) == 2
    assert len(result.cycles) == 2
    assert all(cycle.state is CycleState.COMMITTED for cycle in result.cycles)
    assert all(len(cycle.model_call_digests) == 2 for cycle in result.cycles)
    assert all(
        cycle.intervention is not None and cycle.intervention.action is InterventionAction.SILENCE
        for cycle in result.cycles
    )
    assert len(result.call_receipts) == len(client.requests) == 4
    assert tuple(receipt.model_call_index for receipt in result.call_receipts) == (0, 1, 0, 1)
    assert result.deliveries == ()
    snapshot = await repository.snapshot(RUN_ID)
    assert snapshot.ingestion_cursor == snapshot.memory_cursor == 4
    budget = await repository.budget_snapshot(RUN_ID)
    assert budget.reserved.model_calls == 0
    assert budget.consumed.model_calls == 4
    assert budget.consumed.interventions == 0
    assert result.rebuild_equivalent is True


@pytest.mark.asyncio
async def test_native_token_accounting_is_exact_and_preserves_provider_disagreement(
    repository: RunRepository,
) -> None:
    counter = DeterministicModelTokenCounter(
        model_id=_profile().model_id,
        input_token_count=23,
        output_token_count=7,
    )
    runner, client, configuration = _runner(
        repository,
        mode="silence",
        cycle_capacity=1,
        model_token_counter=counter,
    )

    result = await runner.run((_run_start(),))

    assert len(client.requests) == 2
    assert configuration.model_token_counter == counter.identity
    assert result.model_token_usage == ModelTokenUsageAttestation(
        configured_counter=counter.identity,
        usage_sources=(
            ModelTokenUsageSource.LOCAL_COUNTER,
            ModelTokenUsageSource.PROVIDER_REPORTED,
        ),
        provider_input_tokens=22,
        provider_output_tokens=14,
        canonical_input_tokens=46,
        canonical_output_tokens=14,
        canonical_token_equivalents=60,
        provider_canonical_disagreement=True,
    )
    assert result.cycles[0].budget_settlement is not None
    assert result.cycles[0].budget_settlement.canonical_token_equivalents == 60
    _assert_recalculated_tamper_rejected(
        result,
        lambda payload: cast(dict[str, object], payload["model_token_usage"]).update(
            canonical_token_equivalents=61
        ),
    )


@pytest.mark.asyncio
async def test_counter_mismatch_fails_unknown_before_settlement_or_memory_commit(
    repository: RunRepository,
) -> None:
    configured = DeterministicModelTokenCounter(
        model_id=_profile().model_id,
        input_token_count=23,
        output_token_count=7,
    )
    wrong = DeterministicModelTokenCounter(
        model_id=_profile().model_id,
        input_token_count=24,
        output_token_count=7,
    )
    runner, client, configuration = _runner(
        repository,
        mode="silence",
        cycle_capacity=1,
        model_token_counter=configured,
    )
    client.model_token_counter = wrong

    with pytest.raises(FixedStepExecutionError):
        await runner.run((_run_start(),))

    assert len(client.requests) == 2
    cycles = tuple(
        entry.record
        for entry in await repository.ledger(RUN_ID)
        if type(entry.record) is CycleRecord
    )
    assert cycles[-1].state is CycleState.FAILED
    assert cycles[-1].failure_reason is ReasonCode.FAILED_UNKNOWN_COST
    assert cycles[-1].budget_settlement == configuration.cycle_reservation
    snapshot = await repository.snapshot(RUN_ID)
    assert snapshot.ingestion_cursor == 1
    assert snapshot.memory_cursor == 0
    assert snapshot.records == ()


@pytest.mark.asyncio
async def test_exhausted_budget_records_one_decision_per_event_without_partial_cycle(
    repository: RunRepository,
) -> None:
    result, client = await _run(
        repository,
        _multi_step_events(),
        mode="silence",
        cycle_capacity=1,
    )

    assert len(result.decisions) == 4
    assert result.decisions[-1].invoke is False
    assert result.decisions[-1].reason_codes == (ReasonCode.BUDGET_EXHAUSTED,)
    assert result.schedule.invocation_count == 2
    assert len(result.windows) == 2
    assert len(result.cycles) == 1
    assert len(result.call_receipts) == len(client.requests) == 2
    snapshot = await repository.snapshot(RUN_ID)
    assert snapshot.ingestion_cursor == 4
    assert snapshot.memory_cursor == 1
    budget = await repository.budget_snapshot(RUN_ID)
    assert budget.reserved.model_calls == 0
    assert budget.consumed.model_calls == 2
    assert result.rebuild_equivalent is True


@pytest.mark.asyncio
async def test_runner_rejects_invalid_inputs_and_cannot_reuse_a_run(tmp_path: Path) -> None:
    repository = _make_repository("memory", tmp_path / "unused.sqlite3")
    with pytest.raises(FixedStepInputError):
        _runner(
            repository,
            mode="silence",
            cycle_capacity=1,
            policy_version="not-fixed-step/v1",
        )

    runner, client, _configuration_attestation = _runner(
        repository,
        mode="silence",
        cycle_capacity=1,
    )
    with pytest.raises(FixedStepInputError):
        await runner.run(())
    with pytest.raises(FixedStepInputError):
        await runner.run(cast(tuple[FixedStepEventInput, ...], (object(),)))
    with pytest.raises(FixedStepInputError):
        await runner.run((_run_start().model_copy(update={"task_description": None}),))
    wrong_start = _run_start().model_copy(
        update={
            "draft": _run_start().draft.model_copy(update={"event_type": EventType.OBSERVATION})
        }
    )
    with pytest.raises(FixedStepInputError):
        await runner.run((wrong_start,))

    invalid_inputs = (
        _run_start().model_copy(
            update={
                "draft": _run_start().draft.model_copy(update={"payload": {"task": "", "step": 1}})
            }
        ),
        _run_start().model_copy(
            update={
                "logical_messages": (
                    LogicalMessageBinding(
                        role=LogicalMessageRole.USER,
                        selector=EventTextSelector(field_path="/payload/task"),
                    ),
                )
            }
        ),
        _run_start().model_copy(
            update={"task_description": EventTextSelector(field_path="/payload/missing")}
        ),
        _run_start().model_copy(
            update={"task_description": EventTextSelector(field_path="/payload/step")}
        ),
        _run_start().model_copy(
            update={
                "draft": _run_start().draft.model_copy(
                    update={"payload": {"task": "é", "step": 1}}
                ),
                "task_description": EventTextSelector(
                    field_path="/payload/task",
                    span=TextSpan(start_byte=0, end_byte=1),
                ),
            }
        ),
        _run_start().model_copy(
            update={
                "draft": _run_start().draft.model_copy(
                    update={"payload": {"task": "short", "step": 1}}
                ),
                "task_description": EventTextSelector(
                    field_path="/payload/task",
                    span=TextSpan(start_byte=0, end_byte=99),
                ),
            }
        ),
        _run_start().model_copy(
            update={
                "draft": _run_start().draft.model_copy(
                    update={"payload": {"task": "Task", "step": 0}}
                )
            }
        ),
        _run_start().model_copy(
            update={
                "draft": _run_start().draft.model_copy(
                    update={"parent_ids": (UUID("00000000-0000-4000-8000-00000000afff"),)}
                )
            }
        ),
        _run_start().model_copy(
            update={
                "draft": _run_start().draft.model_copy(
                    update={"payload": {"items": ["Task"], "step": 1}}
                ),
                "task_description": EventTextSelector(field_path="/payload/items/01"),
            }
        ),
    )
    for invalid in invalid_inputs:
        with pytest.raises(FixedStepInputError):
            await runner.run((invalid,))

    step_two = _run_start().model_copy(
        update={
            "draft": _run_start().draft.model_copy(update={"payload": {"task": "Task", "step": 2}})
        }
    )
    with pytest.raises(FixedStepInputError):
        await runner.run(
            (
                step_two,
                _message_event(2, step=1, role=LogicalMessageRole.ASSISTANT),
            )
        )

    no_step = _run_start().model_copy(update={"action_step": None})
    result = await runner.run((no_step,))
    assert len(result.call_receipts) == len(client.requests) == 2
    with pytest.raises(FixedStepInputError):
        await runner.run((no_step,))


@pytest.mark.asyncio
async def test_missing_reminder_routing_fails_known_without_publishing_memory(
    repository: RunRepository,
) -> None:
    adapter = _DeliveryAdapter("deliver")

    result, client = await _run(
        repository,
        (_run_start(),),
        mode="reminder",
        cycle_capacity=1,
        delivery_adapter=adapter,
        requested_delivery_target=DeliveryTarget.NEXT_MODEL_CALL,
    )

    assert len(client.requests) == 2
    assert len(result.cycles) == len(result.executions) == 1
    cycle = result.cycles[0]
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is ReasonCode.TARGET_UNAVAILABLE
    assert cycle.validated_delta is None
    assert cycle.intervention is None
    assert cycle.budget_settlement is not None
    assert cycle.budget_settlement.interventions == 0
    assert result.deliveries == result.outcomes == ()
    assert adapter.calls == []
    snapshot = await repository.snapshot(RUN_ID)
    assert snapshot.ingestion_cursor == 1
    assert snapshot.memory_cursor == 0
    assert snapshot.records == ()


@pytest.mark.asyncio
async def test_executor_request_mismatch_fails_closed_without_committing_memory(
    tmp_path: Path,
) -> None:
    repository = _make_repository("memory", tmp_path / "unused.sqlite3")
    runner, client, configuration = _runner(
        repository,
        mode="silence",
        cycle_capacity=1,
        mismatching_executor=True,
    )

    with pytest.raises(FixedStepExecutionError):
        await runner.run((_run_start(),))

    assert len(client.requests) == 2
    snapshot = await repository.snapshot(RUN_ID)
    assert snapshot.ingestion_cursor == 1
    assert snapshot.memory_cursor == 0
    assert snapshot.records == ()
    budget = await repository.budget_snapshot(RUN_ID)
    assert budget.reserved == BudgetAmounts()
    assert budget.consumed == configuration.cycle_reservation
    cycles = tuple(
        entry.record
        for entry in await repository.ledger(RUN_ID)
        if type(entry.record) is CycleRecord
    )
    assert cycles[-1].state is CycleState.FAILED
    assert cycles[-1].failure_reason is ReasonCode.FAILED_UNKNOWN_COST


@pytest.mark.asyncio
async def test_fresh_backends_are_byte_identical_and_result_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    memory = _make_repository("memory", tmp_path / "unused.sqlite3")
    sqlite = _make_repository("sqlite", tmp_path / "deterministic.sqlite3")
    try:
        memory_result, memory_client = await _run(
            memory,
            _multi_step_events(),
            mode="silence",
            cycle_capacity=2,
        )
        sqlite_result, sqlite_client = await _run(
            sqlite,
            _multi_step_events(),
            mode="silence",
            cycle_capacity=2,
        )
    finally:
        assert isinstance(sqlite, SQLiteRunRepository)
        sqlite.close()

    assert len(memory_client.requests) == len(sqlite_client.requests) == 4
    assert canonical_json(memory_result) == canonical_json(sqlite_result)
    assert memory_result.result_digest == sqlite_result.result_digest
    assert memory_result.result_digest == (
        "81ba7e91f2f0c7adb148f6da2f4a1f9d892abeb8ecafa3d3650a8db748984c46"
    )
    assert memory_result.rebuild_equivalent is sqlite_result.rebuild_equivalent is True
    rebuilt = await memory.rebuild(RUN_ID)
    assert rebuilt.equivalent is True
    assert memory_result.projection_digests == rebuilt.after
    assert memory_result.ledger_head == await memory.ledger_head(RUN_ID)

    first_cycle = memory_result.cycles[0]
    first_receipts = tuple(
        receipt
        for receipt in memory_result.call_receipts
        if receipt.cycle_id == first_cycle.cycle_id
    )
    without_settlement = first_cycle.model_copy(update={"budget_settlement": None})
    with pytest.raises(ValueError, match="unaccounted"):
        memory_result._validate_cycle_settlement(without_settlement, first_receipts)
    memory_result._validate_cycle_settlement(without_settlement, ())
    assert first_cycle.budget_settlement is not None
    wrong_call_count = first_cycle.model_copy(
        update={
            "budget_settlement": first_cycle.budget_settlement.model_copy(update={"model_calls": 1})
        }
    )
    with pytest.raises(ValueError, match="model-call settlement"):
        memory_result._validate_cycle_settlement(wrong_call_count, first_receipts)

    def reverse(field_name: str) -> Callable[[dict[str, object]], None]:
        def mutate(payload: dict[str, object]) -> None:
            payload[field_name] = list(reversed(cast(list[object], payload[field_name])))

        return mutate

    def decision_reason(payload: dict[str, object]) -> None:
        decisions = cast(list[dict[str, object]], payload["decisions"])
        decisions[0]["reason_codes"] = [ReasonCode.POLICY_ALWAYS.value]

    def decision_configuration(payload: dict[str, object]) -> None:
        decisions = cast(list[dict[str, object]], payload["decisions"])
        decisions[0]["configuration_digest"] = "0" * 64

    def ledger_count(payload: dict[str, object]) -> None:
        payload["ledger_entry_count"] = cast(int, payload["ledger_entry_count"]) + 1

    def ledger_head(payload: dict[str, object]) -> None:
        head = cast(dict[str, object], payload["ledger_head"])
        head["entry_count"] = cast(int, head["entry_count"]) + 1

    def projection_algorithm(payload: dict[str, object]) -> None:
        projection = cast(dict[str, dict[str, object]], payload["projection_digests"])
        projection["overall"]["algorithm"] = PayloadDigestAlgorithm.SYNTHETIC_SHA256.value

    def rebuild_claim(payload: dict[str, object]) -> None:
        payload["rebuild_equivalent"] = False

    def drop_last(field_name: str) -> Callable[[dict[str, object]], None]:
        def mutate(payload: dict[str, object]) -> None:
            payload[field_name] = cast(list[object], payload[field_name])[:-1]

        return mutate

    def decision_risk(payload: dict[str, object]) -> None:
        decisions = cast(list[dict[str, object]], payload["decisions"])
        decisions[0]["risk_score"] = 0.5

    def invoking_without_budget(payload: dict[str, object]) -> None:
        decisions = cast(list[dict[str, object]], payload["decisions"])
        snapshot = cast(dict[str, dict[str, object]], decisions[0]["budget_snapshot"])
        limits = snapshot["limits"]
        consumed = snapshot["consumed"]
        consumed["model_calls"] = limits["model_calls"]

    def invalid_budget_demotion(payload: dict[str, object]) -> None:
        decisions = cast(list[dict[str, object]], payload["decisions"])
        decisions[0]["invoke"] = False
        decisions[0]["reason_codes"] = [ReasonCode.BUDGET_EXHAUSTED.value]

    def invalid_scripted_silence(payload: dict[str, object]) -> None:
        decisions = cast(list[dict[str, object]], payload["decisions"])
        decisions[1]["reason_codes"] = [ReasonCode.POLICY_NEVER.value]

    def duplicate_decision(payload: dict[str, object]) -> None:
        decisions = cast(list[dict[str, object]], payload["decisions"])
        decisions[1]["decision_id"] = decisions[0]["decision_id"]

    def future_parent(payload: dict[str, object]) -> None:
        prefix = cast(dict[str, object], payload["trajectory_prefix"])
        items = cast(list[dict[str, object]], prefix["items"])
        final_event = cast(dict[str, object], items[-1]["event"])
        second_event = cast(dict[str, object], items[1]["event"])
        second_event["parent_ids"] = [final_event["event_id"]]

    def settlement_mismatch(payload: dict[str, object]) -> None:
        cycles = cast(list[dict[str, object]], payload["cycles"])
        settlement = cast(dict[str, object], cycles[0]["budget_settlement"])
        settlement["input_tokens"] = cast(int, settlement["input_tokens"]) + 1

    def unsupported_outcome(payload: dict[str, object]) -> None:
        outcomes = cast(list[dict[str, object]], payload["outcomes"])
        outcomes[0]["memory_calls"] = 1

    def duplicate_outcome(payload: dict[str, object]) -> None:
        outcomes = cast(list[dict[str, object]], payload["outcomes"])
        outcomes[1]["outcome_id"] = outcomes[0]["outcome_id"]

    def regroup_calls(payload: dict[str, object]) -> None:
        calls = cast(list[dict[str, object]], payload["call_receipts"])
        payload["call_receipts"] = calls[2:] + calls[:2]

    def rebind_execution_request_digest(payload: dict[str, object]) -> None:
        executions = cast(list[dict[str, object]], payload["executions"])
        executions[0]["request_digest"] = "0" * 64
        executions[0]["result_digest"] = two_phase_module._result_digest(executions[0])

    def rebind_complete_cycle_request(payload: dict[str, object]) -> None:
        requests = cast(list[dict[str, object]], payload["cycle_requests"])
        executions = cast(list[dict[str, object]], payload["executions"])
        requests[0]["intervention_id"] = "00000000-0000-4000-8000-00000000cafe"
        requests[0]["request_digest"] = two_phase_module._cycle_request_digest(requests[0])
        executions[0]["request_digest"] = requests[0]["request_digest"]
        executions[0]["result_digest"] = two_phase_module._result_digest(executions[0])

    def rebind_unused_memory_identity(payload: dict[str, object]) -> None:
        requests = cast(list[dict[str, object]], payload["cycle_requests"])
        executions = cast(list[dict[str, object]], payload["executions"])
        assigned = cast(list[str], requests[0]["assigned_memory_ids"])
        assigned[-1] = "00000000-0000-4000-8000-00000000dead"
        requests[0]["request_digest"] = two_phase_module._cycle_request_digest(requests[0])
        executions[0]["request_digest"] = requests[0]["request_digest"]
        executions[0]["result_digest"] = two_phase_module._result_digest(executions[0])

    def rebind_grounding_away_from_trajectory(payload: dict[str, object]) -> None:
        requests = cast(list[dict[str, object]], payload["cycle_requests"])
        executions = cast(list[dict[str, object]], payload["executions"])
        calls = cast(list[dict[str, object]], payload["call_receipts"])
        request_grounding = cast(dict[str, object], requests[0]["grounding_state"])
        execution_grounding = cast(dict[str, object], executions[0]["grounding_state"])
        request_events = cast(list[dict[str, object]], request_grounding["events"])
        execution_events = cast(list[dict[str, object]], execution_grounding["events"])
        request_events[0]["source_adapter"] = "tampered-grounding/v1"
        execution_events[0]["source_adapter"] = "tampered-grounding/v1"
        grounding = GroundingState.model_validate_json(canonical_json(execution_grounding))
        execution_receipts = cast(list[dict[str, object]], executions[0]["call_receipts"])
        execution_receipts[-1]["grounding_state_digest"] = two_phase_module._grounding_state_digest(
            grounding
        )
        execution_receipts[-1]["receipt_digest"] = two_phase_module._call_receipt_digest(
            execution_receipts[-1]
        )
        calls[1] = dict(execution_receipts[-1])
        requests[0]["request_digest"] = two_phase_module._cycle_request_digest(requests[0])
        executions[0]["request_digest"] = requests[0]["request_digest"]
        executions[0]["result_digest"] = two_phase_module._result_digest(executions[0])

    for mutate in (
        reverse("normalized_draft_digests"),
        reverse("persisted_event_draft_digests"),
        future_parent,
        reverse("windows"),
        drop_last("windows"),
        drop_last("decisions"),
        decision_risk,
        invoking_without_budget,
        invalid_budget_demotion,
        invalid_scripted_silence,
        duplicate_decision,
        decision_reason,
        decision_configuration,
        drop_last("cycles"),
        drop_last("cycle_requests"),
        drop_last("executions"),
        reverse("cycles"),
        reverse("cycle_requests"),
        reverse("executions"),
        rebind_execution_request_digest,
        rebind_complete_cycle_request,
        rebind_unused_memory_identity,
        rebind_grounding_away_from_trajectory,
        reverse("call_receipts"),
        regroup_calls,
        reverse("outcomes"),
        unsupported_outcome,
        duplicate_outcome,
        settlement_mismatch,
        ledger_count,
        ledger_head,
        projection_algorithm,
        rebuild_claim,
    ):
        _assert_recalculated_tamper_rejected(memory_result, mutate)

    tampered = memory_result.model_dump(mode="json", warnings=False)
    tampered["result_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="result digest"):
        AlgorithmRunResult.model_validate_json(canonical_json(tampered))
