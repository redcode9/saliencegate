from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest

from saliencegate.adapters import JSONLReplayAdapter, JsonlReplayEvent, encode_jsonl_trace
from saliencegate.domain import (
    BudgetAmounts,
    BudgetLimits,
    BudgetSnapshot,
    ClaimKind,
    ConstraintStatus,
    CycleRecord,
    CycleState,
    DeliveryRecord,
    DeliveryState,
    DeliveryTarget,
    EventPhase,
    EventType,
    EvidenceReference,
    EvidenceSource,
    InterventionAction,
    InterventionDecision,
    InterventionOutcome,
    InvocationDecision,
    MemoryCreate,
    MemoryDelta,
    MemoryKind,
    MemoryUpdate,
    NormalizedTraceEventDraft,
    OutcomeEvidenceMode,
    PrivateStatusReplacement,
    ReasonCode,
    RepeatedErrorStatus,
    Signal,
    SignalType,
    TrustLabel,
    canonical_digest,
    canonical_json,
)
from saliencegate.intervention import (
    GroundingConfig,
    GroundingContext,
    GroundingPipeline,
    GroundingReceipt,
    GroundingState,
    ProposalParseStatus,
    ProposedClaim,
    RenderingConfig,
)
from saliencegate.models import ReplayModel, ReplayRecord
from saliencegate.policy import ScriptedPolicy, ScriptedPolicyConfig
from saliencegate.policy.config import RunState
from saliencegate.ports.adapters import (
    AdapterCapabilities,
    AdapterDeliveryFailedError,
    DeduplicationGuarantee,
    DeliveryAdapter,
    DeliveryChannel,
    DeliveryEnvelope,
    DeliveryReceipt,
    DeliveryRole,
    InjectionMapping,
)
from saliencegate.ports.memory import GroundingObservation, MemoryCycleOutput
from saliencegate.ports.models import (
    ModelCallStatus,
    ModelRequest,
    ModelResult,
    ModelUsage,
    StructuredModel,
)
from saliencegate.ports.outcomes import (
    OutcomeRecorder,
    OutcomeRecordingError,
    PolicyReplayOutcomeRecorder,
)
from saliencegate.ports.repository import (
    AppendReceipt,
    BeginCycle,
    BeginDeliveryAttempt,
    CommitCycle,
    CompleteDelivery,
    CycleReceipt,
    DeliveryAttemptReceipt,
    DeliveryTransitionReceipt,
    FailCycle,
    LedgerEntry,
    MarkDeliveryUnknown,
    RebuildReceipt,
    RepositoryError,
    ReserveCycle,
    RunNotFoundError,
)
from saliencegate.repository import MemoryRunRepository, SQLiteRunRepository
from saliencegate.runtime import BatchConfig, BatchPriorityKind
from saliencegate.runtime.engine import (
    ReplayEngine,
    ReplayEngineConfig,
    ReplayEngineInputError,
    ReplayEngineInvariantError,
    ReplayEngineModelError,
    ReplayEventResult,
    ReplayModelPayload,
    ReplayRunResult,
    ReplaySignalExtractor,
    ReplayTraceAdapter,
    ReplayTriggerPolicy,
    _memory_assignments,
    _model_fixture_state,
    _normalized_event_draft_digest,
    _projection,
    _ReplayFixtureState,
    _result_digest,
    _semantic_uuid,
    _validated_signals,
    normalized_trace_digest,
)
from saliencegate.security import RedactionPolicy
from saliencegate.signals import DetectionContext, DeterministicSignalExtractor

RUN_ID = UUID("00000000-0000-4000-8000-00000000d001")
EVENT_1_ID = UUID("00000000-0000-4000-8000-00000000d011")
EVENT_2_ID = UUID("00000000-0000-4000-8000-00000000d012")
EVENT_3_ID = UUID("00000000-0000-4000-8000-00000000d013")
EVENT_4_ID = UUID("00000000-0000-4000-8000-00000000d014")
DELIVERY_ID = UUID("00000000-0000-4000-8000-00000000d015")
DELTA_1_ID = UUID("00000000-0000-4000-8000-00000000d021")
DELTA_2_ID = UUID("00000000-0000-4000-8000-00000000d022")
DELTA_3_ID = UUID("00000000-0000-4000-8000-00000000d023")
NOW = datetime(2026, 7, 11, 21, 0, tzinfo=UTC)
PROMPT_DIGEST = "a" * 64
ADAPTER_ID = "engine-fixture/1"
FIXTURES = Path(__file__).parents[1] / "fixtures"


def _draft(
    ordinal: int,
    *,
    parent_ids: tuple[UUID, ...] = (),
) -> NormalizedTraceEventDraft:
    event_type = EventType.RUN_START if ordinal == 1 else EventType.OBSERVATION
    phase = EventPhase.INTERNAL if ordinal == 1 else EventPhase.POST_ACTION
    return NormalizedTraceEventDraft(
        run_id=RUN_ID,
        source_event_id=f"engine-event-{ordinal}",
        timestamp=NOW + timedelta(seconds=ordinal),
        event_type=event_type,
        phase=phase,
        payload={"message": f"verified event {ordinal}"},
        parent_ids=parent_ids,
        source_adapter="engine-fixture/1",
        trust_label=TrustLabel.SYNTHETIC_FIXTURE,
    )


def trace_adapter() -> JSONLReplayAdapter:
    events = (
        JsonlReplayEvent.create(
            ordinal=1,
            expected_event_id=EVENT_1_ID,
            draft=_draft(1),
        ),
        JsonlReplayEvent.create(
            ordinal=2,
            expected_event_id=EVENT_2_ID,
            draft=_draft(2, parent_ids=(EVENT_1_ID,)),
        ),
        JsonlReplayEvent.create(
            ordinal=3,
            expected_event_id=EVENT_3_ID,
            draft=_draft(3, parent_ids=(EVENT_2_ID,)),
        ),
        JsonlReplayEvent.create(
            ordinal=4,
            expected_event_id=EVENT_4_ID,
            draft=_draft(4, parent_ids=(EVENT_3_ID,)),
            next_model_call_target_request_id="engine-request-4",
        ),
    )
    return JSONLReplayAdapter.from_bytes(encode_jsonl_trace(events))


def grounding_pipeline() -> GroundingPipeline:
    rendering = RenderingConfig(
        schema_version="1.0",
        renderer_version="fixed-ascii/v1",
        token_counter_version="utf8-bytes-ceil-div-4-v1",
        max_claims=2,
        max_evidence_bytes=1_024,
        max_output_bytes=4_096,
        max_token_equivalents=1_024,
        include_provenance=False,
    )
    return GroundingPipeline(
        GroundingConfig(
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
            allowed_delivery_targets=(DeliveryTarget.NEXT_MODEL_CALL,),
            rendering=rendering,
        )
    )


def engine_config(*, batch_bytes: int = 32_000) -> ReplayEngineConfig:
    return ReplayEngineConfig(
        model_id="replay-fixture/1",
        prompt_template_digest=PROMPT_DIGEST,
        budget_limits=BudgetLimits(
            model_calls=10,
            input_tokens=10_000,
            output_tokens=10_000,
            canonical_token_equivalents=20_000,
            latency_us=1_000_000,
            interventions=10,
            schema_repairs=2,
            max_call_latency_us=100_000,
        ),
        reservation=BudgetAmounts(
            model_calls=1,
            input_tokens=1_000,
            output_tokens=1_000,
            canonical_token_equivalents=2_000,
            latency_us=100_000,
            interventions=1,
            schema_repairs=1,
        ),
        batch=BatchConfig(
            max_utf8_bytes=batch_bytes,
            max_approximate_tokens=max(1, batch_bytes // 4),
            recent_event_count=4,
            max_controller_errors=4,
            max_action_proposals=4,
            max_tool_errors=4,
            max_test_failures=4,
            max_conflicts=4,
        ),
        requested_delivery_target=DeliveryTarget.NEXT_MODEL_CALL,
    )


def delivery_capabilities() -> AdapterCapabilities:
    return AdapterCapabilities(
        schema_version="1.0",
        adapter_id=ADAPTER_ID,
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


class FailingDeliveryAdapter:
    def __init__(self) -> None:
        self.calls: list[DeliveryEnvelope] = []

    def capabilities(self) -> AdapterCapabilities:
        return delivery_capabilities()

    async def deliver(self, delivery: DeliveryEnvelope) -> DeliveryReceipt:
        self.calls.append(delivery)
        raise AdapterDeliveryFailedError()


class ScriptedMemoryModel:
    def __init__(self, trace_digest: str) -> None:
        self.trace_digest = trace_digest
        self.requests: list[ModelRequest] = []
        self.results: list[ModelResult] = []
        self.memory_id: UUID | None = None

    @staticmethod
    def _usage() -> ModelUsage:
        return ModelUsage(
            input_tokens=10,
            output_tokens=5,
            canonical_token_equivalents=15,
            latency_us=100,
        )

    async def generate(self, request: ModelRequest) -> ModelResult:
        self.requests.append(request)
        call = len(self.requests)
        event_id = (EVENT_2_ID, EVENT_3_ID, EVENT_4_ID)[call - 1]
        delta_id = (DELTA_1_ID, DELTA_2_ID, DELTA_3_ID)[call - 1]
        created_at = NOW + timedelta(seconds=call + 1)
        if call == 1:
            self.memory_id = _semantic_uuid(
                self.trace_digest,
                "memory",
                request.cycle_id,
                "verified-requirement",
            )
            delta = MemoryDelta(
                delta_id=delta_id,
                run_id=RUN_ID,
                creates=(
                    MemoryCreate(
                        handle="verified-requirement",
                        kind=MemoryKind.KNOWLEDGE,
                        content="Run the verified test suite before delivery.",
                        provenance=(
                            EvidenceReference(
                                source=EvidenceSource.EVENT,
                                source_id=event_id,
                                field_path="/payload/message",
                            ),
                        ),
                        confidence=1.0,
                        trust_label=TrustLabel.SYNTHETIC_FIXTURE,
                    ),
                ),
                created_at=created_at,
            )
            observation = GroundingObservation(
                parse_status=ProposalParseStatus.VALID,
                proposal_action=InterventionAction.SILENCE,
                claims=(),
                confidence=1.0,
            )
        elif call == 2:
            delta = MemoryDelta(
                delta_id=delta_id,
                run_id=RUN_ID,
                creates=(
                    MemoryCreate(
                        handle="verified-procedure",
                        kind=MemoryKind.PROCEDURAL,
                        content="The schema-invalid response still preserves a valid delta.",
                        provenance=(
                            EvidenceReference(
                                source=EvidenceSource.EVENT,
                                source_id=event_id,
                                field_path="/payload/message",
                            ),
                        ),
                        confidence=1.0,
                        trust_label=TrustLabel.SYNTHETIC_FIXTURE,
                    ),
                ),
                created_at=created_at,
            )
            observation = GroundingObservation(
                parse_status=ProposalParseStatus.SCHEMA_INVALID,
                proposal_action=None,
                claims=(),
                confidence=1.0,
            )
        else:
            assert self.memory_id is not None
            delta = MemoryDelta(
                delta_id=delta_id,
                run_id=RUN_ID,
                created_at=created_at,
            )
            observation = GroundingObservation(
                parse_status=ProposalParseStatus.VALID,
                proposal_action=InterventionAction.REMIND,
                claims=(
                    ProposedClaim(
                        kind=ClaimKind.REQUIREMENT,
                        evidence=EvidenceReference(
                            source=EvidenceSource.MEMORY,
                            source_id=self.memory_id,
                            revision=1,
                            field_path="/content",
                        ),
                    ),
                ),
                confidence=1.0,
            )
        result = ModelResult(
            status=ModelCallStatus.COMPLETED,
            request_digest=request.request_digest,
            output=MemoryCycleOutput(delta=delta, observation=observation),
            usage=self._usage(),
        )
        self.results.append(result)
        return result


def repository_ids() -> Callable[[], UUID]:
    values = iter((EVENT_1_ID, EVENT_2_ID, EVENT_3_ID, EVENT_4_ID, DELIVERY_ID))
    return values.__next__


async def run_replay(
    repository: MemoryRunRepository | SQLiteRunRepository,
) -> tuple[
    ReplayRunResult,
    ScriptedMemoryModel,
    FailingDeliveryAdapter,
    PolicyReplayOutcomeRecorder,
]:
    trace = trace_adapter()
    model = ScriptedMemoryModel(trace.manifest.trace_digest)
    delivery = FailingDeliveryAdapter()
    recorder = PolicyReplayOutcomeRecorder()
    engine = ReplayEngine(
        repository=repository,
        adapter=trace,
        extractor=DeterministicSignalExtractor(()),
        policy=ScriptedPolicy(
            ScriptedPolicyConfig(
                schema_version="1.0",
                policy_kind="scripted",
                decisions=(False, True, True, True),
                on_exhaustion="silence",
            )
        ),
        model=model,
        grounding=grounding_pipeline(),
        config=engine_config(),
        delivery_adapter=delivery,
        outcome_recorder=recorder,
    )
    result = await engine.run(trace.events, trace_digest=trace.manifest.trace_digest)
    return result, model, delivery, recorder


async def test_engine_connects_silence_schema_invalid_reminder_and_failed_delivery() -> None:
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=repository_ids(),
    )

    result, model, delivery, recorder = await run_replay(repository)

    assert len(result.events) == 4
    assert result.events[0].cycle is None
    assert tuple(item.cycle.state for item in result.events[1:] if item.cycle is not None) == (
        CycleState.COMMITTED,
        CycleState.COMMITTED,
        CycleState.COMMITTED,
    )
    assert len(model.requests) == 3
    assert result.events[1].cycle is not None
    assert result.events[1].cycle.intervention is not None
    assert result.events[1].cycle.intervention.action is InterventionAction.SILENCE
    assert result.events[2].cycle is not None
    assert result.events[2].cycle.intervention is not None
    assert result.events[2].cycle.intervention.reason_code is ReasonCode.SCHEMA_INVALID
    assert result.events[3].cycle is not None
    assert result.events[3].cycle.intervention is not None
    assert result.events[3].cycle.intervention.action is InterventionAction.REMIND
    assert result.events[3].delivery is not None
    assert result.events[3].delivery.state is DeliveryState.FAILED
    assert len(delivery.calls) == 1
    assert result.outcomes == recorder.outcomes
    assert len(result.outcomes) == 3
    assert all(
        outcome.evidence_mode is OutcomeEvidenceMode.POLICY_REPLAY
        and outcome.utility is None
        and outcome.task_reward is None
        for outcome in result.outcomes
    )
    snapshot = await repository.snapshot(RUN_ID)
    assert {memory.kind for memory in snapshot.records} == {
        MemoryKind.KNOWLEDGE,
        MemoryKind.PROCEDURAL,
    }
    assert result.rebuild_equivalent


async def test_two_fresh_backends_produce_byte_identical_replay_results(
    tmp_path: Path,
) -> None:
    memory = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=repository_ids(),
    )
    sqlite = SQLiteRunRepository(
        tmp_path / "engine-replay.sqlite3",
        synthetic_benchmark=True,
        id_factory=repository_ids(),
    )
    try:
        memory_result, *_ = await run_replay(memory)
        sqlite_result, *_ = await run_replay(sqlite)
    finally:
        sqlite.close()

    assert memory_result.decisions_json == sqlite_result.decisions_json
    assert memory_result.projection_digests == sqlite_result.projection_digests
    assert canonical_json(memory_result) == canonical_json(sqlite_result)


async def _run_frozen_replay(model: ReplayModel | None = None) -> ReplayRunResult:
    trace = JSONLReplayAdapter.from_path(FIXTURES / "runs" / "basic.jsonl")
    if model is None:
        model = ReplayModel.from_path(FIXTURES / "models" / "basic_responses.jsonl")
    delivery = FailingDeliveryAdapter()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=repository_ids(),
    )
    engine = ReplayEngine(
        repository=repository,
        adapter=trace,
        extractor=DeterministicSignalExtractor(()),
        policy=ScriptedPolicy(
            ScriptedPolicyConfig(
                schema_version="1.0",
                policy_kind="scripted",
                decisions=(False, True, True, True),
                on_exhaustion="silence",
            )
        ),
        model=model,
        grounding=grounding_pipeline(),
        config=engine_config(),
        delivery_adapter=delivery,
    )

    result = await engine.run(trace.events, trace_digest=trace.manifest.trace_digest)

    assert model.remaining_responses == 0
    assert result.events[-1].delivery is not None
    assert result.events[-1].delivery.state is DeliveryState.FAILED
    return result


async def test_frozen_trace_and_replay_model_are_byte_identical_across_two_fresh_runs() -> None:
    first = await _run_frozen_replay()
    second = await _run_frozen_replay()

    assert first.decisions_json == second.decisions_json
    assert first.projection_digests == second.projection_digests
    assert canonical_json(first) == canonical_json(second)
    assert first.model_execution_mode == "frozen_replay"
    assert first.replay_id == "replay-model/v1"
    assert (
        first.fixture_digest == "59f0f7016600c1010c19dba01e21bc4e34bb6345e5cc42349e90a0aa0a16020b"
    )
    assert first.fixture_response_count == first.fixture_consumed_count == 3


async def test_frozen_replay_rejects_an_unconsumed_extra_response() -> None:
    fixture_lines = (FIXTURES / "models" / "basic_responses.jsonl").read_text().splitlines()
    records = tuple(ReplayRecord.model_validate_json(line) for line in fixture_lines)
    unused_request = ModelRequest(
        run_id=RUN_ID,
        cycle_id="f" * 64,
        model_call_index=99,
        model_id="unused-fixture/1",
        prompt_template_digest="e" * 64,
        payload={"unused": True},
    )
    unused_result = ModelResult(
        status=ModelCallStatus.MODEL_ERROR,
        request_digest=unused_request.request_digest,
        usage=ModelUsage(),
    )
    altered = ReplayModel(
        (
            *records,
            ReplayRecord(
                ordinal=4,
                request_digest=unused_request.request_digest,
                result=unused_result,
            ),
        ),
        replay_id="fixture/with-unused-v1",
    )

    with pytest.raises(ReplayEngineModelError):
        await _run_frozen_replay(altered)

    assert altered.remaining_responses == 1


class ExplodingModel:
    async def generate(self, request: ModelRequest) -> ModelResult:
        raise RuntimeError(f"private model error for {request.request_digest}")


async def test_model_exception_terminalizes_the_running_cycle_before_raising() -> None:
    event = JsonlReplayEvent.create(
        ordinal=1,
        expected_event_id=EVENT_1_ID,
        draft=_draft(1),
    )
    trace = JSONLReplayAdapter.from_bytes(encode_jsonl_trace((event,)))
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )
    engine = ReplayEngine(
        repository=repository,
        adapter=trace,
        extractor=DeterministicSignalExtractor(()),
        policy=ScriptedPolicy(
            ScriptedPolicyConfig(
                schema_version="1.0",
                policy_kind="scripted",
                decisions=(True,),
                on_exhaustion="silence",
            )
        ),
        model=ExplodingModel(),
        grounding=grounding_pipeline(),
        config=engine_config(),
    )

    with pytest.raises(ReplayEngineModelError):
        await engine.run(trace.events, trace_digest=trace.manifest.trace_digest)

    cycles = tuple(
        entry.record
        for entry in await repository.ledger(RUN_ID)
        if isinstance(entry.record, CycleRecord)
    )
    assert cycles[-1].state is CycleState.FAILED
    assert cycles[-1].failure_reason is ReasonCode.FAILED_UNKNOWN_COST
    assert cycles[-1].budget_settlement == engine_config().reservation
    assert not any(
        isinstance(entry.record, DeliveryRecord) for entry in await repository.ledger(RUN_ID)
    )


def one_event_trace(
    *,
    target_request_id: str | None = None,
    pre_action_target_request_id: str | None = None,
) -> JSONLReplayAdapter:
    event = JsonlReplayEvent.create(
        ordinal=1,
        expected_event_id=EVENT_1_ID,
        draft=_draft(1),
        next_model_call_target_request_id=target_request_id,
        pre_action_target_request_id=pre_action_target_request_id,
    )
    return JSONLReplayAdapter.from_bytes(encode_jsonl_trace((event,)))


def invoking_policy() -> ScriptedPolicy:
    return ScriptedPolicy(
        ScriptedPolicyConfig(
            schema_version="1.0",
            policy_kind="scripted",
            decisions=(True,),
            on_exhaustion="silence",
        )
    )


class UncalledModel:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, request: ModelRequest) -> ModelResult:
        self.calls += 1
        raise AssertionError(f"overflow must not call model {request.request_digest}")


async def test_mandatory_batch_overflow_fails_reserved_cycle_without_a_model_call() -> None:
    trace = one_event_trace()
    model = UncalledModel()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )
    engine = ReplayEngine(
        repository=repository,
        adapter=trace,
        extractor=DeterministicSignalExtractor(()),
        policy=invoking_policy(),
        model=model,
        grounding=grounding_pipeline(),
        config=engine_config(batch_bytes=1),
    )

    result = await engine.run(trace.events, trace_digest=trace.manifest.trace_digest)

    assert model.calls == 0
    assert result.events[0].cycle is not None
    assert result.events[0].cycle.state is CycleState.FAILED
    assert result.events[0].cycle.failure_reason is ReasonCode.MANDATORY_INPUT_OVERFLOW
    assert result.events[0].cycle.budget_settlement == BudgetAmounts()
    assert result.outcomes == ()


class TimeoutModel:
    async def generate(self, request: ModelRequest) -> ModelResult:
        return ModelResult(
            status=ModelCallStatus.MODEL_TIMEOUT,
            request_digest=request.request_digest,
            output=None,
            usage=ModelUsage(latency_us=100),
        )


async def test_recorded_model_timeout_settles_known_usage_and_fails_the_cycle() -> None:
    trace = one_event_trace()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )
    engine = ReplayEngine(
        repository=repository,
        adapter=trace,
        extractor=DeterministicSignalExtractor(()),
        policy=invoking_policy(),
        model=TimeoutModel(),
        grounding=grounding_pipeline(),
        config=engine_config(),
    )

    result = await engine.run(trace.events, trace_digest=trace.manifest.trace_digest)

    cycle = result.events[0].cycle
    assert cycle is not None
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is ReasonCode.MODEL_TIMEOUT
    assert cycle.budget_settlement is not None
    assert cycle.budget_settlement.model_calls == 1
    assert cycle.model_call_latencies_us == (100,)


class ReminderWithoutTargetModel:
    async def generate(self, request: ModelRequest) -> ModelResult:
        output = MemoryCycleOutput(
            delta=MemoryDelta(
                delta_id=DELTA_1_ID,
                run_id=RUN_ID,
                created_at=NOW + timedelta(seconds=1),
            ),
            observation=GroundingObservation(
                parse_status=ProposalParseStatus.VALID,
                proposal_action=InterventionAction.REMIND,
                claims=(
                    ProposedClaim(
                        kind=ClaimKind.REQUIREMENT,
                        evidence=EvidenceReference(
                            source=EvidenceSource.EVENT,
                            source_id=EVENT_1_ID,
                            field_path="/payload/message",
                        ),
                    ),
                ),
                confidence=1.0,
            ),
        )
        return ModelResult(
            status=ModelCallStatus.COMPLETED,
            request_digest=request.request_digest,
            output=output,
            usage=ModelUsage(latency_us=100),
        )


async def test_unavailable_delivery_target_commits_grounded_silence_without_outbox() -> None:
    trace = one_event_trace(target_request_id="request-without-delivery-adapter")
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )
    engine = ReplayEngine(
        repository=repository,
        adapter=trace,
        extractor=DeterministicSignalExtractor(()),
        policy=invoking_policy(),
        model=ReminderWithoutTargetModel(),
        grounding=grounding_pipeline(),
        config=engine_config(),
        delivery_adapter=None,
    )

    result = await engine.run(trace.events, trace_digest=trace.manifest.trace_digest)

    cycle = result.events[0].cycle
    assert cycle is not None and cycle.intervention is not None
    assert cycle.state is CycleState.COMMITTED
    assert cycle.intervention.action is InterventionAction.SILENCE
    assert cycle.intervention.reason_code is ReasonCode.UNSUPPORTED_DELIVERY_TARGET
    assert result.events[0].delivery is None
    assert not any(
        isinstance(entry.record, DeliveryRecord) for entry in await repository.ledger(RUN_ID)
    )


OTHER_RUN_ID = UUID("00000000-0000-4000-8000-00000000d099")
OTHER_EVENT_ID = UUID("00000000-0000-4000-8000-00000000d098")


def _silent_output(
    *,
    run_id: UUID = RUN_ID,
    created_at: datetime = NOW + timedelta(seconds=1),
    updates: tuple[MemoryUpdate, ...] = (),
) -> MemoryCycleOutput:
    return MemoryCycleOutput(
        delta=MemoryDelta(
            delta_id=DELTA_1_ID,
            run_id=run_id,
            updates=updates,
            created_at=created_at,
        ),
        observation=GroundingObservation(
            parse_status=ProposalParseStatus.VALID,
            proposal_action=InterventionAction.SILENCE,
            claims=(),
            confidence=1.0,
        ),
    )


class CompletedModel:
    def __init__(
        self,
        output: MemoryCycleOutput,
        *,
        usage: ModelUsage | None = None,
        request_digest: str | None = None,
    ) -> None:
        self.output = output
        self.usage = ModelUsage(latency_us=100) if usage is None else usage
        self.request_digest = request_digest

    async def generate(self, request: ModelRequest) -> ModelResult:
        return ModelResult(
            status=ModelCallStatus.COMPLETED,
            request_digest=(
                request.request_digest if self.request_digest is None else self.request_digest
            ),
            output=self.output,
            usage=self.usage,
        )


def _engine(
    repository: MemoryRunRepository,
    adapter: ReplayTraceAdapter,
    *,
    model: StructuredModel | None = None,
    extractor: ReplaySignalExtractor | None = None,
    policy: ReplayTriggerPolicy | None = None,
    grounding: GroundingPipeline | None = None,
    config: ReplayEngineConfig | None = None,
    delivery_adapter: DeliveryAdapter | None = None,
    outcome_recorder: OutcomeRecorder | None = None,
) -> ReplayEngine:
    return ReplayEngine(
        repository=repository,
        adapter=adapter,
        extractor=(DeterministicSignalExtractor(()) if extractor is None else extractor),
        policy=(invoking_policy() if policy is None else policy),
        model=(CompletedModel(_silent_output()) if model is None else model),
        grounding=(grounding_pipeline() if grounding is None else grounding),
        config=(engine_config() if config is None else config),
        delivery_adapter=delivery_adapter,
        outcome_recorder=outcome_recorder,
    )


@pytest.mark.parametrize(
    "field_name",
    (
        "model_calls",
        "input_tokens",
        "output_tokens",
        "canonical_token_equivalents",
        "latency_us",
        "interventions",
        "schema_repairs",
    ),
)
def test_engine_config_rejects_every_reservation_dimension_above_its_limit(
    field_name: str,
) -> None:
    config = engine_config()
    values = config.reservation.model_dump(mode="python")
    values[field_name] = getattr(config.budget_limits, field_name) + 1

    with pytest.raises(ValueError, match="reservation exceeds the run budget"):
        ReplayEngineConfig(
            model_id=config.model_id,
            prompt_template_digest=config.prompt_template_digest,
            budget_limits=config.budget_limits,
            reservation=BudgetAmounts.model_validate(values),
            batch=config.batch,
            requested_delivery_target=config.requested_delivery_target,
        )


def test_engine_config_requires_at_least_one_reserved_model_call() -> None:
    config = engine_config()

    with pytest.raises(ValueError, match="reservation exceeds the run budget"):
        ReplayEngineConfig(
            model_id=config.model_id,
            prompt_template_digest=config.prompt_template_digest,
            budget_limits=config.budget_limits,
            reservation=BudgetAmounts(),
            batch=config.batch,
        )


def _validate_event_result(value: ReplayEventResult) -> ReplayEventResult:
    validator = cast(Callable[[], ReplayEventResult], value.records_belong_to_the_event_run)
    return validator()


def _validate_run_result(value: ReplayRunResult) -> ReplayRunResult:
    digest = _result_digest(
        value.model_dump(mode="python", exclude={"result_digest"}, warnings=False)
    )
    value = value.model_copy(update={"result_digest": digest})
    validator = cast(Callable[[], ReplayRunResult], value.result_is_replay_safe)
    return validator()


async def test_event_and_run_result_validators_reject_cross_run_and_causal_forgery() -> None:
    result, *_ = await run_replay(
        MemoryRunRepository(synthetic_benchmark=True, id_factory=repository_ids())
    )
    first = result.events[0]
    reminder = result.events[-1]

    assert result.ledger_head.entry_count == result.ledger_entry_count
    assert result.ledger_head.head_tag != result.ledger_head.chain_tag
    forged_head = result.ledger_head.model_copy(
        update={"entry_count": result.ledger_entry_count + 1}
    )
    with pytest.raises(ValueError, match="ledger attestation"):
        _validate_run_result(result.model_copy(update={"ledger_head": forged_head}))

    stale_digest = result.model_copy(update={"trace_digest": "f" * 64})
    stale_validator = cast(Callable[[], ReplayRunResult], stale_digest.result_is_replay_safe)
    with pytest.raises(ValueError, match="result digest"):
        stale_validator()

    cross_run_decision = first.decision.model_copy(update={"run_id": OTHER_RUN_ID})
    with pytest.raises(ValueError, match="different runs"):
        _validate_event_result(first.model_copy(update={"decision": cross_run_decision}))

    cross_run_signal = Signal(
        signal_id=UUID("00000000-0000-4000-8000-00000000d091"),
        run_id=OTHER_RUN_ID,
        created_at=first.event.timestamp,
        signal_type=SignalType.TOOL_ERROR,
        strength=1.0,
        evidence_event_ids=(first.event.event_id,),
        detector_version="engine-fixture/1",
        reason_code=ReasonCode.TOOL_ERROR,
    )
    with pytest.raises(ValueError, match="different runs"):
        _validate_event_result(first.model_copy(update={"signals": (cross_run_signal,)}))

    assert reminder.cycle is not None
    cross_run_cycle = reminder.cycle.model_copy(update={"run_id": OTHER_RUN_ID})
    with pytest.raises(ValueError, match="different run"):
        _validate_event_result(reminder.model_copy(update={"cycle": cross_run_cycle}))

    assert reminder.delivery is not None
    cross_run_delivery = reminder.delivery.model_copy(update={"run_id": OTHER_RUN_ID})
    with pytest.raises(ValueError, match="different run"):
        _validate_event_result(
            reminder.model_copy(update={"cycle": None, "delivery": cross_run_delivery})
        )

    with pytest.raises(ValueError, match="decision export"):
        _validate_run_result(result.model_copy(update={"decisions_json": "[]"}))
    with pytest.raises(ValueError, match="decision export"):
        _validate_run_result(result.model_copy(update={"decisions_digest": "f" * 64}))

    cross_run_event = first.event.model_copy(update={"run_id": OTHER_RUN_ID})
    cross_run_event_result = first.model_copy(update={"event": cross_run_event})
    with pytest.raises(ValueError, match="different run"):
        _validate_run_result(
            result.model_copy(update={"events": (cross_run_event_result, *result.events[1:])})
        )

    causal_updates: tuple[dict[str, object], ...] = (
        {"run_id": OTHER_RUN_ID},
        {"evidence_mode": OutcomeEvidenceMode.LIVE_OBSERVATION},
        {"next_action_fingerprint": "e" * 64},
        {"repeated_error_status": RepeatedErrorStatus.REPEATED},
        {"constraint_status": ConstraintStatus.VIOLATED},
        {"utility": "helpful"},
        {"action_changed": True},
        {"task_reward": 1.0},
        {"task_passed": True},
    )
    for update in causal_updates:
        causal = result.outcomes[0].model_copy(update=update)
        with pytest.raises(ValueError, match="causal outcome claim"):
            _validate_run_result(result.model_copy(update={"outcomes": (causal,)}))

    with pytest.raises(ValueError, match="equivalent rebuild"):
        _validate_run_result(result.model_copy(update={"rebuild_equivalent": False}))
    with pytest.raises(ValueError, match="configuration attestation"):
        _validate_run_result(result.model_copy(update={"engine_configuration_digest": "f" * 64}))


class DirectDraftAdapter:
    def normalize(self, native_event: object) -> NormalizedTraceEventDraft:
        if isinstance(native_event, ExpectedIdTrap):
            return native_event.draft
        if type(native_event) is not NormalizedTraceEventDraft:
            raise ValueError("not a normalized fixture")
        return native_event

    def resolve_event_id(self, native_event: object, ordinal: int) -> UUID | None:
        if isinstance(native_event, ExpectedIdTrap):
            return native_event.expected_event_id
        return (EVENT_1_ID, EVENT_2_ID, EVENT_3_ID, EVENT_4_ID)[ordinal - 1]

    def resolve_target_request_id(
        self,
        native_event: object,
        target: DeliveryTarget,
    ) -> str | None:
        return None


class ExpectedIdTrap:
    def __init__(self, draft: NormalizedTraceEventDraft) -> None:
        self.draft = draft

    @property
    def expected_event_id(self) -> UUID:
        raise RuntimeError("private expected-ID callback failure")


class InvalidTraceAttestationAdapter(DirectDraftAdapter):
    def __init__(self, *, raises: bool) -> None:
        self.raises = raises

    @property
    def trace_digest(self) -> str:
        if self.raises:
            raise RuntimeError("private attestation callback failure")
        return "f" * 64


def _never_policy(count: int = 2) -> ScriptedPolicy:
    return ScriptedPolicy(
        ScriptedPolicyConfig(
            schema_version="1.0",
            policy_kind="scripted",
            decisions=(False,) * count,
            on_exhaustion="silence",
        )
    )


@pytest.mark.parametrize(
    ("native_events", "trace_digest"),
    (
        ([], "0" * 64),
        ((), "0" * 64),
        ((_draft(1),), cast(str, object())),
        ((_draft(1),), "0" * 63),
        ((_draft(1),), "g" * 64),
    ),
)
async def test_replay_rejects_malformed_top_level_inputs_without_ingestion(
    native_events: object,
    trace_digest: object,
) -> None:
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )
    engine = _engine(
        repository,
        DirectDraftAdapter(),
        policy=_never_policy(1),
    )

    with pytest.raises(ReplayEngineInputError):
        await engine.run(
            cast(tuple[object, ...], native_events),
            trace_digest=cast(str, trace_digest),
        )


async def test_replay_enforces_the_event_count_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saliencegate.runtime.engine as engine_module

    monkeypatch.setattr(engine_module, "_MAX_REPLAY_EVENTS", 0)
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )

    with pytest.raises(ReplayEngineInputError):
        await _engine(repository, DirectDraftAdapter()).run((_draft(1),), trace_digest="0" * 64)


@pytest.mark.parametrize("attestation_raises", (False, True))
async def test_replay_rejects_mismatched_or_unreadable_trace_attestation(
    attestation_raises: bool,
) -> None:
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )

    with pytest.raises(ReplayEngineInputError):
        await _engine(
            repository,
            InvalidTraceAttestationAdapter(raises=attestation_raises),
            policy=_never_policy(1),
        ).run((_draft(1),), trace_digest="0" * 64)


def _other_run_draft(ordinal: int, *, timestamp: datetime) -> NormalizedTraceEventDraft:
    values = _draft(ordinal).model_dump(mode="python")
    values.update(
        run_id=OTHER_RUN_ID,
        source_event_id=f"other-run-{ordinal}",
        timestamp=timestamp,
        parent_ids=(),
    )
    return NormalizedTraceEventDraft.model_validate(values)


@pytest.mark.parametrize("second_kind", ("cross_run", "non_monotonic"))
async def test_replay_rejects_cross_run_and_non_monotonic_event_sequences(
    second_kind: str,
) -> None:
    first = _draft(1)
    second = (
        _other_run_draft(2, timestamp=first.timestamp + timedelta(seconds=1))
        if second_kind == "cross_run"
        else NormalizedTraceEventDraft.model_validate(
            {
                **_draft(2).model_dump(mode="python"),
                "timestamp": first.timestamp - timedelta(microseconds=1),
                "parent_ids": (),
            }
        )
    )
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID, EVENT_2_ID)).__next__,
    )

    with pytest.raises(ReplayEngineInputError):
        await _engine(
            repository,
            DirectDraftAdapter(),
            policy=_never_policy(),
        ).run((first, second), trace_digest="0" * 64)

    with pytest.raises(RunNotFoundError):
        await repository.ledger(RUN_ID)


async def test_replay_rejects_duplicate_append_and_expected_id_callback_failure() -> None:
    duplicate_repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )
    draft = _draft(1)
    with pytest.raises(ReplayEngineInputError):
        await _engine(
            duplicate_repository,
            DirectDraftAdapter(),
            policy=_never_policy(),
        ).run((draft, draft), trace_digest="0" * 64)

    trap_repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )
    with pytest.raises(ReplayEngineInputError):
        await _engine(
            trap_repository,
            DirectDraftAdapter(),
            policy=_never_policy(1),
        ).run((ExpectedIdTrap(draft),), trace_digest="0" * 64)


class MismatchedEventIdRepository(MemoryRunRepository):
    def __init__(self) -> None:
        super().__init__(synthetic_benchmark=True)

    async def append(
        self,
        event: NormalizedTraceEventDraft,
        *,
        event_id: UUID | None = None,
    ) -> AppendReceipt:
        return await super().append(event, event_id=OTHER_EVENT_ID)


async def test_replay_rejects_an_authoritative_event_id_mismatch() -> None:
    trace = one_event_trace()
    repository = MismatchedEventIdRepository()

    with pytest.raises(ReplayEngineInvariantError):
        await _engine(repository, trace, policy=_never_policy(1)).run(
            trace.events,
            trace_digest=trace.manifest.trace_digest,
        )


class CrossRunReceiptRepository(MemoryRunRepository):
    def __init__(self) -> None:
        super().__init__(
            synthetic_benchmark=True,
            id_factory=iter((EVENT_1_ID, EVENT_2_ID)).__next__,
        )
        self.append_count = 0

    async def append(
        self,
        event: NormalizedTraceEventDraft,
        *,
        event_id: UUID | None = None,
    ) -> AppendReceipt:
        receipt = await super().append(event, event_id=event_id)
        self.append_count += 1
        if self.append_count == 2:
            forged_event = receipt.event.model_copy(update={"run_id": OTHER_RUN_ID})
            return receipt.model_copy(update={"event": forged_event})
        return receipt


async def test_replay_revalidates_the_authoritative_event_run_after_append() -> None:
    drafts = (_draft(1), _draft(2))
    repository = CrossRunReceiptRepository()

    with pytest.raises(ReplayEngineInputError):
        await _engine(
            repository,
            DirectDraftAdapter(),
            policy=_never_policy(),
        ).run(drafts, trace_digest="0" * 64)


class ExplodingExtractor:
    def extract(self, context: DetectionContext) -> tuple[Signal, ...]:
        raise RuntimeError(f"private extraction error for {context.run_id}")


class ExplodingPolicy:
    def decide(
        self,
        signals: list[Signal],
        state: RunState,
        budget: BudgetSnapshot,
    ) -> InvocationDecision:
        raise RuntimeError(f"private policy error at {state.event_sequence}")


class InvalidPolicy:
    def decide(
        self,
        signals: list[Signal],
        state: RunState,
        budget: BudgetSnapshot,
    ) -> InvocationDecision:
        return cast(InvocationDecision, object())


@pytest.mark.parametrize("boundary", ("extractor", "policy_exception", "policy_value"))
async def test_replay_fails_closed_at_extractor_and_policy_boundaries(boundary: str) -> None:
    trace = one_event_trace()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )
    extractor: ReplaySignalExtractor = (
        ExplodingExtractor() if boundary == "extractor" else DeterministicSignalExtractor(())
    )
    policy: ReplayTriggerPolicy = (
        ExplodingPolicy()
        if boundary == "policy_exception"
        else InvalidPolicy()
        if boundary == "policy_value"
        else _never_policy(1)
    )

    with pytest.raises(ReplayEngineInputError):
        await _engine(
            repository,
            trace,
            extractor=extractor,
            policy=policy,
        ).run(trace.events, trace_digest=trace.manifest.trace_digest)


class OneSignalExtractor:
    def extract(self, context: DetectionContext) -> tuple[Signal, ...]:
        event = context.events[-1]
        return (
            Signal(
                signal_id=UUID("00000000-0000-4000-8000-00000000d090"),
                run_id=context.run_id,
                created_at=event.timestamp,
                signal_type=SignalType.TOOL_ERROR,
                strength=1.0,
                evidence_event_ids=(event.event_id,),
                detector_version="engine-fixture/1",
                reason_code=ReasonCode.TOOL_ERROR,
            ),
        )


async def test_replay_persists_extracted_signals_before_a_silence_decision() -> None:
    trace = one_event_trace()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )

    result = await _engine(
        repository,
        trace,
        extractor=OneSignalExtractor(),
        policy=_never_policy(1),
    ).run(trace.events, trace_digest=trace.manifest.trace_digest)

    assert len(result.events[0].signals) == 1
    assert any(isinstance(entry.record, Signal) for entry in await repository.ledger(RUN_ID))


def _last_cycle(entries: tuple[LedgerEntry, ...]) -> CycleRecord:
    cycles = tuple(entry.record for entry in entries if isinstance(entry.record, CycleRecord))
    assert cycles
    return cycles[-1]


@pytest.mark.parametrize("failure_kind", ("wrong_request_digest", "usage_overflow"))
async def test_model_identity_and_usage_failures_consume_the_reserved_unknown_cost(
    failure_kind: str,
) -> None:
    trace = one_event_trace()
    config = engine_config()
    model = CompletedModel(
        _silent_output(),
        request_digest=("b" * 64 if failure_kind == "wrong_request_digest" else None),
        usage=(
            ModelUsage(
                input_tokens=config.reservation.input_tokens + 1,
                latency_us=100,
            )
            if failure_kind == "usage_overflow"
            else None
        ),
    )
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )

    with pytest.raises(ReplayEngineModelError):
        await _engine(repository, trace, model=model).run(
            trace.events,
            trace_digest=trace.manifest.trace_digest,
        )

    cycle = _last_cycle(await repository.ledger(RUN_ID))
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is ReasonCode.FAILED_UNKNOWN_COST
    assert cycle.budget_settlement == config.reservation


class KnownModelError:
    async def generate(self, request: ModelRequest) -> ModelResult:
        return ModelResult(
            status=ModelCallStatus.MODEL_ERROR,
            request_digest=request.request_digest,
            output=None,
            usage=ModelUsage(latency_us=100),
        )


async def test_known_model_error_records_known_usage_without_an_outcome() -> None:
    trace = one_event_trace()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )

    result = await _engine(repository, trace, model=KnownModelError()).run(
        trace.events,
        trace_digest=trace.manifest.trace_digest,
    )

    cycle = result.events[0].cycle
    assert cycle is not None
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is ReasonCode.MODEL_ERROR
    assert cycle.budget_settlement is not None
    assert cycle.budget_settlement.model_calls == 1
    assert result.outcomes == ()


@pytest.mark.parametrize(
    "invalid_delta",
    ("timestamp", "future_timestamp", "run", "memory_conflict"),
)
async def test_invalid_model_deltas_terminalize_without_mutating_memory(
    invalid_delta: str,
) -> None:
    trace = one_event_trace()
    updates: tuple[MemoryUpdate, ...] = ()
    run_id = RUN_ID
    created_at = NOW + timedelta(seconds=1)
    expected_reason = ReasonCode.INVALID_STRUCTURED_OUTPUT
    if invalid_delta == "timestamp":
        created_at = NOW
    elif invalid_delta == "future_timestamp":
        created_at = NOW + timedelta(days=365)
    elif invalid_delta == "run":
        run_id = OTHER_RUN_ID
    else:
        updates = (
            MemoryUpdate(
                memory_id=UUID("00000000-0000-4000-8000-00000000d092"),
                expected_revision=1,
                content="A missing memory cannot be updated by replay.",
            ),
        )
        expected_reason = ReasonCode.MEMORY_CONFLICT
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )

    result = await _engine(
        repository,
        trace,
        model=CompletedModel(_silent_output(run_id=run_id, created_at=created_at, updates=updates)),
    ).run(trace.events, trace_digest=trace.manifest.trace_digest)

    cycle = result.events[0].cycle
    assert cycle is not None
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is expected_reason
    assert result.outcomes == ()
    assert (await repository.snapshot(RUN_ID)).records == ()


class ExplodingGroundingPipeline(GroundingPipeline):
    def replay_receipt(
        self,
        receipt: GroundingReceipt,
        *,
        context: GroundingContext,
        state: GroundingState,
    ) -> InterventionDecision:
        raise RuntimeError(f"private grounding error for {context.cycle_id}")


async def test_grounding_failure_terminalizes_as_invalid_structured_output() -> None:
    trace = one_event_trace()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )
    grounding = ExplodingGroundingPipeline(grounding_pipeline().configuration)

    result = await _engine(repository, trace, grounding=grounding).run(
        trace.events,
        trace_digest=trace.manifest.trace_digest,
    )

    cycle = result.events[0].cycle
    assert cycle is not None
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is ReasonCode.INVALID_STRUCTURED_OUTPUT
    assert result.outcomes == ()


class MissingEvidenceReminderModel:
    async def generate(self, request: ModelRequest) -> ModelResult:
        output = MemoryCycleOutput(
            delta=MemoryDelta(
                delta_id=DELTA_1_ID,
                run_id=RUN_ID,
                created_at=NOW + timedelta(seconds=1),
            ),
            observation=GroundingObservation(
                parse_status=ProposalParseStatus.VALID,
                proposal_action=InterventionAction.REMIND,
                claims=(
                    ProposedClaim(
                        kind=ClaimKind.REQUIREMENT,
                        evidence=EvidenceReference(
                            source=EvidenceSource.EVENT,
                            source_id=OTHER_EVENT_ID,
                            field_path="/payload/message",
                        ),
                    ),
                ),
                confidence=1.0,
            ),
        )
        return ModelResult(
            status=ModelCallStatus.COMPLETED,
            request_digest=request.request_digest,
            output=output,
            usage=ModelUsage(latency_us=100),
        )


async def test_missing_grounding_evidence_is_a_committed_safe_silence() -> None:
    trace = one_event_trace(target_request_id="engine-request-1")
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )

    result = await _engine(
        repository,
        trace,
        model=MissingEvidenceReminderModel(),
        delivery_adapter=FailingDeliveryAdapter(),
    ).run(trace.events, trace_digest=trace.manifest.trace_digest)

    cycle = result.events[0].cycle
    assert cycle is not None and cycle.intervention is not None
    assert cycle.state is CycleState.COMMITTED
    assert cycle.intervention.action is InterventionAction.SILENCE
    assert cycle.intervention.reason_code is ReasonCode.CITATION_MISSING
    assert result.events[0].delivery is None


def _config_with(
    *,
    target: DeliveryTarget | None = DeliveryTarget.NEXT_MODEL_CALL,
    reservation: BudgetAmounts | None = None,
) -> ReplayEngineConfig:
    values = engine_config().model_dump(mode="python")
    values["requested_delivery_target"] = target
    if reservation is not None:
        values["reservation"] = reservation
    return ReplayEngineConfig.model_validate(values)


async def test_intervention_usage_cannot_exceed_its_reserved_budget() -> None:
    trace = one_event_trace(target_request_id="engine-request-1")
    config = engine_config()
    reservation_values = config.reservation.model_dump(mode="python")
    reservation_values["interventions"] = 0
    zero_intervention_reservation = BudgetAmounts.model_validate(reservation_values)
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )

    with pytest.raises(ReplayEngineModelError):
        await _engine(
            repository,
            trace,
            model=ReminderWithoutTargetModel(),
            config=_config_with(reservation=zero_intervention_reservation),
            delivery_adapter=FailingDeliveryAdapter(),
        ).run(trace.events, trace_digest=trace.manifest.trace_digest)

    cycle = _last_cycle(await repository.ledger(RUN_ID))
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is ReasonCode.FAILED_UNKNOWN_COST
    assert cycle.budget_settlement == zero_intervention_reservation


class RejectingCommitRepository(MemoryRunRepository):
    async def commit_cycle(self, command: CommitCycle) -> CycleReceipt:
        raise RepositoryError("fixture commit rejection")


class RejectingCommitAndFailRepository(RejectingCommitRepository):
    async def fail_cycle(self, command: FailCycle) -> CycleReceipt:
        raise RepositoryError("fixture terminalization rejection")


@pytest.mark.parametrize("terminalization_fails", (False, True))
async def test_commit_rejection_is_terminalized_or_escalated_as_an_invariant(
    terminalization_fails: bool,
) -> None:
    trace = one_event_trace()
    repository = (
        RejectingCommitAndFailRepository(
            synthetic_benchmark=True,
            id_factory=iter((EVENT_1_ID,)).__next__,
        )
        if terminalization_fails
        else RejectingCommitRepository(
            synthetic_benchmark=True,
            id_factory=iter((EVENT_1_ID,)).__next__,
        )
    )
    engine = _engine(repository, trace)

    if terminalization_fails:
        with pytest.raises(ReplayEngineInvariantError):
            await engine.run(trace.events, trace_digest=trace.manifest.trace_digest)
        assert _last_cycle(await repository.ledger(RUN_ID)).state is CycleState.RUNNING
    else:
        result = await engine.run(trace.events, trace_digest=trace.manifest.trace_digest)
        cycle = result.events[0].cycle
        assert cycle is not None
        assert cycle.state is CycleState.FAILED
        assert cycle.failure_reason is ReasonCode.INVALID_STRUCTURED_OUTPUT
        assert result.outcomes == ()


class ExplodingCapabilitiesAdapter(FailingDeliveryAdapter):
    def capabilities(self) -> AdapterCapabilities:
        raise RuntimeError("private capabilities callback failure")


@pytest.mark.parametrize(
    "binding_case",
    ("target_disabled", "pre_action_unsupported", "target_unresolved", "capabilities_error"),
)
async def test_unavailable_delivery_bindings_fail_closed_to_grounded_silence(
    binding_case: str,
) -> None:
    target = (
        None
        if binding_case == "target_disabled"
        else DeliveryTarget.PRE_ACTION_REPLAN
        if binding_case == "pre_action_unsupported"
        else DeliveryTarget.NEXT_MODEL_CALL
    )
    trace = one_event_trace(
        target_request_id=(None if binding_case == "target_unresolved" else "engine-request-1"),
        pre_action_target_request_id="engine-action-1",
    )
    adapter: DeliveryAdapter = (
        ExplodingCapabilitiesAdapter()
        if binding_case == "capabilities_error"
        else FailingDeliveryAdapter()
    )
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )

    result = await _engine(
        repository,
        trace,
        model=ReminderWithoutTargetModel(),
        config=_config_with(target=target),
        delivery_adapter=adapter,
    ).run(trace.events, trace_digest=trace.manifest.trace_digest)

    cycle = result.events[0].cycle
    assert cycle is not None and cycle.intervention is not None
    assert cycle.state is CycleState.COMMITTED
    assert cycle.intervention.action is InterventionAction.SILENCE
    assert cycle.intervention.reason_code is ReasonCode.UNSUPPORTED_DELIVERY_TARGET
    assert result.events[0].delivery is None


class ExplodingOutcomeRecorder:
    async def record(self, outcome: InterventionOutcome) -> None:
        raise RuntimeError(f"private outcome sink error for {outcome.outcome_id}")


async def test_outcome_sink_failure_is_sanitized_after_authoritative_persistence() -> None:
    trace = one_event_trace()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )

    with pytest.raises(OutcomeRecordingError) as captured:
        await _engine(
            repository,
            trace,
            outcome_recorder=ExplodingOutcomeRecorder(),
        ).run(trace.events, trace_digest=trace.manifest.trace_digest)

    assert "private" not in str(captured.value)
    entries = await repository.ledger(RUN_ID)
    assert _last_cycle(entries).state is CycleState.COMMITTED
    assert sum(isinstance(entry.record, InterventionOutcome) for entry in entries) == 1


class NonEquivalentRebuildRepository(MemoryRunRepository):
    async def rebuild(self, run_id: UUID) -> RebuildReceipt:
        receipt = await super().rebuild(run_id)
        return receipt.model_copy(update={"equivalent": False})


async def test_non_equivalent_rebuild_is_never_returned_as_a_replay_result() -> None:
    trace = one_event_trace()
    repository = NonEquivalentRebuildRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )

    with pytest.raises(ReplayEngineInvariantError):
        await _engine(repository, trace, policy=_never_policy(1)).run(
            trace.events,
            trace_digest=trace.manifest.trace_digest,
        )


class MismatchedBudgetRepository(MemoryRunRepository):
    async def budget_snapshot(self, run_id: UUID) -> BudgetSnapshot:
        snapshot = await super().budget_snapshot(run_id)
        limits = snapshot.limits.model_copy(
            update={"input_tokens": snapshot.limits.input_tokens + 1}
        )
        return snapshot.model_copy(update={"limits": limits})


async def test_replay_rejects_repository_budget_configuration_drift() -> None:
    events = (
        JsonlReplayEvent.create(ordinal=1, expected_event_id=EVENT_1_ID, draft=_draft(1)),
        JsonlReplayEvent.create(ordinal=2, expected_event_id=EVENT_2_ID, draft=_draft(2)),
    )
    trace = JSONLReplayAdapter.from_bytes(encode_jsonl_trace(events))
    repository = MismatchedBudgetRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID, EVENT_2_ID)).__next__,
    )

    with pytest.raises(ReplayEngineInvariantError):
        await _engine(repository, trace, policy=_never_policy()).run(
            trace.events,
            trace_digest=trace.manifest.trace_digest,
        )


def test_projection_helper_sanitizes_malformed_ledger_entries() -> None:
    with pytest.raises(ReplayEngineInvariantError):
        _projection(RUN_ID, (cast(LedgerEntry, object()),))


def two_event_trace() -> JSONLReplayAdapter:
    events = (
        JsonlReplayEvent.create(
            ordinal=1,
            expected_event_id=EVENT_1_ID,
            draft=_draft(1),
        ),
        JsonlReplayEvent.create(
            ordinal=2,
            expected_event_id=EVENT_2_ID,
            draft=_draft(2, parent_ids=(EVENT_1_ID,)),
        ),
    )
    return JSONLReplayAdapter.from_bytes(encode_jsonl_trace(events))


async def test_attested_trace_rejects_prefix_before_any_ingestion() -> None:
    trace = trace_adapter()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )

    with pytest.raises(ReplayEngineInputError):
        await _engine(repository, trace, policy=_never_policy(1)).run(
            trace.events[:1],
            trace_digest=trace.manifest.trace_digest,
        )

    with pytest.raises(RunNotFoundError):
        await repository.ledger(RUN_ID)


async def test_replay_requires_a_fresh_authoritative_run() -> None:
    draft = _draft(1)
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )
    await repository.append(draft)

    with pytest.raises(ReplayEngineInputError):
        await _engine(
            repository,
            DirectDraftAdapter(),
            policy=_never_policy(1),
        ).run((draft,), trace_digest="0" * 64)


async def test_generic_adapter_uses_engine_owned_normalized_trace_identity() -> None:
    draft = _draft(1)
    results: list[ReplayRunResult] = []
    for claimed_digest, repository_event_id in (
        ("0" * 64, EVENT_1_ID),
        ("f" * 64, OTHER_EVENT_ID),
    ):
        repository = MemoryRunRepository(
            synthetic_benchmark=True,
            id_factory=iter((repository_event_id,)).__next__,
        )
        results.append(
            await _engine(
                repository,
                DirectDraftAdapter(),
                policy=_never_policy(1),
            ).run((draft,), trace_digest=claimed_digest)
        )

    expected = normalized_trace_digest((draft,))
    assert results[0].trace_attestation_mode == "engine_normalized"
    assert results[0].normalized_trace_digest == expected
    assert results[0].trace_digest != expected
    assert results[0].trace_digest == results[1].trace_digest
    assert results[0].trace_expected_event_ids == results[1].trace_expected_event_ids
    assert results[0].events[0].event.event_id == results[1].events[0].event.event_id
    assert canonical_json(results[0]) == canonical_json(results[1])
    with pytest.raises(ValueError, match="engine-normalized trace attestation"):
        _validate_run_result(results[0].model_copy(update={"trace_digest": "f" * 64}))


async def test_input_trace_and_persisted_redacted_event_have_separate_attestations() -> None:
    draft = _draft(1).model_copy(update={"payload": {"message": "contains configured-literal"}})
    repository = MemoryRunRepository(
        redaction_policy=RedactionPolicy(literal_secrets=("configured-literal",)),
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )

    result = await _engine(
        repository,
        DirectDraftAdapter(),
        policy=_never_policy(1),
    ).run((draft,), trace_digest="0" * 64)

    assert result.normalized_draft_digests != result.persisted_event_draft_digests
    assert result.events[0].event.payload["message"] == "contains [REDACTED]"
    assert _validate_run_result(result) == result


async def test_generic_adapter_event_id_mapping_preserves_parent_graphs() -> None:
    drafts = (
        _draft(1),
        _draft(2, parent_ids=(EVENT_1_ID,)),
    )
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((OTHER_EVENT_ID,)).__next__,
    )

    result = await _engine(
        repository,
        DirectDraftAdapter(),
        policy=_never_policy(),
    ).run(drafts, trace_digest="0" * 64)

    assert tuple(item.event.event_id for item in result.events) == (EVENT_1_ID, EVENT_2_ID)
    assert result.events[1].event.parent_ids == (EVENT_1_ID,)


class RoutedDraftAdapter(DirectDraftAdapter):
    def __init__(self, target_request_id: str) -> None:
        self.target_request_id = target_request_id
        self.resolve_calls = 0

    def resolve_target_request_id(
        self,
        native_event: object,
        target: DeliveryTarget,
    ) -> str | None:
        self.resolve_calls += 1
        return self.target_request_id


async def test_generic_routing_is_frozen_once_and_bound_to_execution_identity() -> None:
    draft = _draft(1)
    results: list[ReplayRunResult] = []
    adapters: list[RoutedDraftAdapter] = []
    for target_request_id in ("request-a", "request-b"):
        adapter = RoutedDraftAdapter(target_request_id)
        adapters.append(adapter)
        results.append(
            await _engine(
                MemoryRunRepository(
                    synthetic_benchmark=True,
                    id_factory=iter((EVENT_1_ID,)).__next__,
                ),
                adapter,
                policy=_never_policy(1),
                delivery_adapter=FailingDeliveryAdapter(),
            ).run((draft,), trace_digest="0" * 64)
        )

    assert tuple(adapter.resolve_calls for adapter in adapters) == (1, 1)
    assert results[0].normalized_trace_digest == results[1].normalized_trace_digest
    assert results[0].routing_digest != results[1].routing_digest
    assert results[0].trace_digest != results[1].trace_digest
    assert results[0].events[0].decision.decision_id != results[1].events[0].decision.decision_id


class PerEventSilentModel:
    def __init__(self, *, first_procedure: str | None = None) -> None:
        self.first_procedure = first_procedure
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResult:
        self.requests.append(request)
        call = len(self.requests)
        creates: tuple[MemoryCreate, ...] = ()
        if call == 1 and self.first_procedure is not None:
            creates = (
                MemoryCreate(
                    handle="bounded-procedure",
                    kind=MemoryKind.PROCEDURAL,
                    content=self.first_procedure,
                    provenance=(
                        EvidenceReference(
                            source=EvidenceSource.EVENT,
                            source_id=EVENT_1_ID,
                            field_path="/payload/message",
                        ),
                    ),
                    confidence=1.0,
                    trust_label=TrustLabel.TRUSTED_CONTROLLER,
                ),
            )
        output = MemoryCycleOutput(
            delta=MemoryDelta(
                delta_id=(DELTA_1_ID, DELTA_2_ID)[call - 1],
                run_id=RUN_ID,
                creates=creates,
                created_at=NOW + timedelta(seconds=call),
            ),
            observation=GroundingObservation(
                parse_status=ProposalParseStatus.VALID,
                proposal_action=InterventionAction.SILENCE,
                claims=(),
                confidence=1.0,
            ),
        )
        return ModelResult(
            status=ModelCallStatus.COMPLETED,
            request_digest=request.request_digest,
            output=output,
            usage=ModelUsage(
                input_tokens=10,
                output_tokens=5,
                canonical_token_equivalents=15,
                latency_us=100,
            ),
        )


async def test_budget_denial_never_leaves_a_pending_cycle() -> None:
    trace = two_event_trace()
    config_values = engine_config().model_dump(mode="python")
    limit_values = cast(dict[str, object], config_values["budget_limits"])
    config_values["budget_limits"] = {**limit_values, "model_calls": 1}
    config = ReplayEngineConfig.model_validate(config_values)
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )

    with pytest.raises(ReplayEngineInputError):
        await _engine(
            repository,
            trace,
            model=PerEventSilentModel(),
            policy=ScriptedPolicy(
                ScriptedPolicyConfig(
                    schema_version="1.0",
                    policy_kind="scripted",
                    decisions=(True, True),
                    on_exhaustion="silence",
                )
            ),
            config=config,
        ).run(trace.events, trace_digest=trace.manifest.trace_digest)

    projection = _projection(RUN_ID, await repository.ledger(RUN_ID))
    assert projection.cycles
    assert all(
        cycle.state in (CycleState.COMMITTED, CycleState.FAILED)
        for cycle in projection.cycles.values()
    )


async def test_oversized_procedural_candidate_is_skipped_before_model_start() -> None:
    trace = two_event_trace()
    model = PerEventSilentModel(first_procedure="x" * 20_000)
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )

    result = await _engine(
        repository,
        trace,
        model=model,
        policy=ScriptedPolicy(
            ScriptedPolicyConfig(
                schema_version="1.0",
                policy_kind="scripted",
                decisions=(True, True),
                on_exhaustion="silence",
            )
        ),
        config=_config_with(
            reservation=engine_config().reservation.model_copy(
                update={"output_tokens": 6_000, "canonical_token_equivalents": 8_000}
            )
        ),
    ).run(trace.events, trace_digest=trace.manifest.trace_digest)

    assert len(model.requests) == 2
    second_payload = ReplayModelPayload.model_validate_json(
        canonical_json(model.requests[1].payload)
    )
    assert second_payload.candidate_memories == ()
    assert all(
        item.cycle is not None and item.cycle.state is CycleState.COMMITTED
        for item in result.events
    )
    snapshot = await repository.snapshot(RUN_ID)
    assert snapshot.records[0].trust_label is TrustLabel.UNTRUSTED_MODEL_OUTPUT


class OrderedSignalsExtractor:
    def __init__(self, *, reverse: bool) -> None:
        self.reverse = reverse

    def extract(self, context: DetectionContext) -> tuple[Signal, ...]:
        current = context.current
        values = (
            Signal(
                signal_id=UUID("00000000-0000-4000-8000-00000000d094"),
                run_id=context.run_id,
                created_at=current.timestamp,
                signal_type=SignalType.TOOL_ERROR,
                strength=1.0,
                evidence_event_ids=(current.event_id,),
                detector_version="ordered-fixture/1",
                reason_code=ReasonCode.TOOL_ERROR,
            ),
            Signal(
                signal_id=UUID("00000000-0000-4000-8000-00000000d095"),
                run_id=context.run_id,
                created_at=current.timestamp,
                signal_type=SignalType.TEST_FAILURE,
                strength=1.0,
                evidence_event_ids=(current.event_id,),
                detector_version="ordered-fixture/1",
                reason_code=ReasonCode.TEST_FAILURE,
            ),
        )
        return tuple(reversed(values)) if self.reverse else values


async def test_signal_order_cannot_change_replay_artifact_bytes() -> None:
    trace = one_event_trace()
    results: list[ReplayRunResult] = []
    for reverse in (False, True):
        repository = MemoryRunRepository(
            synthetic_benchmark=True,
            id_factory=trace.event_id_factory(),
        )
        results.append(
            await _engine(
                repository,
                trace,
                extractor=OrderedSignalsExtractor(reverse=reverse),
                policy=_never_policy(1),
            ).run(trace.events, trace_digest=trace.manifest.trace_digest)
        )

    assert canonical_json(results[0]) == canonical_json(results[1])


class CrossCursorSignalExtractor:
    def extract(self, context: DetectionContext) -> tuple[Signal, ...]:
        if len(context.events) == 1:
            return ()
        return (
            Signal(
                signal_id=UUID("00000000-0000-4000-8000-00000000d096"),
                run_id=context.run_id,
                created_at=context.current.timestamp,
                signal_type=SignalType.TOOL_ERROR,
                strength=1.0,
                evidence_event_ids=(
                    context.events[0].event_id,
                    context.current.event_id,
                ),
                detector_version="cross-cursor-fixture/1",
                reason_code=ReasonCode.TOOL_ERROR,
            ),
        )


class ReverseEvidenceSignalExtractor(CrossCursorSignalExtractor):
    def extract(self, context: DetectionContext) -> tuple[Signal, ...]:
        signals = super().extract(context)
        if not signals:
            return ()
        return (
            signals[0].model_copy(
                update={"evidence_event_ids": tuple(reversed(signals[0].evidence_event_ids))}
            ),
        )


async def test_signal_evidence_order_is_canonical() -> None:
    trace = two_event_trace()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )

    with pytest.raises(ReplayEngineInputError):
        await _engine(
            repository,
            trace,
            extractor=ReverseEvidenceSignalExtractor(),
            policy=_never_policy(),
        ).run(trace.events, trace_digest=trace.manifest.trace_digest)


async def test_cross_cursor_signal_keeps_current_event_priority() -> None:
    trace = two_event_trace()
    model = PerEventSilentModel()
    config = engine_config()
    config = ReplayEngineConfig.model_validate(
        {
            **config.model_dump(mode="python"),
            "batch": config.batch.model_copy(
                update={
                    "recent_event_count": 0,
                    "max_tool_errors": 1,
                }
            ),
        }
    )
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )

    await _engine(
        repository,
        trace,
        model=model,
        extractor=CrossCursorSignalExtractor(),
        policy=ScriptedPolicy(
            ScriptedPolicyConfig(
                schema_version="1.0",
                policy_kind="scripted",
                decisions=(True, True),
                on_exhaustion="silence",
            )
        ),
        config=config,
    ).run(trace.events, trace_digest=trace.manifest.trace_digest)

    payload = ReplayModelPayload.model_validate_json(canonical_json(model.requests[1].payload))
    assert tuple(item.event.event_id for item in payload.batch.payload.verbatim_events) == (
        EVENT_2_ID,
    )
    assert payload.batch.payload.verbatim_events[0].priority_kinds == (
        BatchPriorityKind.TOOL_ERROR,
    )


async def test_unexpected_preview_failure_terminalizes_running_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saliencegate.runtime.engine as engine_module

    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("private preview failure")

    monkeypatch.setattr(engine_module, "preview_memory_delta", explode)
    trace = one_event_trace()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )

    with pytest.raises(ReplayEngineInvariantError):
        await _engine(repository, trace).run(
            trace.events,
            trace_digest=trace.manifest.trace_digest,
        )

    cycle = _last_cycle(await repository.ledger(RUN_ID))
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is ReasonCode.FAILED_UNKNOWN_COST


class ConcurrentAppendRepository(MemoryRunRepository):
    def __init__(self) -> None:
        super().__init__(
            synthetic_benchmark=True,
            id_factory=iter((EVENT_1_ID, EVENT_2_ID)).__next__,
        )
        self.injected = False

    async def append(
        self,
        event: NormalizedTraceEventDraft,
        *,
        event_id: UUID | None = None,
    ) -> AppendReceipt:
        receipt = await super().append(event, event_id=event_id)
        if not self.injected:
            self.injected = True
            await super().append(
                _draft(2, parent_ids=(EVENT_1_ID,)),
                event_id=EVENT_2_ID,
            )
        return receipt


async def test_observed_concurrent_append_never_contaminates_a_replay_result() -> None:
    trace = one_event_trace()
    repository = ConcurrentAppendRepository()

    with pytest.raises(ReplayEngineInvariantError):
        await _engine(repository, trace, policy=_never_policy(1)).run(
            trace.events,
            trace_digest=trace.manifest.trace_digest,
        )


class ConcurrentSignalRepository(MemoryRunRepository):
    def __init__(self) -> None:
        super().__init__(
            synthetic_benchmark=True,
            id_factory=iter((EVENT_1_ID,)).__next__,
        )

    async def append(
        self,
        event: NormalizedTraceEventDraft,
        *,
        event_id: UUID | None = None,
    ) -> AppendReceipt:
        receipt = await super().append(event, event_id=event_id)
        await super().record_signal(
            Signal(
                signal_id=UUID("00000000-0000-4000-8000-00000000d097"),
                run_id=receipt.event.run_id,
                created_at=receipt.event.timestamp,
                signal_type=SignalType.TOOL_ERROR,
                strength=1.0,
                evidence_event_ids=(receipt.event.event_id,),
                detector_version="concurrent-fixture/1",
                reason_code=ReasonCode.TOOL_ERROR,
            )
        )
        return receipt


async def test_concurrent_non_event_record_never_contaminates_replay_result() -> None:
    trace = one_event_trace()

    with pytest.raises(ReplayEngineInvariantError):
        await _engine(
            ConcurrentSignalRepository(),
            trace,
            policy=_never_policy(1),
        ).run(trace.events, trace_digest=trace.manifest.trace_digest)


class OneSignalPerEventExtractor:
    def extract(self, context: DetectionContext) -> tuple[Signal, ...]:
        current = context.current
        return (
            Signal(
                signal_id=(
                    UUID("00000000-0000-4000-8000-00000000d094")
                    if current.sequence == 1
                    else UUID("00000000-0000-4000-8000-00000000d095")
                ),
                run_id=context.run_id,
                created_at=current.timestamp,
                signal_type=SignalType.TOOL_ERROR,
                strength=1.0,
                evidence_event_ids=(current.event_id,),
                detector_version="bounded-signal-fixture/1",
                reason_code=ReasonCode.TOOL_ERROR,
            ),
        )


async def test_signals_have_a_cumulative_run_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    import saliencegate.runtime.engine as engine_module

    monkeypatch.setattr(engine_module, "_MAX_SIGNALS_PER_RUN", 1)
    trace = two_event_trace()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )

    with pytest.raises(ReplayEngineInputError):
        await _engine(
            repository,
            trace,
            extractor=OneSignalPerEventExtractor(),
            policy=_never_policy(),
        ).run(trace.events, trace_digest=trace.manifest.trace_digest)

    signals = tuple(
        entry.record
        for entry in await repository.ledger(RUN_ID)
        if isinstance(entry.record, Signal)
    )
    assert len(signals) == 1


async def test_forged_signal_evidence_cardinality_is_rejected_before_copying() -> None:
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID,)).__next__,
    )
    event = (await repository.append(_draft(1))).event
    context = DetectionContext(run_id=RUN_ID, events=(event,))
    valid = OneSignalPerEventExtractor().extract(context)[0]
    forged = valid.model_copy(update={"evidence_event_ids": (EVENT_1_ID,) * 65})

    with pytest.raises(ReplayEngineInputError):
        _validated_signals((forged,), context)


async def test_model_request_budget_overflow_fails_before_model_start() -> None:
    trace = one_event_trace()
    model = UncalledModel()
    reservation = engine_config().reservation.model_copy(
        update={"input_tokens": 1, "canonical_token_equivalents": 1}
    )
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )

    result = await _engine(
        repository,
        trace,
        model=model,
        config=_config_with(reservation=reservation),
    ).run(trace.events, trace_digest=trace.manifest.trace_digest)

    assert model.calls == 0
    cycle = result.events[0].cycle
    assert cycle is not None
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is ReasonCode.MANDATORY_INPUT_OVERFLOW
    assert cycle.budget_settlement == BudgetAmounts()


async def test_model_request_wrapper_is_budgeted_before_model_start() -> None:
    trace = one_event_trace()
    sizing_model = PerEventSilentModel()
    sizing_repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )
    await _engine(sizing_repository, trace, model=sizing_model).run(
        trace.events,
        trace_digest=trace.manifest.trace_digest,
    )
    sized_request = sizing_model.requests[0]
    payload_bytes = len(canonical_json(sized_request.payload))
    request_bytes = len(canonical_json(sized_request))
    payload_token_equivalents = (payload_bytes + 3) // 4
    assert request_bytes > payload_token_equivalents * 4

    model = UncalledModel()
    reservation = engine_config().reservation.model_copy(
        update={"canonical_token_equivalents": payload_token_equivalents}
    )
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )
    result = await _engine(
        repository,
        trace,
        model=model,
        config=_config_with(reservation=reservation),
    ).run(trace.events, trace_digest=trace.manifest.trace_digest)

    assert model.calls == 0
    cycle = result.events[0].cycle
    assert cycle is not None
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is ReasonCode.MANDATORY_INPUT_OVERFLOW
    assert cycle.budget_settlement == BudgetAmounts()


class OversizedDeltaModel:
    async def generate(self, request: ModelRequest) -> ModelResult:
        create = MemoryCreate(
            handle="bounded-item",
            kind=MemoryKind.PROCEDURAL,
            content="bounded",
            provenance=(
                EvidenceReference(
                    source=EvidenceSource.EVENT,
                    source_id=EVENT_1_ID,
                    field_path="/payload/message",
                ),
            ),
            confidence=1.0,
            trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
        )
        valid_output = _silent_output()
        forged_delta = valid_output.delta.model_copy(update={"creates": (create,) * 65})
        forged_output = valid_output.model_copy(update={"delta": forged_delta})
        valid_result = ModelResult(
            status=ModelCallStatus.COMPLETED,
            request_digest=request.request_digest,
            output=valid_output,
            usage=ModelUsage(latency_us=100),
        )
        return valid_result.model_copy(update={"output": forged_output})


class OversizedEvidenceModel:
    async def generate(self, request: ModelRequest) -> ModelResult:
        reference = EvidenceReference(
            source=EvidenceSource.EVENT,
            source_id=EVENT_1_ID,
            field_path="/payload/message",
        )
        forged_reference = reference.model_copy(update={"field_path": "/" + "x" * 5_000})
        create = MemoryCreate(
            handle="bounded-item",
            kind=MemoryKind.PROCEDURAL,
            content="bounded",
            provenance=(reference,),
            confidence=1.0,
            trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
        ).model_copy(update={"provenance": (forged_reference,)})
        output = _silent_output()
        forged_delta = output.delta.model_copy(update={"creates": (create,)})
        forged_output = output.model_copy(update={"delta": forged_delta})
        valid_result = ModelResult(
            status=ModelCallStatus.COMPLETED,
            request_digest=request.request_digest,
            output=output,
            usage=ModelUsage(latency_us=100),
        )
        return valid_result.model_copy(update={"output": forged_output})


async def test_model_delta_cardinality_is_bounded_before_revalidation() -> None:
    trace = one_event_trace()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )

    with pytest.raises(ReplayEngineModelError):
        await _engine(repository, trace, model=OversizedDeltaModel()).run(
            trace.events,
            trace_digest=trace.manifest.trace_digest,
        )

    cycle = _last_cycle(await repository.ledger(RUN_ID))
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is ReasonCode.FAILED_UNKNOWN_COST


async def test_model_evidence_bytes_are_bounded_before_revalidation() -> None:
    trace = one_event_trace()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )

    with pytest.raises(ReplayEngineModelError):
        await _engine(repository, trace, model=OversizedEvidenceModel()).run(
            trace.events,
            trace_digest=trace.manifest.trace_digest,
        )

    cycle = _last_cycle(await repository.ledger(RUN_ID))
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is ReasonCode.FAILED_UNKNOWN_COST


async def test_model_output_bytes_cannot_exceed_the_reserved_output_budget() -> None:
    trace = one_event_trace()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )
    reservation = engine_config().reservation.model_copy(update={"output_tokens": 1})

    with pytest.raises(ReplayEngineModelError):
        await _engine(
            repository,
            trace,
            model=PerEventSilentModel(first_procedure="x" * 5_000),
            config=_config_with(reservation=reservation),
        ).run(trace.events, trace_digest=trace.manifest.trace_digest)

    cycle = _last_cycle(await repository.ledger(RUN_ID))
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is ReasonCode.FAILED_UNKNOWN_COST


async def test_canonical_byte_floor_prevents_zero_usage_underreporting() -> None:
    trace = one_event_trace()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )

    result = await _engine(repository, trace).run(
        trace.events,
        trace_digest=trace.manifest.trace_digest,
    )

    cycle = result.events[0].cycle
    assert cycle is not None and cycle.budget_settlement is not None
    assert cycle.budget_settlement.canonical_token_equivalents > 0


async def test_model_output_bytes_have_a_cumulative_run_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saliencegate.runtime.engine as engine_module

    output_bytes = len(canonical_json(_silent_output()))
    monkeypatch.setattr(engine_module, "_MAX_MODEL_OUTPUT_BYTES_PER_RUN", output_bytes * 2 - 1)
    trace = two_event_trace()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )

    with pytest.raises(ReplayEngineModelError):
        await _engine(
            repository,
            trace,
            model=PerEventSilentModel(),
            policy=ScriptedPolicy(
                ScriptedPolicyConfig(
                    schema_version="1.0",
                    policy_kind="scripted",
                    decisions=(True, True),
                    on_exhaustion="silence",
                )
            ),
        ).run(trace.events, trace_digest=trace.manifest.trace_digest)

    cycles = tuple(
        entry.record
        for entry in await repository.ledger(RUN_ID)
        if isinstance(entry.record, CycleRecord)
    )
    assert cycles[-1].state is CycleState.FAILED
    assert cycles[-1].failure_reason is ReasonCode.FAILED_UNKNOWN_COST


class BlockingModel:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, request: ModelRequest) -> ModelResult:
        self.entered.set()
        await self.release.wait()
        return ModelResult(
            status=ModelCallStatus.COMPLETED,
            request_digest=request.request_digest,
            output=_silent_output(),
            usage=ModelUsage(output_tokens=1, canonical_token_equivalents=1),
        )


class BlockingFailureCleanupRepository(MemoryRunRepository):
    def __init__(self) -> None:
        super().__init__(
            synthetic_benchmark=True,
            id_factory=iter((EVENT_1_ID,)).__next__,
        )
        self.failure_cleanup_entered = asyncio.Event()
        self.release_failure_cleanup = asyncio.Event()

    async def fail_cycle(self, command: FailCycle) -> CycleReceipt:
        self.failure_cleanup_entered.set()
        await self.release_failure_cleanup.wait()
        return await super().fail_cycle(command)


async def test_repeated_cancellation_waits_for_cycle_cleanup() -> None:
    trace = one_event_trace()
    model = BlockingModel()
    repository = BlockingFailureCleanupRepository()
    task = asyncio.create_task(
        _engine(repository, trace, model=model).run(
            trace.events,
            trace_digest=trace.manifest.trace_digest,
        )
    )

    await model.entered.wait()
    task.cancel()
    await repository.failure_cleanup_entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert _last_cycle(await repository.ledger(RUN_ID)).state is CycleState.RUNNING

    repository.release_failure_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _last_cycle(await repository.ledger(RUN_ID)).state is CycleState.FAILED


class BlockingBeginReceiptRepository(MemoryRunRepository):
    def __init__(self) -> None:
        super().__init__(
            synthetic_benchmark=True,
            id_factory=iter((EVENT_1_ID,)).__next__,
        )
        self.begin_persisted = asyncio.Event()
        self.release_receipt = asyncio.Event()

    async def begin_cycle(self, command: BeginCycle) -> CycleReceipt:
        receipt = await super().begin_cycle(command)
        self.begin_persisted.set()
        await self.release_receipt.wait()
        return receipt


async def test_cancelling_after_begin_persistence_terminalizes_the_pending_cycle() -> None:
    trace = one_event_trace()
    repository = BlockingBeginReceiptRepository()
    task = asyncio.create_task(
        _engine(repository, trace).run(
            trace.events,
            trace_digest=trace.manifest.trace_digest,
        )
    )

    await repository.begin_persisted.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    cycle = _last_cycle(await repository.ledger(RUN_ID))
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is ReasonCode.MODEL_ERROR
    assert cycle.budget_settlement is None


async def test_cancelling_a_running_model_terminalizes_unknown_cost_before_reraise() -> None:
    trace = one_event_trace()
    model = BlockingModel()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )
    task = asyncio.create_task(
        _engine(repository, trace, model=model).run(
            trace.events,
            trace_digest=trace.manifest.trace_digest,
        )
    )

    await model.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    cycle = _last_cycle(await repository.ledger(RUN_ID))
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is ReasonCode.FAILED_UNKNOWN_COST
    assert cycle.budget_settlement == engine_config().reservation


class BlockingDeliveryAdapter(FailingDeliveryAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.capability_calls = 0

    def capabilities(self) -> AdapterCapabilities:
        self.capability_calls += 1
        return delivery_capabilities()

    async def deliver(self, delivery: DeliveryEnvelope) -> DeliveryReceipt:
        self.calls.append(delivery)
        self.entered.set()
        await self.release.wait()
        raise AssertionError("fixture delivery should be cancelled")


async def test_cancelling_delivery_persists_unknown_before_reraise() -> None:
    trace = one_event_trace(target_request_id="request-cancelled")
    adapter = BlockingDeliveryAdapter()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=repository_ids(),
    )
    task = asyncio.create_task(
        _engine(
            repository,
            trace,
            model=ReminderWithoutTargetModel(),
            delivery_adapter=adapter,
        ).run(trace.events, trace_digest=trace.manifest.trace_digest)
    )

    await adapter.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    entries = await repository.ledger(RUN_ID)
    assert _last_cycle(entries).state is CycleState.COMMITTED
    deliveries = tuple(
        entry.record for entry in entries if isinstance(entry.record, DeliveryRecord)
    )
    assert deliveries[-1].state is DeliveryState.UNKNOWN
    assert deliveries[-1].reason_code is ReasonCode.DELIVERY_UNKNOWN
    assert adapter.capability_calls == 1


class BlockingUnknownCleanupRepository(MemoryRunRepository):
    def __init__(self) -> None:
        super().__init__(
            synthetic_benchmark=True,
            id_factory=repository_ids(),
        )
        self.unknown_cleanup_entered = asyncio.Event()
        self.release_unknown_cleanup = asyncio.Event()

    async def mark_delivery_unknown(
        self,
        command: MarkDeliveryUnknown,
    ) -> DeliveryTransitionReceipt:
        self.unknown_cleanup_entered.set()
        await self.release_unknown_cleanup.wait()
        return await super().mark_delivery_unknown(command)


async def test_repeated_cancellation_waits_for_delivery_cleanup() -> None:
    trace = one_event_trace(target_request_id="request-double-cancel")
    adapter = BlockingDeliveryAdapter()
    repository = BlockingUnknownCleanupRepository()
    task = asyncio.create_task(
        _engine(
            repository,
            trace,
            model=ReminderWithoutTargetModel(),
            delivery_adapter=adapter,
        ).run(trace.events, trace_digest=trace.manifest.trace_digest)
    )

    await adapter.entered.wait()
    task.cancel()
    await repository.unknown_cleanup_entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    deliveries = tuple(
        entry.record
        for entry in await repository.ledger(RUN_ID)
        if isinstance(entry.record, DeliveryRecord)
    )
    assert deliveries[-1].state is DeliveryState.ATTEMPTING

    repository.release_unknown_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    deliveries = tuple(
        entry.record
        for entry in await repository.ledger(RUN_ID)
        if isinstance(entry.record, DeliveryRecord)
    )
    assert deliveries[-1].state is DeliveryState.UNKNOWN


class BlockingAttemptReceiptRepository(MemoryRunRepository):
    def __init__(self) -> None:
        super().__init__(
            synthetic_benchmark=True,
            id_factory=repository_ids(),
        )
        self.attempt_persisted = asyncio.Event()
        self.release_receipt = asyncio.Event()

    async def begin_delivery_attempt(
        self,
        command: BeginDeliveryAttempt,
    ) -> DeliveryAttemptReceipt:
        receipt = await super().begin_delivery_attempt(command)
        self.attempt_persisted.set()
        await self.release_receipt.wait()
        return receipt


async def test_cancelling_after_attempt_persistence_marks_delivery_unknown() -> None:
    trace = one_event_trace(target_request_id="request-cancelled-attempt")
    repository = BlockingAttemptReceiptRepository()
    adapter = ImmediateDeliveryAdapter()
    task = asyncio.create_task(
        _engine(
            repository,
            trace,
            model=ReminderWithoutTargetModel(),
            delivery_adapter=adapter,
        ).run(trace.events, trace_digest=trace.manifest.trace_digest)
    )

    await repository.attempt_persisted.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    entries = await repository.ledger(RUN_ID)
    deliveries = tuple(
        entry.record for entry in entries if isinstance(entry.record, DeliveryRecord)
    )
    assert deliveries[-1].state is DeliveryState.UNKNOWN
    assert deliveries[-1].reason_code is ReasonCode.DELIVERY_UNKNOWN
    assert adapter.calls == []


class ImmediateDeliveryAdapter(FailingDeliveryAdapter):
    async def deliver(self, delivery: DeliveryEnvelope) -> DeliveryReceipt:
        self.calls.append(delivery)
        return DeliveryReceipt(
            schema_version="1.0",
            delivery_id=delivery.delivery_id,
            attempt_id=delivery.attempt_id,
            attempt_number=delivery.attempt_number,
            adapter_id=delivery.adapter_id,
            target_request_id=delivery.target_request_id,
            delivered_at=delivery.created_at,
            provider_receipt_id="provider-receipt-cancelled-complete",
        )


class BlockingCompleteDeliveryRepository(MemoryRunRepository):
    def __init__(self) -> None:
        super().__init__(
            synthetic_benchmark=True,
            id_factory=repository_ids(),
        )
        self.complete_entered = asyncio.Event()
        self.release_complete = asyncio.Event()

    async def complete_delivery(
        self,
        command: CompleteDelivery,
    ) -> DeliveryTransitionReceipt:
        self.complete_entered.set()
        await self.release_complete.wait()
        return await super().complete_delivery(command)


async def test_cancelling_delivery_completion_persists_unknown_before_reraise() -> None:
    trace = one_event_trace(target_request_id="request-cancelled-complete")
    repository = BlockingCompleteDeliveryRepository()
    task = asyncio.create_task(
        _engine(
            repository,
            trace,
            model=ReminderWithoutTargetModel(),
            delivery_adapter=ImmediateDeliveryAdapter(),
        ).run(trace.events, trace_digest=trace.manifest.trace_digest)
    )

    await repository.complete_entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    entries = await repository.ledger(RUN_ID)
    deliveries = tuple(
        entry.record for entry in entries if isinstance(entry.record, DeliveryRecord)
    )
    assert deliveries[-1].state is DeliveryState.UNKNOWN
    assert deliveries[-1].reason_code is ReasonCode.DELIVERY_UNKNOWN


def test_normalized_trace_digest_enforces_its_total_byte_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saliencegate.runtime.engine as engine_module

    monkeypatch.setattr(engine_module, "_MAX_NORMALIZED_TRACE_BYTES", 1)

    with pytest.raises(ReplayEngineInputError):
        normalized_trace_digest((_draft(1),))


class CountingDraftAdapter(DirectDraftAdapter):
    def __init__(self) -> None:
        self.normalize_calls = 0

    def normalize(self, native_event: object) -> NormalizedTraceEventDraft:
        self.normalize_calls += 1
        return super().normalize(native_event)


async def test_preflight_stops_normalizing_at_the_total_trace_byte_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saliencegate.runtime.engine as engine_module

    monkeypatch.setattr(engine_module, "_MAX_NORMALIZED_TRACE_BYTES", 1)
    adapter = CountingDraftAdapter()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=iter((EVENT_1_ID, EVENT_2_ID)).__next__,
    )

    with pytest.raises(ReplayEngineInputError):
        await _engine(repository, adapter, policy=_never_policy()).run(
            (_draft(1), _draft(2)),
            trace_digest="0" * 64,
        )

    assert adapter.normalize_calls == 1


class RejectingReserveRepository(MemoryRunRepository):
    async def reserve_cycle(self, command: ReserveCycle) -> CycleReceipt:
        raise RepositoryError("fixture reserve rejection")


async def test_unexpected_reservation_failure_terminalizes_pending_cycle() -> None:
    trace = one_event_trace()
    repository = RejectingReserveRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )

    with pytest.raises(ReplayEngineInvariantError):
        await _engine(repository, trace).run(
            trace.events,
            trace_digest=trace.manifest.trace_digest,
        )

    cycle = _last_cycle(await repository.ledger(RUN_ID))
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is ReasonCode.MODEL_ERROR
    assert cycle.budget_settlement is None


async def test_unexpected_request_builder_failure_terminalizes_reserved_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode_request(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("private request construction failure")

    monkeypatch.setattr(ReplayEngine, "_model_request", explode_request)
    trace = one_event_trace()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )

    with pytest.raises(ReplayEngineInvariantError):
        await _engine(repository, trace).run(
            trace.events,
            trace_digest=trace.manifest.trace_digest,
        )

    cycle = _last_cycle(await repository.ledger(RUN_ID))
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is ReasonCode.MODEL_ERROR
    assert cycle.budget_settlement == BudgetAmounts()


async def test_result_validators_cover_fixture_order_outcome_and_delivery_forgery() -> None:
    result, *_ = await run_replay(
        MemoryRunRepository(
            synthetic_benchmark=True,
            id_factory=repository_ids(),
        )
    )

    serialized = result.model_dump(mode="python", warnings=False)
    serialized.pop("result_digest")
    with pytest.raises(ValueError):
        ReplayRunResult.model_validate(serialized)
    with pytest.raises(ValueError, match="result digest"):
        ReplayRunResult.model_validate({**serialized, "result_digest": "0" * 64})

    with pytest.raises(ValueError, match="cannot claim"):
        _validate_run_result(
            result.model_copy(
                update={
                    "replay_id": "forged/1",
                    "fixture_digest": "f" * 64,
                    "fixture_response_count": 1,
                    "fixture_consumed_count": 1,
                }
            )
        )
    with pytest.raises(ValueError, match="completely consumed"):
        _validate_run_result(
            result.model_copy(
                update={
                    "model_execution_mode": "frozen_replay",
                    "replay_id": "fixture/1",
                    "fixture_digest": "f" * 64,
                    "fixture_response_count": 2,
                    "fixture_consumed_count": 1,
                }
            )
        )

    reordered = (result.events[1], result.events[0], *result.events[2:])
    reordered_decisions = tuple(item.decision.model_dump(mode="json") for item in reordered)
    with pytest.raises(ValueError, match=r"normalized trace attestation|ordered trace"):
        _validate_run_result(
            result.model_copy(
                update={
                    "events": reordered,
                    "events_digest": canonical_digest(
                        tuple(item.event.model_dump(mode="json") for item in reordered)
                    ),
                    "decisions_json": canonical_json(reordered_decisions).decode(),
                    "decisions_digest": canonical_digest(reordered_decisions),
                }
            )
        )

    forged_decision = result.events[0].decision.model_copy(update={"decision_id": OTHER_EVENT_ID})
    forged_decision_event = result.events[0].model_copy(update={"decision": forged_decision})
    forged_decision_events = (forged_decision_event, *result.events[1:])
    forged_decisions = tuple(
        item.decision.model_dump(mode="json") for item in forged_decision_events
    )
    with pytest.raises(ValueError, match="decision identity"):
        _validate_run_result(
            result.model_copy(
                update={
                    "events": forged_decision_events,
                    "decisions_json": canonical_json(forged_decisions).decode(),
                    "decisions_digest": canonical_digest(forged_decisions),
                }
            )
        )

    future_signal = Signal(
        signal_id=UUID("00000000-0000-4000-8000-00000000d093"),
        run_id=RUN_ID,
        created_at=result.events[0].event.timestamp,
        signal_type=SignalType.TOOL_ERROR,
        strength=1.0,
        evidence_event_ids=(
            result.events[0].event.event_id,
            result.events[1].event.event_id,
        ),
        detector_version="future-evidence-fixture/1",
        reason_code=ReasonCode.TOOL_ERROR,
    )
    future_signal_events = (
        result.events[0].model_copy(update={"signals": (future_signal,)}),
        *result.events[1:],
    )
    with pytest.raises(ValueError, match="signal evidence"):
        _validate_run_result(result.model_copy(update={"events": future_signal_events}))

    future_parent_event = result.events[0].event.model_copy(
        update={"parent_ids": (result.events[1].event.event_id,)}
    )
    future_parent_events = (
        result.events[0].model_copy(update={"event": future_parent_event}),
        *result.events[1:],
    )
    with pytest.raises(ValueError, match="parent graph"):
        _validate_run_result(
            result.model_copy(
                update={
                    "events": future_parent_events,
                    "events_digest": canonical_digest(
                        tuple(item.event.model_dump(mode="json") for item in future_parent_events)
                    ),
                    "persisted_event_draft_digests": tuple(
                        _normalized_event_draft_digest(item.event) for item in future_parent_events
                    ),
                }
            )
        )
    with pytest.raises(ValueError, match="exactly attest"):
        _validate_run_result(result.model_copy(update={"outcomes": ()}))

    reminder = result.events[-1]
    assert reminder.cycle is not None
    assert reminder.cycle.intervention is not None
    silenced = reminder.cycle.intervention.model_copy(update={"action": InterventionAction.SILENCE})
    forged_cycle = reminder.cycle.model_copy(update={"intervention": silenced})
    with pytest.raises(ValueError, match="committed reminder"):
        _validate_event_result(reminder.model_copy(update={"cycle": forged_cycle}))

    with pytest.raises(ValueError, match="committed reminder"):
        _validate_event_result(reminder.model_copy(update={"delivery": None}))
    running_cycle = reminder.cycle.model_copy(update={"state": CycleState.RUNNING})
    with pytest.raises(ValueError, match="cycle belongs"):
        _validate_event_result(reminder.model_copy(update={"cycle": running_cycle}))
    impossible_revision = reminder.cycle.model_copy(update={"revision": 5})
    with pytest.raises(ValueError, match="impossible terminal revision"):
        _validate_event_result(reminder.model_copy(update={"cycle": impossible_revision}))
    with pytest.raises(ValueError, match="model request digest"):
        _validate_event_result(reminder.model_copy(update={"model_request_digest": None}))

    unavailable_binding = result.routing_bindings[-1].model_copy(
        update={
            "target": None,
            "target_request_id_digest": None,
            "adapter_id": None,
            "adapter_capabilities_digest": None,
        }
    )
    forged_routing = (*result.routing_bindings[:-1], unavailable_binding)
    with pytest.raises(ValueError, match="routing binding"):
        _validate_run_result(
            result.model_copy(
                update={
                    "routing_bindings": forged_routing,
                    "routing_digest": canonical_digest(
                        tuple(item.model_dump(mode="json") for item in forged_routing)
                    ),
                }
            )
        )

    prefix = result.events[:3]
    prefix_decisions = tuple(item.decision.model_dump(mode="json") for item in prefix)
    prefix_interventions = {
        item.cycle.intervention.intervention_id
        for item in prefix
        if item.cycle is not None and item.cycle.intervention is not None
    }
    prefix_draft_digests = result.normalized_draft_digests[:3]
    prefix_persisted_digests = result.persisted_event_draft_digests[:3]
    prefix_routing = result.routing_bindings[:3]
    with pytest.raises(ValueError, match="adapter trace manifest"):
        _validate_run_result(
            result.model_copy(
                update={
                    "trace_event_count": len(prefix),
                    "normalized_draft_digests": prefix_draft_digests,
                    "persisted_event_draft_digests": prefix_persisted_digests,
                    "normalized_trace_digest": canonical_digest(
                        {
                            "schema_version": "engine-normalized-trace/v1",
                            "draft_digests": prefix_draft_digests,
                        }
                    ),
                    "routing_bindings": prefix_routing,
                    "routing_digest": canonical_digest(
                        tuple(item.model_dump(mode="json") for item in prefix_routing)
                    ),
                    "events": prefix,
                    "events_digest": canonical_digest(
                        tuple(item.event.model_dump(mode="json") for item in prefix)
                    ),
                    "decisions_json": canonical_json(prefix_decisions).decode(),
                    "decisions_digest": canonical_digest(prefix_decisions),
                    "outcomes": tuple(
                        outcome
                        for outcome in result.outcomes
                        if outcome.intervention_id in prefix_interventions
                    ),
                }
            )
        )


async def test_internal_fixture_signal_and_private_status_boundaries_fail_closed() -> None:
    with pytest.raises(ValueError):
        _ReplayFixtureState(
            replay_id="fixture/1",
            fixture_digest="f" * 64,
            response_count=0,
            remaining_count=1,
        )

    class PartialFixture:
        replay_id = "fixture/partial-v1"

    with pytest.raises(ReplayEngineModelError):
        _model_fixture_state(PartialFixture())

    trace = one_event_trace()
    replay = await _engine(
        MemoryRunRepository(
            synthetic_benchmark=True,
            id_factory=trace.event_id_factory(),
        ),
        trace,
        policy=_never_policy(1),
    ).run(trace.events, trace_digest=trace.manifest.trace_digest)
    event = replay.events[0].event
    context = DetectionContext(run_id=RUN_ID, events=(event,))
    valid = OneSignalExtractor().extract(context)[0]
    wrong_evidence = valid.model_copy(update={"evidence_event_ids": (OTHER_EVENT_ID,)})

    for invalid in (
        [valid],
        (object(),),
        (wrong_evidence,),
        (valid, valid),
    ):
        with pytest.raises(ReplayEngineInputError):
            _validated_signals(invalid, context)

    replacement = PrivateStatusReplacement(
        replacement=MemoryCreate(
            handle="private-status",
            kind=MemoryKind.PRIVATE_STATUS,
            content="Untrusted private model state.",
            provenance=(
                EvidenceReference(
                    source=EvidenceSource.EVENT,
                    source_id=EVENT_1_ID,
                    field_path="/payload/message",
                ),
            ),
            confidence=1.0,
            trust_label=TrustLabel.TRUSTED_CONTROLLER,
        )
    )
    output = MemoryCycleOutput(
        delta=MemoryDelta(
            delta_id=DELTA_1_ID,
            run_id=RUN_ID,
            private_status_replacement=replacement,
            created_at=NOW + timedelta(seconds=1),
        ),
        observation=GroundingObservation(
            parse_status=ProposalParseStatus.VALID,
            proposal_action=InterventionAction.SILENCE,
            claims=(),
            confidence=1.0,
        ),
    )
    model_result = ModelResult(
        status=ModelCallStatus.COMPLETED,
        request_digest="f" * 64,
        output=output,
        usage=ModelUsage(),
    )
    assignments = _memory_assignments("e" * 64, "d" * 64, model_result)
    assert tuple(item.handle for item in assignments) == ("private-status",)
