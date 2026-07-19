from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

import pytest

from saliencegate.domain import (
    BudgetAmounts,
    CycleRecord,
    CycleState,
    DeliveryRecord,
    DeliveryState,
    InterventionOutcome,
    InvocationDecision,
    NormalizedTraceEventDraft,
    ReasonCode,
    TraceEvent,
    canonical_digest,
)
from saliencegate.experiments import (
    Stage2ConditionId,
    Stage2ExperimentError,
    Stage2ExperimentRunner,
    Stage2Trajectory,
    load_stage2_trajectory,
)
from saliencegate.experiments import runner as runner_module
from saliencegate.models.replay_two_phase import TwoPhaseReplayClient
from saliencegate.ports.adapters import (
    AdapterCapabilities,
    DeliveryEnvelope,
    DeliveryReceipt,
)
from saliencegate.ports.model_calls import StructuredCallRequest, StructuredCallResult
from saliencegate.ports.repository import (
    AppendDisposition,
    AppendReceipt,
    BeginCycle,
    CommitCycle,
    CycleReceipt,
    FailCycle,
    LedgerEntry,
    LedgerReceipt,
    ReserveCycle,
    RunNotFoundError,
    StartCycle,
)
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.runtime.algorithm_result import (
    algorithm_runtime_uuid,
    algorithm_trace_digest,
)
from saliencegate.runtime.budget import BudgetGovernor

ROOT = Path(__file__).resolve().parents[2]
TRAJECTORY_FIXTURE = ROOT / "tests/fixtures/runs/paper_two_phase_basic.jsonl"
FIXED_RESPONSES = ROOT / "tests/fixtures/models/paper_two_phase_fixed_step_responses.jsonl"
RUN_ID = UUID("00000000-0000-4000-8000-000000009000")


def _id_factory(trace_digest: str) -> Callable[[], UUID]:
    ordinal = 0

    def next_identifier() -> UUID:
        nonlocal ordinal
        ordinal += 1
        return algorithm_runtime_uuid(trace_digest, "stage2-repository", ordinal)

    return next_identifier


class _BlockingCommitRepository(MemoryRunRepository):
    def __init__(self, trace_digest: str) -> None:
        super().__init__(synthetic_benchmark=True, id_factory=_id_factory(trace_digest))
        self.commit_is_durable = asyncio.Event()
        self.release_commit = asyncio.Event()

    async def commit_cycle(self, command: CommitCycle) -> CycleReceipt:
        receipt = await super().commit_cycle(command)
        self.commit_is_durable.set()
        await self.release_commit.wait()
        return receipt


class _BlockingLifecycleRepository(MemoryRunRepository):
    def __init__(self, trace_digest: str, transition: str) -> None:
        super().__init__(synthetic_benchmark=True, id_factory=_id_factory(trace_digest))
        self.transition = transition
        self.transition_is_durable = asyncio.Event()
        self.release_transition = asyncio.Event()

    async def _block(self, transition: str, receipt: CycleReceipt) -> CycleReceipt:
        if self.transition == transition:
            self.transition_is_durable.set()
            await self.release_transition.wait()
        return receipt

    async def begin_cycle(self, command: BeginCycle) -> CycleReceipt:
        return await self._block("pending", await super().begin_cycle(command))

    async def reserve_cycle(self, command: ReserveCycle) -> CycleReceipt:
        return await self._block("reserved", await super().reserve_cycle(command))

    async def mark_cycle_running(self, command: StartCycle) -> CycleReceipt:
        return await self._block("running", await super().mark_cycle_running(command))


class _PostAppendTransitionErrorRepository(MemoryRunRepository):
    def __init__(self, trace_digest: str, transition: str) -> None:
        super().__init__(synthetic_benchmark=True, id_factory=_id_factory(trace_digest))
        self.transition = transition

    def _lose_acknowledgement(
        self,
        transition: str,
        receipt: CycleReceipt,
    ) -> CycleReceipt:
        if self.transition == transition:
            raise RuntimeError("simulated lost transition acknowledgement")
        return receipt

    async def begin_cycle(self, command: BeginCycle) -> CycleReceipt:
        return self._lose_acknowledgement(
            "pending",
            await super().begin_cycle(command),
        )

    async def reserve_cycle(self, command: ReserveCycle) -> CycleReceipt:
        return self._lose_acknowledgement(
            "reserved",
            await super().reserve_cycle(command),
        )

    async def mark_cycle_running(self, command: StartCycle) -> CycleReceipt:
        return self._lose_acknowledgement(
            "running",
            await super().mark_cycle_running(command),
        )

    async def commit_cycle(self, command: CommitCycle) -> CycleReceipt:
        return self._lose_acknowledgement(
            "committed",
            await super().commit_cycle(command),
        )


class _TraceBoundaryFaultRepository(MemoryRunRepository):
    def __init__(
        self,
        trace_digest: str,
        boundary: str,
        *,
        post_append: bool,
        cancellation: bool,
    ) -> None:
        super().__init__(synthetic_benchmark=True, id_factory=_id_factory(trace_digest))
        self.boundary = boundary
        self.post_append = post_append
        self.cancellation = cancellation
        self.triggered = False

    def _fail_once(self, boundary: str) -> None:
        if self.boundary != boundary or self.triggered:
            return
        self.triggered = True
        if self.cancellation:
            raise asyncio.CancelledError()
        raise RuntimeError("simulated trace-boundary acknowledgement failure")

    async def append(
        self,
        event: NormalizedTraceEventDraft,
        *,
        event_id: UUID | None = None,
    ) -> AppendReceipt:
        if not self.post_append:
            self._fail_once("event")
        receipt = await super().append(event, event_id=event_id)
        if self.post_append:
            self._fail_once("event")
        return receipt

    async def record_invocation_decision(
        self,
        decision: InvocationDecision,
    ) -> LedgerReceipt:
        if not self.post_append:
            self._fail_once("decision")
        receipt = await super().record_invocation_decision(decision)
        if self.post_append:
            self._fail_once("decision")
        return receipt


class _BlockingTraceBoundaryRepository(MemoryRunRepository):
    def __init__(self, trace_digest: str, boundary: str) -> None:
        super().__init__(synthetic_benchmark=True, id_factory=_id_factory(trace_digest))
        self.boundary = boundary
        self.boundary_is_durable = asyncio.Event()
        self.release_boundary = asyncio.Event()
        self.triggered = False

    async def _block_once(self, boundary: str) -> None:
        if self.boundary != boundary or self.triggered:
            return
        self.triggered = True
        self.boundary_is_durable.set()
        await self.release_boundary.wait()

    async def append(
        self,
        event: NormalizedTraceEventDraft,
        *,
        event_id: UUID | None = None,
    ) -> AppendReceipt:
        receipt = await super().append(event, event_id=event_id)
        await self._block_once("event")
        return receipt

    async def record_invocation_decision(
        self,
        decision: InvocationDecision,
    ) -> LedgerReceipt:
        receipt = await super().record_invocation_decision(decision)
        await self._block_once("decision")
        return receipt


class _BlockingLostAckTraceBoundaryRepository(_BlockingTraceBoundaryRepository):
    async def append(
        self,
        event: NormalizedTraceEventDraft,
        *,
        event_id: UUID | None = None,
    ) -> AppendReceipt:
        receipt = await super().append(event, event_id=event_id)
        if self.boundary == "event":
            raise RuntimeError("simulated blocked event acknowledgement loss")
        return receipt

    async def record_invocation_decision(
        self,
        decision: InvocationDecision,
    ) -> LedgerReceipt:
        receipt = await super().record_invocation_decision(decision)
        if self.boundary == "decision":
            raise RuntimeError("simulated blocked decision acknowledgement loss")
        return receipt


class _BlockingLostAckLifecycleRepository(_BlockingLifecycleRepository):
    def _lose_acknowledgement(
        self,
        transition: str,
        receipt: CycleReceipt,
    ) -> CycleReceipt:
        if self.transition == transition:
            raise RuntimeError("simulated blocked lifecycle acknowledgement loss")
        return receipt

    async def begin_cycle(self, command: BeginCycle) -> CycleReceipt:
        return self._lose_acknowledgement(
            "pending",
            await super().begin_cycle(command),
        )

    async def reserve_cycle(self, command: ReserveCycle) -> CycleReceipt:
        return self._lose_acknowledgement(
            "reserved",
            await super().reserve_cycle(command),
        )

    async def mark_cycle_running(self, command: StartCycle) -> CycleReceipt:
        return self._lose_acknowledgement(
            "running",
            await super().mark_cycle_running(command),
        )


class _CorruptTraceBoundaryRepository(MemoryRunRepository):
    def __init__(self, trace_digest: str, corruption: str) -> None:
        super().__init__(synthetic_benchmark=True, id_factory=_id_factory(trace_digest))
        self.corruption = corruption
        self.extra_event_written = False

    async def append(
        self,
        event: NormalizedTraceEventDraft,
        *,
        event_id: UUID | None = None,
    ) -> AppendReceipt:
        receipt = await super().append(event, event_id=event_id)
        if self.corruption == "extra_event" and not self.extra_event_written:
            self.extra_event_written = True
            await super().append(
                event.model_copy(update={"source_event_id": "concurrent-extra-event"}),
                event_id=UUID("00000000-0000-4000-8000-000000009999"),
            )
        if self.corruption == "append_disposition":
            return receipt.model_copy(update={"disposition": AppendDisposition.DUPLICATE})
        if self.corruption == "append_position":
            return receipt.model_copy(update={"ledger_position": receipt.ledger_position + 1})
        return receipt

    async def record_invocation_decision(
        self,
        decision: InvocationDecision,
    ) -> LedgerReceipt:
        receipt = await super().record_invocation_decision(decision)
        if self.corruption == "decision_receipt":
            return receipt.model_copy(update={"appended": False})
        return receipt

    async def ledger(self, run_id: UUID) -> tuple[LedgerEntry, ...]:
        ledger = await super().ledger(run_id)
        record_type = (
            TraceEvent
            if self.corruption == "event_envelope"
            else InvocationDecision
            if self.corruption == "decision_envelope"
            else None
        )
        if record_type is None:
            return ledger
        return tuple(
            entry.model_copy(update={"record_key": "corrupt-envelope"})
            if type(entry.record) is record_type
            else entry
            for entry in ledger
        )


class _PostAppendCommitCancellationRepository(MemoryRunRepository):
    def __init__(self, trace_digest: str) -> None:
        super().__init__(synthetic_benchmark=True, id_factory=_id_factory(trace_digest))
        self.commit_ordinal = 0

    async def commit_cycle(self, command: CommitCycle) -> CycleReceipt:
        receipt = await super().commit_cycle(command)
        self.commit_ordinal += 1
        if self.commit_ordinal == 2:
            raise asyncio.CancelledError()
        return receipt


class _PreAppendCommitErrorRepository(MemoryRunRepository):
    async def commit_cycle(self, command: CommitCycle) -> CycleReceipt:
        del command
        raise RuntimeError("simulated pre-append commit failure")


class _PostAppendFailureLifecycleRepository(_BlockingLifecycleRepository):
    async def fail_cycle(self, command: FailCycle) -> CycleReceipt:
        await super().fail_cycle(command)
        raise RuntimeError("simulated lost failure acknowledgement")


class _BlockingFailureRepository(MemoryRunRepository):
    def __init__(self, trace_digest: str) -> None:
        super().__init__(synthetic_benchmark=True, id_factory=_id_factory(trace_digest))
        self.failure_started = asyncio.Event()
        self.release_failure = asyncio.Event()

    async def fail_cycle(self, command: FailCycle) -> CycleReceipt:
        self.failure_started.set()
        await self.release_failure.wait()
        return await super().fail_cycle(command)


class _PostAppendOutcomeErrorRepository(MemoryRunRepository):
    async def record_outcome(self, outcome: InterventionOutcome) -> LedgerReceipt:
        await super().record_outcome(outcome)
        raise RuntimeError("simulated lost outcome acknowledgement")


class _BlockingClient:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def generate(self, request: StructuredCallRequest) -> StructuredCallResult:
        del request
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("the blocked call unexpectedly resumed")


class _FailingDeliveryAdapter:
    def __init__(self, capabilities: AdapterCapabilities) -> None:
        self._capabilities = capabilities

    def capabilities(self) -> AdapterCapabilities:
        return self._capabilities

    async def deliver(self, delivery: DeliveryEnvelope) -> DeliveryReceipt:
        del delivery
        raise RuntimeError("simulated offline delivery failure")


def _trajectory_and_digest() -> tuple[Stage2Trajectory, str]:
    trajectory = load_stage2_trajectory(TRAJECTORY_FIXTURE)
    trace_digest = algorithm_trace_digest(
        tuple(canonical_digest(item.draft) for item in trajectory.inputs)
    )
    return trajectory, trace_digest


def _latest_cycles(ledger: tuple[LedgerEntry, ...]) -> tuple[CycleRecord, ...]:
    cycles: dict[str, CycleRecord] = {}
    for entry in ledger:
        if type(entry.record) is CycleRecord:
            cycles[entry.record.cycle_id] = entry.record
    return tuple(cycles.values())


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_a_boundary_operation() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def operation() -> str:
        started.set()
        await release.wait()
        return "done"

    task = asyncio.create_task(runner_module._complete_boundary(operation()))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release.set()

    assert await task == ("done", True)


@pytest.mark.asyncio
async def test_cancelled_unwritten_transition_is_never_reconciled() -> None:
    _trajectory, trace_digest = _trajectory_and_digest()
    runner = Stage2ExperimentRunner(
        repository=MemoryRunRepository(
            synthetic_benchmark=True,
            id_factory=_id_factory(trace_digest),
        ),
        condition=Stage2ConditionId.NO_MEMORY,
        client=None,
    )

    async def operation() -> CycleReceipt:
        raise asyncio.CancelledError()

    async def reconcile() -> CycleReceipt | None:
        return None

    with pytest.raises(asyncio.CancelledError):
        await runner._complete_reconciled_transition(operation(), reconcile)


@pytest.mark.asyncio
async def test_cancellation_during_absent_reconciliation_wins_over_operation_error() -> None:
    _trajectory, trace_digest = _trajectory_and_digest()
    runner = Stage2ExperimentRunner(
        repository=MemoryRunRepository(
            synthetic_benchmark=True,
            id_factory=_id_factory(trace_digest),
        ),
        condition=Stage2ConditionId.NO_MEMORY,
        client=None,
    )
    reconciliation_started = asyncio.Event()
    release_reconciliation = asyncio.Event()

    async def operation() -> CycleReceipt:
        raise RuntimeError("simulated operation failure")

    async def reconcile() -> CycleReceipt | None:
        reconciliation_started.set()
        await release_reconciliation.wait()
        return None

    task = asyncio.create_task(runner._complete_reconciled_transition(operation(), reconcile))
    await reconciliation_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release_reconciliation.set()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_reconciliation_rejects_wrong_ledger_and_budget_evidence() -> None:
    trajectory, trace_digest = _trajectory_and_digest()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=_id_factory(trace_digest),
    )
    runner = Stage2ExperimentRunner(
        repository=repository,
        condition=Stage2ConditionId.FIXED_STEP,
        client=TwoPhaseReplayClient.from_path(FIXED_RESPONSES),
    )
    result = await runner.run(trajectory)
    first = result.boundaries[0]
    second = result.boundaries[1]
    third = result.boundaries[2]
    assert first.cycle is not None and first.request is not None
    assert second.cycle is not None and second.request is not None
    assert third.cycle is not None and third.request is not None
    assert third.cycle.intervention is not None

    with pytest.raises(Stage2ExperimentError):
        await runner._reconcile_cycle_transition(
            expected=first.cycle,
            previous=first.request.cycle_receipt,
            expected_budget=result.final_budget_snapshot,
            decision=result.decisions[5],
        )
    with pytest.raises(Stage2ExperimentError):
        await runner._reconcile_cycle_transition(
            expected=first.cycle,
            previous=first.request.cycle_receipt.model_copy(
                update={"ledger_position": len(result.ledger) + 1}
            ),
            expected_budget=result.final_budget_snapshot,
        )
    with pytest.raises(Stage2ExperimentError):
        await runner._reconcile_cycle_transition(
            expected=first.cycle,
            previous=first.request.cycle_receipt.model_copy(
                update={
                    "record_tag": first.request.cycle_receipt.record_tag.model_copy(
                        update={"value": "f" * 64}
                    )
                }
            ),
            expected_budget=result.final_budget_snapshot,
        )
    with pytest.raises(Stage2ExperimentError):
        await runner._reconcile_cycle_transition(
            expected=first.cycle.model_copy(update={"revision": first.cycle.revision + 1}),
            previous=first.request.cycle_receipt,
            expected_budget=result.final_budget_snapshot,
        )

    absent = first.cycle.model_copy(update={"cycle_id": "f" * 64})
    assert (
        await runner._reconcile_cycle_transition(
            expected=absent,
            previous=None,
            expected_budget=result.final_budget_snapshot,
        )
        is None
    )
    with pytest.raises(Stage2ExperimentError):
        await runner._reconcile_cycle_transition(
            expected=absent,
            previous=first.request.cycle_receipt,
            expected_budget=result.final_budget_snapshot,
        )

    reservation = second.request.cycle_receipt.cycle.budget_reservation
    settlement = second.cycle.budget_settlement
    assert reservation is not None and settlement is not None
    second_budget = BudgetGovernor().settle(
        second.request.cycle_receipt.budget_snapshot,
        reservation,
        settlement,
        model_call_latencies_us=second.cycle.model_call_latencies_us,
    )
    with pytest.raises(Stage2ExperimentError):
        await runner._reconcile_cycle_transition(
            expected=second.cycle,
            previous=second.request.cycle_receipt,
            expected_budget=second_budget,
            expected_delivery=None,
        )

    reconciled = await runner._reconcile_cycle_transition(
        expected=third.cycle,
        previous=third.request.cycle_receipt,
        expected_budget=result.final_budget_snapshot,
    )
    assert reconciled is not None
    assert reconciled.cycle == third.cycle
    with pytest.raises(Stage2ExperimentError):
        await runner._reconcile_cycle_transition(
            expected=third.cycle,
            previous=third.request.cycle_receipt,
            expected_budget=result.final_budget_snapshot.model_copy(
                update={"consumed": BudgetAmounts()}
            ),
        )
    durable_outcome = next(
        entry.record
        for entry in result.ledger
        if type(entry.record) is InterventionOutcome
        and entry.record.intervention_id == third.cycle.intervention.intervention_id
    )
    await runner._record_outcome_idempotently(durable_outcome)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transition", "reason"),
    (
        ("pending", ReasonCode.MODEL_ERROR),
        ("reserved", ReasonCode.MODEL_ERROR),
        ("running", ReasonCode.FAILED_UNKNOWN_COST),
    ),
)
async def test_cancellation_terminalizes_every_active_lifecycle_state(
    transition: str,
    reason: ReasonCode,
) -> None:
    trajectory, trace_digest = _trajectory_and_digest()
    repository = _BlockingLifecycleRepository(trace_digest, transition)
    runner = Stage2ExperimentRunner(
        repository=repository,
        condition=Stage2ConditionId.FIXED_STEP,
        client=TwoPhaseReplayClient.from_path(FIXED_RESPONSES),
    )
    task = asyncio.create_task(runner.run(trajectory))
    await repository.transition_is_durable.wait()

    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    repository.release_transition.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    ledger = await repository.ledger(RUN_ID)
    cycles = _latest_cycles(ledger)
    assert len(cycles) == 1
    assert cycles[0].state is CycleState.FAILED
    assert cycles[0].failure_reason is reason
    if transition == "pending":
        assert cycles[0].budget_settlement is None
    else:
        assert cycles[0].budget_settlement is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transition", "reason"),
    (
        ("pending", ReasonCode.MODEL_ERROR),
        ("reserved", ReasonCode.MODEL_ERROR),
        ("running", ReasonCode.FAILED_UNKNOWN_COST),
    ),
)
async def test_caller_cancellation_survives_overlapping_lost_lifecycle_ack(
    transition: str,
    reason: ReasonCode,
) -> None:
    trajectory, trace_digest = _trajectory_and_digest()
    repository = _BlockingLostAckLifecycleRepository(trace_digest, transition)
    task = asyncio.create_task(
        Stage2ExperimentRunner(
            repository=repository,
            condition=Stage2ConditionId.FIXED_STEP,
            client=TwoPhaseReplayClient.from_path(FIXED_RESPONSES),
        ).run(trajectory)
    )
    await repository.transition_is_durable.wait()

    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    repository.release_transition.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    cycles = _latest_cycles(await repository.ledger(RUN_ID))
    assert len(cycles) == 1
    assert cycles[0].state is CycleState.FAILED
    assert cycles[0].failure_reason is reason


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ("pending", "reserved", "running"))
async def test_lost_active_transition_acknowledgement_reconciles_exact_state(
    transition: str,
) -> None:
    trajectory, trace_digest = _trajectory_and_digest()
    repository = _PostAppendTransitionErrorRepository(trace_digest, transition)

    result = await Stage2ExperimentRunner(
        repository=repository,
        condition=Stage2ConditionId.FIXED_STEP,
        client=TwoPhaseReplayClient.from_path(FIXED_RESPONSES),
    ).run(trajectory)

    assert len(result.boundaries) == 3
    assert all(cycle.state is CycleState.COMMITTED for cycle in _latest_cycles(result.ledger))
    assert sum(type(entry.record) is InterventionOutcome for entry in result.ledger) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("event", "decision"))
async def test_lost_trace_boundary_acknowledgement_replays_exact_result(
    boundary: str,
) -> None:
    trajectory, trace_digest = _trajectory_and_digest()
    baseline = await Stage2ExperimentRunner(
        repository=MemoryRunRepository(
            synthetic_benchmark=True,
            id_factory=_id_factory(trace_digest),
        ),
        condition=Stage2ConditionId.FIXED_STEP,
        client=TwoPhaseReplayClient.from_path(FIXED_RESPONSES),
    ).run(trajectory)
    repository = _TraceBoundaryFaultRepository(
        trace_digest,
        boundary,
        post_append=True,
        cancellation=False,
    )

    recovered = await Stage2ExperimentRunner(
        repository=repository,
        condition=Stage2ConditionId.FIXED_STEP,
        client=TwoPhaseReplayClient.from_path(FIXED_RESPONSES),
    ).run(trajectory)

    assert repository.triggered is True
    assert recovered == baseline


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("event", "decision"))
async def test_post_append_trace_boundary_cancellation_finishes_exact_boundary(
    boundary: str,
) -> None:
    trajectory, trace_digest = _trajectory_and_digest()
    repository = _TraceBoundaryFaultRepository(
        trace_digest,
        boundary,
        post_append=True,
        cancellation=True,
    )

    with pytest.raises(asyncio.CancelledError):
        await Stage2ExperimentRunner(
            repository=repository,
            condition=Stage2ConditionId.FIXED_STEP,
            client=TwoPhaseReplayClient.from_path(FIXED_RESPONSES),
        ).run(trajectory)

    ledger = await repository.ledger(RUN_ID)
    assert sum(type(entry.record) is TraceEvent for entry in ledger) == 1
    assert sum(type(entry.record) is InvocationDecision for entry in ledger) == 1
    cycles = _latest_cycles(ledger)
    assert len(cycles) == 1
    assert cycles[0].state is CycleState.COMMITTED
    assert sum(type(entry.record) is InterventionOutcome for entry in ledger) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("event", "decision"))
async def test_caller_cancellation_after_durable_trace_boundary_is_commit_through(
    boundary: str,
) -> None:
    trajectory, trace_digest = _trajectory_and_digest()
    repository = _BlockingTraceBoundaryRepository(trace_digest, boundary)
    task = asyncio.create_task(
        Stage2ExperimentRunner(
            repository=repository,
            condition=Stage2ConditionId.FIXED_STEP,
            client=TwoPhaseReplayClient.from_path(FIXED_RESPONSES),
        ).run(trajectory)
    )
    await repository.boundary_is_durable.wait()

    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    repository.release_boundary.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    ledger = await repository.ledger(RUN_ID)
    events = tuple(entry.record for entry in ledger if type(entry.record) is TraceEvent)
    decisions = tuple(entry.record for entry in ledger if type(entry.record) is InvocationDecision)
    assert len(events) == len(decisions) == 1
    assert decisions[0].event_sequence == events[0].sequence
    cycles = _latest_cycles(ledger)
    assert len(cycles) == 1
    assert cycles[0].state is CycleState.COMMITTED
    assert sum(type(entry.record) is InterventionOutcome for entry in ledger) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("event", "decision"))
async def test_caller_cancellation_survives_overlapping_lost_trace_ack(
    boundary: str,
) -> None:
    trajectory, trace_digest = _trajectory_and_digest()
    repository = _BlockingLostAckTraceBoundaryRepository(trace_digest, boundary)
    task = asyncio.create_task(
        Stage2ExperimentRunner(
            repository=repository,
            condition=Stage2ConditionId.FIXED_STEP,
            client=TwoPhaseReplayClient.from_path(FIXED_RESPONSES),
        ).run(trajectory)
    )
    await repository.boundary_is_durable.wait()

    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    repository.release_boundary.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    ledger = await repository.ledger(RUN_ID)
    assert sum(type(entry.record) is TraceEvent for entry in ledger) == 1
    assert sum(type(entry.record) is InvocationDecision for entry in ledger) == 1
    cycles = _latest_cycles(ledger)
    assert len(cycles) == 1
    assert cycles[0].state is CycleState.COMMITTED
    assert sum(type(entry.record) is InterventionOutcome for entry in ledger) == 1


@pytest.mark.asyncio
async def test_new_cancellation_after_decision_ack_reaches_the_model_cycle() -> None:
    trajectory, trace_digest = _trajectory_and_digest()
    repository = _BlockingTraceBoundaryRepository(trace_digest, "decision")
    client = _BlockingClient()
    task = asyncio.create_task(
        Stage2ExperimentRunner(
            repository=repository,
            condition=Stage2ConditionId.FIXED_STEP,
            client=client,
        ).run(trajectory)
    )
    await repository.boundary_is_durable.wait()

    task.cancel()
    repository.release_boundary.set()
    await client.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    ledger = await repository.ledger(RUN_ID)
    assert sum(type(entry.record) is TraceEvent for entry in ledger) == 1
    assert sum(type(entry.record) is InvocationDecision for entry in ledger) == 1
    cycles = _latest_cycles(ledger)
    assert len(cycles) == 1
    assert cycles[0].state is CycleState.FAILED
    assert cycles[0].failure_reason is ReasonCode.FAILED_UNKNOWN_COST


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("event", "decision"))
@pytest.mark.parametrize("cancellation", (False, True))
async def test_pre_append_trace_boundary_failure_never_invents_a_record(
    boundary: str,
    cancellation: bool,
) -> None:
    trajectory, trace_digest = _trajectory_and_digest()
    repository = _TraceBoundaryFaultRepository(
        trace_digest,
        boundary,
        post_append=False,
        cancellation=cancellation,
    )
    runner = Stage2ExperimentRunner(
        repository=repository,
        condition=Stage2ConditionId.FIXED_STEP,
        client=TwoPhaseReplayClient.from_path(FIXED_RESPONSES),
    )

    expected_error = asyncio.CancelledError if cancellation else Stage2ExperimentError
    with pytest.raises(expected_error):
        await runner.run(trajectory)

    if boundary == "event":
        with pytest.raises(RunNotFoundError):
            await repository.ledger(RUN_ID)
    else:
        ledger = await repository.ledger(RUN_ID)
        assert sum(type(entry.record) is TraceEvent for entry in ledger) == 1
        assert not any(type(entry.record) is InvocationDecision for entry in ledger)
        assert not any(type(entry.record) is CycleRecord for entry in ledger)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption",
    (
        "append_disposition",
        "append_position",
        "event_envelope",
        "decision_envelope",
        "decision_receipt",
        "extra_event",
    ),
)
async def test_trace_boundary_reconciliation_rejects_corrupt_authoritative_state(
    corruption: str,
) -> None:
    trajectory, trace_digest = _trajectory_and_digest()
    repository = _CorruptTraceBoundaryRepository(trace_digest, corruption)

    with pytest.raises(Stage2ExperimentError):
        await Stage2ExperimentRunner(
            repository=repository,
            condition=Stage2ConditionId.FIXED_STEP,
            client=TwoPhaseReplayClient.from_path(FIXED_RESPONSES),
        ).run(trajectory)


@pytest.mark.asyncio
async def test_lost_commit_acknowledgement_reconciles_before_finalization() -> None:
    trajectory, trace_digest = _trajectory_and_digest()
    repository = _PostAppendTransitionErrorRepository(trace_digest, "committed")

    result = await Stage2ExperimentRunner(
        repository=repository,
        condition=Stage2ConditionId.FIXED_STEP,
        client=TwoPhaseReplayClient.from_path(FIXED_RESPONSES),
    ).run(trajectory)

    assert len(result.boundaries) == 3
    assert all(cycle.state is CycleState.COMMITTED for cycle in _latest_cycles(result.ledger))
    deliveries = tuple(
        entry.record for entry in result.ledger if type(entry.record) is DeliveryRecord
    )
    delivery_ids = {delivery.delivery_id for delivery in deliveries}
    assert len(delivery_ids) == 1
    assert deliveries[-1].state is DeliveryState.DELIVERED
    outcomes = tuple(
        entry.record for entry in result.ledger if type(entry.record) is InterventionOutcome
    )
    assert any(outcome.intervention_id == deliveries[-1].intervention_id for outcome in outcomes)


@pytest.mark.asyncio
async def test_post_append_commit_cancellation_finalizes_before_propagating() -> None:
    trajectory, trace_digest = _trajectory_and_digest()
    repository = _PostAppendCommitCancellationRepository(trace_digest)
    runner = Stage2ExperimentRunner(
        repository=repository,
        condition=Stage2ConditionId.FIXED_STEP,
        client=TwoPhaseReplayClient.from_path(FIXED_RESPONSES),
    )

    with pytest.raises(asyncio.CancelledError):
        await runner.run(trajectory)

    ledger = await repository.ledger(RUN_ID)
    cycles = _latest_cycles(ledger)
    assert len(cycles) == 2
    assert all(cycle.state is CycleState.COMMITTED for cycle in cycles)
    deliveries = tuple(entry.record for entry in ledger if type(entry.record) is DeliveryRecord)
    assert deliveries[-1].state is DeliveryState.DELIVERED
    outcomes = tuple(entry.record for entry in ledger if type(entry.record) is InterventionOutcome)
    assert len(outcomes) == 2
    assert any(outcome.intervention_id == deliveries[-1].intervention_id for outcome in outcomes)


@pytest.mark.asyncio
async def test_pre_append_commit_failure_never_invents_a_committed_transition() -> None:
    trajectory, trace_digest = _trajectory_and_digest()
    repository = _PreAppendCommitErrorRepository(
        synthetic_benchmark=True,
        id_factory=_id_factory(trace_digest),
    )

    with pytest.raises(Stage2ExperimentError):
        await Stage2ExperimentRunner(
            repository=repository,
            condition=Stage2ConditionId.FIXED_STEP,
            client=TwoPhaseReplayClient.from_path(FIXED_RESPONSES),
        ).run(trajectory)

    ledger = await repository.ledger(RUN_ID)
    cycles = _latest_cycles(ledger)
    assert len(cycles) == 1
    assert cycles[0].state is CycleState.FAILED
    assert cycles[0].failure_reason is ReasonCode.FAILED_UNKNOWN_COST
    assert not any(type(entry.record) is DeliveryRecord for entry in ledger)
    assert not any(type(entry.record) is InterventionOutcome for entry in ledger)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transition", "reason"),
    (
        ("pending", ReasonCode.MODEL_ERROR),
        ("reserved", ReasonCode.MODEL_ERROR),
        ("running", ReasonCode.FAILED_UNKNOWN_COST),
    ),
)
async def test_lost_failure_acknowledgement_reconciles_terminal_state(
    transition: str,
    reason: ReasonCode,
) -> None:
    trajectory, trace_digest = _trajectory_and_digest()
    repository = _PostAppendFailureLifecycleRepository(trace_digest, transition)
    runner = Stage2ExperimentRunner(
        repository=repository,
        condition=Stage2ConditionId.FIXED_STEP,
        client=TwoPhaseReplayClient.from_path(FIXED_RESPONSES),
    )
    task = asyncio.create_task(runner.run(trajectory))
    await repository.transition_is_durable.wait()

    task.cancel()
    repository.release_transition.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    ledger = await repository.ledger(RUN_ID)
    cycle_revisions = tuple(entry.record for entry in ledger if type(entry.record) is CycleRecord)
    assert tuple(cycle.revision for cycle in cycle_revisions) == tuple(
        range(1, len(cycle_revisions) + 1)
    )
    assert sum(cycle.state is CycleState.FAILED for cycle in cycle_revisions) == 1
    assert cycle_revisions[-1].state is CycleState.FAILED
    assert cycle_revisions[-1].failure_reason is reason
    assert (await repository.budget_snapshot(RUN_ID)).reserved == BudgetAmounts()


@pytest.mark.asyncio
async def test_cancellation_after_commit_finishes_outcome_before_propagating() -> None:
    trajectory, trace_digest = _trajectory_and_digest()
    repository = _BlockingCommitRepository(trace_digest)
    runner = Stage2ExperimentRunner(
        repository=repository,
        condition=Stage2ConditionId.FIXED_STEP,
        client=TwoPhaseReplayClient.from_path(FIXED_RESPONSES),
    )
    task = asyncio.create_task(runner.run(trajectory))
    await repository.commit_is_durable.wait()

    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    repository.release_commit.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    ledger = await repository.ledger(RUN_ID)
    cycles = _latest_cycles(ledger)
    assert len(cycles) == 1
    assert cycles[0].state is CycleState.COMMITTED
    assert sum(type(entry.record) is InterventionOutcome for entry in ledger) == 1


@pytest.mark.asyncio
async def test_repeated_cancellation_terminalizes_a_running_cycle() -> None:
    trajectory, trace_digest = _trajectory_and_digest()
    repository = _BlockingFailureRepository(trace_digest)
    client = _BlockingClient()
    runner = Stage2ExperimentRunner(
        repository=repository,
        condition=Stage2ConditionId.FIXED_STEP,
        client=client,
    )
    task = asyncio.create_task(runner.run(trajectory))
    await client.started.wait()

    task.cancel()
    await repository.failure_started.wait()
    task.cancel()
    repository.release_failure.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    ledger = await repository.ledger(RUN_ID)
    cycles = _latest_cycles(ledger)
    assert len(cycles) == 1
    assert cycles[0].state is CycleState.FAILED
    assert cycles[0].failure_reason is ReasonCode.FAILED_UNKNOWN_COST
    assert cycles[0].budget_settlement is not None


@pytest.mark.asyncio
async def test_lost_outcome_acknowledgement_reconciles_from_the_ledger() -> None:
    trajectory, trace_digest = _trajectory_and_digest()
    repository = _PostAppendOutcomeErrorRepository(
        synthetic_benchmark=True,
        id_factory=_id_factory(trace_digest),
    )
    result = await Stage2ExperimentRunner(
        repository=repository,
        condition=Stage2ConditionId.FIXED_STEP,
        client=TwoPhaseReplayClient.from_path(FIXED_RESPONSES),
    ).run(trajectory)

    assert len(result.boundaries) == 3
    assert sum(type(entry.record) is InterventionOutcome for entry in result.ledger) == 3


@pytest.mark.asyncio
async def test_delivery_failure_still_records_the_neutral_outcome() -> None:
    trajectory, trace_digest = _trajectory_and_digest()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=_id_factory(trace_digest),
    )
    runner = Stage2ExperimentRunner(
        repository=repository,
        condition=Stage2ConditionId.FIXED_STEP,
        client=TwoPhaseReplayClient.from_path(FIXED_RESPONSES),
    )
    runner._delivery_adapter = _FailingDeliveryAdapter(runner._delivery_adapter.capabilities())

    with pytest.raises(Stage2ExperimentError):
        await runner.run(trajectory)
    ledger = await repository.ledger(RUN_ID)
    assert sum(type(entry.record) is InterventionOutcome for entry in ledger) == 2
    deliveries = tuple(entry.record for entry in ledger if type(entry.record) is DeliveryRecord)
    assert deliveries[-1].state is DeliveryState.UNKNOWN
