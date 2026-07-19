from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import timedelta
from typing import Any, TypeVar
from uuid import UUID

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule
from tests.repository.conformance import (
    CYCLE_DECISION_1_ID,
    CYCLE_RUN_ID,
    begin_cycle_context,
    cycle_commit_command,
    cycle_records,
)

from saliencegate.domain import BudgetAmounts, CycleState, DeliveryTarget, ReasonCode
from saliencegate.ports.repository import (
    BeginCycle,
    CommitCycle,
    CycleConflictError,
    CycleReceipt,
    CycleRevisionConflictError,
    FailCycle,
    InvalidCycleStateError,
    ProjectionInvariantError,
)
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.security import InstallationKey

ResultT = TypeVar("ResultT")


def repository() -> MemoryRunRepository:
    identifiers = iter(
        UUID(f"00000000-0000-4000-8000-{value:012x}") for value in range(0x400, 0x500)
    )
    return MemoryRunRepository(
        installation_key=InstallationKey(b"c" * 32),
        id_factory=lambda: next(identifiers),
    )


@settings(
    max_examples=24,
    stateful_step_count=18,
    deadline=None,
    derandomize=True,
)
class CycleLifecycleStateMachine(RuleBasedStateMachine):
    """Exercise one in-memory cycle through legal and adversarial histories."""

    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        try:
            self.repo = repository()
            self.context = self.execute(begin_cycle_context(self.repo))
        except BaseException:
            self.loop.close()
            raise
        self.state = CycleState.PENDING
        self.reserved_receipt: CycleReceipt | None = None
        self.running_receipt: CycleReceipt | None = None
        self.commit_receipt: CycleReceipt | None = None
        self.commit_request: CommitCycle | None = None
        self.failure_receipt: CycleReceipt | None = None
        self.failure_request: FailCycle | None = None

    def execute(self, operation: Coroutine[Any, Any, ResultT]) -> ResultT:
        return self.loop.run_until_complete(operation)

    def authoritative_ledger(self) -> tuple[object, ...]:
        return self.execute(self.repo.ledger(CYCLE_RUN_ID))

    def assert_ledger_unchanged(self, before: tuple[object, ...]) -> None:
        assert self.authoritative_ledger() == before

    @precondition(lambda self: self.state is CycleState.PENDING)
    @rule()
    def reserve(self) -> None:
        self.reserved_receipt = self.execute(self.repo.reserve_cycle(self.context.reserve))
        assert self.reserved_receipt.appended
        self.state = CycleState.RESERVED

    @precondition(lambda self: self.state is CycleState.RESERVED)
    @rule()
    def start(self) -> None:
        self.running_receipt = self.execute(self.repo.mark_cycle_running(self.context.start))
        assert self.running_receipt.appended
        self.state = CycleState.RUNNING

    @precondition(lambda self: self.state is CycleState.RUNNING)
    @rule()
    def commit(self) -> None:
        request = cycle_commit_command(self.context)
        self.commit_receipt = self.execute(self.repo.commit_cycle(request))
        self.commit_request = request
        assert self.commit_receipt.appended
        self.state = CycleState.COMMITTED

    @precondition(
        lambda self: self.state in (CycleState.PENDING, CycleState.RESERVED, CycleState.RUNNING)
    )
    @rule()
    def fail(self) -> None:
        if self.state is CycleState.PENDING:
            base = self.context.pending.cycle
            settlement = None
            model_call_digests: tuple[str, ...] = ()
        elif self.state is CycleState.RESERVED:
            assert self.reserved_receipt is not None
            base = self.reserved_receipt.cycle
            settlement = BudgetAmounts()
            model_call_digests = ()
        else:
            assert self.running_receipt is not None
            base = self.running_receipt.cycle
            settlement = BudgetAmounts(model_calls=1, latency_us=1_000)
            model_call_digests = ("b" * 64,)
        request = FailCycle(
            run_id=CYCLE_RUN_ID,
            cycle_id=self.context.cycle_id,
            expected_revision=base.revision,
            reason=ReasonCode.MODEL_ERROR,
            settlement=settlement,
            model_call_digests=model_call_digests,
            model_call_latencies_us=((1_000,) if model_call_digests else ()),
            updated_at=base.updated_at + timedelta(seconds=1),
        )
        self.failure_receipt = self.execute(self.repo.fail_cycle(request))
        self.failure_request = request
        assert self.failure_receipt.appended
        self.state = CycleState.FAILED

    @rule()
    def retry_begin_from_history(self) -> None:
        before = self.authoritative_ledger()
        retry = self.execute(
            self.repo.begin_cycle(
                BeginCycle(
                    run_id=CYCLE_RUN_ID,
                    invocation_decision_id=CYCLE_DECISION_1_ID,
                    grounding_version=self.context.pending.cycle.grounding_version,
                    grounding_configuration=(self.context.pending.cycle.grounding_configuration),
                    grounding_configuration_digest=(
                        self.context.pending.cycle.grounding_configuration_digest
                    ),
                    requested_delivery_target=(
                        self.context.pending.cycle.requested_delivery_target
                    ),
                    created_at=self.context.pending.cycle.created_at,
                )
            )
        )
        assert retry == self.context.pending.model_copy(update={"appended": False})
        self.assert_ledger_unchanged(before)

    @precondition(lambda self: self.reserved_receipt is not None)
    @rule()
    def retry_reservation_from_history(self) -> None:
        assert self.reserved_receipt is not None
        before = self.authoritative_ledger()
        retry = self.execute(self.repo.reserve_cycle(self.context.reserve))
        assert retry == self.reserved_receipt.model_copy(update={"appended": False})
        self.assert_ledger_unchanged(before)

    @precondition(lambda self: self.running_receipt is not None)
    @rule()
    def retry_start_from_history(self) -> None:
        assert self.running_receipt is not None
        before = self.authoritative_ledger()
        retry = self.execute(self.repo.mark_cycle_running(self.context.start))
        assert retry == self.running_receipt.model_copy(update={"appended": False})
        self.assert_ledger_unchanged(before)

    @precondition(lambda self: self.commit_receipt is not None)
    @rule()
    def retry_commit_from_history(self) -> None:
        assert self.commit_receipt is not None
        assert self.commit_request is not None
        before = self.authoritative_ledger()
        retry = self.execute(self.repo.commit_cycle(self.commit_request))
        assert retry == self.commit_receipt.model_copy(update={"appended": False})
        self.assert_ledger_unchanged(before)

    @precondition(lambda self: self.failure_receipt is not None)
    @rule()
    def retry_failure_from_history(self) -> None:
        assert self.failure_receipt is not None
        assert self.failure_request is not None
        before = self.authoritative_ledger()
        retry = self.execute(self.repo.fail_cycle(self.failure_request))
        assert retry == self.failure_receipt.model_copy(update={"appended": False})
        self.assert_ledger_unchanged(before)

    @precondition(lambda self: self.state is CycleState.PENDING)
    @rule(excess=st.integers(min_value=1, max_value=50))
    def reject_over_budget_reservation(self, excess: int) -> None:
        before = self.authoritative_ledger()
        invalid = self.context.reserve.model_copy(
            update={
                "reservation": BudgetAmounts(
                    model_calls=1,
                    input_tokens=1_000 + excess,
                )
            }
        )
        with pytest.raises(ProjectionInvariantError, match="reservation"):
            self.execute(self.repo.reserve_cycle(invalid))
        self.assert_ledger_unchanged(before)

    @precondition(lambda self: self.state is CycleState.PENDING)
    @rule()
    def reject_start_before_reservation(self) -> None:
        before = self.authoritative_ledger()
        with pytest.raises(CycleRevisionConflictError):
            self.execute(self.repo.mark_cycle_running(self.context.start))
        self.assert_ledger_unchanged(before)

    @precondition(lambda self: self.state in (CycleState.PENDING, CycleState.RESERVED))
    @rule()
    def reject_commit_before_running(self) -> None:
        before = self.authoritative_ledger()
        with pytest.raises(CycleRevisionConflictError):
            self.execute(self.repo.commit_cycle(cycle_commit_command(self.context)))
        self.assert_ledger_unchanged(before)

    @precondition(lambda self: self.state in (CycleState.PENDING, CycleState.RESERVED))
    @rule()
    def reject_unknown_cost_before_running(self) -> None:
        if self.state is CycleState.PENDING:
            base = self.context.pending.cycle
            settlement = None
        else:
            assert self.reserved_receipt is not None
            base = self.reserved_receipt.cycle
            settlement = BudgetAmounts()
        invalid = FailCycle(
            run_id=CYCLE_RUN_ID,
            cycle_id=self.context.cycle_id,
            expected_revision=base.revision,
            reason=ReasonCode.FAILED_UNKNOWN_COST,
            settlement=settlement,
            updated_at=base.updated_at + timedelta(seconds=1),
        )
        before = self.authoritative_ledger()
        with pytest.raises(CycleConflictError):
            self.execute(self.repo.fail_cycle(invalid))
        self.assert_ledger_unchanged(before)

    @precondition(lambda self: self.reserved_receipt is not None)
    @rule(changed_tokens=st.integers(min_value=101, max_value=150))
    def reject_changed_historical_reservation(self, changed_tokens: int) -> None:
        before = self.authoritative_ledger()
        changed = self.context.reserve.model_copy(
            update={
                "reservation": self.context.reserve.reservation.model_copy(
                    update={"input_tokens": changed_tokens}
                )
            }
        )
        with pytest.raises(CycleConflictError):
            self.execute(self.repo.reserve_cycle(changed))
        self.assert_ledger_unchanged(before)

    @rule(revision=st.integers(min_value=10, max_value=100))
    def reject_unknown_revision(self, revision: int) -> None:
        before = self.authoritative_ledger()
        invalid = self.context.start.model_copy(update={"expected_revision": revision})
        with pytest.raises(CycleRevisionConflictError):
            self.execute(self.repo.mark_cycle_running(invalid))
        self.assert_ledger_unchanged(before)

    @precondition(lambda self: self.state in (CycleState.COMMITTED, CycleState.FAILED))
    @rule()
    def reject_transition_from_terminal_revision(self) -> None:
        latest = cycle_records(self.authoritative_ledger())[-1]
        invalid = FailCycle(
            run_id=CYCLE_RUN_ID,
            cycle_id=self.context.cycle_id,
            expected_revision=latest.revision,
            reason=ReasonCode.MODEL_ERROR,
            settlement=latest.budget_settlement,
            model_call_digests=latest.model_call_digests,
            model_call_latencies_us=latest.model_call_latencies_us,
            updated_at=latest.updated_at + timedelta(seconds=1),
        )
        before = self.authoritative_ledger()
        with pytest.raises(InvalidCycleStateError):
            self.execute(self.repo.fail_cycle(invalid))
        self.assert_ledger_unchanged(before)

    @invariant()
    def ledger_budget_and_cursor_follow_latest_revision(self) -> None:
        ledger = self.authoritative_ledger()
        records = cycle_records(ledger)
        assert len(ledger) == len(records) + 2
        assert tuple(record.revision for record in records) == tuple(range(1, len(records) + 1))
        states = tuple(record.state for record in records)
        allowed_histories = {
            (CycleState.PENDING,),
            (CycleState.PENDING, CycleState.FAILED),
            (CycleState.PENDING, CycleState.RESERVED),
            (CycleState.PENDING, CycleState.RESERVED, CycleState.FAILED),
            (CycleState.PENDING, CycleState.RESERVED, CycleState.RUNNING),
            (
                CycleState.PENDING,
                CycleState.RESERVED,
                CycleState.RUNNING,
                CycleState.COMMITTED,
            ),
            (
                CycleState.PENDING,
                CycleState.RESERVED,
                CycleState.RUNNING,
                CycleState.FAILED,
            ),
        }
        assert states in allowed_histories
        latest = records[-1]
        assert latest.state is self.state

        budget = self.execute(self.repo.budget_snapshot(CYCLE_RUN_ID))
        expected_reserved = BudgetAmounts()
        expected_consumed = BudgetAmounts()
        if latest.state in (CycleState.RESERVED, CycleState.RUNNING):
            assert latest.budget_reservation is not None
            expected_reserved = latest.budget_reservation
        elif latest.state in (CycleState.COMMITTED, CycleState.FAILED):
            if latest.budget_settlement is not None:
                expected_consumed = latest.budget_settlement
        assert budget.reserved == expected_reserved
        assert budget.consumed == expected_consumed

        snapshot = self.execute(self.repo.snapshot(CYCLE_RUN_ID))
        expected_memory_cursor = 1 if latest.state is CycleState.COMMITTED else 0
        assert snapshot.ingestion_cursor == 1
        assert snapshot.memory_cursor == expected_memory_cursor
        assert snapshot.records == ()

    def teardown(self) -> None:
        if self.loop.is_closed():
            return
        try:
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        finally:
            self.loop.close()


TestCycleLifecycleStateMachine = CycleLifecycleStateMachine.TestCase


@pytest.mark.asyncio
async def test_begin_retry_with_a_changed_grounding_pin_conflicts_without_append() -> None:
    repo = repository()
    context = await begin_cycle_context(repo)
    cycle = context.pending.cycle
    base = BeginCycle(
        run_id=CYCLE_RUN_ID,
        invocation_decision_id=CYCLE_DECISION_1_ID,
        grounding_version=cycle.grounding_version,
        grounding_configuration=cycle.grounding_configuration,
        grounding_configuration_digest=cycle.grounding_configuration_digest,
        requested_delivery_target=cycle.requested_delivery_target,
        created_at=cycle.created_at,
    )
    changes = (
        {"grounding_version": "different-grounding/1"},
        {"grounding_configuration": {"different": True}},
        {"grounding_configuration_digest": "8" * 64},
        {"requested_delivery_target": DeliveryTarget.PRE_ACTION_REPLAN},
    )
    before = await repo.ledger(CYCLE_RUN_ID)

    for change in changes:
        with pytest.raises(CycleConflictError):
            await repo.begin_cycle(base.model_copy(update=change))
        assert await repo.ledger(CYCLE_RUN_ID) == before
