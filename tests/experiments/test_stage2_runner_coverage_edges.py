"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.experiments.test_stage2_runner import (
    TRAJECTORY_DIGEST,
    TRAJECTORY_FIXTURE,
    _repository,
    _reviewed_run,
    _ReviewedClient,
)

import saliencegate.experiments.runner as runner_module
from saliencegate.domain import (
    BudgetAmounts,
    CycleState,
    DeliveryRecord,
    DeliveryState,
    ReasonCode,
)
from saliencegate.experiments import (
    Stage2ConditionId,
    Stage2ExperimentError,
    Stage2ExperimentRunner,
    Stage2ExperimentRunResult,
    load_stage2_trajectory,
)
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.runtime.algorithm_result import algorithm_trace_digest


@pytest.fixture(scope="module")
def fixed_result() -> Stage2ExperimentRunResult:
    result, _records = asyncio.run(_reviewed_run(Stage2ConditionId.FIXED_STEP))
    return result


@pytest.fixture(scope="module")
def no_memory_result() -> Stage2ExperimentRunResult:
    trajectory = load_stage2_trajectory(
        TRAJECTORY_FIXTURE,
        expected_fixture_digest=TRAJECTORY_DIGEST,
    )
    trace_digest = algorithm_trace_digest(
        tuple(runner_module.canonical_digest(item.draft) for item in trajectory.inputs)
    )
    return asyncio.run(
        Stage2ExperimentRunner(
            repository=_repository(trace_digest),
            condition=Stage2ConditionId.NO_MEMORY,
            client=None,
        ).run(trajectory)
    )


def _runner(repository: Any) -> Stage2ExperimentRunner:
    return Stage2ExperimentRunner(
        repository=repository,
        condition=Stage2ConditionId.NO_MEMORY,
        client=None,
    )


def _reminder_boundary(result: Stage2ExperimentRunResult):
    return next(
        boundary
        for boundary in result.boundaries
        if boundary.cycle is not None
        and boundary.cycle.intervention is not None
        and boundary.cycle.intervention.action.value == "remind"
    )


def _silence_boundary(result: Stage2ExperimentRunResult):
    return next(
        boundary
        for boundary in result.boundaries
        if boundary.cycle is not None
        and boundary.cycle.intervention is not None
        and boundary.cycle.intervention.action.value == "silence"
    )


def _identity_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    boundaries: tuple[Any, ...],
) -> None:
    values = iter(boundaries)
    monkeypatch.setattr(
        runner_module.Stage2BoundaryEvidence,
        "model_validate_json",
        classmethod(lambda _cls, *_args, **_kwargs: next(values)),
    )


class _Repository:
    def __init__(
        self,
        *,
        ledger: tuple[Any, ...] = (),
        budget: Any = None,
        snapshot: Any = None,
        ledger_error: BaseException | None = None,
        outcome_error: BaseException | None = None,
        outcome_receipt: Any = None,
    ) -> None:
        self.ledger_value = ledger
        self.budget_value = budget
        self.snapshot_value = snapshot
        self.ledger_error = ledger_error
        self.outcome_error = outcome_error
        self.outcome_receipt = outcome_receipt

    async def ledger(self, _run_id: Any) -> tuple[Any, ...]:
        if self.ledger_error is not None:
            raise self.ledger_error
        return self.ledger_value

    async def budget_snapshot(self, _run_id: Any) -> Any:
        return self.budget_value

    async def snapshot(self, _run_id: Any) -> Any:
        return self.snapshot_value

    async def record_outcome(self, _outcome: Any) -> Any:
        if self.outcome_error is not None:
            raise self.outcome_error
        return self.outcome_receipt


def test_result_source_validation_rejects_ledger_envelope_and_empty_ledger(
    fixed_result: Stage2ExperimentRunResult,
) -> None:
    first = fixed_result.ledger[0].model_copy(update={"position": 2})
    with pytest.raises(ValueError, match="sources failed"):
        fixed_result.model_copy(
            update={"ledger": (first, *fixed_result.ledger[1:])}
        ).authoritative_sources_rebuild_the_result()

    with pytest.raises(ValueError, match="sources failed"):
        fixed_result.model_copy(update={"ledger": ()}).authoritative_sources_rebuild_the_result()


def test_result_source_validation_rejects_event_count_drift(
    fixed_result: Stage2ExperimentRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_keys = {id(entry.record): entry.record_key for entry in fixed_result.ledger}
    monkeypatch.setattr(
        runner_module,
        "_ledger_record_key",
        lambda record: record_keys[id(record)],
    )
    monkeypatch.setattr(runner_module, "TraceEvent", type("NoTraceEvent", (), {}))

    with pytest.raises(ValueError, match="sources failed"):
        fixed_result.authoritative_sources_rebuild_the_result()


def test_result_reconciliation_marks_decision_drift(
    fixed_result: Stage2ExperimentRunResult,
) -> None:
    decision = fixed_result.decisions[0].model_copy(update={"cooldown_active": True})
    changed = fixed_result.model_copy(update={"decisions": (decision, *fixed_result.decisions[1:])})

    with pytest.raises(ValueError, match="do not reconcile"):
        changed.authoritative_sources_rebuild_the_result()


def test_result_reconciliation_marks_missing_boundary_source(
    fixed_result: Stage2ExperimentRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = fixed_result.boundaries[0]
    event = first.boundary_event.model_copy(update={"sequence": 2})
    forged = first.model_copy(update={"boundary_event": event})
    boundaries = (forged, *fixed_result.boundaries[1:])
    _identity_boundaries(monkeypatch, boundaries)

    with pytest.raises(ValueError, match="do not reconcile"):
        fixed_result.model_copy(
            update={"boundaries": boundaries}
        ).authoritative_sources_rebuild_the_result()


def test_result_reconciliation_marks_boundary_source_drift(
    fixed_result: Stage2ExperimentRunResult,
    no_memory_result: Stage2ExperimentRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = fixed_result.boundaries[0].model_copy(update={"condition": no_memory_result.condition})
    boundaries = (first, *fixed_result.boundaries[1:])
    _identity_boundaries(monkeypatch, boundaries)

    with pytest.raises(ValueError, match="do not reconcile"):
        fixed_result.model_copy(
            update={"boundaries": boundaries}
        ).authoritative_sources_rebuild_the_result()


def test_result_reconciliation_rejects_execution_without_cycle(
    fixed_result: Stage2ExperimentRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = fixed_result.boundaries[0].model_copy(update={"cycle": None})
    boundaries = (first, *fixed_result.boundaries[1:])
    _identity_boundaries(monkeypatch, boundaries)

    with pytest.raises(ValueError, match="do not reconcile"):
        fixed_result.model_copy(
            update={"boundaries": boundaries}
        ).authoritative_sources_rebuild_the_result()


def test_result_reconciliation_rejects_incomplete_cycle_evidence(
    fixed_result: Stage2ExperimentRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = fixed_result.boundaries[0].model_copy(update={"request": None})
    boundaries = (first, *fixed_result.boundaries[1:])
    _identity_boundaries(monkeypatch, boundaries)

    with pytest.raises(ValueError, match="do not reconcile"):
        fixed_result.model_copy(
            update={"boundaries": boundaries}
        ).authoritative_sources_rebuild_the_result()


def test_result_reconciliation_rejects_delivery_presence_drift(
    fixed_result: Stage2ExperimentRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reminder = _reminder_boundary(fixed_result)
    silence = _silence_boundary(fixed_result)
    assert reminder.delivery_record is not None

    forged_silence = silence.model_copy(update={"delivery_record": reminder.delivery_record})
    boundaries = tuple(
        forged_silence if item is silence else item for item in fixed_result.boundaries
    )
    _identity_boundaries(monkeypatch, boundaries)
    with pytest.raises(ValueError, match="do not reconcile"):
        fixed_result.model_copy(
            update={"boundaries": boundaries}
        ).authoritative_sources_rebuild_the_result()


def test_result_reconciliation_rejects_missing_reminder_delivery(
    fixed_result: Stage2ExperimentRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reminder = _reminder_boundary(fixed_result)
    forged = reminder.model_copy(update={"delivery_record": None})
    boundaries = tuple(forged if item is reminder else item for item in fixed_result.boundaries)
    _identity_boundaries(monkeypatch, boundaries)

    with pytest.raises(ValueError, match="do not reconcile"):
        fixed_result.model_copy(
            update={"boundaries": boundaries}
        ).authoritative_sources_rebuild_the_result()


def test_result_reconciliation_rejects_reminder_delivery_drift(
    fixed_result: Stage2ExperimentRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reminder = _reminder_boundary(fixed_result)
    assert reminder.delivery_record is not None
    delivery = reminder.delivery_record.model_copy(update={"state": DeliveryState.FAILED})
    forged = reminder.model_copy(update={"delivery_record": delivery})
    boundaries = tuple(forged if item is reminder else item for item in fixed_result.boundaries)
    _identity_boundaries(monkeypatch, boundaries)

    with pytest.raises(ValueError, match="do not reconcile"):
        fixed_result.model_copy(
            update={"boundaries": boundaries}
        ).authoritative_sources_rebuild_the_result()


def test_runner_constructor_rejects_resolved_grounding_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Grounding:
        def __init__(self, _configuration: Any) -> None:
            self.resolved_configuration = object()

    monkeypatch.setattr(runner_module, "GroundingPipeline", _Grounding)

    with pytest.raises(Stage2ExperimentError):
        Stage2ExperimentRunner(
            repository=MemoryRunRepository(synthetic_benchmark=True),
            condition=Stage2ConditionId.NO_MEMORY,
            client=None,
        )


@pytest.mark.asyncio
async def test_budget_rejects_repository_limit_drift(
    fixed_result: Stage2ExperimentRunResult,
) -> None:
    repository = _Repository(budget=fixed_result.final_budget_snapshot)
    runner = _runner(repository)
    limits = fixed_result.budget_limits.model_copy(
        update={"model_calls": fixed_result.budget_limits.model_calls + 1}
    )

    with pytest.raises(Stage2ExperimentError):
        await runner._budget(fixed_result.run_id, limits=limits, first_decision=False)


@pytest.mark.asyncio
async def test_grounding_state_includes_recent_committed_reminder(
    fixed_result: Stage2ExperimentRunResult,
) -> None:
    runner = Stage2ExperimentRunner(
        repository=_Repository(ledger=fixed_result.ledger),
        condition=Stage2ConditionId.FIXED_STEP,
        client=_ReviewedClient(Stage2ConditionId.FIXED_STEP),
    )
    runner._grounding = SimpleNamespace(
        configuration=runner._grounding.configuration.model_copy(
            update={"duplicate_window_events": 2}
        )
    )
    state = await runner._grounding_state(
        fixed_result.run_id,
        current_sequence=7,
        memories=(),
    )

    assert state.reminder_history


@pytest.mark.asyncio
async def test_cycle_reconciliation_rejects_delivery_drift(
    fixed_result: Stage2ExperimentRunResult,
) -> None:
    boundary = _reminder_boundary(fixed_result)
    assert boundary.cycle is not None
    pending_index = next(
        index
        for index, entry in enumerate(fixed_result.ledger)
        if type(entry.record) is DeliveryRecord
        and entry.record.cycle_id == boundary.cycle.cycle_id
        and entry.record.revision == 1
    )
    ledger = fixed_result.ledger[: pending_index + 1]
    repository = _Repository(ledger=ledger, budget=fixed_result.final_budget_snapshot)
    runner = _runner(repository)

    with pytest.raises(Stage2ExperimentError):
        await runner._reconcile_cycle_transition(
            expected=boundary.cycle,
            previous=None,
            expected_budget=fixed_result.final_budget_snapshot,
            expected_delivery=None,
        )


@pytest.mark.asyncio
async def test_cycle_reconciliation_preserves_stage2_errors(
    fixed_result: Stage2ExperimentRunResult,
) -> None:
    boundary = fixed_result.boundaries[0]
    assert boundary.cycle is not None
    runner = _runner(_Repository(ledger_error=Stage2ExperimentError()))

    with pytest.raises(Stage2ExperimentError):
        await runner._reconcile_cycle_transition(
            expected=boundary.cycle,
            previous=None,
            expected_budget=fixed_result.final_budget_snapshot,
        )


@pytest.mark.asyncio
async def test_begin_reconciliation_distinguishes_absence_and_errors(
    fixed_result: Stage2ExperimentRunResult,
) -> None:
    decision = fixed_result.decisions[0]
    runner = _runner(_Repository(ledger=()))
    grounding = runner._grounding.pin(runner._condition.shared_controls.requested_delivery_target)

    assert (
        await runner._reconcile_begin(
            decision=decision,
            grounding=grounding,
            created_at=decision.created_at,
        )
        is None
    )

    for error in (Stage2ExperimentError(), RuntimeError("ledger failed")):
        failing = _runner(_Repository(ledger_error=error))
        with pytest.raises(Stage2ExperimentError):
            await failing._reconcile_begin(
                decision=decision,
                grounding=grounding,
                created_at=decision.created_at,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("state", (CycleState.RESERVED, CycleState.RUNNING, CycleState.COMMITTED))
async def test_unknown_terminalization_rejects_impossible_guard_state(
    fixed_result: Stage2ExperimentRunResult,
    state: CycleState,
) -> None:
    boundary = fixed_result.boundaries[0]
    assert boundary.request is not None
    receipt = boundary.request.cycle_receipt
    cycle = receipt.cycle.model_copy(update={"state": state, "budget_reservation": BudgetAmounts()})
    guard = runner_module._RunningCycle(receipt.model_copy(update={"cycle": cycle}))
    runner = _runner(_Repository())

    with pytest.raises(Stage2ExperimentError):
        await runner._terminalize_unknown(
            SimpleNamespace(),
            guard,
            updated_at=boundary.boundary_event.timestamp,
        )


@pytest.mark.asyncio
async def test_known_terminalization_and_settlement_require_running_guard(
    fixed_result: Stage2ExperimentRunResult,
) -> None:
    runner = _runner(_Repository())
    timestamp = fixed_result.trajectory.inputs[0].draft.timestamp

    with pytest.raises(Stage2ExperimentError):
        await runner._terminalize_known(
            SimpleNamespace(),
            runner_module._RunningCycle(),
            SimpleNamespace(),
            SimpleNamespace(),
            updated_at=timestamp,
        )
    with pytest.raises(Stage2ExperimentError):
        await runner._settle_known_failure(
            SimpleNamespace(),
            runner_module._RunningCycle(),
            usage=SimpleNamespace(),
            calls=(),
            reason=ReasonCode.MODEL_ERROR,
            updated_at=timestamp,
        )


@pytest.mark.asyncio
async def test_known_settlement_rejects_budget_drift(
    fixed_result: Stage2ExperimentRunResult,
) -> None:
    boundary = fixed_result.boundaries[0]
    assert boundary.request is not None
    execution = boundary.two_phase_result
    assert execution is not None
    receipt = boundary.request.cycle_receipt
    changed_budget = receipt.budget_snapshot.model_copy(update={"consumed": BudgetAmounts()})
    if changed_budget == receipt.budget_snapshot:
        changed_budget = receipt.budget_snapshot.model_copy(update={"reserved": BudgetAmounts()})
    runner = _runner(_Repository(budget=changed_budget))

    with pytest.raises(Stage2ExperimentError):
        await runner._settle_known_failure(
            SimpleNamespace(),
            runner_module._RunningCycle(receipt),
            usage=execution.usage,
            calls=execution.call_receipts,
            reason=ReasonCode.MODEL_ERROR,
            updated_at=boundary.boundary_event.timestamp,
        )


@pytest.mark.asyncio
async def test_known_settlement_restores_cancellation_after_transition(
    fixed_result: Stage2ExperimentRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = fixed_result.boundaries[0]
    assert boundary.request is not None
    execution = boundary.two_phase_result
    assert execution is not None
    receipt = boundary.request.cycle_receipt
    runner = _runner(_Repository(budget=receipt.budget_snapshot))

    async def cancelled_transition(
        _self: Any,
        operation: Any,
        _reconcile: Any,
    ) -> tuple[Any, bool]:
        operation.close()
        return receipt, True

    class _Coordinator:
        async def fail(self, *_args: object, **_kwargs: object) -> Any:
            return receipt

    monkeypatch.setattr(
        Stage2ExperimentRunner,
        "_complete_reconciled_transition",
        cancelled_transition,
    )

    with pytest.raises(asyncio.CancelledError):
        await runner._settle_known_failure(
            _Coordinator(),
            runner_module._RunningCycle(receipt),
            usage=execution.usage,
            calls=execution.call_receipts,
            reason=ReasonCode.MODEL_ERROR,
            updated_at=boundary.boundary_event.timestamp,
        )


@pytest.mark.asyncio
async def test_outcome_durability_rejects_duplicate_records(
    fixed_result: Stage2ExperimentRunResult,
) -> None:
    boundary = fixed_result.boundaries[0]
    assert boundary.cycle is not None and boundary.cycle.intervention is not None
    outcome = Stage2ExperimentRunner._outcome(
        trace_digest=fixed_result.trace_digest,
        intervention=boundary.cycle.intervention,
    )
    entries = (
        SimpleNamespace(record=outcome),
        SimpleNamespace(record=outcome),
    )
    runner = _runner(_Repository(ledger=entries))

    with pytest.raises(Stage2ExperimentError):
        await runner._outcome_is_durable(outcome)


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", (False, True))
async def test_outcome_recording_preserves_cancellation(
    fixed_result: Stage2ExperimentRunResult,
    durable: bool,
) -> None:
    boundary = fixed_result.boundaries[0]
    assert boundary.cycle is not None and boundary.cycle.intervention is not None
    outcome = Stage2ExperimentRunner._outcome(
        trace_digest=fixed_result.trace_digest,
        intervention=boundary.cycle.intervention,
    )
    ledger = (SimpleNamespace(record=outcome),) if durable else ()
    runner = _runner(_Repository(ledger=ledger, outcome_error=asyncio.CancelledError()))

    with pytest.raises(asyncio.CancelledError):
        await runner._record_outcome_idempotently(outcome)


@pytest.mark.asyncio
async def test_outcome_recording_rejects_nondurable_error_and_duplicate_receipt(
    fixed_result: Stage2ExperimentRunResult,
) -> None:
    boundary = fixed_result.boundaries[0]
    assert boundary.cycle is not None and boundary.cycle.intervention is not None
    outcome = Stage2ExperimentRunner._outcome(
        trace_digest=fixed_result.trace_digest,
        intervention=boundary.cycle.intervention,
    )

    with pytest.raises(Stage2ExperimentError):
        await _runner(
            _Repository(ledger=(), outcome_error=RuntimeError("write failed"))
        )._record_outcome_idempotently(outcome)
    with pytest.raises(Stage2ExperimentError):
        await _runner(
            _Repository(ledger=(), outcome_receipt=SimpleNamespace(appended=False))
        )._record_outcome_idempotently(outcome)


@pytest.mark.asyncio
async def test_finalize_committed_cycle_records_outcome_before_reraising_delivery(
    fixed_result: Stage2ExperimentRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = _reminder_boundary(fixed_result)
    assert boundary.cycle is not None and boundary.cycle.intervention is not None

    class _Worker:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def deliver(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("delivery failed")

    async def recorded(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(runner_module, "DeliveryWorker", _Worker)
    monkeypatch.setattr(Stage2ExperimentRunner, "_record_outcome_idempotently", recorded)
    committed = SimpleNamespace(
        delivery=SimpleNamespace(delivery_id=boundary.delivery_record.delivery_id)
    )

    with pytest.raises(RuntimeError, match="delivery failed"):
        await _runner(_Repository())._finalize_committed_cycle(
            trace_digest=fixed_result.trace_digest,
            committed=committed,
            intervention=boundary.cycle.intervention,
            timestamp=boundary.boundary_event.timestamp,
        )


@pytest.mark.asyncio
async def test_execute_cycle_requires_executor_and_window(
    fixed_result: Stage2ExperimentRunResult,
) -> None:
    runner = _runner(_Repository())

    with pytest.raises(Stage2ExperimentError):
        await runner._execute_cycle(
            boundary=SimpleNamespace(window=object()),
            decision=fixed_result.decisions[0],
        )


@pytest.mark.asyncio
async def test_run_rejects_detached_trajectory_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = load_stage2_trajectory(TRAJECTORY_FIXTURE)
    runner = _runner(MemoryRunRepository(synthetic_benchmark=True))
    monkeypatch.setattr(
        runner_module.Stage2Trajectory,
        "model_validate_json",
        classmethod(lambda _cls, *_args, **_kwargs: object()),
    )

    with pytest.raises(Stage2ExperimentError):
        await runner.run(trajectory)


@pytest.mark.asyncio
async def test_run_wraps_budget_reservation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = load_stage2_trajectory(TRAJECTORY_FIXTURE)
    trace_digest = algorithm_trace_digest(
        tuple(runner_module.canonical_digest(item.draft) for item in trajectory.inputs)
    )
    runner = Stage2ExperimentRunner(
        repository=_repository(trace_digest),
        condition=Stage2ConditionId.FIXED_STEP,
        client=_ReviewedClient(Stage2ConditionId.FIXED_STEP),
    )

    def fail_reservation(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("reservation failed")

    monkeypatch.setattr(runner_module.BudgetGovernor, "reserve", fail_reservation)

    with pytest.raises(Stage2ExperimentError):
        await runner.run(trajectory)


@pytest.mark.asyncio
async def test_single_replay_preserves_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def cancel_run(*_args: object, **_kwargs: object) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(Stage2ExperimentRunner, "run", cancel_run)

    with pytest.raises(asyncio.CancelledError):
        await runner_module._replay_stage2_fixture_once(
            TRAJECTORY_FIXTURE,
            condition=Stage2ConditionId.NO_MEMORY,
            responses_path=None,
            expected_trajectory_fixture_digest=TRAJECTORY_DIGEST,
            expected_response_fixture_digest=None,
        )


@pytest.mark.asyncio
async def test_double_replay_rejects_nondeterminism(
    fixed_result: Stage2ExperimentRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter(
        (
            fixed_result,
            fixed_result.model_copy(update={"result_digest": "f" * 64}),
        )
    )

    async def replay_once(*_args: object, **_kwargs: object) -> Stage2ExperimentRunResult:
        return next(results)

    monkeypatch.setattr(runner_module, "_replay_stage2_fixture_once", replay_once)

    with pytest.raises(Stage2ExperimentError):
        await runner_module.replay_stage2_fixture_twice(
            Path("unused-stage2-runner.jsonl"),
            condition=Stage2ConditionId.NO_MEMORY,
        )
