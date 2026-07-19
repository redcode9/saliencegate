from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal, cast

import pytest
from tests.runtime import test_fixed_step_runtime as fixtures

import saliencegate.runtime.fixed_step as fixed_step_module
from saliencegate.domain import (
    BudgetAmounts,
    CycleRecord,
    CycleState,
    DeliveryTarget,
    PayloadDigest,
    ReasonCode,
)
from saliencegate.intervention import GroundingPipeline
from saliencegate.memory.two_phase import (
    PaperTwoPhaseCycleExecutor,
    RepositoryOperationMaterializer,
)
from saliencegate.ports.adapters import AdapterCapabilities
from saliencegate.ports.model_calls import (
    ProviderUsageProvenance,
    StructuredCallParseStatus,
    StructuredCallRequest,
    StructuredCallResult,
    StructuredCallStatus,
    StructuredCallUsage,
)
from saliencegate.ports.repository import (
    MemoryDeltaPreview,
    ReserveCycle,
    RunRepository,
    StartCycle,
)
from saliencegate.ports.two_phase import (
    TwoPhaseCycleFailure,
    TwoPhaseCycleOutcome,
    TwoPhaseCycleRequest,
    TwoPhaseFailureReason,
)
from saliencegate.repository.sqlite import SQLiteRunRepository
from saliencegate.runtime.fixed_step import FixedStepExecutionError, FixedStepRunner

Fault = Literal["error", "cancel"]
Transition = Literal["after_begin", "after_reserve"]


class _ExplodingCapabilitiesAdapter(fixtures._DeliveryAdapter):
    def capabilities(self):
        raise RuntimeError("injected capabilities failure")


def test_value_free_helpers_cover_invalid_values_and_closed_failure_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = fixtures._run_start().model_copy(
        update={
            "draft": fixtures._run_start().draft.model_copy(
                update={"payload": {"task": "Task", "items": ["first"], "step": 1}}
            )
        }
    )

    assert fixed_step_module._draft_pointer(item, "/payload/items/0") == "first"
    with pytest.raises(fixed_step_module.FixedStepInputError):
        fixed_step_module._draft_pointer(item, "/payload/items/nope")
    with pytest.raises(fixed_step_module.FixedStepInputError):
        fixed_step_module._draft_pointer(item, "/payload/task/deeper")
    with pytest.raises(fixed_step_module.FixedStepInputError):
        fixed_step_module._utc(None)
    assert str(fixed_step_module.FixedStepInvariantError()) == (
        "fixed-step authoritative state diverged"
    )
    assert (
        fixed_step_module._failure_reason(TwoPhaseFailureReason.MODEL_ERROR)
        is ReasonCode.MODEL_ERROR
    )
    assert (
        fixed_step_module._failure_reason(TwoPhaseFailureReason.SCHEMA_INVALID)
        is ReasonCode.INVALID_STRUCTURED_OUTPUT
    )
    valid = fixtures._run_start()
    monkeypatch.setattr(
        fixed_step_module,
        "normalized_trace_event_draft_is_bounded",
        lambda _value: False,
    )
    with pytest.raises(ValueError, match="normalized draft"):
        valid.draft_and_selectors_are_exact()


def test_capability_and_input_guards_fail_closed_before_execution(tmp_path: Path) -> None:
    repository = fixtures._make_repository("memory", tmp_path / "unused.sqlite3")
    pre_action_runner, _client, _configuration = fixtures._runner(
        repository,
        mode="silence",
        cycle_capacity=1,
        delivery_adapter=fixtures._DeliveryAdapter("deliver"),
        requested_delivery_target=DeliveryTarget.PRE_ACTION_REPLAN,
    )
    assert pre_action_runner._capabilities() is None

    exploding_runner, _client, _configuration = fixtures._runner(
        repository,
        mode="silence",
        cycle_capacity=1,
        delivery_adapter=_ExplodingCapabilitiesAdapter("deliver"),
        requested_delivery_target=DeliveryTarget.NEXT_MODEL_CALL,
    )
    assert exploding_runner._capabilities() is None
    assert exploding_runner._routing(
        fixtures._run_start(target_request_id="next-request"),
        cast(AdapterCapabilities, object()),
    ) == (DeliveryTarget.NEXT_MODEL_CALL, None)

    malformed = fixed_step_module.FixedStepEventInput.model_construct(
        draft=object(),
        expected_event_id=fixtures.EVENT_IDS[0],
        task_description=None,
    )
    with pytest.raises(ValueError, match="normalized draft"):
        malformed.draft_and_selectors_are_exact()


def _raise_fault(fault: Fault) -> None:
    if fault == "cancel":
        raise asyncio.CancelledError
    raise RuntimeError("injected repository boundary failure")


class _TransitionFaultRepository:
    def __init__(
        self,
        inner: RunRepository,
        *,
        transition: Transition,
        fault: Fault,
    ) -> None:
        self.inner = inner
        self.transition = transition
        self.fault = fault

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    async def reserve_cycle(self, command: ReserveCycle):
        if self.transition == "after_begin":
            _raise_fault(self.fault)
        return await self.inner.reserve_cycle(command)

    async def mark_cycle_running(self, command: StartCycle):
        if self.transition == "after_reserve":
            _raise_fault(self.fault)
        return await self.inner.mark_cycle_running(command)


class _ExecutorFault:
    def __init__(self, fault: Fault) -> None:
        self.fault = fault

    async def execute(self, request: TwoPhaseCycleRequest) -> TwoPhaseCycleOutcome:
        del request
        _raise_fault(self.fault)


class _SnapshotFaultRepository:
    def __init__(self, inner: RunRepository, fault: Fault) -> None:
        self.inner = inner
        self.fault = fault

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    async def snapshot(self, run_id):
        del run_id
        _raise_fault(self.fault)


class _RunningLedgerFaultRepository:
    def __init__(self, inner: RunRepository, fault: Fault) -> None:
        self.inner = inner
        self.fault = fault

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    async def ledger(self, run_id):
        entries = await self.inner.ledger(run_id)
        if any(
            type(entry.record) is CycleRecord and entry.record.state is CycleState.RUNNING
            for entry in entries
        ):
            _raise_fault(self.fault)
        return entries


class _BudgetFaultRepository:
    def __init__(self, inner: RunRepository, fault: Fault) -> None:
        self.inner = inner
        self.fault = fault

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    async def budget_snapshot(self, run_id):
        del run_id
        _raise_fault(self.fault)


class _UnknownUsageClient(fixtures._RequestAwareClient):
    async def generate(self, request: StructuredCallRequest) -> StructuredCallResult:
        completed = await super().generate(request)
        values = completed.model_dump(
            mode="python",
            exclude={"call_digest", "usage"},
            warnings=False,
        )
        # Preserve the exact discriminated output model under strict validation.
        values["output"] = completed.output
        values["completion_digest"] = completed.completion_digest
        return StructuredCallResult(
            **values,
            usage=StructuredCallUsage(
                schema_version="structured-call-usage/v1",
                provider_input_tokens=None,
                provider_output_tokens=None,
                provider_usage_provenance=ProviderUsageProvenance.UNAVAILABLE,
                latency_us=completed.usage.latency_us,
            ),
        )


class _TimeoutClient:
    async def generate(self, request: StructuredCallRequest) -> StructuredCallResult:
        return StructuredCallResult(
            schema_version="structured-call-result/v1",
            request_digest=request.request_digest,
            model_call_index=request.model_call_index,
            phase=request.phase,
            attempt=request.attempt,
            response_schema_version=request.response_schema_version,
            status=StructuredCallStatus.MODEL_TIMEOUT,
            parse_status=StructuredCallParseStatus.NOT_ATTEMPTED,
            output=None,
            completion_digest=None,
            completion_byte_count=None,
            usage=StructuredCallUsage(
                schema_version="structured-call-usage/v1",
                provider_input_tokens=3,
                provider_output_tokens=4,
                provider_usage_provenance=ProviderUsageProvenance.REPLAY_ATTESTED,
                latency_us=77,
            ),
        )


class _PreviewDriftRepository:
    def __init__(self, inner: RunRepository) -> None:
        self.inner = inner
        self.preview_count = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    async def preview_memory_delta(self, command: object) -> MemoryDeltaPreview:
        self.preview_count += 1
        preview = await self.inner.preview_memory_delta(command)  # type: ignore[arg-type]
        if self.preview_count != 2:
            return preview
        divergent = PayloadDigest(
            algorithm=preview.preview_projection_digest.algorithm,
            value="d" * 64,
        )
        return preview.model_copy(update={"preview_projection_digest": divergent})


class _PreviewFaultRepository:
    def __init__(self, inner: RunRepository, fault: Fault) -> None:
        self.inner = inner
        self.fault = fault
        self.preview_count = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    async def preview_memory_delta(self, command: object) -> MemoryDeltaPreview:
        self.preview_count += 1
        preview = await self.inner.preview_memory_delta(command)  # type: ignore[arg-type]
        if self.preview_count == 2:
            _raise_fault(self.fault)
        return preview


@pytest.fixture(params=("memory", "sqlite"))
def repository(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Iterator[RunRepository]:
    current = fixtures._make_repository(
        cast(str, request.param),
        tmp_path / "fixed-step-boundaries.sqlite3",
    )
    try:
        yield current
    finally:
        if isinstance(current, SQLiteRunRepository):
            current.close()


async def _latest_cycle(repository: RunRepository) -> CycleRecord:
    cycles = tuple(
        entry.record
        for entry in await repository.ledger(fixtures.RUN_ID)
        if type(entry.record) is CycleRecord
    )
    assert cycles
    return cycles[-1]


async def _assert_no_memory(repository: RunRepository) -> None:
    snapshot = await repository.snapshot(fixtures.RUN_ID)
    assert snapshot.ingestion_cursor == 1
    assert snapshot.memory_cursor == 0
    assert snapshot.records == ()


async def _assert_full_unknown_failure(
    repository: RunRepository,
    reservation: BudgetAmounts,
) -> None:
    cycle = await _latest_cycle(repository)
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is ReasonCode.FAILED_UNKNOWN_COST
    assert cycle.budget_reservation == reservation
    assert cycle.budget_settlement == reservation
    budget = await repository.budget_snapshot(fixtures.RUN_ID)
    assert budget.reserved == BudgetAmounts()
    assert budget.consumed == reservation
    await _assert_no_memory(repository)


async def _expect_fault(
    fault: Fault,
    operation: Callable[[], object],
) -> None:
    if fault == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await operation()  # type: ignore[misc]
    else:
        with pytest.raises(FixedStepExecutionError):
            await operation()  # type: ignore[misc]


@pytest.mark.asyncio
async def test_boundary_helpers_finish_under_cancellation_and_propagate_errors() -> None:
    completion_gate = asyncio.Event()

    async def complete() -> str:
        await completion_gate.wait()
        return "complete"

    boundary = asyncio.create_task(fixed_step_module._complete_boundary(complete()))
    await asyncio.sleep(0)
    boundary.cancel()
    completion_gate.set()
    assert await boundary == ("complete", True)

    error_gate = asyncio.Event()

    async def explode() -> None:
        await error_gate.wait()
        raise RuntimeError("injected boundary failure")

    error_boundary = asyncio.create_task(fixed_step_module._complete_boundary(explode()))
    await asyncio.sleep(0)
    error_gate.set()
    with pytest.raises(RuntimeError, match="injected boundary failure"):
        await error_boundary

    cleanup_gate = asyncio.Event()
    cleanup = asyncio.create_task(cleanup_gate.wait())
    drainer = asyncio.create_task(fixed_step_module._drain_cleanup(cleanup))  # type: ignore[arg-type]
    await asyncio.sleep(0)
    drainer.cancel()
    cleanup_gate.set()
    await drainer

    cleanup_error_gate = asyncio.Event()
    cleanup_error = asyncio.create_task(explode_after(cleanup_error_gate))
    error_drainer = asyncio.create_task(fixed_step_module._drain_cleanup(cleanup_error))
    await asyncio.sleep(0)
    cleanup_error_gate.set()
    with pytest.raises(RuntimeError, match="injected cleanup failure"):
        await error_drainer


async def explode_after(gate: asyncio.Event) -> None:
    await gate.wait()
    raise RuntimeError("injected cleanup failure")


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("error", "cancel"))
@pytest.mark.parametrize("transition", ("after_begin", "after_reserve"))
async def test_transition_faults_terminalize_unfinished_cycles(
    repository: RunRepository,
    transition: Transition,
    fault: Fault,
) -> None:
    wrapped = _TransitionFaultRepository(
        repository,
        transition=transition,
        fault=fault,
    )
    runner, _client, configuration = fixtures._runner(
        cast(RunRepository, wrapped),
        mode="silence",
        cycle_capacity=1,
    )

    await _expect_fault(fault, lambda: runner.run((fixtures._run_start(),)))

    cycle = await _latest_cycle(repository)
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is ReasonCode.MODEL_ERROR
    if transition == "after_begin":
        assert cycle.budget_reservation is None
        assert cycle.budget_settlement is None
    else:
        assert cycle.budget_reservation == configuration.cycle_reservation
        assert cycle.budget_settlement == BudgetAmounts()
    budget = await repository.budget_snapshot(fixtures.RUN_ID)
    assert budget.reserved == BudgetAmounts()
    assert budget.consumed == BudgetAmounts()
    await _assert_no_memory(repository)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled_transition", (1, 2, 3))
async def test_cancellation_after_confirmed_transition_uses_latest_guard(
    repository: RunRepository,
    monkeypatch: pytest.MonkeyPatch,
    cancelled_transition: int,
) -> None:
    original = fixed_step_module._complete_boundary
    completed = 0

    async def force_cancellation(operation):
        nonlocal completed
        result, _cancelled = await original(operation)
        completed += 1
        return result, completed == cancelled_transition

    monkeypatch.setattr(fixed_step_module, "_complete_boundary", force_cancellation)
    runner, _client, configuration = fixtures._runner(
        repository,
        mode="silence",
        cycle_capacity=1,
    )

    with pytest.raises(asyncio.CancelledError):
        await runner.run((fixtures._run_start(),))

    cycle = await _latest_cycle(repository)
    assert cycle.state is CycleState.FAILED
    if cancelled_transition == 1:
        assert cycle.failure_reason is ReasonCode.MODEL_ERROR
        assert cycle.budget_reservation is None
        assert cycle.budget_settlement is None
    elif cancelled_transition == 2:
        assert cycle.failure_reason is ReasonCode.MODEL_ERROR
        assert cycle.budget_reservation == configuration.cycle_reservation
        assert cycle.budget_settlement == BudgetAmounts()
    else:
        assert cycle.failure_reason is ReasonCode.FAILED_UNKNOWN_COST
        assert cycle.budget_settlement == configuration.cycle_reservation


@pytest.mark.asyncio
async def test_cancellation_after_confirmed_commit_never_rewrites_terminal_state(
    repository: RunRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = fixed_step_module._complete_boundary
    completed = 0

    async def cancel_fourth_boundary(operation):
        nonlocal completed
        result, _cancelled = await original(operation)
        completed += 1
        return result, completed == 4

    monkeypatch.setattr(fixed_step_module, "_complete_boundary", cancel_fourth_boundary)
    runner, _client, _configuration = fixtures._runner(
        repository,
        mode="silence",
        cycle_capacity=1,
    )

    with pytest.raises(asyncio.CancelledError):
        await runner.run((fixtures._run_start(),))

    cycle = await _latest_cycle(repository)
    assert cycle.state is CycleState.COMMITTED
    assert cycle.revision == 4
    snapshot = await repository.snapshot(fixtures.RUN_ID)
    assert snapshot.ingestion_cursor == snapshot.memory_cursor == 1
    budget = await repository.budget_snapshot(fixtures.RUN_ID)
    assert budget.reserved == BudgetAmounts()
    assert budget.consumed == cycle.budget_settlement


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("error", "cancel"))
async def test_executor_faults_charge_full_reservation_and_fail_unknown_cost(
    repository: RunRepository,
    fault: Fault,
) -> None:
    grounding = GroundingPipeline(fixtures._grounding_config())
    configuration = fixtures._configuration(grounding, cycle_capacity=1)
    runner = FixedStepRunner(
        repository=repository,
        executor=_ExecutorFault(fault),
        grounding_pipeline=grounding,
        configuration=configuration,
    )

    await _expect_fault(fault, lambda: runner.run((fixtures._run_start(),)))

    cycle = await _latest_cycle(repository)
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is ReasonCode.FAILED_UNKNOWN_COST
    assert cycle.budget_reservation == configuration.cycle_reservation
    assert cycle.budget_settlement == configuration.cycle_reservation
    assert cycle.model_call_digests == ()
    assert cycle.model_call_latencies_us == ()
    budget = await repository.budget_snapshot(fixtures.RUN_ID)
    assert budget.reserved == BudgetAmounts()
    assert budget.consumed == configuration.cycle_reservation
    await _assert_no_memory(repository)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("error", "cancel"))
async def test_snapshot_faults_after_running_are_terminalized(
    repository: RunRepository,
    fault: Fault,
) -> None:
    wrapped = _SnapshotFaultRepository(repository, fault)
    runner, _client, configuration = fixtures._runner(
        cast(RunRepository, wrapped),
        mode="silence",
        cycle_capacity=1,
    )

    await _expect_fault(fault, lambda: runner.run((fixtures._run_start(),)))

    await _assert_full_unknown_failure(repository, configuration.cycle_reservation)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("error", "cancel"))
async def test_grounding_ledger_faults_after_running_are_terminalized(
    repository: RunRepository,
    fault: Fault,
) -> None:
    wrapped = _RunningLedgerFaultRepository(repository, fault)
    runner, _client, configuration = fixtures._runner(
        cast(RunRepository, wrapped),
        mode="silence",
        cycle_capacity=1,
    )

    if fault == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await runner.run((fixtures._run_start(),))
    else:
        with pytest.raises(fixed_step_module.FixedStepInvariantError):
            await runner.run((fixtures._run_start(),))

    await _assert_full_unknown_failure(repository, configuration.cycle_reservation)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("error", "cancel"))
@pytest.mark.parametrize("outcome", ("success", "known_failure"))
async def test_budget_settlement_faults_after_calls_are_terminalized(
    repository: RunRepository,
    fault: Fault,
    outcome: Literal["success", "known_failure"],
) -> None:
    wrapped = _BudgetFaultRepository(repository, fault)
    grounding = GroundingPipeline(fixtures._grounding_config())
    configuration = fixtures._configuration(grounding, cycle_capacity=1)
    client = (
        fixtures._RequestAwareClient(cast(RunRepository, wrapped), "silence")
        if outcome == "success"
        else _TimeoutClient()
    )
    executor = PaperTwoPhaseCycleExecutor(
        materializer=RepositoryOperationMaterializer(cast(RunRepository, wrapped)),
        client=client,
        prompt_bundle=fixtures.PAPER_TWO_PHASE_V1,
        grounding_pipeline=grounding,
        model_profile=fixtures._profile(),
        call_policy=fixtures._policy(),
    )
    runner = FixedStepRunner(
        repository=cast(RunRepository, wrapped),
        executor=executor,
        grounding_pipeline=grounding,
        configuration=configuration,
    )

    await _expect_fault(fault, lambda: runner.run((fixtures._run_start(),)))

    await _assert_full_unknown_failure(repository, configuration.cycle_reservation)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("error", "cancel"))
async def test_authoritative_materialization_faults_are_terminalized(
    repository: RunRepository,
    fault: Fault,
) -> None:
    wrapped = _PreviewFaultRepository(repository, fault)
    runner, client, configuration = fixtures._runner(
        cast(RunRepository, wrapped),
        mode="reminder",
        cycle_capacity=1,
    )

    await _expect_fault(fault, lambda: runner.run((fixtures._run_start(),)))

    assert wrapped.preview_count == 2
    assert len(client.requests) == 2
    await _assert_full_unknown_failure(repository, configuration.cycle_reservation)


@pytest.mark.asyncio
async def test_unknown_provider_usage_consumes_full_token_reservation(
    repository: RunRepository,
) -> None:
    grounding = GroundingPipeline(fixtures._grounding_config())
    configuration = fixtures._configuration(grounding, cycle_capacity=1)
    client = _UnknownUsageClient(repository, "silence")
    executor = PaperTwoPhaseCycleExecutor(
        materializer=RepositoryOperationMaterializer(repository),
        client=client,
        prompt_bundle=fixtures.PAPER_TWO_PHASE_V1,
        grounding_pipeline=grounding,
        model_profile=fixtures._profile(),
        call_policy=fixtures._policy(),
    )
    runner = FixedStepRunner(
        repository=repository,
        executor=executor,
        grounding_pipeline=grounding,
        configuration=configuration,
    )

    result = await runner.run((fixtures._run_start(),))

    assert len(result.call_receipts) == 2
    assert result.executions[0].usage.provider_input_tokens is None
    assert result.executions[0].usage.provider_output_tokens is None
    settlement = result.cycles[0].budget_settlement
    assert settlement is not None
    assert settlement.input_tokens == configuration.cycle_reservation.input_tokens
    assert settlement.output_tokens == configuration.cycle_reservation.output_tokens
    assert (
        settlement.canonical_token_equivalents
        == configuration.cycle_reservation.canonical_token_equivalents
    )
    assert settlement.model_calls == 2
    assert settlement.latency_us == 201


@pytest.mark.asyncio
async def test_known_phase_one_timeout_is_attested_and_settled_exactly(
    repository: RunRepository,
) -> None:
    grounding = GroundingPipeline(fixtures._grounding_config())
    configuration = fixtures._configuration(grounding, cycle_capacity=1)
    executor = PaperTwoPhaseCycleExecutor(
        materializer=RepositoryOperationMaterializer(repository),
        client=_TimeoutClient(),
        prompt_bundle=fixtures.PAPER_TWO_PHASE_V1,
        grounding_pipeline=grounding,
        model_profile=fixtures._profile(),
        call_policy=fixtures._policy(),
    )
    runner = FixedStepRunner(
        repository=repository,
        executor=executor,
        grounding_pipeline=grounding,
        configuration=configuration,
    )

    result = await runner.run((fixtures._run_start(),))

    assert len(result.cycles) == len(result.executions) == 1
    cycle = result.cycles[0]
    execution = result.executions[0]
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is ReasonCode.MODEL_TIMEOUT
    assert cycle.budget_settlement == BudgetAmounts(
        model_calls=1,
        input_tokens=3,
        output_tokens=4,
        canonical_token_equivalents=(configuration.cycle_reservation.canonical_token_equivalents),
        latency_us=77,
        interventions=0,
        schema_repairs=0,
    )
    assert type(execution) is TwoPhaseCycleFailure
    assert execution.reason is TwoPhaseFailureReason.MODEL_TIMEOUT
    assert len(result.call_receipts) == 1
    assert result.deliveries == result.outcomes == ()
    await _assert_no_memory(repository)

    for field_name, replacement in (
        ("delta_id", "00000000-0000-4000-8000-00000000da7a"),
        ("intervention_id", "00000000-0000-4000-8000-000000001a7e"),
    ):

        def rebind_failure_request(
            payload: dict[str, object],
            field: str = field_name,
            value: str = replacement,
        ) -> None:
            requests = cast(list[dict[str, object]], payload["cycle_requests"])
            executions = cast(list[dict[str, object]], payload["executions"])
            requests[0][field] = value
            requests[0]["request_digest"] = fixtures.two_phase_module._cycle_request_digest(
                requests[0]
            )
            executions[0]["request_digest"] = requests[0]["request_digest"]
            executions[0]["failure_digest"] = fixtures.two_phase_module._failure_digest(
                executions[0]
            )

        fixtures._assert_recalculated_tamper_rejected(result, rebind_failure_request)


@pytest.mark.asyncio
async def test_authoritative_materialization_drift_is_rejected_before_commit(
    repository: RunRepository,
) -> None:
    wrapped = _PreviewDriftRepository(repository)
    runner, client, configuration = fixtures._runner(
        cast(RunRepository, wrapped),
        mode="reminder",
        cycle_capacity=1,
    )

    with pytest.raises(FixedStepExecutionError):
        await runner.run((fixtures._run_start(),))

    assert wrapped.preview_count == 2
    assert len(client.requests) == 2
    cycle = await _latest_cycle(repository)
    assert cycle.state is CycleState.FAILED
    assert cycle.failure_reason is ReasonCode.FAILED_UNKNOWN_COST
    assert cycle.budget_settlement == configuration.cycle_reservation
    await _assert_no_memory(repository)
