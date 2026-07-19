from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Literal, cast
from uuid import UUID

import pytest
from pydantic import ValidationError
from tests.runtime import test_fixed_step_runtime as fixtures

from saliencegate.domain import (
    BudgetAmounts,
    CycleRecord,
    CycleState,
    DeduplicationGuarantee,
    DeliveryState,
    DeliveryTarget,
    InterventionOutcome,
    ReasonCode,
    canonical_digest,
    canonical_json,
)
from saliencegate.intervention import GroundingPipeline
from saliencegate.memory.two_phase import (
    PaperTwoPhaseCycleExecutor,
    RepositoryOperationMaterializer,
)
from saliencegate.ports.adapters import (
    AdapterCapabilities,
    DeliveryAdapter,
    DeliveryChannel,
    DeliveryEnvelope,
    DeliveryReceipt,
    DeliveryRole,
    InjectionMapping,
)
from saliencegate.ports.model_calls import (
    ProviderUsageProvenance,
    StructuredCallParseStatus,
    StructuredCallPhase,
    StructuredCallRequest,
    StructuredCallResult,
    StructuredCallStatus,
    StructuredCallUsage,
)
from saliencegate.ports.repository import (
    CommitCycle,
    MemoryDeltaPreview,
    PreviewConflictError,
    ReserveCycle,
    RunRepository,
    StartCycle,
)
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.repository.sqlite import SQLiteRunRepository
from saliencegate.runtime.algorithm_result import (
    FixedStepRecoveryResult,
    fixed_step_recovery_digest,
)
from saliencegate.runtime.fixed_step import (
    FixedStepExecutionError,
    FixedStepInputError,
    FixedStepInvariantError,
    FixedStepRunner,
)

Backend = Literal["memory", "sqlite"]
CrashTransition = Literal["reserved", "running"]
CrashPoint = Literal["after_phase_one", "before_grounding", "before_commit"]


class _ProcessCrash(BaseException):
    pass


def _close(repository: RunRepository) -> None:
    if isinstance(repository, SQLiteRunRepository):
        repository.close()


@pytest.fixture(params=("memory", "sqlite"))
def backend(request: pytest.FixtureRequest) -> Backend:
    return cast(Backend, request.param)


async def _reopen(
    backend: Backend,
    repository: RunRepository,
    path: Path,
) -> RunRepository:
    if backend == "sqlite":
        assert isinstance(repository, SQLiteRunRepository)
        repository.close()
        return fixtures._make_repository("sqlite", path)
    assert isinstance(repository, MemoryRunRepository)
    state = await repository._verified_state(fixtures.RUN_ID)
    restored = fixtures._make_repository("memory", path)
    assert isinstance(restored, MemoryRunRepository)
    await restored._restore_run(state.ledger, state.ledger_head)
    return restored


async def _latest_cycle(repository: RunRepository) -> CycleRecord:
    cycles = tuple(
        entry.record
        for entry in await repository.ledger(fixtures.RUN_ID)
        if type(entry.record) is CycleRecord
    )
    assert cycles
    return cycles[-1]


class _AfterTransitionCrashRepository:
    def __init__(self, inner: RunRepository, transition: CrashTransition) -> None:
        self.inner = inner
        self.transition = transition

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    async def reserve_cycle(self, command: ReserveCycle):
        receipt = await self.inner.reserve_cycle(command)
        if self.transition == "reserved":
            raise _ProcessCrash()
        return receipt

    async def mark_cycle_running(self, command: StartCycle):
        receipt = await self.inner.mark_cycle_running(command)
        if self.transition == "running":
            raise _ProcessCrash()
        return receipt


class _CommitFaultRepository:
    def __init__(
        self,
        inner: RunRepository,
        *,
        after_commit: bool,
        crash: bool = False,
    ) -> None:
        self.inner = inner
        self.after_commit = after_commit
        self.crash = crash

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    async def commit_cycle(self, command: CommitCycle):
        if self.after_commit:
            await self.inner.commit_cycle(command)
        if self.crash:
            raise _ProcessCrash()
        if self.after_commit:
            raise RuntimeError("injected ambiguous commit response")
        raise RuntimeError("injected pre-commit failure")


class _ConflictOnAuthoritativePreviewRepository:
    def __init__(self, inner: RunRepository) -> None:
        self.inner = inner
        self.preview_count = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    async def preview_memory_delta(self, command: object) -> MemoryDeltaPreview:
        self.preview_count += 1
        if self.preview_count == 2:
            raise PreviewConflictError()
        return await self.inner.preview_memory_delta(command)  # type: ignore[arg-type]


class _ConflictOnFirstCommitRepository:
    def __init__(self, inner: RunRepository) -> None:
        self.inner = inner
        self.commit_count = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    async def commit_cycle(self, command: CommitCycle):
        self.commit_count += 1
        if self.commit_count == 1:
            raise PreviewConflictError()
        return await self.inner.commit_cycle(command)


class _CancelFirstPreviewRepository:
    def __init__(self, inner: RunRepository) -> None:
        self.inner = inner
        self.preview_count = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    async def preview_memory_delta(self, command: object) -> MemoryDeltaPreview:
        self.preview_count += 1
        if self.preview_count == 1:
            raise asyncio.CancelledError()
        return await self.inner.preview_memory_delta(command)  # type: ignore[arg-type]


class _PhaseTwoTimeoutClient(fixtures._RequestAwareClient):
    async def generate(self, request: StructuredCallRequest) -> StructuredCallResult:
        if request.phase is StructuredCallPhase.MEMORY_EDIT:
            return await super().generate(request)
        self.requests.append(request)
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


class _CrashBeforePhaseTwoClient(fixtures._RequestAwareClient):
    async def generate(self, request: StructuredCallRequest) -> StructuredCallResult:
        if request.phase is StructuredCallPhase.INTERVENTION:
            raise _ProcessCrash()
        return await super().generate(request)


class _OutcomeResponseFaultRepository:
    def __init__(self, inner: RunRepository) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    async def record_outcome(self, outcome):
        await self.inner.record_outcome(outcome)
        raise RuntimeError("injected lost outcome response")


class _OutcomeCancellationRepository:
    def __init__(self, inner: RunRepository) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    async def record_outcome(self, outcome):
        await self.inner.record_outcome(outcome)
        raise asyncio.CancelledError()


class _InvalidCapabilitiesAdapter(fixtures._DeliveryAdapter):
    def capabilities(self):
        raise RuntimeError("injected capability failure")


class _CancelledAfterCommitRepository:
    def __init__(self, inner: RunRepository) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    async def commit_cycle(self, command: CommitCycle):
        await self.inner.commit_cycle(command)
        raise asyncio.CancelledError()


class _CrashDeliveryAdapter:
    def __init__(self, *, deduplicates: bool = True) -> None:
        self.deduplicates = deduplicates
        self.crash_once = True
        self.calls: list[DeliveryEnvelope] = []
        self.effects: set[object] = set()

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            schema_version="1.0",
            adapter_id=fixtures.ADAPTER_ID,
            pre_action_interception=False,
            deduplicates_delivery_id=self.deduplicates,
            deduplication_guarantee=(
                DeduplicationGuarantee.DURABLE_DELIVERY_ID
                if self.deduplicates
                else DeduplicationGuarantee.AT_MOST_ONCE_ATTEMPT
            ),
            injection_mappings=(
                InjectionMapping(
                    channel=DeliveryChannel.PROVIDER_DATA,
                    role=DeliveryRole.DATA,
                    provider_channel="context",
                ),
            ),
        )

    async def deliver(self, envelope: DeliveryEnvelope) -> DeliveryReceipt:
        self.calls.append(envelope)
        self.effects.add(envelope.delivery_id)
        if self.crash_once:
            self.crash_once = False
            raise _ProcessCrash()
        return DeliveryReceipt(
            schema_version="1.0",
            delivery_id=envelope.delivery_id,
            attempt_id=envelope.attempt_id,
            attempt_number=envelope.attempt_number,
            adapter_id=envelope.adapter_id,
            target_request_id=envelope.target_request_id,
            delivered_at=envelope.created_at + timedelta(microseconds=1),
            provider_receipt_id="recovered-delivery/v1",
        )


def _recovery_runner(
    repository: RunRepository,
    *,
    cycle_capacity: int,
    adapter: DeliveryAdapter | None = None,
    target: DeliveryTarget | None = None,
) -> tuple[FixedStepRunner, fixtures._RequestAwareClient]:
    runner, client, _configuration = fixtures._runner(
        repository,
        mode="silence",
        cycle_capacity=cycle_capacity,
        delivery_adapter=adapter,
        requested_delivery_target=target,
    )
    return runner, client


def _paper_runner(
    repository: RunRepository,
    client: fixtures._RequestAwareClient,
    *,
    cycle_capacity: int = 1,
) -> tuple[FixedStepRunner, object]:
    grounding = GroundingPipeline(fixtures._grounding_config())
    configuration = fixtures._configuration(grounding, cycle_capacity=cycle_capacity)
    executor = PaperTwoPhaseCycleExecutor(
        materializer=RepositoryOperationMaterializer(repository),
        client=client,
        prompt_bundle=fixtures.PAPER_TWO_PHASE_V1,
        grounding_pipeline=grounding,
        model_profile=fixtures._profile(),
        call_policy=fixtures._policy(),
    )
    return (
        FixedStepRunner(
            repository=repository,
            executor=executor,
            grounding_pipeline=grounding,
            configuration=configuration,
        ),
        configuration,
    )


@pytest.mark.asyncio
async def test_phase_two_transport_failure_settles_two_calls_without_memory(
    backend: Backend,
    tmp_path: Path,
) -> None:
    path = tmp_path / f"phase-two-{backend}.sqlite3"
    repository = fixtures._make_repository(backend, path)
    try:
        grounding = GroundingPipeline(fixtures._grounding_config())
        configuration = fixtures._configuration(grounding, cycle_capacity=1)
        client = _PhaseTwoTimeoutClient(repository, "silence")
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

        cycle = result.cycles[0]
        assert cycle.state is CycleState.FAILED
        assert cycle.failure_reason is ReasonCode.MODEL_TIMEOUT
        assert cycle.budget_settlement == BudgetAmounts(
            model_calls=2,
            input_tokens=14,
            output_tokens=11,
            canonical_token_equivalents=(
                configuration.cycle_reservation.canonical_token_equivalents
            ),
            latency_us=177,
            interventions=0,
            schema_repairs=0,
        )
        assert len(result.call_receipts) == len(client.requests) == 2
        snapshot = await repository.snapshot(fixtures.RUN_ID)
        assert snapshot.memory_cursor == 0
        assert snapshot.records == ()
    finally:
        _close(repository)


@pytest.mark.asyncio
async def test_recovery_rejects_invalid_input_and_missing_delivery_authority(
    tmp_path: Path,
) -> None:
    empty = fixtures._make_repository("memory", tmp_path / "unused.sqlite3")
    runner, _client = _recovery_runner(empty, cycle_capacity=1)
    with pytest.raises(FixedStepInputError):
        await runner.recover(cast(UUID, "not-a-run-id"), recovered_at=fixtures.NOW)
    with pytest.raises(FixedStepInputError):
        await runner.recover(fixtures.RUN_ID, recovered_at=fixtures.NOW.replace(tzinfo=None))
    with pytest.raises(FixedStepInputError):
        await runner.recover(fixtures.RUN_ID, recovered_at=fixtures.NOW)

    repository = fixtures._make_repository("memory", tmp_path / "pending.sqlite3")
    adapter = fixtures._DeliveryAdapter("deliver")
    wrapped = _CommitFaultRepository(repository, after_commit=True, crash=True)
    crash_runner, _calls, _configuration = fixtures._runner(
        cast(RunRepository, wrapped),
        mode="reminder",
        cycle_capacity=1,
        delivery_adapter=adapter,
        requested_delivery_target=DeliveryTarget.NEXT_MODEL_CALL,
    )
    with pytest.raises(_ProcessCrash):
        await crash_runner.run((fixtures._run_start(target_request_id="pending-authority"),))

    unauthorized, recovery_client, _configuration = fixtures._runner(
        repository,
        mode="silence",
        cycle_capacity=1,
        delivery_adapter=None,
        requested_delivery_target=DeliveryTarget.NEXT_MODEL_CALL,
    )
    with pytest.raises(FixedStepInvariantError):
        await unauthorized.recover(
            fixtures.RUN_ID,
            recovered_at=fixtures.NOW + timedelta(seconds=30),
        )
    assert recovery_client.requests == []
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_cancellation_after_phase_one_consumes_full_reservation_without_preview(
    backend: Backend,
    tmp_path: Path,
) -> None:
    path = tmp_path / f"phase-one-cancel-{backend}.sqlite3"
    repository = fixtures._make_repository(backend, path)
    try:
        wrapped = _CancelFirstPreviewRepository(repository)
        runner, client, configuration = fixtures._runner(
            cast(RunRepository, wrapped),
            mode="silence",
            cycle_capacity=1,
        )

        with pytest.raises(asyncio.CancelledError):
            await runner.run((fixtures._run_start(),))

        assert wrapped.preview_count == 1
        assert len(client.requests) == 1
        cycle = await _latest_cycle(repository)
        assert cycle.state is CycleState.FAILED
        assert cycle.failure_reason is ReasonCode.FAILED_UNKNOWN_COST
        assert cycle.budget_settlement == configuration.cycle_reservation
        snapshot = await repository.snapshot(fixtures.RUN_ID)
        assert snapshot.memory_cursor == 0
        assert snapshot.records == ()
    finally:
        _close(repository)


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ("reserved", "running"))
async def test_process_restart_recovers_only_durable_cycle_state(
    backend: Backend,
    tmp_path: Path,
    transition: CrashTransition,
) -> None:
    path = tmp_path / f"{backend}-{transition}.sqlite3"
    repository = fixtures._make_repository(backend, path)
    current = repository
    try:
        wrapped = _AfterTransitionCrashRepository(repository, transition)
        runner, client, configuration = fixtures._runner(
            cast(RunRepository, wrapped),
            mode="silence",
            cycle_capacity=1,
        )
        with pytest.raises(_ProcessCrash):
            await runner.run((fixtures._run_start(),))
        assert client.requests == []

        current = await _reopen(backend, repository, path)
        recovery_runner, recovery_client = _recovery_runner(current, cycle_capacity=1)
        recovered = await recovery_runner.recover(
            fixtures.RUN_ID,
            recovered_at=fixtures.NOW + timedelta(seconds=30),
        )
        assert recovery_client.requests == []
        assert recovered.rebuild_equivalent is True
        assert recovered.memory_snapshot.memory_cursor == 0
        assert recovered.memory_snapshot.records == ()
        if transition == "reserved":
            assert len(recovered.cycle_recovery.resumable_reserved) == 1
            assert recovered.cycle_recovery.failed_unknown_cost == ()
            assert recovered.budget_snapshot.reserved == configuration.cycle_reservation
            assert recovered.budget_snapshot.consumed == BudgetAmounts()
            payload = recovered.model_dump(mode="json", warnings=False)
            recovery = cast(dict[str, object], payload["cycle_recovery"])
            reserved = cast(list[dict[str, object]], recovery["resumable_reserved"])
            reserved[0]["state"] = CycleState.PENDING.value
            reserved[0]["budget_reservation"] = None
            payload["result_digest"] = fixed_step_recovery_digest(payload)
            with pytest.raises(ValidationError, match="reserved recovery"):
                FixedStepRecoveryResult.model_validate_json(canonical_json(payload))
            payload = recovered.model_dump(mode="json", warnings=False)
            recovery = cast(dict[str, object], payload["cycle_recovery"])
            reserved = cast(list[dict[str, object]], recovery["resumable_reserved"])
            recovery["resumable_pending"] = reserved
            recovery["resumable_reserved"] = []
            payload["result_digest"] = fixed_step_recovery_digest(payload)
            with pytest.raises(ValidationError, match="pending recovery"):
                FixedStepRecoveryResult.model_validate_json(canonical_json(payload))
        else:
            assert recovered.cycle_recovery.resumable_reserved == ()
            assert len(recovered.cycle_recovery.failed_unknown_cost) == 1
            failed = recovered.cycle_recovery.failed_unknown_cost[0].cycle
            assert failed.revision == 4
            assert failed.failure_reason is ReasonCode.FAILED_UNKNOWN_COST
            assert failed.budget_settlement == configuration.cycle_reservation
            assert recovered.budget_snapshot.reserved == BudgetAmounts()
            assert recovered.budget_snapshot.consumed == configuration.cycle_reservation

        ledger_before = await current.ledger(fixtures.RUN_ID)
        second = await recovery_runner.recover(
            fixtures.RUN_ID,
            recovered_at=fixtures.NOW + timedelta(seconds=31),
        )
        assert second.cycle_recovery.failed_unknown_cost == ()
        assert await current.ledger(fixtures.RUN_ID) == ledger_before
    finally:
        _close(current)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("point", "expected_calls"),
    (("after_phase_one", 1), ("before_grounding", 2), ("before_commit", 2)),
)
async def test_precommit_process_crash_never_publishes_phase_one_preview(
    backend: Backend,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    point: CrashPoint,
    expected_calls: int,
) -> None:
    path = tmp_path / f"precommit-{backend}-{point}.sqlite3"
    repository = fixtures._make_repository(backend, path)
    current = repository
    try:
        if point == "after_phase_one":
            client = _CrashBeforePhaseTwoClient(repository, "silence")
            runner, configuration = _paper_runner(repository, client)
        elif point == "before_grounding":
            runner, client, configuration = fixtures._runner(
                repository,
                mode="silence",
                cycle_capacity=1,
            )

            def crash_grounding(*_args: object, **_kwargs: object) -> None:
                raise _ProcessCrash()

            monkeypatch.setattr(GroundingPipeline, "ground", crash_grounding)
        else:
            wrapped = _CommitFaultRepository(repository, after_commit=False, crash=True)
            runner, client, configuration = fixtures._runner(
                cast(RunRepository, wrapped),
                mode="silence",
                cycle_capacity=1,
            )

        with pytest.raises(_ProcessCrash):
            await runner.run((fixtures._run_start(),))
        assert len(client.requests) == expected_calls
        running = await _latest_cycle(repository)
        assert running.state is CycleState.RUNNING
        before = await repository.snapshot(fixtures.RUN_ID)
        assert before.memory_cursor == 0
        assert before.records == ()

        current = await _reopen(backend, repository, path)
        recovery_runner, recovery_client = _recovery_runner(current, cycle_capacity=1)
        recovered = await recovery_runner.recover(
            fixtures.RUN_ID,
            recovered_at=fixtures.NOW + timedelta(seconds=30),
        )

        assert recovery_client.requests == []
        assert len(recovered.cycle_recovery.failed_unknown_cost) == 1
        failed = recovered.cycle_recovery.failed_unknown_cost[0].cycle
        assert failed.state is CycleState.FAILED
        assert failed.failure_reason is ReasonCode.FAILED_UNKNOWN_COST
        assert failed.budget_settlement == configuration.cycle_reservation
        assert recovered.memory_snapshot.memory_cursor == 0
        assert recovered.memory_snapshot.records == ()
    finally:
        _close(current)


@pytest.mark.asyncio
async def test_generic_commit_failure_is_unknown_but_durable_commit_is_reconciled(
    backend: Backend,
    tmp_path: Path,
) -> None:
    for after_commit in (False, True):
        path = tmp_path / f"commit-{backend}-{after_commit}.sqlite3"
        repository = fixtures._make_repository(backend, path)
        try:
            wrapped = _CommitFaultRepository(repository, after_commit=after_commit)
            runner, client, configuration = fixtures._runner(
                cast(RunRepository, wrapped),
                mode="silence",
                cycle_capacity=1,
            )
            if after_commit:
                result = await runner.run((fixtures._run_start(),))
                assert result.cycles[0].state is CycleState.COMMITTED
                assert (await repository.snapshot(fixtures.RUN_ID)).memory_cursor == 1
            else:
                with pytest.raises(FixedStepExecutionError):
                    await runner.run((fixtures._run_start(),))
                cycle = await _latest_cycle(repository)
                assert cycle.state is CycleState.FAILED
                assert cycle.failure_reason is ReasonCode.FAILED_UNKNOWN_COST
                assert cycle.budget_settlement == configuration.cycle_reservation
                assert (await repository.snapshot(fixtures.RUN_ID)).memory_cursor == 0
            assert len(client.requests) == 2
        finally:
            _close(repository)


@pytest.mark.asyncio
async def test_cancellation_after_durable_commit_preserves_committed_state(
    backend: Backend,
    tmp_path: Path,
) -> None:
    path = tmp_path / f"cancelled-commit-{backend}.sqlite3"
    repository = fixtures._make_repository(backend, path)
    try:
        wrapped = _CancelledAfterCommitRepository(repository)
        runner, client, _configuration = fixtures._runner(
            cast(RunRepository, wrapped),
            mode="silence",
            cycle_capacity=1,
        )

        with pytest.raises(asyncio.CancelledError):
            await runner.run((fixtures._run_start(),))

        assert len(client.requests) == 2
        cycle = await _latest_cycle(repository)
        assert cycle.state is CycleState.COMMITTED
        assert cycle.revision == 4
        snapshot = await repository.snapshot(fixtures.RUN_ID)
        assert snapshot.ingestion_cursor == snapshot.memory_cursor == 1
    finally:
        _close(repository)


@pytest.mark.asyncio
async def test_lost_outcome_response_is_reconciled_idempotently(
    backend: Backend,
    tmp_path: Path,
) -> None:
    path = tmp_path / f"outcome-{backend}.sqlite3"
    repository = fixtures._make_repository(backend, path)
    try:
        wrapped = _OutcomeResponseFaultRepository(repository)
        runner, client, _configuration = fixtures._runner(
            cast(RunRepository, wrapped),
            mode="silence",
            cycle_capacity=1,
        )

        result = await runner.run((fixtures._run_start(),))

        assert len(client.requests) == 2
        assert len(result.outcomes) == 1
        stored = tuple(
            entry.record
            for entry in await repository.ledger(fixtures.RUN_ID)
            if type(entry.record) is type(result.outcomes[0])
        )
        assert stored == result.outcomes
        assert (await repository.rebuild(fixtures.RUN_ID)).equivalent is True
    finally:
        _close(repository)


@pytest.mark.asyncio
async def test_cancellation_after_durable_outcome_preserves_exactly_one_record(
    backend: Backend,
    tmp_path: Path,
) -> None:
    path = tmp_path / f"cancelled-outcome-{backend}.sqlite3"
    repository = fixtures._make_repository(backend, path)
    try:
        wrapped = _OutcomeCancellationRepository(repository)
        runner, client, _configuration = fixtures._runner(
            cast(RunRepository, wrapped),
            mode="silence",
            cycle_capacity=1,
        )

        with pytest.raises(asyncio.CancelledError):
            await runner.run((fixtures._run_start(),))

        assert len(client.requests) == 2
        cycle = await _latest_cycle(repository)
        assert cycle.state is CycleState.COMMITTED
        snapshot = await repository.snapshot(fixtures.RUN_ID)
        assert snapshot.ingestion_cursor == snapshot.memory_cursor == 1
        outcomes = tuple(
            entry.record
            for entry in await repository.ledger(fixtures.RUN_ID)
            if isinstance(entry.record, InterventionOutcome)
        )
        assert len(outcomes) == 1
        assert (await repository.rebuild(fixtures.RUN_ID)).equivalent is True
    finally:
        _close(repository)


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict_at", ("authoritative_preview", "commit"))
async def test_memory_conflict_keeps_range_eligible_for_next_boundary(
    backend: Backend,
    tmp_path: Path,
    conflict_at: Literal["authoritative_preview", "commit"],
) -> None:
    path = tmp_path / f"conflict-{backend}-{conflict_at}.sqlite3"
    repository = fixtures._make_repository(backend, path)
    try:
        wrapped = (
            _ConflictOnAuthoritativePreviewRepository(repository)
            if conflict_at == "authoritative_preview"
            else _ConflictOnFirstCommitRepository(repository)
        )
        runner, client, _configuration = fixtures._runner(
            cast(RunRepository, wrapped),
            mode="silence",
            cycle_capacity=2,
        )
        result = await runner.run(
            (
                fixtures._run_start(),
                fixtures._message_event(
                    2,
                    step=2,
                    role=fixtures.LogicalMessageRole.ASSISTANT,
                ),
            )
        )

        assert tuple(cycle.state for cycle in result.cycles) == (
            CycleState.FAILED,
            CycleState.COMMITTED,
        )
        assert result.cycles[0].failure_reason is ReasonCode.MEMORY_CONFLICT
        assert result.cycles[0].budget_settlement is not None
        assert result.cycles[0].budget_settlement.model_calls == 2
        assert result.cycles[1].first_event_sequence == 1
        assert result.cycles[1].last_event_sequence == 2
        assert len(client.requests) == 4
        snapshot = await repository.snapshot(fixtures.RUN_ID)
        assert snapshot.ingestion_cursor == snapshot.memory_cursor == 2
    finally:
        _close(repository)


@pytest.mark.asyncio
async def test_post_commit_crash_recovers_delivery_without_model_retry(
    backend: Backend,
    tmp_path: Path,
) -> None:
    path = tmp_path / f"post-commit-{backend}.sqlite3"
    repository = fixtures._make_repository(backend, path)
    current = repository
    adapter = fixtures._DeliveryAdapter("deliver")
    try:
        wrapped = _CommitFaultRepository(repository, after_commit=True, crash=True)
        runner, client, _configuration = fixtures._runner(
            cast(RunRepository, wrapped),
            mode="reminder",
            cycle_capacity=1,
            delivery_adapter=adapter,
            requested_delivery_target=DeliveryTarget.NEXT_MODEL_CALL,
        )
        with pytest.raises(_ProcessCrash):
            await runner.run((fixtures._run_start(target_request_id="after-restart"),))
        assert len(client.requests) == 2
        assert adapter.calls == []
        cycle = await _latest_cycle(repository)
        assert cycle.state is CycleState.COMMITTED
        before = await repository.snapshot(fixtures.RUN_ID)
        assert before.ingestion_cursor == before.memory_cursor == 1
        pending = tuple(
            entry.record
            for entry in await repository.ledger(fixtures.RUN_ID)
            if getattr(entry.record, "state", None) is DeliveryState.PENDING
        )
        assert len(pending) == 1

        current = await _reopen(backend, repository, path)
        wrong_runner, wrong_client = _recovery_runner(
            current,
            cycle_capacity=2,
            adapter=adapter,
            target=DeliveryTarget.NEXT_MODEL_CALL,
        )
        with pytest.raises(FixedStepInvariantError):
            await wrong_runner.recover(
                fixtures.RUN_ID,
                recovered_at=fixtures.NOW + timedelta(seconds=29),
            )
        assert wrong_client.requests == []
        assert adapter.calls == []
        invalid_adapter = _InvalidCapabilitiesAdapter("deliver")
        invalid_runner, invalid_client = _recovery_runner(
            current,
            cycle_capacity=1,
            adapter=invalid_adapter,
            target=DeliveryTarget.NEXT_MODEL_CALL,
        )
        with pytest.raises(FixedStepInvariantError):
            await invalid_runner.recover(
                fixtures.RUN_ID,
                recovered_at=fixtures.NOW + timedelta(seconds=29),
            )
        assert invalid_client.requests == []
        assert invalid_adapter.calls == []
        assert adapter.calls == []
        recovery_runner, recovery_client = _recovery_runner(
            current,
            cycle_capacity=1,
            adapter=adapter,
            target=DeliveryTarget.NEXT_MODEL_CALL,
        )
        recovered = await recovery_runner.recover(
            fixtures.RUN_ID,
            recovered_at=fixtures.NOW + timedelta(seconds=30),
        )

        assert recovery_client.requests == []
        assert len(adapter.calls) == 1
        assert len(recovered.deliveries) == 1
        assert recovered.deliveries[0].state is DeliveryState.DELIVERED
        assert recovered.memory_snapshot.memory_cursor == 1
        assert len(recovered.memory_snapshot.records) == 1
        assert (await _latest_cycle(current)).state is CycleState.COMMITTED

        omitted_delivery = recovered.model_dump(mode="json", warnings=False)
        omitted_delivery["deliveries"] = []
        omitted_delivery["result_digest"] = fixed_step_recovery_digest(omitted_delivery)
        with pytest.raises(ValidationError, match="deliveries differ"):
            FixedStepRecoveryResult.model_validate_json(canonical_json(omitted_delivery))
    finally:
        _close(current)


@pytest.mark.asyncio
@pytest.mark.parametrize("deduplicates", (True, False))
async def test_delivery_crash_retries_only_with_durable_deduplication(
    backend: Backend,
    tmp_path: Path,
    deduplicates: bool,
) -> None:
    path = tmp_path / f"delivery-{backend}-{deduplicates}.sqlite3"
    repository = fixtures._make_repository(backend, path)
    current = repository
    adapter = _CrashDeliveryAdapter(deduplicates=deduplicates)
    try:
        runner, client, _configuration = fixtures._runner(
            repository,
            mode="reminder",
            cycle_capacity=1,
            delivery_adapter=cast(DeliveryAdapter, adapter),
            requested_delivery_target=DeliveryTarget.NEXT_MODEL_CALL,
        )
        with pytest.raises(_ProcessCrash):
            await runner.run((fixtures._run_start(target_request_id="delivery-restart"),))
        assert len(client.requests) == 2
        assert len(adapter.calls) == 1
        assert len(adapter.effects) == 1
        attempting = tuple(
            entry.record
            for entry in await repository.ledger(fixtures.RUN_ID)
            if getattr(entry.record, "state", None) is DeliveryState.ATTEMPTING
        )
        assert attempting

        current = await _reopen(backend, repository, path)
        recovery_runner, recovery_client = _recovery_runner(
            current,
            cycle_capacity=1,
            adapter=(cast(DeliveryAdapter, adapter) if deduplicates else None),
            target=DeliveryTarget.NEXT_MODEL_CALL,
        )
        recovered = await recovery_runner.recover(
            fixtures.RUN_ID,
            recovered_at=fixtures.NOW + timedelta(seconds=30),
        )

        assert recovery_client.requests == []
        assert len(adapter.effects) == 1
        assert len(recovered.deliveries) == 1
        if deduplicates:
            assert len(adapter.calls) == 2
            assert recovered.deliveries[0].state is DeliveryState.DELIVERED
            payload = recovered.model_dump(mode="json", warnings=False)
            deliveries = cast(list[dict[str, object]], payload["deliveries"])
            deliveries.append(dict(deliveries[0]))
            payload["result_digest"] = fixed_step_recovery_digest(payload)
            with pytest.raises(ValidationError, match="deliveries"):
                FixedStepRecoveryResult.model_validate_json(canonical_json(payload))
            payload = recovered.model_dump(mode="json", warnings=False)
            deliveries = cast(list[dict[str, object]], payload["deliveries"])
            deliveries[0]["updated_at"] = (
                (fixtures.NOW + timedelta(seconds=31)).isoformat().replace("+00:00", "Z")
            )
            payload["result_digest"] = fixed_step_recovery_digest(payload)
            with pytest.raises(ValidationError, match="ledger replay"):
                FixedStepRecoveryResult.model_validate_json(canonical_json(payload))
        else:
            assert len(adapter.calls) == 1
            assert recovered.deliveries[0].state is DeliveryState.UNKNOWN
        assert (await _latest_cycle(current)).state is CycleState.COMMITTED
        assert recovered.memory_snapshot.memory_cursor == 1
    finally:
        _close(current)


@pytest.mark.asyncio
async def test_running_recovery_is_byte_identical_across_backends(tmp_path: Path) -> None:
    encoded: list[str] = []
    for backend in cast(tuple[Backend, ...], ("memory", "sqlite")):
        path = tmp_path / f"deterministic-{backend}.sqlite3"
        repository = fixtures._make_repository(backend, path)
        current = repository
        try:
            wrapped = _AfterTransitionCrashRepository(repository, "running")
            runner, _client, _configuration = fixtures._runner(
                cast(RunRepository, wrapped),
                mode="silence",
                cycle_capacity=1,
            )
            with pytest.raises(_ProcessCrash):
                await runner.run((fixtures._run_start(),))
            current = await _reopen(backend, repository, path)
            recovery_runner, recovery_client = _recovery_runner(current, cycle_capacity=1)
            recovered = await recovery_runner.recover(
                fixtures.RUN_ID,
                recovered_at=fixtures.NOW + timedelta(seconds=30),
            )
            assert recovery_client.requests == []
            encoded.append(canonical_json(recovered))
        finally:
            _close(current)

    assert encoded[0] == encoded[1]


@pytest.mark.asyncio
async def test_recovery_result_rejects_recalculated_semantic_change(tmp_path: Path) -> None:
    repository = fixtures._make_repository("memory", tmp_path / "unused.sqlite3")
    wrapped = _AfterTransitionCrashRepository(repository, "running")
    runner, _client, _configuration = fixtures._runner(
        cast(RunRepository, wrapped),
        mode="silence",
        cycle_capacity=1,
    )
    with pytest.raises(_ProcessCrash):
        await runner.run((fixtures._run_start(),))
    recovery_runner, _client = _recovery_runner(repository, cycle_capacity=1)
    recovered = await recovery_runner.recover(
        fixtures.RUN_ID,
        recovered_at=fixtures.NOW + timedelta(seconds=30),
    )

    def deny_rebuild(payload: dict[str, object]) -> None:
        payload["rebuild_equivalent"] = False

    def advance_memory_cursor(payload: dict[str, object]) -> None:
        snapshot = cast(dict[str, object], payload["memory_snapshot"])
        snapshot["memory_cursor"] = 1

    def undercharge_unknown_cycle(payload: dict[str, object]) -> None:
        recovery = cast(dict[str, object], payload["cycle_recovery"])
        failures = cast(list[dict[str, object]], recovery["failed_unknown_cost"])
        cycle = cast(dict[str, object], failures[0]["cycle"])
        settlement = cast(dict[str, object], cycle["budget_settlement"])
        settlement["model_calls"] = 1

    def wrong_recovery_run(payload: dict[str, object]) -> None:
        recovery = cast(dict[str, object], payload["cycle_recovery"])
        recovery["run_id"] = "00000000-0000-4000-8000-00000000beef"

    def duplicate_recovered_cycle(payload: dict[str, object]) -> None:
        recovery = cast(dict[str, object], payload["cycle_recovery"])
        failures = cast(list[dict[str, object]], recovery["failed_unknown_cost"])
        failures.append(dict(failures[0]))

    def deny_new_recovery_receipt(payload: dict[str, object]) -> None:
        recovery = cast(dict[str, object], payload["cycle_recovery"])
        failures = cast(list[dict[str, object]], recovery["failed_unknown_cost"])
        failures[0]["appended"] = False

    def alter_budget_projection(payload: dict[str, object]) -> None:
        budget = cast(dict[str, dict[str, object]], payload["budget_snapshot"])
        consumed = budget["consumed"]
        consumed["model_calls"] = 1

    def truncate_ledger(payload: dict[str, object]) -> None:
        ledger = cast(list[object], payload["ledger"])
        payload["ledger"] = ledger[:-1]

    def wrong_ledger_count(payload: dict[str, object]) -> None:
        payload["ledger_entry_count"] = cast(int, payload["ledger_entry_count"]) + 1

    def omit_recovered_cycle(payload: dict[str, object]) -> None:
        recovery = cast(dict[str, object], payload["cycle_recovery"])
        recovery["failed_unknown_cost"] = []

    def corrupt_recovery_receipt_tag(payload: dict[str, object]) -> None:
        recovery = cast(dict[str, object], payload["cycle_recovery"])
        failures = cast(list[dict[str, object]], recovery["failed_unknown_cost"])
        record_tag = cast(dict[str, object], failures[0]["record_tag"])
        record_tag["value"] = "0" * 64

    def make_cycle_newer_than_recovery(payload: dict[str, object]) -> None:
        payload["recovered_at"] = (
            (fixtures.NOW - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
        )

    def break_semantic_ledger_replay(payload: dict[str, object]) -> None:
        ledger = cast(list[dict[str, object]], payload["ledger"])
        final_record = cast(dict[str, object], ledger[-1]["record"])
        final_record["revision"] = cast(int, final_record["revision"]) + 1

    def alter_semantic_projection_digest(payload: dict[str, object]) -> None:
        digests = cast(dict[str, dict[str, object]], payload["semantic_projection_digests"])
        digests["events"]["value"] = "0" * 64

    def replace_semantic_projection_with_self_consistent_values(
        payload: dict[str, object],
    ) -> None:
        digests = cast(dict[str, dict[str, object]], payload["semantic_projection_digests"])
        digests["events"]["value"] = "0" * 64
        digests["overall"]["value"] = canonical_digest(digests)

    wrong_grounding = GroundingPipeline(fixtures._grounding_config())
    wrong_configuration = fixtures._configuration(wrong_grounding, cycle_capacity=2)

    def substitute_valid_wrong_configuration(payload: dict[str, object]) -> None:
        payload["configuration"] = wrong_configuration.model_dump(mode="json", warnings=False)

    for mutate in (
        deny_rebuild,
        advance_memory_cursor,
        undercharge_unknown_cycle,
        wrong_recovery_run,
        duplicate_recovered_cycle,
        deny_new_recovery_receipt,
        alter_budget_projection,
        truncate_ledger,
        wrong_ledger_count,
        omit_recovered_cycle,
        corrupt_recovery_receipt_tag,
        make_cycle_newer_than_recovery,
        break_semantic_ledger_replay,
        alter_semantic_projection_digest,
        replace_semantic_projection_with_self_consistent_values,
        substitute_valid_wrong_configuration,
    ):
        payload = recovered.model_dump(mode="json", warnings=False)
        mutate(payload)
        payload["result_digest"] = fixed_step_recovery_digest(payload)
        with pytest.raises(ValidationError):
            FixedStepRecoveryResult.model_validate_json(canonical_json(payload))

    invalid_digest = recovered.model_dump(mode="json", warnings=False)
    invalid_digest["result_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="recovery digest"):
        FixedStepRecoveryResult.model_validate_json(canonical_json(invalid_digest))
