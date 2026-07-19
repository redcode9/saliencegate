from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TypeVar, cast
from uuid import UUID

from pydantic import BaseModel

from saliencegate.domain import (
    BudgetAmounts,
    CycleRecord,
    CycleState,
    InterventionDecision,
    InvocationDecision,
    JsonObject,
    MemoryDelta,
    MemoryIdAssignment,
    ReasonCode,
)
from saliencegate.ports.repository import (
    BeginCycle,
    CommitCycle,
    CycleReceipt,
    CycleRecoveryReceipt,
    EnqueueDelivery,
    FailCycle,
    GroundingPin,
    ReserveCycle,
    RunRepository,
    StartCycle,
)

ModelT = TypeVar("ModelT", bound=BaseModel)
CycleSource = CycleReceipt | CycleRecord


class CycleCoordinatorError(ValueError):
    """Base class for safe, stateless cycle-command failures."""


class CycleCoordinatorInputError(CycleCoordinatorError):
    def __init__(self, operation: str) -> None:
        super().__init__(f"{operation} input failed cycle validation")


class CycleCoordinatorStateError(CycleCoordinatorError):
    def __init__(self, operation: str, expected: str) -> None:
        super().__init__(f"{operation} requires {expected}")


class CycleCoordinatorIdentityError(CycleCoordinatorError):
    def __init__(self, operation: str) -> None:
        super().__init__(f"{operation} input does not belong to the cycle")


def _validated_model(value: object, expected_type: type[ModelT], operation: str) -> ModelT:
    if type(value) is not expected_type:
        raise CycleCoordinatorInputError(operation)
    try:
        encoded = cast(BaseModel, value).model_dump_json(warnings=False)
        return expected_type.model_validate_json(encoded)
    except (AttributeError, TypeError, ValueError):
        raise CycleCoordinatorInputError(operation) from None


def _cycle(source: object, operation: str) -> CycleRecord:
    if type(source) is CycleReceipt:
        if type(source.cycle) is not CycleRecord:
            raise CycleCoordinatorInputError(operation)
        receipt = _validated_model(source, CycleReceipt, operation)
        if type(receipt.cycle) is not CycleRecord:  # pragma: no cover - Pydantic invariant
            raise CycleCoordinatorInputError(operation)
        return receipt.cycle
    return _validated_model(source, CycleRecord, operation)


def _timestamp(value: object, operation: str) -> datetime:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise CycleCoordinatorInputError(operation)
    return value.astimezone(UTC)


def _updated_at(cycle: CycleRecord, value: object, operation: str) -> datetime:
    updated_at = _timestamp(value, operation)
    if updated_at < cycle.updated_at:
        raise CycleCoordinatorInputError(operation)
    return updated_at


def _require_state(
    cycle: CycleRecord,
    operation: str,
    expected: CycleState | tuple[CycleState, ...],
) -> None:
    expected_states = (expected,) if isinstance(expected, CycleState) else expected
    if cycle.state not in expected_states:
        label = " or ".join(f"a {state.value} cycle" for state in expected_states)
        raise CycleCoordinatorStateError(operation, label)


def _digests(value: object, operation: str) -> tuple[str, ...]:
    if type(value) is not tuple or any(type(item) is not str for item in value):
        raise CycleCoordinatorInputError(operation)
    return cast(tuple[str, ...], value)


def _latencies(value: object, operation: str) -> tuple[int, ...]:
    if type(value) is not tuple or any(type(item) is not int or item < 0 for item in value):
        raise CycleCoordinatorInputError(operation)
    return cast(tuple[int, ...], value)


def _assignments(
    value: object,
    operation: str,
) -> tuple[MemoryIdAssignment, ...]:
    if type(value) is not tuple:
        raise CycleCoordinatorInputError(operation)
    return tuple(
        _validated_model(item, MemoryIdAssignment, operation)
        for item in cast(tuple[object, ...], value)
    )


def _candidate(
    cycle: CycleRecord,
    operation: str,
    **updates: object,
) -> CycleRecord:
    values = cycle.model_dump(mode="python", warnings=False)
    values.update(revision=cycle.revision + 1, **updates)
    try:
        return CycleRecord.model_validate(values)
    except (TypeError, ValueError):
        raise CycleCoordinatorInputError(operation) from None


def _validated_command(command: ModelT, operation: str) -> ModelT:
    try:
        return type(command).model_validate_json(command.model_dump_json(warnings=False))
    except (TypeError, ValueError):
        raise CycleCoordinatorInputError(operation) from None


class CycleCommandFactory:
    """Build repository commands solely from validated authoritative cycle values."""

    __slots__ = ()

    def begin(
        self,
        decision: InvocationDecision,
        *,
        grounding: GroundingPin,
        created_at: datetime,
    ) -> BeginCycle:
        operation = "begin_cycle"
        validated = _validated_model(decision, InvocationDecision, operation)
        timestamp = _timestamp(created_at, operation)
        if not validated.invoke:
            raise CycleCoordinatorStateError(operation, "an invoking decision")
        if timestamp < validated.created_at:
            raise CycleCoordinatorInputError(operation)
        pinned = _validated_model(grounding, GroundingPin, operation)
        try:
            command = BeginCycle(
                run_id=validated.run_id,
                invocation_decision_id=validated.decision_id,
                grounding_version=pinned.grounding_version,
                grounding_configuration=pinned.grounding_configuration,
                grounding_configuration_digest=pinned.grounding_configuration_digest,
                requested_delivery_target=pinned.requested_delivery_target,
                created_at=timestamp,
            )
        except (TypeError, ValueError):
            raise CycleCoordinatorInputError(operation) from None
        return _validated_command(command, operation)

    def reserve(
        self,
        source: CycleSource,
        *,
        reservation: BudgetAmounts,
        updated_at: datetime,
    ) -> ReserveCycle:
        operation = "reserve_cycle"
        cycle = _cycle(source, operation)
        _require_state(cycle, operation, CycleState.PENDING)
        held = _validated_model(reservation, BudgetAmounts, operation)
        timestamp = _updated_at(cycle, updated_at, operation)
        _candidate(
            cycle,
            operation,
            state=CycleState.RESERVED,
            budget_reservation=held,
            updated_at=timestamp,
        )
        return _validated_command(
            ReserveCycle(
                run_id=cycle.run_id,
                cycle_id=cycle.cycle_id,
                expected_revision=cycle.revision,
                reservation=held,
                updated_at=timestamp,
            ),
            operation,
        )

    def start(
        self,
        source: CycleSource,
        *,
        batch_digest: str,
        updated_at: datetime,
    ) -> StartCycle:
        operation = "mark_cycle_running"
        cycle = _cycle(source, operation)
        _require_state(cycle, operation, CycleState.RESERVED)
        if type(batch_digest) is not str:
            raise CycleCoordinatorInputError(operation)
        timestamp = _updated_at(cycle, updated_at, operation)
        _candidate(
            cycle,
            operation,
            state=CycleState.RUNNING,
            batch_digest=batch_digest,
            updated_at=timestamp,
        )
        return _validated_command(
            StartCycle(
                run_id=cycle.run_id,
                cycle_id=cycle.cycle_id,
                expected_revision=cycle.revision,
                batch_digest=batch_digest,
                updated_at=timestamp,
            ),
            operation,
        )

    def commit(
        self,
        source: CycleSource,
        *,
        settlement: BudgetAmounts,
        validated_delta: MemoryDelta,
        memory_id_assignments: tuple[MemoryIdAssignment, ...],
        intervention: InterventionDecision,
        selector_provenance: JsonObject | None = None,
        delivery: EnqueueDelivery | None = None,
        updated_at: datetime,
        model_call_digests: tuple[str, ...] = (),
        model_call_latencies_us: tuple[int, ...] = (),
    ) -> CommitCycle:
        operation = "commit_cycle"
        cycle = _cycle(source, operation)
        _require_state(cycle, operation, CycleState.RUNNING)
        actual = _validated_model(settlement, BudgetAmounts, operation)
        delta = _validated_model(validated_delta, MemoryDelta, operation)
        assignments = _assignments(memory_id_assignments, operation)
        verdict = _validated_model(intervention, InterventionDecision, operation)
        enqueue = (
            None if delivery is None else _validated_model(delivery, EnqueueDelivery, operation)
        )
        digests = _digests(model_call_digests, operation)
        call_latencies = _latencies(model_call_latencies_us, operation)
        timestamp = _updated_at(cycle, updated_at, operation)
        if (
            delta.run_id != cycle.run_id
            or verdict.run_id != cycle.run_id
            or verdict.cycle_id != cycle.cycle_id
        ):
            raise CycleCoordinatorIdentityError(operation)
        if not (
            cycle.updated_at <= delta.created_at <= timestamp
            and cycle.updated_at <= verdict.created_at <= timestamp
        ):
            raise CycleCoordinatorInputError(operation)
        if verdict.created_at < delta.created_at:
            raise CycleCoordinatorInputError(operation)
        _candidate(
            cycle,
            operation,
            state=CycleState.COMMITTED,
            budget_settlement=actual,
            model_call_digests=digests,
            model_call_latencies_us=call_latencies,
            validated_delta=delta,
            memory_id_assignments=assignments,
            intervention=verdict,
            selector_provenance=selector_provenance,
            updated_at=timestamp,
        )
        return _validated_command(
            CommitCycle(
                run_id=cycle.run_id,
                cycle_id=cycle.cycle_id,
                expected_revision=cycle.revision,
                settlement=actual,
                model_call_digests=digests,
                model_call_latencies_us=call_latencies,
                validated_delta=delta,
                memory_id_assignments=assignments,
                intervention=verdict,
                selector_provenance=selector_provenance,
                delivery=enqueue,
                updated_at=timestamp,
            ),
            operation,
        )

    def fail(
        self,
        source: CycleSource,
        *,
        reason: ReasonCode,
        updated_at: datetime,
        settlement: BudgetAmounts | None = None,
        model_call_digests: tuple[str, ...] = (),
        model_call_latencies_us: tuple[int, ...] = (),
    ) -> FailCycle:
        operation = "fail_cycle"
        cycle = _cycle(source, operation)
        _require_state(
            cycle,
            operation,
            (CycleState.PENDING, CycleState.RESERVED, CycleState.RUNNING),
        )
        if type(reason) is not ReasonCode:
            raise CycleCoordinatorInputError(operation)
        actual = (
            None if settlement is None else _validated_model(settlement, BudgetAmounts, operation)
        )
        digests = _digests(model_call_digests, operation)
        call_latencies = _latencies(model_call_latencies_us, operation)
        timestamp = _updated_at(cycle, updated_at, operation)
        if reason is ReasonCode.FAILED_UNKNOWN_COST:
            if cycle.state is not CycleState.RUNNING:
                raise CycleCoordinatorStateError(operation, "a running cycle")
            if actual is not None and actual != cycle.budget_reservation:
                raise CycleCoordinatorInputError(operation)
            actual = cycle.budget_reservation
        if cycle.state is CycleState.PENDING and (actual is not None or digests or call_latencies):
            raise CycleCoordinatorInputError(operation)
        if cycle.state is CycleState.RESERVED and (actual is None or digests or call_latencies):
            raise CycleCoordinatorInputError(operation)
        if cycle.state is CycleState.RUNNING and actual is None:
            raise CycleCoordinatorInputError(operation)
        _candidate(
            cycle,
            operation,
            state=CycleState.FAILED,
            budget_settlement=actual,
            model_call_digests=digests,
            model_call_latencies_us=call_latencies,
            failure_reason=reason,
            updated_at=timestamp,
        )
        return _validated_command(
            FailCycle(
                run_id=cycle.run_id,
                cycle_id=cycle.cycle_id,
                expected_revision=cycle.revision,
                reason=reason,
                settlement=actual,
                model_call_digests=digests,
                model_call_latencies_us=call_latencies,
                updated_at=timestamp,
            ),
            operation,
        )


_COMMANDS = CycleCommandFactory()


class CycleCoordinator:
    """Thin repository coordinator with no cycle cache or parallel mutable state."""

    __slots__ = ("_repository",)

    def __init__(self, repository: RunRepository) -> None:
        self._repository = repository

    async def begin(
        self,
        decision: InvocationDecision,
        *,
        grounding: GroundingPin,
        created_at: datetime,
    ) -> CycleReceipt:
        return await self._repository.begin_cycle(
            _COMMANDS.begin(
                decision,
                grounding=grounding,
                created_at=created_at,
            )
        )

    async def reserve(
        self,
        source: CycleSource,
        *,
        reservation: BudgetAmounts,
        updated_at: datetime,
    ) -> CycleReceipt:
        return await self._repository.reserve_cycle(
            _COMMANDS.reserve(source, reservation=reservation, updated_at=updated_at)
        )

    async def start(
        self,
        source: CycleSource,
        *,
        batch_digest: str,
        updated_at: datetime,
    ) -> CycleReceipt:
        return await self._repository.mark_cycle_running(
            _COMMANDS.start(source, batch_digest=batch_digest, updated_at=updated_at)
        )

    async def commit(
        self,
        source: CycleSource,
        *,
        settlement: BudgetAmounts,
        validated_delta: MemoryDelta,
        memory_id_assignments: tuple[MemoryIdAssignment, ...],
        intervention: InterventionDecision,
        selector_provenance: JsonObject | None = None,
        delivery: EnqueueDelivery | None = None,
        updated_at: datetime,
        model_call_digests: tuple[str, ...] = (),
        model_call_latencies_us: tuple[int, ...] = (),
    ) -> CycleReceipt:
        command = _COMMANDS.commit(
            source,
            settlement=settlement,
            validated_delta=validated_delta,
            memory_id_assignments=memory_id_assignments,
            intervention=intervention,
            selector_provenance=selector_provenance,
            delivery=delivery,
            updated_at=updated_at,
            model_call_digests=model_call_digests,
            model_call_latencies_us=model_call_latencies_us,
        )
        return await self._repository.commit_cycle(command)

    async def fail(
        self,
        source: CycleSource,
        *,
        reason: ReasonCode,
        updated_at: datetime,
        settlement: BudgetAmounts | None = None,
        model_call_digests: tuple[str, ...] = (),
        model_call_latencies_us: tuple[int, ...] = (),
    ) -> CycleReceipt:
        command = _COMMANDS.fail(
            source,
            reason=reason,
            updated_at=updated_at,
            settlement=settlement,
            model_call_digests=model_call_digests,
            model_call_latencies_us=model_call_latencies_us,
        )
        return await self._repository.fail_cycle(command)

    async def recover(self, run_id: UUID, *, recovered_at: datetime) -> CycleRecoveryReceipt:
        operation = "recover_cycles"
        if type(run_id) is not UUID or run_id.version != 4:
            raise CycleCoordinatorInputError(operation)
        timestamp = _timestamp(recovered_at, operation)
        return await self._repository.recover_cycles(
            UUID(int=run_id.int),
            recovered_at=timestamp,
        )
