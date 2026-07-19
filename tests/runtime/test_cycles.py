from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from saliencegate.domain import (
    BudgetAmounts,
    BudgetLimits,
    BudgetSnapshot,
    CycleRecord,
    CycleState,
    DeliveryTarget,
    InterventionAction,
    InterventionDecision,
    InvocationDecision,
    MemoryDelta,
    MemoryIdAssignment,
    PayloadDigest,
    PayloadDigestAlgorithm,
    ReasonCode,
    cycle_id,
)
from saliencegate.ports.repository import (
    BeginCycle,
    CommitCycle,
    CycleReceipt,
    CycleRecoveryReceipt,
    FailCycle,
    GroundingPin,
    ReserveCycle,
    RunRepository,
    StartCycle,
)
from saliencegate.runtime import (
    CycleCommandFactory,
    CycleCoordinator,
    CycleCoordinatorIdentityError,
    CycleCoordinatorInputError,
    CycleCoordinatorStateError,
)

RUN_ID = UUID("00000000-0000-4000-8000-000000000601")
OTHER_RUN_ID = UUID("00000000-0000-4000-8000-000000000602")
DECISION_ID = UUID("00000000-0000-4000-8000-000000000603")
DELTA_ID = UUID("00000000-0000-4000-8000-000000000604")
INTERVENTION_ID = UUID("00000000-0000-4000-8000-000000000605")
MEMORY_ID = UUID("00000000-0000-4000-8000-000000000606")
CONFIGURATION_DIGEST = "a" * 64
GROUNDING_VERSION = "fixture-grounding/1"
GROUNDING_CONFIGURATION = {"max_claims": 2}
GROUNDING_CONFIGURATION_DIGEST = "9" * 64
REQUESTED_DELIVERY_TARGET = DeliveryTarget.NEXT_MODEL_CALL
BATCH_DIGEST = "b" * 64
MODEL_DIGEST = "c" * 64
NOW = datetime(2026, 7, 11, 12, 30, tzinfo=UTC)
CYCLE_ID = cycle_id(
    RUN_ID,
    1,
    1,
    "fixture/1",
    CONFIGURATION_DIGEST,
    GROUNDING_VERSION,
    GROUNDING_CONFIGURATION_DIGEST,
    REQUESTED_DELIVERY_TARGET,
)


def grounding_pin() -> GroundingPin:
    return GroundingPin(
        grounding_version=GROUNDING_VERSION,
        grounding_configuration=GROUNDING_CONFIGURATION,
        grounding_configuration_digest=GROUNDING_CONFIGURATION_DIGEST,
        requested_delivery_target=REQUESTED_DELIVERY_TARGET,
    )


def limits() -> BudgetLimits:
    return BudgetLimits(
        model_calls=10,
        input_tokens=1_000,
        output_tokens=1_000,
        canonical_token_equivalents=2_000,
        latency_us=1_000_000,
        max_call_latency_us=500_000,
        interventions=10,
        schema_repairs=2,
    )


def reservation() -> BudgetAmounts:
    return BudgetAmounts(
        model_calls=1,
        input_tokens=100,
        output_tokens=100,
        canonical_token_equivalents=200,
        latency_us=100_000,
    )


def decision(*, invoke: bool = True) -> InvocationDecision:
    return InvocationDecision(
        decision_id=DECISION_ID,
        run_id=RUN_ID,
        event_sequence=1,
        invoke=invoke,
        risk_score=0.8,
        reason_codes=(ReasonCode.TOOL_ERROR,),
        policy_version="fixture/1",
        configuration_digest=CONFIGURATION_DIGEST,
        budget_snapshot=BudgetSnapshot(
            limits=limits(),
            reserved=BudgetAmounts(),
            consumed=BudgetAmounts(),
        ),
        cooldown_active=False,
        created_at=NOW,
    )


def cycle(state: CycleState, *, revision: int | None = None) -> CycleRecord:
    values: dict[str, object] = {
        "cycle_id": CYCLE_ID,
        "run_id": RUN_ID,
        "revision": {
            CycleState.PENDING: 1,
            CycleState.RESERVED: 2,
            CycleState.RUNNING: 3,
            CycleState.COMMITTED: 4,
            CycleState.FAILED: 4,
        }[state]
        if revision is None
        else revision,
        "invocation_decision_id": DECISION_ID,
        "policy_version": "fixture/1",
        "configuration_digest": CONFIGURATION_DIGEST,
        "grounding_version": GROUNDING_VERSION,
        "grounding_configuration": GROUNDING_CONFIGURATION,
        "grounding_configuration_digest": GROUNDING_CONFIGURATION_DIGEST,
        "requested_delivery_target": REQUESTED_DELIVERY_TARGET,
        "first_event_sequence": 1,
        "last_event_sequence": 1,
        "state": state,
        "created_at": NOW,
        "updated_at": NOW + timedelta(seconds=2),
    }
    if state in (CycleState.RESERVED, CycleState.RUNNING, CycleState.COMMITTED):
        values["budget_reservation"] = reservation()
    if state in (CycleState.RUNNING, CycleState.COMMITTED):
        values["batch_digest"] = BATCH_DIGEST
    if state is CycleState.COMMITTED:
        delta = memory_delta()
        values.update(
            budget_settlement=reservation(),
            model_call_digests=(MODEL_DIGEST,),
            model_call_latencies_us=(reservation().latency_us,),
            validated_delta=delta,
            intervention=intervention(),
        )
    if state is CycleState.FAILED:
        values["failure_reason"] = ReasonCode.MODEL_TIMEOUT
    return CycleRecord.model_validate(values)


def tag(character: str) -> PayloadDigest:
    return PayloadDigest(
        algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
        value=character * 64,
    )


def receipt(record: CycleRecord) -> CycleReceipt:
    return CycleReceipt(
        appended=True,
        cycle=record,
        record_tag=tag("d"),
        ledger_position=record.revision,
        chain_tag=tag("e"),
        budget_snapshot=BudgetSnapshot(
            limits=limits(),
            reserved=(
                reservation()
                if record.state in (CycleState.RESERVED, CycleState.RUNNING)
                else BudgetAmounts()
            ),
            consumed=BudgetAmounts(),
        ),
    )


def memory_delta(*, run_id: UUID = RUN_ID) -> MemoryDelta:
    return MemoryDelta(
        delta_id=DELTA_ID,
        run_id=run_id,
        created_at=NOW + timedelta(seconds=3),
    )


def intervention(
    *,
    run_id: UUID = RUN_ID,
    identifier: str = CYCLE_ID,
) -> InterventionDecision:
    return InterventionDecision(
        intervention_id=INTERVENTION_ID,
        run_id=run_id,
        cycle_id=identifier,
        grounding_version=GROUNDING_VERSION,
        grounding_configuration=GROUNDING_CONFIGURATION,
        grounding_configuration_digest=GROUNDING_CONFIGURATION_DIGEST,
        grounding_receipt={"status": "fixture-verified"},
        action=InterventionAction.SILENCE,
        confidence=1.0,
        reason_code=ReasonCode.SILENCE_SELECTED,
        created_at=NOW + timedelta(seconds=3),
    )


class CommandRecorder:
    def __init__(self, response: CycleReceipt) -> None:
        self.response = response
        self.commands: list[object] = []

    async def begin_cycle(self, command: BeginCycle) -> CycleReceipt:
        self.commands.append(command)
        return self.response

    async def reserve_cycle(self, command: ReserveCycle) -> CycleReceipt:
        self.commands.append(command)
        return self.response

    async def mark_cycle_running(self, command: StartCycle) -> CycleReceipt:
        self.commands.append(command)
        return self.response

    async def commit_cycle(self, command: CommitCycle) -> CycleReceipt:
        self.commands.append(command)
        return self.response

    async def fail_cycle(self, command: FailCycle) -> CycleReceipt:
        self.commands.append(command)
        return self.response

    async def recover_cycles(
        self,
        run_id: UUID,
        *,
        recovered_at: datetime,
    ) -> CycleRecoveryReceipt:
        self.commands.append((run_id, recovered_at))
        return CycleRecoveryReceipt(
            run_id=run_id,
            resumable_pending=(),
            resumable_reserved=(),
            failed_unknown_cost=(),
        )


def test_begin_command_uses_an_exact_invoking_decision() -> None:
    command = CycleCommandFactory().begin(
        decision(),
        grounding=grounding_pin(),
        created_at=NOW + timedelta(seconds=1),
    )

    assert command == BeginCycle(
        run_id=RUN_ID,
        invocation_decision_id=DECISION_ID,
        grounding_version=GROUNDING_VERSION,
        grounding_configuration=GROUNDING_CONFIGURATION,
        grounding_configuration_digest=GROUNDING_CONFIGURATION_DIGEST,
        requested_delivery_target=REQUESTED_DELIVERY_TARGET,
        created_at=NOW + timedelta(seconds=1),
    )
    with pytest.raises(CycleCoordinatorStateError, match="invoking"):
        CycleCommandFactory().begin(
            decision(invoke=False),
            grounding=grounding_pin(),
            created_at=NOW,
        )


def test_transition_commands_derive_identity_and_revision_from_the_source() -> None:
    factory = CycleCommandFactory()
    pending = cycle(CycleState.PENDING)
    reserved = cycle(CycleState.RESERVED, revision=7)
    running = cycle(CycleState.RUNNING, revision=11)

    reserve = factory.reserve(
        receipt(pending),
        reservation=reservation(),
        updated_at=NOW + timedelta(seconds=3),
    )
    start = factory.start(
        reserved,
        batch_digest=BATCH_DIGEST,
        updated_at=NOW + timedelta(seconds=3),
    )
    commit = factory.commit(
        receipt(running),
        settlement=reservation(),
        validated_delta=memory_delta(),
        memory_id_assignments=(),
        intervention=intervention(),
        model_call_digests=(MODEL_DIGEST,),
        model_call_latencies_us=(reservation().latency_us,),
        updated_at=NOW + timedelta(seconds=4),
    )

    assert (reserve.run_id, reserve.cycle_id, reserve.expected_revision) == (
        RUN_ID,
        CYCLE_ID,
        1,
    )
    assert start.expected_revision == 7
    assert commit.expected_revision == 11


@pytest.mark.parametrize(
    ("method", "source", "kwargs"),
    [
        (
            "reserve",
            cycle(CycleState.RESERVED),
            {"reservation": reservation(), "updated_at": NOW + timedelta(seconds=3)},
        ),
        (
            "start",
            cycle(CycleState.PENDING),
            {"batch_digest": BATCH_DIGEST, "updated_at": NOW + timedelta(seconds=3)},
        ),
        (
            "commit",
            cycle(CycleState.RESERVED),
            {
                "settlement": reservation(),
                "validated_delta": memory_delta(),
                "memory_id_assignments": (),
                "intervention": intervention(),
                "updated_at": NOW + timedelta(seconds=3),
            },
        ),
    ],
)
def test_factory_rejects_commands_for_the_wrong_authoritative_state(
    method: str,
    source: CycleRecord,
    kwargs: dict[str, object],
) -> None:
    operation = getattr(CycleCommandFactory(), method)

    with pytest.raises(CycleCoordinatorStateError):
        operation(source, **kwargs)


def test_commit_validates_run_cycle_and_created_handle_identity() -> None:
    running = cycle(CycleState.RUNNING)
    factory = CycleCommandFactory()

    with pytest.raises(CycleCoordinatorIdentityError):
        factory.commit(
            running,
            settlement=reservation(),
            validated_delta=memory_delta(run_id=OTHER_RUN_ID),
            memory_id_assignments=(),
            intervention=intervention(),
            model_call_digests=(MODEL_DIGEST,),
            model_call_latencies_us=(reservation().latency_us,),
            updated_at=NOW + timedelta(seconds=4),
        )
    with pytest.raises(CycleCoordinatorInputError):
        factory.commit(
            running,
            settlement=reservation(),
            validated_delta=memory_delta(),
            memory_id_assignments=(),
            intervention=intervention().model_copy(
                update={"grounding_configuration_digest": "8" * 64}
            ),
            model_call_digests=(MODEL_DIGEST,),
            model_call_latencies_us=(reservation().latency_us,),
            updated_at=NOW + timedelta(seconds=4),
        )
    with pytest.raises(CycleCoordinatorIdentityError):
        factory.commit(
            running,
            settlement=reservation(),
            validated_delta=memory_delta(),
            memory_id_assignments=(),
            intervention=intervention(identifier="f" * 64),
            updated_at=NOW + timedelta(seconds=4),
        )
    with pytest.raises(CycleCoordinatorInputError):
        factory.commit(
            running,
            settlement=reservation(),
            validated_delta=memory_delta(),
            memory_id_assignments=(MemoryIdAssignment(handle="different", memory_id=MEMORY_ID),),
            intervention=intervention(),
            updated_at=NOW + timedelta(seconds=4),
        )

    early = NOW + timedelta(seconds=1)
    with pytest.raises(CycleCoordinatorInputError):
        factory.commit(
            running,
            settlement=reservation(),
            validated_delta=memory_delta().model_copy(update={"created_at": early}),
            memory_id_assignments=(),
            intervention=intervention(),
            model_call_digests=(MODEL_DIGEST,),
            model_call_latencies_us=(reservation().latency_us,),
            updated_at=NOW + timedelta(seconds=4),
        )
    with pytest.raises(CycleCoordinatorInputError):
        factory.commit(
            running,
            settlement=reservation(),
            validated_delta=memory_delta(),
            memory_id_assignments=(),
            intervention=intervention().model_copy(
                update={"created_at": NOW + timedelta(seconds=2, microseconds=500_000)}
            ),
            model_call_digests=(MODEL_DIGEST,),
            model_call_latencies_us=(reservation().latency_us,),
            updated_at=NOW + timedelta(seconds=4),
        )
    with pytest.raises(CycleCoordinatorInputError):
        factory.commit(
            running,
            settlement=reservation(),
            validated_delta=memory_delta(),
            memory_id_assignments=(),
            intervention=intervention().model_copy(update={"created_at": early}),
            model_call_digests=(MODEL_DIGEST,),
            model_call_latencies_us=(reservation().latency_us,),
            updated_at=NOW + timedelta(seconds=4),
        )


def test_fail_builds_only_state_appropriate_commands() -> None:
    factory = CycleCommandFactory()
    pending = factory.fail(
        cycle(CycleState.PENDING),
        reason=ReasonCode.MODEL_TIMEOUT,
        updated_at=NOW + timedelta(seconds=3),
    )
    reserved = factory.fail(
        cycle(CycleState.RESERVED),
        reason=ReasonCode.MODEL_TIMEOUT,
        settlement=BudgetAmounts(),
        updated_at=NOW + timedelta(seconds=3),
    )
    unknown = factory.fail(
        cycle(CycleState.RUNNING),
        reason=ReasonCode.FAILED_UNKNOWN_COST,
        updated_at=NOW + timedelta(seconds=3),
    )

    assert pending.expected_revision == 1
    assert pending.settlement is None
    assert reserved.expected_revision == 2
    assert unknown.settlement == reservation()
    with pytest.raises(CycleCoordinatorInputError):
        factory.fail(
            cycle(CycleState.RESERVED),
            reason=ReasonCode.MODEL_TIMEOUT,
            updated_at=NOW + timedelta(seconds=3),
        )
    with pytest.raises(CycleCoordinatorStateError):
        factory.fail(
            cycle(CycleState.PENDING),
            reason=ReasonCode.FAILED_UNKNOWN_COST,
            updated_at=NOW + timedelta(seconds=3),
        )


def test_unchecked_or_inexact_sources_fail_without_echoing_content() -> None:
    secret = "fixture-secret-must-not-echo"
    invalid = receipt(cycle(CycleState.PENDING)).model_copy(
        update={"cycle": cycle(CycleState.PENDING).model_copy(update={"revision": secret})}
    )

    with pytest.raises(CycleCoordinatorInputError) as error:
        CycleCommandFactory().reserve(
            invalid,
            reservation=reservation(),
            updated_at=NOW + timedelta(seconds=3),
        )
    assert secret not in str(error.value)
    unserializable = cycle(CycleState.PENDING).model_copy(update={"revision": object()})
    with pytest.raises(CycleCoordinatorInputError) as serialization_error:
        CycleCommandFactory().reserve(
            unserializable,
            reservation=reservation(),
            updated_at=NOW + timedelta(seconds=3),
        )
    assert serialization_error.value.__cause__ is None

    class CycleRecordSubclass(CycleRecord):
        pass

    subclass = CycleRecordSubclass.model_validate(cycle(CycleState.PENDING).model_dump())
    with pytest.raises(CycleCoordinatorInputError):
        CycleCommandFactory().reserve(
            subclass,
            reservation=reservation(),
            updated_at=NOW + timedelta(seconds=3),
        )
    nested_subclass = receipt(cycle(CycleState.PENDING)).model_copy(update={"cycle": subclass})
    with pytest.raises(CycleCoordinatorInputError):
        CycleCommandFactory().reserve(
            nested_subclass,
            reservation=reservation(),
            updated_at=NOW + timedelta(seconds=3),
        )


def test_factory_rejects_non_utc_and_backwards_timestamps() -> None:
    pending = cycle(CycleState.PENDING)

    with pytest.raises(CycleCoordinatorInputError):
        CycleCommandFactory().reserve(
            pending,
            reservation=reservation(),
            updated_at=NOW.replace(tzinfo=timezone(timedelta(hours=1))),
        )
    with pytest.raises(CycleCoordinatorInputError):
        CycleCommandFactory().reserve(
            pending,
            reservation=reservation(),
            updated_at=NOW,
        )


def test_factory_rejects_inexact_transition_arguments() -> None:
    factory = CycleCommandFactory()

    with pytest.raises(CycleCoordinatorInputError):
        factory.begin(
            decision(),
            grounding=grounding_pin().model_copy(
                update={"grounding_configuration_digest": "not-a-digest"}
            ),
            created_at=NOW,
        )
    with pytest.raises(CycleCoordinatorInputError):
        factory.begin(
            decision(),
            grounding=grounding_pin(),
            created_at=NOW - timedelta(seconds=1),
        )
    with pytest.raises(CycleCoordinatorInputError):
        factory.start(
            cycle(CycleState.RESERVED),
            batch_digest=cast(str, 1),
            updated_at=NOW + timedelta(seconds=3),
        )
    with pytest.raises(CycleCoordinatorInputError):
        factory.commit(
            cycle(CycleState.RUNNING),
            settlement=reservation(),
            validated_delta=memory_delta(),
            memory_id_assignments=cast(tuple[MemoryIdAssignment, ...], []),
            intervention=intervention(),
            updated_at=NOW + timedelta(seconds=4),
        )
    with pytest.raises(CycleCoordinatorInputError):
        factory.commit(
            cycle(CycleState.RUNNING),
            settlement=reservation(),
            validated_delta=memory_delta(),
            memory_id_assignments=(),
            intervention=intervention(),
            model_call_digests=cast(tuple[str, ...], [MODEL_DIGEST]),
            updated_at=NOW + timedelta(seconds=4),
        )
    with pytest.raises(CycleCoordinatorInputError):
        factory.commit(
            cycle(CycleState.RUNNING),
            settlement=reservation(),
            validated_delta=memory_delta(),
            memory_id_assignments=(),
            intervention=intervention(),
            model_call_digests=(MODEL_DIGEST,),
            model_call_latencies_us=cast(tuple[int, ...], [1]),
            updated_at=NOW + timedelta(seconds=4),
        )
    with pytest.raises(CycleCoordinatorInputError):
        factory.fail(
            cycle(CycleState.PENDING),
            reason=cast(ReasonCode, "not-a-reason"),
            updated_at=NOW + timedelta(seconds=3),
        )


def test_factory_rejects_inconsistent_failure_accounting() -> None:
    factory = CycleCommandFactory()

    with pytest.raises(CycleCoordinatorInputError):
        factory.fail(
            cycle(CycleState.RUNNING),
            reason=ReasonCode.MODEL_TIMEOUT,
            updated_at=NOW + timedelta(seconds=3),
        )
    with pytest.raises(CycleCoordinatorInputError):
        factory.fail(
            cycle(CycleState.PENDING),
            reason=ReasonCode.MODEL_TIMEOUT,
            settlement=BudgetAmounts(),
            updated_at=NOW + timedelta(seconds=3),
        )
    with pytest.raises(CycleCoordinatorInputError):
        factory.fail(
            cycle(CycleState.RUNNING),
            reason=ReasonCode.FAILED_UNKNOWN_COST,
            settlement=BudgetAmounts(model_calls=1),
            updated_at=NOW + timedelta(seconds=3),
        )


def test_repository_commands_reject_incomplete_call_receipts() -> None:
    valid_commit = CycleCommandFactory().commit(
        cycle(CycleState.RUNNING),
        settlement=reservation(),
        validated_delta=memory_delta(),
        memory_id_assignments=(),
        intervention=intervention(),
        model_call_digests=(MODEL_DIGEST,),
        model_call_latencies_us=(reservation().latency_us,),
        updated_at=NOW + timedelta(seconds=4),
    )
    commit_values = valid_commit.model_dump(mode="python")
    with pytest.raises(ValidationError, match="receipts must match"):
        CommitCycle.model_validate({**commit_values, "model_call_latencies_us": ()})
    with pytest.raises(ValidationError, match="exceeds settled"):
        CommitCycle.model_validate(
            {
                **commit_values,
                "settlement": reservation().model_copy(update={"latency_us": 0}),
            }
        )

    common = {
        "run_id": RUN_ID,
        "cycle_id": CYCLE_ID,
        "expected_revision": 3,
        "reason": ReasonCode.MODEL_TIMEOUT,
        "updated_at": NOW + timedelta(seconds=4),
    }
    with pytest.raises(ValidationError, match="equal length"):
        FailCycle(**common, settlement=reservation(), model_call_digests=(MODEL_DIGEST,))
    with pytest.raises(ValidationError, match="require a settlement"):
        FailCycle(
            **common,
            model_call_digests=(MODEL_DIGEST,),
            model_call_latencies_us=(0,),
        )
    with pytest.raises(ValidationError, match="exceeds settled"):
        FailCycle(
            **common,
            settlement=reservation().model_copy(update={"latency_us": 0}),
            model_call_digests=(MODEL_DIGEST,),
            model_call_latencies_us=(1,),
        )
    with pytest.raises(ValidationError, match="known model-call receipts exceed"):
        FailCycle(
            **{**common, "reason": ReasonCode.FAILED_UNKNOWN_COST},
            settlement=reservation(),
            model_call_digests=(MODEL_DIGEST, "d" * 64),
            model_call_latencies_us=(0, 0),
        )
    with pytest.raises(ValidationError, match="receipts must match"):
        FailCycle(**common, settlement=reservation())


async def test_coordinator_forwards_commands_without_caching_cycle_state() -> None:
    response = receipt(cycle(CycleState.RESERVED))
    recorder = CommandRecorder(response)
    coordinator = CycleCoordinator(cast(RunRepository, recorder))

    actual = await coordinator.reserve(
        receipt(cycle(CycleState.PENDING)),
        reservation=reservation(),
        updated_at=NOW + timedelta(seconds=3),
    )

    assert actual is response
    assert len(recorder.commands) == 1
    assert isinstance(recorder.commands[0], ReserveCycle)
    assert not hasattr(coordinator, "__dict__")


async def test_coordinator_forwards_every_lifecycle_operation() -> None:
    response = receipt(cycle(CycleState.PENDING))
    recorder = CommandRecorder(response)
    coordinator = CycleCoordinator(cast(RunRepository, recorder))

    assert (
        await coordinator.begin(
            decision(),
            grounding=grounding_pin(),
            created_at=NOW,
        )
        is response
    )
    assert (
        await coordinator.start(
            cycle(CycleState.RESERVED),
            batch_digest=BATCH_DIGEST,
            updated_at=NOW + timedelta(seconds=3),
        )
        is response
    )
    assert (
        await coordinator.commit(
            cycle(CycleState.RUNNING),
            settlement=reservation(),
            validated_delta=memory_delta(),
            memory_id_assignments=(),
            intervention=intervention(),
            model_call_digests=(MODEL_DIGEST,),
            model_call_latencies_us=(reservation().latency_us,),
            updated_at=NOW + timedelta(seconds=4),
        )
        is response
    )
    assert (
        await coordinator.fail(
            cycle(CycleState.RUNNING),
            reason=ReasonCode.MODEL_TIMEOUT,
            settlement=BudgetAmounts(model_calls=1),
            model_call_digests=(MODEL_DIGEST,),
            model_call_latencies_us=(0,),
            updated_at=NOW + timedelta(seconds=4),
        )
        is response
    )

    assert [type(command) for command in recorder.commands] == [
        BeginCycle,
        StartCycle,
        CommitCycle,
        FailCycle,
    ]


async def test_recovery_validates_and_copies_the_run_identity() -> None:
    response = receipt(cycle(CycleState.PENDING))
    recorder = CommandRecorder(response)
    coordinator = CycleCoordinator(cast(RunRepository, recorder))

    recovered = await coordinator.recover(
        RUN_ID,
        recovered_at=NOW + timedelta(seconds=5),
    )

    assert recovered.run_id == RUN_ID
    assert recorder.commands == [(RUN_ID, NOW + timedelta(seconds=5))]
    with pytest.raises(CycleCoordinatorInputError):
        await coordinator.recover(UUID(int=1), recovered_at=NOW)
