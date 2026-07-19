from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest
from pydantic import ValidationError
from tests.repository.conformance import (
    CYCLE_RUN_ID,
    advance_cycle_to_running,
    reminder_commit_command,
)

from saliencegate.domain import (
    DeliveryOutcome,
    DeliveryRecord,
    DeliveryState,
    DeliveryTarget,
    PayloadDigest,
    PayloadDigestAlgorithm,
    ReasonCode,
    canonical_digest,
)
from saliencegate.ports.adapters import (
    ADAPTER_CONTRACT_VERSION,
    AdapterCapabilities,
    AdapterDeliveryFailedError,
    DeduplicationGuarantee,
    DeliveryChannel,
    DeliveryEnvelope,
    DeliveryReceipt,
    DeliveryRole,
    InjectionMapping,
    adapter_capabilities_digest,
    enqueue_delivery_binding,
)
from saliencegate.ports.repository import (
    BeginDeliveryAttempt,
    ClaimDelivery,
    CompleteDelivery,
    CycleReceipt,
    DeliveryAttemptEnvelope,
    DeliveryAttemptReceipt,
    DeliveryRecoveryReceipt,
    DeliveryRevisionConflictError,
    DeliveryTransitionReceipt,
    EnqueueDelivery,
    MarkDeliveryUnknown,
    RejectDelivery,
)
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.repository.sqlite import SQLiteRunRepository
from saliencegate.runtime.delivery import (
    DeliveryRuntimeError,
    DeliveryWorker,
    DeliveryWorkerResult,
)
from saliencegate.security import InstallationKey

RUN_ID = UUID("00000000-0000-4000-8000-000000000701")
DELIVERY_ID = UUID("00000000-0000-4000-8000-000000000702")
INTERVENTION_ID = UUID("00000000-0000-4000-8000-000000000703")
CLAIM_1 = UUID("00000000-0000-4000-8000-000000000704")
ATTEMPT_1 = UUID("00000000-0000-4000-8000-000000000705")
CLAIM_2 = UUID("00000000-0000-4000-8000-000000000706")
ATTEMPT_2 = UUID("00000000-0000-4000-8000-000000000707")
FOREIGN_ATTEMPT = UUID("00000000-0000-4000-8000-000000000708")
OTHER_RUN_ID = UUID("00000000-0000-4000-8000-000000000709")
OTHER_INTERVENTION_ID = UUID("00000000-0000-4000-8000-000000000710")
CYCLE_ID = "a" * 64
TARGET_REQUEST_ID = "request-stable-1"
ADAPTER_ID = "fixture-adapter/1"
REMINDER = "[SALIENCEGATE_REMINDER fixed-ascii/v1]\nauthority=none\n[/SALIENCEGATE_REMINDER]"
NOW = datetime(2026, 7, 11, 15, 0, tzinfo=UTC)


def provider_mapping() -> InjectionMapping:
    return InjectionMapping(
        channel=DeliveryChannel.PROVIDER_DATA,
        role=DeliveryRole.DATA,
        provider_channel="context",
    )


def user_mapping() -> InjectionMapping:
    return InjectionMapping(
        channel=DeliveryChannel.EXISTING_USER_TASK,
        role=DeliveryRole.USER,
        provider_channel=None,
    )


def unsafe_mapping(role: DeliveryRole) -> InjectionMapping:
    return user_mapping().model_copy(update={"role": role})


def capabilities(
    *,
    deduplicates: bool = True,
    pre_action: bool = True,
    mappings: tuple[InjectionMapping, ...] | None = None,
) -> AdapterCapabilities:
    return AdapterCapabilities(
        schema_version="1.0",
        adapter_id=ADAPTER_ID,
        pre_action_interception=pre_action,
        deduplicates_delivery_id=deduplicates,
        deduplication_guarantee=(
            DeduplicationGuarantee.DURABLE_DELIVERY_ID
            if deduplicates
            else DeduplicationGuarantee.AT_MOST_ONCE_ATTEMPT
        ),
        injection_mappings=(provider_mapping(),) if mappings is None else mappings,
    )


def delivery_record(
    state: DeliveryState = DeliveryState.PENDING,
    *,
    target: DeliveryTarget = DeliveryTarget.NEXT_MODEL_CALL,
    deduplicates: bool = True,
    supports_pre_action: bool = True,
    mappings: tuple[InjectionMapping, ...] | None = None,
    revision: int | None = None,
    attempt_count: int | None = None,
) -> DeliveryRecord:
    default_revision = {
        DeliveryState.PENDING: 1,
        DeliveryState.CLAIMED: 2,
        DeliveryState.ATTEMPTING: 3,
        DeliveryState.DELIVERED: 4,
        DeliveryState.FAILED: 4,
        DeliveryState.UNKNOWN: 4,
        DeliveryState.REJECTED: 3,
    }[state]
    count = (
        0 if state in (DeliveryState.PENDING, DeliveryState.CLAIMED, DeliveryState.REJECTED) else 1
    )
    if attempt_count is not None:
        count = attempt_count
    pinned = capabilities(
        deduplicates=deduplicates,
        pre_action=supports_pre_action,
        mappings=mappings,
    )
    values: dict[str, object] = {
        "delivery_id": DELIVERY_ID,
        "run_id": RUN_ID,
        "revision": default_revision if revision is None else revision,
        "cycle_id": CYCLE_ID,
        "intervention_id": INTERVENTION_ID,
        "rendered_text_digest": canonical_digest(REMINDER),
        "target_request_id": TARGET_REQUEST_ID,
        "target": target,
        "state": state,
        "attempt_count": count,
        "adapter_id": ADAPTER_ID,
        "adapter_deduplicates": deduplicates,
        "adapter_deduplication_guarantee": pinned.deduplication_guarantee,
        "adapter_supports_pre_action": supports_pre_action,
        "adapter_contract_version": ADAPTER_CONTRACT_VERSION,
        "adapter_capabilities_digest": adapter_capabilities_digest(pinned),
        "claim_id": None if state is DeliveryState.PENDING else CLAIM_1,
        "attempt_id": (
            ATTEMPT_1
            if state
            in (
                DeliveryState.ATTEMPTING,
                DeliveryState.DELIVERED,
                DeliveryState.FAILED,
                DeliveryState.UNKNOWN,
            )
            else None
        ),
        "receipt": None,
        "outcome": None,
        "reason_code": None,
        "created_at": NOW,
        "updated_at": NOW,
    }
    if state is DeliveryState.DELIVERED:
        values.update(
            receipt={"provider_receipt_id": "provider-receipt-1"},
            outcome=DeliveryOutcome.DELIVERED,
            reason_code=ReasonCode.DELIVERY_SUCCEEDED,
        )
    elif state is DeliveryState.FAILED:
        values.update(
            outcome=DeliveryOutcome.FAILED,
            reason_code=ReasonCode.DELIVERY_FAILED,
        )
    elif state is DeliveryState.UNKNOWN:
        values.update(
            outcome=DeliveryOutcome.UNKNOWN,
            reason_code=ReasonCode.DELIVERY_UNKNOWN,
        )
    elif state is DeliveryState.REJECTED:
        values.update(
            outcome=DeliveryOutcome.REFUSED,
            reason_code=ReasonCode.TARGET_UNAVAILABLE,
        )
    return DeliveryRecord.model_validate(values)


def _transition(
    record: DeliveryRecord,
    *,
    state: DeliveryState,
    updated_at: datetime,
    **updates: object,
) -> DeliveryRecord:
    values = record.model_dump(mode="python", warnings=False)
    values.update(
        revision=record.revision + 1,
        state=state,
        updated_at=updated_at,
        **updates,
    )
    return DeliveryRecord.model_validate(values)


def tag(character: str) -> PayloadDigest:
    return PayloadDigest(
        algorithm=PayloadDigestAlgorithm.SYNTHETIC_SHA256,
        value=character * 64,
    )


class FakeOutbox:
    """Strict state-machine fake; adapter effects remain outside this boundary."""

    def __init__(self, current: DeliveryRecord, *, payload: str = REMINDER) -> None:
        self.current = current
        self.payload = payload
        self.commands: list[object] = []
        self.history: list[DeliveryRecord] = [current]

    def _receipt(
        self,
        *,
        appended: bool,
    ) -> DeliveryTransitionReceipt:
        return DeliveryTransitionReceipt(
            appended=appended,
            delivery=self.current,
            record_tag=tag("d"),
            ledger_position=len(self.history),
            chain_tag=tag("e"),
        )

    def _attempt_receipt(
        self,
        *,
        appended: bool,
        envelope: DeliveryAttemptEnvelope | None,
    ) -> DeliveryAttemptReceipt:
        return DeliveryAttemptReceipt(
            appended=appended,
            delivery=self.current,
            record_tag=tag("d"),
            ledger_position=len(self.history),
            chain_tag=tag("e"),
            envelope=envelope,
        )

    def _append(self, record: DeliveryRecord) -> None:
        self.current = record
        self.history.append(record)

    async def delivery(self, run_id: UUID, delivery_id: UUID) -> DeliveryRecord:
        assert run_id == self.current.run_id
        assert delivery_id == self.current.delivery_id
        return self.current

    async def claim_delivery(self, command: ClaimDelivery) -> DeliveryTransitionReceipt:
        self.commands.append(command)
        assert command.run_id == self.current.run_id
        assert command.delivery_id == self.current.delivery_id
        if self.current.state is DeliveryState.PENDING or (
            self.current.state is DeliveryState.UNKNOWN and self.current.adapter_deduplicates
        ):
            self._append(
                _transition(
                    self.current,
                    state=DeliveryState.CLAIMED,
                    updated_at=command.updated_at,
                    claim_id=command.claim_id,
                    attempt_id=None,
                    receipt=None,
                    outcome=None,
                    reason_code=None,
                )
            )
            return self._receipt(appended=True)
        return self._receipt(appended=False)

    async def begin_delivery_attempt(
        self,
        command: BeginDeliveryAttempt,
    ) -> DeliveryAttemptReceipt:
        self.commands.append(command)
        assert command.run_id == self.current.run_id
        assert command.delivery_id == self.current.delivery_id
        if (
            self.current.state is not DeliveryState.CLAIMED
            or command.expected_revision != self.current.revision
            or command.claim_id != self.current.claim_id
        ):
            return self._attempt_receipt(appended=False, envelope=None)
        self._append(
            _transition(
                self.current,
                state=DeliveryState.ATTEMPTING,
                updated_at=command.updated_at,
                attempt_count=self.current.attempt_count + 1,
                attempt_id=command.attempt_id,
            )
        )
        return self._attempt_receipt(
            appended=True,
            envelope=DeliveryAttemptEnvelope(
                delivery_id=self.current.delivery_id,
                run_id=self.current.run_id,
                cycle_id=self.current.cycle_id,
                intervention_id=self.current.intervention_id,
                rendered_text_digest=self.current.rendered_text_digest,
                target_request_id=self.current.target_request_id,
                target=self.current.target,
                claim_id=command.claim_id,
                attempt_id=command.attempt_id,
                attempt_number=self.current.attempt_count,
                adapter_id=self.current.adapter_id,
                adapter_deduplicates=self.current.adapter_deduplicates,
                adapter_deduplication_guarantee=(self.current.adapter_deduplication_guarantee),
                adapter_supports_pre_action=self.current.adapter_supports_pre_action,
                adapter_contract_version=self.current.adapter_contract_version,
                adapter_capabilities_digest=self.current.adapter_capabilities_digest,
                rendered_text=self.payload,
                ttl_steps=1,
            ),
        )

    async def complete_delivery(
        self,
        command: CompleteDelivery,
    ) -> DeliveryTransitionReceipt:
        self.commands.append(command)
        assert command.expected_revision == self.current.revision
        assert command.claim_id == self.current.claim_id
        assert command.attempt_id == self.current.attempt_id
        if command.outcome is DeliveryOutcome.DELIVERED:
            assert command.provider_receipt_id is not None
            self._append(
                _transition(
                    self.current,
                    state=DeliveryState.DELIVERED,
                    updated_at=command.updated_at,
                    receipt={"provider_receipt_id": command.provider_receipt_id},
                    outcome=DeliveryOutcome.DELIVERED,
                    reason_code=ReasonCode.DELIVERY_SUCCEEDED,
                )
            )
        else:
            assert command.outcome is DeliveryOutcome.FAILED
            assert command.provider_receipt_id is None
            self._append(
                _transition(
                    self.current,
                    state=DeliveryState.FAILED,
                    updated_at=command.updated_at,
                    receipt=None,
                    outcome=DeliveryOutcome.FAILED,
                    reason_code=ReasonCode.DELIVERY_FAILED,
                )
            )
        return self._receipt(appended=True)

    async def mark_delivery_unknown(
        self,
        command: MarkDeliveryUnknown,
    ) -> DeliveryTransitionReceipt:
        self.commands.append(command)
        assert command.expected_revision == self.current.revision
        assert command.claim_id == self.current.claim_id
        assert command.attempt_id == self.current.attempt_id
        self._append(
            _transition(
                self.current,
                state=DeliveryState.UNKNOWN,
                updated_at=command.updated_at,
                receipt=None,
                outcome=DeliveryOutcome.UNKNOWN,
                reason_code=ReasonCode.DELIVERY_UNKNOWN,
            )
        )
        return self._receipt(appended=True)

    async def reject_delivery(self, command: RejectDelivery) -> DeliveryTransitionReceipt:
        self.commands.append(command)
        assert command.expected_revision == self.current.revision
        assert command.claim_id == self.current.claim_id
        self._append(
            _transition(
                self.current,
                state=DeliveryState.REJECTED,
                updated_at=command.updated_at,
                receipt=None,
                outcome=DeliveryOutcome.REFUSED,
                reason_code=command.reason_code,
            )
        )
        return self._receipt(appended=True)

    async def recover_deliveries(
        self,
        run_id: UUID,
        *,
        recovered_at: datetime,
    ) -> DeliveryRecoveryReceipt:
        assert run_id == self.current.run_id
        marked_unknown: tuple[DeliveryTransitionReceipt, ...] = ()
        if self.current.state is DeliveryState.ATTEMPTING:
            self._append(
                _transition(
                    self.current,
                    state=DeliveryState.UNKNOWN,
                    updated_at=recovered_at,
                    receipt=None,
                    outcome=DeliveryOutcome.UNKNOWN,
                    reason_code=ReasonCode.DELIVERY_UNKNOWN,
                )
            )
            marked_unknown = (self._receipt(appended=True),)
        return DeliveryRecoveryReceipt(
            run_id=run_id,
            marked_unknown=marked_unknown,
            resumable_pending=(
                (self.current,) if self.current.state is DeliveryState.PENDING else ()
            ),
            resumable_claimed=(
                (self.current,) if self.current.state is DeliveryState.CLAIMED else ()
            ),
            retryable_unknown=(
                (self.current,)
                if self.current.state is DeliveryState.UNKNOWN and self.current.adapter_deduplicates
                else ()
            ),
            non_retryable_unknown=(
                (self.current,)
                if self.current.state is DeliveryState.UNKNOWN
                and not self.current.adapter_deduplicates
                else ()
            ),
        )


class AdapterMode(StrEnum):
    SUCCESS = "success"
    KNOWN_FAILURE = "known_failure"
    UNKNOWN_AFTER_EFFECT = "unknown_after_effect"
    INVALID_RECEIPT = "invalid_receipt"


class FakeAdapter:
    def __init__(
        self,
        declared: AdapterCapabilities,
        *,
        mode: AdapterMode = AdapterMode.SUCCESS,
        prior_effects: set[UUID] | None = None,
    ) -> None:
        self.declared = declared
        self.mode = mode
        self.calls: list[DeliveryEnvelope] = []
        self.effects = set() if prior_effects is None else prior_effects
        self.deferred_targets: list[str] = []

    def capabilities(self) -> AdapterCapabilities:
        return self.declared

    async def deliver(self, envelope: DeliveryEnvelope) -> DeliveryReceipt:
        self.calls.append(envelope)
        if self.mode is AdapterMode.KNOWN_FAILURE:
            raise AdapterDeliveryFailedError()
        self.effects.add(envelope.delivery_id)
        if envelope.target is DeliveryTarget.PRE_ACTION_REPLAN:
            self.deferred_targets.append(envelope.target_request_id)
        if self.mode is AdapterMode.UNKNOWN_AFTER_EFFECT:
            raise RuntimeError("provider leaked sk-fixture-secret after side effect")
        attempt_id = (
            FOREIGN_ATTEMPT if self.mode is AdapterMode.INVALID_RECEIPT else envelope.attempt_id
        )
        return DeliveryReceipt(
            schema_version="1.0",
            delivery_id=envelope.delivery_id,
            target_request_id=envelope.target_request_id,
            adapter_id=self.declared.adapter_id,
            attempt_id=attempt_id,
            attempt_number=envelope.attempt_number,
            provider_receipt_id="provider-receipt-1",
            delivered_at=envelope.created_at + timedelta(milliseconds=1),
        )


def identity_factory(*identities: UUID) -> Callable[[], UUID]:
    remaining = iter(identities)
    return lambda: next(remaining)


def worker(
    outbox: FakeOutbox,
    adapter: FakeAdapter,
    *identities: UUID,
) -> DeliveryWorker:
    return DeliveryWorker(
        repository=outbox,
        adapter=adapter,
        id_factory=identity_factory(*identities),
    )


async def test_success_is_durable_before_the_adapter_side_effect() -> None:
    outbox = FakeOutbox(delivery_record())
    adapter = FakeAdapter(capabilities())

    result = await worker(outbox, adapter, CLAIM_1, ATTEMPT_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.DELIVERED
    assert result.delivery.outcome is DeliveryOutcome.DELIVERED
    assert result.delivered is True
    assert result.guarded is False
    assert result.reason_code is ReasonCode.DELIVERY_SUCCEEDED
    assert tuple(record.state for record in outbox.history) == (
        DeliveryState.PENDING,
        DeliveryState.CLAIMED,
        DeliveryState.ATTEMPTING,
        DeliveryState.DELIVERED,
    )
    assert len(adapter.calls) == 1
    envelope = adapter.calls[0]
    assert envelope.delivery_id == DELIVERY_ID
    assert envelope.target_request_id == TARGET_REQUEST_ID
    assert envelope.claim_id == CLAIM_1
    assert envelope.attempt_id == ATTEMPT_1
    assert envelope.attempt_number == 1
    assert outbox.history[-2].state is DeliveryState.ATTEMPTING


async def test_unknown_retry_uses_new_tokens_but_the_same_deduplication_key() -> None:
    prior_effects = {DELIVERY_ID}
    outbox = FakeOutbox(delivery_record(DeliveryState.UNKNOWN, deduplicates=True))
    adapter = FakeAdapter(capabilities(deduplicates=True), prior_effects=prior_effects)

    result = await worker(outbox, adapter, CLAIM_2, ATTEMPT_2).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=2),
    )

    assert result.delivery.state is DeliveryState.DELIVERED
    assert result.delivery.attempt_count == 2
    assert len(adapter.calls) == 1
    retry = adapter.calls[0]
    assert retry.delivery_id == DELIVERY_ID
    assert retry.target_request_id == TARGET_REQUEST_ID
    assert retry.claim_id == CLAIM_2
    assert retry.attempt_id == ATTEMPT_2
    assert retry.attempt_number == 2
    assert adapter.effects == {DELIVERY_ID}


async def test_non_deduplicating_crash_records_one_unknown_attempt_and_never_retries() -> None:
    outbox = FakeOutbox(delivery_record(deduplicates=False))
    adapter = FakeAdapter(
        capabilities(deduplicates=False),
        mode=AdapterMode.UNKNOWN_AFTER_EFFECT,
    )
    delivery_worker = worker(outbox, adapter, CLAIM_1, ATTEMPT_1)

    first = await delivery_worker.deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )
    second = await delivery_worker.deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=2),
    )

    assert first.delivery.state is DeliveryState.UNKNOWN
    assert second.delivery == first.delivery
    assert first.delivered is second.delivered is False
    assert first.guarded is second.guarded is False
    assert first.reason_code is second.reason_code is ReasonCode.DELIVERY_UNKNOWN
    assert first.delivery.attempt_count == 1
    assert len(adapter.calls) == 1
    assert adapter.effects == {DELIVERY_ID}


async def test_known_adapter_failure_is_failed_not_unknown_or_delivered() -> None:
    outbox = FakeOutbox(delivery_record())
    adapter = FakeAdapter(capabilities(), mode=AdapterMode.KNOWN_FAILURE)

    result = await worker(outbox, adapter, CLAIM_1, ATTEMPT_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.FAILED
    assert result.delivery.outcome is DeliveryOutcome.FAILED
    assert result.delivery.receipt is None
    assert result.delivered is False
    assert result.guarded is False
    assert result.reason_code is ReasonCode.DELIVERY_FAILED
    assert len(adapter.calls) == 1


@pytest.mark.parametrize(
    ("mappings", "reason"),
    [
        ((unsafe_mapping(DeliveryRole.SYSTEM),), ReasonCode.UNSAFE_ROLE_MAPPING),
        ((unsafe_mapping(DeliveryRole.DEVELOPER),), ReasonCode.UNSAFE_ROLE_MAPPING),
        ((), ReasonCode.UNSUPPORTED_DELIVERY_CHANNEL),
    ],
)
async def test_unsafe_role_or_missing_channel_is_rejected_before_side_effect(
    mappings: tuple[InjectionMapping, ...],
    reason: ReasonCode,
) -> None:
    outbox = FakeOutbox(delivery_record())
    declared = capabilities(mappings=())
    if mappings:
        declared = declared.model_copy(update={"injection_mappings": mappings})
    adapter = FakeAdapter(declared)

    result = await worker(outbox, adapter, CLAIM_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.REJECTED
    assert result.delivery.outcome is DeliveryOutcome.REFUSED
    assert result.delivery.attempt_count == 0
    assert result.delivered is False
    assert result.guarded is False
    assert result.reason_code is reason
    assert adapter.calls == []
    assert DeliveryState.ATTEMPTING not in {record.state for record in outbox.history}


async def test_pre_action_without_interception_is_rejected_before_deferral() -> None:
    outbox = FakeOutbox(
        delivery_record(
            target=DeliveryTarget.PRE_ACTION_REPLAN,
            supports_pre_action=False,
        ),
    )
    adapter = FakeAdapter(capabilities(pre_action=False))

    result = await worker(outbox, adapter, CLAIM_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.REJECTED
    assert result.reason_code is ReasonCode.TARGET_UNAVAILABLE
    assert result.delivered is False
    assert result.guarded is False
    assert adapter.calls == []
    assert adapter.deferred_targets == []


async def test_successful_pre_action_delivery_is_the_only_guarded_result() -> None:
    outbox = FakeOutbox(
        delivery_record(target=DeliveryTarget.PRE_ACTION_REPLAN),
    )
    adapter = FakeAdapter(capabilities(pre_action=True))

    result = await worker(outbox, adapter, CLAIM_1, ATTEMPT_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.DELIVERED
    assert result.delivered is True
    assert result.guarded is True
    assert adapter.deferred_targets == [TARGET_REQUEST_ID]


async def test_restart_recovers_attempting_and_retries_only_deduplicating_delivery() -> None:
    deduplicating_outbox = FakeOutbox(
        delivery_record(DeliveryState.ATTEMPTING, deduplicates=True),
    )
    deduplicating_adapter = FakeAdapter(
        capabilities(deduplicates=True),
        prior_effects={DELIVERY_ID},
    )
    delivery_worker = worker(
        deduplicating_outbox,
        deduplicating_adapter,
        CLAIM_2,
        ATTEMPT_2,
    )

    resumed = await delivery_worker.recover(
        RUN_ID,
        recovered_at=NOW + timedelta(seconds=2),
    )

    assert len(resumed) == 1
    assert resumed[0].delivery.state is DeliveryState.DELIVERED
    assert resumed[0].delivery.attempt_count == 2
    assert deduplicating_adapter.effects == {DELIVERY_ID}
    assert deduplicating_adapter.calls[0].target_request_id == TARGET_REQUEST_ID

    non_deduplicating_outbox = FakeOutbox(
        delivery_record(DeliveryState.ATTEMPTING, deduplicates=False),
    )
    non_deduplicating_adapter = FakeAdapter(capabilities(deduplicates=False))
    unresolved = await worker(non_deduplicating_outbox, non_deduplicating_adapter).recover(
        RUN_ID,
        recovered_at=NOW + timedelta(seconds=2),
    )

    assert len(unresolved) == 1
    assert unresolved[0].delivery.state is DeliveryState.UNKNOWN
    assert unresolved[0].delivered is False
    assert unresolved[0].guarded is False
    assert non_deduplicating_adapter.calls == []


class StaleBeginOutbox(FakeOutbox):
    async def begin_delivery_attempt(
        self,
        command: BeginDeliveryAttempt,
    ) -> DeliveryAttemptReceipt:
        self.commands.append(command)
        self._append(
            _transition(
                self.current,
                state=DeliveryState.UNKNOWN,
                updated_at=command.updated_at,
                attempt_count=self.current.attempt_count + 1,
                attempt_id=command.attempt_id,
                outcome=DeliveryOutcome.UNKNOWN,
                reason_code=ReasonCode.DELIVERY_UNKNOWN,
            )
        )
        return self._attempt_receipt(appended=False, envelope=None)


async def test_stale_begin_receipt_never_authorizes_an_adapter_call() -> None:
    outbox = StaleBeginOutbox(delivery_record())
    adapter = FakeAdapter(capabilities())

    result = await worker(outbox, adapter, CLAIM_1, ATTEMPT_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.UNKNOWN
    assert result.delivered is False
    assert result.guarded is False
    assert adapter.calls == []


async def test_invalid_adapter_receipt_is_unknown_because_the_side_effect_is_ambiguous() -> None:
    outbox = FakeOutbox(delivery_record())
    adapter = FakeAdapter(capabilities(), mode=AdapterMode.INVALID_RECEIPT)

    result = await worker(outbox, adapter, CLAIM_1, ATTEMPT_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.UNKNOWN
    assert result.delivery.receipt is None
    assert result.reason_code is ReasonCode.DELIVERY_UNKNOWN
    assert result.delivered is False
    assert result.guarded is False
    assert len(adapter.calls) == 1


def _explode_receipt_serialization(*_args: object, **_kwargs: object) -> str:
    raise RuntimeError("hostile-receipt-serialization-secret")


class ExplodingReceiptAdapter(FakeAdapter):
    async def deliver(self, envelope: DeliveryEnvelope) -> DeliveryReceipt:
        receipt = await super().deliver(envelope)
        object.__setattr__(receipt, "model_dump_json", _explode_receipt_serialization)
        return receipt


async def test_hostile_receipt_exception_after_effect_is_sanitized_to_unknown() -> None:
    outbox = FakeOutbox(delivery_record())
    adapter = ExplodingReceiptAdapter(capabilities())

    result = await worker(outbox, adapter, CLAIM_1, ATTEMPT_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert adapter.effects == {DELIVERY_ID}
    assert result.delivery.state is DeliveryState.UNKNOWN
    assert result.reason_code is ReasonCode.DELIVERY_UNKNOWN
    assert "hostile-receipt-serialization-secret" not in repr(result)
    assert len(adapter.calls) == 1


async def test_provider_exception_text_never_enters_records_results_or_errors() -> None:
    secret = "sk-fixture-secret"
    outbox = FakeOutbox(delivery_record())
    adapter = FakeAdapter(
        capabilities(),
        mode=AdapterMode.UNKNOWN_AFTER_EFFECT,
    )

    result = await worker(outbox, adapter, CLAIM_1, ATTEMPT_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.UNKNOWN
    assert secret not in result.delivery.model_dump_json(warnings=False)
    assert secret not in repr(result)
    assert secret not in " ".join(repr(command) for command in outbox.commands)
    assert all(record.receipt is None for record in outbox.history)


def test_worker_result_cannot_claim_failed_or_unknown_delivery_was_guarded() -> None:
    for state, reason in (
        (DeliveryState.FAILED, ReasonCode.DELIVERY_FAILED),
        (DeliveryState.UNKNOWN, ReasonCode.DELIVERY_UNKNOWN),
    ):
        result = DeliveryWorkerResult.from_delivery(delivery_record(state), reason_code=reason)
        assert result.delivered is False
        assert result.guarded is False


async def test_existing_user_task_fallback_remains_bound_to_the_original_target() -> None:
    declared = capabilities(mappings=(user_mapping(),))
    outbox = FakeOutbox(delivery_record(mappings=(user_mapping(),)))
    adapter = FakeAdapter(declared)

    result = await worker(outbox, adapter, CLAIM_1, ATTEMPT_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.DELIVERED
    assert adapter.calls[0].target_request_id == TARGET_REQUEST_ID
    assert adapter.calls[0].mapping.channel is DeliveryChannel.EXISTING_USER_TASK
    assert adapter.calls[0].mapping.role is DeliveryRole.USER


class ForgedAttemptReceiptOutbox(FakeOutbox):
    def __init__(
        self,
        current: DeliveryRecord,
        *,
        delivery_changes: dict[str, object] | None = None,
        envelope_changes: dict[str, object] | None = None,
    ) -> None:
        super().__init__(current)
        self.delivery_changes = {} if delivery_changes is None else delivery_changes
        self.envelope_changes = {} if envelope_changes is None else envelope_changes

    async def begin_delivery_attempt(
        self,
        command: BeginDeliveryAttempt,
    ) -> DeliveryAttemptReceipt:
        receipt = await super().begin_delivery_attempt(command)
        assert receipt.envelope is not None
        return receipt.model_copy(
            update={
                "delivery": receipt.delivery.model_copy(update=self.delivery_changes),
                "envelope": receipt.envelope.model_copy(update=self.envelope_changes),
            }
        )


@pytest.mark.parametrize(
    ("field_name", "foreign_value"),
    (
        ("target_request_id", "foreign-request"),
        ("target", DeliveryTarget.PRE_ACTION_REPLAN),
        ("attempt_id", FOREIGN_ATTEMPT),
        ("rendered_text", "forged reminder text"),
    ),
)
async def test_forged_attempt_envelope_is_revalidated_before_adapter_authorization(
    field_name: str,
    foreign_value: object,
) -> None:
    outbox = ForgedAttemptReceiptOutbox(
        delivery_record(),
        envelope_changes={field_name: foreign_value},
    )
    adapter = FakeAdapter(capabilities())

    result = await worker(outbox, adapter, CLAIM_1, ATTEMPT_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery == outbox.current
    assert result.delivery.state is DeliveryState.ATTEMPTING
    assert result.reason_code is ReasonCode.DELIVERY_UNKNOWN
    assert adapter.calls == []


@pytest.mark.parametrize(
    ("delivery_changes", "envelope_changes"),
    (
        ({"attempt_id": FOREIGN_ATTEMPT}, {"attempt_id": FOREIGN_ATTEMPT}),
        (
            {"target_request_id": "foreign-request"},
            {"target_request_id": "foreign-request"},
        ),
        (
            {"target": DeliveryTarget.PRE_ACTION_REPLAN},
            {"target": DeliveryTarget.PRE_ACTION_REPLAN},
        ),
        ({"run_id": OTHER_RUN_ID}, {"run_id": OTHER_RUN_ID}),
        ({"cycle_id": "b" * 64}, {"cycle_id": "b" * 64}),
        (
            {"intervention_id": OTHER_INTERVENTION_ID},
            {"intervention_id": OTHER_INTERVENTION_ID},
        ),
        ({"adapter_id": "foreign-adapter/1"}, {"adapter_id": "foreign-adapter/1"}),
        (
            {
                "adapter_deduplicates": False,
                "adapter_deduplication_guarantee": (DeduplicationGuarantee.AT_MOST_ONCE_ATTEMPT),
            },
            {
                "adapter_deduplicates": False,
                "adapter_deduplication_guarantee": (DeduplicationGuarantee.AT_MOST_ONCE_ATTEMPT),
            },
        ),
        (
            {"adapter_supports_pre_action": False},
            {"adapter_supports_pre_action": False},
        ),
        (
            {"adapter_contract_version": "adapter-contract/v2"},
            {"adapter_contract_version": "adapter-contract/v2"},
        ),
        (
            {"adapter_capabilities_digest": "f" * 64},
            {"adapter_capabilities_digest": "f" * 64},
        ),
        (
            {"rendered_text_digest": canonical_digest("forged reminder text")},
            {
                "rendered_text": "forged reminder text",
                "rendered_text_digest": canonical_digest("forged reminder text"),
            },
        ),
        ({"created_at": NOW - timedelta(seconds=1)}, {}),
    ),
    ids=(
        "attempt",
        "target-request",
        "target",
        "run",
        "cycle",
        "intervention",
        "adapter",
        "deduplication",
        "pre-action",
        "contract",
        "capabilities-digest",
        "rendered-text-digest",
        "creation-time",
    ),
)
async def test_internally_coherent_stale_attempt_receipt_is_bound_to_current_command(
    delivery_changes: dict[str, object],
    envelope_changes: dict[str, object],
) -> None:
    outbox = ForgedAttemptReceiptOutbox(
        delivery_record(),
        delivery_changes=delivery_changes,
        envelope_changes=envelope_changes,
    )
    adapter = FakeAdapter(capabilities())

    result = await worker(outbox, adapter, CLAIM_1, ATTEMPT_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery == outbox.current
    assert result.delivery.state is DeliveryState.ATTEMPTING
    assert result.reason_code is ReasonCode.DELIVERY_UNKNOWN
    assert adapter.calls == []


async def test_capability_mapping_drift_is_rejected_before_claim_or_side_effect() -> None:
    outbox = FakeOutbox(delivery_record())
    adapter = FakeAdapter(capabilities(mappings=(user_mapping(),)))

    result = await worker(outbox, adapter).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.REJECTED
    assert result.reason_code is ReasonCode.TARGET_UNAVAILABLE
    assert result.delivery.attempt_count == 0
    assert adapter.calls == []


@pytest.mark.parametrize(
    "mismatch",
    ("adapter-id", "deduplication", "pre-action", "contract-version"),
)
async def test_pinned_adapter_contract_mismatch_is_rejected(mismatch: str) -> None:
    current = delivery_record()
    declared = capabilities()
    if mismatch == "adapter-id":
        declared = declared.model_copy(update={"adapter_id": "foreign-adapter/1"})
    elif mismatch == "deduplication":
        declared = capabilities(deduplicates=False)
    elif mismatch == "pre-action":
        declared = capabilities(pre_action=False)
    else:
        current = current.model_copy(update={"adapter_contract_version": "adapter-contract/v2"})
    outbox = FakeOutbox(current)
    adapter = FakeAdapter(declared)

    result = await worker(outbox, adapter).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.REJECTED
    assert result.reason_code is ReasonCode.TARGET_UNAVAILABLE
    assert adapter.calls == []


@pytest.mark.parametrize(
    ("state", "reason"),
    (
        (DeliveryState.DELIVERED, ReasonCode.DELIVERY_SUCCEEDED),
        (DeliveryState.FAILED, ReasonCode.DELIVERY_FAILED),
        (DeliveryState.REJECTED, ReasonCode.TARGET_UNAVAILABLE),
    ),
)
async def test_terminal_delivery_is_a_side_effect_free_fast_path(
    state: DeliveryState,
    reason: ReasonCode,
) -> None:
    outbox = FakeOutbox(delivery_record(state))
    adapter = FakeAdapter(capabilities())

    result = await worker(outbox, adapter).deliver(RUN_ID, DELIVERY_ID, now=NOW)

    assert result.delivery == outbox.current
    assert result.reason_code is reason
    assert adapter.calls == []
    assert outbox.commands == []


async def test_attempting_delivery_is_not_reauthorized_by_a_second_worker() -> None:
    outbox = FakeOutbox(delivery_record(DeliveryState.ATTEMPTING))
    adapter = FakeAdapter(capabilities())

    result = await worker(outbox, adapter).deliver(RUN_ID, DELIVERY_ID, now=NOW)

    assert result.delivery.state is DeliveryState.ATTEMPTING
    assert result.reason_code is ReasonCode.DELIVERY_UNKNOWN
    assert result.delivered is False
    assert result.guarded is False
    assert adapter.calls == []
    assert outbox.commands == []


@pytest.mark.parametrize(
    ("run_id", "delivery_id", "now"),
    (
        (UUID(int=1), DELIVERY_ID, NOW),
        (RUN_ID, UUID(int=2), NOW),
        (RUN_ID, DELIVERY_ID, NOW.replace(tzinfo=None)),
    ),
    ids=("run-id", "delivery-id", "timestamp"),
)
async def test_worker_rejects_invalid_public_identifiers_and_time(
    run_id: UUID,
    delivery_id: UUID,
    now: datetime,
) -> None:
    outbox = FakeOutbox(delivery_record())

    with pytest.raises(DeliveryRuntimeError):
        await worker(outbox, FakeAdapter(capabilities())).deliver(
            run_id,
            delivery_id,
            now=now,
        )

    assert outbox.commands == []


@pytest.mark.parametrize(
    ("identities", "expected_state"),
    (
        ((), DeliveryState.PENDING),
        ((CLAIM_1,), DeliveryState.CLAIMED),
        ((UUID(int=3),), DeliveryState.PENDING),
        ((CLAIM_1, UUID(int=4)), DeliveryState.CLAIMED),
    ),
    ids=("missing-claim", "missing-attempt", "invalid-claim", "invalid-attempt"),
)
async def test_invalid_identity_factory_never_authorizes_an_attempt(
    identities: tuple[UUID, ...],
    expected_state: DeliveryState,
) -> None:
    outbox = FakeOutbox(delivery_record())
    delivery_worker = worker(outbox, FakeAdapter(capabilities()), *identities)

    with pytest.raises(DeliveryRuntimeError):
        await delivery_worker.deliver(
            RUN_ID,
            DELIVERY_ID,
            now=NOW + timedelta(seconds=1),
        )

    assert outbox.current.state is expected_state
    assert DeliveryState.ATTEMPTING not in {record.state for record in outbox.history}


class ClaimConflictOutbox(FakeOutbox):
    async def claim_delivery(self, command: ClaimDelivery) -> DeliveryTransitionReceipt:
        self.commands.append(command)
        self._append(
            _transition(
                self.current,
                state=DeliveryState.CLAIMED,
                updated_at=command.updated_at,
                claim_id=CLAIM_2,
                attempt_id=None,
                receipt=None,
                outcome=None,
                reason_code=None,
            )
        )
        raise DeliveryRevisionConflictError(command.expected_revision, self.current.revision)


class BeginConflictOutbox(FakeOutbox):
    async def begin_delivery_attempt(
        self,
        command: BeginDeliveryAttempt,
    ) -> DeliveryAttemptReceipt:
        self.commands.append(command)
        self._append(
            _transition(
                self.current,
                state=DeliveryState.ATTEMPTING,
                updated_at=command.updated_at,
                attempt_count=self.current.attempt_count + 1,
                attempt_id=FOREIGN_ATTEMPT,
            )
        )
        raise DeliveryRevisionConflictError(command.expected_revision, self.current.revision)


class RejectConflictOutbox(FakeOutbox):
    async def reject_delivery(self, command: RejectDelivery) -> DeliveryTransitionReceipt:
        self.commands.append(command)
        self._append(
            _transition(
                self.current,
                state=DeliveryState.REJECTED,
                updated_at=command.updated_at,
                receipt=None,
                outcome=DeliveryOutcome.REFUSED,
                reason_code=ReasonCode.TARGET_UNAVAILABLE,
            )
        )
        raise DeliveryRevisionConflictError(command.expected_revision, self.current.revision)


async def test_claim_revision_conflict_returns_the_authoritative_owner() -> None:
    outbox = ClaimConflictOutbox(delivery_record())
    adapter = FakeAdapter(capabilities())

    result = await worker(outbox, adapter, CLAIM_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.CLAIMED
    assert result.delivery.claim_id == CLAIM_2
    assert adapter.calls == []


async def test_begin_revision_conflict_never_authorizes_the_competing_attempt() -> None:
    outbox = BeginConflictOutbox(delivery_record(DeliveryState.CLAIMED))
    adapter = FakeAdapter(capabilities())

    result = await worker(outbox, adapter, ATTEMPT_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.ATTEMPTING
    assert result.delivery.attempt_id == FOREIGN_ATTEMPT
    assert result.reason_code is ReasonCode.DELIVERY_UNKNOWN
    assert adapter.calls == []


async def test_reject_revision_conflict_uses_the_authoritative_refusal_reason() -> None:
    outbox = RejectConflictOutbox(delivery_record())
    adapter = FakeAdapter(capabilities(mappings=()))

    result = await worker(outbox, adapter).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.REJECTED
    assert result.reason_code is ReasonCode.TARGET_UNAVAILABLE
    assert adapter.calls == []


class LateCompletionOutbox(FakeOutbox):
    def __init__(self, current: DeliveryRecord) -> None:
        super().__init__(current)
        self.completion_calls = 0

    async def complete_delivery(
        self,
        command: CompleteDelivery,
    ) -> DeliveryTransitionReceipt:
        self.completion_calls += 1
        if self.completion_calls == 1:
            self.commands.append(command)
            self._append(
                _transition(
                    self.current,
                    state=DeliveryState.UNKNOWN,
                    updated_at=command.updated_at,
                    receipt=None,
                    outcome=DeliveryOutcome.UNKNOWN,
                    reason_code=ReasonCode.DELIVERY_UNKNOWN,
                )
            )
            raise DeliveryRevisionConflictError(
                command.expected_revision,
                self.current.revision,
            )
        return await super().complete_delivery(command)


class ContendedCompletionOutbox(FakeOutbox):
    def __init__(self, current: DeliveryRecord) -> None:
        super().__init__(current)
        self.completion_calls = 0

    async def complete_delivery(
        self,
        command: CompleteDelivery,
    ) -> DeliveryTransitionReceipt:
        self.completion_calls += 1
        values = self.current.model_dump(mode="python", warnings=False)
        values.update(
            revision=self.current.revision + 3,
            state=DeliveryState.ATTEMPTING,
            attempt_count=self.current.attempt_count + 1,
            claim_id=CLAIM_2,
            attempt_id=ATTEMPT_2,
            receipt=None,
            outcome=None,
            reason_code=None,
            updated_at=command.updated_at,
        )
        self._append(DeliveryRecord.model_validate(values))
        raise DeliveryRevisionConflictError(command.expected_revision, self.current.revision)


class UnknownConflictOutbox(FakeOutbox):
    async def mark_delivery_unknown(
        self,
        command: MarkDeliveryUnknown,
    ) -> DeliveryTransitionReceipt:
        self.commands.append(command)
        self._append(
            _transition(
                self.current,
                state=DeliveryState.DELIVERED,
                updated_at=command.updated_at,
                receipt={"provider_receipt_id": "concurrent-receipt"},
                outcome=DeliveryOutcome.DELIVERED,
                reason_code=ReasonCode.DELIVERY_SUCCEEDED,
            )
        )
        raise DeliveryRevisionConflictError(command.expected_revision, self.current.revision)


async def test_late_success_completes_unknown_with_the_same_attempt_ownership() -> None:
    outbox = LateCompletionOutbox(delivery_record())
    adapter = FakeAdapter(capabilities())

    result = await worker(outbox, adapter, CLAIM_1, ATTEMPT_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.DELIVERED
    assert result.delivery.revision == 5
    assert result.delivery.attempt_id == ATTEMPT_1
    assert outbox.completion_calls == 2
    assert len(adapter.calls) == 1


async def test_late_success_does_not_complete_a_competing_attempt() -> None:
    outbox = ContendedCompletionOutbox(delivery_record())
    adapter = FakeAdapter(capabilities())

    result = await worker(outbox, adapter, CLAIM_1, ATTEMPT_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.ATTEMPTING
    assert result.delivery.claim_id == CLAIM_2
    assert result.delivery.attempt_id == ATTEMPT_2
    assert result.reason_code is ReasonCode.DELIVERY_UNKNOWN
    assert outbox.completion_calls == 1
    assert len(adapter.calls) == 1


async def test_mark_unknown_revision_conflict_returns_concurrent_completion() -> None:
    outbox = UnknownConflictOutbox(delivery_record())
    adapter = FakeAdapter(capabilities(), mode=AdapterMode.UNKNOWN_AFTER_EFFECT)

    result = await worker(outbox, adapter, CLAIM_1, ATTEMPT_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.DELIVERED
    assert result.reason_code is ReasonCode.DELIVERY_SUCCEEDED
    assert result.delivered is True
    assert len(adapter.calls) == 1


async def test_invalid_repository_payload_becomes_unknown_without_adapter_effect() -> None:
    payload = "é" * 3_000
    current = delivery_record().model_copy(
        update={"rendered_text_digest": canonical_digest(payload)}
    )
    outbox = FakeOutbox(current, payload=payload)
    adapter = FakeAdapter(capabilities())

    result = await worker(outbox, adapter, CLAIM_1, ATTEMPT_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.UNKNOWN
    assert result.reason_code is ReasonCode.DELIVERY_UNKNOWN
    assert result.delivery.attempt_count == 1
    assert adapter.calls == []


def test_worker_result_reason_must_match_the_authoritative_record() -> None:
    failed = delivery_record(DeliveryState.FAILED)

    with pytest.raises(ValidationError, match="reason"):
        DeliveryWorkerResult(
            delivery=failed,
            reason_code=ReasonCode.DELIVERY_UNKNOWN,
            delivered=False,
            guarded=False,
        )


def test_worker_result_guard_requires_delivered_pre_action_capability() -> None:
    delivered_without_interception = delivery_record(DeliveryState.DELIVERED)

    result = DeliveryWorkerResult.from_delivery(delivered_without_interception)

    assert result.delivered is True
    assert result.guarded is False
    with pytest.raises(ValidationError, match="overstates"):
        DeliveryWorkerResult(
            delivery=delivered_without_interception,
            reason_code=ReasonCode.DELIVERY_SUCCEEDED,
            delivered=True,
            guarded=True,
        )


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_real_repository_binding_commit_and_worker_delivery(
    backend: str,
    tmp_path: Path,
) -> None:
    identifiers = iter(
        UUID(f"00000000-0000-4000-8000-{value:012x}") for value in range(0x900, 0xA00)
    )
    key = InstallationKey(b"i" * 32)
    repository = (
        MemoryRunRepository(
            installation_key=key,
            id_factory=lambda: next(identifiers),
        )
        if backend == "memory"
        else SQLiteRunRepository(
            tmp_path / "delivery-integration.sqlite3",
            installation_key=key,
            id_factory=lambda: next(identifiers),
        )
    )
    try:
        context, _reserved, _running = await advance_cycle_to_running(repository)
        command = reminder_commit_command(context)
        assert command.delivery is not None
        declared = capabilities()
        command = command.model_copy(
            update={
                "delivery": enqueue_delivery_binding(
                    target_request_id=command.delivery.target_request_id,
                    capabilities=declared,
                )
            }
        )
        committed = await repository.commit_cycle(command)
        pending = committed.delivery
        assert pending is not None
        forged_receipt = committed.model_dump(mode="python", warnings=False)
        forged_receipt["delivery"] = None
        with pytest.raises(ValidationError, match="committed reminder"):
            CycleReceipt.model_validate(forged_receipt)

        adapter = FakeAdapter(declared)
        result = await DeliveryWorker(
            repository=repository,
            adapter=adapter,
            id_factory=identity_factory(CLAIM_1, ATTEMPT_1),
        ).deliver(
            CYCLE_RUN_ID,
            pending.delivery_id,
            now=context.commit_time + timedelta(seconds=1),
        )

        assert result.delivery.state is DeliveryState.DELIVERED
        assert result.delivered
        assert len(adapter.calls) == 1
        delivery_states = tuple(
            entry.record.state
            for entry in await repository.ledger(CYCLE_RUN_ID)
            if isinstance(entry.record, DeliveryRecord)
        )
        assert delivery_states == (
            DeliveryState.PENDING,
            DeliveryState.CLAIMED,
            DeliveryState.ATTEMPTING,
            DeliveryState.DELIVERED,
        )
        assert (await repository.rebuild(CYCLE_RUN_ID)).equivalent
    finally:
        close = getattr(repository, "close", None)
        if close is not None:
            close()


@pytest.mark.parametrize(
    ("delivery", "reason", "message"),
    (
        (
            delivery_record(DeliveryState.DELIVERED),
            ReasonCode.DELIVERY_FAILED,
            "success reason",
        ),
        (
            delivery_record(DeliveryState.FAILED),
            ReasonCode.DELIVERY_SUCCEEDED,
            "cannot carry",
        ),
    ),
)
def test_worker_result_rejects_success_reason_contradictions(
    delivery: DeliveryRecord,
    reason: ReasonCode,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        DeliveryWorkerResult(
            delivery=delivery,
            reason_code=reason,
            delivered=delivery.state is DeliveryState.DELIVERED,
            guarded=False,
        )


class ExplodingCapabilitiesAdapter(FakeAdapter):
    def capabilities(self) -> AdapterCapabilities:
        raise RuntimeError("adapter capability secret")


async def test_capability_exception_is_sanitized_into_a_safe_refusal() -> None:
    outbox = FakeOutbox(delivery_record())
    adapter = ExplodingCapabilitiesAdapter(capabilities())

    result = await worker(outbox, adapter).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.REJECTED
    assert result.reason_code is ReasonCode.UNSUPPORTED_DELIVERY_CHANNEL
    assert "secret" not in repr(result)
    assert adapter.calls == []


async def test_capability_drift_cannot_reject_an_already_unknown_attempt() -> None:
    outbox = FakeOutbox(delivery_record(DeliveryState.UNKNOWN))
    adapter = FakeAdapter(capabilities(mappings=(user_mapping(),)))

    result = await worker(outbox, adapter).deliver(RUN_ID, DELIVERY_ID, now=NOW)

    assert result.delivery == outbox.current
    assert result.delivery.state is DeliveryState.UNKNOWN
    assert result.reason_code is ReasonCode.DELIVERY_UNKNOWN
    assert adapter.calls == []
    assert outbox.commands == []


class WrongRunOutbox(FakeOutbox):
    async def delivery(self, run_id: UUID, delivery_id: UUID) -> DeliveryRecord:
        assert run_id == RUN_ID
        assert delivery_id == DELIVERY_ID
        return self.current.model_copy(update={"run_id": OTHER_RUN_ID})


async def test_repository_cannot_substitute_a_delivery_from_another_run() -> None:
    outbox = WrongRunOutbox(delivery_record())

    with pytest.raises(DeliveryRuntimeError):
        await worker(outbox, FakeAdapter(capabilities())).deliver(
            RUN_ID,
            DELIVERY_ID,
            now=NOW,
        )


class ClaimReceiptOutbox(FakeOutbox):
    def __init__(self, current: DeliveryRecord, *, appended: bool) -> None:
        super().__init__(current)
        self.appended = appended

    async def claim_delivery(self, command: ClaimDelivery) -> DeliveryTransitionReceipt:
        self.commands.append(command)
        return self._receipt(appended=self.appended)


@pytest.mark.parametrize("appended", (False, True))
async def test_stale_claim_receipt_cannot_reach_an_attempt(appended: bool) -> None:
    outbox = ClaimReceiptOutbox(delivery_record(), appended=appended)
    adapter = FakeAdapter(capabilities())

    result = await worker(outbox, adapter, CLAIM_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.PENDING
    assert adapter.calls == []


class ForgedClaimBindingOutbox(FakeOutbox):
    async def claim_delivery(self, command: ClaimDelivery) -> DeliveryTransitionReceipt:
        receipt = await super().claim_delivery(command)
        forged = receipt.delivery.model_copy(
            update={"rendered_text_digest": canonical_digest("forged reminder text")}
        )
        return receipt.model_copy(update={"delivery": forged})

    async def begin_delivery_attempt(
        self,
        _command: BeginDeliveryAttempt,
    ) -> DeliveryAttemptReceipt:
        raise AssertionError("a forged claim receipt must not authorize an attempt")


async def test_claim_receipt_cannot_replace_the_committed_reminder_binding() -> None:
    outbox = ForgedClaimBindingOutbox(delivery_record())
    adapter = FakeAdapter(capabilities())

    result = await worker(outbox, adapter, CLAIM_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.CLAIMED
    assert result.delivery.rendered_text_digest == canonical_digest(REMINDER)
    assert adapter.calls == []


class ExplodingClaimReceiptOutbox(FakeOutbox):
    async def claim_delivery(self, command: ClaimDelivery) -> DeliveryTransitionReceipt:
        receipt = await super().claim_delivery(command)
        object.__setattr__(receipt, "model_dump_json", _explode_receipt_serialization)
        return receipt


async def test_hostile_claim_receipt_serialization_never_authorizes_an_attempt() -> None:
    outbox = ExplodingClaimReceiptOutbox(delivery_record())
    adapter = FakeAdapter(capabilities())

    result = await worker(outbox, adapter, CLAIM_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.CLAIMED
    assert "hostile-receipt-serialization-secret" not in repr(result)
    assert adapter.calls == []


async def test_unowned_claimed_record_never_authorizes_an_attempt() -> None:
    forged = delivery_record(DeliveryState.CLAIMED).model_copy(update={"claim_id": None})
    outbox = FakeOutbox(forged)
    adapter = FakeAdapter(capabilities())

    with pytest.raises(DeliveryRuntimeError, match="input failed validation"):
        await worker(outbox, adapter).deliver(
            RUN_ID,
            DELIVERY_ID,
            now=NOW + timedelta(seconds=1),
        )

    assert adapter.calls == []


class NonReceiptBeginOutbox(FakeOutbox):
    async def begin_delivery_attempt(
        self,
        command: BeginDeliveryAttempt,
    ) -> DeliveryAttemptReceipt:
        await super().begin_delivery_attempt(command)
        return cast(DeliveryAttemptReceipt, object())


async def test_non_receipt_begin_result_never_authorizes_an_adapter_call() -> None:
    outbox = NonReceiptBeginOutbox(delivery_record())
    adapter = FakeAdapter(capabilities())

    result = await worker(outbox, adapter, CLAIM_1, ATTEMPT_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.ATTEMPTING
    assert adapter.calls == []


class ExplodingCompletionOutbox(FakeOutbox):
    async def complete_delivery(
        self,
        command: CompleteDelivery,
    ) -> DeliveryTransitionReceipt:
        self.commands.append(command)
        raise RuntimeError("completion transport failed")


@pytest.mark.parametrize("mode", (AdapterMode.SUCCESS, AdapterMode.KNOWN_FAILURE))
async def test_completion_transport_failure_is_recorded_unknown(mode: AdapterMode) -> None:
    outbox = ExplodingCompletionOutbox(delivery_record())
    adapter = FakeAdapter(capabilities(), mode=mode)

    result = await worker(outbox, adapter, CLAIM_1, ATTEMPT_1).deliver(
        RUN_ID,
        DELIVERY_ID,
        now=NOW + timedelta(seconds=1),
    )

    assert result.delivery.state is DeliveryState.UNKNOWN
    assert result.reason_code is ReasonCode.DELIVERY_UNKNOWN
    assert len(adapter.calls) == 1


@pytest.mark.parametrize("operation", ("complete", "complete-failed", "unknown"))
async def test_terminal_persistence_helpers_require_attempt_ownership_tokens(
    operation: str,
) -> None:
    outbox = FakeOutbox(delivery_record(DeliveryState.ATTEMPTING))
    subject = worker(outbox, FakeAdapter(capabilities()))
    unowned = outbox.current.model_copy(update={"claim_id": None})

    with pytest.raises(DeliveryRuntimeError):
        if operation == "complete":
            await subject._complete(
                unowned,
                outcome=DeliveryOutcome.DELIVERED,
                provider_receipt_id="provider-receipt-1",
                now=NOW,
            )
        elif operation == "complete-failed":
            await subject._complete_failed(unowned, NOW)
        else:
            await subject._unknown(unowned, NOW)

    assert outbox.commands == []


def test_enqueue_binding_rejects_a_forged_deduplication_flag() -> None:
    binding = enqueue_delivery_binding(
        target_request_id=TARGET_REQUEST_ID,
        capabilities=capabilities(),
    )
    values = binding.model_dump(mode="python", warnings=False)
    values["adapter_deduplicates"] = False

    with pytest.raises(ValidationError, match="deduplication flag"):
        EnqueueDelivery.model_validate(values)


@pytest.mark.parametrize(
    ("outcome", "provider_receipt_id", "message"),
    (
        (DeliveryOutcome.UNKNOWN, None, "completion outcome"),
        (DeliveryOutcome.DELIVERED, None, "provider receipt"),
        (DeliveryOutcome.FAILED, "unexpected-receipt", "provider receipt"),
    ),
)
def test_completion_command_rejects_non_terminal_or_mismatched_receipts(
    outcome: DeliveryOutcome,
    provider_receipt_id: str | None,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        CompleteDelivery(
            run_id=RUN_ID,
            delivery_id=DELIVERY_ID,
            expected_revision=3,
            claim_id=CLAIM_1,
            attempt_id=ATTEMPT_1,
            outcome=outcome,
            provider_receipt_id=provider_receipt_id,
            updated_at=NOW,
        )


def test_rejection_command_accepts_only_delivery_refusal_reasons() -> None:
    with pytest.raises(ValidationError, match="delivery refusal reason"):
        RejectDelivery(
            run_id=RUN_ID,
            delivery_id=DELIVERY_ID,
            expected_revision=1,
            reason_code=ReasonCode.DELIVERY_FAILED,
            updated_at=NOW,
        )


def test_transition_receipt_rejects_mixed_integrity_algorithms() -> None:
    with pytest.raises(ValidationError, match="integrity algorithms differ"):
        DeliveryTransitionReceipt(
            appended=True,
            delivery=delivery_record(),
            record_tag=tag("d"),
            ledger_position=1,
            chain_tag=PayloadDigest(
                algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
                value="e" * 64,
            ),
        )


async def test_attempt_receipt_cannot_expose_an_envelope_without_a_new_append() -> None:
    outbox = FakeOutbox(delivery_record(DeliveryState.CLAIMED))
    receipt = await outbox.begin_delivery_attempt(
        BeginDeliveryAttempt(
            run_id=RUN_ID,
            delivery_id=DELIVERY_ID,
            expected_revision=2,
            claim_id=CLAIM_1,
            attempt_id=ATTEMPT_1,
            updated_at=NOW + timedelta(seconds=1),
        )
    )
    values = receipt.model_dump(mode="python", warnings=False)
    values["appended"] = False

    with pytest.raises(ValidationError, match="newly persisted attempt"):
        DeliveryAttemptReceipt.model_validate(values)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("wrong-state", "wrong run or state"),
        ("duplicate", "must be disjoint"),
        ("retry-without-dedup", "durable deduplication"),
        ("non-retry-with-dedup", "cannot support deduplication"),
        ("stale-marked-unknown", "newly marked unknown"),
    ),
)
def test_recovery_receipt_rejects_unsafe_or_ambiguous_classification(
    case: str,
    message: str,
) -> None:
    values: dict[str, object] = {
        "run_id": RUN_ID,
        "marked_unknown": (),
        "resumable_pending": (),
        "resumable_claimed": (),
        "retryable_unknown": (),
        "non_retryable_unknown": (),
    }
    if case == "wrong-state":
        values["resumable_pending"] = (delivery_record(DeliveryState.CLAIMED),)
    elif case == "duplicate":
        pending = delivery_record()
        values["resumable_pending"] = (pending, pending)
    elif case == "retry-without-dedup":
        values["retryable_unknown"] = (delivery_record(DeliveryState.UNKNOWN, deduplicates=False),)
    elif case == "non-retry-with-dedup":
        values["non_retryable_unknown"] = (delivery_record(DeliveryState.UNKNOWN),)
    else:
        unknown = delivery_record(DeliveryState.UNKNOWN)
        values["marked_unknown"] = (
            DeliveryTransitionReceipt(
                appended=False,
                delivery=unknown,
                record_tag=tag("d"),
                ledger_position=1,
                chain_tag=tag("e"),
            ),
        )

    with pytest.raises(ValidationError, match=message):
        DeliveryRecoveryReceipt.model_validate(values)
