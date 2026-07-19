from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Protocol, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator

from saliencegate.domain import (
    DeliveryOutcome,
    DeliveryRecord,
    DeliveryState,
    DeliveryTarget,
    ReasonCode,
    canonical_digest,
    new_repository_id,
)
from saliencegate.ports.adapters import (
    ADAPTER_CONTRACT_VERSION,
    AdapterCapabilities,
    AdapterDeliveryFailedError,
    AdapterDeliveryRefusedError,
    DeliveryAdapter,
    DeliveryEnvelope,
    adapter_capabilities_digest,
    delivery_payload,
    select_injection_mapping,
    validate_delivery_receipt,
    validated_capabilities,
)
from saliencegate.ports.repository import (
    BeginDeliveryAttempt,
    ClaimDelivery,
    CompleteDelivery,
    DeliveryAttemptReceipt,
    DeliveryRecoveryReceipt,
    DeliveryRevisionConflictError,
    DeliveryTransitionReceipt,
    MarkDeliveryUnknown,
    RejectDelivery,
)


class DeliveryRuntimeError(ValueError):
    def __init__(self) -> None:
        super().__init__("delivery worker input failed validation")


async def _drain_cleanup(task: asyncio.Task[object]) -> None:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    with suppress(asyncio.CancelledError, Exception):
        task.result()


_DELIVERY_BINDING_FIELDS = (
    "cycle_id",
    "intervention_id",
    "rendered_text_digest",
    "target_request_id",
    "target",
    "adapter_id",
    "adapter_deduplicates",
    "adapter_deduplication_guarantee",
    "adapter_supports_pre_action",
    "adapter_contract_version",
    "adapter_capabilities_digest",
    "created_at",
)


def _same_delivery_binding(left: DeliveryRecord, right: DeliveryRecord) -> bool:
    return all(
        getattr(left, field_name) == getattr(right, field_name)
        for field_name in _DELIVERY_BINDING_FIELDS
    )


class DeliveryOutbox(Protocol):
    async def delivery(self, run_id: UUID, delivery_id: UUID) -> DeliveryRecord: ...

    async def claim_delivery(
        self,
        command: ClaimDelivery,
    ) -> DeliveryTransitionReceipt: ...

    async def begin_delivery_attempt(
        self,
        command: BeginDeliveryAttempt,
    ) -> DeliveryAttemptReceipt: ...

    async def complete_delivery(
        self,
        command: CompleteDelivery,
    ) -> DeliveryTransitionReceipt: ...

    async def mark_delivery_unknown(
        self,
        command: MarkDeliveryUnknown,
    ) -> DeliveryTransitionReceipt: ...

    async def reject_delivery(
        self,
        command: RejectDelivery,
    ) -> DeliveryTransitionReceipt: ...

    async def recover_deliveries(
        self,
        run_id: UUID,
        *,
        recovered_at: datetime,
    ) -> DeliveryRecoveryReceipt: ...


class DeliveryWorkerResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    delivery: DeliveryRecord
    reason_code: ReasonCode
    delivered: bool
    guarded: bool

    @model_validator(mode="after")
    def claims_match_authoritative_delivery(self) -> Self:
        expected_delivered = self.delivery.state is DeliveryState.DELIVERED
        expected_guarded = (
            expected_delivered
            and self.delivery.target is DeliveryTarget.PRE_ACTION_REPLAN
            and self.delivery.adapter_supports_pre_action
        )
        if self.delivered is not expected_delivered or self.guarded is not expected_guarded:
            raise ValueError("delivery result overstates the authoritative outcome")
        if expected_delivered and self.reason_code is not ReasonCode.DELIVERY_SUCCEEDED:
            raise ValueError("delivered result requires the success reason")
        if not expected_delivered and self.reason_code is ReasonCode.DELIVERY_SUCCEEDED:
            raise ValueError("undelivered result cannot carry the success reason")
        if (
            self.delivery.reason_code is not None
            and self.reason_code is not self.delivery.reason_code
        ):
            raise ValueError("delivery result reason disagrees with the authoritative record")
        return self

    @classmethod
    def from_delivery(
        cls,
        delivery: DeliveryRecord,
        *,
        reason_code: ReasonCode | None = None,
    ) -> DeliveryWorkerResult:
        reason = delivery.reason_code if reason_code is None else reason_code
        if reason is None:
            reason = (
                ReasonCode.DELIVERY_UNKNOWN
                if delivery.state is DeliveryState.ATTEMPTING
                else ReasonCode.TARGET_UNAVAILABLE
            )
        delivered = delivery.state is DeliveryState.DELIVERED
        return cls(
            delivery=delivery,
            reason_code=reason,
            delivered=delivered,
            guarded=(
                delivered
                and delivery.target is DeliveryTarget.PRE_ACTION_REPLAN
                and delivery.adapter_supports_pre_action
            ),
        )


def _utc_timestamp(value: object) -> datetime:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise DeliveryRuntimeError()
    return value.astimezone(UTC)


def _uuid4(value: object) -> UUID:
    if type(value) is not UUID or value.version != 4:
        raise DeliveryRuntimeError()
    return UUID(int=value.int)


class DeliveryWorker:
    """Persist adapter ownership before every potentially ambiguous side effect."""

    __slots__ = ("_adapter", "_id_factory", "_repository")

    def __init__(
        self,
        repository: DeliveryOutbox,
        adapter: DeliveryAdapter,
        id_factory: Callable[[], UUID] = new_repository_id,
    ) -> None:
        self._repository = repository
        self._adapter = adapter
        self._id_factory = id_factory

    def _new_id(self) -> UUID:
        try:
            return _uuid4(self._id_factory())
        except (StopIteration, TypeError, ValueError):
            raise DeliveryRuntimeError() from None

    def _capabilities(self, delivery: DeliveryRecord) -> AdapterCapabilities:
        try:
            capabilities = validated_capabilities(self._adapter.capabilities())
        except AdapterDeliveryRefusedError:
            raise
        except Exception:
            raise AdapterDeliveryRefusedError(ReasonCode.UNSUPPORTED_DELIVERY_CHANNEL) from None
        if (
            capabilities.adapter_id != delivery.adapter_id
            or capabilities.deduplicates_delivery_id is not delivery.adapter_deduplicates
            or capabilities.deduplication_guarantee is not delivery.adapter_deduplication_guarantee
            or capabilities.pre_action_interception is not delivery.adapter_supports_pre_action
            or delivery.adapter_contract_version != ADAPTER_CONTRACT_VERSION
        ):
            raise AdapterDeliveryRefusedError(ReasonCode.TARGET_UNAVAILABLE)
        return capabilities

    async def _current(self, run_id: UUID, delivery_id: UUID) -> DeliveryRecord:
        untrusted = await self._repository.delivery(run_id, delivery_id)
        current: DeliveryRecord | None = None
        try:
            if type(untrusted) is DeliveryRecord:
                current = DeliveryRecord.model_validate_json(
                    untrusted.model_dump_json(warnings=False)
                )
        except Exception:
            pass
        if current is None:
            raise DeliveryRuntimeError()
        return current

    async def _reject(
        self,
        delivery: DeliveryRecord,
        *,
        reason_code: ReasonCode,
        now: datetime,
    ) -> DeliveryWorkerResult:
        if delivery.state not in (DeliveryState.PENDING, DeliveryState.CLAIMED):
            return DeliveryWorkerResult.from_delivery(delivery)
        try:
            receipt = await self._repository.reject_delivery(
                RejectDelivery(
                    run_id=delivery.run_id,
                    delivery_id=delivery.delivery_id,
                    expected_revision=delivery.revision,
                    claim_id=delivery.claim_id,
                    reason_code=reason_code,
                    updated_at=now,
                )
            )
        except DeliveryRevisionConflictError:
            current = await self._current(delivery.run_id, delivery.delivery_id)
            return DeliveryWorkerResult.from_delivery(current)
        return DeliveryWorkerResult.from_delivery(
            receipt.delivery,
            reason_code=reason_code,
        )

    async def _unknown_after_cancellation(
        self,
        delivery: DeliveryRecord,
        now: datetime,
    ) -> None:
        cleanup = asyncio.create_task(self._unknown(delivery, now))
        await _drain_cleanup(cleanup)

    async def _unknown_persisted_attempt_after_cancellation(
        self,
        *,
        run_id: UUID,
        delivery_id: UUID,
        claim_id: UUID,
        attempt_id: UUID,
        now: datetime,
    ) -> None:
        async def recover() -> None:
            current = await self._current(run_id, delivery_id)
            if (
                current.state is DeliveryState.ATTEMPTING
                and current.claim_id == claim_id
                and current.attempt_id == attempt_id
            ):
                await self._unknown(current, now)

        cleanup = asyncio.create_task(recover())
        await _drain_cleanup(cleanup)

    async def deliver(
        self,
        run_id: UUID,
        delivery_id: UUID,
        *,
        now: datetime,
    ) -> DeliveryWorkerResult:
        run = _uuid4(run_id)
        identifier = _uuid4(delivery_id)
        timestamp = _utc_timestamp(now)
        current = await self._current(run, identifier)
        if current.run_id != run:
            raise DeliveryRuntimeError()
        if current.state in (
            DeliveryState.DELIVERED,
            DeliveryState.FAILED,
            DeliveryState.REJECTED,
        ):
            return DeliveryWorkerResult.from_delivery(current)
        if current.state is DeliveryState.ATTEMPTING:
            return DeliveryWorkerResult.from_delivery(
                current,
                reason_code=ReasonCode.DELIVERY_UNKNOWN,
            )
        if current.state is DeliveryState.UNKNOWN and not current.adapter_deduplicates:
            return DeliveryWorkerResult.from_delivery(current)

        try:
            capabilities = self._capabilities(current)
            mapping = select_injection_mapping(capabilities, current.target)
            if adapter_capabilities_digest(capabilities) != current.adapter_capabilities_digest:
                raise AdapterDeliveryRefusedError(ReasonCode.TARGET_UNAVAILABLE)
        except AdapterDeliveryRefusedError as error:
            return await self._reject(
                current,
                reason_code=error.reason_code,
                now=timestamp,
            )

        if current.state in (DeliveryState.PENDING, DeliveryState.UNKNOWN):
            claim_id = self._new_id()
            try:
                claimed = await self._repository.claim_delivery(
                    ClaimDelivery(
                        run_id=run,
                        delivery_id=identifier,
                        expected_revision=current.revision,
                        claim_id=claim_id,
                        updated_at=timestamp,
                    )
                )
            except DeliveryRevisionConflictError:
                latest = await self._current(run, identifier)
                return DeliveryWorkerResult.from_delivery(latest)
            valid_claim: DeliveryTransitionReceipt | None = None
            try:
                if type(claimed) is DeliveryTransitionReceipt:
                    valid_claim = DeliveryTransitionReceipt.model_validate_json(
                        claimed.model_dump_json(warnings=False)
                    )
            except Exception:
                pass
            if (
                valid_claim is None
                or not valid_claim.appended
                or valid_claim.delivery.delivery_id != identifier
                or valid_claim.delivery.run_id != run
                or valid_claim.delivery.revision != current.revision + 1
                or valid_claim.delivery.state is not DeliveryState.CLAIMED
                or valid_claim.delivery.claim_id != claim_id
                or valid_claim.delivery.attempt_id is not None
                or valid_claim.delivery.attempt_count != current.attempt_count
                or valid_claim.delivery.updated_at != timestamp
                or not _same_delivery_binding(valid_claim.delivery, current)
            ):
                latest = await self._current(run, identifier)
                return DeliveryWorkerResult.from_delivery(latest)
            current = valid_claim.delivery
        if current.state is not DeliveryState.CLAIMED or current.claim_id is None:
            return DeliveryWorkerResult.from_delivery(current)

        attempt_id = self._new_id()
        try:
            attempted = await self._repository.begin_delivery_attempt(
                BeginDeliveryAttempt(
                    run_id=run,
                    delivery_id=identifier,
                    expected_revision=current.revision,
                    claim_id=current.claim_id,
                    attempt_id=attempt_id,
                    updated_at=timestamp,
                )
            )
        except asyncio.CancelledError:
            await self._unknown_persisted_attempt_after_cancellation(
                run_id=run,
                delivery_id=identifier,
                claim_id=current.claim_id,
                attempt_id=attempt_id,
                now=timestamp,
            )
            raise
        except DeliveryRevisionConflictError:
            latest = await self._current(run, identifier)
            return DeliveryWorkerResult.from_delivery(latest)
        valid_attempt: DeliveryAttemptReceipt | None = None
        try:
            if type(attempted) is DeliveryAttemptReceipt:
                valid_attempt = DeliveryAttemptReceipt.model_validate_json(
                    attempted.model_dump_json(warnings=False)
                )
        except (AttributeError, TypeError, ValueError):
            pass
        if valid_attempt is None:
            latest = await self._current(run, identifier)
            return DeliveryWorkerResult.from_delivery(latest)
        attempted = valid_attempt
        if not attempted.appended or attempted.envelope is None:
            return DeliveryWorkerResult.from_delivery(attempted.delivery)
        if (
            attempted.delivery.delivery_id != identifier
            or attempted.delivery.run_id != run
            or attempted.delivery.revision != current.revision + 1
            or attempted.delivery.state is not DeliveryState.ATTEMPTING
            or attempted.delivery.claim_id != current.claim_id
            or attempted.delivery.attempt_id != attempt_id
            or attempted.delivery.attempt_count != current.attempt_count + 1
            or attempted.delivery.updated_at != timestamp
            or not _same_delivery_binding(attempted.delivery, current)
        ):
            latest = await self._current(run, identifier)
            return DeliveryWorkerResult.from_delivery(latest)
        raw = attempted.envelope
        if canonical_digest(raw.rendered_text) != raw.rendered_text_digest:
            latest = await self._current(run, identifier)
            return DeliveryWorkerResult.from_delivery(latest)
        try:
            envelope = DeliveryEnvelope(
                schema_version="1.0",
                delivery_id=raw.delivery_id,
                run_id=raw.run_id,
                cycle_id=raw.cycle_id,
                intervention_id=raw.intervention_id,
                claim_id=raw.claim_id,
                attempt_id=raw.attempt_id,
                attempt_number=raw.attempt_number,
                adapter_id=raw.adapter_id,
                target_request_id=raw.target_request_id,
                target=raw.target,
                mapping=mapping,
                payload=delivery_payload(raw.rendered_text, mapping),
                ttl_steps=1,
                created_at=attempted.delivery.updated_at,
            )
        except (TypeError, ValueError):
            return await self._unknown(attempted.delivery, timestamp)

        try:
            untrusted_receipt = await self._adapter.deliver(envelope)
        except asyncio.CancelledError:
            await self._unknown_after_cancellation(attempted.delivery, timestamp)
            raise
        except AdapterDeliveryFailedError:
            try:
                return await self._complete_failed(attempted.delivery, timestamp)
            except asyncio.CancelledError:
                await self._unknown_after_cancellation(attempted.delivery, timestamp)
                raise
            except Exception:
                return await self._unknown(attempted.delivery, timestamp)
        except Exception:
            return await self._unknown(attempted.delivery, timestamp)
        try:
            receipt = validate_delivery_receipt(envelope, untrusted_receipt)
        except Exception:
            return await self._unknown(attempted.delivery, timestamp)
        try:
            return await self._complete(
                attempted.delivery,
                outcome=DeliveryOutcome.DELIVERED,
                provider_receipt_id=receipt.provider_receipt_id,
                now=timestamp,
            )
        except asyncio.CancelledError:
            await self._unknown_after_cancellation(attempted.delivery, timestamp)
            raise
        except Exception:
            return await self._unknown(attempted.delivery, timestamp)

    async def _complete(
        self,
        delivery: DeliveryRecord,
        *,
        outcome: DeliveryOutcome,
        provider_receipt_id: str | None,
        now: datetime,
    ) -> DeliveryWorkerResult:
        if delivery.claim_id is None or delivery.attempt_id is None:
            raise DeliveryRuntimeError()
        expected = delivery
        try:
            completed = await self._repository.complete_delivery(
                CompleteDelivery(
                    run_id=delivery.run_id,
                    delivery_id=delivery.delivery_id,
                    expected_revision=expected.revision,
                    claim_id=delivery.claim_id,
                    attempt_id=delivery.attempt_id,
                    outcome=outcome,
                    provider_receipt_id=provider_receipt_id,
                    updated_at=now,
                )
            )
        except DeliveryRevisionConflictError:
            current = await self._current(delivery.run_id, delivery.delivery_id)
            if (
                current.state is DeliveryState.UNKNOWN
                and current.claim_id == delivery.claim_id
                and current.attempt_id == delivery.attempt_id
            ):
                completed = await self._repository.complete_delivery(
                    CompleteDelivery(
                        run_id=delivery.run_id,
                        delivery_id=delivery.delivery_id,
                        expected_revision=current.revision,
                        claim_id=delivery.claim_id,
                        attempt_id=delivery.attempt_id,
                        outcome=outcome,
                        provider_receipt_id=provider_receipt_id,
                        updated_at=max(now, current.updated_at),
                    )
                )
            else:
                return DeliveryWorkerResult.from_delivery(current)
        return DeliveryWorkerResult.from_delivery(completed.delivery)

    async def _complete_failed(
        self,
        delivery: DeliveryRecord,
        now: datetime,
    ) -> DeliveryWorkerResult:
        if delivery.claim_id is None or delivery.attempt_id is None:
            raise DeliveryRuntimeError()
        return await self._complete(
            delivery,
            outcome=DeliveryOutcome.FAILED,
            provider_receipt_id=None,
            now=now,
        )

    async def _unknown(
        self,
        delivery: DeliveryRecord,
        now: datetime,
    ) -> DeliveryWorkerResult:
        if delivery.claim_id is None or delivery.attempt_id is None:
            raise DeliveryRuntimeError()
        try:
            unknown = await self._repository.mark_delivery_unknown(
                MarkDeliveryUnknown(
                    run_id=delivery.run_id,
                    delivery_id=delivery.delivery_id,
                    expected_revision=delivery.revision,
                    claim_id=delivery.claim_id,
                    attempt_id=delivery.attempt_id,
                    updated_at=now,
                )
            )
        except DeliveryRevisionConflictError:
            current = await self._current(delivery.run_id, delivery.delivery_id)
            return DeliveryWorkerResult.from_delivery(current)
        return DeliveryWorkerResult.from_delivery(unknown.delivery)

    async def recover(
        self,
        run_id: UUID,
        *,
        recovered_at: datetime,
    ) -> tuple[DeliveryWorkerResult, ...]:
        run = _uuid4(run_id)
        timestamp = _utc_timestamp(recovered_at)
        recovered = await self._repository.recover_deliveries(
            run,
            recovered_at=timestamp,
        )
        results: list[DeliveryWorkerResult] = []
        for delivery in (
            *recovered.resumable_pending,
            *recovered.resumable_claimed,
            *recovered.retryable_unknown,
        ):
            results.append(
                await self.deliver(
                    run,
                    delivery.delivery_id,
                    now=timestamp,
                )
            )
        results.extend(
            DeliveryWorkerResult.from_delivery(delivery)
            for delivery in recovered.non_retryable_unknown
        )
        return tuple(results)


__all__ = [
    "DeliveryOutbox",
    "DeliveryRuntimeError",
    "DeliveryWorker",
    "DeliveryWorkerResult",
]
