from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TypeVar
from uuid import UUID

from pydantic import BaseModel, ValidationError

from saliencegate.domain import (
    BudgetAmounts,
    BudgetSnapshot,
    CycleRecord,
    CycleState,
    DeliveryOutcome,
    DeliveryRecord,
    DeliveryState,
    EventPhase,
    EventType,
    InterventionOutcome,
    InvocationDecision,
    LedgerRecord,
    NormalizedTraceEventDraft,
    PayloadDigest,
    PayloadDigestAlgorithm,
    ReasonCode,
    RedactedTraceEventDraft,
    Signal,
    TraceEvent,
    TrustLabel,
    canonical_digest,
    canonical_json,
    cycle_id,
    new_repository_id,
    validate_normalized_trace_event_draft,
)
from saliencegate.domain import (
    delivery_id as derive_delivery_id,
)
from saliencegate.ports.repository import (
    MAX_CONDITIONAL_BATCH_EVENTS,
    MAX_CONDITIONAL_BATCH_OPERATIONS,
    MAX_CONDITIONAL_BATCH_REQUEST_BYTES,
    MAX_CONDITIONAL_BATCH_SIGNALS,
    AppendDisposition,
    AppendReceipt,
    BeginCycle,
    BeginDeliveryAttempt,
    ClaimDelivery,
    CommitCycle,
    CompleteDelivery,
    ConditionalAppendOperation,
    ConditionalBatchReceipt,
    ConditionalEventAppend,
    ConditionalSignalAppend,
    CrossRunReferenceError,
    CycleConflictError,
    CycleReceipt,
    CycleRecoveryReceipt,
    CycleRevisionConflictError,
    DeliveryAttemptEnvelope,
    DeliveryAttemptReceipt,
    DeliveryNotFoundError,
    DeliveryOwnershipError,
    DeliveryRecoveryReceipt,
    DeliveryRevisionConflictError,
    DeliveryTransitionReceipt,
    DigestVerificationError,
    DirectLedgerRecord,
    FailCycle,
    InvalidAppendTypeError,
    InvalidCycleStateError,
    InvalidDeliveryStateError,
    InvalidDraftError,
    InvalidQueryError,
    InvalidRecordError,
    InvalidRecordTypeError,
    InvalidRecoveryTimeError,
    InvalidRunIdError,
    LedgerEntry,
    LedgerHead,
    LedgerHeadConflictError,
    LedgerReceipt,
    MarkDeliveryUnknown,
    MemoryDeltaPreview,
    MemoryHit,
    MemoryQuery,
    MemorySnapshot,
    PreviewConflictError,
    PreviewMemoryDelta,
    ProjectionDigests,
    ProjectionInvariantError,
    RebuildError,
    RebuildReceipt,
    RecordCollisionError,
    RejectDelivery,
    RepositoryError,
    ReserveCycle,
    RunNotFoundError,
    StartCycle,
    UnsafeEventMetadataError,
    UnsafeRecordContentError,
)
from saliencegate.repository.integrity import IntegrityContext
from saliencegate.repository.projector import (
    Projection,
    apply_entry,
    empty_projection,
    projection_digests,
    validate_complete_projection,
)
from saliencegate.repository.projector import (
    budget_snapshot as projected_budget_snapshot,
)
from saliencegate.repository.projector import (
    preview_memory_delta as preview_projected_memory_delta,
)
from saliencegate.repository.projector import (
    search as search_projection,
)
from saliencegate.repository.projector import (
    snapshot as projection_snapshot,
)
from saliencegate.security import (
    InstallationKey,
    RedactionPolicy,
    Redactor,
    load_or_create_installation_key,
)
from saliencegate.security.redaction import verify_redacted_event

RecordT = TypeVar("RecordT", Signal, InvocationDecision, InterventionOutcome)
ModelT = TypeVar("ModelT", bound=BaseModel)


def _copy_model(value: ModelT) -> ModelT:
    return type(value).model_validate_json(value.model_dump_json(warnings=False))


@dataclass(slots=True)
class _DirectRecord:
    record: DirectLedgerRecord
    receipt: LedgerReceipt


@dataclass(slots=True)
class _CycleRevision:
    record: CycleRecord
    receipt: CycleReceipt


@dataclass(frozen=True, slots=True)
class _CycleReceiptSeed:
    record: CycleRecord
    record_tag: PayloadDigest
    ledger_position: int
    chain_tag: PayloadDigest
    budget_snapshot: BudgetSnapshot


@dataclass(slots=True)
class _DeliveryRevision:
    record: DeliveryRecord
    receipt: DeliveryTransitionReceipt


@dataclass(frozen=True, slots=True)
class _VerifiedRunState:
    run_id: UUID
    ledger: tuple[LedgerEntry, ...]
    ledger_head: LedgerHead
    projection: Projection
    digests: ProjectionDigests


@dataclass(frozen=True, slots=True)
class _ReplayedRun:
    state: _VerifiedRunState
    direct_records: dict[tuple[str, UUID], _DirectRecord]
    cycle_records: dict[tuple[str, int], _CycleRevision]
    delivery_records: dict[tuple[UUID, int], _DeliveryRevision]
    collision_receipts: dict[str, AppendReceipt]


@dataclass(frozen=True, slots=True)
class _PreparedConditionalEvent:
    draft: NormalizedTraceEventDraft
    redacted: RedactedTraceEventDraft
    event_id: UUID


@dataclass(frozen=True, slots=True)
class _PreparedConditionalSignal:
    signal: Signal


_PreparedConditionalOperation = _PreparedConditionalEvent | _PreparedConditionalSignal


@dataclass(slots=True)
class _RunSlot:
    run_id: UUID
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ledger: tuple[LedgerEntry, ...] = ()
    ledger_head: LedgerHead | None = None
    projection: Projection | None = None
    direct_records: dict[tuple[str, UUID], _DirectRecord] = field(default_factory=dict)
    cycle_records: dict[tuple[str, int], _CycleRevision] = field(default_factory=dict)
    delivery_records: dict[tuple[UUID, int], _DeliveryRevision] = field(default_factory=dict)
    collision_receipts: dict[str, AppendReceipt] = field(default_factory=dict)
    append_leases: int = 0

    def __post_init__(self) -> None:
        if self.projection is None:
            self.projection = empty_projection(self.run_id)


class MemoryRunRepository:
    """Single-event-loop, transactional in-memory implementation of the run repository."""

    def __init__(
        self,
        *,
        redaction_policy: RedactionPolicy | None = None,
        installation_key: InstallationKey | None = None,
        synthetic_benchmark: bool = False,
        id_factory: Callable[[], UUID] = new_repository_id,
    ) -> None:
        if installation_key is None and not synthetic_benchmark:
            installation_key = load_or_create_installation_key()
        if redaction_policy is not None and type(redaction_policy) is not RedactionPolicy:
            raise TypeError("redaction_policy must be exactly RedactionPolicy")
        policy = (
            RedactionPolicy()
            if redaction_policy is None
            else RedactionPolicy(
                literal_secrets=redaction_policy.literal_secrets,
                structured_field_names=redaction_policy.structured_field_names,
            )
        )
        self._redactor = Redactor(
            literal_secrets=policy.literal_secrets,
            structured_field_names=policy.structured_field_names,
        )
        self._integrity = IntegrityContext(
            key=installation_key,
            synthetic_benchmark=synthetic_benchmark,
        )
        self._id_factory = id_factory
        self._slots: dict[UUID, _RunSlot] = {}
        self._trusted_heads: dict[UUID, LedgerHead] = {}
        self._trusted_projections: dict[UUID, Projection] = {}
        self._slots_lock = asyncio.Lock()

    def __repr__(self) -> str:
        run_count = sum(slot.ledger_head is not None for slot in self._slots.values())
        return f"MemoryRunRepository(runs={run_count})"

    async def _slot(self, run_id: UUID) -> _RunSlot:
        async with self._slots_lock:
            slot = self._slots.get(run_id)
            if slot is None:
                if self._has_trusted_state(run_id):
                    raise DigestVerificationError("trusted ledger anchor")
                raise RunNotFoundError(run_id)
            if not slot.ledger and slot.ledger_head is None:
                if self._has_trusted_state(run_id):
                    raise DigestVerificationError("trusted ledger anchor")
                raise RunNotFoundError(run_id)
            return slot

    async def _acquire_append_slot(self, run_id: UUID) -> _RunSlot:
        async with self._slots_lock:
            slot = self._slots.get(run_id)
            if slot is None:
                if self._has_trusted_state(run_id):
                    raise DigestVerificationError("trusted ledger anchor")
                slot = _RunSlot(run_id=run_id)
                self._slots[run_id] = slot
            elif not slot.ledger and slot.ledger_head is None and self._has_trusted_state(run_id):
                raise DigestVerificationError("trusted ledger anchor")
            slot.append_leases += 1
            return slot

    def _has_trusted_state(self, run_id: UUID) -> bool:
        return run_id in self._trusted_heads or run_id in self._trusted_projections

    async def _release_append_slot(self, slot: _RunSlot) -> None:
        async with self._slots_lock:
            slot.append_leases -= 1
            if (
                slot.append_leases == 0
                and not slot.ledger
                and slot.ledger_head is None
                and self._slots.get(slot.run_id) is slot
            ):
                del self._slots[slot.run_id]

    async def _release_append_slot_safely(self, slot: _RunSlot) -> None:
        release = asyncio.create_task(self._release_append_slot(slot))
        cancellation_received = False
        while True:
            try:
                await asyncio.shield(release)
            except asyncio.CancelledError:
                if release.cancelled():  # pragma: no cover - release is never cancelled internally
                    raise
                cancellation_received = True
                continue
            except BaseException:
                if cancellation_received:
                    raise asyncio.CancelledError from None
                raise
            if cancellation_received:
                raise asyncio.CancelledError
            return

    def _new_id(self) -> UUID:
        identifier = self._id_factory()
        if type(identifier) is not UUID or identifier.version != 4:
            raise ProjectionInvariantError("repository ID factory must return UUID4 values")
        return UUID(int=identifier.int)

    @staticmethod
    def _validate_run_id(run_id: UUID) -> UUID:
        if type(run_id) is not UUID or run_id.version != 4:
            raise InvalidRunIdError()
        return UUID(int=run_id.int)

    def _entry(
        self,
        slot: _RunSlot,
        record: LedgerRecord,
        *,
        record_key: str,
    ) -> LedgerEntry:
        position = len(slot.ledger) + 1
        previous = slot.ledger[-1].chain_tag if slot.ledger else None
        record_tag = self._integrity.tag(
            record,
            domain="saliencegate:ledger-record:v1",
        )
        chain_value = {
            "run_id": str(slot.run_id),
            "position": position,
            "record_key": record_key,
            "record_tag": record_tag,
            "previous_chain_tag": previous,
        }
        chain_tag = self._integrity.tag(
            chain_value,
            domain="saliencegate:ledger-chain:v1",
        )
        entry = LedgerEntry(
            run_id=slot.run_id,
            position=position,
            record_key=record_key,
            record_tag=record_tag,
            previous_chain_tag=previous,
            chain_tag=chain_tag,
            record=record,
        )
        self._verify_entry(entry, expected_position=position, previous=previous)
        return entry

    def _verify_entry(
        self,
        entry: LedgerEntry,
        *,
        expected_position: int,
        previous: PayloadDigest | None,
    ) -> None:
        if entry.position != expected_position or entry.previous_chain_tag != previous:
            raise DigestVerificationError("ledger chain")
        if entry.record_key != self._record_key(entry.record):
            raise DigestVerificationError("ledger record key")
        if not self._integrity.verify(
            entry.record,
            entry.record_tag,
            domain="saliencegate:ledger-record:v1",
        ):
            raise DigestVerificationError("ledger record")
        chain_value = {
            "run_id": str(entry.run_id),
            "position": entry.position,
            "record_key": entry.record_key,
            "record_tag": entry.record_tag,
            "previous_chain_tag": entry.previous_chain_tag,
        }
        if not self._integrity.verify(
            chain_value,
            entry.chain_tag,
            domain="saliencegate:ledger-chain:v1",
        ):
            raise DigestVerificationError("ledger chain")

    @staticmethod
    def _head_value(
        run_id: UUID,
        entry_count: int,
        chain_tag: PayloadDigest,
        projection_tag: PayloadDigest,
    ) -> dict[str, object]:
        return {
            "run_id": str(run_id),
            "entry_count": entry_count,
            "chain_tag": chain_tag,
            "projection_tag": projection_tag,
        }

    def _head(
        self,
        slot: _RunSlot,
        entry: LedgerEntry,
        projection: Projection,
    ) -> LedgerHead:
        previous_projection_tag = (
            None if slot.ledger_head is None else slot.ledger_head.projection_tag
        )
        projection_tag = self._projection_checkpoint_tag(
            entry,
            projection,
            previous=previous_projection_tag,
        )
        head_tag = self._integrity.tag(
            self._head_value(
                slot.run_id,
                entry.position,
                entry.chain_tag,
                projection_tag,
            ),
            domain="saliencegate:ledger-head:v1",
        )
        return LedgerHead(
            run_id=slot.run_id,
            entry_count=entry.position,
            chain_tag=entry.chain_tag,
            projection_tag=projection_tag,
            head_tag=head_tag,
        )

    def _projection_checkpoint_tag(
        self,
        entry: LedgerEntry,
        projection: Projection,
        *,
        previous: PayloadDigest | None,
    ) -> PayloadDigest:
        return self._integrity.tag(
            {
                "previous_projection_tag": previous,
                "entry_chain_tag": entry.chain_tag,
                "ledger_position": entry.position,
                "ingestion_cursor": projection.ingestion_cursor,
                "memory_cursor": projection.memory_cursor,
                "counts": {
                    "events": len(projection.events_by_id),
                    "signals": len(projection.signals),
                    "decisions": len(projection.decisions),
                    "cycle_revisions": len(projection.cycle_history),
                    "memories": len(projection.memories),
                    "memory_revisions": len(projection.memory_history),
                    "interventions": len(projection.interventions),
                    "outcomes": len(projection.outcomes),
                    "delivery_revisions": len(projection.delivery_history),
                },
            },
            domain="saliencegate:projection-checkpoint:v1",
        )

    def _verify_head(self, slot: _RunSlot, *, verify_projection: bool = True) -> None:
        head = slot.ledger_head
        if head is None or not slot.ledger:
            raise DigestVerificationError("ledger head")
        if self._trusted_heads.get(slot.run_id) != head:
            raise DigestVerificationError("trusted ledger anchor")
        last = slot.ledger[-1]
        if (
            head.run_id != slot.run_id
            or head.entry_count != len(slot.ledger)
            or head.chain_tag != last.chain_tag
        ):
            raise DigestVerificationError("ledger head")
        if verify_projection and (
            slot.projection is None
            or self._trusted_projections.get(slot.run_id) is not slot.projection
        ):
            raise DigestVerificationError("projection head")
        if not self._integrity.verify(
            self._head_value(
                head.run_id,
                head.entry_count,
                head.chain_tag,
                head.projection_tag,
            ),
            head.head_tag,
            domain="saliencegate:ledger-head:v1",
        ):
            raise DigestVerificationError("ledger head")

    def _verify_ledger(self, slot: _RunSlot) -> None:
        self._verify_head(slot)
        previous: PayloadDigest | None = None
        for expected_position, entry in enumerate(slot.ledger, start=1):
            self._verify_entry(
                entry,
                expected_position=expected_position,
                previous=previous,
            )
            previous = entry.chain_tag

    @staticmethod
    def _record_key(record: LedgerRecord) -> str:
        if isinstance(record, TraceEvent):
            return f"trace_event:{record.event_id}"
        if isinstance(record, Signal):
            return f"signal:{record.signal_id}"
        if isinstance(record, InvocationDecision):
            return f"invocation_decision:{record.decision_id}"
        if isinstance(record, CycleRecord):
            return f"cycle:{record.cycle_id}:{record.revision}"
        if isinstance(record, InterventionOutcome):
            return f"intervention_outcome:{record.outcome_id}"
        if isinstance(record, DeliveryRecord):
            return f"delivery:{record.delivery_id}:{record.revision}"
        raise ProjectionInvariantError("unsupported ledger record key")

    @staticmethod
    def _draft_values(draft: NormalizedTraceEventDraft) -> dict[str, object]:
        return {name: getattr(draft, name) for name in NormalizedTraceEventDraft.model_fields}

    def _validate_draft(self, draft: object) -> NormalizedTraceEventDraft:
        if type(draft) is not NormalizedTraceEventDraft:
            raise InvalidAppendTypeError(type(draft))
        try:
            validated = validate_normalized_trace_event_draft(self._draft_values(draft))
            return NormalizedTraceEventDraft.model_validate_json(
                validated.model_dump_json(warnings=False)
            )
        except (AttributeError, ValidationError):
            pass
        raise InvalidDraftError()

    def _validate_metadata(self, draft: NormalizedTraceEventDraft) -> None:
        metadata = {
            "source_event_id": draft.source_event_id,
            "source_adapter": draft.source_adapter,
        }
        result = self._redactor.redact_payload(metadata)
        changed = tuple(
            field_name
            for field_name, value in metadata.items()
            if result.payload.root[field_name] != value
        )
        reserved = tuple(
            field_name
            for field_name, is_reserved in (
                (
                    "source_event_id",
                    draft.source_event_id.casefold().startswith("saliencegate:"),
                ),
                (
                    "source_adapter",
                    draft.source_adapter.casefold() == "saliencegate.repository",
                ),
                ("trust_label", draft.trust_label is TrustLabel.TRUSTED_RUNTIME),
            )
            if is_reserved
        )
        changed = tuple(dict.fromkeys((*changed, *reserved)))
        if changed:
            raise UnsafeEventMetadataError(changed)

    def _redact_draft(self, draft: NormalizedTraceEventDraft) -> RedactedTraceEventDraft:
        try:
            result = self._redactor.redact_event(
                draft,
                key=self._integrity.key,
                synthetic_benchmark=self._integrity.synthetic_benchmark,
            )
        except ValueError:
            pass
        else:
            if verify_redacted_event(
                result.event,
                redactor=self._redactor,
                key=self._integrity.key,
                synthetic_benchmark=self._integrity.synthetic_benchmark,
            ):
                return result.event
            raise DigestVerificationError("redacted event")
        raise InvalidDraftError()

    @staticmethod
    def _event_identity(
        event: TraceEvent | RedactedTraceEventDraft,
    ) -> dict[str, object]:
        fields = (
            "schema_version",
            "timestamp",
            "event_type",
            "phase",
            "payload",
            "payload_digest",
            "parent_ids",
            "source_adapter",
            "trust_label",
        )
        values = event.model_dump(mode="json")
        return {field_name: values[field_name] for field_name in fields}

    @classmethod
    def _differing_fields(
        cls,
        existing: TraceEvent,
        incoming: RedactedTraceEventDraft,
    ) -> tuple[str, ...]:
        existing_identity = cls._event_identity(existing)
        incoming_identity = cls._event_identity(incoming)
        return tuple(
            field_name
            for field_name in existing_identity
            if canonical_json(existing_identity[field_name])
            != canonical_json(incoming_identity[field_name])
        )

    def _trace_event(
        self,
        redacted: RedactedTraceEventDraft,
        *,
        sequence: int,
        event_id: UUID | None = None,
    ) -> TraceEvent:
        values = {
            name: getattr(redacted, name)
            for name in (
                "schema_version",
                "run_id",
                "source_event_id",
                "timestamp",
                "event_type",
                "phase",
                "payload",
                "payload_digest",
                "parent_ids",
                "source_adapter",
                "trust_label",
            )
        }
        identifier = self._new_id() if event_id is None else self._validate_run_id(event_id)
        values.update(event_id=identifier, sequence=sequence)
        try:
            return TraceEvent.model_validate(values)
        except ValidationError:
            pass
        raise ProjectionInvariantError("repository-created trace event is invalid")

    def _stage_entry(
        self, slot: _RunSlot, record: LedgerRecord, *, key: str
    ) -> tuple[
        LedgerEntry,
        Projection,
        LedgerHead,
    ]:
        entry = self._entry(slot, record, record_key=key)
        if slot.projection is None:  # pragma: no cover - slot invariant
            raise ProjectionInvariantError("run projection is unavailable")
        projected = apply_entry(slot.projection, entry)
        return entry, projected, self._head(slot, entry, projected)

    async def append(
        self,
        event: NormalizedTraceEventDraft,
        *,
        event_id: UUID | None = None,
    ) -> AppendReceipt:
        draft = self._validate_draft(event)
        if event_id is not None:
            event_id = self._validate_run_id(event_id)
        self._validate_metadata(draft)
        redacted = self._redact_draft(draft)
        slot = await self._acquire_append_slot(draft.run_id)
        try:
            async with slot.lock:
                return self._append_to_slot(slot, draft, redacted, event_id=event_id)
        finally:
            await self._release_append_slot(slot)

    async def append_event_if_head(
        self,
        event: NormalizedTraceEventDraft,
        *,
        event_id: UUID,
        expected_head: LedgerHead | None,
    ) -> AppendReceipt:
        draft = self._validate_draft(event)
        event_id = self._validate_run_id(event_id)
        copied_head = None if expected_head is None else self._copy_expected_head(expected_head)
        self._validate_metadata(draft)
        redacted = self._redact_draft(draft)
        slot = await self._acquire_append_slot(draft.run_id)
        try:
            async with slot.lock:
                self._require_expected_head(slot, copied_head)
                return self._append_conditional_to_slot(
                    slot,
                    draft,
                    redacted,
                    event_id=event_id,
                )
        finally:
            await self._release_append_slot(slot)

    @staticmethod
    def _copy_expected_head(value: object) -> LedgerHead:
        try:
            if type(value) is not LedgerHead:
                raise TypeError
            if type(value.run_id) is not UUID or type(value.entry_count) is not int:
                raise TypeError
            digests = (value.chain_tag, value.projection_tag, value.head_tag)
            if any(type(digest) is not PayloadDigest for digest in digests):
                raise TypeError
            if any(
                type(digest.algorithm) is not PayloadDigestAlgorithm
                or type(digest.value) is not str
                for digest in digests
            ):
                raise TypeError
            values = LedgerHead.model_dump(value, mode="python", warnings=False)
            return LedgerHead.model_validate(values)
        except Exception:
            raise LedgerHeadConflictError() from None

    def _require_expected_head(
        self,
        slot: _RunSlot,
        expected_head: LedgerHead | None,
    ) -> None:
        if slot.ledger_head is None and not slot.ledger:
            if expected_head is not None:
                raise LedgerHeadConflictError()
            return
        self._verify_head(slot)
        if expected_head is None or slot.ledger_head != expected_head:
            raise LedgerHeadConflictError()

    def _append_to_slot(
        self,
        slot: _RunSlot,
        draft: NormalizedTraceEventDraft,
        redacted: RedactedTraceEventDraft,
        *,
        event_id: UUID | None,
    ) -> AppendReceipt:
        if slot.projection is None:  # pragma: no cover - slot invariant
            raise ProjectionInvariantError("run projection is unavailable")
        if slot.ledger or slot.ledger_head is not None:
            self._verify_head(slot)
        existing = slot.projection.events_by_source.get(draft.source_event_id)
        if existing is not None:
            if canonical_json(self._event_identity(existing)) == canonical_json(
                self._event_identity(redacted)
            ):
                return self._duplicate_event_receipt(slot, existing)
            return self._record_collision(slot, existing, redacted)

        return self._append_new_event_to_slot(
            slot,
            draft,
            redacted,
            event_id=event_id,
        )

    def _append_conditional_to_slot(
        self,
        slot: _RunSlot,
        draft: NormalizedTraceEventDraft,
        redacted: RedactedTraceEventDraft,
        *,
        event_id: UUID,
        publish: bool = True,
    ) -> AppendReceipt:
        if slot.projection is None:  # pragma: no cover - slot invariant
            raise ProjectionInvariantError("run projection is unavailable")
        existing = slot.projection.events_by_source.get(draft.source_event_id)
        if existing is not None:
            if canonical_json(self._event_identity(existing)) == canonical_json(
                self._event_identity(redacted)
            ):
                return self._duplicate_event_receipt(slot, existing)
            raise RecordCollisionError("trace_event", existing.event_id)

        return self._append_new_event_to_slot(
            slot,
            draft,
            redacted,
            event_id=event_id,
            publish=publish,
        )

    @staticmethod
    def _duplicate_event_receipt(
        slot: _RunSlot,
        existing: TraceEvent,
    ) -> AppendReceipt:
        if slot.projection is None:  # pragma: no cover - slot invariant
            raise ProjectionInvariantError("run projection is unavailable")
        return _copy_model(
            AppendReceipt(
                disposition=AppendDisposition.DUPLICATE,
                event=existing,
                ledger_position=slot.projection.event_positions[existing.event_id],
                ingestion_cursor=slot.projection.ingestion_cursor,
            )
        )

    def _append_new_event_to_slot(
        self,
        slot: _RunSlot,
        draft: NormalizedTraceEventDraft,
        redacted: RedactedTraceEventDraft,
        *,
        event_id: UUID | None,
        publish: bool = True,
    ) -> AppendReceipt:
        if slot.projection is None:  # pragma: no cover - slot invariant
            raise ProjectionInvariantError("run projection is unavailable")

        if any(parent_id not in slot.projection.events_by_id for parent_id in draft.parent_ids):
            raise CrossRunReferenceError("event parent")

        trace_event = self._trace_event(
            redacted,
            sequence=slot.projection.ingestion_cursor + 1,
            event_id=event_id,
        )
        entry, projected, head = self._stage_entry(
            slot,
            trace_event,
            key=self._record_key(trace_event),
        )
        receipt = _copy_model(
            AppendReceipt(
                disposition=AppendDisposition.APPENDED,
                event=trace_event,
                ledger_position=entry.position,
                ingestion_cursor=projected.ingestion_cursor,
            )
        )
        slot.ledger += (entry,)
        slot.ledger_head = head
        slot.projection = projected
        if publish:
            self._trusted_heads[slot.run_id] = head
            self._trusted_projections[slot.run_id] = projected
        return receipt

    def _record_collision(
        self,
        slot: _RunSlot,
        existing: TraceEvent,
        incoming: RedactedTraceEventDraft,
    ) -> AppendReceipt:
        if slot.projection is None:  # pragma: no cover - slot invariant
            raise ProjectionInvariantError("run projection is unavailable")
        differing_fields = self._differing_fields(existing, incoming)
        fingerprint = self._integrity.tag(
            {
                "existing_event_id": str(existing.event_id),
                "incoming_identity": self._event_identity(incoming),
            },
            domain="saliencegate:source-collision:v1",
        )
        cached = slot.collision_receipts.get(fingerprint.value)
        if cached is not None:
            return _copy_model(cached)

        audit_draft = NormalizedTraceEventDraft(
            run_id=existing.run_id,
            source_event_id=f"saliencegate:collision:{fingerprint.value}",
            timestamp=incoming.timestamp,
            event_type=EventType.CONTROLLER_ERROR,
            phase=EventPhase.INTERNAL,
            payload={
                "reason_code": ReasonCode.SOURCE_EVENT_COLLISION.value,
                "existing_event_id": str(existing.event_id),
                "collision_fingerprint": fingerprint.model_dump(mode="json"),
                "differing_fields": differing_fields,
            },
            parent_ids=(existing.event_id,),
            source_adapter="saliencegate.repository",
            trust_label=TrustLabel.TRUSTED_RUNTIME,
        )
        redacted_audit = self._redact_draft(audit_draft)
        collision_event = self._trace_event(
            redacted_audit,
            sequence=slot.projection.ingestion_cursor + 1,
        )
        entry, projected, head = self._stage_entry(
            slot,
            collision_event,
            key=self._record_key(collision_event),
        )
        receipt = AppendReceipt(
            disposition=AppendDisposition.COLLISION,
            event=existing,
            ledger_position=slot.projection.event_positions[existing.event_id],
            ingestion_cursor=projected.ingestion_cursor,
            collision_event=collision_event,
        )
        public_receipt = _copy_model(receipt)
        slot.ledger += (entry,)
        slot.ledger_head = head
        slot.projection = projected
        self._trusted_heads[slot.run_id] = head
        self._trusted_projections[slot.run_id] = projected
        slot.collision_receipts = {
            **slot.collision_receipts,
            fingerprint.value: receipt,
        }
        return public_receipt

    async def _record_direct(
        self,
        record: RecordT,
        *,
        operation: str,
        record_id: UUID,
    ) -> LedgerReceipt:
        slot = await self._slot(record.run_id)
        async with slot.lock:
            return self._record_direct_locked(
                slot,
                record,
                operation=operation,
                record_id=record_id,
            )

    def _record_direct_locked(
        self,
        slot: _RunSlot,
        record: RecordT,
        *,
        operation: str,
        record_id: UUID,
        verify_head: bool = True,
        publish: bool = True,
    ) -> LedgerReceipt:
        if verify_head:
            self._verify_head(slot)
        key = (operation, record_id)
        existing = slot.direct_records.get(key)
        if existing is not None:
            if canonical_json(existing.record) != canonical_json(record):
                raise RecordCollisionError(operation, record_id)
            return _copy_model(existing.receipt.model_copy(update={"appended": False}))
        entry, projected, head = self._stage_entry(
            slot,
            record,
            key=self._record_key(record),
        )
        receipt = LedgerReceipt(
            appended=True,
            record_id=record_id,
            record_tag=entry.record_tag,
            ledger_position=entry.position,
            chain_tag=entry.chain_tag,
        )
        public_receipt = _copy_model(receipt)
        slot.ledger += (entry,)
        slot.ledger_head = head
        slot.projection = projected
        if publish:
            self._trusted_heads[slot.run_id] = head
            self._trusted_projections[slot.run_id] = projected
            slot.direct_records = {
                **slot.direct_records,
                key: _DirectRecord(record=record, receipt=receipt),
            }
        else:
            slot.direct_records[key] = _DirectRecord(record=record, receipt=receipt)
        return public_receipt

    @staticmethod
    def _validate_direct_record(
        record: object,
        expected_type: type[RecordT],
        operation: str,
    ) -> RecordT:
        if type(record) is not expected_type:
            raise InvalidRecordTypeError(operation, type(record))
        try:
            values = expected_type.model_dump(record, mode="python", warnings=False)
            validated = expected_type.model_validate(values)
            encoded = expected_type.model_dump_json(validated, warnings=False)
            return expected_type.model_validate_json(encoded)
        except Exception:
            raise InvalidRecordError(operation) from None

    @staticmethod
    def _validate_cycle_command(
        command: object,
        expected_type: type[ModelT],
        operation: str,
    ) -> ModelT:
        if type(command) is not expected_type:
            raise InvalidRecordTypeError(operation, type(command))
        try:
            values = command.model_dump(mode="python", warnings=False)
            validated = expected_type.model_validate(values)
            return expected_type.model_validate_json(validated.model_dump_json(warnings=False))
        except (AttributeError, ValidationError):
            raise InvalidRecordError(operation) from None

    def _validate_record_content(self, record: BaseModel, operation: str) -> None:
        try:
            values = record.model_dump(mode="json", warnings=False)
            redacted = self._redactor.redact_payload(values)
        except ValueError:
            pass
        else:
            if canonical_json(redacted.payload.root) == canonical_json(values):
                return
        raise UnsafeRecordContentError(operation)

    def _prepare_conditional_batch(
        self,
        operations: object,
    ) -> tuple[UUID, tuple[_PreparedConditionalOperation, ...]]:
        operation_name = "conditional_batch"
        if type(operations) is not tuple:
            raise InvalidRecordTypeError(operation_name, type(operations))
        if not 1 <= len(operations) <= MAX_CONDITIONAL_BATCH_OPERATIONS:
            raise InvalidRecordError(operation_name)

        event_count = 0
        signal_count = 0
        run_id: UUID | None = None
        copied_operations: list[ConditionalAppendOperation] = []
        prepared: list[_PreparedConditionalOperation] = []
        for operation in operations:
            if type(operation) is ConditionalEventAppend:
                event_count += 1
                if event_count > MAX_CONDITIONAL_BATCH_EVENTS:
                    raise InvalidRecordError(operation_name)
                try:
                    copied_event = ConditionalEventAppend.model_validate_json(
                        ConditionalEventAppend.model_dump_json(operation, warnings=False)
                    )
                    draft = self._validate_draft(copied_event.event)
                    event_id = self._validate_run_id(copied_event.event_id)
                except Exception:
                    raise InvalidRecordError(operation_name) from None
                self._validate_metadata(draft)
                redacted = self._redact_draft(draft)
                copied_operations.append(copied_event)
                prepared.append(
                    _PreparedConditionalEvent(
                        draft=draft,
                        redacted=redacted,
                        event_id=event_id,
                    )
                )
                operation_run_id = draft.run_id
            elif type(operation) is ConditionalSignalAppend:
                signal_count += 1
                if signal_count > MAX_CONDITIONAL_BATCH_SIGNALS:
                    raise InvalidRecordError(operation_name)
                try:
                    copied_signal = ConditionalSignalAppend.model_validate_json(
                        ConditionalSignalAppend.model_dump_json(operation, warnings=False)
                    )
                    signal = self._validate_direct_record(
                        copied_signal.signal,
                        Signal,
                        "signal",
                    )
                except Exception:
                    raise InvalidRecordError(operation_name) from None
                self._validate_record_content(signal, "signal")
                copied_operations.append(copied_signal)
                prepared.append(_PreparedConditionalSignal(signal=signal))
                operation_run_id = signal.run_id
            else:
                raise InvalidRecordTypeError(operation_name, type(operation))

            if run_id is None:
                run_id = operation_run_id
            elif operation_run_id != run_id:
                raise InvalidRecordError(operation_name)

        try:
            request_bytes = len(canonical_json(tuple(copied_operations)))
        except Exception:
            raise InvalidRecordError(operation_name) from None
        if request_bytes > MAX_CONDITIONAL_BATCH_REQUEST_BYTES:
            raise InvalidRecordError(operation_name)
        if run_id is None:  # pragma: no cover - guarded by the non-empty bound
            raise ProjectionInvariantError("conditional batch run is unavailable")
        return run_id, tuple(prepared)

    @staticmethod
    def _bounded_conditional_batch_receipt(
        *,
        initial_head: LedgerHead | None,
        receipts: tuple[AppendReceipt | LedgerReceipt, ...],
        final_head: LedgerHead,
    ) -> ConditionalBatchReceipt:
        try:
            return ConditionalBatchReceipt(
                initial_head=initial_head,
                receipts=receipts,
                final_head=final_head,
            )
        except ValidationError as error:
            if "canonical byte limit" in str(error):
                raise InvalidRecordError("conditional_batch") from None
            raise ProjectionInvariantError("conditional batch receipt is invalid") from None

    async def record_signal(self, signal: Signal) -> LedgerReceipt:
        validated = self._validate_direct_record(signal, Signal, "signal")
        self._validate_record_content(validated, "signal")
        return await self._record_direct(
            validated,
            operation="signal",
            record_id=validated.signal_id,
        )

    async def record_signal_if_head(
        self,
        signal: Signal,
        *,
        expected_head: LedgerHead,
    ) -> LedgerReceipt:
        validated = self._validate_direct_record(signal, Signal, "signal")
        copied_head = self._copy_expected_head(expected_head)
        self._validate_record_content(validated, "signal")
        try:
            slot = await self._slot(validated.run_id)
        except RunNotFoundError:
            raise LedgerHeadConflictError() from None
        async with slot.lock:
            self._require_expected_head(slot, copied_head)
            return self._record_direct_locked(
                slot,
                validated,
                operation="signal",
                record_id=validated.signal_id,
            )

    async def append_records_if_head(
        self,
        operations: tuple[ConditionalAppendOperation, ...],
        *,
        expected_head: LedgerHead | None,
    ) -> ConditionalBatchReceipt:
        run_id, prepared = self._prepare_conditional_batch(operations)
        copied_head = None if expected_head is None else self._copy_expected_head(expected_head)
        slot = await self._acquire_append_slot(run_id)
        try:
            async with slot.lock:
                self._require_expected_head(slot, copied_head)
                initial_head = None if slot.ledger_head is None else _copy_model(slot.ledger_head)
                shadow = self._shadow_slot(slot)
                receipts: list[AppendReceipt | LedgerReceipt] = []
                mutated = False
                for index, operation in enumerate(prepared, start=1):
                    if isinstance(operation, _PreparedConditionalEvent):
                        event_receipt = self._append_conditional_to_slot(
                            shadow,
                            operation.draft,
                            operation.redacted,
                            event_id=operation.event_id,
                            publish=False,
                        )
                        receipts.append(event_receipt)
                        mutated = mutated or (
                            event_receipt.disposition is AppendDisposition.APPENDED
                        )
                    else:
                        signal_receipt = self._record_direct_locked(
                            shadow,
                            operation.signal,
                            operation="signal",
                            record_id=operation.signal.signal_id,
                            verify_head=False,
                            publish=False,
                        )
                        receipts.append(signal_receipt)
                        mutated = mutated or signal_receipt.appended
                    if index % 128 == 0 and index != len(prepared):
                        await asyncio.sleep(0)

                if shadow.projection is None or shadow.ledger_head is None:
                    raise ProjectionInvariantError("conditional batch state is unavailable")
                if mutated:
                    self._verify_conditional_shadow(slot, shadow)
                else:
                    validate_complete_projection(shadow.projection)
                public_receipt = self._bounded_conditional_batch_receipt(
                    initial_head=initial_head,
                    receipts=tuple(receipts),
                    final_head=shadow.ledger_head,
                )
                if mutated:
                    self._commit_conditional_shadow(slot, shadow)
                return public_receipt
        finally:
            await self._release_append_slot_safely(slot)

    async def record_invocation_decision(
        self,
        decision: InvocationDecision,
    ) -> LedgerReceipt:
        validated = self._validate_direct_record(
            decision,
            InvocationDecision,
            "invocation_decision",
        )
        self._validate_record_content(validated, "invocation_decision")
        return await self._record_direct(
            validated,
            operation="invocation_decision",
            record_id=validated.decision_id,
        )

    async def record_outcome(self, outcome: InterventionOutcome) -> LedgerReceipt:
        validated = self._validate_direct_record(
            outcome,
            InterventionOutcome,
            "intervention_outcome",
        )
        self._validate_record_content(validated, "intervention_outcome")
        return await self._record_direct(
            validated,
            operation="intervention_outcome",
            record_id=validated.outcome_id,
        )

    @staticmethod
    def _cycle_base(
        slot: _RunSlot,
        cycle_identifier: str,
        expected_revision: int,
        expected_state: CycleState | None = None,
    ) -> CycleRecord:
        if slot.projection is None:  # pragma: no cover - slot invariant
            raise ProjectionInvariantError("run projection is unavailable")
        base = slot.projection.cycle_history.get((cycle_identifier, expected_revision))
        if base is None:
            latest = slot.projection.cycles.get(cycle_identifier)
            raise CycleRevisionConflictError(
                expected_revision,
                None if latest is None else latest.revision,
            )
        if expected_state is not None and base.state is not expected_state:
            raise InvalidCycleStateError("cycle transition", expected_state)
        return base

    def _record_cycle_locked(
        self,
        slot: _RunSlot,
        record: CycleRecord,
        *,
        publish: bool,
        delivery: DeliveryRecord | None = None,
    ) -> CycleReceipt:
        key = (record.cycle_id, record.revision)
        existing = slot.cycle_records.get(key)
        if existing is not None:
            if canonical_json(existing.record) != canonical_json(record):
                raise CycleConflictError()
            return _copy_model(existing.receipt.model_copy(update={"appended": False}))
        entry, projected, head = self._stage_entry(
            slot,
            record,
            key=self._record_key(record),
        )
        receipt = CycleReceipt(
            appended=True,
            cycle=record,
            record_tag=entry.record_tag,
            ledger_position=entry.position,
            chain_tag=entry.chain_tag,
            budget_snapshot=projected_budget_snapshot(projected),
            delivery=delivery,
        )
        public_receipt = _copy_model(receipt)
        slot.ledger += (entry,)
        slot.ledger_head = head
        slot.projection = projected
        slot.cycle_records = {
            **slot.cycle_records,
            key: _CycleRevision(record=record, receipt=receipt),
        }
        if publish:
            self._trusted_heads[slot.run_id] = head
            self._trusted_projections[slot.run_id] = projected
        return public_receipt

    @staticmethod
    def _shadow_slot(slot: _RunSlot) -> _RunSlot:
        return _RunSlot(
            run_id=slot.run_id,
            ledger=slot.ledger,
            ledger_head=slot.ledger_head,
            projection=slot.projection,
            direct_records=dict(slot.direct_records),
            cycle_records=dict(slot.cycle_records),
            delivery_records=dict(slot.delivery_records),
            collision_receipts=dict(slot.collision_receipts),
        )

    @staticmethod
    def _publish_shadow(slot: _RunSlot, shadow: _RunSlot) -> None:
        slot.ledger = shadow.ledger
        slot.ledger_head = shadow.ledger_head
        slot.projection = shadow.projection
        slot.direct_records = shadow.direct_records
        slot.cycle_records = shadow.cycle_records
        slot.delivery_records = shadow.delivery_records
        slot.collision_receipts = shadow.collision_receipts

    def _verify_conditional_shadow(self, slot: _RunSlot, shadow: _RunSlot) -> None:
        """Authenticate a staged suffix without replaying the trusted prefix."""

        if slot.projection is None or shadow.projection is None or shadow.ledger_head is None:
            raise ProjectionInvariantError("conditional batch state is unavailable")
        base_count = len(slot.ledger)
        if (
            shadow.run_id != slot.run_id
            or len(shadow.ledger) <= base_count
            or shadow.ledger[:base_count] != slot.ledger
        ):
            raise DigestVerificationError("conditional batch candidate")

        projected = slot.projection
        previous_chain = None if slot.ledger_head is None else slot.ledger_head.chain_tag
        projection_tag = None if slot.ledger_head is None else slot.ledger_head.projection_tag
        expected_direct_records = dict(slot.direct_records)
        for expected_position, entry in enumerate(
            shadow.ledger[base_count:],
            start=base_count + 1,
        ):
            if entry.run_id != slot.run_id:
                raise DigestVerificationError("conditional batch candidate")
            self._verify_entry(
                entry,
                expected_position=expected_position,
                previous=previous_chain,
            )
            projected = apply_entry(projected, entry)
            projection_tag = self._projection_checkpoint_tag(
                entry,
                projected,
                previous=projection_tag,
            )
            direct = self._direct_record_from_entry(entry)
            if direct is not None:
                expected_direct_records[direct[0]] = direct[1]
            previous_chain = entry.chain_tag

        head = shadow.ledger_head
        if (
            head.run_id != slot.run_id
            or head.entry_count != len(shadow.ledger)
            or head.chain_tag != previous_chain
            or head.projection_tag != projection_tag
            or not self._integrity.verify(
                self._head_value(
                    head.run_id,
                    head.entry_count,
                    head.chain_tag,
                    head.projection_tag,
                ),
                head.head_tag,
                domain="saliencegate:ledger-head:v1",
            )
        ):
            raise DigestVerificationError("conditional batch candidate")
        validate_complete_projection(projected)
        if (
            projected != shadow.projection
            or expected_direct_records != shadow.direct_records
            or shadow.cycle_records != slot.cycle_records
            or shadow.delivery_records != slot.delivery_records
            or shadow.collision_receipts != slot.collision_receipts
        ):
            raise DigestVerificationError("conditional batch candidate")

    def _commit_conditional_shadow(self, slot: _RunSlot, shadow: _RunSlot) -> None:
        """Publish all staged references, restoring the exact prior state on failure."""

        if shadow.ledger_head is None or shadow.projection is None:  # pragma: no cover
            raise ProjectionInvariantError("conditional batch state is unavailable")
        previous_slot_state = (
            slot.ledger,
            slot.ledger_head,
            slot.projection,
            slot.direct_records,
            slot.cycle_records,
            slot.delivery_records,
            slot.collision_receipts,
        )
        missing = object()
        previous_head: LedgerHead | object = self._trusted_heads.get(slot.run_id, missing)
        previous_projection: Projection | object = self._trusted_projections.get(
            slot.run_id,
            missing,
        )
        try:
            self._publish_shadow(slot, shadow)
            self._trusted_heads[slot.run_id] = shadow.ledger_head
            self._trusted_projections[slot.run_id] = shadow.projection
        except BaseException:
            (
                slot.ledger,
                slot.ledger_head,
                slot.projection,
                slot.direct_records,
                slot.cycle_records,
                slot.delivery_records,
                slot.collision_receipts,
            ) = previous_slot_state
            if previous_head is missing:
                self._trusted_heads.pop(slot.run_id, None)
            else:
                self._trusted_heads[slot.run_id] = previous_head  # type: ignore[assignment]
            if previous_projection is missing:
                self._trusted_projections.pop(slot.run_id, None)
            else:
                self._trusted_projections[slot.run_id] = previous_projection  # type: ignore[assignment]
            raise

    def _record_delivery_locked(
        self,
        slot: _RunSlot,
        record: DeliveryRecord,
        *,
        publish: bool,
    ) -> DeliveryTransitionReceipt:
        key = (record.delivery_id, record.revision)
        existing = slot.delivery_records.get(key)
        if existing is not None:
            if canonical_json(existing.record) != canonical_json(record):
                raise DeliveryRevisionConflictError(record.revision, record.revision)
            return _copy_model(existing.receipt.model_copy(update={"appended": False}))
        entry, projected, head = self._stage_entry(
            slot,
            record,
            key=self._record_key(record),
        )
        receipt = DeliveryTransitionReceipt(
            appended=True,
            delivery=record,
            record_tag=entry.record_tag,
            ledger_position=entry.position,
            chain_tag=entry.chain_tag,
        )
        public_receipt = _copy_model(receipt)
        slot.ledger += (entry,)
        slot.ledger_head = head
        slot.projection = projected
        slot.delivery_records = {
            **slot.delivery_records,
            key: _DeliveryRevision(record=record, receipt=receipt),
        }
        if publish:
            self._trusted_heads[slot.run_id] = head
            self._trusted_projections[slot.run_id] = projected
        return public_receipt

    @staticmethod
    def _delivery_base(
        slot: _RunSlot,
        delivery_id: UUID,
        expected_revision: int,
    ) -> DeliveryRecord:
        if slot.projection is None:  # pragma: no cover - slot invariant
            raise ProjectionInvariantError("run projection is unavailable")
        latest = slot.projection.deliveries.get(delivery_id)
        if latest is None:
            raise DeliveryNotFoundError()
        if latest.revision != expected_revision:
            raise DeliveryRevisionConflictError(expected_revision, latest.revision)
        return latest

    @staticmethod
    def _transition_delivery(
        base: DeliveryRecord,
        operation: str,
        **updates: object,
    ) -> DeliveryRecord:
        values = base.model_dump(mode="python", warnings=False)
        values.update(revision=base.revision + 1, **updates)
        try:
            record = DeliveryRecord.model_validate(values)
            return DeliveryRecord.model_validate_json(record.model_dump_json(warnings=False))
        except ValidationError:
            raise InvalidRecordError(operation) from None

    @staticmethod
    def _delivery_retry(
        slot: _RunSlot,
        *,
        delivery_id: UUID,
        expected_revision: int,
        candidate: DeliveryRecord,
    ) -> DeliveryTransitionReceipt | None:
        if slot.projection is None:  # pragma: no cover - slot invariant
            raise ProjectionInvariantError("run projection is unavailable")
        latest = slot.projection.deliveries.get(delivery_id)
        if latest is None or latest.revision != expected_revision + 1:
            return None
        if canonical_json(latest) != canonical_json(candidate):
            return None
        cached = slot.delivery_records.get((delivery_id, latest.revision))
        if cached is None:  # pragma: no cover - cache invariant
            raise ProjectionInvariantError("delivery receipt cache is unavailable")
        return _copy_model(cached.receipt.model_copy(update={"appended": False}))

    @staticmethod
    def _attempt_envelope(
        projection: Projection,
        delivery: DeliveryRecord,
    ) -> DeliveryAttemptEnvelope:
        intervention = projection.interventions.get(delivery.intervention_id)
        if (
            delivery.state is not DeliveryState.ATTEMPTING
            or delivery.claim_id is None
            or delivery.attempt_id is None
            or intervention is None
            or intervention.rendered_text is None
        ):
            raise ProjectionInvariantError("delivery attempt has no grounded reminder")
        return DeliveryAttemptEnvelope(
            delivery_id=delivery.delivery_id,
            run_id=delivery.run_id,
            cycle_id=delivery.cycle_id,
            intervention_id=delivery.intervention_id,
            rendered_text_digest=delivery.rendered_text_digest,
            claim_id=delivery.claim_id,
            attempt_id=delivery.attempt_id,
            attempt_number=delivery.attempt_count,
            target_request_id=delivery.target_request_id,
            target=delivery.target,
            adapter_id=delivery.adapter_id,
            adapter_deduplicates=delivery.adapter_deduplicates,
            adapter_deduplication_guarantee=(delivery.adapter_deduplication_guarantee),
            adapter_supports_pre_action=delivery.adapter_supports_pre_action,
            adapter_contract_version=delivery.adapter_contract_version,
            adapter_capabilities_digest=delivery.adapter_capabilities_digest,
            rendered_text=intervention.rendered_text,
        )

    @staticmethod
    def _transition_cycle(
        base: CycleRecord,
        operation: str,
        **updates: object,
    ) -> CycleRecord:
        values = base.model_dump(mode="python")
        values.update(updates)
        reservation = updates.get("budget_reservation", base.budget_reservation)
        settlement = updates.get("budget_settlement", base.budget_settlement)
        if type(reservation) is BudgetAmounts and type(settlement) is BudgetAmounts:
            fields = (
                "model_calls",
                "input_tokens",
                "output_tokens",
                "canonical_token_equivalents",
                "latency_us",
                "interventions",
                "schema_repairs",
            )
            if any(
                getattr(settlement, field_name) > getattr(reservation, field_name)
                for field_name in fields
            ):
                raise ProjectionInvariantError("cycle settlement exceeds its reservation")
        try:
            cycle = CycleRecord.model_validate(values)
            return CycleRecord.model_validate_json(cycle.model_dump_json(warnings=False))
        except ValidationError:
            raise InvalidRecordError(operation) from None

    async def begin_cycle(self, command: BeginCycle) -> CycleReceipt:
        validated = self._validate_cycle_command(command, BeginCycle, "begin_cycle")
        self._validate_record_content(validated, "begin_cycle")
        slot = await self._slot(validated.run_id)
        async with slot.lock:
            self._verify_head(slot)
            if slot.projection is None:  # pragma: no cover - slot invariant
                raise ProjectionInvariantError("run projection is unavailable")
            decision = slot.projection.decisions.get(validated.invocation_decision_id)
            if decision is None:
                raise CrossRunReferenceError("cycle invocation decision")
            existing_beginnings = tuple(
                cycle
                for (identifier, revision), cycle in slot.projection.cycle_history.items()
                if revision == 1
                and cycle.invocation_decision_id == validated.invocation_decision_id
            )
            if len(existing_beginnings) > 1:  # pragma: no cover - projection invariant
                raise ProjectionInvariantError("invocation decision has multiple cycles")
            if existing_beginnings:
                existing = existing_beginnings[0]
                if (
                    existing.grounding_version != validated.grounding_version
                    or existing.grounding_configuration != validated.grounding_configuration
                    or existing.grounding_configuration_digest
                    != validated.grounding_configuration_digest
                    or existing.requested_delivery_target is not validated.requested_delivery_target
                ):
                    raise CycleConflictError()
                retry = self._transition_cycle(
                    existing,
                    "begin_cycle",
                    created_at=validated.created_at,
                    updated_at=validated.created_at,
                )
                return self._record_cycle_locked(slot, retry, publish=True)
            first_event_sequence = slot.projection.memory_cursor + 1
            if decision.event_sequence < first_event_sequence:
                raise ProjectionInvariantError("cycle has no unprocessed event range")
            identifier = cycle_id(
                validated.run_id,
                first_event_sequence,
                decision.event_sequence,
                decision.policy_version,
                decision.configuration_digest,
                validated.grounding_version,
                validated.grounding_configuration_digest,
                validated.requested_delivery_target,
            )
            for active in slot.projection.cycles.values():
                if (
                    active.state
                    in (
                        CycleState.PENDING,
                        CycleState.RESERVED,
                        CycleState.RUNNING,
                    )
                    and active.cycle_id != identifier
                ):
                    raise CycleConflictError()
            cycle = CycleRecord(
                cycle_id=identifier,
                run_id=validated.run_id,
                revision=1,
                invocation_decision_id=validated.invocation_decision_id,
                policy_version=decision.policy_version,
                configuration_digest=decision.configuration_digest,
                grounding_version=validated.grounding_version,
                grounding_configuration=validated.grounding_configuration,
                grounding_configuration_digest=validated.grounding_configuration_digest,
                requested_delivery_target=validated.requested_delivery_target,
                first_event_sequence=first_event_sequence,
                last_event_sequence=decision.event_sequence,
                state=CycleState.PENDING,
                created_at=validated.created_at,
                updated_at=validated.created_at,
            )
            return self._record_cycle_locked(slot, cycle, publish=True)

    async def reserve_cycle(self, command: ReserveCycle) -> CycleReceipt:
        validated = self._validate_cycle_command(command, ReserveCycle, "reserve_cycle")
        self._validate_record_content(validated, "reserve_cycle")
        slot = await self._slot(validated.run_id)
        async with slot.lock:
            self._verify_head(slot)
            base = self._cycle_base(
                slot,
                validated.cycle_id,
                validated.expected_revision,
                CycleState.PENDING,
            )
            cycle = self._transition_cycle(
                base,
                "reserve_cycle",
                revision=base.revision + 1,
                state=CycleState.RESERVED,
                budget_reservation=validated.reservation,
                updated_at=validated.updated_at,
            )
            return self._record_cycle_locked(slot, cycle, publish=True)

    async def mark_cycle_running(self, command: StartCycle) -> CycleReceipt:
        validated = self._validate_cycle_command(command, StartCycle, "mark_cycle_running")
        self._validate_record_content(validated, "mark_cycle_running")
        slot = await self._slot(validated.run_id)
        async with slot.lock:
            self._verify_head(slot)
            base = self._cycle_base(
                slot,
                validated.cycle_id,
                validated.expected_revision,
                CycleState.RESERVED,
            )
            cycle = self._transition_cycle(
                base,
                "mark_cycle_running",
                revision=base.revision + 1,
                state=CycleState.RUNNING,
                batch_digest=validated.batch_digest,
                updated_at=validated.updated_at,
            )
            return self._record_cycle_locked(slot, cycle, publish=True)

    async def commit_cycle(self, command: CommitCycle) -> CycleReceipt:
        validated = self._validate_cycle_command(command, CommitCycle, "commit_cycle")
        self._validate_record_content(validated, "commit_cycle")
        slot = await self._slot(validated.run_id)
        async with slot.lock:
            self._verify_head(slot)
            base = self._cycle_base(
                slot,
                validated.cycle_id,
                validated.expected_revision,
                CycleState.RUNNING,
            )
            if (
                validated.intervention.grounding_version != base.grounding_version
                or validated.intervention.grounding_configuration_digest
                != base.grounding_configuration_digest
                or canonical_json(validated.intervention.grounding_configuration)
                != canonical_json(base.grounding_configuration)
            ):
                raise ProjectionInvariantError(
                    "grounded intervention failed authoritative verification"
                )
            cycle = self._transition_cycle(
                base,
                "commit_cycle",
                revision=base.revision + 1,
                state=CycleState.COMMITTED,
                budget_settlement=validated.settlement,
                model_call_digests=validated.model_call_digests,
                model_call_latencies_us=validated.model_call_latencies_us,
                validated_delta=validated.validated_delta,
                memory_id_assignments=validated.memory_id_assignments,
                intervention=validated.intervention,
                selector_provenance=validated.selector_provenance,
                updated_at=validated.updated_at,
            )
            existing_revision = slot.cycle_records.get((cycle.cycle_id, cycle.revision))
            if existing_revision is not None:
                if canonical_json(existing_revision.record) != canonical_json(cycle):
                    raise CycleConflictError()
                existing_delivery = existing_revision.receipt.delivery
                plan = validated.delivery
                plan_matches = (plan is None) == (existing_delivery is None)
                if plan is not None and existing_delivery is not None:
                    plan_matches = plan_matches and (
                        plan.target_request_id == existing_delivery.target_request_id
                        and plan.adapter_id == existing_delivery.adapter_id
                        and plan.adapter_deduplicates is existing_delivery.adapter_deduplicates
                        and plan.adapter_deduplication_guarantee
                        is existing_delivery.adapter_deduplication_guarantee
                        and plan.adapter_supports_pre_action
                        is existing_delivery.adapter_supports_pre_action
                        and plan.adapter_contract_version
                        == existing_delivery.adapter_contract_version
                        and plan.adapter_capabilities_digest
                        == existing_delivery.adapter_capabilities_digest
                    )
                if not plan_matches:
                    raise CycleConflictError()
                return _copy_model(existing_revision.receipt.model_copy(update={"appended": False}))

            delivery: DeliveryRecord | None = None
            if validated.delivery is not None:
                target = validated.intervention.delivery_target
                if target is None:  # pragma: no cover - command invariant
                    raise ProjectionInvariantError("reminder delivery target is unavailable")
                delivery = DeliveryRecord(
                    delivery_id=derive_delivery_id(
                        cycle.run_id,
                        cycle.cycle_id,
                        validated.intervention.intervention_id,
                        validated.delivery.target_request_id,
                        target,
                        validated.delivery.adapter_id,
                        validated.delivery.adapter_capabilities_digest,
                        canonical_digest(validated.intervention.rendered_text),
                    ),
                    run_id=cycle.run_id,
                    revision=1,
                    cycle_id=cycle.cycle_id,
                    intervention_id=validated.intervention.intervention_id,
                    rendered_text_digest=canonical_digest(validated.intervention.rendered_text),
                    target_request_id=validated.delivery.target_request_id,
                    target=target,
                    state=DeliveryState.PENDING,
                    attempt_count=0,
                    adapter_id=validated.delivery.adapter_id,
                    adapter_deduplicates=validated.delivery.adapter_deduplicates,
                    adapter_deduplication_guarantee=(
                        validated.delivery.adapter_deduplication_guarantee
                    ),
                    adapter_supports_pre_action=(validated.delivery.adapter_supports_pre_action),
                    adapter_contract_version=(validated.delivery.adapter_contract_version),
                    adapter_capabilities_digest=(validated.delivery.adapter_capabilities_digest),
                    created_at=validated.updated_at,
                    updated_at=validated.updated_at,
                )
            shadow = self._shadow_slot(slot)
            cycle_receipt = self._record_cycle_locked(
                shadow,
                cycle,
                publish=False,
                delivery=delivery,
            )
            if delivery is not None:
                self._record_delivery_locked(shadow, delivery, publish=False)
            if shadow.projection is None:  # pragma: no cover - shadow invariant
                raise ProjectionInvariantError("cycle commit shadow is incomplete")
            validate_complete_projection(shadow.projection)
            self._publish_shadow(slot, shadow)
            if slot.ledger_head is None or slot.projection is None:  # pragma: no cover
                raise ProjectionInvariantError("cycle commit shadow is incomplete")
            self._trusted_heads[slot.run_id] = slot.ledger_head
            self._trusted_projections[slot.run_id] = slot.projection
            return _copy_model(cycle_receipt)

    async def fail_cycle(self, command: FailCycle) -> CycleReceipt:
        validated = self._validate_cycle_command(command, FailCycle, "fail_cycle")
        self._validate_record_content(validated, "fail_cycle")
        slot = await self._slot(validated.run_id)
        async with slot.lock:
            self._verify_head(slot)
            base = self._cycle_base(
                slot,
                validated.cycle_id,
                validated.expected_revision,
            )
            if base.state not in (
                CycleState.PENDING,
                CycleState.RESERVED,
                CycleState.RUNNING,
            ):
                raise InvalidCycleStateError("fail_cycle", CycleState.RUNNING)
            settlement = validated.settlement
            if base.state is CycleState.PENDING:
                if (
                    settlement is not None
                    or validated.model_call_digests
                    or validated.model_call_latencies_us
                ):
                    raise CycleConflictError()
            else:
                if settlement is None:
                    raise CycleConflictError()
                if base.state is CycleState.RESERVED and (
                    validated.model_call_digests or validated.model_call_latencies_us
                ):
                    raise CycleConflictError()
            if validated.reason is ReasonCode.FAILED_UNKNOWN_COST:
                if base.state is not CycleState.RUNNING:
                    raise CycleConflictError()
                if settlement is not None and settlement != base.budget_reservation:
                    raise CycleConflictError()
                settlement = base.budget_reservation
            cycle = self._transition_cycle(
                base,
                "fail_cycle",
                revision=base.revision + 1,
                state=CycleState.FAILED,
                budget_settlement=settlement,
                model_call_digests=validated.model_call_digests,
                model_call_latencies_us=validated.model_call_latencies_us,
                failure_reason=validated.reason,
                updated_at=validated.updated_at,
            )
            return self._record_cycle_locked(slot, cycle, publish=True)

    async def preview_memory_delta(self, command: PreviewMemoryDelta) -> MemoryDeltaPreview:
        validated = self._validate_cycle_command(
            command,
            PreviewMemoryDelta,
            "preview_memory_delta",
        )
        self._validate_record_content(validated, "preview_memory_delta")
        slot = await self._slot(validated.run_id)
        async with slot.lock:
            self._verify_head(slot)
            if slot.projection is None:  # pragma: no cover - slot invariant
                raise ProjectionInvariantError("run projection is unavailable")
            source = projection_snapshot(
                slot.projection,
                self._integrity,
                ledger_position=len(slot.ledger),
            )
            if (
                validated.expected_ledger_position != source.ledger_position
                or validated.expected_ingestion_cursor != source.ingestion_cursor
                or validated.expected_memory_cursor != source.memory_cursor
                or validated.expected_projection_digest != source.projection_digest
            ):
                raise PreviewConflictError()
            projected = preview_projected_memory_delta(
                slot.projection,
                validated.delta,
                validated.memory_id_assignments,
                last_event_sequence=validated.last_event_sequence,
            )
            preview = projection_snapshot(
                projected,
                self._integrity,
                ledger_position=source.ledger_position,
            )
            return _copy_model(
                MemoryDeltaPreview(
                    schema_version="memory-delta-preview/v1",
                    run_id=validated.run_id,
                    command_digest=validated.command_digest,
                    source_ledger_position=source.ledger_position,
                    source_ingestion_cursor=source.ingestion_cursor,
                    source_memory_cursor=source.memory_cursor,
                    source_projection_digest=source.projection_digest,
                    records=preview.records,
                    current_private_status_id=projected.current_private_status_id,
                    preview_projection_digest=preview.projection_digest,
                )
            )

    async def budget_snapshot(self, run_id: UUID) -> BudgetSnapshot:
        run_id = self._validate_run_id(run_id)
        slot = await self._slot(run_id)
        async with slot.lock:
            self._verify_head(slot)
            if slot.projection is None:  # pragma: no cover - slot invariant
                raise ProjectionInvariantError("run projection is unavailable")
            return _copy_model(projected_budget_snapshot(slot.projection))

    async def recover_cycles(
        self,
        run_id: UUID,
        *,
        recovered_at: datetime,
    ) -> CycleRecoveryReceipt:
        run_id = self._validate_run_id(run_id)
        if (
            type(recovered_at) is not datetime
            or recovered_at.tzinfo is None
            or recovered_at.utcoffset() != timedelta(0)
        ):
            raise InvalidRecoveryTimeError()
        recovered_at = recovered_at.astimezone(UTC)
        slot = await self._slot(run_id)
        async with slot.lock:
            self._verify_head(slot)
            if slot.projection is None:  # pragma: no cover - slot invariant
                raise ProjectionInvariantError("run projection is unavailable")
            resumable_pending = tuple(
                cycle
                for _, cycle in sorted(slot.projection.cycles.items())
                if cycle.state is CycleState.PENDING
            )
            resumable_reserved = tuple(
                cycle
                for _, cycle in sorted(slot.projection.cycles.items())
                if cycle.state is CycleState.RESERVED
            )
            running = tuple(
                cycle
                for _, cycle in sorted(slot.projection.cycles.items())
                if cycle.state is CycleState.RUNNING
            )
            shadow = self._shadow_slot(slot)
            failed_receipts: list[CycleReceipt] = []
            for cycle in running:
                values = cycle.model_dump(mode="python")
                values.update(
                    revision=cycle.revision + 1,
                    state=CycleState.FAILED,
                    budget_settlement=cycle.budget_reservation,
                    failure_reason=ReasonCode.FAILED_UNKNOWN_COST,
                    updated_at=max(recovered_at, cycle.updated_at),
                )
                failed = CycleRecord.model_validate(values)
                failed_receipts.append(self._record_cycle_locked(shadow, failed, publish=False))
            receipt = CycleRecoveryReceipt(
                run_id=run_id,
                resumable_pending=resumable_pending,
                resumable_reserved=resumable_reserved,
                failed_unknown_cost=tuple(failed_receipts),
            )
            public_receipt = _copy_model(receipt)
            if failed_receipts:
                self._publish_shadow(slot, shadow)
                if shadow.ledger_head is None or shadow.projection is None:  # pragma: no cover
                    raise ProjectionInvariantError("recovery shadow state is incomplete")
                self._trusted_heads[run_id] = shadow.ledger_head
                self._trusted_projections[run_id] = shadow.projection
            return public_receipt

    async def delivery(self, run_id: UUID, delivery_id: UUID) -> DeliveryRecord:
        run_id = self._validate_run_id(run_id)
        if type(delivery_id) is not UUID or delivery_id.version != 4:
            raise DeliveryNotFoundError()
        slot = await self._slot(run_id)
        async with slot.lock:
            self._verify_head(slot)
            if slot.projection is None:  # pragma: no cover - slot invariant
                raise ProjectionInvariantError("run projection is unavailable")
            record = slot.projection.deliveries.get(delivery_id)
            if record is None:
                raise DeliveryNotFoundError()
            return _copy_model(record)

    async def claim_delivery(
        self,
        command: ClaimDelivery,
    ) -> DeliveryTransitionReceipt:
        validated = self._validate_cycle_command(command, ClaimDelivery, "claim_delivery")
        self._validate_record_content(validated, "claim_delivery")
        slot = await self._slot(validated.run_id)
        async with slot.lock:
            self._verify_head(slot)
            if slot.projection is None:  # pragma: no cover - slot invariant
                raise ProjectionInvariantError("run projection is unavailable")
            previous = slot.projection.delivery_history.get(
                (validated.delivery_id, validated.expected_revision)
            )
            if previous is not None and previous.state in (
                DeliveryState.PENDING,
                DeliveryState.UNKNOWN,
            ):
                if previous.state is DeliveryState.UNKNOWN and not previous.adapter_deduplicates:
                    raise InvalidDeliveryStateError("claim_delivery")
                candidate = self._transition_delivery(
                    previous,
                    "claim_delivery",
                    state=DeliveryState.CLAIMED,
                    claim_id=validated.claim_id,
                    attempt_id=None,
                    receipt=None,
                    outcome=None,
                    reason_code=None,
                    updated_at=validated.updated_at,
                )
                retry = self._delivery_retry(
                    slot,
                    delivery_id=validated.delivery_id,
                    expected_revision=validated.expected_revision,
                    candidate=candidate,
                )
                if retry is not None:
                    return retry
            base = self._delivery_base(
                slot,
                validated.delivery_id,
                validated.expected_revision,
            )
            if base.state not in (DeliveryState.PENDING, DeliveryState.UNKNOWN):
                raise InvalidDeliveryStateError("claim_delivery")
            if base.state is DeliveryState.UNKNOWN and not base.adapter_deduplicates:
                raise InvalidDeliveryStateError("claim_delivery")
            record = self._transition_delivery(
                base,
                "claim_delivery",
                state=DeliveryState.CLAIMED,
                claim_id=validated.claim_id,
                attempt_id=None,
                receipt=None,
                outcome=None,
                reason_code=None,
                updated_at=validated.updated_at,
            )
            return self._record_delivery_locked(slot, record, publish=True)

    async def begin_delivery_attempt(
        self,
        command: BeginDeliveryAttempt,
    ) -> DeliveryAttemptReceipt:
        validated = self._validate_cycle_command(
            command,
            BeginDeliveryAttempt,
            "begin_delivery_attempt",
        )
        self._validate_record_content(validated, "begin_delivery_attempt")
        slot = await self._slot(validated.run_id)
        async with slot.lock:
            self._verify_head(slot)
            if slot.projection is None:  # pragma: no cover - slot invariant
                raise ProjectionInvariantError("run projection is unavailable")
            previous = slot.projection.delivery_history.get(
                (validated.delivery_id, validated.expected_revision)
            )
            if previous is not None and previous.state is DeliveryState.CLAIMED:
                if previous.claim_id != validated.claim_id:
                    raise DeliveryOwnershipError()
                latest = slot.projection.deliveries.get(validated.delivery_id)
                if (
                    latest is not None
                    and latest.revision == validated.expected_revision + 1
                    and latest.state is DeliveryState.ATTEMPTING
                    and latest.attempt_id != validated.attempt_id
                ):
                    raise DeliveryOwnershipError()
                candidate = self._transition_delivery(
                    previous,
                    "begin_delivery_attempt",
                    state=DeliveryState.ATTEMPTING,
                    attempt_count=previous.attempt_count + 1,
                    attempt_id=validated.attempt_id,
                    updated_at=validated.updated_at,
                )
                retry = self._delivery_retry(
                    slot,
                    delivery_id=validated.delivery_id,
                    expected_revision=validated.expected_revision,
                    candidate=candidate,
                )
                if retry is not None:
                    return DeliveryAttemptReceipt(
                        **retry.model_dump(mode="python", warnings=False),
                        envelope=None,
                    )
            base = self._delivery_base(
                slot,
                validated.delivery_id,
                validated.expected_revision,
            )
            if base.state is not DeliveryState.CLAIMED:
                raise InvalidDeliveryStateError("begin_delivery_attempt")
            if base.claim_id != validated.claim_id:
                raise DeliveryOwnershipError()
            record = self._transition_delivery(
                base,
                "begin_delivery_attempt",
                state=DeliveryState.ATTEMPTING,
                attempt_count=base.attempt_count + 1,
                attempt_id=validated.attempt_id,
                updated_at=validated.updated_at,
            )
            envelope = self._attempt_envelope(slot.projection, record)
            receipt = self._record_delivery_locked(slot, record, publish=True)
            return _copy_model(
                DeliveryAttemptReceipt(
                    **receipt.model_dump(mode="python", warnings=False),
                    envelope=envelope,
                )
            )

    async def complete_delivery(
        self,
        command: CompleteDelivery,
    ) -> DeliveryTransitionReceipt:
        validated = self._validate_cycle_command(
            command,
            CompleteDelivery,
            "complete_delivery",
        )
        self._validate_record_content(validated, "complete_delivery")
        slot = await self._slot(validated.run_id)
        async with slot.lock:
            self._verify_head(slot)
            if slot.projection is None:  # pragma: no cover - slot invariant
                raise ProjectionInvariantError("run projection is unavailable")
            target_state = (
                DeliveryState.DELIVERED
                if validated.outcome is DeliveryOutcome.DELIVERED
                else DeliveryState.FAILED
            )
            reason = (
                ReasonCode.DELIVERY_SUCCEEDED
                if validated.outcome is DeliveryOutcome.DELIVERED
                else ReasonCode.DELIVERY_FAILED
            )
            receipt_data = (
                {"provider_receipt_id": validated.provider_receipt_id}
                if validated.provider_receipt_id is not None
                else None
            )
            previous = slot.projection.delivery_history.get(
                (validated.delivery_id, validated.expected_revision)
            )
            if previous is not None and previous.state in (
                DeliveryState.ATTEMPTING,
                DeliveryState.UNKNOWN,
            ):
                if (
                    previous.claim_id != validated.claim_id
                    or previous.attempt_id != validated.attempt_id
                ):
                    raise DeliveryOwnershipError()
                candidate = self._transition_delivery(
                    previous,
                    "complete_delivery",
                    state=target_state,
                    receipt=receipt_data,
                    outcome=validated.outcome,
                    reason_code=reason,
                    updated_at=validated.updated_at,
                )
                retry = self._delivery_retry(
                    slot,
                    delivery_id=validated.delivery_id,
                    expected_revision=validated.expected_revision,
                    candidate=candidate,
                )
                if retry is not None:
                    return retry
            base = self._delivery_base(
                slot,
                validated.delivery_id,
                validated.expected_revision,
            )
            if base.state not in (DeliveryState.ATTEMPTING, DeliveryState.UNKNOWN):
                raise InvalidDeliveryStateError("complete_delivery")
            if base.claim_id != validated.claim_id or base.attempt_id != validated.attempt_id:
                raise DeliveryOwnershipError()
            record = self._transition_delivery(
                base,
                "complete_delivery",
                state=target_state,
                receipt=receipt_data,
                outcome=validated.outcome,
                reason_code=reason,
                updated_at=validated.updated_at,
            )
            return self._record_delivery_locked(slot, record, publish=True)

    async def mark_delivery_unknown(
        self,
        command: MarkDeliveryUnknown,
    ) -> DeliveryTransitionReceipt:
        validated = self._validate_cycle_command(
            command,
            MarkDeliveryUnknown,
            "mark_delivery_unknown",
        )
        self._validate_record_content(validated, "mark_delivery_unknown")
        slot = await self._slot(validated.run_id)
        async with slot.lock:
            self._verify_head(slot)
            if slot.projection is None:  # pragma: no cover - slot invariant
                raise ProjectionInvariantError("run projection is unavailable")
            previous = slot.projection.delivery_history.get(
                (validated.delivery_id, validated.expected_revision)
            )
            if previous is not None and previous.state is DeliveryState.ATTEMPTING:
                if (
                    previous.claim_id != validated.claim_id
                    or previous.attempt_id != validated.attempt_id
                ):
                    raise DeliveryOwnershipError()
                candidate = self._transition_delivery(
                    previous,
                    "mark_delivery_unknown",
                    state=DeliveryState.UNKNOWN,
                    receipt=None,
                    outcome=DeliveryOutcome.UNKNOWN,
                    reason_code=ReasonCode.DELIVERY_UNKNOWN,
                    updated_at=validated.updated_at,
                )
                retry = self._delivery_retry(
                    slot,
                    delivery_id=validated.delivery_id,
                    expected_revision=validated.expected_revision,
                    candidate=candidate,
                )
                if retry is not None:
                    return retry
            base = self._delivery_base(
                slot,
                validated.delivery_id,
                validated.expected_revision,
            )
            if base.state is not DeliveryState.ATTEMPTING:
                raise InvalidDeliveryStateError("mark_delivery_unknown")
            if base.claim_id != validated.claim_id or base.attempt_id != validated.attempt_id:
                raise DeliveryOwnershipError()
            record = self._transition_delivery(
                base,
                "mark_delivery_unknown",
                state=DeliveryState.UNKNOWN,
                receipt=None,
                outcome=DeliveryOutcome.UNKNOWN,
                reason_code=ReasonCode.DELIVERY_UNKNOWN,
                updated_at=validated.updated_at,
            )
            return self._record_delivery_locked(slot, record, publish=True)

    async def reject_delivery(
        self,
        command: RejectDelivery,
    ) -> DeliveryTransitionReceipt:
        validated = self._validate_cycle_command(
            command,
            RejectDelivery,
            "reject_delivery",
        )
        self._validate_record_content(validated, "reject_delivery")
        slot = await self._slot(validated.run_id)
        async with slot.lock:
            self._verify_head(slot)
            if slot.projection is None:  # pragma: no cover - slot invariant
                raise ProjectionInvariantError("run projection is unavailable")
            previous = slot.projection.delivery_history.get(
                (validated.delivery_id, validated.expected_revision)
            )
            if previous is not None and previous.state in (
                DeliveryState.PENDING,
                DeliveryState.CLAIMED,
            ):
                if previous.state is DeliveryState.PENDING:
                    if validated.claim_id is not None:
                        raise DeliveryOwnershipError()
                elif previous.claim_id != validated.claim_id:
                    raise DeliveryOwnershipError()
                candidate = self._transition_delivery(
                    previous,
                    "reject_delivery",
                    state=DeliveryState.REJECTED,
                    receipt=None,
                    outcome=DeliveryOutcome.REFUSED,
                    reason_code=validated.reason_code,
                    updated_at=validated.updated_at,
                )
                retry = self._delivery_retry(
                    slot,
                    delivery_id=validated.delivery_id,
                    expected_revision=validated.expected_revision,
                    candidate=candidate,
                )
                if retry is not None:
                    return retry
            base = self._delivery_base(
                slot,
                validated.delivery_id,
                validated.expected_revision,
            )
            if base.state not in (DeliveryState.PENDING, DeliveryState.CLAIMED):
                raise InvalidDeliveryStateError("reject_delivery")
            if base.state is DeliveryState.PENDING:
                if validated.claim_id is not None:
                    raise DeliveryOwnershipError()
            elif base.claim_id != validated.claim_id:
                raise DeliveryOwnershipError()
            record = self._transition_delivery(
                base,
                "reject_delivery",
                state=DeliveryState.REJECTED,
                receipt=None,
                outcome=DeliveryOutcome.REFUSED,
                reason_code=validated.reason_code,
                updated_at=validated.updated_at,
            )
            return self._record_delivery_locked(slot, record, publish=True)

    async def recover_deliveries(
        self,
        run_id: UUID,
        *,
        recovered_at: datetime,
    ) -> DeliveryRecoveryReceipt:
        run_id = self._validate_run_id(run_id)
        if (
            type(recovered_at) is not datetime
            or recovered_at.tzinfo is None
            or recovered_at.utcoffset() != timedelta(0)
        ):
            raise InvalidRecoveryTimeError()
        recovered_at = recovered_at.astimezone(UTC)
        slot = await self._slot(run_id)
        async with slot.lock:
            self._verify_head(slot)
            if slot.projection is None:  # pragma: no cover - slot invariant
                raise ProjectionInvariantError("run projection is unavailable")
            attempting = tuple(
                delivery
                for _, delivery in sorted(
                    slot.projection.deliveries.items(), key=lambda item: str(item[0])
                )
                if delivery.state is DeliveryState.ATTEMPTING
            )
            if any(recovered_at < delivery.updated_at for delivery in attempting):
                raise InvalidRecoveryTimeError()
            shadow = self._shadow_slot(slot)
            marked_unknown: list[DeliveryTransitionReceipt] = []
            for delivery in attempting:
                unknown = self._transition_delivery(
                    delivery,
                    "recover_deliveries",
                    state=DeliveryState.UNKNOWN,
                    receipt=None,
                    outcome=DeliveryOutcome.UNKNOWN,
                    reason_code=ReasonCode.DELIVERY_UNKNOWN,
                    updated_at=recovered_at,
                )
                marked_unknown.append(self._record_delivery_locked(shadow, unknown, publish=False))
            if marked_unknown:
                self._publish_shadow(slot, shadow)
                if slot.ledger_head is None or slot.projection is None:  # pragma: no cover
                    raise ProjectionInvariantError("delivery recovery shadow is incomplete")
                self._trusted_heads[run_id] = slot.ledger_head
                self._trusted_projections[run_id] = slot.projection
            projection = shadow.projection if marked_unknown else slot.projection
            if projection is None:  # pragma: no cover - slot invariant
                raise ProjectionInvariantError("run projection is unavailable")
            latest = tuple(
                delivery
                for _, delivery in sorted(
                    projection.deliveries.items(), key=lambda item: str(item[0])
                )
            )
            return _copy_model(
                DeliveryRecoveryReceipt(
                    run_id=run_id,
                    marked_unknown=tuple(marked_unknown),
                    resumable_pending=tuple(
                        item for item in latest if item.state is DeliveryState.PENDING
                    ),
                    resumable_claimed=tuple(
                        item for item in latest if item.state is DeliveryState.CLAIMED
                    ),
                    retryable_unknown=tuple(
                        item
                        for item in latest
                        if item.state is DeliveryState.UNKNOWN and item.adapter_deduplicates
                    ),
                    non_retryable_unknown=tuple(
                        item
                        for item in latest
                        if item.state is DeliveryState.UNKNOWN and not item.adapter_deduplicates
                    ),
                )
            )

    async def ledger(self, run_id: UUID) -> tuple[LedgerEntry, ...]:
        run_id = self._validate_run_id(run_id)
        slot = await self._slot(run_id)
        async with slot.lock:
            self._verify_ledger(slot)
            return tuple(_copy_model(entry) for entry in slot.ledger)

    async def ledger_head(self, run_id: UUID) -> LedgerHead:
        run_id = self._validate_run_id(run_id)
        slot = await self._slot(run_id)
        async with slot.lock:
            self._verify_ledger(slot)
            if slot.ledger_head is None:  # pragma: no cover - verified ledger invariant
                raise DigestVerificationError("durable ledger head")
            return _copy_model(slot.ledger_head)

    def _replay_run(
        self,
        entries: tuple[LedgerEntry, ...],
        head: LedgerHead,
    ) -> _ReplayedRun:
        if type(entries) is not tuple or not entries or type(head) is not LedgerHead:
            raise RebuildError()
        if any(type(entry) is not LedgerEntry for entry in entries):
            raise RebuildError()
        try:
            copied_entries = tuple(_copy_model(entry) for entry in entries)
            copied_head = _copy_model(head)
        except (AttributeError, TypeError, ValidationError):
            raise RebuildError() from None
        run_id = copied_head.run_id
        if (
            copied_head.entry_count != len(copied_entries)
            or copied_entries[-1].run_id != run_id
            or copied_head.chain_tag != copied_entries[-1].chain_tag
        ):
            raise DigestVerificationError("durable ledger head")
        if not self._integrity.verify(
            self._head_value(
                run_id,
                copied_head.entry_count,
                copied_head.chain_tag,
                copied_head.projection_tag,
            ),
            copied_head.head_tag,
            domain="saliencegate:ledger-head:v1",
        ):
            raise DigestVerificationError("durable ledger head")

        projected = empty_projection(run_id)
        previous: PayloadDigest | None = None
        projection_tag: PayloadDigest | None = None
        direct_records: dict[tuple[str, UUID], _DirectRecord] = {}
        cycle_receipt_seeds: dict[tuple[str, int], _CycleReceiptSeed] = {}
        delivery_records: dict[tuple[UUID, int], _DeliveryRevision] = {}
        try:
            for expected_position, entry in enumerate(copied_entries, start=1):
                if entry.run_id != run_id:
                    raise DigestVerificationError("durable ledger run")
                self._verify_entry(
                    entry,
                    expected_position=expected_position,
                    previous=previous,
                )
                projected = apply_entry(projected, entry)
                projection_tag = self._projection_checkpoint_tag(
                    entry,
                    projected,
                    previous=projection_tag,
                )
                previous = entry.chain_tag
                direct = self._direct_record_from_entry(entry)
                if direct is not None:
                    direct_records[direct[0]] = direct[1]
                cycle_receipt_seed = self._cycle_record_from_entry(entry, projected)
                if cycle_receipt_seed is not None:
                    cycle_receipt_seeds[cycle_receipt_seed[0]] = cycle_receipt_seed[1]
                delivery_revision = self._delivery_record_from_entry(entry)
                if delivery_revision is not None:
                    delivery_records[delivery_revision[0]] = delivery_revision[1]
            validate_complete_projection(projected)
            cycle_records = self._attach_cycle_deliveries(cycle_receipt_seeds, projected)
            if projection_tag != copied_head.projection_tag:
                raise DigestVerificationError("durable projection head")
            collision_receipts = self._collision_receipts(projected)
            digests = projection_digests(
                projected,
                self._integrity,
                ledger_position=len(copied_entries),
            )
        except DigestVerificationError:
            raise
        except (RepositoryError, ValidationError):
            raise RebuildError() from None
        state = _VerifiedRunState(
            run_id=run_id,
            ledger=copied_entries,
            ledger_head=copied_head,
            projection=projected,
            digests=digests,
        )
        return _ReplayedRun(
            state=state,
            direct_records=direct_records,
            cycle_records=cycle_records,
            delivery_records=delivery_records,
            collision_receipts=collision_receipts,
        )

    async def _verified_state(self, run_id: UUID) -> _VerifiedRunState:
        """Replay before exporting detached, authenticated durable-backend state."""

        run_id = self._validate_run_id(run_id)
        slot = await self._slot(run_id)
        async with slot.lock:
            self._verify_ledger(slot)
            if slot.ledger_head is None or slot.projection is None:  # pragma: no cover
                raise DigestVerificationError("verified run state")
            replayed = self._replay_run(slot.ledger, slot.ledger_head)
            try:
                live_digests = projection_digests(
                    slot.projection,
                    self._integrity,
                    ledger_position=len(slot.ledger),
                )
            except (AttributeError, TypeError, ValueError, ValidationError):
                raise DigestVerificationError("live projection") from None
            if live_digests != replayed.state.digests:
                raise DigestVerificationError("live projection")
            return replayed.state

    async def _restore_run(
        self,
        entries: tuple[LedgerEntry, ...],
        head: LedgerHead,
    ) -> _VerifiedRunState:
        """Verify and atomically install one durable ledger into an empty run slot."""

        replayed = self._replay_run(entries, head)
        state = replayed.state
        detached = self._replay_run(state.ledger, state.ledger_head).state
        slot = _RunSlot(
            run_id=state.run_id,
            ledger=state.ledger,
            ledger_head=state.ledger_head,
            projection=state.projection,
            direct_records=replayed.direct_records,
            cycle_records=replayed.cycle_records,
            delivery_records=replayed.delivery_records,
            collision_receipts=replayed.collision_receipts,
        )
        async with self._slots_lock:
            if state.run_id in self._slots or state.run_id in self._trusted_heads:
                raise ProjectionInvariantError("durable run is already installed")
            self._slots[state.run_id] = slot
            self._trusted_heads[state.run_id] = state.ledger_head
            self._trusted_projections[state.run_id] = state.projection
        return detached

    async def search(self, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        if type(query) is not MemoryQuery:
            raise InvalidRecordTypeError("search", type(query))
        validation_failed = False
        try:
            values = {name: getattr(query, name) for name in MemoryQuery.model_fields}
            validated_query = MemoryQuery.model_validate(values)
            validated_query = MemoryQuery.model_validate_json(
                validated_query.model_dump_json(warnings=False)
            )
        except (AttributeError, ValidationError):
            validation_failed = True
        if validation_failed:
            raise InvalidQueryError()
        slot = await self._slot(validated_query.run_id)
        async with slot.lock:
            if slot.projection is None:  # pragma: no cover - slot invariant
                raise ProjectionInvariantError("run projection is unavailable")
            self._verify_head(slot)
            return tuple(
                _copy_model(hit) for hit in search_projection(slot.projection, validated_query)
            )

    async def snapshot(self, run_id: UUID) -> MemorySnapshot:
        run_id = self._validate_run_id(run_id)
        slot = await self._slot(run_id)
        async with slot.lock:
            if slot.projection is None:  # pragma: no cover - slot invariant
                raise ProjectionInvariantError("run projection is unavailable")
            self._verify_head(slot)
            return _copy_model(
                projection_snapshot(
                    slot.projection,
                    self._integrity,
                    ledger_position=len(slot.ledger),
                )
            )

    async def rebuild(self, run_id: UUID) -> RebuildReceipt:
        run_id = self._validate_run_id(run_id)
        slot = await self._slot(run_id)
        async with slot.lock:
            if slot.projection is None:  # pragma: no cover - slot invariant
                raise ProjectionInvariantError("run projection is unavailable")
            before: ProjectionDigests | None = None
            with suppress(AttributeError, TypeError, ValueError):
                before = projection_digests(
                    slot.projection,
                    self._integrity,
                    ledger_position=len(slot.ledger),
                )
            failure = False
            direct_records: dict[tuple[str, UUID], _DirectRecord] = {}
            cycle_receipt_seeds: dict[tuple[str, int], _CycleReceiptSeed] = {}
            delivery_records: dict[tuple[UUID, int], _DeliveryRevision] = {}
            try:
                self._verify_head(slot, verify_projection=False)
                projected = empty_projection(run_id)
                previous: PayloadDigest | None = None
                projection_tag: PayloadDigest | None = None
                for expected_position, entry in enumerate(slot.ledger, start=1):
                    self._verify_entry(
                        entry,
                        expected_position=expected_position,
                        previous=previous,
                    )
                    projected = apply_entry(projected, entry)
                    projection_tag = self._projection_checkpoint_tag(
                        entry,
                        projected,
                        previous=projection_tag,
                    )
                    previous = entry.chain_tag
                    direct = self._direct_record_from_entry(entry)
                    if direct is not None:
                        direct_records[direct[0]] = direct[1]
                    cycle_receipt_seed = self._cycle_record_from_entry(entry, projected)
                    if cycle_receipt_seed is not None:
                        cycle_receipt_seeds[cycle_receipt_seed[0]] = cycle_receipt_seed[1]
                    delivery_revision = self._delivery_record_from_entry(entry)
                    if delivery_revision is not None:
                        delivery_records[delivery_revision[0]] = delivery_revision[1]
                validate_complete_projection(projected)
                cycle_records = self._attach_cycle_deliveries(cycle_receipt_seeds, projected)
                after = projection_digests(
                    projected,
                    self._integrity,
                    ledger_position=len(slot.ledger),
                )
                if slot.ledger_head is None or projection_tag != slot.ledger_head.projection_tag:
                    raise DigestVerificationError("projection head")
                collision_receipts = self._collision_receipts(projected)
            except (RepositoryError, ValidationError):
                failure = True
            if failure:
                raise RebuildError()
            equivalent = before is not None and before == after
            receipt = RebuildReceipt(
                run_id=run_id,
                entries_replayed=len(slot.ledger),
                before=before,
                after=after,
                equivalent=equivalent,
            )
            slot.projection = projected
            self._trusted_projections[run_id] = projected
            slot.direct_records = direct_records
            slot.cycle_records = cycle_records
            slot.delivery_records = delivery_records
            slot.collision_receipts = collision_receipts
            return receipt

    @staticmethod
    def _direct_record_from_entry(
        entry: LedgerEntry,
    ) -> tuple[tuple[str, UUID], _DirectRecord] | None:
        record = entry.record
        if isinstance(record, Signal):
            operation = "signal"
            record_id = record.signal_id
        elif isinstance(record, InvocationDecision):
            operation = "invocation_decision"
            record_id = record.decision_id
        elif isinstance(record, InterventionOutcome):
            operation = "intervention_outcome"
            record_id = record.outcome_id
        else:
            return None
        receipt = LedgerReceipt(
            appended=True,
            record_id=record_id,
            record_tag=entry.record_tag,
            ledger_position=entry.position,
            chain_tag=entry.chain_tag,
        )
        return (operation, record_id), _DirectRecord(record=record, receipt=receipt)

    @staticmethod
    def _cycle_record_from_entry(
        entry: LedgerEntry,
        projection: Projection,
    ) -> tuple[tuple[str, int], _CycleReceiptSeed] | None:
        record = entry.record
        if not isinstance(record, CycleRecord):
            return None
        seed = _CycleReceiptSeed(
            record=record,
            record_tag=entry.record_tag,
            ledger_position=entry.position,
            chain_tag=entry.chain_tag,
            budget_snapshot=projected_budget_snapshot(projection),
        )
        return (record.cycle_id, record.revision), seed

    @staticmethod
    def _delivery_record_from_entry(
        entry: LedgerEntry,
    ) -> tuple[tuple[UUID, int], _DeliveryRevision] | None:
        record = entry.record
        if not isinstance(record, DeliveryRecord):
            return None
        receipt = DeliveryTransitionReceipt(
            appended=True,
            delivery=record,
            record_tag=entry.record_tag,
            ledger_position=entry.position,
            chain_tag=entry.chain_tag,
        )
        return (record.delivery_id, record.revision), _DeliveryRevision(
            record=record,
            receipt=receipt,
        )

    @staticmethod
    def _attach_cycle_deliveries(
        cycle_receipt_seeds: dict[tuple[str, int], _CycleReceiptSeed],
        projection: Projection,
    ) -> dict[tuple[str, int], _CycleRevision]:
        pending_by_cycle = {
            delivery.cycle_id: delivery
            for (delivery_id, revision), delivery in projection.delivery_history.items()
            if revision == 1 and delivery.delivery_id == delivery_id
        }
        attached: dict[tuple[str, int], _CycleRevision] = {}
        for key, seed in cycle_receipt_seeds.items():
            delivery = pending_by_cycle.get(seed.record.cycle_id)
            if seed.record.state is not CycleState.COMMITTED:
                delivery = None
            receipt = CycleReceipt(
                appended=True,
                cycle=seed.record,
                record_tag=seed.record_tag,
                ledger_position=seed.ledger_position,
                chain_tag=seed.chain_tag,
                budget_snapshot=seed.budget_snapshot,
                delivery=delivery,
            )
            attached[key] = _CycleRevision(record=seed.record, receipt=receipt)
        return attached

    def _collision_receipts(self, projection: Projection) -> dict[str, AppendReceipt]:
        receipts: dict[str, AppendReceipt] = {}
        for event in projection.events_by_sequence.values():
            if not event.source_event_id.startswith("saliencegate:collision:"):
                continue
            if (
                event.event_type is not EventType.CONTROLLER_ERROR
                or event.phase is not EventPhase.INTERNAL
                or event.source_adapter != "saliencegate.repository"
                or event.trust_label is not TrustLabel.TRUSTED_RUNTIME
                or event.payload.get("reason_code") != ReasonCode.SOURCE_EVENT_COLLISION.value
                or set(event.payload)
                != {
                    "reason_code",
                    "existing_event_id",
                    "collision_fingerprint",
                    "differing_fields",
                }
            ):
                raise ProjectionInvariantError("invalid collision audit event")
            fingerprint = event.payload.get("collision_fingerprint")
            differing_fields = event.payload.get("differing_fields")
            if not isinstance(fingerprint, Mapping):
                raise ProjectionInvariantError("invalid collision audit event")
            algorithm = fingerprint.get("algorithm")
            value = fingerprint.get("value")
            if (
                set(fingerprint) != {"algorithm", "value"}
                or algorithm != event.payload_digest.algorithm.value
                or not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                or not isinstance(differing_fields, tuple)
                or not differing_fields
                or not all(isinstance(field_name, str) for field_name in differing_fields)
                or len(set(differing_fields)) != len(differing_fields)
                or any(
                    field_name
                    not in {
                        "schema_version",
                        "timestamp",
                        "event_type",
                        "phase",
                        "payload",
                        "payload_digest",
                        "parent_ids",
                        "source_adapter",
                        "trust_label",
                    }
                    for field_name in differing_fields
                )
                or len(event.parent_ids) != 1
            ):
                raise ProjectionInvariantError("invalid collision audit event")
            if not isinstance(value, str):  # pragma: no cover - narrowed above
                raise ProjectionInvariantError("invalid collision audit event")
            if event.source_event_id != f"saliencegate:collision:{value}":
                raise ProjectionInvariantError("invalid collision audit event")
            existing = projection.events_by_id.get(event.parent_ids[0])
            if (
                existing is None
                or event.payload.get("existing_event_id") != str(existing.event_id)
                or value in receipts
            ):
                raise ProjectionInvariantError("invalid collision audit event")
            receipts[value] = AppendReceipt(
                disposition=AppendDisposition.COLLISION,
                event=existing,
                ledger_position=projection.event_positions[existing.event_id],
                ingestion_cursor=event.sequence,
                collision_event=event,
            )
        return receipts
