from __future__ import annotations

import asyncio
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from saliencegate.domain import (
    BudgetAmounts,
    BudgetLimits,
    BudgetSnapshot,
    ClaimKind,
    ConstraintStatus,
    CycleRecord,
    CycleState,
    DeduplicationGuarantee,
    DeliveryOutcome,
    DeliveryRecord,
    DeliveryState,
    DeliveryTarget,
    EventPhase,
    EventType,
    EvidenceReference,
    EvidenceSource,
    InterventionAction,
    InterventionDecision,
    InterventionOutcome,
    InvocationDecision,
    MemoryCreate,
    MemoryDelta,
    MemoryIdAssignment,
    MemoryKind,
    MemoryUpdate,
    NormalizedTraceEventDraft,
    OutcomeEvidenceMode,
    PayloadDigest,
    PayloadDigestAlgorithm,
    ReasonCode,
    RedactedTraceEventDraft,
    RepeatedErrorStatus,
    Signal,
    SignalType,
    TraceEvent,
    TrustLabel,
    canonical_json,
)
from saliencegate.intervention.claims import InterventionProposal, ProposedClaim
from saliencegate.intervention.grounding import (
    GroundingConfig,
    GroundingContext,
    GroundingPipeline,
    GroundingState,
    resolve_grounding_configuration,
)
from saliencegate.intervention.rendering import RenderingConfig
from saliencegate.ports.repository import (
    AppendDisposition,
    AppendReceipt,
    BeginCycle,
    BeginDeliveryAttempt,
    ClaimDelivery,
    CommitCycle,
    CompleteDelivery,
    ConditionalBatchReceipt,
    ConditionalEventAppend,
    ConditionalSignalAppend,
    CrossRunReferenceError,
    CycleConflictError,
    CycleReceipt,
    CycleRevisionConflictError,
    DeliveryOwnershipError,
    DeliveryRevisionConflictError,
    EnqueueDelivery,
    FailCycle,
    InvalidAppendTypeError,
    InvalidDeliveryStateError,
    InvalidDraftError,
    InvalidQueryError,
    InvalidRecordError,
    InvalidRecordTypeError,
    InvalidRunIdError,
    LedgerHead,
    LedgerHeadConflictError,
    LedgerReceipt,
    MarkDeliveryUnknown,
    MemoryQuery,
    ProjectionInvariantError,
    RecordCollisionError,
    RejectDelivery,
    ReserveCycle,
    RevisionConflictError,
    RunNotFoundError,
    RunRepository,
    StartCycle,
    UnsafeEventMetadataError,
    UnsafeRecordContentError,
)

RUN_A = UUID("00000000-0000-4000-8000-000000000101")
RUN_B = UUID("00000000-0000-4000-8000-000000000102")
PARENT_ID = UUID("00000000-0000-4000-8000-000000000103")
DECISION_ID = UUID("00000000-0000-4000-8000-000000000112")
OUTCOME_ID = UUID("00000000-0000-4000-8000-000000000113")
INTERVENTION_ID = UUID("00000000-0000-4000-8000-000000000114")
CONDITIONAL_EVENT_ID_A = UUID("00000000-0000-4000-8000-000000000121")
CONDITIONAL_EVENT_ID_B = UUID("00000000-0000-4000-8000-000000000122")
CONDITIONAL_SIGNAL_ID = UUID("00000000-0000-4000-8000-000000000123")
NOW = datetime(2026, 7, 11, 12, 30, tzinfo=UTC)
DELIVERY_CLAIM_A = UUID("00000000-0000-4000-8000-000000000381")
DELIVERY_CLAIM_B = UUID("00000000-0000-4000-8000-000000000382")
DELIVERY_CLAIM_C = UUID("00000000-0000-4000-8000-000000000385")
DELIVERY_ATTEMPT_A = UUID("00000000-0000-4000-8000-000000000383")
DELIVERY_ATTEMPT_B = UUID("00000000-0000-4000-8000-000000000384")


class AliasUUID(UUID):
    def __hash__(self) -> int:
        return hash(RUN_A)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UUID)


class AliasLedgerHead(LedgerHead):
    pass


class AliasPayloadDigest(PayloadDigest):
    pass


RepositoryFactory = Callable[[], RunRepository]


def event_draft(
    *,
    run_id: UUID = RUN_A,
    source_event_id: str = "source-1",
    payload: dict[str, object] | None = None,
    timestamp: datetime = NOW,
    event_type: EventType = EventType.OBSERVATION,
    phase: EventPhase = EventPhase.POST_ACTION,
    parent_ids: tuple[UUID, ...] = (),
    source_adapter: str = "fixture-adapter",
    trust_label: TrustLabel = TrustLabel.UNTRUSTED_TOOL_OUTPUT,
) -> NormalizedTraceEventDraft:
    return NormalizedTraceEventDraft(
        run_id=run_id,
        source_event_id=source_event_id,
        timestamp=timestamp,
        event_type=event_type,
        phase=phase,
        payload={"message": "safe"} if payload is None else payload,
        parent_ids=parent_ids,
        source_adapter=source_adapter,
        trust_label=trust_label,
    )


def invocation_decision() -> InvocationDecision:
    limits = BudgetLimits(
        model_calls=10,
        input_tokens=1_000,
        output_tokens=1_000,
        canonical_token_equivalents=2_000,
        latency_us=1_000_000,
        max_call_latency_us=500_000,
        interventions=5,
        schema_repairs=2,
    )
    return InvocationDecision(
        decision_id=DECISION_ID,
        run_id=RUN_A,
        event_sequence=1,
        invoke=False,
        risk_score=0.1,
        reason_codes=(ReasonCode.RISK_BELOW_THRESHOLD,),
        policy_version="fixture/1",
        configuration_digest="f" * 64,
        budget_snapshot=BudgetSnapshot(
            limits=limits,
            reserved=BudgetAmounts(),
            consumed=BudgetAmounts(),
        ),
        cooldown_active=False,
        created_at=NOW,
    )


def conditional_signal(event: TraceEvent, *, run_id: UUID = RUN_A) -> Signal:
    return Signal(
        signal_id=CONDITIONAL_SIGNAL_ID,
        run_id=run_id,
        created_at=NOW,
        signal_type=SignalType.TOOL_ERROR,
        strength=0.8,
        evidence_event_ids=(event.event_id,),
        detector_version="fixture/1",
        reason_code=ReasonCode.TOOL_ERROR,
    )


class RepositoryConformance:
    @pytest.fixture
    def repository_factory(self) -> RepositoryFactory:
        raise NotImplementedError

    @pytest.fixture
    def repository(self, repository_factory: RepositoryFactory) -> RunRepository:
        return repository_factory()

    async def test_append_redacts_and_authenticates_before_persistence(
        self,
        repository: RunRepository,
    ) -> None:
        secret = "fixture-secret-value"
        receipt = await repository.append(
            event_draft(payload={"password": secret, "safe": "preserved"})
        )

        assert receipt.disposition is AppendDisposition.APPENDED
        assert receipt.event.sequence == 1
        assert receipt.event.event_id.version == 4
        assert receipt.event.payload == {"password": "[REDACTED]", "safe": "preserved"}
        assert receipt.event.payload_digest.algorithm is PayloadDigestAlgorithm.HMAC_SHA256
        assert secret not in canonical_json(receipt).decode()
        ledger = await repository.ledger(RUN_A)
        assert tuple(entry.position for entry in ledger) == (1,)
        assert ledger[0].record == receipt.event
        assert ledger[0].chain_tag.algorithm is PayloadDigestAlgorithm.HMAC_SHA256

    async def test_identical_retry_returns_the_original_event_without_mutation(
        self,
        repository: RunRepository,
    ) -> None:
        draft = event_draft()
        first = await repository.append(draft)
        second = await repository.append(draft)

        assert first.disposition is AppendDisposition.APPENDED
        assert second.disposition is AppendDisposition.DUPLICATE
        assert second.event == first.event
        assert second.ledger_position == first.ledger_position
        assert second.ingestion_cursor == first.ingestion_cursor
        assert len(await repository.ledger(RUN_A)) == 1

    async def test_conditional_event_append_requires_the_exact_complete_head(
        self,
        repository: RunRepository,
    ) -> None:
        first = await repository.append_event_if_head(
            event_draft(source_event_id="conditional-first"),
            event_id=CONDITIONAL_EVENT_ID_A,
            expected_head=None,
        )
        first_head = await repository.ledger_head(RUN_A)

        with pytest.raises(LedgerHeadConflictError):
            await repository.append_event_if_head(
                event_draft(source_event_id="absent-conflict"),
                event_id=CONDITIONAL_EVENT_ID_B,
                expected_head=None,
            )

        second = await repository.append_event_if_head(
            event_draft(source_event_id="conditional-second"),
            event_id=CONDITIONAL_EVENT_ID_B,
            expected_head=first_head,
        )
        second_head = await repository.ledger_head(RUN_A)

        with pytest.raises(LedgerHeadConflictError):
            await repository.append_event_if_head(
                event_draft(source_event_id="stale-conflict"),
                event_id=UUID("00000000-0000-4000-8000-000000000124"),
                expected_head=first_head,
            )
        with pytest.raises(LedgerHeadConflictError):
            await repository.append_event_if_head(
                event_draft(source_event_id="conditional-first"),
                event_id=CONDITIONAL_EVENT_ID_A,
                expected_head=first_head,
            )

        duplicate = await repository.append_event_if_head(
            event_draft(source_event_id="conditional-first"),
            event_id=CONDITIONAL_EVENT_ID_A,
            expected_head=second_head,
        )

        assert first.event.event_id == CONDITIONAL_EVENT_ID_A
        assert second.event.event_id == CONDITIONAL_EVENT_ID_B
        assert duplicate.disposition is AppendDisposition.DUPLICATE
        assert duplicate.event == first.event
        assert await repository.ledger_head(RUN_A) == second_head
        assert len(await repository.ledger(RUN_A)) == 2

    async def test_conditional_event_collision_does_not_append_an_audit_event(
        self,
        repository: RunRepository,
    ) -> None:
        first = await repository.append_event_if_head(
            event_draft(payload={"password": "first-secret"}),
            event_id=CONDITIONAL_EVENT_ID_A,
            expected_head=None,
        )
        head = await repository.ledger_head(RUN_A)
        duplicate = await repository.append_event_if_head(
            event_draft(payload={"password": "second-secret"}),
            event_id=CONDITIONAL_EVENT_ID_A,
            expected_head=head,
        )

        assert duplicate.disposition is AppendDisposition.DUPLICATE
        assert duplicate.event == first.event
        assert await repository.ledger_head(RUN_A) == head
        snapshot = await repository.snapshot(RUN_A)
        mismatched_head = head.model_copy(update={"entry_count": head.entry_count + 1})

        with pytest.raises(LedgerHeadConflictError):
            await repository.append_event_if_head(
                event_draft(payload={"safe": "changed"}),
                event_id=CONDITIONAL_EVENT_ID_A,
                expected_head=mismatched_head,
            )
        with pytest.raises(RecordCollisionError):
            await repository.append_event_if_head(
                event_draft(payload={"safe": "changed"}),
                event_id=CONDITIONAL_EVENT_ID_A,
                expected_head=head,
            )

        ledger = await repository.ledger(RUN_A)
        assert len(ledger) == 1
        assert all(
            not (
                isinstance(entry.record, TraceEvent)
                and entry.record.event_type is EventType.CONTROLLER_ERROR
            )
            for entry in ledger
        )
        assert await repository.ledger_head(RUN_A) == head
        assert await repository.snapshot(RUN_A) == snapshot

    async def test_conditional_signal_write_is_exact_head_idempotent(
        self,
        repository: RunRepository,
    ) -> None:
        event = (
            await repository.append_event_if_head(
                event_draft(),
                event_id=CONDITIONAL_EVENT_ID_A,
                expected_head=None,
            )
        ).event
        event_head = await repository.ledger_head(RUN_A)
        signal = conditional_signal(event)
        first = await repository.record_signal_if_head(signal, expected_head=event_head)
        signal_head = await repository.ledger_head(RUN_A)

        with pytest.raises(LedgerHeadConflictError):
            await repository.record_signal_if_head(signal, expected_head=event_head)

        duplicate = await repository.record_signal_if_head(signal, expected_head=signal_head)
        missing_run_head = signal_head.model_copy(update={"run_id": RUN_B})
        with pytest.raises(LedgerHeadConflictError):
            await repository.record_signal_if_head(
                conditional_signal(event, run_id=RUN_B),
                expected_head=missing_run_head,
            )

        assert first.appended
        assert duplicate == first.model_copy(update={"appended": False})
        assert await repository.ledger_head(RUN_A) == signal_head
        assert len(await repository.ledger(RUN_A)) == 2

    async def test_conditional_batch_matches_ordered_sequential_writes(
        self,
        repository_factory: RepositoryFactory,
    ) -> None:
        sequential = repository_factory()
        batched = repository_factory()
        first_draft = event_draft(source_event_id="batch-first")
        second_draft = event_draft(
            source_event_id="batch-second",
            parent_ids=(CONDITIONAL_EVENT_ID_A,),
        )

        first = await sequential.append_event_if_head(
            first_draft,
            event_id=CONDITIONAL_EVENT_ID_A,
            expected_head=None,
        )
        first_head = await sequential.ledger_head(RUN_A)
        signal = conditional_signal(first.event)
        signal_receipt = await sequential.record_signal_if_head(
            signal,
            expected_head=first_head,
        )
        signal_head = await sequential.ledger_head(RUN_A)
        second = await sequential.append_event_if_head(
            second_draft,
            event_id=CONDITIONAL_EVENT_ID_B,
            expected_head=signal_head,
        )

        receipt = await batched.append_records_if_head(
            (
                ConditionalEventAppend(
                    event=first_draft,
                    event_id=CONDITIONAL_EVENT_ID_A,
                ),
                ConditionalSignalAppend(signal=signal),
                ConditionalEventAppend(
                    event=second_draft,
                    event_id=CONDITIONAL_EVENT_ID_B,
                ),
            ),
            expected_head=None,
        )

        assert receipt.initial_head is None
        assert receipt.receipts == (first, signal_receipt, second)
        assert receipt.final_head == await sequential.ledger_head(RUN_A)
        assert await batched.ledger(RUN_A) == await sequential.ledger(RUN_A)
        assert await batched.snapshot(RUN_A) == await sequential.snapshot(RUN_A)

    async def test_conditional_batch_matches_a_sequential_suffix_after_a_prefix(
        self,
        repository_factory: RepositoryFactory,
    ) -> None:
        sequential = repository_factory()
        batched = repository_factory()
        prefix_id = UUID("00000000-0000-4000-8000-000000000124")
        prefix_draft = event_draft(source_event_id="batch-common-prefix")
        await sequential.append_event_if_head(
            prefix_draft,
            event_id=prefix_id,
            expected_head=None,
        )
        await batched.append_event_if_head(
            prefix_draft,
            event_id=prefix_id,
            expected_head=None,
        )
        initial_head = await sequential.ledger_head(RUN_A)
        assert await batched.ledger_head(RUN_A) == initial_head

        suffix_draft = event_draft(
            source_event_id="batch-after-prefix",
            parent_ids=(prefix_id,),
        )
        event_receipt = await sequential.append_event_if_head(
            suffix_draft,
            event_id=CONDITIONAL_EVENT_ID_A,
            expected_head=initial_head,
        )
        event_head = await sequential.ledger_head(RUN_A)
        signal = conditional_signal(event_receipt.event)
        signal_receipt = await sequential.record_signal_if_head(signal, expected_head=event_head)

        batch_receipt = await batched.append_records_if_head(
            (
                ConditionalEventAppend(
                    event=suffix_draft,
                    event_id=CONDITIONAL_EVENT_ID_A,
                ),
                ConditionalSignalAppend(signal=signal),
            ),
            expected_head=initial_head,
        )

        assert batch_receipt.initial_head == initial_head
        assert batch_receipt.receipts == (event_receipt, signal_receipt)
        assert batch_receipt.final_head == await sequential.ledger_head(RUN_A)
        assert await batched.ledger(RUN_A) == await sequential.ledger(RUN_A)
        assert await batched.snapshot(RUN_A) == await sequential.snapshot(RUN_A)

    async def test_conditional_batch_rolls_back_a_staged_valid_prefix(
        self,
        repository: RunRepository,
    ) -> None:
        origin = await repository.append_event_if_head(
            event_draft(source_event_id="batch-origin"),
            event_id=CONDITIONAL_EVENT_ID_A,
            expected_head=None,
        )
        head = await repository.ledger_head(RUN_A)
        ledger = await repository.ledger(RUN_A)
        snapshot = await repository.snapshot(RUN_A)
        invalid_signal = conditional_signal(origin.event).model_copy(
            update={"evidence_event_ids": (PARENT_ID,)}
        )

        with pytest.raises(CrossRunReferenceError, match="signal evidence"):
            await repository.append_records_if_head(
                (
                    ConditionalEventAppend(
                        event=event_draft(source_event_id="batch-staged"),
                        event_id=CONDITIONAL_EVENT_ID_B,
                    ),
                    ConditionalSignalAppend(signal=invalid_signal),
                ),
                expected_head=head,
            )

        assert await repository.ledger_head(RUN_A) == head
        assert await repository.ledger(RUN_A) == ledger
        assert await repository.snapshot(RUN_A) == snapshot

    async def test_conditional_batch_rejects_invalid_aggregate_before_run_creation(
        self,
        repository: RunRepository,
    ) -> None:
        operation = ConditionalEventAppend(
            event=event_draft(source_event_id="batch-bounds"),
            event_id=CONDITIONAL_EVENT_ID_A,
        )

        with pytest.raises(InvalidRecordError, match="conditional_batch"):
            await repository.append_records_if_head((), expected_head=None)
        with pytest.raises(InvalidRecordTypeError, match="conditional_batch"):
            await repository.append_records_if_head(cast(Any, [operation]), expected_head=None)
        with pytest.raises(InvalidRecordError, match="conditional_batch"):
            await repository.append_records_if_head((operation,) * 5_001, expected_head=None)
        with pytest.raises(InvalidRecordError, match="conditional_batch"):
            await repository.append_records_if_head((operation,) * 1_001, expected_head=None)
        with pytest.raises(InvalidRecordError, match="conditional_batch"):
            await repository.append_records_if_head(
                (
                    operation,
                    ConditionalEventAppend(
                        event=event_draft(run_id=RUN_B, source_event_id="batch-other-run"),
                        event_id=CONDITIONAL_EVENT_ID_B,
                    ),
                ),
                expected_head=None,
            )

        corrupted = ConditionalEventAppend(
            event=event_draft(source_event_id="batch-corrupted"),
            event_id=CONDITIONAL_EVENT_ID_B,
        )
        object.__setattr__(corrupted, "event", object())
        with pytest.raises(InvalidRecordError, match="conditional_batch"):
            await repository.append_records_if_head((corrupted,), expected_head=None)

        with pytest.raises(RunNotFoundError):
            await repository.ledger(RUN_A)
        with pytest.raises(RunNotFoundError):
            await repository.ledger(RUN_B)

    async def test_conditional_batch_rejects_signal_limit_and_stale_head_without_mutation(
        self,
        repository: RunRepository,
    ) -> None:
        origin = await repository.append_event_if_head(
            event_draft(source_event_id="batch-signal-limit-origin"),
            event_id=CONDITIONAL_EVENT_ID_A,
            expected_head=None,
        )
        head = await repository.ledger_head(RUN_A)
        ledger = await repository.ledger(RUN_A)
        signal_operation = ConditionalSignalAppend(signal=conditional_signal(origin.event))

        with pytest.raises(InvalidRecordError, match="conditional_batch"):
            await repository.append_records_if_head(
                (signal_operation,) * 4_001,
                expected_head=head,
            )
        with pytest.raises(LedgerHeadConflictError):
            await repository.append_records_if_head(
                (
                    ConditionalEventAppend(
                        event=event_draft(source_event_id="batch-stale-head"),
                        event_id=CONDITIONAL_EVENT_ID_B,
                    ),
                ),
                expected_head=head.model_copy(update={"entry_count": head.entry_count + 1}),
            )

        assert await repository.ledger_head(RUN_A) == head
        assert await repository.ledger(RUN_A) == ledger

    async def test_conditional_batch_repairs_signals_and_retries_as_an_exact_noop(
        self,
        repository: RunRepository,
    ) -> None:
        origin = await repository.append_event_if_head(
            event_draft(source_event_id="batch-signal-repair-origin"),
            event_id=CONDITIONAL_EVENT_ID_A,
            expected_head=None,
        )
        event_head = await repository.ledger_head(RUN_A)
        operation = ConditionalSignalAppend(signal=conditional_signal(origin.event))

        repaired = await repository.append_records_if_head(
            (operation,),
            expected_head=event_head,
        )
        repaired_head = await repository.ledger_head(RUN_A)
        repaired_ledger = await repository.ledger(RUN_A)
        retried = await repository.append_records_if_head(
            (operation,),
            expected_head=repaired_head,
        )

        assert isinstance(repaired.receipts[0], LedgerReceipt)
        assert isinstance(retried.receipts[0], LedgerReceipt)
        assert repaired.initial_head == event_head
        assert repaired.receipts[0].appended
        assert repaired.final_head == repaired_head
        assert retried.initial_head == repaired_head
        assert not retried.receipts[0].appended
        assert retried.final_head == repaired_head
        assert await repository.ledger(RUN_A) == repaired_ledger

    async def test_two_conditional_batches_on_one_head_have_one_winner(
        self,
        repository: RunRepository,
    ) -> None:
        await repository.append_event_if_head(
            event_draft(source_event_id="batch-race-origin"),
            event_id=CONDITIONAL_EVENT_ID_A,
            expected_head=None,
        )
        head = await repository.ledger_head(RUN_A)
        operations = tuple(
            (
                ConditionalEventAppend(
                    event=event_draft(source_event_id=f"batch-race-{index}"),
                    event_id=UUID(f"00000000-0000-4000-8000-{300 + index:012x}"),
                ),
            )
            for index in range(2)
        )

        results = await asyncio.gather(
            *(repository.append_records_if_head(batch, expected_head=head) for batch in operations),
            return_exceptions=True,
        )

        assert sum(isinstance(result, ConditionalBatchReceipt) for result in results) == 1
        assert sum(isinstance(result, LedgerHeadConflictError) for result in results) == 1
        assert len(await repository.ledger(RUN_A)) == 2

    async def test_conditional_batch_redacts_payload_and_rejects_unsafe_metadata(
        self,
        repository: RunRepository,
    ) -> None:
        secret = "fixture-batch-secret-value"
        receipt = await repository.append_records_if_head(
            (
                ConditionalEventAppend(
                    event=event_draft(
                        source_event_id="batch-redaction",
                        payload={"password": secret, "safe": "preserved"},
                    ),
                    event_id=CONDITIONAL_EVENT_ID_A,
                ),
            ),
            expected_head=None,
        )
        event_receipt = receipt.receipts[0]
        assert isinstance(event_receipt, AppendReceipt)
        assert event_receipt.event.payload == {
            "password": "[REDACTED]",
            "safe": "preserved",
        }
        assert secret not in canonical_json(receipt).decode()
        head = await repository.ledger_head(RUN_A)
        ledger = await repository.ledger(RUN_A)

        with pytest.raises(UnsafeEventMetadataError, match="source_event_id"):
            await repository.append_records_if_head(
                (
                    ConditionalEventAppend(
                        event=event_draft(
                            source_event_id="sk-aaaaaaaaaaaaaaaa",
                        ),
                        event_id=CONDITIONAL_EVENT_ID_B,
                    ),
                ),
                expected_head=head,
            )

        assert await repository.ledger_head(RUN_A) == head
        assert await repository.ledger(RUN_A) == ledger

    async def test_conditional_batch_collision_discards_prefix_without_audit_event(
        self,
        repository: RunRepository,
    ) -> None:
        original_draft = event_draft(
            source_event_id="batch-collision-origin",
            payload={"message": "original"},
        )
        original = await repository.append_event_if_head(
            original_draft,
            event_id=CONDITIONAL_EVENT_ID_A,
            expected_head=None,
        )
        head = await repository.ledger_head(RUN_A)
        ledger = await repository.ledger(RUN_A)

        with pytest.raises(RecordCollisionError):
            await repository.append_records_if_head(
                (
                    ConditionalEventAppend(
                        event=event_draft(source_event_id="batch-collision-staged"),
                        event_id=CONDITIONAL_EVENT_ID_B,
                    ),
                    ConditionalEventAppend(
                        event=original_draft.model_copy(
                            update={"payload": {"message": "divergent"}}
                        ),
                        event_id=UUID("00000000-0000-4000-8000-000000000125"),
                    ),
                ),
                expected_head=head,
            )

        assert original.disposition is AppendDisposition.APPENDED
        assert await repository.ledger_head(RUN_A) == head
        assert await repository.ledger(RUN_A) == ledger
        assert not any(
            isinstance(entry.record, TraceEvent)
            and entry.record.event_type is EventType.CONTROLLER_ERROR
            for entry in await repository.ledger(RUN_A)
        )

    async def test_conditional_head_tokens_are_recursively_revalidated(
        self,
        repository: RunRepository,
    ) -> None:
        await repository.append_event_if_head(
            event_draft(),
            event_id=CONDITIONAL_EVENT_ID_A,
            expected_head=None,
        )
        head = await repository.ledger_head(RUN_A)
        alias_head = AliasLedgerHead.model_validate(head.model_dump(mode="python"))
        alias_digest = AliasPayloadDigest.model_validate(head.head_tag.model_dump(mode="python"))
        poisoned_head = head.model_copy(update={"entry_count": head.entry_count + 1})
        secret = "forged-head-secret"

        def reject_instance_dump(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError(secret)

        object.__setattr__(poisoned_head, "model_dump", reject_instance_dump)
        malformed = (
            alias_head,
            poisoned_head,
            head.model_copy(update={"entry_count": True}),
            head.model_copy(update={"run_id": AliasUUID(str(head.run_id))}),
            head.model_copy(update={"run_id": UUID(int=1)}),
            head.model_copy(update={"head_tag": alias_digest}),
            head.model_copy(
                update={"head_tag": head.head_tag.model_copy(update={"value": "invalid"})}
            ),
            head.model_copy(
                update={
                    "head_tag": head.head_tag.model_copy(
                        update={"algorithm": head.head_tag.algorithm.value}
                    )
                }
            ),
        )
        valid_mismatches = (
            head.model_copy(update={"run_id": RUN_B}),
            head.model_copy(update={"entry_count": head.entry_count + 1}),
            head.model_copy(
                update={"chain_tag": head.chain_tag.model_copy(update={"value": "0" * 64})}
            ),
            head.model_copy(
                update={
                    "projection_tag": head.projection_tag.model_copy(update={"value": "1" * 64})
                }
            ),
            head.model_copy(
                update={"head_tag": head.head_tag.model_copy(update={"value": "2" * 64})}
            ),
        )

        for index, expected_head in enumerate((*malformed, *valid_mismatches), start=1):
            with pytest.raises(LedgerHeadConflictError) as error:
                await repository.append_event_if_head(
                    event_draft(source_event_id=f"malformed-head-{index}"),
                    event_id=UUID(f"00000000-0000-4000-8000-{200 + index:012x}"),
                    expected_head=cast(Any, expected_head),
                )
            assert secret not in str(error.value)
            assert secret not in repr(error.value)

        assert await repository.ledger_head(RUN_A) == head
        assert len(await repository.ledger(RUN_A)) == 1

    async def test_secrets_that_redact_to_the_same_payload_are_duplicates(
        self,
        repository: RunRepository,
    ) -> None:
        first = await repository.append(event_draft(payload={"password": "first-secret"}))
        second = await repository.append(event_draft(payload={"password": "second-secret"}))

        assert second.disposition is AppendDisposition.DUPLICATE
        assert second.event == first.event
        assert len(await repository.ledger(RUN_A)) == 1

    @pytest.mark.parametrize(
        "change",
        [
            {"payload": {"message": "changed-safe-value"}},
            {"timestamp": NOW + timedelta(seconds=1)},
            {"event_type": EventType.TOOL_COMPLETION},
            {"phase": EventPhase.PRE_ACTION},
            {"parent_ids": (PARENT_ID,)},
            {"source_adapter": "other-adapter"},
            {"trust_label": TrustLabel.UNTRUSTED_TASK_INPUT},
        ],
    )
    async def test_source_collision_is_rejected_with_one_sanitized_audit_event(
        self,
        repository: RunRepository,
        change: dict[str, object],
    ) -> None:
        original = await repository.append(event_draft())
        changed = event_draft(**change)

        first_collision = await repository.append(changed)
        repeated_collision = await repository.append(changed)

        assert first_collision.disposition is AppendDisposition.COLLISION
        assert first_collision.event == original.event
        assert first_collision.collision_event is not None
        assert first_collision.collision_event.event_type is EventType.CONTROLLER_ERROR
        assert first_collision.collision_event.sequence == 2
        assert repeated_collision == first_collision
        ledger = await repository.ledger(RUN_A)
        assert tuple(entry.position for entry in ledger) == (1, 2)
        assert tuple(cast(TraceEvent, entry.record).sequence for entry in ledger) == (1, 2)
        encoded = canonical_json(ledger).decode()
        assert "changed-safe-value" not in encoded
        assert "other-adapter" not in encoded

    async def test_rebuild_ignores_caller_events_that_mimic_collision_audits(
        self,
        repository: RunRepository,
    ) -> None:
        original = await repository.append(event_draft())
        changed = event_draft(payload={"message": "changed"})
        collision = await repository.append(changed)
        assert collision.collision_event is not None
        decoy = await repository.append(event_draft(source_event_id="decoy"))
        fake_payload = dict(collision.collision_event.payload)
        fake_payload["existing_event_id"] = str(decoy.event.event_id)
        await repository.append(
            event_draft(
                source_event_id="caller-audit-lookalike",
                event_type=EventType.CONTROLLER_ERROR,
                phase=EventPhase.INTERNAL,
                payload=fake_payload,
                parent_ids=(decoy.event.event_id,),
            )
        )

        await repository.rebuild(RUN_A)
        repeated = await repository.append(changed)

        assert repeated.event == original.event
        assert repeated.collision_event == collision.collision_event
        assert len(await repository.ledger(RUN_A)) == 4

    async def test_redacted_drafts_and_other_runtime_types_are_rejected_without_state(
        self,
        repository: RunRepository,
    ) -> None:
        normalized = event_draft()
        forged = RedactedTraceEventDraft(
            **normalized.model_dump(mode="python", exclude={"record_type"}),
            payload_digest={
                "algorithm": PayloadDigestAlgorithm.HMAC_SHA256,
                "value": "a" * 64,
            },
        )

        with pytest.raises(InvalidAppendTypeError):
            await repository.append(cast(Any, forged))
        with pytest.raises(InvalidAppendTypeError):
            await repository.append(cast(Any, normalized.model_dump(mode="python")))
        with pytest.raises(RunNotFoundError):
            await repository.ledger(RUN_A)

    async def test_constructed_or_copied_invalid_draft_is_revalidated_without_echoing_input(
        self,
        repository: RunRepository,
    ) -> None:
        secret = "fixture-secret-that-must-not-echo"
        invalid = event_draft().model_copy(update={"payload": {"value": {secret}}})

        with pytest.raises(InvalidDraftError) as error:
            await repository.append(invalid)

        assert secret not in str(error.value)
        assert secret not in repr(error.value)
        with pytest.raises(RunNotFoundError):
            await repository.ledger(RUN_A)

        missing_fields = NormalizedTraceEventDraft.model_construct()
        with pytest.raises(InvalidDraftError):
            await repository.append(missing_fields)

    async def test_uuid_subclasses_cannot_cross_repository_boundaries(
        self,
        repository: RunRepository,
    ) -> None:
        alias = AliasUUID(str(RUN_B))
        invalid_draft = event_draft().model_copy(update={"run_id": alias})

        with pytest.raises(InvalidDraftError):
            await repository.append(invalid_draft)
        with pytest.raises(InvalidRunIdError):
            await repository.ledger(alias)

        event = (await repository.append(event_draft())).event
        signal = Signal(
            signal_id=UUID("00000000-0000-4000-8000-000000000117"),
            run_id=RUN_A,
            created_at=NOW,
            signal_type=SignalType.TOOL_ERROR,
            strength=0.8,
            evidence_event_ids=(event.event_id,),
            detector_version="fixture/1",
            reason_code=ReasonCode.TOOL_ERROR,
        ).model_copy(update={"run_id": alias})
        with pytest.raises(InvalidRecordError):
            await repository.record_signal(signal)

        query = MemoryQuery(run_id=RUN_A).model_copy(update={"run_id": alias})
        with pytest.raises(InvalidQueryError):
            await repository.search(query)
        assert len(await repository.ledger(RUN_A)) == 1

        caller_run_id = UUID(str(RUN_B))
        copied_draft = event_draft(run_id=caller_run_id)
        assert copied_draft.run_id == caller_run_id
        assert copied_draft.run_id is not caller_run_id

    @pytest.mark.parametrize(
        "change",
        [
            {"source_event_id": "raw metadata with spaces"},
            {"source_adapter": "adapter\nsecond-line"},
            {"parent_ids": (PARENT_ID, PARENT_ID)},
        ],
    )
    async def test_invalid_metadata_cannot_bypass_validation_through_model_copy(
        self,
        repository: RunRepository,
        change: dict[str, object],
    ) -> None:
        invalid = event_draft().model_copy(update=change)

        with pytest.raises(InvalidDraftError):
            await repository.append(invalid)
        with pytest.raises(RunNotFoundError):
            await repository.ledger(RUN_A)

    @pytest.mark.parametrize("field", ["source_event_id", "source_adapter"])
    async def test_recognized_secrets_are_rejected_in_metadata(
        self,
        repository: RunRepository,
        field: str,
    ) -> None:
        values: dict[str, object] = {field: "sk-aaaaaaaaaaaaaaaa"}
        draft = event_draft(**values)

        with pytest.raises(UnsafeEventMetadataError, match=field):
            await repository.append(draft)
        with pytest.raises(RunNotFoundError):
            await repository.ledger(RUN_A)

    @pytest.mark.parametrize(
        "change",
        [
            {"source_event_id": "saliencegate:forged-internal"},
            {"source_event_id": "SalienceGate:forged-internal"},
            {"source_adapter": "saliencegate.repository"},
            {"source_adapter": "SalienceGate.Repository"},
            {"trust_label": TrustLabel.TRUSTED_RUNTIME},
        ],
    )
    async def test_internal_event_metadata_is_reserved(
        self,
        repository: RunRepository,
        change: dict[str, object],
    ) -> None:
        with pytest.raises(UnsafeEventMetadataError):
            await repository.append(event_draft(**change))
        with pytest.raises(RunNotFoundError):
            await repository.ledger(RUN_A)

    async def test_concurrent_appends_have_contiguous_per_run_sequences(
        self,
        repository: RunRepository,
    ) -> None:
        drafts = tuple(event_draft(source_event_id=f"source-{index}") for index in range(40))
        receipts = await asyncio.gather(*(repository.append(draft) for draft in drafts))

        assert all(receipt.disposition is AppendDisposition.APPENDED for receipt in receipts)
        ledger = await repository.ledger(RUN_A)
        events = tuple(cast(TraceEvent, entry.record) for entry in ledger)
        assert tuple(event.sequence for event in events) == tuple(range(1, 41))
        assert len({event.event_id for event in events}) == 40

    async def test_concurrent_identical_appends_create_one_event(
        self,
        repository: RunRepository,
    ) -> None:
        receipts = await asyncio.gather(*(repository.append(event_draft()) for _ in range(40)))

        assert sum(receipt.disposition is AppendDisposition.APPENDED for receipt in receipts) == 1
        assert sum(receipt.disposition is AppendDisposition.DUPLICATE for receipt in receipts) == 39
        assert len({receipt.event.event_id for receipt in receipts}) == 1
        assert len(await repository.ledger(RUN_A)) == 1

    async def test_runs_have_independent_event_sequences(
        self,
        repository: RunRepository,
    ) -> None:
        first_a, first_b, second_a, second_b = await asyncio.gather(
            repository.append(event_draft(run_id=RUN_A, source_event_id="a-1")),
            repository.append(event_draft(run_id=RUN_B, source_event_id="b-1")),
            repository.append(event_draft(run_id=RUN_A, source_event_id="a-2")),
            repository.append(event_draft(run_id=RUN_B, source_event_id="b-2")),
        )

        assert {first_a.event.sequence, second_a.event.sequence} == {1, 2}
        assert {first_b.event.sequence, second_b.event.sequence} == {1, 2}

    async def test_event_parents_must_exist_in_the_same_run(
        self,
        repository: RunRepository,
    ) -> None:
        parent = (await repository.append(event_draft(run_id=RUN_A))).event
        child = await repository.append(
            event_draft(
                run_id=RUN_A,
                source_event_id="source-child",
                parent_ids=(parent.event_id,),
            )
        )

        assert child.event.parent_ids == (parent.event_id,)
        with pytest.raises(CrossRunReferenceError, match="parent"):
            await repository.append(
                event_draft(
                    run_id=RUN_B,
                    parent_ids=(parent.event_id,),
                )
            )
        with pytest.raises(RunNotFoundError):
            await repository.ledger(RUN_B)

    async def test_parent_ids_are_an_order_independent_set(
        self,
        repository: RunRepository,
    ) -> None:
        first = (await repository.append(event_draft(source_event_id="parent-first"))).event
        second = (await repository.append(event_draft(source_event_id="parent-second"))).event
        forward = await repository.append(
            event_draft(
                source_event_id="child",
                parent_ids=(first.event_id, second.event_id),
            )
        )
        reverse = await repository.append(
            event_draft(
                source_event_id="child",
                parent_ids=(second.event_id, first.event_id),
            )
        )

        assert reverse.disposition is AppendDisposition.DUPLICATE
        assert reverse.event == forward.event
        assert forward.event.parent_ids == tuple(sorted((first.event_id, second.event_id), key=str))
        assert len(await repository.ledger(RUN_A)) == 3

    async def test_direct_recording_is_typed_idempotent_and_ordered(
        self,
        repository: RunRepository,
    ) -> None:
        event = (await repository.append(event_draft())).event
        signal = Signal(
            signal_id=UUID("00000000-0000-4000-8000-000000000111"),
            run_id=RUN_A,
            created_at=NOW,
            signal_type=SignalType.TOOL_ERROR,
            strength=0.8,
            evidence_event_ids=(event.event_id,),
            detector_version="fixture/1",
            reason_code=ReasonCode.TOOL_ERROR,
        )

        first = await repository.record_signal(signal)
        duplicate = await repository.record_signal(signal)
        assert first.appended
        assert not duplicate.appended
        assert duplicate == first.model_copy(update={"appended": False})
        assert tuple(entry.position for entry in await repository.ledger(RUN_A)) == (1, 2)

        changed = signal.model_copy(update={"strength": 0.4})
        with pytest.raises(RecordCollisionError, match="collision"):
            await repository.record_signal(changed)
        with pytest.raises(InvalidRecordTypeError):
            await repository.record_signal(cast(Any, event))
        invalid = signal.model_copy(update={"evidence_event_ids": ()})
        with pytest.raises(InvalidRecordError):
            await repository.record_signal(invalid)
        duplicate_evidence = signal.model_copy(
            update={"evidence_event_ids": (event.event_id, event.event_id)}
        )
        with pytest.raises(InvalidRecordError):
            await repository.record_signal(duplicate_evidence)
        mismatched_reason = signal.model_copy(update={"reason_code": ReasonCode.CONFLICT})
        with pytest.raises(InvalidRecordError):
            await repository.record_signal(mismatched_reason)
        assert len(await repository.ledger(RUN_A)) == 2
        assert (await repository.rebuild(RUN_A)).equivalent

    async def test_direct_records_are_recursively_revalidated(
        self,
        repository: RunRepository,
    ) -> None:
        await repository.append(event_draft())
        decision = invocation_decision()
        invalid_consumed = decision.budget_snapshot.consumed.model_copy(update={"model_calls": -7})
        invalid_budget = decision.budget_snapshot.model_copy(update={"consumed": invalid_consumed})
        invalid = decision.model_copy(update={"budget_snapshot": invalid_budget})

        with pytest.raises(InvalidRecordError):
            await repository.record_invocation_decision(invalid)

        invalid_reason = decision.model_copy(
            update={"reason_codes": (ReasonCode.DELIVERY_SUCCEEDED,)}
        )
        contradictory_reason = decision.model_copy(
            update={
                "invoke": True,
                "reason_codes": (ReasonCode.BUDGET_EXHAUSTED,),
            }
        )
        for invalid_decision in (invalid_reason, contradictory_reason):
            with pytest.raises(InvalidRecordError):
                await repository.record_invocation_decision(invalid_decision)

        secret = "fixture-secret-must-not-echo"
        invalid_text = decision.model_copy(update={"policy_version": {secret}})
        with (
            warnings.catch_warnings(record=True) as caught,
            pytest.raises(InvalidRecordError) as error,
        ):
            await repository.record_invocation_decision(invalid_text)

        assert secret not in str(error.value)
        assert secret not in "".join(str(item.message) for item in caught)
        assert len(await repository.ledger(RUN_A)) == 1

    async def test_direct_record_text_must_be_unchanged_by_redaction(
        self,
        repository: RunRepository,
    ) -> None:
        event = (await repository.append(event_draft())).event
        secret = "sk-aaaaaaaaaaaaaaaa"
        signal = Signal(
            signal_id=UUID("00000000-0000-4000-8000-000000000116"),
            run_id=RUN_A,
            created_at=NOW,
            signal_type=SignalType.TOOL_ERROR,
            strength=0.8,
            evidence_event_ids=(event.event_id,),
            detector_version=secret,
            reason_code=ReasonCode.TOOL_ERROR,
        )
        decision = invocation_decision().model_copy(update={"policy_version": secret})

        with pytest.raises(UnsafeRecordContentError) as signal_error:
            await repository.record_signal(signal)
        with pytest.raises(UnsafeRecordContentError) as decision_error:
            await repository.record_invocation_decision(decision)

        assert secret not in str(signal_error.value)
        assert secret not in str(decision_error.value)
        assert len(await repository.ledger(RUN_A)) == 1

    async def test_decisions_require_an_existing_event_and_outcomes_require_intervention(
        self,
        repository: RunRepository,
    ) -> None:
        await repository.append(event_draft())
        decision = invocation_decision()
        first = await repository.record_invocation_decision(decision)
        duplicate = await repository.record_invocation_decision(decision)

        assert first.appended
        assert not duplicate.appended
        assert len(await repository.ledger(RUN_A)) == 2

        outcome = InterventionOutcome(
            outcome_id=OUTCOME_ID,
            run_id=RUN_A,
            intervention_id=INTERVENTION_ID,
            repeated_error_status=RepeatedErrorStatus.UNKNOWN,
            constraint_status=ConstraintStatus.UNKNOWN,
            evidence_mode=OutcomeEvidenceMode.POLICY_REPLAY,
            created_at=NOW,
        )
        with pytest.raises(CrossRunReferenceError, match="intervention"):
            await repository.record_outcome(outcome)
        assert len(await repository.ledger(RUN_A)) == 2
        assert (await repository.rebuild(RUN_A)).equivalent

    async def test_decision_timestamp_cannot_precede_its_referenced_event(
        self,
        repository: RunRepository,
    ) -> None:
        await repository.append(event_draft())
        decision = invocation_decision().model_copy(
            update={"created_at": NOW - timedelta(microseconds=1)}
        )

        with pytest.raises(ProjectionInvariantError, match="precedes its event"):
            await repository.record_invocation_decision(decision)

        assert len(await repository.ledger(RUN_A)) == 1
        assert (await repository.rebuild(RUN_A)).equivalent

    async def test_event_accepts_only_one_authoritative_invocation_decision(
        self,
        repository: RunRepository,
    ) -> None:
        await repository.append(event_draft())
        decision = invocation_decision()
        first = await repository.record_invocation_decision(decision)
        duplicate = await repository.record_invocation_decision(decision)
        competing = decision.model_copy(
            update={
                "decision_id": UUID("00000000-0000-4000-8000-000000000117"),
                "policy_version": "fixture/2",
            }
        )

        assert first.appended
        assert duplicate == first.model_copy(update={"appended": False})
        with pytest.raises(ProjectionInvariantError, match="already has a decision"):
            await repository.record_invocation_decision(competing)

        assert len(await repository.ledger(RUN_A)) == 2
        assert (await repository.rebuild(RUN_A)).equivalent

    async def test_signal_cannot_reference_an_event_from_another_run(
        self,
        repository: RunRepository,
    ) -> None:
        event_a = (await repository.append(event_draft())).event
        await repository.append(event_draft(run_id=RUN_B))
        cross_run_signal = Signal(
            signal_id=UUID("00000000-0000-4000-8000-000000000115"),
            run_id=RUN_B,
            created_at=NOW,
            signal_type=SignalType.CONFLICT,
            strength=1.0,
            evidence_event_ids=(event_a.event_id,),
            detector_version="fixture/1",
            reason_code=ReasonCode.CONFLICT,
        )

        with pytest.raises(CrossRunReferenceError, match="evidence"):
            await repository.record_signal(cross_run_signal)
        assert len(await repository.ledger(RUN_B)) == 1

    async def test_snapshot_is_immutable_and_not_a_live_view(
        self,
        repository: RunRepository,
    ) -> None:
        await repository.append(event_draft(source_event_id="source-1"))
        before = await repository.snapshot(RUN_A)
        await repository.append(event_draft(source_event_id="source-2"))
        after = await repository.snapshot(RUN_A)

        assert before.ingestion_cursor == 1
        assert before.ledger_position == 1
        assert after.ingestion_cursor == 2
        assert after.ledger_position == 2
        assert before.projection_digest != after.projection_digest
        with pytest.raises(ValidationError, match="frozen"):
            before.ingestion_cursor = 99

        invalid_query = MemoryQuery(run_id=RUN_A).model_copy(update={"limit": 0})
        with pytest.raises(InvalidQueryError):
            await repository.search(invalid_query)
        with pytest.raises(InvalidRecordTypeError):
            await repository.search(cast(Any, {"run_id": RUN_A}))

    async def test_public_records_are_defensive_copies(
        self,
        repository: RunRepository,
    ) -> None:
        receipt = await repository.append(event_draft())
        expected_event_id = UUID(str(receipt.event.event_id))
        object.__setattr__(receipt.event, "source_adapter", "mutated-return")
        object.__setattr__(
            receipt.event.event_id,
            "int",
            UUID("00000000-0000-4000-8000-000000000199").int,
        )

        first_read = await repository.ledger(RUN_A)
        first_event = cast(TraceEvent, first_read[0].record)
        assert first_event.source_adapter == "fixture-adapter"
        assert first_event.event_id == expected_event_id
        object.__setattr__(first_event, "source_adapter", "mutated-ledger-read")
        object.__setattr__(
            first_event.event_id,
            "int",
            UUID("00000000-0000-4000-8000-000000000198").int,
        )

        second_read = await repository.ledger(RUN_A)
        second_event = cast(TraceEvent, second_read[0].record)
        assert second_event.source_adapter == "fixture-adapter"
        assert second_event.event_id == expected_event_id
        assert (await repository.rebuild(RUN_A)).equivalent

    async def test_rebuild_is_idempotent_and_does_not_change_the_ledger(
        self,
        repository: RunRepository,
    ) -> None:
        await repository.append(event_draft(source_event_id="source-1"))
        await repository.append(event_draft(source_event_id="source-2"))
        before = await repository.ledger(RUN_A)

        first = await repository.rebuild(RUN_A)
        second = await repository.rebuild(RUN_A)

        assert first.equivalent
        assert second.equivalent
        assert first.after == second.after
        assert await repository.ledger(RUN_A) == before

    async def test_ledger_tags_make_fresh_replay_deterministic(
        self,
        repository_factory: RepositoryFactory,
    ) -> None:
        first = repository_factory()
        second = repository_factory()
        drafts = (
            event_draft(source_event_id="source-1"),
            event_draft(source_event_id="source-2"),
        )
        for draft in drafts:
            await first.append(draft)
            await second.append(draft)

        assert canonical_json(await first.ledger(RUN_A)) == canonical_json(
            await second.ledger(RUN_A)
        )


CYCLE_RUN_ID = UUID("00000000-0000-4000-8000-000000000301")
CYCLE_DECISION_1_ID = UUID("00000000-0000-4000-8000-000000000302")
CYCLE_DECISION_2_ID = UUID("00000000-0000-4000-8000-000000000303")
CYCLE_INTERVENTION_1_ID = UUID("00000000-0000-4000-8000-000000000304")
CYCLE_MEMORY_ID = UUID("00000000-0000-4000-8000-000000000306")
CYCLE_MISSING_MEMORY_ID = UUID("00000000-0000-4000-8000-000000000307")
CYCLE_NOW = datetime(2026, 7, 11, 15, 0, tzinfo=UTC)


def cycle_grounding_config() -> GroundingConfig:
    return GroundingConfig(
        schema_version="1.0",
        pipeline_version="grounding-pipeline/v1",
        claim_schema_version="citation-only-claims/v1",
        max_claims=2,
        max_evidence_per_claim=1,
        max_pointer_segments=32,
        max_pointer_utf8_bytes=1_024,
        duplicate_window_events=0,
        cooldown_events=0,
        ttl_steps=1,
        allowed_delivery_targets=(
            DeliveryTarget.NEXT_MODEL_CALL,
            DeliveryTarget.PRE_ACTION_REPLAN,
        ),
        rendering=RenderingConfig(
            schema_version="1.0",
            renderer_version="fixed-ascii/v1",
            token_counter_version="utf8-bytes-ceil-div-4-v1",
            max_claims=2,
            max_evidence_bytes=1_024,
            max_output_bytes=4_096,
            max_token_equivalents=1_024,
            include_provenance=False,
        ),
    )


def cycle_budget_limits(**updates: int) -> BudgetLimits:
    values = {
        "model_calls": 10,
        "input_tokens": 1_000,
        "output_tokens": 1_000,
        "canonical_token_equivalents": 2_000,
        "latency_us": 1_000_000,
        "max_call_latency_us": 500_000,
        "interventions": 10,
        "schema_repairs": 10,
    }
    values.update(updates)
    return BudgetLimits(**values)


def _empty_cycle_budget(limits: BudgetLimits) -> BudgetSnapshot:
    return BudgetSnapshot(
        limits=limits,
        reserved=BudgetAmounts(),
        consumed=BudgetAmounts(),
    )


@dataclass(frozen=True, slots=True)
class CycleContext:
    event: TraceEvent
    pending: CycleReceipt
    reserve: ReserveCycle
    start: StartCycle
    ordinal: int

    @property
    def cycle_id(self) -> str:
        return self.pending.cycle.cycle_id

    @property
    def commit_time(self) -> datetime:
        return CYCLE_NOW + timedelta(seconds=self.ordinal * 20 + 5)


def _cycle_event_draft(ordinal: int) -> NormalizedTraceEventDraft:
    return NormalizedTraceEventDraft(
        run_id=CYCLE_RUN_ID,
        source_event_id=f"cycle-source-{ordinal}",
        timestamp=CYCLE_NOW + timedelta(seconds=ordinal * 20),
        event_type=EventType.OBSERVATION,
        phase=EventPhase.POST_ACTION,
        payload={"message": f"safe cycle event {ordinal}"},
        source_adapter="cycle-fixture/1",
        trust_label=TrustLabel.SYNTHETIC_FIXTURE,
    )


def _cycle_invocation(
    *,
    event: TraceEvent,
    decision_id: UUID,
    snapshot: BudgetSnapshot,
    ordinal: int,
) -> InvocationDecision:
    return InvocationDecision(
        decision_id=decision_id,
        run_id=CYCLE_RUN_ID,
        event_sequence=event.sequence,
        invoke=True,
        risk_score=0.8,
        reason_codes=(ReasonCode.SCRIPTED_INVOKE,),
        policy_version="cycle-fixture/1",
        configuration_digest="f" * 64,
        budget_snapshot=snapshot,
        cooldown_active=False,
        created_at=CYCLE_NOW + timedelta(seconds=ordinal * 20 + 1),
    )


async def begin_cycle_context(
    repository: RunRepository,
    *,
    ordinal: int = 1,
    limits: BudgetLimits | None = None,
) -> CycleContext:
    event = (await repository.append(_cycle_event_draft(ordinal))).event
    if ordinal == 1:
        snapshot = _empty_cycle_budget(cycle_budget_limits() if limits is None else limits)
        decision_id = CYCLE_DECISION_1_ID
    else:
        snapshot = await repository.budget_snapshot(CYCLE_RUN_ID)
        decision_id = CYCLE_DECISION_2_ID
    await repository.record_invocation_decision(
        _cycle_invocation(
            event=event,
            decision_id=decision_id,
            snapshot=snapshot,
            ordinal=ordinal,
        )
    )
    grounding_configuration = cycle_grounding_config()
    resolved_grounding = resolve_grounding_configuration(grounding_configuration)
    pending = await repository.begin_cycle(
        BeginCycle(
            run_id=CYCLE_RUN_ID,
            invocation_decision_id=decision_id,
            grounding_version=resolved_grounding.pipeline_version,
            grounding_configuration=resolved_grounding.configuration,
            grounding_configuration_digest=resolved_grounding.configuration_digest,
            requested_delivery_target=DeliveryTarget.NEXT_MODEL_CALL,
            created_at=CYCLE_NOW + timedelta(seconds=ordinal * 20 + 2),
        )
    )
    return CycleContext(
        event=event,
        pending=pending,
        reserve=ReserveCycle(
            run_id=CYCLE_RUN_ID,
            cycle_id=pending.cycle.cycle_id,
            reservation=BudgetAmounts(
                model_calls=1,
                input_tokens=100,
                output_tokens=50,
                canonical_token_equivalents=150,
                latency_us=10_000,
                interventions=1,
                schema_repairs=1,
            ),
            updated_at=CYCLE_NOW + timedelta(seconds=ordinal * 20 + 3),
        ),
        start=StartCycle(
            run_id=CYCLE_RUN_ID,
            cycle_id=pending.cycle.cycle_id,
            batch_digest=f"{ordinal:x}" * 64,
            updated_at=CYCLE_NOW + timedelta(seconds=ordinal * 20 + 4),
        ),
        ordinal=ordinal,
    )


def _cycle_silence(
    context: CycleContext,
    intervention_id: UUID,
) -> InterventionDecision:
    configuration = cycle_grounding_config()
    return GroundingPipeline(configuration).ground(
        InterventionProposal(
            action=InterventionAction.SILENCE,
            claims=(),
            confidence=1.0,
            model_free_text=None,
        ),
        context=GroundingContext(
            schema_version="1.0",
            intervention_id=intervention_id,
            run_id=CYCLE_RUN_ID,
            cycle_id=context.cycle_id,
            current_event_sequence=context.event.sequence,
            created_at=context.commit_time,
            requested_delivery_target=DeliveryTarget.NEXT_MODEL_CALL,
            model_call_index=0,
            model_call_digest="a" * 64,
        ),
        state=GroundingState(
            schema_version="1.0",
            events=(context.event,),
            memories=(),
            reminder_history=(),
        ),
    )


def _cycle_reminder(context: CycleContext) -> InterventionDecision:
    return GroundingPipeline(cycle_grounding_config()).ground(
        InterventionProposal(
            action=InterventionAction.REMIND,
            claims=(
                ProposedClaim(
                    kind=ClaimKind.ENVIRONMENT_FACT,
                    evidence=EvidenceReference(
                        source=EvidenceSource.EVENT,
                        source_id=context.event.event_id,
                        field_path="/payload/message",
                    ),
                ),
            ),
            confidence=0.9,
            model_free_text="ignored by authoritative grounding",
        ),
        context=GroundingContext(
            schema_version="1.0",
            intervention_id=CYCLE_INTERVENTION_1_ID,
            run_id=CYCLE_RUN_ID,
            cycle_id=context.cycle_id,
            current_event_sequence=context.event.sequence,
            created_at=context.commit_time,
            requested_delivery_target=DeliveryTarget.NEXT_MODEL_CALL,
            model_call_index=0,
            model_call_digest="a" * 64,
        ),
        state=GroundingState(
            schema_version="1.0",
            events=(context.event,),
            memories=(),
            reminder_history=(),
        ),
    )


def cycle_commit_command(
    context: CycleContext,
    *,
    settlement: BudgetAmounts | None = None,
    delta: MemoryDelta | None = None,
    assignments: tuple[MemoryIdAssignment, ...] = (),
    intervention_id: UUID = CYCLE_INTERVENTION_1_ID,
    intervention: InterventionDecision | None = None,
    model_call_digests: tuple[str, ...] | None = None,
    model_call_latencies_us: tuple[int, ...] | None = None,
) -> CommitCycle:
    if settlement is None:
        settlement = BudgetAmounts(
            model_calls=1,
            input_tokens=80,
            output_tokens=20,
            canonical_token_equivalents=100,
            latency_us=8_000,
            interventions=0,
            schema_repairs=0,
        )
    if delta is None:
        delta = MemoryDelta(
            delta_id=UUID(f"00000000-0000-4000-8000-{0x500 + context.ordinal:012x}"),
            run_id=CYCLE_RUN_ID,
            created_at=context.commit_time,
        )
    if model_call_digests is None:
        model_call_digests = tuple("a" * 64 for _ in range(settlement.model_calls))
    if model_call_latencies_us is None:
        quotient, remainder = divmod(
            settlement.latency_us,
            max(settlement.model_calls, 1),
        )
        model_call_latencies_us = tuple(
            quotient + int(index < remainder) for index in range(settlement.model_calls)
        )
    verdict = _cycle_silence(context, intervention_id) if intervention is None else intervention
    delivery = (
        EnqueueDelivery(
            target_request_id=f"request-{context.ordinal}",
            adapter_id="fixture.adapter/1",
            adapter_deduplicates=True,
            adapter_deduplication_guarantee=(DeduplicationGuarantee.DURABLE_DELIVERY_ID),
            adapter_supports_pre_action=True,
            adapter_contract_version="adapter-contract/v1",
            adapter_capabilities_digest="8" * 64,
        )
        if verdict.action is InterventionAction.REMIND
        else None
    )
    return CommitCycle(
        run_id=CYCLE_RUN_ID,
        cycle_id=context.cycle_id,
        settlement=settlement,
        model_call_digests=model_call_digests,
        model_call_latencies_us=model_call_latencies_us,
        validated_delta=delta,
        memory_id_assignments=assignments,
        intervention=verdict,
        delivery=delivery,
        updated_at=context.commit_time,
    )


async def advance_cycle_to_running(
    repository: RunRepository,
    *,
    limits: BudgetLimits | None = None,
    reservation: BudgetAmounts | None = None,
) -> tuple[CycleContext, CycleReceipt, CycleReceipt]:
    context = await begin_cycle_context(repository, limits=limits)
    reserve = context.reserve
    if reservation is not None:
        reserve = reserve.model_copy(update={"reservation": reservation})
    reserved = await repository.reserve_cycle(reserve)
    running = await repository.mark_cycle_running(context.start)
    return context, reserved, running


def cycle_records(entries: tuple[object, ...]) -> tuple[CycleRecord, ...]:
    return tuple(
        entry.record
        for entry in entries
        if hasattr(entry, "record") and isinstance(entry.record, CycleRecord)
    )


def delivery_records(entries: tuple[object, ...]) -> tuple[DeliveryRecord, ...]:
    return tuple(
        entry.record
        for entry in entries
        if hasattr(entry, "record") and isinstance(entry.record, DeliveryRecord)
    )


def reminder_commit_command(
    context: CycleContext,
    *,
    adapter_deduplicates: bool = True,
) -> CommitCycle:
    command = cycle_commit_command(
        context,
        settlement=BudgetAmounts(
            model_calls=1,
            input_tokens=80,
            output_tokens=20,
            canonical_token_equivalents=100,
            latency_us=8_000,
            interventions=1,
        ),
        intervention=_cycle_reminder(context),
    )
    assert command.delivery is not None
    return command.model_copy(
        update={
            "delivery": command.delivery.model_copy(
                update={
                    "adapter_deduplicates": adapter_deduplicates,
                    "adapter_deduplication_guarantee": (
                        DeduplicationGuarantee.DURABLE_DELIVERY_ID
                        if adapter_deduplicates
                        else DeduplicationGuarantee.AT_MOST_ONCE_ATTEMPT
                    ),
                }
            )
        }
    )


async def commit_grounded_delivery(
    repository: RunRepository,
    *,
    adapter_deduplicates: bool = True,
) -> tuple[CycleContext, CommitCycle, CycleReceipt, DeliveryRecord]:
    context, _reserved, _running = await advance_cycle_to_running(repository)
    command = reminder_commit_command(
        context,
        adapter_deduplicates=adapter_deduplicates,
    )
    committed = await repository.commit_cycle(command)
    delivery = committed.delivery
    assert delivery is not None
    return context, command, committed, delivery


def claim_command(
    delivery: DeliveryRecord,
    claim_id: UUID,
    *,
    seconds: int = 1,
) -> ClaimDelivery:
    return ClaimDelivery(
        run_id=delivery.run_id,
        delivery_id=delivery.delivery_id,
        expected_revision=delivery.revision,
        claim_id=claim_id,
        updated_at=delivery.updated_at + timedelta(seconds=seconds),
    )


def attempt_command(
    delivery: DeliveryRecord,
    claim_id: UUID,
    attempt_id: UUID,
    *,
    seconds: int = 1,
) -> BeginDeliveryAttempt:
    return BeginDeliveryAttempt(
        run_id=delivery.run_id,
        delivery_id=delivery.delivery_id,
        expected_revision=delivery.revision,
        claim_id=claim_id,
        attempt_id=attempt_id,
        updated_at=delivery.updated_at + timedelta(seconds=seconds),
    )


def unknown_command(
    delivery: DeliveryRecord,
    claim_id: UUID,
    attempt_id: UUID,
    *,
    seconds: int = 1,
) -> MarkDeliveryUnknown:
    return MarkDeliveryUnknown(
        run_id=delivery.run_id,
        delivery_id=delivery.delivery_id,
        expected_revision=delivery.revision,
        claim_id=claim_id,
        attempt_id=attempt_id,
        updated_at=delivery.updated_at + timedelta(seconds=seconds),
    )


def complete_command(
    delivery: DeliveryRecord,
    claim_id: UUID,
    attempt_id: UUID,
    outcome: DeliveryOutcome,
    *,
    seconds: int = 1,
) -> CompleteDelivery:
    return CompleteDelivery(
        run_id=delivery.run_id,
        delivery_id=delivery.delivery_id,
        expected_revision=delivery.revision,
        claim_id=claim_id,
        attempt_id=attempt_id,
        outcome=outcome,
        provider_receipt_id=(
            "provider-receipt-1" if outcome is DeliveryOutcome.DELIVERED else None
        ),
        updated_at=delivery.updated_at + timedelta(seconds=seconds),
    )


class CycleRepositoryConformance:
    """Lifecycle contract shared by every authoritative repository backend."""

    async def test_valid_grounded_reminder_survives_authoritative_rebuild(
        self,
        repository: RunRepository,
    ) -> None:
        context, _reserved, _running = await advance_cycle_to_running(repository)
        reminder = _cycle_reminder(context)
        settlement = BudgetAmounts(
            model_calls=1,
            input_tokens=80,
            output_tokens=20,
            canonical_token_equivalents=100,
            latency_us=8_000,
            interventions=1,
        )

        committed = await repository.commit_cycle(
            cycle_commit_command(
                context,
                settlement=settlement,
                intervention=reminder,
            )
        )

        assert committed.cycle.intervention == reminder
        assert (await repository.rebuild(CYCLE_RUN_ID)).equivalent
        rebuilt = cycle_records(await repository.ledger(CYCLE_RUN_ID))[-1]
        assert rebuilt.intervention == reminder

    async def test_forged_grounding_output_and_configuration_are_rejected_atomically(
        self,
        repository: RunRepository,
    ) -> None:
        context, _reserved, _running = await advance_cycle_to_running(repository)
        reminder = _cycle_reminder(context)
        settlement = BudgetAmounts(
            model_calls=1,
            input_tokens=80,
            output_tokens=20,
            canonical_token_equivalents=100,
            latency_us=8_000,
            interventions=1,
        )
        secret = "forged-output-must-not-echo"
        forged = (
            reminder.model_copy(update={"rendered_text": secret}),
            reminder.model_copy(update={"grounding_configuration": {"attacker": secret}}),
        )
        before = await repository.ledger(CYCLE_RUN_ID)

        for intervention in forged:
            with pytest.raises(
                ProjectionInvariantError,
                match="grounded intervention failed authoritative verification",
            ) as error:
                await repository.commit_cycle(
                    cycle_commit_command(
                        context,
                        settlement=settlement,
                        intervention=intervention,
                    )
                )
            assert secret not in str(error.value)
            assert secret not in repr(error.value)
            assert await repository.ledger(CYCLE_RUN_ID) == before

    async def test_cycle_happy_path_commits_memory_budget_and_cursor_atomically(
        self,
        repository: RunRepository,
    ) -> None:
        context, reserved, running = await advance_cycle_to_running(repository)
        delta = MemoryDelta(
            delta_id=UUID("00000000-0000-4000-8000-000000000308"),
            run_id=CYCLE_RUN_ID,
            creates=(
                MemoryCreate(
                    handle="created-memory",
                    kind=MemoryKind.KNOWLEDGE,
                    content="Run the focused repository tests.",
                    provenance=(
                        EvidenceReference(
                            source=EvidenceSource.EVENT,
                            source_id=context.event.event_id,
                            field_path="/payload/message",
                        ),
                    ),
                    confidence=0.9,
                    trust_label=TrustLabel.TRUSTED_CONTROLLER,
                ),
            ),
            created_at=context.commit_time,
        )
        settlement = BudgetAmounts(
            model_calls=1,
            input_tokens=75,
            output_tokens=15,
            canonical_token_equivalents=90,
            latency_us=7_500,
        )
        committed = await repository.commit_cycle(
            cycle_commit_command(
                context,
                settlement=settlement,
                delta=delta,
                assignments=(
                    MemoryIdAssignment(
                        handle="created-memory",
                        memory_id=CYCLE_MEMORY_ID,
                    ),
                ),
            )
        )

        assert tuple(
            receipt.cycle.state for receipt in (context.pending, reserved, running, committed)
        ) == (
            CycleState.PENDING,
            CycleState.RESERVED,
            CycleState.RUNNING,
            CycleState.COMMITTED,
        )
        assert committed.cycle.revision == 4
        assert committed.budget_snapshot.reserved == BudgetAmounts()
        assert committed.budget_snapshot.consumed == settlement
        snapshot = await repository.snapshot(CYCLE_RUN_ID)
        assert snapshot.ingestion_cursor == 1
        assert snapshot.memory_cursor == 1
        assert tuple(record.memory_id for record in snapshot.records) == (CYCLE_MEMORY_ID,)
        assert tuple(
            record.state for record in cycle_records(await repository.ledger(CYCLE_RUN_ID))
        ) == (
            CycleState.PENDING,
            CycleState.RESERVED,
            CycleState.RUNNING,
            CycleState.COMMITTED,
        )

    async def test_transition_retries_return_historical_receipts_after_commit(
        self,
        repository: RunRepository,
    ) -> None:
        context, reserved, running = await advance_cycle_to_running(repository)
        command = cycle_commit_command(context)
        committed = await repository.commit_cycle(command)
        ledger_before = await repository.ledger(CYCLE_RUN_ID)

        reserve_retry = await repository.reserve_cycle(context.reserve)
        start_retry = await repository.mark_cycle_running(context.start)
        commit_retry = await repository.commit_cycle(command)

        assert reserve_retry == reserved.model_copy(update={"appended": False})
        assert start_retry == running.model_copy(update={"appended": False})
        assert commit_retry == committed.model_copy(update={"appended": False})
        assert await repository.ledger(CYCLE_RUN_ID) == ledger_before

    async def test_begin_retry_survives_memory_cursor_advancement(
        self,
        repository: RunRepository,
    ) -> None:
        context, _reserved, _running = await advance_cycle_to_running(repository)
        await repository.commit_cycle(cycle_commit_command(context))
        ledger_before = await repository.ledger(CYCLE_RUN_ID)

        retry = await repository.begin_cycle(
            BeginCycle(
                run_id=CYCLE_RUN_ID,
                invocation_decision_id=CYCLE_DECISION_1_ID,
                grounding_version=context.pending.cycle.grounding_version,
                grounding_configuration=context.pending.cycle.grounding_configuration,
                grounding_configuration_digest=(
                    context.pending.cycle.grounding_configuration_digest
                ),
                requested_delivery_target=context.pending.cycle.requested_delivery_target,
                created_at=context.pending.cycle.created_at,
            )
        )

        assert retry == context.pending.model_copy(update={"appended": False})
        assert await repository.ledger(CYCLE_RUN_ID) == ledger_before

    async def test_changed_retry_conflicts_and_unknown_revision_is_rejected(
        self,
        repository: RunRepository,
    ) -> None:
        context, _reserved, _running = await advance_cycle_to_running(repository)
        ledger_before = await repository.ledger(CYCLE_RUN_ID)
        changed_reservation = context.reserve.model_copy(
            update={"reservation": BudgetAmounts(model_calls=1, input_tokens=99)}
        )

        with pytest.raises(CycleConflictError):
            await repository.reserve_cycle(changed_reservation)
        missing_revision = context.start.model_copy(update={"expected_revision": 99})
        with pytest.raises(CycleRevisionConflictError) as error:
            await repository.mark_cycle_running(missing_revision)

        assert error.value.expected == 99
        assert error.value.actual == 3
        assert await repository.ledger(CYCLE_RUN_ID) == ledger_before

    async def test_changed_terminal_commit_retry_conflicts_without_mutation(
        self,
        repository: RunRepository,
    ) -> None:
        context, _reserved, _running = await advance_cycle_to_running(repository)
        command = cycle_commit_command(context)
        await repository.commit_cycle(command)
        ledger_before = await repository.ledger(CYCLE_RUN_ID)
        changed = command.model_copy(
            update={"settlement": command.settlement.model_copy(update={"input_tokens": 79})}
        )

        with pytest.raises(CycleConflictError):
            await repository.commit_cycle(changed)
        assert await repository.ledger(CYCLE_RUN_ID) == ledger_before

    @pytest.mark.parametrize(
        "field_name",
        (
            "model_calls",
            "input_tokens",
            "output_tokens",
            "canonical_token_equivalents",
            "latency_us",
            "interventions",
            "schema_repairs",
        ),
    )
    async def test_each_cycle_budget_dimension_enforces_its_boundary(
        self,
        repository: RunRepository,
        field_name: str,
    ) -> None:
        limit_updates = {field_name: 1}
        if field_name == "latency_us":
            limit_updates["max_call_latency_us"] = 100
        context = await begin_cycle_context(
            repository,
            limits=cycle_budget_limits(**limit_updates),
        )
        exact_values = {"model_calls": 1, field_name: 1}
        overflow_values = {**exact_values, field_name: 2}
        overflow = context.reserve.model_copy(
            update={"reservation": BudgetAmounts(**overflow_values)}
        )
        ledger_before = await repository.ledger(CYCLE_RUN_ID)

        with pytest.raises(ProjectionInvariantError, match="reservation"):
            await repository.reserve_cycle(overflow)
        assert await repository.ledger(CYCLE_RUN_ID) == ledger_before

        exact = context.reserve.model_copy(update={"reservation": BudgetAmounts(**exact_values)})
        receipt = await repository.reserve_cycle(exact)
        assert receipt.budget_snapshot.reserved == exact.reservation

    async def test_latency_reservation_covers_total_cycle_overhead(
        self,
        repository: RunRepository,
    ) -> None:
        context = await begin_cycle_context(
            repository,
            limits=cycle_budget_limits(
                model_calls=2,
                latency_us=100,
                max_call_latency_us=10,
            ),
        )
        total_overhead = context.reserve.model_copy(
            update={"reservation": BudgetAmounts(model_calls=1, latency_us=11)}
        )

        receipt = await repository.reserve_cycle(total_overhead)
        assert receipt.budget_snapshot.reserved == total_overhead.reservation

    async def test_settlement_honors_actual_per_call_latency_ceiling(
        self,
        repository: RunRepository,
    ) -> None:
        held = BudgetAmounts(model_calls=2, latency_us=20)
        context, _reserved, _running = await advance_cycle_to_running(
            repository,
            limits=cycle_budget_limits(
                model_calls=2,
                latency_us=20,
                max_call_latency_us=10,
            ),
            reservation=held,
        )
        before = await repository.ledger(CYCLE_RUN_ID)

        with pytest.raises(ProjectionInvariantError, match="per-call latency"):
            await repository.commit_cycle(
                cycle_commit_command(
                    context,
                    settlement=BudgetAmounts(model_calls=2, latency_us=20),
                    model_call_digests=("a" * 64, "b" * 64),
                    model_call_latencies_us=(20, 0),
                )
            )

        assert await repository.ledger(CYCLE_RUN_ID) == before
        assert (await repository.budget_snapshot(CYCLE_RUN_ID)).reserved == held

    async def test_consumed_budget_reduces_next_cycle_available_balance(
        self,
        repository: RunRepository,
    ) -> None:
        limits = cycle_budget_limits(
            model_calls=2,
            input_tokens=10,
            output_tokens=10,
            canonical_token_equivalents=20,
            latency_us=100,
            max_call_latency_us=100,
        )
        first_reservation = BudgetAmounts(
            model_calls=1,
            input_tokens=10,
            output_tokens=5,
            canonical_token_equivalents=15,
            latency_us=50,
        )
        context, _reserved, _running = await advance_cycle_to_running(
            repository,
            limits=limits,
            reservation=first_reservation,
        )
        first_settlement = BudgetAmounts(
            model_calls=1,
            input_tokens=6,
            output_tokens=2,
            canonical_token_equivalents=8,
            latency_us=20,
        )
        await repository.commit_cycle(cycle_commit_command(context, settlement=first_settlement))
        second = await begin_cycle_context(repository, ordinal=2)
        over_remaining = second.reserve.model_copy(
            update={"reservation": BudgetAmounts(model_calls=1, input_tokens=5)}
        )

        with pytest.raises(ProjectionInvariantError, match="reservation"):
            await repository.reserve_cycle(over_remaining)

        exact_remaining = second.reserve.model_copy(
            update={"reservation": BudgetAmounts(model_calls=1, input_tokens=4)}
        )
        reserved = await repository.reserve_cycle(exact_remaining)
        assert reserved.budget_snapshot.consumed == first_settlement
        assert reserved.budget_snapshot.reserved == exact_remaining.reservation

    async def test_over_settlement_rejects_entire_commit(
        self,
        repository: RunRepository,
    ) -> None:
        context, _reserved, _running = await advance_cycle_to_running(repository)
        before_ledger = await repository.ledger(CYCLE_RUN_ID)
        before_snapshot = await repository.snapshot(CYCLE_RUN_ID)
        before_budget = await repository.budget_snapshot(CYCLE_RUN_ID)
        settlement = cycle_commit_command(context).settlement
        over = settlement.model_copy(
            update={
                "input_tokens": context.reserve.reservation.input_tokens + 1,
            }
        )

        with pytest.raises(ProjectionInvariantError, match="settlement"):
            await repository.commit_cycle(cycle_commit_command(context, settlement=over))

        assert await repository.ledger(CYCLE_RUN_ID) == before_ledger
        assert await repository.snapshot(CYCLE_RUN_ID) == before_snapshot
        assert await repository.budget_snapshot(CYCLE_RUN_ID) == before_budget

    async def test_failed_commit_validation_is_atomic_and_cycle_remains_recoverable(
        self,
        repository: RunRepository,
    ) -> None:
        context, _reserved, _running = await advance_cycle_to_running(repository)
        invalid_delta = MemoryDelta(
            delta_id=UUID("00000000-0000-4000-8000-000000000309"),
            run_id=CYCLE_RUN_ID,
            updates=(
                MemoryUpdate(
                    memory_id=CYCLE_MISSING_MEMORY_ID,
                    expected_revision=1,
                    content="This update must never be projected.",
                ),
            ),
            created_at=context.commit_time,
        )
        before_ledger = await repository.ledger(CYCLE_RUN_ID)
        before_snapshot = await repository.snapshot(CYCLE_RUN_ID)
        before_budget = await repository.budget_snapshot(CYCLE_RUN_ID)

        with pytest.raises(RevisionConflictError):
            await repository.commit_cycle(cycle_commit_command(context, delta=invalid_delta))

        assert await repository.ledger(CYCLE_RUN_ID) == before_ledger
        assert await repository.snapshot(CYCLE_RUN_ID) == before_snapshot
        assert await repository.budget_snapshot(CYCLE_RUN_ID) == before_budget
        assert cycle_records(before_ledger)[-1].state is CycleState.RUNNING

        committed = await repository.commit_cycle(cycle_commit_command(context))
        assert committed.cycle.state is CycleState.COMMITTED

    async def test_commit_outputs_cannot_predate_running_revision(
        self,
        repository: RunRepository,
    ) -> None:
        context, _reserved, running = await advance_cycle_to_running(repository)
        command = cycle_commit_command(context)
        before_ledger = await repository.ledger(CYCLE_RUN_ID)
        early = running.cycle.updated_at - timedelta(microseconds=1)
        early_delta = command.model_copy(
            update={
                "validated_delta": command.validated_delta.model_copy(update={"created_at": early})
            }
        )
        early_intervention = command.model_copy(
            update={"intervention": command.intervention.model_copy(update={"created_at": early})}
        )

        for invalid in (early_delta, early_intervention):
            with pytest.raises(ProjectionInvariantError, match="timestamp"):
                await repository.commit_cycle(invalid)
            assert await repository.ledger(CYCLE_RUN_ID) == before_ledger

        before_delta = command.model_copy(
            update={
                "intervention": command.intervention.model_copy(
                    update={
                        "created_at": command.validated_delta.created_at - timedelta(microseconds=1)
                    }
                )
            }
        )
        with pytest.raises(ProjectionInvariantError, match="precede"):
            await repository.commit_cycle(before_delta)
        assert await repository.ledger(CYCLE_RUN_ID) == before_ledger

    async def test_failure_before_running_releases_tokens_but_retains_latency(
        self,
        repository: RunRepository,
    ) -> None:
        context = await begin_cycle_context(repository)
        reserved = await repository.reserve_cycle(context.reserve)
        invalid = FailCycle(
            run_id=CYCLE_RUN_ID,
            cycle_id=context.cycle_id,
            expected_revision=reserved.cycle.revision,
            reason=ReasonCode.MODEL_ERROR,
            settlement=BudgetAmounts(
                canonical_token_equivalents=1,
                latency_us=1,
            ),
            updated_at=reserved.cycle.updated_at + timedelta(seconds=1),
        )
        before = await repository.ledger(CYCLE_RUN_ID)

        with pytest.raises(InvalidRecordError):
            await repository.fail_cycle(invalid)
        assert await repository.ledger(CYCLE_RUN_ID) == before

        latency_only = invalid.model_copy(update={"settlement": BudgetAmounts(latency_us=1)})
        failed = await repository.fail_cycle(latency_only)
        assert failed.cycle.state is CycleState.FAILED
        assert failed.budget_snapshot.consumed == BudgetAmounts(latency_us=1)

    @pytest.mark.parametrize("stage", ("pending", "reserved", "running"))
    async def test_explicit_failure_is_idempotent_and_never_advances_memory(
        self,
        repository: RunRepository,
        stage: str,
    ) -> None:
        context = await begin_cycle_context(repository)
        base = context.pending
        settlement: BudgetAmounts | None = None
        digests: tuple[str, ...] = ()
        if stage in ("reserved", "running"):
            base = await repository.reserve_cycle(context.reserve)
            settlement = BudgetAmounts()
        if stage == "running":
            base = await repository.mark_cycle_running(context.start)
            settlement = BudgetAmounts(model_calls=1, latency_us=1_000)
            digests = ("b" * 64,)
        failure = FailCycle(
            run_id=CYCLE_RUN_ID,
            cycle_id=context.cycle_id,
            expected_revision=base.cycle.revision,
            reason=ReasonCode.MODEL_ERROR,
            settlement=settlement,
            model_call_digests=digests,
            model_call_latencies_us=((1_000,) if digests else ()),
            updated_at=base.cycle.updated_at + timedelta(seconds=1),
        )
        failed = await repository.fail_cycle(failure)
        ledger_after_failure = await repository.ledger(CYCLE_RUN_ID)

        retry = await repository.fail_cycle(failure)

        assert failed.cycle.state is CycleState.FAILED
        assert retry == failed.model_copy(update={"appended": False})
        assert await repository.ledger(CYCLE_RUN_ID) == ledger_after_failure
        snapshot = await repository.snapshot(CYCLE_RUN_ID)
        assert snapshot.memory_cursor == 0
        assert snapshot.records == ()

    async def test_recovery_leaves_pending_cycle_resumable_without_writing(
        self,
        repository: RunRepository,
    ) -> None:
        context = await begin_cycle_context(repository)
        ledger_before = await repository.ledger(CYCLE_RUN_ID)

        first = await repository.recover_cycles(
            CYCLE_RUN_ID,
            recovered_at=context.commit_time,
        )
        second = await repository.recover_cycles(
            CYCLE_RUN_ID,
            recovered_at=context.commit_time,
        )

        assert first.resumable_pending == (context.pending.cycle,)
        assert first.resumable_reserved == ()
        assert first.failed_unknown_cost == ()
        assert second == first
        assert await repository.ledger(CYCLE_RUN_ID) == ledger_before

    async def test_recovery_leaves_reserved_cycle_and_hold_resumable(
        self,
        repository: RunRepository,
    ) -> None:
        context = await begin_cycle_context(repository)
        reserved = await repository.reserve_cycle(context.reserve)
        ledger_before = await repository.ledger(CYCLE_RUN_ID)

        first = await repository.recover_cycles(
            CYCLE_RUN_ID,
            recovered_at=context.commit_time,
        )
        second = await repository.recover_cycles(
            CYCLE_RUN_ID,
            recovered_at=context.commit_time,
        )

        assert first.resumable_pending == ()
        assert first.resumable_reserved == (reserved.cycle,)
        assert first.failed_unknown_cost == ()
        assert second == first
        assert await repository.ledger(CYCLE_RUN_ID) == ledger_before
        assert (
            await repository.budget_snapshot(CYCLE_RUN_ID)
        ).reserved == context.reserve.reservation

    async def test_running_recovery_consumes_full_reservation_once(
        self,
        repository: RunRepository,
    ) -> None:
        context, _reserved, _running = await advance_cycle_to_running(repository)
        ledger_before = await repository.ledger(CYCLE_RUN_ID)

        first = await repository.recover_cycles(
            CYCLE_RUN_ID,
            recovered_at=context.commit_time,
        )
        ledger_after = await repository.ledger(CYCLE_RUN_ID)
        second = await repository.recover_cycles(
            CYCLE_RUN_ID,
            recovered_at=context.commit_time + timedelta(seconds=1),
        )

        assert len(first.failed_unknown_cost) == 1
        failed = first.failed_unknown_cost[0]
        assert failed.cycle.state is CycleState.FAILED
        assert failed.cycle.failure_reason is ReasonCode.FAILED_UNKNOWN_COST
        assert failed.cycle.budget_settlement == context.reserve.reservation
        assert len(ledger_after) == len(ledger_before) + 1
        assert second.failed_unknown_cost == ()
        assert await repository.ledger(CYCLE_RUN_ID) == ledger_after
        budget = await repository.budget_snapshot(CYCLE_RUN_ID)
        assert budget.reserved == BudgetAmounts()
        assert budget.consumed == context.reserve.reservation
        snapshot = await repository.snapshot(CYCLE_RUN_ID)
        assert snapshot.memory_cursor == 0
        assert snapshot.records == ()
        assert (await repository.rebuild(CYCLE_RUN_ID)).equivalent

    async def test_concurrent_identical_reservations_create_one_hold(
        self,
        repository: RunRepository,
    ) -> None:
        context = await begin_cycle_context(repository)

        first, second = await asyncio.gather(
            repository.reserve_cycle(context.reserve),
            repository.reserve_cycle(context.reserve),
        )

        assert sorted((first.appended, second.appended)) == [False, True]
        assert first.cycle == second.cycle
        records = cycle_records(await repository.ledger(CYCLE_RUN_ID))
        assert tuple(record.state for record in records) == (
            CycleState.PENDING,
            CycleState.RESERVED,
        )
        assert (
            await repository.budget_snapshot(CYCLE_RUN_ID)
        ).reserved == context.reserve.reservation

    async def test_reminder_cycle_and_pending_delivery_commit_as_adjacent_entries(
        self,
        repository: RunRepository,
    ) -> None:
        context, _reserved, _running = await advance_cycle_to_running(repository)
        command = reminder_commit_command(context)
        before = await repository.ledger(CYCLE_RUN_ID)

        committed = await repository.commit_cycle(command)
        after = await repository.ledger(CYCLE_RUN_ID)

        delivery = committed.delivery
        assert delivery is not None
        assert committed.appended
        assert delivery.state is DeliveryState.PENDING
        assert delivery.revision == 1
        assert delivery.cycle_id == committed.cycle.cycle_id
        assert delivery.intervention_id == command.intervention.intervention_id
        assert command.delivery is not None
        assert delivery.target_request_id == command.delivery.target_request_id
        assert delivery.created_at == committed.cycle.updated_at
        assert len(after) == len(before) + 2
        assert after[-2].record == committed.cycle
        assert after[-1].record == delivery
        assert await repository.delivery(CYCLE_RUN_ID, delivery.delivery_id) == delivery

    async def test_silence_cycle_enqueues_no_delivery(
        self,
        repository: RunRepository,
    ) -> None:
        context, _reserved, _running = await advance_cycle_to_running(repository)
        command = cycle_commit_command(context)
        before = await repository.ledger(CYCLE_RUN_ID)

        committed = await repository.commit_cycle(command)
        after = await repository.ledger(CYCLE_RUN_ID)

        assert command.intervention.action is InterventionAction.SILENCE
        assert command.delivery is None
        assert committed.delivery is None
        assert len(after) == len(before) + 1
        assert after[-1].record == committed.cycle
        assert delivery_records(after) == ()

    async def test_delivery_plan_retry_is_stable_and_changed_plan_conflicts(
        self,
        repository: RunRepository,
    ) -> None:
        context, _reserved, _running = await advance_cycle_to_running(repository)
        command = reminder_commit_command(context)
        committed = await repository.commit_cycle(command)
        ledger_after_commit = await repository.ledger(CYCLE_RUN_ID)

        retry = await repository.commit_cycle(command)

        assert retry == committed.model_copy(update={"appended": False})
        assert retry.delivery is not None
        assert committed.delivery is not None
        assert retry.delivery.delivery_id == committed.delivery.delivery_id
        assert await repository.ledger(CYCLE_RUN_ID) == ledger_after_commit

        assert command.delivery is not None
        changed = command.model_copy(
            update={
                "delivery": command.delivery.model_copy(
                    update={"target_request_id": "different-request"}
                )
            }
        )
        with pytest.raises(CycleConflictError):
            await repository.commit_cycle(changed)
        assert await repository.ledger(CYCLE_RUN_ID) == ledger_after_commit

    async def test_delivery_tokens_fence_claim_races_and_stale_attempt_envelopes(
        self,
        repository: RunRepository,
    ) -> None:
        _context, _commit, _committed, pending = await commit_grounded_delivery(repository)
        claim_a = claim_command(pending, DELIVERY_CLAIM_A)
        claim_b = claim_command(pending, DELIVERY_CLAIM_B)

        raced = await asyncio.gather(
            repository.claim_delivery(claim_a),
            repository.claim_delivery(claim_b),
            return_exceptions=True,
        )
        receipts = tuple(item for item in raced if not isinstance(item, BaseException))
        failures = tuple(item for item in raced if isinstance(item, BaseException))

        assert len(receipts) == 1
        assert len(failures) == 1
        assert isinstance(failures[0], DeliveryRevisionConflictError)
        claimed_receipt = cast(Any, receipts[0])
        claimed = claimed_receipt.delivery
        winning_claim = claimed.claim_id
        assert winning_claim in (DELIVERY_CLAIM_A, DELIVERY_CLAIM_B)
        winning_command = claim_a if winning_claim == DELIVERY_CLAIM_A else claim_b
        losing_claim = DELIVERY_CLAIM_B if winning_claim == DELIVERY_CLAIM_A else DELIVERY_CLAIM_A

        claim_retry = await repository.claim_delivery(winning_command)
        assert not claim_retry.appended
        assert claim_retry.delivery == claimed

        wrong_attempt = attempt_command(
            claimed,
            losing_claim,
            DELIVERY_ATTEMPT_A,
        )
        with pytest.raises(DeliveryOwnershipError):
            await repository.begin_delivery_attempt(wrong_attempt)

        begin = attempt_command(claimed, winning_claim, DELIVERY_ATTEMPT_A)
        attempted_receipt = await repository.begin_delivery_attempt(begin)
        attempted = attempted_receipt.delivery
        committed_intervention = _committed.cycle.intervention
        assert committed_intervention is not None
        assert attempted_receipt.appended
        assert attempted_receipt.envelope is not None
        assert attempted_receipt.envelope.claim_id == winning_claim
        assert attempted_receipt.envelope.attempt_id == DELIVERY_ATTEMPT_A
        assert attempted_receipt.envelope.rendered_text == committed_intervention.rendered_text

        begin_retry = await repository.begin_delivery_attempt(begin)
        assert not begin_retry.appended
        assert begin_retry.envelope is None
        assert begin_retry.delivery == attempted

        unknown = await repository.mark_delivery_unknown(
            unknown_command(attempted, winning_claim, DELIVERY_ATTEMPT_A)
        )
        ledger_after_unknown = await repository.ledger(CYCLE_RUN_ID)
        with pytest.raises(DeliveryRevisionConflictError):
            await repository.begin_delivery_attempt(begin)
        assert await repository.ledger(CYCLE_RUN_ID) == ledger_after_unknown
        assert unknown.delivery.state is DeliveryState.UNKNOWN

    async def test_successful_and_failed_delivery_completions_are_idempotent(
        self,
        repository_factory: RepositoryFactory,
    ) -> None:
        successful_repository = repository_factory()
        _context, _command, _committed, pending = await commit_grounded_delivery(
            successful_repository
        )
        claimed = (
            await successful_repository.claim_delivery(claim_command(pending, DELIVERY_CLAIM_A))
        ).delivery
        attempted = (
            await successful_repository.begin_delivery_attempt(
                attempt_command(claimed, DELIVERY_CLAIM_A, DELIVERY_ATTEMPT_A)
            )
        ).delivery
        success_command = complete_command(
            attempted,
            DELIVERY_CLAIM_A,
            DELIVERY_ATTEMPT_A,
            DeliveryOutcome.DELIVERED,
        )

        delivered = await successful_repository.complete_delivery(success_command)
        delivered_retry = await successful_repository.complete_delivery(success_command)

        assert delivered.delivery.state is DeliveryState.DELIVERED
        assert delivered.delivery.outcome is DeliveryOutcome.DELIVERED
        assert delivered.delivery.reason_code is ReasonCode.DELIVERY_SUCCEEDED
        assert delivered.delivery.receipt == {"provider_receipt_id": "provider-receipt-1"}
        assert delivered_retry == delivered.model_copy(update={"appended": False})

        failed_repository = repository_factory()
        _context, _command, _committed, pending = await commit_grounded_delivery(failed_repository)
        claimed = (
            await failed_repository.claim_delivery(claim_command(pending, DELIVERY_CLAIM_A))
        ).delivery
        attempted = (
            await failed_repository.begin_delivery_attempt(
                attempt_command(claimed, DELIVERY_CLAIM_A, DELIVERY_ATTEMPT_A)
            )
        ).delivery
        failure_command = complete_command(
            attempted,
            DELIVERY_CLAIM_A,
            DELIVERY_ATTEMPT_A,
            DeliveryOutcome.FAILED,
        )

        failed = await failed_repository.complete_delivery(failure_command)
        failed_retry = await failed_repository.complete_delivery(failure_command)

        assert failed.delivery.state is DeliveryState.FAILED
        assert failed.delivery.outcome is DeliveryOutcome.FAILED
        assert failed.delivery.reason_code is ReasonCode.DELIVERY_FAILED
        assert failed.delivery.receipt is None
        assert failed_retry == failed.model_copy(update={"appended": False})

    async def test_historical_delivery_retries_reject_changed_ownership_tokens(
        self,
        repository_factory: RepositoryFactory,
    ) -> None:
        begin_repository = repository_factory()
        _context, _command, _committed, pending = await commit_grounded_delivery(begin_repository)
        claimed = (
            await begin_repository.claim_delivery(claim_command(pending, DELIVERY_CLAIM_A))
        ).delivery
        begin = attempt_command(claimed, DELIVERY_CLAIM_A, DELIVERY_ATTEMPT_A)
        await begin_repository.begin_delivery_attempt(begin)
        begin_ledger = await begin_repository.ledger(CYCLE_RUN_ID)

        for changed in (
            begin.model_copy(update={"claim_id": DELIVERY_CLAIM_B}),
            begin.model_copy(update={"attempt_id": DELIVERY_ATTEMPT_B}),
        ):
            with pytest.raises(DeliveryOwnershipError):
                await begin_repository.begin_delivery_attempt(changed)
        assert await begin_repository.ledger(CYCLE_RUN_ID) == begin_ledger

        complete_repository = repository_factory()
        _context, _command, _committed, pending = await commit_grounded_delivery(
            complete_repository
        )
        claimed = (
            await complete_repository.claim_delivery(claim_command(pending, DELIVERY_CLAIM_A))
        ).delivery
        attempted = (
            await complete_repository.begin_delivery_attempt(
                attempt_command(claimed, DELIVERY_CLAIM_A, DELIVERY_ATTEMPT_A)
            )
        ).delivery
        complete = complete_command(
            attempted,
            DELIVERY_CLAIM_A,
            DELIVERY_ATTEMPT_A,
            DeliveryOutcome.DELIVERED,
        )
        await complete_repository.complete_delivery(complete)
        complete_ledger = await complete_repository.ledger(CYCLE_RUN_ID)

        for changed in (
            complete.model_copy(update={"claim_id": DELIVERY_CLAIM_B}),
            complete.model_copy(update={"attempt_id": DELIVERY_ATTEMPT_B}),
        ):
            with pytest.raises(DeliveryOwnershipError):
                await complete_repository.complete_delivery(changed)
        assert await complete_repository.ledger(CYCLE_RUN_ID) == complete_ledger

        unknown_repository = repository_factory()
        _context, _command, _committed, pending = await commit_grounded_delivery(unknown_repository)
        claimed = (
            await unknown_repository.claim_delivery(claim_command(pending, DELIVERY_CLAIM_A))
        ).delivery
        attempted = (
            await unknown_repository.begin_delivery_attempt(
                attempt_command(claimed, DELIVERY_CLAIM_A, DELIVERY_ATTEMPT_A)
            )
        ).delivery
        mark_unknown = unknown_command(
            attempted,
            DELIVERY_CLAIM_A,
            DELIVERY_ATTEMPT_A,
        )
        await unknown_repository.mark_delivery_unknown(mark_unknown)
        unknown_ledger = await unknown_repository.ledger(CYCLE_RUN_ID)

        for changed in (
            mark_unknown.model_copy(update={"claim_id": DELIVERY_CLAIM_B}),
            mark_unknown.model_copy(update={"attempt_id": DELIVERY_ATTEMPT_B}),
        ):
            with pytest.raises(DeliveryOwnershipError):
                await unknown_repository.mark_delivery_unknown(changed)
        assert await unknown_repository.ledger(CYCLE_RUN_ID) == unknown_ledger

        reject_repository = repository_factory()
        _context, _command, _committed, pending = await commit_grounded_delivery(reject_repository)
        claimed = (
            await reject_repository.claim_delivery(claim_command(pending, DELIVERY_CLAIM_A))
        ).delivery
        reject = RejectDelivery(
            run_id=claimed.run_id,
            delivery_id=claimed.delivery_id,
            expected_revision=claimed.revision,
            claim_id=DELIVERY_CLAIM_A,
            reason_code=ReasonCode.UNSAFE_ROLE_MAPPING,
            updated_at=claimed.updated_at + timedelta(seconds=1),
        )
        await reject_repository.reject_delivery(reject)
        reject_ledger = await reject_repository.ledger(CYCLE_RUN_ID)

        with pytest.raises(DeliveryOwnershipError):
            await reject_repository.reject_delivery(
                reject.model_copy(update={"claim_id": DELIVERY_CLAIM_B})
            )
        assert await reject_repository.ledger(CYCLE_RUN_ID) == reject_ledger

    async def test_unknown_retry_requires_deduplication_and_new_tokens(
        self,
        repository_factory: RepositoryFactory,
    ) -> None:
        deduplicating = repository_factory()
        _context, _command, _committed, pending = await commit_grounded_delivery(deduplicating)
        claimed = (
            await deduplicating.claim_delivery(claim_command(pending, DELIVERY_CLAIM_A))
        ).delivery
        attempted = (
            await deduplicating.begin_delivery_attempt(
                attempt_command(claimed, DELIVERY_CLAIM_A, DELIVERY_ATTEMPT_A)
            )
        ).delivery
        unknown = (
            await deduplicating.mark_delivery_unknown(
                unknown_command(attempted, DELIVERY_CLAIM_A, DELIVERY_ATTEMPT_A)
            )
        ).delivery

        with pytest.raises(ProjectionInvariantError, match="claim owner"):
            await deduplicating.claim_delivery(claim_command(unknown, DELIVERY_CLAIM_A))

        reclaimed = (
            await deduplicating.claim_delivery(claim_command(unknown, DELIVERY_CLAIM_B))
        ).delivery
        retried_receipt = await deduplicating.begin_delivery_attempt(
            attempt_command(reclaimed, DELIVERY_CLAIM_B, DELIVERY_ATTEMPT_B)
        )
        assert reclaimed.claim_id == DELIVERY_CLAIM_B
        assert reclaimed.attempt_id is None
        assert reclaimed.attempt_count == 1
        assert retried_receipt.delivery.attempt_count == 2
        assert retried_receipt.envelope is not None
        assert retried_receipt.envelope.attempt_id == DELIVERY_ATTEMPT_B
        assert retried_receipt.envelope.attempt_number == 2

        second_unknown = (
            await deduplicating.mark_delivery_unknown(
                unknown_command(
                    retried_receipt.delivery,
                    DELIVERY_CLAIM_B,
                    DELIVERY_ATTEMPT_B,
                )
            )
        ).delivery
        with pytest.raises(ProjectionInvariantError, match="claim owner token was reused"):
            await deduplicating.claim_delivery(claim_command(second_unknown, DELIVERY_CLAIM_A))
        third_claim = (
            await deduplicating.claim_delivery(claim_command(second_unknown, DELIVERY_CLAIM_C))
        ).delivery
        with pytest.raises(ProjectionInvariantError, match="attempt token was reused"):
            await deduplicating.begin_delivery_attempt(
                attempt_command(third_claim, DELIVERY_CLAIM_C, DELIVERY_ATTEMPT_A)
            )

        non_deduplicating = repository_factory()
        _context, _command, _committed, pending = await commit_grounded_delivery(
            non_deduplicating,
            adapter_deduplicates=False,
        )
        claimed = (
            await non_deduplicating.claim_delivery(claim_command(pending, DELIVERY_CLAIM_A))
        ).delivery
        attempted = (
            await non_deduplicating.begin_delivery_attempt(
                attempt_command(claimed, DELIVERY_CLAIM_A, DELIVERY_ATTEMPT_A)
            )
        ).delivery
        unknown = (
            await non_deduplicating.mark_delivery_unknown(
                unknown_command(attempted, DELIVERY_CLAIM_A, DELIVERY_ATTEMPT_A)
            )
        ).delivery
        before = await non_deduplicating.ledger(CYCLE_RUN_ID)

        with pytest.raises(InvalidDeliveryStateError):
            await non_deduplicating.claim_delivery(claim_command(unknown, DELIVERY_CLAIM_B))
        assert await non_deduplicating.ledger(CYCLE_RUN_ID) == before

    async def test_late_completion_only_wins_before_an_unknown_retry_is_claimed(
        self,
        repository_factory: RepositoryFactory,
    ) -> None:
        uncontested = repository_factory()
        _context, _command, _committed, pending = await commit_grounded_delivery(uncontested)
        claimed = (
            await uncontested.claim_delivery(claim_command(pending, DELIVERY_CLAIM_A))
        ).delivery
        attempted = (
            await uncontested.begin_delivery_attempt(
                attempt_command(claimed, DELIVERY_CLAIM_A, DELIVERY_ATTEMPT_A)
            )
        ).delivery
        unknown = (
            await uncontested.mark_delivery_unknown(
                unknown_command(attempted, DELIVERY_CLAIM_A, DELIVERY_ATTEMPT_A)
            )
        ).delivery

        late = await uncontested.complete_delivery(
            complete_command(
                unknown,
                DELIVERY_CLAIM_A,
                DELIVERY_ATTEMPT_A,
                DeliveryOutcome.DELIVERED,
            )
        )
        assert late.delivery.state is DeliveryState.DELIVERED
        assert late.delivery.attempt_count == 1

        contested = repository_factory()
        _context, _command, _committed, pending = await commit_grounded_delivery(contested)
        claimed = (
            await contested.claim_delivery(claim_command(pending, DELIVERY_CLAIM_A))
        ).delivery
        attempted = (
            await contested.begin_delivery_attempt(
                attempt_command(claimed, DELIVERY_CLAIM_A, DELIVERY_ATTEMPT_A)
            )
        ).delivery
        unknown = (
            await contested.mark_delivery_unknown(
                unknown_command(attempted, DELIVERY_CLAIM_A, DELIVERY_ATTEMPT_A)
            )
        ).delivery
        reclaimed = (
            await contested.claim_delivery(claim_command(unknown, DELIVERY_CLAIM_B))
        ).delivery
        before = await contested.ledger(CYCLE_RUN_ID)

        with pytest.raises(DeliveryRevisionConflictError):
            await contested.complete_delivery(
                complete_command(
                    unknown,
                    DELIVERY_CLAIM_A,
                    DELIVERY_ATTEMPT_A,
                    DeliveryOutcome.DELIVERED,
                )
            )
        assert await contested.ledger(CYCLE_RUN_ID) == before
        assert await contested.delivery(CYCLE_RUN_ID, reclaimed.delivery_id) == reclaimed

    async def test_delivery_recovery_is_idempotent_and_classifies_retry_safety(
        self,
        repository_factory: RepositoryFactory,
    ) -> None:
        deduplicating = repository_factory()
        _context, _command, _committed, pending = await commit_grounded_delivery(deduplicating)
        claimed = (
            await deduplicating.claim_delivery(claim_command(pending, DELIVERY_CLAIM_A))
        ).delivery
        attempted = (
            await deduplicating.begin_delivery_attempt(
                attempt_command(claimed, DELIVERY_CLAIM_A, DELIVERY_ATTEMPT_A)
            )
        ).delivery
        recovered_at = attempted.updated_at + timedelta(seconds=1)

        first = await deduplicating.recover_deliveries(
            CYCLE_RUN_ID,
            recovered_at=recovered_at,
        )
        ledger_after_first = await deduplicating.ledger(CYCLE_RUN_ID)
        second = await deduplicating.recover_deliveries(
            CYCLE_RUN_ID,
            recovered_at=recovered_at + timedelta(seconds=1),
        )

        assert len(first.marked_unknown) == 1
        recovered_unknown = first.marked_unknown[0].delivery
        assert recovered_unknown.state is DeliveryState.UNKNOWN
        assert first.retryable_unknown == (recovered_unknown,)
        assert first.non_retryable_unknown == ()
        assert second.marked_unknown == ()
        assert second.retryable_unknown == first.retryable_unknown
        assert await deduplicating.ledger(CYCLE_RUN_ID) == ledger_after_first

        reclaimed = (
            await deduplicating.claim_delivery(claim_command(recovered_unknown, DELIVERY_CLAIM_B))
        ).delivery
        claimed_recovery = await deduplicating.recover_deliveries(
            CYCLE_RUN_ID,
            recovered_at=reclaimed.updated_at + timedelta(seconds=1),
        )
        assert claimed_recovery.marked_unknown == ()
        assert claimed_recovery.resumable_claimed == (reclaimed,)

        non_deduplicating = repository_factory()
        _context, _command, _committed, pending = await commit_grounded_delivery(
            non_deduplicating,
            adapter_deduplicates=False,
        )
        claimed = (
            await non_deduplicating.claim_delivery(claim_command(pending, DELIVERY_CLAIM_A))
        ).delivery
        attempted = (
            await non_deduplicating.begin_delivery_attempt(
                attempt_command(claimed, DELIVERY_CLAIM_A, DELIVERY_ATTEMPT_A)
            )
        ).delivery
        recovered = await non_deduplicating.recover_deliveries(
            CYCLE_RUN_ID,
            recovered_at=attempted.updated_at + timedelta(seconds=1),
        )
        assert recovered.retryable_unknown == ()
        assert recovered.non_retryable_unknown == (recovered.marked_unknown[0].delivery,)

        pending_repository = repository_factory()
        _context, _command, _committed, pending = await commit_grounded_delivery(pending_repository)
        pending_recovery = await pending_repository.recover_deliveries(
            CYCLE_RUN_ID,
            recovered_at=pending.updated_at + timedelta(seconds=1),
        )
        assert pending_recovery.marked_unknown == ()
        assert pending_recovery.resumable_pending == (pending,)

    async def test_delivery_rebuild_preserves_outbox_and_historical_retry_receipts(
        self,
        repository: RunRepository,
    ) -> None:
        _context, commit, committed, pending = await commit_grounded_delivery(repository)
        claimed = (
            await repository.claim_delivery(claim_command(pending, DELIVERY_CLAIM_A))
        ).delivery
        attempted = (
            await repository.begin_delivery_attempt(
                attempt_command(claimed, DELIVERY_CLAIM_A, DELIVERY_ATTEMPT_A)
            )
        ).delivery
        completion = complete_command(
            attempted,
            DELIVERY_CLAIM_A,
            DELIVERY_ATTEMPT_A,
            DeliveryOutcome.DELIVERED,
        )
        delivered = await repository.complete_delivery(completion)
        ledger_before = await repository.ledger(CYCLE_RUN_ID)

        rebuilt = await repository.rebuild(CYCLE_RUN_ID)

        assert rebuilt.equivalent
        assert await repository.delivery(CYCLE_RUN_ID, pending.delivery_id) == delivered.delivery
        complete_retry = await repository.complete_delivery(completion)
        commit_retry = await repository.commit_cycle(commit)
        assert complete_retry == delivered.model_copy(update={"appended": False})
        assert commit_retry == committed.model_copy(update={"appended": False})
        assert await repository.ledger(CYCLE_RUN_ID) == ledger_before
