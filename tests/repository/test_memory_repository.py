from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import replace
from typing import Any, cast
from uuid import UUID

import pytest
from tests.repository.conformance import (
    CONDITIONAL_EVENT_ID_A,
    CONDITIONAL_EVENT_ID_B,
    RUN_A,
    CycleRepositoryConformance,
    RepositoryConformance,
    RepositoryFactory,
    conditional_signal,
    event_draft,
    invocation_decision,
)

from saliencegate.domain import PayloadDigest, PayloadDigestAlgorithm, canonical_json
from saliencegate.ports.repository import (
    AppendDisposition,
    AppendReceipt,
    ConditionalBatchReceipt,
    ConditionalEventAppend,
    ConditionalSignalAppend,
    CrossRunReferenceError,
    DigestVerificationError,
    InvalidDraftError,
    InvalidRecordError,
    LedgerHeadConflictError,
    MemoryQuery,
    ProjectionInvariantError,
    RebuildError,
    RecordCollisionError,
    RunNotFoundError,
    UnsafeRecordContentError,
)
from saliencegate.repository import memory as memory_module
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.repository.projector import empty_projection
from saliencegate.security import (
    InstallationKey,
    RedactionFinding,
    RedactionPolicy,
    Redactor,
)


def deterministic_ids() -> Iterator[UUID]:
    for value in range(1, 10_000):
        yield UUID(f"00000000-0000-4000-8000-{value:012x}")


def create_repository() -> MemoryRunRepository:
    ids = deterministic_ids()
    return MemoryRunRepository(
        installation_key=InstallationKey(b"k" * 32),
        id_factory=lambda: next(ids),
    )


class TestMemoryRunRepository(RepositoryConformance, CycleRepositoryConformance):
    @pytest.fixture
    def repository_factory(self) -> RepositoryFactory:
        return create_repository


class NoopRedactor(Redactor):
    def _redact_value(
        self,
        value: object,
        path: str,
        findings: list[RedactionFinding],
    ) -> object:
        return value


def test_repository_does_not_accept_an_injectable_redactor() -> None:
    constructor = cast(Any, MemoryRunRepository)

    with pytest.raises(TypeError, match="redactor"):
        constructor(
            redactor=NoopRedactor(),
            installation_key=InstallationKey(b"k" * 32),
        )


def test_repository_rejects_redaction_policy_subclasses() -> None:
    class PolicySubclass(RedactionPolicy):
        pass

    with pytest.raises(TypeError, match="exactly RedactionPolicy"):
        MemoryRunRepository(
            redaction_policy=PolicySubclass(),
            installation_key=InstallationKey(b"k" * 32),
        )


async def test_repository_copies_immutable_redaction_policy_data() -> None:
    policy = RedactionPolicy(literal_secrets=("configured-literal",))
    repository = MemoryRunRepository(
        redaction_policy=policy,
        installation_key=InstallationKey(b"k" * 32),
    )
    object.__setattr__(policy, "literal_secrets", ())

    receipt = await repository.append(
        event_draft(payload={"message": "contains configured-literal"})
    )

    assert receipt.event.payload == {"message": "contains [REDACTED]"}


async def test_rebuild_rejects_tampered_ledger_without_replacing_projection() -> None:
    repository = create_repository()
    await repository.append(event_draft())
    before = await repository.snapshot(RUN_A)
    slot = repository._slots[RUN_A]
    original_entry = slot.ledger[0]
    tampered_event = original_entry.record.model_copy(update={"source_adapter": "tampered-adapter"})
    slot.ledger = (original_entry.model_copy(update={"record": tampered_event}),)

    with pytest.raises(DigestVerificationError, match="record"):
        await repository.ledger(RUN_A)
    with pytest.raises(RebuildError):
        await repository.rebuild(RUN_A)

    assert await repository.snapshot(RUN_A) == before


async def test_authenticated_head_detects_ledger_suffix_truncation() -> None:
    repository = create_repository()
    await repository.append(event_draft(source_event_id="source-1"))
    await repository.append(event_draft(source_event_id="source-2"))
    slot = repository._slots[RUN_A]
    before = slot.projection
    slot.ledger = slot.ledger[:1]

    with pytest.raises(DigestVerificationError, match="head"):
        await repository.ledger(RUN_A)
    with pytest.raises(DigestVerificationError, match="head"):
        await repository.snapshot(RUN_A)
    with pytest.raises(DigestVerificationError, match="head"):
        await repository.search(MemoryQuery(run_id=RUN_A))
    with pytest.raises(RebuildError):
        await repository.rebuild(RUN_A)

    assert slot.projection == before


async def test_mutation_cannot_overwrite_evidence_of_a_truncated_suffix() -> None:
    repository = create_repository()
    await repository.append(event_draft())
    await repository.record_invocation_decision(invocation_decision())
    slot = repository._slots[RUN_A]
    original_head = slot.ledger_head
    slot.ledger = slot.ledger[:1]

    with pytest.raises(DigestVerificationError, match="head"):
        await repository.record_invocation_decision(invocation_decision())

    assert slot.ledger_head == original_head
    assert len(slot.ledger) == 1


async def test_repository_owns_a_private_copy_of_the_integrity_key() -> None:
    ids = deterministic_ids()
    key = InstallationKey(b"k" * 32)
    repository = MemoryRunRepository(
        installation_key=key,
        id_factory=lambda: next(ids),
    )
    await repository.append(event_draft(source_event_id="source-1"))
    object.__setattr__(key, "_material", b"z" * 32)
    await repository.append(event_draft(source_event_id="source-2"))

    assert (await repository.rebuild(RUN_A)).equivalent


async def test_repository_copies_ids_returned_by_the_id_factory() -> None:
    factory_id = UUID("00000000-0000-4000-8000-000000000177")
    repository = MemoryRunRepository(
        installation_key=InstallationKey(b"k" * 32),
        id_factory=lambda: factory_id,
    )
    receipt = await repository.append(event_draft())
    expected = UUID(str(receipt.event.event_id))
    object.__setattr__(factory_id, "int", UUID(int=178).int)

    event = (await repository.ledger(RUN_A))[0].record
    assert event.event_id == expected
    assert (await repository.rebuild(RUN_A)).equivalent


async def test_process_anchor_rejects_a_valid_historical_ledger_rollback() -> None:
    repository = create_repository()
    await repository.append(event_draft(source_event_id="source-1"))
    slot = repository._slots[RUN_A]
    prefix = (slot.ledger, slot.ledger_head, slot.projection)
    await repository.append(event_draft(source_event_id="source-2"))
    current_anchor = repository._trusted_heads[RUN_A]
    slot.ledger, slot.ledger_head, slot.projection = prefix

    with pytest.raises(DigestVerificationError, match="trusted ledger anchor"):
        await repository.ledger(RUN_A)
    with pytest.raises(DigestVerificationError, match="trusted ledger anchor"):
        await repository.append(event_draft(source_event_id="fork"))

    assert repository._trusted_heads[RUN_A] == current_anchor


async def test_process_anchor_rejects_removing_an_existing_run_slot() -> None:
    repository = create_repository()
    await repository.append(event_draft())
    del repository._slots[RUN_A]

    with pytest.raises(DigestVerificationError, match="trusted ledger anchor"):
        await repository.append(event_draft(source_event_id="replacement"))


async def test_process_anchor_rejects_emptying_an_existing_run_slot() -> None:
    repository = create_repository()
    event = (await repository.append(event_draft())).event
    trusted_head = repository._trusted_heads[RUN_A]
    trusted_projection = repository._trusted_projections[RUN_A]
    slot = repository._slots[RUN_A]
    slot.ledger = ()
    slot.ledger_head = None
    slot.projection = empty_projection(RUN_A)

    with pytest.raises(DigestVerificationError, match="trusted ledger anchor"):
        await repository.append_event_if_head(
            event_draft(source_event_id="conditional-replacement"),
            event_id=CONDITIONAL_EVENT_ID_A,
            expected_head=None,
        )
    with pytest.raises(DigestVerificationError, match="trusted ledger anchor"):
        await repository.record_signal_if_head(
            conditional_signal(event),
            expected_head=trusted_head,
        )
    with pytest.raises(DigestVerificationError, match="trusted ledger anchor"):
        await repository.append(event_draft(source_event_id="legacy-replacement"))

    assert repository._trusted_heads[RUN_A] == trusted_head
    assert repository._trusted_projections[RUN_A] == trusted_projection
    assert slot.ledger == ()
    assert slot.ledger_head is None


async def test_conditional_memory_writers_cannot_both_commit_the_same_head() -> None:
    repository = create_repository()
    await repository.append_event_if_head(
        event_draft(source_event_id="race-origin"),
        event_id=CONDITIONAL_EVENT_ID_A,
        expected_head=None,
    )
    expected_head = await repository.ledger_head(RUN_A)

    results = await asyncio.gather(
        repository.append_event_if_head(
            event_draft(source_event_id="race-first"),
            event_id=CONDITIONAL_EVENT_ID_B,
            expected_head=expected_head,
        ),
        repository.append_event_if_head(
            event_draft(source_event_id="race-second"),
            event_id=UUID("00000000-0000-4000-8000-000000000124"),
            expected_head=expected_head,
        ),
        return_exceptions=True,
    )

    successes = tuple(result for result in results if isinstance(result, AppendReceipt))
    conflicts = tuple(result for result in results if isinstance(result, LedgerHeadConflictError))
    assert len(successes) == 1
    assert successes[0].disposition is AppendDisposition.APPENDED
    assert len(conflicts) == 1
    assert len(await repository.ledger(RUN_A)) == 2


async def test_conditional_collision_preserves_projection_and_collision_cache() -> None:
    repository = create_repository()
    await repository.append_event_if_head(
        event_draft(payload={"safe": "original"}),
        event_id=CONDITIONAL_EVENT_ID_A,
        expected_head=None,
    )
    expected_head = await repository.ledger_head(RUN_A)
    expected_snapshot = await repository.snapshot(RUN_A)
    slot = repository._slots[RUN_A]
    expected_cache = dict(slot.collision_receipts)
    collision = event_draft(payload={"safe": "changed"})

    with pytest.raises(RecordCollisionError):
        await repository.append_event_if_head(
            collision,
            event_id=CONDITIONAL_EVENT_ID_A,
            expected_head=expected_head,
        )

    assert slot.collision_receipts == expected_cache
    assert await repository.snapshot(RUN_A) == expected_snapshot
    assert await repository.ledger_head(RUN_A) == expected_head
    assert len(await repository.ledger(RUN_A)) == 1

    legacy = await repository.append(collision)
    assert legacy.disposition is AppendDisposition.COLLISION
    assert len(slot.collision_receipts) == 1
    assert len(await repository.ledger(RUN_A)) == 2


async def test_head_authentication_rejects_a_retagged_in_memory_head() -> None:
    repository = create_repository()
    await repository.append(event_draft())
    slot = repository._slots[RUN_A]
    assert slot.ledger_head is not None
    tampered = slot.ledger_head.model_copy(
        update={
            "head_tag": PayloadDigest(
                algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
                value="0" * 64,
            )
        }
    )
    slot.ledger_head = tampered
    repository._trusted_heads[RUN_A] = tampered

    with pytest.raises(DigestVerificationError, match="ledger head"):
        await repository.ledger(RUN_A)


@pytest.mark.parametrize(
    "change",
    [
        {"position": 2},
        {"record_key": "trace_event:00000000-0000-4000-8000-000000000099"},
        {
            "chain_tag": PayloadDigest(
                algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
                value="0" * 64,
            )
        },
    ],
)
async def test_rebuild_rejects_tampered_order_key_or_chain(
    change: dict[str, object],
) -> None:
    repository = create_repository()
    await repository.append(event_draft())
    slot = repository._slots[RUN_A]
    before = slot.projection
    slot.ledger = (slot.ledger[0].model_copy(update=change),)

    with pytest.raises(RebuildError):
        await repository.rebuild(RUN_A)

    assert slot.projection == before


async def test_rebuild_repairs_projection_from_authenticated_ledger() -> None:
    repository = create_repository()
    await repository.append(event_draft())
    expected = await repository.snapshot(RUN_A)
    slot = repository._slots[RUN_A]
    assert slot.projection is not None
    original_head = slot.ledger_head
    slot.projection = replace(slot.projection, ingestion_cursor=0)

    with pytest.raises(DigestVerificationError, match="projection head"):
        await repository.snapshot(RUN_A)
    with pytest.raises(DigestVerificationError, match="projection head"):
        await repository.append(event_draft(source_event_id="source-2"))

    assert len(slot.ledger) == 1
    assert slot.ledger_head == original_head

    receipt = await repository.rebuild(RUN_A)

    assert not receipt.equivalent
    assert await repository.snapshot(RUN_A) == expected
    second = await repository.append(event_draft(source_event_id="source-2"))
    assert second.event.sequence == 2


async def test_rebuild_repairs_a_projection_that_cannot_be_digested() -> None:
    repository = create_repository()
    await repository.append(event_draft())
    expected = await repository.snapshot(RUN_A)
    slot = repository._slots[RUN_A]
    assert slot.projection is not None
    slot.projection = replace(
        slot.projection,
        signals={UUID("00000000-0000-4000-8000-000000000199"): cast(Any, object())},
    )

    receipt = await repository.rebuild(RUN_A)

    assert receipt.before is None
    assert not receipt.equivalent
    assert await repository.snapshot(RUN_A) == expected


async def test_rebuild_restores_collision_deduplication_index() -> None:
    repository = create_repository()
    await repository.append(event_draft())
    changed = event_draft(payload={"message": "changed"})
    collision = await repository.append(changed)
    assert collision.collision_event is not None
    slot = repository._slots[RUN_A]
    slot.collision_receipts = {}

    await repository.rebuild(RUN_A)
    repeated = await repository.append(changed)

    assert repeated.disposition is AppendDisposition.COLLISION
    assert repeated.collision_event == collision.collision_event
    assert len(await repository.ledger(RUN_A)) == 2


async def test_blocked_run_does_not_block_an_independent_run() -> None:
    repository = create_repository()
    await repository.append(event_draft(run_id=RUN_A, source_event_id="a-1"))
    slot = repository._slots[RUN_A]
    await slot.lock.acquire()
    blocked = asyncio.create_task(
        repository.append(event_draft(run_id=RUN_A, source_event_id="a-2"))
    )
    await asyncio.sleep(0)
    try:
        assert not blocked.done()
        independent = await asyncio.wait_for(
            repository.append(event_draft(run_id=UUID("00000000-0000-4000-8000-000000000102"))),
            timeout=0.2,
        )
        assert independent.event.sequence == 1
    finally:
        slot.lock.release()
    assert (await blocked).event.sequence == 2


async def test_cancelled_waiter_does_not_consume_an_id_or_sequence() -> None:
    repository = create_repository()
    first = await repository.append(event_draft(source_event_id="source-1"))
    slot = repository._slots[RUN_A]
    await slot.lock.acquire()
    cancelled = asyncio.create_task(
        repository.append(event_draft(source_event_id="source-cancelled"))
    )
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    slot.lock.release()

    second = await repository.append(event_draft(source_event_id="source-2"))

    assert first.event.event_id == UUID("00000000-0000-4000-8000-000000000001")
    assert second.event.event_id == UUID("00000000-0000-4000-8000-000000000002")
    assert second.event.sequence == 2


async def test_conditional_batch_verifies_and_publishes_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = create_repository()
    origin = (
        await repository.append_event_if_head(
            event_draft(source_event_id="batch-spy-origin"),
            event_id=CONDITIONAL_EVENT_ID_A,
            expected_head=None,
        )
    ).event
    head = await repository.ledger_head(RUN_A)
    next_draft = event_draft(
        source_event_id="batch-spy-next",
        parent_ids=(origin.event_id,),
    )
    signal = conditional_signal(origin).model_copy(
        update={"evidence_event_ids": (CONDITIONAL_EVENT_ID_B,)}
    )
    calls = {"verify": 0, "publish": 0}
    original_verify = repository._verify_head
    original_publish = repository._publish_shadow

    def verify_spy(*args: object, **kwargs: object) -> None:
        calls["verify"] += 1
        original_verify(*args, **kwargs)

    def publish_spy(*args: object, **kwargs: object) -> None:
        calls["publish"] += 1
        original_publish(*args, **kwargs)

    monkeypatch.setattr(repository, "_verify_head", verify_spy)
    monkeypatch.setattr(repository, "_publish_shadow", publish_spy)

    receipt = await repository.append_records_if_head(
        (
            ConditionalEventAppend(event=next_draft, event_id=CONDITIONAL_EVENT_ID_B),
            ConditionalSignalAppend(signal=signal),
        ),
        expected_head=head,
    )

    assert calls == {"verify": 1, "publish": 1}
    assert receipt.final_head.entry_count == head.entry_count + 2


async def test_conditional_batch_cas_is_checked_under_the_run_lock() -> None:
    repository = create_repository()
    await repository.append_event_if_head(
        event_draft(source_event_id="batch-lock-origin"),
        event_id=CONDITIONAL_EVENT_ID_A,
        expected_head=None,
    )
    head = await repository.ledger_head(RUN_A)
    slot = repository._slots[RUN_A]
    await slot.lock.acquire()
    tasks = tuple(
        asyncio.create_task(
            repository.append_records_if_head(
                (
                    ConditionalEventAppend(
                        event=event_draft(source_event_id=f"batch-lock-{index}"),
                        event_id=event_id,
                    ),
                ),
                expected_head=head,
            )
        )
        for index, event_id in enumerate(
            (
                CONDITIONAL_EVENT_ID_B,
                UUID("00000000-0000-4000-8000-000000000126"),
            )
        )
    )
    try:
        await asyncio.sleep(0)
        assert all(not task.done() for task in tasks)
    finally:
        slot.lock.release()
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert sum(isinstance(result, ConditionalBatchReceipt) for result in results) == 1
    assert sum(isinstance(result, LedgerHeadConflictError) for result in results) == 1
    assert len(await repository.ledger(RUN_A)) == 2


async def test_conditional_batch_injected_failure_preserves_slot_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = create_repository()
    origin = (
        await repository.append_event_if_head(
            event_draft(source_event_id="batch-cow-origin"),
            event_id=CONDITIONAL_EVENT_ID_A,
            expected_head=None,
        )
    ).event
    head = await repository.ledger_head(RUN_A)
    slot = repository._slots[RUN_A]
    before = (
        slot.ledger,
        slot.ledger_head,
        slot.projection,
        slot.direct_records,
        repository._trusted_heads[RUN_A],
        repository._trusted_projections[RUN_A],
    )
    signal = conditional_signal(origin).model_copy(
        update={"evidence_event_ids": (CONDITIONAL_EVENT_ID_B,)}
    )

    def fail_signal(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected batch failure")

    monkeypatch.setattr(repository, "_record_direct_locked", fail_signal)

    with pytest.raises(RuntimeError, match="injected batch failure"):
        await repository.append_records_if_head(
            (
                ConditionalEventAppend(
                    event=event_draft(source_event_id="batch-cow-staged"),
                    event_id=CONDITIONAL_EVENT_ID_B,
                ),
                ConditionalSignalAppend(signal=signal),
            ),
            expected_head=head,
        )

    after = (
        slot.ledger,
        slot.ledger_head,
        slot.projection,
        slot.direct_records,
        repository._trusted_heads[RUN_A],
        repository._trusted_projections[RUN_A],
    )
    assert all(current is original for current, original in zip(after, before, strict=True))
    assert slot.append_leases == 0


async def test_conditional_batch_restores_exact_state_if_publish_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = create_repository()
    await repository.append_event_if_head(
        event_draft(source_event_id="batch-publish-origin"),
        event_id=CONDITIONAL_EVENT_ID_A,
        expected_head=None,
    )
    head = await repository.ledger_head(RUN_A)
    slot = repository._slots[RUN_A]
    before = (
        slot.ledger,
        slot.ledger_head,
        slot.projection,
        slot.direct_records,
        slot.cycle_records,
        slot.delivery_records,
        slot.collision_receipts,
        repository._trusted_heads[RUN_A],
        repository._trusted_projections[RUN_A],
    )

    def partial_publish(target: object, shadow: object) -> None:
        target_slot = cast(Any, target)
        shadow_slot = cast(Any, shadow)
        target_slot.ledger = shadow_slot.ledger
        raise RuntimeError("injected publish failure")

    monkeypatch.setattr(repository, "_publish_shadow", partial_publish)

    with pytest.raises(RuntimeError, match="injected publish failure"):
        await repository.append_records_if_head(
            (
                ConditionalEventAppend(
                    event=event_draft(source_event_id="batch-publish-staged"),
                    event_id=CONDITIONAL_EVENT_ID_B,
                ),
            ),
            expected_head=head,
        )

    after = (
        slot.ledger,
        slot.ledger_head,
        slot.projection,
        slot.direct_records,
        slot.cycle_records,
        slot.delivery_records,
        slot.collision_receipts,
        repository._trusted_heads[RUN_A],
        repository._trusted_projections[RUN_A],
    )
    assert all(current is original for current, original in zip(after, before, strict=True))
    assert await repository.ledger_head(RUN_A) == head


async def test_conditional_batch_rejects_a_corrupted_staged_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = create_repository()
    await repository.append_event_if_head(
        event_draft(source_event_id="batch-verify-origin"),
        event_id=CONDITIONAL_EVENT_ID_A,
        expected_head=None,
    )
    head = await repository.ledger_head(RUN_A)
    slot = repository._slots[RUN_A]
    before = (slot.ledger, slot.ledger_head, slot.projection, slot.direct_records)
    original_append = repository._append_conditional_to_slot

    def corrupting_append(*args: object, **kwargs: object) -> AppendReceipt:
        receipt = original_append(*args, **kwargs)
        shadow = cast(Any, args[0])
        entry = shadow.ledger[-1]
        shadow.ledger = (
            *shadow.ledger[:-1],
            entry.model_copy(
                update={
                    "chain_tag": PayloadDigest(
                        algorithm=entry.chain_tag.algorithm,
                        value="0" * 64,
                    )
                }
            ),
        )
        return receipt

    monkeypatch.setattr(repository, "_append_conditional_to_slot", corrupting_append)

    with pytest.raises(DigestVerificationError, match="ledger chain"):
        await repository.append_records_if_head(
            (
                ConditionalEventAppend(
                    event=event_draft(source_event_id="batch-verify-staged"),
                    event_id=CONDITIONAL_EVENT_ID_B,
                ),
            ),
            expected_head=head,
        )

    after = (slot.ledger, slot.ledger_head, slot.projection, slot.direct_records)
    assert all(current is original for current, original in zip(after, before, strict=True))
    assert await repository.ledger_head(RUN_A) == head


async def test_conditional_batch_oversize_receipt_is_rejected_before_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = create_repository()
    await repository.append_event_if_head(
        event_draft(source_event_id="batch-receipt-origin"),
        event_id=CONDITIONAL_EVENT_ID_A,
        expected_head=None,
    )
    head = await repository.ledger_head(RUN_A)
    slot = repository._slots[RUN_A]
    trusted_head = repository._trusted_heads[RUN_A]
    before = (slot.ledger, slot.ledger_head, slot.projection)
    monkeypatch.setattr(
        "saliencegate.ports.repository.MAX_CONDITIONAL_BATCH_RECEIPT_BYTES",
        1,
    )

    with pytest.raises(InvalidRecordError, match="conditional_batch"):
        await repository.append_records_if_head(
            (
                ConditionalEventAppend(
                    event=event_draft(source_event_id="batch-oversize-staged"),
                    event_id=CONDITIONAL_EVENT_ID_B,
                ),
            ),
            expected_head=head,
        )

    after = (slot.ledger, slot.ledger_head, slot.projection)
    assert all(current is original for current, original in zip(after, before, strict=True))
    assert repository._trusted_heads[RUN_A] is trusted_head


async def test_conditional_batch_oversize_request_is_rejected_before_run_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = create_repository()
    monkeypatch.setattr(memory_module, "MAX_CONDITIONAL_BATCH_REQUEST_BYTES", 1)

    with pytest.raises(InvalidRecordError, match="conditional_batch"):
        await repository.append_records_if_head(
            (
                ConditionalEventAppend(
                    event=event_draft(source_event_id="batch-request-oversize"),
                    event_id=CONDITIONAL_EVENT_ID_A,
                ),
            ),
            expected_head=None,
        )

    assert RUN_A not in repository._slots
    assert RUN_A not in repository._trusted_heads
    assert RUN_A not in repository._trusted_projections


async def test_conditional_batch_request_bound_is_inclusive_and_counts_utf8_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = ConditionalEventAppend(
        event=event_draft(
            source_event_id="batch-request-boundary",
            payload={"message": "multibyte π🙂"},
        ),
        event_id=CONDITIONAL_EVENT_ID_A,
    )
    encoded_size = len(canonical_json((operation,)))
    assert encoded_size > len(canonical_json((operation,)).decode("utf-8"))

    accepted = create_repository()
    monkeypatch.setattr(
        memory_module,
        "MAX_CONDITIONAL_BATCH_REQUEST_BYTES",
        encoded_size,
    )
    receipt = await accepted.append_records_if_head((operation,), expected_head=None)
    assert receipt.final_head.entry_count == 1

    rejected = create_repository()
    monkeypatch.setattr(
        memory_module,
        "MAX_CONDITIONAL_BATCH_REQUEST_BYTES",
        encoded_size - 1,
    )
    with pytest.raises(InvalidRecordError, match="conditional_batch"):
        await rejected.append_records_if_head((operation,), expected_head=None)
    assert RUN_A not in rejected._slots


async def test_conditional_batch_accepts_the_exact_maximum_operation_mix() -> None:
    repository = create_repository()
    draft = event_draft(source_event_id="batch-maximum-origin")
    origin = await repository.append_event_if_head(
        draft,
        event_id=CONDITIONAL_EVENT_ID_A,
        expected_head=None,
    )
    event_head = await repository.ledger_head(RUN_A)
    signal = conditional_signal(origin.event)
    await repository.record_signal_if_head(signal, expected_head=event_head)
    initial_head = await repository.ledger_head(RUN_A)
    initial_ledger = await repository.ledger(RUN_A)
    event_operation = ConditionalEventAppend(
        event=draft,
        event_id=CONDITIONAL_EVENT_ID_A,
    )
    signal_operation = ConditionalSignalAppend(signal=signal)
    operations = (event_operation,) * 1_000 + (signal_operation,) * 4_000

    receipt = await repository.append_records_if_head(
        operations,
        expected_head=initial_head,
    )

    assert len(receipt.receipts) == 5_000
    assert receipt.initial_head == initial_head
    assert receipt.final_head == initial_head
    assert await repository.ledger(RUN_A) == initial_ledger


async def test_cancelled_conditional_batch_discards_deep_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = create_repository()
    operations = tuple(
        ConditionalEventAppend(
            event=event_draft(
                source_event_id=f"batch-cancel-{index}",
            ),
            event_id=UUID(f"00000000-0000-4000-8002-{index:012x}"),
        )
        for index in range(1, 130)
    )
    original_append = repository._append_conditional_to_slot
    calls = 0

    def cancelling_append(*args: object, **kwargs: object) -> AppendReceipt:
        nonlocal calls
        receipt = original_append(*args, **kwargs)
        calls += 1
        if calls == 2:
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
        return receipt

    monkeypatch.setattr(repository, "_append_conditional_to_slot", cancelling_append)

    task = asyncio.create_task(repository.append_records_if_head(operations, expected_head=None))
    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls == 128
    assert RUN_A not in repository._slots
    assert RUN_A not in repository._trusted_heads
    assert RUN_A not in repository._trusted_projections


async def test_repeated_cancellation_drains_batch_lease_after_full_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = create_repository()
    release_started = asyncio.Event()
    allow_release = asyncio.Event()
    original_release = repository._release_append_slot

    async def delayed_release(slot: Any) -> None:
        release_started.set()
        await allow_release.wait()
        await original_release(slot)

    monkeypatch.setattr(repository, "_release_append_slot", delayed_release)
    operation = ConditionalEventAppend(
        event=event_draft(source_event_id="batch-cancel-after-publish"),
        event_id=CONDITIONAL_EVENT_ID_A,
    )
    task = asyncio.create_task(repository.append_records_if_head((operation,), expected_head=None))
    await release_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    allow_release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(await repository.ledger(RUN_A)) == 1
    assert repository._slots[RUN_A].append_leases == 0


async def test_explicit_synthetic_repository_uses_reproducible_unkeyed_tags() -> None:
    ids = deterministic_ids()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=lambda: next(ids),
    )

    receipt = await repository.append(event_draft())
    ledger = await repository.ledger(RUN_A)

    assert receipt.event.payload_digest.algorithm is PayloadDigestAlgorithm.SYNTHETIC_SHA256
    assert ledger[0].record_tag.algorithm is PayloadDigestAlgorithm.SYNTHETIC_SHA256
    assert ledger[0].chain_tag.algorithm is PayloadDigestAlgorithm.SYNTHETIC_SHA256


async def test_invalid_id_factory_result_leaves_no_observable_run() -> None:
    repository = MemoryRunRepository(
        installation_key=InstallationKey(b"k" * 32),
        id_factory=lambda: UUID("00000000-0000-1000-8000-000000000001"),
    )

    with pytest.raises(ProjectionInvariantError, match="UUID4"):
        await repository.append(event_draft())
    with pytest.raises(RunNotFoundError):
        await repository.ledger(RUN_A)


async def test_rejected_appends_do_not_retain_empty_run_slots() -> None:
    repository = create_repository()

    for value in range(1, 51):
        run_id = UUID(f"00000000-0000-4000-8001-{value:012x}")
        with pytest.raises(CrossRunReferenceError, match="parent"):
            await repository.append(
                event_draft(
                    run_id=run_id,
                    parent_ids=(RUN_A,),
                )
            )

    assert repository._slots == {}


async def test_digest_verification_failure_precedes_run_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "saliencegate.repository.memory.verify_redacted_event",
        lambda *args, **kwargs: False,
    )
    repository = MemoryRunRepository(
        installation_key=InstallationKey(b"k" * 32),
    )

    with pytest.raises(DigestVerificationError):
        await repository.append(event_draft())
    with pytest.raises(RunNotFoundError):
        await repository.ledger(RUN_A)


async def test_redaction_failure_precedes_run_creation() -> None:
    repository = MemoryRunRepository(
        redaction_policy=RedactionPolicy(literal_secrets=("configured-secret",)),
        installation_key=InstallationKey(b"k" * 32),
    )

    with pytest.raises(InvalidDraftError, match="draft"):
        await repository.append(event_draft(payload={"configured-secret": "value"}))
    with pytest.raises(RunNotFoundError):
        await repository.ledger(RUN_A)


async def test_configured_literals_are_rejected_from_typed_records() -> None:
    repository = MemoryRunRepository(
        redaction_policy=RedactionPolicy(literal_secrets=("configured-literal",)),
        installation_key=InstallationKey(b"k" * 32),
    )
    await repository.append(event_draft())
    decision = invocation_decision().model_copy(update={"policy_version": "configured-literal"})

    with pytest.raises(UnsafeRecordContentError):
        await repository.record_invocation_decision(decision)

    assert len(await repository.ledger(RUN_A)) == 1
