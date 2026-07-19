from __future__ import annotations

import asyncio
import sqlite3
import threading
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from tests.repository.conformance import (
    CONDITIONAL_EVENT_ID_A,
    CONDITIONAL_EVENT_ID_B,
    RUN_A,
    RUN_B,
    conditional_signal,
    event_draft,
)
from tests.repository.sqlite_support import KEY, repository

from saliencegate.domain import PayloadDigestAlgorithm, Signal, TraceEvent
from saliencegate.ports.repository import (
    ConditionalEventAppend,
    ConditionalSignalAppend,
    DigestVerificationError,
    InvalidRecordError,
    LedgerHead,
    LedgerHeadConflictError,
    RunNotFoundError,
)
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.repository.projector import empty_projection
from saliencegate.repository.sqlite import (
    ConcurrentWriteError,
    SQLiteRepositoryError,
    SQLiteRunRepository,
)
from saliencegate.security import RedactionPolicy


def durable_dump(stored: SQLiteRunRepository) -> tuple[str, ...]:
    return tuple(stored._connection.iterdump())


@pytest.mark.parametrize("busy_timeout_ms", (0, 60_001, True))
def test_constructor_rejects_invalid_busy_timeouts(
    tmp_path: Path,
    busy_timeout_ms: int,
) -> None:
    with pytest.raises(ValueError, match="busy_timeout_ms"):
        repository(tmp_path / "invalid-timeout.sqlite3", busy_timeout_ms=busy_timeout_ms)


def test_constructor_rejects_policy_subclasses_and_invalid_paths(tmp_path: Path) -> None:
    class PolicySubclass(RedactionPolicy):
        pass

    with pytest.raises(TypeError, match="exactly RedactionPolicy"):
        SQLiteRunRepository(
            tmp_path / "policy.sqlite3",
            installation_key=KEY,
            redaction_policy=PolicySubclass(),
        )
    constructor = cast(Any, SQLiteRunRepository)
    with pytest.raises(TypeError, match="path"):
        constructor(object(), installation_key=KEY)
    with pytest.raises(ValueError, match="path"):
        constructor(b"bytes-path", installation_key=KEY)


def test_constructor_loads_default_key_and_sanitizes_open_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "saliencegate.repository.sqlite.load_or_create_installation_key",
        lambda: KEY,
    )
    in_memory = SQLiteRunRepository(":memory:")
    assert in_memory._connection.execute("PRAGMA journal_mode").fetchone()[0] == "memory"
    in_memory.close()

    with pytest.raises(SQLiteRepositoryError) as error:
        SQLiteRunRepository(tmp_path, installation_key=KEY)
    assert str(tmp_path) not in str(error.value)


async def test_sync_and_async_context_managers_close_the_connection(tmp_path: Path) -> None:
    with repository(tmp_path / "sync-context.sqlite3") as synchronous:
        await synchronous.append(event_draft())
    assert "closed=True" in repr(synchronous)

    async with repository(tmp_path / "async-context.sqlite3") as asynchronous:
        await asynchronous.append(event_draft())
    assert "closed=True" in repr(asynchronous)


async def test_synthetic_mode_round_trips_without_an_installation_key(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.sqlite3"
    first = SQLiteRunRepository(path, synthetic_benchmark=True)
    receipt = await first.append(event_draft())
    assert receipt.event.payload_digest.algorithm is PayloadDigestAlgorithm.SYNTHETIC_SHA256
    first.close()

    reopened = SQLiteRunRepository(path, synthetic_benchmark=True)
    assert await reopened.ledger(RUN_A)
    reopened.close()


@pytest.mark.parametrize(
    "mutation",
    (
        "entry-as-text",
        "tag-as-blob",
        "noncanonical-entry",
        "record-key-sidecar",
        "invalid-head",
        "missing-head",
    ),
)
async def test_malformed_authoritative_storage_is_rejected(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / f"malformed-{mutation}.sqlite3"
    stored = repository(path)
    await stored.append(event_draft())
    stored.close()

    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("PRAGMA ignore_check_constraints = ON")
        if mutation == "entry-as-text":
            connection.execute("UPDATE ledger_entries SET entry_json = 'not-a-blob'")
        elif mutation == "tag-as-blob":
            connection.execute("UPDATE ledger_entries SET record_tag = X'00'")
        elif mutation == "noncanonical-entry":
            encoded = connection.execute("SELECT entry_json FROM ledger_entries").fetchone()[0]
            connection.execute("UPDATE ledger_entries SET entry_json = ?", (b" " + encoded,))
        elif mutation == "record-key-sidecar":
            connection.execute("UPDATE ledger_entries SET record_key = 'trace_event:sidecar'")
        elif mutation == "invalid-head":
            connection.execute("UPDATE ledger_heads SET head_tag = 'bad'")
        else:
            connection.execute("DELETE FROM ledger_heads")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(DigestVerificationError):
        repository(path).close()


async def test_explicit_stale_candidate_loses_the_head_cas(tmp_path: Path) -> None:
    path = tmp_path / "explicit-cas.sqlite3"
    first = repository(path)
    second = repository(path, id_start=0x1800)
    try:
        first_engine, first_loaded = await first._load_engine()
        second_engine, second_loaded = await second._load_engine()
        first_receipt = await first_engine.append(event_draft(source_event_id="winner"))
        second_receipt = await second_engine.append(event_draft(source_event_id="loser"))
        first_state = await first_engine._verified_state(first_receipt.event.run_id)
        second_state = await second_engine._verified_state(second_receipt.event.run_id)

        assert first._commit_candidate(first_state, first_loaded.get(RUN_A))
        assert not second._commit_candidate(second_state, second_loaded.get(RUN_A))
        assert len(await second.ledger(RUN_A)) == 1
    finally:
        first.close()
        second.close()


async def test_conditional_absent_append_makes_one_commit_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "conditional-absent-race.sqlite3"
    stored = repository(path)
    peer = repository(path, id_start=0x1800)
    real_load = stored._load_engine
    real_run_database = stored._run_database
    loads = 0
    database_calls = 0
    peer_dump: tuple[str, ...] | None = None

    async def observed_run_database(operation: Any) -> Any:
        nonlocal database_calls
        database_calls += 1
        return await real_run_database(operation)

    async def load_then_create_peer() -> Any:
        nonlocal loads, peer_dump
        engine, states = await real_load()
        loads += 1
        if loads == 1:
            await peer.append(event_draft(source_event_id="peer-winner"))
            peer_dump = durable_dump(peer)
        return engine, states

    monkeypatch.setattr(stored, "_load_engine", load_then_create_peer)
    monkeypatch.setattr(stored, "_run_database", observed_run_database)
    monkeypatch.setattr(
        stored,
        "_mutate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy retry used")),
    )
    try:
        with pytest.raises(LedgerHeadConflictError):
            await stored.append_event_if_head(
                event_draft(source_event_id="conditional-loser"),
                event_id=CONDITIONAL_EVENT_ID_A,
                expected_head=None,
            )

        assert loads == 1
        assert database_calls == 2
        ledger = await peer.ledger(RUN_A)
        assert len(ledger) == 1
        assert ledger[0].record.source_event_id == "peer-winner"
        assert peer_dump is not None
        assert durable_dump(peer) == peer_dump
    finally:
        stored.close()
        peer.close()


async def test_conditional_event_peer_advance_is_not_rebased(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "conditional-event-race.sqlite3"
    stored = repository(path)
    peer = repository(path, id_start=0x1800)
    await stored.append(event_draft(source_event_id="origin"))
    expected = await stored.ledger_head(RUN_A)
    real_load = stored._load_engine
    loads = 0
    peer_dump: tuple[str, ...] | None = None

    async def load_then_advance_peer() -> Any:
        nonlocal loads, peer_dump
        engine, states = await real_load()
        loads += 1
        if loads == 1:
            await peer.append(event_draft(source_event_id="peer-event"))
            peer_dump = durable_dump(peer)
        return engine, states

    monkeypatch.setattr(stored, "_load_engine", load_then_advance_peer)
    try:
        with pytest.raises(LedgerHeadConflictError):
            await stored.append_event_if_head(
                event_draft(source_event_id="conditional-event"),
                event_id=CONDITIONAL_EVENT_ID_B,
                expected_head=expected,
            )

        assert loads == 1
        ledger = await peer.ledger(RUN_A)
        assert all(isinstance(entry.record, TraceEvent) for entry in ledger)
        assert tuple(cast(TraceEvent, entry.record).source_event_id for entry in ledger) == (
            "origin",
            "peer-event",
        )
        assert peer_dump is not None
        assert durable_dump(peer) == peer_dump
    finally:
        stored.close()
        peer.close()


async def test_conditional_duplicate_still_checks_the_durable_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "conditional-duplicate-race.sqlite3"
    stored = repository(path)
    peer = repository(path, id_start=0x1800)
    await stored.append_event_if_head(
        event_draft(source_event_id="origin"),
        event_id=CONDITIONAL_EVENT_ID_A,
        expected_head=None,
    )
    expected = await stored.ledger_head(RUN_A)
    real_load = stored._load_engine
    loads = 0
    peer_dump: tuple[str, ...] | None = None

    async def load_then_advance_peer() -> Any:
        nonlocal loads, peer_dump
        engine, states = await real_load()
        loads += 1
        if loads == 1:
            await peer.append(event_draft(source_event_id="peer-event"))
            peer_dump = durable_dump(peer)
        return engine, states

    monkeypatch.setattr(stored, "_load_engine", load_then_advance_peer)
    try:
        with pytest.raises(LedgerHeadConflictError):
            await stored.append_event_if_head(
                event_draft(source_event_id="origin"),
                event_id=CONDITIONAL_EVENT_ID_A,
                expected_head=expected,
            )

        assert loads == 1
        ledger = await peer.ledger(RUN_A)
        assert all(isinstance(entry.record, TraceEvent) for entry in ledger)
        assert tuple(cast(TraceEvent, entry.record).source_event_id for entry in ledger) == (
            "origin",
            "peer-event",
        )
        assert peer_dump is not None
        assert durable_dump(peer) == peer_dump
    finally:
        stored.close()
        peer.close()


async def test_conditional_signal_peer_advance_is_not_rebased(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "conditional-signal-race.sqlite3"
    stored = repository(path)
    peer = repository(path, id_start=0x1800)
    event = (await stored.append(event_draft(source_event_id="origin"))).event
    expected = await stored.ledger_head(RUN_A)
    real_load = stored._load_engine
    loads = 0
    peer_dump: tuple[str, ...] | None = None

    async def load_then_advance_peer() -> Any:
        nonlocal loads, peer_dump
        engine, states = await real_load()
        loads += 1
        if loads == 1:
            await peer.append(event_draft(source_event_id="peer-event"))
            peer_dump = durable_dump(peer)
        return engine, states

    monkeypatch.setattr(stored, "_load_engine", load_then_advance_peer)
    try:
        with pytest.raises(LedgerHeadConflictError):
            await stored.record_signal_if_head(
                conditional_signal(event),
                expected_head=expected,
            )

        assert loads == 1
        ledger = await peer.ledger(RUN_A)
        assert len(ledger) == 2
        assert all(entry.record.record_type == "trace_event" for entry in ledger)
        assert peer_dump is not None
        assert durable_dump(peer) == peer_dump
    finally:
        stored.close()
        peer.close()


async def test_conditional_signal_owns_its_commit_target_before_await(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = repository(tmp_path / "conditional-signal-target.sqlite3")
    event = (await stored.append(event_draft(run_id=RUN_A, source_event_id="run-a-origin"))).event
    await stored.append(event_draft(run_id=RUN_B, source_event_id="run-b-origin"))
    expected = await stored.ledger_head(RUN_A)
    signal = conditional_signal(event)
    original_record = MemoryRunRepository.record_signal_if_head

    async def record_then_mutate_caller_signal(
        engine: MemoryRunRepository,
        candidate: Signal,
        *,
        expected_head: LedgerHead,
    ) -> Any:
        receipt = await original_record(
            engine,
            candidate,
            expected_head=expected_head,
        )
        object.__setattr__(signal, "run_id", RUN_B)
        return receipt

    monkeypatch.setattr(
        MemoryRunRepository,
        "record_signal_if_head",
        record_then_mutate_caller_signal,
    )
    try:
        receipt = await stored.record_signal_if_head(signal, expected_head=expected)

        run_a_ledger = await stored.ledger(RUN_A)
        run_b_ledger = await stored.ledger(RUN_B)
        assert receipt.appended
        assert len(run_a_ledger) == 2
        assert isinstance(run_a_ledger[-1].record, Signal)
        assert run_a_ledger[-1].record.run_id == RUN_A
        assert len(run_b_ledger) == 1
        assert isinstance(run_b_ledger[0].record, TraceEvent)
    finally:
        stored.close()


async def test_batch_uses_one_target_replay_and_one_projection_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "batch-single-replay.sqlite3"
    stored = repository(path)
    origin = await stored.append(event_draft(source_event_id="origin"))
    expected = await stored.ledger_head(RUN_A)
    operations = (
        ConditionalSignalAppend(signal=conditional_signal(origin.event)),
        *tuple(
            ConditionalEventAppend(
                event=event_draft(source_event_id=f"batch-{index}"),
                event_id=UUID(f"00000000-0000-4000-8000-{0x1900 + index:012x}"),
            )
            for index in range(1, 3)
        ),
    )
    original_replay = MemoryRunRepository._replay_run
    original_write_projection = stored._write_projection
    target_replays = 0
    projection_writes = 0
    statements: list[str] = []

    def counted_replay(
        engine: MemoryRunRepository,
        entries: Any,
        head: LedgerHead,
    ) -> Any:
        nonlocal target_replays
        target_replays += int(head.run_id == RUN_A)
        return original_replay(engine, entries, head)

    def counted_projection_write(state: Any) -> None:
        nonlocal projection_writes
        projection_writes += 1
        original_write_projection(state)

    async def reject_legacy_load() -> Any:
        raise AssertionError("batch used the legacy replaying loader")

    monkeypatch.setattr(MemoryRunRepository, "_replay_run", counted_replay)
    monkeypatch.setattr(stored, "_write_projection", counted_projection_write)
    monkeypatch.setattr(stored, "_load_engine", reject_legacy_load)
    stored._connection.set_trace_callback(statements.append)
    try:
        receipt = await stored.append_records_if_head(operations, expected_head=expected)

        assert receipt.initial_head == expected
        assert receipt.final_head.entry_count == 4
        assert target_replays == 1
        assert projection_writes == 1
        assert sum(statement == "BEGIN IMMEDIATE" for statement in statements) == 1
        write_begin = statements.index("BEGIN IMMEDIATE")
        assert sum(statement == "COMMIT" for statement in statements[write_begin:]) == 1
        verifier = sqlite3.connect(path)
        try:
            assert verifier.execute("SELECT count(*) FROM ledger_entries").fetchone()[0] == 4
        finally:
            verifier.close()
    finally:
        stored._connection.set_trace_callback(None)
        stored.close()


async def test_sqlite_batch_accepts_the_exact_maximum_operation_mix(
    tmp_path: Path,
) -> None:
    stored = repository(tmp_path / "batch-maximum.sqlite3")
    draft = event_draft(source_event_id="sqlite-batch-maximum-origin")
    origin = await stored.append_event_if_head(
        draft,
        event_id=CONDITIONAL_EVENT_ID_A,
        expected_head=None,
    )
    event_head = await stored.ledger_head(RUN_A)
    signal = conditional_signal(origin.event)
    await stored.record_signal_if_head(signal, expected_head=event_head)
    initial_head = await stored.ledger_head(RUN_A)
    operations = (ConditionalEventAppend(event=draft, event_id=CONDITIONAL_EVENT_ID_A),) * 1_000 + (
        ConditionalSignalAppend(signal=signal),
    ) * 4_000

    receipt = await stored.append_records_if_head(
        operations,
        expected_head=initial_head,
    )

    assert len(receipt.receipts) == 5_000
    assert receipt.initial_head == initial_head
    assert receipt.final_head == initial_head
    assert len(await stored.ledger(RUN_A)) == 2
    stored.close()


async def test_batch_target_race_is_not_rebased(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "batch-target-race.sqlite3"
    stored = repository(path)
    peer = repository(path, id_start=0x1800)
    await stored.append(event_draft(source_event_id="origin"))
    expected = await stored.ledger_head(RUN_A)
    real_load = stored._load_batch_engine
    peer_dump: tuple[str, ...] | None = None

    async def load_then_advance_peer() -> Any:
        nonlocal peer_dump
        engine, states = await real_load()
        await peer.append(event_draft(source_event_id="peer-event"))
        peer_dump = durable_dump(peer)
        return engine, states

    monkeypatch.setattr(stored, "_load_batch_engine", load_then_advance_peer)
    operation = ConditionalEventAppend(
        event=event_draft(source_event_id="batch-loser"),
        event_id=CONDITIONAL_EVENT_ID_B,
    )
    try:
        with pytest.raises(LedgerHeadConflictError):
            await stored.append_records_if_head((operation,), expected_head=expected)

        assert peer_dump is not None
        assert durable_dump(peer) == peer_dump
        ledger = await peer.ledger(RUN_A)
        assert tuple(cast(TraceEvent, entry.record).source_event_id for entry in ledger) == (
            "origin",
            "peer-event",
        )
    finally:
        stored.close()
        peer.close()


async def test_invalid_batch_is_rejected_before_catalog_load_or_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = repository(tmp_path / "batch-preflight.sqlite3")
    operation = ConditionalEventAppend(
        event=event_draft(),
        event_id=CONDITIONAL_EVENT_ID_A,
    )
    loads = 0
    commits = 0

    async def forbidden_load() -> Any:
        nonlocal loads
        loads += 1
        raise AssertionError("invalid batch reached the durable catalog")

    def forbidden_commit(*_args: Any, **_kwargs: Any) -> bool:
        nonlocal commits
        commits += 1
        raise AssertionError("invalid batch reached commit")

    monkeypatch.setattr(
        "saliencegate.repository.memory.MAX_CONDITIONAL_BATCH_REQUEST_BYTES",
        1,
    )
    monkeypatch.setattr(stored, "_load_batch_engine", forbidden_load)
    monkeypatch.setattr(stored, "_commit_batch_candidate", forbidden_commit)

    with pytest.raises(InvalidRecordError, match="conditional_batch"):
        await stored.append_records_if_head((operation,), expected_head=None)

    assert loads == 0
    assert commits == 0
    assert not stored._anchors
    assert stored._connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
    stored.close()


async def test_batch_oversize_receipt_is_rejected_before_sql_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = repository(tmp_path / "batch-receipt-precommit.sqlite3")
    commits = 0

    def forbidden_commit(*_args: Any, **_kwargs: Any) -> bool:
        nonlocal commits
        commits += 1
        raise AssertionError("oversize receipt reached SQL commit")

    monkeypatch.setattr(
        "saliencegate.ports.repository.MAX_CONDITIONAL_BATCH_RECEIPT_BYTES",
        1,
    )
    monkeypatch.setattr(stored, "_commit_batch_candidate", forbidden_commit)
    operation = ConditionalEventAppend(
        event=event_draft(source_event_id="batch-receipt-oversize"),
        event_id=CONDITIONAL_EVENT_ID_A,
    )

    with pytest.raises(InvalidRecordError, match="conditional_batch"):
        await stored.append_records_if_head((operation,), expected_head=None)

    assert commits == 0
    assert not stored._connection.in_transaction
    assert stored._connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
    assert stored._connection.execute("SELECT count(*) FROM ledger_entries").fetchone()[0] == 0
    stored.close()


async def test_batch_mid_suffix_failure_rolls_back_every_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = repository(tmp_path / "batch-mid-suffix-failure.sqlite3")
    operations = (
        ConditionalEventAppend(
            event=event_draft(source_event_id="batch-first"),
            event_id=CONDITIONAL_EVENT_ID_A,
        ),
        ConditionalEventAppend(
            event=event_draft(source_event_id="batch-second"),
            event_id=CONDITIONAL_EVENT_ID_B,
        ),
    )
    original_insert = stored._insert_ledger_entry
    inserts = 0

    def fail_second_insert(entry: Any) -> None:
        nonlocal inserts
        inserts += 1
        if inserts == 2:
            raise DigestVerificationError("fixture batch insert")
        original_insert(entry)

    monkeypatch.setattr(stored, "_insert_ledger_entry", fail_second_insert)

    with pytest.raises(DigestVerificationError, match="fixture batch insert"):
        await stored.append_records_if_head(operations, expected_head=None)

    assert not stored._connection.in_transaction
    assert stored._connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
    assert stored._connection.execute("SELECT count(*) FROM ledger_entries").fetchone()[0] == 0
    stored.close()


async def test_batch_reauthenticates_the_exported_candidate_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = repository(tmp_path / "batch-corrupt-candidate.sqlite3")
    original_candidate = stored._batch_candidate_state

    async def corrupt_candidate(engine: MemoryRunRepository, run_id: UUID) -> Any:
        candidate = await original_candidate(engine, run_id)
        entry = candidate.ledger[-1]
        corrupted_entry = entry.model_copy(
            update={
                "chain_tag": entry.chain_tag.model_copy(update={"value": "0" * 64}),
            }
        )
        return replace(candidate, ledger=(*candidate.ledger[:-1], corrupted_entry))

    monkeypatch.setattr(stored, "_batch_candidate_state", corrupt_candidate)
    operation = ConditionalEventAppend(
        event=event_draft(source_event_id="batch-corrupt-candidate"),
        event_id=CONDITIONAL_EVENT_ID_A,
    )

    with pytest.raises(DigestVerificationError, match="ledger chain"):
        await stored.append_records_if_head((operation,), expected_head=None)

    assert not stored._connection.in_transaction
    assert stored._connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
    assert stored._connection.execute("SELECT count(*) FROM ledger_entries").fetchone()[0] == 0
    stored.close()


async def test_batch_post_write_attestation_rolls_back_tampered_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = repository(tmp_path / "batch-post-write-tamper.sqlite3")
    original_write = stored._write_projection

    def write_then_tamper(state: Any) -> None:
        original_write(state)
        stored._connection.execute(
            "UPDATE ledger_entries SET chain_tag = ? WHERE run_id = ? AND position = 1",
            ("0" * 64, str(RUN_A)),
        )

    monkeypatch.setattr(stored, "_write_projection", write_then_tamper)
    operation = ConditionalEventAppend(
        event=event_draft(source_event_id="batch-post-write-tamper"),
        event_id=CONDITIONAL_EVENT_ID_A,
    )

    with pytest.raises(DigestVerificationError, match="durable ledger entry"):
        await stored.append_records_if_head((operation,), expected_head=None)

    assert not stored._connection.in_transaction
    assert stored._connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
    assert stored._connection.execute("SELECT count(*) FROM ledger_entries").fetchone()[0] == 0
    stored.close()


async def test_semantic_failure_retries_when_a_peer_advanced_the_head(tmp_path: Path) -> None:
    path = tmp_path / "semantic-retry.sqlite3"
    first = repository(path)
    peer = repository(path, id_start=0x1800)
    calls = 0

    async def operation(engine: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            await peer.append(event_draft(source_event_id="peer"))
            raise RunNotFoundError(RUN_A)
        return await engine.append(event_draft(source_event_id="after-peer"))

    try:
        receipt = await first._mutate(
            operation,
            lambda result: result.event.run_id,
            retry_run_id=RUN_A,
        )
        assert calls == 2
        assert receipt.event.sequence == 2
    finally:
        first.close()
        peer.close()


async def test_bounded_cas_retries_do_not_publish_a_phantom_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = repository(tmp_path / "retry-limit.sqlite3")
    monkeypatch.setattr(
        stored,
        "_commit_candidate",
        lambda _candidate, _base, **_kwargs: False,
    )

    with pytest.raises(ConcurrentWriteError, match="concurrent"):
        await stored.append(event_draft())

    assert stored._connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
    stored.close()


async def test_live_anchor_rejects_deleted_runs_and_valid_forks(tmp_path: Path) -> None:
    deleted_path = tmp_path / "deleted-run.sqlite3"
    deleted = repository(deleted_path)
    await deleted.append(event_draft())
    external = sqlite3.connect(deleted_path)
    try:
        external.execute("PRAGMA foreign_keys = OFF")
        external.execute("DELETE FROM ledger_heads")
        external.execute("DELETE FROM ledger_entries")
        external.execute("DELETE FROM runs")
        external.commit()
    finally:
        external.close()
    with pytest.raises(DigestVerificationError, match="rollback"):
        await deleted.ledger(RUN_A)
    deleted.close()

    live_path = tmp_path / "live-fork.sqlite3"
    alternate_path = tmp_path / "alternate-fork.sqlite3"
    live = repository(live_path)
    alternate = repository(alternate_path)
    await live.append(event_draft(source_event_id="live"))
    await alternate.append(event_draft(source_event_id="alternate"))
    alternate.close()

    connection = sqlite3.connect(live_path)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("ATTACH DATABASE ? AS alternate", (str(alternate_path),))
        connection.execute("DELETE FROM ledger_heads")
        connection.execute("DELETE FROM ledger_entries")
        connection.execute("DELETE FROM runs")
        connection.execute("INSERT INTO runs SELECT * FROM alternate.runs")
        connection.execute("INSERT INTO ledger_entries SELECT * FROM alternate.ledger_entries")
        connection.execute("INSERT INTO ledger_heads SELECT * FROM alternate.ledger_heads")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(DigestVerificationError, match="fork"):
        await live.ledger(RUN_A)
    live.close()


async def test_in_process_tamper_rolls_back_read_transaction(tmp_path: Path) -> None:
    stored = repository(tmp_path / "read-rollback.sqlite3")
    await stored.append(event_draft())
    stored._connection.execute("PRAGMA ignore_check_constraints = ON")
    stored._connection.execute("UPDATE ledger_heads SET head_tag = 'bad'")

    with pytest.raises(DigestVerificationError):
        await stored.ledger(RUN_A)
    assert not stored._connection.in_transaction
    stored.close()


async def test_unexpected_closed_connection_is_sanitized(tmp_path: Path) -> None:
    stored = repository(tmp_path / "connection-error.sqlite3")
    stored._connection.close()

    with pytest.raises(SQLiteRepositoryError):
        await stored.ledger(RUN_A)


async def test_projection_source_attestation_rejects_an_internal_mismatch(
    tmp_path: Path,
) -> None:
    stored = repository(tmp_path / "source-attestation.sqlite3")
    await stored.append(event_draft())
    engine, states = await stored._load_engine()
    del engine
    state = states[RUN_A]
    mismatched = replace(state, projection=empty_projection(RUN_A))

    with pytest.raises(DigestVerificationError, match="materialization"):
        stored._projection_sources(mismatched)
    stored.close()


async def test_candidate_prefix_guards_reject_truncation(tmp_path: Path) -> None:
    stored = repository(tmp_path / "candidate-prefix.sqlite3")
    await stored.append(event_draft())
    _, states_one = await stored._load_engine()
    await stored.append(event_draft(source_event_id="source-2"))
    _, states_two = await stored._load_engine()
    shorter = states_one[RUN_A]
    longer = states_two[RUN_A]

    assert not stored._prefix_matches(longer, shorter)
    with pytest.raises(DigestVerificationError, match="candidate ledger fork"):
        stored._commit_candidate(shorter, longer)
    stored.close()


async def test_repository_integrity_failure_during_projection_write_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = repository(tmp_path / "projection-integrity.sqlite3")

    def fail_projection(_state: object) -> None:
        raise DigestVerificationError("fixture projection")

    monkeypatch.setattr(stored, "_write_projection", fail_projection)
    with pytest.raises(DigestVerificationError, match="fixture projection"):
        await stored.append(event_draft())
    assert not stored._connection.in_transaction
    assert stored._connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
    stored.close()


async def test_recovery_key_error_is_sanitized_and_closes_constructor_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "recovery-key-error.sqlite3"
    seeded = repository(path)
    await seeded.append(event_draft())
    seeded.close()

    def fail_projection(_self: object, _state: object) -> None:
        raise KeyError("fixture-secret")

    monkeypatch.setattr(SQLiteRunRepository, "_write_projection", fail_projection)
    with pytest.raises(SQLiteRepositoryError) as error:
        repository(path)
    assert "fixture-secret" not in str(error.value)


async def test_catalog_rejects_noncanonical_uuid_aliases(tmp_path: Path) -> None:
    path = tmp_path / "uuid-alias.sqlite3"
    run_id = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    stored = repository(path)
    await stored.append(event_draft(run_id=run_id))
    stored.close()

    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        canonical = str(run_id)
        alias = canonical.upper()
        connection.execute(
            "INSERT INTO runs SELECT ?, created_at FROM runs WHERE run_id = ?",
            (alias, canonical),
        )
        connection.execute(
            """
            INSERT INTO ledger_entries
            SELECT ?, position, record_key, record_type, entry_json, record_algorithm,
                   record_tag, previous_chain_algorithm, previous_chain_tag,
                   chain_algorithm, chain_tag
            FROM ledger_entries WHERE run_id = ?
            """,
            (alias, canonical),
        )
        connection.execute(
            """
            INSERT INTO ledger_heads
            SELECT ?, entry_count, algorithm, chain_tag, projection_tag, head_tag
            FROM ledger_heads WHERE run_id = ?
            """,
            (alias, canonical),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(DigestVerificationError, match="catalog"):
        repository(path).close()


async def test_post_write_attestation_rolls_back_authoritative_trigger_mutation(
    tmp_path: Path,
) -> None:
    stored = repository(tmp_path / "trigger-tamper.sqlite3")
    await stored.append(event_draft())
    before = await stored.ledger(RUN_A)
    stored._connection.execute(
        """
        CREATE TRIGGER mutate_ledger_after_projection
        AFTER UPDATE ON projection_state
        BEGIN
            UPDATE ledger_entries SET entry_json = X'7b7d' WHERE position = 1;
        END
        """
    )

    with pytest.raises(DigestVerificationError):
        await stored.append(event_draft(source_event_id="source-2"))

    assert await stored.ledger(RUN_A) == before
    stored._connection.execute("DROP TRIGGER mutate_ledger_after_projection")
    stored.close()


async def test_busy_writer_does_not_block_the_event_loop(tmp_path: Path) -> None:
    path = tmp_path / "nonblocking-busy.sqlite3"
    stored = repository(path, busy_timeout_ms=1_000)
    blocker = sqlite3.connect(path, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    loop = asyncio.get_running_loop()
    ticks: list[float] = []

    async def heartbeat() -> None:
        for _ in range(8):
            ticks.append(loop.time())
            await asyncio.sleep(0.025)

    append_task = asyncio.create_task(stored.append(event_draft()))
    heartbeat_task = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.12)
    blocker.rollback()
    await append_task
    await heartbeat_task

    assert len(ticks) == 8
    assert max(right - left for left, right in pairwise(ticks)) < 0.1
    blocker.close()
    stored.close()


@pytest.mark.parametrize("operation_kind", ("legacy", "conditional", "batch"))
async def test_double_cancellation_drains_worker_before_async_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation_kind: str,
) -> None:
    path = tmp_path / f"double-cancel-{operation_kind}.sqlite3"
    stored = repository(path, busy_timeout_ms=1_000)
    blocker = sqlite3.connect(path, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    worker_started = threading.Event()
    original_commit = (
        stored._commit_batch_candidate if operation_kind == "batch" else stored._commit_candidate
    )

    def observed_commit(*args: Any, **kwargs: Any) -> bool:
        worker_started.set()
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(
        stored,
        "_commit_batch_candidate" if operation_kind == "batch" else "_commit_candidate",
        observed_commit,
    )
    if operation_kind == "conditional":
        append_operation = stored.append_event_if_head(
            event_draft(),
            event_id=CONDITIONAL_EVENT_ID_A,
            expected_head=None,
        )
    elif operation_kind == "batch":
        append_operation = stored.append_records_if_head(
            (
                ConditionalEventAppend(
                    event=event_draft(source_event_id="double-cancel-batch-1"),
                    event_id=CONDITIONAL_EVENT_ID_A,
                ),
                ConditionalEventAppend(
                    event=event_draft(source_event_id="double-cancel-batch-2"),
                    event_id=CONDITIONAL_EVENT_ID_B,
                ),
            ),
            expected_head=None,
        )
    else:
        append_operation = stored.append(event_draft())
    append_task = asyncio.create_task(append_operation)
    assert await asyncio.to_thread(worker_started.wait, 1)
    append_task.cancel()
    await asyncio.sleep(0)
    append_task.cancel()
    close_task = asyncio.create_task(stored.aclose())
    await asyncio.sleep(0.05)
    assert not close_task.done()

    blocker.rollback()
    with pytest.raises(asyncio.CancelledError):
        await append_task
    await close_task
    assert "closed=True" in repr(stored)
    verifier = sqlite3.connect(path)
    try:
        expected_entries = 2 if operation_kind == "batch" else 1
        assert (
            verifier.execute("SELECT count(*) FROM ledger_entries").fetchone()[0]
            == expected_entries
        )
    finally:
        verifier.close()
    blocker.close()


async def test_commit_rejects_rollback_of_a_different_anchored_run(tmp_path: Path) -> None:
    path = tmp_path / "cross-run-rollback.sqlite3"
    stored = repository(path)
    await stored.append(event_draft(run_id=RUN_A, source_event_id="run-a-1"))
    await stored.append(event_draft(run_id=RUN_B, source_event_id="run-b-1"))
    engine, loaded = await stored._load_engine()
    receipt = await engine.append(event_draft(run_id=RUN_B, source_event_id="run-b-2"))
    candidate = await engine._verified_state(RUN_B)

    external = sqlite3.connect(path)
    try:
        external.execute("PRAGMA foreign_keys = ON")
        external.execute("DELETE FROM runs WHERE run_id = ?", (str(RUN_A),))
        external.commit()
    finally:
        external.close()

    with pytest.raises(DigestVerificationError, match="rollback"):
        await stored._run_database(
            lambda: stored._commit_candidate(candidate, loaded[receipt.event.run_id])
        )
    verifier = sqlite3.connect(path)
    try:
        assert (
            verifier.execute(
                "SELECT count(*) FROM ledger_entries WHERE run_id = ?",
                (str(RUN_B),),
            ).fetchone()[0]
            == 1
        )
    finally:
        verifier.close()
    stored.close()


async def test_batch_commit_rejects_rollback_of_a_different_anchored_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "batch-cross-run-rollback.sqlite3"
    stored = repository(path)
    await stored.append(event_draft(run_id=RUN_A, source_event_id="run-a-origin"))
    await stored.append(event_draft(run_id=RUN_B, source_event_id="run-b-origin"))
    expected = await stored.ledger_head(RUN_A)
    real_load = stored._load_batch_engine

    async def load_then_delete_other_run() -> Any:
        engine, states = await real_load()
        external = sqlite3.connect(path)
        try:
            external.execute("PRAGMA foreign_keys = ON")
            external.execute("DELETE FROM runs WHERE run_id = ?", (str(RUN_B),))
            external.commit()
        finally:
            external.close()
        return engine, states

    monkeypatch.setattr(stored, "_load_batch_engine", load_then_delete_other_run)
    operation = ConditionalEventAppend(
        event=event_draft(run_id=RUN_A, source_event_id="run-a-batch"),
        event_id=CONDITIONAL_EVENT_ID_B,
    )

    with pytest.raises(DigestVerificationError, match="rollback"):
        await stored.append_records_if_head((operation,), expected_head=expected)

    verifier = sqlite3.connect(path)
    try:
        assert (
            verifier.execute(
                "SELECT count(*) FROM ledger_entries WHERE run_id = ?",
                (str(RUN_A),),
            ).fetchone()[0]
            == 1
        )
    finally:
        verifier.close()
    stored.close()
