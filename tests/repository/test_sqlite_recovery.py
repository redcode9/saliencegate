from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import sys
from pathlib import Path
from uuid import UUID

import pytest
from tests.repository.conformance import (
    CYCLE_INTERVENTION_1_ID,
    CYCLE_MEMORY_ID,
    CYCLE_RUN_ID,
    RUN_A,
    advance_cycle_to_running,
    begin_cycle_context,
    cycle_commit_command,
    event_draft,
    invocation_decision,
)
from tests.repository.sqlite_support import repository

from saliencegate.domain import (
    ConstraintStatus,
    EvidenceReference,
    EvidenceSource,
    InterventionOutcome,
    MemoryCreate,
    MemoryDelta,
    MemoryIdAssignment,
    MemoryKind,
    MemoryUpdate,
    OutcomeEvidenceMode,
    RepeatedErrorStatus,
    TrustLabel,
)
from saliencegate.ports.repository import (
    AppendDisposition,
    BeginCycle,
    DigestVerificationError,
    MemoryQuery,
)
from saliencegate.repository.migrations import LATEST_SCHEMA_VERSION
from saliencegate.repository.sqlite import (
    ClosedSQLiteRepositoryError,
    SQLiteRepositoryError,
)
from saliencegate.security import InstallationKey


async def test_reopen_preserves_ledger_retry_receipts_and_pragmas(tmp_path: Path) -> None:
    path = tmp_path / "reopen.sqlite3"
    first = repository(path)
    original = await first.append(event_draft())
    decision = invocation_decision()
    direct = await first.record_invocation_decision(decision)
    ledger = await first.ledger(RUN_A)
    assert "reopen.sqlite3" not in repr(first)
    assert first._connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert first._connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert first._connection.execute("PRAGMA synchronous").fetchone()[0] == 2
    first.close()

    reopened = repository(path, id_start=0x900)
    retry = await reopened.append(event_draft())
    direct_retry = await reopened.record_invocation_decision(decision)

    assert retry == original.model_copy(update={"disposition": AppendDisposition.DUPLICATE})
    assert direct_retry == direct.model_copy(update={"appended": False})
    assert await reopened.ledger(RUN_A) == ledger
    reopened.close()


async def test_close_is_idempotent_and_operations_fail_cleanly(tmp_path: Path) -> None:
    stored = repository(tmp_path / "closed.sqlite3")
    stored.close()
    stored.close()

    with pytest.raises(ClosedSQLiteRepositoryError, match="closed"):
        await stored.append(event_draft())


async def test_two_instances_publish_distinct_and_duplicate_appends_without_loss(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cas.sqlite3"
    first = repository(path)
    second = repository(path, id_start=0x1800)
    try:
        left, right = await asyncio.gather(
            first.append(event_draft(source_event_id="left")),
            second.append(event_draft(source_event_id="right")),
        )
        assert {left.event.sequence, right.event.sequence} == {1, 2}
        assert {left.disposition, right.disposition} == {AppendDisposition.APPENDED}
        assert tuple(entry.position for entry in await first.ledger(RUN_A)) == (1, 2)
        assert await second.ledger(RUN_A) == await first.ledger(RUN_A)

        duplicate_left, duplicate_right = await asyncio.gather(
            first.append(event_draft(source_event_id="same")),
            second.append(event_draft(source_event_id="same")),
        )
        assert {duplicate_left.disposition, duplicate_right.disposition} == {
            AppendDisposition.APPENDED,
            AppendDisposition.DUPLICATE,
        }
        assert len(await first.ledger(RUN_A)) == 3
    finally:
        first.close()
        second.close()


async def test_wrong_key_and_authoritative_tampering_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "tamper.sqlite3"
    stored = repository(path)
    await stored.append(event_draft())
    stored.close()

    with pytest.raises(DigestVerificationError):
        repository(path, key=InstallationKey(b"w" * 32)).close()

    connection = sqlite3.connect(path)
    try:
        encoded = connection.execute(
            "SELECT entry_json FROM ledger_entries WHERE run_id = ? AND position = 1",
            (str(RUN_A),),
        ).fetchone()[0]
        assert type(encoded) is bytes
        tampered = encoded.replace(b"fixture-adapter", b"tampered-adapter")
        connection.execute(
            "UPDATE ledger_entries SET entry_json = ? WHERE run_id = ? AND position = 1",
            (tampered, str(RUN_A)),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(DigestVerificationError):
        repository(path).close()


async def test_reopen_repairs_derived_tables_without_touching_the_ledger(
    tmp_path: Path,
) -> None:
    path = tmp_path / "repair.sqlite3"
    stored = repository(path)
    receipt = await stored.append(event_draft())
    authoritative = tuple(
        stored._connection.execute(
            "SELECT position, entry_json FROM ledger_entries ORDER BY position"
        ).fetchall()
    )
    head = tuple(stored._connection.execute("SELECT * FROM ledger_heads").fetchone())
    stored.close()

    connection = sqlite3.connect(path)
    try:
        connection.execute("DELETE FROM projection_events")
        connection.execute("UPDATE projection_state SET state_tag = ?", ("0" * 64,))
        connection.execute("DELETE FROM memory_fts")
        connection.commit()
    finally:
        connection.close()

    repaired = repository(path)
    try:
        assert (await repaired.ledger(RUN_A))[0].record == receipt.event
        assert (
            repaired._connection.execute("SELECT count(*) FROM projection_events").fetchone()[0]
            == 1
        )
        assert (
            repaired._connection.execute("SELECT state_tag FROM projection_state").fetchone()[0]
            != "0" * 64
        )
        assert (
            tuple(
                repaired._connection.execute(
                    "SELECT position, entry_json FROM ledger_entries ORDER BY position"
                ).fetchall()
            )
            == authoritative
        )
        assert tuple(repaired._connection.execute("SELECT * FROM ledger_heads").fetchone()) == head
    finally:
        repaired.close()


async def test_v1_upgrade_deduplicates_decision_projection_then_replays_ledger(
    tmp_path: Path,
) -> None:
    path = tmp_path / "decision-upgrade-repair.sqlite3"
    stored = repository(path)
    await stored.append(event_draft())
    await stored.record_invocation_decision(invocation_decision())
    authoritative = await stored.ledger(RUN_A)
    stored.close()

    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP INDEX projection_decisions_authoritative_event_idx")
        connection.execute("DELETE FROM schema_migrations WHERE version = 2")
        connection.execute("PRAGMA user_version = 1")
        connection.execute(
            """
            INSERT INTO projection_decisions(
                run_id,
                decision_id,
                event_sequence,
                ledger_position,
                created_at,
                record_json
            )
            SELECT run_id, ?, event_sequence, ledger_position, created_at, record_json
            FROM projection_decisions
            WHERE run_id = ?
            """,
            ("00000000-0000-4000-8000-000000009903", str(RUN_A)),
        )
        connection.commit()
    finally:
        connection.close()

    repaired = repository(path)
    try:
        assert await repaired.ledger(RUN_A) == authoritative
        assert repaired._connection.execute("PRAGMA user_version").fetchone()[0] == (
            LATEST_SCHEMA_VERSION
        )
        assert (
            repaired._connection.execute("SELECT count(*) FROM projection_decisions").fetchone()[0]
            == 1
        )
        assert (
            repaired._connection.execute(
                """
                SELECT count(*) FROM sqlite_schema
                WHERE type = 'index'
                  AND name = 'projection_decisions_authoritative_event_idx'
                """
            ).fetchone()[0]
            == 1
        )
    finally:
        repaired.close()


async def test_reopen_removes_orphan_projections_and_fts_entries(tmp_path: Path) -> None:
    path = tmp_path / "orphan-repair.sqlite3"
    stored = repository(path)
    await stored.append(event_draft())
    stored.close()
    ghost_run = "00000000-0000-4000-8000-000000009901"
    ghost_memory = "00000000-0000-4000-8000-000000009902"

    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            """
            INSERT INTO projection_memories(
                run_id,
                memory_id,
                revision,
                kind,
                validity,
                trust_label,
                content,
                is_latest,
                source_cycle_id,
                source_cycle_revision,
                record_json
            ) VALUES (?, ?, 1, 'knowledge', 'active', 'synthetic_fixture',
                      'orphan sentinel', 1, ?, 1, ?)
            """,
            (ghost_run, ghost_memory, "b" * 64, b"{}"),
        )
        connection.execute(
            """
            INSERT INTO projection_state(
                run_id,
                ledger_position,
                ingestion_cursor,
                memory_cursor,
                projection_digests_json,
                state_algorithm,
                state_tag
            ) VALUES (?, 1, 1, 0, ?, 'hmac_sha256', ?)
            """,
            (ghost_run, b"{}", "0" * 64),
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM memory_fts WHERE memory_fts MATCH 'orphan'"
            ).fetchone()[0]
            == 1
        )
        connection.commit()
    finally:
        connection.close()

    repaired = repository(path)
    try:
        assert (
            repaired._connection.execute(
                "SELECT count(*) FROM projection_memories WHERE run_id = ?",
                (ghost_run,),
            ).fetchone()[0]
            == 0
        )
        assert (
            repaired._connection.execute(
                "SELECT count(*) FROM projection_state WHERE run_id = ?",
                (ghost_run,),
            ).fetchone()[0]
            == 0
        )
        assert (
            repaired._connection.execute(
                "SELECT count(*) FROM memory_fts WHERE memory_fts MATCH 'orphan'"
            ).fetchone()[0]
            == 0
        )
        assert repaired._connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        repaired.close()


async def test_sql_failure_rolls_back_ledger_head_and_projection_atomically(
    tmp_path: Path,
) -> None:
    path = tmp_path / "atomic.sqlite3"
    stored = repository(path)
    await stored.append(event_draft())
    before = await stored.ledger(RUN_A)
    stored._connection.execute(
        """
        CREATE TRIGGER fail_projection_state_update
        BEFORE UPDATE ON projection_state
        BEGIN
            SELECT RAISE(ABORT, 'fixture-secret-sql-error');
        END
        """
    )

    with pytest.raises(SQLiteRepositoryError) as error:
        await stored.append(event_draft(source_event_id="source-2"))

    assert "fixture-secret" not in str(error.value)
    assert "INSERT" not in str(error.value)
    assert await stored.ledger(RUN_A) == before
    stored._connection.execute("DROP TRIGGER fail_projection_state_update")
    appended = await stored.append(event_draft(source_event_id="source-2"))
    assert appended.event.sequence == 2
    assert tuple(entry.position for entry in await stored.ledger(RUN_A)) == (1, 2)
    stored.close()


async def test_cycle_projection_tracks_memory_origin_and_active_fts(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    stored = repository(path)
    context, _reserved, _running = await advance_cycle_to_running(stored)
    delta = MemoryDelta(
        delta_id=UUID("00000000-0000-4000-8000-000000000881"),
        run_id=CYCLE_RUN_ID,
        creates=(
            MemoryCreate(
                handle="durable-memory",
                kind=MemoryKind.KNOWLEDGE,
                content="Repository resilience marker",
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
    await stored.commit_cycle(
        cycle_commit_command(
            context,
            delta=delta,
            assignments=(MemoryIdAssignment(handle="durable-memory", memory_id=CYCLE_MEMORY_ID),),
        )
    )
    outcome = InterventionOutcome(
        outcome_id=UUID("00000000-0000-4000-8000-000000000882"),
        run_id=CYCLE_RUN_ID,
        intervention_id=CYCLE_INTERVENTION_1_ID,
        repeated_error_status=RepeatedErrorStatus.AVOIDED,
        constraint_status=ConstraintStatus.RESPECTED,
        evidence_mode=OutcomeEvidenceMode.POLICY_REPLAY,
        created_at=context.commit_time,
    )
    await stored.record_outcome(outcome)

    memory_row = stored._connection.execute(
        """
        SELECT source_cycle_id, source_cycle_revision, is_latest
        FROM projection_memories
        WHERE memory_id = ?
        """,
        (str(CYCLE_MEMORY_ID),),
    ).fetchone()
    assert tuple(memory_row) == (context.cycle_id, 4, 1)
    assert (
        stored._connection.execute(
            "SELECT count(*) FROM projection_outcomes WHERE outcome_id = ?",
            (str(outcome.outcome_id),),
        ).fetchone()[0]
        == 1
    )
    assert (
        stored._connection.execute(
            "SELECT count(*) FROM memory_fts WHERE memory_fts MATCH 'resilience'"
        ).fetchone()[0]
        == 1
    )
    hits = await stored.search(MemoryQuery(run_id=CYCLE_RUN_ID, text="repo"))
    assert tuple(hit.memory.memory_id for hit in hits) == (CYCLE_MEMORY_ID,)

    second = await begin_cycle_context(stored, ordinal=2)
    await stored.reserve_cycle(second.reserve)
    await stored.mark_cycle_running(second.start)
    updated_delta = MemoryDelta(
        delta_id=UUID("00000000-0000-4000-8000-000000000883"),
        run_id=CYCLE_RUN_ID,
        updates=(
            MemoryUpdate(
                memory_id=CYCLE_MEMORY_ID,
                expected_revision=1,
                content="Updated durable constraint",
            ),
        ),
        created_at=second.commit_time,
    )
    await stored.commit_cycle(
        cycle_commit_command(
            second,
            delta=updated_delta,
            intervention_id=UUID("00000000-0000-4000-8000-000000000884"),
        )
    )
    memory_rows = stored._connection.execute(
        """
        SELECT revision, source_cycle_id, source_cycle_revision, is_latest
        FROM projection_memories
        WHERE memory_id = ?
        ORDER BY revision
        """,
        (str(CYCLE_MEMORY_ID),),
    ).fetchall()
    assert [tuple(row) for row in memory_rows] == [
        (1, context.cycle_id, 4, 0),
        (2, second.cycle_id, 4, 1),
    ]
    assert (
        stored._connection.execute(
            "SELECT count(*) FROM memory_fts WHERE memory_fts MATCH 'resilience'"
        ).fetchone()[0]
        == 0
    )
    updated_hits = await stored.search(MemoryQuery(run_id=CYCLE_RUN_ID, text="updated"))
    assert tuple(hit.memory.revision for hit in updated_hits) == (2,)
    stored.close()

    reopened = repository(path, id_start=0x900)
    try:
        assert (
            await reopened.search(MemoryQuery(run_id=CYCLE_RUN_ID, text="updated")) == updated_hits
        )
        assert (
            reopened._connection.execute(
                "SELECT count(*) FROM memory_fts WHERE memory_fts MATCH 'constraint'"
            ).fetchone()[0]
            == 1
        )
    finally:
        reopened.close()


async def test_live_anchor_detects_valid_prefix_rollback(tmp_path: Path) -> None:
    path = tmp_path / "rollback.sqlite3"
    stored = repository(path)
    await stored.append(event_draft())
    head_one = tuple(
        stored._connection.execute(
            """
            SELECT entry_count, algorithm, chain_tag, projection_tag, head_tag
            FROM ledger_heads WHERE run_id = ?
            """,
            (str(RUN_A),),
        ).fetchone()
    )
    await stored.append(event_draft(source_event_id="source-2"))

    external = sqlite3.connect(path)
    try:
        external.execute("PRAGMA foreign_keys = OFF")
        external.execute(
            "DELETE FROM ledger_entries WHERE run_id = ? AND position = 2",
            (str(RUN_A),),
        )
        external.execute(
            """
            UPDATE ledger_heads
            SET entry_count = ?, algorithm = ?, chain_tag = ?, projection_tag = ?, head_tag = ?
            WHERE run_id = ?
            """,
            (*head_one, str(RUN_A)),
        )
        external.commit()
    finally:
        external.close()

    with pytest.raises(DigestVerificationError, match="rollback"):
        await stored.ledger(RUN_A)
    stored.close()

    # A fresh process has no monotonic external anchor; the valid prefix is accepted.
    fresh = repository(path, id_start=0x900)
    try:
        assert len(await fresh.ledger(RUN_A)) == 1
    finally:
        fresh.close()


async def test_process_exit_inside_transaction_leaves_no_partial_write(tmp_path: Path) -> None:
    path = tmp_path / "process-crash.sqlite3"
    stored = repository(path)
    await stored.append(event_draft())
    before = await stored.ledger(RUN_A)
    stored.close()
    crash_program = """
import os
import sqlite3
import sys

connection = sqlite3.connect(sys.argv[1], isolation_level=None)
connection.execute("PRAGMA journal_mode = WAL")
connection.execute("BEGIN IMMEDIATE")
connection.execute("UPDATE ledger_entries SET entry_json = X'7b7d' WHERE position = 1")
os._exit(73)
"""

    crashed = subprocess.run(
        [sys.executable, "-c", crash_program, str(path)],
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert crashed.returncode == 73

    recovered = repository(path, id_start=0x900)
    try:
        assert await recovered.ledger(RUN_A) == before
        assert recovered._connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        recovered.close()


async def test_reopen_reconstructs_historical_cycle_receipts(tmp_path: Path) -> None:
    path = tmp_path / "cycle-receipts.sqlite3"
    stored = repository(path)
    context, reserved, running = await advance_cycle_to_running(stored)
    command = cycle_commit_command(context)
    committed = await stored.commit_cycle(command)
    ledger = await stored.ledger(CYCLE_RUN_ID)
    stored.close()

    reopened = repository(path, id_start=0xD00)
    try:
        pending_retry = await reopened.begin_cycle(
            BeginCycle(
                run_id=CYCLE_RUN_ID,
                invocation_decision_id=context.pending.cycle.invocation_decision_id,
                grounding_version=context.pending.cycle.grounding_version,
                grounding_configuration=context.pending.cycle.grounding_configuration,
                grounding_configuration_digest=(
                    context.pending.cycle.grounding_configuration_digest
                ),
                requested_delivery_target=context.pending.cycle.requested_delivery_target,
                created_at=context.pending.cycle.created_at,
            )
        )
        assert pending_retry == context.pending.model_copy(update={"appended": False})
        assert await reopened.reserve_cycle(context.reserve) == reserved.model_copy(
            update={"appended": False}
        )
        assert await reopened.mark_cycle_running(context.start) == running.model_copy(
            update={"appended": False}
        )
        assert await reopened.commit_cycle(command) == committed.model_copy(
            update={"appended": False}
        )
        assert await reopened.ledger(CYCLE_RUN_ID) == ledger
    finally:
        reopened.close()


async def test_running_cycle_recovery_is_exactly_once_across_restarts(tmp_path: Path) -> None:
    path = tmp_path / "running-recovery.sqlite3"
    stored = repository(path)
    context, _reserved, running = await advance_cycle_to_running(stored)
    stored.close()

    recovered = repository(path, id_start=0xD00)
    first = await recovered.recover_cycles(
        CYCLE_RUN_ID,
        recovered_at=context.commit_time,
    )
    assert len(first.failed_unknown_cost) == 1
    assert first.failed_unknown_cost[0].cycle.revision == running.cycle.revision + 1
    expected_budget = first.failed_unknown_cost[0].budget_snapshot
    recovered.close()

    reopened = repository(path, id_start=0xE00)
    try:
        second = await reopened.recover_cycles(
            CYCLE_RUN_ID,
            recovered_at=context.commit_time,
        )
        assert second.failed_unknown_cost == ()
        assert await reopened.budget_snapshot(CYCLE_RUN_ID) == expected_budget
    finally:
        reopened.close()


async def test_separate_processes_share_the_head_cas_without_lost_appends(
    tmp_path: Path,
) -> None:
    path = tmp_path / "multiprocess-cas.sqlite3"
    seeded = repository(path)
    seeded.close()
    worker_program = """
import asyncio
import sys
from datetime import UTC, datetime
from uuid import UUID

from saliencegate.domain import (
    EventPhase,
    EventType,
    NormalizedTraceEventDraft,
    TrustLabel,
)
from saliencegate.repository import SQLiteRunRepository
from saliencegate.security import InstallationKey

async def main():
    values = iter(range(int(sys.argv[3]), int(sys.argv[3]) + 100))
    repository = SQLiteRunRepository(
        sys.argv[1],
        installation_key=InstallationKey(b"q" * 32),
        id_factory=lambda: UUID(f"00000000-0000-4000-8000-{next(values):012x}"),
    )
    await repository.append(
        NormalizedTraceEventDraft(
            run_id=UUID("00000000-0000-4000-8000-000000000101"),
            source_event_id=sys.argv[2],
            timestamp=datetime(2026, 7, 11, 12, 30, tzinfo=UTC),
            event_type=EventType.OBSERVATION,
            phase=EventPhase.POST_ACTION,
            payload={"message": "safe"},
            source_adapter="process-fixture",
            trust_label=TrustLabel.SYNTHETIC_FIXTURE,
        )
    )
    await repository.aclose()

asyncio.run(main())
"""
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                worker_program,
                str(path),
                source,
                str(start),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for source, start in (("process-left", 0xA00), ("process-right", 0xB00))
    ]
    results = [process.communicate(timeout=15) for process in processes]
    assert [
        (process.returncode, stdout, stderr)
        for process, (stdout, stderr) in zip(processes, results, strict=True)
    ] == [(0, b"", b""), (0, b"", b"")]

    recovered = repository(path, id_start=0xC00)
    try:
        ledger = await recovered.ledger(RUN_A)
        assert tuple(entry.position for entry in ledger) == (1, 2)
        assert {entry.record.source_event_id for entry in ledger} == {
            "process-left",
            "process-right",
        }
    finally:
        recovered.close()
