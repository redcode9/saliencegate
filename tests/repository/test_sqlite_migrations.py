from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

import saliencegate.repository.migrations as migration_module
from saliencegate.repository.migrations import (
    APPLICATION_ID,
    LATEST_SCHEMA_VERSION,
    Migration,
    MigrationError,
    MigrationIntegrityError,
    SchemaTooNewError,
    apply_migrations,
    discover_migrations,
    migration_checksum,
    parse_statements,
)

RUN_ID = "00000000-0000-4000-8000-000000000701"
EVENT_ID = "00000000-0000-4000-8000-000000000702"
DECISION_ID = "00000000-0000-4000-8000-000000000703"
CYCLE_ID = "a" * 64
NOW = "2026-07-11T15:30:00+00:00"
INITIAL_MIGRATION_CHECKSUM = "3c2e3c502f81ea32b263c1d8896c8ed91fd16f28526b81d6bbf220add9b574f3"
UNIQUE_INVOCATION_EVENT_MIGRATION_CHECKSUM = (
    "249594513388ca4ecef2dc4df3b8923769967af9c45152d705a7819288571ee6"
)


@contextmanager
def migrated_connection() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(":memory:")
    try:
        apply_migrations(connection)
        yield connection
    finally:
        connection.close()


def insert_run(connection: sqlite3.Connection) -> None:
    connection.execute(
        "INSERT INTO runs(run_id, created_at) VALUES (?, ?)",
        (RUN_ID, NOW),
    )


def insert_ledger_entry(
    connection: sqlite3.Connection,
    position: int,
    record_type: str,
) -> None:
    previous_algorithm = None if position == 1 else "hmac_sha256"
    previous_tag = None if position == 1 else f"{position - 1:x}" * 64
    connection.execute(
        """
        INSERT INTO ledger_entries(
            run_id,
            position,
            record_key,
            record_type,
            entry_json,
            record_algorithm,
            record_tag,
            previous_chain_algorithm,
            previous_chain_tag,
            chain_algorithm,
            chain_tag
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            RUN_ID,
            position,
            f"{record_type}:{position}",
            record_type,
            b"{}",
            "hmac_sha256",
            f"{position + 3:x}" * 64,
            previous_algorithm,
            previous_tag,
            "hmac_sha256",
            f"{position:x}" * 64,
        ),
    )


def insert_projection_parents(connection: sqlite3.Connection) -> None:
    insert_run(connection)
    insert_ledger_entry(connection, 1, "trace_event")
    insert_ledger_entry(connection, 2, "invocation_decision")
    insert_ledger_entry(connection, 3, "cycle_record")
    connection.execute(
        """
        INSERT INTO projection_events(
            run_id, event_id, sequence, source_event_id, ledger_position, record_json
        )
        VALUES (?, ?, 1, 'source-1', 1, ?)
        """,
        (RUN_ID, EVENT_ID, b"{}"),
    )
    connection.execute(
        """
        INSERT INTO projection_decisions(
            run_id, decision_id, event_sequence, ledger_position, created_at, record_json
        )
        VALUES (?, ?, 1, 2, ?, ?)
        """,
        (RUN_ID, DECISION_ID, NOW, b"{}"),
    )
    connection.execute(
        """
        INSERT INTO projection_cycle_revisions(
            run_id,
            cycle_id,
            revision,
            state,
            is_latest,
            ledger_position,
            invocation_decision_id,
            created_at,
            updated_at,
            record_json
        )
        VALUES (?, ?, 1, 'committed', 1, 3, ?, ?, ?, ?)
        """,
        (RUN_ID, CYCLE_ID, DECISION_ID, NOW, NOW, b"{}"),
    )


def apply_initial_schema(connection: sqlite3.Connection) -> Migration:
    initial = discover_migrations()[0]
    connection.execute("BEGIN EXCLUSIVE")
    for statement in parse_statements(initial.sql):
        connection.execute(statement)
    connection.execute(
        """
        INSERT INTO schema_migrations(version, name, checksum, applied_at)
        VALUES (?, ?, ?, ?)
        """,
        (initial.version, initial.name, initial.checksum, NOW),
    )
    connection.execute(f"PRAGMA user_version = {initial.version}")
    connection.commit()
    return initial


def fts_hits(connection: sqlite3.Connection, term: str) -> list[tuple[str, str, int]]:
    rows = connection.execute(
        """
        SELECT run_id, memory_id, revision
        FROM memory_fts
        WHERE memory_fts MATCH ?
        ORDER BY run_id, memory_id, revision
        """,
        (term,),
    ).fetchall()
    return [(str(row[0]), str(row[1]), int(row[2])) for row in rows]


def test_packaged_migrations_are_contiguous_parseable_and_checksum_stable() -> None:
    migrations = discover_migrations()

    assert tuple(migration.version for migration in migrations) == (1, 2)
    assert migrations[0].name == "initial"
    assert migrations[0].checksum == migration_checksum(migrations[0].sql)
    assert migrations[0].checksum == INITIAL_MIGRATION_CHECKSUM
    assert len(migrations[0].checksum) == 64
    assert migrations[1].name == "unique_invocation_event"
    assert migrations[1].checksum == migration_checksum(migrations[1].sql)
    assert migrations[1].checksum == UNIQUE_INVOCATION_EVENT_MIGRATION_CHECKSUM
    assert len(migrations[1].checksum) == 64
    initial_statements = parse_statements(migrations[0].sql)
    decision_statements = parse_statements(migrations[1].sql)
    assert any("CREATE TABLE ledger_entries" in statement for statement in initial_statements)
    assert any("CREATE VIRTUAL TABLE memory_fts" in statement for statement in initial_statements)
    assert any(
        "CREATE TRIGGER projection_memories_fts_update" in statement
        for statement in initial_statements
    )
    assert any(
        "CREATE UNIQUE INDEX projection_decisions_authoritative_event_idx" in statement
        for statement in decision_statements
    )


def test_parser_rejects_incomplete_or_empty_migration_sql() -> None:
    with pytest.raises(MigrationError, match="incomplete"):
        parse_statements("CREATE TABLE unfinished (value INTEGER)")
    with pytest.raises(MigrationError, match="no SQL"):
        parse_statements("")


def test_migration_text_helpers_require_exact_strings() -> None:
    with pytest.raises(TypeError, match="exact string"):
        migration_checksum(b"SELECT 1;")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact string"):
        parse_statements(b"SELECT 1;")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("filename", "contents", "message"),
    [
        ("0001_initial.sql", b"\xef\xbb\xbfSELECT 1;", "BOM"),
        ("0001_initial.sql", b"\xffSELECT 1;", "valid UTF-8"),
        ("0002_second.sql", b"SELECT 1;", "missing or non-contiguous"),
    ],
)
def test_malformed_packaged_migrations_are_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    contents: bytes,
    message: str,
) -> None:
    tmp_path.joinpath(filename).write_bytes(contents)
    monkeypatch.setattr(migration_module.resources, "files", lambda _: tmp_path)

    with pytest.raises(MigrationError, match=message):
        discover_migrations()


def test_apply_migrations_builds_the_complete_schema_once() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        applied = apply_migrations(connection)
        reapplied = apply_migrations(connection)

        assert tuple(migration.version for migration in applied) == (1, 2)
        assert reapplied == ()
        assert connection.execute("PRAGMA user_version").fetchone()[0] == LATEST_SCHEMA_VERSION
        assert connection.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type IN ('table', 'view')"
            )
        }
        assert {
            "schema_migrations",
            "runs",
            "ledger_entries",
            "ledger_heads",
            "projection_state",
            "projection_events",
            "projection_signals",
            "projection_decisions",
            "projection_cycle_revisions",
            "projection_memories",
            "projection_active_memories",
            "projection_interventions",
            "projection_outcomes",
            "projection_delivery_revisions",
            "projection_budgets",
            "memory_fts",
        } <= names
        migration_rows = connection.execute(
            "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [tuple(row) for row in migration_rows] == [
            (1, "initial", applied[0].checksum),
            (2, "unique_invocation_event", applied[1].checksum),
        ]
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()


def test_existing_version_one_database_gains_the_authoritative_decision_index() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        initial = apply_initial_schema(connection)
        insert_run(connection)
        insert_ledger_entry(connection, 1, "trace_event")
        insert_ledger_entry(connection, 2, "invocation_decision")
        connection.execute(
            """
            INSERT INTO projection_events(
                run_id, event_id, sequence, source_event_id, ledger_position, record_json
            ) VALUES (?, ?, 1, 'source-1', 1, ?)
            """,
            (RUN_ID, EVENT_ID, b"{}"),
        )
        connection.execute(
            """
            INSERT INTO projection_decisions(
                run_id, decision_id, event_sequence, ledger_position, created_at, record_json
            ) VALUES (?, ?, 1, 2, ?, ?)
            """,
            (RUN_ID, DECISION_ID, NOW, b"{}"),
        )
        connection.commit()

        applied = apply_migrations(connection)

        assert tuple(migration.version for migration in applied) == (2,)
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        rows = connection.execute(
            "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert rows == [
            (1, "initial", initial.checksum),
            (2, "unique_invocation_event", applied[0].checksum),
        ]
        index = connection.execute(
            """
            SELECT sql
            FROM sqlite_schema
            WHERE type = 'index' AND name = 'projection_decisions_authoritative_event_idx'
            """
        ).fetchone()
        assert index is not None
        assert "UNIQUE INDEX" in str(index[0])
        assert connection.execute("SELECT count(*) FROM ledger_entries").fetchone()[0] == 2
        assert connection.execute("SELECT decision_id FROM projection_decisions").fetchone()[0] == (
            DECISION_ID
        )
    finally:
        connection.close()


def test_version_one_upgrade_repairs_ambiguous_sacrificial_decision_projection() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        apply_initial_schema(connection)
        insert_run(connection)
        insert_ledger_entry(connection, 1, "trace_event")
        insert_ledger_entry(connection, 2, "invocation_decision")
        connection.execute(
            """
            INSERT INTO projection_events(
                run_id, event_id, sequence, source_event_id, ledger_position, record_json
            ) VALUES (?, ?, 1, 'source-1', 1, ?)
            """,
            (RUN_ID, EVENT_ID, b"{}"),
        )
        for decision_id in (
            DECISION_ID,
            "00000000-0000-4000-8000-000000000704",
        ):
            connection.execute(
                """
                INSERT INTO projection_decisions(
                    run_id, decision_id, event_sequence, ledger_position, created_at, record_json
                ) VALUES (?, ?, 1, ?, ?, ?)
                """,
                (RUN_ID, decision_id, 2, NOW, b"{}"),
            )
        connection.commit()

        applied = apply_migrations(connection)

        assert tuple(migration.version for migration in applied) == (2,)
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM schema_migrations").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM projection_decisions").fetchone()[0] == 1
        assert (
            connection.execute(
                """
                SELECT 1 FROM sqlite_schema
                WHERE type = 'index'
                  AND name = 'projection_decisions_authoritative_event_idx'
                """
            ).fetchone()
            is not None
        )
    finally:
        connection.close()


def test_unknown_newer_schema_is_refused_without_creating_tables() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(f"PRAGMA user_version = {LATEST_SCHEMA_VERSION + 1}")

        with pytest.raises(SchemaTooNewError) as error:
            apply_migrations(connection)

        assert error.value.found == LATEST_SCHEMA_VERSION + 1
        assert error.value.supported == LATEST_SCHEMA_VERSION
        assert (
            connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name = 'runs'"
            ).fetchone()
            is None
        )
        assert connection.execute("PRAGMA user_version").fetchone()[0] == LATEST_SCHEMA_VERSION + 1
    finally:
        connection.close()


@pytest.mark.parametrize("table_name", ("unrelated_data", "sqlitex_customer_data"))
def test_unidentified_nonempty_database_is_not_claimed_by_saliencegate(
    table_name: str,
) -> None:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(f"CREATE TABLE {table_name}(value TEXT NOT NULL)")
        connection.execute(f"INSERT INTO {table_name} VALUES ('preserved')")
        connection.commit()

        with pytest.raises(MigrationIntegrityError):
            apply_migrations(connection)

        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute(f"SELECT value FROM {table_name}").fetchone()[0] == "preserved"
        assert (
            connection.execute("SELECT 1 FROM sqlite_schema WHERE name = 'runs'").fetchone() is None
        )
    finally:
        connection.close()


def test_migrations_require_a_sqlite_connection_and_idle_transaction() -> None:
    with pytest.raises(TypeError, match=r"sqlite3\.Connection"):
        apply_migrations(object())  # type: ignore[arg-type]

    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("BEGIN")
        with pytest.raises(MigrationError, match="idle"):
            apply_migrations(connection)
    finally:
        connection.close()


@pytest.mark.parametrize("current_version", range(LATEST_SCHEMA_VERSION + 1))
def test_foreign_database_identity_is_refused(current_version: int) -> None:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("PRAGMA application_id = 42")
        connection.execute(f"PRAGMA user_version = {current_version}")

        with pytest.raises(MigrationIntegrityError):
            apply_migrations(connection)

        assert connection.execute("PRAGMA application_id").fetchone()[0] == 42
    finally:
        connection.close()


def test_existing_version_without_migration_history_is_refused() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        connection.execute("PRAGMA user_version = 1")

        with pytest.raises(MigrationIntegrityError):
            apply_migrations(connection)
    finally:
        connection.close()


def test_changed_applied_migration_checksum_is_refused() -> None:
    with migrated_connection() as connection:
        connection.execute(
            "UPDATE schema_migrations SET checksum = ? WHERE version = 1",
            ("0" * 64,),
        )
        connection.commit()

        with pytest.raises(MigrationIntegrityError):
            apply_migrations(connection)

        assert connection.execute("PRAGMA user_version").fetchone()[0] == LATEST_SCHEMA_VERSION


def test_incomplete_applied_migration_history_is_refused() -> None:
    with migrated_connection() as connection:
        connection.execute("DELETE FROM schema_migrations")
        connection.commit()

        with pytest.raises(MigrationIntegrityError):
            apply_migrations(connection)

        assert connection.execute("PRAGMA user_version").fetchone()[0] == LATEST_SCHEMA_VERSION


def test_interrupted_migration_rolls_back_schema_identity_and_ddl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = discover_migrations()[0]
    broken_sql = f"{original.sql}\nCREATE TABLE runs (duplicate INTEGER);\n"
    broken = Migration(
        version=original.version,
        name=original.name,
        sql=broken_sql,
        checksum=migration_checksum(broken_sql),
    )
    monkeypatch.setattr(migration_module, "discover_migrations", lambda: (broken,))
    connection = sqlite3.connect(":memory:")
    try:
        with pytest.raises(MigrationError, match="failed"):
            apply_migrations(connection)

        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0
        assert (
            connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name = 'runs'"
            ).fetchone()
            is None
        )
    finally:
        connection.close()


def test_migration_that_omits_database_identity_is_rolled_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = discover_migrations()[0]
    sql_without_identity = original.sql.replace(
        "PRAGMA application_id = 0x534C4754;\n\n",
        "",
        1,
    )
    migration = Migration(
        version=original.version,
        name=original.name,
        sql=sql_without_identity,
        checksum=migration_checksum(sql_without_identity),
    )
    monkeypatch.setattr(migration_module, "discover_migrations", lambda: (migration,))
    connection = sqlite3.connect(":memory:")
    try:
        with pytest.raises(MigrationIntegrityError):
            apply_migrations(connection)

        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0
        assert (
            connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name = 'runs'"
            ).fetchone()
            is None
        )
    finally:
        connection.close()


def test_ledger_envelope_constraints_and_foreign_keys_fail_closed() -> None:
    with migrated_connection() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            insert_ledger_entry(connection, 1, "trace_event")

        insert_run(connection)
        insert_ledger_entry(connection, 1, "trace_event")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO ledger_entries(
                    run_id, position, record_key, record_type, entry_json,
                    record_algorithm, record_tag, chain_algorithm, chain_tag
                )
                VALUES (?, 2, 'missing-previous', 'signal', ?, 'hmac_sha256', ?,
                        'hmac_sha256', ?)
                """,
                (RUN_ID, b"{}", "b" * 64, "c" * 64),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO ledger_entries(
                    run_id, position, record_key, record_type, entry_json,
                    record_algorithm, record_tag, previous_chain_algorithm,
                    previous_chain_tag, chain_algorithm, chain_tag
                )
                VALUES (?, 2, 'text-json', 'signal', '{}', 'hmac_sha256', ?,
                        'hmac_sha256', ?, 'hmac_sha256', ?)
                """,
                (RUN_ID, "b" * 64, "a" * 64, "c" * 64),
            )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_head_and_state_must_reference_an_authoritative_ledger_position() -> None:
    with migrated_connection() as connection:
        insert_run(connection)
        insert_ledger_entry(connection, 1, "trace_event")

        with pytest.raises(sqlite3.IntegrityError), connection:
            connection.execute(
                """
                    INSERT INTO ledger_heads(
                        run_id, entry_count, algorithm, chain_tag, projection_tag, head_tag
                    )
                    VALUES (?, 2, 'hmac_sha256', ?, ?, ?)
                    """,
                (RUN_ID, "a" * 64, "b" * 64, "c" * 64),
            )
        with pytest.raises(sqlite3.IntegrityError), connection:
            connection.execute(
                """
                    INSERT INTO projection_state(
                        run_id, ledger_position, ingestion_cursor, memory_cursor,
                        projection_digests_json, state_algorithm, state_tag
                    )
                    VALUES (?, 2, 1, 0, ?, 'hmac_sha256', ?)
                    """,
                (RUN_ID, b"{}", "d" * 64),
            )


def test_fts_tracks_only_latest_active_memory_rows() -> None:
    with migrated_connection() as connection:
        insert_projection_parents(connection)
        rows = (
            (
                "00000000-0000-4000-8000-000000000711",
                1,
                "active",
                1,
                "Repository evidence stays searchable.",
            ),
            (
                "00000000-0000-4000-8000-000000000712",
                1,
                "invalidated",
                1,
                "Repository evidence is inactive.",
            ),
            (
                "00000000-0000-4000-8000-000000000713",
                1,
                "active",
                0,
                "Repository evidence is historical.",
            ),
        )
        for memory_id, revision, validity, is_latest, content in rows:
            connection.execute(
                """
                INSERT INTO projection_memories(
                    run_id, memory_id, revision, kind, validity, trust_label, content,
                    is_latest, source_cycle_id, source_cycle_revision, record_json
                )
                VALUES (?, ?, ?, 'knowledge', ?, 'trusted_controller', ?, ?, ?, 1, ?)
                """,
                (
                    RUN_ID,
                    memory_id,
                    revision,
                    validity,
                    content,
                    is_latest,
                    CYCLE_ID,
                    b"{}",
                ),
            )

        assert fts_hits(connection, "repository") == [
            (RUN_ID, "00000000-0000-4000-8000-000000000711", 1)
        ]
        connection.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild')")
        assert fts_hits(connection, "repository") == [
            (RUN_ID, "00000000-0000-4000-8000-000000000711", 1)
        ]
        memory_id = "00000000-0000-4000-8000-000000000711"
        connection.execute(
            """
            UPDATE projection_memories
            SET content = 'Updated constraint remains searchable.'
            WHERE run_id = ? AND memory_id = ? AND revision = 1
            """,
            (RUN_ID, memory_id),
        )
        assert fts_hits(connection, "repository") == []
        assert fts_hits(connection, "constraint") == [(RUN_ID, memory_id, 1)]

        connection.execute(
            """
            UPDATE projection_memories
            SET validity = 'invalidated'
            WHERE run_id = ? AND memory_id = ? AND revision = 1
            """,
            (RUN_ID, memory_id),
        )
        assert fts_hits(connection, "constraint") == []

        connection.execute(
            """
            UPDATE projection_memories
            SET validity = 'active'
            WHERE run_id = ? AND memory_id = ? AND revision = 1
            """,
            (RUN_ID, memory_id),
        )
        assert fts_hits(connection, "constraint") == [(RUN_ID, memory_id, 1)]
        connection.execute(
            "DELETE FROM projection_memories WHERE run_id = ? AND memory_id = ?",
            (RUN_ID, memory_id),
        )
        assert fts_hits(connection, "constraint") == []
