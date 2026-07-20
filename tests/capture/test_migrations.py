from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

import saliencegate.capture.migrations as migration_module
from saliencegate.capture.migrations import (
    APPLICATION_ID,
    LATEST_SCHEMA_VERSION,
    CaptureMigrationError,
    CaptureMigrationIntegrityError,
    CaptureSchemaTooNewError,
    apply_capture_migrations,
    discover_capture_migrations,
    initialize_capture_store,
    validate_capture_store_schema,
)
from saliencegate.security import (
    InstallationKey,
    StableFileAuthorization,
)

EXPECTED_TABLES = frozenset(
    {
        "capture_events",
        "capture_heads",
        "capture_health",
        "capture_sessions",
        "capture_transport_heads",
        "capture_transport_receipts",
        "connections",
        "deleted_sessions",
        "feedback_labels",
        "schema_migrations",
    }
)
MIGRATION_FAILED_MESSAGE = "capture store migration failed"
MIGRATION_INTEGRITY_MESSAGE = "capture store schema integrity failed"
SCHEMA_TOO_NEW_MESSAGE = "capture store schema is newer than this build"
KEY = InstallationKey(b"m" * 32)


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _tables(connection: sqlite3.Connection) -> frozenset[str]:
    return frozenset(
        str(row[0])
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_schema
            WHERE type = 'table' AND name NOT GLOB 'sqlite_*'
            """
        )
    )


def _sidecar_paths(path: Path) -> tuple[Path, Path, Path]:
    return tuple(path.with_name(f"{path.name}{suffix}") for suffix in ("-wal", "-shm", "-journal"))


def _assert_content_free_error(error: BaseException, expected: str, path: Path) -> None:
    assert str(error) == expected
    assert error.args == (expected,)
    rendered = repr(error)
    assert path.name not in rendered
    assert "sentinel" not in rendered


@pytest.fixture
def initialized_store(tmp_path: Path) -> Path:
    path = tmp_path / "capture.sqlite3"
    receipt = initialize_capture_store(path)
    assert receipt.schema_version == LATEST_SCHEMA_VERSION
    assert receipt.applied_versions == (1, 2)
    assert repr(receipt) == "CaptureMigrationReceipt(<redacted>)"
    return path


def test_capture_migration_constants_and_packaged_resource_are_frozen() -> None:
    migrations = discover_capture_migrations()

    assert APPLICATION_ID == 0x53474350
    assert LATEST_SCHEMA_VERSION == 2
    assert tuple(migration.version for migration in migrations) == (1, 2)
    assert tuple(migration.name for migration in migrations) == (
        "capture_store",
        "transport_receipts",
    )
    assert all(
        migration.checksum == hashlib.sha256(migration.sql.encode("utf-8")).hexdigest()
        for migration in migrations
    )
    assert all(len(migration.checksum) == 64 for migration in migrations)
    assert "PRAGMA application_id = 0x53474350;" in migrations[0].sql
    assert all(
        f"'{label}'" in migrations[0].sql
        for label in ("memory-needed", "not-memory-needed", "uncertain")
    )
    assert "'helpful'" not in migrations[0].sql
    assert "'not_helpful'" not in migrations[0].sql
    assert "applied_at" not in migrations[0].sql


@pytest.mark.parametrize(
    ("resource_name", "payload"),
    (
        ("0001_capture_store.sql", b"\xef\xbb\xbfSELECT 1;\n"),
        ("0001_capture_store.sql", b"CREATE TABLE incomplete("),
        ("0001_capture_store.sql", b"\xff"),
        ("0003_noncontiguous.sql", b"SELECT 1;\n"),
    ),
    ids=("bom", "incomplete-sql", "invalid-utf8", "noncontiguous"),
)
def test_discover_rejects_corrupt_or_noncontiguous_packaged_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource_name: str,
    payload: bytes,
) -> None:
    resource_root = tmp_path / "migration-resources-sentinel"
    resource_root.mkdir()
    (resource_root / resource_name).write_bytes(payload)
    monkeypatch.setattr(migration_module.resources, "files", lambda _package: resource_root)

    with pytest.raises(CaptureMigrationError) as captured:
        discover_capture_migrations()

    _assert_content_free_error(captured.value, MIGRATION_FAILED_MESSAGE, resource_root)


def test_initialize_builds_exact_capture_schema_and_current_store_is_a_no_op(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.sqlite3"

    applied = initialize_capture_store(path, busy_timeout_ms=2_000)
    reapplied = initialize_capture_store(path, busy_timeout_ms=2_000)

    assert (applied.schema_version, applied.applied_versions) == (2, (1, 2))
    assert (reapplied.schema_version, reapplied.applied_versions) == (2, ())
    migrations = discover_capture_migrations()
    with _connect(path) as connection:
        assert _tables(connection) == EXPECTED_TABLES
        assert connection.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID
        assert connection.execute("PRAGMA user_version").fetchone()[0] == LATEST_SCHEMA_VERSION
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        rows = connection.execute(
            "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            (migration.version, migration.name, migration.checksum) for migration in migrations
        ]
        assert validate_capture_store_schema(connection) is None
        assert not connection.in_transaction


def test_initialize_authenticates_and_upgrades_a_version_one_store(tmp_path: Path) -> None:
    path = tmp_path / "capture-v1.sqlite3"
    first = discover_capture_migrations()[0]
    with sqlite3.connect(path, isolation_level=None) as connection:
        for statement in migration_module._parse_statements(first.sql):
            connection.execute(statement)
        connection.execute(
            "INSERT INTO schema_migrations(version, name, checksum) VALUES (?, ?, ?)",
            (first.version, first.name, first.checksum),
        )
        connection.execute("PRAGMA user_version = 1")
    path.chmod(0o600)

    receipt = initialize_capture_store(path)

    assert (receipt.schema_version, receipt.applied_versions) == (2, (2,))
    with _connect(path) as connection:
        assert _tables(connection) == EXPECTED_TABLES
        assert validate_capture_store_schema(connection) is None


@pytest.mark.parametrize("busy_timeout_ms", (True, 0, 60_001, 1.5))
def test_initialize_requires_a_bounded_exact_integer_timeout(
    tmp_path: Path,
    busy_timeout_ms: object,
) -> None:
    path = tmp_path / "capture.sqlite3"

    with pytest.raises(CaptureMigrationError) as captured:
        initialize_capture_store(path, busy_timeout_ms=busy_timeout_ms)  # type: ignore[arg-type]

    _assert_content_free_error(captured.value, MIGRATION_FAILED_MESSAGE, path)
    assert not path.exists()


def test_initialize_rejects_an_empty_path_content_free() -> None:
    marker = Path("empty-path-sentinel.sqlite3")

    with pytest.raises(CaptureMigrationError) as captured:
        initialize_capture_store("")

    _assert_content_free_error(captured.value, MIGRATION_FAILED_MESSAGE, marker)


def test_validate_requires_an_idle_sqlite_connection(initialized_store: Path) -> None:
    with pytest.raises(CaptureMigrationError) as invalid_type:
        validate_capture_store_schema(object())  # type: ignore[arg-type]
    _assert_content_free_error(invalid_type.value, MIGRATION_FAILED_MESSAGE, initialized_store)

    with _connect(initialized_store) as connection:
        connection.execute("BEGIN")
        with pytest.raises(CaptureMigrationError) as captured:
            validate_capture_store_schema(connection)
        _assert_content_free_error(captured.value, MIGRATION_FAILED_MESSAGE, initialized_store)


def test_apply_requires_an_idle_exact_sqlite_connection() -> None:
    marker = Path("apply-guard-sentinel.sqlite3")
    with pytest.raises(CaptureMigrationError) as invalid_type:
        apply_capture_migrations(object())  # type: ignore[arg-type]
    _assert_content_free_error(invalid_type.value, MIGRATION_FAILED_MESSAGE, marker)

    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        connection.execute("BEGIN")
        with pytest.raises(CaptureMigrationError) as active_transaction:
            apply_capture_migrations(connection)
        _assert_content_free_error(
            active_transaction.value,
            MIGRATION_FAILED_MESSAGE,
            marker,
        )
        assert connection.in_transaction
        connection.rollback()
    finally:
        connection.close()


def test_future_capture_schema_is_refused_without_downgrade(initialized_store: Path) -> None:
    future_version = LATEST_SCHEMA_VERSION + 1
    with _connect(initialized_store) as connection:
        connection.execute(f"PRAGMA user_version = {future_version}")
        connection.commit()
        with pytest.raises(CaptureSchemaTooNewError) as direct_validation:
            validate_capture_store_schema(connection)
        _assert_content_free_error(
            direct_validation.value,
            SCHEMA_TOO_NEW_MESSAGE,
            initialized_store,
        )

    with pytest.raises(CaptureSchemaTooNewError) as captured:
        initialize_capture_store(initialized_store)

    _assert_content_free_error(captured.value, SCHEMA_TOO_NEW_MESSAGE, initialized_store)
    with _connect(initialized_store) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == future_version
        assert _tables(connection) == EXPECTED_TABLES


def test_future_main_database_is_refused_during_read_only_preflight(tmp_path: Path) -> None:
    path = tmp_path / "future-main-sentinel.sqlite3"
    future_version = LATEST_SCHEMA_VERSION + 1
    with sqlite3.connect(path, isolation_level=None) as connection:
        connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version = {future_version}")
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)
    path.chmod(0o600)
    before = path.read_bytes()

    with pytest.raises(CaptureSchemaTooNewError) as captured:
        initialize_capture_store(path)

    _assert_content_free_error(captured.value, SCHEMA_TOO_NEW_MESSAGE, path)
    assert path.read_bytes() == before
    assert all(not sidecar.exists() for sidecar in _sidecar_paths(path))


def test_validate_rejects_capture_identity_without_migration_history() -> None:
    marker = Path("missing-history-sentinel.sqlite3")
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version = {LATEST_SCHEMA_VERSION}")

        with pytest.raises(CaptureMigrationIntegrityError) as captured:
            validate_capture_store_schema(connection)

        _assert_content_free_error(captured.value, MIGRATION_INTEGRITY_MESSAGE, marker)
    finally:
        connection.close()


@pytest.mark.parametrize("identity", ("foreign", "unidentified"))
def test_unknown_database_is_refused_before_wal_or_any_byte_mutation(
    tmp_path: Path,
    identity: str,
) -> None:
    path = tmp_path / f"{identity}-sentinel.sqlite3"
    with _connect(path) as connection:
        if identity == "foreign":
            connection.execute("PRAGMA application_id = 42")
            connection.execute("PRAGMA user_version = 1")
        connection.execute("CREATE TABLE sentinel_records(value TEXT NOT NULL)")
        connection.execute("INSERT INTO sentinel_records VALUES ('sentinel-content')")
        connection.commit()
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    path.chmod(0o600)
    before = path.read_bytes()

    with pytest.raises(CaptureMigrationIntegrityError) as captured:
        initialize_capture_store(path)

    _assert_content_free_error(captured.value, MIGRATION_INTEGRITY_MESSAGE, path)
    assert path.read_bytes() == before
    assert all(not sidecar.exists() for sidecar in _sidecar_paths(path))
    with _connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert connection.execute("SELECT value FROM sentinel_records").fetchone()[0] == (
            "sentinel-content"
        )
        assert "schema_migrations" not in _tables(connection)


def test_changed_migration_checksum_is_refused_without_repair(initialized_store: Path) -> None:
    changed_checksum = "f" * 64
    with _connect(initialized_store) as connection:
        connection.execute(
            "UPDATE schema_migrations SET checksum = ? WHERE version = 1",
            (changed_checksum,),
        )
        connection.commit()

    with pytest.raises(CaptureMigrationIntegrityError) as captured:
        initialize_capture_store(initialized_store)

    _assert_content_free_error(captured.value, MIGRATION_INTEGRITY_MESSAGE, initialized_store)
    with _connect(initialized_store) as connection:
        assert (
            connection.execute(
                "SELECT checksum FROM schema_migrations WHERE version = 1"
            ).fetchone()[0]
            == changed_checksum
        )


@pytest.mark.parametrize("history_shape", ("missing", "extra"))
def test_migration_history_cardinality_mismatch_is_refused(
    initialized_store: Path,
    history_shape: str,
) -> None:
    with _connect(initialized_store) as connection:
        if history_shape == "missing":
            connection.execute("DELETE FROM schema_migrations")
        else:
            connection.execute(
                """
                INSERT INTO schema_migrations(version, name, checksum)
                VALUES (3, 'unexpected', ?)
                """,
                ("e" * 64,),
            )
        connection.commit()

    with pytest.raises(CaptureMigrationIntegrityError) as captured:
        initialize_capture_store(initialized_store)

    _assert_content_free_error(captured.value, MIGRATION_INTEGRITY_MESSAGE, initialized_store)
    with _connect(initialized_store) as connection:
        expected_count = 0 if history_shape == "missing" else 3
        assert connection.execute("SELECT count(*) FROM schema_migrations").fetchone()[0] == (
            expected_count
        )


def test_extra_schema_object_is_refused_without_destructive_repair(initialized_store: Path) -> None:
    with _connect(initialized_store) as connection:
        connection.execute("CREATE TABLE sentinel_extra(value TEXT NOT NULL)")
        connection.execute("INSERT INTO sentinel_extra VALUES ('sentinel-content')")
        connection.commit()

    with pytest.raises(CaptureMigrationIntegrityError) as captured:
        initialize_capture_store(initialized_store)

    _assert_content_free_error(captured.value, MIGRATION_INTEGRITY_MESSAGE, initialized_store)
    with _connect(initialized_store) as connection:
        assert _tables(connection) == EXPECTED_TABLES | {"sentinel_extra"}
        assert connection.execute("SELECT value FROM sentinel_extra").fetchone()[0] == (
            "sentinel-content"
        )


def test_corrupt_database_is_refused_without_rewrite_or_sidecars(tmp_path: Path) -> None:
    path = tmp_path / "corrupt-sentinel.sqlite3"
    original = b"not-a-sqlite-database\x00sentinel-content"
    path.write_bytes(original)
    path.chmod(0o600)

    connection = sqlite3.connect(path, isolation_level=None)
    try:
        with pytest.raises(CaptureMigrationIntegrityError) as validation_error:
            validate_capture_store_schema(connection)
        _assert_content_free_error(
            validation_error.value,
            MIGRATION_INTEGRITY_MESSAGE,
            path,
        )
    finally:
        connection.close()

    with pytest.raises(CaptureMigrationError) as captured:
        initialize_capture_store(path)

    _assert_content_free_error(captured.value, MIGRATION_FAILED_MESSAGE, path)
    assert path.read_bytes() == original
    assert all(not sidecar.exists() for sidecar in _sidecar_paths(path))


@pytest.mark.parametrize(
    ("state", "version", "application_id", "has_user_schema"),
    (
        ("current-foreign", LATEST_SCHEMA_VERSION, 42, False),
        ("empty-foreign", 0, 42, False),
        ("unidentified-user-schema", 0, 0, True),
    ),
)
def test_apply_rejects_foreign_identity_or_unidentified_user_schema_and_rolls_back(
    state: str,
    version: int,
    application_id: int,
    has_user_schema: bool,
) -> None:
    marker = Path(f"{state}-sentinel.sqlite3")
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        connection.execute(f"PRAGMA application_id = {application_id}")
        connection.execute(f"PRAGMA user_version = {version}")
        if has_user_schema:
            connection.execute("CREATE TABLE sentinel_records(value TEXT NOT NULL)")

        with pytest.raises(CaptureMigrationIntegrityError) as captured:
            apply_capture_migrations(connection)

        _assert_content_free_error(captured.value, MIGRATION_INTEGRITY_MESSAGE, marker)
        assert not connection.in_transaction
        assert connection.execute("PRAGMA application_id").fetchone() == (application_id,)
        assert connection.execute("PRAGMA user_version").fetchone() == (version,)
        assert ("sentinel_records" in _tables(connection)) is has_user_schema
    finally:
        connection.close()


def test_apply_normalizes_unexpected_connection_errors_and_rolls_back() -> None:
    marker = Path("row-factory-sentinel.sqlite3")
    connection = sqlite3.connect(":memory:", isolation_level=None)

    def failing_row_factory(_cursor: sqlite3.Cursor, _row: tuple[object, ...]) -> object:
        raise RuntimeError("row-factory-sentinel")

    connection.row_factory = failing_row_factory
    try:
        with pytest.raises(CaptureMigrationError) as captured:
            apply_capture_migrations(connection)

        _assert_content_free_error(captured.value, MIGRATION_FAILED_MESSAGE, marker)
        assert not connection.in_transaction
    finally:
        connection.close()


def test_interrupted_initial_migration_rolls_back_identity_history_and_ddl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = discover_capture_migrations()[0]
    broken_sql = f"{original.sql}\nCREATE TABLE connections (duplicate INTEGER);\n"
    broken = replace(
        original,
        sql=broken_sql,
        checksum=hashlib.sha256(broken_sql.encode("utf-8")).hexdigest(),
    )
    monkeypatch.setattr(migration_module, "discover_capture_migrations", lambda: (broken,))
    path = tmp_path / "interrupted.sqlite3"

    with pytest.raises(CaptureMigrationError) as captured:
        initialize_capture_store(path)

    _assert_content_free_error(captured.value, MIGRATION_FAILED_MESSAGE, path)
    with _connect(path) as connection:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert _tables(connection) == frozenset()
    assert all(not sidecar.exists() for sidecar in _sidecar_paths(path))


def test_initialize_recovers_a_preexisting_empty_private_database(tmp_path: Path) -> None:
    path = tmp_path / "empty-existing.sqlite3"
    path.touch(mode=0o600)
    path.chmod(0o600)
    assert path.read_bytes() == b""

    receipt = initialize_capture_store(path)

    assert receipt.applied_versions == (1, 2)
    with _connect(path) as connection:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID
        assert connection.execute("PRAGMA user_version").fetchone()[0] == LATEST_SCHEMA_VERSION
        assert _tables(connection) == EXPECTED_TABLES


@pytest.mark.skipif(os.name != "posix", reason="descriptor-bound claim requires POSIX")
def test_initialize_rejects_database_replacement_between_preflight_and_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "capture.sqlite3"
    initialize_capture_store(path)
    original = tmp_path / "original-capture.sqlite3"
    original_bytes = path.read_bytes()
    original_sidecar_bytes = tuple(
        sidecar.read_bytes() if sidecar.exists() else None for sidecar in _sidecar_paths(path)
    )
    real_preflight = cast(Callable[..., None], migration_module._preflight_existing_store)
    replacement_installed = False

    def replace_after_preflight(
        candidate: Path,
        *authorizations: StableFileAuthorization,
    ) -> None:
        nonlocal replacement_installed
        real_preflight(candidate, *authorizations)
        candidate.rename(original)
        for source, destination in zip(
            _sidecar_paths(candidate),
            _sidecar_paths(original),
            strict=True,
        ):
            if source.exists():
                source.rename(destination)
        candidate.touch(mode=0o600)
        candidate.chmod(0o600)
        replacement_installed = True

    monkeypatch.setattr(
        migration_module,
        "_preflight_existing_store",
        replace_after_preflight,
    )

    with pytest.raises(CaptureMigrationError) as captured:
        initialize_capture_store(path)

    _assert_content_free_error(captured.value, MIGRATION_FAILED_MESSAGE, path)
    assert replacement_installed
    assert original.read_bytes() == original_bytes
    assert path.read_bytes() == b""
    assert stat.S_IMODE(original.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert all(not sidecar.exists() for sidecar in _sidecar_paths(path))
    assert (
        tuple(
            sidecar.read_bytes() if sidecar.exists() else None
            for sidecar in _sidecar_paths(original)
        )
        == original_sidecar_bytes
    )


def test_initialized_store_configuration_refuses_non_wal_connections() -> None:
    marker = Path("non-wal-sentinel.sqlite3")
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        with pytest.raises(CaptureMigrationError) as captured:
            migration_module._configure_initialized_store(connection, busy_timeout_ms=1_000)

        _assert_content_free_error(captured.value, MIGRATION_FAILED_MESSAGE, marker)
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("memory",)
    finally:
        connection.close()


def test_runtime_open_never_creates_a_missing_capture_store(tmp_path: Path) -> None:
    from saliencegate.capture.store import CaptureStore

    path = tmp_path / "missing.sqlite3"

    with pytest.raises(CaptureMigrationError):
        CaptureStore.open(path, installation_key=KEY)

    assert not path.exists()
    assert all(not sidecar.exists() for sidecar in _sidecar_paths(path))


def test_runtime_open_validates_current_schema_without_invoking_initializer(
    initialized_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saliencegate.capture.store as store_module
    from saliencegate.capture.store import CaptureStore

    def unexpected_initializer(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("runtime open invoked the migration initializer")

    monkeypatch.setattr(
        migration_module,
        "initialize_capture_store",
        unexpected_initializer,
    )
    monkeypatch.setattr(
        store_module,
        "initialize_capture_store",
        unexpected_initializer,
        raising=False,
    )

    with CaptureStore.open(initialized_store, installation_key=KEY):
        pass


def test_hook_open_skips_full_data_pragmas_while_maintenance_retains_them(
    initialized_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saliencegate.capture.store as store_module
    from saliencegate.capture.store import CaptureStore, CaptureStoreMode

    original_connect = sqlite3.connect
    statements: list[str] = []

    def tracing_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = original_connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(store_module.sqlite3, "connect", tracing_connect)

    with CaptureStore.open(
        initialized_store,
        installation_key=KEY,
        mode=CaptureStoreMode.HOOK,
    ):
        pass

    hook_statements = tuple(statement.strip().casefold() for statement in statements)
    assert hook_statements.count("pragma user_version") == 2
    assert hook_statements.count("pragma application_id") == 2
    assert "pragma quick_check" not in hook_statements
    assert "pragma foreign_key_check" not in hook_statements

    statements.clear()
    with CaptureStore.open(
        initialized_store,
        installation_key=KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ):
        pass

    maintenance_statements = tuple(statement.strip().casefold() for statement in statements)
    assert maintenance_statements.count("pragma quick_check") == 2
    assert maintenance_statements.count("pragma foreign_key_check") == 2
