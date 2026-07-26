"""Explicit, checksummed migrations for the isolated capture store."""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PureWindowsPath
from typing import Final, Never

from saliencegate.security import (
    SecureFileError,
    StableFileAuthorization,
    claim_private_sqlite_location,
    inspect_private_file_location,
)
from saliencegate.security.windows import (
    NativeWindowsSecurityOperations,
    WindowsPathAuthorization,
    WindowsPathKind,
    WindowsSecurityError,
    WindowsSQLiteAuthorization,
    authorize_windows_private_path,
    authorize_windows_sqlite_path,
)

APPLICATION_ID: Final = 0x53474350  # "SGCP"
LATEST_SCHEMA_VERSION: Final = 3

_MIGRATION_NAME = re.compile(r"^(?P<version>[0-9]{4})_(?P<name>[a-z][a-z0-9_]*)\.sql$")
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


class CaptureMigrationError(RuntimeError):
    """A content-free capture migration failure."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture store migration failed")


class CaptureMigrationIntegrityError(CaptureMigrationError):
    """The database identity, history, or schema is not the capture contract."""

    __slots__ = ()

    def __init__(self) -> None:
        RuntimeError.__init__(self, "capture store schema integrity failed")


class CaptureSchemaTooNewError(CaptureMigrationError):
    """The database was produced by a newer capture schema."""

    __slots__ = ()

    def __init__(self) -> None:
        RuntimeError.__init__(self, "capture store schema is newer than this build")


@dataclass(frozen=True, slots=True)
class CaptureMigration:
    version: int
    name: str
    sql: str
    checksum: str


@dataclass(frozen=True, slots=True)
class CaptureMigrationReceipt:
    schema_version: int
    applied_versions: tuple[int, ...]

    def __repr__(self) -> str:
        return "CaptureMigrationReceipt(<redacted>)"


def _migration_checksum(sql: str) -> str:
    if type(sql) is not str:
        raise CaptureMigrationError()
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def _reject_incomplete_sql(_sql: str) -> Never:
    raise CaptureMigrationError()


def _parse_statements(sql: str) -> tuple[str, ...]:
    if type(sql) is not str:
        raise CaptureMigrationError()
    statements: list[str] = []
    pending: list[str] = []
    for line in sql.splitlines(keepends=True):
        pending.append(line)
        candidate = "".join(pending)
        if sqlite3.complete_statement(candidate):
            statement = candidate.strip()
            if statement:
                statements.append(statement)
            pending = []
    if "".join(pending).strip() or not statements:
        _reject_incomplete_sql(sql)
    return tuple(statements)


def discover_capture_migrations() -> tuple[CaptureMigration, ...]:
    """Load the exact contiguous migration sequence from installed resources."""

    try:
        root = resources.files(__package__)
        migrations: list[CaptureMigration] = []
        for resource in sorted(root.iterdir(), key=lambda item: item.name):
            match = _MIGRATION_NAME.fullmatch(resource.name)
            if match is None or not resource.is_file():
                continue
            raw = resource.read_bytes()
            if raw.startswith(b"\xef\xbb\xbf"):
                raise CaptureMigrationError()
            sql = raw.decode("utf-8", errors="strict")
            _parse_statements(sql)
            migrations.append(
                CaptureMigration(
                    version=int(match.group("version")),
                    name=match.group("name"),
                    sql=sql,
                    checksum=_migration_checksum(sql),
                )
            )
        versions = tuple(item.version for item in migrations)
        expected = tuple(range(1, len(migrations) + 1))
        if not migrations or versions != expected or versions[-1] != LATEST_SCHEMA_VERSION:
            raise CaptureMigrationError()
        return tuple(migrations)
    except CaptureMigrationError:
        raise
    except Exception:
        raise CaptureMigrationError() from None


def _pragma_integer(connection: sqlite3.Connection, name: str) -> int:
    row = connection.execute(f"PRAGMA {name}").fetchone()
    if row is None or type(row[0]) is not int:
        raise CaptureMigrationIntegrityError()
    return row[0]


def _has_user_schema_objects(connection: sqlite3.Connection) -> bool:
    return (
        connection.execute(
            """
            SELECT 1 FROM sqlite_schema
            WHERE name NOT GLOB 'sqlite_*'
            LIMIT 1
            """
        ).fetchone()
        is not None
    )


def _verify_history(
    connection: sqlite3.Connection,
    migrations: tuple[CaptureMigration, ...],
    current_version: int,
) -> None:
    if current_version == 0:
        return
    table = connection.execute(
        "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'schema_migrations'"
    ).fetchone()
    if table is None:
        raise CaptureMigrationIntegrityError()
    rows = connection.execute(
        "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
    ).fetchall()
    expected = migrations[:current_version]
    if len(rows) != len(expected):
        raise CaptureMigrationIntegrityError()
    for row, migration in zip(rows, expected, strict=True):
        if tuple(row) != (migration.version, migration.name, migration.checksum):
            raise CaptureMigrationIntegrityError()


def _schema_inventory(connection: sqlite3.Connection) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(row)
        for row in connection.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_schema
            WHERE name NOT GLOB 'sqlite_*'
            ORDER BY type, name
            """
        ).fetchall()
    )


def _expected_schema_inventory(
    migrations: tuple[CaptureMigration, ...],
) -> tuple[tuple[object, ...], ...]:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        for migration in migrations:
            for statement in _parse_statements(migration.sql):
                connection.execute(statement)
        return _schema_inventory(connection)
    finally:
        connection.close()


def _validate_capture_store_metadata_state(
    connection: sqlite3.Connection,
    *,
    require_current: bool,
) -> None:
    if type(connection) is not sqlite3.Connection or connection.in_transaction:
        raise CaptureMigrationError()
    try:
        migrations = discover_capture_migrations()
        version = _pragma_integer(connection, "user_version")
        if version > LATEST_SCHEMA_VERSION:
            raise CaptureSchemaTooNewError()
        if (
            version == 0
            or (require_current and version != LATEST_SCHEMA_VERSION)
            or _pragma_integer(connection, "application_id") != APPLICATION_ID
        ):
            raise CaptureMigrationIntegrityError()
        _verify_history(connection, migrations, version)
        if _schema_inventory(connection) != _expected_schema_inventory(migrations[:version]):
            raise CaptureMigrationIntegrityError()
    except (CaptureMigrationError, CaptureMigrationIntegrityError, CaptureSchemaTooNewError):
        raise
    except sqlite3.Error:
        raise CaptureMigrationIntegrityError() from None
    except Exception:
        raise CaptureMigrationError() from None


def _validate_capture_store_data_state(connection: sqlite3.Connection) -> None:
    """Run whole-database checks for initialization, migration, maintenance, and audit."""

    if type(connection) is not sqlite3.Connection or connection.in_transaction:
        raise CaptureMigrationError()
    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchall()
        if (
            len(quick_check) != 1
            or tuple(quick_check[0]) != ("ok",)
            or connection.execute("PRAGMA foreign_key_check").fetchone()
        ):
            raise CaptureMigrationIntegrityError()
    except CaptureMigrationIntegrityError:
        raise
    except sqlite3.Error:
        raise CaptureMigrationIntegrityError() from None
    except Exception:
        raise CaptureMigrationError() from None


def _validate_capture_store_schema_metadata(connection: sqlite3.Connection) -> None:
    """Validate the exact current schema contract without scanning retained data."""

    _validate_capture_store_metadata_state(connection, require_current=True)


def validate_capture_store_schema(connection: sqlite3.Connection) -> None:
    """Validate the current closed schema and all retained database pages."""

    _validate_capture_store_schema_metadata(connection)
    _validate_capture_store_data_state(connection)


def apply_capture_migrations(connection: sqlite3.Connection) -> tuple[CaptureMigration, ...]:
    """Apply capture migrations only from the explicit initialization path."""

    if type(connection) is not sqlite3.Connection or connection.in_transaction:
        raise CaptureMigrationError()
    migrations = discover_capture_migrations()
    applied: list[CaptureMigration] = []
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN EXCLUSIVE")
        version = _pragma_integer(connection, "user_version")
        if version > LATEST_SCHEMA_VERSION:
            raise CaptureSchemaTooNewError()
        application_id = _pragma_integer(connection, "application_id")
        if version > 0 and application_id != APPLICATION_ID:
            raise CaptureMigrationIntegrityError()
        if version == 0 and application_id not in (0, APPLICATION_ID):
            raise CaptureMigrationIntegrityError()
        if version == 0 and _has_user_schema_objects(connection):
            raise CaptureMigrationIntegrityError()
        _verify_history(connection, migrations, version)
        for migration in migrations[version:]:
            for statement in _parse_statements(migration.sql):
                connection.execute(statement)
            connection.execute(
                """
                INSERT INTO schema_migrations(version, name, checksum)
                VALUES (?, ?, ?)
                """,
                (
                    migration.version,
                    migration.name,
                    migration.checksum,
                ),
            )
            connection.execute(f"PRAGMA user_version = {migration.version}")
            applied.append(migration)
        connection.commit()
        validate_capture_store_schema(connection)
        return tuple(applied)
    except (CaptureMigrationError, CaptureMigrationIntegrityError, CaptureSchemaTooNewError):
        connection.rollback()
        raise
    except sqlite3.Error:
        connection.rollback()
        raise CaptureMigrationError() from None
    except Exception:
        connection.rollback()
        raise CaptureMigrationError() from None


def _preflight_existing_store(
    path: Path,
    location: StableFileAuthorization,
) -> None:
    location.revalidate()
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"{path.as_uri()}?mode=ro&immutable=1",
            isolation_level=None,
            uri=True,
        )
        version = _pragma_integer(connection, "user_version")
        application_id = _pragma_integer(connection, "application_id")
        if version > LATEST_SCHEMA_VERSION:
            raise CaptureSchemaTooNewError()
        if version == 0:
            if application_id not in (0, APPLICATION_ID) or _has_user_schema_objects(connection):
                raise CaptureMigrationIntegrityError()
        else:
            _validate_capture_store_metadata_state(connection, require_current=False)
            _validate_capture_store_data_state(connection)
    finally:
        if connection is not None:
            connection.close()
    location.revalidate()


def _preflight_existing_windows_store(
    path: Path,
    authorization: WindowsPathAuthorization,
) -> None:  # pragma: no cover - exercised by native Windows R01
    authorization.revalidate()
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"{path.as_uri()}?mode=ro&immutable=1",
            isolation_level=None,
            uri=True,
        )
        version = _pragma_integer(connection, "user_version")
        application_id = _pragma_integer(connection, "application_id")
        if version > LATEST_SCHEMA_VERSION:
            raise CaptureSchemaTooNewError()
        if version == 0:
            if application_id not in (0, APPLICATION_ID) or _has_user_schema_objects(connection):
                raise CaptureMigrationIntegrityError()
        else:
            _validate_capture_store_metadata_state(connection, require_current=False)
            _validate_capture_store_data_state(connection)
        authorization.revalidate()
    finally:
        if connection is not None:
            connection.close()


def _configure_initialized_store(
    connection: sqlite3.Connection,
    *,
    busy_timeout_ms: int,
) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
    connection.execute("PRAGMA trusted_schema = OFF")
    connection.execute("PRAGMA temp_store = MEMORY")
    journal = connection.execute("PRAGMA journal_mode = WAL").fetchone()
    connection.execute("PRAGMA synchronous = FULL")
    if (
        journal is None
        or str(journal[0]).casefold() != "wal"
        or connection.execute("PRAGMA foreign_keys").fetchone() != (1,)
        or connection.execute("PRAGMA busy_timeout").fetchone() != (busy_timeout_ms,)
        or connection.execute("PRAGMA trusted_schema").fetchone() != (0,)
        or connection.execute("PRAGMA temp_store").fetchone() != (2,)
        or connection.execute("PRAGMA synchronous").fetchone() != (2,)
    ):
        raise CaptureMigrationError()


def initialize_capture_store(
    path: str | os.PathLike[str],
    *,
    busy_timeout_ms: int = 5_000,
) -> CaptureMigrationReceipt:
    """Explicitly initialize or validate the capture database and migrations."""

    if type(busy_timeout_ms) is not int or not 1 <= busy_timeout_ms <= 60_000:
        raise CaptureMigrationError()
    authorization: StableFileAuthorization | WindowsSQLiteAuthorization | None = None
    connection: sqlite3.Connection | None = None
    try:
        raw_path = os.fspath(path)
        if type(raw_path) is not str or not raw_path:
            raise CaptureMigrationError()
        database_path = Path(raw_path).absolute()
        if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
            operations = NativeWindowsSecurityOperations()
            windows_path = PureWindowsPath(str(database_path))
            existing = operations.inspect_path(windows_path)
            database_authorization: WindowsPathAuthorization | None = None
            if existing is not None:
                database_authorization = authorize_windows_private_path(
                    windows_path,
                    kind=WindowsPathKind.FILE,
                    operations=operations,
                )
                _preflight_existing_windows_store(database_path, database_authorization)
            authorization = authorize_windows_sqlite_path(
                windows_path,
                operations=operations,
                create_database=True,
                database_authorization=database_authorization,
            )
        else:
            location = inspect_private_file_location(database_path)
            sidecar_locations = tuple(
                inspect_private_file_location(f"{database_path}{suffix}")
                for suffix in _SQLITE_SIDECAR_SUFFIXES
            )
            if location.target_exists:
                _preflight_existing_store(database_path, location)
            authorization = claim_private_sqlite_location(
                location,
                sidecar_locations=sidecar_locations,
            )
        authorization._revalidate_before_sqlite_statements()
        connection = sqlite3.connect(
            f"{Path(authorization.path).as_uri()}?mode=rw",
            timeout=busy_timeout_ms / 1_000,
            isolation_level=None,
            uri=True,
        )
        authorization._revalidate_before_sqlite_statements()
        applied = apply_capture_migrations(connection)
        _configure_initialized_store(connection, busy_timeout_ms=busy_timeout_ms)
        validate_capture_store_schema(connection)
        authorization._revalidate_mutable_sqlite()
        return CaptureMigrationReceipt(
            schema_version=LATEST_SCHEMA_VERSION,
            applied_versions=tuple(item.version for item in applied),
        )
    except (CaptureMigrationError, CaptureMigrationIntegrityError, CaptureSchemaTooNewError):
        raise
    except (
        OSError,
        SecureFileError,
        WindowsSecurityError,
        sqlite3.Error,
        TypeError,
        ValueError,
    ):
        raise CaptureMigrationError() from None
    finally:
        closed = True
        if connection is not None:
            try:
                connection.close()
            except sqlite3.Error:
                closed = False
        if closed and authorization is not None:
            authorization._cleanup_created_sqlite_sidecars()


__all__ = [
    "APPLICATION_ID",
    "LATEST_SCHEMA_VERSION",
    "CaptureMigration",
    "CaptureMigrationError",
    "CaptureMigrationIntegrityError",
    "CaptureMigrationReceipt",
    "CaptureSchemaTooNewError",
    "apply_capture_migrations",
    "discover_capture_migrations",
    "initialize_capture_store",
    "validate_capture_store_schema",
]
