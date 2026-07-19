from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from typing import Final

APPLICATION_ID: Final = 0x534C4754  # "SLGT"
LATEST_SCHEMA_VERSION: Final = 2

_MIGRATION_NAME = re.compile(r"^(?P<version>[0-9]{4})_(?P<name>[a-z][a-z0-9_]*)\.sql$")


class MigrationError(RuntimeError):
    """A migration failed without exposing database contents."""


class SchemaTooNewError(MigrationError):
    def __init__(self, found: int, supported: int) -> None:
        self.found = found
        self.supported = supported
        super().__init__("database schema is newer than this SalienceGate build")


class MigrationIntegrityError(MigrationError):
    def __init__(self) -> None:
        super().__init__("database migration history failed integrity validation")


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sql: str
    checksum: str


def migration_checksum(sql: str) -> str:
    if type(sql) is not str:
        raise TypeError("migration SQL must be an exact string")
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def parse_statements(sql: str) -> tuple[str, ...]:
    """Split a migration without the implicit commits caused by ``executescript``."""

    if type(sql) is not str:
        raise TypeError("migration SQL must be an exact string")
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
    if "".join(pending).strip():
        raise MigrationError("migration resource contains incomplete SQL")
    if not statements:
        raise MigrationError("migration resource contains no SQL statements")
    return tuple(statements)


def discover_migrations() -> tuple[Migration, ...]:
    root = resources.files(__package__)
    migrations: list[Migration] = []
    for resource in sorted(root.iterdir(), key=lambda item: item.name):
        match = _MIGRATION_NAME.fullmatch(resource.name)
        if match is None or not resource.is_file():
            continue
        try:
            raw = resource.read_bytes()
            if raw.startswith(b"\xef\xbb\xbf"):
                raise MigrationError("migration resources cannot contain a UTF-8 BOM")
            sql = raw.decode("utf-8")
        except UnicodeError:
            raise MigrationError("migration resource is not valid UTF-8") from None
        parse_statements(sql)
        migrations.append(
            Migration(
                version=int(match.group("version")),
                name=match.group("name"),
                sql=sql,
                checksum=migration_checksum(sql),
            )
        )
    versions = tuple(migration.version for migration in migrations)
    expected = tuple(range(1, len(migrations) + 1))
    if not migrations or versions != expected or versions[-1] != LATEST_SCHEMA_VERSION:
        raise MigrationError("migration resources are missing or non-contiguous")
    return tuple(migrations)


def _pragma_integer(connection: sqlite3.Connection, name: str) -> int:
    row = connection.execute(f"PRAGMA {name}").fetchone()
    if row is None or type(row[0]) is not int:
        raise MigrationIntegrityError()
    return row[0]


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _has_user_schema_objects(connection: sqlite3.Connection) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM sqlite_schema
        WHERE name NOT GLOB 'sqlite_*'
        LIMIT 1
        """
    ).fetchone()
    return row is not None


def _verify_applied_history(
    connection: sqlite3.Connection,
    migrations: tuple[Migration, ...],
    current_version: int,
) -> None:
    if current_version == 0:
        return
    if not _table_exists(connection, "schema_migrations"):
        raise MigrationIntegrityError()
    rows = connection.execute(
        "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
    ).fetchall()
    expected = migrations[:current_version]
    if len(rows) != len(expected):
        raise MigrationIntegrityError()
    for row, migration in zip(rows, expected, strict=True):
        if tuple(row) != (migration.version, migration.name, migration.checksum):
            raise MigrationIntegrityError()


def apply_migrations(connection: sqlite3.Connection) -> tuple[Migration, ...]:
    """Apply packaged migrations once under an exclusive SQLite transaction."""

    if not isinstance(connection, sqlite3.Connection):
        raise TypeError("connection must be a sqlite3.Connection")
    if connection.in_transaction:
        raise MigrationError("migrations require an idle SQLite connection")
    migrations = discover_migrations()
    connection.execute("PRAGMA foreign_keys = ON")
    applied: list[Migration] = []
    try:
        connection.execute("BEGIN EXCLUSIVE")
        current_version = _pragma_integer(connection, "user_version")
        if current_version > LATEST_SCHEMA_VERSION:
            raise SchemaTooNewError(current_version, LATEST_SCHEMA_VERSION)
        application_id = _pragma_integer(connection, "application_id")
        if current_version > 0 and application_id != APPLICATION_ID:
            raise MigrationIntegrityError()
        if current_version == 0 and application_id not in (0, APPLICATION_ID):
            raise MigrationIntegrityError()
        if current_version == 0 and _has_user_schema_objects(connection):
            raise MigrationIntegrityError()
        _verify_applied_history(connection, migrations, current_version)

        for migration in migrations[current_version:]:
            for statement in parse_statements(migration.sql):
                connection.execute(statement)
            connection.execute(
                """
                INSERT INTO schema_migrations(version, name, checksum, applied_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    migration.version,
                    migration.name,
                    migration.checksum,
                    datetime.now(UTC).isoformat(),
                ),
            )
            connection.execute(f"PRAGMA user_version = {migration.version}")
            applied.append(migration)

        if _pragma_integer(connection, "application_id") != APPLICATION_ID:
            raise MigrationIntegrityError()
        connection.commit()
    except MigrationError:
        connection.rollback()
        raise
    except sqlite3.Error:
        connection.rollback()
        raise MigrationError("database migration failed") from None
    return tuple(applied)


__all__ = [
    "APPLICATION_ID",
    "LATEST_SCHEMA_VERSION",
    "Migration",
    "MigrationError",
    "MigrationIntegrityError",
    "SchemaTooNewError",
    "apply_migrations",
    "discover_migrations",
    "migration_checksum",
    "parse_statements",
]
