"""Synchronous, integrity-checked SQLite storage for capture events."""

from __future__ import annotations

import base64
import hmac
import os
import re
import sqlite3
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PureWindowsPath
from threading import Lock
from types import TracebackType
from typing import Annotated, ClassVar, Final, Never, Self, cast

from pydantic import BaseModel, ConfigDict, Field

from saliencegate.capture.capabilities import (
    CaptureProfile,
    CompatibilityStatus,
    validate_capture_capability_binding,
)
from saliencegate.capture.health import (
    CaptureHealthCode,
    capture_health_identity_material,
    capture_health_integrity_material,
)
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.capture.migrations import (
    CaptureMigrationError,
    validate_capture_store_schema,
)
from saliencegate.capture.publication import verify_capture_intake_authentication
from saliencegate.capture.schema import (
    CaptureEvent,
    CaptureIntake,
    canonical_capture_event,
    canonical_capture_intake,
    load_capture_event,
)
from saliencegate.domain import canonical_json
from saliencegate.domain.records import ComponentIdentifier, Sha256Digest
from saliencegate.security import (
    InstallationKey,
    SecureFileError,
    StableFileAuthorization,
    claim_private_sqlite_location,
    inspect_private_file_location,
)
from saliencegate.security.windows import (
    NativeWindowsSecurityOperations,
    WindowsPathKind,
    WindowsSecurityError,
    WindowsSQLiteAuthorization,
    authorize_windows_private_path,
    authorize_windows_sqlite_path,
)

MAX_CAPTURE_EVENTS_PER_SESSION: Final = 1_000
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
_HOST_VERSION = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){1,3}$")


class CaptureStoreError(RuntimeError):
    """A content-free capture store failure."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture store operation failed")


class CaptureStoreBusyError(CaptureStoreError):
    """The bounded SQLite contention budget was exhausted."""

    __slots__ = ()

    def __init__(self) -> None:
        RuntimeError.__init__(self, "capture store is busy")


class CaptureStoreIntegrityError(CaptureStoreError):
    """Authenticated store state could not be verified."""

    __slots__ = ()

    def __init__(self) -> None:
        RuntimeError.__init__(self, "capture store integrity failed")


class CaptureStoreStateError(CaptureStoreError):
    """A requested lifecycle transition is not authorized."""

    __slots__ = ()

    def __init__(self) -> None:
        RuntimeError.__init__(self, "capture store state transition failed")


class CaptureStoreClosedError(CaptureStoreError):
    """An operation targeted a closed store."""

    __slots__ = ()

    def __init__(self) -> None:
        RuntimeError.__init__(self, "capture store is closed")


class CaptureStoreMode(StrEnum):
    HOOK = "hook"
    MAINTENANCE = "maintenance"


class CaptureAdmissionSource(StrEnum):
    DIRECT = "direct"
    SPOOL_DRAIN = "spool_drain"


class CaptureConnectionState(StrEnum):
    PENDING = "pending"
    ENABLED = "enabled"
    DRAINING = "draining"
    DISABLED = "disabled"
    DELETING = "deleting"


class CaptureSessionState(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    QUARANTINED = "quarantined"
    DELETING = "deleting"


class CaptureAppendDisposition(StrEnum):
    ADMITTED = "admitted"
    REPLAYED = "replayed"
    QUARANTINED = "quarantined"
    OVERFLOW = "overflow"


class _CaptureStoreModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<redacted>)"

    __str__ = __repr__


class CaptureConnectionRegistration(_CaptureStoreModel):
    connection_id: Annotated[ComponentIdentifier, Field(repr=False)]
    project_digest: Annotated[Sha256Digest, Field(repr=False)]
    profile_id: CaptureProfile
    capability_manifest_digest: Annotated[Sha256Digest, Field(repr=False)]
    host_version: Annotated[ComponentIdentifier, Field(repr=False)]
    state: CaptureConnectionState = CaptureConnectionState.PENDING


class CaptureConnectionTransition(_CaptureStoreModel):
    connection_id: Annotated[ComponentIdentifier, Field(repr=False)]
    previous_state: CaptureConnectionState
    state: CaptureConnectionState


class CaptureAppendReceipt(_CaptureStoreModel):
    disposition: CaptureAppendDisposition
    connection_id: Annotated[ComponentIdentifier, Field(repr=False)]
    session_id: Annotated[Sha256Digest, Field(repr=False)]
    producer_event_digest: Annotated[Sha256Digest, Field(repr=False)]
    receipt_ordinal: Annotated[int | None, Field(ge=1, le=MAX_CAPTURE_EVENTS_PER_SESSION)]
    previous_event_tag: Annotated[Sha256Digest | None, Field(repr=False)]
    event_tag: Annotated[Sha256Digest | None, Field(repr=False)]
    session_state: CaptureSessionState
    event_count: Annotated[int, Field(ge=0, le=MAX_CAPTURE_EVENTS_PER_SESSION)]


class CaptureSessionVerification(_CaptureStoreModel):
    connection_id: Annotated[ComponentIdentifier, Field(repr=False)]
    session_id: Annotated[Sha256Digest, Field(repr=False)]
    state: CaptureSessionState
    event_count: Annotated[int, Field(ge=0, le=MAX_CAPTURE_EVENTS_PER_SESSION)]
    last_receipt_ordinal: Annotated[
        int | None,
        Field(ge=1, le=MAX_CAPTURE_EVENTS_PER_SESSION),
    ]
    head_event_tag: Annotated[Sha256Digest | None, Field(repr=False)]
    head_tag: Annotated[Sha256Digest, Field(repr=False)]


class _CaptureStoreIntegrity:
    __slots__ = ("__key",)

    _DOMAINS: ClassVar[dict[str, bytes]] = {
        "connection": b"saliencegate:capture-store:connection:v1",
        "session": b"saliencegate:capture-store:session:v1",
        "event": b"saliencegate:capture-store:event:v1",
        "head": b"saliencegate:capture-store:head:v1",
        "health_id": b"saliencegate:capture-store:health-id:v1",
        "health": b"saliencegate:capture-store:health:v1",
        "health_set": b"saliencegate:capture-store:health-set:v1",
        "human_id": b"saliencegate:capture-store:human-id:v1",
        "feedback": b"saliencegate:capture-store:feedback:v1",
        "tombstone": b"saliencegate:capture-store:tombstone:v1",
    }

    def __init__(self, key: InstallationKey) -> None:
        if type(key) is not InstallationKey:
            raise CaptureStoreError()
        object.__setattr__(self, "_CaptureStoreIntegrity__key", key._copy())

    def __setattr__(self, _name: str, _value: object) -> Never:
        raise AttributeError("capture store integrity is immutable")

    def __repr__(self) -> str:
        return "_CaptureStoreIntegrity(<redacted>)"

    def tag(self, purpose: str, value: object) -> str:
        try:
            key = cast(
                InstallationKey,
                object.__getattribute__(self, "_CaptureStoreIntegrity__key"),
            )
            return key._hmac_sha256(
                canonical_json(value),
                domain=self._DOMAINS[purpose],
            )
        except Exception:
            raise CaptureStoreIntegrityError() from None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _connection_material(row: sqlite3.Row | dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "capture-connection-integrity/v1",
        "connection_id": row["connection_id"],
        "project_digest": row["project_digest"],
        "profile_id": row["profile_id"],
        "capability_manifest_digest": row["capability_manifest_digest"],
        "host_version": row["host_version"],
        "compatibility_status": row["compatibility_status"],
        "state": row["state"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _session_material(row: sqlite3.Row | dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "capture-session-integrity/v1",
        "connection_id": row["connection_id"],
        "session_id": row["session_id"],
        "human_id": row["human_id"],
        "state": row["state"],
        "event_count": row["event_count"],
        "coverage_degraded": row["coverage_degraded"],
        "unattributed_drop": row["unattributed_drop"],
        "health_marker_count": row["health_marker_count"],
        "health_set_digest": row["health_set_digest"],
        "opened_at": row["opened_at"],
        "updated_at": row["updated_at"],
        "closed_at": row["closed_at"],
    }


def _event_material(
    *,
    connection_id: str,
    session_id: str,
    receipt_ordinal: int,
    producer_event_digest: str,
    event_kind: str,
    previous_event_tag: str | None,
    admission_source: str,
    admitted_at: str,
    intake: CaptureIntake,
) -> dict[str, object]:
    return {
        "schema_version": "capture-event-integrity/v1",
        "connection_id": connection_id,
        "session_id": session_id,
        "receipt_ordinal": receipt_ordinal,
        "producer_event_digest": producer_event_digest,
        "event_kind": event_kind,
        "previous_event_tag": previous_event_tag,
        "admission_source": admission_source,
        "admitted_at": admitted_at,
        "intake": intake.model_dump(mode="json", warnings="error"),
    }


def _head_material(
    *,
    connection_id: str,
    session_id: str,
    receipt_count: int,
    head_event_tag: str | None,
) -> dict[str, object]:
    return {
        "schema_version": "capture-head-integrity/v1",
        "connection_id": connection_id,
        "session_id": session_id,
        "receipt_count": receipt_count,
        "head_event_tag": head_event_tag,
    }


def _feedback_material(row: sqlite3.Row | dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "capture-feedback-integrity/v1",
        "label_id": row["label_id"],
        "connection_id": row["connection_id"],
        "session_id": row["session_id"],
        "label": row["label"],
        "created_at": row["created_at"],
    }


def _tombstone_material(row: sqlite3.Row | dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "capture-tombstone-integrity/v1",
        "connection_id": row["connection_id"],
        "session_id": row["session_id"],
        "project_digest": row["project_digest"],
        "deleted_at": row["deleted_at"],
    }


_TRANSITIONS = {
    CaptureConnectionState.PENDING: frozenset({CaptureConnectionState.ENABLED}),
    CaptureConnectionState.ENABLED: frozenset({CaptureConnectionState.DRAINING}),
    CaptureConnectionState.DRAINING: frozenset({CaptureConnectionState.DISABLED}),
    CaptureConnectionState.DISABLED: frozenset({CaptureConnectionState.DELETING}),
    CaptureConnectionState.DELETING: frozenset(),
}


class CaptureStore:
    """One current capture database with cross-process SQLite transactions."""

    __slots__ = (
        "_authorization",
        "_closed",
        "_connection",
        "_context",
        "_fault_injector",
        "_integrity",
        "_lock",
        "_mode",
    )

    _authorization: StableFileAuthorization | WindowsSQLiteAuthorization
    _closed: bool
    _connection: sqlite3.Connection
    _context: CaptureDigestContext
    _fault_injector: Callable[[str], None] | None
    _integrity: _CaptureStoreIntegrity
    _lock: Lock
    _mode: CaptureStoreMode

    def __init__(self) -> None:
        raise CaptureStoreError()

    @classmethod
    def open(
        cls,
        path: str | os.PathLike[str],
        *,
        installation_key: InstallationKey,
        busy_timeout_ms: int = 5_000,
        mode: CaptureStoreMode = CaptureStoreMode.HOOK,
        _fault_injector: Callable[[str], None] | None = None,
    ) -> Self:
        """Open one existing current store without creating or migrating it."""

        if (
            type(installation_key) is not InstallationKey
            or type(busy_timeout_ms) is not int
            or not 1 <= busy_timeout_ms <= 60_000
            or type(mode) is not CaptureStoreMode
            or (_fault_injector is not None and not callable(_fault_injector))
        ):
            raise CaptureStoreError()
        authorization: StableFileAuthorization | WindowsSQLiteAuthorization | None = None
        connection: sqlite3.Connection | None = None
        opened = False
        try:
            raw_path = os.fspath(path)
            if type(raw_path) is not str or not raw_path:
                raise CaptureStoreError()
            database_path = Path(raw_path).absolute()
            if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
                operations = NativeWindowsSecurityOperations()
                windows_path = PureWindowsPath(str(database_path))
                if operations.inspect_path(windows_path) is None:
                    raise CaptureMigrationError()
                database_authorization = authorize_windows_private_path(
                    windows_path,
                    kind=WindowsPathKind.FILE,
                    operations=operations,
                )
                database_authorization.revalidate()
                preflight = sqlite3.connect(
                    f"{database_path.as_uri()}?mode=ro&immutable=1",
                    isolation_level=None,
                    uri=True,
                )
                try:
                    validate_capture_store_schema(preflight)
                finally:
                    preflight.close()
                database_authorization.revalidate()
                authorization = authorize_windows_sqlite_path(
                    windows_path,
                    operations=operations,
                    create_database=False,
                    database_authorization=database_authorization,
                )
            else:
                location = inspect_private_file_location(database_path)
                if not location.target_exists:
                    raise CaptureMigrationError()
                sidecars = tuple(
                    inspect_private_file_location(f"{database_path}{suffix}")
                    for suffix in _SQLITE_SIDECAR_SUFFIXES
                )
                location.revalidate()
                preflight = sqlite3.connect(
                    f"{database_path.as_uri()}?mode=ro&immutable=1",
                    isolation_level=None,
                    uri=True,
                )
                try:
                    validate_capture_store_schema(preflight)
                finally:
                    preflight.close()
                location.revalidate()
                authorization = claim_private_sqlite_location(
                    location,
                    sidecar_locations=sidecars,
                )
            authorization._revalidate_before_sqlite_statements()
            connection = sqlite3.connect(
                f"{Path(authorization.path).as_uri()}?mode=rw",
                timeout=busy_timeout_ms / 1_000,
                isolation_level=None,
                check_same_thread=False,
                uri=True,
            )
            connection.row_factory = sqlite3.Row
            authorization._revalidate_before_sqlite_statements()
            validate_capture_store_schema(connection)
            cls._configure_connection(connection, busy_timeout_ms=busy_timeout_ms)
            authorization.revalidate()
            instance = cls.__new__(cls)
            instance._authorization = authorization
            instance._connection = connection
            instance._context = CaptureDigestContext(installation_key)
            instance._integrity = _CaptureStoreIntegrity(installation_key)
            instance._lock = Lock()
            instance._mode = mode
            instance._closed = False
            instance._fault_injector = _fault_injector
            instance._verify_all_state()
            opened = True
            return instance
        except CaptureMigrationError:
            raise
        except CaptureStoreError:
            raise
        except (
            OSError,
            SecureFileError,
            WindowsSecurityError,
            sqlite3.Error,
            TypeError,
            ValueError,
        ):
            raise CaptureStoreError() from None
        finally:
            if connection is not None and not opened:
                with suppress(sqlite3.Error):
                    connection.close()
            if authorization is not None and not opened:
                authorization._cleanup_created_sqlite_sidecars()

    @staticmethod
    def _configure_connection(
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
            or connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1
            or connection.execute("PRAGMA busy_timeout").fetchone()[0] != busy_timeout_ms
            or connection.execute("PRAGMA trusted_schema").fetchone()[0] != 0
            or connection.execute("PRAGMA temp_store").fetchone()[0] != 2
            or connection.execute("PRAGMA synchronous").fetchone()[0] != 2
        ):
            raise CaptureStoreError()

    def __repr__(self) -> str:
        return f"CaptureStore(closed={self._closed})"

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _fault(self, stage: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(stage)

    def _ensure_open(self) -> None:
        if self._closed:
            raise CaptureStoreClosedError()

    def _revalidate_boundary(self) -> None:
        failed = False
        try:
            # DB/WAL/SHM contents can change under a live peer, so this pins their
            # exact security identities without requiring byte-level quiescence.
            self._authorization._revalidate_mutable_sqlite()
        except Exception:
            failed = True
        if failed:
            raise CaptureStoreIntegrityError()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            boundary_failed = False
            try:
                self._authorization.revalidate()
            except (SecureFileError, WindowsSecurityError):
                boundary_failed = True
            try:
                self._connection.close()
            except sqlite3.Error:
                raise CaptureStoreError() from None
            self._closed = True
            self._authorization._cleanup_created_sqlite_sidecars()
            if boundary_failed:
                raise CaptureStoreError()

    def _begin_immediate(self) -> None:
        self._fault("before_begin")
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._fault("after_begin")
        except sqlite3.Error as error:
            code = getattr(error, "sqlite_errorcode", None)
            if type(code) is int and code & 0xFF == sqlite3.SQLITE_BUSY:
                raise CaptureStoreBusyError() from None
            raise CaptureStoreError() from None

    def _rollback(self) -> None:
        with suppress(sqlite3.Error):
            self._connection.rollback()

    def _connection_row(self, connection_id: str) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT * FROM connections WHERE connection_id = ?",
            (connection_id,),
        ).fetchone()
        if row is None:
            raise CaptureStoreStateError()
        expected = self._integrity.tag("connection", _connection_material(row))
        if type(row["row_tag"]) is not str or not hmac.compare_digest(row["row_tag"], expected):
            raise CaptureStoreIntegrityError()
        return cast(sqlite3.Row, row)

    def register_connection(
        self,
        *,
        connection_id: str,
        project_digest: str,
        profile_id: CaptureProfile,
        capability_manifest_digest: str,
        host_version: str,
    ) -> CaptureConnectionRegistration:
        """Idempotently register one pending, manifest-bound connector."""

        self._ensure_open()
        try:
            registration = CaptureConnectionRegistration(
                connection_id=connection_id,
                project_digest=project_digest,
                profile_id=profile_id,
                capability_manifest_digest=capability_manifest_digest,
                host_version=host_version,
            )
            profile = validate_capture_capability_binding(
                registration.profile_id,
                registration.capability_manifest_digest,
            )
            if _HOST_VERSION.fullmatch(registration.host_version) is None:
                raise CaptureStoreStateError()
            compatibility = (
                CompatibilityStatus.VERIFIED
                if registration.host_version == profile.host_version
                else CompatibilityStatus.SCHEMA_COMPATIBLE_UNVERIFIED_VERSION
            )
            with self._lock:
                self._ensure_open()
                self._revalidate_boundary()
                self._begin_immediate()
                try:
                    existing = self._connection.execute(
                        "SELECT * FROM connections WHERE connection_id = ?",
                        (registration.connection_id,),
                    ).fetchone()
                    if existing is not None:
                        checked = self._connection_row(registration.connection_id)
                        if any(
                            checked[field] != expected
                            for field, expected in (
                                ("project_digest", registration.project_digest),
                                ("profile_id", registration.profile_id.value),
                                (
                                    "capability_manifest_digest",
                                    registration.capability_manifest_digest,
                                ),
                                ("host_version", registration.host_version),
                            )
                        ):
                            raise CaptureStoreStateError()
                        self._connection.rollback()
                        self._revalidate_boundary()
                        values = registration.model_dump(mode="python")
                        values["state"] = CaptureConnectionState(checked["state"])
                        return CaptureConnectionRegistration.model_validate(values)
                    timestamp = _now()
                    material: dict[str, object] = {
                        "connection_id": registration.connection_id,
                        "project_digest": registration.project_digest,
                        "profile_id": registration.profile_id.value,
                        "capability_manifest_digest": registration.capability_manifest_digest,
                        "host_version": registration.host_version,
                        "compatibility_status": compatibility.value,
                        "state": CaptureConnectionState.PENDING.value,
                        "created_at": timestamp,
                        "updated_at": timestamp,
                    }
                    row_tag = self._integrity.tag("connection", _connection_material(material))
                    self._connection.execute(
                        """
                        INSERT INTO connections(
                            connection_id, project_digest, profile_id,
                            capability_manifest_digest, host_version,
                            compatibility_status, state, created_at, updated_at, row_tag
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            *material.values(),
                            row_tag,
                        ),
                    )
                    self._connection.commit()
                except BaseException:
                    self._rollback()
                    raise
                self._revalidate_boundary()
                return registration
        except CaptureStoreError:
            raise
        except Exception:
            raise CaptureStoreError() from None

    def transition_connection(
        self,
        connection_id: str,
        *,
        expected_state: CaptureConnectionState,
        target_state: CaptureConnectionState,
    ) -> CaptureConnectionTransition:
        """Apply one authenticated compare-and-swap lifecycle transition."""

        self._ensure_open()
        if (
            type(connection_id) is not str
            or type(expected_state) is not CaptureConnectionState
            or type(target_state) is not CaptureConnectionState
            or target_state not in _TRANSITIONS[expected_state]
        ):
            raise CaptureStoreStateError()
        with self._lock:
            self._ensure_open()
            self._revalidate_boundary()
            self._begin_immediate()
            try:
                row = self._connection_row(connection_id)
                if row["state"] != expected_state.value:
                    raise CaptureStoreStateError()
                updated = dict(row)
                updated["state"] = target_state.value
                updated["updated_at"] = _now()
                row_tag = self._integrity.tag("connection", _connection_material(updated))
                result = self._connection.execute(
                    """
                    UPDATE connections SET state = ?, updated_at = ?, row_tag = ?
                    WHERE connection_id = ? AND state = ?
                    """,
                    (
                        target_state.value,
                        updated["updated_at"],
                        row_tag,
                        connection_id,
                        expected_state.value,
                    ),
                )
                if result.rowcount != 1:
                    raise CaptureStoreStateError()
                self._connection.commit()
            except BaseException:
                self._rollback()
                raise
            self._revalidate_boundary()
        return CaptureConnectionTransition(
            connection_id=connection_id,
            previous_state=expected_state,
            state=target_state,
        )

    def _session_row(self, connection_id: str, session_id: str) -> sqlite3.Row | None:
        row = self._connection.execute(
            """
            SELECT * FROM capture_sessions
            WHERE connection_id = ? AND session_id = ?
            """,
            (connection_id, session_id),
        ).fetchone()
        if row is None:
            return None
        expected = self._integrity.tag("session", _session_material(row))
        if type(row["row_tag"]) is not str or not hmac.compare_digest(row["row_tag"], expected):
            raise CaptureStoreIntegrityError()
        return cast(sqlite3.Row, row)

    def _head_row(self, connection_id: str, session_id: str) -> sqlite3.Row:
        row = self._connection.execute(
            """
            SELECT * FROM capture_heads
            WHERE connection_id = ? AND session_id = ?
            """,
            (connection_id, session_id),
        ).fetchone()
        if row is None:
            raise CaptureStoreIntegrityError()
        expected = self._integrity.tag(
            "head",
            _head_material(
                connection_id=connection_id,
                session_id=session_id,
                receipt_count=row["receipt_count"],
                head_event_tag=row["head_event_tag"],
            ),
        )
        if type(row["head_tag"]) is not str or not hmac.compare_digest(row["head_tag"], expected):
            raise CaptureStoreIntegrityError()
        return cast(sqlite3.Row, row)

    def _tombstone_row(self, connection_id: str, session_id: str) -> sqlite3.Row | None:
        row = self._connection.execute(
            """
            SELECT * FROM deleted_sessions
            WHERE connection_id = ? AND session_id = ?
            """,
            (connection_id, session_id),
        ).fetchone()
        if row is None:
            return None
        expected = self._integrity.tag("tombstone", _tombstone_material(row))
        if type(row["tombstone_tag"]) is not str or not hmac.compare_digest(
            row["tombstone_tag"], expected
        ):
            raise CaptureStoreIntegrityError()
        return cast(sqlite3.Row, row)

    def _load_verified_health_rows(
        self,
        connection_id: str,
        session_id: str,
    ) -> tuple[sqlite3.Row, ...]:
        rows = self._connection.execute(
            """
            SELECT * FROM capture_health
            WHERE connection_id = ? AND session_id = ?
            ORDER BY code, marker_id
            """,
            (connection_id, session_id),
        ).fetchall()
        verified: list[sqlite3.Row] = []
        for row in rows:
            try:
                code = CaptureHealthCode(row["code"])
                identity = capture_health_identity_material(
                    connection_id=connection_id,
                    session_id=session_id,
                    code=code,
                )
                material = capture_health_integrity_material(
                    marker_id=row["marker_id"],
                    connection_id=row["connection_id"],
                    session_id=row["session_id"],
                    code=code,
                    count=row["count"],
                    lower_bound=row["lower_bound"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                expected_marker_id = self._integrity.tag("health_id", identity)
                expected_row_tag = self._integrity.tag("health", material)
                if (
                    row["connection_id"] != connection_id
                    or row["session_id"] != session_id
                    or type(row["marker_id"]) is not str
                    or type(row["row_tag"]) is not str
                    or not hmac.compare_digest(row["marker_id"], expected_marker_id)
                    or not hmac.compare_digest(row["row_tag"], expected_row_tag)
                ):
                    raise CaptureStoreIntegrityError()
            except Exception:
                raise CaptureStoreIntegrityError() from None
            verified.append(cast(sqlite3.Row, row))
        return tuple(verified)

    def _health_set_digest(self, rows: tuple[sqlite3.Row, ...]) -> str:
        return self._integrity.tag(
            "health_set",
            {
                "schema_version": "capture-health-set-integrity/v1",
                "rows": [
                    {
                        "marker_id": row["marker_id"],
                        "row_tag": row["row_tag"],
                    }
                    for row in rows
                ],
            },
        )

    def _verify_health_set(
        self,
        connection_id: str,
        session_id: str,
        session: sqlite3.Row,
    ) -> tuple[sqlite3.Row, ...]:
        rows = self._load_verified_health_rows(connection_id, session_id)
        expected_digest = self._health_set_digest(rows)
        if (
            session["health_marker_count"] != len(rows)
            or type(session["health_set_digest"]) is not str
            or not hmac.compare_digest(session["health_set_digest"], expected_digest)
        ):
            raise CaptureStoreIntegrityError()
        return rows

    def _load_verified_event(self, row: sqlite3.Row) -> CaptureEvent:
        event: CaptureEvent | None = None
        try:
            blob = row["event_json"]
            if type(blob) is bytes:
                event = load_capture_event(blob)
            if event is None:
                raise CaptureStoreIntegrityError()
            intake = verify_capture_intake_authentication(event.intake, context=self._context)
            expected = self._integrity.tag(
                "event",
                _event_material(
                    connection_id=row["connection_id"],
                    session_id=row["session_id"],
                    receipt_ordinal=row["receipt_ordinal"],
                    producer_event_digest=row["producer_event_digest"],
                    event_kind=row["event_kind"],
                    previous_event_tag=row["previous_event_tag"],
                    admission_source=row["admission_source"],
                    admitted_at=row["admitted_at"],
                    intake=intake,
                ),
            )
            if (
                canonical_capture_event(event) != blob
                or row["connection_id"] != intake.connection_id
                or row["session_id"] != intake.session_id
                or row["receipt_ordinal"] != event.receipt_ordinal
                or row["producer_event_digest"] != intake.producer_event_digest
                or row["event_kind"] != intake.kind
                or row["previous_event_tag"] != event.previous_event_tag
                or row["event_tag"] != event.event_tag
                or type(row["admission_source"]) is not str
                or type(row["admitted_at"]) is not str
                or not hmac.compare_digest(event.event_tag, expected)
            ):
                raise CaptureStoreIntegrityError()
        except Exception:
            raise CaptureStoreIntegrityError() from None
        if event is None:
            raise CaptureStoreIntegrityError()
        return event

    def _verify_chain(
        self,
        connection_id: str,
        session_id: str,
        session: sqlite3.Row,
        head: sqlite3.Row,
    ) -> tuple[CaptureEvent, ...]:
        rows = self._connection.execute(
            """
            SELECT *
            FROM capture_events
            WHERE connection_id = ? AND session_id = ?
            ORDER BY receipt_ordinal
            """,
            (connection_id, session_id),
        ).fetchall()
        events: list[CaptureEvent] = []
        previous: str | None = None
        for ordinal, row in enumerate(rows, start=1):
            event = self._load_verified_event(row)
            if (
                row["receipt_ordinal"] != ordinal
                or event.receipt_ordinal != ordinal
                or event.intake.connection_id != connection_id
                or event.intake.session_id != session_id
                or row["previous_event_tag"] != previous
                or event.previous_event_tag != previous
                or row["event_tag"] != event.event_tag
            ):
                raise CaptureStoreIntegrityError()
            previous = event.event_tag
            events.append(event)
        if (
            len(events) != session["event_count"]
            or len(events) != head["receipt_count"]
            or previous != head["head_event_tag"]
        ):
            raise CaptureStoreIntegrityError()
        self._verify_health_set(connection_id, session_id, session)
        return tuple(events)

    def _verify_all_state(self) -> None:
        """Authenticate every currently persisted mutable row before use."""

        self._revalidate_boundary()
        try:
            self._connection.execute("BEGIN")
            connection_rows = self._connection.execute(
                "SELECT connection_id FROM connections ORDER BY connection_id"
            ).fetchall()
            for connection_row in connection_rows:
                self._connection_row(connection_row["connection_id"])
            session_rows = self._connection.execute(
                """
                SELECT connection_id, session_id
                FROM capture_sessions
                ORDER BY connection_id, session_id
                """
            ).fetchall()
            for session_identity in session_rows:
                connection_id = session_identity["connection_id"]
                session_id = session_identity["session_id"]
                session = self._session_row(connection_id, session_id)
                if session is None:
                    raise CaptureStoreIntegrityError()
                head = self._head_row(connection_id, session_id)
                self._verify_chain(connection_id, session_id, session, head)
            feedback_rows = self._connection.execute(
                "SELECT * FROM feedback_labels ORDER BY label_id"
            ).fetchall()
            for feedback in feedback_rows:
                expected = self._integrity.tag("feedback", _feedback_material(feedback))
                if type(feedback["row_tag"]) is not str or not hmac.compare_digest(
                    feedback["row_tag"], expected
                ):
                    raise CaptureStoreIntegrityError()
            tombstone_rows = self._connection.execute(
                """
                SELECT connection_id, session_id
                FROM deleted_sessions
                ORDER BY connection_id, session_id
                """
            ).fetchall()
            for tombstone in tombstone_rows:
                if self._tombstone_row(tombstone["connection_id"], tombstone["session_id"]) is None:
                    raise CaptureStoreIntegrityError()
            self._connection.commit()
        except CaptureStoreError:
            self._rollback()
            raise
        except Exception:
            self._rollback()
            raise CaptureStoreIntegrityError() from None
        self._revalidate_boundary()

    def _human_session_id(self, connection_id: str, session_id: str) -> str:
        digest = self._integrity.tag(
            "human_id",
            {
                "schema_version": "capture-human-session-id/v1",
                "connection_id": connection_id,
                "session_id": session_id,
            },
        )
        encoded = base64.b32encode(bytes.fromhex(digest)).decode("ascii").lower().rstrip("=")
        for length in range(12, len(encoded) + 1):
            candidate = encoded[:length]
            row = self._connection.execute(
                "SELECT connection_id, session_id FROM capture_sessions WHERE human_id = ?",
                (candidate,),
            ).fetchone()
            if row is None or (
                row["connection_id"] == connection_id and row["session_id"] == session_id
            ):
                return candidate
        raise CaptureStoreIntegrityError()

    def _create_session(self, intake: CaptureIntake) -> sqlite3.Row:
        timestamp = _now()
        material: dict[str, object] = {
            "connection_id": intake.connection_id,
            "session_id": intake.session_id,
            "human_id": self._human_session_id(intake.connection_id, intake.session_id),
            "state": CaptureSessionState.OPEN.value,
            "event_count": 0,
            "coverage_degraded": int(intake.capture_disposition != "captured"),
            "unattributed_drop": 0,
            "health_marker_count": 0,
            "health_set_digest": self._health_set_digest(()),
            "opened_at": timestamp,
            "updated_at": timestamp,
            "closed_at": None,
        }
        row_tag = self._integrity.tag("session", _session_material(material))
        self._connection.execute(
            """
            INSERT INTO capture_sessions(
                connection_id, session_id, human_id, state, event_count,
                coverage_degraded, unattributed_drop, health_marker_count,
                health_set_digest, opened_at, updated_at, closed_at, row_tag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (*material.values(), row_tag),
        )
        head_tag = self._integrity.tag(
            "head",
            _head_material(
                connection_id=intake.connection_id,
                session_id=intake.session_id,
                receipt_count=0,
                head_event_tag=None,
            ),
        )
        self._connection.execute(
            """
            INSERT INTO capture_heads(
                connection_id, session_id, receipt_count, head_event_tag, head_tag
            ) VALUES (?, ?, 0, NULL, ?)
            """,
            (intake.connection_id, intake.session_id, head_tag),
        )
        row = self._session_row(intake.connection_id, intake.session_id)
        if row is None:
            raise CaptureStoreIntegrityError()
        return row

    def _update_session(
        self,
        row: sqlite3.Row,
        *,
        state: CaptureSessionState,
        event_count: int,
        coverage_degraded: bool,
    ) -> sqlite3.Row:
        material = dict(row)
        material["state"] = state.value
        material["event_count"] = event_count
        material["coverage_degraded"] = int(coverage_degraded)
        material["updated_at"] = _now()
        material["closed_at"] = (
            material["updated_at"] if state is CaptureSessionState.CLOSED else None
        )
        row_tag = self._integrity.tag("session", _session_material(material))
        result = self._connection.execute(
            """
            UPDATE capture_sessions
            SET state = ?, event_count = ?, coverage_degraded = ?,
                updated_at = ?, closed_at = ?, row_tag = ?
            WHERE connection_id = ? AND session_id = ?
            """,
            (
                material["state"],
                material["event_count"],
                material["coverage_degraded"],
                material["updated_at"],
                material["closed_at"],
                row_tag,
                material["connection_id"],
                material["session_id"],
            ),
        )
        if result.rowcount != 1:
            raise CaptureStoreIntegrityError()
        updated = self._session_row(material["connection_id"], material["session_id"])
        if updated is None:
            raise CaptureStoreIntegrityError()
        return updated

    def _record_health(
        self,
        *,
        connection_id: str,
        session_id: str,
        code: CaptureHealthCode,
    ) -> None:
        session = self._session_row(connection_id, session_id)
        if session is None:
            raise CaptureStoreIntegrityError()
        current_rows = self._verify_health_set(connection_id, session_id, session)
        identity = capture_health_identity_material(
            connection_id=connection_id,
            session_id=session_id,
            code=code,
        )
        marker_id = self._integrity.tag("health_id", identity)
        existing = next((row for row in current_rows if row["marker_id"] == marker_id), None)
        timestamp = _now()
        count = 1
        created_at = timestamp
        if existing is not None:
            material = capture_health_integrity_material(
                marker_id=existing["marker_id"],
                connection_id=existing["connection_id"],
                session_id=existing["session_id"],
                code=CaptureHealthCode(existing["code"]),
                count=existing["count"],
                lower_bound=existing["lower_bound"],
                created_at=existing["created_at"],
                updated_at=existing["updated_at"],
            )
            if type(existing["row_tag"]) is not str or not hmac.compare_digest(
                existing["row_tag"],
                self._integrity.tag("health", material),
            ):
                raise CaptureStoreIntegrityError()
            count = existing["count"] + 1
            created_at = existing["created_at"]
        material = capture_health_integrity_material(
            marker_id=marker_id,
            connection_id=connection_id,
            session_id=session_id,
            code=code,
            count=count,
            lower_bound=0,
            created_at=created_at,
            updated_at=timestamp,
        )
        row_tag = self._integrity.tag("health", material)
        self._connection.execute(
            """
            INSERT INTO capture_health(
                marker_id, connection_id, session_id, code, count,
                lower_bound, created_at, updated_at, row_tag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(marker_id) DO UPDATE SET
                count = excluded.count,
                updated_at = excluded.updated_at,
                row_tag = excluded.row_tag
            """,
            (
                marker_id,
                connection_id,
                session_id,
                code.value,
                count,
                0,
                created_at,
                timestamp,
                row_tag,
            ),
        )
        updated_rows = self._load_verified_health_rows(connection_id, session_id)
        session_material = dict(session)
        session_material["health_marker_count"] = len(updated_rows)
        session_material["health_set_digest"] = self._health_set_digest(updated_rows)
        session_material["updated_at"] = timestamp
        session_row_tag = self._integrity.tag(
            "session",
            _session_material(session_material),
        )
        result = self._connection.execute(
            """
            UPDATE capture_sessions
            SET health_marker_count = ?, health_set_digest = ?,
                updated_at = ?, row_tag = ?
            WHERE connection_id = ? AND session_id = ?
            """,
            (
                session_material["health_marker_count"],
                session_material["health_set_digest"],
                session_material["updated_at"],
                session_row_tag,
                connection_id,
                session_id,
            ),
        )
        if result.rowcount != 1:
            raise CaptureStoreIntegrityError()
        updated_session = self._session_row(connection_id, session_id)
        if updated_session is None:
            raise CaptureStoreIntegrityError()
        self._verify_health_set(connection_id, session_id, updated_session)

    def _replay_receipt(self, event: CaptureEvent, session: sqlite3.Row) -> CaptureAppendReceipt:
        return CaptureAppendReceipt(
            disposition=CaptureAppendDisposition.REPLAYED,
            connection_id=event.intake.connection_id,
            session_id=event.intake.session_id,
            producer_event_digest=event.intake.producer_event_digest,
            receipt_ordinal=event.receipt_ordinal,
            previous_event_tag=event.previous_event_tag,
            event_tag=event.event_tag,
            session_state=CaptureSessionState(session["state"]),
            event_count=session["event_count"],
        )

    def append(
        self,
        intake: CaptureIntake,
        *,
        source: CaptureAdmissionSource = CaptureAdmissionSource.DIRECT,
    ) -> CaptureAppendReceipt:
        """Atomically authenticate, deduplicate, chain, and admit one intake."""

        self._ensure_open()
        if type(source) is not CaptureAdmissionSource:
            raise CaptureStoreError()
        try:
            authenticated = verify_capture_intake_authentication(
                intake,
                context=self._context,
            )
            canonical_intake = canonical_capture_intake(authenticated)
        except Exception:
            raise CaptureStoreIntegrityError() from None
        with self._lock:
            self._ensure_open()
            self._revalidate_boundary()
            self._begin_immediate()
            try:
                connection = self._connection_row(authenticated.connection_id)
                if (
                    connection["profile_id"] != authenticated.adapter_profile
                    or connection["capability_manifest_digest"]
                    != authenticated.capability_manifest_digest
                ):
                    raise CaptureStoreStateError()
                allowed_states = (
                    {CaptureConnectionState.ENABLED.value}
                    if source is CaptureAdmissionSource.DIRECT
                    else {
                        CaptureConnectionState.ENABLED.value,
                        CaptureConnectionState.DRAINING.value,
                        CaptureConnectionState.DISABLED.value,
                    }
                )
                existing_row = self._connection.execute(
                    """
                    SELECT * FROM capture_events
                    WHERE connection_id = ? AND producer_event_digest = ?
                    """,
                    (
                        authenticated.connection_id,
                        authenticated.producer_event_digest,
                    ),
                ).fetchone()
                if existing_row is not None:
                    existing = self._load_verified_event(existing_row)
                    existing_session = self._session_row(
                        existing.intake.connection_id,
                        existing.intake.session_id,
                    )
                    if existing_session is None:
                        raise CaptureStoreIntegrityError()
                    existing_head = self._head_row(
                        existing.intake.connection_id,
                        existing.intake.session_id,
                    )
                    self._verify_chain(
                        existing.intake.connection_id,
                        existing.intake.session_id,
                        existing_session,
                        existing_head,
                    )
                    if canonical_capture_intake(existing.intake) == canonical_intake:
                        self._connection.rollback()
                        self._revalidate_boundary()
                        return self._replay_receipt(existing, existing_session)
                    if connection["state"] not in allowed_states:
                        raise CaptureStoreStateError()
                    if (
                        self._tombstone_row(
                            authenticated.connection_id,
                            authenticated.session_id,
                        )
                        is not None
                    ):
                        raise CaptureStoreStateError()
                    affected = {(authenticated.connection_id, authenticated.session_id)}
                    affected.add((existing.intake.connection_id, existing.intake.session_id))
                    for affected_connection, affected_session_id in affected:
                        if (
                            self._tombstone_row(
                                affected_connection,
                                affected_session_id,
                            )
                            is not None
                        ):
                            raise CaptureStoreStateError()
                        session = self._session_row(affected_connection, affected_session_id)
                        if session is None:
                            if (affected_connection, affected_session_id) != (
                                authenticated.connection_id,
                                authenticated.session_id,
                            ):
                                raise CaptureStoreIntegrityError()
                            session = self._create_session(authenticated)
                        elif (
                            affected_connection,
                            affected_session_id,
                        ) != (
                            existing.intake.connection_id,
                            existing.intake.session_id,
                        ):
                            affected_head = self._head_row(
                                affected_connection,
                                affected_session_id,
                            )
                            self._verify_chain(
                                affected_connection,
                                affected_session_id,
                                session,
                                affected_head,
                            )
                        self._update_session(
                            session,
                            state=CaptureSessionState.QUARANTINED,
                            event_count=session["event_count"],
                            coverage_degraded=True,
                        )
                        self._record_health(
                            connection_id=affected_connection,
                            session_id=affected_session_id,
                            code=CaptureHealthCode.PRODUCER_COLLISION,
                        )
                    incoming_session = self._session_row(
                        authenticated.connection_id,
                        authenticated.session_id,
                    )
                    event_count = 0 if incoming_session is None else incoming_session["event_count"]
                    self._connection.commit()
                    self._revalidate_boundary()
                    return CaptureAppendReceipt(
                        disposition=CaptureAppendDisposition.QUARANTINED,
                        connection_id=authenticated.connection_id,
                        session_id=authenticated.session_id,
                        producer_event_digest=authenticated.producer_event_digest,
                        receipt_ordinal=None,
                        previous_event_tag=None,
                        event_tag=None,
                        session_state=CaptureSessionState.QUARANTINED,
                        event_count=event_count,
                    )
                if connection["state"] not in allowed_states:
                    raise CaptureStoreStateError()
                tombstone = self._tombstone_row(
                    authenticated.connection_id,
                    authenticated.session_id,
                )
                if tombstone is not None:
                    raise CaptureStoreStateError()
                session = self._session_row(
                    authenticated.connection_id,
                    authenticated.session_id,
                )
                if session is None:
                    if authenticated.kind != "session_started":
                        raise CaptureStoreStateError()
                    session = self._create_session(authenticated)
                elif (
                    session["state"] != CaptureSessionState.OPEN.value
                    or authenticated.kind == "session_started"
                ):
                    raise CaptureStoreStateError()
                head = self._head_row(authenticated.connection_id, authenticated.session_id)
                self._verify_chain(
                    authenticated.connection_id,
                    authenticated.session_id,
                    session,
                    head,
                )
                self._fault("after_session_insert_or_load")
                if head["receipt_count"] >= MAX_CAPTURE_EVENTS_PER_SESSION:
                    session = self._update_session(
                        session,
                        state=CaptureSessionState.QUARANTINED,
                        event_count=session["event_count"],
                        coverage_degraded=True,
                    )
                    self._record_health(
                        connection_id=authenticated.connection_id,
                        session_id=authenticated.session_id,
                        code=CaptureHealthCode.SESSION_OVERFLOW,
                    )
                    self._connection.commit()
                    self._revalidate_boundary()
                    return CaptureAppendReceipt(
                        disposition=CaptureAppendDisposition.OVERFLOW,
                        connection_id=authenticated.connection_id,
                        session_id=authenticated.session_id,
                        producer_event_digest=authenticated.producer_event_digest,
                        receipt_ordinal=None,
                        previous_event_tag=None,
                        event_tag=None,
                        session_state=CaptureSessionState.QUARANTINED,
                        event_count=session["event_count"],
                    )
                ordinal = head["receipt_count"] + 1
                previous = head["head_event_tag"]
                admitted_at = _now()
                event_tag = self._integrity.tag(
                    "event",
                    _event_material(
                        connection_id=authenticated.connection_id,
                        session_id=authenticated.session_id,
                        receipt_ordinal=ordinal,
                        producer_event_digest=authenticated.producer_event_digest,
                        event_kind=authenticated.kind,
                        previous_event_tag=previous,
                        admission_source=source.value,
                        admitted_at=admitted_at,
                        intake=authenticated,
                    ),
                )
                event = CaptureEvent(
                    receipt_ordinal=ordinal,
                    previous_event_tag=previous,
                    event_tag=event_tag,
                    intake=authenticated,
                )
                event_bytes = canonical_capture_event(event)
                self._connection.execute(
                    """
                    INSERT INTO capture_events(
                        connection_id, session_id, receipt_ordinal,
                        producer_event_digest, event_kind, event_json,
                        previous_event_tag, event_tag, admission_source, admitted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        authenticated.connection_id,
                        authenticated.session_id,
                        ordinal,
                        authenticated.producer_event_digest,
                        authenticated.kind,
                        event_bytes,
                        previous,
                        event_tag,
                        source.value,
                        admitted_at,
                    ),
                )
                self._fault("after_event_insert")
                head_tag = self._integrity.tag(
                    "head",
                    _head_material(
                        connection_id=authenticated.connection_id,
                        session_id=authenticated.session_id,
                        receipt_count=ordinal,
                        head_event_tag=event_tag,
                    ),
                )
                self._connection.execute(
                    """
                    UPDATE capture_heads
                    SET receipt_count = ?, head_event_tag = ?, head_tag = ?
                    WHERE connection_id = ? AND session_id = ?
                    """,
                    (
                        ordinal,
                        event_tag,
                        head_tag,
                        authenticated.connection_id,
                        authenticated.session_id,
                    ),
                )
                self._fault("after_head_write")
                next_state = (
                    CaptureSessionState.CLOSED
                    if authenticated.kind == "session_finished"
                    else CaptureSessionState.OPEN
                )
                session = self._update_session(
                    session,
                    state=next_state,
                    event_count=ordinal,
                    coverage_degraded=(
                        bool(session["coverage_degraded"])
                        or authenticated.capture_disposition != "captured"
                    ),
                )
                self._fault("after_session_health_write")
                self._fault("before_commit")
                self._connection.commit()
                self._fault("after_commit")
            except BaseException:
                self._rollback()
                raise
            self._revalidate_boundary()
            return CaptureAppendReceipt(
                disposition=CaptureAppendDisposition.ADMITTED,
                connection_id=authenticated.connection_id,
                session_id=authenticated.session_id,
                producer_event_digest=authenticated.producer_event_digest,
                receipt_ordinal=ordinal,
                previous_event_tag=previous,
                event_tag=event_tag,
                session_state=CaptureSessionState(session["state"]),
                event_count=session["event_count"],
            )

    def verify_session(
        self,
        connection_id: str,
        session_id: str,
    ) -> CaptureSessionVerification:
        """Verify one complete bounded chain without repairing any state."""

        self._ensure_open()
        if type(connection_id) is not str or type(session_id) is not str:
            raise CaptureStoreError()
        with self._lock:
            self._ensure_open()
            self._revalidate_boundary()
            try:
                self._connection.execute("BEGIN")
                self._connection_row(connection_id)
                session = self._session_row(connection_id, session_id)
                if session is None:
                    raise CaptureStoreStateError()
                head = self._head_row(connection_id, session_id)
                events = self._verify_chain(connection_id, session_id, session, head)
                self._connection.commit()
            except BaseException:
                self._rollback()
                raise
            self._revalidate_boundary()
        return CaptureSessionVerification(
            connection_id=connection_id,
            session_id=session_id,
            state=CaptureSessionState(session["state"]),
            event_count=session["event_count"],
            last_receipt_ordinal=None if not events else events[-1].receipt_ordinal,
            head_event_tag=head["head_event_tag"],
            head_tag=head["head_tag"],
        )


__all__ = [
    "MAX_CAPTURE_EVENTS_PER_SESSION",
    "CaptureAdmissionSource",
    "CaptureAppendDisposition",
    "CaptureAppendReceipt",
    "CaptureConnectionRegistration",
    "CaptureConnectionState",
    "CaptureConnectionTransition",
    "CaptureSessionState",
    "CaptureSessionVerification",
    "CaptureStore",
    "CaptureStoreBusyError",
    "CaptureStoreClosedError",
    "CaptureStoreError",
    "CaptureStoreIntegrityError",
    "CaptureStoreMode",
    "CaptureStoreStateError",
]
