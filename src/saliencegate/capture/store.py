"""Synchronous, integrity-checked SQLite storage for capture events."""

from __future__ import annotations

import base64
import hmac
import os
import re
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path, PureWindowsPath
from threading import Lock
from types import TracebackType
from typing import TYPE_CHECKING, Annotated, ClassVar, Final, Never, Self, cast

from pydantic import BaseModel, ConfigDict, Field

import saliencegate.security.files as security_files
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
from saliencegate.capture.locations import _capture_spool_boundary_digest
from saliencegate.capture.migrations import (
    CaptureMigrationError,
    _validate_capture_store_schema_metadata,
    validate_capture_store_schema,
)
from saliencegate.capture.publication import verify_capture_intake_authentication
from saliencegate.capture.schema import (
    CaptureEvent,
    CaptureIntake,
    _read_canonical_capture_event_document,
    canonical_capture_event,
    canonical_capture_intake,
    load_capture_event,
)
from saliencegate.capture.transport import (
    MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION,
    CaptureTransportChunk,
    CaptureTransportDisposition,
    CaptureTransportReceipt,
    validate_capture_transport_chunk,
)
from saliencegate.domain import canonical_json
from saliencegate.domain.primitives import ComponentIdentifier, Sha256Digest
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

if TYPE_CHECKING:
    from saliencegate.capture.connections import (
        CaptureConnectionSummary,
        CaptureSessionInventory,
        CaptureSessionSummary,
    )
    from saliencegate.capture.delete import (
        CaptureProjectDeleteReceipt,
        CaptureSessionDeleteReceipt,
    )
    from saliencegate.capture.feedback import (
        CaptureFeedbackLabel,
        CaptureFeedbackReceipt,
        CaptureFeedbackRecord,
        CaptureFeedbackRevision,
    )
    from saliencegate.capture.sessions import CaptureSessionSnapshot

MAX_CAPTURE_EVENTS_PER_SESSION: Final = 1_000
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
_HOST_VERSION = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){1,3}$")
_PROJECT_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_COMPONENT_IDENTIFIER = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._:/+\-]{0,255}$")
_HUMAN_SESSION_ID = re.compile(r"^[a-z2-7]{12,52}$")
_MAX_CAPTURE_QUERY_RESULTS: Final = 1_000
_TRANSPORT_PROFILES: Final = frozenset(
    (
        CaptureProfile.OPENCODE_PLUGIN_V1.value,
        CaptureProfile.PI_EXTENSION_V1.value,
    )
)


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


@dataclass(frozen=True, slots=True, repr=False)
class _CaptureHookConnection:
    """Authenticated connection fields required by a provider hook."""

    connection_id: str
    project_digest: str
    profile_id: CaptureProfile
    capability_manifest_digest: str
    host_version: str
    state: CaptureConnectionState

    def __post_init__(self) -> None:
        if (
            type(self.connection_id) is not str
            or _COMPONENT_IDENTIFIER.fullmatch(self.connection_id) is None
            or type(self.project_digest) is not str
            or _PROJECT_DIGEST.fullmatch(self.project_digest) is None
            or type(self.profile_id) is not CaptureProfile
            or type(self.capability_manifest_digest) is not str
            or _PROJECT_DIGEST.fullmatch(self.capability_manifest_digest) is None
            or type(self.host_version) is not str
            or _HOST_VERSION.fullmatch(self.host_version) is None
            or type(self.state) is not CaptureConnectionState
        ):
            raise CaptureStoreIntegrityError()

    def __repr__(self) -> str:
        return "_CaptureHookConnection(<redacted>)"


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
        "transport_receipt": b"saliencegate:capture-store:transport-receipt:v1",
        "transport_head": b"saliencegate:capture-store:transport-head:v1",
        "transport_intake_set": b"saliencegate:capture-store:transport-intake-set:v1",
        "health_id": b"saliencegate:capture-store:health-id:v1",
        "health": b"saliencegate:capture-store:health:v1",
        "health_set": b"saliencegate:capture-store:health-set:v1",
        "human_id": b"saliencegate:capture-store:human-id:v1",
        "feedback": b"saliencegate:capture-store:feedback:v1",
        "feedback_id": b"saliencegate:capture-store:feedback-id:v1",
        "feedback_anchor_id": b"saliencegate:capture-store:feedback-anchor-id:v1",
        "feedback_record": b"saliencegate:capture:feedback-record:v1",
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


def _stored_timestamp(value: object) -> datetime:
    if type(value) is not str:
        raise CaptureStoreIntegrityError()
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise CaptureStoreIntegrityError() from None


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
    material: dict[str, object] = {
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
    if row["transport_required"] == 1:
        material["schema_version"] = "capture-session-integrity/v2"
        material["transport_required"] = 1
        material["transport_head_tag"] = row["transport_head_tag"]
    return material


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
    intake: CaptureIntake | Mapping[str, object],
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
        "intake": (
            intake
            if isinstance(intake, Mapping)
            else intake.model_dump(mode="json", warnings="error")
        ),
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


def _transport_receipt_material(
    row: sqlite3.Row | dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "capture-transport-receipt-integrity/v1",
        "connection_id": row["connection_id"],
        "session_id": row["session_id"],
        "transport_ordinal": row["transport_ordinal"],
        "batch_ref": row["batch_ref"],
        "chunk_index": row["chunk_index"],
        "chunk_count": row["chunk_count"],
        "chunk_digest": row["chunk_digest"],
        "intake_count": row["intake_count"],
        "intake_set_digest": row["intake_set_digest"],
        "post_event_count": row["post_event_count"],
        "post_head_event_tag": row["post_head_event_tag"],
        "previous_receipt_tag": row["previous_receipt_tag"],
        "admitted_at": row["admitted_at"],
    }


def _transport_head_material(
    *,
    connection_id: str,
    session_id: str,
    receipt_count: int,
    head_receipt_tag: str | None,
) -> dict[str, object]:
    return {
        "schema_version": "capture-transport-head-integrity/v1",
        "connection_id": connection_id,
        "session_id": session_id,
        "receipt_count": receipt_count,
        "head_receipt_tag": head_receipt_tag,
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


def _feedback_identity_material(
    *,
    connection_id: str,
    session_id: str,
    revision: int,
    previous_label_id: str | None,
    label: str,
    created_at: str,
) -> dict[str, object]:
    return {
        "schema_version": "capture-feedback-id/v1",
        "connection_id": connection_id,
        "session_id": session_id,
        "revision": revision,
        "previous_label_id": previous_label_id,
        "label": label,
        "created_at": created_at,
    }


def _feedback_anchor_identity_material(
    *,
    connection_id: str,
    session_id: str,
    revision_count: int,
    head_label_id: str,
    label: str,
    created_at: str,
) -> dict[str, object]:
    return {
        "schema_version": "capture-feedback-anchor-id/v1",
        "connection_id": connection_id,
        "session_id": session_id,
        "revision_count": revision_count,
        "head_label_id": head_label_id,
        "label": label,
        "created_at": created_at,
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
    CaptureConnectionState.DISABLED: frozenset(
        {
            CaptureConnectionState.ENABLED,
            CaptureConnectionState.DELETING,
        }
    ),
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
        "_verified_append_heads",
        "_verified_data_version",
    )

    _authorization: StableFileAuthorization | WindowsSQLiteAuthorization
    _closed: bool
    _connection: sqlite3.Connection
    _context: CaptureDigestContext
    _fault_injector: Callable[[str], None] | None
    _integrity: _CaptureStoreIntegrity
    _lock: Lock
    _mode: CaptureStoreMode
    _verified_append_heads: dict[tuple[str, str], tuple[int, str | None]]
    _verified_data_version: int

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
        validate_schema = (
            _validate_capture_store_schema_metadata
            if mode is CaptureStoreMode.HOOK
            else validate_capture_store_schema
        )
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
                    validate_schema(preflight)
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
                    validate_schema(preflight)
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
            validate_schema(connection)
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
            instance._verified_append_heads = {}
            instance._verified_data_version = instance._database_data_version()
            # Hook open defers a whole-database audit, but the first mutation of each
            # selected session authenticates its complete retained chain.  An
            # unchanged cached head permits bounded verification on later appends;
            # any peer commit clears that cache.  Maintenance callers retain the
            # fail-closed whole-database audit before use.
            if mode is CaptureStoreMode.MAINTENANCE:
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

    @classmethod
    def audit_read_only(
        cls,
        path: str | os.PathLike[str],
        *,
        installation_key: InstallationKey,
    ) -> None:
        """Authenticate a quiescent store through an immutable SQLite snapshot."""

        if type(installation_key) is not InstallationKey:
            raise CaptureStoreError()
        connection: sqlite3.Connection | None = None
        authorization: StableFileAuthorization | WindowsSQLiteAuthorization | None = None
        sidecars: tuple[StableFileAuthorization, ...] = ()
        windows_operations: NativeWindowsSecurityOperations | None = None
        windows_path: PureWindowsPath | None = None
        try:
            raw_path = os.fspath(path)
            if type(raw_path) is not str or not raw_path:
                raise CaptureStoreError()
            database_path = Path(raw_path).absolute()
            if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
                windows_operations = NativeWindowsSecurityOperations()
                windows_path = PureWindowsPath(str(database_path))
                if windows_operations.inspect_path(windows_path) is None:
                    raise CaptureStoreError()
                authorization = cast(
                    StableFileAuthorization | WindowsSQLiteAuthorization,
                    authorize_windows_private_path(
                        windows_path,
                        kind=WindowsPathKind.FILE,
                        operations=windows_operations,
                    ),
                )
                for suffix in ("-wal", "-journal"):
                    sidecar_path = PureWindowsPath(f"{windows_path}{suffix}")
                    if (
                        windows_operations.inspect_path(sidecar_path) is not None
                        and Path(str(sidecar_path)).stat().st_size != 0
                    ):
                        raise CaptureStoreError()
            else:
                authorization = inspect_private_file_location(database_path)
                if not authorization.target_exists:
                    raise CaptureStoreError()
                sidecars = tuple(
                    inspect_private_file_location(f"{database_path}{suffix}")
                    for suffix in _SQLITE_SIDECAR_SUFFIXES
                )
                if any(
                    item.target_exists
                    and item._target_complete_identity is not None
                    and item._target_complete_identity.size != 0
                    for suffix, item in zip(_SQLITE_SIDECAR_SUFFIXES, sidecars, strict=True)
                    if suffix in ("-wal", "-journal")
                ):
                    raise CaptureStoreError()
            authorization.revalidate()
            connection = sqlite3.connect(
                f"{database_path.as_uri()}?mode=ro&immutable=1",
                isolation_level=None,
                uri=True,
            )
            connection.row_factory = sqlite3.Row
            validate_capture_store_schema(connection)
            quick_check = connection.execute("PRAGMA quick_check").fetchone()
            if quick_check is None or tuple(quick_check) != ("ok",):
                raise CaptureStoreIntegrityError()
            instance = cls.__new__(cls)
            instance._authorization = authorization
            instance._connection = connection
            instance._context = CaptureDigestContext(installation_key)
            instance._integrity = _CaptureStoreIntegrity(installation_key)
            instance._lock = Lock()
            instance._mode = CaptureStoreMode.MAINTENANCE
            instance._closed = False
            instance._fault_injector = None
            instance._verified_append_heads = {}
            instance._verified_data_version = instance._database_data_version()
            instance._verify_all_state(immutable=True)
            connection.close()
            connection = None
            authorization.revalidate()
            for sidecar in sidecars:
                sidecar.revalidate()
            if (
                windows_operations is not None
                and windows_path is not None
                and any(
                    windows_operations.inspect_path(PureWindowsPath(f"{windows_path}{suffix}"))
                    is not None
                    and Path(f"{windows_path}{suffix}").stat().st_size != 0
                    for suffix in ("-wal", "-journal")
                )
            ):
                raise CaptureStoreError()
        except CaptureStoreError:
            raise
        except CaptureMigrationError:
            raise CaptureStoreIntegrityError() from None
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
            if connection is not None:
                with suppress(sqlite3.Error):
                    connection.close()

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
        journal = connection.execute("PRAGMA journal_mode").fetchone()
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

    def _database_data_version(self) -> int:
        try:
            row = self._connection.execute("PRAGMA data_version").fetchone()
        except sqlite3.Error:
            raise CaptureStoreIntegrityError() from None
        if row is None or type(row[0]) is not int or row[0] < 1:
            raise CaptureStoreIntegrityError()
        return row[0]

    def _require_maintenance(self) -> None:
        self._ensure_open()
        if self._mode is not CaptureStoreMode.MAINTENANCE:
            raise CaptureStoreStateError()

    @staticmethod
    def _validate_project_digest(project_digest: object) -> str:
        if type(project_digest) is not str or _PROJECT_DIGEST.fullmatch(project_digest) is None:
            raise CaptureStoreStateError()
        return project_digest

    @staticmethod
    def _validate_human_id(human_id: object) -> str:
        if type(human_id) is not str or _HUMAN_SESSION_ID.fullmatch(human_id) is None:
            raise CaptureStoreStateError()
        return human_id

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
                # Peer writers may legitimately change SQLite-managed bytes while
                # this connection is closing.  Pin ownership, type, link count,
                # inode and sidecar security identities without requiring a
                # byte-quiescent stat window.
                self._authorization._revalidate_mutable_sqlite()
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

        self._require_maintenance()
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
                self._require_maintenance()
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

        self._require_maintenance()
        if (
            type(connection_id) is not str
            or type(expected_state) is not CaptureConnectionState
            or type(target_state) is not CaptureConnectionState
            or target_state not in _TRANSITIONS[expected_state]
        ):
            raise CaptureStoreStateError()
        with self._lock:
            self._require_maintenance()
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

    def _connection_summary(self, row: sqlite3.Row) -> CaptureConnectionSummary:
        from saliencegate.capture.connections import CaptureConnectionSummary

        profile_id = CaptureProfile(row["profile_id"])
        validate_capture_capability_binding(
            profile_id,
            row["capability_manifest_digest"],
        )
        return CaptureConnectionSummary(
            connection_id=row["connection_id"],
            project_digest=row["project_digest"],
            profile_id=profile_id,
            capability_manifest_digest=row["capability_manifest_digest"],
            host_version=row["host_version"],
            compatibility_status=CompatibilityStatus(row["compatibility_status"]),
            state=CaptureConnectionState(row["state"]),
            created_at=_stored_timestamp(row["created_at"]),
            updated_at=_stored_timestamp(row["updated_at"]),
        )

    def _hook_connection(self, row: sqlite3.Row) -> _CaptureHookConnection:
        profile_id = CaptureProfile(row["profile_id"])
        validate_capture_capability_binding(
            profile_id,
            row["capability_manifest_digest"],
        )
        return _CaptureHookConnection(
            connection_id=row["connection_id"],
            project_digest=row["project_digest"],
            profile_id=profile_id,
            capability_manifest_digest=row["capability_manifest_digest"],
            host_version=row["host_version"],
            state=CaptureConnectionState(row["state"]),
        )

    def _session_summary(
        self,
        connection: sqlite3.Row,
        session: sqlite3.Row,
    ) -> CaptureSessionSummary:
        from saliencegate.capture.connections import CaptureSessionSummary

        if (
            type(session["coverage_degraded"]) is not int
            or session["coverage_degraded"] not in (0, 1)
            or type(session["unattributed_drop"]) is not int
            or session["unattributed_drop"] not in (0, 1)
        ):
            raise CaptureStoreIntegrityError()
        head = self._head_row(session["connection_id"], session["session_id"])
        self._verify_chain(
            session["connection_id"],
            session["session_id"],
            session,
            head,
        )
        profile_id = CaptureProfile(connection["profile_id"])
        validate_capture_capability_binding(
            profile_id,
            connection["capability_manifest_digest"],
        )
        return CaptureSessionSummary(
            connection_id=session["connection_id"],
            project_digest=connection["project_digest"],
            profile_id=profile_id,
            session_id=session["session_id"],
            human_id=session["human_id"],
            state=CaptureSessionState(session["state"]),
            event_count=session["event_count"],
            coverage_degraded=bool(session["coverage_degraded"]),
            unattributed_drop=bool(session["unattributed_drop"]),
            opened_at=_stored_timestamp(session["opened_at"]),
            updated_at=_stored_timestamp(session["updated_at"]),
            closed_at=(
                None if session["closed_at"] is None else _stored_timestamp(session["closed_at"])
            ),
        )

    def list_connections(
        self,
        *,
        project_digest: str | None = None,
        profile_id: CaptureProfile | None = None,
    ) -> tuple[CaptureConnectionSummary, ...]:
        """Return authenticated connection summaries in stable identity order."""

        self._require_maintenance()
        if project_digest is not None:
            project_digest = self._validate_project_digest(project_digest)
        if profile_id is not None and type(profile_id) is not CaptureProfile:
            raise CaptureStoreStateError()
        with self._lock:
            self._require_maintenance()
            self._revalidate_boundary()
            try:
                self._connection.execute("BEGIN")
                identities = self._connection.execute(
                    """
                    SELECT connection_id
                    FROM connections
                    WHERE (? IS NULL OR project_digest = ?)
                      AND (? IS NULL OR profile_id = ?)
                    ORDER BY connection_id
                    """,
                    (
                        project_digest,
                        project_digest,
                        None if profile_id is None else profile_id.value,
                        None if profile_id is None else profile_id.value,
                    ),
                ).fetchall()
                summaries = tuple(
                    self._connection_summary(self._connection_row(identity["connection_id"]))
                    for identity in identities
                )
                self._connection.commit()
            except CaptureStoreError:
                self._rollback()
                raise
            except Exception:
                self._rollback()
                raise CaptureStoreIntegrityError() from None
            except BaseException:
                self._rollback()
                raise
            self._revalidate_boundary()
            return summaries

    def get_connection(self, connection_id: str) -> CaptureConnectionSummary:
        """Return one authenticated connection without scanning unrelated history."""

        self._ensure_open()
        if type(connection_id) is not str:
            raise CaptureStoreError()
        with self._lock:
            self._ensure_open()
            self._revalidate_boundary()
            try:
                self._connection.execute("BEGIN")
                summary = self._connection_summary(self._connection_row(connection_id))
                self._connection.commit()
            except CaptureStoreError:
                self._rollback()
                raise
            except Exception:
                self._rollback()
                raise CaptureStoreIntegrityError() from None
            except BaseException:
                self._rollback()
                raise
            self._revalidate_boundary()
            return summary

    def _get_hook_connection(self, connection_id: str) -> _CaptureHookConnection:
        """Return only the authenticated connection fields used by a hook."""

        self._ensure_open()
        if type(connection_id) is not str:
            raise CaptureStoreError()
        with self._lock:
            self._ensure_open()
            self._revalidate_boundary()
            try:
                self._connection.execute("BEGIN")
                connection = self._hook_connection(self._connection_row(connection_id))
                self._connection.commit()
            except CaptureStoreError:
                self._rollback()
                raise
            except Exception:
                self._rollback()
                raise CaptureStoreIntegrityError() from None
            except BaseException:
                self._rollback()
                raise
            self._revalidate_boundary()
            return connection

    def list_sessions(
        self,
        *,
        project_digest: str | None = None,
        profile_id: CaptureProfile | None = None,
        state: CaptureSessionState | None = None,
        limit: int = 100,
    ) -> tuple[CaptureSessionSummary, ...]:
        """Return fully verified sessions in deterministic latest-first order."""

        self._require_maintenance()
        if project_digest is not None:
            project_digest = self._validate_project_digest(project_digest)
        if (
            (profile_id is not None and type(profile_id) is not CaptureProfile)
            or (state is not None and type(state) is not CaptureSessionState)
            or type(limit) is not int
            or not 1 <= limit <= _MAX_CAPTURE_QUERY_RESULTS
        ):
            raise CaptureStoreStateError()
        with self._lock:
            self._require_maintenance()
            self._revalidate_boundary()
            try:
                self._connection.execute("BEGIN")
                identities = self._connection.execute(
                    """
                    SELECT sessions.connection_id, sessions.session_id
                    FROM capture_sessions AS sessions
                    JOIN connections
                      ON connections.connection_id = sessions.connection_id
                    WHERE (? IS NULL OR connections.project_digest = ?)
                      AND (? IS NULL OR connections.profile_id = ?)
                      AND (? IS NULL OR sessions.state = ?)
                    ORDER BY sessions.updated_at DESC, sessions.human_id ASC
                    LIMIT ?
                    """,
                    (
                        project_digest,
                        project_digest,
                        None if profile_id is None else profile_id.value,
                        None if profile_id is None else profile_id.value,
                        None if state is None else state.value,
                        None if state is None else state.value,
                        limit,
                    ),
                ).fetchall()
                summaries: list[CaptureSessionSummary] = []
                for identity in identities:
                    connection = self._connection_row(identity["connection_id"])
                    session = self._session_row(
                        identity["connection_id"],
                        identity["session_id"],
                    )
                    if session is None:
                        raise CaptureStoreIntegrityError()
                    summaries.append(self._session_summary(connection, session))
                result = tuple(summaries)
                self._connection.commit()
            except CaptureStoreError:
                self._rollback()
                raise
            except Exception:
                self._rollback()
                raise CaptureStoreIntegrityError() from None
            except BaseException:
                self._rollback()
                raise
            self._revalidate_boundary()
            return result

    def session_inventory(
        self,
        *,
        project_digest: str | None = None,
        profile_id: CaptureProfile | None = None,
    ) -> CaptureSessionInventory:
        """Verify and summarize every matching session without a display limit."""

        from saliencegate.capture.connections import CaptureSessionInventory

        self._require_maintenance()
        if project_digest is not None:
            project_digest = self._validate_project_digest(project_digest)
        if profile_id is not None and type(profile_id) is not CaptureProfile:
            raise CaptureStoreStateError()
        with self._lock:
            self._require_maintenance()
            self._revalidate_boundary()
            try:
                self._connection.execute("BEGIN")
                identities = self._connection.execute(
                    """
                    SELECT sessions.connection_id, sessions.session_id
                    FROM capture_sessions AS sessions
                    JOIN connections
                      ON connections.connection_id = sessions.connection_id
                    WHERE (? IS NULL OR connections.project_digest = ?)
                      AND (? IS NULL OR connections.profile_id = ?)
                    ORDER BY sessions.connection_id, sessions.session_id
                    """,
                    (
                        project_digest,
                        project_digest,
                        None if profile_id is None else profile_id.value,
                        None if profile_id is None else profile_id.value,
                    ),
                ).fetchall()
                summaries: list[CaptureSessionSummary] = []
                for identity in identities:
                    connection = self._connection_row(identity["connection_id"])
                    session = self._session_row(
                        identity["connection_id"],
                        identity["session_id"],
                    )
                    if session is None:
                        raise CaptureStoreIntegrityError()
                    summaries.append(self._session_summary(connection, session))
                oldest = (
                    min(summaries, key=lambda item: (item.opened_at, item.human_id))
                    if summaries
                    else None
                )
                result = CaptureSessionInventory(
                    session_count=len(summaries),
                    quarantined_sessions=sum(
                        item.state is CaptureSessionState.QUARANTINED for item in summaries
                    ),
                    degraded_sessions=sum(item.coverage_degraded for item in summaries),
                    oldest_session=None if oldest is None else oldest.human_id,
                )
                self._connection.commit()
            except CaptureStoreError:
                self._rollback()
                raise
            except Exception:
                self._rollback()
                raise CaptureStoreIntegrityError() from None
            except BaseException:
                self._rollback()
                raise
            self._revalidate_boundary()
            return result

    def session_by_human_id(self, human_id: str) -> CaptureSessionSummary:
        """Resolve and fully verify one live human-addressed session."""

        self._require_maintenance()
        human_id = self._validate_human_id(human_id)
        with self._lock:
            self._require_maintenance()
            self._revalidate_boundary()
            try:
                self._connection.execute("BEGIN")
                identity = self._connection.execute(
                    """
                    SELECT connection_id, session_id
                    FROM capture_sessions
                    WHERE human_id = ?
                    """,
                    (human_id,),
                ).fetchone()
                if identity is None:
                    raise CaptureStoreStateError()
                connection = self._connection_row(identity["connection_id"])
                session = self._session_row(
                    identity["connection_id"],
                    identity["session_id"],
                )
                if session is None:
                    raise CaptureStoreIntegrityError()
                result = self._session_summary(connection, session)
                self._connection.commit()
            except CaptureStoreError:
                self._rollback()
                raise
            except Exception:
                self._rollback()
                raise CaptureStoreIntegrityError() from None
            except BaseException:
                self._rollback()
                raise
            self._revalidate_boundary()
            return result

    def session_belongs_to_project(
        self,
        human_id: str,
        project_digest: str,
        *,
        include_deleted: bool = False,
    ) -> bool:
        """Authenticate one live or deleted session's project ownership."""

        self._require_maintenance()
        human_id = self._validate_human_id(human_id)
        project_digest = self._validate_project_digest(project_digest)
        if type(include_deleted) is not bool:
            raise CaptureStoreStateError()
        with self._lock:
            self._require_maintenance()
            self._revalidate_boundary()
            try:
                self._connection.execute("BEGIN")
                identity = self._connection.execute(
                    """
                    SELECT connection_id, session_id
                    FROM capture_sessions
                    WHERE human_id = ?
                    """,
                    (human_id,),
                ).fetchone()
                if identity is not None:
                    connection = self._connection_row(identity["connection_id"])
                    session = self._session_row(
                        identity["connection_id"],
                        identity["session_id"],
                    )
                    if session is None:
                        raise CaptureStoreIntegrityError()
                    summary = self._session_summary(connection, session)
                    result = hmac.compare_digest(summary.project_digest, project_digest)
                elif include_deleted:
                    tombstone = self._deleted_session_for_human_id(human_id)
                    result = tombstone is not None and hmac.compare_digest(
                        tombstone["project_digest"],
                        project_digest,
                    )
                else:
                    result = False
                self._connection.commit()
            except CaptureStoreError:
                self._rollback()
                raise
            except sqlite3.Error:
                self._rollback()
                raise CaptureStoreError() from None
            except Exception:
                self._rollback()
                raise CaptureStoreIntegrityError() from None
            except BaseException:
                self._rollback()
                raise
            self._revalidate_boundary()
            return result

    def record_feedback(
        self,
        human_id: str,
        label: CaptureFeedbackLabel,
        *,
        project_digest: str,
    ) -> CaptureFeedbackReceipt:
        """Append one authenticated label revision to an explicitly bound session."""

        from saliencegate.capture.feedback import (
            MAX_CAPTURE_FEEDBACK_REVISIONS_PER_SESSION,
            CaptureFeedbackLabel,
            CaptureFeedbackReceipt,
            CaptureFeedbackWriteDisposition,
        )

        self._require_maintenance()
        human_id = self._validate_human_id(human_id)
        project_digest = self._validate_project_digest(project_digest)
        if type(label) is not CaptureFeedbackLabel:
            raise CaptureStoreStateError()
        with self._lock:
            self._require_maintenance()
            self._revalidate_boundary()
            self._begin_immediate()
            try:
                identity = self._connection.execute(
                    """
                    SELECT connection_id, session_id
                    FROM capture_sessions
                    WHERE human_id = ?
                    """,
                    (human_id,),
                ).fetchone()
                if identity is None:
                    raise CaptureStoreStateError()
                connection_id = identity["connection_id"]
                session_id = identity["session_id"]
                connection = self._connection_row(connection_id)
                if not hmac.compare_digest(connection["project_digest"], project_digest):
                    raise CaptureStoreStateError()
                session = self._session_row(connection_id, session_id)
                if session is None:
                    raise CaptureStoreIntegrityError()
                if (
                    connection["state"] == CaptureConnectionState.DELETING.value
                    or session["state"] != CaptureSessionState.CLOSED.value
                ):
                    raise CaptureStoreStateError()
                self._session_summary(connection, session)
                rows = self._verified_feedback_rows(connection_id, session_id)
                if rows and rows[-1]["label"] == label.value:
                    result = CaptureFeedbackReceipt(
                        session_id=human_id,
                        label=label,
                        disposition=CaptureFeedbackWriteDisposition.UNCHANGED,
                        revision_count=len(rows),
                        labeled_at=_stored_timestamp(rows[-1]["created_at"]),
                    )
                    self._connection.commit()
                else:
                    if len(rows) >= MAX_CAPTURE_FEEDBACK_REVISIONS_PER_SESSION:
                        raise CaptureStoreStateError()
                    revision = len(rows) + 1
                    created_at = _now()
                    timestamp = _stored_timestamp(created_at)
                    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
                        raise CaptureStoreIntegrityError()
                    timestamp = timestamp.astimezone(UTC)
                    created_at = timestamp.isoformat()
                    previous_label_id = None if not rows else rows[-1]["label_id"]
                    if not rows:
                        closed_at = _stored_timestamp(session["closed_at"])
                        if timestamp < closed_at:
                            timestamp = closed_at
                            created_at = timestamp.isoformat()
                    else:
                        previous_anchor = self._connection.execute(
                            """
                            SELECT * FROM feedback_labels
                            WHERE connection_id = ? AND session_id = ?
                            ORDER BY created_at DESC, label_id DESC
                            LIMIT 1
                            """,
                            (connection_id, session_id),
                        ).fetchone()
                        if previous_anchor is None:
                            raise CaptureStoreIntegrityError()
                        previous_timestamp = _stored_timestamp(previous_anchor["created_at"])
                        if timestamp <= previous_timestamp:
                            timestamp = previous_timestamp + timedelta(microseconds=1)
                            created_at = timestamp.isoformat()
                        deleted_anchor = self._connection.execute(
                            "DELETE FROM feedback_labels WHERE label_id = ?",
                            (previous_anchor["label_id"],),
                        )
                        if deleted_anchor.rowcount != 1:
                            raise CaptureStoreIntegrityError()
                    label_id = self._integrity.tag(
                        "feedback_id",
                        _feedback_identity_material(
                            connection_id=connection_id,
                            session_id=session_id,
                            revision=revision,
                            previous_label_id=previous_label_id,
                            label=label.value,
                            created_at=created_at,
                        ),
                    )
                    material: dict[str, object] = {
                        "label_id": label_id,
                        "connection_id": connection_id,
                        "session_id": session_id,
                        "label": label.value,
                        "created_at": created_at,
                    }
                    row_tag = self._integrity.tag("feedback", _feedback_material(material))
                    self._connection.execute(
                        """
                        INSERT INTO feedback_labels(
                            label_id, connection_id, session_id, label, created_at, row_tag
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            label_id,
                            connection_id,
                            session_id,
                            label.value,
                            created_at,
                            row_tag,
                        ),
                    )
                    anchor_timestamp = timestamp + timedelta(microseconds=1)
                    anchor_created_at = anchor_timestamp.isoformat()
                    anchor_id = self._integrity.tag(
                        "feedback_anchor_id",
                        _feedback_anchor_identity_material(
                            connection_id=connection_id,
                            session_id=session_id,
                            revision_count=revision,
                            head_label_id=label_id,
                            label=label.value,
                            created_at=anchor_created_at,
                        ),
                    )
                    anchor_material: dict[str, object] = {
                        "label_id": anchor_id,
                        "connection_id": connection_id,
                        "session_id": session_id,
                        "label": label.value,
                        "created_at": anchor_created_at,
                    }
                    anchor_tag = self._integrity.tag(
                        "feedback",
                        _feedback_material(anchor_material),
                    )
                    self._connection.execute(
                        """
                        INSERT INTO feedback_labels(
                            label_id, connection_id, session_id, label, created_at, row_tag
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            anchor_id,
                            connection_id,
                            session_id,
                            label.value,
                            anchor_created_at,
                            anchor_tag,
                        ),
                    )
                    disposition = (
                        CaptureFeedbackWriteDisposition.RECORDED
                        if revision == 1
                        else CaptureFeedbackWriteDisposition.CHANGED
                    )
                    result = CaptureFeedbackReceipt(
                        session_id=human_id,
                        label=label,
                        disposition=disposition,
                        revision_count=revision,
                        labeled_at=timestamp,
                    )
                    self._fault("feedback_before_commit")
                    self._connection.commit()
                    self._fault("feedback_after_commit")
            except CaptureStoreError:
                self._rollback()
                raise
            except sqlite3.Error:
                self._rollback()
                raise CaptureStoreError() from None
            except Exception:
                self._rollback()
                raise CaptureStoreIntegrityError() from None
            except BaseException:
                self._rollback()
                raise
            self._revalidate_boundary()
            return result

    def feedback_history(
        self,
        human_id: str,
        *,
        project_digest: str,
    ) -> tuple[CaptureFeedbackRevision, ...]:
        """Return every authenticated label revision in deterministic order."""

        from saliencegate.capture.feedback import (
            CaptureFeedbackLabel,
            CaptureFeedbackRevision,
        )

        self._require_maintenance()
        human_id = self._validate_human_id(human_id)
        project_digest = self._validate_project_digest(project_digest)
        with self._lock:
            self._require_maintenance()
            self._revalidate_boundary()
            self._begin_immediate()
            try:
                identity = self._connection.execute(
                    """
                    SELECT connection_id, session_id
                    FROM capture_sessions
                    WHERE human_id = ?
                    """,
                    (human_id,),
                ).fetchone()
                if identity is None:
                    raise CaptureStoreStateError()
                connection_id = identity["connection_id"]
                session_id = identity["session_id"]
                connection = self._connection_row(connection_id)
                if not hmac.compare_digest(connection["project_digest"], project_digest):
                    raise CaptureStoreStateError()
                session = self._session_row(connection_id, session_id)
                if session is None:
                    raise CaptureStoreIntegrityError()
                if (
                    connection["state"] == CaptureConnectionState.DELETING.value
                    or session["state"] == CaptureSessionState.DELETING.value
                ):
                    raise CaptureStoreStateError()
                self._session_summary(connection, session)
                rows = self._verified_feedback_rows(connection_id, session_id)
                result = tuple(
                    CaptureFeedbackRevision(
                        session_id=human_id,
                        label=CaptureFeedbackLabel(row["label"]),
                        revision=revision,
                        created_at=_stored_timestamp(row["created_at"]),
                    )
                    for revision, row in enumerate(rows, start=1)
                )
                self._connection.commit()
            except CaptureStoreError:
                self._rollback()
                raise
            except sqlite3.Error:
                self._rollback()
                raise CaptureStoreError() from None
            except Exception:
                self._rollback()
                raise CaptureStoreIntegrityError() from None
            except BaseException:
                self._rollback()
                raise
            self._revalidate_boundary()
            return result

    def list_feedback(
        self,
        *,
        project_digest: str | None = None,
        label_freeze: datetime | None = None,
        limit: int = 1_000,
    ) -> tuple[CaptureFeedbackRecord, ...]:
        """Return authenticated labels strictly before an optional UTC freeze."""

        from saliencegate.capture.feedback import (
            CaptureFeedbackRecord,
            _feedback_record_material,
        )

        self._require_maintenance()
        if project_digest is not None:
            project_digest = self._validate_project_digest(project_digest)
        if (
            (
                label_freeze is not None
                and (
                    type(label_freeze) is not datetime
                    or label_freeze.tzinfo is None
                    or label_freeze.utcoffset() != timedelta(0)
                    or label_freeze != label_freeze.astimezone(UTC)
                )
            )
            or type(limit) is not int
            or not 1 <= limit <= _MAX_CAPTURE_QUERY_RESULTS
        ):
            raise CaptureStoreStateError()
        freeze_text = None if label_freeze is None else label_freeze.isoformat()
        with self._lock:
            self._require_maintenance()
            self._revalidate_boundary()
            self._begin_immediate()
            try:
                identities = self._connection.execute(
                    """
                    SELECT sessions.connection_id, sessions.session_id
                    FROM capture_sessions AS sessions
                    JOIN connections
                      ON connections.connection_id = sessions.connection_id
                    WHERE (? IS NULL OR connections.project_digest = ?)
                      AND EXISTS (
                          SELECT 1
                          FROM feedback_labels AS feedback
                          WHERE feedback.connection_id = sessions.connection_id
                            AND feedback.session_id = sessions.session_id
                            AND (? IS NULL OR feedback.created_at < ?)
                      )
                    ORDER BY sessions.human_id ASC
                    LIMIT ?
                    """,
                    (
                        project_digest,
                        project_digest,
                        freeze_text,
                        freeze_text,
                        limit + 1,
                    ),
                ).fetchall()
                if len(identities) > limit:
                    raise CaptureStoreStateError()
                records: list[CaptureFeedbackRecord] = []
                for identity in identities:
                    connection = self._connection_row(identity["connection_id"])
                    if project_digest is not None and not hmac.compare_digest(
                        connection["project_digest"],
                        project_digest,
                    ):
                        raise CaptureStoreIntegrityError()
                    session = self._session_row(
                        identity["connection_id"],
                        identity["session_id"],
                    )
                    if session is None:
                        raise CaptureStoreIntegrityError()
                    if (
                        connection["state"] == CaptureConnectionState.DELETING.value
                        or session["state"] == CaptureSessionState.DELETING.value
                    ):
                        raise CaptureStoreStateError()
                    summary = self._session_summary(connection, session)
                    rows = self._verified_feedback_rows(
                        identity["connection_id"],
                        identity["session_id"],
                    )
                    if not rows:
                        raise CaptureStoreIntegrityError()
                    selected_rows = (
                        rows
                        if label_freeze is None
                        else tuple(
                            row
                            for row in rows
                            if _stored_timestamp(row["created_at"]) < label_freeze
                        )
                    )
                    if not selected_rows:
                        raise CaptureStoreIntegrityError()
                    latest = selected_rows[-1]
                    unsigned = CaptureFeedbackRecord.model_validate_json(
                        canonical_json(
                            {
                                "schema_version": "capture-feedback-record/v1",
                                "project_digest": summary.project_digest,
                                "profile_id": summary.profile_id.value,
                                "session_id": summary.session_id,
                                "human_id": summary.human_id,
                                "label": latest["label"],
                                "revision_count": len(selected_rows),
                                "labeled_at": latest["created_at"],
                                "record_tag": "0" * 64,
                            }
                        )
                    )
                    body = _feedback_record_material(unsigned)
                    records.append(
                        CaptureFeedbackRecord.model_validate_json(
                            canonical_json(
                                {
                                    **body,
                                    "record_tag": self._integrity.tag(
                                        "feedback_record",
                                        body,
                                    ),
                                }
                            )
                        )
                    )
                result = tuple(records)
                self._connection.commit()
            except CaptureStoreError:
                self._rollback()
                raise
            except sqlite3.Error:
                self._rollback()
                raise CaptureStoreError() from None
            except Exception:
                self._rollback()
                raise CaptureStoreIntegrityError() from None
            except BaseException:
                self._rollback()
                raise
            self._revalidate_boundary()
            return result

    def latest_session(
        self,
        *,
        project_digest: str,
        profile_id: CaptureProfile | None = None,
    ) -> CaptureSessionSummary:
        """Return the deterministic latest session within one explicit project."""

        self._require_maintenance()
        project_digest = self._validate_project_digest(project_digest)
        if profile_id is not None and type(profile_id) is not CaptureProfile:
            raise CaptureStoreStateError()
        sessions = self.list_sessions(
            project_digest=project_digest,
            profile_id=profile_id,
            limit=1,
        )
        if not sessions:
            raise CaptureStoreStateError()
        return sessions[0]

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
        if (
            type(row["transport_required"]) is not int
            or row["transport_required"] not in (0, 1)
            or (row["transport_required"] == 0 and row["transport_head_tag"] is not None)
            or (
                row["transport_required"] == 1
                and (
                    type(row["transport_head_tag"]) is not str
                    or _PROJECT_DIGEST.fullmatch(row["transport_head_tag"]) is None
                )
            )
        ):
            raise CaptureStoreIntegrityError()
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

    def _create_transport_head(
        self,
        connection_id: str,
        session_id: str,
        *,
        profile_id: str,
    ) -> None:
        if profile_id not in _TRANSPORT_PROFILES:
            return
        head_tag = self._transport_head_tag(
            connection_id,
            session_id,
            receipt_count=0,
            head_receipt_tag=None,
        )
        self._connection.execute(
            """
            INSERT INTO capture_transport_heads(
                connection_id, session_id, receipt_count,
                head_receipt_tag, head_tag
            ) VALUES (?, ?, 0, NULL, ?)
            """,
            (connection_id, session_id, head_tag),
        )

    def _transport_head_tag(
        self,
        connection_id: str,
        session_id: str,
        *,
        receipt_count: int,
        head_receipt_tag: str | None,
    ) -> str:
        return self._integrity.tag(
            "transport_head",
            _transport_head_material(
                connection_id=connection_id,
                session_id=session_id,
                receipt_count=receipt_count,
                head_receipt_tag=head_receipt_tag,
            ),
        )

    def _transport_head_row(
        self,
        connection_id: str,
        session_id: str,
    ) -> sqlite3.Row | None:
        row = self._connection.execute(
            """
            SELECT * FROM capture_transport_heads
            WHERE connection_id = ? AND session_id = ?
            """,
            (connection_id, session_id),
        ).fetchone()
        if row is None:
            return None
        expected = self._integrity.tag(
            "transport_head",
            _transport_head_material(
                connection_id=connection_id,
                session_id=session_id,
                receipt_count=row["receipt_count"],
                head_receipt_tag=row["head_receipt_tag"],
            ),
        )
        if type(row["head_tag"]) is not str or not hmac.compare_digest(row["head_tag"], expected):
            raise CaptureStoreIntegrityError()
        return cast(sqlite3.Row, row)

    def _load_verified_transport_receipt(self, row: sqlite3.Row) -> sqlite3.Row:
        try:
            expected = self._integrity.tag(
                "transport_receipt",
                _transport_receipt_material(row),
            )
            if (
                type(row["connection_id"]) is not str
                or type(row["session_id"]) is not str
                or type(row["transport_ordinal"]) is not int
                or type(row["batch_ref"]) is not str
                or type(row["chunk_index"]) is not int
                or type(row["chunk_count"]) is not int
                or type(row["chunk_digest"]) is not str
                or type(row["intake_count"]) is not int
                or type(row["intake_set_digest"]) is not str
                or type(row["post_event_count"]) is not int
                or (
                    row["post_head_event_tag"] is not None
                    and type(row["post_head_event_tag"]) is not str
                )
                or (
                    row["previous_receipt_tag"] is not None
                    and type(row["previous_receipt_tag"]) is not str
                )
                or type(row["receipt_tag"]) is not str
                or type(row["admitted_at"]) is not str
                or not hmac.compare_digest(row["receipt_tag"], expected)
            ):
                raise CaptureStoreIntegrityError()
        except Exception:
            raise CaptureStoreIntegrityError() from None
        return row

    def _verify_transport_chain(
        self,
        connection_id: str,
        session_id: str,
        *,
        allow_pending_event_tail: bool = False,
    ) -> tuple[tuple[sqlite3.Row, ...], int, sqlite3.Row | None]:
        if type(allow_pending_event_tail) is not bool:
            raise CaptureStoreError()
        transport_profile = self._connection_row(connection_id)["profile_id"] in _TRANSPORT_PROFILES
        session = self._session_row(connection_id, session_id)
        if session is None:
            raise CaptureStoreIntegrityError()
        transport_required = session["transport_required"] == 1
        head = self._transport_head_row(connection_id, session_id)
        rows = self._connection.execute(
            """
            SELECT * FROM capture_transport_receipts
            WHERE connection_id = ? AND session_id = ?
            ORDER BY transport_ordinal
            """,
            (connection_id, session_id),
        ).fetchall()
        if not transport_profile:
            if transport_required or rows or head is not None:
                raise CaptureStoreIntegrityError()
            return (), 0, None
        if not transport_required:
            if rows or head is not None:
                raise CaptureStoreIntegrityError()
            # Version-one stores may contain bridge sessions that predate the
            # receiver-owned transport ledger. Their v1 session tag binds this
            # state until the first v2 transport attempt upgrades it atomically.
            return (), 0, None
        if head is None:
            raise CaptureStoreIntegrityError()
        if not hmac.compare_digest(session["transport_head_tag"], head["head_tag"]):
            raise CaptureStoreIntegrityError()
        event_rows = self._connection.execute(
            """
            SELECT * FROM capture_events
            WHERE connection_id = ? AND session_id = ?
            ORDER BY receipt_ordinal
            """,
            (connection_id, session_id),
        ).fetchall()
        event_tags: list[str] = []
        event_sources: list[str] = []
        for ordinal, event_row in enumerate(event_rows, start=1):
            event = self._load_verified_event(event_row)
            if event.receipt_ordinal != ordinal or event.event_tag != event_row["event_tag"]:
                raise CaptureStoreIntegrityError()
            event_tags.append(event.event_tag)
            event_sources.append(event_row["admission_source"])
        verified: list[sqlite3.Row] = []
        previous: str | None = None
        previous_event_count = 0
        chunk_counts: dict[str, int] = {}
        chunk_indices: dict[str, set[int]] = {}
        for ordinal, candidate in enumerate(rows, start=1):
            row = self._load_verified_transport_receipt(candidate)
            batch_ref = row["batch_ref"]
            chunk_count = row["chunk_count"]
            if (
                row["connection_id"] != connection_id
                or row["session_id"] != session_id
                or row["transport_ordinal"] != ordinal
                or row["previous_receipt_tag"] != previous
                or row["chunk_index"] >= chunk_count
                or chunk_counts.setdefault(batch_ref, chunk_count) != chunk_count
                or row["post_event_count"] < previous_event_count
                or row["post_event_count"] > len(event_tags)
                or (
                    None
                    if row["post_event_count"] == 0
                    else event_tags[row["post_event_count"] - 1]
                )
                != row["post_head_event_tag"]
            ):
                raise CaptureStoreIntegrityError()
            indices = chunk_indices.setdefault(batch_ref, set())
            if row["chunk_index"] in indices:
                raise CaptureStoreIntegrityError()
            indices.add(row["chunk_index"])
            previous = row["receipt_tag"]
            previous_event_count = row["post_event_count"]
            verified.append(row)
        if len(verified) != head["receipt_count"] or previous != head["head_receipt_tag"]:
            raise CaptureStoreIntegrityError()
        if previous_event_count != len(event_tags) and not allow_pending_event_tail:
            unreceipted_sources = event_sources[previous_event_count:]
            if not bool(session["coverage_degraded"]) or any(
                source != CaptureAdmissionSource.SPOOL_DRAIN.value for source in unreceipted_sources
            ):
                raise CaptureStoreIntegrityError()
        incomplete = sum(
            len(indices) != chunk_counts[batch_ref] for batch_ref, indices in chunk_indices.items()
        )
        return tuple(verified), incomplete, head

    def _transport_intake_set_digest(self, intakes: tuple[CaptureIntake, ...]) -> str:
        return self._integrity.tag(
            "transport_intake_set",
            {
                "schema_version": "capture-transport-intake-set-integrity/v1",
                "intakes": [
                    {
                        "producer_event_digest": intake.producer_event_digest,
                        "intake_tag": intake.intake_tag,
                    }
                    for intake in intakes
                ],
            },
        )

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

    def _append_session_row(self, connection_id: str, session_id: str) -> sqlite3.Row | None:
        """Load one authenticated session behind the durable deletion barrier."""

        if self._tombstone_row(connection_id, session_id) is not None:
            raise CaptureStoreStateError()
        session = self._session_row(connection_id, session_id)
        if session is not None and session["state"] == CaptureSessionState.DELETING.value:
            raise CaptureStoreStateError()
        return session

    def _verified_feedback_rows(
        self,
        connection_id: str,
        session_id: str,
    ) -> tuple[sqlite3.Row, ...]:
        from saliencegate.capture.feedback import (
            MAX_CAPTURE_FEEDBACK_REVISIONS_PER_SESSION,
            CaptureFeedbackLabel,
        )

        rows = self._connection.execute(
            """
            SELECT * FROM feedback_labels
            WHERE connection_id = ? AND session_id = ?
            ORDER BY created_at, label_id
            """,
            (connection_id, session_id),
        ).fetchall()
        if not rows:
            return ()
        if not 2 <= len(rows) <= MAX_CAPTURE_FEEDBACK_REVISIONS_PER_SESSION + 1:
            raise CaptureStoreIntegrityError()
        anchor = rows[-1]
        revision_rows = rows[:-1]
        verified: list[sqlite3.Row] = []
        previous_label_id: str | None = None
        previous_label: CaptureFeedbackLabel | None = None
        previous_created_at: datetime | None = None
        for revision, row in enumerate(revision_rows, start=1):
            try:
                label = CaptureFeedbackLabel(row["label"])
            except (TypeError, ValueError):
                raise CaptureStoreIntegrityError() from None
            created_at = _stored_timestamp(row["created_at"])
            if (
                row["connection_id"] != connection_id
                or row["session_id"] != session_id
                or created_at.tzinfo is None
                or created_at.utcoffset() != timedelta(0)
                or row["created_at"] != created_at.astimezone(UTC).isoformat()
                or (previous_created_at is not None and created_at <= previous_created_at)
                or (previous_label is not None and label is previous_label)
            ):
                raise CaptureStoreIntegrityError()
            expected_label_id = self._integrity.tag(
                "feedback_id",
                _feedback_identity_material(
                    connection_id=connection_id,
                    session_id=session_id,
                    revision=revision,
                    previous_label_id=previous_label_id,
                    label=label.value,
                    created_at=row["created_at"],
                ),
            )
            if type(row["label_id"]) is not str or not hmac.compare_digest(
                row["label_id"],
                expected_label_id,
            ):
                raise CaptureStoreIntegrityError()
            expected = self._integrity.tag("feedback", _feedback_material(row))
            if type(row["row_tag"]) is not str or not hmac.compare_digest(row["row_tag"], expected):
                raise CaptureStoreIntegrityError()
            verified.append(cast(sqlite3.Row, row))
            previous_label_id = row["label_id"]
            previous_label = label
            previous_created_at = created_at
        if previous_label_id is None or previous_label is None or previous_created_at is None:
            raise CaptureStoreIntegrityError()
        try:
            anchor_label = CaptureFeedbackLabel(anchor["label"])
        except (TypeError, ValueError):
            raise CaptureStoreIntegrityError() from None
        anchor_created_at = _stored_timestamp(anchor["created_at"])
        expected_anchor_id = self._integrity.tag(
            "feedback_anchor_id",
            _feedback_anchor_identity_material(
                connection_id=connection_id,
                session_id=session_id,
                revision_count=len(revision_rows),
                head_label_id=previous_label_id,
                label=previous_label.value,
                created_at=anchor["created_at"],
            ),
        )
        expected_anchor_tag = self._integrity.tag(
            "feedback",
            _feedback_material(anchor),
        )
        if (
            anchor["connection_id"] != connection_id
            or anchor["session_id"] != session_id
            or anchor_label is not previous_label
            or anchor_created_at.tzinfo is None
            or anchor_created_at.utcoffset() != timedelta(0)
            or anchor["created_at"] != anchor_created_at.astimezone(UTC).isoformat()
            or anchor_created_at <= previous_created_at
            or type(anchor["label_id"]) is not str
            or type(anchor["row_tag"]) is not str
            or not hmac.compare_digest(anchor["label_id"], expected_anchor_id)
            or not hmac.compare_digest(anchor["row_tag"], expected_anchor_tag)
        ):
            raise CaptureStoreIntegrityError()
        return tuple(verified)

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
                row["connection_id"] != intake.connection_id
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

    def _load_verified_append_event(
        self,
        row: sqlite3.Row,
    ) -> tuple[int, str | None, str]:
        """Authenticate one retained row without rebuilding its validated model."""

        try:
            blob = row["event_json"]
            if type(blob) is not bytes:
                raise CaptureStoreIntegrityError()
            document = _read_canonical_capture_event_document(blob)
            if set(document) != {
                "event_tag",
                "intake",
                "previous_event_tag",
                "receipt_ordinal",
                "schema_version",
            }:
                raise CaptureStoreIntegrityError()
            receipt_ordinal = document["receipt_ordinal"]
            previous_event_tag = document["previous_event_tag"]
            event_tag = document["event_tag"]
            intake = document["intake"]
            if (
                document["schema_version"] != "capture-event/v1"
                or type(receipt_ordinal) is not int
                or not 1 <= receipt_ordinal <= MAX_CAPTURE_EVENTS_PER_SESSION
                or (receipt_ordinal == 1) != (previous_event_tag is None)
                or (
                    previous_event_tag is not None
                    and (
                        type(previous_event_tag) is not str
                        or _PROJECT_DIGEST.fullmatch(previous_event_tag) is None
                    )
                )
                or type(event_tag) is not str
                or _PROJECT_DIGEST.fullmatch(event_tag) is None
                or not isinstance(intake, Mapping)
            ):
                raise CaptureStoreIntegrityError()
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
                    intake=cast(Mapping[str, object], intake),
                ),
            )
            if (
                row["receipt_ordinal"] != receipt_ordinal
                or row["previous_event_tag"] != previous_event_tag
                or row["event_tag"] != event_tag
                or row["connection_id"] != intake.get("connection_id")
                or row["session_id"] != intake.get("session_id")
                or row["producer_event_digest"] != intake.get("producer_event_digest")
                or row["event_kind"] != intake.get("kind")
                or type(row["admission_source"]) is not str
                or type(row["admitted_at"]) is not str
                or not hmac.compare_digest(event_tag, expected)
            ):
                raise CaptureStoreIntegrityError()
            return receipt_ordinal, previous_event_tag, event_tag
        except Exception:
            raise CaptureStoreIntegrityError() from None

    def _verify_chain_rows(
        self,
        connection_id: str,
        session_id: str,
        session: sqlite3.Row,
        head: sqlite3.Row,
    ) -> tuple[tuple[tuple[CaptureEvent, sqlite3.Row], ...], tuple[sqlite3.Row, ...]]:
        rows = self._connection.execute(
            """
            SELECT *
            FROM capture_events
            WHERE connection_id = ? AND session_id = ?
            ORDER BY receipt_ordinal
            """,
            (connection_id, session_id),
        ).fetchall()
        events: list[tuple[CaptureEvent, sqlite3.Row]] = []
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
            events.append((event, cast(sqlite3.Row, row)))
        if (
            len(events) != session["event_count"]
            or len(events) != head["receipt_count"]
            or previous != head["head_event_tag"]
        ):
            raise CaptureStoreIntegrityError()
        health = self._verify_health_set(connection_id, session_id, session)
        return tuple(events), health

    def _verify_chain(
        self,
        connection_id: str,
        session_id: str,
        session: sqlite3.Row,
        head: sqlite3.Row,
        *,
        allow_pending_transport_tail: bool = False,
    ) -> tuple[CaptureEvent, ...]:
        events, _health = self._verify_chain_rows(
            connection_id,
            session_id,
            session,
            head,
        )
        self._verify_transport_chain(
            connection_id,
            session_id,
            allow_pending_event_tail=allow_pending_transport_tail,
        )
        return tuple(event for event, _row in events)

    def _verify_append_chain(
        self,
        connection_id: str,
        session_id: str,
        session: sqlite3.Row,
        head: sqlite3.Row,
    ) -> CaptureEvent | None:
        """Authenticate every retained row before a cold append.

        Each event tag was created only after strict intake authentication and
        canonical CaptureEvent construction.  Recomputing that tag over the exact
        bounded canonical document therefore re-establishes admission provenance
        without constructing a Pydantic object for every historical row.  The tip
        is still loaded through the full schema boundary before it is returned.
        """

        rows = self._connection.execute(
            """
            SELECT *
            FROM capture_events
            WHERE connection_id = ? AND session_id = ?
            ORDER BY receipt_ordinal
            """,
            (connection_id, session_id),
        ).fetchall()
        previous: str | None = None
        latest: sqlite3.Row | None = None
        for ordinal, row in enumerate(rows, start=1):
            receipt_ordinal, previous_event_tag, event_tag = self._load_verified_append_event(row)
            if (
                row["receipt_ordinal"] != ordinal
                or receipt_ordinal != ordinal
                or row["connection_id"] != connection_id
                or row["session_id"] != session_id
                or previous_event_tag != previous
            ):
                raise CaptureStoreIntegrityError()
            previous = event_tag
            latest = cast(sqlite3.Row, row)
        if (
            len(rows) != session["event_count"]
            or len(rows) != head["receipt_count"]
            or previous != head["head_event_tag"]
        ):
            raise CaptureStoreIntegrityError()
        self._verify_health_set(connection_id, session_id, session)
        self._verify_transport_chain(connection_id, session_id)
        return None if latest is None else self._load_verified_event(latest)

    def _verify_append_commitment(
        self,
        connection_id: str,
        session_id: str,
        session: sqlite3.Row,
        head: sqlite3.Row,
    ) -> CaptureEvent | None:
        """Authenticate the bounded state needed to extend one session safely."""

        event_count = session["event_count"]
        receipt_count = head["receipt_count"]
        if event_count != receipt_count:
            raise CaptureStoreIntegrityError()
        data_version = self._database_data_version()
        if data_version != self._verified_data_version:
            self._verified_append_heads.clear()
            self._verified_data_version = data_version
        key = (connection_id, session_id)
        commitment = (receipt_count, head["head_event_tag"])
        cache_miss = self._verified_append_heads.get(key) != commitment
        if cache_miss:
            latest = self._verify_append_chain(connection_id, session_id, session, head)
            self._verified_append_heads[key] = commitment
            return latest
        inventory = self._connection.execute(
            """
            SELECT COUNT(*) AS event_count
            FROM capture_events
            WHERE connection_id = ? AND session_id = ?
            """,
            (connection_id, session_id),
        ).fetchone()
        if inventory is None or inventory["event_count"] != receipt_count:
            raise CaptureStoreIntegrityError()
        latest = self._connection.execute(
            """
            SELECT *
            FROM capture_events
            WHERE connection_id = ? AND session_id = ?
            ORDER BY receipt_ordinal DESC
            LIMIT 1
            """,
            (connection_id, session_id),
        ).fetchone()
        if receipt_count == 0:
            if latest is not None or head["head_event_tag"] is not None:
                raise CaptureStoreIntegrityError()
            self._verify_health_set(connection_id, session_id, session)
            return None
        if latest is None or latest["receipt_ordinal"] != receipt_count:
            raise CaptureStoreIntegrityError()
        event = self._load_verified_event(latest)
        if (
            event.intake.connection_id != connection_id
            or event.intake.session_id != session_id
            or event.receipt_ordinal != receipt_count
            or latest["event_tag"] != head["head_event_tag"]
            or event.event_tag != head["head_event_tag"]
        ):
            raise CaptureStoreIntegrityError()
        self._verify_health_set(connection_id, session_id, session)
        return event

    def _verify_all_state(self, *, immutable: bool = False) -> None:
        """Authenticate every currently persisted mutable row before use."""

        if type(immutable) is not bool:
            raise CaptureStoreError()
        if immutable:
            self._authorization.revalidate()
        else:
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
            verified_feedback_count = 0
            for session_identity in session_rows:
                connection_id = session_identity["connection_id"]
                session_id = session_identity["session_id"]
                session = self._session_row(connection_id, session_id)
                if session is None:
                    raise CaptureStoreIntegrityError()
                head = self._head_row(connection_id, session_id)
                self._verify_chain(connection_id, session_id, session, head)
                feedback_rows = self._verified_feedback_rows(connection_id, session_id)
                verified_feedback_count += len(feedback_rows) + bool(feedback_rows)
            persisted_feedback_count = self._connection.execute(
                "SELECT COUNT(*) FROM feedback_labels"
            ).fetchone()[0]
            if (
                type(persisted_feedback_count) is not int
                or persisted_feedback_count != verified_feedback_count
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
        if immutable:
            self._authorization.revalidate()
        else:
            self._revalidate_boundary()

    def _encoded_human_session_id(self, connection_id: str, session_id: str) -> str:
        digest = self._integrity.tag(
            "human_id",
            {
                "schema_version": "capture-human-session-id/v1",
                "connection_id": connection_id,
                "session_id": session_id,
            },
        )
        return base64.b32encode(bytes.fromhex(digest)).decode("ascii").lower().rstrip("=")

    def _human_session_id(self, connection_id: str, session_id: str) -> str:
        encoded = self._encoded_human_session_id(connection_id, session_id)
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

    def _deleted_session_for_human_id(self, human_id: str) -> sqlite3.Row | None:
        identities = self._connection.execute(
            """
            SELECT connection_id, session_id
            FROM deleted_sessions
            ORDER BY connection_id, session_id
            """
        ).fetchall()
        matched: sqlite3.Row | None = None
        for identity in identities:
            tombstone = self._tombstone_row(
                identity["connection_id"],
                identity["session_id"],
            )
            if tombstone is None:
                raise CaptureStoreIntegrityError()
            encoded = self._encoded_human_session_id(
                tombstone["connection_id"],
                tombstone["session_id"],
            )
            if encoded.startswith(human_id):
                if matched is not None:
                    raise CaptureStoreIntegrityError()
                matched = tombstone
        return matched

    def _enable_secure_delete(self) -> None:
        try:
            self._connection.execute("PRAGMA secure_delete = ON")
            setting = self._connection.execute("PRAGMA secure_delete").fetchone()
        except sqlite3.Error:
            raise CaptureStoreError() from None
        if setting is None or len(setting) != 1 or setting[0] != 1:
            raise CaptureStoreError()

    def _checkpoint_wal(self) -> None:
        try:
            result = self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        except sqlite3.Error as error:
            code = getattr(error, "sqlite_errorcode", None)
            if type(code) is int and code & 0xFF == sqlite3.SQLITE_BUSY:
                raise CaptureStoreBusyError() from None
            raise CaptureStoreError() from None
        if result is None or len(result) != 3 or any(type(value) is not int for value in result):
            raise CaptureStoreError()
        if result[0] != 0:
            raise CaptureStoreBusyError()
        self._fault("delete_after_checkpoint")

    def _require_project_connections_disabled(self, project_digest: str) -> bool:
        self._require_maintenance()
        project_digest = self._validate_project_digest(project_digest)
        with self._lock:
            self._require_maintenance()
            self._revalidate_boundary()
            try:
                self._connection.execute("BEGIN")
                identities = self._connection.execute(
                    """
                    SELECT connection_id
                    FROM connections
                    WHERE project_digest = ?
                    ORDER BY connection_id
                    """,
                    (project_digest,),
                ).fetchall()
                states: list[CaptureConnectionState] = []
                for identity in identities:
                    connection = self._connection_row(identity["connection_id"])
                    states.append(CaptureConnectionState(connection["state"]))
                state_set = set(states)
                if state_set not in (
                    set(),
                    {CaptureConnectionState.DISABLED},
                    {CaptureConnectionState.DELETING},
                ):
                    raise CaptureStoreStateError()
                should_drain = state_set == {CaptureConnectionState.DISABLED}
                self._connection.commit()
            except CaptureStoreError:
                self._rollback()
                raise
            except sqlite3.Error:
                self._rollback()
                raise CaptureStoreError() from None
            except Exception:
                self._rollback()
                raise CaptureStoreIntegrityError() from None
            except BaseException:
                self._rollback()
                raise
            self._revalidate_boundary()
            return should_drain

    def _session_delete_requires_drain(self, human_id: str) -> bool:
        self._require_maintenance()
        human_id = self._validate_human_id(human_id)
        with self._lock:
            self._require_maintenance()
            self._revalidate_boundary()
            try:
                self._connection.execute("BEGIN")
                identity = self._connection.execute(
                    """
                    SELECT connection_id, session_id
                    FROM capture_sessions
                    WHERE human_id = ?
                    """,
                    (human_id,),
                ).fetchone()
                if identity is None:
                    if self._deleted_session_for_human_id(human_id) is None:
                        raise CaptureStoreStateError()
                    should_drain = False
                else:
                    connection_id = identity["connection_id"]
                    session_id = identity["session_id"]
                    self._connection_row(connection_id)
                    session = self._session_row(connection_id, session_id)
                    if session is None:
                        raise CaptureStoreIntegrityError()
                    head = self._head_row(connection_id, session_id)
                    self._verify_chain(connection_id, session_id, session, head)
                    self._verified_feedback_rows(connection_id, session_id)
                    should_drain = session["state"] != CaptureSessionState.DELETING.value
                self._connection.commit()
            except CaptureStoreError:
                self._rollback()
                raise
            except sqlite3.Error:
                self._rollback()
                raise CaptureStoreError() from None
            except Exception:
                self._rollback()
                raise CaptureStoreIntegrityError() from None
            except BaseException:
                self._rollback()
                raise
            self._revalidate_boundary()
            return should_drain

    def _delete_session(self, human_id: str) -> CaptureSessionDeleteReceipt:
        from saliencegate.capture.delete import (
            CaptureDeleteDisposition,
            CaptureSessionDeleteReceipt,
        )

        self._require_maintenance()
        human_id = self._validate_human_id(human_id)
        with self._lock:
            self._require_maintenance()
            self._revalidate_boundary()
            self._enable_secure_delete()
            self._begin_immediate()
            try:
                identity = self._connection.execute(
                    """
                    SELECT connection_id, session_id
                    FROM capture_sessions
                    WHERE human_id = ?
                    """,
                    (human_id,),
                ).fetchone()
                if identity is None:
                    tombstone = self._deleted_session_for_human_id(human_id)
                    if tombstone is None:
                        raise CaptureStoreStateError()
                    self._connection.commit()
                    self._revalidate_boundary()
                    self._checkpoint_wal()
                    self._revalidate_boundary()
                    return CaptureSessionDeleteReceipt(
                        disposition=CaptureDeleteDisposition.ALREADY_DELETED,
                        human_id=human_id,
                    )
                connection_id = identity["connection_id"]
                session_id = identity["session_id"]
                self._connection_row(connection_id)
                session = self._session_row(connection_id, session_id)
                if session is None:
                    raise CaptureStoreIntegrityError()
                head = self._head_row(connection_id, session_id)
                self._verify_chain(connection_id, session_id, session, head)
                self._verified_feedback_rows(connection_id, session_id)
                if session["state"] != CaptureSessionState.DELETING.value:
                    self._update_session(
                        session,
                        state=CaptureSessionState.DELETING,
                        event_count=session["event_count"],
                        coverage_degraded=bool(session["coverage_degraded"]),
                    )
                self._connection.commit()
                self._fault("delete_after_mark_commit")
            except sqlite3.Error:
                self._rollback()
                raise CaptureStoreError() from None
            except BaseException:
                self._rollback()
                raise

            self._begin_immediate()
            try:
                connection = self._connection_row(connection_id)
                session = self._session_row(connection_id, session_id)
                if session is None:
                    tombstone = self._tombstone_row(connection_id, session_id)
                    if tombstone is None:
                        raise CaptureStoreIntegrityError()
                    self._connection.commit()
                    disposition = CaptureDeleteDisposition.ALREADY_DELETED
                else:
                    if session["state"] != CaptureSessionState.DELETING.value:
                        raise CaptureStoreIntegrityError()
                    head = self._head_row(connection_id, session_id)
                    self._verify_chain(connection_id, session_id, session, head)
                    self._verified_feedback_rows(connection_id, session_id)
                    tombstone = self._tombstone_row(connection_id, session_id)
                    if tombstone is None:
                        material: dict[str, object] = {
                            "connection_id": connection_id,
                            "session_id": session_id,
                            "project_digest": connection["project_digest"],
                            "deleted_at": _now(),
                        }
                        tombstone_tag = self._integrity.tag(
                            "tombstone",
                            _tombstone_material(material),
                        )
                        self._connection.execute(
                            """
                            INSERT INTO deleted_sessions(
                                connection_id, session_id, project_digest,
                                deleted_at, tombstone_tag
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (*material.values(), tombstone_tag),
                        )
                        tombstone = self._tombstone_row(connection_id, session_id)
                    if (
                        tombstone is None
                        or tombstone["project_digest"] != connection["project_digest"]
                    ):
                        raise CaptureStoreIntegrityError()
                    self._fault("delete_after_tombstone_write")
                    deleted = self._connection.execute(
                        """
                        DELETE FROM capture_sessions
                        WHERE connection_id = ? AND session_id = ?
                        """,
                        (connection_id, session_id),
                    )
                    if deleted.rowcount != 1:
                        raise CaptureStoreIntegrityError()
                    self._fault("delete_before_purge_commit")
                    self._connection.commit()
                    self._fault("delete_after_purge_commit")
                    disposition = CaptureDeleteDisposition.DELETED
            except sqlite3.Error:
                self._rollback()
                raise CaptureStoreError() from None
            except BaseException:
                self._rollback()
                raise
            self._revalidate_boundary()
            self._checkpoint_wal()
            self._revalidate_boundary()
            return CaptureSessionDeleteReceipt(
                disposition=disposition,
                human_id=human_id,
            )

    def _delete_project(self, project_digest: str) -> CaptureProjectDeleteReceipt:
        from saliencegate.capture.delete import (
            CaptureDeleteDisposition,
            CaptureProjectDeleteReceipt,
        )

        self._require_maintenance()
        project_digest = self._validate_project_digest(project_digest)
        with self._lock:
            self._require_maintenance()
            self._revalidate_boundary()
            self._enable_secure_delete()
            self._begin_immediate()
            try:
                identity_rows = self._connection.execute(
                    """
                    SELECT connection_id
                    FROM connections
                    WHERE project_digest = ?
                    ORDER BY connection_id
                    """,
                    (project_digest,),
                ).fetchall()
                connection_ids = tuple(row["connection_id"] for row in identity_rows)
                if not connection_ids:
                    self._connection.commit()
                    self._revalidate_boundary()
                    self._checkpoint_wal()
                    self._revalidate_boundary()
                    return CaptureProjectDeleteReceipt(
                        disposition=CaptureDeleteDisposition.ALREADY_DELETED,
                        project_digest=project_digest,
                        deleted_connections=0,
                        deleted_sessions=0,
                        deleted_tombstones=0,
                    )
                for connection_id in connection_ids:
                    connection = self._connection_row(connection_id)
                    state = CaptureConnectionState(connection["state"])
                    if state not in {
                        CaptureConnectionState.DISABLED,
                        CaptureConnectionState.DELETING,
                    }:
                        raise CaptureStoreStateError()
                    if state is CaptureConnectionState.DISABLED:
                        material = dict(connection)
                        material["state"] = CaptureConnectionState.DELETING.value
                        material["updated_at"] = _now()
                        row_tag = self._integrity.tag(
                            "connection",
                            _connection_material(material),
                        )
                        updated = self._connection.execute(
                            """
                            UPDATE connections
                            SET state = ?, updated_at = ?, row_tag = ?
                            WHERE connection_id = ? AND state = ?
                            """,
                            (
                                material["state"],
                                material["updated_at"],
                                row_tag,
                                connection_id,
                                CaptureConnectionState.DISABLED.value,
                            ),
                        )
                        if updated.rowcount != 1:
                            raise CaptureStoreStateError()
                self._connection.commit()
                self._fault("delete_project_after_mark_commit")
            except sqlite3.Error:
                self._rollback()
                raise CaptureStoreError() from None
            except BaseException:
                self._rollback()
                raise

            self._begin_immediate()
            try:
                current_rows = self._connection.execute(
                    """
                    SELECT connection_id
                    FROM connections
                    WHERE project_digest = ?
                    ORDER BY connection_id
                    """,
                    (project_digest,),
                ).fetchall()
                current_ids = tuple(row["connection_id"] for row in current_rows)
                if current_ids != connection_ids:
                    raise CaptureStoreStateError()
                deleted_sessions = 0
                deleted_tombstones = 0
                for connection_id in connection_ids:
                    connection = self._connection_row(connection_id)
                    if connection["state"] != CaptureConnectionState.DELETING.value:
                        raise CaptureStoreStateError()
                    sessions = self._connection.execute(
                        """
                        SELECT session_id
                        FROM capture_sessions
                        WHERE connection_id = ?
                        ORDER BY session_id
                        """,
                        (connection_id,),
                    ).fetchall()
                    for identity in sessions:
                        session_id = identity["session_id"]
                        session = self._session_row(connection_id, session_id)
                        if session is None:
                            raise CaptureStoreIntegrityError()
                        head = self._head_row(connection_id, session_id)
                        self._verify_chain(connection_id, session_id, session, head)
                        self._verified_feedback_rows(connection_id, session_id)
                    tombstones = self._connection.execute(
                        """
                        SELECT session_id
                        FROM deleted_sessions
                        WHERE connection_id = ?
                        ORDER BY session_id
                        """,
                        (connection_id,),
                    ).fetchall()
                    for identity in tombstones:
                        tombstone = self._tombstone_row(
                            connection_id,
                            identity["session_id"],
                        )
                        if tombstone is None or tombstone["project_digest"] != project_digest:
                            raise CaptureStoreIntegrityError()
                    deleted_sessions += len(sessions)
                    deleted_tombstones += len(tombstones)
                deleted = self._connection.execute(
                    "DELETE FROM connections WHERE project_digest = ?",
                    (project_digest,),
                )
                if deleted.rowcount != len(connection_ids):
                    raise CaptureStoreIntegrityError()
                self._fault("delete_project_before_purge_commit")
                self._connection.commit()
                self._fault("delete_project_after_purge_commit")
            except sqlite3.Error:
                self._rollback()
                raise CaptureStoreError() from None
            except BaseException:
                self._rollback()
                raise
            self._revalidate_boundary()
            self._checkpoint_wal()
            self._revalidate_boundary()
            return CaptureProjectDeleteReceipt(
                disposition=CaptureDeleteDisposition.DELETED,
                project_digest=project_digest,
                deleted_connections=len(connection_ids),
                deleted_sessions=deleted_sessions,
                deleted_tombstones=deleted_tombstones,
            )

    def _create_session(
        self,
        intake: CaptureIntake,
        *,
        force_coverage_degraded: bool = False,
    ) -> sqlite3.Row:
        timestamp = _now()
        transport_required = intake.adapter_profile in _TRANSPORT_PROFILES
        transport_head_tag = (
            self._transport_head_tag(
                intake.connection_id,
                intake.session_id,
                receipt_count=0,
                head_receipt_tag=None,
            )
            if transport_required
            else None
        )
        material: dict[str, object] = {
            "connection_id": intake.connection_id,
            "session_id": intake.session_id,
            "human_id": self._human_session_id(intake.connection_id, intake.session_id),
            "state": CaptureSessionState.OPEN.value,
            "event_count": 0,
            "coverage_degraded": int(
                force_coverage_degraded or intake.capture_disposition != "captured"
            ),
            "unattributed_drop": 0,
            "transport_required": int(transport_required),
            "transport_head_tag": transport_head_tag,
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
                coverage_degraded, unattributed_drop, transport_required,
                transport_head_tag, health_marker_count, health_set_digest,
                opened_at, updated_at, closed_at, row_tag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        self._create_transport_head(
            intake.connection_id,
            intake.session_id,
            profile_id=intake.adapter_profile,
        )
        row = self._session_row(intake.connection_id, intake.session_id)
        if row is None:
            raise CaptureStoreIntegrityError()
        return row

    def _create_degraded_session(self, connection_id: str, session_id: str) -> sqlite3.Row:
        """Create a content-free quarantine when a callback cannot form evidence."""

        timestamp = _now()
        connection = self._connection_row(connection_id)
        transport_required = connection["profile_id"] in _TRANSPORT_PROFILES
        transport_head_tag = (
            self._transport_head_tag(
                connection_id,
                session_id,
                receipt_count=0,
                head_receipt_tag=None,
            )
            if transport_required
            else None
        )
        material: dict[str, object] = {
            "connection_id": connection_id,
            "session_id": session_id,
            "human_id": self._human_session_id(connection_id, session_id),
            "state": CaptureSessionState.QUARANTINED.value,
            "event_count": 0,
            "coverage_degraded": 1,
            "unattributed_drop": 0,
            "transport_required": int(transport_required),
            "transport_head_tag": transport_head_tag,
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
                coverage_degraded, unattributed_drop, transport_required,
                transport_head_tag, health_marker_count, health_set_digest,
                opened_at, updated_at, closed_at, row_tag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (*material.values(), row_tag),
        )
        head_tag = self._integrity.tag(
            "head",
            _head_material(
                connection_id=connection_id,
                session_id=session_id,
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
            (connection_id, session_id, head_tag),
        )
        self._create_transport_head(
            connection_id,
            session_id,
            profile_id=connection["profile_id"],
        )
        row = self._session_row(connection_id, session_id)
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
        transport_required: bool | None = None,
        transport_head_tag: str | None = None,
    ) -> sqlite3.Row:
        material = dict(row)
        material["state"] = state.value
        material["event_count"] = event_count
        material["coverage_degraded"] = int(coverage_degraded)
        if transport_required is not None:
            material["transport_required"] = int(transport_required)
        if transport_head_tag is not None:
            material["transport_head_tag"] = transport_head_tag
        material["updated_at"] = _now()
        if state is CaptureSessionState.CLOSED:
            material["closed_at"] = (
                row["closed_at"]
                if row["state"] == CaptureSessionState.CLOSED.value and row["closed_at"] is not None
                else material["updated_at"]
            )
        else:
            material["closed_at"] = None
        row_tag = self._integrity.tag("session", _session_material(material))
        result = self._connection.execute(
            """
            UPDATE capture_sessions
            SET state = ?, event_count = ?, coverage_degraded = ?,
                transport_required = ?, transport_head_tag = ?, updated_at = ?,
                closed_at = ?, row_tag = ?
            WHERE connection_id = ? AND session_id = ?
            """,
            (
                material["state"],
                material["event_count"],
                material["coverage_degraded"],
                material["transport_required"],
                material["transport_head_tag"],
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

    def _mark_transport_fallback_session(self, row: sqlite3.Row) -> sqlite3.Row:
        """Make a receipt-less bridge path's coverage loss durable."""

        connection = self._connection_row(row["connection_id"])
        if connection["profile_id"] not in _TRANSPORT_PROFILES:
            raise CaptureStoreIntegrityError()
        updated = self._update_session(
            row,
            state=CaptureSessionState(row["state"]),
            event_count=row["event_count"],
            coverage_degraded=True,
        )
        self._verify_transport_chain(row["connection_id"], row["session_id"])
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

    def mark_session_health(
        self,
        connection_id: str,
        session_id: str,
        code: CaptureHealthCode,
    ) -> None:
        """Atomically attach one authenticated health marker to an existing session."""

        self._ensure_open()
        if (
            type(connection_id) is not str
            or type(session_id) is not str
            or _PROJECT_DIGEST.fullmatch(session_id) is None
            or type(code) is not CaptureHealthCode
        ):
            raise CaptureStoreError()
        with self._lock:
            self._ensure_open()
            self._revalidate_boundary()
            self._begin_immediate()
            try:
                connection = self._connection_row(connection_id)
                if connection["state"] != CaptureConnectionState.ENABLED.value:
                    raise CaptureStoreStateError()
                session = self._append_session_row(connection_id, session_id)
                if session is None:
                    session = self._create_degraded_session(connection_id, session_id)
                    head = self._head_row(connection_id, session_id)
                else:
                    head = self._head_row(connection_id, session_id)
                    self._verify_append_commitment(connection_id, session_id, session, head)
                    session = self._update_session(
                        session,
                        state=CaptureSessionState(session["state"]),
                        event_count=session["event_count"],
                        coverage_degraded=True,
                    )
                self._record_health(
                    connection_id=connection_id,
                    session_id=session_id,
                    code=code,
                )
                self._verified_append_heads[(connection_id, session_id)] = (
                    head["receipt_count"],
                    head["head_event_tag"],
                )
                self._connection.commit()
            except BaseException:
                self._rollback()
                raise
            self._revalidate_boundary()

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

    def _append_authenticated_in_transaction(
        self,
        authenticated: CaptureIntake,
        canonical_intake: bytes,
        *,
        source: CaptureAdmissionSource,
        transport_bound: bool = False,
    ) -> CaptureAppendReceipt:
        """Extend one event chain inside the caller-owned write transaction."""

        connection = self._connection_row(authenticated.connection_id)
        if (
            connection["profile_id"] != authenticated.adapter_profile
            or connection["capability_manifest_digest"] != authenticated.capability_manifest_digest
        ):
            raise CaptureStoreStateError()
        if connection["state"] == CaptureConnectionState.DELETING.value:
            raise CaptureStoreStateError()
        transport_profile = connection["profile_id"] in _TRANSPORT_PROFILES
        if transport_profile and source is CaptureAdmissionSource.DIRECT and not transport_bound:
            raise CaptureStoreStateError()
        transport_fallback = (
            transport_profile
            and source is CaptureAdmissionSource.SPOOL_DRAIN
            and not transport_bound
        )
        allowed_states = (
            {CaptureConnectionState.ENABLED.value}
            if source is CaptureAdmissionSource.DIRECT
            else {
                CaptureConnectionState.ENABLED.value,
                CaptureConnectionState.DRAINING.value,
                CaptureConnectionState.DISABLED.value,
            }
        )
        incoming_session = self._append_session_row(
            authenticated.connection_id,
            authenticated.session_id,
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
            existing_session = self._append_session_row(
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
                allow_pending_transport_tail=transport_bound,
            )
            if canonical_capture_intake(existing.intake) == canonical_intake:
                if transport_fallback:
                    existing_session = self._mark_transport_fallback_session(existing_session)
                return self._replay_receipt(existing, existing_session)
            if connection["state"] not in allowed_states:
                raise CaptureStoreStateError()
            affected = {(authenticated.connection_id, authenticated.session_id)}
            affected.add((existing.intake.connection_id, existing.intake.session_id))
            for affected_connection, affected_session_id in affected:
                session = self._append_session_row(
                    affected_connection,
                    affected_session_id,
                )
                if session is None:
                    if (affected_connection, affected_session_id) != (
                        authenticated.connection_id,
                        authenticated.session_id,
                    ):
                        raise CaptureStoreIntegrityError()
                    session = self._create_session(
                        authenticated,
                        force_coverage_degraded=transport_fallback,
                    )
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
                    self._verify_append_commitment(
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
        session = incoming_session
        if session is None:
            if authenticated.kind != "session_started":
                raise CaptureStoreStateError()
            session = self._create_session(
                authenticated,
                force_coverage_degraded=transport_fallback,
            )
        elif transport_fallback and session["state"] in {
            CaptureSessionState.CLOSED.value,
            CaptureSessionState.QUARANTINED.value,
        }:
            session = self._quarantine_transport_session(
                authenticated.connection_id,
                authenticated.session_id,
                code=CaptureHealthCode.GAP_DETECTED,
            )
            return CaptureAppendReceipt(
                disposition=CaptureAppendDisposition.QUARANTINED,
                connection_id=authenticated.connection_id,
                session_id=authenticated.session_id,
                producer_event_digest=authenticated.producer_event_digest,
                receipt_ordinal=None,
                previous_event_tag=None,
                event_tag=None,
                session_state=CaptureSessionState.QUARANTINED,
                event_count=session["event_count"],
            )
        elif (
            session["state"] != CaptureSessionState.OPEN.value
            or authenticated.kind == "session_started"
        ):
            raise CaptureStoreStateError()
        head = self._head_row(authenticated.connection_id, authenticated.session_id)
        self._verify_append_commitment(
            authenticated.connection_id,
            authenticated.session_id,
            session,
            head,
        )
        if transport_fallback:
            session = self._mark_transport_fallback_session(session)
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
                or transport_fallback
            ),
        )
        self._fault("after_session_health_write")
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
                receipt = self._append_authenticated_in_transaction(
                    authenticated,
                    canonical_intake,
                    source=source,
                )
                if receipt.disposition is CaptureAppendDisposition.REPLAYED:
                    self._connection.rollback()
                else:
                    if receipt.disposition is CaptureAppendDisposition.ADMITTED:
                        self._fault("before_commit")
                    self._connection.commit()
                    if receipt.disposition is CaptureAppendDisposition.ADMITTED:
                        assert receipt.receipt_ordinal is not None
                        self._verified_append_heads[
                            (authenticated.connection_id, authenticated.session_id)
                        ] = (receipt.receipt_ordinal, receipt.event_tag)
                        self._fault("after_commit")
            except BaseException:
                self._rollback()
                raise
            self._revalidate_boundary()
            return receipt

    def _quarantine_transport_session(
        self,
        connection_id: str,
        session_id: str,
        *,
        code: CaptureHealthCode,
    ) -> sqlite3.Row:
        session = self._append_session_row(connection_id, session_id)
        if session is None:
            session = self._create_degraded_session(connection_id, session_id)
        else:
            head = self._head_row(connection_id, session_id)
            self._verify_append_commitment(connection_id, session_id, session, head)
            if (
                session["state"] != CaptureSessionState.QUARANTINED.value
                or session["coverage_degraded"] != 1
            ):
                session = self._update_session(
                    session,
                    state=CaptureSessionState.QUARANTINED,
                    event_count=session["event_count"],
                    coverage_degraded=True,
                )
        health = self._verify_health_set(connection_id, session_id, session)
        if not any(row["code"] == code.value for row in health):
            self._record_health(
                connection_id=connection_id,
                session_id=session_id,
                code=code,
            )
        updated = self._session_row(connection_id, session_id)
        if updated is None:
            raise CaptureStoreIntegrityError()
        return updated

    def _transport_failure_receipt(
        self,
        descriptor: CaptureTransportChunk,
        *,
        disposition: CaptureTransportDisposition,
        intake_count: int,
        session: sqlite3.Row,
    ) -> CaptureTransportReceipt:
        _rows, incomplete, _head = self._verify_transport_chain(
            descriptor.connection_id,
            descriptor.session_id,
        )
        return CaptureTransportReceipt(
            disposition=disposition,
            connection_id=descriptor.connection_id,
            session_id=descriptor.session_id,
            batch_ref=descriptor.batch_ref,
            chunk_index=descriptor.chunk_index,
            chunk_count=descriptor.chunk_count,
            intake_count=intake_count,
            transport_ordinal=None,
            previous_receipt_tag=None,
            receipt_tag=None,
            incomplete_batch_count=incomplete,
            event_count=session["event_count"],
        )

    def append_transport_chunk(
        self,
        chunk: CaptureTransportChunk,
        intakes: tuple[CaptureIntake, ...],
    ) -> CaptureTransportReceipt:
        """Atomically admit one pseudonymized intake set and its transport receipt."""

        self._ensure_open()
        try:
            descriptor = validate_capture_transport_chunk(chunk)
            if (
                type(intakes) is not tuple
                or len(intakes) > MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION
            ):
                raise CaptureStoreError()
            authenticated = tuple(
                verify_capture_intake_authentication(intake, context=self._context)
                for intake in intakes
            )
            canonical = tuple(canonical_capture_intake(intake) for intake in authenticated)
            if any(
                intake.connection_id != descriptor.connection_id
                or intake.session_id != descriptor.session_id
                for intake in authenticated
            ):
                raise CaptureStoreStateError()
            intake_set_digest = self._transport_intake_set_digest(authenticated)
        except CaptureStoreError:
            raise
        except Exception:
            raise CaptureStoreIntegrityError() from None

        with self._lock:
            self._ensure_open()
            self._revalidate_boundary()
            self._begin_immediate()
            try:
                connection = self._connection_row(descriptor.connection_id)
                if (
                    connection["profile_id"] not in _TRANSPORT_PROFILES
                    or connection["state"] != CaptureConnectionState.ENABLED.value
                    or any(
                        intake.adapter_profile != connection["profile_id"]
                        or intake.capability_manifest_digest
                        != connection["capability_manifest_digest"]
                        for intake in authenticated
                    )
                ):
                    raise CaptureStoreStateError()

                existing_candidate = self._connection.execute(
                    """
                    SELECT * FROM capture_transport_receipts
                    WHERE connection_id = ? AND batch_ref = ? AND chunk_index = ?
                    """,
                    (
                        descriptor.connection_id,
                        descriptor.batch_ref,
                        descriptor.chunk_index,
                    ),
                ).fetchone()
                if existing_candidate is not None:
                    existing_rows, incomplete, _existing_head = self._verify_transport_chain(
                        existing_candidate["connection_id"],
                        existing_candidate["session_id"],
                    )
                    existing = next(
                        row
                        for row in existing_rows
                        if row["batch_ref"] == descriptor.batch_ref
                        and row["chunk_index"] == descriptor.chunk_index
                    )
                    if (
                        existing["session_id"] == descriptor.session_id
                        and existing["chunk_count"] == descriptor.chunk_count
                        and hmac.compare_digest(existing["chunk_digest"], descriptor.chunk_digest)
                        and existing["intake_count"] == len(authenticated)
                        and hmac.compare_digest(existing["intake_set_digest"], intake_set_digest)
                    ):
                        session = self._session_row(
                            descriptor.connection_id,
                            descriptor.session_id,
                        )
                        if session is None:
                            raise CaptureStoreIntegrityError()
                        event_head = self._head_row(
                            descriptor.connection_id,
                            descriptor.session_id,
                        )
                        verified_events = self._verify_chain(
                            descriptor.connection_id,
                            descriptor.session_id,
                            session,
                            event_head,
                        )
                        events_by_digest = {
                            event.intake.producer_event_digest: event for event in verified_events
                        }
                        if any(
                            (existing_event := events_by_digest.get(intake.producer_event_digest))
                            is None
                            or canonical_capture_intake(existing_event.intake) != encoded
                            for intake, encoded in zip(authenticated, canonical, strict=True)
                        ):
                            raise CaptureStoreIntegrityError()
                        self._connection.rollback()
                        self._revalidate_boundary()
                        return CaptureTransportReceipt(
                            disposition=CaptureTransportDisposition.REPLAYED,
                            connection_id=descriptor.connection_id,
                            session_id=descriptor.session_id,
                            batch_ref=descriptor.batch_ref,
                            chunk_index=descriptor.chunk_index,
                            chunk_count=descriptor.chunk_count,
                            intake_count=len(authenticated),
                            transport_ordinal=existing["transport_ordinal"],
                            previous_receipt_tag=existing["previous_receipt_tag"],
                            receipt_tag=existing["receipt_tag"],
                            incomplete_batch_count=incomplete,
                            event_count=session["event_count"],
                        )
                    affected = {
                        (existing["connection_id"], existing["session_id"]),
                        (descriptor.connection_id, descriptor.session_id),
                    }
                    incoming_session: sqlite3.Row | None = None
                    for affected_connection, affected_session in sorted(affected):
                        quarantined = self._quarantine_transport_session(
                            affected_connection,
                            affected_session,
                            code=CaptureHealthCode.PRODUCER_COLLISION,
                        )
                        if (affected_connection, affected_session) == (
                            descriptor.connection_id,
                            descriptor.session_id,
                        ):
                            incoming_session = quarantined
                    if incoming_session is None:
                        raise CaptureStoreIntegrityError()
                    receipt = self._transport_failure_receipt(
                        descriptor,
                        disposition=CaptureTransportDisposition.QUARANTINED,
                        intake_count=len(authenticated),
                        session=incoming_session,
                    )
                    self._connection.commit()
                    self._revalidate_boundary()
                    return receipt

                batch_candidates = self._connection.execute(
                    """
                    SELECT connection_id, session_id, chunk_count
                    FROM capture_transport_receipts
                    WHERE connection_id = ? AND batch_ref = ?
                    ORDER BY transport_ordinal
                    """,
                    (descriptor.connection_id, descriptor.batch_ref),
                ).fetchall()
                if batch_candidates:
                    candidate_sessions = {
                        (row["connection_id"], row["session_id"]) for row in batch_candidates
                    }
                    for candidate_connection, candidate_session in sorted(candidate_sessions):
                        self._verify_transport_chain(
                            candidate_connection,
                            candidate_session,
                        )
                    if any(
                        row["session_id"] != descriptor.session_id
                        or row["chunk_count"] != descriptor.chunk_count
                        for row in batch_candidates
                    ):
                        affected = candidate_sessions | {
                            (descriptor.connection_id, descriptor.session_id)
                        }
                        incoming_quarantined: sqlite3.Row | None = None
                        for affected_connection, affected_session in sorted(affected):
                            quarantined = self._quarantine_transport_session(
                                affected_connection,
                                affected_session,
                                code=CaptureHealthCode.PRODUCER_COLLISION,
                            )
                            if (affected_connection, affected_session) == (
                                descriptor.connection_id,
                                descriptor.session_id,
                            ):
                                incoming_quarantined = quarantined
                        if incoming_quarantined is None:
                            raise CaptureStoreIntegrityError()
                        receipt = self._transport_failure_receipt(
                            descriptor,
                            disposition=CaptureTransportDisposition.QUARANTINED,
                            intake_count=len(authenticated),
                            session=incoming_quarantined,
                        )
                        self._connection.commit()
                        self._revalidate_boundary()
                        return receipt

                existing_session = self._append_session_row(
                    descriptor.connection_id,
                    descriptor.session_id,
                )
                if existing_session is not None:
                    event_head = self._head_row(
                        descriptor.connection_id,
                        descriptor.session_id,
                    )
                    self._verify_append_commitment(
                        descriptor.connection_id,
                        descriptor.session_id,
                        existing_session,
                        event_head,
                    )
                    if existing_session["state"] != CaptureSessionState.OPEN.value:
                        health = self._verify_health_set(
                            descriptor.connection_id,
                            descriptor.session_id,
                            existing_session,
                        )
                        if existing_session["state"] == CaptureSessionState.QUARANTINED.value:
                            health_codes = {row["code"] for row in health}
                            disposition = (
                                CaptureTransportDisposition.OVERFLOW
                                if CaptureHealthCode.SESSION_OVERFLOW.value in health_codes
                                else CaptureTransportDisposition.QUARANTINED
                            )
                            receipt = self._transport_failure_receipt(
                                descriptor,
                                disposition=disposition,
                                intake_count=len(authenticated),
                                session=existing_session,
                            )
                            self._connection.rollback()
                            self._revalidate_boundary()
                            return receipt
                        session = self._quarantine_transport_session(
                            descriptor.connection_id,
                            descriptor.session_id,
                            code=CaptureHealthCode.PRODUCER_COLLISION,
                        )
                        receipt = self._transport_failure_receipt(
                            descriptor,
                            disposition=CaptureTransportDisposition.QUARANTINED,
                            intake_count=len(authenticated),
                            session=session,
                        )
                        self._connection.commit()
                        self._revalidate_boundary()
                        return receipt
                    _rows, _incomplete, existing_transport_head = self._verify_transport_chain(
                        descriptor.connection_id,
                        descriptor.session_id,
                    )
                    if existing_transport_head is None:
                        empty_transport_head_tag = self._transport_head_tag(
                            descriptor.connection_id,
                            descriptor.session_id,
                            receipt_count=0,
                            head_receipt_tag=None,
                        )
                        existing_session = self._update_session(
                            existing_session,
                            state=CaptureSessionState.OPEN,
                            event_count=existing_session["event_count"],
                            coverage_degraded=True,
                            transport_required=True,
                            transport_head_tag=empty_transport_head_tag,
                        )
                        self._create_transport_head(
                            descriptor.connection_id,
                            descriptor.session_id,
                            profile_id=connection["profile_id"],
                        )
                        _rows, _incomplete, existing_transport_head = self._verify_transport_chain(
                            descriptor.connection_id,
                            descriptor.session_id,
                            allow_pending_event_tail=True,
                        )
                    if existing_transport_head is None:
                        raise CaptureStoreIntegrityError()
                    if (
                        existing_transport_head["receipt_count"]
                        >= MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION
                    ):
                        session = self._quarantine_transport_session(
                            descriptor.connection_id,
                            descriptor.session_id,
                            code=CaptureHealthCode.SESSION_OVERFLOW,
                        )
                        receipt = self._transport_failure_receipt(
                            descriptor,
                            disposition=CaptureTransportDisposition.OVERFLOW,
                            intake_count=len(authenticated),
                            session=session,
                        )
                        self._connection.commit()
                        self._revalidate_boundary()
                        return receipt
                elif not authenticated or authenticated[0].kind != "session_started":
                    raise CaptureStoreStateError()

                self._connection.execute("SAVEPOINT capture_transport_intakes")
                for intake, encoded in zip(authenticated, canonical, strict=True):
                    prior_collision = self._connection.execute(
                        """
                        SELECT connection_id, session_id
                        FROM capture_events
                        WHERE connection_id = ? AND producer_event_digest = ?
                        """,
                        (intake.connection_id, intake.producer_event_digest),
                    ).fetchone()
                    appended = self._append_authenticated_in_transaction(
                        intake,
                        encoded,
                        source=CaptureAdmissionSource.DIRECT,
                        transport_bound=True,
                    )
                    if appended.disposition is CaptureAppendDisposition.ADMITTED:
                        assert appended.receipt_ordinal is not None
                        self._verified_append_heads[
                            (descriptor.connection_id, descriptor.session_id)
                        ] = (appended.receipt_ordinal, appended.event_tag)
                    if appended.disposition not in {
                        CaptureAppendDisposition.ADMITTED,
                        CaptureAppendDisposition.REPLAYED,
                    }:
                        self._connection.execute("ROLLBACK TO capture_transport_intakes")
                        self._connection.execute("RELEASE capture_transport_intakes")
                        code = (
                            CaptureHealthCode.SESSION_OVERFLOW
                            if appended.disposition is CaptureAppendDisposition.OVERFLOW
                            else CaptureHealthCode.PRODUCER_COLLISION
                        )
                        affected = {(descriptor.connection_id, descriptor.session_id)}
                        if (
                            appended.disposition is CaptureAppendDisposition.QUARANTINED
                            and prior_collision is not None
                        ):
                            affected.add(
                                (
                                    prior_collision["connection_id"],
                                    prior_collision["session_id"],
                                )
                            )
                        incoming_quarantine: sqlite3.Row | None = None
                        for affected_connection, affected_session in sorted(affected):
                            quarantined = self._quarantine_transport_session(
                                affected_connection,
                                affected_session,
                                code=code,
                            )
                            if (affected_connection, affected_session) == (
                                descriptor.connection_id,
                                descriptor.session_id,
                            ):
                                incoming_quarantine = quarantined
                        if incoming_quarantine is None:
                            raise CaptureStoreIntegrityError()
                        session = incoming_quarantine
                        disposition = (
                            CaptureTransportDisposition.OVERFLOW
                            if appended.disposition is CaptureAppendDisposition.OVERFLOW
                            else CaptureTransportDisposition.QUARANTINED
                        )
                        receipt = self._transport_failure_receipt(
                            descriptor,
                            disposition=disposition,
                            intake_count=len(authenticated),
                            session=session,
                        )
                        self._connection.commit()
                        self._revalidate_boundary()
                        return receipt
                session = self._session_row(
                    descriptor.connection_id,
                    descriptor.session_id,
                )
                if session is None:
                    raise CaptureStoreIntegrityError()
                transport_rows, _incomplete, transport_head = self._verify_transport_chain(
                    descriptor.connection_id,
                    descriptor.session_id,
                    allow_pending_event_tail=True,
                )
                if transport_head is None:
                    raise CaptureStoreIntegrityError()
                if transport_head["receipt_count"] >= MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION:
                    self._connection.execute("ROLLBACK TO capture_transport_intakes")
                    self._connection.execute("RELEASE capture_transport_intakes")
                    session = self._quarantine_transport_session(
                        descriptor.connection_id,
                        descriptor.session_id,
                        code=CaptureHealthCode.SESSION_OVERFLOW,
                    )
                    receipt = self._transport_failure_receipt(
                        descriptor,
                        disposition=CaptureTransportDisposition.OVERFLOW,
                        intake_count=len(authenticated),
                        session=session,
                    )
                    self._connection.commit()
                    self._revalidate_boundary()
                    return receipt
                self._connection.execute("RELEASE capture_transport_intakes")
                self._fault("transport_after_intake_admission")
                ordinal = transport_head["receipt_count"] + 1
                previous = transport_head["head_receipt_tag"]
                event_head = self._head_row(
                    descriptor.connection_id,
                    descriptor.session_id,
                )
                admitted_at = _now()
                material: dict[str, object] = {
                    "connection_id": descriptor.connection_id,
                    "session_id": descriptor.session_id,
                    "transport_ordinal": ordinal,
                    "batch_ref": descriptor.batch_ref,
                    "chunk_index": descriptor.chunk_index,
                    "chunk_count": descriptor.chunk_count,
                    "chunk_digest": descriptor.chunk_digest,
                    "intake_count": len(authenticated),
                    "intake_set_digest": intake_set_digest,
                    "post_event_count": session["event_count"],
                    "post_head_event_tag": event_head["head_event_tag"],
                    "previous_receipt_tag": previous,
                    "admitted_at": admitted_at,
                }
                receipt_tag = self._integrity.tag(
                    "transport_receipt",
                    _transport_receipt_material(material),
                )
                self._connection.execute(
                    """
                    INSERT INTO capture_transport_receipts(
                        connection_id, session_id, transport_ordinal,
                        batch_ref, chunk_index, chunk_count, chunk_digest,
                        intake_count, intake_set_digest, post_event_count,
                        post_head_event_tag, previous_receipt_tag, receipt_tag,
                        admitted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        descriptor.connection_id,
                        descriptor.session_id,
                        ordinal,
                        descriptor.batch_ref,
                        descriptor.chunk_index,
                        descriptor.chunk_count,
                        descriptor.chunk_digest,
                        len(authenticated),
                        intake_set_digest,
                        session["event_count"],
                        event_head["head_event_tag"],
                        previous,
                        receipt_tag,
                        admitted_at,
                    ),
                )
                self._fault("transport_after_receipt_insert")
                transport_head_tag = self._transport_head_tag(
                    descriptor.connection_id,
                    descriptor.session_id,
                    receipt_count=ordinal,
                    head_receipt_tag=receipt_tag,
                )
                updated = self._connection.execute(
                    """
                    UPDATE capture_transport_heads
                    SET receipt_count = ?, head_receipt_tag = ?, head_tag = ?
                    WHERE connection_id = ? AND session_id = ?
                    """,
                    (
                        ordinal,
                        receipt_tag,
                        transport_head_tag,
                        descriptor.connection_id,
                        descriptor.session_id,
                    ),
                )
                if updated.rowcount != 1:
                    raise CaptureStoreIntegrityError()
                session = self._update_session(
                    session,
                    state=CaptureSessionState(session["state"]),
                    event_count=session["event_count"],
                    coverage_degraded=bool(session["coverage_degraded"]),
                    transport_head_tag=transport_head_tag,
                )
                self._fault("transport_after_head_write")
                _verified_rows, incomplete, _verified_head = self._verify_transport_chain(
                    descriptor.connection_id,
                    descriptor.session_id,
                )
                if len(_verified_rows) != len(transport_rows) + 1:
                    raise CaptureStoreIntegrityError()
                self._fault("transport_before_commit")
                self._connection.commit()
                event_head = self._head_row(
                    descriptor.connection_id,
                    descriptor.session_id,
                )
                self._verified_append_heads[(descriptor.connection_id, descriptor.session_id)] = (
                    event_head["receipt_count"],
                    event_head["head_event_tag"],
                )
                self._fault("transport_after_commit")
            except BaseException:
                self._rollback()
                raise
            self._revalidate_boundary()
            return CaptureTransportReceipt(
                disposition=CaptureTransportDisposition.ADMITTED,
                connection_id=descriptor.connection_id,
                session_id=descriptor.session_id,
                batch_ref=descriptor.batch_ref,
                chunk_index=descriptor.chunk_index,
                chunk_count=descriptor.chunk_count,
                intake_count=len(authenticated),
                transport_ordinal=ordinal,
                previous_receipt_tag=previous,
                receipt_tag=receipt_tag,
                incomplete_batch_count=incomplete,
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

    def snapshot_session(
        self,
        connection_id: str,
        session_id: str,
    ) -> CaptureSessionSnapshot:
        """Return one authenticated, coherent MVCC session snapshot."""

        from saliencegate.capture.sessions import (
            CaptureSessionSnapshot,
            CaptureSnapshotEvent,
            CaptureSnapshotHealth,
            _authenticate_capture_session_snapshot,
        )

        self._ensure_open()
        if type(connection_id) is not str or type(session_id) is not str:
            raise CaptureStoreError()
        with self._lock:
            self._ensure_open()
            self._revalidate_boundary()
            try:
                # A deferred read transaction establishes an SQLite MVCC view on
                # the first SELECT without excluding a concurrent WAL writer.
                self._connection.execute("BEGIN")
                connection = self._connection_row(connection_id)
                if connection["state"] == CaptureConnectionState.DELETING.value:
                    raise CaptureStoreStateError()
                session = self._session_row(connection_id, session_id)
                if session is None:
                    raise CaptureStoreStateError()
                if session["state"] == CaptureSessionState.DELETING.value:
                    raise CaptureStoreStateError()
                if (
                    type(session["coverage_degraded"]) is not int
                    or session["coverage_degraded"] not in (0, 1)
                    or type(session["unattributed_drop"]) is not int
                    or session["unattributed_drop"] not in (0, 1)
                ):
                    raise CaptureStoreIntegrityError()
                head = self._head_row(connection_id, session_id)
                event_rows, health_rows = self._verify_chain_rows(
                    connection_id,
                    session_id,
                    session,
                    head,
                )
                transport_rows, incomplete_transport_batches, transport_head = (
                    self._verify_transport_chain(connection_id, session_id)
                )

                # Re-read the authenticated commitments in the same transaction.
                # A peer may have committed a newer revision, but this reader must
                # keep observing and returning one coherent older revision.
                checked_connection = self._connection_row(connection_id)
                checked_session = self._session_row(connection_id, session_id)
                checked_head = self._head_row(connection_id, session_id)
                checked_health = self._verify_health_set(
                    connection_id,
                    session_id,
                    session,
                )
                (
                    checked_transport_rows,
                    checked_incomplete_transport_batches,
                    checked_transport_head,
                ) = self._verify_transport_chain(connection_id, session_id)
                if (
                    checked_session is None
                    or checked_connection["row_tag"] != connection["row_tag"]
                    or checked_session["row_tag"] != session["row_tag"]
                    or checked_head["head_tag"] != head["head_tag"]
                    or tuple((row["marker_id"], row["row_tag"]) for row in checked_health)
                    != tuple((row["marker_id"], row["row_tag"]) for row in health_rows)
                    or checked_incomplete_transport_batches != incomplete_transport_batches
                    or tuple(
                        (row["transport_ordinal"], row["receipt_tag"])
                        for row in checked_transport_rows
                    )
                    != tuple(
                        (row["transport_ordinal"], row["receipt_tag"]) for row in transport_rows
                    )
                    or (
                        None
                        if checked_transport_head is None
                        else checked_transport_head["head_tag"]
                    )
                    != (None if transport_head is None else transport_head["head_tag"])
                ):
                    raise CaptureStoreIntegrityError()

                profile_id = CaptureProfile(connection["profile_id"])
                legacy_transport = (
                    connection["profile_id"] in _TRANSPORT_PROFILES
                    and session["transport_required"] == 0
                )
                validate_capture_capability_binding(
                    profile_id,
                    connection["capability_manifest_digest"],
                )
                snapshot = CaptureSessionSnapshot(
                    connection_id=connection["connection_id"],
                    project_digest=connection["project_digest"],
                    profile_id=profile_id,
                    capability_manifest_digest=connection["capability_manifest_digest"],
                    host_version=connection["host_version"],
                    compatibility_status=CompatibilityStatus(connection["compatibility_status"]),
                    connection_state=CaptureConnectionState(connection["state"]),
                    session_id=session["session_id"],
                    human_id=session["human_id"],
                    state=CaptureSessionState(session["state"]),
                    event_count=session["event_count"],
                    transport_receipt_count=len(transport_rows),
                    incomplete_transport_batch_count=incomplete_transport_batches,
                    coverage_degraded=(
                        bool(session["coverage_degraded"])
                        or incomplete_transport_batches > 0
                        or legacy_transport
                    ),
                    unattributed_drop=bool(session["unattributed_drop"]),
                    opened_at=_stored_timestamp(session["opened_at"]),
                    updated_at=_stored_timestamp(session["updated_at"]),
                    closed_at=(
                        None
                        if session["closed_at"] is None
                        else _stored_timestamp(session["closed_at"])
                    ),
                    events=tuple(
                        CaptureSnapshotEvent(
                            receipt_ordinal=event.receipt_ordinal,
                            admission_source=CaptureAdmissionSource(row["admission_source"]),
                            admitted_at=_stored_timestamp(row["admitted_at"]),
                            event=event,
                        )
                        for event, row in event_rows
                    ),
                    health=tuple(
                        CaptureSnapshotHealth(
                            code=CaptureHealthCode(row["code"]),
                            count=row["count"],
                            lower_bound=row["lower_bound"],
                            created_at=_stored_timestamp(row["created_at"]),
                            updated_at=_stored_timestamp(row["updated_at"]),
                        )
                        for row in health_rows
                    ),
                    spool_boundary_digest=self._snapshot_spool_boundary_digest(),
                    snapshot_digest="0" * 64,
                )
                authenticated = _authenticate_capture_session_snapshot(
                    snapshot,
                    context=self._context,
                )
                self._fault("snapshot_after_verification")
                self._connection.commit()
            except CaptureStoreError:
                self._rollback()
                raise
            except Exception:
                self._rollback()
                raise CaptureStoreIntegrityError() from None
            except BaseException:
                self._rollback()
                raise
            self._revalidate_boundary()
        return authenticated

    def _snapshot_spool_boundary_digest(self) -> str | None:
        """Observe the exact sibling spool directory without creating one."""

        authorization = self._authorization
        if isinstance(authorization, WindowsSQLiteAuthorization):  # pragma: no cover - R01
            state = authorization._parent
            spool_path = state.path / "capture-spool"
            try:
                if authorization._operations.inspect_path(spool_path) is None:
                    return None
                spool = authorize_windows_private_path(
                    spool_path,
                    kind=WindowsPathKind.DIRECTORY,
                    operations=authorization._operations,
                )
                digest = _capture_spool_boundary_digest(
                    state.security.identity,
                    spool.security.identity,
                    platform="windows",
                    context=self._context,
                )
                state.revalidate()
                spool.revalidate()
                return digest
            except (SecureFileError, WindowsSecurityError):
                return None

        parent_identity = authorization._parent_identity
        if parent_identity is None:
            return None
        posix_state = security_files._PrivateDirectoryAuthorization(
            path=os.fspath(Path(authorization.path).parent),
            _identity=parent_identity,
        )
        try:
            posix_spool = security_files._authorize_private_directory_child(
                posix_state,
                "capture-spool",
                create=False,
            )
            digest = _capture_spool_boundary_digest(
                posix_state._identity,
                posix_spool._identity,
                platform="posix",
                context=self._context,
            )
            posix_state.revalidate()
            posix_spool.revalidate()
            return digest
        except SecureFileError:
            return None


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
