"""Authenticated, bounded fallback spool for capture intake."""

from __future__ import annotations

import hmac
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Annotated, Final, Literal, Protocol, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

import saliencegate.security.files as security_files
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.capture.locations import (
    CaptureStoreLocations,
    _capture_spool_boundary_digest,
)
from saliencegate.capture.publication import verify_capture_intake_authentication
from saliencegate.capture.schema import (
    MAX_CAPTURE_EVENT_BYTES,
    CaptureIntake,
    canonical_capture_intake,
    load_capture_intake,
)
from saliencegate.domain import canonical_json
from saliencegate.domain.records import Sha256Digest
from saliencegate.security import (
    InstallationKey,
    SecureFileError,
    StableFileRead,
)
from saliencegate.security.windows import (
    NativeWindowsSecurityOperations,
    WindowsPathAuthorization,
    WindowsPathKind,
    WindowsStableFileRead,
    authorize_windows_private_path,
)

MAX_CAPTURE_SPOOL_BYTES: Final = 32 * 1_024 * 1_024
MAX_CAPTURE_SPOOL_EVENTS: Final = 10_000

_ENTRY_SUFFIX = ".capture-intake"
_ENTRY_NAME = re.compile(r"^[0-9a-f]{64}\.capture-intake$")
_ENTRY_HEADER = b"capture-spool/v1"
_ENTRY_DOMAIN = b"saliencegate:capture-spool:record:v1"
_SESSION_MARKER_SUFFIX = ".capture-session"
_SESSION_MARKER_NAME = re.compile(r"^[0-9a-f]{64}\.capture-session$")
_SESSION_MARKER_HEADER = b"capture-spool-session/v1"
_SESSION_MARKER_NAME_DOMAIN = b"saliencegate:capture-spool:session-name:v1"
_SESSION_MARKER_DOMAIN = b"saliencegate:capture-spool:session:v1"
_COMPONENT_IDENTIFIER = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._:/+\-]{0,255}$")
_HEALTH_NAME = ".capture-spool-health"
_HEALTH_HEADER = b"capture-spool-health/v1"
_HEALTH_DOMAIN = b"saliencegate:capture-spool:health:v1"
_OBSERVATION_DOMAIN = b"saliencegate:capture-spool:observation:v1"
_LOCK_NAME = ".capture-spool-lock"
_MAX_ENTRY_BYTES = MAX_CAPTURE_EVENT_BYTES + 256
_MAX_SESSION_MARKER_BYTES = 1_024
_MAX_HEALTH_BYTES = 4_096
_MAX_LOCK_BYTES = 4_096


def _safe_lock_metadata(value: os.stat_result) -> bool:
    return (
        stat.S_ISREG(value.st_mode)
        and stat.S_IMODE(value.st_mode) == 0o600
        and value.st_uid == security_files._current_user_id()
        and value.st_nlink == 1
    )


class CaptureSpoolError(RuntimeError):
    """A content-free capture spool operation failure."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture spool operation failed")


class CaptureSpoolIntegrityError(CaptureSpoolError):
    """A spooled record or health marker failed authentication."""

    __slots__ = ()

    def __init__(self) -> None:
        RuntimeError.__init__(self, "capture spool integrity check failed")


class CaptureSpoolUnavailableError(CaptureSpoolError):
    """The spool fence was unavailable before admission began."""

    __slots__ = ()


class CaptureSpoolObservationError(ValueError):
    """A spool observation failed validation or key-bound authentication."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture spool observation is invalid")


@dataclass(frozen=True, slots=True, repr=False)
class CaptureSpoolEnqueueReceipt:
    disposition: Literal["queued", "already_queued", "dropped_quota"]

    def __repr__(self) -> str:
        return "CaptureSpoolEnqueueReceipt(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class CaptureSpoolDrainReceipt:
    admitted_events: int
    remaining_events: int

    def __repr__(self) -> str:
        return "CaptureSpoolDrainReceipt(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class CaptureSpoolHealth:
    queued_events: int
    queued_bytes: int
    dropped_events: int
    coverage_degraded: bool
    last_drop_reason: Literal["spool_incomplete", "spool_quota"] | None

    def __repr__(self) -> str:
        return "CaptureSpoolHealth(<redacted>)"


class CaptureSpoolObservation(BaseModel):
    """A deterministic, snapshot-bound proof of one locked spool health view."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    schema_version: Literal["capture-spool-observation/v1"] = "capture-spool-observation/v1"
    snapshot_digest: Sha256Digest = Field(repr=False)
    spool_boundary_digest: Sha256Digest = Field(repr=False)
    queued_events: Annotated[int, Field(ge=0, le=MAX_CAPTURE_SPOOL_EVENTS)]
    queued_bytes: Annotated[int, Field(ge=0, le=MAX_CAPTURE_SPOOL_BYTES)]
    dropped_events: Annotated[int, Field(ge=0, le=(1 << 63) - 1)]
    coverage_degraded: bool
    last_drop_reason: Literal["spool_incomplete", "spool_quota"] | None
    observation_tag: Sha256Digest = Field(repr=False)

    @model_validator(mode="after")
    def health_state_is_exact(self) -> Self:
        if (
            type(self.snapshot_digest) is not str
            or type(self.spool_boundary_digest) is not str
            or type(self.queued_events) is not int
            or type(self.queued_bytes) is not int
            or type(self.dropped_events) is not int
            or type(self.coverage_degraded) is not bool
            or (self.last_drop_reason is not None and type(self.last_drop_reason) is not str)
            or type(self.observation_tag) is not str
            or ((self.queued_events == 0) != (self.queued_bytes == 0))
            or self.coverage_degraded is not (self.dropped_events > 0)
            or ((self.dropped_events == 0) != (self.last_drop_reason is None))
        ):
            raise ValueError("capture spool observation health is invalid")
        return self

    @property
    def health(self) -> CaptureSpoolHealth:
        """Return a fresh compatibility DTO for the authenticated health values."""

        return CaptureSpoolHealth(
            queued_events=self.queued_events,
            queued_bytes=self.queued_bytes,
            dropped_events=self.dropped_events,
            coverage_degraded=self.coverage_degraded,
            last_drop_reason=self.last_drop_reason,
        )

    def __repr__(self) -> str:
        return "CaptureSpoolObservation(<redacted>)"

    __str__ = __repr__


def _validated_spool_observation(value: object) -> CaptureSpoolObservation:
    if type(value) is CaptureSpoolObservation:
        value = value.model_dump(mode="python", warnings="error")
    return CaptureSpoolObservation.model_validate(value)


def _spool_observation_preimage(observation: CaptureSpoolObservation) -> bytes:
    return canonical_json(
        {
            "schema_version": "capture-spool-observation-integrity/v1",
            "observation": observation.model_dump(
                mode="json",
                exclude={"observation_tag"},
                warnings="error",
            ),
        }
    )


def _seal_spool_observation(
    health: CaptureSpoolHealth,
    *,
    snapshot_digest: str,
    spool_boundary_digest: str,
    installation_key: InstallationKey,
) -> CaptureSpoolObservation:
    result: CaptureSpoolObservation | None = None
    try:
        if type(health) is not CaptureSpoolHealth or type(installation_key) is not InstallationKey:
            raise CaptureSpoolObservationError()
        draft = CaptureSpoolObservation(
            snapshot_digest=snapshot_digest,
            spool_boundary_digest=spool_boundary_digest,
            queued_events=health.queued_events,
            queued_bytes=health.queued_bytes,
            dropped_events=health.dropped_events,
            coverage_degraded=health.coverage_degraded,
            last_drop_reason=health.last_drop_reason,
            observation_tag="0" * 64,
        )
        tag = installation_key._hmac_sha256(
            _spool_observation_preimage(draft),
            domain=_OBSERVATION_DOMAIN,
        )
        result = _validated_spool_observation(draft.model_copy(update={"observation_tag": tag}))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        result = None
    if result is None:
        raise CaptureSpoolObservationError()
    return result


def verify_capture_spool_observation(
    observation: object,
    *,
    expected_snapshot_digest: str,
    expected_spool_boundary_digest: str,
    installation_key: InstallationKey,
) -> CaptureSpoolObservation:
    """Defensively copy and authenticate one snapshot-bound spool observation."""

    result: CaptureSpoolObservation | None = None
    try:
        if (
            type(expected_snapshot_digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_snapshot_digest) is None
            or type(expected_spool_boundary_digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_spool_boundary_digest) is None
            or type(installation_key) is not InstallationKey
        ):
            raise CaptureSpoolObservationError()
        validated = _validated_spool_observation(observation)
        if not hmac.compare_digest(validated.snapshot_digest, expected_snapshot_digest):
            raise CaptureSpoolObservationError()
        if not hmac.compare_digest(
            validated.spool_boundary_digest,
            expected_spool_boundary_digest,
        ):
            raise CaptureSpoolObservationError()
        expected = installation_key._hmac_sha256(
            _spool_observation_preimage(validated),
            domain=_OBSERVATION_DOMAIN,
        )
        if not hmac.compare_digest(validated.observation_tag, expected):
            raise CaptureSpoolObservationError()
        result = validated
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        result = None
    if result is None:
        raise CaptureSpoolObservationError()
    return result


class _AppendStore(Protocol):
    def append(self, intake: CaptureIntake) -> object: ...


_SpoolFileRead = StableFileRead | WindowsStableFileRead
_SpoolDirectoryIdentity = security_files._PrivateDirectoryAuthorization | WindowsPathAuthorization
_SpoolEntry = tuple[Path, CaptureIntake, _SpoolFileRead]
_SpoolSessionKey = tuple[str, str]
_SpoolSessionMarkerState = Literal["acknowledged", "pending"]
_SpoolSessionMarker = tuple[Path, _SpoolSessionMarkerState, _SpoolFileRead]


class CaptureSpoolMaintenance:
    """One short-lived exclusive spool lease for lifecycle maintenance."""

    __slots__ = ("_active", "_directory_fd", "_spool")

    def __init__(self, spool: CaptureSpool, directory_fd: int | None) -> None:
        self._spool = spool
        self._directory_fd = directory_fd
        self._active = True

    def __repr__(self) -> str:
        return "CaptureSpoolMaintenance(<redacted>)"

    def drain(self, store: _AppendStore) -> CaptureSpoolDrainReceipt:
        if not self._active:
            raise CaptureSpoolError()
        return self._spool._drain_locked(store, self._directory_fd)

    def clear_drop_health_if_empty(self) -> bool:
        """Clear global quota-drop health only after the queue is empty."""

        if not self._active:
            raise CaptureSpoolError()
        return self._spool._clear_drop_health_if_empty_locked(self._directory_fd)

    def _close(self) -> None:
        self._active = False


class CaptureSpool:
    """One installation-key-bound spool directory."""

    __slots__ = (
        "_boundary_digest",
        "_context",
        "_key",
        "_locations",
        "_spool_identity",
        "_state_identity",
        "_windows_operations",
    )

    _context: CaptureDigestContext
    _boundary_digest: str
    _key: InstallationKey
    _locations: CaptureStoreLocations
    _spool_identity: _SpoolDirectoryIdentity
    _state_identity: _SpoolDirectoryIdentity
    _windows_operations: NativeWindowsSecurityOperations | None

    def __init__(self) -> None:
        raise CaptureSpoolError()

    @classmethod
    def open(
        cls,
        locations: CaptureStoreLocations,
        installation_key: InstallationKey,
    ) -> CaptureSpool:
        """Open or create the exact owner-private spool boundary."""

        return cls._open(locations, installation_key, create=True)

    @classmethod
    def _open(
        cls,
        locations: CaptureStoreLocations,
        installation_key: InstallationKey,
        *,
        create: bool,
    ) -> CaptureSpool:
        """Authorize one exact spool boundary with explicit creation policy."""

        if (
            type(locations) is not CaptureStoreLocations
            or type(installation_key) is not InstallationKey
            or type(create) is not bool
        ):
            raise CaptureSpoolError()
        result: CaptureSpool | None = None
        failed = False
        try:
            if locations.platform != ("windows" if os.name == "nt" else "posix"):
                raise SecureFileError()
            windows_operations: NativeWindowsSecurityOperations | None = None
            state_identity: _SpoolDirectoryIdentity
            spool_identity: _SpoolDirectoryIdentity
            if locations.platform == "windows":  # pragma: no cover - native Windows R01
                windows_operations = NativeWindowsSecurityOperations()
                state_identity = authorize_windows_private_path(
                    PureWindowsPath(str(locations.state_directory)),
                    kind=WindowsPathKind.DIRECTORY,
                    operations=windows_operations,
                    create=create,
                )
                spool_identity = authorize_windows_private_path(
                    PureWindowsPath(str(locations.spool_directory)),
                    kind=WindowsPathKind.DIRECTORY,
                    operations=windows_operations,
                    create=create,
                )
                state_identity.revalidate()
                spool_identity.revalidate()
            else:
                state_identity = security_files._authorize_private_directory(
                    locations.state_directory,
                    create=create,
                )
                spool_identity = security_files._authorize_private_directory_child(
                    state_identity,
                    locations.spool_directory.name,
                    create=create,
                )
                state_identity.revalidate()
                spool_identity.revalidate()
            instance = cls.__new__(cls)
            instance._locations = locations
            instance._key = installation_key._copy()
            instance._context = CaptureDigestContext(installation_key)
            instance._boundary_digest = _capture_spool_boundary_digest(
                (
                    state_identity.security.identity
                    if isinstance(state_identity, WindowsPathAuthorization)
                    else state_identity._identity
                ),
                (
                    spool_identity.security.identity
                    if isinstance(spool_identity, WindowsPathAuthorization)
                    else spool_identity._identity
                ),
                platform=locations.platform,
                context=instance._context,
            )
            instance._state_identity = state_identity
            instance._spool_identity = spool_identity
            instance._windows_operations = windows_operations
            result = instance
        except Exception:
            failed = True
        if failed or result is None:
            raise CaptureSpoolError()
        return result

    @classmethod
    def audit_read_only(
        cls,
        locations: CaptureStoreLocations,
        *,
        installation_key: InstallationKey,
    ) -> CaptureSpoolHealth:
        """Authenticate a quiescent existing spool without creating or draining."""

        try:
            instance = cls._open(locations, installation_key, create=False)
            return instance._audit_read_only()
        except CaptureSpoolError:
            raise
        except Exception:
            raise CaptureSpoolError() from None

    def __repr__(self) -> str:
        return "CaptureSpool(<redacted>)"

    def _revalidate(self) -> None:
        if self._windows_operations is not None:  # pragma: no cover - native Windows R01
            if (
                type(self._state_identity) is not WindowsPathAuthorization
                or type(self._spool_identity) is not WindowsPathAuthorization
            ):
                raise SecureFileError()
            self._state_identity.revalidate()
            self._spool_identity.revalidate()
            return
        if (
            type(self._state_identity) is not security_files._PrivateDirectoryAuthorization
            or type(self._spool_identity) is not security_files._PrivateDirectoryAuthorization
        ):
            raise SecureFileError()
        self._state_identity.revalidate()
        self._spool_identity.revalidate()

    @contextmanager
    def _locked(self, *, blocking: bool = True) -> Iterator[int | None]:
        if type(blocking) is not bool:
            raise CaptureSpoolError()
        self._revalidate()
        lock_path = self._locations.spool_directory / _LOCK_NAME
        if self._windows_operations is not None:  # pragma: no cover - native Windows R01
            with self._windows_operations.private_file_lock(
                PureWindowsPath(str(lock_path)),
                blocking=blocking,
            ):
                self._revalidate()
                yield None
                self._revalidate()
            return
        if type(self._spool_identity) is not security_files._PrivateDirectoryAuthorization:
            raise SecureFileError()
        directory_fd = security_files._open_authorized_private_directory(self._spool_identity)
        flags = os.O_RDWR | os.O_CREAT
        for name in ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK"):
            flags |= cast(int, getattr(os, name, 0))
        descriptor: int | None = None
        operation_failure: BaseException | None = None
        try:
            descriptor = os.open(_LOCK_NAME, flags, 0o600, dir_fd=directory_fd)
            metadata = os.fstat(descriptor)
            security_files._require_safe_acl(descriptor)
            if not _safe_lock_metadata(metadata):
                raise SecureFileError()
            import fcntl

            fcntl.flock(
                descriptor,
                fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB),
            )
            security_files._require_safe_acl(descriptor)
            named = os.stat(_LOCK_NAME, dir_fd=directory_fd, follow_symlinks=False)
            opened_locked = os.fstat(descriptor)
            if (
                (named.st_dev, named.st_ino) != (opened_locked.st_dev, opened_locked.st_ino)
                or not _safe_lock_metadata(named)
                or not _safe_lock_metadata(opened_locked)
            ):
                raise SecureFileError()
            self._revalidate()
            security_files._require_authorized_private_directory_descriptor(
                self._spool_identity,
                directory_fd,
            )
            yield directory_fd
            named_after = os.stat(_LOCK_NAME, dir_fd=directory_fd, follow_symlinks=False)
            opened_after = os.fstat(descriptor)
            security_files._require_safe_acl(descriptor)
            if (
                (named_after.st_dev, named_after.st_ino)
                != (opened_after.st_dev, opened_after.st_ino)
                or not _safe_lock_metadata(named_after)
                or not _safe_lock_metadata(opened_after)
            ):
                raise SecureFileError()
            self._revalidate()
        except BaseException as error:
            operation_failure = error
        finally:
            cleanup_failure: BaseException | None = None
            if descriptor is not None:
                try:
                    with suppress(OSError):
                        import fcntl

                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                except BaseException as error:
                    cleanup_failure = error
            close_failure = security_files._close_independent_descriptors(
                descriptor,
                directory_fd,
            )
            cleanup_failure = security_files._preferred_failure(
                cleanup_failure,
                close_failure,
            )
            operation_failure = security_files._preferred_failure(
                operation_failure,
                cleanup_failure,
            )
        if operation_failure is not None:
            raise operation_failure

    def _record_tag(self, intake_bytes: bytes) -> str:
        return self._key._hmac_sha256(intake_bytes, domain=_ENTRY_DOMAIN)

    def _frame(self, intake: CaptureIntake) -> tuple[str, bytes]:
        verified = verify_capture_intake_authentication(intake, context=self._context)
        intake_bytes = canonical_capture_intake(verified)
        tag = self._record_tag(intake_bytes)
        return tag, b"\n".join((_ENTRY_HEADER, tag.encode("ascii"), intake_bytes))

    def _session_marker_name_tag(
        self,
        connection_id: str,
        session_id: str,
    ) -> str:
        if (
            type(connection_id) is not str
            or _COMPONENT_IDENTIFIER.fullmatch(connection_id) is None
            or type(session_id) is not str
            or re.fullmatch(r"[0-9a-f]{64}", session_id) is None
        ):
            raise CaptureSpoolIntegrityError()
        identity = canonical_json(
            {
                "schema_version": "capture-spool-session-name/v1",
                "connection_id": connection_id,
                "session_id": session_id,
            }
        )
        return self._key._hmac_sha256(identity, domain=_SESSION_MARKER_NAME_DOMAIN)

    def _session_marker_frame(
        self,
        connection_id: str,
        session_id: str,
        state: _SpoolSessionMarkerState = "pending",
    ) -> tuple[str, bytes]:
        name_tag = self._session_marker_name_tag(connection_id, session_id)
        if state not in ("acknowledged", "pending"):
            raise CaptureSpoolIntegrityError()
        payload = canonical_json(
            {
                "schema_version": "capture-spool-session/v1",
                "connection_id": connection_id,
                "session_id": session_id,
                "state": state,
            }
        )
        content_tag = self._key._hmac_sha256(payload, domain=_SESSION_MARKER_DOMAIN)
        framed = b"\n".join((_SESSION_MARKER_HEADER, content_tag.encode("ascii"), payload))
        return name_tag, framed

    def _read_spool_file(
        self,
        path: Path,
        directory_fd: int | None,
        *,
        maximum_bytes: int,
    ) -> _SpoolFileRead:
        if self._windows_operations is not None:  # pragma: no cover - native Windows R01
            return self._windows_operations.read_private_file(
                PureWindowsPath(str(path)),
                maximum_bytes=maximum_bytes,
            )
        if (
            type(directory_fd) is not int
            or type(self._spool_identity) is not security_files._PrivateDirectoryAuthorization
        ):
            raise SecureFileError()
        return security_files._read_private_file_at_descriptor(
            self._spool_identity,
            directory_fd,
            path.name,
            maximum_bytes,
        )

    def _named_spool_file_exists(self, name: str, directory_fd: int | None) -> bool:
        path = self._locations.spool_directory / name
        if self._windows_operations is not None:  # pragma: no cover - native Windows R01
            return self._windows_operations.inspect_path(PureWindowsPath(str(path))) is not None
        if type(directory_fd) is not int:
            raise SecureFileError()
        try:
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True

    def _read_session_marker(
        self,
        connection_id: str,
        session_id: str,
        directory_fd: int | None,
    ) -> _SpoolSessionMarker | None:
        tag = self._session_marker_name_tag(connection_id, session_id)
        path = self._locations.spool_directory / f"{tag}{_SESSION_MARKER_SUFFIX}"
        if not self._named_spool_file_exists(path.name, directory_fd):
            return None
        decoded = self._decode_session_marker(path, directory_fd)
        if decoded is None:
            raise CaptureSpoolIntegrityError()
        key, state, stable = decoded
        if key != (connection_id, session_id):
            raise CaptureSpoolIntegrityError()
        return path, state, stable

    def _decode_session_marker(
        self,
        path: Path,
        directory_fd: int | None,
    ) -> tuple[_SpoolSessionKey, _SpoolSessionMarkerState, _SpoolFileRead] | None:
        try:
            stable = self._read_spool_file(
                path,
                directory_fd,
                maximum_bytes=_MAX_SESSION_MARKER_BYTES,
            )
            parts = stable.data.split(b"\n", maxsplit=2)
            if len(parts) != 3 or parts[0] != _SESSION_MARKER_HEADER:
                return None
            encoded_tag, payload = parts[1:]
            if re.fullmatch(rb"[0-9a-f]{64}", encoded_tag) is None:
                return None
            expected = self._key._hmac_sha256(payload, domain=_SESSION_MARKER_DOMAIN)
            if not hmac.compare_digest(encoded_tag.decode("ascii"), expected):
                return None
            import json

            value = json.loads(payload)
            if (
                type(value) is not dict
                or set(value) != {"schema_version", "connection_id", "session_id", "state"}
                or value.get("schema_version") != "capture-spool-session/v1"
                or type(value.get("connection_id")) is not str
                or _COMPONENT_IDENTIFIER.fullmatch(value["connection_id"]) is None
                or type(value.get("session_id")) is not str
                or re.fullmatch(r"[0-9a-f]{64}", value["session_id"]) is None
                or value.get("state") not in ("acknowledged", "pending")
                or canonical_json(value) != payload
            ):
                return None
            name_tag = self._session_marker_name_tag(
                value["connection_id"],
                value["session_id"],
            )
            if path.name != f"{name_tag}{_SESSION_MARKER_SUFFIX}":
                return None
            stable.authorization.revalidate()
            return (
                (value["connection_id"], value["session_id"]),
                cast(_SpoolSessionMarkerState, value["state"]),
                stable,
            )
        except Exception:
            return None

    def _audit_lock_boundary(self, directory_fd: int | None) -> _SpoolFileRead | None:
        path = self._locations.spool_directory / _LOCK_NAME
        if self._windows_operations is not None:  # pragma: no cover - native Windows R01
            windows_path = PureWindowsPath(str(path))
            if self._windows_operations.inspect_path(windows_path) is None:
                return None
            stable: _SpoolFileRead = self._windows_operations.read_private_file(
                windows_path,
                maximum_bytes=_MAX_LOCK_BYTES,
            )
        else:
            if (
                type(directory_fd) is not int
                or type(self._spool_identity) is not security_files._PrivateDirectoryAuthorization
            ):
                raise SecureFileError()
            try:
                named = os.stat(_LOCK_NAME, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                return None
            if not _safe_lock_metadata(named):
                raise SecureFileError()
            stable = security_files._read_private_file_at_descriptor(
                self._spool_identity,
                directory_fd,
                _LOCK_NAME,
                _MAX_LOCK_BYTES,
            )
            named_after = os.stat(_LOCK_NAME, dir_fd=directory_fd, follow_symlinks=False)
            if not _safe_lock_metadata(named_after):
                raise SecureFileError()
        stable.authorization.revalidate()
        return stable

    def _audit_read_only(self) -> CaptureSpoolHealth:
        directory_fd: int | None = None
        try:
            self._revalidate()
            if self._windows_operations is None:
                if type(self._spool_identity) is not security_files._PrivateDirectoryAuthorization:
                    raise SecureFileError()
                directory_fd = security_files._open_authorized_private_directory(
                    self._spool_identity
                )
            initial_child_names = tuple(path.name for path in self._spool_child_paths(directory_fd))
            lock = self._audit_lock_boundary(directory_fd)
            entries = self._entries(directory_fd)
            markers = self._session_markers(directory_fd)
            orphan_markers = self._reconcile_session_markers(entries, markers)
            (
                dropped_events,
                last_drop_reason,
                acknowledged_marker,
                health,
            ) = self._drop_state_record(directory_fd)
            for _path, _intake, stable in entries:
                stable.authorization.revalidate()
            if health is not None:
                health.authorization.revalidate()
            if lock is not None:
                lock.authorization.revalidate()
            for _path, _state, marker in markers.values():
                marker.authorization.revalidate()
            final_child_names = tuple(path.name for path in self._spool_child_paths(directory_fd))
            if final_child_names != initial_child_names:
                raise CaptureSpoolIntegrityError()
            self._revalidate()
            return self._health_from_state(
                entries,
                dropped_events=dropped_events,
                last_drop_reason=last_drop_reason,
                acknowledged_marker=acknowledged_marker,
                markers=markers,
                orphan_markers=orphan_markers,
            )
        except CaptureSpoolError:
            raise
        except Exception:
            raise CaptureSpoolError() from None
        finally:
            if directory_fd is not None:
                with suppress(OSError):
                    os.close(directory_fd)

    def _spool_child_paths(self, directory_fd: int | None) -> tuple[Path, ...]:
        if self._windows_operations is not None:  # pragma: no cover - native Windows R01
            candidates = tuple(sorted(self._locations.spool_directory.iterdir()))
        else:
            if (
                type(directory_fd) is not int
                or type(self._spool_identity) is not security_files._PrivateDirectoryAuthorization
            ):
                raise SecureFileError()
            security_files._require_authorized_private_directory_descriptor(
                self._spool_identity,
                directory_fd,
            )
            candidates = tuple(
                self._locations.spool_directory / name for name in sorted(os.listdir(directory_fd))
            )
        for path in candidates:
            if (
                _ENTRY_NAME.fullmatch(path.name) is None
                and _SESSION_MARKER_NAME.fullmatch(path.name) is None
                and path.name not in {_HEALTH_NAME, _LOCK_NAME}
            ):
                raise CaptureSpoolIntegrityError()
        if self._windows_operations is None:
            if (
                type(directory_fd) is not int
                or type(self._spool_identity) is not security_files._PrivateDirectoryAuthorization
            ):
                raise SecureFileError()
            security_files._require_authorized_private_directory_descriptor(
                self._spool_identity,
                directory_fd,
            )
        return candidates

    def _entry_paths(self, directory_fd: int | None) -> tuple[Path, ...]:
        return tuple(
            path
            for path in self._spool_child_paths(directory_fd)
            if _ENTRY_NAME.fullmatch(path.name)
        )

    def _session_marker_paths(self, directory_fd: int | None) -> tuple[Path, ...]:
        return tuple(
            path
            for path in self._spool_child_paths(directory_fd)
            if _SESSION_MARKER_NAME.fullmatch(path.name)
        )

    def _decode_entry(
        self,
        path: Path,
        directory_fd: int | None,
    ) -> tuple[CaptureIntake, _SpoolFileRead] | None:
        try:
            if self._windows_operations is not None:  # pragma: no cover - native Windows R01
                stable: _SpoolFileRead = self._windows_operations.read_private_file(
                    PureWindowsPath(str(path)),
                    maximum_bytes=_MAX_ENTRY_BYTES,
                )
            else:
                if (
                    type(directory_fd) is not int
                    or type(self._spool_identity)
                    is not security_files._PrivateDirectoryAuthorization
                ):
                    return None
                stable = security_files._read_private_file_at_descriptor(
                    self._spool_identity,
                    directory_fd,
                    path.name,
                    _MAX_ENTRY_BYTES,
                )
            parts = stable.data.split(b"\n", maxsplit=2)
            if len(parts) != 3 or parts[0] != _ENTRY_HEADER:
                return None
            encoded_tag, intake_bytes = parts[1:]
            if re.fullmatch(rb"[0-9a-f]{64}", encoded_tag) is None:
                return None
            tag = encoded_tag.decode("ascii")
            if path.name != f"{tag}{_ENTRY_SUFFIX}":
                return None
            expected = self._record_tag(intake_bytes)
            if not hmac.compare_digest(tag, expected):
                return None
            intake = load_capture_intake(intake_bytes)
            verified = verify_capture_intake_authentication(intake, context=self._context)
            if canonical_capture_intake(verified) != intake_bytes:
                return None
            stable.authorization.revalidate()
            return verified, stable
        except Exception:
            return None

    def _entries(
        self,
        directory_fd: int | None,
    ) -> tuple[_SpoolEntry, ...]:
        result: list[_SpoolEntry] = []
        for path in self._entry_paths(directory_fd):
            decoded = self._decode_entry(path, directory_fd)
            if decoded is None:
                raise CaptureSpoolIntegrityError()
            intake, stable = decoded
            result.append((path, intake, stable))
        return tuple(sorted(result, key=self._drain_order_key))

    def _session_markers(
        self,
        directory_fd: int | None,
    ) -> dict[_SpoolSessionKey, _SpoolSessionMarker]:
        result: dict[_SpoolSessionKey, _SpoolSessionMarker] = {}
        for path in self._session_marker_paths(directory_fd):
            decoded = self._decode_session_marker(path, directory_fd)
            if decoded is None:
                raise CaptureSpoolIntegrityError()
            key, state, stable = decoded
            if key in result:
                raise CaptureSpoolIntegrityError()
            result[key] = (path, state, stable)
        return result

    @staticmethod
    def _reconcile_session_markers(
        entries: tuple[_SpoolEntry, ...],
        markers: dict[_SpoolSessionKey, _SpoolSessionMarker],
    ) -> set[_SpoolSessionKey]:
        queued_sessions = {
            (intake.connection_id, intake.session_id) for _path, intake, _stable in entries
        }
        if not queued_sessions.issubset(markers):
            raise CaptureSpoolIntegrityError()
        return set(markers).difference(queued_sessions)

    @staticmethod
    def _drain_order_key(
        entry: _SpoolEntry,
    ) -> tuple[str, str, int, int, int, str, str]:
        path, intake, _stable = entry
        lifecycle_rank = (
            0 if intake.kind == "session_started" else 2 if intake.kind == "session_finished" else 1
        )
        if intake.producer_sequence is not None:
            authority_rank = 0
            producer_sequence = intake.producer_sequence
            occurred_at = ""
        elif intake.occurred_at is not None:
            authority_rank = 1
            producer_sequence = 0
            occurred_at = intake.occurred_at.isoformat()
        else:
            authority_rank = 2
            producer_sequence = 0
            occurred_at = ""
        return (
            intake.connection_id,
            intake.session_id,
            lifecycle_rank,
            authority_rank,
            producer_sequence,
            occurred_at,
            path.name,
        )

    def _health_path(self) -> Path:
        return self._locations.spool_directory / _HEALTH_NAME

    def _decode_drop_state(
        self,
        data: bytes,
    ) -> tuple[int, str | None, str | None] | None:
        try:
            parts = data.split(b"\n", maxsplit=2)
            if len(parts) != 3 or parts[0] != _HEALTH_HEADER:
                return None
            encoded_tag, payload = parts[1:]
            if re.fullmatch(rb"[0-9a-f]{64}", encoded_tag) is None:
                return None
            expected = self._key._hmac_sha256(payload, domain=_HEALTH_DOMAIN)
            if not hmac.compare_digest(encoded_tag.decode("ascii"), expected):
                return None
            import json

            value = json.loads(payload)
            if (
                type(value) is not dict
                or set(value)
                != {
                    "schema_version",
                    "dropped_events",
                    "last_drop_reason",
                    "acknowledged_marker",
                }
                or value.get("schema_version") != "capture-spool-health/v1"
                or type(value.get("dropped_events")) is not int
                or value["dropped_events"] < 0
                or value.get("last_drop_reason") not in (None, "spool_incomplete", "spool_quota")
                or (
                    value.get("acknowledged_marker") is not None
                    and (
                        type(value["acknowledged_marker"]) is not str
                        or re.fullmatch(r"[0-9a-f]{64}", value["acknowledged_marker"]) is None
                    )
                )
                or ((value["dropped_events"] == 0) != (value["last_drop_reason"] is None))
                or (value["dropped_events"] == 0 and value["acknowledged_marker"] is not None)
                or canonical_json(value) != payload
            ):
                return None
            return (
                value["dropped_events"],
                value["last_drop_reason"],
                value["acknowledged_marker"],
            )
        except Exception:
            return None

    def _drop_state_record(
        self,
        directory_fd: int | None,
    ) -> tuple[int, str | None, str | None, _SpoolFileRead | None]:
        path = self._health_path()
        stable: _SpoolFileRead | None = None
        try:
            if self._windows_operations is not None:  # pragma: no cover - native Windows R01
                windows_path = PureWindowsPath(str(path))
                if self._windows_operations.inspect_path(windows_path) is None:
                    return 0, None, None, None
                stable = self._windows_operations.read_private_file(
                    windows_path,
                    maximum_bytes=_MAX_HEALTH_BYTES,
                )
            else:
                if (
                    type(directory_fd) is not int
                    or type(self._spool_identity)
                    is not security_files._PrivateDirectoryAuthorization
                ):
                    raise SecureFileError()
                try:
                    os.stat(_HEALTH_NAME, dir_fd=directory_fd, follow_symlinks=False)
                except FileNotFoundError:
                    return 0, None, None, None
                stable = security_files._read_private_file_at_descriptor(
                    self._spool_identity,
                    directory_fd,
                    _HEALTH_NAME,
                    _MAX_HEALTH_BYTES,
                )
        except Exception:
            stable = None
        if stable is None:
            raise CaptureSpoolIntegrityError()
        state = self._decode_drop_state(stable.data)
        if state is None:
            raise CaptureSpoolIntegrityError()
        stable.authorization.revalidate()
        return state[0], state[1], state[2], stable

    def _drop_state(self, directory_fd: int | None) -> tuple[int, str | None, str | None]:
        dropped_events, last_drop_reason, acknowledged_marker, _stable = self._drop_state_record(
            directory_fd
        )
        return dropped_events, last_drop_reason, acknowledged_marker

    def _write_drop_state(
        self,
        dropped_events: int,
        reason: str,
        directory_fd: int | None,
        *,
        acknowledged_marker: str | None = None,
    ) -> None:
        if (
            type(dropped_events) is not int
            or dropped_events < 1
            or reason not in ("spool_incomplete", "spool_quota")
            or (
                acknowledged_marker is not None
                and (
                    type(acknowledged_marker) is not str
                    or re.fullmatch(r"[0-9a-f]{64}", acknowledged_marker) is None
                )
            )
        ):
            raise CaptureSpoolIntegrityError()
        payload = canonical_json(
            {
                "schema_version": "capture-spool-health/v1",
                "dropped_events": dropped_events,
                "last_drop_reason": reason,
                "acknowledged_marker": acknowledged_marker,
            }
        )
        tag = self._key._hmac_sha256(payload, domain=_HEALTH_DOMAIN)
        framed = b"\n".join((_HEALTH_HEADER, tag.encode("ascii"), payload))
        path = self._health_path()
        if self._windows_operations is not None:  # pragma: no cover - native Windows R01
            self._windows_operations.publish_private_file(
                PureWindowsPath(str(path)),
                framed,
                maximum_bytes=_MAX_HEALTH_BYTES,
                validate_replacement=lambda value: self._decode_drop_state(value) is not None,
                validate_published=lambda value: (
                    self._decode_drop_state(value) == (dropped_events, reason, acknowledged_marker)
                ),
            )
        else:
            if (
                type(directory_fd) is not int
                or type(self._spool_identity) is not security_files._PrivateDirectoryAuthorization
            ):
                raise SecureFileError()
            security_files._publish_private_file_at_descriptor(
                self._spool_identity,
                directory_fd,
                _HEALTH_NAME,
                framed,
                maximum_bytes=_MAX_HEALTH_BYTES,
                validate_replacement=lambda value: self._decode_drop_state(value) is not None,
                validate_published=lambda value: (
                    self._decode_drop_state(value) == (dropped_events, reason, acknowledged_marker)
                ),
            )

    def _health_locked(self, directory_fd: int | None) -> CaptureSpoolHealth:
        entries = self._entries(directory_fd)
        markers = self._session_markers(directory_fd)
        orphan_markers = self._reconcile_session_markers(entries, markers)
        dropped_events, last_drop_reason, acknowledged_marker = self._drop_state(directory_fd)
        return self._health_from_state(
            entries,
            dropped_events=dropped_events,
            last_drop_reason=last_drop_reason,
            acknowledged_marker=acknowledged_marker,
            markers=markers,
            orphan_markers=orphan_markers,
        )

    @staticmethod
    def _health_from_state(
        entries: tuple[_SpoolEntry, ...],
        *,
        dropped_events: int,
        last_drop_reason: str | None,
        acknowledged_marker: str | None,
        markers: dict[_SpoolSessionKey, _SpoolSessionMarker],
        orphan_markers: set[_SpoolSessionKey],
    ) -> CaptureSpoolHealth:
        synthetic_drops = 0
        for key, (path, state, _stable) in markers.items():
            marker_tag = path.name.removesuffix(_SESSION_MARKER_SUFFIX)
            if state == "acknowledged" and dropped_events == 0:
                raise CaptureSpoolIntegrityError()
            if (
                state == "pending"
                and marker_tag == acknowledged_marker
                and key not in orphan_markers
            ):
                raise CaptureSpoolIntegrityError()
            if key in orphan_markers and state == "pending" and marker_tag != acknowledged_marker:
                synthetic_drops += 1
        if synthetic_drops:
            dropped_events += synthetic_drops
            last_drop_reason = "spool_incomplete"
        queued_bytes = sum(len(stable.data) for _, _, stable in entries)
        return CaptureSpoolHealth(
            queued_events=len(entries),
            queued_bytes=int(queued_bytes),
            dropped_events=dropped_events,
            coverage_degraded=dropped_events > 0,
            last_drop_reason=cast(
                Literal["spool_incomplete", "spool_quota"] | None,
                last_drop_reason,
            ),
        )

    def _delete_stable_file_locked(
        self,
        stable: _SpoolFileRead,
        directory_fd: int | None,
    ) -> None:
        stable.authorization.revalidate()
        if self._windows_operations is not None:  # pragma: no cover - native Windows R01
            if type(stable) is not WindowsStableFileRead:
                raise CaptureSpoolIntegrityError()
            self._windows_operations.delete_authorized_file(stable.authorization)
            self._revalidate()
            return
        if (
            type(stable) is not StableFileRead
            or type(directory_fd) is not int
            or type(self._spool_identity) is not security_files._PrivateDirectoryAuthorization
        ):
            raise CaptureSpoolIntegrityError()
        security_files._delete_authorized_private_file_at_descriptor(
            self._spool_identity,
            directory_fd,
            stable.authorization,
        )
        security_files._require_authorized_private_directory_descriptor(
            self._spool_identity,
            directory_fd,
        )

    def _persist_orphan_degradation_locked(
        self,
        orphan_markers: set[_SpoolSessionKey],
        markers: dict[_SpoolSessionKey, _SpoolSessionMarker],
        directory_fd: int | None,
        *,
        remove: bool,
    ) -> None:
        if not orphan_markers:
            return
        dropped_events, last_drop_reason, acknowledged_marker = self._drop_state(directory_fd)
        remaining_orphans = set(orphan_markers)
        if acknowledged_marker is not None:
            for key, (path, state, stable) in tuple(markers.items()):
                marker_tag = path.name.removesuffix(_SESSION_MARKER_SUFFIX)
                if marker_tag != acknowledged_marker or state != "pending":
                    continue
                if key not in orphan_markers:
                    raise CaptureSpoolIntegrityError()
                if remove:
                    self._delete_stable_file_locked(stable, directory_fd)
                    markers.pop(key)
                    remaining_orphans.remove(key)
                else:
                    markers[key] = self._set_session_marker_state_locked(
                        key,
                        "acknowledged",
                        directory_fd,
                    )
                break
        for key in sorted(remaining_orphans):
            path, state, stable = markers[key]
            marker_tag = path.name.removesuffix(_SESSION_MARKER_SUFFIX)
            if state == "acknowledged" and dropped_events == 0:
                raise CaptureSpoolIntegrityError()
            if state == "pending" and acknowledged_marker != marker_tag:
                dropped_events += 1
                last_drop_reason = "spool_incomplete"
                acknowledged_marker = marker_tag
                self._write_drop_state(
                    dropped_events,
                    last_drop_reason,
                    directory_fd,
                    acknowledged_marker=acknowledged_marker,
                )
            if remove:
                self._delete_stable_file_locked(stable, directory_fd)
                markers.pop(key)
            elif state == "pending":
                markers[key] = self._set_session_marker_state_locked(
                    key,
                    "acknowledged",
                    directory_fd,
                )

    def _set_session_marker_state_locked(
        self,
        key: _SpoolSessionKey,
        state: _SpoolSessionMarkerState,
        directory_fd: int | None,
    ) -> _SpoolSessionMarker:
        connection_id, session_id = key
        tag, framed = self._session_marker_frame(connection_id, session_id, state)
        path = self._locations.spool_directory / f"{tag}{_SESSION_MARKER_SUFFIX}"
        valid_frames = frozenset(
            (
                self._session_marker_frame(
                    connection_id,
                    session_id,
                    "acknowledged",
                )[1],
                self._session_marker_frame(connection_id, session_id, "pending")[1],
            )
        )
        if self._windows_operations is not None:  # pragma: no cover - native Windows R01
            self._windows_operations.publish_private_file(
                PureWindowsPath(str(path)),
                framed,
                maximum_bytes=_MAX_SESSION_MARKER_BYTES,
                validate_replacement=lambda value: value in valid_frames,
                validate_published=lambda value: value == framed,
            )
        else:
            if (
                type(directory_fd) is not int
                or type(self._spool_identity) is not security_files._PrivateDirectoryAuthorization
            ):
                raise SecureFileError()
            security_files._publish_private_file_at_descriptor(
                self._spool_identity,
                directory_fd,
                path.name,
                framed,
                maximum_bytes=_MAX_SESSION_MARKER_BYTES,
                validate_replacement=lambda value: value in valid_frames,
                validate_published=lambda value: value == framed,
            )
        marker = self._read_session_marker(connection_id, session_id, directory_fd)
        if marker is None or marker[1] != state:
            raise CaptureSpoolIntegrityError()
        return marker

    def _ensure_session_marker_locked(
        self,
        connection_id: str,
        session_id: str,
        directory_fd: int | None,
    ) -> _SpoolSessionMarker:
        existing = self._read_session_marker(connection_id, session_id, directory_fd)
        if existing is not None:
            return existing
        tag, framed = self._session_marker_frame(connection_id, session_id, "pending")
        path = self._locations.spool_directory / f"{tag}{_SESSION_MARKER_SUFFIX}"
        if self._windows_operations is not None:  # pragma: no cover - native Windows R01
            self._windows_operations.publish_private_file(
                PureWindowsPath(str(path)),
                framed,
                maximum_bytes=_MAX_SESSION_MARKER_BYTES,
                validate_published=lambda value: value == framed,
            )
        else:
            if (
                type(directory_fd) is not int
                or type(self._spool_identity) is not security_files._PrivateDirectoryAuthorization
            ):
                raise SecureFileError()
            security_files._publish_private_file_at_descriptor(
                self._spool_identity,
                directory_fd,
                path.name,
                framed,
                maximum_bytes=_MAX_SESSION_MARKER_BYTES,
                validate_published=lambda value: value == framed,
            )
        marker = self._read_session_marker(connection_id, session_id, directory_fd)
        if marker is None or marker[1] != "pending":
            raise CaptureSpoolIntegrityError()
        return marker

    def _clear_drop_health_if_empty_locked(self, directory_fd: int | None) -> bool:
        entries = self._entries(directory_fd)
        markers = self._session_markers(directory_fd)
        orphan_markers = self._reconcile_session_markers(entries, markers)
        if entries:
            return False
        self._persist_orphan_degradation_locked(
            orphan_markers,
            markers,
            directory_fd,
            remove=True,
        )
        _dropped_events, _last_drop_reason, _acknowledged_marker, stable = self._drop_state_record(
            directory_fd
        )
        if stable is not None:
            self._delete_stable_file_locked(stable, directory_fd)
        return True

    def health(self) -> CaptureSpoolHealth:
        failed = False
        try:
            with self._locked() as directory_fd:
                return self._health_locked(directory_fd)
        except CaptureSpoolError:
            raise
        except Exception:
            failed = True
        if failed:
            raise CaptureSpoolError()
        raise CaptureSpoolError()

    def observe_health(self, snapshot_digest: str) -> CaptureSpoolObservation:
        """Authenticate one locked spool health view against a session snapshot."""

        try:
            if (
                type(snapshot_digest) is not str
                or re.fullmatch(r"[0-9a-f]{64}", snapshot_digest) is None
            ):
                raise CaptureSpoolObservationError()
            with self._locked(blocking=False) as directory_fd:
                health = self._health_locked(directory_fd)
                return _seal_spool_observation(
                    health,
                    snapshot_digest=snapshot_digest,
                    spool_boundary_digest=self._boundary_digest,
                    installation_key=self._key,
                )
        except (CaptureSpoolError, CaptureSpoolObservationError):
            raise
        except Exception:
            raise CaptureSpoolError() from None

    def _enqueue_locked(
        self,
        intake: CaptureIntake,
        tag: str,
        framed: bytes,
        directory_fd: int | None,
    ) -> CaptureSpoolEnqueueReceipt:
        entries = self._entries(directory_fd)
        markers = self._session_markers(directory_fd)
        orphan_markers = self._reconcile_session_markers(entries, markers)
        self._persist_orphan_degradation_locked(
            orphan_markers,
            markers,
            directory_fd,
            remove=False,
        )
        path = self._locations.spool_directory / f"{tag}{_ENTRY_SUFFIX}"
        existing = next((item for item in entries if item[0] == path), None)
        if existing is not None:
            if existing[2].data != framed:
                raise CaptureSpoolIntegrityError()
            return CaptureSpoolEnqueueReceipt(disposition="already_queued")
        queued_bytes = sum(len(stable.data) for _, _, stable in entries)
        if (
            len(entries) >= MAX_CAPTURE_SPOOL_EVENTS
            or queued_bytes + len(framed) > MAX_CAPTURE_SPOOL_BYTES
        ):
            key = (intake.connection_id, intake.session_id)
            marker = markers.get(key)
            pending_quota_barrier = marker is None
            if marker is None:
                marker = self._ensure_session_marker_locked(
                    intake.connection_id,
                    intake.session_id,
                    directory_fd,
                )
                markers[key] = marker
            dropped_events, _reason, acknowledged_marker = self._drop_state(directory_fd)
            marker_tag = marker[0].name.removesuffix(_SESSION_MARKER_SUFFIX)
            self._write_drop_state(
                dropped_events + 1,
                "spool_quota",
                directory_fd,
                acknowledged_marker=(marker_tag if pending_quota_barrier else acknowledged_marker),
            )
            if pending_quota_barrier:
                markers[key] = self._set_session_marker_state_locked(
                    key,
                    "acknowledged",
                    directory_fd,
                )
            return CaptureSpoolEnqueueReceipt(disposition="dropped_quota")
        self._ensure_session_marker_locked(
            intake.connection_id,
            intake.session_id,
            directory_fd,
        )
        if self._windows_operations is not None:  # pragma: no cover - native Windows R01
            self._windows_operations.publish_private_file(
                PureWindowsPath(str(path)),
                framed,
                maximum_bytes=_MAX_ENTRY_BYTES,
                validate_published=lambda value: value == framed,
            )
        else:
            if (
                type(directory_fd) is not int
                or type(self._spool_identity) is not security_files._PrivateDirectoryAuthorization
            ):
                raise SecureFileError()
            security_files._publish_private_file_at_descriptor(
                self._spool_identity,
                directory_fd,
                path.name,
                framed,
                maximum_bytes=_MAX_ENTRY_BYTES,
                validate_published=lambda value: value == framed,
            )
        return CaptureSpoolEnqueueReceipt(disposition="queued")

    def enqueue(self, intake: CaptureIntake) -> CaptureSpoolEnqueueReceipt:
        failed = False
        try:
            tag, framed = self._frame(intake)
            with self._locked() as directory_fd:
                return self._enqueue_locked(intake, tag, framed, directory_fd)
        except CaptureSpoolError:
            raise
        except Exception:
            failed = True
        if failed:
            raise CaptureSpoolError()
        raise CaptureSpoolError()

    def _admit_locked(
        self,
        store: _AppendStore,
        intake: CaptureIntake,
        tag: str,
        framed: bytes,
        directory_fd: int | None,
    ) -> object:
        marker = self._read_session_marker(
            intake.connection_id,
            intake.session_id,
            directory_fd,
        )
        if marker is not None:
            return self._enqueue_locked(intake, tag, framed, directory_fd)
        from saliencegate.capture.store import CaptureStoreBusyError

        try:
            return store.append(intake)
        except Exception as error:
            if not isinstance(error, CaptureStoreBusyError):
                raise
        return self._enqueue_locked(intake, tag, framed, directory_fd)

    def admit(self, store: _AppendStore, intake: CaptureIntake) -> object:
        """Admit once while preserving any same-session spool backlog."""

        try:
            tag, framed = self._frame(intake)
        except CaptureSpoolError:
            raise
        except Exception:
            raise CaptureSpoolError() from None
        entered = False
        try:
            with self._locked() as directory_fd:
                entered = True
                return self._admit_locked(
                    store,
                    intake,
                    tag,
                    framed,
                    directory_fd,
                )
        except CaptureSpoolError:
            raise
        except RuntimeError:
            raise
        except Exception:
            if not entered:
                raise CaptureSpoolUnavailableError() from None
            raise CaptureSpoolError() from None

    @contextmanager
    def maintenance(self) -> Iterator[CaptureSpoolMaintenance]:
        """Hold the spool fence across drain and a store lifecycle mutation."""

        token: CaptureSpoolMaintenance | None = None
        try:
            with self._locked() as directory_fd:
                token = CaptureSpoolMaintenance(self, directory_fd)
                try:
                    yield token
                finally:
                    token._close()
        except CaptureSpoolError:
            raise
        except RuntimeError:
            raise
        except Exception:
            raise CaptureSpoolError() from None

    def _append_from_spool(self, store: _AppendStore, intake: CaptureIntake) -> object:
        from saliencegate.capture.store import (
            CaptureAdmissionSource,
            CaptureStore,
        )

        if isinstance(store, CaptureStore):
            return store.append(intake, source=CaptureAdmissionSource.SPOOL_DRAIN)
        return store.append(intake)

    def _drain_locked(
        self,
        store: _AppendStore,
        directory_fd: int | None,
    ) -> CaptureSpoolDrainReceipt:
        entries = self._entries(directory_fd)
        markers = self._session_markers(directory_fd)
        orphan_markers = self._reconcile_session_markers(entries, markers)
        self._persist_orphan_degradation_locked(
            orphan_markers,
            markers,
            directory_fd,
            remove=True,
        )
        remaining_by_session: dict[_SpoolSessionKey, int] = {}
        for _path, intake, _stable in entries:
            key = (intake.connection_id, intake.session_id)
            remaining_by_session[key] = remaining_by_session.get(key, 0) + 1
        admitted = 0
        for _path, intake, stable in entries:
            try:
                self._append_from_spool(store, intake)
            except Exception as error:
                from saliencegate.capture.store import CaptureStoreBusyError

                if isinstance(error, CaptureStoreBusyError):
                    return CaptureSpoolDrainReceipt(
                        admitted_events=admitted,
                        remaining_events=len(entries) - admitted,
                    )
                raise
            self._delete_stable_file_locked(stable, directory_fd)
            key = (intake.connection_id, intake.session_id)
            remaining_by_session[key] -= 1
            if remaining_by_session[key] == 0:
                _marker_path, _state, marker = markers.pop(key)
                self._delete_stable_file_locked(marker, directory_fd)
            admitted += 1
        return CaptureSpoolDrainReceipt(admitted_events=admitted, remaining_events=0)

    def drain(self, store: _AppendStore) -> CaptureSpoolDrainReceipt:
        failed = False
        try:
            with self._locked() as directory_fd:
                return self._drain_locked(store, directory_fd)
        except CaptureSpoolError:
            raise
        except RuntimeError:
            raise
        except Exception:
            failed = True
        if failed:
            raise CaptureSpoolError()
        raise CaptureSpoolError()


def admit_capture_intake(
    store: _AppendStore,
    spool: CaptureSpool,
    intake: CaptureIntake,
) -> object:
    """Admit under the cross-process spool ordering fence."""

    return spool.admit(store, intake)


__all__ = [
    "MAX_CAPTURE_SPOOL_BYTES",
    "MAX_CAPTURE_SPOOL_EVENTS",
    "CaptureSpool",
    "CaptureSpoolDrainReceipt",
    "CaptureSpoolEnqueueReceipt",
    "CaptureSpoolError",
    "CaptureSpoolHealth",
    "CaptureSpoolIntegrityError",
    "CaptureSpoolMaintenance",
    "CaptureSpoolObservation",
    "CaptureSpoolObservationError",
    "CaptureSpoolUnavailableError",
    "admit_capture_intake",
    "verify_capture_spool_observation",
]
