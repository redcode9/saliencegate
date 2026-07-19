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
from typing import Final, Literal, Protocol, cast

import saliencegate.security.files as security_files
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.capture.locations import CaptureStoreLocations
from saliencegate.capture.publication import verify_capture_intake_authentication
from saliencegate.capture.schema import (
    MAX_CAPTURE_EVENT_BYTES,
    CaptureIntake,
    canonical_capture_intake,
    load_capture_intake,
)
from saliencegate.domain import canonical_json
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
_HEALTH_NAME = ".capture-spool-health"
_HEALTH_HEADER = b"capture-spool-health/v1"
_HEALTH_DOMAIN = b"saliencegate:capture-spool:health:v1"
_LOCK_NAME = ".capture-spool-lock"
_MAX_ENTRY_BYTES = MAX_CAPTURE_EVENT_BYTES + 256
_MAX_HEALTH_BYTES = 4_096


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
    last_drop_reason: Literal["spool_quota"] | None

    def __repr__(self) -> str:
        return "CaptureSpoolHealth(<redacted>)"


class _AppendStore(Protocol):
    def append(self, intake: CaptureIntake) -> object: ...


_SpoolFileRead = StableFileRead | WindowsStableFileRead
_SpoolDirectoryIdentity = security_files._PrivateDirectoryAuthorization | WindowsPathAuthorization


class CaptureSpool:
    """One installation-key-bound spool directory."""

    __slots__ = (
        "_context",
        "_key",
        "_locations",
        "_spool_identity",
        "_state_identity",
        "_windows_operations",
    )

    _context: CaptureDigestContext
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

        if (
            type(locations) is not CaptureStoreLocations
            or type(installation_key) is not InstallationKey
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
                    create=True,
                )
                spool_identity = authorize_windows_private_path(
                    PureWindowsPath(str(locations.spool_directory)),
                    kind=WindowsPathKind.DIRECTORY,
                    operations=windows_operations,
                    create=True,
                )
                state_identity.revalidate()
                spool_identity.revalidate()
            else:
                state_identity = security_files._authorize_private_directory(
                    locations.state_directory,
                    create=True,
                )
                spool_identity = security_files._authorize_private_directory_child(
                    state_identity,
                    locations.spool_directory.name,
                    create=True,
                )
                state_identity.revalidate()
                spool_identity.revalidate()
            instance = cls.__new__(cls)
            instance._locations = locations
            instance._key = installation_key._copy()
            instance._context = CaptureDigestContext(installation_key)
            instance._state_identity = state_identity
            instance._spool_identity = spool_identity
            instance._windows_operations = windows_operations
            result = instance
        except Exception:
            failed = True
        if failed or result is None:
            raise CaptureSpoolError()
        return result

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
    def _locked(self) -> Iterator[int | None]:
        self._revalidate()
        lock_path = self._locations.spool_directory / _LOCK_NAME
        if self._windows_operations is not None:  # pragma: no cover - native Windows R01
            with self._windows_operations.private_file_lock(PureWindowsPath(str(lock_path))):
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

            fcntl.flock(descriptor, fcntl.LOCK_EX)
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

    def _entry_paths(self, directory_fd: int | None) -> tuple[Path, ...]:
        paths: list[Path] = []
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
            if _ENTRY_NAME.fullmatch(path.name):
                paths.append(path)
            elif path.name not in {_HEALTH_NAME, _LOCK_NAME}:
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
        return tuple(paths)

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
    ) -> tuple[tuple[Path, CaptureIntake, _SpoolFileRead], ...]:
        result: list[tuple[Path, CaptureIntake, _SpoolFileRead]] = []
        for path in self._entry_paths(directory_fd):
            decoded = self._decode_entry(path, directory_fd)
            if decoded is None:
                raise CaptureSpoolIntegrityError()
            intake, stable = decoded
            result.append((path, intake, stable))
        return tuple(sorted(result, key=self._drain_order_key))

    @staticmethod
    def _drain_order_key(
        entry: tuple[Path, CaptureIntake, _SpoolFileRead],
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

    def _decode_drop_state(self, data: bytes) -> tuple[int, str | None] | None:
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
                or value.get("schema_version") != "capture-spool-health/v1"
                or type(value.get("dropped_events")) is not int
                or value["dropped_events"] < 0
                or value.get("last_drop_reason") not in (None, "spool_quota")
                or canonical_json(value) != payload
            ):
                return None
            return value["dropped_events"], value["last_drop_reason"]
        except Exception:
            return None

    def _drop_state(self, directory_fd: int | None) -> tuple[int, str | None]:
        path = self._health_path()
        stable: _SpoolFileRead | None = None
        try:
            if self._windows_operations is not None:  # pragma: no cover - native Windows R01
                windows_path = PureWindowsPath(str(path))
                if self._windows_operations.inspect_path(windows_path) is None:
                    return 0, None
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
                    return 0, None
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
        return state

    def _write_drop_state(
        self,
        dropped_events: int,
        reason: str,
        directory_fd: int | None,
    ) -> None:
        payload = canonical_json(
            {
                "schema_version": "capture-spool-health/v1",
                "dropped_events": dropped_events,
                "last_drop_reason": reason,
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
                    self._decode_drop_state(value) == (dropped_events, reason)
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
                    self._decode_drop_state(value) == (dropped_events, reason)
                ),
            )

    def _health_locked(self, directory_fd: int | None) -> CaptureSpoolHealth:
        entries = self._entries(directory_fd)
        dropped_events, last_drop_reason = self._drop_state(directory_fd)
        queued_bytes = sum(len(stable.data) for _, _, stable in entries)
        return CaptureSpoolHealth(
            queued_events=len(entries),
            queued_bytes=int(queued_bytes),
            dropped_events=dropped_events,
            coverage_degraded=dropped_events > 0,
            last_drop_reason=cast(Literal["spool_quota"] | None, last_drop_reason),
        )

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

    def enqueue(self, intake: CaptureIntake) -> CaptureSpoolEnqueueReceipt:
        failed = False
        try:
            tag, framed = self._frame(intake)
            with self._locked() as directory_fd:
                entries = self._entries(directory_fd)
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
                    dropped_events, _reason = self._drop_state(directory_fd)
                    self._write_drop_state(
                        dropped_events + 1,
                        "spool_quota",
                        directory_fd,
                    )
                    return CaptureSpoolEnqueueReceipt(disposition="dropped_quota")
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
                        or type(self._spool_identity)
                        is not security_files._PrivateDirectoryAuthorization
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
        except CaptureSpoolError:
            raise
        except Exception:
            failed = True
        if failed:
            raise CaptureSpoolError()
        raise CaptureSpoolError()

    def _append_from_spool(self, store: _AppendStore, intake: CaptureIntake) -> object:
        from saliencegate.capture.store import (
            CaptureAdmissionSource,
            CaptureStore,
        )

        if isinstance(store, CaptureStore):
            return store.append(intake, source=CaptureAdmissionSource.SPOOL_DRAIN)
        return store.append(intake)

    def drain(self, store: _AppendStore) -> CaptureSpoolDrainReceipt:
        failed = False
        try:
            with self._locked() as directory_fd:
                entries = self._entries(directory_fd)
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
                    stable.authorization.revalidate()
                    if self._windows_operations is not None:  # pragma: no cover - Windows R01
                        if type(stable) is not WindowsStableFileRead:
                            raise CaptureSpoolIntegrityError()
                        self._windows_operations.delete_authorized_file(stable.authorization)
                        self._revalidate()
                    else:
                        if (
                            type(stable) is not StableFileRead
                            or type(directory_fd) is not int
                            or type(self._spool_identity)
                            is not security_files._PrivateDirectoryAuthorization
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
                    admitted += 1
                return CaptureSpoolDrainReceipt(admitted_events=admitted, remaining_events=0)
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
    """Append directly, falling back only for the store's primary BUSY signal."""

    from saliencegate.capture.store import CaptureStoreBusyError

    try:
        return store.append(intake)
    except CaptureStoreBusyError:
        return spool.enqueue(intake)


__all__ = [
    "MAX_CAPTURE_SPOOL_BYTES",
    "MAX_CAPTURE_SPOOL_EVENTS",
    "CaptureSpool",
    "CaptureSpoolDrainReceipt",
    "CaptureSpoolEnqueueReceipt",
    "CaptureSpoolError",
    "CaptureSpoolHealth",
    "CaptureSpoolIntegrityError",
    "admit_capture_intake",
]
