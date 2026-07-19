"""Fail-closed stable reads, private locations, and SQLite path authorization.

The private boundary walks directory descriptors without following symlinks and pins exact
file identities.  A deliberately separate legacy read profile preserves the JSONL adapter's
historical target-only acceptance while retaining regular-file, single-link, stability, and
byte bounds.  SQLite authorization also pins its WAL, shared-memory, and rollback-journal
names before SQLite may use them.  Python's standard ``sqlite3`` module does not expose an
fd-bound VFS, so a same-uid adversary can still race the micro-window between checks; every
observable boundary is nevertheless rechecked and persistent replacements fail closed.
"""

from __future__ import annotations

import ctypes
import errno
import os
import secrets
import stat
import sys
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Never

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - publication is gated to POSIX hosts
    _fcntl = None  # type: ignore[assignment]

_SQLITE_DURABLE_SIDECAR_SUFFIXES = ("-wal", "-shm")
_SQLITE_TRANSIENT_SIDECAR_SUFFIXES = ("-journal",)
_SQLITE_SIDECAR_SUFFIXES = (
    *_SQLITE_DURABLE_SIDECAR_SUFFIXES,
    *_SQLITE_TRANSIENT_SIDECAR_SUFFIXES,
)
_DARWIN_ACL_TYPE_EXTENDED = 0x00000100
_DARWIN_ACL_FIRST_ENTRY = 0
_DARWIN_ACL_NEXT_ENTRY = -1
_DARWIN_ACL_EXTENDED_DENY = 2
_UNSUPPORTED_OPERATION_ERRNOS = frozenset(
    {
        errno.ENOSYS,
        *(
            value
            for name in ("ENOTSUP", "EOPNOTSUPP")
            if (value := getattr(errno, name, None)) is not None
        ),
    }
)


class SecureFileError(ValueError):
    """A value-free failure to establish or revalidate a secure file boundary."""

    def __init__(self) -> None:
        super().__init__("secure file authorization failed")


class SecureFileBoundError(SecureFileError):
    """A value-free failure caused by an explicit caller-owned byte or line bound."""

    def __init__(self) -> None:
        ValueError.__init__(self, "secure file bound exceeded")


class SecureFileUnsupportedError(SecureFileError):
    """The host cannot provide the requested secure filesystem operation."""

    def __init__(self) -> None:
        ValueError.__init__(self, "secure file operation unsupported")


class _UnsafeFilePathError(Exception):
    pass


class _UnsupportedFileOperationError(Exception):
    pass


class StableReadPolicy(StrEnum):
    """Fixed filesystem policy profiles for schema-neutral bounded reads."""

    LEGACY_COMPATIBILITY = "legacy_compatibility"
    PRIVATE_OWNER = "private_owner"


class _AuthorizationKind(StrEnum):
    SQLITE = "sqlite"
    STABLE_READ = "stable_read"
    PRIVATE_LOCATION = "private_location"


@dataclass(frozen=True, slots=True)
class _StableIdentity:
    device: int
    inode: int
    mode: int
    owner: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _StableIdentity:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            mode=value.st_mode,
            owner=value.st_uid,
        )

    def matches(self, value: os.stat_result) -> bool:
        return self == type(self).from_stat(value)


@dataclass(frozen=True, slots=True)
class _CompleteIdentity:
    stable: _StableIdentity
    link_count: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _CompleteIdentity:
        return cls(
            stable=_StableIdentity.from_stat(value),
            link_count=value.st_nlink,
            size=value.st_size,
            modified_ns=value.st_mtime_ns,
            changed_ns=value.st_ctime_ns,
        )

    def matches(self, value: os.stat_result) -> bool:
        return self == type(self).from_stat(value)


@dataclass(frozen=True, slots=True)
class _SQLiteSidecarAuthorization:
    suffix: str
    identity: _StableIdentity
    created: bool
    transient: bool
    cleanup_identity: _CompleteIdentity | None


@dataclass(frozen=True, slots=True, repr=False)
class StableFileAuthorization:
    """A copied path bound to the exact filesystem boundary that was checked."""

    path: str
    _parent_identity: _StableIdentity | None
    _target_identity: _StableIdentity | None
    _sqlite_sidecars: tuple[_SQLiteSidecarAuthorization, ...] = ()
    _target_complete_identity: _CompleteIdentity | None = None
    _kind: _AuthorizationKind = _AuthorizationKind.SQLITE
    _read_policy: StableReadPolicy | None = None

    def revalidate(self) -> None:
        """Fail if the parent, database, or SQLite sidecar boundary became unsafe."""

        self._checked_revalidate(strict_transient=False)

    def _revalidate_before_sqlite_statements(self) -> None:
        """Pin every journal identity before SQLite is allowed to execute SQL."""

        if self._kind is not _AuthorizationKind.SQLITE:
            raise SecureFileError()
        self._checked_revalidate(strict_transient=True)

    def _checked_revalidate(self, *, strict_transient: bool) -> None:
        failed = False
        try:
            _revalidate(self, strict_transient=strict_transient)
        except Exception:
            failed = True
        if failed:
            raise SecureFileError()

    def aliases(self, other: StableFileAuthorization) -> bool:
        """Compare captured names and inodes without following either path again."""

        if type(other) is not StableFileAuthorization:
            raise SecureFileError()
        return _authorizations_alias(self, other)

    def _cleanup_created_sqlite_sidecars(self) -> None:
        """Best-effort removal of unchanged placeholders created by this authorization.

        SQLite owns existing and claimed journal files.  This hook is only for a
        failed constructor after its connection has been closed; pre-existing files
        and names whose identity or security metadata changed are never removed.
        """

        _cleanup_authorized_sqlite_sidecars(self)

    def __repr__(self) -> str:
        return "StableFileAuthorization(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class StableFileRead:
    """Exact bounded bytes and the stable authorization captured with them."""

    data: bytes
    authorization: StableFileAuthorization

    def __post_init__(self) -> None:
        if type(self.data) is not bytes or type(self.authorization) is not StableFileAuthorization:
            raise SecureFileError()

    def iter_lines(
        self,
        *,
        maximum_line_bytes: int,
        maximum_lines: int,
    ) -> Iterator[bytes]:
        """Yield LF/CRLF-delimited records without materializing a split copy."""

        if (
            type(maximum_line_bytes) is not int
            or maximum_line_bytes < 1
            or type(maximum_lines) is not int
            or maximum_lines < 1
        ):
            raise SecureFileBoundError()
        start = 0
        line_count = 0
        data = self.data
        while start < len(data):
            newline = data.find(b"\n", start)
            if newline < 0:
                raw_end = len(data)
                next_start = len(data)
            else:
                raw_end = newline
                next_start = newline + 1
            if raw_end - start > maximum_line_bytes or line_count >= maximum_lines:
                raise SecureFileBoundError()
            end = raw_end
            if newline >= 0 and end > start and data[end - 1] == 0x0D:
                end -= 1
            line_count += 1
            yield data[start:end]
            start = next_start


_ATOMIC_PUBLICATION_TOKEN = object()


class AtomicFilePublication:
    """A one-shot owner-only publication authorized for one exact path state."""

    __slots__ = (
        "_lock",
        "_maximum_bytes",
        "_replacement_data",
        "_used",
        "authorization",
    )

    def __init__(
        self,
        authorization: StableFileAuthorization,
        maximum_bytes: int,
        replacement_data: bytes | None,
        *,
        _token: object,
    ) -> None:
        if (
            _token is not _ATOMIC_PUBLICATION_TOKEN
            or type(authorization) is not StableFileAuthorization
            or authorization._kind is not _AuthorizationKind.PRIVATE_LOCATION
            or type(maximum_bytes) is not int
            or not 1 <= maximum_bytes < sys.maxsize
            or (replacement_data is not None and type(replacement_data) is not bytes)
        ):
            raise SecureFileError()
        self.authorization = authorization
        self._maximum_bytes = maximum_bytes
        self._replacement_data = replacement_data
        self._used = False
        self._lock = Lock()

    def publish(
        self,
        data: bytes,
        *,
        validate_published: Callable[[bytes], bool] | None = None,
    ) -> StableFileRead:
        """Publish exact bytes once, retaining rollback until validation succeeds."""

        if type(data) is not bytes:
            raise SecureFileError()
        if len(data) > self._maximum_bytes:
            raise SecureFileBoundError()
        if validate_published is not None and not callable(validate_published):
            raise SecureFileError()
        with self._lock:
            if self._used:
                raise SecureFileError()
            self._used = True
        result: StableFileRead | None = None
        failed = False
        unsupported = False
        try:
            result = _publish_atomic_file(self, data, validate_published)
        except _UnsupportedFileOperationError:
            unsupported = True
        except Exception:
            failed = True
        if unsupported:
            raise SecureFileUnsupportedError()
        if failed or result is None:
            raise SecureFileError()
        return result

    def __repr__(self) -> str:
        return "AtomicFilePublication(<redacted>)"


def _fail() -> Never:
    raise _UnsafeFilePathError


def _current_user_id() -> int:
    return os.getuid()


def _require_secure_platform() -> None:
    required_flags = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    if (
        os.name != "posix"
        or not hasattr(os, "getuid")
        or not hasattr(os, "fchmod")
        or not all(hasattr(os, name) for name in required_flags)
        or os.open not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.stat not in os.supports_follow_symlinks
        or os.unlink not in os.supports_dir_fd
    ):
        _fail()


def _require_atomic_publication_platform() -> None:
    """Require every primitive used by the no-clobber and replacement protocols."""

    try:
        _require_secure_platform()
    except Exception:
        raise _UnsupportedFileOperationError from None
    if (
        not hasattr(os, "fsync")
        or not hasattr(os, "link")
        or not hasattr(os, "rename")
        or not hasattr(os, "write")
        or _fcntl is None
        or os.link not in os.supports_dir_fd
        or os.link not in os.supports_follow_symlinks
        or os.rename not in os.supports_dir_fd
    ):
        raise _UnsupportedFileOperationError


def _probe_directory_fsync(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError as error:
        unsupported = {*_UNSUPPORTED_OPERATION_ERRNOS, errno.EINVAL}
        if error.errno in unsupported:
            raise _UnsupportedFileOperationError from None
        raise


def _copy_path(value: str | os.PathLike[str]) -> Path:
    raw = os.fspath(value)
    if type(raw) is not str or not raw or "\0" in raw:
        _fail()
    os.fsencode(raw)
    if any(component in (".", "..") for component in raw.split(os.sep)):
        _fail()
    path = Path(os.path.abspath(raw))
    if path.name in ("", ".", ".."):
        _fail()
    return path


def _copy_legacy_read_path(value: str | os.PathLike[str]) -> Path:
    """Copy a legacy path without narrowing the historical JSONL acceptance set."""

    raw = os.fspath(value)
    if type(raw) is not str or not raw or "\0" in raw:
        _fail()
    os.fsencode(raw)
    # ``abspath``/``normpath`` cannot be used here: collapsing ``..`` before the
    # kernel resolves a preceding symlink can select a different file than the
    # legacy raw-path open.  Prefixing the cwd makes the snapshot absolute while
    # deliberately retaining the original traversal components.
    declared = raw if os.path.isabs(raw) else os.path.join(os.getcwd(), raw)
    path = Path(declared)
    if path.name in ("", ".", ".."):
        _fail()
    return path


def _safe_parent(value: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(value.st_mode)
        and value.st_uid == _current_user_id()
        and stat.S_IMODE(value.st_mode) & 0o022 == 0
    )


def _safe_ancestor(value: os.stat_result) -> bool:
    mode = stat.S_IMODE(value.st_mode)
    trusted_owner = value.st_uid in (0, _current_user_id())
    private = mode & 0o022 == 0
    sticky_shared = mode & stat.S_ISVTX != 0
    return stat.S_ISDIR(value.st_mode) and trusted_owner and (private or sticky_shared)


def _safe_target(value: os.stat_result) -> bool:
    return (
        stat.S_ISREG(value.st_mode)
        and value.st_nlink == 1
        and value.st_uid == _current_user_id()
        and stat.S_IMODE(value.st_mode) == 0o600
    )


def _safe_read_target(value: os.stat_result, policy: StableReadPolicy) -> bool:
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        return False
    if policy is StableReadPolicy.LEGACY_COMPATIBILITY:
        return True
    return value.st_uid == _current_user_id() and stat.S_IMODE(value.st_mode) & 0o022 == 0


def _read_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _read_descriptor_bounded(descriptor: int, maximum_bytes: int) -> bytes:
    result = bytearray()
    while len(result) <= maximum_bytes:
        remaining = maximum_bytes + 1 - len(result)
        chunk = os.read(descriptor, min(64 * 1024, remaining))
        if not chunk:
            break
        result.extend(chunk)
    return bytes(result)


def _require_stable_read_metadata(
    expected: _CompleteIdentity,
    values: tuple[os.stat_result, ...],
    *,
    policy: StableReadPolicy,
) -> None:
    if any(not _safe_read_target(value, policy) or not expected.matches(value) for value in values):
        _fail()


def _read_opened_stable_file(
    descriptor: int,
    named_before: os.stat_result,
    named_after: Callable[[], os.stat_result],
    *,
    maximum_bytes: int,
    policy: StableReadPolicy,
    check_acl: bool,
) -> tuple[bytes, _CompleteIdentity]:
    opened_before = os.fstat(descriptor)
    expected = _CompleteIdentity.from_stat(named_before)
    _require_stable_read_metadata(
        expected,
        (named_before, opened_before),
        policy=policy,
    )
    if check_acl:
        _require_safe_acl(descriptor)

    initially_oversized = expected.size > maximum_bytes
    data = b"" if initially_oversized else _read_descriptor_bounded(descriptor, maximum_bytes)

    opened_after = os.fstat(descriptor)
    if check_acl:
        _require_safe_acl(descriptor)
    current_named = named_after()
    _require_stable_read_metadata(
        expected,
        (opened_after, current_named),
        policy=policy,
    )
    if initially_oversized:
        raise SecureFileBoundError()
    if len(data) > maximum_bytes or len(data) != expected.size:
        _fail()
    return data, expected


def _read_legacy_file(path: Path, maximum_bytes: int) -> StableFileRead:
    named_before = os.stat(path, follow_symlinks=False)
    descriptor = os.open(path, _read_open_flags())
    try:
        data, identity = _read_opened_stable_file(
            descriptor,
            named_before,
            lambda: os.stat(path, follow_symlinks=False),
            maximum_bytes=maximum_bytes,
            policy=StableReadPolicy.LEGACY_COMPATIBILITY,
            check_acl=False,
        )
    finally:
        os.close(descriptor)
    authorization = StableFileAuthorization(
        path=os.fspath(path),
        _parent_identity=None,
        _target_identity=identity.stable,
        _target_complete_identity=identity,
        _kind=_AuthorizationKind.STABLE_READ,
        _read_policy=StableReadPolicy.LEGACY_COMPATIBILITY,
    )
    return StableFileRead(data=data, authorization=authorization)


def _read_private_file(path: Path, maximum_bytes: int) -> StableFileRead:
    directory_fd, parent_identity = _open_parent(path)
    try:
        named_before = _named_stat(path.name, directory_fd)
        descriptor = os.open(path.name, _read_open_flags(), dir_fd=directory_fd)
        try:
            data, identity = _read_opened_stable_file(
                descriptor,
                named_before,
                lambda: _named_stat(path.name, directory_fd),
                maximum_bytes=maximum_bytes,
                policy=StableReadPolicy.PRIVATE_OWNER,
                check_acl=True,
            )
        finally:
            os.close(descriptor)
        _verify_parent(path, directory_fd, parent_identity)
        final_named = _named_stat(path.name, directory_fd)
        _require_stable_read_metadata(
            identity,
            (final_named,),
            policy=StableReadPolicy.PRIVATE_OWNER,
        )
    finally:
        os.close(directory_fd)
    authorization = StableFileAuthorization(
        path=os.fspath(path),
        _parent_identity=parent_identity,
        _target_identity=identity.stable,
        _target_complete_identity=identity,
        _kind=_AuthorizationKind.STABLE_READ,
        _read_policy=StableReadPolicy.PRIVATE_OWNER,
    )
    return StableFileRead(data=data, authorization=authorization)


def _darwin_acl_is_unsafe(descriptor: int, *, deny_only_allowed: bool) -> bool:
    """Fail-closed detection of unsafe macOS extended ACL entries.

    Deny-only ACLs are safe and common on macOS home-directory ancestors.  Any
    allow or unknown entry can weaken the mode-bit boundary and is rejected.
    The final parent, database, and sidecars reject every extended entry.
    """

    if sys.platform != "darwin":
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        get_acl = libc.acl_get_fd_np
        get_acl.argtypes = [ctypes.c_int, ctypes.c_int]
        get_acl.restype = ctypes.c_void_p
        get_entry = libc.acl_get_entry
        get_entry.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        get_entry.restype = ctypes.c_int
        get_tag = libc.acl_get_tag_type
        get_tag.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        get_tag.restype = ctypes.c_int
        free_acl = libc.acl_free
        free_acl.argtypes = [ctypes.c_void_p]
        free_acl.restype = ctypes.c_int

        ctypes.set_errno(0)
        acl = get_acl(descriptor, _DARWIN_ACL_TYPE_EXTENDED)
        if acl is None:
            return ctypes.get_errno() != errno.ENOENT

        unsafe = not deny_only_allowed
        try:
            if deny_only_allowed:
                entry = ctypes.c_void_p()
                entry_id = _DARWIN_ACL_FIRST_ENTRY
                while True:
                    ctypes.set_errno(0)
                    result = get_entry(acl, entry_id, ctypes.byref(entry))
                    if result == -1:
                        unsafe = ctypes.get_errno() != errno.EINVAL
                        break
                    if result != 0 or entry.value is None:
                        unsafe = True
                        break
                    tag = ctypes.c_int()
                    if get_tag(entry, ctypes.byref(tag)) != 0:
                        unsafe = True
                        break
                    if tag.value != _DARWIN_ACL_EXTENDED_DENY:
                        unsafe = True
                        break
                    entry_id = _DARWIN_ACL_NEXT_ENTRY
        finally:
            if free_acl(acl) != 0:
                unsafe = True
        return unsafe
    except Exception:
        return True


def _require_safe_acl(descriptor: int, *, deny_only_allowed: bool = False) -> None:
    if _darwin_acl_is_unsafe(descriptor, deny_only_allowed=deny_only_allowed):
        _fail()


def _named_stat(name: str, directory_fd: int) -> os.stat_result:
    return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)


def _open_directory_chain(path: Path) -> tuple[int, _StableIdentity]:
    """Open an absolute directory without following any component symlink."""

    if not path.is_absolute() or path.anchor != os.sep:
        _fail()
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(os.sep, flags)
    try:
        if not _safe_ancestor(os.fstat(descriptor)):
            _fail()
        _require_safe_acl(descriptor, deny_only_allowed=True)
        for component in path.parts[1:]:
            if component in ("", ".", ".."):
                _fail()
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            try:
                if not _safe_ancestor(os.fstat(next_descriptor)):
                    _fail()
                _require_safe_acl(next_descriptor, deny_only_allowed=True)
            except BaseException:
                os.close(next_descriptor)
                raise
            previous_descriptor = descriptor
            descriptor = next_descriptor
            os.close(previous_descriptor)
        opened = os.fstat(descriptor)
        if not _safe_parent(opened):
            _fail()
        _require_safe_acl(descriptor)
        return descriptor, _StableIdentity.from_stat(opened)
    except BaseException:
        os.close(descriptor)
        raise


def _open_parent(path: Path) -> tuple[int, _StableIdentity]:
    descriptor, opened_identity = _open_directory_chain(path.parent)
    try:
        named_descriptor, named_identity = _open_directory_chain(path.parent)
        try:
            if opened_identity != named_identity:
                _fail()
        finally:
            os.close(named_descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, opened_identity


def _verify_parent(path: Path, descriptor: int, expected: _StableIdentity) -> None:
    opened = os.fstat(descriptor)
    _require_safe_acl(descriptor)
    named_descriptor, named_identity = _open_directory_chain(path.parent)
    try:
        if not _safe_parent(opened) or not expected.matches(opened) or expected != named_identity:
            _fail()
    finally:
        os.close(named_descriptor)


def _open_existing_target(name: str, directory_fd: int) -> int:
    flags = os.O_RDWR | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW
    return os.open(name, flags, dir_fd=directory_fd)


def _validate_existing_target(
    directory_fd: int,
    name: str,
    *,
    named_before: os.stat_result | None = None,
) -> _StableIdentity:
    if named_before is None:
        named_before = _named_stat(name, directory_fd)
    if not _safe_target(named_before):
        _fail()

    descriptor = _open_existing_target(name, directory_fd)
    try:
        before = os.fstat(descriptor)
        if not _safe_target(before):
            _fail()
        _require_safe_acl(descriptor)
        identity = _CompleteIdentity.from_stat(before)
        after = os.fstat(descriptor)
        _require_safe_acl(descriptor)
        named_after = _named_stat(name, directory_fd)
        if (
            not _safe_target(after)
            or not _safe_target(named_after)
            or not identity.matches(named_before)
            or not identity.matches(after)
            or not identity.matches(named_after)
        ):
            _fail()
        return identity.stable
    finally:
        os.close(descriptor)


def _unlink_unchanged_private_file(
    directory_fd: int,
    name: str,
    expected: _CompleteIdentity,
) -> None:
    """Unlink only the still-private, single-link name that this boundary created."""

    with suppress(OSError):
        named = _named_stat(name, directory_fd)
        if expected.matches(named) and _safe_target(named):
            os.unlink(name, dir_fd=directory_fd)


def _create_target(directory_fd: int, name: str) -> _CompleteIdentity:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    cleanup_identity: _CompleteIdentity | None = None
    try:
        os.fchmod(descriptor, 0o600)
        before = os.fstat(descriptor)
        if not _safe_target(before) or before.st_size != 0:
            _fail()
        _require_safe_acl(descriptor)
        complete = _CompleteIdentity.from_stat(before)
        cleanup_identity = complete
        after = os.fstat(descriptor)
        _require_safe_acl(descriptor)
        named = _named_stat(name, directory_fd)
        if (
            not _safe_target(after)
            or not _safe_target(named)
            or not complete.matches(after)
            or not complete.matches(named)
        ):
            _fail()
    except BaseException:
        if cleanup_identity is not None:
            _unlink_unchanged_private_file(directory_fd, name, cleanup_identity)
        with suppress(OSError):
            os.close(descriptor)
        raise
    try:
        os.close(descriptor)
    except BaseException:
        _unlink_unchanged_private_file(directory_fd, name, complete)
        raise
    return complete


def _authorize_named_target(
    directory_fd: int,
    name: str,
) -> tuple[_StableIdentity, bool, _CompleteIdentity | None]:
    """Authorize one exact name, creating it only after an initial absent stat."""

    try:
        named_before = _named_stat(name, directory_fd)
    except FileNotFoundError:
        created = _create_target(directory_fd, name)
        return created.stable, True, created
    return (
        _validate_existing_target(
            directory_fd,
            name,
            named_before=named_before,
        ),
        False,
        None,
    )


def _validate_sqlite_sidecars(
    directory_fd: int,
    database_name: str,
    sidecars: tuple[_SQLiteSidecarAuthorization, ...] | list[_SQLiteSidecarAuthorization],
    *,
    strict_transient: bool,
) -> None:
    if tuple(sidecar.suffix for sidecar in sidecars) != _SQLITE_SIDECAR_SUFFIXES:
        _fail()
    for sidecar in sidecars:
        if sidecar.created != (sidecar.cleanup_identity is not None):
            _fail()
        try:
            current_identity = _validate_existing_target(
                directory_fd,
                f"{database_name}{sidecar.suffix}",
            )
        except FileNotFoundError:
            if sidecar.transient and not strict_transient:
                continue
            raise
        if sidecar.transient and not strict_transient:
            continue
        if current_identity != sidecar.identity:
            _fail()


def _matches_authorized_target(
    directory_fd: int,
    database_name: str,
    expected: _StableIdentity,
) -> bool:
    try:
        return _validate_existing_target(directory_fd, database_name) == expected
    except Exception:
        return False


def _cleanup_sqlite_sidecars(
    directory_fd: int,
    database_name: str,
    sidecars: tuple[_SQLiteSidecarAuthorization, ...] | list[_SQLiteSidecarAuthorization],
) -> None:
    for sidecar in reversed(sidecars):
        if sidecar.cleanup_identity is not None:
            _unlink_unchanged_private_file(
                directory_fd,
                f"{database_name}{sidecar.suffix}",
                sidecar.cleanup_identity,
            )


def _authorize(path: Path) -> StableFileAuthorization:
    directory_fd, parent_identity = _open_parent(path)
    target_identity: _StableIdentity | None = None
    target_cleanup_identity: _CompleteIdentity | None = None
    sidecars: list[_SQLiteSidecarAuthorization] = []
    authorization: StableFileAuthorization | None = None
    try:
        target_identity, _target_created, target_cleanup_identity = _authorize_named_target(
            directory_fd,
            path.name,
        )
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            sidecar_identity, sidecar_created, cleanup_identity = _authorize_named_target(
                directory_fd,
                f"{path.name}{suffix}",
            )
            sidecars.append(
                _SQLiteSidecarAuthorization(
                    suffix=suffix,
                    identity=sidecar_identity,
                    created=sidecar_created,
                    transient=suffix in _SQLITE_TRANSIENT_SIDECAR_SUFFIXES,
                    cleanup_identity=cleanup_identity,
                )
            )
        _verify_parent(path, directory_fd, parent_identity)
        if _validate_existing_target(directory_fd, path.name) != target_identity:
            _fail()
        _validate_sqlite_sidecars(
            directory_fd,
            path.name,
            sidecars,
            strict_transient=True,
        )
        authorization = StableFileAuthorization(
            path=os.fspath(path),
            _parent_identity=parent_identity,
            _target_identity=target_identity,
            _sqlite_sidecars=tuple(sidecars),
        )
    except BaseException:
        if target_identity is not None and _matches_authorized_target(
            directory_fd,
            path.name,
            target_identity,
        ):
            _cleanup_sqlite_sidecars(directory_fd, path.name, sidecars)
        if target_cleanup_identity is not None:
            _unlink_unchanged_private_file(
                directory_fd,
                path.name,
                target_cleanup_identity,
            )
        with suppress(OSError):
            os.close(directory_fd)
        raise
    try:
        os.close(directory_fd)
    except BaseException:
        with suppress(Exception):
            cleanup_fd, cleanup_parent = _open_parent(path)
            try:
                if (
                    cleanup_parent == parent_identity
                    and target_identity is not None
                    and _matches_authorized_target(
                        cleanup_fd,
                        path.name,
                        target_identity,
                    )
                ):
                    _cleanup_sqlite_sidecars(cleanup_fd, path.name, sidecars)
                    if target_cleanup_identity is not None:
                        _unlink_unchanged_private_file(
                            cleanup_fd,
                            path.name,
                            target_cleanup_identity,
                        )
            finally:
                os.close(cleanup_fd)
        # The descriptor state is unspecified after a failed close; never reuse a
        # numeric descriptor that another thread could already have acquired.
        raise
    if authorization is None:  # pragma: no cover - guarded by construction above
        _fail()
    return authorization


def _claim_inspected_sqlite_location(
    location: StableFileAuthorization,
    sidecar_locations: tuple[StableFileAuthorization, ...] | None = None,
) -> StableFileAuthorization:
    """Atomically turn exact private-location snapshots into SQLite authority."""

    if (
        type(location) is not StableFileAuthorization
        or location._kind is not _AuthorizationKind.PRIVATE_LOCATION
        or type(location.path) is not str
        or type(location._parent_identity) is not _StableIdentity
        or type(location._sqlite_sidecars) is not tuple
        or location._sqlite_sidecars
        or location._read_policy is not None
    ):
        _fail()
    expected_target = location._target_complete_identity
    expected_stable = location._target_identity
    if (expected_target is None) != (expected_stable is None) or (
        expected_target is not None
        and (
            type(expected_target) is not _CompleteIdentity
            or type(expected_stable) is not _StableIdentity
            or expected_target.stable != expected_stable
        )
    ):
        _fail()

    path = _copy_path(location.path)
    if os.fspath(path) != location.path:
        _fail()
    expected_parent = location._parent_identity
    expected_sidecars: (
        tuple[tuple[str, _CompleteIdentity | None, _StableIdentity | None], ...] | None
    ) = None
    if sidecar_locations is not None:
        if type(sidecar_locations) is not tuple or len(sidecar_locations) != len(
            _SQLITE_SIDECAR_SUFFIXES
        ):
            _fail()
        copied_sidecars: list[tuple[str, _CompleteIdentity | None, _StableIdentity | None]] = []
        for suffix, sidecar_location in zip(
            _SQLITE_SIDECAR_SUFFIXES,
            sidecar_locations,
            strict=True,
        ):
            if (
                type(sidecar_location) is not StableFileAuthorization
                or sidecar_location._kind is not _AuthorizationKind.PRIVATE_LOCATION
                or sidecar_location.path != f"{location.path}{suffix}"
                or sidecar_location._parent_identity != expected_parent
                or sidecar_location._sqlite_sidecars
                or sidecar_location._read_policy is not None
            ):
                _fail()
            complete = sidecar_location._target_complete_identity
            stable = sidecar_location._target_identity
            if (complete is None) != (stable is None) or (
                complete is not None
                and (
                    type(complete) is not _CompleteIdentity
                    or type(stable) is not _StableIdentity
                    or complete.stable != stable
                )
            ):
                _fail()
            copied_sidecars.append((suffix, complete, stable))
        expected_sidecars = tuple(copied_sidecars)

    directory_fd, parent_identity = _open_parent(path)
    target_identity: _StableIdentity | None = None
    target_cleanup_identity: _CompleteIdentity | None = None
    sidecars: list[_SQLiteSidecarAuthorization] = []
    authorization: StableFileAuthorization | None = None
    try:
        if parent_identity != expected_parent:
            _fail()
        _verify_parent(path, directory_fd, expected_parent)

        if expected_target is None:
            _require_absent_target(directory_fd, path.name)
        else:
            initial_target = _validate_private_location_target(
                directory_fd,
                path.name,
                expected=expected_target,
            )
            if initial_target.stable != expected_stable:
                _fail()
        if expected_sidecars is not None:
            for suffix, expected_complete, expected_sidecar_stable in expected_sidecars:
                sidecar_name = f"{path.name}{suffix}"
                if expected_complete is None:
                    _require_absent_target(directory_fd, sidecar_name)
                else:
                    initial_sidecar = _validate_private_location_target(
                        directory_fd,
                        sidecar_name,
                        expected=expected_complete,
                    )
                    if initial_sidecar.stable != expected_sidecar_stable:
                        _fail()

        if expected_target is None:
            target_cleanup_identity = _create_target(directory_fd, path.name)
            target_identity = target_cleanup_identity.stable
        else:
            target_identity = _validate_private_location_target(
                directory_fd,
                path.name,
                expected=expected_target,
            ).stable
            if target_identity != expected_stable:
                _fail()

        sidecar_expectations = (
            expected_sidecars
            if expected_sidecars is not None
            else tuple((suffix, None, None) for suffix in _SQLITE_SIDECAR_SUFFIXES)
        )
        for suffix, expected_complete, expected_sidecar_stable in sidecar_expectations:
            sidecar_name = f"{path.name}{suffix}"
            if expected_sidecars is None:
                sidecar_identity, sidecar_created, cleanup_identity = _authorize_named_target(
                    directory_fd, sidecar_name
                )
            elif expected_complete is None:
                cleanup_identity = _create_target(directory_fd, sidecar_name)
                sidecar_identity = cleanup_identity.stable
                sidecar_created = True
            else:
                sidecar_identity = _validate_private_location_target(
                    directory_fd,
                    sidecar_name,
                    expected=expected_complete,
                ).stable
                if sidecar_identity != expected_sidecar_stable:
                    _fail()
                sidecar_created = False
                cleanup_identity = None
            sidecars.append(
                _SQLiteSidecarAuthorization(
                    suffix=suffix,
                    identity=sidecar_identity,
                    created=sidecar_created,
                    transient=suffix in _SQLITE_TRANSIENT_SIDECAR_SUFFIXES,
                    cleanup_identity=cleanup_identity,
                )
            )

        _verify_parent(path, directory_fd, expected_parent)
        final_expected = target_cleanup_identity if expected_target is None else expected_target
        if final_expected is None:  # pragma: no cover - guarded by the branches above
            _fail()
        final_target = _validate_private_location_target(
            directory_fd,
            path.name,
            expected=final_expected,
        )
        if final_target.stable != target_identity:
            _fail()
        _validate_sqlite_sidecars(
            directory_fd,
            path.name,
            sidecars,
            strict_transient=True,
        )
        authorization = StableFileAuthorization(
            path=location.path,
            _parent_identity=expected_parent,
            _target_identity=target_identity,
            _sqlite_sidecars=tuple(sidecars),
        )
    except BaseException:
        if target_identity is not None and _matches_authorized_target(
            directory_fd,
            path.name,
            target_identity,
        ):
            _cleanup_sqlite_sidecars(directory_fd, path.name, sidecars)
        if target_cleanup_identity is not None:
            _unlink_unchanged_private_file(
                directory_fd,
                path.name,
                target_cleanup_identity,
            )
        with suppress(OSError):
            os.close(directory_fd)
        raise
    try:
        os.close(directory_fd)
    except BaseException:
        with suppress(Exception):
            cleanup_fd, cleanup_parent = _open_parent(path)
            try:
                if (
                    cleanup_parent == expected_parent
                    and target_identity is not None
                    and _matches_authorized_target(
                        cleanup_fd,
                        path.name,
                        target_identity,
                    )
                ):
                    _cleanup_sqlite_sidecars(cleanup_fd, path.name, sidecars)
                    if target_cleanup_identity is not None:
                        _unlink_unchanged_private_file(
                            cleanup_fd,
                            path.name,
                            target_cleanup_identity,
                        )
            finally:
                os.close(cleanup_fd)
        raise
    if authorization is None:  # pragma: no cover - guarded by construction above
        _fail()
    return authorization


def _revalidate_sqlite(
    authorization: StableFileAuthorization,
    *,
    strict_transient: bool,
) -> None:
    path = Path(authorization.path)
    expected_parent = authorization._parent_identity
    expected_target = authorization._target_identity
    if expected_parent is None or expected_target is None:
        _fail()
    directory_fd, parent_identity = _open_parent(path)
    try:
        if parent_identity != expected_parent:
            _fail()
        target_identity = _validate_existing_target(directory_fd, path.name)
        if target_identity != expected_target:
            _fail()
        _validate_sqlite_sidecars(
            directory_fd,
            path.name,
            authorization._sqlite_sidecars,
            strict_transient=strict_transient,
        )
        _verify_parent(path, directory_fd, expected_parent)
        if _validate_existing_target(directory_fd, path.name) != expected_target:
            _fail()
        _validate_sqlite_sidecars(
            directory_fd,
            path.name,
            authorization._sqlite_sidecars,
            strict_transient=strict_transient,
        )
    finally:
        os.close(directory_fd)


def _validate_read_authorization(authorization: StableFileAuthorization) -> None:
    expected = authorization._target_complete_identity
    policy = authorization._read_policy
    if (
        expected is None
        or authorization._target_identity != expected.stable
        or type(policy) is not StableReadPolicy
    ):
        _fail()
    path = Path(authorization.path)
    if policy is StableReadPolicy.LEGACY_COMPATIBILITY:
        named_before = os.stat(path, follow_symlinks=False)
        descriptor = os.open(path, _read_open_flags())
        try:
            opened_before = os.fstat(descriptor)
            opened_after = os.fstat(descriptor)
            named_after = os.stat(path, follow_symlinks=False)
            _require_stable_read_metadata(
                expected,
                (named_before, opened_before, opened_after, named_after),
                policy=policy,
            )
        finally:
            os.close(descriptor)
        return

    expected_parent = authorization._parent_identity
    if expected_parent is None:
        _fail()
    directory_fd, parent_identity = _open_parent(path)
    try:
        if parent_identity != expected_parent:
            _fail()
        named_before = _named_stat(path.name, directory_fd)
        descriptor = os.open(path.name, _read_open_flags(), dir_fd=directory_fd)
        try:
            opened_before = os.fstat(descriptor)
            _require_safe_acl(descriptor)
            opened_after = os.fstat(descriptor)
            _require_safe_acl(descriptor)
            named_after = _named_stat(path.name, directory_fd)
            _require_stable_read_metadata(
                expected,
                (named_before, opened_before, opened_after, named_after),
                policy=policy,
            )
        finally:
            os.close(descriptor)
        _verify_parent(path, directory_fd, expected_parent)
        _require_stable_read_metadata(
            expected,
            (_named_stat(path.name, directory_fd),),
            policy=policy,
        )
    finally:
        os.close(directory_fd)


def _validate_private_location_target(
    directory_fd: int,
    name: str,
    *,
    expected: _CompleteIdentity | None = None,
) -> _CompleteIdentity:
    named_before = _named_stat(name, directory_fd)
    if not _safe_target(named_before):
        _fail()
    descriptor = _open_existing_target(name, directory_fd)
    try:
        opened_before = os.fstat(descriptor)
        _require_safe_acl(descriptor)
        identity = _CompleteIdentity.from_stat(opened_before)
        opened_after = os.fstat(descriptor)
        _require_safe_acl(descriptor)
        named_after = _named_stat(name, directory_fd)
        if (
            not _safe_target(opened_before)
            or not _safe_target(opened_after)
            or not _safe_target(named_after)
            or not identity.matches(named_before)
            or not identity.matches(opened_after)
            or not identity.matches(named_after)
            or (expected is not None and identity != expected)
        ):
            _fail()
        return identity
    finally:
        os.close(descriptor)


def _require_absent_target(directory_fd: int, name: str) -> None:
    try:
        _named_stat(name, directory_fd)
    except FileNotFoundError:
        return
    _fail()


def _revalidate_private_location(authorization: StableFileAuthorization) -> None:
    path = Path(authorization.path)
    expected_parent = authorization._parent_identity
    expected = authorization._target_complete_identity
    if expected_parent is None or authorization._read_policy is not None:
        _fail()
    if (expected is None) != (authorization._target_identity is None):
        _fail()
    directory_fd, parent_identity = _open_parent(path)
    try:
        if parent_identity != expected_parent:
            _fail()
        if expected is None:
            _require_absent_target(directory_fd, path.name)
        else:
            _validate_private_location_target(directory_fd, path.name, expected=expected)
        _verify_parent(path, directory_fd, expected_parent)
        if expected is None:
            _require_absent_target(directory_fd, path.name)
        else:
            _validate_private_location_target(directory_fd, path.name, expected=expected)
    finally:
        os.close(directory_fd)


def _revalidate(
    authorization: StableFileAuthorization,
    *,
    strict_transient: bool,
) -> None:
    if type(authorization) is not StableFileAuthorization:
        _fail()
    if authorization._kind is _AuthorizationKind.SQLITE:
        _revalidate_sqlite(authorization, strict_transient=strict_transient)
    elif authorization._kind is _AuthorizationKind.STABLE_READ:
        _validate_read_authorization(authorization)
    elif authorization._kind is _AuthorizationKind.PRIVATE_LOCATION:
        _revalidate_private_location(authorization)
    else:  # pragma: no cover - Enum makes this unreachable without forged internals
        _fail()


def _cleanup_authorized_sqlite_sidecars(
    authorization: StableFileAuthorization,
) -> None:
    """Best-effort, identity-checked cleanup without reusing failed descriptors."""

    directory_fd: int | None = None
    try:
        path = Path(authorization.path)
        expected_parent = authorization._parent_identity
        expected_target = authorization._target_identity
        if (
            authorization._kind is not _AuthorizationKind.SQLITE
            or expected_parent is None
            or expected_target is None
        ):
            return
        directory_fd, parent_identity = _open_parent(path)
        if parent_identity != expected_parent:
            return
        _verify_parent(path, directory_fd, expected_parent)
        if not _matches_authorized_target(
            directory_fd,
            path.name,
            expected_target,
        ):
            return
        _cleanup_sqlite_sidecars(
            directory_fd,
            path.name,
            authorization._sqlite_sidecars,
        )
    except Exception:
        return
    finally:
        if directory_fd is not None:
            with suppress(OSError):
                os.close(directory_fd)


@dataclass(frozen=True, slots=True)
class _AuthorizationSlot:
    path_key: str
    parent_identity: _StableIdentity | None
    name_key: str
    target_identity: _StableIdentity | None


def _conservative_name_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _conservative_path_key(value: str) -> str:
    return _conservative_name_key(os.path.normpath(value))


def _authorization_slots(
    authorization: StableFileAuthorization,
) -> tuple[_AuthorizationSlot, ...]:
    path = Path(authorization.path)
    slots = [
        _AuthorizationSlot(
            path_key=_conservative_path_key(authorization.path),
            parent_identity=authorization._parent_identity,
            name_key=_conservative_name_key(path.name),
            target_identity=authorization._target_identity,
        )
    ]
    for sidecar in authorization._sqlite_sidecars:
        sidecar_path = f"{authorization.path}{sidecar.suffix}"
        slots.append(
            _AuthorizationSlot(
                path_key=_conservative_path_key(sidecar_path),
                parent_identity=authorization._parent_identity,
                name_key=_conservative_name_key(f"{path.name}{sidecar.suffix}"),
                target_identity=sidecar.identity,
            )
        )
    return tuple(slots)


def _same_inode(left: _StableIdentity, right: _StableIdentity) -> bool:
    return left.device == right.device and left.inode == right.inode


def _same_parent_slot(left: _AuthorizationSlot, right: _AuthorizationSlot) -> bool:
    return (
        left.parent_identity is not None
        and right.parent_identity is not None
        and _same_inode(left.parent_identity, right.parent_identity)
        and left.name_key == right.name_key
    )


def _authorizations_alias(
    left: StableFileAuthorization,
    right: StableFileAuthorization,
) -> bool:
    for left_slot in _authorization_slots(left):
        for right_slot in _authorization_slots(right):
            if left_slot.path_key == right_slot.path_key or _same_parent_slot(
                left_slot,
                right_slot,
            ):
                return True
            if (
                left_slot.target_identity is not None
                and right_slot.target_identity is not None
                and _same_inode(left_slot.target_identity, right_slot.target_identity)
            ):
                return True
    return False


@dataclass(frozen=True, slots=True)
class _AtomicTemporaryFile:
    name: str
    identity: _StableIdentity
    complete_identity: _CompleteIdentity


def _optional_named_stat(name: str, directory_fd: int) -> os.stat_result | None:
    try:
        return _named_stat(name, directory_fd)
    except FileNotFoundError:
        return None


def _atomic_inode_matches(
    metadata: os.stat_result | None,
    expected: _StableIdentity,
) -> bool:
    return (
        metadata is not None
        and stat.S_ISREG(metadata.st_mode)
        and metadata.st_uid == _current_user_id()
        and _same_inode(_StableIdentity.from_stat(metadata), expected)
    )


def _unlink_atomic_name(
    directory_fd: int,
    name: str,
    expected: _StableIdentity,
) -> bool:
    """Remove only a still-owned name, classifying unlink-then-raise correctly."""

    for _attempt in range(2):
        before = _optional_named_stat(name, directory_fd)
        if before is None:
            return True
        if not _atomic_inode_matches(before, expected):
            return False
        with suppress(Exception):
            os.unlink(name, dir_fd=directory_fd)
    return _optional_named_stat(name, directory_fd) is None


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if type(count) is not int or not 1 <= count <= len(view) - written:
            _fail()
        written += count


def _create_atomic_temporary_file(
    directory_fd: int,
    data: bytes,
    *,
    forbidden_names: frozenset[str],
) -> _AtomicTemporaryFile:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW
    descriptor: int | None = None
    name: str | None = None
    cleanup_identity: _StableIdentity | None = None
    for _attempt in range(32):
        candidate = f".saliencegate-atomic-{secrets.token_hex(16)}"
        if candidate in forbidden_names:
            continue
        try:
            descriptor = os.open(candidate, flags, 0o600, dir_fd=directory_fd)
        except FileExistsError:
            continue
        name = candidate
        break
    if descriptor is None or name is None:
        _fail()
    try:
        opened = os.fstat(descriptor)
        cleanup_identity = _StableIdentity.from_stat(opened)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != _current_user_id()
            or opened.st_nlink != 1
            or opened.st_size != 0
        ):
            _fail()
        os.fchmod(descriptor, 0o600)
        _require_safe_acl(descriptor)
        _write_all(descriptor, data)
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        _require_safe_acl(descriptor)
        named = _named_stat(name, directory_fd)
        complete = _CompleteIdentity.from_stat(after)
        if not _safe_target(after) or after.st_size != len(data) or not complete.matches(named):
            _fail()
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        if cleanup_identity is not None:
            _unlink_atomic_name(directory_fd, name, cleanup_identity)
        raise
    try:
        os.close(descriptor)
    except BaseException:
        _unlink_atomic_name(directory_fd, name, complete.stable)
        raise
    return _AtomicTemporaryFile(
        name=name,
        identity=complete.stable,
        complete_identity=complete,
    )


def _lock_replacement_target(
    directory_fd: int,
    authorization: StableFileAuthorization,
) -> int:
    expected = authorization._target_complete_identity
    if expected is None or _fcntl is None:
        _fail()
    name = Path(authorization.path).name
    descriptor = _open_existing_target(name, directory_fd)
    try:
        if not expected.matches(os.fstat(descriptor)):
            _fail()
        _require_safe_acl(descriptor)
        try:
            _fcntl.flock(descriptor, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in _UNSUPPORTED_OPERATION_ERRNOS:
                raise _UnsupportedFileOperationError from None
            raise
        if not expected.matches(os.fstat(descriptor)):
            _fail()
        _validate_private_location_target(directory_fd, name, expected=expected)
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        raise
    return descriptor


def _validate_publication_location_at(
    authorization: StableFileAuthorization,
    directory_fd: int,
) -> None:
    path = Path(authorization.path)
    expected_parent = authorization._parent_identity
    expected = authorization._target_complete_identity
    if (
        authorization._kind is not _AuthorizationKind.PRIVATE_LOCATION
        or expected_parent is None
        or not expected_parent.matches(os.fstat(directory_fd))
    ):
        _fail()
    if expected is None:
        _require_absent_target(directory_fd, path.name)
    else:
        _validate_private_location_target(directory_fd, path.name, expected=expected)
    _verify_parent(path, directory_fd, expected_parent)
    if expected is None:
        _require_absent_target(directory_fd, path.name)
    else:
        _validate_private_location_target(directory_fd, path.name, expected=expected)


def _target_is_atomic_inode(
    directory_fd: int,
    target_name: str,
    expected: _StableIdentity,
) -> bool:
    return _atomic_inode_matches(
        _optional_named_stat(target_name, directory_fd),
        expected,
    )


def _temporary_is_complete(
    directory_fd: int,
    temporary: _AtomicTemporaryFile,
) -> bool:
    metadata = _optional_named_stat(temporary.name, directory_fd)
    return (
        metadata is not None
        and _safe_target(metadata)
        and temporary.complete_identity.matches(metadata)
    )


def _verify_published_target(
    directory_fd: int,
    target_name: str,
    expected: _StableIdentity,
    expected_size: int,
) -> None:
    metadata = _named_stat(target_name, directory_fd)
    if (
        not _atomic_inode_matches(metadata, expected)
        or not _safe_target(metadata)
        or metadata.st_size != expected_size
    ):
        _fail()


def _link_temporary_no_clobber(
    directory_fd: int,
    temporary: _AtomicTemporaryFile,
    target_name: str,
) -> None:
    operation_error: Exception | None = None
    try:
        os.link(
            temporary.name,
            target_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except Exception as error:
        operation_error = error
    source = _optional_named_stat(temporary.name, directory_fd)
    target = _optional_named_stat(target_name, directory_fd)
    if (
        not _atomic_inode_matches(source, temporary.identity)
        or not _atomic_inode_matches(target, temporary.identity)
        or source is None
        or target is None
        or source.st_nlink != 2
        or target.st_nlink != 2
        or source.st_size != temporary.complete_identity.size
        or target.st_size != temporary.complete_identity.size
    ):
        if (
            isinstance(operation_error, OSError)
            and operation_error.errno in _UNSUPPORTED_OPERATION_ERRNOS
        ):
            raise _UnsupportedFileOperationError from None
        _fail()
    if not _unlink_atomic_name(directory_fd, temporary.name, temporary.identity):
        _fail()
    _verify_published_target(
        directory_fd,
        target_name,
        temporary.identity,
        temporary.complete_identity.size,
    )


def _rename_temporary_over_target(
    directory_fd: int,
    temporary: _AtomicTemporaryFile,
    target_name: str,
) -> None:
    operation_error: Exception | None = None
    try:
        os.rename(
            temporary.name,
            target_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
    except Exception as error:
        operation_error = error
    if _optional_named_stat(temporary.name, directory_fd) is not None:
        if (
            isinstance(operation_error, OSError)
            and operation_error.errno in _UNSUPPORTED_OPERATION_ERRNOS
        ):
            raise _UnsupportedFileOperationError from None
        _fail()
    _verify_published_target(
        directory_fd,
        target_name,
        temporary.identity,
        temporary.complete_identity.size,
    )


def _rollback_absent_publication(
    directory_fd: int,
    target_name: str,
    new_identity: _StableIdentity,
) -> bool:
    if not _target_is_atomic_inode(directory_fd, target_name, new_identity):
        return False
    removed = _unlink_atomic_name(directory_fd, target_name, new_identity)
    if removed:
        with suppress(OSError):
            os.fsync(directory_fd)
    return removed


def _rollback_replacement(
    directory_fd: int,
    path: Path,
    target_name: str,
    new_identity: _StableIdentity,
    backup: _AtomicTemporaryFile,
    old_data: bytes,
) -> bool:
    if not _target_is_atomic_inode(directory_fd, target_name, new_identity):
        return False
    source = backup
    if not _temporary_is_complete(directory_fd, backup):
        try:
            source = _create_atomic_temporary_file(
                directory_fd,
                old_data,
                forbidden_names=frozenset({target_name, backup.name}),
            )
        except Exception:
            return False
    for _attempt in range(2):
        if _optional_named_stat(source.name, directory_fd) is None:
            break
        if not _temporary_is_complete(directory_fd, source):
            return False
        with suppress(Exception):
            os.rename(
                source.name,
                target_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
    if _optional_named_stat(source.name, directory_fd) is not None:
        if source is not backup:
            _unlink_atomic_name(directory_fd, source.name, source.identity)
        return False
    if not _target_is_atomic_inode(directory_fd, target_name, source.identity):
        return False
    try:
        os.fsync(directory_fd)
        restored = _read_private_file(path, max(1, len(old_data)))
    except Exception:
        return False
    restored_exactly = restored.data == old_data
    if source is not backup:
        _unlink_atomic_name(directory_fd, backup.name, backup.identity)
    return restored_exactly


def _publish_atomic_file(
    publication: AtomicFilePublication,
    data: bytes,
    validate_published: Callable[[bytes], bool] | None,
) -> StableFileRead:
    authorization = publication.authorization
    path = Path(authorization.path)
    old_data = publication._replacement_data
    directory_fd, parent_identity = _open_parent(path)
    new_temporary: _AtomicTemporaryFile | None = None
    backup: _AtomicTemporaryFile | None = None
    replacement_lock_fd: int | None = None
    published = False
    committed = False
    preserve_backup = False
    try:
        if parent_identity != authorization._parent_identity:
            _fail()
        _validate_publication_location_at(authorization, directory_fd)
        if old_data is not None:
            replacement_lock_fd = _lock_replacement_target(directory_fd, authorization)
            _validate_publication_location_at(authorization, directory_fd)
        forbidden = frozenset({path.name})
        if old_data is not None:
            backup = _create_atomic_temporary_file(
                directory_fd,
                old_data,
                forbidden_names=forbidden,
            )
            forbidden = frozenset({path.name, backup.name})
        new_temporary = _create_atomic_temporary_file(
            directory_fd,
            data,
            forbidden_names=forbidden,
        )
        if backup is not None:
            os.fsync(directory_fd)
        _validate_publication_location_at(authorization, directory_fd)
        try:
            if old_data is None:
                _link_temporary_no_clobber(directory_fd, new_temporary, path.name)
            else:
                _rename_temporary_over_target(directory_fd, new_temporary, path.name)
        except BaseException:
            published = _target_is_atomic_inode(
                directory_fd,
                path.name,
                new_temporary.identity,
            )
            if (
                old_data is not None
                and not published
                and authorization._target_identity is not None
                and not _target_is_atomic_inode(
                    directory_fd,
                    path.name,
                    authorization._target_identity,
                )
            ):
                preserve_backup = True
            raise
        published = True
        os.fsync(directory_fd)
        reopened = _read_private_file(path, publication._maximum_bytes)
        if (
            reopened.data != data
            or reopened.authorization._target_identity != new_temporary.identity
        ):
            _fail()
        if validate_published is not None:
            try:
                validated = validate_published(reopened.data)
            except Exception:
                _fail()
            if validated is not True:
                _fail()
        reopened.authorization.revalidate()
        if backup is not None and not _unlink_atomic_name(
            directory_fd,
            backup.name,
            backup.identity,
        ):
            _fail()
        committed = True
        return reopened
    except BaseException:
        if published and new_temporary is not None:
            if old_data is None:
                _rollback_absent_publication(
                    directory_fd,
                    path.name,
                    new_temporary.identity,
                )
            elif backup is not None:
                restored = _rollback_replacement(
                    directory_fd,
                    path,
                    path.name,
                    new_temporary.identity,
                    backup,
                    old_data,
                )
                preserve_backup = not restored
        raise
    finally:
        if new_temporary is not None:
            _unlink_atomic_name(directory_fd, new_temporary.name, new_temporary.identity)
        if backup is not None and not preserve_backup and (committed or not published):
            _unlink_atomic_name(directory_fd, backup.name, backup.identity)
        if replacement_lock_fd is not None:
            if _fcntl is not None:
                with suppress(OSError):
                    _fcntl.flock(replacement_lock_fd, _fcntl.LOCK_UN)
            with suppress(OSError):
                os.close(replacement_lock_fd)
        with suppress(OSError):
            os.close(directory_fd)


def authorize_atomic_file_publication(
    path: str | os.PathLike[str],
    *,
    maximum_bytes: int,
    validate_replacement: Callable[[bytes], bool] | None = None,
) -> AtomicFilePublication:
    """Authorize one absent output or one callback-approved exact replacement."""

    if type(maximum_bytes) is not int or not 1 <= maximum_bytes < sys.maxsize:
        raise SecureFileBoundError()
    if validate_replacement is not None and not callable(validate_replacement):
        raise SecureFileError()
    result: AtomicFilePublication | None = None
    bound_exceeded = False
    unsupported = False
    failed = False
    try:
        _require_atomic_publication_platform()
        authorization = inspect_private_file_location(path)
        copied_path = Path(authorization.path)
        directory_fd, parent_identity = _open_parent(copied_path)
        try:
            if parent_identity != authorization._parent_identity:
                _fail()
            _validate_publication_location_at(authorization, directory_fd)
            _probe_directory_fsync(directory_fd)
        finally:
            os.close(directory_fd)

        replacement_data: bytes | None = None
        if authorization._target_complete_identity is not None:
            if validate_replacement is None:
                _fail()
            existing = _read_private_file(copied_path, maximum_bytes)
            if (
                existing.authorization._target_complete_identity
                != authorization._target_complete_identity
            ):
                _fail()
            try:
                accepted = validate_replacement(existing.data)
            except Exception:
                _fail()
            if accepted is not True:
                _fail()
            authorization.revalidate()
            replacement_data = existing.data
        result = AtomicFilePublication(
            authorization,
            maximum_bytes,
            replacement_data,
            _token=_ATOMIC_PUBLICATION_TOKEN,
        )
    except SecureFileBoundError:
        bound_exceeded = True
    except (SecureFileUnsupportedError, _UnsupportedFileOperationError):
        unsupported = True
    except KeyboardInterrupt:
        raise
    except Exception:
        failed = True
    if bound_exceeded:
        raise SecureFileBoundError()
    if unsupported:
        raise SecureFileUnsupportedError()
    if failed or result is None:
        raise SecureFileError()
    return result


def read_stable_file(
    path: str | os.PathLike[str],
    *,
    maximum_bytes: int,
    policy: StableReadPolicy,
) -> StableFileRead:
    """Read one regular, single-link file through a fixed stability policy."""

    if type(maximum_bytes) is not int or not 1 <= maximum_bytes < sys.maxsize:
        raise SecureFileBoundError()
    if type(policy) is not StableReadPolicy:
        raise SecureFileError()
    result: StableFileRead | None = None
    bound_exceeded = False
    try:
        if policy is StableReadPolicy.LEGACY_COMPATIBILITY:
            result = _read_legacy_file(_copy_legacy_read_path(path), maximum_bytes)
        else:
            _require_secure_platform()
            result = _read_private_file(_copy_path(path), maximum_bytes)
    except SecureFileBoundError:
        bound_exceeded = True
    except Exception:
        pass
    if bound_exceeded:
        raise SecureFileBoundError()
    if result is None:
        raise SecureFileError()
    return result


def inspect_private_file_location(
    path: str | os.PathLike[str],
) -> StableFileAuthorization:
    """Snapshot an existing private file or one exact absent slot without mutation."""

    authorization: StableFileAuthorization | None = None
    unsupported = False
    try:
        try:
            _require_secure_platform()
        except Exception:
            raise _UnsupportedFileOperationError from None
        copied_path = _copy_path(path)
        directory_fd, parent_identity = _open_parent(copied_path)
        try:
            try:
                identity = _validate_private_location_target(
                    directory_fd,
                    copied_path.name,
                )
            except FileNotFoundError:
                identity = None
                _require_absent_target(directory_fd, copied_path.name)
            _verify_parent(copied_path, directory_fd, parent_identity)
            if identity is None:
                _require_absent_target(directory_fd, copied_path.name)
            else:
                _validate_private_location_target(
                    directory_fd,
                    copied_path.name,
                    expected=identity,
                )
        finally:
            os.close(directory_fd)
        authorization = StableFileAuthorization(
            path=os.fspath(copied_path),
            _parent_identity=parent_identity,
            _target_identity=None if identity is None else identity.stable,
            _target_complete_identity=identity,
            _kind=_AuthorizationKind.PRIVATE_LOCATION,
        )
    except _UnsupportedFileOperationError:
        unsupported = True
    except OSError as error:
        unsupported = error.errno in _UNSUPPORTED_OPERATION_ERRNOS
    except Exception:
        pass
    if unsupported:
        raise SecureFileUnsupportedError()
    if authorization is None:
        raise SecureFileError()
    return authorization


def _claim_private_sqlite_location(
    location: StableFileAuthorization,
    *,
    sidecar_locations: tuple[StableFileAuthorization, ...] | None = None,
) -> StableFileAuthorization:
    """Claim inspected private locations without reopening a raw path boundary."""

    authorization: StableFileAuthorization | None = None
    failed = False
    try:
        _require_secure_platform()
        authorization = _claim_inspected_sqlite_location(location, sidecar_locations)
    except Exception:
        failed = True
    if failed or authorization is None:
        raise SecureFileError()
    return authorization


def authorize_private_sqlite_path(
    path: str | os.PathLike[str],
) -> StableFileAuthorization:
    """Precreate or authorize a private SQLite database and its journal sidecars."""

    authorization: StableFileAuthorization | None = None
    failed = False
    try:
        _require_secure_platform()
        authorization = _authorize(_copy_path(path))
    except Exception:
        failed = True
    if failed or authorization is None:
        raise SecureFileError()
    return authorization


__all__ = [
    "AtomicFilePublication",
    "SecureFileBoundError",
    "SecureFileError",
    "SecureFileUnsupportedError",
    "StableFileAuthorization",
    "StableFileRead",
    "StableReadPolicy",
    "authorize_atomic_file_publication",
    "authorize_private_sqlite_path",
    "inspect_private_file_location",
    "read_stable_file",
]
