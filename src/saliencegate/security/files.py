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
from functools import lru_cache
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
    PRIVATE_EXACT = "private_exact"
    PRIVATE_EXECUTABLE = "private_executable"


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


@dataclass(frozen=True, slots=True, repr=False)
class _PrivateDirectoryAuthorization:
    """An exact owner-private directory reached without following symlinks."""

    path: str
    _identity: _StableIdentity

    def revalidate(self) -> None:
        descriptor: int | None = None
        failed = False
        try:
            descriptor = _open_authorized_private_directory(self)
        except Exception:
            failed = True
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except Exception:
                    failed = True
        if failed:
            raise SecureFileError()

    def __repr__(self) -> str:
        return "_PrivateDirectoryAuthorization(<redacted>)"


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

    @property
    def target_exists(self) -> bool:
        """Whether the inspected private-location slot contained an exact target."""

        if self._kind is not _AuthorizationKind.PRIVATE_LOCATION:
            raise SecureFileError()
        return self._target_complete_identity is not None

    def _revalidate_before_sqlite_statements(self) -> None:
        """Pin every journal identity before SQLite is allowed to execute SQL."""

        if self._kind is not _AuthorizationKind.SQLITE:
            raise SecureFileError()
        self._checked_revalidate(strict_transient=True)

    def _revalidate_mutable_sqlite(self) -> None:
        """Revalidate a live multi-process SQLite boundary as contents change."""

        failed = False
        try:
            if self._kind is not _AuthorizationKind.SQLITE:
                _fail()
            _revalidate_mutable_sqlite(self)
        except Exception:
            failed = True
        if failed:
            raise SecureFileError()

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


def _preferred_failure(
    primary: BaseException | None,
    secondary: BaseException | None,
) -> BaseException | None:
    """Preserve the primary failure without swallowing cleanup cancellation."""

    if primary is None:
        return secondary
    if not isinstance(primary, Exception):
        return primary
    if secondary is not None and not isinstance(secondary, Exception):
        return secondary
    return primary


def _close_independent_descriptors(
    *descriptors: int | None,
) -> BaseException | None:
    """Attempt every distinct close and return the preferred failure, if any."""

    failure: BaseException | None = None
    attempted: set[int] = set()
    for descriptor in descriptors:
        if descriptor is None or descriptor in attempted:
            continue
        attempted.add(descriptor)
        try:
            os.close(descriptor)
        except BaseException as error:
            failure = _preferred_failure(failure, error)
    return failure


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


def _safe_executable_target(value: os.stat_result) -> bool:
    return (
        stat.S_ISREG(value.st_mode)
        and value.st_nlink == 1
        and value.st_uid == _current_user_id()
        and stat.S_IMODE(value.st_mode) == 0o700
    )


def _safe_read_target(value: os.stat_result, policy: StableReadPolicy) -> bool:
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        return False
    if policy is StableReadPolicy.LEGACY_COMPATIBILITY:
        return True
    if policy is StableReadPolicy.PRIVATE_EXACT:
        return _safe_target(value)
    if policy is StableReadPolicy.PRIVATE_EXECUTABLE:
        return _safe_executable_target(value)
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


def _read_private_file(
    path: Path,
    maximum_bytes: int,
    *,
    policy: StableReadPolicy,
) -> StableFileRead:
    if policy not in (
        StableReadPolicy.PRIVATE_OWNER,
        StableReadPolicy.PRIVATE_EXACT,
        StableReadPolicy.PRIVATE_EXECUTABLE,
    ):
        _fail()
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
                policy=policy,
                check_acl=True,
            )
        finally:
            os.close(descriptor)
        _verify_parent(path, directory_fd, parent_identity)
        final_named = _named_stat(path.name, directory_fd)
        _require_stable_read_metadata(
            identity,
            (final_named,),
            policy=policy,
        )
    finally:
        os.close(directory_fd)
    authorization = StableFileAuthorization(
        path=os.fspath(path),
        _parent_identity=parent_identity,
        _target_identity=identity.stable,
        _target_complete_identity=identity,
        _kind=_AuthorizationKind.STABLE_READ,
        _read_policy=policy,
    )
    return StableFileRead(data=data, authorization=authorization)


def _require_authorized_private_directory_descriptor(
    directory: _PrivateDirectoryAuthorization,
    directory_fd: int,
) -> None:
    if type(directory_fd) is not int or directory_fd < 0:
        _fail()
    opened = os.fstat(directory_fd)
    if not _safe_private_directory(opened) or not directory._identity.matches(opened):
        _fail()
    _require_safe_acl(directory_fd)


def _read_private_file_at_descriptor(
    directory: _PrivateDirectoryAuthorization,
    directory_fd: int,
    name: str,
    maximum_bytes: int,
) -> StableFileRead:
    """Read one child relative to an already pinned authorized directory."""

    copied_name = _copy_private_child_name(name)
    path = Path(directory.path) / copied_name
    _require_authorized_private_directory_descriptor(directory, directory_fd)
    named_before = _named_stat(copied_name, directory_fd)
    descriptor = os.open(copied_name, _read_open_flags(), dir_fd=directory_fd)
    try:
        data, identity = _read_opened_stable_file(
            descriptor,
            named_before,
            lambda: _named_stat(copied_name, directory_fd),
            maximum_bytes=maximum_bytes,
            policy=StableReadPolicy.PRIVATE_OWNER,
            check_acl=True,
        )
    finally:
        os.close(descriptor)
    _require_authorized_private_directory_descriptor(directory, directory_fd)
    final_named = _named_stat(copied_name, directory_fd)
    _require_stable_read_metadata(
        identity,
        (final_named,),
        policy=StableReadPolicy.PRIVATE_OWNER,
    )
    return StableFileRead(
        data=data,
        authorization=StableFileAuthorization(
            path=os.fspath(path),
            _parent_identity=directory._identity,
            _target_identity=identity.stable,
            _target_complete_identity=identity,
            _kind=_AuthorizationKind.STABLE_READ,
            _read_policy=StableReadPolicy.PRIVATE_OWNER,
        ),
    )


def _read_private_file_at(
    directory: _PrivateDirectoryAuthorization,
    name: str,
    maximum_bytes: int,
) -> StableFileRead:
    descriptor = _open_authorized_private_directory(directory)
    try:
        result = _read_private_file_at_descriptor(
            directory,
            descriptor,
            name,
            maximum_bytes,
        )
        directory.revalidate()
        return result
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class _DarwinAclApi:
    library: object
    get_acl: Callable[[int, int], int | None]
    get_entry: Callable[[int, int, object], int]
    get_tag: Callable[[object, object], int]
    free_acl: Callable[[int], int]


@lru_cache(maxsize=1)
def _darwin_acl_api() -> _DarwinAclApi:
    """Bind the macOS ACL API once while retaining its owning library handle."""

    library = ctypes.CDLL(None, use_errno=True)
    get_acl = library.acl_get_fd_np
    get_acl.argtypes = [ctypes.c_int, ctypes.c_int]
    get_acl.restype = ctypes.c_void_p
    get_entry = library.acl_get_entry
    get_entry.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    get_entry.restype = ctypes.c_int
    get_tag = library.acl_get_tag_type
    get_tag.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    get_tag.restype = ctypes.c_int
    free_acl = library.acl_free
    free_acl.argtypes = [ctypes.c_void_p]
    free_acl.restype = ctypes.c_int
    return _DarwinAclApi(
        library=library,
        get_acl=get_acl,
        get_entry=get_entry,
        get_tag=get_tag,
        free_acl=free_acl,
    )


def _darwin_acl_is_unsafe(descriptor: int, *, deny_only_allowed: bool) -> bool:
    """Fail-closed detection of unsafe macOS extended ACL entries.

    Deny-only ACLs are safe and common on macOS home-directory ancestors.  Any
    allow or unknown entry can weaken the mode-bit boundary and is rejected.
    The final parent, database, and sidecars reject every extended entry.
    """

    if sys.platform != "darwin":
        return False
    try:
        api = _darwin_acl_api()

        ctypes.set_errno(0)
        acl = api.get_acl(descriptor, _DARWIN_ACL_TYPE_EXTENDED)
        if acl is None:
            return ctypes.get_errno() != errno.ENOENT

        unsafe = not deny_only_allowed
        try:
            if deny_only_allowed:
                entry = ctypes.c_void_p()
                entry_id = _DARWIN_ACL_FIRST_ENTRY
                while True:
                    ctypes.set_errno(0)
                    result = api.get_entry(acl, entry_id, ctypes.byref(entry))
                    if result == -1:
                        unsafe = ctypes.get_errno() != errno.EINVAL
                        break
                    if result != 0 or entry.value is None:
                        unsafe = True
                        break
                    tag = ctypes.c_int()
                    if api.get_tag(entry, ctypes.byref(tag)) != 0:
                        unsafe = True
                        break
                    if tag.value != _DARWIN_ACL_EXTENDED_DENY:
                        unsafe = True
                        break
                    entry_id = _DARWIN_ACL_NEXT_ENTRY
        finally:
            if api.free_acl(acl) != 0:
                unsafe = True
        return unsafe
    except Exception:
        return True


def _require_safe_acl(descriptor: int, *, deny_only_allowed: bool = False) -> None:
    if _darwin_acl_is_unsafe(descriptor, deny_only_allowed=deny_only_allowed):
        _fail()


def _named_stat(name: str, directory_fd: int) -> os.stat_result:
    return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)


def _safe_private_directory(value: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(value.st_mode)
        and value.st_uid == _current_user_id()
        and stat.S_IMODE(value.st_mode) == 0o700
    )


def _require_private_directory_platform() -> None:
    _require_secure_platform()
    if os.mkdir not in os.supports_dir_fd:
        _fail()


def _open_or_create_private_directory_chain(path: Path) -> tuple[int, _StableIdentity]:
    """Create a missing suffix with mkdirat while every ancestor remains pinned."""

    if not path.is_absolute() or path.anchor != os.sep or path == Path(os.sep):
        _fail()
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(os.sep, flags)
    try:
        root = os.fstat(descriptor)
        if not _safe_ancestor(root):
            _fail()
        _require_safe_acl(descriptor, deny_only_allowed=True)
        components = path.parts[1:]
        for index, component in enumerate(components):
            if component in ("", ".", ".."):
                _fail()
            created = False
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                    created = True
                except FileExistsError:
                    pass
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            try:
                if created:
                    os.fchmod(next_descriptor, 0o700)
                opened = os.fstat(next_descriptor)
                named = _named_stat(component, descriptor)
                identity = _StableIdentity.from_stat(opened)
                is_leaf = index == len(components) - 1
                if not identity.matches(named):
                    _fail()
                if created or is_leaf:
                    if not _safe_private_directory(opened) or not _safe_private_directory(named):
                        _fail()
                    _require_safe_acl(next_descriptor)
                else:
                    if not _safe_ancestor(opened) or not _safe_ancestor(named):
                        _fail()
                    _require_safe_acl(next_descriptor, deny_only_allowed=True)
            except BaseException:
                os.close(next_descriptor)
                raise
            previous_descriptor = descriptor
            descriptor = next_descriptor
            os.close(previous_descriptor)
        opened = os.fstat(descriptor)
        if not _safe_private_directory(opened):
            _fail()
        _require_safe_acl(descriptor)
        return descriptor, _StableIdentity.from_stat(opened)
    except BaseException:
        os.close(descriptor)
        raise


def _open_safe_ancestor_directory(path: Path) -> tuple[int, _StableIdentity]:
    """Open one existing safe ancestor, allowing a sticky shared leaf."""

    if not path.is_absolute() or path.anchor != os.sep:
        _fail()
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(os.sep, flags)
    try:
        opened = os.fstat(descriptor)
        if not _safe_ancestor(opened):
            _fail()
        _require_safe_acl(descriptor, deny_only_allowed=True)
        for component in path.parts[1:]:
            if component in ("", ".", ".."):
                _fail()
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            try:
                opened = os.fstat(next_descriptor)
                named = _named_stat(component, descriptor)
                identity = _StableIdentity.from_stat(opened)
                if (
                    not _safe_ancestor(opened)
                    or not _safe_ancestor(named)
                    or not identity.matches(named)
                ):
                    _fail()
                _require_safe_acl(next_descriptor, deny_only_allowed=True)
            except BaseException:
                os.close(next_descriptor)
                raise
            previous_descriptor = descriptor
            descriptor = next_descriptor
            os.close(previous_descriptor)
        opened = os.fstat(descriptor)
        if not _safe_ancestor(opened):
            _fail()
        _require_safe_acl(descriptor, deny_only_allowed=True)
        return descriptor, _StableIdentity.from_stat(opened)
    except BaseException:
        os.close(descriptor)
        raise


def _inspect_private_directory_boundary(path: Path) -> bool:
    """Return presence after authenticating an exact leaf or absent suffix."""

    if not path.is_absolute() or path.anchor != os.sep or path == Path(os.sep):
        _fail()
    components = path.parts[1:]
    if not components or any(component in ("", ".", "..") for component in components):
        _fail()
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(os.sep, flags)
    current_path = Path(os.sep)
    try:
        root = os.fstat(descriptor)
        if not _safe_ancestor(root):
            _fail()
        _require_safe_acl(descriptor, deny_only_allowed=True)
        for index, component in enumerate(components):
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                expected_parent = _StableIdentity.from_stat(os.fstat(descriptor))
                for _attempt in range(2):
                    _require_absent_target(descriptor, component)
                    fresh_descriptor, fresh_identity = _open_safe_ancestor_directory(current_path)
                    try:
                        if fresh_identity != expected_parent:
                            _fail()
                        _require_absent_target(fresh_descriptor, component)
                    finally:
                        os.close(fresh_descriptor)
                _require_absent_target(descriptor, component)
                return False
            try:
                opened = os.fstat(next_descriptor)
                named = _named_stat(component, descriptor)
                identity = _StableIdentity.from_stat(opened)
                is_leaf = index == len(components) - 1
                if not identity.matches(named):
                    _fail()
                if is_leaf:
                    if not _safe_private_directory(opened) or not _safe_private_directory(named):
                        _fail()
                    _require_safe_acl(next_descriptor)
                else:
                    if not _safe_ancestor(opened) or not _safe_ancestor(named):
                        _fail()
                    _require_safe_acl(next_descriptor, deny_only_allowed=True)
            except BaseException:
                os.close(next_descriptor)
                raise
            previous_descriptor = descriptor
            descriptor = next_descriptor
            os.close(previous_descriptor)
            current_path /= component
        snapshot_descriptor, _identity = _open_private_directory_snapshot(path)
        os.close(snapshot_descriptor)
        return True
    finally:
        os.close(descriptor)


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


def _open_private_directory_snapshot(path: Path) -> tuple[int, _StableIdentity]:
    """Open one exact 0700 leaf and confirm a second fresh walk names it."""

    descriptor, identity = _open_directory_chain(path)
    try:
        opened = os.fstat(descriptor)
        if not _safe_private_directory(opened) or not identity.matches(opened):
            _fail()
        named_descriptor, named_identity = _open_directory_chain(path)
        try:
            named = os.fstat(named_descriptor)
            if (
                not _safe_private_directory(named)
                or not identity.matches(named)
                or named_identity != identity
            ):
                _fail()
        finally:
            os.close(named_descriptor)
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _authorize_private_directory(
    path: str | os.PathLike[str],
    *,
    create: bool,
) -> _PrivateDirectoryAuthorization:
    """Authorize an exact private directory, optionally creating a missing suffix."""

    authorization: _PrivateDirectoryAuthorization | None = None
    descriptor: int | None = None
    operation_failure: BaseException | None = None
    try:
        if type(create) is not bool:
            _fail()
        if create:
            _require_private_directory_platform()
        else:
            _require_secure_platform()
        copied_path = _copy_path(path)
        if create:
            descriptor, created_identity = _open_or_create_private_directory_chain(copied_path)
            close_failure = _close_independent_descriptors(descriptor)
            descriptor = None
            if close_failure is not None:
                raise close_failure
            descriptor, identity = _open_private_directory_snapshot(copied_path)
            if identity != created_identity:
                _fail()
        else:
            descriptor, identity = _open_private_directory_snapshot(copied_path)
        authorization = _PrivateDirectoryAuthorization(
            path=os.fspath(copied_path),
            _identity=identity,
        )
    except BaseException as error:
        operation_failure = error
    finally:
        close_failure = _close_independent_descriptors(descriptor)
        operation_failure = _preferred_failure(operation_failure, close_failure)
    if operation_failure is not None and not isinstance(operation_failure, Exception):
        raise operation_failure
    if operation_failure is not None or authorization is None:
        raise SecureFileError()
    return authorization


def _open_authorized_private_directory(
    authorization: _PrivateDirectoryAuthorization,
) -> int:
    """Return a descriptor for the exact directory captured by an authorization."""

    if (
        type(authorization) is not _PrivateDirectoryAuthorization
        or type(authorization.path) is not str
        or type(authorization._identity) is not _StableIdentity
    ):
        _fail()
    copied_path = _copy_path(authorization.path)
    if os.fspath(copied_path) != authorization.path:
        _fail()
    descriptor, identity = _open_private_directory_snapshot(copied_path)
    if identity != authorization._identity:
        os.close(descriptor)
        _fail()
    return descriptor


def _authorize_private_directory_child(
    parent: _PrivateDirectoryAuthorization,
    name: str,
    *,
    create: bool,
) -> _PrivateDirectoryAuthorization:
    """Authorize one direct child relative to an already authorized parent."""

    authorization: _PrivateDirectoryAuthorization | None = None
    parent_fd: int | None = None
    child_fd: int | None = None
    operation_failure: BaseException | None = None
    try:
        if type(create) is not bool:
            _fail()
        _require_private_directory_platform()
        copied_name = _copy_private_child_name(name)
        parent_fd = _open_authorized_private_directory(parent)
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        created = False
        try:
            child_fd = os.open(copied_name, flags, dir_fd=parent_fd)
        except FileNotFoundError:
            if not create:
                raise
            try:
                os.mkdir(copied_name, 0o700, dir_fd=parent_fd)
                created = True
            except FileExistsError:
                pass
            child_fd = os.open(copied_name, flags, dir_fd=parent_fd)
        if created:
            os.fchmod(child_fd, 0o700)
        opened = os.fstat(child_fd)
        named = _named_stat(copied_name, parent_fd)
        identity = _StableIdentity.from_stat(opened)
        if (
            not _safe_private_directory(opened)
            or not _safe_private_directory(named)
            or not identity.matches(named)
        ):
            _fail()
        _require_safe_acl(child_fd)
        _require_authorized_private_directory_descriptor(parent, parent_fd)
        authorization = _PrivateDirectoryAuthorization(
            path=os.fspath(Path(parent.path) / copied_name),
            _identity=identity,
        )
    except BaseException as error:
        operation_failure = error
    finally:
        close_failure = _close_independent_descriptors(child_fd, parent_fd)
        operation_failure = _preferred_failure(operation_failure, close_failure)
    if operation_failure is not None and not isinstance(operation_failure, Exception):
        raise operation_failure
    if operation_failure is not None or authorization is None:
        raise SecureFileError()
    parent.revalidate()
    authorization.revalidate()
    return authorization


def _copy_private_child_name(name: str) -> str:
    if (
        type(name) is not str
        or not name
        or name in (".", "..")
        or os.sep in name
        or (os.altsep is not None and os.altsep in name)
        or "\0" in name
    ):
        _fail()
    os.fsencode(name)
    return name


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


def _validate_existing_mutable_target(
    directory_fd: int,
    name: str,
    *,
    expected: _StableIdentity | None = None,
) -> _StableIdentity:
    """Pin security identity while allowing SQLite-managed bytes to change."""

    named_before = _named_stat(name, directory_fd)
    if not _safe_target(named_before):
        _fail()
    descriptor = _open_existing_target(name, directory_fd)
    try:
        opened_before = os.fstat(descriptor)
        if not _safe_target(opened_before):
            _fail()
        _require_safe_acl(descriptor)
        identity = _StableIdentity.from_stat(opened_before)
        opened_after = os.fstat(descriptor)
        _require_safe_acl(descriptor)
        named_after = _named_stat(name, directory_fd)
        if (
            not _safe_target(opened_after)
            or not _safe_target(named_after)
            or _StableIdentity.from_stat(named_before) != identity
            or _StableIdentity.from_stat(opened_after) != identity
            or _StableIdentity.from_stat(named_after) != identity
            or (expected is not None and identity != expected)
        ):
            _fail()
        return identity
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


def _validate_mutable_sqlite_sidecars(
    directory_fd: int,
    database_name: str,
    sidecars: tuple[_SQLiteSidecarAuthorization, ...],
) -> None:
    if tuple(sidecar.suffix for sidecar in sidecars) != _SQLITE_SIDECAR_SUFFIXES:
        _fail()
    for sidecar in sidecars:
        try:
            _validate_existing_mutable_target(
                directory_fd,
                f"{database_name}{sidecar.suffix}",
                expected=None if sidecar.transient else sidecar.identity,
            )
        except FileNotFoundError:
            if sidecar.transient:
                continue
            raise


def _revalidate_mutable_sqlite(authorization: StableFileAuthorization) -> None:
    path = Path(authorization.path)
    expected_parent = authorization._parent_identity
    expected_target = authorization._target_identity
    if expected_parent is None or expected_target is None:
        _fail()
    directory_fd, parent_identity = _open_parent(path)
    try:
        if parent_identity != expected_parent:
            _fail()
        _validate_existing_mutable_target(
            directory_fd,
            path.name,
            expected=expected_target,
        )
        _validate_mutable_sqlite_sidecars(
            directory_fd,
            path.name,
            authorization._sqlite_sidecars,
        )
        _verify_parent(path, directory_fd, expected_parent)
        _validate_existing_mutable_target(
            directory_fd,
            path.name,
            expected=expected_target,
        )
        _validate_mutable_sqlite_sidecars(
            directory_fd,
            path.name,
            authorization._sqlite_sidecars,
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


def _validate_private_read_target_at(
    directory_fd: int,
    name: str,
    expected: _CompleteIdentity,
) -> None:
    """Revalidate one PRIVATE_OWNER read against an already pinned parent."""

    if type(expected) is not _CompleteIdentity:
        _fail()
    named_before = _named_stat(name, directory_fd)
    descriptor = os.open(name, _read_open_flags(), dir_fd=directory_fd)
    try:
        opened_before = os.fstat(descriptor)
        _require_safe_acl(descriptor)
        opened_after = os.fstat(descriptor)
        _require_safe_acl(descriptor)
        named_after = _named_stat(name, directory_fd)
        _require_stable_read_metadata(
            expected,
            (named_before, opened_before, opened_after, named_after),
            policy=StableReadPolicy.PRIVATE_OWNER,
        )
    finally:
        os.close(descriptor)


def _staged_private_read_metadata_matches(
    expected: _CompleteIdentity,
    value: os.stat_result,
) -> bool:
    """Match a renamed target while allowing the rename-induced ctime change."""

    return (
        _safe_read_target(value, StableReadPolicy.PRIVATE_OWNER)
        and expected.stable.matches(value)
        and value.st_nlink == expected.link_count
        and value.st_size == expected.size
        and value.st_mtime_ns == expected.modified_ns
    )


def _validate_staged_private_read_target_at(
    directory_fd: int,
    name: str,
    expected: _CompleteIdentity,
) -> None:
    if type(expected) is not _CompleteIdentity:
        _fail()
    named_before = _named_stat(name, directory_fd)
    descriptor = os.open(name, _read_open_flags(), dir_fd=directory_fd)
    try:
        opened_before = os.fstat(descriptor)
        _require_safe_acl(descriptor)
        opened_after = os.fstat(descriptor)
        _require_safe_acl(descriptor)
        named_after = _named_stat(name, directory_fd)
        if any(
            not _staged_private_read_metadata_matches(expected, value)
            for value in (named_before, opened_before, opened_after, named_after)
        ):
            _fail()
    finally:
        os.close(descriptor)


def _require_private_delete_platform() -> None:
    try:
        _require_secure_platform()
    except Exception:
        raise _UnsupportedFileOperationError from None
    if (
        not hasattr(os, "fsync")
        or not hasattr(os, "link")
        or not hasattr(os, "mkdir")
        or not hasattr(os, "rename")
        or not hasattr(os, "rmdir")
        or os.link not in os.supports_dir_fd
        or os.link not in os.supports_follow_symlinks
        or os.mkdir not in os.supports_dir_fd
        or os.rename not in os.supports_dir_fd
        or os.rmdir not in os.supports_dir_fd
    ):
        raise _UnsupportedFileOperationError


def _private_delete_stage_matches(
    metadata: os.stat_result | None,
    expected: _StableIdentity,
) -> bool:
    return (
        metadata is not None
        and stat.S_ISDIR(metadata.st_mode)
        and stat.S_IMODE(metadata.st_mode) == 0o700
        and metadata.st_uid == _current_user_id()
        and expected.matches(metadata)
    )


def _create_private_delete_stage(
    directory_fd: int,
    *,
    forbidden_name: str,
) -> tuple[str, int, _StableIdentity]:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    for _attempt in range(32):
        name = f".saliencegate-delete-{secrets.token_hex(16)}"
        if name == forbidden_name:
            continue
        try:
            os.mkdir(name, 0o700, dir_fd=directory_fd)
        except FileExistsError:
            continue
        stage_fd: int | None = None
        identity: _StableIdentity | None = None
        try:
            stage_fd = os.open(name, flags, dir_fd=directory_fd)
            os.fchmod(stage_fd, 0o700)
            opened = os.fstat(stage_fd)
            identity = _StableIdentity.from_stat(opened)
            _require_safe_acl(stage_fd)
            named = _named_stat(name, directory_fd)
            if not _private_delete_stage_matches(opened, identity) or not identity.matches(named):
                _fail()
            return name, stage_fd, identity
        except BaseException:
            if stage_fd is not None:
                with suppress(OSError):
                    os.close(stage_fd)
            if identity is not None and _private_delete_stage_matches(
                _optional_named_stat(name, directory_fd),
                identity,
            ):
                with suppress(OSError):
                    os.rmdir(name, dir_fd=directory_fd)
            raise
    _fail()


def _stage_authorized_private_name(
    directory_fd: int,
    name: str,
    stage_fd: int,
) -> None:
    """Atomically move the current name into a freshly created private stage."""

    _require_absent_target(stage_fd, "entry")
    os.rename(
        name,
        "entry",
        src_dir_fd=directory_fd,
        dst_dir_fd=stage_fd,
    )


def _same_named_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return _same_inode(_StableIdentity.from_stat(left), _StableIdentity.from_stat(right))


def _restore_staged_private_name(
    directory_fd: int,
    name: str,
    stage_fd: int,
) -> bool:
    """Best-effort no-clobber restoration; never overwrite an occupied name."""

    try:
        staged = _optional_named_stat("entry", stage_fd)
        if staged is None:
            return True
        if _optional_named_stat(name, directory_fd) is not None:
            return False
        os.link(
            "entry",
            name,
            src_dir_fd=stage_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        restored = _named_stat(name, directory_fd)
        staged_after = _named_stat("entry", stage_fd)
        if not _same_named_inode(staged, restored) or not _same_named_inode(
            restored,
            staged_after,
        ):
            return False
        os.unlink("entry", dir_fd=stage_fd)
        if _optional_named_stat("entry", stage_fd) is not None:
            return False
        _probe_directory_fsync(stage_fd)
        _probe_directory_fsync(directory_fd)
        restored_after = _named_stat(name, directory_fd)
        return _same_named_inode(restored, restored_after)
    except Exception:
        return False


def _remove_private_delete_stage(
    directory_fd: int,
    name: str,
    expected: _StableIdentity,
) -> bool:
    try:
        metadata = _optional_named_stat(name, directory_fd)
        if metadata is None:
            return True
        if not _private_delete_stage_matches(metadata, expected):
            return False
        os.rmdir(name, dir_fd=directory_fd)
        return _optional_named_stat(name, directory_fd) is None
    except Exception:
        return False


def _delete_authorized_private_file(authorization: StableFileAuthorization) -> None:
    """Atomically stage and then delete one exact private stable read."""

    expected = authorization._target_complete_identity
    expected_parent = authorization._parent_identity
    if (
        authorization._kind is not _AuthorizationKind.STABLE_READ
        or authorization._read_policy
        not in (
            StableReadPolicy.PRIVATE_OWNER,
            StableReadPolicy.PRIVATE_EXACT,
            StableReadPolicy.PRIVATE_EXECUTABLE,
        )
        or authorization._sqlite_sidecars
        or type(expected) is not _CompleteIdentity
        or type(expected_parent) is not _StableIdentity
        or authorization._target_identity != expected.stable
        or type(authorization.path) is not str
    ):
        _fail()
    path = _copy_path(authorization.path)
    if os.fspath(path) != authorization.path:
        _fail()
    directory_fd, parent_identity = _open_parent(path)
    stage_name: str | None = None
    stage_fd: int | None = None
    stage_identity: _StableIdentity | None = None
    staged = False
    try:
        if parent_identity != expected_parent:
            _fail()
        _probe_directory_fsync(directory_fd)
        _validate_private_read_target_at(directory_fd, path.name, expected)
        stage_name, stage_fd, stage_identity = _create_private_delete_stage(
            directory_fd,
            forbidden_name=path.name,
        )
        _probe_directory_fsync(stage_fd)
        _verify_parent(path, directory_fd, expected_parent)
        _validate_private_read_target_at(directory_fd, path.name, expected)
        _stage_authorized_private_name(directory_fd, path.name, stage_fd)
        staged = True
        _require_absent_target(directory_fd, path.name)
        _probe_directory_fsync(directory_fd)
        _probe_directory_fsync(stage_fd)
        try:
            _validate_staged_private_read_target_at(stage_fd, "entry", expected)
        except Exception:
            if _restore_staged_private_name(directory_fd, path.name, stage_fd):
                staged = False
            _fail()
        os.unlink("entry", dir_fd=stage_fd)
        staged = _optional_named_stat("entry", stage_fd) is not None
        if staged:
            _fail()
        _probe_directory_fsync(stage_fd)
        os.close(stage_fd)
        stage_fd = None
        if not _remove_private_delete_stage(directory_fd, stage_name, stage_identity):
            _fail()
        _probe_directory_fsync(directory_fd)
        _verify_parent(path, directory_fd, expected_parent)
        _require_absent_target(directory_fd, path.name)
    finally:
        if (
            staged
            and stage_fd is not None
            and _restore_staged_private_name(
                directory_fd,
                path.name,
                stage_fd,
            )
        ):
            staged = False
        if stage_fd is not None:
            with suppress(OSError):
                os.close(stage_fd)
        if not staged and stage_name is not None and stage_identity is not None:
            _remove_private_delete_stage(directory_fd, stage_name, stage_identity)
        os.close(directory_fd)


def _delete_authorized_private_file_at_descriptor(
    directory: _PrivateDirectoryAuthorization,
    directory_fd: int,
    authorization: StableFileAuthorization,
) -> None:
    """Delete one stable child using the transaction's pinned directory descriptor."""

    _require_private_delete_platform()
    expected = authorization._target_complete_identity
    if (
        type(directory) is not _PrivateDirectoryAuthorization
        or type(authorization) is not StableFileAuthorization
        or authorization._kind is not _AuthorizationKind.STABLE_READ
        or authorization._read_policy
        not in (
            StableReadPolicy.PRIVATE_OWNER,
            StableReadPolicy.PRIVATE_EXACT,
            StableReadPolicy.PRIVATE_EXECUTABLE,
        )
        or authorization._sqlite_sidecars
        or type(expected) is not _CompleteIdentity
        or authorization._parent_identity != directory._identity
        or authorization._target_identity != expected.stable
        or type(authorization.path) is not str
    ):
        _fail()
    path = _copy_path(authorization.path)
    if path.parent != Path(directory.path) or os.fspath(path) != authorization.path:
        _fail()
    name = _copy_private_child_name(path.name)
    stage_name: str | None = None
    stage_fd: int | None = None
    stage_identity: _StableIdentity | None = None
    staged = False
    _require_authorized_private_directory_descriptor(directory, directory_fd)
    try:
        _probe_directory_fsync(directory_fd)
        _validate_private_read_target_at(directory_fd, name, expected)
        stage_name, stage_fd, stage_identity = _create_private_delete_stage(
            directory_fd,
            forbidden_name=name,
        )
        _probe_directory_fsync(stage_fd)
        _require_authorized_private_directory_descriptor(directory, directory_fd)
        _validate_private_read_target_at(directory_fd, name, expected)
        _stage_authorized_private_name(directory_fd, name, stage_fd)
        staged = True
        _require_absent_target(directory_fd, name)
        _probe_directory_fsync(directory_fd)
        _probe_directory_fsync(stage_fd)
        try:
            _validate_staged_private_read_target_at(stage_fd, "entry", expected)
        except Exception:
            if _restore_staged_private_name(directory_fd, name, stage_fd):
                staged = False
            _fail()
        os.unlink("entry", dir_fd=stage_fd)
        staged = _optional_named_stat("entry", stage_fd) is not None
        if staged:
            _fail()
        _probe_directory_fsync(stage_fd)
        os.close(stage_fd)
        stage_fd = None
        if not _remove_private_delete_stage(directory_fd, stage_name, stage_identity):
            _fail()
        _probe_directory_fsync(directory_fd)
        _require_authorized_private_directory_descriptor(directory, directory_fd)
        _require_absent_target(directory_fd, name)
    finally:
        if (
            staged
            and stage_fd is not None
            and _restore_staged_private_name(directory_fd, name, stage_fd)
        ):
            staged = False
        if stage_fd is not None:
            with suppress(OSError):
                os.close(stage_fd)
        if not staged and stage_name is not None and stage_identity is not None:
            _remove_private_delete_stage(directory_fd, stage_name, stage_identity)


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


def _inspect_private_file_location_at_descriptor(
    directory: _PrivateDirectoryAuthorization,
    directory_fd: int,
    name: str,
) -> StableFileAuthorization:
    """Snapshot an exact child slot without resolving its directory path again."""

    copied_name = _copy_private_child_name(name)
    _require_authorized_private_directory_descriptor(directory, directory_fd)
    try:
        identity = _validate_private_location_target(directory_fd, copied_name)
    except FileNotFoundError:
        identity = None
        _require_absent_target(directory_fd, copied_name)
    _require_authorized_private_directory_descriptor(directory, directory_fd)
    if identity is None:
        _require_absent_target(directory_fd, copied_name)
    else:
        _validate_private_location_target(
            directory_fd,
            copied_name,
            expected=identity,
        )
    return StableFileAuthorization(
        path=os.fspath(Path(directory.path) / copied_name),
        _parent_identity=directory._identity,
        _target_identity=None if identity is None else identity.stable,
        _target_complete_identity=identity,
        _kind=_AuthorizationKind.PRIVATE_LOCATION,
    )


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


def _validate_publication_location_at_descriptor(
    directory: _PrivateDirectoryAuthorization,
    directory_fd: int,
    authorization: StableFileAuthorization,
) -> None:
    path = Path(authorization.path)
    expected = authorization._target_complete_identity
    if (
        authorization._kind is not _AuthorizationKind.PRIVATE_LOCATION
        or authorization._parent_identity != directory._identity
        or path.parent != Path(directory.path)
    ):
        _fail()
    _require_authorized_private_directory_descriptor(directory, directory_fd)
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
        restored = _read_private_file(
            path,
            max(1, len(old_data)),
            policy=StableReadPolicy.PRIVATE_EXACT,
        )
    except Exception:
        return False
    restored_exactly = restored.data == old_data
    if source is not backup:
        _unlink_atomic_name(directory_fd, backup.name, backup.identity)
    return restored_exactly


def _rollback_replacement_at_descriptor(
    directory: _PrivateDirectoryAuthorization,
    directory_fd: int,
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
        restored = _read_private_file_at_descriptor(
            directory,
            directory_fd,
            target_name,
            max(1, len(old_data)),
        )
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
        reopened = _read_private_file(
            path,
            publication._maximum_bytes,
            policy=StableReadPolicy.PRIVATE_EXACT,
        )
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


def _publish_private_file_at_descriptor_unchecked(
    directory: _PrivateDirectoryAuthorization,
    directory_fd: int,
    authorization: StableFileAuthorization,
    replacement_data: bytes | None,
    data: bytes,
    maximum_bytes: int,
    validate_published: Callable[[bytes], bool] | None,
) -> StableFileRead:
    """Publish through one pinned directory without resolving its path."""

    path = Path(authorization.path)
    new_temporary: _AtomicTemporaryFile | None = None
    backup: _AtomicTemporaryFile | None = None
    replacement_lock_fd: int | None = None
    published = False
    committed = False
    preserve_backup = False
    try:
        _validate_publication_location_at_descriptor(
            directory,
            directory_fd,
            authorization,
        )
        if replacement_data is not None:
            replacement_lock_fd = _lock_replacement_target(directory_fd, authorization)
            _validate_publication_location_at_descriptor(
                directory,
                directory_fd,
                authorization,
            )
        forbidden = frozenset({path.name})
        if replacement_data is not None:
            backup = _create_atomic_temporary_file(
                directory_fd,
                replacement_data,
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
        _validate_publication_location_at_descriptor(
            directory,
            directory_fd,
            authorization,
        )
        try:
            if replacement_data is None:
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
                replacement_data is not None
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
        reopened = _read_private_file_at_descriptor(
            directory,
            directory_fd,
            path.name,
            maximum_bytes,
        )
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
        reopened_identity = reopened.authorization._target_complete_identity
        if type(reopened_identity) is not _CompleteIdentity:
            _fail()
        _validate_private_read_target_at(
            directory_fd,
            path.name,
            reopened_identity,
        )
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
            if replacement_data is None:
                _rollback_absent_publication(
                    directory_fd,
                    path.name,
                    new_temporary.identity,
                )
            elif backup is not None:
                restored = _rollback_replacement_at_descriptor(
                    directory,
                    directory_fd,
                    path.name,
                    new_temporary.identity,
                    backup,
                    replacement_data,
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


def _publish_private_file_at_descriptor(
    directory: _PrivateDirectoryAuthorization,
    directory_fd: int,
    name: str,
    data: bytes,
    *,
    maximum_bytes: int,
    validate_replacement: Callable[[bytes], bool] | None = None,
    validate_published: Callable[[bytes], bool] | None = None,
) -> StableFileRead:
    """Authorize and publish one private child within a pinned transaction."""

    if type(maximum_bytes) is not int or not 1 <= maximum_bytes < sys.maxsize:
        raise SecureFileBoundError()
    if type(data) is not bytes:
        raise SecureFileError()
    if len(data) > maximum_bytes:
        raise SecureFileBoundError()
    if validate_replacement is not None and not callable(validate_replacement):
        raise SecureFileError()
    if validate_published is not None and not callable(validate_published):
        raise SecureFileError()
    result: StableFileRead | None = None
    bound_exceeded = False
    unsupported = False
    try:
        _require_atomic_publication_platform()
        authorization = _inspect_private_file_location_at_descriptor(
            directory,
            directory_fd,
            name,
        )
        _probe_directory_fsync(directory_fd)
        replacement_data: bytes | None = None
        if authorization._target_complete_identity is not None:
            if validate_replacement is None:
                _fail()
            existing = _read_private_file_at_descriptor(
                directory,
                directory_fd,
                name,
                maximum_bytes,
            )
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
            _validate_publication_location_at_descriptor(
                directory,
                directory_fd,
                authorization,
            )
            replacement_data = existing.data
        result = _publish_private_file_at_descriptor_unchecked(
            directory,
            directory_fd,
            authorization,
            replacement_data,
            data,
            maximum_bytes,
            validate_published,
        )
    except SecureFileBoundError:
        bound_exceeded = True
    except _UnsupportedFileOperationError:
        unsupported = True
    except OSError as error:
        unsupported = error.errno in _UNSUPPORTED_OPERATION_ERRNOS
    except KeyboardInterrupt:
        raise
    except Exception:
        pass
    if bound_exceeded:
        raise SecureFileBoundError()
    if unsupported:
        raise SecureFileUnsupportedError()
    if result is None:
        raise SecureFileError()
    return result


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
            existing = _read_private_file(
                copied_path,
                maximum_bytes,
                policy=StableReadPolicy.PRIVATE_EXACT,
            )
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
            result = _read_private_file(
                _copy_path(path),
                maximum_bytes,
                policy=policy,
            )
    except SecureFileBoundError:
        bound_exceeded = True
    except Exception:
        pass
    if bound_exceeded:
        raise SecureFileBoundError()
    if result is None:
        raise SecureFileError()
    return result


def delete_authorized_private_file(
    authorization: StableFileAuthorization,
) -> None:
    """Delete only the exact file authenticated by a private stable read."""

    unsupported = False
    failed = False
    try:
        _require_private_delete_platform()
        if type(authorization) is not StableFileAuthorization:
            _fail()
        _delete_authorized_private_file(authorization)
    except (SecureFileUnsupportedError, _UnsupportedFileOperationError):
        unsupported = True
    except OSError as error:
        unsupported = error.errno in _UNSUPPORTED_OPERATION_ERRNOS
        failed = not unsupported
    except KeyboardInterrupt:
        raise
    except Exception:
        failed = True
    if unsupported:
        raise SecureFileUnsupportedError()
    if failed:
        raise SecureFileError()


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


def ensure_private_directory(path: str | os.PathLike[str]) -> None:
    """Create or authorize one exact owner-only directory without following links."""

    unsupported = False
    failed = False
    try:
        copied_path = _copy_path(path)
        if not _inspect_private_directory_boundary(copied_path):
            try:
                authorization = _authorize_private_directory(copied_path, create=True)
            except Exception:
                # Another process may have won the absent-leaf creation race.
                # Accept only a fresh full private-boundary authorization.
                if not _inspect_private_directory_boundary(copied_path):
                    raise
            else:
                authorization.revalidate()
    except (SecureFileUnsupportedError, _UnsupportedFileOperationError):
        unsupported = True
    except OSError as error:
        unsupported = error.errno in _UNSUPPORTED_OPERATION_ERRNOS
        failed = not unsupported
    except KeyboardInterrupt:
        raise
    except Exception:
        failed = True
    if unsupported:
        raise SecureFileUnsupportedError()
    if failed:
        raise SecureFileError()


def inspect_private_directory(path: str | os.PathLike[str]) -> bool:
    """Inspect an exact private leaf or a safely absent suffix without mutation."""

    unsupported = False
    failed = False
    result: bool | None = None
    try:
        _require_secure_platform()
        result = _inspect_private_directory_boundary(_copy_path(path))
    except (SecureFileUnsupportedError, _UnsupportedFileOperationError):
        unsupported = True
    except OSError as error:
        unsupported = error.errno in _UNSUPPORTED_OPERATION_ERRNOS
        failed = not unsupported
    except KeyboardInterrupt:
        raise
    except Exception:
        failed = True
    if unsupported:
        raise SecureFileUnsupportedError()
    if failed or result is None:
        raise SecureFileError()
    return result


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


def claim_private_sqlite_location(
    location: StableFileAuthorization,
    *,
    sidecar_locations: tuple[StableFileAuthorization, ...] | None = None,
) -> StableFileAuthorization:
    """Claim an exact inspected database and sidecar layout without path fallback."""

    try:
        return _claim_private_sqlite_location(
            location,
            sidecar_locations=sidecar_locations,
        )
    except SecureFileError:
        raise
    except Exception:
        raise SecureFileError() from None


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
    "claim_private_sqlite_location",
    "delete_authorized_private_file",
    "ensure_private_directory",
    "inspect_private_directory",
    "inspect_private_file_location",
    "read_stable_file",
]
