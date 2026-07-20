"""Fail-closed owner-private Windows path authorization.

The public authorization boundary is deliberately backend-injectable so its race and
security rules can be exercised on every development host.  The native backend opens
every named component itself (rather than following a final reparse point), inspects its
stable file ID, owner, DACL, link count, and reparse metadata, and creates new objects
with a protected owner-only ACL.

Win32 does not expose a Python-standard handle-relative filesystem API.  Every
authorization therefore captures and revalidates the complete existing ancestor chain
as well as the exact final object at each observable boundary.
"""

from __future__ import annotations

import ctypes
import os
import re
import secrets
from collections.abc import Callable
from contextlib import suppress
from ctypes import wintypes
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PureWindowsPath
from time import sleep
from types import TracebackType
from typing import Any, Final, Protocol, TypeVar, cast, runtime_checkable

_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_SID_PATTERN = re.compile(r"S-[0-9]+(?:-[0-9]+)+", re.ASCII)
_RESERVED_BASENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)

_AUTHORIZATION_TOKEN = object()
_MISSING = object()
_T = TypeVar("_T")


class WindowsSecurityError(ValueError):
    """A value-free failure to establish or revalidate a Windows path boundary."""

    def __init__(self) -> None:
        super().__init__("Windows private path authorization failed")


class _NativeWindowsError(Exception):
    pass


class WindowsPathKind(StrEnum):
    """The two filesystem object types accepted by the private boundary."""

    FILE = "file"
    DIRECTORY = "directory"


class _WindowsDaclPolicy(StrEnum):
    PRIVATE = "private"
    MANAGED = "managed"


@dataclass(frozen=True, slots=True, repr=False)
class WindowsFileIdentity:
    """The volume-scoped, 128-bit identity returned by ``FileIdInfo``."""

    volume_serial_number: int
    file_id: bytes

    def __post_init__(self) -> None:
        if (
            type(self.volume_serial_number) is not int
            or not 0 <= self.volume_serial_number <= _UINT64_MAX
            or type(self.file_id) is not bytes
            or len(self.file_id) != 16
        ):
            raise WindowsSecurityError()

    def __repr__(self) -> str:
        return "WindowsFileIdentity(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class WindowsPathSecurity:
    """Security and aliasing metadata captured from one no-follow handle."""

    identity: WindowsFileIdentity
    kind: WindowsPathKind
    owner_sid: str
    owner_private_dacl: bool
    owner_write_protected_dacl: bool
    owner_traversal_protected_dacl: bool
    hardlink_count: int
    reparse_tag: int | None

    def __post_init__(self) -> None:
        valid_reparse_tag = self.reparse_tag is None or (
            type(self.reparse_tag) is int and 0 <= self.reparse_tag <= _UINT32_MAX
        )
        if (
            type(self.identity) is not WindowsFileIdentity
            or type(self.kind) is not WindowsPathKind
            or not _is_valid_sid_text(self.owner_sid)
            or type(self.owner_private_dacl) is not bool
            or type(self.owner_write_protected_dacl) is not bool
            or type(self.owner_traversal_protected_dacl) is not bool
            or (self.owner_private_dacl and not self.owner_write_protected_dacl)
            or (self.owner_write_protected_dacl and not self.owner_traversal_protected_dacl)
            or type(self.hardlink_count) is not int
            or not 1 <= self.hardlink_count <= _UINT32_MAX
            or not valid_reparse_tag
        ):
            raise WindowsSecurityError()

    def __repr__(self) -> str:
        return "WindowsPathSecurity(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class _WindowsFileSnapshot:
    security: WindowsPathSecurity
    size: int
    last_write_time: int
    change_time: int

    def __post_init__(self) -> None:
        if (
            type(self.security) is not WindowsPathSecurity
            or type(self.size) is not int
            or self.size < 0
            or type(self.last_write_time) is not int
            or self.last_write_time < 0
            or type(self.change_time) is not int
            or self.change_time < 0
        ):
            raise WindowsSecurityError()

    def __repr__(self) -> str:
        return "_WindowsFileSnapshot(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class WindowsStableFileRead:
    """Exact bounded bytes plus the Windows identity captured while reading them."""

    data: bytes
    authorization: WindowsPathAuthorization

    def __post_init__(self) -> None:
        if type(self.data) is not bytes or type(self.authorization) is not WindowsPathAuthorization:
            raise WindowsSecurityError()

    def __repr__(self) -> str:
        return "WindowsStableFileRead(<redacted>)"


@runtime_checkable
class WindowsSecurityOperations(Protocol):
    """Injectable operations needed to authorize one Windows filesystem object."""

    def current_user_sid(self) -> str:
        """Return the current process token's user SID."""

    def inspect_path(self, path: PureWindowsPath) -> WindowsPathSecurity | None:
        """Inspect the final named object without following a reparse point."""

    def inspect_ancestor_directories(
        self,
        path: PureWindowsPath,
    ) -> tuple[tuple[PureWindowsPath, WindowsPathSecurity], ...]:
        """Inspect every ancestor, from the path anchor through its direct parent."""

    def create_private_path(self, path: PureWindowsPath, kind: WindowsPathKind) -> None:
        """Exclusively create a path with a protected owner-only DACL."""


@dataclass(frozen=True, slots=True, repr=False)
class WindowsPathAuthorization:
    """An exact Windows path identity and its revalidation capability."""

    path: PureWindowsPath
    kind: WindowsPathKind
    security: WindowsPathSecurity
    _owner_sid: str = field(repr=False)
    _operations: WindowsSecurityOperations = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)
    _ancestor_chain: tuple[tuple[PureWindowsPath, WindowsPathSecurity], ...] = field(
        default=(),
        repr=False,
    )
    _dacl_policy: _WindowsDaclPolicy = field(
        default=_WindowsDaclPolicy.PRIVATE,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            self._token is not _AUTHORIZATION_TOKEN
            or not _is_valid_windows_path(self.path)
            or type(self.kind) is not WindowsPathKind
            or type(self.security) is not WindowsPathSecurity
            or not _is_valid_sid_text(self._owner_sid)
            or not isinstance(self._operations, WindowsSecurityOperations)
            or type(self._dacl_policy) is not _WindowsDaclPolicy
            or not _is_valid_ancestor_chain(
                self.path,
                self._ancestor_chain,
                owner_sid=self._owner_sid,
            )
        ):
            raise WindowsSecurityError()

    @property
    def component_security(self) -> tuple[WindowsPathSecurity, ...]:
        """Return the complete redaction-safe security snapshot in traversal order."""

        return (*tuple(security for _path, security in self._ancestor_chain), self.security)

    def revalidate(self) -> None:
        """Fail if the name, identity, owner, DACL, links, or kind changed."""

        _content_free_call(self._checked_revalidate)

    def _checked_revalidate(self) -> None:
        current_owner_sid = self._operations.current_user_sid()
        if current_owner_sid != self._owner_sid:
            raise _NativeWindowsError()
        before = _capture_ancestor_chain(
            self.path,
            operations=self._operations,
            owner_sid=self._owner_sid,
        )
        if before != self._ancestor_chain:
            raise _NativeWindowsError()
        inspected = self._operations.inspect_path(self.path)
        checked = _validate_authorized_security(
            inspected,
            kind=self.kind,
            owner_sid=self._owner_sid,
            dacl_policy=self._dacl_policy,
        )
        after = _capture_ancestor_chain(
            self.path,
            operations=self._operations,
            owner_sid=self._owner_sid,
        )
        if (
            before != after
            or after != self._ancestor_chain
            or checked.identity != self.security.identity
        ):
            raise _NativeWindowsError()

    def __repr__(self) -> str:
        return "WindowsPathAuthorization(<redacted>)"


def authorize_windows_private_path(
    path: PureWindowsPath,
    *,
    kind: WindowsPathKind,
    operations: WindowsSecurityOperations,
    create: bool = False,
) -> WindowsPathAuthorization:
    """Authorize an existing exact path, optionally creating it owner-private first."""

    return _content_free_call(
        lambda: _authorize_windows_path(
            path,
            kind=kind,
            operations=operations,
            create=create,
            dacl_policy=_WindowsDaclPolicy.PRIVATE,
        )
    )


def authorize_windows_managed_path(
    path: PureWindowsPath,
    *,
    kind: WindowsPathKind,
    operations: WindowsSecurityOperations,
) -> WindowsPathAuthorization:
    """Authorize a current-user-owned path whose DACL denies untrusted writes."""

    return _content_free_call(
        lambda: _authorize_windows_path(
            path,
            kind=kind,
            operations=operations,
            create=False,
            dacl_policy=_WindowsDaclPolicy.MANAGED,
        )
    )


def ensure_windows_private_directory(
    path: PureWindowsPath,
    *,
    operations: WindowsSecurityOperations,
) -> WindowsPathAuthorization:
    """Create a missing suffix with private DACLs below one safe existing ancestor."""

    return _content_free_call(
        lambda: _ensure_windows_private_directory(path, operations=operations)
    )


def _ensure_windows_private_directory(
    path: PureWindowsPath,
    *,
    operations: WindowsSecurityOperations,
) -> WindowsPathAuthorization:
    if not _is_valid_windows_path(path) or not isinstance(operations, WindowsSecurityOperations):
        raise _NativeWindowsError()
    if operations.inspect_path(path) is not None:
        return _authorize_windows_path(
            path,
            kind=WindowsPathKind.DIRECTORY,
            operations=operations,
            create=False,
            dacl_policy=_WindowsDaclPolicy.PRIVATE,
        )
    missing: list[PureWindowsPath] = []
    current = path
    while operations.inspect_path(current) is None:
        missing.append(current)
        parent = current.parent
        if parent == current:
            raise _NativeWindowsError()
        current = parent
    parent_authorization = _authorize_windows_path(
        current,
        kind=WindowsPathKind.DIRECTORY,
        operations=operations,
        create=False,
        dacl_policy=_WindowsDaclPolicy.MANAGED,
    )
    result: WindowsPathAuthorization | None = None
    for item in reversed(missing):
        parent_authorization._checked_revalidate()
        try:
            operations.create_private_path(item, WindowsPathKind.DIRECTORY)
        except Exception:
            if operations.inspect_path(item) is None:
                raise
        parent_authorization._checked_revalidate()
        result = _authorize_windows_path(
            item,
            kind=WindowsPathKind.DIRECTORY,
            operations=operations,
            create=False,
            dacl_policy=_WindowsDaclPolicy.PRIVATE,
        )
        parent_authorization = result
    if result is None:
        raise _NativeWindowsError()
    result._checked_revalidate()
    return result


def _authorize_windows_path(
    path: PureWindowsPath,
    *,
    kind: WindowsPathKind,
    operations: WindowsSecurityOperations,
    create: bool,
    dacl_policy: _WindowsDaclPolicy,
) -> WindowsPathAuthorization:
    if (
        not _is_valid_windows_path(path)
        or type(kind) is not WindowsPathKind
        or not isinstance(operations, WindowsSecurityOperations)
        or type(create) is not bool
        or type(dacl_policy) is not _WindowsDaclPolicy
    ):
        raise _NativeWindowsError()
    owner_sid = operations.current_user_sid()
    if not _is_valid_sid_text(owner_sid):
        raise _NativeWindowsError()
    ancestors = _capture_ancestor_chain(
        path,
        operations=operations,
        owner_sid=owner_sid,
    )
    inspected = operations.inspect_path(path)
    if inspected is None:
        if not create:
            raise _NativeWindowsError()
        if (
            _capture_ancestor_chain(
                path,
                operations=operations,
                owner_sid=owner_sid,
            )
            != ancestors
        ):
            raise _NativeWindowsError()
        operations.create_private_path(path, kind)
        inspected = operations.inspect_path(path)
    security = _validate_authorized_security(
        inspected,
        kind=kind,
        owner_sid=owner_sid,
        dacl_policy=dacl_policy,
    )
    after = _capture_ancestor_chain(
        path,
        operations=operations,
        owner_sid=owner_sid,
    )
    if after != ancestors:
        raise _NativeWindowsError()
    return WindowsPathAuthorization(
        path=path,
        kind=kind,
        security=security,
        _owner_sid=owner_sid,
        _operations=operations,
        _token=_AUTHORIZATION_TOKEN,
        _ancestor_chain=ancestors,
        _dacl_policy=dacl_policy,
    )


def _capture_ancestor_chain(
    path: PureWindowsPath,
    *,
    operations: WindowsSecurityOperations,
    owner_sid: str,
) -> tuple[tuple[PureWindowsPath, WindowsPathSecurity], ...]:
    chain = operations.inspect_ancestor_directories(path)
    if not _is_valid_ancestor_chain(path, chain, owner_sid=owner_sid):
        raise _NativeWindowsError()
    return chain


def _is_valid_ancestor_chain(
    path: PureWindowsPath,
    chain: object,
    *,
    owner_sid: str,
) -> bool:
    if type(chain) is not tuple or not _is_valid_sid_text(owner_sid):
        return False
    expected_paths = _windows_ancestor_paths(path)
    if len(chain) != len(expected_paths):
        return False
    for index, item in enumerate(chain):
        if type(item) is not tuple or len(item) != 2:
            return False
        ancestor, security = item
        if ancestor != expected_paths[index] or not _is_safe_ancestor_security(
            security,
            owner_sid=owner_sid,
            is_anchor=index == 0,
        ):
            return False
    return True


def _is_safe_ancestor_security(
    security: object,
    *,
    owner_sid: str,
    is_anchor: bool,
) -> bool:
    return bool(
        type(security) is WindowsPathSecurity
        and security.kind is WindowsPathKind.DIRECTORY
        and security.owner_sid in {owner_sid, *_TRUSTED_PRIVILEGED_SIDS}
        and (
            security.owner_traversal_protected_dacl
            if is_anchor
            else security.owner_write_protected_dacl
        )
        and security.hardlink_count == 1
        and security.reparse_tag is None
    )


def _validate_authorized_security(
    security: WindowsPathSecurity | None,
    *,
    kind: WindowsPathKind,
    owner_sid: str,
    dacl_policy: _WindowsDaclPolicy,
) -> WindowsPathSecurity:
    if type(dacl_policy) is not _WindowsDaclPolicy:
        raise _NativeWindowsError()
    if dacl_policy is _WindowsDaclPolicy.PRIVATE:
        return _validate_private_security(security, kind=kind, owner_sid=owner_sid)
    if (
        type(security) is not WindowsPathSecurity
        or security.kind is not kind
        or security.owner_sid != owner_sid
        or not security.owner_write_protected_dacl
        or security.hardlink_count != 1
        or security.reparse_tag is not None
    ):
        raise _NativeWindowsError()
    return security


def _validate_private_security(
    security: WindowsPathSecurity | None,
    *,
    kind: WindowsPathKind,
    owner_sid: str,
) -> WindowsPathSecurity:
    if (
        type(security) is not WindowsPathSecurity
        or security.kind is not kind
        or security.owner_sid != owner_sid
        or not security.owner_private_dacl
        or security.hardlink_count != 1
        or security.reparse_tag is not None
    ):
        raise _NativeWindowsError()
    return security


def _content_free_call(operation: Callable[[], _T]) -> _T:
    result: object = _MISSING
    failed = False
    try:
        result = operation()
    except Exception:
        failed = True
    if failed or result is _MISSING:
        raise WindowsSecurityError()
    return cast(_T, result)


def _is_valid_sid_text(value: object) -> bool:
    if (
        type(value) is not str
        or not 5 <= len(value) <= 184
        or _SID_PATTERN.fullmatch(value) is None
    ):
        return False
    components = value.split("-")
    try:
        revision = int(components[1])
        authority = int(components[2])
        subauthorities = tuple(int(component) for component in components[3:])
    except (IndexError, ValueError):
        return False
    return (
        0 <= revision <= 0xFF
        and 0 <= authority < (1 << 48)
        and len(subauthorities) <= 15
        and all(0 <= component <= _UINT32_MAX for component in subauthorities)
    )


def _is_valid_windows_path(value: object) -> bool:
    return (
        isinstance(value, PureWindowsPath)
        and _is_valid_windows_inspection_path(value)
        and bool(value.name)
    )


def _is_valid_windows_inspection_path(value: object) -> bool:
    if not isinstance(value, PureWindowsPath) or not value.is_absolute():
        return False
    rendered = str(value)
    if "\x00" in rendered or value.drive.casefold().startswith(("\\\\.\\", "\\\\?\\")):
        return False
    relative_parts = value.parts[1:]
    for part in relative_parts:
        if (
            not part
            or part in {".", ".."}
            or part.endswith((" ", "."))
            or any(character in '<>:"|?*' or ord(character) < 32 for character in part)
            or part.split(".", 1)[0].upper() in _RESERVED_BASENAMES
        ):
            return False
    return True


def _windows_ancestor_paths(path: PureWindowsPath) -> tuple[PureWindowsPath, ...]:
    if not _is_valid_windows_path(path):
        return ()
    return tuple(reversed(path.parents))


# Win32 constants and structures are declared explicitly instead of relying on pywin32 so
# the security boundary remains part of the dependency-free core package.
_INVALID_HANDLE_VALUE: Final[int] = cast(int, ctypes.c_void_p(-1).value)
_ERROR_FILE_NOT_FOUND = 2
_ERROR_PATH_NOT_FOUND = 3
_ERROR_SHARING_VIOLATION = 32
_ERROR_INVALID_FUNCTION = 1
_ERROR_NOT_SUPPORTED = 50
_ERROR_INVALID_PARAMETER = 87
_ERROR_INSUFFICIENT_BUFFER = 122
_TOKEN_QUERY = 0x0008
_TOKEN_USER = 1
_FILE_READ_ATTRIBUTES = 0x0080
_FILE_WRITE_ATTRIBUTES = 0x0100
_READ_CONTROL = 0x00020000
_DELETE = 0x00010000
_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_FILE_SHARE_DELETE = 0x00000004
_CREATE_NEW = 1
_OPEN_EXISTING = 3
_MOVEFILE_WRITE_THROUGH = 0x00000008
# ReplaceFileW exposes historical write-through constants, but Microsoft documents
# them as unsupported.  The replacement payload is flushed before this atomic swap.
_REPLACEFILE_FLAGS = 0
_FILE_ATTRIBUTE_DIRECTORY = 0x00000010
_FILE_ATTRIBUTE_NORMAL = 0x00000080
_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_FILE_ID_INFO_CLASS = 18
_FILE_BASIC_INFO_CLASS = 0
_FILE_STANDARD_INFO_CLASS = 1
_FILE_DISPOSITION_INFO_CLASS = 4
_FSCTL_GET_REPARSE_POINT = 0x000900A8
_MAXIMUM_REPARSE_DATA_BUFFER_SIZE = 16 * 1024
_SE_FILE_OBJECT = 1
_OWNER_SECURITY_INFORMATION = 0x00000001
_DACL_SECURITY_INFORMATION = 0x00000004
_SE_DACL_PRESENT = 0x0004
_SE_DACL_PROTECTED = 0x1000
_ACL_SIZE_INFORMATION_CLASS = 2
_ACCESS_ALLOWED_ACE_TYPE = 0
_ACCESS_DENIED_ACE_TYPE = 1
_INHERIT_ONLY_ACE = 0x08
_INHERITED_ACE = 0x10
_FILE_ALL_ACCESS = 0x001F01FF
_UNTRUSTED_WRITE_ACCESS = (
    0x00000002  # FILE_WRITE_DATA / FILE_ADD_FILE
    | 0x00000004  # FILE_APPEND_DATA / FILE_ADD_SUBDIRECTORY
    | 0x00000010  # FILE_WRITE_EA
    | 0x00000040  # FILE_DELETE_CHILD
    | 0x00000100  # FILE_WRITE_ATTRIBUTES
    | 0x00010000  # DELETE
    | 0x00040000  # WRITE_DAC
    | 0x00080000  # WRITE_OWNER
    | 0x01000000  # ACCESS_SYSTEM_SECURITY
    | 0x02000000  # MAXIMUM_ALLOWED
    | 0x10000000  # GENERIC_ALL
    | 0x40000000  # GENERIC_WRITE
)
_UNTRUSTED_TRAVERSAL_ACCESS = (
    0x00000040  # FILE_DELETE_CHILD
    | 0x00010000  # DELETE
    | 0x00040000  # WRITE_DAC
    | 0x00080000  # WRITE_OWNER
    | 0x01000000  # ACCESS_SYSTEM_SECURITY
    | 0x02000000  # MAXIMUM_ALLOWED
    | 0x10000000  # GENERIC_ALL
    | 0x40000000  # GENERIC_WRITE
)
_TRUSTED_PRIVILEGED_SIDS = frozenset(
    {
        "S-1-5-18",  # LocalSystem, analogous to POSIX root for this boundary.
        "S-1-5-32-544",  # Builtin Administrators.
        # NT SERVICE\TrustedInstaller, the normal owner for protected Windows roots.
        "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464",
    }
)
_SDDL_REVISION_1 = 1
_LOCKFILE_FAIL_IMMEDIATELY = 0x00000001
_LOCKFILE_EXCLUSIVE_LOCK = 0x00000002


class _FileId128(ctypes.Structure):
    _fields_ = [("Identifier", wintypes.BYTE * 16)]


class _FileIdInfo(ctypes.Structure):
    _fields_ = [
        ("VolumeSerialNumber", ctypes.c_ulonglong),
        ("FileId", _FileId128),
    ]


class _ByHandleFileInformation(ctypes.Structure):
    _fields_ = [
        ("dwFileAttributes", wintypes.DWORD),
        ("ftCreationTime", wintypes.FILETIME),
        ("ftLastAccessTime", wintypes.FILETIME),
        ("ftLastWriteTime", wintypes.FILETIME),
        ("dwVolumeSerialNumber", wintypes.DWORD),
        ("nFileSizeHigh", wintypes.DWORD),
        ("nFileSizeLow", wintypes.DWORD),
        ("nNumberOfLinks", wintypes.DWORD),
        ("nFileIndexHigh", wintypes.DWORD),
        ("nFileIndexLow", wintypes.DWORD),
    ]


class _FileBasicInfo(ctypes.Structure):
    _fields_ = [
        ("CreationTime", ctypes.c_longlong),
        ("LastAccessTime", ctypes.c_longlong),
        ("LastWriteTime", ctypes.c_longlong),
        ("ChangeTime", ctypes.c_longlong),
        ("FileAttributes", wintypes.DWORD),
    ]


class _FileStandardInfo(ctypes.Structure):
    _fields_ = [
        ("AllocationSize", ctypes.c_longlong),
        ("EndOfFile", ctypes.c_longlong),
        ("NumberOfLinks", wintypes.DWORD),
        ("DeletePending", wintypes.BOOL),
        ("Directory", wintypes.BOOL),
    ]


class _FileDispositionInfo(ctypes.Structure):
    _fields_ = [("DeleteFile", wintypes.BOOL)]


class _Overlapped(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", wintypes.DWORD),
        ("OffsetHigh", wintypes.DWORD),
        ("hEvent", wintypes.HANDLE),
    ]


class _SidAndAttributes(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]


class _SecurityAttributes(ctypes.Structure):
    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", ctypes.c_void_p),
        ("bInheritHandle", wintypes.BOOL),
    ]


class _AclSizeInformation(ctypes.Structure):
    _fields_ = [
        ("AceCount", wintypes.DWORD),
        ("AclBytesInUse", wintypes.DWORD),
        ("AclBytesFree", wintypes.DWORD),
    ]


class _AceHeader(ctypes.Structure):
    _fields_ = [
        ("AceType", wintypes.BYTE),
        ("AceFlags", wintypes.BYTE),
        ("AceSize", wintypes.WORD),
    ]


class _WindowsBindings:  # pragma: no cover - exercised by native Windows R01
    kernel32: Any
    advapi32: Any
    get_last_error: Callable[[], int]

    def __init__(self) -> None:
        loader = getattr(ctypes, "WinDLL", None)
        get_last_error = getattr(ctypes, "get_last_error", None)
        if os.name != "nt" or loader is None or get_last_error is None:
            raise _NativeWindowsError()
        self.get_last_error = cast(Callable[[], int], get_last_error)
        self.kernel32 = loader("kernel32", use_last_error=True)
        self.advapi32 = loader("advapi32", use_last_error=True)
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        self.kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(_SecurityAttributes),
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        self.kernel32.CreateFileW.restype = wintypes.HANDLE
        self.kernel32.CreateDirectoryW.argtypes = [
            wintypes.LPCWSTR,
            ctypes.POINTER(_SecurityAttributes),
        ]
        self.kernel32.CreateDirectoryW.restype = wintypes.BOOL
        self.kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel32.CloseHandle.restype = wintypes.BOOL
        self.kernel32.GetCurrentProcess.argtypes = []
        self.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        self.kernel32.GetFileInformationByHandle.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ByHandleFileInformation),
        ]
        self.kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
        self.kernel32.GetFileInformationByHandleEx.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self.kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL
        self.kernel32.SetFileInformationByHandle.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self.kernel32.SetFileInformationByHandle.restype = wintypes.BOOL
        self.kernel32.ReadFile.argtypes = [
            wintypes.HANDLE,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.c_void_p,
        ]
        self.kernel32.ReadFile.restype = wintypes.BOOL
        self.kernel32.WriteFile.argtypes = [
            wintypes.HANDLE,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.c_void_p,
        ]
        self.kernel32.WriteFile.restype = wintypes.BOOL
        self.kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
        self.kernel32.FlushFileBuffers.restype = wintypes.BOOL
        self.kernel32.LockFileEx.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(_Overlapped),
        ]
        self.kernel32.LockFileEx.restype = wintypes.BOOL
        self.kernel32.UnlockFileEx.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(_Overlapped),
        ]
        self.kernel32.UnlockFileEx.restype = wintypes.BOOL
        self.kernel32.MoveFileExW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
        ]
        self.kernel32.MoveFileExW.restype = wintypes.BOOL
        self.kernel32.ReplaceFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.kernel32.ReplaceFileW.restype = wintypes.BOOL
        self.kernel32.DeviceIoControl.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.c_void_p,
        ]
        self.kernel32.DeviceIoControl.restype = wintypes.BOOL
        self.kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        self.kernel32.LocalFree.restype = ctypes.c_void_p

        self.advapi32.OpenProcessToken.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        ]
        self.advapi32.OpenProcessToken.restype = wintypes.BOOL
        self.advapi32.GetTokenInformation.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.advapi32.GetTokenInformation.restype = wintypes.BOOL
        self.advapi32.GetSecurityInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.advapi32.GetSecurityInfo.restype = wintypes.DWORD
        self.advapi32.ConvertSidToStringSidW.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.LPWSTR),
        ]
        self.advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
        self.advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
        self.advapi32.GetSecurityDescriptorControl.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.WORD),
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.advapi32.GetSecurityDescriptorControl.restype = wintypes.BOOL
        self.advapi32.GetAclInformation.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_int,
        ]
        self.advapi32.GetAclInformation.restype = wintypes.BOOL
        self.advapi32.GetAce.argtypes = [
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.advapi32.GetAce.restype = wintypes.BOOL
        self.advapi32.EqualSid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.advapi32.EqualSid.restype = wintypes.BOOL


class NativeWindowsSecurityOperations:  # pragma: no cover - exercised by native Windows R01
    """ctypes-backed implementation of the private Windows path operations."""

    __slots__ = ("_bindings",)

    def __init__(self) -> None:
        self._bindings = _content_free_call(_WindowsBindings)

    def current_user_sid(self) -> str:
        """Return a copied string form of the current process token's user SID."""

        return _content_free_call(self._current_user_sid)

    def inspect_path(self, path: PureWindowsPath) -> WindowsPathSecurity | None:
        """Inspect a file or directory through a no-follow Win32 handle."""

        return _content_free_call(lambda: self._inspect_path(path))

    def inspect_ancestor_directories(
        self,
        path: PureWindowsPath,
    ) -> tuple[tuple[PureWindowsPath, WindowsPathSecurity], ...]:
        """Inspect every existing traversal component through no-follow handles."""

        return _content_free_call(lambda: self._inspect_ancestor_directories(path))

    def create_private_path(self, path: PureWindowsPath, kind: WindowsPathKind) -> None:
        """Exclusively create a file or directory with an owner-only protected DACL."""

        _content_free_call(lambda: self._create_private_path(path, kind))

    def read_private_file(
        self,
        path: PureWindowsPath,
        *,
        maximum_bytes: int,
    ) -> WindowsStableFileRead:
        """Read exact bounded bytes while denying writers and path replacement."""

        return _content_free_call(
            lambda: self._read_file(
                path,
                maximum_bytes=maximum_bytes,
                dacl_policy=_WindowsDaclPolicy.PRIVATE,
            )
        )

    def read_managed_file(
        self,
        path: PureWindowsPath,
        *,
        maximum_bytes: int,
    ) -> WindowsStableFileRead:
        """Read bytes whose DACL prevents writes by untrusted principals."""

        return _content_free_call(
            lambda: self._read_file(
                path,
                maximum_bytes=maximum_bytes,
                dacl_policy=_WindowsDaclPolicy.MANAGED,
            )
        )

    def publish_private_file(
        self,
        path: PureWindowsPath,
        data: bytes,
        *,
        maximum_bytes: int,
        validate_replacement: Callable[[bytes], bool] | None = None,
        validate_published: Callable[[bytes], bool] | None = None,
    ) -> WindowsStableFileRead:
        """Atomically publish owner-private bytes in one authorized directory."""

        return _content_free_call(
            lambda: self._publish_private_file(
                path,
                data,
                maximum_bytes=maximum_bytes,
                validate_replacement=validate_replacement,
                validate_published=validate_published,
                parent_dacl_policy=_WindowsDaclPolicy.PRIVATE,
                target_dacl_policy=_WindowsDaclPolicy.PRIVATE,
            )
        )

    def publish_private_file_in_managed_directory(
        self,
        path: PureWindowsPath,
        data: bytes,
        *,
        maximum_bytes: int,
        validate_replacement: Callable[[bytes], bool] | None = None,
        validate_published: Callable[[bytes], bool] | None = None,
    ) -> WindowsStableFileRead:
        """Publish a private file without rewriting one safe provider-owned parent."""

        return _content_free_call(
            lambda: self._publish_private_file(
                path,
                data,
                maximum_bytes=maximum_bytes,
                validate_replacement=validate_replacement,
                validate_published=validate_published,
                parent_dacl_policy=_WindowsDaclPolicy.MANAGED,
                target_dacl_policy=_WindowsDaclPolicy.PRIVATE,
            )
        )

    def publish_managed_file(
        self,
        path: PureWindowsPath,
        data: bytes,
        *,
        maximum_bytes: int,
        validate_replacement: Callable[[bytes], bool] | None = None,
        validate_published: Callable[[bytes], bool] | None = None,
    ) -> WindowsStableFileRead:
        """Publish into a safe shared directory and permit a safe existing target."""

        return _content_free_call(
            lambda: self._publish_private_file(
                path,
                data,
                maximum_bytes=maximum_bytes,
                validate_replacement=validate_replacement,
                validate_published=validate_published,
                parent_dacl_policy=_WindowsDaclPolicy.MANAGED,
                target_dacl_policy=_WindowsDaclPolicy.MANAGED,
            )
        )

    def delete_authorized_file(
        self,
        authorization: WindowsPathAuthorization,
    ) -> None:
        """Delete only the exact still-authorized file object."""

        _content_free_call(lambda: self._delete_authorized_file(authorization))

    def private_file_lock(
        self,
        path: PureWindowsPath,
        *,
        blocking: bool = True,
    ) -> _WindowsPrivateFileLock:
        """Return a cross-process lock over one exact private file."""

        return _content_free_call(lambda: self._private_file_lock(path, blocking=blocking))

    def _current_user_sid(self) -> str:
        token = wintypes.HANDLE()
        current_process = self._bindings.kernel32.GetCurrentProcess()
        if not self._bindings.advapi32.OpenProcessToken(
            current_process,
            _TOKEN_QUERY,
            ctypes.byref(token),
        ):
            raise _NativeWindowsError()
        if token.value is None:
            raise _NativeWindowsError()
        try:
            required = wintypes.DWORD()
            first_result = self._bindings.advapi32.GetTokenInformation(
                token,
                _TOKEN_USER,
                None,
                0,
                ctypes.byref(required),
            )
            if first_result or self._bindings.get_last_error() != _ERROR_INSUFFICIENT_BUFFER:
                raise _NativeWindowsError()
            if required.value < ctypes.sizeof(_SidAndAttributes):
                raise _NativeWindowsError()
            buffer = ctypes.create_string_buffer(required.value)
            if not self._bindings.advapi32.GetTokenInformation(
                token,
                _TOKEN_USER,
                buffer,
                required,
                ctypes.byref(required),
            ):
                raise _NativeWindowsError()
            token_user = ctypes.cast(buffer, ctypes.POINTER(_SidAndAttributes)).contents
            if token_user.Sid is None:
                raise _NativeWindowsError()
            return self._sid_to_text(token_user.Sid)
        finally:
            self._close_handle(token.value)

    def _inspect_path(self, path: PureWindowsPath) -> WindowsPathSecurity | None:
        if not _is_valid_windows_inspection_path(path):
            raise _NativeWindowsError()
        handle = self._open_path_no_follow(path)
        if handle is None:
            return None
        try:
            return self._snapshot_from_handle(handle).security
        finally:
            self._close_handle(handle)

    def _inspect_ancestor_directories(
        self,
        path: PureWindowsPath,
    ) -> tuple[tuple[PureWindowsPath, WindowsPathSecurity], ...]:
        if not _is_valid_windows_path(path):
            raise _NativeWindowsError()
        result: list[tuple[PureWindowsPath, WindowsPathSecurity]] = []
        for ancestor in _windows_ancestor_paths(path):
            security = self._inspect_path(ancestor)
            if security is None:
                raise _NativeWindowsError()
            result.append((ancestor, security))
        return tuple(result)

    def _snapshot_from_handle(self, handle: int) -> _WindowsFileSnapshot:
        legacy = _ByHandleFileInformation()
        if not self._bindings.kernel32.GetFileInformationByHandle(
            handle,
            ctypes.byref(legacy),
        ):
            raise _NativeWindowsError()
        basic = _FileBasicInfo()
        standard = _FileStandardInfo()
        if not self._bindings.kernel32.GetFileInformationByHandleEx(
            handle,
            _FILE_BASIC_INFO_CLASS,
            ctypes.byref(basic),
            ctypes.sizeof(basic),
        ) or not self._bindings.kernel32.GetFileInformationByHandleEx(
            handle,
            _FILE_STANDARD_INFO_CLASS,
            ctypes.byref(standard),
            ctypes.sizeof(standard),
        ):
            raise _NativeWindowsError()
        attributes = int(legacy.dwFileAttributes)
        kind = (
            WindowsPathKind.DIRECTORY
            if attributes & _FILE_ATTRIBUTE_DIRECTORY
            else WindowsPathKind.FILE
        )
        reparse_tag = (
            self._read_reparse_tag(handle) if attributes & _FILE_ATTRIBUTE_REPARSE_POINT else None
        )
        (
            owner_sid,
            owner_private_dacl,
            owner_write_protected_dacl,
            owner_traversal_protected_dacl,
        ) = self._read_owner_and_dacl(handle)
        return _WindowsFileSnapshot(
            security=WindowsPathSecurity(
                identity=self._read_file_identity(handle, legacy),
                kind=kind,
                owner_sid=owner_sid,
                owner_private_dacl=owner_private_dacl,
                owner_write_protected_dacl=owner_write_protected_dacl,
                owner_traversal_protected_dacl=owner_traversal_protected_dacl,
                hardlink_count=int(standard.NumberOfLinks),
                reparse_tag=reparse_tag,
            ),
            size=int(standard.EndOfFile),
            last_write_time=int(basic.LastWriteTime),
            change_time=int(basic.ChangeTime),
        )

    def _create_private_path(self, path: PureWindowsPath, kind: WindowsPathKind) -> None:
        if not _is_valid_windows_path(path) or type(kind) is not WindowsPathKind:
            raise _NativeWindowsError()
        owner_sid = self._current_user_sid()
        security_descriptor = ctypes.c_void_p()
        descriptor_size = wintypes.DWORD()
        sddl = f"O:{owner_sid}D:P(A;;FA;;;{owner_sid})"
        if not self._bindings.advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl,
            _SDDL_REVISION_1,
            ctypes.byref(security_descriptor),
            ctypes.byref(descriptor_size),
        ):
            raise _NativeWindowsError()
        if security_descriptor.value is None:
            raise _NativeWindowsError()
        attributes = _SecurityAttributes(
            nLength=ctypes.sizeof(_SecurityAttributes),
            lpSecurityDescriptor=security_descriptor,
            bInheritHandle=False,
        )
        try:
            if kind is WindowsPathKind.DIRECTORY:
                if not self._bindings.kernel32.CreateDirectoryW(
                    str(path),
                    ctypes.byref(attributes),
                ):
                    raise _NativeWindowsError()
                return
            handle = self._bindings.kernel32.CreateFileW(
                str(path),
                _GENERIC_READ | _GENERIC_WRITE | _DELETE,
                0,
                ctypes.byref(attributes),
                _CREATE_NEW,
                _FILE_ATTRIBUTE_NORMAL,
                None,
            )
            if handle == _INVALID_HANDLE_VALUE or handle is None:
                raise _NativeWindowsError()
            self._close_handle(cast(int, handle))
        finally:
            self._bindings.kernel32.LocalFree(security_descriptor)

    def _open_path_no_follow(self, path: PureWindowsPath) -> int | None:
        handle = self._bindings.kernel32.CreateFileW(
            str(path),
            _READ_CONTROL | _FILE_READ_ATTRIBUTES,
            _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
            None,
            _OPEN_EXISTING,
            _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        if handle == _INVALID_HANDLE_VALUE or handle is None:
            if self._bindings.get_last_error() in {_ERROR_FILE_NOT_FOUND, _ERROR_PATH_NOT_FOUND}:
                return None
            raise _NativeWindowsError()
        return cast(int, handle)

    def _authorization_from_snapshot(
        self,
        path: PureWindowsPath,
        snapshot: _WindowsFileSnapshot,
        *,
        kind: WindowsPathKind,
        dacl_policy: _WindowsDaclPolicy = _WindowsDaclPolicy.PRIVATE,
    ) -> WindowsPathAuthorization:
        owner_sid = self._current_user_sid()
        security = _validate_authorized_security(
            snapshot.security,
            kind=kind,
            owner_sid=owner_sid,
            dacl_policy=dacl_policy,
        )
        ancestors = _capture_ancestor_chain(
            path,
            operations=self,
            owner_sid=owner_sid,
        )
        authorization = WindowsPathAuthorization(
            path=path,
            kind=kind,
            security=security,
            _owner_sid=owner_sid,
            _operations=self,
            _token=_AUTHORIZATION_TOKEN,
            _ancestor_chain=ancestors,
            _dacl_policy=dacl_policy,
        )
        authorization._checked_revalidate()
        return authorization

    def _read_file(
        self,
        path: PureWindowsPath,
        *,
        maximum_bytes: int,
        dacl_policy: _WindowsDaclPolicy,
    ) -> WindowsStableFileRead:
        if (
            not _is_valid_windows_path(path)
            or type(maximum_bytes) is not int
            or not 1 <= maximum_bytes < (1 << 63)
            or type(dacl_policy) is not _WindowsDaclPolicy
        ):
            raise _NativeWindowsError()
        handle = self._bindings.kernel32.CreateFileW(
            str(path),
            _GENERIC_READ | _READ_CONTROL | _FILE_READ_ATTRIBUTES,
            _FILE_SHARE_READ,
            None,
            _OPEN_EXISTING,
            _FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        if handle == _INVALID_HANDLE_VALUE or handle is None:
            raise _NativeWindowsError()
        checked_handle = cast(int, handle)
        try:
            before = self._snapshot_from_handle(checked_handle)
            if before.size > maximum_bytes:
                raise _NativeWindowsError()
            authorization = self._authorization_from_snapshot(
                path,
                before,
                kind=WindowsPathKind.FILE,
                dacl_policy=dacl_policy,
            )
            result = bytearray()
            while len(result) < before.size:
                requested = min(64 * 1024, before.size - len(result))
                buffer = ctypes.create_string_buffer(requested)
                received = wintypes.DWORD()
                if not self._bindings.kernel32.ReadFile(
                    checked_handle,
                    buffer,
                    requested,
                    ctypes.byref(received),
                    None,
                ):
                    raise _NativeWindowsError()
                if not 1 <= received.value <= requested:
                    raise _NativeWindowsError()
                result.extend(buffer.raw[: received.value])
            after = self._snapshot_from_handle(checked_handle)
            if after != before or len(result) != before.size:
                raise _NativeWindowsError()
            authorization._checked_revalidate()
            return WindowsStableFileRead(data=bytes(result), authorization=authorization)
        finally:
            self._close_handle(checked_handle)

    def _new_private_file(
        self,
        path: PureWindowsPath,
        data: bytes,
    ) -> tuple[WindowsPathAuthorization, _WindowsFileSnapshot]:
        if not _is_valid_windows_path(path) or type(data) is not bytes:
            raise _NativeWindowsError()
        owner_sid = self._current_user_sid()
        security_descriptor = ctypes.c_void_p()
        descriptor_size = wintypes.DWORD()
        sddl = f"O:{owner_sid}D:P(A;;FA;;;{owner_sid})"
        if (
            not self._bindings.advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl,
                _SDDL_REVISION_1,
                ctypes.byref(security_descriptor),
                ctypes.byref(descriptor_size),
            )
            or security_descriptor.value is None
        ):
            raise _NativeWindowsError()
        attributes = _SecurityAttributes(
            nLength=ctypes.sizeof(_SecurityAttributes),
            lpSecurityDescriptor=security_descriptor,
            bInheritHandle=False,
        )
        handle: int | None = None
        snapshot: _WindowsFileSnapshot | None = None
        try:
            raw_handle = self._bindings.kernel32.CreateFileW(
                str(path),
                _GENERIC_READ | _GENERIC_WRITE | _DELETE | _READ_CONTROL | _FILE_READ_ATTRIBUTES,
                _FILE_SHARE_READ,
                ctypes.byref(attributes),
                _CREATE_NEW,
                _FILE_ATTRIBUTE_NORMAL,
                None,
            )
            if raw_handle == _INVALID_HANDLE_VALUE or raw_handle is None:
                raise _NativeWindowsError()
            handle = cast(int, raw_handle)
            offset = 0
            while offset < len(data):
                chunk = data[offset : offset + 64 * 1024]
                buffer = ctypes.create_string_buffer(chunk, len(chunk))
                written = wintypes.DWORD()
                if not self._bindings.kernel32.WriteFile(
                    handle,
                    buffer,
                    len(chunk),
                    ctypes.byref(written),
                    None,
                ) or written.value != len(chunk):
                    raise _NativeWindowsError()
                offset += written.value
            if not self._bindings.kernel32.FlushFileBuffers(handle):
                raise _NativeWindowsError()
            snapshot = self._snapshot_from_handle(handle)
            if snapshot.size != len(data):
                raise _NativeWindowsError()
            _validate_private_security(
                snapshot.security,
                kind=WindowsPathKind.FILE,
                owner_sid=owner_sid,
            )
        finally:
            if handle is not None:
                self._close_handle(handle)
            self._bindings.kernel32.LocalFree(security_descriptor)
        if snapshot is None:
            raise _NativeWindowsError()
        authorization = self._authorization_from_snapshot(
            path,
            snapshot,
            kind=WindowsPathKind.FILE,
        )
        return authorization, snapshot

    def _temporary_path(self, parent: PureWindowsPath, prefix: str) -> PureWindowsPath:
        for _attempt in range(32):
            candidate = parent / f".{prefix}-{secrets.token_hex(16)}"
            if self._inspect_path(candidate) is None:
                return candidate
        raise _NativeWindowsError()

    def _publish_private_file(
        self,
        path: PureWindowsPath,
        data: bytes,
        *,
        maximum_bytes: int,
        validate_replacement: Callable[[bytes], bool] | None,
        validate_published: Callable[[bytes], bool] | None,
        parent_dacl_policy: _WindowsDaclPolicy,
        target_dacl_policy: _WindowsDaclPolicy,
    ) -> WindowsStableFileRead:
        if (
            not _is_valid_windows_path(path)
            or type(data) is not bytes
            or type(maximum_bytes) is not int
            or not 1 <= maximum_bytes < (1 << 63)
            or len(data) > maximum_bytes
            or (validate_replacement is not None and not callable(validate_replacement))
            or (validate_published is not None and not callable(validate_published))
            or type(parent_dacl_policy) is not _WindowsDaclPolicy
            or type(target_dacl_policy) is not _WindowsDaclPolicy
        ):
            raise _NativeWindowsError()
        parent = _authorize_windows_path(
            path.parent,
            kind=WindowsPathKind.DIRECTORY,
            operations=self,
            create=False,
            dacl_policy=parent_dacl_policy,
        )
        existing_security = self._inspect_path(path)
        existing_read: WindowsStableFileRead | None = None
        if existing_security is not None:
            if validate_replacement is None:
                raise _NativeWindowsError()
            existing_read = self._read_file(
                path,
                maximum_bytes=maximum_bytes,
                dacl_policy=target_dacl_policy,
            )
            if validate_replacement(existing_read.data) is not True:
                raise _NativeWindowsError()
        temporary_path = self._temporary_path(path.parent, "saliencegate-private")
        temporary_authorization, temporary_snapshot = self._new_private_file(
            temporary_path,
            data,
        )
        backup_path: PureWindowsPath | None = None
        published = False
        try:
            parent._checked_revalidate()
            if existing_read is None:
                if not self._bindings.kernel32.MoveFileExW(
                    str(temporary_path),
                    str(path),
                    _MOVEFILE_WRITE_THROUGH,
                ):
                    raise _NativeWindowsError()
            else:
                existing_read.authorization._checked_revalidate()
                backup_path = self._temporary_path(path.parent, "saliencegate-backup")
                if not self._bindings.kernel32.ReplaceFileW(
                    str(path),
                    str(temporary_path),
                    str(backup_path),
                    _REPLACEFILE_FLAGS,
                    None,
                    None,
                ):
                    raise _NativeWindowsError()
            published = True
            parent._checked_revalidate()
            published_read = self._read_file(
                path,
                maximum_bytes=maximum_bytes,
                dacl_policy=target_dacl_policy,
            )
            if (
                published_read.data != data
                or published_read.authorization.security.identity
                != temporary_snapshot.security.identity
                or (
                    validate_published is not None
                    and validate_published(published_read.data) is not True
                )
            ):
                raise _NativeWindowsError()
            if backup_path is not None and existing_read is not None:
                backup_read = self._read_file(
                    backup_path,
                    maximum_bytes=maximum_bytes,
                    dacl_policy=target_dacl_policy,
                )
                if (
                    backup_read.data != existing_read.data
                    or backup_read.authorization.security.identity
                    != existing_read.authorization.security.identity
                ):
                    raise _NativeWindowsError()
                parent._checked_revalidate()
                self._delete_authorized_file(backup_read.authorization)
                backup_path = None
            else:
                parent._checked_revalidate()
            return published_read
        except BaseException:
            if published:
                if backup_path is None:
                    current = self._inspect_path(path)
                    if (
                        current is not None
                        and current.identity == temporary_snapshot.security.identity
                    ):
                        with suppress(Exception):
                            authorization = _authorize_windows_path(
                                path,
                                kind=WindowsPathKind.FILE,
                                operations=self,
                                create=False,
                                dacl_policy=target_dacl_policy,
                            )
                            self._delete_authorized_file(authorization)
                else:
                    with suppress(Exception):
                        self._bindings.kernel32.ReplaceFileW(
                            str(path),
                            str(backup_path),
                            None,
                            _REPLACEFILE_FLAGS,
                            None,
                            None,
                        )
            raise
        finally:
            with suppress(Exception):
                temporary_authorization._checked_revalidate()
                self._delete_authorized_file(temporary_authorization)

    def _delete_authorized_file(
        self,
        authorization: WindowsPathAuthorization,
        *,
        expected_snapshot: _WindowsFileSnapshot | None = None,
    ) -> None:
        if (
            type(authorization) is not WindowsPathAuthorization
            or authorization.kind is not WindowsPathKind.FILE
            or authorization._operations is not self
        ):
            raise _NativeWindowsError()
        authorization._checked_revalidate()
        handle = self._bindings.kernel32.CreateFileW(
            str(authorization.path),
            _DELETE | _READ_CONTROL | _FILE_READ_ATTRIBUTES,
            _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
            None,
            _OPEN_EXISTING,
            _FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        if handle == _INVALID_HANDLE_VALUE or handle is None:
            raise _NativeWindowsError()
        checked_handle = cast(int, handle)
        try:
            snapshot = self._snapshot_from_handle(checked_handle)
            _validate_authorized_security(
                snapshot.security,
                kind=WindowsPathKind.FILE,
                owner_sid=authorization._owner_sid,
                dacl_policy=authorization._dacl_policy,
            )
            if snapshot.security.identity != authorization.security.identity or (
                expected_snapshot is not None and snapshot != expected_snapshot
            ):
                raise _NativeWindowsError()
            disposition = _FileDispositionInfo(DeleteFile=True)
            if not self._bindings.kernel32.SetFileInformationByHandle(
                checked_handle,
                _FILE_DISPOSITION_INFO_CLASS,
                ctypes.byref(disposition),
                ctypes.sizeof(disposition),
            ):
                raise _NativeWindowsError()
        finally:
            self._close_handle(checked_handle)
        if self._inspect_path(authorization.path) is not None:
            raise _NativeWindowsError()

    def _snapshot_authorized_file(
        self,
        authorization: WindowsPathAuthorization,
    ) -> _WindowsFileSnapshot:
        if (
            type(authorization) is not WindowsPathAuthorization
            or authorization.kind is not WindowsPathKind.FILE
            or authorization._operations is not self
        ):
            raise _NativeWindowsError()
        authorization._checked_revalidate()
        handle = self._open_path_no_follow(authorization.path)
        if handle is None:
            raise _NativeWindowsError()
        try:
            snapshot = self._snapshot_from_handle(handle)
            if snapshot.security.identity != authorization.security.identity:
                raise _NativeWindowsError()
            return snapshot
        finally:
            self._close_handle(handle)

    def _private_file_lock(
        self,
        path: PureWindowsPath,
        *,
        blocking: bool,
    ) -> _WindowsPrivateFileLock:
        if not _is_valid_windows_path(path) or type(blocking) is not bool:
            raise _NativeWindowsError()
        parent = _authorize_windows_path(
            path.parent,
            kind=WindowsPathKind.DIRECTORY,
            operations=self,
            create=False,
            dacl_policy=_WindowsDaclPolicy.PRIVATE,
        )
        if self._inspect_path(path) is None:
            parent._checked_revalidate()
            try:
                self._new_private_file(path, b"")
            except Exception:
                # CREATE_NEW can lose to another first opener.  Continue only when the
                # winner's exact object is now inspectable; every later check still
                # enforces owner/DACL/link/reparse/kind and exact identity.
                if self._inspect_path(path) is None:
                    raise
            parent._checked_revalidate()
        else:
            parent._checked_revalidate()
        checked_handle: int | None = None
        for attempt in range(16):
            handle = self._bindings.kernel32.CreateFileW(
                str(path),
                _GENERIC_READ
                | _GENERIC_WRITE
                | _READ_CONTROL
                | _FILE_READ_ATTRIBUTES
                | _FILE_WRITE_ATTRIBUTES,
                _FILE_SHARE_READ | _FILE_SHARE_WRITE,
                None,
                _OPEN_EXISTING,
                _FILE_FLAG_OPEN_REPARSE_POINT,
                None,
            )
            if handle != _INVALID_HANDLE_VALUE and handle is not None:
                checked_handle = cast(int, handle)
                break
            if self._bindings.get_last_error() != _ERROR_SHARING_VIOLATION or attempt == 15:
                raise _NativeWindowsError()
            sleep(0.001)
        if checked_handle is None:
            raise _NativeWindowsError()
        try:
            snapshot = self._snapshot_from_handle(checked_handle)
            authorization = self._authorization_from_snapshot(
                path,
                snapshot,
                kind=WindowsPathKind.FILE,
            )
            parent._checked_revalidate()
            overlapped = _Overlapped()
            if not self._bindings.kernel32.LockFileEx(
                checked_handle,
                _LOCKFILE_EXCLUSIVE_LOCK | (0 if blocking else _LOCKFILE_FAIL_IMMEDIATELY),
                0,
                1,
                0,
                ctypes.byref(overlapped),
            ):
                raise _NativeWindowsError()
            return _WindowsPrivateFileLock(
                operations=self,
                handle=checked_handle,
                authorization=authorization,
                overlapped=overlapped,
                _token=_AUTHORIZATION_TOKEN,
            )
        except BaseException:
            self._close_handle(checked_handle)
            raise

    def _release_private_lock(
        self,
        handle: int,
        authorization: WindowsPathAuthorization,
        overlapped: _Overlapped,
    ) -> None:
        failed = False
        try:
            authorization._checked_revalidate()
            if not self._bindings.kernel32.UnlockFileEx(
                handle,
                0,
                1,
                0,
                ctypes.byref(overlapped),
            ):
                failed = True
        except Exception:
            failed = True
        try:
            self._close_handle(handle)
        except Exception:
            failed = True
        if failed:
            raise _NativeWindowsError()

    def _read_file_identity(
        self,
        handle: int,
        basic: _ByHandleFileInformation,
    ) -> WindowsFileIdentity:
        identity = _FileIdInfo()
        if self._bindings.kernel32.GetFileInformationByHandleEx(
            handle,
            _FILE_ID_INFO_CLASS,
            ctypes.byref(identity),
            ctypes.sizeof(identity),
        ):
            return WindowsFileIdentity(
                volume_serial_number=int(identity.VolumeSerialNumber),
                file_id=bytes(identity.FileId.Identifier),
            )
        if self._bindings.get_last_error() not in {
            _ERROR_INVALID_FUNCTION,
            _ERROR_NOT_SUPPORTED,
            _ERROR_INVALID_PARAMETER,
        }:
            raise _NativeWindowsError()
        legacy_file_id = (int(basic.nFileIndexHigh) << 32) | int(basic.nFileIndexLow)
        return WindowsFileIdentity(
            volume_serial_number=int(basic.dwVolumeSerialNumber),
            file_id=legacy_file_id.to_bytes(8, "little") + (b"\x00" * 8),
        )

    def _read_reparse_tag(self, handle: int) -> int:
        buffer = ctypes.create_string_buffer(_MAXIMUM_REPARSE_DATA_BUFFER_SIZE)
        returned = wintypes.DWORD()
        if not self._bindings.kernel32.DeviceIoControl(
            handle,
            _FSCTL_GET_REPARSE_POINT,
            None,
            0,
            buffer,
            len(buffer),
            ctypes.byref(returned),
            None,
        ):
            # The reparse attribute itself is sufficient to reject authorization.  A fixed
            # sentinel avoids weakening the boundary if a filesystem withholds its tag.
            return _UINT32_MAX
        if returned.value < ctypes.sizeof(wintypes.DWORD):
            return _UINT32_MAX
        return int.from_bytes(buffer.raw[:4], "little")

    def _read_owner_and_dacl(self, handle: int) -> tuple[str, bool, bool, bool]:
        owner = ctypes.c_void_p()
        dacl = ctypes.c_void_p()
        descriptor = ctypes.c_void_p()
        result = self._bindings.advapi32.GetSecurityInfo(
            handle,
            _SE_FILE_OBJECT,
            _OWNER_SECURITY_INFORMATION | _DACL_SECURITY_INFORMATION,
            ctypes.byref(owner),
            None,
            ctypes.byref(dacl),
            None,
            ctypes.byref(descriptor),
        )
        try:
            if result != 0 or owner.value is None or descriptor.value is None:
                raise _NativeWindowsError()
            owner_sid = self._sid_to_text(owner.value)
            private, write_protected, traversal_protected = self._dacl_properties(
                descriptor.value,
                dacl.value,
                owner.value,
            )
            return owner_sid, private, write_protected, traversal_protected
        finally:
            if descriptor.value is not None:
                self._bindings.kernel32.LocalFree(descriptor)

    def _dacl_properties(
        self,
        descriptor: int,
        dacl: int | None,
        owner_sid: int,
    ) -> tuple[bool, bool, bool]:
        if dacl is None:
            return False, False, False
        control = wintypes.WORD()
        revision = wintypes.DWORD()
        if not self._bindings.advapi32.GetSecurityDescriptorControl(
            descriptor,
            ctypes.byref(control),
            ctypes.byref(revision),
        ):
            raise _NativeWindowsError()
        if not control.value & _SE_DACL_PRESENT:
            return False, False, False
        information = _AclSizeInformation()
        if not self._bindings.advapi32.GetAclInformation(
            dacl,
            ctypes.byref(information),
            ctypes.sizeof(information),
            _ACL_SIZE_INFORMATION_CLASS,
        ):
            raise _NativeWindowsError()
        private = bool(control.value & _SE_DACL_PROTECTED and information.AceCount == 1)
        write_protected = True
        traversal_protected = True
        for index in range(information.AceCount):
            ace = ctypes.c_void_p()
            if not self._bindings.advapi32.GetAce(dacl, index, ctypes.byref(ace)):
                raise _NativeWindowsError()
            if ace.value is None:
                raise _NativeWindowsError()
            header = ctypes.cast(ace, ctypes.POINTER(_AceHeader)).contents
            if header.AceSize < 12 or header.AceType not in {
                _ACCESS_ALLOWED_ACE_TYPE,
                _ACCESS_DENIED_ACE_TYPE,
            }:
                private = False
                write_protected = False
                traversal_protected = False
                continue
            mask = ctypes.c_uint32.from_address(ace.value + 4).value
            ace_sid = ace.value + 8
            owner_ace = bool(self._bindings.advapi32.EqualSid(ace_sid, owner_sid))
            if index == 0:
                private = bool(
                    private
                    and header.AceType == _ACCESS_ALLOWED_ACE_TYPE
                    and not header.AceFlags & _INHERITED_ACE
                    and owner_ace
                    and mask & _FILE_ALL_ACCESS == _FILE_ALL_ACCESS
                )
            if (
                header.AceType == _ACCESS_ALLOWED_ACE_TYPE
                and not header.AceFlags & _INHERIT_ONLY_ACE
                and mask & _UNTRUSTED_WRITE_ACCESS
                and not owner_ace
            ):
                ace_sid_text = self._sid_to_text(ace_sid)
                if ace_sid_text not in _TRUSTED_PRIVILEGED_SIDS:
                    write_protected = False
                    if mask & _UNTRUSTED_TRAVERSAL_ACCESS:
                        traversal_protected = False
        return private, write_protected, traversal_protected

    def _sid_to_text(self, sid: int) -> str:
        rendered = wintypes.LPWSTR()
        if not self._bindings.advapi32.ConvertSidToStringSidW(
            sid,
            ctypes.byref(rendered),
        ):
            raise _NativeWindowsError()
        try:
            value = rendered.value
            if not _is_valid_sid_text(value):
                raise _NativeWindowsError()
            return cast(str, value)
        finally:
            self._bindings.kernel32.LocalFree(ctypes.cast(rendered, ctypes.c_void_p))

    def _close_handle(self, handle: int) -> None:
        if not self._bindings.kernel32.CloseHandle(handle):
            raise _NativeWindowsError()

    def __repr__(self) -> str:
        return "NativeWindowsSecurityOperations(<redacted>)"


class _WindowsPrivateFileLock:  # pragma: no cover - exercised by native Windows R01
    __slots__ = ("_authorization", "_handle", "_operations", "_overlapped", "_released")

    def __init__(
        self,
        *,
        operations: NativeWindowsSecurityOperations,
        handle: int,
        authorization: WindowsPathAuthorization,
        overlapped: _Overlapped,
        _token: object,
    ) -> None:
        if (
            _token is not _AUTHORIZATION_TOKEN
            or type(operations) is not NativeWindowsSecurityOperations
            or type(handle) is not int
            or type(authorization) is not WindowsPathAuthorization
            or type(overlapped) is not _Overlapped
        ):
            raise WindowsSecurityError()
        self._operations = operations
        self._handle = handle
        self._authorization = authorization
        self._overlapped = overlapped
        self._released = False

    def __enter__(self) -> None:
        if self._released:
            raise WindowsSecurityError()
        self._authorization.revalidate()

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if self._released:
            raise WindowsSecurityError()
        self._released = True
        _content_free_call(
            lambda: self._operations._release_private_lock(
                self._handle,
                self._authorization,
                self._overlapped,
            )
        )

    def __repr__(self) -> str:
        return "_WindowsPrivateFileLock(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class _WindowsSQLiteSidecar:  # pragma: no cover - exercised by native Windows R01
    suffix: str
    authorization: WindowsPathAuthorization
    transient: bool
    cleanup_snapshot: _WindowsFileSnapshot | None

    def __repr__(self) -> str:
        return "_WindowsSQLiteSidecar(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class WindowsSQLiteAuthorization:  # pragma: no cover - exercised by native Windows R01
    """Exact Windows authority for one database and its SQLite sidecar lifecycle."""

    path: str
    _parent: WindowsPathAuthorization
    _database: WindowsPathAuthorization
    _sidecars: tuple[_WindowsSQLiteSidecar, ...]
    _operations: NativeWindowsSecurityOperations = field(compare=False, repr=False)
    _token: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if (
            self._token is not _AUTHORIZATION_TOKEN
            or type(self.path) is not str
            or type(self._parent) is not WindowsPathAuthorization
            or type(self._database) is not WindowsPathAuthorization
            or type(self._sidecars) is not tuple
            or tuple(sidecar.suffix for sidecar in self._sidecars) != ("-wal", "-shm", "-journal")
            or type(self._operations) is not NativeWindowsSecurityOperations
        ):
            raise WindowsSecurityError()

    def _revalidate_before_sqlite_statements(self) -> None:
        self._parent.revalidate()
        self._database.revalidate()
        for sidecar in self._sidecars:
            sidecar.authorization.revalidate()
        self._parent.revalidate()
        self._database.revalidate()

    def _revalidate_mutable_sqlite(self) -> None:
        self._checked_relaxed_revalidate()

    def revalidate(self) -> None:
        self._checked_relaxed_revalidate()

    def _checked_relaxed_revalidate(self) -> None:
        self._parent.revalidate()
        self._database.revalidate()
        database_path = PureWindowsPath(self.path)
        for sidecar in self._sidecars:
            if not sidecar.transient:
                sidecar.authorization.revalidate()
                continue
            path = PureWindowsPath(f"{database_path}{sidecar.suffix}")
            if self._operations.inspect_path(path) is not None:
                authorize_windows_private_path(
                    path,
                    kind=WindowsPathKind.FILE,
                    operations=self._operations,
                ).revalidate()
        self._parent.revalidate()
        self._database.revalidate()

    def _cleanup_created_sqlite_sidecars(self) -> None:
        for sidecar in reversed(self._sidecars):
            snapshot = sidecar.cleanup_snapshot
            if snapshot is None:
                continue
            path = PureWindowsPath(f"{self.path}{sidecar.suffix}")
            try:
                if self._operations.inspect_path(path) is None:
                    continue
                authorization = authorize_windows_private_path(
                    path,
                    kind=WindowsPathKind.FILE,
                    operations=self._operations,
                )
                if authorization.security.identity != snapshot.security.identity:
                    continue
                self._operations._delete_authorized_file(
                    authorization,
                    expected_snapshot=snapshot,
                )
            except Exception:
                continue

    def __repr__(self) -> str:
        return "WindowsSQLiteAuthorization(<redacted>)"


def authorize_windows_sqlite_path(  # pragma: no cover - exercised by native Windows R01
    path: PureWindowsPath,
    *,
    operations: NativeWindowsSecurityOperations,
    create_database: bool,
    database_authorization: WindowsPathAuthorization | None = None,
) -> WindowsSQLiteAuthorization:
    """Claim an exact Windows database and owner-private WAL/SHM/journal names."""

    return _content_free_call(
        lambda: _authorize_windows_sqlite_path(
            path,
            operations=operations,
            create_database=create_database,
            database_authorization=database_authorization,
        )
    )


def _authorize_windows_sqlite_path(  # pragma: no cover - exercised by native Windows R01
    path: PureWindowsPath,
    *,
    operations: NativeWindowsSecurityOperations,
    create_database: bool,
    database_authorization: WindowsPathAuthorization | None,
) -> WindowsSQLiteAuthorization:
    if (
        not _is_valid_windows_path(path)
        or type(operations) is not NativeWindowsSecurityOperations
        or type(create_database) is not bool
        or (
            database_authorization is not None
            and type(database_authorization) is not WindowsPathAuthorization
        )
    ):
        raise _NativeWindowsError()
    parent = authorize_windows_private_path(
        path.parent,
        kind=WindowsPathKind.DIRECTORY,
        operations=operations,
    )
    database_created: tuple[WindowsPathAuthorization, _WindowsFileSnapshot] | None = None
    created_sidecars: list[tuple[WindowsPathAuthorization, _WindowsFileSnapshot]] = []
    database: WindowsPathAuthorization
    try:
        if database_authorization is not None:
            if (
                database_authorization.path != path
                or database_authorization.kind is not WindowsPathKind.FILE
                or database_authorization._operations is not operations
            ):
                raise _NativeWindowsError()
            database_authorization._checked_revalidate()
            database = database_authorization
        elif operations._inspect_path(path) is None:
            if not create_database:
                raise _NativeWindowsError()
            database_created = operations._new_private_file(path, b"")
            database = database_created[0]
        else:
            database = authorize_windows_private_path(
                path,
                kind=WindowsPathKind.FILE,
                operations=operations,
            )

        sidecars: list[_WindowsSQLiteSidecar] = []
        for suffix, transient in (("-wal", False), ("-shm", False), ("-journal", True)):
            sidecar_path = PureWindowsPath(f"{path}{suffix}")
            cleanup_snapshot: _WindowsFileSnapshot | None = None
            if operations._inspect_path(sidecar_path) is None:
                sidecar_authorization, cleanup_snapshot = operations._new_private_file(
                    sidecar_path,
                    b"",
                )
                created_sidecars.append((sidecar_authorization, cleanup_snapshot))
            else:
                sidecar_authorization = authorize_windows_private_path(
                    sidecar_path,
                    kind=WindowsPathKind.FILE,
                    operations=operations,
                )
            sidecars.append(
                _WindowsSQLiteSidecar(
                    suffix=suffix,
                    authorization=sidecar_authorization,
                    transient=transient,
                    cleanup_snapshot=cleanup_snapshot,
                )
            )
        parent._checked_revalidate()
        database._checked_revalidate()
        result = WindowsSQLiteAuthorization(
            path=str(path),
            _parent=parent,
            _database=database,
            _sidecars=tuple(sidecars),
            _operations=operations,
            _token=_AUTHORIZATION_TOKEN,
        )
        result._revalidate_before_sqlite_statements()
        return result
    except BaseException:
        for authorization, snapshot in reversed(created_sidecars):
            with suppress(Exception):
                operations._delete_authorized_file(
                    authorization,
                    expected_snapshot=snapshot,
                )
        if database_created is not None:
            with suppress(Exception):
                operations._delete_authorized_file(
                    database_created[0],
                    expected_snapshot=database_created[1],
                )
        raise


__all__ = [
    "NativeWindowsSecurityOperations",
    "WindowsFileIdentity",
    "WindowsPathAuthorization",
    "WindowsPathKind",
    "WindowsPathSecurity",
    "WindowsSQLiteAuthorization",
    "WindowsSecurityError",
    "WindowsSecurityOperations",
    "WindowsStableFileRead",
    "authorize_windows_managed_path",
    "authorize_windows_private_path",
    "authorize_windows_sqlite_path",
    "ensure_windows_private_directory",
]
