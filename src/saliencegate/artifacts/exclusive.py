"""POSIX-only primitives for immutable flat artifact storage.

This module is deliberately schema-neutral. Benchmark adapters provide already validated bytes and
their own complete-tree validator; this layer owns descriptor-relative creation, exact modes,
identity checks, bounded reads, and fail-closed manifest-last publication.
"""

from __future__ import annotations

import os
import re
import stat
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Never

_MAX_PUBLICATION_FILES = 10_000
_MAX_LOCKED_ENTRIES = 65_536
_MAX_FLAT_FILE_BYTES = 128 * 1024 * 1024
_SAFE_FILE_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,199}$")


class ExclusiveStorageError(RuntimeError):
    """A value-free immutable-storage failure."""


class ExclusiveStorageExistsError(ExclusiveStorageError):
    """The destination exists but is not the exact complete immutable tree."""


class ExclusiveStorageUnsupportedError(ExclusiveStorageError):
    """The host cannot enforce the required POSIX storage boundary."""


@dataclass(frozen=True, slots=True)
class _Identity:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int
    mode: int
    link_count: int
    owner: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _Identity:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            size=value.st_size,
            modified_ns=value.st_mtime_ns,
            changed_ns=value.st_ctime_ns,
            mode=value.st_mode,
            link_count=value.st_nlink,
            owner=value.st_uid,
        )

    def matches(self, value: os.stat_result) -> bool:
        return self == type(self).from_stat(value)

    def same_object(self, value: os.stat_result) -> bool:
        return (
            self.device == value.st_dev
            and self.inode == value.st_ino
            and stat.S_IFMT(self.mode) == stat.S_IFMT(value.st_mode)
            and self.owner == value.st_uid
        )


@dataclass(frozen=True, slots=True)
class _FlatTreeIdentity:
    directory: _Identity
    files: tuple[tuple[str, _Identity], ...]


def _fail(message: str = "immutable artifact storage failed") -> Never:
    raise ExclusiveStorageError(message)


def _required_posix_primitives_available() -> bool:
    required_dir_fd = (os.open, os.mkdir, os.stat, os.unlink)
    required_flags = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    return (
        os.name == "posix"
        and all(function in os.supports_dir_fd for function in required_dir_fd)
        and os.scandir in os.supports_fd
        and all(hasattr(os, name) for name in required_flags)
    )


def _require_posix_primitives() -> None:
    if not _required_posix_primitives_available():
        raise ExclusiveStorageUnsupportedError("immutable artifact storage is unsupported")
    try:
        import fcntl

        if not callable(fcntl.flock):
            raise AttributeError
    except (AttributeError, ImportError):
        raise ExclusiveStorageUnsupportedError(
            "immutable artifact storage is unsupported"
        ) from None


def _safe_flat_name(name: object) -> bool:
    return (
        type(name) is str
        and _SAFE_FILE_NAME.fullmatch(name) is not None
        and name not in (".", "..")
        and os.path.basename(name) == name
        and "/" not in name
        and "\\" not in name
    )


def _snapshot_files(files: Mapping[str, bytes], manifest_name: str) -> dict[str, bytes]:
    try:
        copied = dict(files)
    except Exception:
        _fail("immutable artifact input failed validation")
    aliases: set[str] = set()
    if (
        not _safe_flat_name(manifest_name)
        or not 1 <= len(copied) <= _MAX_PUBLICATION_FILES
        or manifest_name not in copied
    ):
        _fail("immutable artifact input failed validation")
    for name, data in copied.items():
        alias = name.casefold() if type(name) is str else ""
        if (
            not _safe_flat_name(name)
            or alias in aliases
            or type(data) is not bytes
            or not 1 <= len(data) <= _MAX_FLAT_FILE_BYTES
        ):
            _fail("immutable artifact input failed validation")
        aliases.add(alias)
    return copied


def _safe_output(output: os.PathLike[str] | str) -> Path:
    if isinstance(output, bytes):
        _fail("immutable artifact destination failed validation")
    try:
        raw = os.fspath(output)
        if type(raw) is not str or "\0" in raw:
            _fail("immutable artifact destination failed validation")
        os.fsencode(raw)
        destination = Path(raw)
    except ExclusiveStorageError:
        raise
    except Exception:
        _fail("immutable artifact destination failed validation")
    if destination.name in ("", ".", ".."):
        _fail("immutable artifact destination failed validation")
    return destination


def _current_owner(metadata: os.stat_result) -> bool:
    return metadata.st_uid == os.getuid()


def _safe_parent(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and _current_owner(metadata)
        and stat.S_IMODE(metadata.st_mode) & 0o022 == 0
    )


def _verify_parent_identity(parent: Path, parent_fd: int, identity: _Identity) -> None:
    try:
        opened = os.fstat(parent_fd)
        named = parent.lstat()
    except OSError:
        _fail("immutable artifact destination parent identity changed")
    if (
        not _safe_parent(opened)
        or not _safe_parent(named)
        or not identity.same_object(opened)
        or not identity.same_object(named)
    ):
        _fail("immutable artifact destination parent identity changed")


def _open_directory_at(parent_fd: int, name: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    return os.open(name, flags, dir_fd=parent_fd)


def _scan_names(
    directory_fd: int,
    *,
    maximum_entries: int = _MAX_PUBLICATION_FILES,
) -> tuple[str, ...]:
    names: list[str] = []
    try:
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                names.append(entry.name)
                if len(names) > maximum_entries:
                    _fail()
    except ExclusiveStorageError:
        raise
    except Exception:
        _fail("immutable artifact directory scan failed")
    return tuple(sorted(names))


def _read_regular_exact(
    directory_fd: int,
    name: str,
    expected: bytes,
    *,
    synchronize: bool = False,
) -> _Identity:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError:
        _fail()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or not _current_owner(before)
            or before.st_size != len(expected)
        ):
            _fail()
        data = bytearray()
        while len(data) <= len(expected):
            chunk = os.read(descriptor, min(64 * 1024, len(expected) + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        if synchronize:
            os.fsync(descriptor)
        after = os.fstat(descriptor)
        identity = _Identity.from_stat(before)
        try:
            named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError:
            _fail()
        if (
            bytes(data) != expected
            or not identity.matches(after)
            or not identity.matches(named)
            or not stat.S_ISREG(named.st_mode)
        ):
            _fail()
        return identity
    except ExclusiveStorageError:
        raise
    except OSError:
        _fail()
    finally:
        os.close(descriptor)


def _create_regular_exclusive(directory_fd: int, name: str, data: bytes) -> _Identity:
    """Create one owner-only regular file relative to a verified directory descriptor."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    except OSError:
        _fail()
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail()
            view = view[written:]
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or not _current_owner(metadata)
            or metadata.st_size != len(data)
        ):
            _fail()
        identity = _Identity.from_stat(metadata)
    except ExclusiveStorageError:
        raise
    except OSError:
        _fail()
    finally:
        os.close(descriptor)
    try:
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError:
        _fail()
    if not identity.matches(named) or not stat.S_ISREG(named.st_mode):
        _fail()
    return identity


def _read_regular_bounded(
    directory_fd: int,
    name: str,
    *,
    maximum_bytes: int,
) -> tuple[bytes, _Identity]:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError:
        _fail("locked artifact directory entry is unsafe")
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or not _current_owner(before)
            or not 1 <= before.st_size <= maximum_bytes
        ):
            _fail("locked artifact directory entry is unsafe")
        data = bytearray()
        while len(data) <= maximum_bytes:
            chunk = os.read(descriptor, min(64 * 1024, maximum_bytes + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(descriptor)
        identity = _Identity.from_stat(before)
        try:
            named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError:
            _fail("locked artifact directory entry is unsafe")
        if (
            len(data) > maximum_bytes
            or len(data) != before.st_size
            or not identity.matches(after)
            or not identity.matches(named)
            or not stat.S_ISREG(named.st_mode)
        ):
            _fail("locked artifact directory entry is unsafe")
        return bytes(data), identity
    except ExclusiveStorageError:
        raise
    except OSError:
        _fail("locked artifact directory entry is unsafe")
    finally:
        os.close(descriptor)


def _verify_exact_tree(
    parent_fd: int,
    destination_name: str,
    files: Mapping[str, bytes],
    *,
    synchronize: bool = False,
) -> _FlatTreeIdentity:
    try:
        directory_fd = _open_directory_at(parent_fd, destination_name)
    except OSError:
        _fail()
    try:
        before = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o700
            or not _current_owner(before)
        ):
            _fail()
        expected_names = tuple(sorted(files))
        if _scan_names(directory_fd) != expected_names:
            _fail()
        identities: list[tuple[str, _Identity]] = []
        for name in expected_names:
            identities.append(
                (
                    name,
                    _read_regular_exact(
                        directory_fd,
                        name,
                        files[name],
                        synchronize=synchronize,
                    ),
                )
            )
        if _scan_names(directory_fd) != expected_names:
            _fail()
        for name, file_identity in identities:
            try:
                current_file = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError:
                _fail()
            if not stat.S_ISREG(current_file.st_mode) or not file_identity.matches(current_file):
                _fail()
        if synchronize:
            os.fsync(directory_fd)
        after = os.fstat(directory_fd)
        identity = _Identity.from_stat(after)
        try:
            named = os.stat(destination_name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError:
            _fail()
        if not stat.S_ISDIR(named.st_mode) or not identity.matches(named):
            _fail()
        return _FlatTreeIdentity(directory=identity, files=tuple(identities))
    finally:
        os.close(directory_fd)


def _destination_exists(parent_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError:
        raise ExclusiveStorageExistsError("immutable artifact destination already exists") from None
    return True


def _publish_manifest_last_directory(
    output: os.PathLike[str] | str,
    files: Mapping[str, bytes],
    *,
    manifest_name: str,
    validate_complete: Callable[[Path], bool],
) -> None:
    """Publish one immutable directory directly, preserving every partial failure state."""

    _require_posix_primitives()
    copied = _snapshot_files(files, manifest_name)
    destination = _safe_output(output)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        parent = destination.parent.resolve(strict=True)
        parent_metadata = parent.lstat()
    except (TypeError, ValueError, UnicodeError, OSError, RuntimeError):
        _fail("immutable artifact destination is unavailable")
    if not _safe_parent(parent_metadata):
        _fail("immutable artifact destination parent is unsafe")
    destination = parent / destination.name
    parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        parent_fd = os.open(parent, parent_flags)
    except OSError:
        _fail("immutable artifact destination is unavailable")
    created = False
    try:
        parent_identity = _Identity.from_stat(parent_metadata)
        _verify_parent_identity(parent, parent_fd, parent_identity)
        if _destination_exists(parent_fd, destination.name):
            try:
                before_validation = _verify_exact_tree(parent_fd, destination.name, copied)
                if not validate_complete(destination):
                    _fail()
                after_validation = _verify_exact_tree(
                    parent_fd,
                    destination.name,
                    copied,
                    synchronize=True,
                )
                if after_validation != before_validation:
                    _fail()
                _verify_parent_identity(parent, parent_fd, parent_identity)
                os.fsync(parent_fd)
                return
            except ExclusiveStorageError:
                raise ExclusiveStorageExistsError(
                    "immutable artifact destination already exists"
                ) from None
            except Exception:
                raise ExclusiveStorageExistsError(
                    "immutable artifact destination already exists"
                ) from None
        try:
            os.mkdir(destination.name, 0o700, dir_fd=parent_fd)
            created = True
            directory_fd = _open_directory_at(parent_fd, destination.name)
        except OSError:
            if _destination_exists(parent_fd, destination.name):
                raise ExclusiveStorageExistsError(
                    "immutable artifact destination already exists"
                ) from None
            _fail()
        try:
            os.fchmod(directory_fd, 0o700)
            initial = os.fstat(directory_fd)
            if (
                not stat.S_ISDIR(initial.st_mode)
                or stat.S_IMODE(initial.st_mode) != 0o700
                or not _current_owner(initial)
            ):
                _fail()
            directory_object = _Identity.from_stat(initial)
            for name in sorted(set(copied) - {manifest_name}):
                _create_regular_exclusive(directory_fd, name, copied[name])
            os.fsync(directory_fd)
            _create_regular_exclusive(directory_fd, manifest_name, copied[manifest_name])
            os.fsync(directory_fd)
            current = os.fstat(directory_fd)
            named = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not directory_object.same_object(current)
                or not directory_object.same_object(named)
                or stat.S_IMODE(current.st_mode) != 0o700
            ):
                _fail()
        finally:
            os.close(directory_fd)
        before_validation = _verify_exact_tree(parent_fd, destination.name, copied)
        if not validate_complete(destination):
            _fail()
        after_validation = _verify_exact_tree(
            parent_fd,
            destination.name,
            copied,
            synchronize=True,
        )
        if after_validation != before_validation:
            _fail()
        _verify_parent_identity(parent, parent_fd, parent_identity)
        os.fsync(parent_fd)
    except (ExclusiveStorageError, ExclusiveStorageExistsError):
        raise
    except Exception:
        _fail()
    finally:
        os.close(parent_fd)
    if not created:  # pragma: no cover - documents the intentional no-cleanup boundary
        _fail()


def _validate_locked_bounds(maximum_entries: object, maximum_file_bytes: object) -> None:
    if (
        type(maximum_entries) is not int
        or not 1 <= maximum_entries <= _MAX_LOCKED_ENTRIES
        or type(maximum_file_bytes) is not int
        or not 1 <= maximum_file_bytes <= _MAX_FLAT_FILE_BYTES
    ):
        _fail("locked artifact directory bounds failed validation")


def _validate_entry_name(
    name: str,
    validator: Callable[[str], bool] | None,
) -> None:
    if validator is None:
        return
    try:
        accepted = validator(name)
    except Exception:
        _fail("locked artifact directory entry name failed validation")
    if type(accepted) is not bool or not accepted:
        _fail("locked artifact directory entry name failed validation")


def _locked_entry_identities(
    directory_fd: int,
    *,
    lock_name: str,
    maximum_entries: int,
    maximum_file_bytes: int,
    entry_name_validator: Callable[[str], bool] | None,
) -> dict[str, _Identity]:
    names = _scan_names(directory_fd, maximum_entries=maximum_entries + 1)
    if lock_name not in names or not 1 <= len(names) <= maximum_entries + 1:
        _fail("locked artifact directory inventory is unsafe")
    aliases: set[str] = set()
    identities: dict[str, _Identity] = {}
    for name in names:
        alias = name.casefold()
        if not _safe_flat_name(name) or alias in aliases:
            _fail("locked artifact directory inventory is unsafe")
        aliases.add(alias)
        if name == lock_name:
            continue
        _validate_entry_name(name, entry_name_validator)
        try:
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError:
            _fail("locked artifact directory inventory is unsafe")
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or not _current_owner(metadata)
            or not 1 <= metadata.st_size <= maximum_file_bytes
        ):
            _fail("locked artifact directory inventory is unsafe")
        identities[name] = _Identity.from_stat(metadata)
    if _scan_names(directory_fd, maximum_entries=maximum_entries + 1) != names:
        _fail("locked artifact directory inventory is unsafe")
    for name, identity in identities.items():
        try:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError:
            _fail("locked artifact directory inventory is unsafe")
        if not stat.S_ISREG(current.st_mode) or not identity.matches(current):
            _fail("locked artifact directory inventory is unsafe")
    return identities


class LockedFlatDirectory:
    """A bounded flat-directory transaction held under one advisory exclusive lock."""

    __slots__ = (
        "_active",
        "_directory_fd",
        "_directory_identity",
        "_directory_name",
        "_entries",
        "_entry_name_validator",
        "_lock_fd",
        "_lock_identity",
        "_lock_name",
        "_maximum_entries",
        "_maximum_file_bytes",
        "_parent_fd",
        "_parent_identity",
        "_parent_path",
    )

    def __init__(
        self,
        *,
        parent_fd: int,
        directory_fd: int,
        lock_fd: int,
        parent_path: Path,
        parent_identity: _Identity,
        directory_name: str,
        lock_name: str,
        maximum_entries: int,
        maximum_file_bytes: int,
        directory_identity: _Identity,
        lock_identity: _Identity,
        entries: dict[str, _Identity],
        entry_name_validator: Callable[[str], bool] | None,
    ) -> None:
        self._active = True
        self._parent_fd = parent_fd
        self._parent_path = parent_path
        self._parent_identity = parent_identity
        self._directory_fd = directory_fd
        self._lock_fd = lock_fd
        self._directory_name = directory_name
        self._lock_name = lock_name
        self._maximum_entries = maximum_entries
        self._maximum_file_bytes = maximum_file_bytes
        self._directory_identity = directory_identity
        self._lock_identity = lock_identity
        self._entries = entries
        self._entry_name_validator = entry_name_validator

    def _require_active(self) -> None:
        if not self._active:
            _fail("locked artifact directory handle is closed")

    def _refresh_directory_identity(self) -> None:
        current = os.fstat(self._directory_fd)
        try:
            named = os.stat(
                self._directory_name,
                dir_fd=self._parent_fd,
                follow_symlinks=False,
            )
        except OSError:
            _fail("locked artifact directory identity changed")
        identity = _Identity.from_stat(current)
        if (
            not stat.S_ISDIR(current.st_mode)
            or stat.S_IMODE(current.st_mode) != 0o700
            or not _current_owner(current)
            or not identity.matches(named)
        ):
            _fail("locked artifact directory identity changed")
        self._directory_identity = identity

    def _verify_container(self) -> None:
        self._require_active()
        _verify_parent_identity(self._parent_path, self._parent_fd, self._parent_identity)
        try:
            directory_metadata = os.fstat(self._directory_fd)
            lock_metadata = os.fstat(self._lock_fd)
            named_directory = os.stat(
                self._directory_name,
                dir_fd=self._parent_fd,
                follow_symlinks=False,
            )
            named_lock = os.stat(
                self._lock_name,
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
        except OSError:
            _fail("locked artifact directory identity changed")
        if (
            not self._directory_identity.matches(directory_metadata)
            or not self._directory_identity.matches(named_directory)
            or not self._lock_identity.matches(lock_metadata)
            or not self._lock_identity.matches(named_lock)
        ):
            _fail("locked artifact directory identity changed")

    def _verify_inventory(self) -> None:
        self._verify_container()
        current = _locked_entry_identities(
            self._directory_fd,
            lock_name=self._lock_name,
            maximum_entries=self._maximum_entries,
            maximum_file_bytes=self._maximum_file_bytes,
            entry_name_validator=self._entry_name_validator,
        )
        if current != self._entries:
            _fail("locked artifact directory inventory changed")
        self._verify_container()

    @property
    def names(self) -> tuple[str, ...]:
        self._verify_container()
        return tuple(sorted(self._entries))

    def read_regular(self, name: str, *, maximum_bytes: int) -> bytes:
        self._verify_container()
        if (
            not _safe_flat_name(name)
            or name == self._lock_name
            or name not in self._entries
            or type(maximum_bytes) is not int
            or not 1 <= maximum_bytes <= self._maximum_file_bytes
        ):
            _fail("locked artifact directory read failed validation")
        data, identity = _read_regular_bounded(
            self._directory_fd,
            name,
            maximum_bytes=maximum_bytes,
        )
        if identity != self._entries[name]:
            _fail("locked artifact directory entry identity changed")
        self._verify_container()
        return data

    def create_regular_exclusive(
        self,
        name: str,
        data: bytes,
        *,
        maximum_bytes: int,
    ) -> None:
        self._verify_container()
        if (
            not _safe_flat_name(name)
            or name == self._lock_name
            or name in self._entries
            or len(self._entries) >= self._maximum_entries
            or type(data) is not bytes
            or type(maximum_bytes) is not int
            or not 1 <= maximum_bytes <= self._maximum_file_bytes
            or not 1 <= len(data) <= maximum_bytes
        ):
            _fail("locked artifact directory create failed validation")
        _validate_entry_name(name, self._entry_name_validator)
        identity = _create_regular_exclusive(self._directory_fd, name, data)
        self._entries[name] = identity
        try:
            os.fsync(self._directory_fd)
        except OSError:
            _fail("locked artifact directory create failed")
        self._refresh_directory_identity()
        self._verify_container()

    def _close(self) -> None:
        self._active = False


def _open_lock_file(directory_fd: int, lock_name: str, *, create: bool) -> tuple[int, _Identity]:
    flags = os.O_RDWR | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW
    if create:
        flags |= os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(lock_name, flags, 0o600, dir_fd=directory_fd)
    except OSError:
        _fail("locked artifact directory lock is unavailable")
    try:
        if create:
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or not _current_owner(metadata)
            or metadata.st_size != 0
        ):
            _fail("locked artifact directory lock is unsafe")
        identity = _Identity.from_stat(metadata)
        try:
            named = os.stat(lock_name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError:
            _fail("locked artifact directory lock is unsafe")
        if not identity.matches(named):
            _fail("locked artifact directory lock is unsafe")
        return descriptor, identity
    except Exception:
        os.close(descriptor)
        raise


def _verify_lock_identity(
    directory_fd: int,
    lock_fd: int,
    lock_name: str,
    lock_identity: _Identity,
) -> None:
    try:
        opened = os.fstat(lock_fd)
        named = os.stat(lock_name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError:
        _fail("locked artifact directory lock identity changed")
    if not lock_identity.matches(opened) or not lock_identity.matches(named):
        _fail("locked artifact directory lock identity changed")


def _synchronize_regular_identity(
    directory_fd: int,
    name: str,
    identity: _Identity,
) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError:
        _fail("locked artifact directory durability check failed")
    try:
        before = os.fstat(descriptor)
        if not identity.matches(before):
            _fail("locked artifact directory durability check failed")
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not identity.matches(after) or not identity.matches(named):
            _fail("locked artifact directory durability check failed")
    except ExclusiveStorageError:
        raise
    except OSError:
        _fail("locked artifact directory durability check failed")
    finally:
        os.close(descriptor)


def _synchronize_locked_directory(
    directory_fd: int,
    lock_fd: int,
    lock_identity: _Identity,
    entries: Mapping[str, _Identity],
) -> None:
    try:
        before_lock = os.fstat(lock_fd)
        if not lock_identity.matches(before_lock):
            _fail("locked artifact directory durability check failed")
        os.fsync(lock_fd)
        if not lock_identity.matches(os.fstat(lock_fd)):
            _fail("locked artifact directory durability check failed")
        for name, identity in entries.items():
            _synchronize_regular_identity(directory_fd, name, identity)
        os.fsync(directory_fd)
    except ExclusiveStorageError:
        raise
    except OSError:
        _fail("locked artifact directory durability check failed")


@contextmanager
def open_locked_flat_directory(
    directory: os.PathLike[str] | str,
    *,
    create: bool,
    lock_name: str = "review.lock",
    maximum_entries: int,
    maximum_file_bytes: int,
    entry_name_validator: Callable[[str], bool] | None = None,
) -> Iterator[LockedFlatDirectory]:
    """Open one owner-only flat directory and hold its lock across scan/read/create."""

    _require_posix_primitives()
    if (
        type(create) is not bool
        or not _safe_flat_name(lock_name)
        or (entry_name_validator is not None and not callable(entry_name_validator))
    ):
        _fail("locked artifact directory input failed validation")
    _validate_locked_bounds(maximum_entries, maximum_file_bytes)
    destination = _safe_output(directory)
    try:
        if create:
            destination.parent.mkdir(parents=True, exist_ok=True)
        parent = destination.parent.resolve(strict=True)
        parent_metadata = parent.lstat()
    except (TypeError, ValueError, UnicodeError, OSError, RuntimeError):
        _fail("locked artifact directory is unavailable")
    if not _safe_parent(parent_metadata):
        _fail("locked artifact directory parent is unsafe")
    destination = parent / destination.name
    parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    parent_fd: int | None = None
    directory_fd: int | None = None
    lock_fd: int | None = None
    locked = False
    handle: LockedFlatDirectory | None = None
    try:
        parent_fd = os.open(parent, parent_flags)
        opened_parent = os.fstat(parent_fd)
        parent_identity = _Identity.from_stat(parent_metadata)
        if not _safe_parent(opened_parent) or not parent_identity.same_object(opened_parent):
            _fail("locked artifact directory parent is unsafe")
        exists = _destination_exists(parent_fd, destination.name)
        created = False
        if not exists:
            if not create:
                _fail("locked artifact directory is unavailable")
            try:
                os.mkdir(destination.name, 0o700, dir_fd=parent_fd)
                created = True
            except OSError:
                _fail("locked artifact directory is unavailable")
        directory_fd = _open_directory_at(parent_fd, destination.name)
        if created:
            os.fchmod(directory_fd, 0o700)
        directory_metadata = os.fstat(directory_fd)
        named_directory = os.stat(
            destination.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        directory_identity = _Identity.from_stat(directory_metadata)
        if (
            not stat.S_ISDIR(directory_metadata.st_mode)
            or stat.S_IMODE(directory_metadata.st_mode) != 0o700
            or not _current_owner(directory_metadata)
            or not directory_identity.matches(named_directory)
        ):
            _fail("locked artifact directory is unsafe")
        lock_fd, lock_identity = _open_lock_file(directory_fd, lock_name, create=created)
        import fcntl

        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        locked = True
        _verify_lock_identity(directory_fd, lock_fd, lock_name, lock_identity)
        directory_metadata = os.fstat(directory_fd)
        named_directory = os.stat(
            destination.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        directory_identity = _Identity.from_stat(directory_metadata)
        if not directory_identity.matches(named_directory):
            _fail("locked artifact directory identity changed")
        entries = _locked_entry_identities(
            directory_fd,
            lock_name=lock_name,
            maximum_entries=maximum_entries,
            maximum_file_bytes=maximum_file_bytes,
            entry_name_validator=entry_name_validator,
        )
        _synchronize_locked_directory(directory_fd, lock_fd, lock_identity, entries)
        os.fsync(parent_fd)
        revalidated_entries = _locked_entry_identities(
            directory_fd,
            lock_name=lock_name,
            maximum_entries=maximum_entries,
            maximum_file_bytes=maximum_file_bytes,
            entry_name_validator=entry_name_validator,
        )
        if revalidated_entries != entries:
            _fail("locked artifact directory inventory changed")
        directory_metadata = os.fstat(directory_fd)
        named_directory = os.stat(
            destination.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if not directory_identity.matches(directory_metadata) or not directory_identity.matches(
            named_directory
        ):
            _fail("locked artifact directory identity changed")
        _verify_parent_identity(parent, parent_fd, parent_identity)
        _verify_lock_identity(directory_fd, lock_fd, lock_name, lock_identity)
        handle = LockedFlatDirectory(
            parent_fd=parent_fd,
            directory_fd=directory_fd,
            lock_fd=lock_fd,
            parent_path=parent,
            parent_identity=parent_identity,
            directory_name=destination.name,
            lock_name=lock_name,
            maximum_entries=maximum_entries,
            maximum_file_bytes=maximum_file_bytes,
            directory_identity=directory_identity,
            lock_identity=lock_identity,
            entries=entries,
            entry_name_validator=entry_name_validator,
        )
        try:
            yield handle
        except BaseException:
            handle._verify_inventory()
            raise
        else:
            handle._verify_inventory()
    except ExclusiveStorageError:
        raise
    except OSError:
        _fail("locked artifact directory operation failed")
    finally:
        had_prior_failure = sys.exc_info()[0] is not None
        cleanup_failed = False
        if handle is not None:
            try:
                handle._close()
            except Exception:
                cleanup_failed = True
        if locked and lock_fd is not None:
            try:
                import fcntl

                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except Exception:
                cleanup_failed = True
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except Exception:
                cleanup_failed = True
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except Exception:
                cleanup_failed = True
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except Exception:
                cleanup_failed = True
        if cleanup_failed and not had_prior_failure:
            _fail("locked artifact directory cleanup failed")


__all__ = [
    "ExclusiveStorageError",
    "ExclusiveStorageExistsError",
    "ExclusiveStorageUnsupportedError",
    "LockedFlatDirectory",
    "open_locked_flat_directory",
]
