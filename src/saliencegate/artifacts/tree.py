"""Internal, schema-neutral I/O for closed artifact trees.

The replay and algorithm artifact layers supply schema callbacks; this module owns only
bounded filesystem reads and crash-safe sibling publication. It is intentionally not
re-exported from :mod:`saliencegate.artifacts`.
"""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Hashable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Generic, Never, TypeVar

from saliencegate.artifacts.exclusive import (
    ExclusiveStorageError,
    ExclusiveStorageExistsError,
    ExclusiveStorageUnsupportedError,
    _publish_manifest_last_directory,
)
from saliencegate.domain import canonical_json

_ManifestT = TypeVar("_ManifestT")
_FileKeyT = TypeVar("_FileKeyT", bound=Hashable)
_PartT = TypeVar("_PartT")
_ResultT = TypeVar("_ResultT")

_MAX_TREE_FILES = 10_000
_MAX_TREE_FILE_BYTES = 128 * 1024 * 1024
_MAX_MARKER_BYTES = 4096
_SAFE_FILE_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,199}$")
_SAFE_REPLACEMENT_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ArtifactExportError(RuntimeError):
    """A value-free artifact export failure."""


class ArtifactDestinationError(ArtifactExportError):
    """The selected artifact destination cannot safely complete the export."""


class ArtifactExistsError(ArtifactDestinationError):
    def __init__(self) -> None:
        super().__init__("artifact destination already exists")


class ClosedTreeReadErrorKind(StrEnum):
    UNSAFE_PATH = "unsafe_path"
    MISSING_ENTRY = "missing_entry"
    UNSAFE_ENTRY = "unsafe_entry"
    INVALID_DESCRIPTOR = "invalid_descriptor"


_READ_ERROR_MESSAGES: dict[ClosedTreeReadErrorKind, str] = {
    ClosedTreeReadErrorKind.UNSAFE_PATH: "closed artifact path is unsafe",
    ClosedTreeReadErrorKind.MISSING_ENTRY: "closed artifact entry is missing",
    ClosedTreeReadErrorKind.UNSAFE_ENTRY: "closed artifact entry is unsafe",
    ClosedTreeReadErrorKind.INVALID_DESCRIPTOR: "closed artifact descriptor is invalid",
}


class ClosedTreeReadError(ValueError):
    def __init__(self, kind: ClosedTreeReadErrorKind) -> None:
        self.kind = kind
        super().__init__(_READ_ERROR_MESSAGES[kind])


@dataclass(frozen=True, slots=True)
class ClosedTreeFileSpec(Generic[_FileKeyT]):
    key: _FileKeyT
    name: str
    maximum_bytes: int
    expected_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class ClosedTreeDescriptor(Generic[_ManifestT, _FileKeyT]):
    manifest: _ManifestT
    manifest_name: str
    manifest_digest: str
    replacement_key: str
    files: tuple[ClosedTreeFileSpec[_FileKeyT], ...]


@dataclass(frozen=True, slots=True)
class _PathIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int
    mode: int
    link_count: int
    owner: int

    @classmethod
    def from_stat(cls, metadata: os.stat_result) -> _PathIdentity:
        return cls(
            device=metadata.st_dev,
            inode=metadata.st_ino,
            size=metadata.st_size,
            modified_ns=metadata.st_mtime_ns,
            changed_ns=metadata.st_ctime_ns,
            mode=metadata.st_mode,
            link_count=metadata.st_nlink,
            owner=getattr(metadata, "st_uid", 0),
        )

    def payload(self) -> dict[str, int]:
        return {
            "device": self.device,
            "inode": self.inode,
            "size": self.size,
            "modified_ns": self.modified_ns,
            "changed_ns": self.changed_ns,
            "mode": self.mode,
            "link_count": self.link_count,
            "owner": self.owner,
        }

    @classmethod
    def from_payload(cls, value: object) -> _PathIdentity | None:
        fields = (
            "device",
            "inode",
            "size",
            "modified_ns",
            "changed_ns",
            "mode",
            "link_count",
            "owner",
        )
        if type(value) is not dict or set(value) != set(fields):
            return None
        if any(type(value.get(field)) is not int or value[field] < 0 for field in fields):
            return None
        return cls(**{field: value[field] for field in fields})

    def matches(self, metadata: os.stat_result) -> bool:
        return self == type(self).from_stat(metadata)

    def same_object(self, metadata: os.stat_result) -> bool:
        return (
            self.device == metadata.st_dev
            and self.inode == metadata.st_ino
            and stat.S_IFMT(self.mode) == stat.S_IFMT(metadata.st_mode)
            and self.owner == getattr(metadata, "st_uid", 0)
        )

    def matches_after_rename(self, metadata: os.stat_result) -> bool:
        current = type(self).from_stat(metadata)
        return (
            self.device == current.device
            and self.inode == current.inode
            and self.size == current.size
            and self.modified_ns == current.modified_ns
            and self.mode == current.mode
            and self.link_count == current.link_count
            and self.owner == current.owner
        )


@dataclass(frozen=True, slots=True)
class ClosedTreeRead(Generic[_ResultT, _ManifestT, _FileKeyT]):
    value: _ResultT
    descriptor: ClosedTreeDescriptor[_ManifestT, _FileKeyT]
    directory_identity: _PathIdentity

    @property
    def manifest(self) -> _ManifestT:
        return self.descriptor.manifest

    @property
    def manifest_digest(self) -> str:
        return self.descriptor.manifest_digest

    @property
    def replacement_key(self) -> str:
        return self.descriptor.replacement_key


@dataclass(frozen=True, slots=True)
class _ReadFile:
    data: bytes
    identity: _PathIdentity


def _read_failure(kind: ClosedTreeReadErrorKind) -> Never:
    raise ClosedTreeReadError(kind)


def _current_owner(metadata: os.stat_result) -> bool:
    return not (os.name == "posix" and hasattr(os, "getuid") and metadata.st_uid != os.getuid())


def _safe_read_mode(metadata: os.stat_result, *, required_posix_mode: int) -> bool:
    if not _current_owner(metadata):
        return False
    mode = stat.S_IMODE(metadata.st_mode)
    if os.name == "posix":
        return mode == required_posix_mode
    return not mode & 0o077


def _safe_flat_name(name: object) -> bool:
    return (
        type(name) is str
        and _SAFE_FILE_NAME.fullmatch(name) is not None
        and name not in (".", "..")
        and os.path.basename(name) == name
        and "/" not in name
        and "\\" not in name
    )


def _checked_descriptor(
    value: ClosedTreeDescriptor[_ManifestT, _FileKeyT],
    *,
    manifest_name: str,
    maximum_manifest_bytes: int,
) -> ClosedTreeDescriptor[_ManifestT, _FileKeyT]:
    if type(value) is not ClosedTreeDescriptor:
        _read_failure(ClosedTreeReadErrorKind.INVALID_DESCRIPTOR)
    descriptor = value
    if (
        not _safe_flat_name(manifest_name)
        or type(descriptor.manifest_name) is not str
        or descriptor.manifest_name != manifest_name
        or type(descriptor.manifest_digest) is not str
        or _SHA256.fullmatch(descriptor.manifest_digest) is None
        or type(descriptor.replacement_key) is not str
        or _SAFE_REPLACEMENT_KEY.fullmatch(descriptor.replacement_key) is None
        or type(descriptor.files) is not tuple
        or type(maximum_manifest_bytes) is not int
        or not 2 <= maximum_manifest_bytes <= _MAX_TREE_FILE_BYTES
        or not 0 <= len(descriptor.files) <= _MAX_TREE_FILES
    ):
        _read_failure(ClosedTreeReadErrorKind.INVALID_DESCRIPTOR)
    names: set[str] = set()
    aliases: set[str] = set()
    keys: set[_FileKeyT] = set()
    for item in descriptor.files:
        if type(item) is not ClosedTreeFileSpec:
            _read_failure(ClosedTreeReadErrorKind.INVALID_DESCRIPTOR)
        try:
            hash(item.key)
            alias = os.path.normcase(item.name).casefold()
            duplicate_key = item.key in keys
        except Exception:
            _read_failure(ClosedTreeReadErrorKind.INVALID_DESCRIPTOR)
        if (
            not _safe_flat_name(item.name)
            or item.name == manifest_name
            or type(item.maximum_bytes) is not int
            or not 2 <= item.maximum_bytes <= _MAX_TREE_FILE_BYTES
            or (
                item.expected_bytes is not None
                and (
                    type(item.expected_bytes) is not int
                    or not 2 <= item.expected_bytes <= item.maximum_bytes
                )
            )
            or item.name in names
            or alias in aliases
            or duplicate_key
        ):
            _read_failure(ClosedTreeReadErrorKind.INVALID_DESCRIPTOR)
        names.add(item.name)
        aliases.add(alias)
        keys.add(item.key)
    return descriptor


def _read_regular_file(
    directory_fd: int,
    name: str,
    *,
    maximum: int,
    missing_kind: ClosedTreeReadErrorKind = ClosedTreeReadErrorKind.MISSING_ENTRY,
) -> _ReadFile:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        _read_failure(missing_kind)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.EMLINK, errno.ENXIO, errno.ENOTDIR):
            _read_failure(ClosedTreeReadErrorKind.UNSAFE_ENTRY)
        _read_failure(missing_kind)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not _safe_read_mode(before, required_posix_mode=0o600)
            or before.st_size < 2
            or before.st_size > maximum
        ):
            _read_failure(ClosedTreeReadErrorKind.UNSAFE_ENTRY)
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(data) > maximum
            or len(data) != before.st_size
            or _PathIdentity.from_stat(before) != _PathIdentity.from_stat(after)
        ):
            _read_failure(ClosedTreeReadErrorKind.UNSAFE_ENTRY)
        try:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError:
            _read_failure(ClosedTreeReadErrorKind.UNSAFE_ENTRY)
        if not stat.S_ISREG(current.st_mode) or _PathIdentity.from_stat(
            current
        ) != _PathIdentity.from_stat(before):
            _read_failure(ClosedTreeReadErrorKind.UNSAFE_ENTRY)
        return _ReadFile(data=data, identity=_PathIdentity.from_stat(before))
    except ClosedTreeReadError:
        raise
    except OSError:
        _read_failure(ClosedTreeReadErrorKind.UNSAFE_ENTRY)
    finally:
        with suppress(OSError):
            os.close(descriptor)


def read_closed_tree(
    manifest_path: os.PathLike[str] | str,
    *,
    maximum_manifest_bytes: int,
    parse_manifest: Callable[[bytes], ClosedTreeDescriptor[_ManifestT, _FileKeyT]],
    parse_file: Callable[[_FileKeyT, bytes], _PartT],
    finish: Callable[[_ManifestT, Mapping[_FileKeyT, _PartT]], _ResultT],
) -> ClosedTreeRead[_ResultT, _ManifestT, _FileKeyT]:
    """Read one closed flat tree and recheck every filesystem identity before returning."""

    if isinstance(manifest_path, bytes):
        _read_failure(ClosedTreeReadErrorKind.UNSAFE_PATH)
    if (
        type(maximum_manifest_bytes) is not int
        or not 2 <= maximum_manifest_bytes <= _MAX_TREE_FILE_BYTES
    ):
        _read_failure(ClosedTreeReadErrorKind.INVALID_DESCRIPTOR)
    try:
        path = Path(os.fspath(manifest_path))
    except (TypeError, ValueError, OSError):
        _read_failure(ClosedTreeReadErrorKind.UNSAFE_PATH)
    if not _safe_flat_name(path.name):
        _read_failure(ClosedTreeReadErrorKind.UNSAFE_PATH)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(path.parent, flags)
    except OSError:
        _read_failure(ClosedTreeReadErrorKind.MISSING_ENTRY)
    try:
        try:
            directory_metadata = os.fstat(directory_fd)
        except OSError:
            _read_failure(ClosedTreeReadErrorKind.UNSAFE_ENTRY)
        directory_identity = _PathIdentity.from_stat(directory_metadata)
        try:
            named_directory = os.stat(path.parent, follow_symlinks=False)
        except OSError:
            _read_failure(ClosedTreeReadErrorKind.UNSAFE_ENTRY)
        if (
            not stat.S_ISDIR(directory_metadata.st_mode)
            or stat.S_ISLNK(named_directory.st_mode)
            or not _safe_read_mode(directory_metadata, required_posix_mode=0o700)
            or not directory_identity.matches(named_directory)
        ):
            _read_failure(ClosedTreeReadErrorKind.UNSAFE_ENTRY)
        manifest_file = _read_regular_file(
            directory_fd,
            path.name,
            maximum=maximum_manifest_bytes,
        )
        descriptor = _checked_descriptor(
            parse_manifest(manifest_file.data),
            manifest_name=path.name,
            maximum_manifest_bytes=maximum_manifest_bytes,
        )
        identities = {path.name: manifest_file.identity}
        expected_files = {path.name, *(item.name for item in descriptor.files)}
        try:
            listed_files = set(os.listdir(directory_fd))
        except OSError:
            _read_failure(ClosedTreeReadErrorKind.UNSAFE_ENTRY)
        if expected_files - listed_files:
            _read_failure(ClosedTreeReadErrorKind.MISSING_ENTRY)
        if listed_files != expected_files:
            _read_failure(ClosedTreeReadErrorKind.UNSAFE_ENTRY)
        parsed: dict[_FileKeyT, _PartT] = {}
        for item in descriptor.files:
            raw = _read_regular_file(
                directory_fd,
                item.name,
                maximum=item.maximum_bytes,
            )
            identities[item.name] = raw.identity
            if item.expected_bytes is not None and len(raw.data) != item.expected_bytes:
                _read_failure(ClosedTreeReadErrorKind.UNSAFE_ENTRY)
            parsed[item.key] = parse_file(item.key, raw.data)
        value = finish(descriptor.manifest, MappingProxyType(parsed))
        try:
            final_files = set(os.listdir(directory_fd))
        except OSError:
            _read_failure(ClosedTreeReadErrorKind.UNSAFE_ENTRY)
        if final_files != expected_files:
            _read_failure(ClosedTreeReadErrorKind.UNSAFE_ENTRY)
        for name, identity in identities.items():
            try:
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError:
                _read_failure(ClosedTreeReadErrorKind.UNSAFE_ENTRY)
            if not stat.S_ISREG(current.st_mode) or not identity.matches(current):
                _read_failure(ClosedTreeReadErrorKind.UNSAFE_ENTRY)
        try:
            final_directory = os.fstat(directory_fd)
            named_final_directory = os.stat(path.parent, follow_symlinks=False)
        except OSError:
            _read_failure(ClosedTreeReadErrorKind.UNSAFE_ENTRY)
        if not directory_identity.matches(final_directory) or not directory_identity.matches(
            named_final_directory
        ):
            _read_failure(ClosedTreeReadErrorKind.UNSAFE_ENTRY)
        return ClosedTreeRead(
            value=value,
            descriptor=descriptor,
            directory_identity=directory_identity,
        )
    finally:
        with suppress(OSError):
            os.close(directory_fd)


def _write_file(directory: Path, name: str, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(directory / name, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short artifact write")
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
            raise OSError("unsafe artifact write")
        identity = _PathIdentity.from_stat(metadata)
    finally:
        os.close(descriptor)
    if not identity.matches((directory / name).lstat()):
        raise OSError("raced artifact write")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_destination(output: os.PathLike[str] | str) -> Path:
    if isinstance(output, bytes):
        raise ArtifactDestinationError("artifact destination must be a text path")
    try:
        destination = Path(os.fspath(output))
    except (TypeError, ValueError, OSError):
        raise ArtifactDestinationError("artifact destination failed validation") from None
    if destination.name in ("", ".", ".."):
        raise ArtifactDestinationError("artifact destination failed validation")
    return destination


def _safe_parent(metadata: os.stat_result, expected: _PathIdentity) -> bool:
    return (
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and expected.same_object(metadata)
        and _current_owner(metadata)
        and not (os.name == "posix" and stat.S_IMODE(metadata.st_mode) & 0o022)
    )


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _owned_regular(path: Path, expected: _PathIdentity) -> bool:
    current = _lstat_or_none(path)
    return (
        current is not None
        and stat.S_ISREG(current.st_mode)
        and current.st_nlink == 1
        and expected.matches(current)
    )


def _owned_directory(path: Path, expected: _PathIdentity) -> bool:
    current = _lstat_or_none(path)
    return (
        current is not None
        and stat.S_ISDIR(current.st_mode)
        and not stat.S_ISLNK(current.st_mode)
        and expected.matches(current)
    )


def _remove_owned_directory(path: Path, expected: _PathIdentity) -> bool:
    if not _owned_directory(path, expected):
        return False
    shutil.rmtree(path)
    return _lstat_or_none(path) is None


def _remove_owned_staging(path: Path, expected: _PathIdentity) -> bool:
    current = _lstat_or_none(path)
    if current is None:
        return True
    if not stat.S_ISDIR(current.st_mode) or not expected.same_object(current):
        return False
    shutil.rmtree(path)
    return _lstat_or_none(path) is None


def _unlink_owned_regular(path: Path, expected: _PathIdentity) -> bool:
    if not _owned_regular(path, expected):
        return False
    path.unlink()
    return _lstat_or_none(path) is None


@contextmanager
def _destination_lock(
    destination: Path,
    parent: Path,
    parent_identity: _PathIdentity | None = None,
) -> Iterator[None]:
    lock_path = parent / f".{destination.name}.lock"
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor: int | None = None
    try:
        try:
            if parent_identity is not None and not _safe_parent(parent.lstat(), parent_identity):
                raise ArtifactDestinationError("artifact destination parent is unsafe")
            descriptor = os.open(lock_path, flags, 0o600)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or not _current_owner(metadata)
            ):
                raise ArtifactDestinationError("artifact destination lock is unsafe")
            lock_identity = _PathIdentity.from_stat(metadata)
            if not lock_identity.matches(lock_path.lstat()):
                raise ArtifactDestinationError("artifact destination lock is unsafe")
            if os.name == "posix":
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX)
            elif os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt

                if metadata.st_size == 0:
                    os.write(descriptor, b"\0")
                    os.fsync(descriptor)
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
                lock_identity = _PathIdentity.from_stat(os.fstat(descriptor))
            if not lock_identity.matches(lock_path.lstat()) or (
                parent_identity is not None and not _safe_parent(parent.lstat(), parent_identity)
            ):
                raise ArtifactDestinationError("artifact destination lock is unsafe")
        except ArtifactExportError:
            raise
        except OSError:
            raise ArtifactDestinationError("artifact destination lock is unavailable") from None
        yield
    finally:
        if descriptor is not None:
            if os.name == "posix":
                with suppress(OSError):
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            elif os.name == "nt":  # pragma: no cover - exercised on Windows CI
                with suppress(OSError):
                    import msvcrt

                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(  # type: ignore[attr-defined]
                        descriptor,
                        msvcrt.LK_UNLCK,  # type: ignore[attr-defined]
                        1,
                    )
            with suppress(OSError):
                os.close(descriptor)


def _replacement_paths(destination: Path, parent: Path) -> tuple[Path, Path]:
    return (
        parent / f".{destination.name}.backup",
        parent / f".{destination.name}.replace.json",
    )


@dataclass(frozen=True, slots=True)
class _ReplacementMarker:
    destination_name: str
    replacement_key: str
    original: _PathIdentity
    replacement: _PathIdentity
    original_manifest_digest: str
    replacement_manifest_digest: str
    file_identity: _PathIdentity


def _replacement_marker_bytes(
    destination: Path,
    original_metadata: os.stat_result,
    replacement_metadata: os.stat_result,
    *,
    replacement_key: str,
    original_manifest_digest: str,
    replacement_manifest_digest: str,
) -> bytes:
    return canonical_json(
        {
            "schema_version": "artifact-replacement/v1",
            "destination_name": destination.name,
            "run_id": replacement_key,
            "original": _PathIdentity.from_stat(original_metadata).payload(),
            "replacement": _PathIdentity.from_stat(replacement_metadata).payload(),
            "original_manifest_digest": original_manifest_digest,
            "replacement_manifest_digest": replacement_manifest_digest,
        }
    )


def _read_regular_path(path: Path, *, maximum: int) -> tuple[bytes, _PathIdentity] | None:
    metadata = _lstat_or_none(path)
    if metadata is None:
        return None
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or not _current_owner(metadata)
        or not 2 <= metadata.st_size <= maximum
    ):
        raise ArtifactExportError("artifact replacement recovery is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not _PathIdentity.from_stat(metadata).matches(before):
            raise ArtifactExportError("artifact replacement recovery is unsafe")
        data = bytearray()
        while len(data) <= maximum:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = _PathIdentity.from_stat(before)
    current = _lstat_or_none(path)
    if (
        len(data) > maximum
        or len(data) != before.st_size
        or not identity.matches(after)
        or current is None
        or not identity.matches(current)
    ):
        raise ArtifactExportError("artifact replacement recovery is unsafe")
    return bytes(data), identity


def _read_replacement_marker(marker: Path, destination: Path) -> _ReplacementMarker | None:
    loaded = _read_regular_path(marker, maximum=_MAX_MARKER_BYTES)
    if loaded is None:
        return None
    data, identity = loaded
    try:
        payload = json.loads(data)
        canonical_payload = canonical_json(payload)
    except Exception:
        raise ArtifactExportError("artifact replacement recovery is unsafe") from None
    original = _PathIdentity.from_payload(
        payload.get("original") if type(payload) is dict else None
    )
    replacement = _PathIdentity.from_payload(
        payload.get("replacement") if type(payload) is dict else None
    )
    replacement_key = payload.get("run_id") if type(payload) is dict else None
    original_digest = payload.get("original_manifest_digest") if type(payload) is dict else None
    replacement_digest = (
        payload.get("replacement_manifest_digest") if type(payload) is dict else None
    )
    if (
        type(payload) is not dict
        or canonical_payload != data
        or payload.get("schema_version") != "artifact-replacement/v1"
        or payload.get("destination_name") != destination.name
        or set(payload)
        != {
            "schema_version",
            "destination_name",
            "run_id",
            "original",
            "replacement",
            "original_manifest_digest",
            "replacement_manifest_digest",
        }
        or type(replacement_key) is not str
        or _SAFE_REPLACEMENT_KEY.fullmatch(replacement_key) is None
        or original is None
        or replacement is None
        or type(original_digest) is not str
        or type(replacement_digest) is not str
        or _SHA256.fullmatch(original_digest) is None
        or _SHA256.fullmatch(replacement_digest) is None
    ):
        raise ArtifactExportError("artifact replacement recovery is unsafe")
    return _ReplacementMarker(
        destination_name=destination.name,
        replacement_key=replacement_key,
        original=original,
        replacement=replacement,
        original_manifest_digest=original_digest,
        replacement_manifest_digest=replacement_digest,
        file_identity=identity,
    )


_ValidateTree = Callable[[Path, str | None], ClosedTreeDescriptor[_ManifestT, _FileKeyT]]


def _validated_tree_descriptor(
    path: Path,
    *,
    expected_digest: str | None,
    validate_tree: _ValidateTree[_ManifestT, _FileKeyT],
    manifest_name: str,
    maximum_manifest_bytes: int,
) -> ClosedTreeDescriptor[_ManifestT, _FileKeyT] | None:
    try:
        descriptor = validate_tree(path, expected_digest)
        checked = _checked_descriptor(
            descriptor,
            manifest_name=manifest_name,
            maximum_manifest_bytes=maximum_manifest_bytes,
        )
        if expected_digest is not None and checked.manifest_digest != expected_digest:
            return None
        return descriptor
    except Exception:
        return None


def _recover_interrupted_replacement(
    destination: Path,
    parent: Path,
    *,
    validate_tree: _ValidateTree[_ManifestT, _FileKeyT],
    manifest_name: str,
    maximum_manifest_bytes: int,
) -> None:
    backup, marker = _replacement_paths(destination, parent)

    def matches(path: Path, replacement_key: str, digest: str) -> bool:
        descriptor = _validated_tree_descriptor(
            path,
            expected_digest=digest,
            validate_tree=validate_tree,
            manifest_name=manifest_name,
            maximum_manifest_bytes=maximum_manifest_bytes,
        )
        return descriptor is not None and descriptor.replacement_key == replacement_key

    try:
        destination_metadata = _lstat_or_none(destination)
        backup_metadata = _lstat_or_none(backup)
        replacement = _read_replacement_marker(marker, destination)
        if backup_metadata is None:
            if replacement is not None:
                original_present = (
                    destination_metadata is not None
                    and replacement.original.matches_after_rename(destination_metadata)
                )
                published_present = (
                    destination_metadata is not None
                    and replacement.replacement.matches_after_rename(destination_metadata)
                )
                digest = (
                    replacement.original_manifest_digest
                    if original_present
                    else replacement.replacement_manifest_digest
                    if published_present
                    else None
                )
                if digest is None or not matches(
                    destination,
                    replacement.replacement_key,
                    digest,
                ):
                    raise ArtifactExportError("artifact replacement recovery is unsafe")
                if not _unlink_owned_regular(marker, replacement.file_identity):
                    raise ArtifactExportError("artifact replacement recovery is unsafe")
                _fsync_directory(parent)
            return
        if (
            replacement is None
            or stat.S_ISLNK(backup_metadata.st_mode)
            or not stat.S_ISDIR(backup_metadata.st_mode)
            or not replacement.original.matches_after_rename(backup_metadata)
            or not matches(
                backup,
                replacement.replacement_key,
                replacement.original_manifest_digest,
            )
        ):
            raise ArtifactExportError("artifact replacement recovery is unsafe")
        current_backup_identity = _PathIdentity.from_stat(backup_metadata)
        if destination_metadata is None:
            os.rename(backup, destination)
            restored = _lstat_or_none(destination)
            if (
                restored is None
                or not replacement.original.matches_after_rename(restored)
                or not matches(
                    destination,
                    replacement.replacement_key,
                    replacement.original_manifest_digest,
                )
                or not _unlink_owned_regular(marker, replacement.file_identity)
            ):
                raise ArtifactExportError("artifact replacement recovery is unsafe")
            _fsync_directory(parent)
            return
        if stat.S_ISLNK(destination_metadata.st_mode) or not stat.S_ISDIR(
            destination_metadata.st_mode
        ):
            raise ArtifactExportError("artifact replacement recovery is unsafe")
        if not replacement.replacement.matches_after_rename(destination_metadata) or not (
            matches(
                destination,
                replacement.replacement_key,
                replacement.replacement_manifest_digest,
            )
            and _owned_directory(backup, current_backup_identity)
            and matches(
                backup,
                replacement.replacement_key,
                replacement.original_manifest_digest,
            )
        ):
            raise ArtifactExportError("artifact replacement recovery is unsafe")
        if not _remove_owned_directory(
            backup,
            current_backup_identity,
        ) or not _unlink_owned_regular(marker, replacement.file_identity):
            raise ArtifactExportError("artifact replacement recovery is unsafe")
        _fsync_directory(parent)
    except ArtifactExportError:
        raise
    except OSError:
        raise ArtifactExportError("artifact replacement recovery failed") from None


@dataclass(frozen=True, slots=True)
class _ReplacementAuthorization:
    directory: _PathIdentity
    manifest: _PathIdentity
    manifest_digest: str
    replacement_key: str


def _authorized_replace_target(
    destination: Path,
    metadata: os.stat_result,
    expected_replacement_key: str,
    *,
    validate_tree: _ValidateTree[_ManifestT, _FileKeyT],
    manifest_name: str,
    maximum_manifest_bytes: int,
) -> _ReplacementAuthorization:
    descriptor = _validated_tree_descriptor(
        destination,
        expected_digest=None,
        validate_tree=validate_tree,
        manifest_name=manifest_name,
        maximum_manifest_bytes=maximum_manifest_bytes,
    )
    try:
        current = destination.lstat()
        manifest = (destination / manifest_name).lstat()
    except OSError:
        raise ArtifactExistsError() from None
    if (
        descriptor is None
        or descriptor.replacement_key != expected_replacement_key
        or not stat.S_ISDIR(current.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or not stat.S_ISREG(manifest.st_mode)
        or manifest.st_nlink != 1
        or not _PathIdentity.from_stat(metadata).matches(current)
    ):
        raise ArtifactExistsError()
    return _ReplacementAuthorization(
        directory=_PathIdentity.from_stat(current),
        manifest=_PathIdentity.from_stat(manifest),
        manifest_digest=descriptor.manifest_digest,
        replacement_key=descriptor.replacement_key,
    )


def _replacement_is_still_authorized(
    destination: Path,
    authorization: _ReplacementAuthorization,
    expected_replacement_key: str,
    *,
    validate_tree: _ValidateTree[_ManifestT, _FileKeyT],
    manifest_name: str,
    maximum_manifest_bytes: int,
) -> bool:
    try:
        before_directory = destination.lstat()
        before_manifest = (destination / manifest_name).lstat()
        if not authorization.directory.matches(before_directory) or not (
            stat.S_ISREG(before_manifest.st_mode)
            and before_manifest.st_nlink == 1
            and authorization.manifest.matches(before_manifest)
        ):
            return False
        descriptor = _validated_tree_descriptor(
            destination,
            expected_digest=authorization.manifest_digest,
            validate_tree=validate_tree,
            manifest_name=manifest_name,
            maximum_manifest_bytes=maximum_manifest_bytes,
        )
        after_directory = destination.lstat()
        after_manifest = (destination / manifest_name).lstat()
        return (
            descriptor is not None
            and descriptor.replacement_key == expected_replacement_key
            and authorization.replacement_key == expected_replacement_key
            and authorization.directory.matches(after_directory)
            and stat.S_ISREG(after_manifest.st_mode)
            and after_manifest.st_nlink == 1
            and authorization.manifest.matches(after_manifest)
        )
    except Exception:
        return False


def _validated_publication_files(
    files: Mapping[str, bytes],
    descriptor: ClosedTreeDescriptor[_ManifestT, _FileKeyT],
    *,
    maximum_manifest_bytes: int,
) -> dict[str, bytes]:
    try:
        copied = dict(files)
    except Exception:
        raise ArtifactExportError("artifact publish input failed validation") from None
    expected = {descriptor.manifest_name, *(item.name for item in descriptor.files)}
    if set(copied) != expected or any(
        type(name) is not str or type(data) is not bytes for name, data in copied.items()
    ):
        raise ArtifactExportError("artifact publish input failed validation")
    limits = {
        descriptor.manifest_name: maximum_manifest_bytes,
        **{item.name: item.maximum_bytes for item in descriptor.files},
    }
    expected_sizes = {
        item.name: item.expected_bytes
        for item in descriptor.files
        if item.expected_bytes is not None
    }
    for name, data in copied.items():
        if not 2 <= len(data) <= limits[name] or (
            name in expected_sizes and len(data) != expected_sizes[name]
        ):
            raise ArtifactExportError("artifact publish input failed validation")
    return copied


def publish_closed_tree(
    output: os.PathLike[str] | str,
    files: Mapping[str, bytes],
    *,
    manifest_name: str,
    maximum_manifest_bytes: int,
    parse_manifest: Callable[[bytes], ClosedTreeDescriptor[_ManifestT, _FileKeyT]],
    validate_tree: _ValidateTree[_ManifestT, _FileKeyT],
    replace: bool,
) -> ClosedTreeDescriptor[_ManifestT, _FileKeyT]:
    """Validate and atomically publish one closed flat tree through a sibling rename."""

    if type(replace) is not bool:
        raise ArtifactExportError("artifact replace flag failed validation")
    if not _safe_flat_name(manifest_name) or type(maximum_manifest_bytes) is not int:
        raise ArtifactExportError("artifact publish input failed validation")
    try:
        manifest_bytes = files[manifest_name]
        descriptor = parse_manifest(manifest_bytes)
        checked = _checked_descriptor(
            descriptor,
            manifest_name=manifest_name,
            maximum_manifest_bytes=maximum_manifest_bytes,
        )
    except ClosedTreeReadError:
        raise ArtifactExportError("artifact publish input failed validation") from None
    except Exception:
        raise ArtifactExportError("artifact publish input failed validation") from None
    copied = _validated_publication_files(
        files,
        checked,
        maximum_manifest_bytes=maximum_manifest_bytes,
    )
    destination = _safe_destination(output)
    parent = destination.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        parent = parent.resolve(strict=True)
        parent_metadata = parent.lstat()
    except (OSError, RuntimeError):
        raise ArtifactDestinationError("artifact destination is unavailable") from None
    if (
        not stat.S_ISDIR(parent_metadata.st_mode)
        or stat.S_ISLNK(parent_metadata.st_mode)
        or not _current_owner(parent_metadata)
        or (os.name == "posix" and stat.S_IMODE(parent_metadata.st_mode) & 0o022)
    ):
        raise ArtifactDestinationError("artifact destination parent is unsafe")
    parent_identity = _PathIdentity.from_stat(parent_metadata)
    destination = parent / destination.name
    try:
        with _destination_lock(destination, parent, parent_identity):
            locked_parent_identity = _PathIdentity.from_stat(parent.lstat())
            _publish_locked(
                destination,
                parent,
                copied,
                descriptor=descriptor,
                validate_tree=validate_tree,
                replace=replace,
                parent_identity=locked_parent_identity,
                maximum_manifest_bytes=maximum_manifest_bytes,
            )
    except ArtifactExportError:
        raise
    except Exception:
        raise ArtifactDestinationError("artifact atomic publish failed") from None
    return descriptor


def publish_closed_tree_exclusive(
    output: os.PathLike[str] | str,
    files: Mapping[str, bytes],
    *,
    manifest_name: str,
    maximum_manifest_bytes: int,
    parse_manifest: Callable[[bytes], ClosedTreeDescriptor[_ManifestT, _FileKeyT]],
    validate_tree: _ValidateTree[_ManifestT, _FileKeyT],
) -> ClosedTreeDescriptor[_ManifestT, _FileKeyT]:
    """Publish one immutable closed tree directly, with its manifest written last."""

    if not _safe_flat_name(manifest_name) or type(maximum_manifest_bytes) is not int:
        raise ArtifactExportError("artifact publish input failed validation")
    try:
        snapshot = dict(files)
        manifest_bytes = snapshot[manifest_name]
        descriptor = parse_manifest(manifest_bytes)
        checked = _checked_descriptor(
            descriptor,
            manifest_name=manifest_name,
            maximum_manifest_bytes=maximum_manifest_bytes,
        )
        copied = _validated_publication_files(
            snapshot,
            checked,
            maximum_manifest_bytes=maximum_manifest_bytes,
        )
    except ClosedTreeReadError:
        raise ArtifactExportError("artifact publish input failed validation") from None
    except Exception:
        raise ArtifactExportError("artifact publish input failed validation") from None

    def validate_complete(path: Path) -> bool:
        try:
            validated = validate_tree(path, checked.manifest_digest)
            return (
                _checked_descriptor(
                    validated,
                    manifest_name=manifest_name,
                    maximum_manifest_bytes=maximum_manifest_bytes,
                )
                == checked
            )
        except Exception:
            return False

    try:
        _publish_manifest_last_directory(
            output,
            copied,
            manifest_name=manifest_name,
            validate_complete=validate_complete,
        )
    except ExclusiveStorageExistsError:
        raise ArtifactExistsError() from None
    except ExclusiveStorageUnsupportedError:
        raise ArtifactDestinationError("exclusive artifact storage is unsupported") from None
    except ExclusiveStorageError:
        raise ArtifactDestinationError("exclusive artifact publish failed") from None
    except Exception:
        raise ArtifactDestinationError("exclusive artifact publish failed") from None
    return descriptor


def _publish_locked(
    destination: Path,
    parent: Path,
    files: Mapping[str, bytes],
    *,
    descriptor: ClosedTreeDescriptor[_ManifestT, _FileKeyT],
    validate_tree: _ValidateTree[_ManifestT, _FileKeyT],
    replace: bool,
    parent_identity: _PathIdentity,
    maximum_manifest_bytes: int,
) -> None:
    manifest_name = descriptor.manifest_name
    authorization: _ReplacementAuthorization | None = None
    _recover_interrupted_replacement(
        destination,
        parent,
        validate_tree=validate_tree,
        manifest_name=manifest_name,
        maximum_manifest_bytes=maximum_manifest_bytes,
    )
    if not _safe_parent(parent.lstat(), parent_identity):
        raise ArtifactDestinationError("artifact destination parent is unsafe")
    existing = _lstat_or_none(destination)
    if existing is not None:
        if stat.S_ISLNK(existing.st_mode) or not stat.S_ISDIR(existing.st_mode):
            raise ArtifactExistsError()
        if not replace:
            raise ArtifactExistsError()
        authorization = _authorized_replace_target(
            destination,
            existing,
            descriptor.replacement_key,
            validate_tree=validate_tree,
            manifest_name=manifest_name,
            maximum_manifest_bytes=maximum_manifest_bytes,
        )
        existing = destination.lstat()
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=parent))
    staging_metadata = staging.lstat()
    staging_identity = _PathIdentity.from_stat(staging_metadata)
    backup: Path | None = None
    backup_identity: _PathIdentity | None = None
    marker: Path | None = None
    marker_identity: _PathIdentity | None = None
    published = False
    try:
        if (
            not stat.S_ISDIR(staging_metadata.st_mode)
            or stat.S_IMODE(staging_metadata.st_mode) != 0o700
            or not _current_owner(staging_metadata)
        ):
            raise ArtifactDestinationError("artifact staging directory is unsafe")
        for name in sorted(files):
            _write_file(staging, name, files[name])
        _fsync_directory(staging)
        current_staging = staging.lstat()
        if not staging_identity.same_object(current_staging):
            raise ArtifactExistsError()
        staging_identity = _PathIdentity.from_stat(current_staging)
        staged = _validated_tree_descriptor(
            staging,
            expected_digest=descriptor.manifest_digest,
            validate_tree=validate_tree,
            manifest_name=manifest_name,
            maximum_manifest_bytes=maximum_manifest_bytes,
        )
        if staged != descriptor:
            raise ArtifactExportError("artifact staging validation failed")
        if existing is not None:
            assert authorization is not None
            if not _replacement_is_still_authorized(
                destination,
                authorization,
                descriptor.replacement_key,
                validate_tree=validate_tree,
                manifest_name=manifest_name,
                maximum_manifest_bytes=maximum_manifest_bytes,
            ):
                raise ArtifactExistsError()
            backup, marker = _replacement_paths(destination, parent)
            _write_file(
                parent,
                marker.name,
                _replacement_marker_bytes(
                    destination,
                    existing,
                    staging.lstat(),
                    replacement_key=descriptor.replacement_key,
                    original_manifest_digest=authorization.manifest_digest,
                    replacement_manifest_digest=descriptor.manifest_digest,
                ),
            )
            marker_metadata = marker.lstat()
            marker_identity = _PathIdentity.from_stat(marker_metadata)
            if not _owned_regular(marker, marker_identity):
                raise ArtifactExistsError()
            _fsync_directory(parent)
            if not _replacement_is_still_authorized(
                destination,
                authorization,
                descriptor.replacement_key,
                validate_tree=validate_tree,
                manifest_name=manifest_name,
                maximum_manifest_bytes=maximum_manifest_bytes,
            ) or not _owned_regular(marker, marker_identity):
                raise ArtifactExistsError()
            os.replace(destination, backup)
            moved_backup = _lstat_or_none(backup)
            if moved_backup is None or not authorization.directory.matches_after_rename(
                moved_backup
            ):
                raise ArtifactExistsError()
            backup_identity = _PathIdentity.from_stat(moved_backup)
            old = _validated_tree_descriptor(
                backup,
                expected_digest=authorization.manifest_digest,
                validate_tree=validate_tree,
                manifest_name=manifest_name,
                maximum_manifest_bytes=maximum_manifest_bytes,
            )
            if old is None or old.replacement_key != authorization.replacement_key:
                raise ArtifactExistsError()
        try:
            staging_before_publish = _lstat_or_none(staging)
            if staging_before_publish is None or not staging_identity.matches(
                staging_before_publish
            ):
                raise ArtifactExistsError()
            os.replace(staging, destination)
            moved_staging = _lstat_or_none(destination)
            if moved_staging is None or not staging_identity.matches_after_rename(moved_staging):
                raise ArtifactExistsError()
            published_descriptor = _validated_tree_descriptor(
                destination,
                expected_digest=descriptor.manifest_digest,
                validate_tree=validate_tree,
                manifest_name=manifest_name,
                maximum_manifest_bytes=maximum_manifest_bytes,
            )
            if published_descriptor != descriptor:
                raise ArtifactDestinationError("published artifact failed validation")
            published = True
        except Exception:
            if (
                backup is not None
                and backup_identity is not None
                and _owned_directory(backup, backup_identity)
                and _lstat_or_none(destination) is None
            ):
                os.rename(backup, destination)
                restored = _lstat_or_none(destination)
                if restored is not None and backup_identity.matches_after_rename(restored):
                    backup = None
                    backup_identity = None
                    if (
                        marker is not None
                        and marker_identity is not None
                        and _unlink_owned_regular(marker, marker_identity)
                    ):
                        marker = None
                        marker_identity = None
                    _fsync_directory(parent)
            raise
        _fsync_directory(parent)
        if backup is not None and backup_identity is not None:
            old = _validated_tree_descriptor(
                backup,
                expected_digest=(None if authorization is None else authorization.manifest_digest),
                validate_tree=validate_tree,
                manifest_name=manifest_name,
                maximum_manifest_bytes=maximum_manifest_bytes,
            )
            if not (
                _owned_directory(backup, backup_identity)
                and authorization is not None
                and old is not None
                and old.replacement_key == authorization.replacement_key
                and _remove_owned_directory(backup, backup_identity)
            ):
                raise ArtifactDestinationError("artifact replacement cleanup is unsafe")
            backup = None
            backup_identity = None
        if marker is not None and marker_identity is not None:
            if not _unlink_owned_regular(marker, marker_identity):
                raise ArtifactDestinationError("artifact replacement cleanup is unsafe")
            marker = None
            marker_identity = None
        _fsync_directory(parent)
        if not _safe_parent(parent.lstat(), parent_identity):
            raise ArtifactDestinationError("artifact destination parent is unsafe")
    except ArtifactExportError:
        raise
    except Exception:
        raise ArtifactDestinationError("artifact atomic publish failed") from None
    finally:
        if not published:
            with suppress(OSError):
                _remove_owned_staging(staging, staging_identity)
        if (
            backup is not None
            and backup_identity is not None
            and _owned_directory(backup, backup_identity)
            and _lstat_or_none(destination) is None
        ):
            with suppress(OSError):
                os.rename(backup, destination)
                restored = _lstat_or_none(destination)
                if restored is not None and backup_identity.matches_after_rename(restored):
                    backup = None
                    backup_identity = None
        if (
            backup is None
            and marker is not None
            and marker_identity is not None
            and _lstat_or_none(destination) is not None
        ):
            with suppress(OSError):
                if _unlink_owned_regular(marker, marker_identity):
                    marker = None
                    marker_identity = None
