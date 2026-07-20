"""Opaque, byte-preserving edits for provider-owned configuration spans."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import tomllib
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PureWindowsPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from saliencegate.security.windows import (
    NativeWindowsSecurityOperations,
    WindowsPathKind,
    authorize_windows_managed_path,
)

MAX_PROVIDER_CONFIG_BYTES = 2 * 1_024 * 1_024
MAX_OWNED_CONFIG_BYTES = 256 * 1_024
_MARKER = re.compile(r"^saliencegate-owned:[a-z0-9][a-z0-9._-]{0,95}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOML_KEY = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ConfigFileError(ValueError):
    """A content-free configuration boundary failure."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("provider configuration edit failed")


class ConfigSyntax(StrEnum):
    JSON_OBJECT = "json_object"
    OPAQUE_TEXT = "opaque_text"
    TOML_DOCUMENT = "toml_document"


class _ConfigModel(BaseModel):
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


class TomlBooleanConstraint(_ConfigModel):
    """Require one existing TOML boolean path to have a safe value."""

    path: Annotated[
        tuple[
            Annotated[
                str,
                StringConstraints(min_length=1, max_length=64, pattern=_TOML_KEY.pattern),
            ],
            ...,
        ],
        Field(min_length=1, max_length=8),
    ]
    expected: bool


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigFileError()
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ConfigFileError()


def _strict_json_loads(value: bytes) -> object:
    try:
        return json.loads(
            value,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except ConfigFileError:
        raise
    except Exception:
        raise ConfigFileError() from None


def _strict_toml_loads(value: bytes) -> dict[str, object]:
    try:
        return tomllib.loads(value.decode("utf-8", errors="strict"))
    except Exception:
        raise ConfigFileError() from None


class OwnedConfigSpec(_ConfigModel):
    """One provider-owned fragment and its schema-neutral ownership marker."""

    syntax: ConfigSyntax
    marker: Annotated[
        str,
        StringConstraints(min_length=20, max_length=114, pattern=_MARKER.pattern),
    ]
    owned_fragment: Annotated[bytes, Field(min_length=1, max_length=MAX_OWNED_CONFIG_BYTES)] = (
        Field(repr=False)
    )
    toml_boolean_constraints: Annotated[
        tuple[TomlBooleanConstraint, ...],
        Field(max_length=16),
    ] = ()

    @model_validator(mode="after")
    def fragment_is_closed_and_uniquely_marked(self) -> Self:
        marker = self.marker.encode("ascii")
        if self.owned_fragment.count(marker) != 1 or b"\x00" in self.owned_fragment:
            raise ValueError("owned configuration marker is invalid")
        try:
            self.owned_fragment.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("owned configuration fragment is invalid") from None
        if self.syntax is ConfigSyntax.JSON_OBJECT:
            decoded = _strict_json_loads(b"{" + self.owned_fragment + b"}")
            if type(decoded) is not dict or len(decoded) != 1:
                raise ValueError("owned JSON configuration fragment is invalid")
        elif self.syntax is ConfigSyntax.TOML_DOCUMENT:
            _strict_toml_loads(self.owned_fragment)
        if self.toml_boolean_constraints:
            paths = tuple(item.path for item in self.toml_boolean_constraints)
            if self.syntax is not ConfigSyntax.TOML_DOCUMENT or paths != tuple(sorted(set(paths))):
                raise ValueError("owned TOML constraints are invalid")
        return self


class OwnedConfigReverseEdit(_ConfigModel):
    """Minimal reverse edit; it never contains the foreign configuration preimage."""

    schema_version: Literal["owned-config-reverse-edit/v1"] = "owned-config-reverse-edit/v1"
    syntax: ConfigSyntax
    target_existed: bool
    preimage_digest: Annotated[str, StringConstraints(pattern=_SHA256.pattern)]
    installed_digest: Annotated[str, StringConstraints(pattern=_SHA256.pattern)]
    marker: Annotated[str, StringConstraints(pattern=_MARKER.pattern)]
    start: Annotated[int, Field(ge=0, le=MAX_PROVIDER_CONFIG_BYTES)]
    end: Annotated[int, Field(ge=1, le=MAX_PROVIDER_CONFIG_BYTES)]
    owned_span_digest: Annotated[str, StringConstraints(pattern=_SHA256.pattern)]
    owned_span_base64: Annotated[str, StringConstraints(min_length=4, max_length=400_000)] = Field(
        repr=False
    )

    @model_validator(mode="after")
    def reverse_edit_is_self_consistent(self) -> Self:
        try:
            span = base64.b64decode(self.owned_span_base64, validate=True)
        except Exception:
            raise ValueError("owned configuration reverse edit is invalid") from None
        if (
            self.end <= self.start
            or self.end - self.start != len(span)
            or len(span) > MAX_OWNED_CONFIG_BYTES + 8
            or hashlib.sha256(span).hexdigest() != self.owned_span_digest
            or span.count(self.marker.encode("ascii")) != 1
        ):
            raise ValueError("owned configuration reverse edit is invalid")
        return self

    def owned_span(self) -> bytes:
        try:
            return base64.b64decode(self.owned_span_base64, validate=True)
        except Exception:
            raise ConfigFileError() from None


class OwnedConfigPlan(_ConfigModel):
    installed_bytes: Annotated[
        bytes,
        Field(min_length=1, max_length=MAX_PROVIDER_CONFIG_BYTES, repr=False),
    ]
    reverse_edit: OwnedConfigReverseEdit


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validated_spec(value: object) -> OwnedConfigSpec:
    try:
        payload = (
            value.model_dump(mode="python", warnings="error")
            if type(value) is OwnedConfigSpec
            else value
        )
        return OwnedConfigSpec.model_validate(payload)
    except Exception:
        raise ConfigFileError() from None


def _json_object_insertion(current: bytes, fragment: bytes) -> tuple[bytes, int, bytes]:
    decoded = _strict_json_loads(current)
    if type(decoded) is not dict:
        raise ConfigFileError()
    owned = _strict_json_loads(b"{" + fragment + b"}")
    if type(owned) is not dict or len(owned) != 1 or next(iter(owned)) in decoded:
        raise ConfigFileError()
    end = len(current)
    while end and current[end - 1] in b" \t\r\n":
        end -= 1
    if end < 2 or current[end - 1 : end] != b"}":
        raise ConfigFileError()
    close = end - 1
    start = 0
    while start < close and current[start] in b" \t\r\n":
        start += 1
    if current[start : start + 1] != b"{":
        raise ConfigFileError()
    separator = b"" if not decoded else b","
    span = separator + fragment
    return current[:close] + span + current[close:], close, span


def _opaque_insertion(current: bytes, fragment: bytes) -> tuple[bytes, int, bytes]:
    try:
        current.decode("utf-8")
    except UnicodeDecodeError:
        raise ConfigFileError() from None
    separator = b"" if not current or current.endswith((b"\n", b"\r")) else b"\n"
    span = separator + fragment
    return current + span, len(current), span


def _validate_toml_boolean_constraints(
    document: dict[str, object],
    constraints: tuple[TomlBooleanConstraint, ...],
) -> None:
    for constraint in constraints:
        current: object = document
        missing = False
        for component in constraint.path:
            if type(current) is not dict:
                raise ConfigFileError()
            if component not in current:
                missing = True
                break
            current = current[component]
        if missing:
            continue
        if type(current) is not bool or current is not constraint.expected:
            raise ConfigFileError()


def plan_owned_config_install(
    current: bytes | None,
    spec: OwnedConfigSpec,
) -> OwnedConfigPlan:
    """Plan one insertion without normalizing or retaining any foreign bytes."""

    try:
        checked = _validated_spec(spec)
        if current is not None and type(current) is not bytes:
            raise ConfigFileError()
        target_existed = current is not None
        original = (
            b"{}"
            if current is None and checked.syntax is ConfigSyntax.JSON_OBJECT
            else (b"" if current is None else current)
        )
        if len(original) > MAX_PROVIDER_CONFIG_BYTES:
            raise ConfigFileError()
        marker = checked.marker.encode("ascii")
        if marker in original:
            raise ConfigFileError()
        if checked.syntax is ConfigSyntax.JSON_OBJECT:
            installed, start, span = _json_object_insertion(original, checked.owned_fragment)
        else:
            installed, start, span = _opaque_insertion(original, checked.owned_fragment)
        if checked.syntax is ConfigSyntax.TOML_DOCUMENT:
            original_document = _strict_toml_loads(original)
            installed_document = _strict_toml_loads(installed)
            _validate_toml_boolean_constraints(
                original_document,
                checked.toml_boolean_constraints,
            )
            _validate_toml_boolean_constraints(
                installed_document,
                checked.toml_boolean_constraints,
            )
        if len(installed) > MAX_PROVIDER_CONFIG_BYTES:
            raise ConfigFileError()
        reverse = OwnedConfigReverseEdit(
            syntax=checked.syntax,
            target_existed=target_existed,
            preimage_digest=_digest(original),
            installed_digest=_digest(installed),
            marker=checked.marker,
            start=start,
            end=start + len(span),
            owned_span_digest=_digest(span),
            owned_span_base64=base64.b64encode(span).decode("ascii"),
        )
        return OwnedConfigPlan(installed_bytes=installed, reverse_edit=reverse)
    except ConfigFileError:
        raise
    except Exception:
        raise ConfigFileError() from None


def remove_owned_config_edit(
    current: bytes,
    reverse_edit: OwnedConfigReverseEdit,
) -> bytes | None:
    """Reverse an untouched edit exactly, or remove one unique owned span after drift."""

    try:
        if type(current) is not bytes or len(current) > MAX_PROVIDER_CONFIG_BYTES:
            raise ConfigFileError()
        payload = (
            reverse_edit.model_dump(mode="python", warnings="error")
            if type(reverse_edit) is OwnedConfigReverseEdit
            else reverse_edit
        )
        checked = OwnedConfigReverseEdit.model_validate(payload)
        span = checked.owned_span()
        if _digest(current) == checked.installed_digest:
            if current[checked.start : checked.end] != span:
                raise ConfigFileError()
            restored = current[: checked.start] + current[checked.end :]
            if _digest(restored) != checked.preimage_digest:
                raise ConfigFileError()
            return None if not checked.target_existed else restored
        if current.count(span) != 1 or current.count(checked.marker.encode("ascii")) != 1:
            raise ConfigFileError()
        start = current.index(span)
        before = current[:start]
        after = current[start + len(span) :]
        restored = before + after
        if checked.syntax is ConfigSyntax.JSON_OBJECT:
            if type(_strict_json_loads(current)) is not dict:
                raise ConfigFileError()
            candidates = [restored]
            following = next(
                (index for index, value in enumerate(after) if value not in b" \t\r\n"),
                None,
            )
            if following is not None and after[following : following + 1] == b",":
                candidates.append(before + after[:following] + after[following + 1 :])
            preceding = next(
                (
                    index
                    for index in range(len(before) - 1, -1, -1)
                    if before[index] not in b" \t\r\n"
                ),
                None,
            )
            if preceding is not None and before[preceding : preceding + 1] == b",":
                candidates.append(before[:preceding] + before[preceding + 1 :] + after)
            valid: list[bytes] = []
            for candidate in candidates:
                try:
                    decoded = _strict_json_loads(candidate)
                except ConfigFileError:
                    continue
                if type(decoded) is dict and candidate not in valid:
                    valid.append(candidate)
            if len(valid) != 1:
                raise ConfigFileError()
            restored = valid[0]
        elif checked.syntax is ConfigSyntax.TOML_DOCUMENT:
            _strict_toml_loads(current)
            _strict_toml_loads(restored)
        if len(restored) > MAX_PROVIDER_CONFIG_BYTES:
            raise ConfigFileError()
        return restored
    except ConfigFileError:
        raise
    except Exception:
        raise ConfigFileError() from None


@dataclass(frozen=True, slots=True, repr=False)
class _ConfigSnapshot:
    data: bytes | None
    identity: tuple[int, int, int, int, int, int] | None
    mode: int


def _safe_parent(path: Path) -> os.stat_result:
    metadata = path.parent.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or (os.name == "posix" and hasattr(os, "getuid") and metadata.st_uid != os.getuid())
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise ConfigFileError()
    return metadata


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_mode,
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        metadata.st_mode,
    )


def _safe_ancestor(metadata: os.stat_result, *, leaf: bool) -> bool:
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        return False
    mode = stat.S_IMODE(metadata.st_mode)
    if leaf:
        return metadata.st_uid == os.getuid() and not mode & 0o022
    return metadata.st_uid in (0, os.getuid()) and (not mode & 0o022 or bool(mode & stat.S_ISVTX))


def _open_config_parent(path: Path) -> tuple[int, tuple[int, int, int, int]]:
    if os.name != "posix" or not hasattr(os, "getuid") or path.anchor != os.sep:
        raise ConfigFileError()
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(os.sep, flags)
    try:
        components = path.parent.parts[1:]
        for index, component in enumerate(components):
            child = os.open(component, flags, dir_fd=descriptor)
            try:
                opened = os.fstat(child)
                named = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
                if (
                    _directory_identity(opened) != _directory_identity(named)
                    or not _safe_ancestor(opened, leaf=index == len(components) - 1)
                    or not _safe_ancestor(named, leaf=index == len(components) - 1)
                ):
                    raise ConfigFileError()
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        parent_identity = _directory_identity(os.fstat(descriptor))
        return descriptor, parent_identity
    except BaseException:
        os.close(descriptor)
        raise


def _revalidate_config_parent(
    path: Path,
    descriptor: int,
    expected: tuple[int, int, int, int],
) -> None:
    if _directory_identity(os.fstat(descriptor)) != expected or not _safe_ancestor(
        os.fstat(descriptor), leaf=True
    ):
        raise ConfigFileError()
    fresh, fresh_identity = _open_config_parent(path)
    try:
        if fresh_identity != expected:
            raise ConfigFileError()
    finally:
        os.close(fresh)


def _snapshot_at(
    path: Path,
    descriptor: int,
    parent_identity: tuple[int, int, int, int],
) -> _ConfigSnapshot:
    _revalidate_config_parent(path, descriptor, parent_identity)
    try:
        named = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        _revalidate_config_parent(path, descriptor, parent_identity)
        try:
            os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return _ConfigSnapshot(data=None, identity=None, mode=0o600)
        raise ConfigFileError() from None
    if (
        not stat.S_ISREG(named.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or named.st_nlink != 1
        or named.st_size > MAX_PROVIDER_CONFIG_BYTES
        or named.st_uid != os.getuid()
        or stat.S_IMODE(named.st_mode) & 0o022
    ):
        raise ConfigFileError()
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    opened_descriptor = os.open(path.name, flags, dir_fd=descriptor)
    try:
        opened = os.fstat(opened_descriptor)
        data = bytearray()
        while len(data) <= MAX_PROVIDER_CONFIG_BYTES:
            chunk = os.read(
                opened_descriptor,
                min(64 * 1024, MAX_PROVIDER_CONFIG_BYTES + 1 - len(data)),
            )
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(opened_descriptor)
    finally:
        os.close(opened_descriptor)
    current = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
    expected = _identity(named)
    if (
        _identity(opened) != expected
        or _identity(after) != expected
        or _identity(current) != expected
        or len(data) != named.st_size
        or len(data) > MAX_PROVIDER_CONFIG_BYTES
    ):
        raise ConfigFileError()
    _revalidate_config_parent(path, descriptor, parent_identity)
    return _ConfigSnapshot(
        data=bytes(data),
        identity=expected,
        mode=stat.S_IMODE(named.st_mode),
    )


def _snapshot(path: Path) -> _ConfigSnapshot:
    if os.name == "posix":
        descriptor, parent_identity = _open_config_parent(path)
        try:
            return _snapshot_at(path, descriptor, parent_identity)
        finally:
            os.close(descriptor)
    _safe_parent(path)
    try:
        named = path.lstat()
    except FileNotFoundError:
        return _ConfigSnapshot(data=None, identity=None, mode=0o600)
    if (
        not stat.S_ISREG(named.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or named.st_nlink != 1
        or named.st_size > MAX_PROVIDER_CONFIG_BYTES
        or (os.name == "posix" and hasattr(os, "getuid") and named.st_uid != os.getuid())
    ):
        raise ConfigFileError()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        data = bytearray()
        while len(data) <= MAX_PROVIDER_CONFIG_BYTES:
            chunk = os.read(descriptor, min(64 * 1024, MAX_PROVIDER_CONFIG_BYTES + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = path.lstat()
    expected = _identity(named)
    if (
        _identity(opened) != expected
        or _identity(after) != expected
        or _identity(current) != expected
        or len(data) != named.st_size
        or len(data) > MAX_PROVIDER_CONFIG_BYTES
    ):
        raise ConfigFileError()
    return _ConfigSnapshot(
        data=bytes(data),
        identity=expected,
        mode=stat.S_IMODE(named.st_mode),
    )


def read_config_bytes(path: Path) -> bytes | None:
    try:
        if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
            raise ConfigFileError()
        if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
            operations = NativeWindowsSecurityOperations()
            windows_path = PureWindowsPath(os.fspath(path))
            parent = authorize_windows_managed_path(
                windows_path.parent,
                kind=WindowsPathKind.DIRECTORY,
                operations=operations,
            )
            if operations.inspect_path(windows_path) is None:
                parent.revalidate()
                if operations.inspect_path(windows_path) is not None:
                    raise ConfigFileError()
                return None
            stable = operations.read_managed_file(
                windows_path,
                maximum_bytes=MAX_PROVIDER_CONFIG_BYTES,
            )
            parent.revalidate()
            stable.authorization.revalidate()
            return stable.data
        return _snapshot(path).data
    except ConfigFileError:
        raise
    except Exception:
        raise ConfigFileError() from None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_config_bytes(path: Path, *, expected: bytes | None, data: bytes) -> None:
    """Atomically replace exactly the bytes observed by the caller."""

    temporary: str | None = None
    parent_descriptor: int | None = None
    try:
        if (
            not isinstance(path, Path)
            or not path.is_absolute()
            or ".." in path.parts
            or (expected is not None and type(expected) is not bytes)
            or type(data) is not bytes
            or len(data) > MAX_PROVIDER_CONFIG_BYTES
            or (not data and expected is None)
        ):
            raise ConfigFileError()
        if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
            operations = NativeWindowsSecurityOperations()
            published = operations.publish_managed_file(
                PureWindowsPath(os.fspath(path)),
                data,
                maximum_bytes=MAX_PROVIDER_CONFIG_BYTES,
                validate_replacement=(
                    None
                    if expected is None
                    else lambda current: hmac.compare_digest(current, expected)
                ),
                validate_published=lambda current: hmac.compare_digest(current, data),
            )
            if not hmac.compare_digest(published.data, data):
                raise ConfigFileError()
            return
        parent_descriptor, parent_identity = _open_config_parent(path)
        snapshot = _snapshot_at(path, parent_descriptor, parent_identity)
        if snapshot.data != expected:
            raise ConfigFileError()
        temporary = f".{path.name}.{secrets.token_hex(16)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            temporary,
            flags,
            snapshot.mode,
            dir_fd=parent_descriptor,
        )
        try:
            os.fchmod(descriptor, snapshot.mode)
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise ConfigFileError()
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if _snapshot_at(path, parent_descriptor, parent_identity).identity != snapshot.identity:
            raise ConfigFileError()
        if snapshot.identity is None:
            os.link(
                temporary,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            os.unlink(temporary, dir_fd=parent_descriptor)
            temporary = None
        else:
            os.replace(
                temporary,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            temporary = None
        os.fsync(parent_descriptor)
        if _snapshot_at(path, parent_descriptor, parent_identity).data != data:
            raise ConfigFileError()
        _revalidate_config_parent(path, parent_descriptor, parent_identity)
    except ConfigFileError:
        raise
    except Exception:
        raise ConfigFileError() from None
    finally:
        if temporary is not None and parent_descriptor is not None:
            with suppress(OSError):
                os.unlink(temporary, dir_fd=parent_descriptor)
        if parent_descriptor is not None:
            with suppress(OSError):
                os.close(parent_descriptor)


def delete_config_bytes(path: Path, *, expected: bytes) -> None:
    """Delete only the exact user-owned config bytes supplied by the caller."""

    try:
        if type(expected) is not bytes:
            raise ConfigFileError()
        if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
            operations = NativeWindowsSecurityOperations()
            windows_path = PureWindowsPath(os.fspath(path))
            parent = authorize_windows_managed_path(
                windows_path.parent,
                kind=WindowsPathKind.DIRECTORY,
                operations=operations,
            )
            stable = operations.read_managed_file(
                windows_path,
                maximum_bytes=MAX_PROVIDER_CONFIG_BYTES,
            )
            if not hmac.compare_digest(stable.data, expected):
                raise ConfigFileError()
            parent.revalidate()
            operations.delete_authorized_file(stable.authorization)
            parent.revalidate()
            if operations.inspect_path(windows_path) is not None:
                raise ConfigFileError()
            return
        if os.name == "posix":
            descriptor, parent_identity = _open_config_parent(path)
            try:
                if _snapshot_at(path, descriptor, parent_identity).data != expected:
                    raise ConfigFileError()
                os.unlink(path.name, dir_fd=descriptor)
                os.fsync(descriptor)
                if _snapshot_at(path, descriptor, parent_identity).data is not None:
                    raise ConfigFileError()
                _revalidate_config_parent(path, descriptor, parent_identity)
                return
            finally:
                os.close(descriptor)
        if _snapshot(path).data != expected:
            raise ConfigFileError()
        path.unlink()
        _fsync_directory(path.parent)
        if path.exists() or path.is_symlink():
            raise ConfigFileError()
    except ConfigFileError:
        raise
    except Exception:
        raise ConfigFileError() from None


__all__ = [
    "MAX_PROVIDER_CONFIG_BYTES",
    "ConfigFileError",
    "ConfigSyntax",
    "OwnedConfigPlan",
    "OwnedConfigReverseEdit",
    "OwnedConfigSpec",
    "delete_config_bytes",
    "plan_owned_config_install",
    "publish_config_bytes",
    "read_config_bytes",
    "remove_owned_config_edit",
]
