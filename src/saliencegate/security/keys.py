from __future__ import annotations

import hmac
import os
import secrets
import stat
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from threading import Lock

from saliencegate.security.files import SecureFileError, ensure_private_directory
from saliencegate.security.windows import (
    NativeWindowsSecurityOperations,
    WindowsPathKind,
    WindowsSecurityError,
    authorize_windows_private_path,
    ensure_windows_private_directory,
)


class InvalidInstallationKeyError(ValueError):
    pass


class InsecureKeyFileError(PermissionError):
    pass


class InsecureKeyPathError(ValueError):
    pass


_PROCESS_KEY_LOCK = Lock()


class InstallationKey:
    __slots__ = ("_material",)
    _material: bytes

    def __init__(self, material: bytes) -> None:
        if len(material) != 32:
            raise InvalidInstallationKeyError("installation key must contain exactly 32 bytes")
        object.__setattr__(self, "_material", bytes(material))

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("InstallationKey is immutable")

    def __repr__(self) -> str:
        return "InstallationKey(<redacted>)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InstallationKey):
            return NotImplemented
        return hmac.compare_digest(self._material, other._material)

    def _hmac_sha256(self, message: bytes, *, domain: bytes) -> str:
        framed = (
            len(domain).to_bytes(8, byteorder="big", signed=False)
            + domain
            + len(message).to_bytes(8, byteorder="big", signed=False)
            + message
        )
        return hmac.new(self._material, framed, sha256).hexdigest()

    def _serialized(self) -> bytes:
        return self._material

    def _copy(self) -> InstallationKey:
        return InstallationKey(self._material)


def generate_installation_key() -> InstallationKey:
    return InstallationKey(secrets.token_bytes(32))


def default_installation_key_path(
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    environment = os.environ if environ is None else environ
    if xdg_root := environment.get("XDG_CONFIG_HOME"):
        config_root = Path(xdg_root).expanduser()
    elif os.name == "nt" and (app_data := environment.get("APPDATA")):
        config_root = Path(app_data).expanduser()
    else:
        home = Path(environment.get("HOME", str(Path.home()))).expanduser()
        config_root = home / ".config"
    if not config_root.is_absolute():
        raise InsecureKeyPathError("installation key configuration root must be absolute")
    return config_root / "saliencegate" / "installation.key"


def _validate_key_file(file_stat: os.stat_result, path: Path) -> None:
    if not stat.S_ISREG(file_stat.st_mode):
        raise InsecureKeyFileError(f"installation key is not a regular file: {path}")
    if os.name == "posix":
        if stat.S_IMODE(file_stat.st_mode) & 0o077:
            raise InsecureKeyFileError("installation key permissions must be owner-only")
        if hasattr(os, "getuid") and file_stat.st_uid != os.getuid():
            raise InsecureKeyFileError("installation key must be owned by the current user")


def _read_key(path: Path, *, attempts: int = 50) -> InstallationKey:
    if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        try:
            operations = NativeWindowsSecurityOperations()
            windows_path = PureWindowsPath(os.fspath(path))
            parent_security = operations.inspect_path(windows_path.parent)
            if parent_security is None:
                raise FileNotFoundError(os.fspath(path))
            parent = authorize_windows_private_path(
                windows_path.parent,
                kind=WindowsPathKind.DIRECTORY,
                operations=operations,
            )
            if operations.inspect_path(windows_path) is None:
                parent.revalidate()
                raise FileNotFoundError(os.fspath(path))
            stable = operations.read_private_file(windows_path, maximum_bytes=64)
            parent.revalidate()
            stable.authorization.revalidate()
            if len(stable.data) != 32:
                raise InvalidInstallationKeyError("installation key must contain exactly 32 bytes")
            return InstallationKey(stable.data)
        except (FileNotFoundError, InvalidInstallationKeyError):
            raise
        except WindowsSecurityError:
            raise InsecureKeyFileError(
                "installation key Windows security boundary is invalid"
            ) from None
    if path.is_symlink():
        raise InsecureKeyFileError("installation key cannot be a symbolic link")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    for attempt in range(attempts):
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            if path.is_symlink():
                raise InsecureKeyFileError("installation key cannot be a symbolic link") from error
            raise
        try:
            _validate_key_file(os.fstat(descriptor), path)
            material = os.read(descriptor, 33)
        finally:
            os.close(descriptor)
        if len(material) == 32:
            return InstallationKey(material)
        if attempt + 1 < attempts:
            time.sleep(0.001)
    raise InvalidInstallationKeyError("installation key must contain exactly 32 bytes")


def _write_all(descriptor: int, material: bytes) -> None:
    offset = 0
    while offset < len(material):
        written = os.write(descriptor, material[offset:])
        if written <= 0:
            raise OSError("installation key write made no progress")
        offset += written


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _lock_descriptor(descriptor: int) -> None:
    if os.name == "posix":
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX)
    elif os.name == "nt":  # pragma: no cover - exercised on Windows CI
        import msvcrt

        if os.fstat(descriptor).st_size == 0:
            _write_all(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]


def _unlock_descriptor(descriptor: int) -> None:
    if os.name == "posix":
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_UN)
    elif os.name == "nt":  # pragma: no cover - exercised on Windows CI
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]


@contextmanager
def _installation_key_lock(target: Path) -> Iterator[None]:
    path = target.with_name(f".{target.name}.lock")
    if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        try:
            operations = NativeWindowsSecurityOperations()
            windows_path = PureWindowsPath(os.fspath(path))
            authorize_windows_private_path(
                windows_path.parent,
                kind=WindowsPathKind.DIRECTORY,
                operations=operations,
            ).revalidate()
            with operations.private_file_lock(windows_path):
                yield
            return
        except WindowsSecurityError:
            raise InsecureKeyFileError(
                "installation key lock Windows security boundary is invalid"
            ) from None
    if path.is_symlink():
        raise InsecureKeyFileError("installation key lock cannot be a symbolic link")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    descriptor = os.open(path, flags, 0o600)
    try:
        _validate_key_file(os.fstat(descriptor), path)
        _lock_descriptor(descriptor)
        try:
            yield
        finally:
            _unlock_descriptor(descriptor)
    finally:
        os.close(descriptor)


def _load_or_create_locked(target: Path) -> InstallationKey:
    try:
        return _read_key(target)
    except FileNotFoundError:
        pass

    key = generate_installation_key()
    if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        try:
            operations = NativeWindowsSecurityOperations()
            published = operations.publish_private_file(
                PureWindowsPath(os.fspath(target)),
                key._serialized(),
                maximum_bytes=32,
                validate_published=lambda current: hmac.compare_digest(
                    current,
                    key._serialized(),
                ),
            )
            if not hmac.compare_digest(published.data, key._serialized()):
                raise WindowsSecurityError()
            return InstallationKey(published.data)
        except WindowsSecurityError:
            raise InsecureKeyFileError("installation key Windows publication failed") from None
    temporary = target.with_name(f".{target.name}.{secrets.token_hex(16)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(temporary, flags, 0o600)
        try:
            _write_all(descriptor, key._serialized())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            return _read_key(target)
        _fsync_directory(target.parent)
        return _read_key(target)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def load_or_create_installation_key(path: Path | None = None) -> InstallationKey:
    target = default_installation_key_path() if path is None else Path(path).expanduser()
    if not target.is_absolute():
        raise InsecureKeyPathError("installation key path must be absolute")
    with _PROCESS_KEY_LOCK:
        if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
            try:
                operations = NativeWindowsSecurityOperations()
                ensure_windows_private_directory(
                    PureWindowsPath(os.fspath(target.parent)),
                    operations=operations,
                ).revalidate()
            except WindowsSecurityError:
                raise InsecureKeyPathError(
                    "installation key Windows directory boundary is invalid"
                ) from None
        else:
            try:
                ensure_private_directory(target.parent)
            except SecureFileError:
                raise InsecureKeyPathError(
                    "installation key directory boundary is invalid"
                ) from None

        with _installation_key_lock(target):
            return _load_or_create_locked(target)


def load_installation_key(
    path: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> InstallationKey:
    """Load one existing installation key without creating files or directories."""

    target = (
        default_installation_key_path(environ=environ) if path is None else Path(path).expanduser()
    )
    if not target.is_absolute():
        raise InsecureKeyPathError("installation key path must be absolute")
    return _read_key(target, attempts=1)
