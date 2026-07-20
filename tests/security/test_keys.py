from __future__ import annotations

import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PureWindowsPath
from threading import Event

import pytest

import saliencegate.security.keys as keys_module
from saliencegate.security import (
    InsecureKeyFileError,
    InsecureKeyPathError,
    InstallationKey,
    InvalidInstallationKeyError,
    default_installation_key_path,
    generate_installation_key,
    load_installation_key,
    load_or_create_installation_key,
)
from saliencegate.security.windows import (
    NativeWindowsSecurityOperations,
    WindowsPathKind,
    ensure_windows_private_directory,
)


def _write_private_key_fixture(path: Path, data: bytes) -> None:
    if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        operations = NativeWindowsSecurityOperations()
        ensure_windows_private_directory(
            PureWindowsPath(os.fspath(path.parent)),
            operations=operations,
        )
        operations.publish_private_file(
            PureWindowsPath(os.fspath(path)),
            data,
            maximum_bytes=max(1, len(data)),
            validate_published=lambda current: current == data,
        )
        return
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(0o600)


@pytest.mark.parametrize("size", [0, 1, 31, 33, 64])
def test_installation_key_requires_exactly_32_bytes(size: int) -> None:
    with pytest.raises(InvalidInstallationKeyError, match="32 bytes"):
        InstallationKey(b"k" * size)


def test_key_representation_never_exposes_material() -> None:
    material = b"visible-test-material-32-bytes!!"
    key = InstallationKey(material)

    assert material.decode() not in repr(key)
    assert "redacted" in repr(key).lower()
    assert not hasattr(key, "hmac_sha256")


def test_installation_key_is_immutable() -> None:
    key = InstallationKey(b"k" * 32)

    with pytest.raises(AttributeError, match="immutable"):
        key._material = b"z" * 32


def test_generated_keys_are_distinct() -> None:
    assert generate_installation_key() != generate_installation_key()
    assert generate_installation_key() != object()


def test_default_key_path_uses_the_user_configuration_directory(tmp_path: Path) -> None:
    path = default_installation_key_path(
        environ={"XDG_CONFIG_HOME": str(tmp_path / "configuration")}
    )

    assert path == tmp_path / "configuration" / "saliencegate" / "installation.key"

    home_path = default_installation_key_path(environ={"HOME": str(tmp_path / "home")})
    assert home_path == tmp_path / "home" / ".config" / "saliencegate" / "installation.key"


def test_relative_configuration_roots_cannot_write_into_the_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(InsecureKeyPathError, match="absolute"):
        default_installation_key_path(environ={"XDG_CONFIG_HOME": ".config"})
    with pytest.raises(InsecureKeyPathError, match="absolute"):
        load_or_create_installation_key(Path(".config/installation.key"))


def test_key_file_is_created_once_and_reloaded(tmp_path: Path) -> None:
    path = tmp_path / "keys" / "installation.key"
    first = load_or_create_installation_key(path)
    second = load_or_create_installation_key(path)

    assert first == second
    assert path.read_bytes() != b""
    assert len(path.read_bytes()) == 32
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.with_name(f".{path.name}.lock").stat().st_mode) == 0o600


def test_existing_key_loader_is_read_only_and_never_creates_parent_or_lock(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "installation.key"

    with pytest.raises(FileNotFoundError):
        load_installation_key(missing)
    assert not missing.parent.exists()

    path = tmp_path / "private" / "installation.key"
    _write_private_key_fixture(path, b"k" * 32)
    before = path.stat()

    assert load_installation_key(path) == InstallationKey(b"k" * 32)
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert not path.with_name(f".{path.name}.lock").exists()


def test_key_is_published_only_after_its_contents_are_synced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "posix":
        pytest.skip("directory durability checks are POSIX-specific")
    path = tmp_path / "keys" / "installation.key"
    publication_state_at_fsync: list[bool] = []
    real_fsync = os.fsync

    def observe_fsync(descriptor: int) -> None:
        publication_state_at_fsync.append(path.exists())
        real_fsync(descriptor)

    monkeypatch.setattr("saliencegate.security.keys.os.fsync", observe_fsync)
    load_or_create_installation_key(path)

    assert publication_state_at_fsync == [False, True]


def test_concurrent_reader_waits_until_the_key_directory_is_synced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "posix":
        pytest.skip("directory fsync scheduling is POSIX-specific")
    path = tmp_path / "keys" / "installation.key"
    sync_started = Event()
    release_sync = Event()
    reader_started = Event()
    real_fsync_directory = keys_module._fsync_directory

    def pause_directory_sync(directory: Path) -> None:
        sync_started.set()
        assert release_sync.wait(timeout=2)
        real_fsync_directory(directory)

    def load_as_reader() -> InstallationKey:
        reader_started.set()
        return load_or_create_installation_key(path)

    monkeypatch.setattr(keys_module, "_fsync_directory", pause_directory_sync)
    with ThreadPoolExecutor(max_workers=2) as executor:
        creator = executor.submit(load_or_create_installation_key, path)
        assert sync_started.wait(timeout=2)
        reader = executor.submit(load_as_reader)
        assert reader_started.wait(timeout=2)
        try:
            assert not reader.done()
        finally:
            release_sync.set()

        assert creator.result(timeout=2) == reader.result(timeout=2)


def test_concurrent_key_creation_converges_on_one_key(tmp_path: Path) -> None:
    path = tmp_path / "keys" / "installation.key"
    with ThreadPoolExecutor(max_workers=8) as executor:
        keys = tuple(executor.map(lambda _: load_or_create_installation_key(path), range(32)))

    assert all(key == keys[0] for key in keys)


def test_insecure_existing_key_permissions_are_rejected(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("POSIX permission bits are not available")
    path = tmp_path / "installation.key"
    path.write_bytes(b"k" * 32)
    path.chmod(0o644)

    with pytest.raises(InsecureKeyFileError, match="owner-only"):
        load_or_create_installation_key(path)


def test_symlink_key_file_is_rejected(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("symlink protection is platform-specific")
    target = tmp_path / "target.key"
    target.write_bytes(b"k" * 32)
    target.chmod(0o600)
    link = tmp_path / "installation.key"
    link.symlink_to(target)

    with pytest.raises(InsecureKeyFileError, match="symbolic link"):
        load_or_create_installation_key(link)


def test_symlink_key_lock_is_rejected(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("symlink protection is platform-specific")
    path = tmp_path / "installation.key"
    target = tmp_path / "target.lock"
    target.touch(mode=0o600)
    path.with_name(f".{path.name}.lock").symlink_to(target)

    with pytest.raises(InsecureKeyFileError, match="lock cannot be a symbolic link"):
        load_or_create_installation_key(path)


def test_corrupt_key_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "private" / "installation.key"
    _write_private_key_fixture(path, b"too short")

    with pytest.raises(InvalidInstallationKeyError, match="32 bytes"):
        load_or_create_installation_key(path)


def test_non_regular_key_path_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "private" / "installation.key"
    if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        operations = NativeWindowsSecurityOperations()
        ensure_windows_private_directory(
            PureWindowsPath(os.fspath(path.parent)),
            operations=operations,
        )
        operations.create_private_path(
            PureWindowsPath(os.fspath(path)),
            WindowsPathKind.DIRECTORY,
        )
    else:
        path.parent.mkdir(mode=0o700, parents=True)
        path.parent.chmod(0o700)
        path.mkdir()

    with pytest.raises(InsecureKeyFileError, match="regular file"):
        load_or_create_installation_key(path)


def test_fifo_key_path_is_rejected_without_blocking(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO files are not available")
    path = tmp_path / "installation.key"
    os.mkfifo(path, mode=0o600)
    source_root = Path(__file__).parents[2] / "src"
    script = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "sys.path.insert(0, sys.argv[1])",
            "from saliencegate.security import InsecureKeyFileError",
            "from saliencegate.security import load_or_create_installation_key",
            "try:",
            "    load_or_create_installation_key(Path(sys.argv[2]))",
            "except InsecureKeyFileError:",
            "    raise SystemExit(0)",
            "raise SystemExit(1)",
        )
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(source_root), str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=2,
    )

    assert completed.returncode == 0, completed.stderr


def test_failed_key_write_removes_the_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "posix":
        pytest.skip("descriptor write failure injection is POSIX-specific")
    path = tmp_path / "keys" / "installation.key"

    def fail_write(_descriptor: int, _material: bytes) -> int:
        raise OSError("synthetic write failure")

    monkeypatch.setattr("saliencegate.security.keys.os.write", fail_write)
    with pytest.raises(OSError, match="synthetic write failure"):
        load_or_create_installation_key(path)
    assert not path.exists()
    assert tuple(path.parent.glob("*.tmp")) == ()


@pytest.mark.skipif(
    os.name != "nt",
    reason="native Win32 key security is the remote R01 gate",
)
def test_native_windows_key_directory_file_and_lock_are_owner_private(tmp_path: Path) -> None:
    path = tmp_path / "keys" / "installation.key"

    first = load_or_create_installation_key(path)
    second = load_installation_key(path)

    assert first == second
    operations = NativeWindowsSecurityOperations()
    for checked in (
        path.parent,
        path,
        path.with_name(f".{path.name}.lock"),
    ):
        security = operations.inspect_path(PureWindowsPath(os.fspath(checked)))
        assert security is not None
        assert security.owner_private_dacl is True
        assert security.hardlink_count == 1
        assert security.reparse_tag is None
