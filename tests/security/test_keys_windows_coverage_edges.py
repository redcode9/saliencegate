"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

import os
import stat
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import cast

import pytest
from tests.security.test_windows import (
    _OWNER_SID,
    _FakeWindowsOperations,
    _identity,
    _security,
)

from saliencegate.security import keys as keys_module
from saliencegate.security import windows as windows_module
from saliencegate.security.files import SecureFileError
from saliencegate.security.keys import (
    InsecureKeyFileError,
    InsecureKeyPathError,
    InstallationKey,
)
from saliencegate.security.windows import (
    WindowsPathKind,
    WindowsPathSecurity,
    WindowsSecurityError,
)


def test_default_key_path_uses_windows_appdata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WindowsPathProbe:
        def __init__(self, value: object) -> None:
            self.value = PureWindowsPath(value)

        def expanduser(self) -> WindowsPathProbe:
            return self

        def is_absolute(self) -> bool:
            return self.value.is_absolute()

        def __truediv__(self, value: str) -> WindowsPathProbe:
            return WindowsPathProbe(self.value / value)

    appdata = PureWindowsPath(r"C:\Users\fixture\AppData")
    monkeypatch.setattr(keys_module.os, "name", "nt")
    monkeypatch.setattr(keys_module, "Path", WindowsPathProbe)

    resolved = keys_module.default_installation_key_path(environ={"APPDATA": str(appdata)})
    assert resolved.value == appdata / "saliencegate" / "installation.key"  # type: ignore[attr-defined]


def test_key_file_rejects_foreign_posix_owner(tmp_path: Path) -> None:
    metadata = SimpleNamespace(
        st_mode=stat.S_IFREG | 0o600,
        st_uid=os.getuid() + 1,
    )

    with pytest.raises(InsecureKeyFileError, match="current user"):
        keys_module._validate_key_file(cast(os.stat_result, metadata), tmp_path / "key")


def test_key_file_validation_skips_posix_owner_checks_on_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_uid=-1)
    monkeypatch.setattr(keys_module.os, "name", "nt")

    keys_module._validate_key_file(cast(os.stat_result, metadata), tmp_path / "key")


def test_key_read_works_without_optional_open_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "installation.key"
    path.write_bytes(b"k" * 32)
    path.chmod(0o600)
    monkeypatch.delattr(keys_module.os, "O_NOFOLLOW")
    monkeypatch.delattr(keys_module.os, "O_NONBLOCK")

    assert keys_module._read_key(path, attempts=1) == InstallationKey(b"k" * 32)


def test_key_read_normalizes_symlink_race(monkeypatch: pytest.MonkeyPatch) -> None:
    class RacingPath:
        calls = 0

        def is_symlink(self) -> bool:
            self.calls += 1
            return self.calls > 1

    path = RacingPath()

    def fail_open(*_args: object, **_kwargs: object) -> int:
        raise OSError("open failed")

    monkeypatch.setattr(keys_module.os, "open", fail_open)

    with pytest.raises(InsecureKeyFileError, match="symbolic link"):
        keys_module._read_key(cast(Path, path), attempts=1)


def test_write_all_rejects_zero_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(keys_module.os, "write", lambda _descriptor, _data: 0)

    with pytest.raises(OSError, match="no progress"):
        keys_module._write_all(7, b"material")


def test_directory_fsync_is_a_noop_off_posix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keys_module.os, "name", "nt")

    keys_module._fsync_directory(tmp_path)


def test_directory_fsync_works_without_directory_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(keys_module.os, "O_DIRECTORY")

    keys_module._fsync_directory(tmp_path)


def test_installation_key_lock_works_without_optional_open_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(keys_module.os, "O_NOFOLLOW")
    monkeypatch.delattr(keys_module.os, "O_NONBLOCK")

    with keys_module._installation_key_lock(tmp_path / "installation.key"):
        pass


def test_load_or_create_handles_publish_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "installation.key"
    target.write_bytes(b"e" * 32)
    target.chmod(0o600)
    existing = InstallationKey(b"e" * 32)
    calls = 0

    def read(_path: Path, *, attempts: int = 50) -> InstallationKey:
        nonlocal calls
        del attempts
        calls += 1
        if calls == 1:
            raise FileNotFoundError
        return existing

    monkeypatch.setattr(keys_module, "_read_key", read)
    monkeypatch.setattr(
        keys_module, "generate_installation_key", lambda: InstallationKey(b"n" * 32)
    )

    assert keys_module._load_or_create_locked(target) == existing
    assert calls == 2


def test_load_or_create_normalizes_private_directory_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_path: Path) -> None:
        raise SecureFileError()

    monkeypatch.setattr(keys_module, "ensure_private_directory", fail)

    with pytest.raises(InsecureKeyPathError, match="directory boundary"):
        keys_module.load_or_create_installation_key(tmp_path / "keys" / "installation.key")


def test_load_existing_key_rejects_relative_path() -> None:
    with pytest.raises(InsecureKeyPathError, match="absolute"):
        keys_module.load_installation_key(Path("relative.key"))


class _DirectoryOperations(_FakeWindowsOperations):
    def __init__(
        self,
        existing: dict[PureWindowsPath, WindowsPathSecurity],
        *,
        fail_create: bool = False,
        publish_before_failure: bool = False,
    ) -> None:
        super().__init__(None)
        self.existing = existing
        self.fail_create = fail_create
        self.publish_before_failure = publish_before_failure

    def inspect_path(self, path: PureWindowsPath) -> WindowsPathSecurity | None:
        self.inspected.append(path)
        return self.existing.get(path)

    def create_private_path(self, path: PureWindowsPath, kind: WindowsPathKind) -> None:
        self.created.append((path, kind))
        if not self.fail_create or self.publish_before_failure:
            self.existing[path] = _security(
                identity=_identity(len(self.existing).to_bytes(16, "little")),
                kind=WindowsPathKind.DIRECTORY,
            )
        if self.fail_create:
            raise RuntimeError("create failed")


def _directory_fixture() -> tuple[
    PureWindowsPath, PureWindowsPath, dict[PureWindowsPath, WindowsPathSecurity]
]:
    base = PureWindowsPath(r"C:\Users\synthetic")
    target = base / "private"
    existing = {
        base: _security(
            kind=WindowsPathKind.DIRECTORY,
            owner_private_dacl=False,
        )
    }
    return base, target, existing


def test_private_directory_rejects_invalid_path() -> None:
    operations = _FakeWindowsOperations(_security(kind=WindowsPathKind.DIRECTORY))

    with pytest.raises(WindowsSecurityError):
        windows_module.ensure_windows_private_directory(
            PureWindowsPath("relative"),
            operations=operations,
        )


def test_private_directory_authorizes_existing_path() -> None:
    path = PureWindowsPath(r"C:\Users\synthetic\private")
    operations = _FakeWindowsOperations(_security(kind=WindowsPathKind.DIRECTORY))

    authorization = windows_module.ensure_windows_private_directory(
        path,
        operations=operations,
    )

    assert authorization.path == path


def test_private_directory_rejects_missing_anchor() -> None:
    operations = _FakeWindowsOperations(None)

    with pytest.raises(WindowsSecurityError):
        windows_module.ensure_windows_private_directory(
            PureWindowsPath(r"C:\Users\synthetic\private"),
            operations=operations,
        )


def test_private_directory_creates_missing_suffix() -> None:
    _base, target, existing = _directory_fixture()
    operations = _DirectoryOperations(existing)

    authorization = windows_module.ensure_windows_private_directory(
        target,
        operations=operations,
    )

    assert authorization.path == target
    assert operations.created == [(target, WindowsPathKind.DIRECTORY)]


@pytest.mark.parametrize("publish_before_failure", [False, True])
def test_private_directory_handles_create_failure(
    publish_before_failure: bool,
) -> None:
    _base, target, existing = _directory_fixture()
    operations = _DirectoryOperations(
        existing,
        fail_create=True,
        publish_before_failure=publish_before_failure,
    )

    if publish_before_failure:
        authorization = windows_module.ensure_windows_private_directory(
            target,
            operations=operations,
        )
        assert authorization.path == target
    else:
        with pytest.raises(WindowsSecurityError):
            windows_module.ensure_windows_private_directory(
                target,
                operations=operations,
            )


class _ChangingAncestors(_FakeWindowsOperations):
    def __init__(self, security: WindowsPathSecurity | None) -> None:
        super().__init__(security)
        self.captures = 0

    def inspect_ancestor_directories(
        self,
        path: PureWindowsPath,
    ) -> tuple[tuple[PureWindowsPath, WindowsPathSecurity], ...]:
        chain = list(super().inspect_ancestor_directories(path))
        self.captures += 1
        if self.captures > 1:
            ancestor, _security_snapshot = chain[-1]
            chain[-1] = (
                ancestor,
                _security(
                    identity=_identity(b"z" * 16),
                    kind=WindowsPathKind.DIRECTORY,
                    owner_private_dacl=False,
                ),
            )
        return tuple(chain)


def test_authorizer_rejects_ancestor_change_before_create() -> None:
    operations = _ChangingAncestors(None)

    with pytest.raises(WindowsSecurityError):
        windows_module.authorize_windows_private_path(
            PureWindowsPath(r"C:\Users\synthetic\new.bin"),
            kind=WindowsPathKind.FILE,
            operations=operations,
            create=True,
        )


def test_authorizer_rejects_ancestor_change_after_inspection() -> None:
    operations = _ChangingAncestors(_security())

    with pytest.raises(WindowsSecurityError):
        windows_module.authorize_windows_private_path(
            PureWindowsPath(r"C:\Users\synthetic\current.bin"),
            kind=WindowsPathKind.FILE,
            operations=operations,
        )


@pytest.mark.parametrize(
    "chain",
    [
        [],
        (),
        ((PureWindowsPath("C:\\"),),),
    ],
)
def test_ancestor_chain_rejects_invalid_container_shape(chain: object) -> None:
    path = PureWindowsPath(r"C:\Users\synthetic\current.bin")

    assert not windows_module._is_valid_ancestor_chain(
        path,
        chain,
        owner_sid=_OWNER_SID,
    )


def test_authorized_security_rejects_unknown_dacl_policy() -> None:
    with pytest.raises(windows_module._NativeWindowsError):
        windows_module._validate_authorized_security(
            _security(),
            kind=WindowsPathKind.FILE,
            owner_sid=_OWNER_SID,
            dacl_policy=object(),  # type: ignore[arg-type]
        )


def test_windows_ancestor_paths_reject_invalid_path() -> None:
    assert windows_module._windows_ancestor_paths(PureWindowsPath("relative")) == ()
