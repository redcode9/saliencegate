from __future__ import annotations

import errno
import os
import sqlite3
import stat
from dataclasses import FrozenInstanceError, replace
from pathlib import Path, PureWindowsPath

import pytest
from tests.capture.store_support import (
    CONNECTION_ID,
    INSTALLATION_KEY,
    PROJECT_DIGEST,
    authenticated_intake,
    initialized_store,
    register_connection,
)

import saliencegate.capture.spool as spool_module
import saliencegate.security.files as files_module
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.capture.locations import (
    CaptureLocationError,
    CaptureStoreLocations,
    resolve_capture_store_locations,
)
from saliencegate.capture.migrations import (
    CaptureMigrationIntegrityError,
    initialize_capture_store,
)
from saliencegate.capture.schema import CaptureIntake
from saliencegate.capture.spool import CaptureSpool, CaptureSpoolError
from saliencegate.capture.store import (
    CaptureStore,
    CaptureStoreError,
    CaptureStoreIntegrityError,
    CaptureStoreMode,
)
from saliencegate.security import load_or_create_installation_key
from saliencegate.security.windows import (
    NativeWindowsSecurityOperations,
    WindowsFileIdentity,
    WindowsPathAuthorization,
    WindowsPathKind,
    WindowsPathSecurity,
    WindowsSecurityError,
    WindowsSecurityOperations,
    authorize_windows_private_path,
)

_OWNER_SID = "S-1-5-21-1000"
_OTHER_SID = "S-1-5-21-2000"


def _locations(tmp_path: Path) -> CaptureStoreLocations:
    return resolve_capture_store_locations(
        environ={"XDG_STATE_HOME": str(tmp_path / "state")},
        home=tmp_path / "home",
        platform="posix",
    )


def _private_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    path.chmod(0o700)
    return path


def _spool_entries(locations: CaptureStoreLocations) -> tuple[Path, ...]:
    return tuple(sorted(locations.spool_directory.glob("*.capture-intake")))


_CAPTURE_TABLE_QUERIES = {
    "schema_migrations": "SELECT * FROM schema_migrations ORDER BY version",
    "connections": "SELECT * FROM connections ORDER BY connection_id",
    "capture_sessions": ("SELECT * FROM capture_sessions ORDER BY connection_id, session_id"),
    "capture_events": (
        "SELECT * FROM capture_events ORDER BY connection_id, session_id, receipt_ordinal"
    ),
    "capture_heads": "SELECT * FROM capture_heads ORDER BY connection_id, session_id",
    "capture_health": "SELECT * FROM capture_health ORDER BY marker_id",
    "feedback_labels": "SELECT * FROM feedback_labels ORDER BY label_id",
    "deleted_sessions": ("SELECT * FROM deleted_sessions ORDER BY connection_id, session_id"),
}


def _database_rows(path: Path) -> dict[str, tuple[tuple[object, ...], ...]]:
    connection = sqlite3.connect(path)
    try:
        return {
            table: tuple(connection.execute(query).fetchall())
            for table, query in _CAPTURE_TABLE_QUERIES.items()
        }
    finally:
        connection.close()


def _create_admitted_session(path: Path) -> CaptureIntake:
    initialize_capture_store(path)
    intake = authenticated_intake("session_started", producer_index=1)
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        register_connection(store)
        store.append(intake)
    return intake


def _assert_open_rejected_without_repair(
    path: Path,
    expected_rows: dict[str, tuple[tuple[object, ...], ...]],
) -> None:
    unexpectedly_opened: CaptureStore | None = None
    try:
        with pytest.raises(CaptureStoreIntegrityError):
            unexpectedly_opened = CaptureStore.open(
                path,
                installation_key=INSTALLATION_KEY,
                busy_timeout_ms=250,
                mode=CaptureStoreMode.MAINTENANCE,
            )
    finally:
        if unexpectedly_opened is not None:
            unexpectedly_opened.close()
        assert _database_rows(path) == expected_rows


class _RecordingStore:
    def __init__(self) -> None:
        self.received: list[CaptureIntake] = []

    def append(self, intake: CaptureIntake) -> object:
        self.received.append(intake)
        return object()


class _FakeWindowsOperations:
    def __init__(self, inspections: list[WindowsPathSecurity | None]) -> None:
        self._inspections = inspections
        self.inspected: list[PureWindowsPath] = []
        self.ancestor_inspected: list[PureWindowsPath] = []
        self.created: list[tuple[PureWindowsPath, WindowsPathKind]] = []

    def current_user_sid(self) -> str:
        return _OWNER_SID

    def inspect_path(self, path: PureWindowsPath) -> WindowsPathSecurity | None:
        self.inspected.append(path)
        if not self._inspections:
            raise AssertionError("unexpected Windows inspection")
        return self._inspections.pop(0)

    def inspect_ancestor_directories(
        self,
        path: PureWindowsPath,
    ) -> tuple[tuple[PureWindowsPath, WindowsPathSecurity], ...]:
        self.ancestor_inspected.append(path)
        return tuple(
            (
                ancestor,
                _windows_security(
                    kind=WindowsPathKind.DIRECTORY,
                    identity=(index + 1).to_bytes(16, "little"),
                ),
            )
            for index, ancestor in enumerate(reversed(path.parents))
        )

    def create_private_path(self, path: PureWindowsPath, kind: WindowsPathKind) -> None:
        self.created.append((path, kind))


def _windows_security(
    *,
    kind: WindowsPathKind = WindowsPathKind.FILE,
    identity: bytes = b"a" * 16,
) -> WindowsPathSecurity:
    return WindowsPathSecurity(
        identity=WindowsFileIdentity(volume_serial_number=7, file_id=identity),
        kind=kind,
        owner_sid=_OWNER_SID,
        owner_private_dacl=True,
        owner_write_protected_dacl=True,
        owner_traversal_protected_dacl=True,
        hardlink_count=1,
        reparse_tag=None,
    )


def test_posix_store_locations_use_xdg_state_and_do_not_touch_the_filesystem(
    tmp_path: Path,
) -> None:
    root = tmp_path / "xdg-state"

    locations = resolve_capture_store_locations(
        environ={"XDG_STATE_HOME": str(root), "HOME": "provider-native-secret"},
        home=tmp_path / "explicit-home",
        platform="posix",
    )

    assert locations == CaptureStoreLocations(
        platform="posix",
        state_directory=root / "saliencegate",
        database_path=root / "saliencegate" / "capture.sqlite3",
        spool_directory=root / "saliencegate" / "capture-spool",
    )
    assert not root.exists()
    assert repr(locations) == "CaptureStoreLocations(<redacted>)"
    with pytest.raises(FrozenInstanceError):
        locations.database_path = tmp_path / "replacement"  # type: ignore[misc]


def test_posix_store_locations_fall_back_to_the_explicit_home(tmp_path: Path) -> None:
    home = tmp_path / "home"

    locations = resolve_capture_store_locations(
        environ={},
        home=home,
        platform="posix",
    )

    assert locations.state_directory == home / ".local" / "state" / "saliencegate"
    assert locations.database_path == locations.state_directory / "capture.sqlite3"
    assert locations.spool_directory == locations.state_directory / "capture-spool"


def test_windows_store_locations_use_local_appdata_without_posix_path_faking(
    tmp_path: Path,
) -> None:
    local_appdata = tmp_path / "LocalAppData"

    locations = resolve_capture_store_locations(
        environ={"LOCALAPPDATA": str(local_appdata)},
        home=tmp_path / "unused-home",
        platform="windows",
    )

    assert locations.platform == "windows"
    assert locations.state_directory == local_appdata / "SalienceGate"
    assert locations.database_path == local_appdata / "SalienceGate" / "capture.sqlite3"
    assert locations.spool_directory == local_appdata / "SalienceGate" / "capture-spool"
    assert not local_appdata.exists()


@pytest.mark.parametrize(
    ("environ", "home", "platform"),
    (
        ({"XDG_STATE_HOME": "relative-state"}, Path("/safe/home"), "posix"),
        ({"XDG_STATE_HOME": "/safe/../escape"}, Path("/safe/home"), "posix"),
        ({}, Path("relative-home"), "posix"),
        ({"LOCALAPPDATA": "relative-data"}, Path("/safe/home"), "windows"),
        ({}, Path("/safe/home"), "windows"),
        ({}, Path("/safe/home"), "plan9"),
    ),
)
def test_store_location_roots_fail_closed_and_content_free(
    environ: dict[str, str],
    home: Path,
    platform: str,
) -> None:
    with pytest.raises(CaptureLocationError) as captured:
        resolve_capture_store_locations(environ=environ, home=home, platform=platform)

    assert str(captured.value) == "capture store location is invalid"
    assert "relative" not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.skipif(os.name != "posix", reason="owner-private modes require POSIX")
def test_spool_creates_owner_private_single_link_boundaries(tmp_path: Path) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    spool.enqueue(authenticated_intake("session_started", producer_index=1))
    entry = _spool_entries(locations)[0]

    for directory in (locations.state_directory, locations.spool_directory):
        metadata = directory.lstat()
        assert stat.S_ISDIR(metadata.st_mode)
        assert stat.S_IMODE(metadata.st_mode) == 0o700
        assert metadata.st_uid == os.getuid()
    metadata = entry.lstat()
    assert stat.S_ISREG(metadata.st_mode)
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_uid == os.getuid()
    assert metadata.st_nlink == 1
    assert not locations.database_path.exists()


@pytest.mark.skipif(os.name != "posix", reason="mkdirat requires POSIX")
def test_spool_child_creation_stays_bound_to_state_during_a_transient_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _locations(tmp_path)
    _private_directory(locations.state_directory.parent)
    _private_directory(locations.state_directory)
    displaced = locations.state_directory.with_name("saliencegate-displaced")
    replacement_state = _private_directory(tmp_path / "replacement-state")
    marker = replacement_state / "do-not-touch"
    marker.write_bytes(b"unchanged")
    marker.chmod(0o600)
    real_mkdir = os.mkdir
    injected = False

    def create_child_while_swapped(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal injected
        if os.fsdecode(path) == "capture-spool" and dir_fd is not None and not injected:
            injected = True
            locations.state_directory.rename(displaced)
            replacement_state.rename(locations.state_directory)
            try:
                real_mkdir(path, mode, dir_fd=dir_fd)
            finally:
                locations.state_directory.rename(replacement_state)
                displaced.rename(locations.state_directory)
            return
        real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", create_child_while_swapped)
    monkeypatch.setattr(
        os,
        "supports_dir_fd",
        os.supports_dir_fd | {create_child_while_swapped},
    )

    spool = CaptureSpool.open(locations, INSTALLATION_KEY)

    assert injected
    assert locations.spool_directory.is_dir()
    assert marker.read_bytes() == b"unchanged"
    assert not (replacement_state / "capture-spool").exists()
    assert repr(spool) == "CaptureSpool(<redacted>)"


@pytest.mark.skipif(os.name != "posix", reason="descriptor lifecycle requires POSIX")
def test_spool_open_fails_content_free_if_directory_authorization_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _locations(tmp_path)
    real_snapshot = files_module._open_private_directory_snapshot
    real_close = os.close
    failed_descriptor: int | None = None
    close_attempted = False

    def capture_state_descriptor(path: Path):
        nonlocal failed_descriptor
        descriptor, identity = real_snapshot(path)
        if path == locations.state_directory and failed_descriptor is None:
            failed_descriptor = descriptor
        return descriptor, identity

    def close_then_fail(descriptor: int) -> None:
        nonlocal close_attempted
        if descriptor == failed_descriptor:
            close_attempted = True
            real_close(descriptor)
            raise OSError(errno.EIO, "provider-native-secret-close-failure")
        real_close(descriptor)

    monkeypatch.setattr(
        files_module,
        "_open_private_directory_snapshot",
        capture_state_descriptor,
    )
    monkeypatch.setattr(os, "close", close_then_fail)

    with pytest.raises(CaptureSpoolError) as captured:
        CaptureSpool.open(locations, INSTALLATION_KEY)

    assert close_attempted
    assert failed_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(failed_descriptor)
    assert str(captured.value) == "capture spool operation failed"
    assert "provider-native-secret" not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert not locations.spool_directory.exists()


@pytest.mark.skipif(os.name != "posix", reason="symlink boundaries require POSIX")
def test_spool_rejects_a_symlink_directory_without_touching_the_target(
    tmp_path: Path,
) -> None:
    locations = _locations(tmp_path)
    locations.state_directory.mkdir(parents=True, mode=0o700)
    target = _private_directory(tmp_path / "target-spool")
    marker = target / "do-not-touch"
    marker.write_bytes(b"unchanged")
    marker.chmod(0o600)
    locations.spool_directory.symlink_to(target, target_is_directory=True)

    with pytest.raises(CaptureSpoolError):
        CaptureSpool.open(locations, INSTALLATION_KEY)

    assert locations.spool_directory.is_symlink()
    assert marker.read_bytes() == b"unchanged"


@pytest.mark.skipif(os.name != "posix", reason="owner-private modes require POSIX")
def test_spool_rejects_non_private_or_foreign_directory_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe_locations = _locations(tmp_path / "mode")
    unsafe_locations.state_directory.mkdir(parents=True, mode=0o700)
    unsafe_locations.state_directory.chmod(0o755)
    with pytest.raises(CaptureSpoolError):
        CaptureSpool.open(unsafe_locations, INSTALLATION_KEY)

    foreign_locations = _locations(tmp_path / "owner")
    foreign_locations.state_directory.mkdir(parents=True, mode=0o700)
    monkeypatch.setattr(files_module, "_current_user_id", lambda: os.getuid() + 1)
    with pytest.raises(CaptureSpoolError):
        CaptureSpool.open(foreign_locations, INSTALLATION_KEY)


@pytest.mark.skipif(os.name != "posix", reason="file alias checks require POSIX")
@pytest.mark.parametrize("mutation", ("mode", "hardlink", "symlink"))
def test_spool_drain_rejects_unsafe_entry_mutations_without_following_them(
    tmp_path: Path,
    mutation: str,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    spool.enqueue(authenticated_intake("session_started", producer_index=1))
    entry = _spool_entries(locations)[0]
    victim = locations.spool_directory / "victim"
    victim.write_bytes(b"do-not-touch")
    victim.chmod(0o600)
    if mutation == "mode":
        entry.chmod(0o644)
    elif mutation == "hardlink":
        (locations.spool_directory / "alias").hardlink_to(entry)
    else:
        displaced = locations.spool_directory / "displaced"
        entry.rename(displaced)
        entry.symlink_to(victim)
    store = _RecordingStore()

    with pytest.raises(CaptureSpoolError) as captured:
        spool.drain(store)

    assert store.received == []
    assert victim.read_bytes() == b"do-not-touch"
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.skipif(os.name != "posix", reason="directory identity checks require POSIX")
def test_open_spool_rejects_directory_replacement_before_enqueue(tmp_path: Path) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    displaced = locations.state_directory / "capture-spool-displaced"
    locations.spool_directory.rename(displaced)
    locations.spool_directory.mkdir(mode=0o700)

    with pytest.raises(CaptureSpoolError):
        spool.enqueue(authenticated_intake("session_started", producer_index=1))

    assert _spool_entries(locations) == ()
    assert tuple(displaced.glob("*.capture-intake")) == ()


@pytest.mark.skipif(os.name != "posix", reason="descriptor-relative locks require POSIX")
def test_spool_lock_never_targets_an_ancestor_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    displaced = locations.state_directory.with_name("saliencegate-displaced")
    replacement_state = _private_directory(tmp_path / "replacement-state")
    replacement_spool = _private_directory(replacement_state / "capture-spool")
    marker = replacement_spool / "do-not-touch"
    marker.write_bytes(b"unchanged")
    marker.chmod(0o600)
    real_open = os.open
    injected = False

    def replace_before_lock_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal injected
        if os.fsdecode(path).endswith(".capture-spool-lock") and not injected:
            injected = True
            locations.state_directory.rename(displaced)
            replacement_state.rename(locations.state_directory)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replace_before_lock_open)
    monkeypatch.setattr(
        os,
        "supports_dir_fd",
        os.supports_dir_fd | {replace_before_lock_open},
    )

    with pytest.raises(CaptureSpoolError):
        spool.enqueue(authenticated_intake("session_started", producer_index=1))

    assert injected
    replacement_spool_after = locations.spool_directory
    assert (replacement_spool_after / marker.name).read_bytes() == b"unchanged"
    assert not (replacement_spool_after / ".capture-spool-lock").exists()
    assert tuple(replacement_spool_after.glob("*.capture-intake")) == ()


@pytest.mark.skipif(os.name != "posix", reason="descriptor lifecycle requires POSIX")
def test_spool_locked_cleanup_attempts_both_descriptor_closes_and_fails_content_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    real_revalidate = CaptureSpool._revalidate
    real_close = os.close
    revalidation_calls = 0
    armed = False
    close_attempts: list[tuple[str, int]] = []

    def arm_after_final_revalidation(self: CaptureSpool) -> None:
        nonlocal armed, revalidation_calls
        real_revalidate(self)
        revalidation_calls += 1
        if revalidation_calls == 3:
            armed = True

    def close_then_fail(descriptor: int) -> None:
        if not armed:
            real_close(descriptor)
            return
        metadata = os.fstat(descriptor)
        spool_metadata = locations.spool_directory.stat()
        if stat.S_ISREG(metadata.st_mode):
            close_attempts.append(("lock", descriptor))
            real_close(descriptor)
            raise OSError(errno.EIO, "provider-native-secret-lock-close")
        if (metadata.st_dev, metadata.st_ino) == (
            spool_metadata.st_dev,
            spool_metadata.st_ino,
        ):
            close_attempts.append(("directory", descriptor))
            real_close(descriptor)
            raise OSError(errno.EIO, "provider-native-secret-directory-close")
        real_close(descriptor)

    monkeypatch.setattr(CaptureSpool, "_revalidate", arm_after_final_revalidation)
    monkeypatch.setattr(os, "close", close_then_fail)

    with pytest.raises(CaptureSpoolError) as captured:
        spool.health()

    assert [kind for kind, _descriptor in close_attempts] == ["lock", "directory"]
    for _kind, descriptor in close_attempts:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert str(captured.value) == "capture spool operation failed"
    assert "provider-native-secret" not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.skipif(os.name != "posix", reason="descriptor-bound publication requires POSIX")
def test_spool_publication_never_targets_an_ancestor_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    displaced = locations.state_directory.with_name("saliencegate-displaced")
    replacement_state = _private_directory(tmp_path / "replacement-state")
    replacement_spool = _private_directory(replacement_state / "capture-spool")
    marker = replacement_spool / "do-not-touch"
    marker.write_bytes(b"unchanged")
    marker.chmod(0o600)
    real_publish = files_module._publish_private_file_at_descriptor
    injected = False

    def replace_before_publication(*args: object, **kwargs: object):
        nonlocal injected
        if not injected:
            injected = True
            locations.state_directory.rename(displaced)
            replacement_state.rename(locations.state_directory)
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(
        files_module,
        "_publish_private_file_at_descriptor",
        replace_before_publication,
    )

    with pytest.raises(CaptureSpoolError):
        spool.enqueue(authenticated_intake("session_started", producer_index=1))

    assert injected
    replacement_spool_after = locations.spool_directory
    assert (replacement_spool_after / marker.name).read_bytes() == b"unchanged"
    assert tuple(replacement_spool_after.glob("*.capture-intake")) == ()


@pytest.mark.skipif(os.name != "posix", reason="descriptor-bound publication requires POSIX")
def test_spool_entry_publication_stays_on_the_pinned_leaf_during_a_transient_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    displaced = locations.state_directory.with_name("saliencegate-displaced")
    replacement_state = _private_directory(tmp_path / "replacement-state")
    replacement_spool = _private_directory(replacement_state / "capture-spool")
    marker = replacement_spool / "do-not-touch"
    marker.write_bytes(b"unchanged")
    marker.chmod(0o600)
    real_publish = files_module._publish_private_file_at_descriptor
    swapped = False

    def publish_while_swapped(*args: object, **kwargs: object):
        nonlocal swapped
        locations.state_directory.rename(displaced)
        replacement_state.rename(locations.state_directory)
        try:
            result = real_publish(*args, **kwargs)
            swapped = True
            return result
        finally:
            locations.state_directory.rename(replacement_state)
            displaced.rename(locations.state_directory)

    monkeypatch.setattr(
        files_module,
        "_publish_private_file_at_descriptor",
        publish_while_swapped,
    )

    receipt = spool.enqueue(authenticated_intake("session_started", producer_index=1))

    assert swapped
    assert receipt.disposition == "queued"
    assert len(_spool_entries(locations)) == 1
    assert marker.read_bytes() == b"unchanged"
    assert tuple(replacement_spool.glob("*.capture-intake")) == ()


@pytest.mark.skipif(os.name != "posix", reason="descriptor-bound drain requires POSIX")
def test_spool_list_and_delete_stay_on_the_pinned_leaf_during_transient_swaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    intake = authenticated_intake("session_started", producer_index=1)
    spool.enqueue(intake)
    displaced = locations.state_directory.with_name("saliencegate-displaced")
    replacement_state = _private_directory(tmp_path / "replacement-state")
    replacement_spool = _private_directory(replacement_state / "capture-spool")
    marker = replacement_spool / "do-not-touch"
    marker.write_bytes(b"unchanged")
    marker.chmod(0o600)
    real_listdir = os.listdir
    real_delete = files_module._delete_authorized_private_file_at_descriptor
    listed_while_swapped = False
    deleted_while_swapped = False

    def swap_for(call, *args: object, **kwargs: object):
        locations.state_directory.rename(displaced)
        replacement_state.rename(locations.state_directory)
        try:
            return call(*args, **kwargs)
        finally:
            locations.state_directory.rename(replacement_state)
            displaced.rename(locations.state_directory)

    def list_while_swapped(path: str | bytes | os.PathLike[str] | int):
        nonlocal listed_while_swapped
        if type(path) is int and not listed_while_swapped:
            listed_while_swapped = True
            return swap_for(real_listdir, path)
        return real_listdir(path)

    def delete_while_swapped(*args: object, **kwargs: object) -> None:
        nonlocal deleted_while_swapped
        deleted_while_swapped = True
        swap_for(real_delete, *args, **kwargs)

    monkeypatch.setattr(os, "listdir", list_while_swapped)
    monkeypatch.setattr(
        files_module,
        "_delete_authorized_private_file_at_descriptor",
        delete_while_swapped,
    )
    store = _RecordingStore()

    receipt = spool.drain(store)

    assert listed_while_swapped
    assert deleted_while_swapped
    assert receipt.admitted_events == 1
    assert receipt.remaining_events == 0
    assert store.received == [intake]
    assert _spool_entries(locations) == ()
    assert marker.read_bytes() == b"unchanged"


@pytest.mark.skipif(os.name != "posix", reason="descriptor-bound health requires POSIX")
def test_spool_health_update_and_read_stay_on_the_pinned_leaf_during_transient_swaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    displaced = locations.state_directory.with_name("saliencegate-displaced")
    replacement_state = _private_directory(tmp_path / "replacement-state")
    replacement_spool = _private_directory(replacement_state / "capture-spool")
    marker = replacement_spool / "do-not-touch"
    marker.write_bytes(b"unchanged")
    marker.chmod(0o600)
    real_publish = files_module._publish_private_file_at_descriptor
    real_read = files_module._read_private_file_at_descriptor
    did_publish_while_swapped = False
    did_read_while_swapped = False
    inside_swap = False

    def swap_for(call, *args: object, **kwargs: object):
        nonlocal inside_swap
        assert not inside_swap
        inside_swap = True
        locations.state_directory.rename(displaced)
        replacement_state.rename(locations.state_directory)
        try:
            return call(*args, **kwargs)
        finally:
            locations.state_directory.rename(replacement_state)
            displaced.rename(locations.state_directory)
            inside_swap = False

    def publish_while_swapped(*args: object, **kwargs: object):
        nonlocal did_publish_while_swapped
        did_publish_while_swapped = True
        return swap_for(real_publish, *args, **kwargs)

    def read_while_swapped(*args: object, **kwargs: object):
        nonlocal did_read_while_swapped
        name = args[2]
        if name == ".capture-spool-health" and not inside_swap:
            did_read_while_swapped = True
            return swap_for(real_read, *args, **kwargs)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(spool_module, "MAX_CAPTURE_SPOOL_EVENTS", 0)
    monkeypatch.setattr(
        files_module,
        "_publish_private_file_at_descriptor",
        publish_while_swapped,
    )
    monkeypatch.setattr(
        files_module,
        "_read_private_file_at_descriptor",
        read_while_swapped,
    )

    receipt = spool.enqueue(authenticated_intake("session_started", producer_index=1))
    health = spool.health()

    assert did_publish_while_swapped
    assert did_read_while_swapped
    assert receipt.disposition == "dropped_quota"
    assert health.dropped_events == 1
    assert health.coverage_degraded
    assert marker.read_bytes() == b"unchanged"
    assert not (replacement_spool / ".capture-spool-health").exists()


@pytest.mark.skipif(os.name != "posix", reason="SQLite boundary aliases require POSIX")
@pytest.mark.parametrize("suffix", ("", "-wal", "-shm", "-journal"))
@pytest.mark.parametrize("mutation", ("mode", "hardlink", "symlink"))
def test_live_store_rejects_database_and_sidecar_mutations_before_sql(
    tmp_path: Path,
    suffix: str,
    mutation: str,
) -> None:
    locations = _locations(tmp_path)
    locations.state_directory.mkdir(parents=True, mode=0o700)
    initialize_capture_store(locations.database_path)
    store = CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    )
    register_connection(store)
    target = Path(f"{locations.database_path}{suffix}")
    assert target.exists()
    victim = locations.state_directory / f"victim-{mutation}-{suffix.removeprefix('-')}"
    victim.write_bytes(b"do-not-touch")
    victim.chmod(0o600)
    displaced = target.with_name(f"{target.name}.displaced")
    alias = target.with_name(f"{target.name}.alias")
    try:
        if mutation == "mode":
            target.chmod(0o644)
        elif mutation == "hardlink":
            alias.hardlink_to(target)
        else:
            target.rename(displaced)
            target.symlink_to(victim)

        with pytest.raises(CaptureStoreIntegrityError):
            store.append(authenticated_intake("session_started", producer_index=1))
        assert victim.read_bytes() == b"do-not-touch"
    finally:
        if mutation == "mode":
            target.chmod(0o600)
        elif mutation == "hardlink":
            alias.unlink()
        else:
            target.unlink()
            displaced.rename(target)
        store.close()

    connection = sqlite3.connect(locations.database_path)
    try:
        assert connection.execute("SELECT count(*) FROM capture_sessions").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM capture_events").fetchone() == (0,)
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("column_index", "statement", "replacement"),
    (
        (
            3,
            "UPDATE capture_events SET producer_event_digest = ? WHERE receipt_ordinal = 1",
            "f" * 64,
        ),
        (
            4,
            "UPDATE capture_events SET event_kind = ? WHERE receipt_ordinal = 1",
            "turn_finished",
        ),
        (
            8,
            "UPDATE capture_events SET admission_source = ? WHERE receipt_ordinal = 1",
            "spool_drain",
        ),
        (
            9,
            "UPDATE capture_events SET admitted_at = ? WHERE receipt_ordinal = 1",
            "2000-01-01T00:00:00Z",
        ),
    ),
    ids=("producer-event-digest", "event-kind", "admission-source", "admitted-at"),
)
def test_open_rejects_external_event_metadata_tampering_without_repair(
    tmp_path: Path,
    column_index: int,
    statement: str,
    replacement: object,
) -> None:
    path = tmp_path / f"event-metadata-{column_index}.sqlite3"
    _create_admitted_session(path)
    original_rows = _database_rows(path)
    original_event = original_rows["capture_events"][0]
    assert original_event[column_index] != replacement

    connection = sqlite3.connect(path)
    try:
        result = connection.execute(statement, (replacement,))
        assert result.rowcount == 1
        connection.commit()
    finally:
        connection.close()

    tampered_rows = _database_rows(path)
    expected_event = (
        *original_event[:column_index],
        replacement,
        *original_event[column_index + 1 :],
    )
    assert tampered_rows["capture_events"] == (expected_event,)
    assert {table: rows for table, rows in tampered_rows.items() if table != "capture_events"} == {
        table: rows for table, rows in original_rows.items() if table != "capture_events"
    }

    _assert_open_rejected_without_repair(path, tampered_rows)


@pytest.mark.parametrize(
    ("column", "replacement"),
    (
        ("receipt_count", 2),
        ("head_event_tag", "0" * 64),
        ("head_tag", "0" * 64),
    ),
)
def test_open_rejects_capture_head_tampering_without_repair(
    tmp_path: Path,
    column: str,
    replacement: object,
) -> None:
    path = tmp_path / f"head-{column}.sqlite3"
    _create_admitted_session(path)
    connection = sqlite3.connect(path)
    try:
        result = connection.execute(
            f"UPDATE capture_heads SET {column} = ?",
            (replacement,),
        )
        assert result.rowcount == 1
        connection.commit()
    finally:
        connection.close()

    tampered_rows = _database_rows(path)
    _assert_open_rejected_without_repair(path, tampered_rows)


def test_open_rejects_capture_session_row_tag_tampering_without_repair(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session-row-tag.sqlite3"
    _create_admitted_session(path)
    connection = sqlite3.connect(path)
    try:
        result = connection.execute("UPDATE capture_sessions SET row_tag = ?", ("0" * 64,))
        assert result.rowcount == 1
        connection.commit()
    finally:
        connection.close()

    tampered_rows = _database_rows(path)
    assert tampered_rows["capture_sessions"][0][-3] == "0" * 64
    _assert_open_rejected_without_repair(path, tampered_rows)


@pytest.mark.parametrize("access", ("reopen", "live"))
def test_deleted_authenticated_health_marker_is_rejected_without_repair(
    tmp_path: Path,
    access: str,
) -> None:
    path = tmp_path / f"deleted-health-marker-{access}.sqlite3"
    initialize_capture_store(path)
    original = authenticated_intake("session_started", producer_index=1)
    collision = authenticated_intake(
        "turn_finished",
        producer_index=2,
        changes={"producer_event_digest": original.producer_event_digest},
    )
    store = CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    )
    try:
        register_connection(store)
        store.append(original)
        store.append(collision)
        intact_rows = _database_rows(path)
        assert len(intact_rows["capture_health"]) == 1
        connection = sqlite3.connect(path)
        try:
            assert connection.execute(
                """
                SELECT health_marker_count FROM capture_sessions
                WHERE connection_id = ? AND session_id = ?
                """,
                (CONNECTION_ID, original.session_id),
            ).fetchone() == (1,)
        finally:
            connection.close()

        if access == "reopen":
            store.close()
        connection = sqlite3.connect(path)
        try:
            result = connection.execute(
                """
                DELETE FROM capture_health
                WHERE connection_id = ? AND session_id = ?
                """,
                (CONNECTION_ID, original.session_id),
            )
            assert result.rowcount == 1
            connection.commit()
        finally:
            connection.close()
        tampered_rows = _database_rows(path)
        assert tampered_rows["capture_health"] == ()
        assert tampered_rows["capture_sessions"] == intact_rows["capture_sessions"]

        if access == "reopen":
            _assert_open_rejected_without_repair(path, tampered_rows)
        else:
            with pytest.raises(CaptureStoreIntegrityError):
                store.verify_session(CONNECTION_ID, original.session_id)
            assert _database_rows(path) == tampered_rows
    finally:
        store.close()


@pytest.mark.parametrize("mutation", ("missing-head", "missing-interior-event", "orphan-head"))
def test_open_rejects_missing_or_orphaned_chain_rows_without_repair(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / f"chain-shape-{mutation}.sqlite3"
    initialize_capture_store(path)
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        register_connection(store)
        start = authenticated_intake("session_started", producer_index=1)
        store.append(start)
        if mutation == "missing-interior-event":
            store.append(authenticated_intake("turn_finished", producer_index=2))
            store.append(authenticated_intake("turn_finished", producer_index=3))

    connection = sqlite3.connect(path)
    try:
        if mutation == "missing-head":
            connection.execute("DELETE FROM capture_heads")
        elif mutation == "missing-interior-event":
            connection.execute("DELETE FROM capture_events WHERE receipt_ordinal = 2")
        else:
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.execute(
                """
                INSERT INTO capture_heads(
                    connection_id, session_id, receipt_count,
                    head_event_tag, head_tag
                ) VALUES (?, ?, 0, NULL, ?)
                """,
                (CONNECTION_ID, "f" * 64, "0" * 64),
            )
        connection.commit()
    finally:
        connection.close()

    tampered_rows = _database_rows(path)
    with pytest.raises((CaptureStoreIntegrityError, CaptureMigrationIntegrityError)):
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            busy_timeout_ms=250,
            mode=CaptureStoreMode.MAINTENANCE,
        )
    assert _database_rows(path) == tampered_rows


@pytest.mark.skipif(os.name != "posix", reason="owner-private modes require POSIX")
def test_close_rejects_an_unsafe_database_boundary_after_closing_without_repair(
    tmp_path: Path,
) -> None:
    locations = _locations(tmp_path)
    locations.state_directory.mkdir(parents=True, mode=0o700)
    initialize_capture_store(locations.database_path)
    store = CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    )
    try:
        locations.database_path.chmod(0o644)
        with pytest.raises(CaptureStoreError) as captured:
            store.close()

        assert repr(store) == "CaptureStore(closed=True)"
        assert stat.S_IMODE(locations.database_path.lstat().st_mode) == 0o644
        assert str(captured.value) == "capture store operation failed"
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None
    finally:
        locations.database_path.chmod(0o600)
        store.close()


@pytest.mark.parametrize("attempt", ("identical-replay", "digest-collision"))
def test_live_store_rejects_event_tag_tampering_before_replay_or_collision(
    tmp_path: Path,
    attempt: str,
) -> None:
    path = tmp_path / f"live-event-tag-{attempt}.sqlite3"
    initialize_capture_store(path)
    admitted = authenticated_intake("session_started", producer_index=1)
    candidate = admitted
    if attempt == "digest-collision":
        candidate = authenticated_intake(
            "turn_finished",
            producer_index=2,
            changes={"producer_event_digest": admitted.producer_event_digest},
        )

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        register_connection(store)
        store.append(admitted)
        connection = sqlite3.connect(path)
        try:
            result = connection.execute(
                "UPDATE capture_events SET event_tag = ? WHERE receipt_ordinal = 1",
                ("0" * 64,),
            )
            assert result.rowcount == 1
            connection.commit()
        finally:
            connection.close()
        tampered_rows = _database_rows(path)
        assert tampered_rows["capture_events"][0][7] == "0" * 64

        try:
            with pytest.raises(CaptureStoreIntegrityError):
                store.append(candidate)
        finally:
            assert _database_rows(path) == tampered_rows


def test_hook_replay_rejects_an_authenticated_event_unreachable_from_the_head(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unreachable-replay.sqlite3"
    fork_path = tmp_path / "unreachable-replay-fork.sqlite3"
    start = authenticated_intake("session_started", producer_index=1)
    current = authenticated_intake("action_started", producer_index=2)
    tail = authenticated_intake("turn_finished", producer_index=3)
    alternate = authenticated_intake("controller_failed", producer_index=4)
    fresh = authenticated_intake("turn_finished", producer_index=5)

    with initialized_store(path) as store:
        register_connection(store)
        store.append(start)
    fork_path.write_bytes(path.read_bytes())
    fork_path.chmod(0o600)

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        store.append(current)
        store.append(tail)
    with CaptureStore.open(
        fork_path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.HOOK,
    ) as fork:
        fork.append(alternate)

    fork_connection = sqlite3.connect(fork_path)
    try:
        alternate_row = fork_connection.execute(
            """
            SELECT producer_event_digest, event_kind, event_json,
                   previous_event_tag, event_tag, admission_source, admitted_at
            FROM capture_events
            WHERE connection_id = ? AND session_id = ? AND receipt_ordinal = 2
            """,
            (CONNECTION_ID, start.session_id),
        ).fetchone()
    finally:
        fork_connection.close()
    assert alternate_row is not None

    connection = sqlite3.connect(path)
    try:
        updated = connection.execute(
            """
            UPDATE capture_events
            SET producer_event_digest = ?, event_kind = ?, event_json = ?,
                previous_event_tag = ?, event_tag = ?, admission_source = ?, admitted_at = ?
            WHERE connection_id = ? AND session_id = ? AND receipt_ordinal = 2
            """,
            (*alternate_row, CONNECTION_ID, start.session_id),
        )
        assert updated.rowcount == 1
        connection.commit()
    finally:
        connection.close()

    tampered_rows = _database_rows(path)
    with (
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            busy_timeout_ms=250,
            mode=CaptureStoreMode.HOOK,
        ) as store,
        pytest.raises(CaptureStoreIntegrityError),
    ):
        store.append(alternate)
    assert _database_rows(path) == tampered_rows
    with (
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            busy_timeout_ms=250,
            mode=CaptureStoreMode.HOOK,
        ) as store,
        pytest.raises(CaptureStoreIntegrityError),
    ):
        store.append(fresh)
    assert _database_rows(path) == tampered_rows
    _assert_open_rejected_without_repair(path, tampered_rows)


def test_hook_replay_rejects_an_authenticated_event_deleted_from_the_chain(
    tmp_path: Path,
) -> None:
    path = tmp_path / "deleted-replay.sqlite3"
    start = authenticated_intake("session_started", producer_index=1)
    deleted = authenticated_intake("action_started", producer_index=2)
    tail = authenticated_intake("turn_finished", producer_index=3)

    with initialized_store(path) as store:
        register_connection(store)
        store.append(start)
        store.append(deleted)
        store.append(tail)

    connection = sqlite3.connect(path)
    try:
        removed = connection.execute(
            """
            DELETE FROM capture_events
            WHERE connection_id = ? AND session_id = ? AND receipt_ordinal = 2
            """,
            (CONNECTION_ID, start.session_id),
        )
        assert removed.rowcount == 1
        connection.commit()
    finally:
        connection.close()

    tampered_rows = _database_rows(path)
    with (
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            busy_timeout_ms=250,
            mode=CaptureStoreMode.HOOK,
        ) as store,
        pytest.raises(CaptureStoreIntegrityError),
    ):
        store.append(deleted)
    assert _database_rows(path) == tampered_rows
    _assert_open_rejected_without_repair(path, tampered_rows)


def test_hook_append_reaudits_a_cached_chain_after_a_peer_commit(tmp_path: Path) -> None:
    path = tmp_path / "peer-tamper-after-cache.sqlite3"
    start = authenticated_intake("session_started", producer_index=1)
    current = authenticated_intake("action_started", producer_index=2)
    tail = authenticated_intake("turn_finished", producer_index=3)
    fresh = authenticated_intake("turn_finished", producer_index=4)

    with initialized_store(path) as maintenance:
        register_connection(maintenance)

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        store.append(start)
        store.append(current)
        store.append(tail)

        connection = sqlite3.connect(path)
        try:
            updated = connection.execute(
                """
                UPDATE capture_events SET event_tag = ?
                WHERE connection_id = ? AND session_id = ? AND receipt_ordinal = 2
                """,
                ("0" * 64, CONNECTION_ID, start.session_id),
            )
            assert updated.rowcount == 1
            connection.commit()
        finally:
            connection.close()

        tampered_rows = _database_rows(path)
        with pytest.raises(CaptureStoreIntegrityError):
            store.append(fresh)
        assert _database_rows(path) == tampered_rows

    _assert_open_rejected_without_repair(path, tampered_rows)


@pytest.mark.parametrize("forged_row", ("deleted-session", "feedback-label"))
def test_open_rejects_forged_auxiliary_integrity_tags_without_repair(
    tmp_path: Path,
    forged_row: str,
) -> None:
    path = tmp_path / f"forged-{forged_row}.sqlite3"
    intake = _create_admitted_session(path)
    connection = sqlite3.connect(path)
    try:
        if forged_row == "deleted-session":
            connection.execute(
                """
                INSERT INTO deleted_sessions(
                    connection_id, session_id, project_digest,
                    deleted_at, tombstone_tag
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    CONNECTION_ID,
                    intake.session_id,
                    PROJECT_DIGEST,
                    "2000-01-01T00:00:00Z",
                    "0" * 64,
                ),
            )
        else:
            connection.execute(
                """
                INSERT INTO feedback_labels(
                    label_id, connection_id, session_id,
                    label, created_at, row_tag
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "f" * 64,
                    CONNECTION_ID,
                    intake.session_id,
                    "memory-needed",
                    "2000-01-01T00:00:00Z",
                    "0" * 64,
                ),
            )
        connection.commit()
    finally:
        connection.close()

    forged_rows = _database_rows(path)
    expected_table = "deleted_sessions" if forged_row == "deleted-session" else "feedback_labels"
    assert len(forged_rows[expected_table]) == 1
    _assert_open_rejected_without_repair(path, forged_rows)


def test_validly_authenticated_database_rollback_is_explicitly_outside_the_threat_model(
    tmp_path: Path,
) -> None:
    locations = _locations(tmp_path)
    locations.state_directory.mkdir(parents=True, mode=0o700)
    initialize_capture_store(locations.database_path)
    first = authenticated_intake("session_started", producer_index=1)
    second = authenticated_intake("turn_finished", producer_index=2)
    with CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        register_connection(store)
        store.append(first)
    valid_older_copy = locations.database_path.read_bytes()

    with CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        store.append(second)
        assert store.verify_session(CONNECTION_ID, first.session_id).event_count == 2

    locations.database_path.write_bytes(valid_older_copy)
    locations.database_path.chmod(0o600)
    with CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        accepted_rollback = store.verify_session(CONNECTION_ID, first.session_id)

    assert accepted_rollback.event_count == 1
    assert accepted_rollback.last_receipt_ordinal == 1


def test_windows_operations_contract_creates_and_revalidates_exact_identity() -> None:
    path = PureWindowsPath(r"C:\Users\synthetic\capture-spool")
    safe = _windows_security(kind=WindowsPathKind.DIRECTORY)
    operations = _FakeWindowsOperations([None, safe, safe])

    assert isinstance(operations, WindowsSecurityOperations)
    authorization = authorize_windows_private_path(
        path,
        kind=WindowsPathKind.DIRECTORY,
        operations=operations,
        create=True,
    )
    authorization.revalidate()

    assert isinstance(authorization, WindowsPathAuthorization)
    assert operations.created == [(path, WindowsPathKind.DIRECTORY)]
    assert operations.inspected == [path, path, path]
    assert repr(authorization) == "WindowsPathAuthorization(<redacted>)"


@pytest.mark.parametrize(
    "unsafe",
    (
        replace(_windows_security(), owner_sid=_OTHER_SID),
        replace(_windows_security(), owner_private_dacl=False),
        replace(_windows_security(), hardlink_count=2),
        replace(_windows_security(), reparse_tag=0xA000000C),
        replace(_windows_security(), kind=WindowsPathKind.DIRECTORY),
    ),
)
def test_windows_contract_rejects_owner_dacl_hardlink_reparse_and_kind(
    unsafe: WindowsPathSecurity,
) -> None:
    path = PureWindowsPath(r"C:\Users\synthetic\capture.sqlite3")
    operations = _FakeWindowsOperations([unsafe])

    with pytest.raises(WindowsSecurityError) as captured:
        authorize_windows_private_path(
            path,
            kind=WindowsPathKind.FILE,
            operations=operations,
        )

    assert str(captured.value) == "Windows private path authorization failed"
    assert "synthetic" not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_windows_contract_revalidation_rejects_identity_races() -> None:
    path = PureWindowsPath(r"C:\Users\synthetic\capture.sqlite3")
    original = _windows_security(identity=b"a" * 16)
    replacement = _windows_security(identity=b"b" * 16)
    operations = _FakeWindowsOperations([original, replacement])
    authorization = authorize_windows_private_path(
        path,
        kind=WindowsPathKind.FILE,
        operations=operations,
    )

    with pytest.raises(WindowsSecurityError):
        authorization.revalidate()


@pytest.mark.skipif(
    os.name != "nt",
    reason="native Win32 security verification is the remote R01 gate",
)
def test_native_windows_operations_authorize_a_real_private_directory(tmp_path: Path) -> None:
    path = PureWindowsPath(str(tmp_path / "native-private"))
    operations = NativeWindowsSecurityOperations()

    authorization = authorize_windows_private_path(
        path,
        kind=WindowsPathKind.DIRECTORY,
        operations=operations,
        create=True,
    )

    authorization.revalidate()


def test_provider_sentinel_never_reaches_key_database_sidecars_or_spool(
    tmp_path: Path,
) -> None:
    sentinel = b"SG_PROVIDER_NATIVE_SENTINEL_7c26415e"
    key_directory = tmp_path / "configuration" / "saliencegate"
    key_directory.mkdir(parents=True, mode=0o700)
    key_path = key_directory / "installation.key"
    key_path.write_bytes(b"k" * 32)
    key_path.chmod(0o600)
    installation_key = load_or_create_installation_key(key_path)
    context = CaptureDigestContext(installation_key)
    locations = resolve_capture_store_locations(
        environ={
            "XDG_STATE_HOME": str(tmp_path / "state"),
            "OPENAI_API_KEY": sentinel.decode(),
            "ANTHROPIC_API_KEY": sentinel.decode(),
        },
        home=tmp_path / "home",
        platform="posix",
    )
    spool = CaptureSpool.open(locations, installation_key)
    initialize_capture_store(locations.database_path)
    admitted = authenticated_intake(
        "session_started",
        session_native=sentinel,
        producer_index=1,
        context=context,
    )
    queued = authenticated_intake(
        "session_finished",
        session_native=sentinel,
        producer_index=2,
        context=context,
    )

    with CaptureStore.open(
        locations.database_path,
        installation_key=installation_key,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        register_connection(store, connection_id=CONNECTION_ID)
        store.append(admitted)
        spool.enqueue(queued)
        files = (
            key_path,
            *(path for path in locations.state_directory.rglob("*") if path.is_file()),
        )
        names = {path.name for path in files}
        assert "installation.key" in names
        assert "capture.sqlite3" in names
        assert "capture.sqlite3-wal" in names
        assert "capture.sqlite3-shm" in names
        assert any(path.name.endswith(".capture-intake") for path in files)
        assert all(sentinel not in path.read_bytes() for path in files)

    files_after_close = (
        key_path,
        *(path for path in locations.state_directory.rglob("*") if path.is_file()),
    )
    assert all(sentinel not in path.read_bytes() for path in files_after_close)
