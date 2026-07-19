from __future__ import annotations

import errno
import os
import sqlite3
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import saliencegate.security.files as files_module
from saliencegate.security import (
    AtomicFilePublication,
    SecureFileBoundError,
    SecureFileError,
    SecureFileUnsupportedError,
    StableFileAuthorization,
    StableFileRead,
    StableReadPolicy,
    authorize_atomic_file_publication,
    authorize_private_sqlite_path,
    inspect_private_file_location,
    read_stable_file,
)


def _private_directory(path: Path) -> Path:
    path.mkdir()
    path.chmod(0o700)
    return path


def _private_file(path: Path, data: bytes = b"") -> Path:
    path.write_bytes(data)
    path.chmod(0o600)
    return path


def _sqlite_sidecars(path: Path) -> tuple[Path, Path]:
    return (Path(f"{path}-wal"), Path(f"{path}-shm"))


def _sqlite_journal(path: Path) -> Path:
    return Path(f"{path}-journal")


def _add_macos_acl(path: Path, rule: str) -> None:
    subprocess.run(
        ["chmod", "+a", rule, str(path)],
        check=True,
        capture_output=True,
        timeout=10,
    )


def _clear_macos_acl(path: Path) -> None:
    subprocess.run(
        ["chmod", "-N", str(path)],
        check=True,
        capture_output=True,
        timeout=10,
    )


def _assert_sanitized(error: SecureFileError, secret: str) -> None:
    assert str(error) == "secure file authorization failed"
    assert secret not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.skipif(os.name != "posix", reason="private file modes require POSIX")
def test_missing_sqlite_file_is_created_privately_and_can_be_revalidated(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow.sqlite3"

    authorization = authorize_private_sqlite_path(target)

    metadata = target.lstat()
    assert isinstance(authorization, StableFileAuthorization)
    assert authorization.path == os.path.abspath(target)
    assert target.read_bytes() == b""
    assert stat.S_ISREG(metadata.st_mode)
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_nlink == 1
    assert metadata.st_uid == os.getuid()
    for sidecar in (*_sqlite_sidecars(target), _sqlite_journal(target)):
        sidecar_metadata = sidecar.lstat()
        assert sidecar.read_bytes() == b""
        assert stat.S_ISREG(sidecar_metadata.st_mode)
        assert stat.S_IMODE(sidecar_metadata.st_mode) == 0o600
        assert sidecar_metadata.st_nlink == 1
        assert sidecar_metadata.st_uid == os.getuid()
    authorization.revalidate()


@pytest.mark.skipif(os.name != "posix", reason="private SQLite sidecars require POSIX")
@pytest.mark.parametrize("suffix", ("-wal", "-shm", "-journal"))
@pytest.mark.parametrize("attack", ("symlink", "hardlink"))
def test_unsafe_sqlite_sidecar_is_rejected_without_mutating_its_victim(
    tmp_path: Path,
    suffix: str,
    attack: str,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_file(parent / "shadow.sqlite3", b"database")
    victim = _private_file(parent / "victim", b"do-not-touch")
    sidecar = Path(f"{target}{suffix}")
    if attack == "symlink":
        sidecar.symlink_to(victim)
    else:
        sidecar.hardlink_to(victim)

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)

    assert victim.read_bytes() == b"do-not-touch"
    assert target.read_bytes() == b"database"
    if attack == "symlink":
        assert sidecar.is_symlink()
    else:
        assert sidecar.samefile(victim)


@pytest.mark.skipif(os.name != "posix", reason="private SQLite sidecars require POSIX")
def test_partial_sidecar_authorization_cleans_only_the_placeholder_it_created(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_file(parent / "shadow.sqlite3", b"database")
    wal, shm = _sqlite_sidecars(target)
    victim = _private_file(parent / "victim", b"do-not-touch")
    shm.symlink_to(victim)

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)

    assert not wal.exists()
    assert shm.is_symlink()
    assert victim.read_bytes() == b"do-not-touch"
    assert target.read_bytes() == b"database"


@pytest.mark.skipif(os.name != "posix", reason="private SQLite sidecars require POSIX")
def test_failed_sidecar_authorization_never_removes_a_preexisting_safe_sidecar(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_file(parent / "shadow.sqlite3", b"database")
    wal, shm = _sqlite_sidecars(target)
    _private_file(wal, b"existing-wal")
    victim = _private_file(parent / "victim", b"do-not-touch")
    shm.hardlink_to(victim)

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)

    assert wal.read_bytes() == b"existing-wal"
    assert shm.samefile(victim)
    assert victim.read_bytes() == b"do-not-touch"


@pytest.mark.skipif(os.name != "posix", reason="private SQLite sidecars require POSIX")
@pytest.mark.parametrize("suffix", ("-wal", "-shm"))
@pytest.mark.parametrize("mutation", ("missing", "mode", "hardlink", "replacement"))
def test_revalidation_rejects_every_sidecar_identity_or_security_regression(
    tmp_path: Path,
    suffix: str,
    mutation: str,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow.sqlite3"
    authorization = authorize_private_sqlite_path(target)
    sidecar = Path(f"{target}{suffix}")
    if mutation == "missing":
        sidecar.unlink()
    elif mutation == "mode":
        sidecar.chmod(0o644)
    elif mutation == "hardlink":
        (parent / "sidecar-alias").hardlink_to(sidecar)
    else:
        displaced = parent / "sidecar-displaced"
        sidecar.rename(displaced)
        _private_file(sidecar)

    with pytest.raises(SecureFileError):
        authorization.revalidate()


@pytest.mark.skipif(os.name != "posix", reason="private SQLite journals require POSIX")
@pytest.mark.parametrize("mutation", ("mode", "hardlink", "symlink"))
def test_revalidation_allows_a_missing_transient_journal_but_rejects_unsafe_names(
    tmp_path: Path,
    mutation: str,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow.sqlite3"
    authorization = authorize_private_sqlite_path(target)
    journal = _sqlite_journal(target)
    journal.unlink()
    authorization.revalidate()
    victim = _private_file(parent / "journal-victim", b"do-not-touch")
    if mutation == "mode":
        _private_file(journal)
        journal.chmod(0o644)
    elif mutation == "hardlink":
        journal.hardlink_to(victim)
    else:
        journal.symlink_to(victim)

    with pytest.raises(SecureFileError):
        authorization.revalidate()

    assert victim.read_bytes() == b"do-not-touch"


@pytest.mark.skipif(os.name != "posix", reason="private SQLite journals require POSIX")
def test_pre_statement_revalidation_pins_the_transient_journal_identity(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow.sqlite3"
    authorization = authorize_private_sqlite_path(target)
    journal = _sqlite_journal(target)
    displaced = parent / "journal-displaced"
    journal.rename(displaced)
    _private_file(journal, b"safe-looking-replacement")

    # Once SQLite has started configuration, replacing or removing its rollback
    # journal is a valid lifecycle transition as long as the new name is safe.
    authorization.revalidate()
    with pytest.raises(SecureFileError):
        authorization._revalidate_before_sqlite_statements()

    assert journal.read_bytes() == b"safe-looking-replacement"
    assert displaced.read_bytes() == b""


@pytest.mark.skipif(os.name != "posix", reason="private SQLite sidecars require POSIX")
def test_sidecar_cleanup_is_identity_checked_and_never_removes_preexisting_files(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_file(parent / "shadow.sqlite3")
    wal, shm = _sqlite_sidecars(target)
    _private_file(shm, b"preexisting")
    authorization = authorize_private_sqlite_path(target)
    displaced = parent / "created-wal-displaced"
    wal.rename(displaced)
    _private_file(wal, b"replacement")

    authorization._cleanup_created_sqlite_sidecars()

    assert wal.read_bytes() == b"replacement"
    assert displaced.read_bytes() == b""
    assert shm.read_bytes() == b"preexisting"


@pytest.mark.skipif(os.name != "posix", reason="private SQLite sidecars require POSIX")
def test_sidecar_cleanup_does_nothing_after_the_database_name_is_replaced(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow.sqlite3"
    authorization = authorize_private_sqlite_path(target)
    wal, shm = _sqlite_sidecars(target)
    displaced = parent / "database-displaced"
    target.rename(displaced)
    _private_file(target, b"replacement")

    authorization._cleanup_created_sqlite_sidecars()

    assert target.read_bytes() == b"replacement"
    assert wal.read_bytes() == b""
    assert shm.read_bytes() == b""


@pytest.mark.skipif(os.name != "posix", reason="private SQLite sidecars require POSIX")
def test_sidecar_cleanup_removes_only_unchanged_placeholders_and_is_idempotent(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_file(parent / "shadow.sqlite3")
    wal, shm = _sqlite_sidecars(target)
    _private_file(shm, b"preexisting")
    authorization = authorize_private_sqlite_path(target)

    authorization._cleanup_created_sqlite_sidecars()
    authorization._cleanup_created_sqlite_sidecars()

    assert not wal.exists()
    assert shm.read_bytes() == b"preexisting"


@pytest.mark.skipif(os.name != "posix", reason="SQLite WAL requires POSIX file modes")
def test_sidecar_cleanup_never_unlinks_placeholders_claimed_by_a_live_peer(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow.sqlite3"
    authorization = authorize_private_sqlite_path(target)
    wal, shm = _sqlite_sidecars(target)
    first = sqlite3.connect(target, isolation_level=None)
    second: sqlite3.Connection | None = None
    try:
        assert first.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        first.execute("CREATE TABLE durable(value INTEGER NOT NULL)")
        second = sqlite3.connect(target, isolation_level=None)
        identities = {sidecar.name: sidecar.lstat().st_ino for sidecar in (wal, shm)}
        first.close()

        authorization._cleanup_created_sqlite_sidecars()

        assert {sidecar.name: sidecar.lstat().st_ino for sidecar in (wal, shm)} == identities
        second.execute("INSERT INTO durable VALUES (42)")
        assert second.execute("SELECT value FROM durable").fetchall() == [(42,)]
    finally:
        with suppress(sqlite3.Error):
            first.close()
        if second is not None:
            second.close()


@pytest.mark.skipif(os.name != "posix", reason="SQLite WAL requires POSIX file modes")
def test_authorized_placeholders_keep_their_inodes_when_sqlite_claims_wal(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow.sqlite3"
    authorization = authorize_private_sqlite_path(target)
    wal, shm = _sqlite_sidecars(target)
    identities = {sidecar.name: sidecar.lstat().st_ino for sidecar in (wal, shm)}

    connection = sqlite3.connect(target, isolation_level=None)
    try:
        authorization.revalidate()
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        connection.execute("CREATE TABLE durable(value INTEGER NOT NULL)")
        connection.execute("INSERT INTO durable VALUES (42)")
        authorization.revalidate()
        assert {sidecar.name: sidecar.lstat().st_ino for sidecar in (wal, shm)} == identities
    finally:
        connection.close()
    authorization._cleanup_created_sqlite_sidecars()

    reopened = authorize_private_sqlite_path(target)
    second = sqlite3.connect(reopened.path, isolation_level=None)
    try:
        reopened.revalidate()
        assert second.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        assert second.execute("SELECT value FROM durable").fetchall() == [(42,)]
        reopened.revalidate()
    finally:
        second.close()
    reopened._cleanup_created_sqlite_sidecars()


@pytest.mark.skipif(os.name != "posix", reason="SQLite journals require POSIX file modes")
def test_persist_journal_transitions_to_wal_without_weakening_the_boundary(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow.sqlite3"
    initial = authorize_private_sqlite_path(target)
    connection = sqlite3.connect(initial.path, isolation_level=None)
    try:
        initial.revalidate()
        assert connection.execute("PRAGMA journal_mode = PERSIST").fetchone() == ("persist",)
        connection.execute("CREATE TABLE durable(value INTEGER NOT NULL)")
        connection.execute("INSERT INTO durable VALUES (1)")
    finally:
        connection.close()
    journal = _sqlite_journal(target)
    assert journal.exists()

    reopened = authorize_private_sqlite_path(target)
    second = sqlite3.connect(reopened.path, isolation_level=None)
    try:
        reopened.revalidate()
        assert second.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        second.execute("INSERT INTO durable VALUES (2)")
        reopened.revalidate()
        assert second.execute("SELECT value FROM durable ORDER BY value").fetchall() == [
            (1,),
            (2,),
        ]
    finally:
        second.close()
    reopened._cleanup_created_sqlite_sidecars()


@pytest.mark.skipif(os.name != "posix", reason="SQLite WAL requires POSIX file modes")
def test_crash_style_existing_wal_is_recovered_without_replacing_its_inodes(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow.sqlite3"
    initial = authorize_private_sqlite_path(target)
    wal, shm = _sqlite_sidecars(target)
    initial_identities = {sidecar.name: sidecar.lstat().st_ino for sidecar in (wal, shm)}
    child = """
import os
import sqlite3
import sys

connection = sqlite3.connect(sys.argv[1], isolation_level=None)
connection.execute("PRAGMA journal_mode = WAL")
connection.execute("PRAGMA wal_autocheckpoint = 0")
connection.execute("CREATE TABLE durable(value INTEGER NOT NULL)")
connection.execute("INSERT INTO durable VALUES (42)")
os._exit(0)
"""
    subprocess.run(
        [sys.executable, "-c", child, initial.path],
        check=True,
        timeout=10,
    )
    assert {sidecar.name: sidecar.lstat().st_ino for sidecar in (wal, shm)} == (initial_identities)

    recovered = authorize_private_sqlite_path(target)
    connection = sqlite3.connect(recovered.path, isolation_level=None)
    try:
        recovered.revalidate()
        assert connection.execute("SELECT value FROM durable").fetchall() == [(42,)]
        recovered.revalidate()
        assert {sidecar.name: sidecar.lstat().st_ino for sidecar in (wal, shm)} == (
            initial_identities
        )
        # The recovery authorization observed, rather than created, these files.
        recovered._cleanup_created_sqlite_sidecars()
        assert {sidecar.name: sidecar.lstat().st_ino for sidecar in (wal, shm)} == (
            initial_identities
        )
    finally:
        connection.close()

    # SQLite, not the authorization helper, remains their lifecycle owner.


@pytest.mark.skipif(os.name != "posix", reason="private file modes require POSIX")
def test_existing_sqlite_file_can_change_contents_without_changing_authorized_identity(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_file(parent / "shadow.sqlite3")
    authorization = authorize_private_sqlite_path(target)

    target.write_bytes(b"SQLite format 3\0")
    target.chmod(0o600)

    authorization.revalidate()


@pytest.mark.skipif(os.name != "posix", reason="private file modes require POSIX")
def test_authorization_copies_the_path_and_hides_it_from_its_representation(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "raw-secret-parent")
    target = parent / "raw-secret.sqlite3"

    authorization = authorize_private_sqlite_path(target)

    assert "raw-secret" not in repr(authorization)
    with pytest.raises(AttributeError):
        authorization.path = "replacement"  # type: ignore[misc]


@pytest.mark.parametrize("value", (object(), b"database", "", "bad\0path"))
def test_invalid_paths_fail_with_one_value_free_error(value: object) -> None:
    with pytest.raises(SecureFileError) as raised:
        authorize_private_sqlite_path(cast(str | os.PathLike[str], value))

    _assert_sanitized(raised.value, "bad")


def test_missing_parent_fails_without_creating_any_directory(tmp_path: Path) -> None:
    secret = "raw-secret-parent"
    target = tmp_path / secret / "shadow.sqlite3"

    with pytest.raises(SecureFileError) as raised:
        authorize_private_sqlite_path(target)

    _assert_sanitized(raised.value, secret)
    assert not target.parent.exists()


@pytest.mark.parametrize("component", (".", ".."))
def test_dot_path_components_are_rejected_before_normalization(
    tmp_path: Path,
    component: str,
) -> None:
    parent = _private_directory(tmp_path / "store")
    declared = parent / "missing" if component == ".." else parent
    raw_path = f"{declared}{os.sep}{component}{os.sep}shadow.sqlite3"
    normalized_target = parent / "shadow.sqlite3"

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(raw_path)

    assert not normalized_target.exists()


@pytest.mark.skipif(os.name != "posix", reason="private directory modes require POSIX")
@pytest.mark.parametrize("mode", (0o777, 0o770, 0o707, 0o702))
def test_group_or_world_writable_parent_is_rejected(tmp_path: Path, mode: int) -> None:
    parent = _private_directory(tmp_path / "store")
    parent.chmod(mode)
    target = parent / "shadow.sqlite3"

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)

    assert not target.exists()


@pytest.mark.skipif(os.name != "posix", reason="private directory modes require POSIX")
def test_group_or_world_writable_ancestor_is_rejected(tmp_path: Path) -> None:
    ancestor = _private_directory(tmp_path / "unsafe-ancestor")
    parent = _private_directory(ancestor / "store")
    ancestor.chmod(0o777)
    target = parent / "shadow.sqlite3"

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)

    assert not target.exists()


@pytest.mark.skipif(os.name != "posix", reason="private directory modes require POSIX")
def test_owner_writable_ancestor_owned_by_an_untrusted_uid_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ancestor = _private_directory(tmp_path / "foreign-ancestor")
    ancestor.chmod(0o755)
    parent = _private_directory(ancestor / "store")
    target = parent / "shadow.sqlite3"
    ancestor_metadata = ancestor.stat()
    real_fstat = os.fstat

    def foreign_owner(descriptor: int) -> os.stat_result:
        metadata = real_fstat(descriptor)
        if (
            metadata.st_dev == ancestor_metadata.st_dev
            and metadata.st_ino == ancestor_metadata.st_ino
        ):
            return cast(
                os.stat_result,
                SimpleNamespace(
                    st_mode=metadata.st_mode,
                    st_uid=os.getuid() + 1,
                ),
            )
        return metadata

    monkeypatch.setattr(os, "fstat", foreign_owner)

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)

    assert not target.exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS extended ACL semantics")
def test_permission_granting_acl_on_private_parent_is_rejected(tmp_path: Path) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow.sqlite3"
    _add_macos_acl(
        parent,
        "everyone allow write,delete,add_file,add_subdirectory",
    )
    try:
        with pytest.raises(SecureFileError):
            authorize_private_sqlite_path(target)
    finally:
        _clear_macos_acl(parent)

    assert not target.exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS extended ACL semantics")
@pytest.mark.parametrize("boundary", ("database", "wal", "shm"))
def test_permission_granting_acl_on_database_or_sidecar_is_rejected(
    tmp_path: Path,
    boundary: str,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_file(parent / "shadow.sqlite3", b"database")
    wal, shm = _sqlite_sidecars(target)
    selected = target
    if boundary == "wal":
        selected = _private_file(wal, b"wal")
    elif boundary == "shm":
        selected = _private_file(shm, b"shm")
    _add_macos_acl(selected, "everyone allow write,delete")
    try:
        with pytest.raises(SecureFileError):
            authorize_private_sqlite_path(target)
    finally:
        _clear_macos_acl(selected)

    assert target.read_bytes() == b"database"
    if boundary == "wal":
        assert wal.read_bytes() == b"wal"
    elif boundary == "shm":
        assert shm.read_bytes() == b"shm"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS extended ACL semantics")
def test_even_deny_only_acl_is_rejected_on_the_final_database_boundary(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_file(parent / "shadow.sqlite3", b"database")
    _add_macos_acl(target, "everyone deny delete")
    try:
        with pytest.raises(SecureFileError):
            authorize_private_sqlite_path(target)
    finally:
        _clear_macos_acl(target)

    assert target.read_bytes() == b"database"


@pytest.mark.skipif(os.name != "posix", reason="symbolic links require POSIX semantics")
def test_symbolic_parent_is_rejected_before_target_creation(tmp_path: Path) -> None:
    real_parent = _private_directory(tmp_path / "real")
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    target = linked_parent / "shadow.sqlite3"

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)

    assert not (real_parent / target.name).exists()


@pytest.mark.skipif(os.name != "posix", reason="symbolic links require POSIX semantics")
def test_symbolic_ancestor_is_rejected_before_target_creation(tmp_path: Path) -> None:
    real_ancestor = _private_directory(tmp_path / "real")
    real_parent = _private_directory(real_ancestor / "store")
    linked_ancestor = tmp_path / "linked"
    linked_ancestor.symlink_to(real_ancestor, target_is_directory=True)
    target = linked_ancestor / real_parent.name / "shadow.sqlite3"

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)

    assert not (real_parent / target.name).exists()


@pytest.mark.skipif(os.name != "posix", reason="private file modes require POSIX")
@pytest.mark.parametrize("mode", (0o644, 0o400, 0o700, 0o660))
def test_existing_target_requires_exact_mode_0600(tmp_path: Path, mode: int) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_file(parent / "shadow.sqlite3")
    target.chmod(mode)

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)


@pytest.mark.skipif(os.name != "posix", reason="symbolic links require POSIX semantics")
def test_symbolic_target_is_rejected_without_touching_its_referent(tmp_path: Path) -> None:
    parent = _private_directory(tmp_path / "store")
    referent = _private_file(parent / "referent.sqlite3", b"unchanged")
    target = parent / "shadow.sqlite3"
    target.symlink_to(referent)

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)

    assert referent.read_bytes() == b"unchanged"


@pytest.mark.skipif(os.name != "posix", reason="hard links require POSIX semantics")
def test_hard_linked_target_is_rejected(tmp_path: Path) -> None:
    parent = _private_directory(tmp_path / "store")
    source = _private_file(parent / "source.sqlite3")
    target = parent / "shadow.sqlite3"
    target.hardlink_to(source)

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO files are unavailable")
def test_fifo_target_is_rejected_without_opening_it_for_blocking_io(tmp_path: Path) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow.sqlite3"
    os.mkfifo(target, mode=0o600)

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)


def test_directory_target_is_rejected(tmp_path: Path) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_directory(parent / "shadow.sqlite3")

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)


@pytest.mark.skipif(os.name != "posix", reason="private ownership requires POSIX")
def test_wrong_owner_is_rejected_when_owner_check_is_simulated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow.sqlite3"
    monkeypatch.setattr(files_module, "_current_user_id", lambda: os.getuid() + 1)

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)

    assert not target.exists()


@pytest.mark.skipif(os.name != "posix", reason="private ownership requires POSIX")
def test_wrong_target_owner_is_rejected_when_owner_check_is_simulated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_file(parent / "shadow.sqlite3")
    real_named_stat = files_module._named_stat

    def wrong_owner_stat(name: str, directory_fd: int) -> os.stat_result:
        metadata = real_named_stat(name, directory_fd)
        return cast(
            os.stat_result,
            SimpleNamespace(
                st_dev=metadata.st_dev,
                st_ino=metadata.st_ino,
                st_size=metadata.st_size,
                st_mtime_ns=metadata.st_mtime_ns,
                st_ctime_ns=metadata.st_ctime_ns,
                st_mode=metadata.st_mode,
                st_nlink=metadata.st_nlink,
                st_uid=metadata.st_uid + 1,
            ),
        )

    monkeypatch.setattr(files_module, "_named_stat", wrong_owner_stat)

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)


@pytest.mark.skipif(os.name != "posix", reason="inode identities require POSIX")
def test_revalidation_rejects_target_path_replacement(tmp_path: Path) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_file(parent / "shadow.sqlite3", b"original")
    authorization = authorize_private_sqlite_path(target)
    displaced = parent / "displaced.sqlite3"
    target.rename(displaced)
    _private_file(target, b"replacement")

    with pytest.raises(SecureFileError) as raised:
        authorization.revalidate()

    _assert_sanitized(raised.value, "replacement")


@pytest.mark.skipif(os.name != "posix", reason="inode identities require POSIX")
def test_revalidation_rejects_parent_path_replacement(tmp_path: Path) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_file(parent / "shadow.sqlite3")
    authorization = authorize_private_sqlite_path(target)
    displaced = tmp_path / "displaced"
    parent.rename(displaced)
    replacement = _private_directory(tmp_path / "store")
    _private_file(replacement / "shadow.sqlite3")

    with pytest.raises(SecureFileError):
        authorization.revalidate()


@pytest.mark.skipif(os.name != "posix", reason="private file modes require POSIX")
@pytest.mark.parametrize("mutation", ("mode", "hardlink", "symlink"))
def test_revalidation_rejects_target_security_regressions(
    tmp_path: Path,
    mutation: str,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_file(parent / "shadow.sqlite3")
    authorization = authorize_private_sqlite_path(target)

    if mutation == "mode":
        target.chmod(0o644)
    elif mutation == "hardlink":
        (parent / "alias.sqlite3").hardlink_to(target)
    else:
        displaced = parent / "displaced.sqlite3"
        target.rename(displaced)
        target.symlink_to(displaced)

    with pytest.raises(SecureFileError):
        authorization.revalidate()


@pytest.mark.skipif(os.name != "posix", reason="inode identities require POSIX")
def test_authorization_rejects_a_target_that_changes_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_file(parent / "shadow.sqlite3")
    real_named_stat = files_module._named_stat
    target_stat_calls = 0

    def changing_named_stat(name: str, directory_fd: int) -> os.stat_result:
        nonlocal target_stat_calls
        metadata = real_named_stat(name, directory_fd)
        if name == target.name:
            target_stat_calls += 1
            if target_stat_calls == 2:
                return cast(
                    os.stat_result,
                    SimpleNamespace(
                        st_dev=metadata.st_dev,
                        st_ino=metadata.st_ino + 1,
                        st_size=metadata.st_size,
                        st_mtime_ns=metadata.st_mtime_ns,
                        st_ctime_ns=metadata.st_ctime_ns,
                        st_mode=metadata.st_mode,
                        st_nlink=metadata.st_nlink,
                        st_uid=metadata.st_uid,
                    ),
                )
        return metadata

    monkeypatch.setattr(files_module, "_named_stat", changing_named_stat)

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)


@pytest.mark.skipif(os.name != "posix", reason="descriptor walks require POSIX")
def test_directory_walk_closes_its_descriptor_on_base_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = os.open
    real_close = os.close
    opened: list[int] = []
    closed: list[int] = []

    def interrupt_after_root(path: str, flags: int, *args: object, **kwargs: object) -> int:
        if opened:
            raise KeyboardInterrupt
        descriptor = real_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]
        opened.append(descriptor)
        return descriptor

    def record_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(os, "open", interrupt_after_root)
    monkeypatch.setattr(os, "close", record_close)

    with pytest.raises(KeyboardInterrupt):
        files_module._open_directory_chain(Path("/private"))

    if opened != closed:
        for descriptor in set(opened).difference(closed):
            real_close(descriptor)
    assert closed == opened


@pytest.mark.skipif(os.name != "posix", reason="descriptor walks require POSIX")
def test_parent_open_closes_the_first_walk_on_second_walk_base_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow.sqlite3"
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(parent, flags)
    identity = files_module._StableIdentity.from_stat(os.fstat(descriptor))
    real_close = os.close
    closed: list[int] = []
    calls = 0

    def interrupt_second_walk(_path: Path) -> tuple[int, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return descriptor, identity
        raise KeyboardInterrupt

    def record_close(value: int) -> None:
        closed.append(value)
        real_close(value)

    monkeypatch.setattr(files_module, "_open_directory_chain", interrupt_second_walk)
    monkeypatch.setattr(os, "close", record_close)

    with pytest.raises(KeyboardInterrupt):
        files_module._open_parent(target)

    if descriptor not in closed:
        real_close(descriptor)
    assert descriptor in closed


@pytest.mark.skipif(os.name != "posix", reason="inode identities require POSIX")
@pytest.mark.parametrize("race", ("disappear", "swap"))
def test_authorization_never_recreates_a_target_lost_after_initial_stat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    race: str,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_file(parent / "shadow.sqlite3", b"original")
    displaced = parent / "displaced.sqlite3"
    real_open_existing_target = files_module._open_existing_target

    def race_before_open(name: str, directory_fd: int) -> int:
        if race == "disappear":
            target.unlink()
        else:
            target.rename(displaced)
            _private_file(target, b"replacement")
        return real_open_existing_target(name, directory_fd)

    monkeypatch.setattr(files_module, "_open_existing_target", race_before_open)

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)

    if race == "disappear":
        assert not target.exists()
    else:
        assert target.read_bytes() == b"replacement"


@pytest.mark.skipif(os.name != "posix", reason="private file modes require POSIX")
def test_created_target_is_removed_when_its_descriptor_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow.sqlite3"
    real_close = os.close
    injected = False

    def fail_first_regular_file_close(descriptor: int) -> None:
        nonlocal injected
        metadata = os.fstat(descriptor)
        if stat.S_ISREG(metadata.st_mode) and not injected:
            injected = True
            real_close(descriptor)
            raise OSError("injected close failure")
        real_close(descriptor)

    monkeypatch.setattr(os, "close", fail_first_regular_file_close)

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)

    assert injected
    assert not target.exists()


@pytest.mark.skipif(os.name != "posix", reason="private file modes require POSIX")
def test_descriptor_close_failure_never_unlinks_a_new_target_claimed_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow.sqlite3"
    real_close = os.close
    injected = False

    def claim_then_fail_first_regular_file_close(descriptor: int) -> None:
        nonlocal injected
        metadata = os.fstat(descriptor)
        if stat.S_ISREG(metadata.st_mode) and not injected:
            injected = True
            target.write_bytes(b"claimed")
            real_close(descriptor)
            raise OSError("injected close failure")
        real_close(descriptor)

    monkeypatch.setattr(os, "close", claim_then_fail_first_regular_file_close)

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)

    assert injected
    assert target.read_bytes() == b"claimed"


@pytest.mark.skipif(os.name != "posix", reason="private file modes require POSIX")
def test_created_target_is_removed_when_the_final_parent_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow.sqlite3"
    real_verify_parent = files_module._verify_parent
    real_close = os.close
    armed = False
    injected = False

    def arm_after_parent_check(path: Path, descriptor: int, expected: object) -> None:
        nonlocal armed
        real_verify_parent(path, descriptor, expected)  # type: ignore[arg-type]
        armed = True

    def fail_final_parent_close(descriptor: int) -> None:
        nonlocal injected
        metadata = os.fstat(descriptor)
        if armed and stat.S_ISDIR(metadata.st_mode) and not injected:
            injected = True
            real_close(descriptor)
            raise OSError("injected close failure")
        real_close(descriptor)

    monkeypatch.setattr(files_module, "_verify_parent", arm_after_parent_check)
    monkeypatch.setattr(os, "close", fail_final_parent_close)

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)

    assert injected
    assert not target.exists()


@pytest.mark.skipif(os.name != "posix", reason="private file modes require POSIX")
def test_final_parent_close_failure_cleans_sidecars_for_a_preexisting_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_file(parent / "shadow.sqlite3", b"existing")
    wal, shm = _sqlite_sidecars(target)
    real_verify_parent = files_module._verify_parent
    real_close = os.close
    armed = False
    injected = False

    def arm_after_parent_check(path: Path, descriptor: int, expected: object) -> None:
        nonlocal armed
        real_verify_parent(path, descriptor, expected)  # type: ignore[arg-type]
        armed = True

    def fail_final_parent_close(descriptor: int) -> None:
        nonlocal injected
        metadata = os.fstat(descriptor)
        if armed and stat.S_ISDIR(metadata.st_mode) and not injected:
            injected = True
            real_close(descriptor)
            raise OSError("injected close failure")
        real_close(descriptor)

    monkeypatch.setattr(files_module, "_verify_parent", arm_after_parent_check)
    monkeypatch.setattr(os, "close", fail_final_parent_close)

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)

    assert injected
    assert target.read_bytes() == b"existing"
    assert not wal.exists()
    assert not shm.exists()


@pytest.mark.skipif(os.name != "posix", reason="private file modes require POSIX")
def test_failed_authorization_never_unlinks_a_new_database_claimed_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow.sqlite3"
    wal, shm = _sqlite_sidecars(target)
    real_verify_parent = files_module._verify_parent

    def claim_target_then_fail(path: Path, descriptor: int, expected: object) -> None:
        real_verify_parent(path, descriptor, expected)  # type: ignore[arg-type]
        target.write_bytes(b"claimed")
        raise OSError("injected validation failure")

    monkeypatch.setattr(files_module, "_verify_parent", claim_target_then_fail)

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)

    assert target.read_bytes() == b"claimed"
    assert not wal.exists()
    assert not shm.exists()


@pytest.mark.skipif(os.name != "posix", reason="private file modes require POSIX")
def test_failed_authorization_never_cleans_sidecars_after_target_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow.sqlite3"
    wal, shm = _sqlite_sidecars(target)
    displaced = parent / "database-displaced"
    real_verify_parent = files_module._verify_parent

    def replace_target_then_fail(path: Path, descriptor: int, expected: object) -> None:
        real_verify_parent(path, descriptor, expected)  # type: ignore[arg-type]
        target.rename(displaced)
        _private_file(target, b"replacement")
        raise OSError("injected validation failure")

    monkeypatch.setattr(files_module, "_verify_parent", replace_target_then_fail)

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)

    assert target.read_bytes() == b"replacement"
    assert wal.read_bytes() == b""
    assert shm.read_bytes() == b""


@pytest.mark.skipif(os.name != "posix", reason="hard links require POSIX semantics")
def test_authorization_rechecks_target_after_the_final_parent_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_file(parent / "shadow.sqlite3")
    alias = parent / "alias.sqlite3"
    real_verify_parent = files_module._verify_parent

    def link_after_parent_check(
        path: Path,
        descriptor: int,
        expected: object,
    ) -> None:
        real_verify_parent(path, descriptor, expected)  # type: ignore[arg-type]
        alias.hardlink_to(target)

    monkeypatch.setattr(files_module, "_verify_parent", link_after_parent_check)

    with pytest.raises(SecureFileError):
        authorize_private_sqlite_path(target)

    assert target.stat().st_nlink == 2


@pytest.mark.skipif(os.name != "posix", reason="hard links require POSIX semantics")
def test_revalidation_rechecks_target_after_the_final_parent_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_file(parent / "shadow.sqlite3")
    authorization = authorize_private_sqlite_path(target)
    alias = parent / "alias.sqlite3"
    real_verify_parent = files_module._verify_parent

    def link_after_parent_check(
        path: Path,
        descriptor: int,
        expected: object,
    ) -> None:
        real_verify_parent(path, descriptor, expected)  # type: ignore[arg-type]
        alias.hardlink_to(target)

    monkeypatch.setattr(files_module, "_verify_parent", link_after_parent_check)

    with pytest.raises(SecureFileError):
        authorization.revalidate()

    assert target.stat().st_nlink == 2


def test_legacy_stable_reader_preserves_writable_parent_and_target_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "legacy"
    parent.mkdir(mode=0o777)
    parent.chmod(0o777)
    target = parent / "trace.jsonl"
    target.write_bytes(b"first\nsecond\n")
    target.chmod(0o666)
    monkeypatch.chdir(parent)

    read = read_stable_file(
        Path(".") / "trace.jsonl",
        maximum_bytes=64,
        policy=StableReadPolicy.LEGACY_COMPATIBILITY,
    )

    assert isinstance(read, StableFileRead)
    assert read.data == b"first\nsecond\n"
    assert read.authorization.path == os.path.abspath(target)
    read.authorization.revalidate()


@pytest.mark.skipif(os.name != "posix", reason="symbolic traversal requires POSIX")
def test_legacy_reader_preserves_kernel_symlink_then_parent_traversal(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "outside"
    child = outside / "child"
    child.mkdir(parents=True)
    (base / "link").symlink_to(child, target_is_directory=True)
    _private_file(base / "trace", b"lexically-normalized-wrong-file")
    selected = _private_file(outside / "trace", b"kernel-selected-file")
    declared = base / "link" / ".." / "trace"

    read = read_stable_file(
        os.fspath(declared),
        maximum_bytes=64,
        policy=StableReadPolicy.LEGACY_COMPATIBILITY,
    )

    assert read.data == b"kernel-selected-file"
    assert os.path.samefile(read.authorization.path, selected)
    read.authorization.revalidate()


@pytest.mark.skipif(os.name != "posix", reason="private ownership requires POSIX")
def test_private_reader_accepts_public_read_bits_but_not_public_write_bits(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "private")
    accepted = _private_file(parent / "accepted", b"safe")
    accepted.chmod(0o644)
    rejected = _private_file(parent / "rejected", b"unsafe")
    rejected.chmod(0o622)

    read = read_stable_file(
        accepted,
        maximum_bytes=4,
        policy=StableReadPolicy.PRIVATE_OWNER,
    )

    assert read.data == b"safe"
    with pytest.raises(SecureFileError):
        read_stable_file(
            rejected,
            maximum_bytes=16,
            policy=StableReadPolicy.PRIVATE_OWNER,
        )


@pytest.mark.parametrize("maximum_bytes", (0, -1, True, sys.maxsize))
def test_stable_reader_rejects_invalid_bounds_with_the_distinct_bound_error(
    tmp_path: Path,
    maximum_bytes: object,
) -> None:
    target = _private_file(tmp_path / "input", b"a")

    with pytest.raises(SecureFileBoundError) as raised:
        read_stable_file(
            target,
            maximum_bytes=cast(int, maximum_bytes),
            policy=StableReadPolicy.LEGACY_COMPATIBILITY,
        )

    assert type(raised.value) is SecureFileBoundError
    assert str(raised.value) == "secure file bound exceeded"


def test_initially_oversized_stable_file_has_the_distinct_bound_error(
    tmp_path: Path,
) -> None:
    target = _private_file(tmp_path / "input", b"12345")

    with pytest.raises(SecureFileBoundError):
        read_stable_file(
            target,
            maximum_bytes=4,
            policy=StableReadPolicy.LEGACY_COMPATIBILITY,
        )


def test_growth_during_read_is_an_integrity_error_not_a_bound_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _private_file(tmp_path / "input", b"1234")
    real_read = os.read
    injected = False

    def grow_then_read(descriptor: int, maximum: int) -> bytes:
        nonlocal injected
        if not injected:
            injected = True
            target.write_bytes(b"12345")
        return real_read(descriptor, maximum)

    monkeypatch.setattr(os, "read", grow_then_read)

    with pytest.raises(SecureFileError) as raised:
        read_stable_file(
            target,
            maximum_bytes=4,
            policy=StableReadPolicy.LEGACY_COMPATIBILITY,
        )

    assert type(raised.value) is SecureFileError


def test_stable_reader_rejects_metadata_mutation_after_bytes_are_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _private_file(tmp_path / "input", b"stable")
    real_read = files_module._read_descriptor_bounded

    def mutate_after_read(descriptor: int, maximum: int) -> bytes:
        data = real_read(descriptor, maximum)
        target.chmod(0o400)
        return data

    monkeypatch.setattr(files_module, "_read_descriptor_bounded", mutate_after_read)

    with pytest.raises(SecureFileError):
        read_stable_file(
            target,
            maximum_bytes=16,
            policy=StableReadPolicy.LEGACY_COMPATIBILITY,
        )


@pytest.mark.parametrize(
    ("encoded", "expected"),
    (
        (b"", ()),
        (b"a", (b"a",)),
        (b"a\n", (b"a",)),
        (b"a\n\n", (b"a", b"")),
        (b"\n", (b"",)),
        (b"a\r\nb\r", (b"a", b"b\r")),
        (b"\r", (b"\r",)),
    ),
)
def test_stable_line_iterator_has_strict_lf_and_crlf_edge_semantics(
    tmp_path: Path,
    encoded: bytes,
    expected: tuple[bytes, ...],
) -> None:
    target = _private_file(tmp_path / "input", encoded)
    read = read_stable_file(
        target,
        maximum_bytes=max(1, len(encoded)),
        policy=StableReadPolicy.LEGACY_COMPATIBILITY,
    )

    assert tuple(read.iter_lines(maximum_line_bytes=16, maximum_lines=8)) == expected


@pytest.mark.parametrize(
    ("encoded", "maximum_line_bytes", "maximum_lines"),
    (
        (b"ab\n", 1, 2),
        (b"a\r\n", 1, 2),
        (b"a\nb\n", 2, 1),
    ),
)
def test_stable_line_iterator_enforces_encoded_line_and_count_bounds(
    tmp_path: Path,
    encoded: bytes,
    maximum_line_bytes: int,
    maximum_lines: int,
) -> None:
    target = _private_file(tmp_path / "input", encoded)
    read = read_stable_file(
        target,
        maximum_bytes=len(encoded),
        policy=StableReadPolicy.LEGACY_COMPATIBILITY,
    )

    with pytest.raises(SecureFileBoundError):
        tuple(
            read.iter_lines(
                maximum_line_bytes=maximum_line_bytes,
                maximum_lines=maximum_lines,
            )
        )


def test_stable_read_authorization_rejects_content_or_name_replacement(
    tmp_path: Path,
) -> None:
    target = _private_file(tmp_path / "input", b"original")
    read = read_stable_file(
        target,
        maximum_bytes=16,
        policy=StableReadPolicy.LEGACY_COMPATIBILITY,
    )
    target.write_bytes(b"changed!")

    with pytest.raises(SecureFileError):
        read.authorization.revalidate()

    target.unlink()
    _private_file(target, b"original")
    with pytest.raises(SecureFileError):
        read.authorization.revalidate()


def test_private_location_snapshots_an_absent_slot_without_creating_it(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = parent / "report.json"

    authorization = inspect_private_file_location(target)

    assert not target.exists()
    authorization.revalidate()
    _private_file(target, b"claimed")
    with pytest.raises(SecureFileError):
        authorization.revalidate()


def test_private_location_existing_target_is_exact_and_revalidated(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = _private_file(parent / "report.json", b"old")
    authorization = inspect_private_file_location(target)

    authorization.revalidate()
    target.write_bytes(b"new")

    with pytest.raises(SecureFileError):
        authorization.revalidate()


@pytest.mark.parametrize("mode", (0o644, 0o400, 0o660))
def test_private_location_requires_exact_owner_only_target_mode(
    tmp_path: Path,
    mode: int,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = _private_file(parent / "report.json", b"old")
    target.chmod(mode)

    with pytest.raises(SecureFileError):
        inspect_private_file_location(target)


@pytest.mark.skipif(
    os.open not in os.supports_dir_fd,
    reason="secure-platform capability is already unavailable",
)
def test_private_location_reports_an_unsupported_secure_platform_distinctly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = parent / "report.json"
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd - {os.open})

    with pytest.raises(SecureFileUnsupportedError) as raised:
        inspect_private_file_location(target)

    assert type(raised.value) is SecureFileUnsupportedError
    assert str(raised.value) == "secure file operation unsupported"
    assert not target.exists()


def test_authorization_aliases_are_conservative_for_absent_unicode_and_case_slots(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "private")
    composed = inspect_private_file_location(parent / "Réport.JSON")
    decomposed = inspect_private_file_location(parent / "réport.json")
    distinct = inspect_private_file_location(parent / "other.json")

    assert composed.aliases(decomposed)
    assert decomposed.aliases(composed)
    assert not composed.aliases(distinct)


def test_sqlite_authorization_aliases_every_captured_sidecar_location(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "private")
    database = parent / "shadow.sqlite3"
    database_authorization = authorize_private_sqlite_path(database)

    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = read_stable_file(
            Path(f"{database}{suffix}"),
            maximum_bytes=1,
            policy=StableReadPolicy.PRIVATE_OWNER,
        )
        assert database_authorization.aliases(sidecar.authorization)
        assert sidecar.authorization.aliases(database_authorization)


def test_authorization_aliases_rejects_non_authorization_inputs(tmp_path: Path) -> None:
    parent = _private_directory(tmp_path / "private")
    authorization = inspect_private_file_location(parent / "report.json")

    with pytest.raises(SecureFileError):
        authorization.aliases(cast(StableFileAuthorization, object()))


def test_non_sqlite_authorizations_cannot_cross_the_sqlite_statement_boundary(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = _private_file(parent / "input", b"safe")
    read = read_stable_file(
        target,
        maximum_bytes=4,
        policy=StableReadPolicy.PRIVATE_OWNER,
    )
    location = inspect_private_file_location(parent / "output")

    for authorization in (read.authorization, location):
        with pytest.raises(SecureFileError):
            authorization._revalidate_before_sqlite_statements()


def _atomic_residue(parent: Path) -> tuple[Path, ...]:
    return tuple(parent.glob(".saliencegate-atomic-*"))


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_atomic_publication_creates_one_exact_private_file_without_residue(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = parent / "report.json"
    publication = authorize_atomic_file_publication(target, maximum_bytes=16)

    published = publication.publish(b'{"ok":true}')

    metadata = target.lstat()
    assert isinstance(publication, AtomicFilePublication)
    assert publication.authorization.path == os.path.abspath(target)
    assert published.data == b'{"ok":true}'
    assert target.read_bytes() == published.data
    assert stat.S_ISREG(metadata.st_mode)
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_uid == os.getuid()
    assert metadata.st_nlink == 1
    assert _atomic_residue(parent) == ()
    published.authorization.revalidate()

    with pytest.raises(SecureFileError):
        publication.publish(b"again")


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_replacement_callbacks_are_ordered_exactly_once_around_publication(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = _private_file(parent / "report.json", b"old")
    calls: list[tuple[str, bytes]] = []

    def validate_old(data: bytes) -> bool:
        assert _atomic_residue(parent) == ()
        calls.append(("old", data))
        return True

    publication = authorize_atomic_file_publication(
        target,
        maximum_bytes=16,
        validate_replacement=validate_old,
    )
    assert calls == [("old", b"old")]

    def validate_new(data: bytes) -> bool:
        calls.append(("new", data))
        assert target.read_bytes() == b"new"
        return True

    result = publication.publish(b"new", validate_published=validate_new)

    assert result.data == b"new"
    assert calls == [("old", b"old"), ("new", b"new")]
    assert target.read_bytes() == b"new"
    assert _atomic_residue(parent) == ()


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_replacement_authorization_rejects_without_staging_or_mutation(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = _private_file(parent / "report.json", b"old")

    with pytest.raises(SecureFileError):
        authorize_atomic_file_publication(target, maximum_bytes=16)
    assert target.read_bytes() == b"old"
    assert _atomic_residue(parent) == ()

    calls = 0

    def reject(data: bytes) -> bool:
        nonlocal calls
        calls += 1
        assert data == b"old"
        assert _atomic_residue(parent) == ()
        return False

    with pytest.raises(SecureFileError):
        authorize_atomic_file_publication(
            target,
            maximum_bytes=16,
            validate_replacement=reject,
        )
    assert calls == 1
    assert target.read_bytes() == b"old"
    assert _atomic_residue(parent) == ()

    def explode(_data: bytes) -> bool:
        raise RuntimeError("raw-replacement-secret")

    with pytest.raises(SecureFileError) as raised:
        authorize_atomic_file_publication(
            target,
            maximum_bytes=16,
            validate_replacement=explode,
        )
    _assert_sanitized(raised.value, "raw-replacement-secret")
    assert target.read_bytes() == b"old"
    assert _atomic_residue(parent) == ()


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_replacement_callback_is_not_called_for_an_absent_target(tmp_path: Path) -> None:
    parent = _private_directory(tmp_path / "private")
    target = parent / "report.json"
    calls = 0

    def replacement(_data: bytes) -> bool:
        nonlocal calls
        calls += 1
        return True

    publication = authorize_atomic_file_publication(
        target,
        maximum_bytes=16,
        validate_replacement=replacement,
    )
    publication.publish(b"new")

    assert calls == 0
    assert target.read_bytes() == b"new"


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_atomic_publication_handles_short_writes_and_rejects_zero_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = parent / "short.json"
    publication = authorize_atomic_file_publication(target, maximum_bytes=32)
    real_write = os.write
    shortened = False

    def short_once(descriptor: int, data: object) -> int:
        nonlocal shortened
        view = memoryview(data)  # type: ignore[arg-type]
        if not shortened and len(view) > 1:
            shortened = True
            return real_write(descriptor, view[:2])
        return real_write(descriptor, view)

    monkeypatch.setattr(os, "write", short_once)
    publication.publish(b"complete")
    assert target.read_bytes() == b"complete"

    zero_target = parent / "zero.json"
    zero_publication = authorize_atomic_file_publication(zero_target, maximum_bytes=32)
    monkeypatch.setattr(os, "write", lambda _descriptor, _data: 0)
    with pytest.raises(SecureFileError):
        zero_publication.publish(b"never")
    assert not zero_target.exists()
    assert _atomic_residue(parent) == ()


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_no_clobber_revalidates_an_absent_target_before_staging(tmp_path: Path) -> None:
    parent = _private_directory(tmp_path / "private")
    target = parent / "report.json"
    publication = authorize_atomic_file_publication(target, maximum_bytes=16)
    _private_file(target, b"claimed")

    with pytest.raises(SecureFileError):
        publication.publish(b"new")

    assert target.read_bytes() == b"claimed"
    assert _atomic_residue(parent) == ()


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_postpublication_rejection_rolls_replacement_back_byte_exactly(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = _private_file(parent / "report.json", b"old-report")
    publication = authorize_atomic_file_publication(
        target,
        maximum_bytes=32,
        validate_replacement=lambda data: data == b"old-report",
    )

    with pytest.raises(SecureFileError):
        publication.publish(b"new-report", validate_published=lambda _data: False)

    metadata = target.lstat()
    assert target.read_bytes() == b"old-report"
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_nlink == 1
    assert _atomic_residue(parent) == ()


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
@pytest.mark.parametrize("operation", ("link", "rename"))
def test_namespace_syscall_that_succeeds_then_raises_is_classified_from_inodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    parent = _private_directory(tmp_path / f"private-{operation}")
    target = parent / "report.json"
    if operation == "rename":
        _private_file(target, b"old")
        publication = authorize_atomic_file_publication(
            target,
            maximum_bytes=16,
            validate_replacement=lambda data: data == b"old",
        )
        original = os.rename

        def mutate_then_raise(*args: object, **kwargs: object) -> None:
            original(*args, **kwargs)  # type: ignore[arg-type]
            raise OSError("injected rename result loss")

        monkeypatch.setattr(os, "rename", mutate_then_raise)
    else:
        publication = authorize_atomic_file_publication(target, maximum_bytes=16)
        original = os.link

        def mutate_then_raise(*args: object, **kwargs: object) -> None:
            original(*args, **kwargs)  # type: ignore[arg-type]
            raise OSError("injected link result loss")

        monkeypatch.setattr(os, "link", mutate_then_raise)

    result = publication.publish(b"new")

    assert result.data == b"new"
    assert target.read_bytes() == b"new"
    assert _atomic_residue(parent) == ()


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_directory_fsync_failure_after_replace_restores_old_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = _private_file(parent / "report.json", b"old")
    publication = authorize_atomic_file_publication(
        target,
        maximum_bytes=16,
        validate_replacement=lambda data: data == b"old",
    )
    real_fsync = os.fsync
    directory_calls = 0

    def fail_second_directory_fsync(descriptor: int) -> None:
        nonlocal directory_calls
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_calls += 1
            if directory_calls == 2:
                raise OSError("injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_second_directory_fsync)

    with pytest.raises(SecureFileError):
        publication.publish(b"new")

    assert target.read_bytes() == b"old"
    assert _atomic_residue(parent) == ()


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_atomic_publication_rejects_parent_replacement_without_writing_either_parent(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = parent / "report.json"
    publication = authorize_atomic_file_publication(target, maximum_bytes=16)
    displaced = tmp_path / "displaced"
    parent.rename(displaced)
    _private_directory(parent)

    with pytest.raises(SecureFileError):
        publication.publish(b"new")

    assert not target.exists()
    assert not (displaced / target.name).exists()
    assert _atomic_residue(parent) == ()
    assert _atomic_residue(displaced) == ()


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_two_no_clobber_publishers_produce_one_complete_winner(tmp_path: Path) -> None:
    parent = _private_directory(tmp_path / "private")
    target = parent / "report.json"
    publications = tuple(
        authorize_atomic_file_publication(target, maximum_bytes=16) for _ in range(2)
    )

    def attempt(index: int) -> tuple[bool, bytes]:
        data = f"report-{index}".encode()
        try:
            publications[index].publish(data)
        except SecureFileError:
            return False, data
        return True, data

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(attempt, range(2)))

    winners = tuple(data for success, data in outcomes if success)
    assert len(winners) == 1
    assert target.read_bytes() == winners[0]
    assert _atomic_residue(parent) == ()


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_two_replacers_produce_one_complete_winner(tmp_path: Path) -> None:
    parent = _private_directory(tmp_path / "private")
    target = _private_file(parent / "report.json", b"old")
    publications = tuple(
        authorize_atomic_file_publication(
            target,
            maximum_bytes=16,
            validate_replacement=lambda data: data == b"old",
        )
        for _ in range(2)
    )

    def attempt(index: int) -> tuple[bool, bytes]:
        data = f"report-{index}".encode()
        try:
            publications[index].publish(data)
        except SecureFileError:
            return False, data
        return True, data

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(attempt, range(2)))

    winners = tuple(data for success, data in outcomes if success)
    assert len(winners) == 1
    assert target.read_bytes() == winners[0]
    assert _atomic_residue(parent) == ()


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_replacement_lock_contention_fails_fast_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = _private_file(parent / "report.json", b"old")
    publication = authorize_atomic_file_publication(
        target,
        maximum_bytes=16,
        validate_replacement=lambda data: data == b"old",
    )
    assert files_module._fcntl is not None
    real_flock = files_module._fcntl.flock
    attempted_flags: list[int] = []

    def contend(descriptor: int, flags: int) -> object:
        attempted_flags.append(flags)
        if flags & files_module._fcntl.LOCK_EX:
            raise BlockingIOError(errno.EWOULDBLOCK, "injected contention")
        return real_flock(descriptor, flags)

    monkeypatch.setattr(files_module._fcntl, "flock", contend)

    with pytest.raises(SecureFileError) as raised:
        publication.publish(b"new")

    _assert_sanitized(raised.value, "injected contention")
    assert attempted_flags[0] & files_module._fcntl.LOCK_NB
    assert target.read_bytes() == b"old"
    assert _atomic_residue(parent) == ()


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
@pytest.mark.parametrize("operation", ("link", "rename"))
def test_filesystem_without_atomic_namespace_operation_is_reported_as_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    parent = _private_directory(tmp_path / f"private-{operation}")
    target = parent / "report.json"
    if operation == "rename":
        _private_file(target, b"old")
        publication = authorize_atomic_file_publication(
            target,
            maximum_bytes=16,
            validate_replacement=lambda data: data == b"old",
        )
    else:
        publication = authorize_atomic_file_publication(target, maximum_bytes=16)

    def unsupported(*_args: object, **_kwargs: object) -> None:
        raise OSError(getattr(errno, "EOPNOTSUPP", errno.ENOSYS), "unsupported")

    monkeypatch.setattr(os, operation, unsupported)

    with pytest.raises(SecureFileUnsupportedError):
        publication.publish(b"new")

    if operation == "rename":
        assert target.read_bytes() == b"old"
    else:
        assert not target.exists()
    assert _atomic_residue(parent) == ()


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_transient_cleanup_unlink_failure_is_retried_without_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = parent / "report.json"
    publication = authorize_atomic_file_publication(target, maximum_bytes=16)
    real_unlink = os.unlink
    failed = False

    def fail_one_atomic_cleanup(
        name: str,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal failed
        if name.startswith(".saliencegate-atomic-") and not failed:
            failed = True
            raise OSError("injected cleanup failure")
        real_unlink(name, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", fail_one_atomic_cleanup)

    publication.publish(b"new")

    assert failed
    assert target.read_bytes() == b"new"
    assert _atomic_residue(parent) == ()


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_file_fsync_failure_leaves_old_replacement_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = _private_file(parent / "report.json", b"old")
    publication = authorize_atomic_file_publication(
        target,
        maximum_bytes=16,
        validate_replacement=lambda data: data == b"old",
    )
    real_fsync = os.fsync
    failed = False

    def fail_first_regular_fsync(descriptor: int) -> None:
        nonlocal failed
        if stat.S_ISREG(os.fstat(descriptor).st_mode) and not failed:
            failed = True
            raise OSError("injected file fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_first_regular_fsync)

    with pytest.raises(SecureFileError):
        publication.publish(b"new")

    assert failed
    assert target.read_bytes() == b"old"
    assert _atomic_residue(parent) == ()


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_temp_permission_failure_leaves_no_output_or_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = parent / "report.json"
    publication = authorize_atomic_file_publication(target, maximum_bytes=16)
    monkeypatch.setattr(
        os,
        "fchmod",
        lambda _descriptor, _mode: (_ for _ in ()).throw(OSError("injected permission failure")),
    )

    with pytest.raises(SecureFileError):
        publication.publish(b"new")

    assert not target.exists()
    assert _atomic_residue(parent) == ()


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_directory_fsync_failure_after_no_clobber_link_removes_new_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = parent / "report.json"
    publication = authorize_atomic_file_publication(target, maximum_bytes=16)
    real_fsync = os.fsync
    failed = False

    def fail_first_directory_fsync(descriptor: int) -> None:
        nonlocal failed
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) and not failed:
            failed = True
            raise OSError("injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_first_directory_fsync)

    with pytest.raises(SecureFileError):
        publication.publish(b"new")

    assert failed
    assert not target.exists()
    assert _atomic_residue(parent) == ()


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_postpublication_callback_exception_is_sanitized_and_rolls_back(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = _private_file(parent / "report.json", b"old")
    publication = authorize_atomic_file_publication(
        target,
        maximum_bytes=16,
        validate_replacement=lambda data: data == b"old",
    )

    def explode(_data: bytes) -> bool:
        raise RuntimeError("raw-postpublication-secret")

    with pytest.raises(SecureFileError) as raised:
        publication.publish(b"new", validate_published=explode)

    _assert_sanitized(raised.value, "raw-postpublication-secret")
    assert target.read_bytes() == b"old"
    assert _atomic_residue(parent) == ()


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_rollback_rename_that_succeeds_then_raises_is_classified_as_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = _private_file(parent / "report.json", b"old")
    publication = authorize_atomic_file_publication(
        target,
        maximum_bytes=16,
        validate_replacement=lambda data: data == b"old",
    )
    real_rename = os.rename
    calls = 0

    def raise_after_rollback(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        real_rename(*args, **kwargs)  # type: ignore[arg-type]
        if calls == 2:
            raise OSError("injected rollback result loss")

    monkeypatch.setattr(os, "rename", raise_after_rollback)

    with pytest.raises(SecureFileError):
        publication.publish(b"new", validate_published=lambda _data: False)

    assert calls == 2
    assert target.read_bytes() == b"old"
    assert _atomic_residue(parent) == ()


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_transient_rollback_rename_failure_is_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = _private_file(parent / "report.json", b"old")
    publication = authorize_atomic_file_publication(
        target,
        maximum_bytes=16,
        validate_replacement=lambda data: data == b"old",
    )
    real_rename = os.rename
    calls = 0

    def fail_first_rollback(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected rollback failure")
        real_rename(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "rename", fail_first_rollback)

    with pytest.raises(SecureFileError):
        publication.publish(b"new", validate_published=lambda _data: False)

    assert calls == 3
    assert target.read_bytes() == b"old"
    assert _atomic_residue(parent) == ()


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_postpublication_corruption_of_target_and_backup_still_restores_old_bytes(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = _private_file(parent / "report.json", b"old")
    publication = authorize_atomic_file_publication(
        target,
        maximum_bytes=32,
        validate_replacement=lambda data: data == b"old",
    )

    def corrupt_then_reject(_data: bytes) -> bool:
        target.write_bytes(b"corrupt-new")
        for temporary in _atomic_residue(parent):
            temporary.write_bytes(b"corrupt-backup")
        return False

    with pytest.raises(SecureFileError):
        publication.publish(b"new", validate_published=corrupt_then_reject)

    assert target.read_bytes() == b"old"
    assert _atomic_residue(parent) == ()


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_postreplace_reopen_failure_rolls_back_and_cleans_every_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = _private_file(parent / "report.json", b"old")
    publication = authorize_atomic_file_publication(
        target,
        maximum_bytes=16,
        validate_replacement=lambda data: data == b"old",
    )
    real_read = files_module._read_private_file
    failed = False

    def fail_once(path: Path, maximum_bytes: int) -> StableFileRead:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected reopen failure")
        return real_read(path, maximum_bytes)

    monkeypatch.setattr(files_module, "_read_private_file", fail_once)

    with pytest.raises(SecureFileError):
        publication.publish(b"new")

    assert failed
    assert target.read_bytes() == b"old"
    assert _atomic_residue(parent) == ()


@pytest.mark.skipif(os.name != "posix", reason="atomic publication requires POSIX")
def test_replacement_target_swap_before_publish_is_never_overwritten(tmp_path: Path) -> None:
    parent = _private_directory(tmp_path / "private")
    target = _private_file(parent / "report.json", b"old")
    publication = authorize_atomic_file_publication(
        target,
        maximum_bytes=16,
        validate_replacement=lambda data: data == b"old",
    )
    displaced = parent / "old-displaced"
    target.rename(displaced)
    _private_file(target, b"third-party")

    with pytest.raises(SecureFileError):
        publication.publish(b"new")

    assert target.read_bytes() == b"third-party"
    assert displaced.read_bytes() == b"old"
    assert _atomic_residue(parent) == ()


def test_atomic_publication_reports_an_unsupported_platform_distinctly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = parent / "report.json"
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd - {os.link})

    with pytest.raises(SecureFileUnsupportedError) as raised:
        authorize_atomic_file_publication(target, maximum_bytes=16)

    assert type(raised.value) is SecureFileUnsupportedError
    assert str(raised.value) == "secure file operation unsupported"
    assert not target.exists()
    assert _atomic_residue(parent) == ()


def test_atomic_publication_preserves_unsupported_location_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "private")
    target = parent / "report.json"

    def unsupported(*_args: object, **_kwargs: object) -> None:
        raise OSError(getattr(errno, "EOPNOTSUPP", errno.ENOSYS), "unsupported")

    monkeypatch.setattr(files_module, "_open_parent", unsupported)

    with pytest.raises(SecureFileUnsupportedError):
        authorize_atomic_file_publication(target, maximum_bytes=16)

    assert not target.exists()


def test_atomic_publication_validates_bounds_callbacks_and_public_api(tmp_path: Path) -> None:
    from saliencegate import security

    parent = _private_directory(tmp_path / "private")
    target = parent / "report.json"
    with pytest.raises(SecureFileBoundError):
        authorize_atomic_file_publication(target, maximum_bytes=0)
    with pytest.raises(SecureFileError):
        authorize_atomic_file_publication(
            target,
            maximum_bytes=16,
            validate_replacement=cast(object, "not-callable"),  # type: ignore[arg-type]
        )

    publication = authorize_atomic_file_publication(target, maximum_bytes=1)
    with pytest.raises(SecureFileBoundError):
        publication.publish(b"xx")

    assert security.AtomicFilePublication is AtomicFilePublication
    assert security.authorize_atomic_file_publication is authorize_atomic_file_publication
    assert {
        "AtomicFilePublication",
        "authorize_atomic_file_publication",
    }.issubset(security.__all__)


def test_public_security_api_exports_the_stable_authorization_contract() -> None:
    from saliencegate import security

    assert security.SecureFileBoundError is SecureFileBoundError
    assert security.SecureFileError is SecureFileError
    assert security.SecureFileUnsupportedError is SecureFileUnsupportedError
    assert security.StableFileRead is StableFileRead
    assert security.StableFileAuthorization is StableFileAuthorization
    assert security.StableReadPolicy is StableReadPolicy
    assert security.authorize_private_sqlite_path is authorize_private_sqlite_path
    assert security.inspect_private_file_location is inspect_private_file_location
    assert security.read_stable_file is read_stable_file
    assert {
        "SecureFileBoundError",
        "SecureFileError",
        "SecureFileUnsupportedError",
        "StableFileRead",
        "StableFileAuthorization",
        "StableReadPolicy",
        "authorize_private_sqlite_path",
        "inspect_private_file_location",
        "read_stable_file",
    }.issubset(security.__all__)
