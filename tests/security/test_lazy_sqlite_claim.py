from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import saliencegate.security.files as files_module
from saliencegate.security.files import (
    SecureFileError,
    StableFileAuthorization,
    authorize_private_sqlite_path,
    inspect_private_file_location,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="private SQLite capabilities require POSIX",
)

_SQLITE_SUFFIXES = ("", "-wal", "-shm", "-journal")


def _private_directory(path: Path) -> Path:
    path.mkdir()
    path.chmod(0o700)
    return path


def _private_file(path: Path, data: bytes = b"") -> Path:
    path.write_bytes(data)
    path.chmod(0o600)
    return path


def _sqlite_paths(path: Path) -> tuple[Path, ...]:
    return tuple(Path(f"{path}{suffix}") for suffix in _SQLITE_SUFFIXES)


def _sqlite_snapshot(path: Path) -> tuple[tuple[Path, bytes, int], ...]:
    return tuple(
        (candidate, candidate.read_bytes(), candidate.lstat().st_ino)
        for candidate in _sqlite_paths(path)
    )


def _claim(location: StableFileAuthorization) -> StableFileAuthorization:
    return files_module._claim_private_sqlite_location(location)


def test_inspection_is_non_mutating_until_the_absent_slot_is_claimed(tmp_path: Path) -> None:
    parent = _private_directory(tmp_path / "store")
    database = parent / "shadow.sqlite3"
    paths = _sqlite_paths(database)

    location = inspect_private_file_location(database)

    assert not any(path.exists() for path in paths)
    location.revalidate()

    authorization = _claim(location)

    for path in paths:
        metadata = path.lstat()
        assert path.read_bytes() == b""
        assert stat.S_ISREG(metadata.st_mode)
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_nlink == 1
        assert metadata.st_uid == os.getuid()
    authorization.revalidate()
    authorization._revalidate_before_sqlite_statements()


def test_claim_preserves_the_exact_existing_database_and_authorizes_sidecars(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    database = _private_file(parent / "shadow.sqlite3", b"existing-database")
    before = database.lstat()
    location = inspect_private_file_location(database)

    authorization = _claim(location)

    after = database.lstat()
    assert database.read_bytes() == b"existing-database"
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    for sidecar in _sqlite_paths(database)[1:]:
        assert sidecar.read_bytes() == b""
    authorization.revalidate()
    authorization._revalidate_before_sqlite_statements()


def test_stale_absent_location_never_overwrites_or_removes_a_peer_claim(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    database = parent / "shadow.sqlite3"
    location = inspect_private_file_location(database)
    _private_file(database, b"peer-database")

    with pytest.raises(SecureFileError) as raised:
        _claim(location)

    assert database.read_bytes() == b"peer-database"
    assert not any(path.exists() for path in _sqlite_paths(database)[1:])
    assert str(raised.value) == "secure file authorization failed"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_absent_slot_race_between_check_and_create_is_still_no_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "store")
    database = parent / "shadow.sqlite3"
    location = inspect_private_file_location(database)
    real_create_target = files_module._create_target
    raced = False

    def race_then_create(directory_fd: int, name: str) -> object:
        nonlocal raced
        if name == database.name and not raced:
            raced = True
            _private_file(database, b"peer-database")
        return real_create_target(directory_fd, name)

    monkeypatch.setattr(files_module, "_create_target", race_then_create)

    with pytest.raises(SecureFileError):
        _claim(location)

    assert raced is True
    assert database.read_bytes() == b"peer-database"
    assert not any(path.exists() for path in _sqlite_paths(database)[1:])


@pytest.mark.parametrize("mutation", ("content", "replacement"))
def test_existing_location_drift_fails_before_sidecar_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    parent = _private_directory(tmp_path / "store")
    database = _private_file(parent / "shadow.sqlite3", b"original")
    location = inspect_private_file_location(database)
    displaced = parent / "displaced.sqlite3"
    if mutation == "content":
        database.write_bytes(b"modified")
    else:
        database.rename(displaced)
        _private_file(database, b"replacement")

    with pytest.raises(SecureFileError):
        _claim(location)

    assert database.read_bytes() == (b"modified" if mutation == "content" else b"replacement")
    if mutation == "replacement":
        assert displaced.read_bytes() == b"original"
    assert not any(path.exists() for path in _sqlite_paths(database)[1:])


def test_parent_replacement_invalidates_the_location_without_mutating_either_parent(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    database = parent / "shadow.sqlite3"
    location = inspect_private_file_location(database)
    displaced = tmp_path / "displaced"
    parent.rename(displaced)
    replacement = _private_directory(tmp_path / "store")

    with pytest.raises(SecureFileError):
        _claim(location)

    assert not any(path.exists() for path in _sqlite_paths(displaced / database.name))
    assert not any(path.exists() for path in _sqlite_paths(replacement / database.name))


def test_claim_rejects_an_unsafe_sidecar_and_cleans_only_its_own_placeholder(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    database = _private_file(parent / "shadow.sqlite3", b"database")
    location = inspect_private_file_location(database)
    _database, wal, shm, journal = _sqlite_paths(database)
    victim = _private_file(parent / "victim", b"do-not-touch")
    shm.symlink_to(victim)

    with pytest.raises(SecureFileError):
        _claim(location)

    assert database.read_bytes() == b"database"
    assert victim.read_bytes() == b"do-not-touch"
    assert shm.is_symlink()
    assert shm.resolve() == victim
    assert not wal.exists()
    assert not journal.exists()


def test_claim_rejects_non_location_authorizations_without_mutation(tmp_path: Path) -> None:
    first_parent = _private_directory(tmp_path / "first")
    existing = first_parent / "existing.sqlite3"
    sqlite_authorization = authorize_private_sqlite_path(existing)
    before = _sqlite_snapshot(existing)
    second_parent = _private_directory(tmp_path / "second")
    untouched = second_parent / "untouched.sqlite3"

    with pytest.raises(SecureFileError):
        _claim(sqlite_authorization)
    with pytest.raises(SecureFileError):
        _claim(object())  # type: ignore[arg-type]

    assert _sqlite_snapshot(existing) == before
    assert not any(path.exists() for path in _sqlite_paths(untouched))
