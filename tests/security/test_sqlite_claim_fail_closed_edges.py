from __future__ import annotations

import os
import stat
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import saliencegate.security.files as files_module
from saliencegate.security.files import (
    SecureFileError,
    StableFileAuthorization,
    StableReadPolicy,
    inspect_private_file_location,
    read_stable_file,
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


def _inspect_sqlite_namespace(path: Path) -> tuple[StableFileAuthorization, ...]:
    return tuple(inspect_private_file_location(candidate) for candidate in _sqlite_paths(path))


def _claim_namespace(
    locations: tuple[StableFileAuthorization, ...],
) -> StableFileAuthorization:
    return files_module._claim_private_sqlite_location(
        locations[0],
        sidecar_locations=locations[1:],
    )


def _snapshot(paths: tuple[Path, ...]) -> tuple[tuple[bytes, int, int, int], ...]:
    return tuple(
        (
            path.read_bytes(),
            path.stat().st_dev,
            path.stat().st_ino,
            stat.S_IMODE(path.stat().st_mode),
        )
        for path in paths
    )


def _different_stable(
    identity: files_module._StableIdentity,
) -> files_module._StableIdentity:
    return replace(identity, inode=identity.inode + 1)


def _different_complete(
    identity: files_module._CompleteIdentity,
) -> files_module._CompleteIdentity:
    return replace(identity, stable=_different_stable(identity.stable))


def test_claim_rejects_a_location_with_inconsistent_target_identities(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    database = _private_file(parent / "shadow.sqlite3", b"database")
    location = inspect_private_file_location(database)
    complete = location._target_complete_identity
    assert complete is not None
    forged = replace(location, _target_identity=_different_stable(complete.stable))

    with pytest.raises(SecureFileError):
        files_module._claim_private_sqlite_location(forged)

    assert database.read_bytes() == b"database"
    assert not any(path.exists() for path in _sqlite_paths(database)[1:])


def test_claim_rejects_a_location_whose_copied_path_is_not_canonical(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    database = parent / "shadow.sqlite3"
    location = inspect_private_file_location(database)
    forged = replace(location, path=database.name)

    with pytest.raises(SecureFileError):
        files_module._claim_private_sqlite_location(forged)

    assert not any(path.exists() for path in _sqlite_paths(database))


def test_claim_rejects_an_incomplete_sidecar_snapshot_without_mutation(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    database = parent / "shadow.sqlite3"
    locations = _inspect_sqlite_namespace(database)

    with pytest.raises(SecureFileError):
        files_module._claim_private_sqlite_location(
            locations[0],
            sidecar_locations=locations[1:2],
        )

    assert not any(path.exists() for path in _sqlite_paths(database))


def test_claim_rejects_a_sidecar_snapshot_bound_to_the_wrong_name(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    database = parent / "shadow.sqlite3"
    locations = _inspect_sqlite_namespace(database)
    forged_wal = replace(locations[1], path=f"{database}-other")

    with pytest.raises(SecureFileError):
        files_module._claim_private_sqlite_location(
            locations[0],
            sidecar_locations=(forged_wal, *locations[2:]),
        )

    assert not any(path.exists() for path in _sqlite_paths(database))


def test_claim_rejects_a_sidecar_with_inconsistent_target_identities(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    database = parent / "shadow.sqlite3"
    locations = _inspect_sqlite_namespace(database)
    parent_identity = locations[1]._parent_identity
    assert parent_identity is not None
    forged_wal = replace(locations[1], _target_identity=parent_identity)

    with pytest.raises(SecureFileError):
        files_module._claim_private_sqlite_location(
            locations[0],
            sidecar_locations=(forged_wal, *locations[2:]),
        )

    assert not any(path.exists() for path in _sqlite_paths(database))


@pytest.mark.parametrize("target_validation", (1, 2, 3))
def test_claim_rejects_target_identity_drift_at_every_validation_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_validation: int,
) -> None:
    parent = _private_directory(tmp_path / "store")
    database = parent / "shadow.sqlite3"
    paths = _sqlite_paths(database)
    for index, path in enumerate(paths):
        _private_file(path, f"existing-{index}".encode())
    before = _snapshot(paths)
    locations = _inspect_sqlite_namespace(database)
    real_validate = files_module._validate_private_location_target
    calls = 0

    def drift_target(
        directory_fd: int,
        name: str,
        *,
        expected: files_module._CompleteIdentity | None = None,
    ) -> files_module._CompleteIdentity:
        nonlocal calls
        result = real_validate(directory_fd, name, expected=expected)
        if name == database.name:
            calls += 1
            if calls == target_validation:
                return _different_complete(result)
        return result

    monkeypatch.setattr(files_module, "_validate_private_location_target", drift_target)

    with pytest.raises(SecureFileError):
        _claim_namespace(locations)

    assert calls == target_validation
    assert _snapshot(paths) == before


@pytest.mark.parametrize("sidecar_validation", (1, 2))
def test_claim_rejects_sidecar_identity_drift_at_both_validation_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sidecar_validation: int,
) -> None:
    parent = _private_directory(tmp_path / "store")
    database = parent / "shadow.sqlite3"
    paths = _sqlite_paths(database)
    for index, path in enumerate(paths):
        _private_file(path, f"existing-{index}".encode())
    before = _snapshot(paths)
    locations = _inspect_sqlite_namespace(database)
    real_validate = files_module._validate_private_location_target
    wal_name = f"{database.name}-wal"
    calls = 0

    def drift_wal(
        directory_fd: int,
        name: str,
        *,
        expected: files_module._CompleteIdentity | None = None,
    ) -> files_module._CompleteIdentity:
        nonlocal calls
        result = real_validate(directory_fd, name, expected=expected)
        if name == wal_name:
            calls += 1
            if calls == sidecar_validation:
                return _different_complete(result)
        return result

    monkeypatch.setattr(files_module, "_validate_private_location_target", drift_wal)

    with pytest.raises(SecureFileError):
        _claim_namespace(locations)

    assert calls == sidecar_validation
    assert _snapshot(paths) == before


def test_failed_sidecar_creation_removes_only_the_claim_owned_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "store")
    database = parent / "shadow.sqlite3"
    paths = _sqlite_paths(database)
    locations = _inspect_sqlite_namespace(database)
    real_create = files_module._create_target

    def fail_wal_creation(
        directory_fd: int,
        name: str,
    ) -> files_module._CompleteIdentity:
        if name == f"{database.name}-wal":
            raise OSError("injected sidecar failure")
        return real_create(directory_fd, name)

    monkeypatch.setattr(files_module, "_create_target", fail_wal_creation)

    with pytest.raises(SecureFileError):
        _claim_namespace(locations)

    assert not any(path.exists() for path in paths)


@pytest.mark.parametrize("preexisting", (False, True))
def test_final_parent_close_failure_cleans_only_claim_owned_placeholders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preexisting: bool,
) -> None:
    parent = _private_directory(tmp_path / "store")
    database = parent / "shadow.sqlite3"
    paths = _sqlite_paths(database)
    before: tuple[tuple[bytes, int, int, int], ...] | None = None
    if preexisting:
        for index, path in enumerate(paths):
            _private_file(path, f"existing-{index}".encode())
        before = _snapshot(paths)
    locations = _inspect_sqlite_namespace(database)
    real_validate_sidecars = files_module._validate_sqlite_sidecars
    real_close = os.close
    armed = False
    injected = False

    def arm_after_final_validation(*args: object, **kwargs: object) -> None:
        nonlocal armed
        real_validate_sidecars(*args, **kwargs)  # type: ignore[arg-type]
        armed = True

    def fail_final_parent_close(descriptor: int) -> None:
        nonlocal injected
        metadata = os.fstat(descriptor)
        if armed and stat.S_ISDIR(metadata.st_mode) and not injected:
            injected = True
            real_close(descriptor)
            raise OSError("injected parent close failure")
        real_close(descriptor)

    monkeypatch.setattr(files_module, "_validate_sqlite_sidecars", arm_after_final_validation)
    monkeypatch.setattr(os, "close", fail_final_parent_close)

    with pytest.raises(SecureFileError):
        _claim_namespace(locations)

    assert injected is True
    if before is None:
        assert not any(path.exists() for path in paths)
    else:
        assert _snapshot(paths) == before


def test_close_failure_never_cleans_through_a_replaced_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "store")
    database = parent / "shadow.sqlite3"
    locations = _inspect_sqlite_namespace(database)
    real_validate_sidecars = files_module._validate_sqlite_sidecars
    real_close = os.close
    displaced = tmp_path / "displaced"
    armed = False
    injected = False

    def arm_after_final_validation(*args: object, **kwargs: object) -> None:
        nonlocal armed
        real_validate_sidecars(*args, **kwargs)  # type: ignore[arg-type]
        armed = True

    def replace_parent_then_fail_close(descriptor: int) -> None:
        nonlocal injected
        metadata = os.fstat(descriptor)
        if armed and stat.S_ISDIR(metadata.st_mode) and not injected:
            injected = True
            real_close(descriptor)
            parent.rename(displaced)
            _private_directory(tmp_path / "store")
            raise OSError("injected parent close failure")
        real_close(descriptor)

    monkeypatch.setattr(files_module, "_validate_sqlite_sidecars", arm_after_final_validation)
    monkeypatch.setattr(os, "close", replace_parent_then_fail_close)

    with pytest.raises(SecureFileError):
        _claim_namespace(locations)

    assert injected is True
    assert all(path.read_bytes() == b"" for path in _sqlite_paths(displaced / database.name))
    assert not any(path.exists() for path in _sqlite_paths(tmp_path / "store" / database.name))


def test_sqlite_revalidation_rejects_an_incomplete_capability(tmp_path: Path) -> None:
    parent = _private_directory(tmp_path / "store")
    database = parent / "shadow.sqlite3"
    authorization = _claim_namespace(_inspect_sqlite_namespace(database))
    forged = replace(authorization, _target_identity=None)

    with pytest.raises(SecureFileError):
        forged.revalidate()


def test_sqlite_revalidation_rechecks_target_identity_after_parent_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "store")
    database = parent / "shadow.sqlite3"
    authorization = _claim_namespace(_inspect_sqlite_namespace(database))
    real_validate = files_module._validate_existing_target
    calls = 0

    def drift_after_parent_check(directory_fd: int, name: str, **kwargs: object) -> object:
        nonlocal calls
        result = real_validate(directory_fd, name, **kwargs)
        if name == database.name:
            calls += 1
            if calls == 2:
                return _different_stable(result)
        return result

    monkeypatch.setattr(files_module, "_validate_existing_target", drift_after_parent_check)

    with pytest.raises(SecureFileError):
        authorization.revalidate()

    assert calls == 2


@pytest.mark.parametrize("missing_field", ("target", "parent"))
def test_private_read_revalidation_rejects_an_incomplete_capability(
    tmp_path: Path,
    missing_field: str,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_file(parent / "trace.json", b"{}")
    stable = read_stable_file(
        target,
        maximum_bytes=2,
        policy=StableReadPolicy.PRIVATE_OWNER,
    )
    if missing_field == "target":
        forged = replace(stable.authorization, _target_identity=None)
    else:
        forged = replace(stable.authorization, _parent_identity=None)

    with pytest.raises(SecureFileError):
        forged.revalidate()


def test_private_read_revalidation_rejects_parent_replacement(tmp_path: Path) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_file(parent / "trace.json", b"{}")
    stable = read_stable_file(
        target,
        maximum_bytes=2,
        policy=StableReadPolicy.PRIVATE_OWNER,
    )
    displaced = tmp_path / "displaced"
    parent.rename(displaced)
    _private_directory(tmp_path / "store")

    with pytest.raises(SecureFileError):
        stable.authorization.revalidate()

    assert (displaced / target.name).read_bytes() == b"{}"


def test_claim_rejects_a_non_tuple_sidecar_snapshot_without_mutation(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "store")
    database = parent / "shadow.sqlite3"
    locations = _inspect_sqlite_namespace(database)

    with pytest.raises(SecureFileError):
        files_module._claim_private_sqlite_location(
            locations[0],
            sidecar_locations=cast(
                tuple[StableFileAuthorization, ...],
                list(locations[1:]),
            ),
        )

    assert not any(path.exists() for path in _sqlite_paths(database))
