from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from saliencegate.artifacts import exclusive as exclusive_module
from saliencegate.artifacts.exclusive import (
    ExclusiveStorageError,
    LockedFlatDirectory,
    open_locked_flat_directory,
)
from saliencegate.artifacts.tree import (
    ArtifactDestinationError,
    ArtifactExistsError,
    ArtifactExportError,
    ClosedTreeDescriptor,
    ClosedTreeFileSpec,
    publish_closed_tree_exclusive,
    read_closed_tree,
)
from saliencegate.domain import canonical_json

_MANIFEST_NAME = "manifest.json"
_MANIFEST_MAXIMUM_BYTES = 4096


def _fixture_files(*, generation: int = 1) -> dict[str, bytes]:
    alpha = canonical_json({"generation": generation, "key": "alpha"})
    beta = canonical_json({"generation": generation, "key": "beta"})
    manifest = canonical_json(
        {
            "files": [
                {
                    "expected_bytes": len(beta),
                    "key": "beta",
                    "maximum_bytes": 256,
                    "name": "beta.json",
                },
                {
                    "expected_bytes": len(alpha),
                    "key": "alpha",
                    "maximum_bytes": 256,
                    "name": "alpha.json",
                },
            ],
            "generation": generation,
            "replacement_key": "exclusive-tree",
            "schema_version": "exclusive-tree-test/v1",
        }
    )
    return {
        "beta.json": beta,
        _MANIFEST_NAME: manifest,
        "alpha.json": alpha,
    }


def _parse_manifest(data: bytes) -> ClosedTreeDescriptor:
    value = json.loads(data)
    if type(value) is not dict or canonical_json(value) != data:
        raise ValueError("manifest is not canonical")
    raw_files = value.get("files")
    if type(raw_files) is not list:
        raise ValueError("manifest files are invalid")
    files = tuple(
        ClosedTreeFileSpec(
            key=item["key"],
            name=item["name"],
            maximum_bytes=item["maximum_bytes"],
            expected_bytes=item["expected_bytes"],
        )
        for item in raw_files
    )
    return ClosedTreeDescriptor(
        manifest=value,
        manifest_name=_MANIFEST_NAME,
        manifest_digest=sha256(data).hexdigest(),
        replacement_key=value["replacement_key"],
        files=files,
    )


def _parse_file(key: object, data: bytes) -> object:
    value = json.loads(data)
    if type(value) is not dict or canonical_json(value) != data or value.get("key") != key:
        raise ValueError("component is invalid")
    return value


def _validate_tree(path: Path, expected_digest: str | None) -> ClosedTreeDescriptor:
    loaded = read_closed_tree(
        path / _MANIFEST_NAME,
        maximum_manifest_bytes=_MANIFEST_MAXIMUM_BYTES,
        parse_manifest=_parse_manifest,
        parse_file=_parse_file,
        finish=lambda manifest, parsed: (manifest, parsed),
    )
    if expected_digest is not None and loaded.manifest_digest != expected_digest:
        raise ValueError("tree digest does not match")
    return loaded.descriptor


def _publish(
    root: Path,
    files: Mapping[str, bytes],
    *,
    validate_tree: Callable[[Path, str | None], ClosedTreeDescriptor] = _validate_tree,
) -> ClosedTreeDescriptor:
    return publish_closed_tree_exclusive(
        root,
        files,
        manifest_name=_MANIFEST_NAME,
        maximum_manifest_bytes=_MANIFEST_MAXIMUM_BYTES,
        parse_manifest=_parse_manifest,
        validate_tree=validate_tree,
    )


def _tree(root: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(root.iterdir())}


def _flat_snapshot(root: Path) -> tuple[tuple[str, int, int, int, bytes | None], ...]:
    snapshot: list[tuple[str, int, int, int, bytes | None]] = []
    for path in sorted(root.iterdir()):
        metadata = path.lstat()
        data = path.read_bytes() if stat.S_ISREG(metadata.st_mode) else None
        snapshot.append((path.name, metadata.st_mode, metadata.st_ino, metadata.st_nlink, data))
    return tuple(snapshot)


def _wait_for_path(path: Path, *, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.01)
    return path.exists()


def _metadata_with(metadata: os.stat_result, **changes: object) -> object:
    values = {
        name: getattr(metadata, name)
        for name in (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_mode",
            "st_nlink",
            "st_uid",
        )
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_exclusive_publish_is_owner_only_manifest_last_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "review-pack"
    files = _fixture_files()
    created: list[str] = []
    real_create = exclusive_module._create_regular_exclusive

    def record_create(directory_fd: int, name: str, data: bytes) -> object:
        created.append(name)
        return real_create(directory_fd, name, data)

    monkeypatch.setattr(exclusive_module, "_create_regular_exclusive", record_create)

    descriptor = _publish(root, files)

    assert descriptor == _parse_manifest(files[_MANIFEST_NAME])
    assert _tree(root) == files
    assert created == ["alpha.json", "beta.json", _MANIFEST_NAME]
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in root.iterdir())

    before = {
        path.name: (path.stat().st_dev, path.stat().st_ino, path.stat().st_mtime_ns)
        for path in (root, *sorted(root.iterdir()))
    }
    created.clear()
    assert _publish(root, dict(reversed(tuple(files.items())))) == descriptor
    assert created == []
    assert _tree(root) == files
    assert {
        path.name: (path.stat().st_dev, path.stat().st_ino, path.stat().st_mtime_ns)
        for path in (root, *sorted(root.iterdir()))
    } == before


def test_exclusive_publish_refuses_a_different_complete_destination_unchanged(
    tmp_path: Path,
) -> None:
    root = tmp_path / "review-pack"
    original = _fixture_files(generation=1)
    _publish(root, original)

    try:
        _publish(root, _fixture_files(generation=2))
    except ArtifactExistsError:
        pass
    else:  # pragma: no cover - the assertion below communicates the contract
        raise AssertionError("different destination unexpectedly accepted")

    assert _tree(root) == original


def test_exclusive_publish_rejects_every_unsafe_existing_shape_without_mutation(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"{}")
    outside.chmod(0o600)

    for case in ("incomplete", "extra", "file-mode", "root-mode", "symlink", "hardlink"):
        root = tmp_path / case
        files = _fixture_files()
        _publish(root, files)
        if case == "incomplete":
            (root / _MANIFEST_NAME).unlink()
        elif case == "extra":
            (root / "extra.json").write_bytes(b"{}")
            (root / "extra.json").chmod(0o600)
        elif case == "file-mode":
            (root / "alpha.json").chmod(0o640)
        elif case == "root-mode":
            root.chmod(0o750)
        elif case == "symlink":
            (root / "alpha.json").unlink()
            (root / "alpha.json").symlink_to(outside)
        else:
            os.link(root / "alpha.json", tmp_path / f"{case}-alias.json")
        before = _flat_snapshot(root)

        with pytest.raises(ArtifactExistsError):
            _publish(root, files)

        assert _flat_snapshot(root) == before


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO semantics")
def test_exclusive_publish_rejects_a_fifo_child_without_blocking_or_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fifo-pack"
    files = _fixture_files()
    _publish(root, files)
    (root / "alpha.json").unlink()
    os.mkfifo(root / "alpha.json", mode=0o600)
    before = _flat_snapshot(root)

    with pytest.raises(ArtifactExistsError):
        _publish(root, files)

    assert _flat_snapshot(root) == before


def test_exclusive_publish_rejects_a_wrong_owner_child_value_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "wrong-owner-pack"
    files = _fixture_files()
    _publish(root, files)
    target_inode = (root / "alpha.json").stat().st_ino
    real_owner = exclusive_module._current_owner

    def reject_target(metadata: os.stat_result) -> bool:
        return metadata.st_ino != target_inode and real_owner(metadata)

    monkeypatch.setattr(exclusive_module, "_current_owner", reject_target)

    with pytest.raises(ArtifactExistsError) as error:
        _publish(root, files)

    assert "alpha" not in str(error.value)
    assert _tree(root) == files


def test_exclusive_publish_fails_unsupported_before_destination_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "unsupported"
    monkeypatch.setattr(exclusive_module, "_required_posix_primitives_available", lambda: False)

    with pytest.raises(ArtifactDestinationError):
        _publish(root, _fixture_files())

    assert not root.exists()


def test_exclusive_publish_snapshots_hostile_input_before_destination_io(
    tmp_path: Path,
) -> None:
    root = tmp_path / "hostile-input"
    files = _fixture_files()

    class HostileMapping(Mapping[str, bytes]):
        def __getitem__(self, key: str) -> bytes:
            return files[key]

        def __iter__(self):
            yield _MANIFEST_NAME
            raise RuntimeError("fixture-secret mapping changed")

        def __len__(self) -> int:
            return len(files)

    with pytest.raises(ArtifactExportError) as error:
        _publish(root, HostileMapping())

    assert "fixture-secret" not in str(error.value)
    assert not root.exists()


@pytest.mark.parametrize("invalid_path", ("bad\0parent/pack", "bad-\ud800/pack"))
def test_exclusive_publish_maps_unencodable_paths_value_free(
    tmp_path: Path,
    invalid_path: str,
) -> None:
    root = tmp_path / invalid_path

    with pytest.raises(ArtifactDestinationError) as error:
        _publish(root, _fixture_files())

    assert "bad" not in str(error.value)
    assert "bad" not in repr(error.value)


def test_two_different_exclusive_publishers_never_mix_bytes(tmp_path: Path) -> None:
    root = tmp_path / "contended"
    first = _fixture_files(generation=1)
    second = _fixture_files(generation=2)
    barrier = threading.Barrier(2)

    def publish(files: Mapping[str, bytes]) -> Exception | None:
        barrier.wait(timeout=2)
        try:
            _publish(root, files)
        except Exception as error:
            return error
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(publish, (first, second)))

    assert sum(error is None for error in results) == 1
    assert all(error is None or isinstance(error, ArtifactExistsError) for error in results)
    assert _tree(root) in (first, second)


@pytest.mark.parametrize("mutation", ("root-swap", "same-byte-rewrite"))
def test_exclusive_publish_rejects_identity_changes_during_validation(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / "review-pack"
    files = _fixture_files()
    saved = tmp_path / "saved"
    intruder = tmp_path / "intruder"
    if mutation == "root-swap":
        _publish(intruder, files)

    def mutate_during_validation(
        path: Path,
        expected_digest: str | None,
    ) -> ClosedTreeDescriptor:
        descriptor = _validate_tree(path, expected_digest)
        if mutation == "root-swap":
            os.rename(path, saved)
            os.rename(intruder, path)
        else:
            alpha = path / "alpha.json"
            alpha.write_bytes(alpha.read_bytes())
            alpha.chmod(0o600)
        return descriptor

    with pytest.raises(ArtifactDestinationError):
        _publish(root, files, validate_tree=mutate_during_validation)


def test_exclusive_publish_rejects_parent_identity_change_during_validation(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir(mode=0o700)
    root = parent / "review-pack"
    saved_parent = tmp_path / "saved-parent"

    def swap_parent(
        path: Path,
        expected_digest: str | None,
    ) -> ClosedTreeDescriptor:
        descriptor = _validate_tree(path, expected_digest)
        os.rename(parent, saved_parent)
        parent.mkdir(mode=0o700)
        return descriptor

    with pytest.raises(ArtifactDestinationError):
        _publish(root, _fixture_files(), validate_tree=swap_parent)


def test_exclusive_publish_restats_every_file_after_the_last_validation_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "review-pack"
    files = _fixture_files()
    real_read = exclusive_module._read_regular_exact
    alpha_reads = 0

    def rewrite_after_second_alpha(
        directory_fd: int,
        name: str,
        expected: bytes,
        **kwargs: object,
    ) -> object:
        nonlocal alpha_reads
        identity = real_read(directory_fd, name, expected, **kwargs)
        if name == "alpha.json":
            alpha_reads += 1
            if alpha_reads == 2:
                alpha = root / name
                alpha.write_bytes(alpha.read_bytes())
                alpha.chmod(0o600)
        return identity

    monkeypatch.setattr(exclusive_module, "_read_regular_exact", rewrite_after_second_alpha)

    with pytest.raises(ArtifactDestinationError):
        _publish(root, files)


def test_existing_exclusive_publish_reestablishes_every_durability_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "review-pack"
    files = _fixture_files()
    _publish(root, files)
    expected_inodes = {
        root.stat().st_ino,
        root.parent.stat().st_ino,
        *(path.stat().st_ino for path in root.iterdir()),
    }
    synced_inodes: set[int] = set()
    real_fsync = exclusive_module.os.fsync

    def record_fsync(descriptor: int) -> None:
        synced_inodes.add(os.fstat(descriptor).st_ino)
        real_fsync(descriptor)

    monkeypatch.setattr(exclusive_module.os, "fsync", record_fsync)

    _publish(root, files)

    assert expected_inodes <= synced_inodes


def test_exclusive_adapter_sanitizes_a_final_descriptor_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "review-pack"
    real_verify_parent = exclusive_module._verify_parent_identity
    real_close = exclusive_module.os.close
    parent_checks = 0
    armed = False

    def verify_parent(*args: object, **kwargs: object) -> None:
        nonlocal parent_checks, armed
        real_verify_parent(*args, **kwargs)
        parent_checks += 1
        armed = parent_checks >= 2

    def fail_final_close(descriptor: int) -> None:
        nonlocal armed
        if armed:
            armed = False
            real_close(descriptor)
            raise OSError("fixture-secret close")
        real_close(descriptor)

    monkeypatch.setattr(exclusive_module, "_verify_parent_identity", verify_parent)
    monkeypatch.setattr(exclusive_module.os, "close", fail_final_close)

    with pytest.raises(ArtifactDestinationError) as error:
        _publish(root, _fixture_files())

    assert "fixture-secret" not in str(error.value)
    assert "fixture-secret" not in repr(error.value)


def test_exclusive_publish_preserves_an_incomplete_crash_and_never_repairs_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "interrupted"
    fresh = tmp_path / "fresh"
    files = _fixture_files()
    real_create = exclusive_module._create_regular_exclusive

    def fail_before_manifest(directory_fd: int, name: str, data: bytes) -> object:
        if name == _MANIFEST_NAME:
            raise OSError("fixture-secret crash before completion marker")
        return real_create(directory_fd, name, data)

    monkeypatch.setattr(exclusive_module, "_create_regular_exclusive", fail_before_manifest)

    with pytest.raises(ArtifactDestinationError) as interrupted:
        _publish(root, files)

    assert "fixture-secret" not in str(interrupted.value)
    assert root.is_dir()
    assert _tree(root) == {
        "alpha.json": files["alpha.json"],
        "beta.json": files["beta.json"],
    }
    incomplete = _tree(root)

    monkeypatch.setattr(exclusive_module, "_create_regular_exclusive", real_create)
    with pytest.raises(ArtifactExistsError):
        _publish(root, files)

    assert _tree(root) == incomplete
    _publish(fresh, files)
    assert _tree(fresh) == files


def test_locked_flat_directory_is_bounded_append_only_and_invalid_after_exit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reviews"
    handle: LockedFlatDirectory

    with open_locked_flat_directory(
        root,
        create=True,
        maximum_entries=8,
        maximum_file_bytes=1024,
    ) as locked:
        handle = locked
        assert locked.names == ()
        locked.create_regular_exclusive("first.json", b"{}", maximum_bytes=16)
        assert locked.names == ("first.json",)
        assert locked.read_regular("first.json", maximum_bytes=16) == b"{}"
        with pytest.raises(ExclusiveStorageError):
            locked.create_regular_exclusive("first.json", b"{}", maximum_bytes=16)

    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "review.lock").stat().st_mode) == 0o600
    assert stat.S_IMODE((root / "first.json").stat().st_mode) == 0o600
    with pytest.raises(ExclusiveStorageError):
        handle.read_regular("first.json", maximum_bytes=16)

    with open_locked_flat_directory(
        root,
        create=False,
        maximum_entries=8,
        maximum_file_bytes=1024,
    ) as reopened:
        assert reopened.names == ("first.json",)
        assert reopened.read_regular("first.json", maximum_bytes=16) == b"{}"


def test_locked_flat_directory_rejects_missing_lock_without_repair(tmp_path: Path) -> None:
    root = tmp_path / "reviews"
    root.mkdir(mode=0o700)

    with (
        pytest.raises(ExclusiveStorageError),
        open_locked_flat_directory(
            root,
            create=True,
            maximum_entries=8,
            maximum_file_bytes=1024,
        ),
    ):
        raise AssertionError("unsafe directory unexpectedly opened")

    assert tuple(root.iterdir()) == ()


@pytest.mark.parametrize("invalid_path", ("bad\0parent/reviews", "bad-\ud800/reviews"))
def test_locked_flat_directory_maps_unencodable_paths_value_free(
    tmp_path: Path,
    invalid_path: str,
) -> None:
    root = tmp_path / invalid_path

    with (
        pytest.raises(ExclusiveStorageError) as error,
        open_locked_flat_directory(
            root,
            create=True,
            maximum_entries=8,
            maximum_file_bytes=1024,
        ),
    ):
        raise AssertionError("invalid path unexpectedly opened")

    assert "bad" not in str(error.value)
    assert "bad" not in repr(error.value)


def test_locked_flat_directory_sanitizes_an_unexpected_pathlike_failure() -> None:
    class HostilePath(os.PathLike[str]):
        def __fspath__(self) -> str:
            raise RuntimeError("fixture-secret path")

    with (
        pytest.raises(ExclusiveStorageError) as error,
        open_locked_flat_directory(
            HostilePath(),
            create=True,
            maximum_entries=8,
            maximum_file_bytes=1024,
        ),
    ):
        raise AssertionError("hostile path unexpectedly opened")

    assert "fixture-secret" not in str(error.value)
    assert "fixture-secret" not in repr(error.value)


def test_locked_flat_directory_accepts_the_full_review_history_bound(tmp_path: Path) -> None:
    root = tmp_path / "reviews"

    with open_locked_flat_directory(
        root,
        create=True,
        maximum_entries=23_040,
        maximum_file_bytes=320 * 1024 + 1,
    ) as locked:
        assert locked.names == ()

    with (
        pytest.raises(ExclusiveStorageError),
        open_locked_flat_directory(
            root,
            create=False,
            maximum_entries=65_537,
            maximum_file_bytes=320 * 1024 + 1,
        ),
    ):
        raise AssertionError("invalid history bound unexpectedly accepted")


@pytest.mark.parametrize(
    "case",
    ("nested", "symlink", "hardlink", "mode", "oversized", "unsafe-name"),
)
def test_locked_flat_directory_rejects_unsafe_inventory_unchanged(
    tmp_path: Path,
    case: str,
) -> None:
    root = tmp_path / case
    with open_locked_flat_directory(
        root,
        create=True,
        maximum_entries=8,
        maximum_file_bytes=1024,
    ) as locked:
        locked.create_regular_exclusive("first.json", b"{}", maximum_bytes=16)

    if case == "nested":
        (root / "nested").mkdir()
    elif case == "symlink":
        (root / "linked.json").symlink_to(root / "first.json")
    elif case == "hardlink":
        os.link(root / "first.json", tmp_path / "alias.json")
    elif case == "mode":
        (root / "first.json").chmod(0o640)
    elif case == "oversized":
        (root / "first.json").write_bytes(b"x" * 1025)
        (root / "first.json").chmod(0o600)
    else:
        (root / "Unsafe.json").write_bytes(b"{}")
        (root / "Unsafe.json").chmod(0o600)
    before = _flat_snapshot(root)

    with (
        pytest.raises(ExclusiveStorageError),
        open_locked_flat_directory(
            root,
            create=False,
            maximum_entries=8,
            maximum_file_bytes=1024,
        ),
    ):
        raise AssertionError("unsafe directory unexpectedly opened")

    assert _flat_snapshot(root) == before


@pytest.mark.parametrize("target", ("entry", "lock"))
@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO semantics")
def test_locked_flat_directory_rejects_fifo_entries_without_blocking(
    tmp_path: Path,
    target: str,
) -> None:
    root = tmp_path / target
    with open_locked_flat_directory(
        root,
        create=True,
        maximum_entries=8,
        maximum_file_bytes=1024,
    ):
        pass
    name = "review.lock" if target == "lock" else "pipe.json"
    if target == "lock":
        (root / name).unlink()
    os.mkfifo(root / name, mode=0o600)
    before = _flat_snapshot(root)

    with (
        pytest.raises(ExclusiveStorageError),
        open_locked_flat_directory(
            root,
            create=False,
            maximum_entries=8,
            maximum_file_bytes=1024,
        ),
    ):
        raise AssertionError("FIFO unexpectedly opened")

    assert _flat_snapshot(root) == before


def test_locked_flat_directory_rejects_wrong_owner_entry_value_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "wrong-owner"
    with open_locked_flat_directory(
        root,
        create=True,
        maximum_entries=8,
        maximum_file_bytes=1024,
    ) as locked:
        locked.create_regular_exclusive("first.json", b"{}", maximum_bytes=16)
    target_inode = (root / "first.json").stat().st_ino
    real_owner = exclusive_module._current_owner

    def reject_target(metadata: os.stat_result) -> bool:
        return metadata.st_ino != target_inode and real_owner(metadata)

    monkeypatch.setattr(exclusive_module, "_current_owner", reject_target)

    with (
        pytest.raises(ExclusiveStorageError) as error,
        open_locked_flat_directory(
            root,
            create=False,
            maximum_entries=8,
            maximum_file_bytes=1024,
        ),
    ):
        raise AssertionError("wrong owner unexpectedly accepted")

    assert "first" not in str(error.value)


def test_locked_flat_directory_detects_inventory_change_before_unlock(tmp_path: Path) -> None:
    root = tmp_path / "reviews"

    with (
        pytest.raises(ExclusiveStorageError),
        open_locked_flat_directory(
            root,
            create=True,
            maximum_entries=8,
            maximum_file_bytes=1024,
        ) as locked,
    ):
        locked.create_regular_exclusive("first.json", b"{}", maximum_bytes=16)
        (root / "intruder.json").write_bytes(b"{}")
        (root / "intruder.json").chmod(0o600)

    assert (root / "first.json").read_bytes() == b"{}"
    assert (root / "intruder.json").read_bytes() == b"{}"


def test_locked_flat_directory_verifies_mutation_even_when_body_raises(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reviews"

    with (
        pytest.raises(ExclusiveStorageError) as error,
        open_locked_flat_directory(
            root,
            create=True,
            maximum_entries=8,
            maximum_file_bytes=1024,
        ),
    ):
        (root / "intruder.json").write_bytes(b"{}")
        (root / "intruder.json").chmod(0o600)
        raise RuntimeError("fixture-secret body failure")

    assert "fixture-secret" not in str(error.value)
    assert "fixture-secret" not in repr(error.value)


def test_locked_flat_directory_sanitizes_unexpected_scan_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reviews"
    with open_locked_flat_directory(
        root,
        create=True,
        maximum_entries=8,
        maximum_file_bytes=1024,
    ):
        pass

    monkeypatch.setattr(
        exclusive_module.os,
        "scandir",
        lambda descriptor: (_ for _ in ()).throw(RuntimeError("fixture-secret scandir")),
    )

    with (
        pytest.raises(ExclusiveStorageError) as error,
        open_locked_flat_directory(
            root,
            create=False,
            maximum_entries=8,
            maximum_file_bytes=1024,
        ),
    ):
        raise AssertionError("hostile scan unexpectedly succeeded")

    assert "fixture-secret" not in str(error.value)
    assert "fixture-secret" not in repr(error.value)


def test_locked_flat_directory_enforces_per_operation_bounds_without_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reviews"

    with open_locked_flat_directory(
        root,
        create=True,
        maximum_entries=2,
        maximum_file_bytes=32,
    ) as locked:
        locked.create_regular_exclusive("first.json", b"x" * 16, maximum_bytes=16)
        before = _flat_snapshot(root)
        with pytest.raises(ExclusiveStorageError):
            locked.read_regular("first.json", maximum_bytes=15)
        with pytest.raises(ExclusiveStorageError):
            locked.create_regular_exclusive("second.json", b"x" * 17, maximum_bytes=16)
        assert _flat_snapshot(root) == before


def test_reopened_locked_directory_reestablishes_every_durability_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reviews"
    with open_locked_flat_directory(
        root,
        create=True,
        maximum_entries=8,
        maximum_file_bytes=1024,
    ) as locked:
        locked.create_regular_exclusive("first.json", b"{}", maximum_bytes=16)

    expected_inodes = {
        root.stat().st_ino,
        root.parent.stat().st_ino,
        (root / "review.lock").stat().st_ino,
        (root / "first.json").stat().st_ino,
    }
    synced_inodes: set[int] = set()
    real_fsync = exclusive_module.os.fsync

    def record_fsync(descriptor: int) -> None:
        synced_inodes.add(os.fstat(descriptor).st_ino)
        real_fsync(descriptor)

    monkeypatch.setattr(exclusive_module.os, "fsync", record_fsync)

    with open_locked_flat_directory(
        root,
        create=False,
        maximum_entries=8,
        maximum_file_bytes=1024,
    ):
        pass

    assert expected_inodes <= synced_inodes


def test_locked_directory_rechecks_the_named_lock_after_synchronization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reviews"
    with open_locked_flat_directory(
        root,
        create=True,
        maximum_entries=8,
        maximum_file_bytes=1024,
    ):
        pass

    real_synchronize = exclusive_module._synchronize_locked_directory

    def replace_lock_after_synchronization(*args: object, **kwargs: object) -> None:
        real_synchronize(*args, **kwargs)
        (root / "review.lock").unlink()
        descriptor = os.open(root / "review.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)

    monkeypatch.setattr(
        exclusive_module,
        "_synchronize_locked_directory",
        replace_lock_after_synchronization,
    )
    entered = False

    with (
        pytest.raises(ExclusiveStorageError),
        open_locked_flat_directory(
            root,
            create=False,
            maximum_entries=8,
            maximum_file_bytes=1024,
        ),
    ):
        entered = True

    assert not entered


def test_locked_reads_do_not_rescan_the_whole_history_quadratically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reviews"
    with open_locked_flat_directory(
        root,
        create=True,
        maximum_entries=8,
        maximum_file_bytes=1024,
    ) as locked:
        locked.create_regular_exclusive("first.json", b"{}", maximum_bytes=16)

    real_inventory = exclusive_module._locked_entry_identities
    inventory_scans = 0

    def count_inventory(*args: object, **kwargs: object) -> object:
        nonlocal inventory_scans
        inventory_scans += 1
        return real_inventory(*args, **kwargs)

    monkeypatch.setattr(exclusive_module, "_locked_entry_identities", count_inventory)

    with open_locked_flat_directory(
        root,
        create=False,
        maximum_entries=8,
        maximum_file_bytes=1024,
    ) as locked:
        scans_after_open = inventory_scans
        for _ in range(10):
            assert locked.read_regular("first.json", maximum_bytes=16) == b"{}"
        assert inventory_scans == scans_after_open

    assert inventory_scans == scans_after_open + 1


def test_locked_flat_directory_serializes_scan_and_rescans_after_waiting(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reviews"
    entered = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    future = None

    def reopen() -> tuple[str, ...]:
        with open_locked_flat_directory(
            root,
            create=False,
            maximum_entries=8,
            maximum_file_bytes=1024,
        ) as locked:
            entered.set()
            return locked.names

    try:
        with open_locked_flat_directory(
            root,
            create=True,
            maximum_entries=8,
            maximum_file_bytes=1024,
        ) as first:
            future = executor.submit(reopen)
            assert not entered.wait(timeout=0.1)
            first.create_regular_exclusive("first.json", b"{}", maximum_bytes=16)
        assert entered.wait(timeout=2)
        assert future.result(timeout=2) == ("first.json",)
    finally:
        executor.shutdown(wait=True)


def test_locked_flat_directory_rejects_parent_swap_before_unlock(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir(mode=0o700)
    root = parent / "reviews"
    saved_parent = tmp_path / "saved-parent"

    with (
        pytest.raises(ExclusiveStorageError),
        open_locked_flat_directory(
            root,
            create=True,
            maximum_entries=8,
            maximum_file_bytes=1024,
        ),
    ):
        os.rename(parent, saved_parent)
        parent.mkdir(mode=0o700)


def test_locked_flat_directory_applies_a_schema_neutral_entry_name_gate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reviews"

    def allowed(name: str) -> bool:
        return name.startswith("review--") and name.endswith(".json")

    with open_locked_flat_directory(
        root,
        create=True,
        maximum_entries=8,
        maximum_file_bytes=1024,
        entry_name_validator=allowed,
    ) as locked:
        locked.create_regular_exclusive("review--first.json", b"{}", maximum_bytes=16)
        before = _flat_snapshot(root)
        with pytest.raises(ExclusiveStorageError):
            locked.create_regular_exclusive("fork.json", b"{}", maximum_bytes=16)
        assert _flat_snapshot(root) == before

    (root / "fork.json").write_bytes(b"{}")
    (root / "fork.json").chmod(0o600)
    before = _flat_snapshot(root)
    with (
        pytest.raises(ExclusiveStorageError),
        open_locked_flat_directory(
            root,
            create=False,
            maximum_entries=8,
            maximum_file_bytes=1024,
            entry_name_validator=allowed,
        ),
    ):
        raise AssertionError("unexpected entry passed the name gate")
    assert _flat_snapshot(root) == before


def test_process_crash_releases_the_directory_lock(tmp_path: Path) -> None:
    root = tmp_path / "reviews"
    with open_locked_flat_directory(
        root,
        create=True,
        maximum_entries=8,
        maximum_file_bytes=1024,
    ):
        pass

    holder_marker = tmp_path / "holder-entered"
    contender_marker = tmp_path / "contender-entered"
    holder_code = """
import sys
import time
from pathlib import Path
from saliencegate.artifacts.exclusive import open_locked_flat_directory

with open_locked_flat_directory(
    sys.argv[1], create=False, maximum_entries=8, maximum_file_bytes=1024
):
    Path(sys.argv[2]).write_text("entered", encoding="utf-8")
    time.sleep(60)
"""
    contender_code = """
import sys
from pathlib import Path
from saliencegate.artifacts.exclusive import open_locked_flat_directory

with open_locked_flat_directory(
    sys.argv[1], create=False, maximum_entries=8, maximum_file_bytes=1024
):
    Path(sys.argv[2]).write_text("entered", encoding="utf-8")
"""
    holder = subprocess.Popen(
        [sys.executable, "-c", holder_code, str(root), str(holder_marker)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    contender: subprocess.Popen[bytes] | None = None
    try:
        assert _wait_for_path(holder_marker, timeout=5)
        contender = subprocess.Popen(
            [sys.executable, "-c", contender_code, str(root), str(contender_marker)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.2)
        assert not contender_marker.exists()

        holder.terminate()
        holder.wait(timeout=5)
        assert _wait_for_path(contender_marker, timeout=5)
        assert contender.wait(timeout=5) == 0
    finally:
        for process in (holder, contender):
            if process is None:
                continue
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)


def test_locked_directory_reports_cleanup_failure_and_releases_the_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reviews"
    with open_locked_flat_directory(
        root,
        create=True,
        maximum_entries=8,
        maximum_file_bytes=1024,
    ):
        pass

    lock_inode = (root / "review.lock").stat().st_ino
    real_close = exclusive_module.os.close
    armed = False

    def fail_after_closing_lock(descriptor: int) -> None:
        nonlocal armed
        inode = os.fstat(descriptor).st_ino
        real_close(descriptor)
        if armed and inode == lock_inode:
            armed = False
            raise OSError("fixture-secret cleanup")

    monkeypatch.setattr(exclusive_module.os, "close", fail_after_closing_lock)

    with (
        pytest.raises(ExclusiveStorageError) as error,
        open_locked_flat_directory(
            root,
            create=False,
            maximum_entries=8,
            maximum_file_bytes=1024,
        ),
    ):
        armed = True

    assert "fixture-secret" not in str(error.value)
    assert "fixture-secret" not in repr(error.value)

    contender_code = """
import sys
from saliencegate.artifacts.exclusive import open_locked_flat_directory

with open_locked_flat_directory(
    sys.argv[1], create=False, maximum_entries=8, maximum_file_bytes=1024
):
    pass
"""
    contender = subprocess.run(
        [sys.executable, "-c", contender_code, str(root)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
    )
    assert contender.returncode == 0


def test_exclusive_private_boundary_validators_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fcntl

    with monkeypatch.context() as patch:
        patch.setattr(fcntl, "flock", None)
        with pytest.raises(exclusive_module.ExclusiveStorageUnsupportedError):
            exclusive_module._require_posix_primitives()

    class BrokenMapping(Mapping[str, bytes]):
        def __getitem__(self, key: str) -> bytes:
            raise KeyError(key)

        def __iter__(self):
            raise RuntimeError("fixture-secret mapping")

        def __len__(self) -> int:
            return 1

    invalid_snapshots: tuple[tuple[Mapping[str, bytes], str], ...] = (
        (BrokenMapping(), _MANIFEST_NAME),
        ({}, _MANIFEST_NAME),
        ({_MANIFEST_NAME: b"{}", "Unsafe.json": b"{}"}, _MANIFEST_NAME),
    )
    for files, manifest_name in invalid_snapshots:
        with pytest.raises(ExclusiveStorageError) as error:
            exclusive_module._snapshot_files(files, manifest_name)
        assert "fixture-secret" not in str(error.value)

    for output in (b"fixture-secret", Path(".")):
        with pytest.raises(ExclusiveStorageError) as error:
            exclusive_module._safe_output(output)
        assert "fixture-secret" not in str(error.value)

    with pytest.raises(ExclusiveStorageError):
        exclusive_module._validate_entry_name(
            "review.json",
            lambda name: (_ for _ in ()).throw(RuntimeError(name)),
        )


def test_exclusive_scan_and_parent_helpers_sanitize_low_level_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "one.json").write_bytes(b"{}")
    (tmp_path / "two.json").write_bytes(b"{}")
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ExclusiveStorageError):
            exclusive_module._scan_names(directory_fd, maximum_entries=1)

        with monkeypatch.context() as patch:
            patch.setattr(
                exclusive_module.os,
                "scandir",
                lambda descriptor: (_ for _ in ()).throw(RuntimeError(descriptor)),
            )
            with pytest.raises(ExclusiveStorageError):
                exclusive_module._scan_names(directory_fd)

        parent_identity = exclusive_module._Identity.from_stat(tmp_path.stat())
        with monkeypatch.context() as patch:
            real_lstat = Path.lstat

            def fail_parent(path: Path) -> os.stat_result:
                if path == tmp_path:
                    raise OSError("fixture-secret parent")
                return real_lstat(path)

            patch.setattr(Path, "lstat", fail_parent)
            with pytest.raises(ExclusiveStorageError):
                exclusive_module._verify_parent_identity(
                    tmp_path,
                    directory_fd,
                    parent_identity,
                )

        with monkeypatch.context() as patch:
            patch.setattr(
                exclusive_module.os,
                "stat",
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    PermissionError("fixture-secret destination")
                ),
            )
            with pytest.raises(exclusive_module.ExclusiveStorageExistsError):
                exclusive_module._destination_exists(directory_fd, "artifact")
    finally:
        os.close(directory_fd)


@pytest.mark.parametrize(
    "failure",
    ("open", "write", "metadata", "operation", "named-stat", "named-identity"),
)
def test_create_regular_exclusive_sanitizes_each_posix_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    name = f"{failure}.json"
    try:
        if failure == "open":
            real_open = exclusive_module.os.open

            def fail_open(path: object, *args: object, **kwargs: object) -> int:
                if path == name:
                    raise PermissionError("fixture-secret open")
                return real_open(path, *args, **kwargs)

            monkeypatch.setattr(exclusive_module.os, "open", fail_open)
        elif failure == "write":
            monkeypatch.setattr(exclusive_module.os, "write", lambda descriptor, data: 0)
        elif failure == "metadata":
            monkeypatch.setattr(exclusive_module, "_current_owner", lambda metadata: False)
        elif failure == "operation":
            monkeypatch.setattr(
                exclusive_module.os,
                "fchmod",
                lambda descriptor, mode: (_ for _ in ()).throw(OSError("fixture-secret chmod")),
            )
        elif failure == "named-stat":
            real_stat = exclusive_module.os.stat

            def fail_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
                if path == name:
                    raise OSError("fixture-secret stat")
                return real_stat(path, *args, **kwargs)

            monkeypatch.setattr(exclusive_module.os, "stat", fail_stat)
        else:
            real_stat = exclusive_module.os.stat

            def mismatch_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
                if path == name:
                    return os.fstat(directory_fd)
                return real_stat(path, *args, **kwargs)

            monkeypatch.setattr(exclusive_module.os, "stat", mismatch_stat)

        with pytest.raises(ExclusiveStorageError) as error:
            exclusive_module._create_regular_exclusive(directory_fd, name, b"{}")
        assert "fixture-secret" not in str(error.value)
    finally:
        os.close(directory_fd)


@pytest.mark.parametrize(
    "reader,failure",
    (
        ("exact", "named-stat"),
        ("exact", "sync"),
        ("exact", "read"),
        ("bounded", "open"),
        ("bounded", "named-stat"),
        ("bounded", "identity"),
        ("bounded", "read"),
    ),
)
def test_exclusive_regular_readers_fail_closed_on_descriptor_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader: str,
    failure: str,
) -> None:
    name = "value.json"
    target = tmp_path / name
    target.write_bytes(b"{}")
    target.chmod(0o600)
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        if failure == "open":
            real_open = exclusive_module.os.open

            def fail_open(path: object, *args: object, **kwargs: object) -> int:
                if path == name:
                    raise PermissionError("fixture-secret open")
                return real_open(path, *args, **kwargs)

            monkeypatch.setattr(exclusive_module.os, "open", fail_open)
        elif failure == "named-stat":
            real_stat = exclusive_module.os.stat

            def fail_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
                if path == name:
                    raise OSError("fixture-secret stat")
                return real_stat(path, *args, **kwargs)

            monkeypatch.setattr(exclusive_module.os, "stat", fail_stat)
        elif failure == "identity":
            real_stat = exclusive_module.os.stat

            def mismatch_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
                if path == name:
                    return os.fstat(directory_fd)
                return real_stat(path, *args, **kwargs)

            monkeypatch.setattr(exclusive_module.os, "stat", mismatch_stat)
        elif failure == "sync":
            monkeypatch.setattr(
                exclusive_module.os,
                "fsync",
                lambda descriptor: (_ for _ in ()).throw(OSError("fixture-secret fsync")),
            )
        else:
            monkeypatch.setattr(
                exclusive_module.os,
                "read",
                lambda descriptor, maximum: (_ for _ in ()).throw(OSError("fixture-secret read")),
            )

        with pytest.raises(ExclusiveStorageError) as error:
            if reader == "exact":
                exclusive_module._read_regular_exact(
                    directory_fd,
                    name,
                    b"{}",
                    synchronize=failure == "sync",
                )
            else:
                exclusive_module._read_regular_bounded(
                    directory_fd,
                    name,
                    maximum_bytes=16,
                )
        assert "fixture-secret" not in str(error.value)
    finally:
        os.close(directory_fd)


@pytest.mark.parametrize(
    "failure",
    ("second-scan", "child-stat", "destination-stat", "destination-identity"),
)
def test_exact_tree_verification_rejects_each_final_identity_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    root = tmp_path / "artifact"
    files = _fixture_files()
    _publish(root, files)
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        if failure == "second-scan":
            real_scan = exclusive_module._scan_names
            calls = 0

            def mutate_second_scan(*args: object, **kwargs: object) -> tuple[str, ...]:
                nonlocal calls
                calls += 1
                names = real_scan(*args, **kwargs)
                return (*names, "intruder.json") if calls == 2 else names

            monkeypatch.setattr(exclusive_module, "_scan_names", mutate_second_scan)
        else:
            real_stat = exclusive_module.os.stat
            child_calls = 0

            def raced_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
                nonlocal child_calls
                if failure == "child-stat" and path == "alpha.json":
                    child_calls += 1
                    if child_calls == 2:
                        raise OSError("fixture-secret child stat")
                if failure == "destination-stat" and path == root.name:
                    raise OSError("fixture-secret destination stat")
                if failure == "destination-identity" and path == root.name:
                    return os.fstat(parent_fd)
                return real_stat(path, *args, **kwargs)

            monkeypatch.setattr(exclusive_module.os, "stat", raced_stat)

        with pytest.raises(ExclusiveStorageError) as error:
            exclusive_module._verify_exact_tree(parent_fd, root.name, files)
        assert "fixture-secret" not in str(error.value)
    finally:
        os.close(parent_fd)


def test_manifest_last_publication_rejects_parent_and_validation_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _fixture_files()

    unavailable = tmp_path / "unavailable" / "artifact"
    with monkeypatch.context() as patch:
        real_mkdir = Path.mkdir

        def fail_parent_mkdir(path: Path, *args: object, **kwargs: object) -> None:
            if path == unavailable.parent:
                raise OSError("fixture-secret parent")
            real_mkdir(path, *args, **kwargs)

        patch.setattr(Path, "mkdir", fail_parent_mkdir)
        with pytest.raises(ExclusiveStorageError):
            exclusive_module._publish_manifest_last_directory(
                unavailable,
                files,
                manifest_name=_MANIFEST_NAME,
                validate_complete=lambda path: True,
            )

    unsafe_parent = tmp_path / "unsafe-parent"
    unsafe_parent.mkdir(mode=0o700)
    unsafe_parent.chmod(0o777)
    try:
        with pytest.raises(ExclusiveStorageError):
            exclusive_module._publish_manifest_last_directory(
                unsafe_parent / "artifact",
                files,
                manifest_name=_MANIFEST_NAME,
                validate_complete=lambda path: True,
            )
    finally:
        unsafe_parent.chmod(0o700)

    open_failure = tmp_path / "open-failure"
    with monkeypatch.context() as patch:
        real_open = exclusive_module.os.open

        def fail_parent_open(path: object, *args: object, **kwargs: object) -> int:
            if path == tmp_path:
                raise OSError("fixture-secret parent open")
            return real_open(path, *args, **kwargs)

        patch.setattr(exclusive_module.os, "open", fail_parent_open)
        with pytest.raises(ExclusiveStorageError):
            exclusive_module._publish_manifest_last_directory(
                open_failure,
                files,
                manifest_name=_MANIFEST_NAME,
                validate_complete=lambda path: True,
            )

    existing = tmp_path / "existing"
    exclusive_module._publish_manifest_last_directory(
        existing,
        files,
        manifest_name=_MANIFEST_NAME,
        validate_complete=lambda path: True,
    )
    with pytest.raises(exclusive_module.ExclusiveStorageExistsError):
        exclusive_module._publish_manifest_last_directory(
            existing,
            files,
            manifest_name=_MANIFEST_NAME,
            validate_complete=lambda path: False,
        )
    with pytest.raises(exclusive_module.ExclusiveStorageExistsError):
        exclusive_module._publish_manifest_last_directory(
            existing,
            files,
            manifest_name=_MANIFEST_NAME,
            validate_complete=lambda path: (_ for _ in ()).throw(
                RuntimeError("fixture-secret validator")
            ),
        )

    fresh = tmp_path / "fresh-invalid"
    with pytest.raises(ExclusiveStorageError):
        exclusive_module._publish_manifest_last_directory(
            fresh,
            files,
            manifest_name=_MANIFEST_NAME,
            validate_complete=lambda path: False,
        )
    assert fresh.is_dir()


def test_manifest_last_publication_rejects_identity_and_creation_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _fixture_files()

    existing = tmp_path / "existing-race"
    exclusive_module._publish_manifest_last_directory(
        existing,
        files,
        manifest_name=_MANIFEST_NAME,
        validate_complete=lambda path: True,
    )
    with monkeypatch.context() as patch:
        real_verify = exclusive_module._verify_exact_tree
        calls = 0

        def disagree(*args: object, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            verified = real_verify(*args, **kwargs)
            return object() if calls == 2 else verified

        patch.setattr(exclusive_module, "_verify_exact_tree", disagree)
        with pytest.raises(exclusive_module.ExclusiveStorageExistsError):
            exclusive_module._publish_manifest_last_directory(
                existing,
                files,
                manifest_name=_MANIFEST_NAME,
                validate_complete=lambda path: True,
            )

    mkdir_race = tmp_path / "mkdir-race"
    with monkeypatch.context() as patch:
        real_mkdir = exclusive_module.os.mkdir

        def fail_destination_mkdir(path: object, *args: object, **kwargs: object) -> None:
            if path == mkdir_race.name and kwargs.get("dir_fd") is not None:
                raise OSError("fixture-secret mkdir")
            real_mkdir(path, *args, **kwargs)

        patch.setattr(exclusive_module.os, "mkdir", fail_destination_mkdir)
        with pytest.raises(ExclusiveStorageError):
            exclusive_module._publish_manifest_last_directory(
                mkdir_race,
                files,
                manifest_name=_MANIFEST_NAME,
                validate_complete=lambda path: True,
            )

    unsafe_initial = tmp_path / "unsafe-initial"
    with monkeypatch.context() as patch:
        parent_inode = tmp_path.stat().st_ino
        real_owner = exclusive_module._current_owner

        def reject_created_directory(metadata: os.stat_result) -> bool:
            if stat.S_ISDIR(metadata.st_mode) and metadata.st_ino != parent_inode:
                return False
            return real_owner(metadata)

        patch.setattr(exclusive_module, "_current_owner", reject_created_directory)
        with pytest.raises(ExclusiveStorageError):
            exclusive_module._publish_manifest_last_directory(
                unsafe_initial,
                files,
                manifest_name=_MANIFEST_NAME,
                validate_complete=lambda path: True,
            )

    changed = tmp_path / "changed-directory"
    with monkeypatch.context() as patch:
        real_fstat = exclusive_module.os.fstat
        target_inode: int | None = None
        seen = 0

        def change_final_mode(descriptor: int) -> object:
            nonlocal seen, target_inode
            metadata = real_fstat(descriptor)
            if stat.S_ISDIR(metadata.st_mode) and metadata.st_ino != tmp_path.stat().st_ino:
                target_inode = metadata.st_ino
            if target_inode is not None and metadata.st_ino == target_inode:
                seen += 1
                if seen >= 2:
                    return _metadata_with(metadata, st_mode=stat.S_IFDIR | 0o755)
            return metadata

        patch.setattr(exclusive_module.os, "fstat", change_final_mode)
        with pytest.raises(ExclusiveStorageError):
            exclusive_module._publish_manifest_last_directory(
                changed,
                files,
                manifest_name=_MANIFEST_NAME,
                validate_complete=lambda path: True,
            )


@pytest.mark.parametrize(
    "failure",
    ("missing-lock", "first-stat", "second-scan", "second-stat", "identity"),
)
def test_locked_inventory_rejects_each_scan_and_identity_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    root = tmp_path / failure
    if failure == "missing-lock":
        root.mkdir(mode=0o700)
    else:
        with open_locked_flat_directory(
            root,
            create=True,
            maximum_entries=8,
            maximum_file_bytes=1024,
        ) as locked:
            locked.create_regular_exclusive("first.json", b"{}", maximum_bytes=16)

    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        if failure == "first-stat":
            real_stat = exclusive_module.os.stat

            def fail_first_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
                if path == "first.json":
                    raise OSError("fixture-secret first stat")
                return real_stat(path, *args, **kwargs)

            monkeypatch.setattr(exclusive_module.os, "stat", fail_first_stat)
        elif failure == "second-scan":
            real_scan = exclusive_module._scan_names
            scans = 0

            def mismatch_second_scan(*args: object, **kwargs: object) -> tuple[str, ...]:
                nonlocal scans
                scans += 1
                names = real_scan(*args, **kwargs)
                return ("review.lock",) if scans == 2 else names

            monkeypatch.setattr(exclusive_module, "_scan_names", mismatch_second_scan)
        elif failure in {"second-stat", "identity"}:
            real_stat = exclusive_module.os.stat
            entry_stats = 0

            def race_second_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
                nonlocal entry_stats
                if path == "first.json":
                    entry_stats += 1
                    if entry_stats == 2:
                        if failure == "second-stat":
                            raise OSError("fixture-secret second stat")
                        return os.fstat(directory_fd)
                return real_stat(path, *args, **kwargs)

            monkeypatch.setattr(exclusive_module.os, "stat", race_second_stat)

        with pytest.raises(ExclusiveStorageError) as error:
            exclusive_module._locked_entry_identities(
                directory_fd,
                lock_name="review.lock",
                maximum_entries=8,
                maximum_file_bytes=1024,
                entry_name_validator=None,
            )
        assert "fixture-secret" not in str(error.value)
    finally:
        os.close(directory_fd)


def test_locked_handle_rejects_container_inventory_and_operation_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reviews"
    with open_locked_flat_directory(
        root,
        create=True,
        maximum_entries=8,
        maximum_file_bytes=1024,
    ) as locked:
        locked.create_regular_exclusive("first.json", b"{}", maximum_bytes=16)

        with monkeypatch.context() as patch:
            real_stat = exclusive_module.os.stat

            def fail_named_directory(
                path: object,
                *args: object,
                **kwargs: object,
            ) -> os.stat_result:
                if path == root.name:
                    raise OSError("fixture-secret directory stat")
                return real_stat(path, *args, **kwargs)

            patch.setattr(exclusive_module.os, "stat", fail_named_directory)
            with pytest.raises(ExclusiveStorageError):
                locked._refresh_directory_identity()
            with pytest.raises(ExclusiveStorageError):
                locked._verify_container()

        with monkeypatch.context() as patch:
            real_owner = exclusive_module._current_owner
            root_inode = root.stat().st_ino

            def reject_root(metadata: os.stat_result) -> bool:
                return metadata.st_ino != root_inode and real_owner(metadata)

            patch.setattr(exclusive_module, "_current_owner", reject_root)
            with pytest.raises(ExclusiveStorageError):
                locked._refresh_directory_identity()

        with monkeypatch.context() as patch:
            patch.setattr(exclusive_module, "_locked_entry_identities", lambda *a, **k: {})
            with pytest.raises(ExclusiveStorageError):
                locked._verify_inventory()

        with pytest.raises(ExclusiveStorageError):
            locked.read_regular("missing.json", maximum_bytes=16)

        with monkeypatch.context() as patch:
            fake_identity = exclusive_module._Identity(0, 0, 0, 0, 0, 0, 0, 0)
            patch.setattr(
                exclusive_module,
                "_read_regular_bounded",
                lambda *a, **k: (b"{}", fake_identity),
            )
            with pytest.raises(ExclusiveStorageError):
                locked.read_regular("first.json", maximum_bytes=16)

        with monkeypatch.context() as patch:
            real_fsync = exclusive_module.os.fsync

            def fail_directory_fsync(descriptor: int) -> None:
                if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                    raise OSError("fixture-secret directory fsync")
                real_fsync(descriptor)

            patch.setattr(exclusive_module.os, "fsync", fail_directory_fsync)
            with pytest.raises(ExclusiveStorageError):
                locked.create_regular_exclusive("second.json", b"{}", maximum_bytes=16)
        locked._refresh_directory_identity()


@pytest.mark.parametrize(
    "failure",
    ("open-named", "open-identity", "verify-stat", "verify-identity"),
)
def test_lock_file_identity_helpers_reject_named_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    root = tmp_path / failure
    with open_locked_flat_directory(
        root,
        create=True,
        maximum_entries=8,
        maximum_file_bytes=1024,
    ):
        pass
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    lock_fd = os.open("review.lock", os.O_RDWR, dir_fd=directory_fd)
    lock_identity = exclusive_module._Identity.from_stat(os.fstat(lock_fd))
    try:
        real_stat = exclusive_module.os.stat

        def raced_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
            if path == "review.lock":
                if failure in {"open-named", "verify-stat"}:
                    raise OSError("fixture-secret lock stat")
                return os.fstat(directory_fd)
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(exclusive_module.os, "stat", raced_stat)
        with pytest.raises(ExclusiveStorageError) as error:
            if failure.startswith("open"):
                exclusive_module._open_lock_file(directory_fd, "review.lock", create=False)
            else:
                exclusive_module._verify_lock_identity(
                    directory_fd,
                    lock_fd,
                    "review.lock",
                    lock_identity,
                )
        assert "fixture-secret" not in str(error.value)
    finally:
        os.close(lock_fd)
        os.close(directory_fd)


@pytest.mark.parametrize(
    "failure",
    ("open", "before", "sync", "after"),
)
def test_regular_identity_durability_checks_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    name = "first.json"
    target = tmp_path / name
    target.write_bytes(b"{}")
    target.chmod(0o600)
    identity = exclusive_module._Identity.from_stat(target.stat())
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        if failure == "open":
            real_open = exclusive_module.os.open

            def fail_open(path: object, *args: object, **kwargs: object) -> int:
                if path == name:
                    raise OSError("fixture-secret open")
                return real_open(path, *args, **kwargs)

            monkeypatch.setattr(exclusive_module.os, "open", fail_open)
        elif failure == "before":
            identity = exclusive_module._Identity(0, 0, 0, 0, 0, 0, 0, 0)
        elif failure == "sync":
            monkeypatch.setattr(
                exclusive_module.os,
                "fsync",
                lambda descriptor: (_ for _ in ()).throw(OSError("fixture-secret fsync")),
            )
        else:
            real_stat = exclusive_module.os.stat

            def mismatch_named(path: object, *args: object, **kwargs: object) -> os.stat_result:
                if path == name:
                    return os.fstat(directory_fd)
                return real_stat(path, *args, **kwargs)

            monkeypatch.setattr(exclusive_module.os, "stat", mismatch_named)

        with pytest.raises(ExclusiveStorageError) as error:
            exclusive_module._synchronize_regular_identity(directory_fd, name, identity)
        assert "fixture-secret" not in str(error.value)
    finally:
        os.close(directory_fd)


@pytest.mark.parametrize("failure", ("before", "after", "operation"))
def test_locked_directory_durability_checks_reject_lock_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    lock = tmp_path / "review.lock"
    lock.write_bytes(b"")
    lock.chmod(0o600)
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    lock_fd = os.open(lock, os.O_RDWR)
    identity = exclusive_module._Identity.from_stat(os.fstat(lock_fd))
    try:
        if failure == "before":
            identity = exclusive_module._Identity(0, 0, 0, 0, 0, 0, 0, 0)
        elif failure == "after":
            real_fstat = exclusive_module.os.fstat
            calls = 0

            def mismatch_second_fstat(descriptor: int) -> object:
                nonlocal calls
                metadata = real_fstat(descriptor)
                if descriptor == lock_fd:
                    calls += 1
                    if calls == 2:
                        return _metadata_with(metadata, st_mtime_ns=metadata.st_mtime_ns + 1)
                return metadata

            monkeypatch.setattr(exclusive_module.os, "fstat", mismatch_second_fstat)
        else:
            monkeypatch.setattr(
                exclusive_module.os,
                "fsync",
                lambda descriptor: (_ for _ in ()).throw(OSError("fixture-secret fsync")),
            )

        with pytest.raises(ExclusiveStorageError) as error:
            exclusive_module._synchronize_locked_directory(
                directory_fd,
                lock_fd,
                identity,
                {},
            )
        assert "fixture-secret" not in str(error.value)
    finally:
        os.close(lock_fd)
        os.close(directory_fd)


def test_locked_directory_open_rejects_invalid_and_unsafe_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_arguments = (
        {"create": 1, "lock_name": "review.lock", "entry_name_validator": None},
        {"create": True, "lock_name": "../lock", "entry_name_validator": None},
        {"create": True, "lock_name": "review.lock", "entry_name_validator": 1},
    )
    for arguments in invalid_arguments:
        with (
            pytest.raises(ExclusiveStorageError),
            open_locked_flat_directory(
                tmp_path / "invalid",
                maximum_entries=8,
                maximum_file_bytes=1024,
                **arguments,  # type: ignore[arg-type]
            ),
        ):
            raise AssertionError("invalid boundary unexpectedly opened")

    with (
        pytest.raises(ExclusiveStorageError),
        open_locked_flat_directory(
            tmp_path / "missing-parent" / "reviews",
            create=False,
            maximum_entries=8,
            maximum_file_bytes=1024,
        ),
    ):
        raise AssertionError("missing parent unexpectedly opened")

    unsafe_parent = tmp_path / "unsafe-parent"
    unsafe_parent.mkdir(mode=0o700)
    unsafe_parent.chmod(0o777)
    try:
        with (
            pytest.raises(ExclusiveStorageError),
            open_locked_flat_directory(
                unsafe_parent / "reviews",
                create=True,
                maximum_entries=8,
                maximum_file_bytes=1024,
            ),
        ):
            raise AssertionError("unsafe parent unexpectedly opened")
    finally:
        unsafe_parent.chmod(0o700)

    with (
        pytest.raises(ExclusiveStorageError),
        open_locked_flat_directory(
            tmp_path / "missing-reviews",
            create=False,
            maximum_entries=8,
            maximum_file_bytes=1024,
        ),
    ):
        raise AssertionError("missing directory unexpectedly opened")

    mkdir_failure = tmp_path / "mkdir-failure"
    with monkeypatch.context() as patch:
        real_mkdir = exclusive_module.os.mkdir

        def fail_mkdir(path: object, *args: object, **kwargs: object) -> None:
            if path == mkdir_failure.name and kwargs.get("dir_fd") is not None:
                raise OSError("fixture-secret mkdir")
            real_mkdir(path, *args, **kwargs)

        patch.setattr(exclusive_module.os, "mkdir", fail_mkdir)
        with (
            pytest.raises(ExclusiveStorageError),
            open_locked_flat_directory(
                mkdir_failure,
                create=True,
                maximum_entries=8,
                maximum_file_bytes=1024,
            ),
        ):
            raise AssertionError("failed mkdir unexpectedly opened")

    with monkeypatch.context() as patch:
        real_safe_parent = exclusive_module._safe_parent
        checks = 0

        def reject_opened_parent(metadata: os.stat_result) -> bool:
            nonlocal checks
            checks += 1
            return checks != 2 and real_safe_parent(metadata)

        patch.setattr(exclusive_module, "_safe_parent", reject_opened_parent)
        with (
            pytest.raises(ExclusiveStorageError),
            open_locked_flat_directory(
                tmp_path / "opened-parent",
                create=True,
                maximum_entries=8,
                maximum_file_bytes=1024,
            ),
        ):
            raise AssertionError("raced parent unexpectedly opened")

    unsafe_root = tmp_path / "unsafe-root"
    unsafe_root.mkdir(mode=0o755)
    with (
        pytest.raises(ExclusiveStorageError),
        open_locked_flat_directory(
            unsafe_root,
            create=False,
            maximum_entries=8,
            maximum_file_bytes=1024,
        ),
    ):
        raise AssertionError("unsafe root unexpectedly opened")


def test_locked_directory_open_rejects_post_lock_identity_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reviews"
    with open_locked_flat_directory(
        root,
        create=True,
        maximum_entries=8,
        maximum_file_bytes=1024,
    ):
        pass

    with monkeypatch.context() as patch:
        real_verify = exclusive_module._verify_lock_identity
        calls = 0
        saved = tmp_path / "saved-reviews"

        def mutate_after_lock(*args: object, **kwargs: object) -> None:
            nonlocal calls
            real_verify(*args, **kwargs)
            calls += 1
            if calls == 1:
                os.rename(root, saved)
                root.mkdir(mode=0o700)

        patch.setattr(exclusive_module, "_verify_lock_identity", mutate_after_lock)
        try:
            with (
                pytest.raises(ExclusiveStorageError),
                open_locked_flat_directory(
                    root,
                    create=False,
                    maximum_entries=8,
                    maximum_file_bytes=1024,
                ),
            ):
                raise AssertionError("directory identity race unexpectedly opened")
        finally:
            root.rmdir()
            os.rename(saved, root)

    with monkeypatch.context() as patch:
        real_inventory = exclusive_module._locked_entry_identities
        calls = 0

        def mismatch_revalidation(*args: object, **kwargs: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            entries = real_inventory(*args, **kwargs)
            if calls == 2:
                return {**entries, "intruder.json": object()}
            return entries

        patch.setattr(exclusive_module, "_locked_entry_identities", mismatch_revalidation)
        with (
            pytest.raises(ExclusiveStorageError),
            open_locked_flat_directory(
                root,
                create=False,
                maximum_entries=8,
                maximum_file_bytes=1024,
            ),
        ):
            raise AssertionError("inventory race unexpectedly opened")

    import fcntl

    with monkeypatch.context() as patch:
        patch.setattr(
            fcntl,
            "flock",
            lambda descriptor, operation: (_ for _ in ()).throw(OSError("fixture-secret flock")),
        )
        with (
            pytest.raises(ExclusiveStorageError) as error,
            open_locked_flat_directory(
                root,
                create=False,
                maximum_entries=8,
                maximum_file_bytes=1024,
            ),
        ):
            raise AssertionError("failed lock unexpectedly opened")
        assert "fixture-secret" not in str(error.value)


@pytest.mark.parametrize("failure", ("handle", "unlock", "directory", "parent"))
def test_locked_directory_reports_every_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    import fcntl

    root = tmp_path / failure
    with open_locked_flat_directory(
        root,
        create=True,
        maximum_entries=8,
        maximum_file_bytes=1024,
    ):
        pass

    if failure == "handle":
        real_close_handle = LockedFlatDirectory._close

        def fail_handle_close(handle: LockedFlatDirectory) -> None:
            real_close_handle(handle)
            raise RuntimeError("fixture-secret handle close")

        monkeypatch.setattr(LockedFlatDirectory, "_close", fail_handle_close)
    elif failure == "unlock":
        real_flock = fcntl.flock

        def fail_unlock(descriptor: int, operation: int) -> None:
            real_flock(descriptor, operation)
            if operation == fcntl.LOCK_UN:
                raise OSError("fixture-secret unlock")

        monkeypatch.setattr(fcntl, "flock", fail_unlock)
    else:
        target_inode = (root if failure == "directory" else root.parent).stat().st_ino
        real_close = exclusive_module.os.close
        armed = False

        def fail_target_close(descriptor: int) -> None:
            inode = os.fstat(descriptor).st_ino
            real_close(descriptor)
            if armed and inode == target_inode:
                raise OSError("fixture-secret descriptor close")

        monkeypatch.setattr(exclusive_module.os, "close", fail_target_close)

    with (
        pytest.raises(ExclusiveStorageError) as error,
        open_locked_flat_directory(
            root,
            create=False,
            maximum_entries=8,
            maximum_file_bytes=1024,
        ),
    ):
        if failure in {"directory", "parent"}:
            armed = True

    assert "fixture-secret" not in str(error.value)
