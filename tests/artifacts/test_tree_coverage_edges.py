"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from tests.artifacts.test_artifact_tree import (
    _MANIFEST_MAXIMUM_BYTES,
    _MANIFEST_NAME,
    _caught_publish_error,
    _fixture_files,
    _prepare_recovery_state,
    _publish,
    _tree,
    _validate_tree,
)

from saliencegate.artifacts import tree as tree_module
from saliencegate.artifacts.tree import (
    ArtifactExistsError,
    ArtifactExportError,
    ClosedTreeReadError,
)


def test_descriptor_read_rejects_data_growth_past_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = tmp_path / "value.json"
    value.write_bytes(b"{}")
    value.chmod(0o600)
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)

    def fill_bound(_descriptor: int, maximum: int) -> bytes:
        return b"x" * maximum

    monkeypatch.setattr(tree_module.os, "read", fill_bound)
    try:
        with pytest.raises(ClosedTreeReadError):
            tree_module._read_regular_file(
                directory_fd,
                value.name,
                maximum=2,
            )
    finally:
        os.close(directory_fd)


def test_recovery_file_read_rejects_descriptor_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = tmp_path / "value.json"
    value.write_bytes(b"{}")
    value.chmod(0o600)
    monkeypatch.setattr(tree_module.os, "fstat", lambda _descriptor: tmp_path.stat())

    with pytest.raises(ArtifactExportError, match="recovery is unsafe"):
        tree_module._read_regular_path(value, maximum=16)


def test_recovery_file_read_rejects_growth_past_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = tmp_path / "value.json"
    value.write_bytes(b"{}")
    value.chmod(0o600)

    def fill_bound(_descriptor: int, maximum: int) -> bytes:
        return b"x" * maximum

    monkeypatch.setattr(tree_module.os, "read", fill_bound)

    with pytest.raises(ArtifactExportError, match="recovery is unsafe"):
        tree_module._read_regular_path(value, maximum=2)


def test_recovery_rejects_marker_without_matching_visible_tree(tmp_path: Path) -> None:
    root, _backup, _marker, _original, _replacement = _prepare_recovery_state(
        tmp_path / "artifact",
        tmp_path / "replacement-source",
        state="original-only",
    )
    metadata = root.stat()
    os.utime(root, ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000))

    with pytest.raises(ArtifactExportError, match="recovery is unsafe"):
        tree_module._recover_interrupted_replacement(
            root,
            tmp_path,
            validate_tree=_validate_tree,
            manifest_name=_MANIFEST_NAME,
            maximum_manifest_bytes=_MANIFEST_MAXIMUM_BYTES,
        )


def test_recovery_requires_marker_cleanup_for_visible_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _backup, marker, original, _replacement = _prepare_recovery_state(
        tmp_path / "artifact",
        tmp_path / "replacement-source",
        state="original-only",
    )
    monkeypatch.setattr(tree_module, "_unlink_owned_regular", lambda *_args: False)

    with pytest.raises(ArtifactExportError, match="recovery is unsafe"):
        tree_module._recover_interrupted_replacement(
            root,
            tmp_path,
            validate_tree=_validate_tree,
            manifest_name=_MANIFEST_NAME,
            maximum_manifest_bytes=_MANIFEST_MAXIMUM_BYTES,
        )
    assert _tree(root) == original
    assert marker.exists()


def test_recovery_requires_marker_cleanup_after_backup_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _backup, marker, original, _replacement = _prepare_recovery_state(
        tmp_path / "artifact",
        tmp_path / "replacement-source",
        state="backup-only",
    )
    monkeypatch.setattr(tree_module, "_unlink_owned_regular", lambda *_args: False)

    with pytest.raises(ArtifactExportError, match="recovery is unsafe"):
        tree_module._recover_interrupted_replacement(
            root,
            tmp_path,
            validate_tree=_validate_tree,
            manifest_name=_MANIFEST_NAME,
            maximum_manifest_bytes=_MANIFEST_MAXIMUM_BYTES,
        )
    assert _tree(root) == original
    assert marker.exists()


def test_recovery_rejects_non_directory_published_destination(tmp_path: Path) -> None:
    root, _backup, _marker, _original, replacement = _prepare_recovery_state(
        tmp_path / "artifact",
        tmp_path / "replacement-source",
        state="both",
    )
    saved = tmp_path / "saved-replacement"
    os.rename(root, saved)
    root.write_bytes(b"unsafe")
    root.chmod(0o600)

    with pytest.raises(ArtifactExportError, match="recovery is unsafe"):
        tree_module._recover_interrupted_replacement(
            root,
            tmp_path,
            validate_tree=_validate_tree,
            manifest_name=_MANIFEST_NAME,
            maximum_manifest_bytes=_MANIFEST_MAXIMUM_BYTES,
        )
    assert _tree(saved) == replacement


def test_recovery_requires_backup_cleanup_for_published_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, backup, marker, _original, replacement = _prepare_recovery_state(
        tmp_path / "artifact",
        tmp_path / "replacement-source",
        state="both",
    )
    monkeypatch.setattr(tree_module, "_remove_owned_directory", lambda *_args: False)

    with pytest.raises(ArtifactExportError, match="recovery is unsafe"):
        tree_module._recover_interrupted_replacement(
            root,
            tmp_path,
            validate_tree=_validate_tree,
            manifest_name=_MANIFEST_NAME,
            maximum_manifest_bytes=_MANIFEST_MAXIMUM_BYTES,
        )
    assert _tree(root) == replacement
    assert backup.exists()
    assert marker.exists()


def test_recovery_normalizes_os_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_path: Path) -> object:
        raise OSError("lstat failed")

    monkeypatch.setattr(tree_module, "_lstat_or_none", fail)

    with pytest.raises(ArtifactExportError, match="recovery failed"):
        tree_module._recover_interrupted_replacement(
            tmp_path / "artifact",
            tmp_path,
            validate_tree=_validate_tree,
            manifest_name=_MANIFEST_NAME,
            maximum_manifest_bytes=_MANIFEST_MAXIMUM_BYTES,
        )


def test_replacement_recheck_returns_false_when_destination_disappears(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    saved = tmp_path / "saved"
    _publish(root, _fixture_files())
    authorization = tree_module._authorized_replace_target(
        root,
        root.lstat(),
        "stable-tree",
        validate_tree=_validate_tree,
        manifest_name=_MANIFEST_NAME,
        maximum_manifest_bytes=_MANIFEST_MAXIMUM_BYTES,
    )
    root.rename(saved)

    assert not tree_module._replacement_is_still_authorized(
        root,
        authorization,
        "stable-tree",
        validate_tree=_validate_tree,
        manifest_name=_MANIFEST_NAME,
        maximum_manifest_bytes=_MANIFEST_MAXIMUM_BYTES,
    )


def test_publisher_rejects_staging_identity_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    real_same_object = tree_module._PathIdentity.same_object
    parent_inode = tmp_path.lstat().st_ino

    def reject_second(
        identity: tree_module._PathIdentity,
        metadata: os.stat_result,
    ) -> bool:
        if stat.S_ISDIR(identity.mode) and identity.inode != parent_inode:
            return False
        return real_same_object(identity, metadata)

    monkeypatch.setattr(tree_module._PathIdentity, "same_object", reject_second)

    error = _caught_publish_error(lambda: _publish(root, _fixture_files()))
    assert isinstance(error, ArtifactExistsError)
    assert not root.exists()


def test_publisher_rejects_unsafe_new_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    original = _fixture_files(generation=1)
    _publish(root, original)
    monkeypatch.setattr(tree_module, "_owned_regular", lambda *_args: False)

    error = _caught_publish_error(
        lambda: _publish(root, _fixture_files(generation=2), replace=True)
    )
    assert isinstance(error, ArtifactExistsError)
    assert _tree(root) == original


def test_publisher_rechecks_marker_before_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    original = _fixture_files(generation=1)
    _publish(root, original)
    real_owned_regular = tree_module._owned_regular
    calls = 0

    def reject_second(path: Path, identity: tree_module._PathIdentity) -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:
            return False
        return real_owned_regular(path, identity)

    monkeypatch.setattr(tree_module, "_owned_regular", reject_second)

    error = _caught_publish_error(
        lambda: _publish(root, _fixture_files(generation=2), replace=True)
    )
    assert isinstance(error, ArtifactExistsError)
    assert _tree(root) == original
