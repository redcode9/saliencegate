from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest

from saliencegate.artifacts import export as export_module
from saliencegate.artifacts import tree as tree_module
from saliencegate.artifacts.export import (
    ArtifactExistsError,
    ArtifactExportError,
    SyntheticArtifactContent,
    discover_revision,
    export_replay_artifact,
)
from saliencegate.artifacts.manifest import (
    ArtifactClassification,
    ArtifactEvidenceLevel,
    RevisionEvidence,
    RevisionSource,
)
from saliencegate.artifacts.validate import validate_artifact
from saliencegate.runtime.engine import ReplayRunResult
from saliencegate.security import RedactionPolicy


def _clean_revision() -> RevisionEvidence:
    return RevisionEvidence(
        source=RevisionSource.GIT,
        package_version="0.1.0",
        commit="1" * 40,
        dirty_worktree=False,
        distribution_digest=None,
    )


def _tree(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


async def test_user_export_is_deterministic_valid_and_contains_no_raw_fixture_content(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = export_replay_artifact(
        replay_result,
        first,
        classification=ArtifactClassification.USER_REDACTED,
        revision=_clean_revision(),
    )
    second_manifest = export_replay_artifact(
        replay_result,
        second,
        classification=ArtifactClassification.USER_REDACTED,
        revision=_clean_revision(),
    )

    assert _tree(first) == _tree(second)
    assert set(_tree(first)) == {
        "attestations.json",
        "budgets.json",
        "decisions.json",
        "deliveries.json",
        "manifest.json",
        "outcomes.json",
        "run.json",
    }
    assert first_manifest == second_manifest
    report = validate_artifact(
        first / "manifest.json",
        expected_manifest_digest=first_manifest.manifest_digest,
    )
    assert report.valid
    assert report.manifest_digest == first_manifest.manifest_digest
    assert report.component_count == 6

    exported = b"".join(_tree(first).values())
    assert b"verified event 1" not in exported
    assert b"Run the verified test suite before delivery." not in exported
    assert b"engine-request-4" not in exported
    assert stat.S_IMODE(first.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600 and not path.read_bytes().endswith(b"\n")
        for path in first.iterdir()
    )


async def test_raw_synthetic_content_requires_explicit_synthetic_classification(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    content = SyntheticArtifactContent(
        prompt={"instruction": "fixture-secret-value"},
        responses=({"answer": "redistributable synthetic response"},),
    )

    with pytest.raises(ArtifactExportError, match="synthetic content"):
        export_replay_artifact(
            replay_result,
            tmp_path / "user",
            classification=ArtifactClassification.USER_REDACTED,
            revision=_clean_revision(),
            synthetic_content=content,
        )
    assert not (tmp_path / "user").exists()

    manifest = export_replay_artifact(
        replay_result,
        tmp_path / "synthetic",
        classification=ArtifactClassification.SYNTHETIC_RAW,
        revision=_clean_revision(),
        synthetic_content=content,
    )

    assert manifest.classification is ArtifactClassification.SYNTHETIC_RAW
    assert b"fixture-secret-value" in (tmp_path / "synthetic" / "synthetic.json").read_bytes()
    assert validate_artifact(tmp_path / "synthetic" / "manifest.json").valid


async def test_export_refuses_overwrite_and_replace_produces_a_valid_tree(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    destination = tmp_path / "artifact"
    original = export_replay_artifact(
        replay_result,
        destination,
        classification=ArtifactClassification.USER_REDACTED,
        revision=_clean_revision(),
    )

    with pytest.raises(ArtifactExistsError):
        export_replay_artifact(
            replay_result,
            destination,
            classification=ArtifactClassification.USER_REDACTED,
            revision=_clean_revision(),
        )

    replaced = export_replay_artifact(
        replay_result,
        destination,
        classification=ArtifactClassification.USER_REDACTED,
        revision=_clean_revision(),
        replace=True,
    )
    assert replaced == original
    assert validate_artifact(destination / "manifest.json").valid
    assert tuple(tmp_path.glob(".artifact.*")) == (tmp_path / ".artifact.lock",)


async def test_failed_publish_leaves_no_partial_destination_or_temp_tree(
    tmp_path: Path,
    replay_result: ReplayRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact"

    def fail_replace(source: os.PathLike[str] | str, target: os.PathLike[str] | str) -> None:
        raise OSError("simulated publish failure")

    monkeypatch.setattr("saliencegate.artifacts.export.os.replace", fail_replace)

    with pytest.raises(ArtifactExportError):
        export_replay_artifact(
            replay_result,
            destination,
            classification=ArtifactClassification.USER_REDACTED,
            revision=_clean_revision(),
        )

    assert not destination.exists()
    assert tuple(tmp_path.glob(".artifact.*")) == (tmp_path / ".artifact.lock",)


def test_revision_discovery_never_uses_distribution_to_mask_git_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(export_module, "_git_revision", lambda path: (None, False))
    monkeypatch.setattr(export_module, "_distribution_digest", lambda: "a" * 64)
    distribution = discover_revision(tmp_path)
    assert distribution.source is RevisionSource.DISTRIBUTION
    assert distribution.distribution_digest == "a" * 64

    monkeypatch.setattr(export_module, "_git_revision", lambda path: (None, True))
    guarded = discover_revision(tmp_path)
    assert guarded.source is RevisionSource.UNATTESTED
    assert guarded.distribution_digest is None

    monkeypatch.setattr(export_module, "_git_revision", lambda path: (_clean_revision(), True))
    assert discover_revision(tmp_path) == _clean_revision()


async def test_user_export_fails_closed_on_known_structural_secret_or_confirmatory_dirty_tree(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    secret = replay_result.model_id
    with pytest.raises(ArtifactExportError, match="non-redacted"):
        export_replay_artifact(
            replay_result,
            tmp_path / "secret",
            classification=ArtifactClassification.USER_REDACTED,
            revision=_clean_revision(),
            redaction_policy=RedactionPolicy(literal_secrets=(secret,)),
        )
    assert not (tmp_path / "secret").exists()

    dirty = RevisionEvidence(
        source=RevisionSource.GIT,
        package_version="0.1.0",
        commit="3" * 40,
        dirty_worktree=True,
        distribution_digest=None,
    )
    with pytest.raises(ArtifactExportError, match="construction"):
        export_replay_artifact(
            replay_result,
            tmp_path / "confirmatory",
            classification=ArtifactClassification.USER_REDACTED,
            evidence_level=ArtifactEvidenceLevel.CONFIRMATORY,
            revision=dirty,
        )
    assert not (tmp_path / "confirmatory").exists()


async def test_export_boundary_rejects_invalid_flags_paths_and_existing_non_directory(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    destination = tmp_path / "existing"
    destination.write_text("do not replace")

    with pytest.raises(ArtifactExistsError):
        export_replay_artifact(
            replay_result,
            destination,
            revision=_clean_revision(),
            replace=True,
        )
    with pytest.raises(ArtifactExportError, match="text path"):
        export_replay_artifact(
            replay_result,
            cast("Path", b"artifact"),
            revision=_clean_revision(),
        )
    with pytest.raises(ArtifactExportError, match="classification"):
        export_replay_artifact(
            replay_result,
            tmp_path / "bad-classification",
            classification=cast("ArtifactClassification", "user_redacted"),
            revision=_clean_revision(),
        )
    with pytest.raises(ArtifactExportError, match="replace flag"):
        export_replay_artifact(
            replay_result,
            tmp_path / "bad-replace",
            revision=_clean_revision(),
            replace=cast("bool", 1),
        )


async def test_failed_replacement_restores_original_artifact(
    tmp_path: Path,
    replay_result: ReplayRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact"
    original = export_replay_artifact(
        replay_result,
        destination,
        revision=_clean_revision(),
    )
    original_tree = _tree(destination)
    real_replace = os.replace
    calls = 0

    def fail_new_publish(source: os.PathLike[str] | str, target: os.PathLike[str] | str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated second-rename failure")
        real_replace(source, target)

    monkeypatch.setattr(export_module.os, "replace", fail_new_publish)

    with pytest.raises(ArtifactExportError, match="atomic publish"):
        export_replay_artifact(
            replay_result,
            destination,
            revision=_clean_revision(),
            replace=True,
        )

    assert _tree(destination) == original_tree
    assert validate_artifact(
        destination / "manifest.json",
        expected_manifest_digest=original.manifest_digest,
    ).valid
    assert tuple(tmp_path.glob(".artifact.*")) == (tmp_path / ".artifact.lock",)


async def test_next_export_recovers_destination_after_interrupted_replacement(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    destination = tmp_path / "artifact"
    original = export_replay_artifact(
        replay_result,
        destination,
        revision=_clean_revision(),
    )
    original_tree = _tree(destination)
    backup = tmp_path / ".artifact.backup"
    marker = tmp_path / ".artifact.replace.json"
    replacement = tmp_path / ".replacement-placeholder"
    replacement.mkdir()
    export_module._write_file(
        tmp_path,
        marker.name,
        export_module._replacement_marker_bytes(
            destination,
            destination.lstat(),
            replacement.lstat(),
            run_id=original.run_id,
            original_manifest_digest=original.manifest_digest,
            replacement_manifest_digest=original.manifest_digest,
        ),
    )
    os.rename(destination, backup)
    assert not destination.exists()

    recovered = export_replay_artifact(
        replay_result,
        destination,
        revision=_clean_revision(),
        replace=True,
    )

    assert recovered == original
    assert _tree(destination) == original_tree
    assert not backup.exists()
    assert not marker.exists()
    assert validate_artifact(destination / "manifest.json").valid
    replacement.rmdir()


async def test_unknown_backup_sibling_is_never_deleted_during_recovery(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    destination = tmp_path / "artifact"
    export_replay_artifact(replay_result, destination, revision=_clean_revision())
    unknown_backup = tmp_path / ".artifact.backup"
    unknown_backup.mkdir()
    sentinel = unknown_backup / "unrelated-user-data.txt"
    sentinel.write_text("must survive")

    with pytest.raises(ArtifactExportError, match="recovery is unsafe"):
        export_replay_artifact(replay_result, destination, revision=_clean_revision())

    assert sentinel.read_text() == "must survive"


async def test_failed_immediate_restore_keeps_recovery_state_for_next_export(
    tmp_path: Path,
    replay_result: ReplayRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact"
    original = export_replay_artifact(
        replay_result,
        destination,
        revision=_clean_revision(),
    )
    real_replace = os.replace
    calls = 0

    def fail_publish(source: os.PathLike[str] | str, target: os.PathLike[str] | str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("publish failed")
        real_replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(export_module.os, "replace", fail_publish)
        patch.setattr(
            export_module.os,
            "rename",
            lambda source, target: (_ for _ in ()).throw(OSError("restore failed")),
        )
        with pytest.raises(ArtifactExportError, match="atomic publish"):
            export_replay_artifact(
                replay_result,
                destination,
                revision=_clean_revision(),
                replace=True,
            )

    backup = tmp_path / ".artifact.backup"
    marker = tmp_path / ".artifact.replace.json"
    assert not destination.exists()
    assert backup.is_dir()
    assert marker.is_file()

    recovered = export_replay_artifact(
        replay_result,
        destination,
        revision=_clean_revision(),
        replace=True,
    )
    assert recovered == original
    assert validate_artifact(destination / "manifest.json").valid
    assert not backup.exists()
    assert not marker.exists()


async def test_concurrent_destination_creation_never_authorizes_backup_deletion(
    tmp_path: Path,
    replay_result: ReplayRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact"
    original = export_replay_artifact(
        replay_result,
        destination,
        revision=_clean_revision(),
    )
    real_replace = os.replace

    def collide_with_publish(
        source: os.PathLike[str] | str,
        target: os.PathLike[str] | str,
    ) -> None:
        source_path = Path(source)
        target_path = Path(target)
        if source_path.name.startswith(".artifact.tmp-") and target_path == destination:
            destination.mkdir()
            (destination / "intruder.json").write_text("unrelated")
            raise OSError("destination appeared concurrently")
        real_replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(export_module.os, "replace", collide_with_publish)
        with pytest.raises(ArtifactExportError, match="atomic publish"):
            export_replay_artifact(
                replay_result,
                destination,
                revision=_clean_revision(),
                replace=True,
            )

    backup = tmp_path / ".artifact.backup"
    marker = tmp_path / ".artifact.replace.json"
    assert (destination / "intruder.json").read_text() == "unrelated"
    assert backup.is_dir()
    assert marker.is_file()
    assert validate_artifact(
        backup / "manifest.json",
        expected_manifest_digest=original.manifest_digest,
    ).valid

    with pytest.raises(ArtifactExportError, match="recovery is unsafe"):
        export_replay_artifact(
            replay_result,
            destination,
            revision=_clean_revision(),
            replace=True,
        )
    assert backup.is_dir()
    assert marker.is_file()

    (destination / "intruder.json").unlink()
    destination.rmdir()
    recovered = export_replay_artifact(
        replay_result,
        destination,
        revision=_clean_revision(),
        replace=True,
    )
    assert recovered == original
    assert validate_artifact(destination / "manifest.json").valid


async def test_replace_refuses_non_artifact_directory_without_touching_its_data(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    destination = tmp_path / "artifact"
    destination.mkdir()
    sentinel = destination / "unrelated-user-data.txt"
    sentinel.write_text("must survive")

    with pytest.raises(ArtifactExistsError):
        export_replay_artifact(
            replay_result,
            destination,
            revision=_clean_revision(),
            replace=True,
        )

    assert sentinel.read_text() == "must survive"
    assert tuple(destination.iterdir()) == (sentinel,)


async def test_replace_authorization_is_bound_to_the_same_run(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    destination = tmp_path / "artifact"
    export_replay_artifact(
        replay_result,
        destination,
        revision=_clean_revision(),
    )

    with pytest.raises(ArtifactExistsError):
        export_module._authorized_replace_target(
            destination,
            destination.lstat(),
            uuid4(),
        )

    assert validate_artifact(destination / "manifest.json").valid


async def test_replace_aborts_without_deleting_a_destination_swapped_before_backup_rename(
    tmp_path: Path,
    replay_result: ReplayRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact"
    manifest = export_replay_artifact(replay_result, destination, revision=_clean_revision())
    saved_original = tmp_path / "saved-original"
    intruder = tmp_path / "intruder"
    intruder.mkdir()
    sentinel = intruder / "unrelated-user-data.txt"
    sentinel.write_text("must survive")
    backup = tmp_path / ".artifact.backup"
    marker = tmp_path / ".artifact.replace.json"
    real_replace = os.replace

    def swap_before_backup(
        source: os.PathLike[str] | str,
        target: os.PathLike[str] | str,
    ) -> None:
        if Path(source) == destination and Path(target) == backup:
            os.rename(destination, saved_original)
            os.rename(intruder, destination)
        real_replace(source, target)

    monkeypatch.setattr(export_module.os, "replace", swap_before_backup)

    with pytest.raises(ArtifactExistsError):
        export_replay_artifact(
            replay_result,
            destination,
            revision=_clean_revision(),
            replace=True,
        )

    assert (backup / sentinel.name).read_text() == "must survive"
    assert validate_artifact(
        saved_original / "manifest.json",
        expected_manifest_digest=manifest.manifest_digest,
    ).valid
    assert marker.is_file()


async def test_replace_aborts_and_preserves_a_staging_path_swapped_before_publish(
    tmp_path: Path,
    replay_result: ReplayRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact"
    manifest = export_replay_artifact(replay_result, destination, revision=_clean_revision())
    saved_staging = tmp_path / "saved-staging"
    intruder = tmp_path / "intruder"
    intruder.mkdir()
    sentinel = intruder / "unrelated-user-data.txt"
    sentinel.write_text("must survive")
    backup = tmp_path / ".artifact.backup"
    marker = tmp_path / ".artifact.replace.json"
    real_replace = os.replace

    def swap_staging(
        source: os.PathLike[str] | str,
        target: os.PathLike[str] | str,
    ) -> None:
        source_path = Path(source)
        if source_path.name.startswith(".artifact.tmp-") and Path(target) == destination:
            os.rename(source_path, saved_staging)
            os.rename(intruder, source_path)
        real_replace(source, target)

    monkeypatch.setattr(export_module.os, "replace", swap_staging)

    with pytest.raises(ArtifactExistsError):
        export_replay_artifact(
            replay_result,
            destination,
            revision=_clean_revision(),
            replace=True,
        )

    assert (destination / sentinel.name).read_text() == "must survive"
    assert validate_artifact(
        backup / "manifest.json",
        expected_manifest_digest=manifest.manifest_digest,
    ).valid
    assert validate_artifact(
        saved_staging / "manifest.json",
        expected_manifest_digest=manifest.manifest_digest,
    ).valid
    assert marker.is_file()


async def test_replace_never_deletes_a_backup_path_swapped_before_cleanup(
    tmp_path: Path,
    replay_result: ReplayRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact"
    manifest = export_replay_artifact(replay_result, destination, revision=_clean_revision())
    backup = tmp_path / ".artifact.backup"
    saved_backup = tmp_path / "saved-backup"
    intruder = tmp_path / "intruder"
    intruder.mkdir()
    sentinel = intruder / "unrelated-user-data.txt"
    sentinel.write_text("must survive")
    marker = tmp_path / ".artifact.replace.json"
    real_remove = tree_module._remove_owned_directory

    def swap_backup(path: Path, expected: export_module._PathIdentity) -> bool:
        if path == backup:
            os.rename(backup, saved_backup)
            os.rename(intruder, backup)
        return real_remove(path, expected)

    monkeypatch.setattr(tree_module, "_remove_owned_directory", swap_backup)

    with pytest.raises(ArtifactExportError, match="cleanup is unsafe"):
        export_replay_artifact(
            replay_result,
            destination,
            revision=_clean_revision(),
            replace=True,
        )

    assert (backup / sentinel.name).read_text() == "must survive"
    assert validate_artifact(
        saved_backup / "manifest.json",
        expected_manifest_digest=manifest.manifest_digest,
    ).valid
    assert validate_artifact(destination / "manifest.json").valid
    assert marker.is_file()


async def test_recovery_preserves_a_backup_mutated_after_the_marker_was_written(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    destination = tmp_path / "artifact"
    manifest = export_replay_artifact(replay_result, destination, revision=_clean_revision())
    backup = tmp_path / ".artifact.backup"
    marker = tmp_path / ".artifact.replace.json"
    replacement = tmp_path / "replacement-placeholder"
    replacement.mkdir()
    export_module._write_file(
        tmp_path,
        marker.name,
        export_module._replacement_marker_bytes(
            destination,
            destination.lstat(),
            replacement.lstat(),
            run_id=manifest.run_id,
            original_manifest_digest=manifest.manifest_digest,
            replacement_manifest_digest=manifest.manifest_digest,
        ),
    )
    os.rename(destination, backup)
    sentinel = backup / "unrelated-user-data.txt"
    sentinel.write_text("must survive")

    with pytest.raises(ArtifactExportError, match="recovery is unsafe"):
        export_replay_artifact(
            replay_result,
            destination,
            revision=_clean_revision(),
            replace=True,
        )

    assert sentinel.read_text() == "must survive"
    assert backup.is_dir()
    assert marker.is_file()


async def test_recovery_keeps_the_valid_backup_when_published_content_is_corrupted(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    destination = tmp_path / "artifact"
    replacement = tmp_path / "replacement"
    original = export_replay_artifact(replay_result, destination, revision=_clean_revision())
    new = export_replay_artifact(replay_result, replacement, revision=_clean_revision())
    backup = tmp_path / ".artifact.backup"
    marker = tmp_path / ".artifact.replace.json"
    export_module._write_file(
        tmp_path,
        marker.name,
        export_module._replacement_marker_bytes(
            destination,
            destination.lstat(),
            replacement.lstat(),
            run_id=original.run_id,
            original_manifest_digest=original.manifest_digest,
            replacement_manifest_digest=new.manifest_digest,
        ),
    )
    os.rename(destination, backup)
    os.rename(replacement, destination)
    decisions = destination / "decisions.json"
    decisions.write_bytes(decisions.read_bytes() + b" ")

    with pytest.raises(ArtifactExportError, match="recovery is unsafe"):
        export_replay_artifact(
            replay_result,
            destination,
            revision=_clean_revision(),
            replace=True,
        )

    assert validate_artifact(
        backup / "manifest.json",
        expected_manifest_digest=original.manifest_digest,
    ).valid
    assert decisions.read_bytes().endswith(b" ")
    assert marker.is_file()


async def test_recovery_accepts_an_original_restored_before_marker_cleanup(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    destination = tmp_path / "artifact"
    manifest = export_replay_artifact(replay_result, destination, revision=_clean_revision())
    backup = tmp_path / ".artifact.backup"
    marker = tmp_path / ".artifact.replace.json"
    replacement = tmp_path / "replacement-placeholder"
    replacement.mkdir()
    export_module._write_file(
        tmp_path,
        marker.name,
        export_module._replacement_marker_bytes(
            destination,
            destination.lstat(),
            replacement.lstat(),
            run_id=manifest.run_id,
            original_manifest_digest=manifest.manifest_digest,
            replacement_manifest_digest=manifest.manifest_digest,
        ),
    )
    os.rename(destination, backup)
    os.rename(backup, destination)

    with pytest.raises(ArtifactExistsError):
        export_replay_artifact(replay_result, destination, revision=_clean_revision())

    assert validate_artifact(
        destination / "manifest.json",
        expected_manifest_digest=manifest.manifest_digest,
    ).valid
    assert not backup.exists()
    assert not marker.exists()


async def test_export_refuses_an_unsafe_lock_without_touching_its_target(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    destination = tmp_path / "artifact"
    lock = tmp_path / ".artifact.lock"
    target = tmp_path / "unrelated-user-data.txt"
    target.write_text("must survive")
    try:
        lock.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("symbolic links are unavailable")

    with pytest.raises(ArtifactExportError, match="lock"):
        export_replay_artifact(replay_result, destination, revision=_clean_revision())

    assert target.read_text() == "must survive"
    assert lock.is_symlink()
    assert not destination.exists()

    lock.unlink()
    lock.write_text("must survive")
    lock.chmod(0o644)
    with pytest.raises(ArtifactExportError, match="lock is unsafe"):
        export_replay_artifact(replay_result, destination, revision=_clean_revision())
    assert lock.read_text() == "must survive"
    assert not destination.exists()


def test_filesystem_identity_payload_and_owned_cleanup_fail_closed(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    foreign = tmp_path / "foreign"
    owned.mkdir()
    foreign.mkdir()
    identity = export_module._PathIdentity.from_stat(owned.lstat())

    assert export_module._PathIdentity.from_payload(identity.payload()) == identity
    assert export_module._PathIdentity.from_payload({}) is None
    assert export_module._PathIdentity.from_payload({**identity.payload(), "size": -1}) is None
    assert not export_module._remove_owned_staging(foreign, identity)
    assert not export_module._unlink_owned_regular(foreign, identity)
    assert foreign.is_dir()


@pytest.mark.parametrize("swap_call", (1, 2))
def test_destination_lock_detects_path_identity_swaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_call: int,
) -> None:
    destination = tmp_path / "artifact"
    lock = tmp_path / ".artifact.lock"
    foreign = tmp_path / "foreign.lock"
    foreign.touch(mode=0o600)
    real_lstat = Path.lstat
    calls = 0

    def swapped_lstat(path: Path) -> os.stat_result:
        nonlocal calls
        if path == lock:
            calls += 1
            if calls == swap_call:
                return real_lstat(foreign)
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", swapped_lstat)

    with (
        pytest.raises(ArtifactExportError, match="lock is unsafe"),
        export_module._destination_lock(destination, tmp_path),
    ):
        pytest.fail("an identity-swapped lock must not be acquired")


def test_replacement_marker_parser_rejects_unsafe_malformed_and_raced_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact"
    marker = tmp_path / ".artifact.replace.json"
    marker.mkdir()
    with pytest.raises(ArtifactExportError, match="recovery is unsafe"):
        export_module._read_replacement_marker(marker, destination)

    marker.rmdir()
    marker.write_bytes(b"x" * 5_000)
    marker.chmod(0o600)
    with pytest.raises(ArtifactExportError, match="recovery is unsafe"):
        export_module._read_replacement_marker(marker, destination)

    marker.write_bytes(b"not-json")
    with pytest.raises(ArtifactExportError, match="recovery is unsafe"):
        export_module._read_replacement_marker(marker, destination)

    original = tmp_path / "original"
    replacement = tmp_path / "replacement"
    original.mkdir()
    replacement.mkdir()
    marker.write_bytes(
        export_module._replacement_marker_bytes(
            destination,
            original.lstat(),
            replacement.lstat(),
            run_id=UUID("00000000-0000-4000-8000-00000000d001"),
            original_manifest_digest="a" * 64,
            replacement_manifest_digest="b" * 64,
        )
    )
    marker.chmod(0o600)
    saved_marker = tmp_path / "saved-marker.json"
    real_lstat = Path.lstat
    marker_lstats = 0

    def replace_before_final_lstat(path: Path) -> os.stat_result:
        nonlocal marker_lstats
        if path == marker:
            marker_lstats += 1
            if marker_lstats == 2:
                os.rename(marker, saved_marker)
                marker.write_bytes(saved_marker.read_bytes())
                marker.chmod(0o600)
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", replace_before_final_lstat)
    with pytest.raises(ArtifactExportError, match="recovery is unsafe"):
        export_module._read_replacement_marker(marker, destination)


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership and mode bits are required")
async def test_export_rejects_a_group_or_world_writable_parent(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    unsafe = tmp_path / "unsafe-parent"
    unsafe.mkdir(mode=0o777)
    unsafe.chmod(0o777)

    with pytest.raises(ArtifactExportError, match="parent is unsafe"):
        export_replay_artifact(
            replay_result,
            unsafe / "artifact",
            revision=_clean_revision(),
        )

    assert not (unsafe / "artifact").exists()
