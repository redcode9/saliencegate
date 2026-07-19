from __future__ import annotations

import json
import os
import socket
import stat
from pathlib import Path
from typing import cast

import pytest

import saliencegate.benchmarks.state_decay.runner as runner_module
from saliencegate.benchmarks.state_decay.diagnostic import (
    StateDecayDiagnosticError,
    run_state_decay_diagnostic,
)
from saliencegate.benchmarks.state_decay.generator import (
    SmokeCoverageError,
    encode_scenarios_jsonl,
    generate_smoke_scenarios,
)
from saliencegate.benchmarks.state_decay.runner import (
    BenchmarkArtifactValidationError,
    BenchmarkCommandError,
    BenchmarkCommandReport,
    SmokeManifest,
    build_state_decay_manifest,
    render_benchmark_human,
    render_benchmark_json,
    run_state_decay_smoke,
    state_decay_artifact_files,
    validate_state_decay_artifact,
)
from saliencegate.domain import canonical_json


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _create(tmp_path: Path, name: str = "smoke") -> tuple[Path, BenchmarkCommandReport]:
    output = tmp_path / name
    return output, run_state_decay_smoke(output)


def test_native_files_reports_and_validation_are_byte_deterministic(
    tmp_path: Path,
) -> None:
    first_files = state_decay_artifact_files()
    second_files = state_decay_artifact_files()
    assert first_files == second_files
    assert set(first_files) == {"manifest.json", "smoke.jsonl"}

    manifest = SmokeManifest.model_validate_json(first_files["manifest.json"])
    assert (
        manifest.manifest_digest
        == "32600f0adce1c21d7081cf2fb01d18722eb4d78c0b49f6b255f3f333962dc3f0"
    )
    assert (
        manifest.fixture_digest
        == "34fcf4ab0bee256ad7d091da261eb190b2d7a96f3dff0ef9eaef8846c32e880e"
    )
    assert (
        manifest.oracle_result_digest
        == "f27879d5054a2283a88cc74df1368bdf97ba6ef04eefd386bd7f1bda28a8f0b2"
    )
    assert (
        manifest.overall_content_digest
        == "a210eed1a07cc6de41c27a20b522e5674b05f91a1ac449137d6c80008d2509f7"
    )
    assert manifest.oracle_version == "paired-continuation-oracle/v1"
    assert manifest.oracle_result_schema_version == "state-decay-oracle-result/v1"
    assert manifest.scenario_count == manifest.oracle_passed == 32
    assert manifest.oracle_failed == 0

    first, one = _create(tmp_path, "first")
    second, two = _create(tmp_path, "second")
    assert one == two
    assert _tree(first) == _tree(second) == first_files

    validation = validate_state_decay_artifact(
        first / "manifest.json",
        expected_manifest_digest=one.manifest_digest,
    )
    assert validation.valid is True
    assert validation.assurance == "deterministic_synthetic_oracle"
    assert validation.confirmatory is False
    assert validation.external_claims_supported is False
    assert validation.expected_digest_matched is True
    assert validation.manifest_digest == one.manifest_digest
    assert validate_state_decay_artifact(first).expected_digest_matched is None


def test_output_directory_named_manifest_json_is_validated_as_a_directory(
    tmp_path: Path,
) -> None:
    output, report = _create(tmp_path, "manifest.json")
    assert output.is_dir()
    assert set(_tree(output)) == {"manifest.json", "smoke.jsonl"}
    assert (
        validate_state_decay_artifact(output, expected_manifest_digest=report.manifest_digest).valid
        is True
    )


def test_report_renderers_are_canonical_and_state_the_evidence_boundary(
    tmp_path: Path,
) -> None:
    _, report = _create(tmp_path)
    encoded = render_benchmark_json(report)
    assert encoded.endswith("\n")
    assert json.loads(encoded) == report.model_dump(mode="json")
    assert canonical_json(json.loads(encoded)) == encoded[:-1].encode()
    assert "output" not in json.loads(encoded)

    human = render_benchmark_human(report)
    assert "StateDecayBench smoke complete" in human
    assert "labels: 16 intervene, 16 silence" in human
    assert "oracle: 32 passed, 0 failed" in human
    assert "diagnostic: yes" in human
    assert "synthetic: yes" in human
    assert "balanced: yes" in human
    assert "external claims: insufficient" in human


def test_generation_and_oracle_require_the_exact_frozen_fixture() -> None:
    scenarios = generate_smoke_scenarios()
    fixture = encode_scenarios_jsonl(scenarios)
    manifest = build_state_decay_manifest(scenarios, fixture)
    assert canonical_json(manifest) == state_decay_artifact_files()["manifest.json"]

    with pytest.raises(BenchmarkArtifactValidationError):
        build_state_decay_manifest(scenarios, fixture + b"{}\n")
    with pytest.raises(BenchmarkArtifactValidationError):
        build_state_decay_manifest(scenarios[:-1], fixture)
    with pytest.raises(BenchmarkArtifactValidationError):
        build_state_decay_manifest(scenarios, cast("bytes", bytearray(fixture)))


@pytest.mark.parametrize(
    "tamper",
    [
        "manifest_whitespace",
        "fixture_row",
        "extra_file",
        "missing_file",
        "manifest_symlink",
        "fixture_hardlink",
    ],
)
def test_validation_rejects_noncanonical_unsafe_or_incomplete_trees(
    tmp_path: Path,
    tamper: str,
) -> None:
    output, _ = _create(tmp_path)
    manifest = output / "manifest.json"
    fixture = output / "smoke.jsonl"
    if tamper == "manifest_whitespace":
        manifest.write_bytes(manifest.read_bytes() + b"\n")
    elif tamper == "fixture_row":
        fixture.write_bytes(fixture.read_bytes() + b"{}\n")
    elif tamper == "extra_file":
        (output / "extra.json").write_text("{}")
    elif tamper == "missing_file":
        fixture.unlink()
    elif tamper == "manifest_symlink":
        source = tmp_path / "external-manifest"
        source.write_bytes(manifest.read_bytes())
        manifest.unlink()
        manifest.symlink_to(source)
    else:
        source = tmp_path / "external-fixture"
        source.write_bytes(fixture.read_bytes())
        fixture.unlink()
        os.link(source, fixture)

    with pytest.raises(BenchmarkArtifactValidationError):
        validate_state_decay_artifact(output)


def test_validation_reconstructs_oracle_and_rejects_a_resealed_false_digest(
    tmp_path: Path,
) -> None:
    output, _ = _create(tmp_path)
    manifest_path = output / "manifest.json"
    payload = json.loads(manifest_path.read_bytes())
    payload["oracle_result_digest"] = "a" * 64
    payload["overall_content_digest"] = runner_module._overall_content_digest(
        fixture_byte_count=payload["fixture_byte_count"],
        fixture_digest=payload["fixture_digest"],
        oracle_result_digest=payload["oracle_result_digest"],
    )
    payload["manifest_digest"] = runner_module._manifest_digest(payload)
    SmokeManifest.model_validate(payload)
    manifest_path.write_bytes(canonical_json(payload))

    with pytest.raises(BenchmarkArtifactValidationError):
        validate_state_decay_artifact(output)


def test_replace_is_explicit_and_only_authorizes_the_same_valid_fixture(
    tmp_path: Path,
) -> None:
    output, report = _create(tmp_path)
    before = _tree(output)
    with pytest.raises(BenchmarkCommandError):
        run_state_decay_smoke(output)
    assert _tree(output) == before

    assert run_state_decay_smoke(output, replace=True) == report
    assert _tree(output) == before
    assert tuple(tmp_path.glob(".smoke.*")) == (tmp_path / ".smoke.lock",)

    arbitrary = tmp_path / "arbitrary"
    arbitrary.mkdir()
    sentinel = arbitrary / "user-data.txt"
    sentinel.write_text("preserve")
    with pytest.raises(BenchmarkCommandError):
        run_state_decay_smoke(arbitrary, replace=True)
    assert sentinel.read_text() == "preserve"

    (output / "smoke.jsonl").write_bytes(before["smoke.jsonl"] + b"{}\n")
    corrupted = _tree(output)
    with pytest.raises(BenchmarkArtifactValidationError):
        run_state_decay_smoke(output, replace=True)
    assert _tree(output) == corrupted


def _write_interrupted_marker(
    output: Path,
    staging: Path,
    *,
    publish_replacement: bool,
) -> tuple[Path, Path]:
    parent = output.parent
    files = state_decay_artifact_files()
    staging.mkdir(mode=0o700)
    for name, data in files.items():
        (staging / name).write_bytes(data)
    old = runner_module._load_state_decay_artifact(output)
    new = SmokeManifest.model_validate_json(files["manifest.json"])
    original_identity = runner_module._FileIdentity.from_stat(output.lstat())
    replacement_identity = runner_module._FileIdentity.from_stat(staging.lstat())
    backup, marker = runner_module._replacement_paths(output, parent)
    marker.write_bytes(
        runner_module._replacement_marker_bytes(
            output,
            original_identity,
            replacement_identity,
            original_manifest_digest=old.manifest_digest,
            replacement_manifest_digest=new.manifest_digest,
            fixture_digest=new.fixture_digest,
        )
    )
    marker.chmod(0o600)
    os.replace(output, backup)
    if publish_replacement:
        os.replace(staging, output)
    return backup, marker


@pytest.mark.parametrize("publish_replacement", [False, True])
def test_interrupted_replace_recovers_old_or_completed_publication(
    tmp_path: Path,
    publish_replacement: bool,
) -> None:
    output, expected = _create(tmp_path)
    staging = tmp_path / ".manual-staging"
    backup, marker = _write_interrupted_marker(
        output,
        staging,
        publish_replacement=publish_replacement,
    )

    assert run_state_decay_smoke(output, replace=True) == expected
    assert validate_state_decay_artifact(output).valid is True
    assert not backup.exists()
    assert not marker.exists()
    if not publish_replacement:
        assert staging.exists()


def test_tampered_marker_and_orphan_backup_fail_closed(tmp_path: Path) -> None:
    output, _ = _create(tmp_path, "marked")
    marker = tmp_path / ".marked.replace.json"
    marker.write_bytes(b"{}")
    marker.chmod(0o600)
    before = _tree(output)
    with pytest.raises(BenchmarkArtifactValidationError):
        run_state_decay_smoke(output, replace=True)
    assert _tree(output) == before

    orphan, _ = _create(tmp_path, "orphan")
    backup = tmp_path / ".orphan.backup"
    os.replace(orphan, backup)
    with pytest.raises(BenchmarkArtifactValidationError):
        run_state_decay_smoke(orphan, replace=True)
    assert not orphan.exists()
    assert validate_state_decay_artifact(backup).valid is True


@pytest.mark.parametrize("raise_after_move", [False, True])
def test_replace_failure_never_loses_the_previous_valid_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raise_after_move: bool,
) -> None:
    output, _ = _create(tmp_path)
    before = _tree(output)
    real_replace = runner_module.os.replace

    def fail_staging_publish(
        source: os.PathLike[str] | str,
        target: os.PathLike[str] | str,
    ) -> None:
        source_path = Path(source)
        target_path = Path(target)
        if source_path.name.startswith(".smoke.tmp-") and target_path == output:
            if raise_after_move:
                real_replace(source, target)
            raise OSError("simulated atomic publish failure")
        real_replace(source, target)

    monkeypatch.setattr(runner_module.os, "replace", fail_staging_publish)
    with pytest.raises(BenchmarkCommandError):
        run_state_decay_smoke(output, replace=True)
    assert _tree(output) == before
    assert validate_state_decay_artifact(output).valid is True
    assert not (tmp_path / ".smoke.backup").exists()
    assert not (tmp_path / ".smoke.replace.json").exists()


def test_destination_parent_and_persistent_lock_are_owner_controlled(
    tmp_path: Path,
) -> None:
    unsafe_parent = tmp_path / "unsafe"
    unsafe_parent.mkdir(mode=0o700)
    if os.name == "posix":
        unsafe_parent.chmod(0o777)
        try:
            with pytest.raises(BenchmarkCommandError):
                run_state_decay_smoke(unsafe_parent / "smoke")
        finally:
            unsafe_parent.chmod(0o700)

    lock = tmp_path / ".locked.lock"
    lock.write_bytes(b"")
    lock.chmod(0o644)
    with pytest.raises(BenchmarkCommandError):
        run_state_decay_smoke(tmp_path / "locked")
    assert stat.S_IMODE(lock.stat().st_mode) == 0o644


def test_runner_is_offline_and_boundaries_are_value_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def network_forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", network_forbidden)
    output, report = _create(tmp_path)
    assert report.scenario_count == 32

    with pytest.raises(BenchmarkCommandError, match="benchmark input or output is invalid"):
        run_state_decay_smoke(cast("Path", b"invalid"))
    with pytest.raises(BenchmarkCommandError):
        run_state_decay_smoke(tmp_path / "invalid-replace", replace=cast("bool", 1))
    with pytest.raises(BenchmarkArtifactValidationError):
        validate_state_decay_artifact(output, expected_manifest_digest="invalid")
    with pytest.raises(BenchmarkArtifactValidationError):
        validate_state_decay_artifact(tmp_path / "missing")

    invalid_report = BenchmarkCommandReport.model_construct(
        **{**report.model_dump(), "scenario_count": 31}
    )
    with pytest.raises(BenchmarkCommandError):
        render_benchmark_json(invalid_report)
    with pytest.raises(BenchmarkCommandError):
        render_benchmark_human(invalid_report)


def test_manifest_rejects_independently_forged_invariants() -> None:
    files = state_decay_artifact_files()
    payload = json.loads(files["manifest.json"])
    payload["overall_content_digest"] = "a" * 64
    with pytest.raises(ValueError, match="content digest"):
        SmokeManifest.model_validate(payload)

    payload = json.loads(files["manifest.json"])
    payload["manifest_digest"] = "a" * 64
    with pytest.raises(ValueError, match="manifest digest"):
        SmokeManifest.model_validate(payload)


def test_manifest_builder_consumes_the_shared_diagnostic_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = generate_smoke_scenarios()
    fixture = encode_scenarios_jsonl(scenarios)
    real_evaluate = runner_module.evaluate_state_decay_scenarios
    calls = 0

    def evaluate(values: object) -> object:
        nonlocal calls
        calls += 1
        return real_evaluate(values)  # type: ignore[arg-type]

    monkeypatch.setattr(runner_module, "evaluate_state_decay_scenarios", evaluate)
    assert build_state_decay_manifest(scenarios, fixture).scenario_count == 32
    assert calls == 1

    def diagnostic_failure(values: object) -> object:
        del values
        raise StateDecayDiagnosticError()

    monkeypatch.setattr(runner_module, "evaluate_state_decay_scenarios", diagnostic_failure)
    with pytest.raises(BenchmarkArtifactValidationError):
        build_state_decay_manifest(scenarios, fixture)

    monkeypatch.setattr(runner_module, "evaluate_state_decay_scenarios", real_evaluate)
    monkeypatch.setattr(
        runner_module,
        "encode_scenarios_jsonl",
        lambda scenarios: (_ for _ in ()).throw(RuntimeError()),
    )
    with pytest.raises(BenchmarkArtifactValidationError):
        build_state_decay_manifest(scenarios, fixture)


def test_native_artifact_composes_the_shared_diagnostic_without_reevaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic = run_state_decay_diagnostic()
    calls = 0

    def run_diagnostic() -> object:
        nonlocal calls
        calls += 1
        return diagnostic

    monkeypatch.setattr(runner_module, "run_state_decay_diagnostic", run_diagnostic)
    monkeypatch.setattr(
        runner_module,
        "evaluate_state_decay_scenarios",
        lambda scenarios: (_ for _ in ()).throw(AssertionError("unexpected reevaluation")),
    )

    files = state_decay_artifact_files()

    assert calls == 1
    assert SmokeManifest.model_validate_json(files["manifest.json"]).scenario_count == 32

    monkeypatch.setattr(
        runner_module,
        "run_state_decay_diagnostic",
        lambda: (_ for _ in ()).throw(StateDecayDiagnosticError()),
    )
    with pytest.raises(BenchmarkArtifactValidationError):
        state_decay_artifact_files()


def test_artifact_encoder_coverage_failure_is_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner_module,
        "encode_scenarios_jsonl",
        lambda scenarios: (_ for _ in ()).throw(SmokeCoverageError()),
    )

    with pytest.raises(BenchmarkArtifactValidationError):
        state_decay_artifact_files()
    with pytest.raises(BenchmarkArtifactValidationError):
        run_state_decay_smoke(tmp_path / "encoder-failure")


def test_low_level_canonical_and_path_guards_are_value_free(tmp_path: Path) -> None:
    for invalid in (
        b'{"value":NaN}',
        b'{"value":1,"value":2}',
        b"[]",
        b"not-json",
    ):
        with pytest.raises(BenchmarkArtifactValidationError):
            runner_module._decode_canonical_object(invalid)
    for invalid in (b"", b"{}", b"{}\n"):
        with pytest.raises(BenchmarkArtifactValidationError):
            runner_module._decode_fixture(invalid)

    class BytePath:
        def __fspath__(self) -> bytes:
            return b"unsafe"

    for invalid_path in ("", ".", "..", BytePath()):
        with pytest.raises(BenchmarkCommandError):
            runner_module._path(cast("str", invalid_path))

    identity = runner_module._FileIdentity.from_stat(tmp_path.stat())
    assert runner_module._FileIdentity.from_payload(None) is None
    bad_payload = identity.payload()
    bad_payload["size"] = -1
    assert runner_module._FileIdentity.from_payload(bad_payload) is None


def test_loader_rejects_wrong_digest_manifest_symlink_and_unexpected_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _ = _create(tmp_path)
    with pytest.raises(BenchmarkArtifactValidationError):
        validate_state_decay_artifact(output, expected_manifest_digest="a" * 64)

    link = tmp_path / "manifest-link.json"
    link.symlink_to(output / "manifest.json")
    renamed = tmp_path / "manifest.json"
    renamed.symlink_to(output / "manifest.json")
    with pytest.raises(BenchmarkArtifactValidationError):
        validate_state_decay_artifact(renamed)

    real_read = runner_module._read_artifact
    monkeypatch.setattr(
        runner_module,
        "_read_artifact",
        lambda path: (_ for _ in ()).throw(RuntimeError()),
    )
    with pytest.raises(BenchmarkArtifactValidationError):
        validate_state_decay_artifact(output)
    monkeypatch.setattr(runner_module, "_read_artifact", real_read)
    with pytest.raises(BenchmarkArtifactValidationError):
        validate_state_decay_artifact(cast("Path", b"unsafe"))


@pytest.mark.parametrize("unsafe_kind", ["empty", "oversized", "fifo"])
def test_component_type_and_size_limits_fail_before_parsing(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    output, _ = _create(tmp_path)
    fixture = output / "smoke.jsonl"
    fixture.unlink()
    if unsafe_kind == "empty":
        fixture.write_bytes(b"")
    elif unsafe_kind == "oversized":
        with fixture.open("wb") as stream:
            stream.truncate(runner_module._MAX_FIXTURE_BYTES + 1)
    else:
        os.mkfifo(fixture)
    with pytest.raises(BenchmarkArtifactValidationError):
        validate_state_decay_artifact(output)


def test_read_and_write_os_failures_are_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _ = _create(tmp_path)
    real_listdir = runner_module.os.listdir
    monkeypatch.setattr(
        runner_module.os,
        "listdir",
        lambda path: (_ for _ in ()).throw(OSError()),
    )
    with pytest.raises(BenchmarkArtifactValidationError):
        validate_state_decay_artifact(output)
    monkeypatch.setattr(runner_module.os, "listdir", real_listdir)

    directory = tmp_path / "short-write"
    directory.mkdir()
    real_write = runner_module.os.write
    monkeypatch.setattr(runner_module.os, "write", lambda descriptor, data: 0)
    with pytest.raises(OSError):
        runner_module._write_file(directory, "file", b"data")
    monkeypatch.setattr(runner_module.os, "write", real_write)


def test_stale_valid_marker_is_removed_and_invalid_backup_is_preserved(
    tmp_path: Path,
) -> None:
    output, expected = _create(tmp_path, "stale")
    staging = tmp_path / ".stale-manual"
    parent = output.parent
    files = state_decay_artifact_files()
    staging.mkdir()
    for name, data in files.items():
        (staging / name).write_bytes(data)
    current = runner_module._load_state_decay_artifact(output)
    replacement = SmokeManifest.model_validate_json(files["manifest.json"])
    _, marker = runner_module._replacement_paths(output, parent)
    marker.write_bytes(
        runner_module._replacement_marker_bytes(
            output,
            runner_module._FileIdentity.from_stat(output.lstat()),
            runner_module._FileIdentity.from_stat(staging.lstat()),
            original_manifest_digest=current.manifest_digest,
            replacement_manifest_digest=replacement.manifest_digest,
            fixture_digest=current.fixture_digest,
        )
    )
    marker.chmod(0o600)
    assert run_state_decay_smoke(output, replace=True) == expected
    assert not marker.exists()

    broken, _ = _create(tmp_path, "broken")
    broken_staging = tmp_path / ".broken-manual"
    backup, broken_marker = _write_interrupted_marker(
        broken,
        broken_staging,
        publish_replacement=False,
    )
    (backup / "smoke.jsonl").write_bytes(b"{}\n")
    with pytest.raises(BenchmarkArtifactValidationError):
        run_state_decay_smoke(broken, replace=True)
    assert backup.exists()
    assert broken_marker.exists()


def test_existing_files_symlinks_and_unsafe_locks_are_never_replaced(
    tmp_path: Path,
) -> None:
    regular = tmp_path / "regular"
    regular.write_text("preserve")
    with pytest.raises(BenchmarkCommandError):
        run_state_decay_smoke(regular, replace=True)
    assert regular.read_text() == "preserve"

    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "symlink"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(BenchmarkCommandError):
        run_state_decay_smoke(symlink, replace=True)

    lock_target = tmp_path / "lock-target"
    lock_target.write_bytes(b"")
    lock = tmp_path / ".locked-link.lock"
    lock.symlink_to(lock_target)
    with pytest.raises(BenchmarkCommandError):
        run_state_decay_smoke(tmp_path / "locked-link")

    parent_file = tmp_path / "parent-file"
    parent_file.write_text("preserve")
    with pytest.raises(BenchmarkCommandError):
        run_state_decay_smoke(parent_file / "smoke")


def test_run_maps_diagnostic_contract_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner_module,
        "state_decay_artifact_files",
        lambda: (_ for _ in ()).throw(StateDecayDiagnosticError()),
    )
    with pytest.raises(BenchmarkArtifactValidationError):
        run_state_decay_smoke(tmp_path / "diagnostic")


def test_secure_file_reader_sanitizes_missing_stat_read_and_identity_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"{}")
    directory_fd = os.open(tmp_path, os.O_RDONLY)
    try:
        with pytest.raises(BenchmarkArtifactValidationError):
            runner_module._read_regular_file(directory_fd, "missing", maximum=10)

        real_stat = runner_module.os.stat

        def fail_relative_stat(*args: object, **kwargs: object) -> os.stat_result:
            if kwargs.get("dir_fd") is not None:
                raise OSError
            return real_stat(*args, **kwargs)

        monkeypatch.setattr(runner_module.os, "stat", fail_relative_stat)
        with pytest.raises(BenchmarkArtifactValidationError):
            runner_module._read_regular_file(directory_fd, "source", maximum=10)
        monkeypatch.setattr(runner_module.os, "stat", real_stat)

        real_read = runner_module.os.read
        monkeypatch.setattr(
            runner_module.os,
            "read",
            lambda descriptor, size: (_ for _ in ()).throw(OSError()),
        )
        with pytest.raises(BenchmarkArtifactValidationError):
            runner_module._read_regular_file(directory_fd, "source", maximum=10)
        monkeypatch.setattr(runner_module.os, "read", real_read)

        other = tmp_path / "other"
        other.write_bytes(b"{}")
        real_fstat = runner_module.os.fstat
        calls = 0

        def changed_fstat(descriptor: int) -> os.stat_result:
            nonlocal calls
            calls += 1
            return other.stat() if calls == 2 else real_fstat(descriptor)

        monkeypatch.setattr(runner_module.os, "fstat", changed_fstat)
        with pytest.raises(BenchmarkArtifactValidationError):
            runner_module._read_regular_file(directory_fd, "source", maximum=10)
    finally:
        os.close(directory_fd)


def test_directory_reader_detects_path_disappearance_and_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _ = _create(tmp_path)
    other = tmp_path / "other-directory"
    other.mkdir()
    real_lstat = Path.lstat

    def missing_lstat(path: Path) -> os.stat_result:
        if path == output:
            raise OSError
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", missing_lstat)
    with pytest.raises(BenchmarkArtifactValidationError):
        runner_module._read_artifact(output)
    monkeypatch.setattr(Path, "lstat", real_lstat)

    monkeypatch.setattr(
        Path,
        "lstat",
        lambda path: other.stat() if path == output else real_lstat(path),
    )
    with pytest.raises(BenchmarkArtifactValidationError):
        runner_module._read_artifact(output)


def test_canonical_fixture_and_manifest_must_include_serialized_defaults(
    tmp_path: Path,
) -> None:
    output, _ = _create(tmp_path)
    fixture = output / "smoke.jsonl"
    lines = fixture.read_bytes().splitlines()
    scenario = json.loads(lines[0])
    scenario.pop("schema_version")
    lines[0] = canonical_json(scenario)
    fixture.write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(BenchmarkArtifactValidationError):
        validate_state_decay_artifact(output)

    output, _ = _create(tmp_path, "manifest-default")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest.pop("schema_version")
    manifest_path.write_bytes(canonical_json(manifest))
    with pytest.raises(BenchmarkArtifactValidationError):
        validate_state_decay_artifact(output)

    with pytest.raises(BenchmarkArtifactValidationError):
        validate_state_decay_artifact(tmp_path / "missing" / "manifest.json")


def test_lock_identity_checks_and_owned_cleanup_refuse_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_matches = runner_module._FileIdentity.matches
    calls = 0

    def changed_lock(
        identity: runner_module._FileIdentity,
        metadata: os.stat_result,
    ) -> bool:
        nonlocal calls
        calls += 1
        return calls != 2 and real_matches(identity, metadata)

    monkeypatch.setattr(runner_module._FileIdentity, "matches", changed_lock)
    with pytest.raises(BenchmarkCommandError):
        run_state_decay_smoke(tmp_path / "changed-lock")
    monkeypatch.setattr(runner_module._FileIdentity, "matches", real_matches)

    directory = tmp_path / "owned"
    directory.mkdir()
    file = tmp_path / "owned-file"
    file.write_bytes(b"x")
    wrong = runner_module._FileIdentity.from_stat(tmp_path.stat())
    assert runner_module._remove_owned_tree(directory, wrong) is False
    assert runner_module._unlink_owned_file(file, wrong) is False


def test_recovery_rejects_unrelated_destination_and_failed_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _ = _create(tmp_path, "unrelated")
    staging = tmp_path / ".unrelated-manual"
    backup, marker = _write_interrupted_marker(
        output,
        staging,
        publish_replacement=False,
    )
    output.mkdir()
    (output / "user-data").write_text("preserve")
    with pytest.raises(BenchmarkArtifactValidationError):
        run_state_decay_smoke(output, replace=True)
    assert backup.exists() and marker.exists()
    assert (output / "user-data").read_text() == "preserve"

    completed, _ = _create(tmp_path, "cleanup")
    completed_staging = tmp_path / ".cleanup-manual"
    cleanup_backup, cleanup_marker = _write_interrupted_marker(
        completed,
        completed_staging,
        publish_replacement=True,
    )
    real_remove = runner_module._remove_owned_tree
    monkeypatch.setattr(runner_module, "_remove_owned_tree", lambda path, identity: False)
    with pytest.raises(BenchmarkArtifactValidationError):
        runner_module._recover_replacement(completed, tmp_path)
    monkeypatch.setattr(runner_module, "_remove_owned_tree", real_remove)
    assert cleanup_backup.exists() and cleanup_marker.exists()


def _assert_value_free(error: BaseException) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None
    assert "fixture-secret" not in str(error)
    assert "fixture-secret" not in repr(error)


def test_parser_and_unexpected_failures_drop_raw_exception_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _ = _create(tmp_path, "manifest-secret")
    (output / "manifest.json").write_bytes(b'{"fixture-secret":NaN}')
    with pytest.raises(BenchmarkArtifactValidationError) as manifest_error:
        validate_state_decay_artifact(output)
    _assert_value_free(manifest_error.value)

    output, _ = _create(tmp_path, "row-secret")
    fixture = output / "smoke.jsonl"
    lines = fixture.read_bytes().splitlines()
    lines[0] = canonical_json({"fixture-secret": "raw-model-input"})
    fixture.write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(BenchmarkArtifactValidationError) as row_error:
        validate_state_decay_artifact(output)
    _assert_value_free(row_error.value)

    output, _ = _create(tmp_path, "unexpected-secret")
    monkeypatch.setattr(
        runner_module,
        "_read_artifact",
        lambda path: (_ for _ in ()).throw(ValueError("fixture-secret raw input")),
    )
    with pytest.raises(BenchmarkArtifactValidationError) as unexpected_error:
        validate_state_decay_artifact(output)
    _assert_value_free(unexpected_error.value)

    with pytest.raises(BenchmarkArtifactValidationError) as direct_json_error:
        runner_module._decode_canonical_object(b'{"fixture-secret":NaN}')
    _assert_value_free(direct_json_error.value)

    with pytest.raises(BenchmarkArtifactValidationError) as surrogate_error:
        runner_module._decode_canonical_object(b'{"x":"\\ud800"}')
    _assert_value_free(surrogate_error.value)


def test_run_and_render_boundaries_drop_nested_secret_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nested_oracle_failure() -> dict[str, bytes]:
        try:
            raise ValueError("fixture-secret raw input")
        except ValueError:
            raise StateDecayDiagnosticError() from None

    monkeypatch.setattr(
        runner_module,
        "state_decay_artifact_files",
        nested_oracle_failure,
    )
    with pytest.raises(BenchmarkArtifactValidationError) as run_error:
        run_state_decay_smoke(tmp_path / "secret")
    _assert_value_free(run_error.value)

    valid = state_decay_artifact_files()
    report = BenchmarkCommandReport.model_construct(
        fixture_digest="fixture-secret",
        oracle_result_digest="a" * 64,
        overall_content_digest="b" * 64,
        manifest_digest="c" * 64,
    )
    assert valid
    with pytest.raises(BenchmarkCommandError) as render_error:
        render_benchmark_json(report)
    _assert_value_free(render_error.value)
