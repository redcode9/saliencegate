from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from tests.cli.conftest import FIXTURES, RunCli

from saliencegate.artifacts import ArtifactValidationError
from saliencegate.benchmarks.state_decay.runner import (
    BenchmarkValidationReport,
    run_state_decay_smoke,
)
from saliencegate.commands.validate import (
    ValidationReport,
    render_validate_human,
    render_validate_json,
    run_validate,
)

TRACE = FIXTURES / "runs" / "basic.jsonl"


def _replay(run_cli: RunCli, root: Path) -> dict[str, object]:
    completed = run_cli("replay", str(TRACE), "--output", str(root), "--json")
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert isinstance(payload, dict)
    return payload


def test_validate_json_accepts_manifest_or_directory_and_expected_digest(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    root = tmp_path / "artifact"
    replay = _replay(run_cli, root)

    by_manifest = run_cli(
        "validate",
        str(root / "manifest.json"),
        "--expected-digest",
        str(replay["manifest_digest"]),
        "--json",
    )
    by_directory = run_cli("validate", str(root), "--json")

    assert by_manifest.returncode == by_directory.returncode == 0
    assert by_manifest.stderr == by_directory.stderr == ""
    report = json.loads(by_manifest.stdout)
    assert set(report) == {
        "component_count",
        "confirmatory",
        "expected_digest_matched",
        "grounding_assurance",
        "integrity_valid",
        "manifest_digest",
        "overall_content_digest",
        "schema_version",
        "structurally_valid",
        "valid",
    }
    assert report["valid"] is True
    assert report["integrity_valid"] is True
    assert report["structurally_valid"] is True
    assert report["expected_digest_matched"] is True
    assert report["grounding_assurance"] == "producer_attested_digest_only"
    assert json.loads(by_directory.stdout)["expected_digest_matched"] is None


def test_validate_human_mode_is_concise_and_explicit_about_grounding_assurance(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    root = tmp_path / "artifact"
    _replay(run_cli, root)

    completed = run_cli("validate", str(root / "manifest.json"))

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert "Artifact valid" in completed.stdout
    assert "producer-attested digest-only grounding" in completed.stdout
    assert "manifest digest:" in completed.stdout


def test_validate_maps_corruption_expected_mismatch_and_confirmatory_requirement(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    root = tmp_path / "artifact"
    _replay(run_cli, root)

    mismatch = run_cli(
        "validate",
        str(root),
        "--expected-digest",
        "0" * 64,
        "--json",
    )
    assert mismatch.returncode == 5
    assert mismatch.stdout == ""
    assert mismatch.stderr == "error: artifact validation failed\n"

    confirmatory = run_cli("validate", str(root), "--require-confirmatory", "--json")
    assert confirmatory.returncode == 5
    assert confirmatory.stderr == "error: artifact validation failed\n"

    (root / "decisions.json").write_bytes((root / "decisions.json").read_bytes() + b" ")
    corrupted = run_cli("validate", str(root), "--json")
    assert corrupted.returncode == 5
    assert corrupted.stdout == ""
    assert corrupted.stderr == "error: artifact validation failed\n"


def test_validate_missing_and_unknown_commands_use_stable_invalid_input_exit(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    missing = run_cli(
        "validate",
        str(tmp_path / "fixture-secret-missing"),
        "--json",
    )
    assert missing.returncode == 2
    assert missing.stdout == ""
    assert missing.stderr == "error: artifact path is invalid\n"
    assert "fixture-secret" not in missing.stderr

    unknown = run_cli("fixture-secret-command", "--json")
    assert unknown.returncode == 2
    assert unknown.stdout == ""
    assert unknown.stderr == "error: invalid command line\n"
    assert "fixture-secret" not in unknown.stderr


def test_validate_rejects_empty_and_misnamed_regular_paths(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    wrong = tmp_path / "fixture-secret-not-a-manifest.json"
    wrong.write_text("{}")

    empty = run_cli("validate", "", "--json")
    misnamed = run_cli("validate", str(wrong), "--json")

    assert empty.returncode == misnamed.returncode == 2
    assert empty.stdout == misnamed.stdout == ""
    assert empty.stderr == misnamed.stderr == "error: artifact path is invalid\n"
    assert "fixture-secret" not in misnamed.stderr


def test_native_benchmark_validation_renderers_and_invalid_types(
    tmp_path: Path,
) -> None:
    output = tmp_path / "smoke"
    benchmark = run_state_decay_smoke(output)
    report = run_validate(output, expected_digest=benchmark.manifest_digest)

    assert type(report) is BenchmarkValidationReport
    assert json.loads(render_validate_json(report)) == report.model_dump(mode="json")
    human = render_validate_human(report)
    assert "Benchmark artifact valid" in human
    assert "deterministic synthetic oracle" in human
    assert "confirmatory: no" in human
    assert "external claims: unsupported" in human

    invalid = cast("ValidationReport", object())
    with pytest.raises(ArtifactValidationError):
        render_validate_json(invalid)
    with pytest.raises(ArtifactValidationError):
        render_validate_human(invalid)
