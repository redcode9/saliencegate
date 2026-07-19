from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.cli.conftest import RunCli

from saliencegate.artifacts import ArtifactValidationError
from saliencegate.commands.validate import run_validate


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_state_decay_smoke_json_is_offline_balanced_and_byte_deterministic(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    one = run_cli(
        "benchmark",
        "state-decay-smoke",
        "--output",
        str(first),
        "--json",
    )
    two = run_cli(
        "benchmark",
        "state-decay-smoke",
        "--output",
        str(second),
        "--json",
    )

    assert one.returncode == two.returncode == 0
    assert one.stderr == two.stderr == ""
    assert one.stdout == two.stdout
    report = json.loads(one.stdout)
    assert set(report) == {
        "balanced",
        "diagnostic",
        "external_claims_assessment",
        "external_claims_supported",
        "family_count",
        "fixture_digest",
        "generator_version",
        "intervene_count",
        "manifest_digest",
        "oracle_failed",
        "oracle_passed",
        "oracle_result_digest",
        "overall_content_digest",
        "scenario_count",
        "schema_version",
        "seed",
        "silence_count",
        "status",
        "suite_id",
        "suite_version",
        "synthetic",
    }
    assert report == {
        **report,
        "balanced": True,
        "diagnostic": True,
        "external_claims_assessment": "insufficient",
        "external_claims_supported": False,
        "family_count": 8,
        "intervene_count": 16,
        "oracle_failed": 0,
        "oracle_passed": 32,
        "scenario_count": 32,
        "schema_version": "cli-benchmark-report/v1",
        "silence_count": 16,
        "status": "ok",
        "suite_id": "state-decay-smoke",
        "synthetic": True,
    }
    assert set(_tree(first)) == {"manifest.json", "smoke.jsonl"}
    assert _tree(first) == _tree(second)


def test_state_decay_smoke_human_report_states_the_evidence_boundary(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    completed = run_cli(
        "benchmark",
        "state-decay-smoke",
        "--output",
        str(tmp_path / "smoke"),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert "StateDecayBench smoke complete" in completed.stdout
    assert "scenarios: 32" in completed.stdout
    assert "families: 8" in completed.stdout
    assert "labels: 16 intervene, 16 silence" in completed.stdout
    assert "diagnostic: yes" in completed.stdout
    assert "synthetic: yes" in completed.stdout
    assert "balanced: yes" in completed.stdout
    assert "external claims: insufficient" in completed.stdout


def test_state_decay_smoke_requires_explicit_replace_and_preserves_arbitrary_data(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    output = tmp_path / "smoke"
    created = run_cli(
        "benchmark",
        "state-decay-smoke",
        "--output",
        str(output),
        "--json",
    )
    assert created.returncode == 0
    before = _tree(output)

    refused = run_cli(
        "benchmark",
        "state-decay-smoke",
        "--output",
        str(output),
        "--json",
    )
    assert refused.returncode == 2
    assert refused.stdout == ""
    assert refused.stderr == "error: benchmark input or output is invalid\n"
    assert _tree(output) == before

    replaced = run_cli(
        "benchmark",
        "state-decay-smoke",
        "--output",
        str(output),
        "--replace",
        "--json",
    )
    assert replaced.returncode == 0
    assert _tree(output) == before

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    sentinel = unrelated / "fixture-secret-user-data.txt"
    sentinel.write_text("must survive")
    rejected = run_cli(
        "benchmark",
        "state-decay-smoke",
        "--output",
        str(unrelated),
        "--replace",
        "--json",
    )
    assert rejected.returncode == 2
    assert rejected.stdout == ""
    assert rejected.stderr == "error: benchmark input or output is invalid\n"
    assert sentinel.read_text() == "must survive"
    assert tuple(unrelated.iterdir()) == (sentinel,)


def test_benchmark_usage_and_unknown_suite_are_value_free(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    missing_output = run_cli("benchmark", "state-decay-smoke", "--json")
    assert missing_output.returncode == 2
    assert missing_output.stdout == ""
    assert missing_output.stderr == "error: invalid command line\n"

    unknown = run_cli(
        "benchmark",
        "fixture-secret-suite",
        "--output",
        str(tmp_path / "unknown"),
        "--json",
    )
    assert unknown.returncode == 2
    assert unknown.stdout == ""
    assert unknown.stderr == "error: benchmark input or output is invalid\n"
    assert "fixture-secret" not in unknown.stderr


def test_state_decay_smoke_tampering_is_corruption_and_is_not_replaced(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    output = tmp_path / "smoke"
    assert (
        run_cli(
            "benchmark",
            "state-decay-smoke",
            "--output",
            str(output),
            "--json",
        ).returncode
        == 0
    )
    fixture = output / "smoke.jsonl"
    fixture.write_bytes(fixture.read_bytes() + b"{}\n")
    before = _tree(output)

    completed = run_cli(
        "benchmark",
        "state-decay-smoke",
        "--output",
        str(output),
        "--replace",
        "--json",
    )

    assert completed.returncode == 5
    assert completed.stdout == ""
    assert completed.stderr == "error: benchmark artifact validation failed\n"
    assert _tree(output) == before

    with pytest.raises(ArtifactValidationError) as captured:
        run_validate(output)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_validate_dispatches_to_the_native_benchmark_assurance_report(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    output = tmp_path / "smoke"
    created = run_cli(
        "benchmark",
        "state-decay-smoke",
        "--output",
        str(output),
        "--json",
    )
    assert created.returncode == 0
    benchmark = json.loads(created.stdout)

    completed = run_cli(
        "validate",
        str(output / "manifest.json"),
        "--expected-digest",
        benchmark["manifest_digest"],
        "--json",
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    report = json.loads(completed.stdout)
    assert report == {
        "assurance": "deterministic_synthetic_oracle",
        "confirmatory": False,
        "expected_digest_matched": True,
        "external_claims_supported": False,
        "fixture_digest": benchmark["fixture_digest"],
        "integrity_valid": True,
        "manifest_digest": benchmark["manifest_digest"],
        "oracle_result_digest": benchmark["oracle_result_digest"],
        "overall_content_digest": benchmark["overall_content_digest"],
        "scenario_count": 32,
        "schema_version": "benchmark-validation-report/v1",
        "structurally_valid": True,
        "valid": True,
    }

    refused = run_cli(
        "validate",
        str(output),
        "--require-confirmatory",
        "--json",
    )
    assert refused.returncode == 5
    assert refused.stdout == ""
    assert refused.stderr == "error: artifact validation failed\n"
