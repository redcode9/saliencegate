from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from tests.cli.conftest import FIXTURES, RunCli
from tests.runtime.test_engine import _run_frozen_replay

from saliencegate.artifacts import (
    ArtifactClassification,
    SyntheticArtifactContent,
    export_replay_artifact,
)
from saliencegate.artifacts.manifest import RevisionEvidence, RevisionSource
from saliencegate.commands import inspect as inspect_module
from saliencegate.commands.inspect import (
    InspectInputError,
    InspectReport,
    InspectRunMismatchError,
    render_inspect_human,
    render_inspect_json,
    run_inspect,
)
from saliencegate.domain import CycleState, ReasonCode

TRACE = FIXTURES / "runs" / "basic.jsonl"


def _replay(run_cli: RunCli, root: Path) -> dict[str, object]:
    completed = run_cli("replay", str(TRACE), "--output", str(root), "--json")
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert isinstance(payload, dict)
    return payload


def test_inspect_json_exposes_typed_evidence_and_no_sensitive_fields(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    root = tmp_path / "artifact"
    replay = _replay(run_cli, root)

    completed = run_cli(
        "inspect",
        str(replay["run_id"]),
        "--artifact",
        str(root),
        "--json",
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    report = json.loads(completed.stdout)
    assert set(report) == {
        "attestations",
        "budgets",
        "cycles",
        "decisions",
        "deliveries",
        "execution",
        "manifest",
        "outcomes",
        "run_id",
        "schema_version",
        "status",
    }
    assert report["schema_version"] == "cli-inspect-report/v1"
    assert report["status"] == "ok"
    assert report["run_id"] == replay["run_id"]
    assert report["manifest"]["manifest_digest"] == replay["manifest_digest"]
    assert report["execution"]["trace_digest"] == replay["trace_digest"]
    assert len(report["decisions"]) == 4
    assert sum(decision["invoke"] for decision in report["decisions"]) == 3
    assert len(report["cycles"]) == 3
    assert {cycle["state"] for cycle in report["cycles"]} == {"committed"}
    assert len(report["deliveries"]) == 1
    assert report["deliveries"][0]["state"] == "failed"
    assert len(report["outcomes"]) == 3
    assert all(outcome["unresolved"] for outcome in report["outcomes"])
    assert any(
        cycle["intervention"] is not None
        and (cycle["intervention"]["cited_memory_ids"] or cycle["intervention"]["cited_event_ids"])
        for cycle in report["cycles"]
    )

    encoded = completed.stdout.lower()
    for forbidden in (
        "verified event 1",
        "run the verified test suite before delivery.",
        "engine-request-4",
        '"prompt"',
        '"responses"',
        '"receipt_digest"',
        '"target_request_id_digest"',
        '"path"',
    ):
        assert forbidden not in encoded


def test_inspect_human_is_concise_and_covers_the_runtime_state(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    root = tmp_path / "artifact"
    replay = _replay(run_cli, root)

    completed = run_cli(
        "inspect",
        str(replay["run_id"]),
        "--artifact",
        str(root / "manifest.json"),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert f"Run {replay['run_id']}" in completed.stdout
    assert "decisions: 4 (3 invoked)" in completed.stdout
    assert "cycles: 3 (3 committed, 0 failed)" in completed.stdout
    assert "budget consumed:" in completed.stdout
    assert "deliveries: 1 (0 delivered)" in completed.stdout
    assert "outcomes: 3 (3 unresolved)" in completed.stdout
    assert "decision 1:" in completed.stdout
    assert "event_ids=" in completed.stdout


def test_inspect_maps_run_mismatch_and_corruption_without_echoing_values(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    root = tmp_path / "fixture-secret-artifact"
    replay = _replay(run_cli, root)

    mismatch = run_cli(
        "inspect",
        str(uuid4()),
        "--artifact",
        str(root),
        "--json",
    )
    assert mismatch.returncode == 2
    assert mismatch.stdout == ""
    assert mismatch.stderr == "error: inspect run does not match artifact\n"
    assert "fixture-secret" not in mismatch.stderr

    decisions = root / "decisions.json"
    decisions.write_bytes(decisions.read_bytes() + b" ")
    corrupted = run_cli(
        "inspect",
        str(replay["run_id"]),
        "--artifact",
        str(root),
        "--json",
    )
    assert corrupted.returncode == 5
    assert corrupted.stdout == ""
    assert corrupted.stderr == "error: artifact validation failed\n"
    assert "fixture-secret" not in corrupted.stderr


def test_inspect_rejects_invalid_uuid_and_missing_artifact_with_stable_diagnostics(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    invalid_uuid = run_cli(
        "inspect",
        "fixture-secret-invalid-uuid",
        "--artifact",
        str(tmp_path / "missing"),
        "--json",
    )
    missing = run_cli(
        "inspect",
        "00000000-0000-4000-8000-00000000d001",
        "--artifact",
        str(tmp_path / "fixture-secret-missing"),
        "--json",
    )

    assert invalid_uuid.returncode == missing.returncode == 2
    assert invalid_uuid.stdout == missing.stdout == ""
    assert invalid_uuid.stderr == missing.stderr == "error: artifact inspection input is invalid\n"
    assert "fixture-secret" not in invalid_uuid.stderr + missing.stderr

    non_v4 = run_cli(
        "inspect",
        "00000000-0000-1000-8000-00000000d001",
        "--artifact",
        str(tmp_path / "missing"),
        "--json",
    )
    assert non_v4.returncode == 2
    assert non_v4.stdout == ""
    assert non_v4.stderr == "error: artifact inspection input is invalid\n"


async def test_inspect_never_emits_explicit_raw_synthetic_content(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    secret = "fixture-secret-explicit-synthetic-payload"
    result = await _run_frozen_replay()
    root = tmp_path / "synthetic"
    manifest = export_replay_artifact(
        result,
        root,
        classification=ArtifactClassification.SYNTHETIC_RAW,
        revision=RevisionEvidence(
            source=RevisionSource.GIT,
            package_version="0.1.0",
            commit="7" * 40,
            dirty_worktree=False,
            distribution_digest=None,
        ),
        synthetic_content=SyntheticArtifactContent(
            prompt={"instruction": secret},
            responses=({"answer": secret},),
        ),
    )

    completed = run_cli(
        "inspect",
        str(manifest.run_id),
        "--artifact",
        str(root),
        "--json",
    )

    assert completed.returncode == 0, completed.stderr
    assert secret not in completed.stdout
    assert secret not in completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["manifest"]["classification"] == "synthetic_raw"
    assert "synthetic" not in payload


async def test_inspect_service_rejects_unsafe_inputs_and_revalidates_rendered_reports(
    tmp_path: Path,
) -> None:
    result = await _run_frozen_replay()
    root = tmp_path / "artifact"
    manifest = export_replay_artifact(result, root)
    report = run_inspect(manifest.run_id, artifact_path=root)

    with pytest.raises(InspectInputError):
        run_inspect(manifest.run_id, artifact_path=b"fixture-secret")  # type: ignore[arg-type]
    with pytest.raises(InspectInputError):
        run_inspect(manifest.run_id, artifact_path=tmp_path / "missing")
    invalid_file = tmp_path / "fixture-secret.json"
    invalid_file.write_text("{}")
    with pytest.raises(InspectInputError):
        run_inspect(manifest.run_id, artifact_path=invalid_file)
    with pytest.raises(InspectInputError):
        run_inspect(UUID(int=0), artifact_path=root)
    with pytest.raises(InspectRunMismatchError):
        run_inspect(uuid4(), artifact_path=root)
    assert inspect_module._intervention_report(None) is None

    inconsistent = report.model_copy(update={"run_id": uuid4()})
    with pytest.raises(ValueError, match="inspection summary"):
        render_inspect_json(inconsistent)

    first_cycle = report.cycles[0]
    assert first_cycle.intervention is not None
    omitted_intervention_id = first_cycle.intervention.intervention_id
    outcomes = tuple(
        outcome for outcome in report.outcomes if outcome.intervention_id != omitted_intervention_id
    )
    counters = report.manifest.counters.model_copy(update={"outcomes": len(outcomes)})
    failed_report = InspectReport.model_validate(
        report.model_copy(
            update={
                "manifest": report.manifest.model_copy(update={"counters": counters}),
                "cycles": (
                    first_cycle.model_copy(
                        update={
                            "state": CycleState.FAILED,
                            "intervention": None,
                            "failure_reason": ReasonCode.MODEL_ERROR,
                        }
                    ),
                    *report.cycles[1:],
                ),
                "outcomes": outcomes,
            }
        )
    )
    assert "evidence=none" in render_inspect_human(failed_report)
