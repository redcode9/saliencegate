from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from tests.cli.conftest import FIXTURES, RunCli

from saliencegate.artifacts import (
    AlgorithmArtifactManifest,
    AlgorithmArtifactValidationReport,
    AlgorithmCheckpointAttestation,
    AlgorithmEndpointClassification,
    AlgorithmExecutionAttestation,
    AlgorithmHardwareAttestation,
    AlgorithmResponseFixtureAttestation,
    AlgorithmSamplingAttestation,
    AlgorithmSamplingMode,
    AlgorithmSourceResultAssurance,
    AlgorithmTokenizerAttestation,
    AlgorithmTokenizerStatus,
    AlgorithmWarmupPolicy,
    ArtifactClassification,
    ArtifactValidationCode,
    ArtifactValidationError,
    ArtifactValidationReport,
    RevisionEvidence,
    RevisionSource,
    export_algorithm_artifact,
)
from saliencegate.benchmarks.state_decay.runner import BenchmarkValidationReport
from saliencegate.commands.validate import (
    render_validate_human,
    render_validate_json,
    run_validate,
)
from saliencegate.domain import canonical_json
from saliencegate.experiments import (
    Stage2ConditionId,
    Stage2ExperimentRunResult,
    replay_stage2_fixture_twice,
)

TRACE = FIXTURES / "runs" / "basic.jsonl"
TRAJECTORY = FIXTURES / "runs" / "paper_two_phase_basic.jsonl"
RESPONSES = FIXTURES / "models" / "paper_two_phase_fixed_step_responses.jsonl"
TRAJECTORY_DIGEST = "751489f55ac9d5ea56408ca6f5036b55e895be0fa130c36f19e624ee094d1266"
RESPONSE_DIGEST = "aed71f320f03f4783fc25089f8c0a638323c2381e498b0349df2def6de5179ae"


@pytest.fixture(scope="module")
def fixed_stage2_result() -> Stage2ExperimentRunResult:
    return asyncio.run(
        replay_stage2_fixture_twice(
            TRAJECTORY,
            condition=Stage2ConditionId.FIXED_STEP,
            responses_path=RESPONSES,
            expected_trajectory_fixture_digest=TRAJECTORY_DIGEST,
            expected_response_fixture_digest=RESPONSE_DIGEST,
        )
    )


def _execution() -> AlgorithmExecutionAttestation:
    return AlgorithmExecutionAttestation.create(
        endpoint_classification=AlgorithmEndpointClassification.OFFLINE_REPLAY,
        runtime_id="saliencegate-two-phase-replay",
        runtime_version="1.0.0",
        checkpoint=AlgorithmCheckpointAttestation(
            model_id="gpt-oss:20b",
            model_tag="gpt-oss:20b-fixture/v1",
            checkpoint_digest=None,
            quantization="not-applicable-replay",
        ),
        sampling=AlgorithmSamplingAttestation(
            mode=AlgorithmSamplingMode.FROZEN_REPLAY,
            temperature=None,
            seed=None,
            reasoning_effort=None,
        ),
        tokenizer=AlgorithmTokenizerAttestation(
            status=AlgorithmTokenizerStatus.ATTESTED,
            tokenizer_id="stage2-reviewed-utf8-counter/v1",
            tokenizer_version="utf8-bytes-ceil-div-4/v1",
            configuration_digest="4" * 64,
            model_id="gpt-oss:20b",
        ),
        hardware=AlgorithmHardwareAttestation(
            model="synthetic-test-host",
            architecture="test-arch",
            logical_core_count=8,
            memory_capacity_bytes=16 * 1024**3,
            operating_system="test-os",
            operating_system_version="1.0",
        ),
        warmup_policy=AlgorithmWarmupPolicy.NOT_APPLICABLE,
        response_fixture=AlgorithmResponseFixtureAttestation(
            replay_id="two-phase-replay/v1",
            fixture_digest=RESPONSE_DIGEST,
            response_count=6,
            consumed_count=6,
        ),
    )


def _revision() -> RevisionEvidence:
    return RevisionEvidence(
        source=RevisionSource.GIT,
        package_version="0.1.0",
        commit="a" * 40,
        dirty_worktree=False,
    )


@pytest.fixture
def algorithm_artifact(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
) -> tuple[Path, AlgorithmArtifactManifest]:
    destination = tmp_path / "algorithm"
    manifest = export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=_execution(),
        revision=_revision(),
    )
    return destination, manifest


def _assert_validation_failure(error: pytest.ExceptionInfo[ArtifactValidationError]) -> None:
    assert error.value.code in {
        ArtifactValidationCode.INVALID_MANIFEST,
        ArtifactValidationCode.UNSAFE_COMPONENT,
    }
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


def test_algorithm_validate_json_human_path_forms_and_digest_anchor_are_exact(
    algorithm_artifact: tuple[Path, AlgorithmArtifactManifest],
    run_cli: RunCli,
) -> None:
    destination, manifest = algorithm_artifact

    unanchored = run_cli("validate", str(destination), "--json")
    anchored = run_cli(
        "validate",
        str(destination / "manifest.json"),
        "--expected-digest",
        manifest.manifest_digest,
        "--json",
    )
    human = run_cli("validate", str(destination / "manifest.json"))

    assert unanchored.returncode == anchored.returncode == human.returncode == 0
    assert unanchored.stderr == anchored.stderr == human.stderr == ""
    assert unanchored.stdout == (
        '{"component_count":9,"confirmatory":false,"expected_digest_matched":null,'
        f'"manifest_digest":"{manifest.manifest_digest}",'
        f'"overall_content_digest":"{manifest.overall_content_digest}",'
        '"schema_version":"algorithm-validation-report/v1","self_consistent":true,'
        '"source_result_assurance":"producer_attested","structurally_valid":true,'
        '"valid":true}\n'
    )
    assert anchored.stdout == unanchored.stdout.replace(
        '"expected_digest_matched":null',
        '"expected_digest_matched":true',
    )
    assert human.stdout == (
        "Algorithm artifact valid\n"
        "source result assurance: producer-attested digest only\n"
        "self-consistent: yes\n"
        "confirmatory: no\n"
        f"manifest digest: {manifest.manifest_digest}\n"
        f"content digest: {manifest.overall_content_digest}\n"
    )


def test_algorithm_validate_recomputed_raw_assurance_is_explicit(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
) -> None:
    destination = tmp_path / "raw"
    manifest = export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=_execution(),
        revision=_revision(),
        classification=ArtifactClassification.SYNTHETIC_RAW,
    )

    report = run_validate(destination)

    assert type(report) is AlgorithmArtifactValidationReport
    assert report.source_result_assurance is AlgorithmSourceResultAssurance.RECOMPUTED_FROM_RAW
    assert json.loads(render_validate_json(report))["source_result_assurance"] == (
        "recomputed_from_raw"
    )
    assert render_validate_human(report) == (
        "Algorithm artifact valid\n"
        "source result assurance: recomputed from included synthetic raw result\n"
        "self-consistent: yes\n"
        "confirmatory: no\n"
        f"manifest digest: {manifest.manifest_digest}\n"
        f"content digest: {manifest.overall_content_digest}\n"
    )


def test_algorithm_validate_errors_are_value_free_and_never_emit_a_report(
    algorithm_artifact: tuple[Path, AlgorithmArtifactManifest],
    run_cli: RunCli,
) -> None:
    destination, _ = algorithm_artifact

    malformed = run_cli(
        "validate",
        str(destination),
        "--expected-digest",
        "fixture-secret",
        "--json",
    )
    mismatch = run_cli(
        "validate",
        str(destination),
        "--expected-digest",
        "0" * 64,
        "--json",
    )
    confirmatory = run_cli(
        "validate",
        str(destination),
        "--require-confirmatory",
        "--json",
    )
    metrics = destination / "metrics.json"
    metrics.write_bytes(metrics.read_bytes() + b" ")
    corrupted = run_cli("validate", str(destination), "--json")

    for completed in (malformed, mismatch, confirmatory, corrupted):
        assert completed.returncode == 5
        assert completed.stdout == ""
        assert completed.stderr == "error: artifact validation failed\n"
        assert "fixture-secret" not in completed.stderr


def test_manifest_preflight_rejects_unknown_noncanonical_and_oversized_inputs(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
) -> None:
    unknown = tmp_path / "unknown"
    manifest = export_algorithm_artifact(
        fixed_stage2_result,
        unknown,
        execution=_execution(),
        revision=_revision(),
    )
    values = manifest.model_dump(mode="json")
    values["artifact_kind"] = "fixture-secret"
    unknown.joinpath("manifest.json").write_bytes(canonical_json(values))

    noncanonical = tmp_path / "noncanonical"
    noncanonical.mkdir()
    noncanonical.joinpath("manifest.json").write_bytes(b'{"artifact_kind": "algorithm_run"}')

    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    duplicate.joinpath("manifest.json").write_bytes(
        b'{"artifact_kind":"algorithm_run","artifact_kind":"algorithm_run"}'
    )

    oversized = tmp_path / "oversized"
    oversized.mkdir()
    oversized.joinpath("manifest.json").write_bytes(b"{" + b" " * (1024 * 1024) + b"}")

    for destination in (unknown, noncanonical, duplicate, oversized):
        with pytest.raises(ArtifactValidationError) as error:
            run_validate(destination)
        _assert_validation_failure(error)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFO support")
def test_manifest_preflight_rejects_links_and_fifo_without_blocking(tmp_path: Path) -> None:
    payload = tmp_path / "payload.json"
    payload.write_bytes(canonical_json({"artifact_kind": "algorithm_run"}))

    symbolic = tmp_path / "symbolic"
    symbolic.mkdir()
    symbolic.joinpath("manifest.json").symlink_to(payload)

    hard = tmp_path / "hard"
    hard.mkdir()
    os.link(payload, hard / "manifest.json")

    fifo = tmp_path / "fifo"
    fifo.mkdir()
    os.mkfifo(fifo / "manifest.json", 0o600)

    for destination in (symbolic, hard, fifo):
        with pytest.raises(ArtifactValidationError) as error:
            run_validate(destination)
        assert error.value.code is ArtifactValidationCode.UNSAFE_COMPONENT
        assert error.value.__cause__ is None


def test_validate_preserves_replay_manifest_without_explicit_artifact_kind(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    destination = tmp_path / "legacy-replay"
    replay = run_cli("replay", str(TRACE), "--output", str(destination), "--json")
    assert replay.returncode == 0, replay.stderr
    manifest_path = destination / "manifest.json"
    payload = json.loads(manifest_path.read_bytes())
    del payload["artifact_kind"]
    manifest_path.write_bytes(canonical_json(payload))

    validated = run_cli("validate", str(destination), "--json")

    assert validated.returncode == 0, validated.stderr
    assert validated.stderr == ""
    report = json.loads(validated.stdout)
    assert report["schema_version"] == "artifact-validation-report/v1"
    assert report["valid"] is True
    assert report["expected_digest_matched"] is None


def test_renderer_bytes_remain_exact() -> None:
    replay = ArtifactValidationReport(
        expected_digest_matched=None,
        confirmatory=False,
        manifest_digest="1" * 64,
        overall_content_digest="2" * 64,
        component_count=6,
    )
    benchmark = BenchmarkValidationReport(
        expected_digest_matched=True,
        manifest_digest="3" * 64,
        overall_content_digest="4" * 64,
        fixture_digest="5" * 64,
        oracle_result_digest="6" * 64,
    )

    assert render_validate_json(replay) == (
        '{"component_count":6,"confirmatory":false,"expected_digest_matched":null,'
        '"grounding_assurance":"producer_attested_digest_only","integrity_valid":true,'
        f'"manifest_digest":"{"1" * 64}","overall_content_digest":"{"2" * 64}",'
        '"schema_version":"artifact-validation-report/v1","structurally_valid":true,'
        '"valid":true}\n'
    )
    assert render_validate_human(replay) == (
        "Artifact valid\n"
        "grounding assurance: producer-attested digest-only grounding\n"
        "confirmatory: no\n"
        f"manifest digest: {'1' * 64}\n"
        f"content digest: {'2' * 64}\n"
    )
    assert render_validate_json(benchmark) == (
        '{"assurance":"deterministic_synthetic_oracle","confirmatory":false,'
        '"expected_digest_matched":true,"external_claims_supported":false,'
        f'"fixture_digest":"{"5" * 64}","integrity_valid":true,'
        f'"manifest_digest":"{"3" * 64}","oracle_result_digest":"{"6" * 64}",'
        f'"overall_content_digest":"{"4" * 64}","scenario_count":32,'
        '"schema_version":"benchmark-validation-report/v1","structurally_valid":true,'
        '"valid":true}\n'
    )
    assert render_validate_human(benchmark) == (
        "Benchmark artifact valid\n"
        "assurance: deterministic synthetic oracle\n"
        "confirmatory: no\n"
        "external claims: unsupported\n"
        "scenarios: 32\n"
        f"manifest digest: {'3' * 64}\n"
        f"content digest: {'4' * 64}\n"
    )
