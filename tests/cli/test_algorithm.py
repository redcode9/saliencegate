from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Never, cast

import pytest
from tests.cli.conftest import RunCli

import saliencegate.cli as cli_module
import saliencegate.commands.algorithm as algorithm_module
from saliencegate import __version__
from saliencegate.artifacts import (
    ArtifactClassification,
    ArtifactValidationCode,
    ArtifactValidationError,
    RevisionEvidence,
    RevisionSource,
    load_validated_algorithm_artifact,
)
from saliencegate.artifacts.algorithm_manifest import AlgorithmTokenizerStatus
from saliencegate.commands.algorithm import (
    AlgorithmReplayCommandError,
    AlgorithmReplayCommandReport,
    render_algorithm_replay_human,
    render_algorithm_replay_json,
    run_algorithm_replay,
)
from saliencegate.domain import canonical_json
from saliencegate.experiments import Stage2ConditionId

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
TRAJECTORY = FIXTURES / "runs" / "paper_two_phase_basic.jsonl"
FIXED_RESPONSES = FIXTURES / "models" / "paper_two_phase_fixed_step_responses.jsonl"
RETRIEVAL_RESPONSES = FIXTURES / "models" / "paper_two_phase_retrieval_responses.jsonl"
ALWAYS_RESPONSES = FIXTURES / "models" / "paper_two_phase_always_inject_responses.jsonl"
RUN_ID = "00000000-0000-4000-8000-000000009000"
SHA256 = re.compile(r"^[0-9a-f]{64}$")

CASES = (
    (
        Stage2ConditionId.NO_MEMORY,
        None,
        0,
        0,
        0,
        0,
        0,
        0,
        "2dd8ff174cec4d2c868126eda780136e4f43162cb969a42e2b190fb24bc9bcc5",
    ),
    (
        Stage2ConditionId.FIXED_STEP,
        FIXED_RESPONSES,
        6,
        19_722,
        393,
        20_115,
        1,
        0,
        "4097e78486eb70b11ca9b0626d245521f3300012a4c709cefc2c1107d3e217bb",
    ),
    (
        Stage2ConditionId.RETRIEVAL_ALWAYS,
        RETRIEVAL_RESPONSES,
        3,
        10_509,
        286,
        10_795,
        2,
        0,
        "2267f43967067f159398e626c926007dacfd601a6c141a5562cae94a18e0f63a",
    ),
    (
        Stage2ConditionId.ALWAYS_INJECT,
        ALWAYS_RESPONSES,
        6,
        19_982,
        409,
        20_391,
        2,
        1,
        "6c24ffd44208e97af2112750630b761110b77d9471813963c30956fa5c195456",
    ),
)


def _tree_bytes(path: Path) -> dict[str, bytes]:
    return {item.name: item.read_bytes() for item in sorted(path.iterdir())}


def test_algorithm_cli_help_contract_is_exact(run_cli: RunCli) -> None:
    root = run_cli("--help")
    command = run_cli("algorithm", "--help")
    replay = run_cli("algorithm", "replay", "--help")

    assert root.returncode == command.returncode == replay.returncode == 0
    assert root.stderr == command.stderr == replay.stderr == ""
    assert root.stdout == (
        "usage: saliencegate [-h] [--version]\n"
        "                    {demo,doctor,setup,connect,disconnect,status,sessions,report,"
        "feedback,delete,replay,shadow,algorithm,pilot,benchmark,inspect,validate}\n"
        "                    ...\n"
        "\n"
        "positional arguments:\n"
        "  {demo,doctor,setup,connect,disconnect,status,sessions,report,feedback,delete,"
        "replay,shadow,algorithm,pilot,benchmark,inspect,validate}\n"
        "    setup               Choose providers and connect them per project or\n"
        "                        globally\n"
        "    connect             Install passive capture\n"
        "    disconnect          Remove passive capture\n"
        "    status              Show passive capture status\n"
        "    sessions            List captured project sessions\n"
        "    report              Build a passive capture report\n"
        "    feedback            Record local capture feedback\n"
        "    delete              Delete local passive capture records\n"
        "\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  --version             show program's version number and exit\n"
    )
    assert command.stdout == (
        "usage: saliencegate algorithm [-h] {replay} ...\n"
        "\n"
        "positional arguments:\n"
        "  {replay}\n"
        "\n"
        "options:\n"
        "  -h, --help  show this help message and exit\n"
    )
    assert replay.stdout == (
        "usage: saliencegate algorithm replay [-h] [--responses RESPONSES] --condition\n"
        "                                     "
        "{no_memory,fixed_step,retrieval_always,always_inject}\n"
        "                                     --output OUTPUT [--replace] [--json]\n"
        "                                     trace\n"
        "\n"
        "positional arguments:\n"
        "  trace\n"
        "\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  --responses RESPONSES\n"
        "  --condition {no_memory,fixed_step,retrieval_always,always_inject}\n"
        "  --output OUTPUT\n"
        "  --replace\n"
        "  --json\n"
    )


def test_algorithm_cli_dispatches_json_and_human_reports(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    json_output = tmp_path / "json"
    human_output = tmp_path / "human"
    arguments = (
        "algorithm",
        "replay",
        str(TRAJECTORY),
        "--responses",
        str(FIXED_RESPONSES),
        "--condition",
        "fixed_step",
        "--output",
    )

    structured = run_cli(*arguments, str(json_output), "--json")
    human = run_cli(*arguments, str(human_output))

    assert structured.returncode == human.returncode == 0
    assert structured.stderr == human.stderr == ""
    payload = json.loads(structured.stdout)
    assert payload["schema_version"] == "cli-algorithm-replay-report/v1"
    assert payload["condition"] == "fixed_step"
    assert payload["run_id"] == RUN_ID
    assert payload["result_digest"] == CASES[1][-1]
    assert payload["calls"] == 6
    assert payload["canonical_token_equivalents"] == 20_115
    assert payload["classification"] == "synthetic_digest_only"
    assert payload["confirmatory"] is False
    assert human.stdout == (
        "Algorithm replay complete\n"
        "condition: fixed_step\n"
        f"run: {RUN_ID}\n"
        "calls: 6\n"
        "canonical tokens: 19722 input, 393 output, 20115 total\n"
        "interventions: 1\n"
        "grounding rejections: 0\n"
        "classification: synthetic_digest_only\n"
        "confirmatory: no\n"
        f"run digest: {payload['run_digest']}\n"
        f"result digest: {payload['result_digest']}\n"
        f"manifest digest: {payload['manifest_digest']}\n"
        f"content digest: {payload['overall_content_digest']}\n"
    )
    assert _tree_bytes(json_output) == _tree_bytes(human_output)


def test_algorithm_cli_invalid_inputs_have_stable_value_free_exits(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    missing_output = run_cli(
        "algorithm",
        "replay",
        str(TRAJECTORY),
        "--condition",
        "fixed_step",
    )
    unknown_condition = run_cli(
        "algorithm",
        "replay",
        str(TRAJECTORY),
        "--condition",
        "fixture-secret-condition",
        "--output",
        str(tmp_path / "unknown"),
    )
    missing_responses = run_cli(
        "algorithm",
        "replay",
        str(TRAJECTORY),
        "--condition",
        "fixed_step",
        "--output",
        str(tmp_path / "missing"),
        "--json",
    )
    extra_responses = run_cli(
        "algorithm",
        "replay",
        str(TRAJECTORY),
        "--responses",
        str(FIXED_RESPONSES),
        "--condition",
        "no_memory",
        "--output",
        str(tmp_path / "extra"),
        "--json",
    )

    for completed in (missing_output, unknown_condition):
        assert completed.returncode == 2
        assert completed.stdout == ""
        assert completed.stderr == "error: invalid command line\n"
        assert "fixture-secret" not in completed.stderr
    for completed in (missing_responses, extra_responses):
        assert completed.returncode == 2
        assert completed.stdout == ""
        assert completed.stderr == "error: algorithm replay input or output is invalid\n"


@pytest.mark.parametrize(
    (
        "condition",
        "responses",
        "calls",
        "canonical_input",
        "canonical_output",
        "canonical_total",
        "interventions",
        "grounding_rejections",
        "result_digest",
    ),
    CASES,
    ids=lambda value: value.value if isinstance(value, Stage2ConditionId) else None,
)
async def test_algorithm_replay_runs_every_closed_condition_and_validates_the_artifact(
    tmp_path: Path,
    condition: Stage2ConditionId,
    responses: Path | None,
    calls: int,
    canonical_input: int,
    canonical_output: int,
    canonical_total: int,
    interventions: int,
    grounding_rejections: int,
    result_digest: str,
) -> None:
    output = tmp_path / condition.value

    report = await run_algorithm_replay(
        TRAJECTORY,
        condition=condition,
        responses_path=responses,
        output_path=output,
    )

    assert type(report) is AlgorithmReplayCommandReport
    assert report.condition is condition
    assert report.run_id == RUN_ID
    assert SHA256.fullmatch(report.run_digest)
    assert report.result_digest == result_digest
    assert SHA256.fullmatch(report.manifest_digest)
    assert SHA256.fullmatch(report.overall_content_digest)
    assert report.calls == calls
    assert report.canonical_input_tokens == canonical_input
    assert report.canonical_output_tokens == canonical_output
    assert report.canonical_token_equivalents == canonical_total
    assert report.interventions == interventions
    assert report.grounding_rejections == grounding_rejections
    assert report.classification is ArtifactClassification.SYNTHETIC_DIGEST_ONLY
    assert report.confirmatory is False

    loaded = load_validated_algorithm_artifact(
        output / "manifest.json",
        expected_manifest_digest=report.manifest_digest,
    )
    execution = loaded.run.execution
    assert loaded.run.run_component_digest == report.run_digest
    assert loaded.manifest.result_digest == report.result_digest
    assert loaded.metrics.metrics.model_call_count == report.calls
    assert execution.execution_mode.value == "frozen_replay"
    assert execution.endpoint_classification.value == "offline_replay"
    assert execution.runtime_id == "saliencegate-two-phase-replay"
    assert execution.runtime_version == __version__
    assert execution.checkpoint.model_id == "gpt-oss:20b"
    assert execution.checkpoint.model_tag == "gpt-oss:20b-fixture/v1"
    assert execution.checkpoint.quantization == "not-applicable-replay"
    assert execution.sampling.mode.value == "frozen_replay"
    assert execution.warmup_policy.value == "not_applicable"
    hardware = execution.hardware
    assert (
        hardware.model,
        hardware.architecture,
        hardware.logical_core_count,
        hardware.memory_capacity_bytes,
        hardware.operating_system,
        hardware.operating_system_version,
    ) == (
        "not-applicable-replay",
        "not-applicable-replay",
        1,
        1,
        "not-applicable-replay",
        "not-applicable-replay",
    )
    if calls:
        assert loaded.run.execution.tokenizer.status is AlgorithmTokenizerStatus.ATTESTED
        assert loaded.run.execution.response_fixture is not None
        assert loaded.run.execution.response_fixture.response_count == calls
        assert loaded.run.execution.response_fixture.consumed_count == calls
    else:
        assert loaded.run.execution.tokenizer.status is AlgorithmTokenizerStatus.UNAVAILABLE
        assert loaded.run.execution.response_fixture is None


async def test_algorithm_replay_json_and_human_reports_are_exact_and_value_minimized(
    tmp_path: Path,
) -> None:
    report = await run_algorithm_replay(
        TRAJECTORY,
        condition=Stage2ConditionId.FIXED_STEP,
        responses_path=FIXED_RESPONSES,
        output_path=tmp_path / "report",
    )

    rendered_json = render_algorithm_replay_json(report)
    assert rendered_json == canonical_json(report).decode("utf-8") + "\n"
    assert set(json.loads(rendered_json)) == {
        "calls",
        "canonical_input_tokens",
        "canonical_output_tokens",
        "canonical_token_equivalents",
        "classification",
        "condition",
        "confirmatory",
        "grounding_rejections",
        "interventions",
        "manifest_digest",
        "overall_content_digest",
        "result_digest",
        "run_digest",
        "run_id",
        "schema_version",
        "status",
    }
    assert json.loads(rendered_json)["schema_version"] == "cli-algorithm-replay-report/v1"

    rendered_human = render_algorithm_replay_human(report)
    assert rendered_human == (
        "Algorithm replay complete\n"
        "condition: fixed_step\n"
        f"run: {RUN_ID}\n"
        "calls: 6\n"
        "canonical tokens: 19722 input, 393 output, 20115 total\n"
        "interventions: 1\n"
        "grounding rejections: 0\n"
        "classification: synthetic_digest_only\n"
        "confirmatory: no\n"
        f"run digest: {report.run_digest}\n"
        f"result digest: {report.result_digest}\n"
        f"manifest digest: {report.manifest_digest}\n"
        f"content digest: {report.overall_content_digest}\n"
    )


async def test_algorithm_replay_is_byte_deterministic_and_replace_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    dirty_states = iter((False, True, False, True))

    def changing_revision() -> RevisionEvidence:
        return RevisionEvidence(
            source=RevisionSource.GIT,
            package_version=__version__,
            commit="a" * 40,
            dirty_worktree=next(dirty_states),
        )

    monkeypatch.setattr(algorithm_module, "discover_revision", changing_revision)

    first_report = await run_algorithm_replay(
        TRAJECTORY,
        condition="fixed_step",
        responses_path=FIXED_RESPONSES,
        output_path=first,
    )
    second_report = await run_algorithm_replay(
        TRAJECTORY,
        condition="fixed_step",
        responses_path=FIXED_RESPONSES,
        output_path=second,
    )

    assert first_report == second_report
    assert _tree_bytes(first) == _tree_bytes(second)
    with pytest.raises(AlgorithmReplayCommandError):
        await run_algorithm_replay(
            TRAJECTORY,
            condition="fixed_step",
            responses_path=FIXED_RESPONSES,
            output_path=first,
        )
    replaced = await run_algorithm_replay(
        TRAJECTORY,
        condition="fixed_step",
        responses_path=FIXED_RESPONSES,
        output_path=first,
        replace=True,
    )
    assert replaced == first_report
    assert _tree_bytes(first) == _tree_bytes(second)
    loaded = load_validated_algorithm_artifact(first / "manifest.json")
    assert loaded.manifest.revision.dirty_worktree is True


async def test_algorithm_replay_preserves_unrelated_destinations(
    tmp_path: Path,
) -> None:
    output = tmp_path / "unrelated"
    output.mkdir()
    sentinel = output / "fixture-secret-user-data.txt"
    sentinel.write_text("preserve me", encoding="utf-8")

    with pytest.raises(AlgorithmReplayCommandError) as raised:
        await run_algorithm_replay(
            TRAJECTORY,
            condition=Stage2ConditionId.FIXED_STEP,
            responses_path=FIXED_RESPONSES,
            output_path=output,
            replace=True,
        )

    assert str(raised.value) == "algorithm replay input or output is invalid"
    assert sentinel.read_text(encoding="utf-8") == "preserve me"


async def test_algorithm_replay_replace_preserves_cross_condition_and_corruption(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifact"
    await run_algorithm_replay(
        TRAJECTORY,
        condition=Stage2ConditionId.FIXED_STEP,
        responses_path=FIXED_RESPONSES,
        output_path=output,
    )
    original = _tree_bytes(output)

    with pytest.raises(AlgorithmReplayCommandError):
        await run_algorithm_replay(
            TRAJECTORY,
            condition=Stage2ConditionId.RETRIEVAL_ALWAYS,
            responses_path=RETRIEVAL_RESPONSES,
            output_path=output,
            replace=True,
        )
    assert _tree_bytes(output) == original

    metrics = output / "metrics.json"
    metrics.write_bytes(metrics.read_bytes() + b" ")
    corrupted = _tree_bytes(output)
    with pytest.raises(AlgorithmReplayCommandError):
        await run_algorithm_replay(
            TRAJECTORY,
            condition=Stage2ConditionId.FIXED_STEP,
            responses_path=FIXED_RESPONSES,
            output_path=output,
            replace=True,
        )
    assert _tree_bytes(output) == corrupted


async def test_algorithm_replay_rejects_unavailable_paths_without_disclosure(
    tmp_path: Path,
) -> None:
    class FailingPath(os.PathLike[str]):
        def __fspath__(self) -> str:
            raise OSError("fixture-secret path failure")

    blocker = tmp_path / "fixture-secret-blocker"
    blocker.write_text("preserve me", encoding="utf-8")

    for trajectory, output in (
        (cast(os.PathLike[str], FailingPath()), tmp_path / "failing-path"),
        (TRAJECTORY, blocker / "artifact"),
    ):
        with pytest.raises(AlgorithmReplayCommandError) as raised:
            await run_algorithm_replay(
                trajectory,
                condition=Stage2ConditionId.NO_MEMORY,
                output_path=output,
            )
        assert "fixture-secret" not in str(raised.value)
        assert raised.value.__cause__ is None
        assert raised.value.__context__ is None
    assert blocker.read_text(encoding="utf-8") == "preserve me"


@pytest.mark.parametrize(
    ("condition", "trajectory", "responses", "output", "replace"),
    (
        (Stage2ConditionId.FIXED_STEP, TRAJECTORY, None, "missing-responses", False),
        (Stage2ConditionId.NO_MEMORY, TRAJECTORY, FIXED_RESPONSES, "extra-responses", False),
        ("fixture-secret-condition", TRAJECTORY, None, "unknown-condition", False),
        (Stage2ConditionId.FIXED_STEP, TRAJECTORY, RETRIEVAL_RESPONSES, "wrong-fixture", False),
        (Stage2ConditionId.NO_MEMORY, "", None, "empty-trace", False),
        (Stage2ConditionId.NO_MEMORY, TRAJECTORY, None, "", False),
        (Stage2ConditionId.NO_MEMORY, TRAJECTORY, None, "invalid-replace", cast(bool, 1)),
    ),
)
async def test_algorithm_replay_invalid_inputs_are_value_free(
    tmp_path: Path,
    condition: Stage2ConditionId | str,
    trajectory: Path | str,
    responses: Path | None,
    output: str,
    replace: bool,
) -> None:
    with pytest.raises(AlgorithmReplayCommandError) as raised:
        await run_algorithm_replay(
            trajectory,
            condition=condition,
            responses_path=responses,
            output_path=tmp_path / output if output else output,
            replace=replace,
        )

    assert str(raised.value) == "algorithm replay input or output is invalid"
    assert "fixture-secret" not in str(raised.value)
    assert "fixture-secret" not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_algorithm_cli_maps_validation_failure_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_loaded_validation(*_args: object, **_kwargs: object) -> Never:
        raise ArtifactValidationError(ArtifactValidationCode.INVALID_MANIFEST)

    monkeypatch.setattr(
        algorithm_module,
        "load_validated_algorithm_artifact",
        fail_loaded_validation,
    )

    with pytest.raises(ArtifactValidationError) as raised:
        asyncio.run(
            run_algorithm_replay(
                TRAJECTORY,
                condition=Stage2ConditionId.NO_MEMORY,
                output_path=tmp_path / "direct-artifact",
            )
        )
    assert raised.value.code is ArtifactValidationCode.INVALID_MANIFEST
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None

    code = cli_module.main(
        (
            "algorithm",
            "replay",
            str(TRAJECTORY),
            "--condition",
            "no_memory",
            "--output",
            str(tmp_path / "artifact"),
            "--json",
        )
    )

    captured = capsys.readouterr()
    assert code == cli_module.ExitCode.CORRUPTED_ARTIFACT
    assert captured.out == ""
    assert captured.err == "error: artifact validation failed\n"


def test_algorithm_module_import_does_not_load_the_live_model_runtime() -> None:
    probe = (
        "import socket, sys\n"
        "class NetworkForbiddenSocket(socket.socket):\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        raise AssertionError('algorithm import attempted network access')\n"
        "def network_forbidden(*args, **kwargs):\n"
        "    raise AssertionError('algorithm import attempted network access')\n"
        "socket.socket = NetworkForbiddenSocket\n"
        "socket.create_connection = network_forbidden\n"
        "socket.getaddrinfo = network_forbidden\n"
        "import saliencegate.commands.algorithm\n"
        "assert 'saliencegate.models.openai_compatible' not in sys.modules\n"
    )
    completed = subprocess.run(
        (sys.executable, "-I", "-c", probe),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == completed.stderr == ""


async def test_algorithm_replay_does_not_import_or_call_the_live_model_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def network_forbidden(*args: object, **kwargs: object) -> Never:
        del args, kwargs
        raise AssertionError("algorithm replay attempted network access")

    live_module = "saliencegate.models.openai_compatible"
    sys.modules.pop(live_module, None)
    monkeypatch.setattr(socket, "socket", network_forbidden)

    report = await run_algorithm_replay(
        TRAJECTORY,
        condition=Stage2ConditionId.FIXED_STEP,
        responses_path=FIXED_RESPONSES,
        output_path=tmp_path / "offline",
    )

    assert report.calls == 6
    assert live_module not in sys.modules


@pytest.mark.parametrize("renderer", (render_algorithm_replay_json, render_algorithm_replay_human))
def test_algorithm_replay_renderers_reject_invalid_reports_without_values(
    renderer: Callable[[AlgorithmReplayCommandReport], str],
) -> None:
    invalid = cast(AlgorithmReplayCommandReport, object())
    with pytest.raises(AlgorithmReplayCommandError) as raised:
        renderer(invalid)
    assert str(raised.value) == "algorithm replay input or output is invalid"
