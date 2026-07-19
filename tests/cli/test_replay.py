from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from tests.cli.conftest import FIXTURES, RunCli

import saliencegate.commands.replay as replay_module
from saliencegate.adapters import JSONLReplayAdapter, JsonlReplayEvent, encode_jsonl_trace
from saliencegate.artifacts import validate_artifact
from saliencegate.commands.replay import ReplayCommandError
from saliencegate.domain import TrustLabel

TRACE = FIXTURES / "runs" / "basic.jsonl"


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_replay_json_runs_without_a_model_service_and_exports_a_valid_artifact(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    output = tmp_path / "artifact"

    completed = run_cli("replay", str(TRACE), "--output", str(output), "--json")

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    report = json.loads(completed.stdout)
    assert set(report) == {
        "classification",
        "confirmatory",
        "counters",
        "manifest_digest",
        "overall_content_digest",
        "result_digest",
        "run_id",
        "schema_version",
        "status",
        "trace_digest",
    }
    assert report["schema_version"] == "cli-replay-report/v1"
    assert report["status"] == "ok"
    assert report["classification"] == "synthetic_digest_only"
    assert report["counters"] == {
        "cycles": 3,
        "decisions": 4,
        "delivered": 0,
        "deliveries": 1,
        "events": 4,
        "invoked": 3,
        "model_calls": 3,
        "outcomes": 3,
        "schema_version": "artifact-counters/v1",
    }
    validation = validate_artifact(
        output / "manifest.json",
        expected_manifest_digest=report["manifest_digest"],
    )
    assert validation.valid


def test_replay_is_byte_deterministic_and_replace_is_explicit(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert run_cli("replay", str(TRACE), "--output", str(first), "--json").returncode == 0
    assert run_cli("replay", str(TRACE), "--output", str(second), "--json").returncode == 0
    assert _tree(first) == _tree(second)

    refused = run_cli("replay", str(TRACE), "--output", str(first), "--json")
    assert refused.returncode == 2
    assert refused.stdout == ""
    assert refused.stderr == "error: replay input or output is invalid\n"
    assert _tree(first) == _tree(second)

    replaced = run_cli(
        "replay",
        str(TRACE),
        "--output",
        str(first),
        "--replace",
        "--json",
    )
    assert replaced.returncode == 0
    assert _tree(first) == _tree(second)


def test_replay_human_mode_and_explicit_response_fixture(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    output = tmp_path / "artifact"
    responses = FIXTURES / "models" / "basic_responses.jsonl"

    completed = run_cli(
        "replay",
        str(TRACE),
        "--responses",
        str(responses),
        "--output",
        str(output),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert "Replay complete" in completed.stdout
    assert "events: 4" in completed.stdout
    assert "manifest digest:" in completed.stdout


def test_replay_usage_and_corrupt_inputs_have_stable_exit_and_value_free_diagnostics(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    missing_output = run_cli("replay", str(TRACE), "--json")
    assert missing_output.returncode == 2
    assert missing_output.stdout == ""
    assert missing_output.stderr == "error: invalid command line\n"

    secret_path = tmp_path / "fixture-secret-trace.jsonl"
    secret_path.write_text("not-json")
    invalid = run_cli(
        "replay",
        str(secret_path),
        "--output",
        str(tmp_path / "invalid"),
        "--json",
    )
    assert invalid.returncode == 2
    assert invalid.stdout == ""
    assert invalid.stderr == "error: replay input or output is invalid\n"
    assert "fixture-secret" not in invalid.stderr

    missing_responses = run_cli(
        "replay",
        str(TRACE),
        "--responses",
        str(tmp_path / "fixture-secret-missing.jsonl"),
        "--output",
        str(tmp_path / "missing-response"),
        "--json",
    )
    assert missing_responses.returncode == 2
    assert missing_responses.stderr == "error: replay input or output is invalid\n"
    assert "fixture-secret" not in missing_responses.stderr


def test_replay_replace_preserves_an_unrelated_directory(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    output = tmp_path / "unrelated"
    output.mkdir()
    sentinel = output / "fixture-secret-user-data.txt"
    sentinel.write_text("must survive")

    completed = run_cli(
        "replay",
        str(TRACE),
        "--output",
        str(output),
        "--replace",
        "--json",
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "error: replay input or output is invalid\n"
    assert sentinel.read_text() == "must survive"
    assert tuple(output.iterdir()) == (sentinel,)


def test_replay_maps_an_invalid_local_key_configuration_without_disclosing_it(
    tmp_path: Path,
) -> None:
    source = JSONLReplayAdapter.from_path(TRACE).events[0]
    event = JsonlReplayEvent.create(
        ordinal=1,
        expected_event_id=source.expected_event_id,
        draft=source.draft.model_copy(update={"trust_label": TrustLabel.UNTRUSTED_TASK_INPUT}),
    )
    trace = tmp_path / "user-trace.jsonl"
    trace.write_bytes(encode_jsonl_trace((event,)))
    responses = tmp_path / "empty-responses.jsonl"
    responses.write_bytes(b"")
    output = tmp_path / "artifact"
    environment = os.environ.copy()
    environment.update(
        HOME=str(tmp_path / "home"),
        PYTHONUTF8="1",
        XDG_CONFIG_HOME="relative-fixture-secret-configuration",
    )

    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "saliencegate",
            "replay",
            str(trace),
            "--responses",
            str(responses),
            "--output",
            str(output),
            "--json",
        ),
        cwd=Path(__file__).parents[2],
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 3
    assert completed.stdout == ""
    assert completed.stderr == "error: replay configuration is invalid\n"
    assert "fixture-secret" not in completed.stderr
    assert not output.exists()


def test_replay_maps_an_unavailable_output_parent_to_invalid_input(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    blocker = tmp_path / "fixture-secret-blocker"
    blocker.write_text("must survive")

    completed = run_cli(
        "replay",
        str(TRACE),
        "--output",
        str(blocker / "artifact"),
        "--json",
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "error: replay input or output is invalid\n"
    assert "fixture-secret" not in completed.stderr
    assert blocker.read_text() == "must survive"

    empty_trace = run_cli(
        "replay",
        "",
        "--output",
        str(tmp_path / "empty-trace"),
        "--json",
    )
    assert empty_trace.returncode == 2
    assert empty_trace.stdout == ""
    assert empty_trace.stderr == "error: replay input or output is invalid\n"


def test_default_response_discovery_sanitizes_filesystem_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = tmp_path / "trace.jsonl"

    def deny(_path: Path) -> os.stat_result:
        raise PermissionError("fixture-secret stat failure")

    monkeypatch.setattr(Path, "lstat", deny)

    with pytest.raises(ReplayCommandError) as error:
        replay_module._response_fixture(trace, None)
    assert "fixture-secret" not in str(error.value)


async def test_replay_rejects_more_responses_than_events_before_running_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OversizedFixture:
        total_responses = 5

    monkeypatch.setattr(
        replay_module.ReplayModel,
        "from_path",
        lambda _path: cast(replay_module.ReplayModel, OversizedFixture()),
    )

    with pytest.raises(ReplayCommandError):
        await replay_module.run_replay(
            TRACE,
            responses_path=FIXTURES / "models" / "basic_responses.jsonl",
            output_path=tmp_path / "artifact",
        )
