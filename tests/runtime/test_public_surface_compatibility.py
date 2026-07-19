from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from saliencegate.artifacts import load_validated_artifact
from saliencegate.artifacts.algorithm_manifest import (
    ALGORITHM_ARTIFACT_SCHEMA_VERSION,
    AlgorithmArtifactManifest,
)
from saliencegate.benchmarks.registry import available_benchmarks, get_benchmark
from saliencegate.benchmarks.state_decay.runner import (
    render_benchmark_human,
    run_state_decay_smoke,
)
from saliencegate.commands.replay import run_replay
from saliencegate.experiments import (
    Stage2ConditionId,
    available_stage2_conditions,
)

ROOT = Path(__file__).parents[2]
TRACE = ROOT / "tests" / "fixtures" / "runs" / "basic.jsonl"

REPLAY_RESULT_DIGEST = "09698605fb38e08da4811161513a31e4cc9c37f29e26f015680d96603b81fc92"
REPLAY_CONTENT_DIGEST = "d522e06eb74a20f981222b07eff9535317413d451c45e0174c8603186ecdbb42"
REPLAY_FIXTURE_DIGEST = "59f0f7016600c1010c19dba01e21bc4e34bb6345e5cc42349e90a0aa0a16020b"
SMOKE_MANIFEST_DIGEST = "32600f0adce1c21d7081cf2fb01d18722eb4d78c0b49f6b255f3f333962dc3f0"

_STAGE2_DIGESTS = {
    Stage2ConditionId.NO_MEMORY: "0b3fcd4bb9d260b4fe8a560c20daa18c735848d9eb4215ae6b675cbccd9cbb44",
    Stage2ConditionId.FIXED_STEP: (
        "87a37f9c65fa8ce7ff8eb499a073357c5c858cea066be9d1123c7af098bb9dd7"
    ),
    Stage2ConditionId.RETRIEVAL_ALWAYS: (
        "3ba0335d5166d94e059a9091f2d83ca5be25f8470b02a752db574b2065d300da"
    ),
    Stage2ConditionId.ALWAYS_INJECT: (
        "fa21957aa68b1129ccfbee894fd1407872b70b8ec0039dca942035af69c83863"
    ),
}

_IMPORT_PROBE = r"""
import importlib.abc
import json
import sys
from pathlib import Path

prefixes = (
    "httpx",
    "openai",
    "openai_harmony",
    "saliencegate.model_runtime",
    "saliencegate.benchmarks.state_decay_v2",
)


def prohibited(name):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


class Recorder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.seen = set()

    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if prohibited(fullname):
            self.seen.add(fullname)
        return None


root = Path(sys.argv[1])
observation = Path(sys.argv[2])
preloaded = sorted(name for name in sys.modules if prohibited(name))
recorder = Recorder()
sys.meta_path.insert(0, recorder)

from saliencegate.cli import main

artifact = root / "smoke"
exit_codes = {
    "benchmark": main(
        ["benchmark", "state-decay-smoke", "--output", str(artifact), "--json"]
    ),
    "validate": main(["validate", str(artifact / "manifest.json"), "--json"]),
}
observation.write_text(
    json.dumps(
        {
            "exit_codes": exit_codes,
            "preloaded": preloaded,
            "seen": sorted(recorder.seen),
        },
        sort_keys=True,
    ),
    encoding="utf-8",
)
raise SystemExit(max(exit_codes.values()))
"""

_REVIEW_CLI_COMPATIBILITY_PROBE = r"""
import contextlib
import importlib
import io
import json
import sys
from pathlib import Path

from saliencegate.benchmarks.registry import available_benchmarks
from saliencegate.cli import ExitCode, main as legacy_main


def exit_code_snapshot():
    return [(member.name, int(member)) for member in ExitCode]


def benchmark_snapshot():
    return [definition.model_dump(mode="json") for definition in available_benchmarks()]


observation = Path(sys.argv[1])
before = {
    "exit_codes": exit_code_snapshot(),
    "benchmarks": benchmark_snapshot(),
}

review_cli = importlib.import_module(
    "saliencegate.benchmarks.state_decay_v2.review_cli"
)
review_stdout = io.StringIO()
review_stderr = io.StringIO()
with contextlib.redirect_stdout(review_stdout), contextlib.redirect_stderr(review_stderr):
    review_exit = review_cli.main([])

legacy_stdout = io.StringIO()
legacy_stderr = io.StringIO()
with contextlib.redirect_stdout(legacy_stdout), contextlib.redirect_stderr(legacy_stderr):
    legacy_exit = legacy_main([])

after = {
    "exit_codes": exit_code_snapshot(),
    "benchmarks": benchmark_snapshot(),
}
observation.write_text(
    json.dumps(
        {
            "after": after,
            "before": before,
            "legacy_exit": int(legacy_exit),
            "legacy_main_identity_preserved": (
                importlib.import_module("saliencegate.cli").main is legacy_main
            ),
            "legacy_stderr": legacy_stderr.getvalue(),
            "legacy_stdout": legacy_stdout.getvalue(),
            "review_exit": int(review_exit),
        },
        sort_keys=True,
    ),
    encoding="utf-8",
)
"""


@pytest.mark.asyncio
async def test_public_surface_preserves_replay_and_smoke_digest_anchors(
    tmp_path: Path,
) -> None:
    replay = await run_replay(TRACE, output_path=tmp_path / "replay")
    loaded = load_validated_artifact(tmp_path / "replay" / "manifest.json")
    smoke = run_state_decay_smoke(tmp_path / "smoke")

    assert replay.result_digest == REPLAY_RESULT_DIGEST
    assert replay.overall_content_digest == REPLAY_CONTENT_DIGEST
    assert loaded.run.fixture_digest == REPLAY_FIXTURE_DIGEST
    assert smoke.manifest_digest == SMOKE_MANIFEST_DIGEST


def test_public_surface_freezes_the_v1_registry_and_report_shapes(tmp_path: Path) -> None:
    expected_definition = {
        "schema_version": "benchmark-definition/v1",
        "suite_id": "state-decay-smoke",
        "suite_version": "v1",
        "title": "StateDecayBench smoke",
        "diagnostic": True,
        "synthetic": True,
        "balanced": True,
        "external_claims_supported": False,
        "scenario_count": 32,
    }
    definitions = available_benchmarks()

    assert tuple(item.model_dump(mode="json") for item in definitions) == (expected_definition,)
    assert get_benchmark("state-decay-smoke").model_dump(mode="json") == expected_definition

    report = run_state_decay_smoke(tmp_path / "smoke")
    assert report.model_dump(mode="json") == {
        "schema_version": "cli-benchmark-report/v1",
        "status": "ok",
        "suite_id": "state-decay-smoke",
        "suite_version": "v1",
        "generator_version": "v1",
        "seed": 20260711,
        "diagnostic": True,
        "synthetic": True,
        "balanced": True,
        "external_claims_supported": False,
        "external_claims_assessment": "insufficient",
        "scenario_count": 32,
        "family_count": 8,
        "intervene_count": 16,
        "silence_count": 16,
        "oracle_passed": 32,
        "oracle_failed": 0,
        "fixture_digest": "34fcf4ab0bee256ad7d091da261eb190b2d7a96f3dff0ef9eaef8846c32e880e",
        "oracle_result_digest": (
            "f27879d5054a2283a88cc74df1368bdf97ba6ef04eefd386bd7f1bda28a8f0b2"
        ),
        "overall_content_digest": (
            "a210eed1a07cc6de41c27a20b522e5674b05f91a1ac449137d6c80008d2509f7"
        ),
        "manifest_digest": SMOKE_MANIFEST_DIGEST,
    }
    assert render_benchmark_human(report) == (
        "StateDecayBench smoke complete\n"
        "scenarios: 32\n"
        "families: 8\n"
        "labels: 16 intervene, 16 silence\n"
        "oracle: 32 passed, 0 failed\n"
        "diagnostic: yes\n"
        "synthetic: yes\n"
        "balanced: yes\n"
        "external claims: insufficient\n"
        f"manifest digest: {SMOKE_MANIFEST_DIGEST}\n"
    )


def test_public_surface_freezes_stage2_condition_and_algorithm_schema_identities() -> None:
    conditions = available_stage2_conditions()

    assert tuple(item.condition_id for item in conditions) == tuple(Stage2ConditionId)
    assert {item.condition_id: item.condition_digest for item in conditions} == _STAGE2_DIGESTS
    assert ALGORITHM_ARTIFACT_SCHEMA_VERSION == "algorithm-artifact/v1"
    assert AlgorithmArtifactManifest.model_fields["schema_version"].default == (
        ALGORITHM_ARTIFACT_SCHEMA_VERSION
    )


def test_public_v1_offline_commands_do_not_cross_future_import_boundaries(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    observation = tmp_path / "imports.json"
    environment = os.environ.copy()
    environment.update(HOME=str(home), PYTHONUTF8="1")

    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            "-c",
            _IMPORT_PROBE,
            str(tmp_path),
            str(observation),
        ),
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert len(completed.stdout.splitlines()) == 2
    assert json.loads(observation.read_text(encoding="utf-8")) == {
        "exit_codes": {"benchmark": 0, "validate": 0},
        "preloaded": [],
        "seen": [],
    }


def test_review_cli_is_additive_to_the_frozen_v1_surface(tmp_path: Path) -> None:
    observation = tmp_path / "review-cli-compatibility.json"

    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            "-c",
            _REVIEW_CLI_COMPATIBILITY_PROBE,
            str(observation),
        ),
        cwd=ROOT,
        env={**os.environ, "PYTHONUTF8": "1"},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""
    observed = json.loads(observation.read_text(encoding="utf-8"))
    assert observed["after"] == observed["before"]
    assert observed["before"]["exit_codes"] == [
        ["SUCCESS", 0],
        ["INVALID_INPUT", 2],
        ["CONFIGURATION", 3],
        ["UNAVAILABLE_DEPENDENCY", 4],
        ["CORRUPTED_ARTIFACT", 5],
        ["INTERNAL_ERROR", 70],
    ]
    assert [item["suite_id"] for item in observed["before"]["benchmarks"]] == ["state-decay-smoke"]
    assert observed["legacy_main_identity_preserved"] is True
    assert observed["legacy_exit"] == 2
    assert observed["legacy_stdout"] == ""
    assert observed["legacy_stderr"] == "error: invalid command line\n"
    assert observed["review_exit"] == 2
