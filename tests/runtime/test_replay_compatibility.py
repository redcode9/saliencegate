from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import saliencegate.artifacts.export as artifact_export
from saliencegate.artifacts import RevisionEvidence, RevisionSource, load_validated_artifact
from saliencegate.commands.replay import run_replay
from saliencegate.models import ReplayModel
from saliencegate.ports.memory import MemoryCycleOutput
from saliencegate.ports.models import (
    MEMORY_CYCLE_OUTPUT_SCHEMA_VERSION,
    MODEL_REQUEST_SCHEMA_VERSION,
    MODEL_RESULT_SCHEMA_VERSION,
    ModelRequest,
    ModelResult,
)

ROOT = Path(__file__).parents[2]
TRACE = ROOT / "tests" / "fixtures" / "runs" / "basic.jsonl"

REPLAY_RESULT_DIGEST = "09698605fb38e08da4811161513a31e4cc9c37f29e26f015680d96603b81fc92"
REPLAY_CONTENT_DIGEST = "d522e06eb74a20f981222b07eff9535317413d451c45e0174c8603186ecdbb42"
REPLAY_FIXTURE_DIGEST = "59f0f7016600c1010c19dba01e21bc4e34bb6345e5cc42349e90a0aa0a16020b"

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
    "saliencegate.models.openai_compatible",
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


preloaded = sorted(name for name in sys.modules if prohibited(name))
recorder = Recorder()
sys.meta_path.insert(0, recorder)

from saliencegate.cli import main

trace, root_value, observation = sys.argv[1:]
root = Path(root_value)
artifact = root / "artifact"
commands = (
    (
        "doctor",
        ["doctor", "--repository", str(root / "doctor.sqlite3"), "--json"],
    ),
    (
        "replay",
        ["replay", trace, "--output", str(artifact), "--json"],
    ),
    (
        "validate",
        ["validate", str(artifact / "manifest.json"), "--json"],
    ),
    (
        "inspect",
        [
            "inspect",
            "00000000-0000-4000-8000-00000000d001",
            "--artifact",
            str(artifact),
            "--json",
        ],
    ),
    (
        "benchmark",
        [
            "benchmark",
            "state-decay-smoke",
            "--output",
            str(root / "smoke"),
            "--json",
        ],
    ),
)
exit_codes = {name: main(arguments) for name, arguments in commands}
Path(observation).write_text(
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


def _revision(commit: str) -> RevisionEvidence:
    return RevisionEvidence(
        source=RevisionSource.GIT,
        package_version="0.1.0",
        commit=commit,
        dirty_worktree=False,
    )


@pytest.mark.asyncio
async def test_replay_contract_is_revision_independent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert MODEL_REQUEST_SCHEMA_VERSION == "model-request/v1"
    assert MODEL_RESULT_SCHEMA_VERSION == "model-result/v1"
    assert MEMORY_CYCLE_OUTPUT_SCHEMA_VERSION == "memory-cycle-output/v1"

    requests: list[ModelRequest] = []
    results: list[ModelResult] = []
    original_generate = ReplayModel.generate

    async def capture_generate(
        self: ReplayModel,
        request: ModelRequest,
    ) -> ModelResult:
        requests.append(ModelRequest.model_validate_json(request.model_dump_json(warnings=False)))
        result = await original_generate(self, request)
        results.append(ModelResult.model_validate_json(result.model_dump_json(warnings=False)))
        return result

    monkeypatch.setattr(ReplayModel, "generate", capture_generate)
    monkeypatch.setattr(
        artifact_export,
        "discover_revision",
        lambda source_dir=None: _revision("a" * 40),
    )
    first = await run_replay(TRACE, output_path=tmp_path / "first")

    monkeypatch.setattr(
        artifact_export,
        "discover_revision",
        lambda source_dir=None: _revision("b" * 40),
    )
    second = await run_replay(TRACE, output_path=tmp_path / "second")

    assert first.result_digest == second.result_digest == REPLAY_RESULT_DIGEST
    assert first.overall_content_digest == second.overall_content_digest == REPLAY_CONTENT_DIGEST
    assert first.manifest_digest != second.manifest_digest

    assert {request.schema_version for request in requests} == {MODEL_REQUEST_SCHEMA_VERSION}
    assert {request.response_schema_version for request in requests} == {
        MEMORY_CYCLE_OUTPUT_SCHEMA_VERSION
    }
    assert tuple(request.model_call_index for request in requests) == (0,) * 6
    assert {result.schema_version for result in results} == {MODEL_RESULT_SCHEMA_VERSION}
    assert all(type(result.output) is MemoryCycleOutput for result in results)
    assert set(MemoryCycleOutput.model_fields) == {"schema_version", "delta", "observation"}
    assert {result.output.schema_version for result in results if result.output is not None} == {
        MEMORY_CYCLE_OUTPUT_SCHEMA_VERSION
    }

    first_artifact = load_validated_artifact(tmp_path / "first" / "manifest.json")
    second_artifact = load_validated_artifact(tmp_path / "second" / "manifest.json")
    assert first_artifact.manifest.artifact_kind == "replay_run"
    assert first_artifact.run.fixture_digest == REPLAY_FIXTURE_DIGEST
    assert first_artifact.manifest.revision.commit == "a" * 40
    assert second_artifact.manifest.revision.commit == "b" * 40
    assert first_artifact.manifest.components == second_artifact.manifest.components
    assert first_artifact.manifest.model_dump(
        mode="json",
        exclude={"manifest_digest", "revision"},
    ) == second_artifact.manifest.model_dump(
        mode="json",
        exclude={"manifest_digest", "revision"},
    )


def test_replay_does_not_attempt_optional_runtime_imports(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    observation = tmp_path / "imports.json"
    environment = os.environ.copy()
    environment.update(
        HOME=str(home),
        PYTHONUTF8="1",
    )

    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            "-c",
            _IMPORT_PROBE,
            str(TRACE),
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
    reports = tuple(json.loads(line) for line in completed.stdout.splitlines())
    assert len(reports) == 5
    assert json.loads(observation.read_text(encoding="utf-8")) == {
        "exit_codes": {
            "benchmark": 0,
            "doctor": 0,
            "inspect": 0,
            "replay": 0,
            "validate": 0,
        },
        "preloaded": [],
        "seen": [],
    }
