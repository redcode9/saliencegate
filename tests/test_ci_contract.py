from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import scripts.verify_built_artifacts as artifact_verifier
from scripts.verify_built_artifacts import (
    DOCUMENTED_COMMAND_CASES,
    EXECUTED_COMMAND_CASES,
    SUPPLEMENTAL_ARTIFACT_PROOFS,
    _run,
    artifact_compatible_commands,
)

from saliencegate.security import inspect_private_file_location

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE = ROOT / "Makefile"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
README = ROOT / "README.md"
SOCKET_GUARD = ROOT / "scripts" / "run_without_sockets.py"
CORE_IMPORT_SMOKE = ROOT / "scripts" / "smoke_core_imports.py"
MODEL_RUNTIME_SMOKE = ROOT / "scripts" / "smoke_model_runtime.py"
LAUNCH_CONTRACT_SMOKE = ROOT / "scripts" / "smoke_launch_contracts.py"
PACKAGE_IMPORT_SMOKE = ROOT / "scripts" / "smoke_package_imports.py"
SHADOW_INSTALLED_SMOKE = ROOT / "scripts" / "smoke_shadow_installed.py"
SHADOW_TRACE_BENCHMARK = ROOT / "scripts" / "benchmark_shadow_trace.py"
ARTIFACT_VERIFIER = ROOT / "scripts" / "verify_built_artifacts.py"
FORBIDDEN_CORE_MODULES = ("anthropic", "harbor", "httpx", "openai", "openai_harmony")

REQUIRED_TARGETS = (
    "format",
    "lint",
    "typecheck",
    "test",
    "coverage",
    "docs-check",
    "build",
    "audit",
    "check",
)


def _read(relative: str) -> str:
    path = ROOT / relative
    assert path.is_file(), f"required repository file is missing: {relative}"
    return path.read_text(encoding="utf-8")


def _issue_field(text: str, field_id: str) -> str:
    match = re.search(
        rf"(?ms)^  - type: [^\n]+\n    id: {re.escape(field_id)}\n"
        r"(?P<body>.*?)(?=^  - type:|\Z)",
        text,
    )
    assert match is not None, f"required issue field is missing: {field_id}"
    return match.group("body")


def _job_block(text: str, job_id: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(job_id)}:\n.*?(?=^  [a-z0-9][a-z0-9-]*:\n|\Z)",
        text,
    )
    assert match is not None, f"required CI job is missing: {job_id}"
    return match.group(0)


def _step_containing(job: str, needle: str) -> str:
    blocks = re.findall(
        r"(?ms)^      - name: .*?(?=^      - name:|\Z)",
        job,
    )
    matches = [block for block in blocks if needle in block]
    assert len(matches) == 1, f"expected one CI step containing {needle!r}"
    return matches[0]


def _assert_installed_shadow_smoke(job: str, *, prefix: str) -> None:
    extraction = _step_containing(job, "tar -xzf dist/*.tar.gz")
    shadow = _step_containing(job, '" shadow analyze "')
    packaged_root = f'"$RUNNER_TEMP/{prefix}-package"'
    shadow_root = f'"$RUNNER_TEMP/{prefix}-shadow"'

    assert "--strip-components=1" in extraction
    assert packaged_root in extraction
    assert f'"$RUNNER_TEMP/{prefix}-package/examples/shadow_asyncio.py"' in shadow
    assert re.search(rf"(?m)^\s+mkdir -p [^\n]*{re.escape(shadow_root)}", shadow)
    assert re.search(rf"(?m)^\s+chmod 700 [^\n]*{re.escape(shadow_root)}", shadow)
    assert "umask 077" in shadow
    assert "scripts/smoke_shadow_installed.py write-trace" in shadow
    assert f'"$RUNNER_TEMP/{prefix}-shadow/events.ndjson"' in shadow
    assert f'"$RUNNER_TEMP/{prefix}-shadow/shadow-report.json"' in shadow
    assert f'"$RUNNER_TEMP/{prefix}-shadow/shadow-command.json"' in shadow
    assert "--capture-scope complete_run_declared" in shadow
    assert "scripts/smoke_shadow_installed.py validate-report" in shadow
    assert f'"$RUNNER_TEMP/{prefix}-package/examples/atif-shadow/one_call.py"' in shadow
    assert (
        f'"$RUNNER_TEMP/{prefix}-package/examples/atif-shadow/codex-minimal.trajectory.json"'
    ) in shadow
    assert (
        f'"$RUNNER_TEMP/{prefix}-package/examples/atif-shadow/terminus-minimal.trajectory.json"'
    ) in shadow
    assert shadow.count(" shadow analyze-atif ") == 2
    assert shadow.count(" validate-public-atif ") == 2
    assert "--profile harbor-codex-v1" in shadow
    assert "--profile harbor-terminus-2-v1" in shadow
    assert "harbor-codex/v1" in shadow
    assert "harbor-terminus-2/v1" in shadow
    assert "scripts/smoke_shadow_installed.py exercise-atif" in shadow
    assert 'OPENAI_API_KEY="provider-credential-read-must-fail"' in shadow
    assert 'ANTHROPIC_API_KEY="provider-credential-read-must-fail"' in shadow
    assert "tests/fixtures" not in shadow

    guarded_lines = [line for line in shadow.splitlines() if 'bin/python"' in line]
    assert len(guarded_lines) == 10
    assert all("-I scripts/run_without_sockets.py" in line for line in guarded_lines)


def test_artifact_verifier_reports_captured_failures_without_credentials() -> None:
    credential = "provider-credential-read-must-fail"
    source = (
        "import sys; "
        f"print({credential!r}); "
        "print('actionable stderr', file=sys.stderr); "
        "raise SystemExit(7)"
    )

    try:
        _run((sys.executable, "-I", "-c", source), capture_output=True)
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("the failing proof command unexpectedly succeeded")

    assert "exited with 7" in message
    assert "actionable stderr" in message
    assert credential not in message
    assert "<redacted>" in message


def test_artifact_workspace_canonicalizes_a_symlinked_temp_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    actual_parent = tmp_path / "actual"
    actual_parent.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual_parent, target_is_directory=True)
    created = actual_parent / "proof"
    created.mkdir(mode=0o700)
    returned = alias / "proof"
    monkeypatch.setattr(
        artifact_verifier.tempfile,
        "mkdtemp",
        lambda *, prefix: str(returned),
    )

    with artifact_verifier._workspace(None) as workspace:
        assert workspace == created.resolve(strict=True)
        assert workspace.is_dir()
        inspect_private_file_location(workspace / "absent")
        assert not (workspace / "absent").exists()

    assert not created.exists()
    assert actual_parent.is_dir()
    assert alias.is_symlink()


def test_makefile_exposes_noninteractive_gates_in_the_authoritative_order() -> None:
    text = _read("Makefile")

    for target in REQUIRED_TARGETS:
        assert re.search(rf"(?m)^{re.escape(target)}\s*:", text)
    assert ".PHONY:" in text
    phony = next(line for line in text.splitlines() if line.startswith(".PHONY:"))
    assert all(target in phony.split() for target in REQUIRED_TARGETS)
    assert re.search(r"(?m)^\.NOTPARALLEL\s*:\s*$", text)
    assert re.search(r"(?m)^artifact-smoke\s*:\s*$", text)
    assert "artifact-smoke" in phony.split()

    docs_check = re.search(r"(?ms)^docs-check:\n(?P<body>.*?)(?=^[a-z-]+:)", text)
    assert docs_check is not None
    assert re.findall(r"(?m)^\s+(.+)$", docs_check.group("body")) == [
        "uv run --locked python scripts/check_public_tree.py",
        "uv run --locked python scripts/check_public_docs.py",
        "uv run --locked python scripts/check_readme_visuals.py",
    ]

    check = re.search(r"(?m)^check\s*:\s*(?P<dependencies>[^\n]+)$", text)
    assert check is not None
    assert tuple(check.group("dependencies").split()) == REQUIRED_TARGETS[:-1]

    required_commands = (
        "ruff format --check .",
        "ruff check .",
        "mypy src/saliencegate",
        "pytest --cov=saliencegate --cov-branch",
        "python scripts/check_public_tree.py",
        "python scripts/check_public_docs.py",
        "python scripts/check_readme_visuals.py",
        "uv lock --check",
        "uv build --no-build-isolation --clear --no-create-gitignore",
        "uv export --locked --all-extras --all-groups --no-emit-project",
        "SALIENCEGATE_REQUIRE_DISTRIBUTIONS=1",
        "pytest -q tests/test_package.py",
        "python scripts/verify_built_artifacts.py --dist-dir dist",
        "set -eu",
        "pip-audit --strict",
        "--progress-spinner off",
        "--disable-pip",
    )
    assert all(command in text for command in required_commands)


def test_ci_is_least_privilege_pinned_and_covers_supported_python() -> None:
    text = _read(".github/workflows/ci.yml")

    assert re.search(r"(?m)^permissions:\s*\n\s+contents:\s*read\s*$", text)
    assert len(re.findall(r"(?m)^permissions:", text)) == 1
    assert not re.search(r"(?m)^[ \t]+permissions:", text)
    assert not re.search(r"(?mi)^permissions:\s*(?:write-all|\{[^\n]*write)", text)
    assert not re.search(r"(?m)^\s+[a-z-]+:\s*write\s*$", text)
    assert "pull_request:" in text
    assert "push:" in text
    assert "workflow_dispatch:" in text
    assert "fail-fast: false" in text
    assert "timeout-minutes:" in text

    strategy = re.search(
        r"(?ms)^    strategy:\n(?P<body>.*?)(?=^    steps:)",
        text,
    )
    assert strategy is not None
    matrix_versions = re.findall(
        r'^          - "([0-9]+\.[0-9]+)"$',
        strategy.group("body"),
        flags=re.MULTILINE,
    )
    assert matrix_versions == ["3.11", "3.12", "3.13"]

    uses_lines = re.findall(r"(?m)^\s*-?\s*uses:\s*([^\s#]+)(?:\s+#\s*(\S.*))?$", text)
    assert uses_lines
    for reference, version_comment in uses_lines:
        assert re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", reference)
        assert re.search(r"\bv\d", version_comment)

    assert "astral-sh/setup-uv@" in text
    assert "enable-cache: true" in text
    assert "cache-dependency-glob: uv.lock" in text
    assert 'UV_LOCKED: "1"' in text
    assert "UV_FROZEN" not in text
    first_sync = "uv sync --locked --all-extras --dev --no-install-project"
    second_sync = "uv sync --locked --all-extras --dev --no-build-isolation"
    for job_id in ("test", "quality", "coverage", "build"):
        job = _job_block(text, job_id)
        assert job.count(first_sync) == 1
        assert job.count(second_sync) == 1
        assert job.index(first_sync) < job.index(second_sync)
    assert text.count(first_sync) == 4
    assert text.count(second_sync) == 4
    benchmark_first_sync = "uv sync --locked --dev --no-install-project"
    benchmark_second_sync = "uv sync --locked --dev --no-build-isolation"
    performance = _job_block(text, "shadow-trace-performance")
    assert text.count(benchmark_first_sync) == 1
    assert text.count(benchmark_second_sync) == 1
    assert performance.index(benchmark_first_sync) < performance.index(benchmark_second_sync)
    build_command = "uv build --no-build-isolation --clear --no-create-gitignore"
    build = _job_block(text, "build")
    assert build.count(build_command) == 1
    assert text.count(build_command) == 1
    for alternative in ("hatch build", "pip wheel", "python -m build", "python -m hatchling"):
        assert alternative not in text


def test_ci_separates_static_quality_from_authoritative_coverage() -> None:
    text = _read(".github/workflows/ci.yml")
    quality = _job_block(text, "quality")
    coverage = _job_block(text, "coverage")
    build = _job_block(text, "build")

    assert re.search(r"(?m)^    timeout-minutes: 30$", quality)
    assert re.findall(r"(?m)^        run: (make [^\n]+)$", quality) == [
        "make format lint typecheck docs-check audit"
    ]
    assert 'python-version: "3.12"' in coverage
    assert re.search(r"(?m)^    timeout-minutes: 90$", coverage)
    assert re.findall(r"(?m)^        run: (make [^\n]+)$", coverage) == ["make coverage"]
    assert "coverage" not in re.findall(r"(?m)^        run: (make [^\n]+)$", quality)[0].split()

    needs = re.search(r"(?ms)^    needs:\n(?P<body>.*?)(?=^    [a-z-]+:)", build)
    assert needs is not None
    assert re.findall(r"(?m)^      - ([a-z][a-z-]+)$", needs.group("body")) == [
        "test",
        "quality",
        "coverage",
        "shadow-trace-performance",
    ]

    makefile = _read("Makefile")
    check = re.search(r"(?m)^check\s*:\s*(?P<dependencies>[^\n]+)$", makefile)
    assert check is not None
    assert tuple(check.group("dependencies").split()) == REQUIRED_TARGETS[:-1]
    assert "pytest --cov=saliencegate --cov-branch" in makefile
    assert "fail_under = 95" in _read("pyproject.toml")


def test_ci_gates_the_documented_shadow_trace_reference_budgets() -> None:
    text = _read(".github/workflows/ci.yml")
    performance = _job_block(text, "shadow-trace-performance")
    build = _job_block(text, "build")

    assert "runs-on: ubuntu-24.04" in performance
    assert 'python-version: "3.12"' in performance
    assert re.search(r"(?m)^    timeout-minutes: 30$", performance)
    assert "uv sync --locked --dev --no-install-project" in performance
    assert "uv sync --locked --dev --no-build-isolation" in performance
    assert "SALIENCEGATE_BENCHMARK_RUNNER_IMAGE: ubuntu-24.04" in performance
    assert (
        "uv run --locked python scripts/benchmark_shadow_trace.py --assert-budgets" in performance
    )
    assert "- shadow-trace-performance" in build

    benchmark = SHADOW_TRACE_BENCHMARK.read_text(encoding="utf-8")
    for required in (
        "1_000",
        "5.0",
        "15.0",
        "512",
        "median",
        "peak_rss",
        "cpu",
        "runner_image",
        "--assert-budgets",
    ):
        assert required in benchmark


def test_ci_builds_once_and_gates_distribution_membership_before_upload() -> None:
    text = _read(".github/workflows/ci.yml")
    build = _job_block(text, "build")

    assert text.count("actions/upload-artifact@") == 1
    assert build.count("actions/upload-artifact@") == 1
    assert "SALIENCEGATE_REQUIRE_DISTRIBUTIONS=1" in build
    assert "uv run --locked pytest -q tests/test_package.py" in build
    assert build.index("SALIENCEGATE_REQUIRE_DISTRIBUTIONS=1") < build.index(
        "actions/upload-artifact@"
    )
    assert "*.tar.gz" in text

    step_blocks = re.findall(
        r"(?ms)^      - name: .*?(?=^      - name:|^  [a-z][a-z-]+:|\Z)",
        text,
    )
    upload_steps = [block for block in step_blocks if "actions/upload-artifact@" in block]
    assert len(upload_steps) == 1
    uploaded_paths = re.findall(r"(?m)^\s+path:\s*([^\n]+)$", upload_steps[0])
    assert uploaded_paths == ["dist/"]
    assert all(
        forbidden not in "\n".join(uploaded_paths)
        for forbidden in ("trace", "runs", ".artifacts", "reports")
    )

    assert "Require exactly one wheel and one source distribution" in text
    assert "find dist -mindepth 1 -maxdepth 1" in text
    assert "-name '*.whl'" in text
    assert "-name '*.tar.gz'" in text


def test_ci_proves_public_atif_semantics_from_artifacts_without_checkout() -> None:
    text = _read(".github/workflows/ci.yml")
    artifact = _job_block(text, "artifact-only-atif")

    assert "needs:\n      - build" in artifact
    assert "runs-on: ubuntu-24.04" in artifact
    assert 'PYTHONPATH: ""' in artifact
    assert "actions/download-artifact@" in artifact
    assert "actions/checkout@" not in artifact
    assert "GITHUB_WORKSPACE" not in artifact
    assert "tests/fixtures" not in artifact
    assert "${{ runner.temp }}/saliencegate-artifact-only" in artifact
    assert 'tar -xzf "$ARTIFACT_ROOT"/dist/*.tar.gz --strip-components=1' in artifact
    assert 'test -f "$ARTIFACT_ROOT/launcher/scripts/verify_built_artifacts.py"' in artifact
    assert 'python "$ARTIFACT_ROOT/launcher/scripts/verify_built_artifacts.py"' in artifact
    assert '--dist-dir "$ARTIFACT_ROOT/dist"' in artifact
    assert '--work-dir "$ARTIFACT_ROOT/proof"' in artifact
    assert "--python 3.12" in artifact
    assert " shadow analyze-atif " not in artifact
    assert " validate-public-atif " not in artifact

    assert ARTIFACT_VERIFIER.is_file()
    assert set(DOCUMENTED_COMMAND_CASES) | SUPPLEMENTAL_ARTIFACT_PROOFS == EXECUTED_COMMAND_CASES
    documented: list[str] = []
    for path in (README, ROOT / "examples" / "atif-shadow" / "README.md"):
        documented.extend(artifact_compatible_commands(path.read_text(encoding="utf-8")))
    assert set(documented) == set(DOCUMENTED_COMMAND_CASES.values())


def test_ci_exercises_the_installed_core_wheel_without_optional_runtime() -> None:
    text = _read(".github/workflows/ci.yml")
    core = _job_block(text, "installed-core-wheel")

    assert "needs:\n      - build" in core
    assert 'PYTHONPATH: ""' in core
    assert "actions/download-artifact@" in core
    assert (
        "uv export --locked --no-dev --no-emit-project "
        '--output-file "$RUNNER_TEMP/core-runtime-requirements.txt" --quiet'
    ) in core
    assert '--require-hashes -r "$RUNNER_TEMP/core-runtime-requirements.txt"' in core
    assert "--no-deps dist/*.whl" in core
    assert '"$RUNNER_TEMP/core-wheel-venv/bin/python" -I scripts/run_without_sockets.py' in core
    assert "scripts/smoke_core_imports.py" in core
    assert core.count("scripts/run_without_sockets.py") >= 19
    import_smoke = CORE_IMPORT_SMOKE.read_text(encoding="utf-8")
    assert "import saliencegate.cli" in import_smoke
    assert "import saliencegate.shadow" in import_smoke
    assert "find_spec(optional_module)" in import_smoke
    assert "optional_module in sys.modules" in import_smoke
    for optional_module in FORBIDDEN_CORE_MODULES:
        assert f'"{optional_module}"' in import_smoke
    for command in (
        '"$RUNNER_TEMP/core-wheel-venv/bin/saliencegate" doctor',
        '"$RUNNER_TEMP/core-wheel-venv/bin/saliencegate" replay',
        '"$RUNNER_TEMP/core-wheel-venv/bin/saliencegate" validate',
        '"$RUNNER_TEMP/core-wheel-venv/bin/saliencegate" benchmark state-decay-smoke',
        '"$RUNNER_TEMP/core-wheel-venv/bin/saliencegate" algorithm replay',
    ):
        assert command in core
    assert "tests/fixtures/runs/basic.jsonl" in core
    assert "tests/fixtures/runs/paper_two_phase_basic.jsonl" in core
    assert "tests/fixtures/models/paper_two_phase_fixed_step_responses.jsonl" in core
    assert "--condition fixed_step" in core
    assert '"$RUNNER_TEMP/core-wheel-output/algorithm/manifest.json"' in core
    assert (
        '"$RUNNER_TEMP/core-wheel-venv/bin/saliencegate" demo --json '
        '> "$RUNNER_TEMP/core-wheel-output/launch-demo.json"'
    ) in core
    assert (
        '"$RUNNER_TEMP/core-wheel-venv/bin/saliencegate-review" --help '
        '> "$RUNNER_TEMP/core-wheel-output/review-help.txt"'
    ) in core
    for command in ("build-pack", "review", "status", "build-envelope"):
        assert (
            f'"$RUNNER_TEMP/core-wheel-venv/bin/saliencegate-review" {command} --help '
            '>> "$RUNNER_TEMP/core-wheel-output/review-help.txt"'
        ) in core
    assert (
        "scripts/smoke_launch_contracts.py "
        '"$RUNNER_TEMP/core-wheel-output/launch-demo.json" '
        '"$RUNNER_TEMP/core-wheel-output/review-help.txt"'
    ) in core
    launch_smoke = LAUNCH_CONTRACT_SMOKE.read_text(encoding="utf-8")
    assert "launch-contracts-ok" in launch_smoke
    assert "finalize" in launch_smoke
    assert "build-envelope" in launch_smoke
    _assert_installed_shadow_smoke(core, prefix="core-wheel")


def test_ci_exercises_the_installed_sdist_from_locked_build_dependencies() -> None:
    text = _read(".github/workflows/ci.yml")
    sdist = _job_block(text, "installed-sdist")

    assert "needs:\n      - build" in sdist
    assert 'PYTHONPATH: ""' in sdist
    assert "actions/download-artifact@" in sdist
    assert "--all-extras" not in sdist
    assert "--all-groups" not in sdist
    assert (
        "uv export --locked --no-dev --no-emit-project "
        '--output-file "$RUNNER_TEMP/sdist-core-runtime-requirements.txt" --quiet'
    ) in sdist
    assert (
        "uv export --locked --only-dev --no-emit-project "
        '--output-file "$RUNNER_TEMP/sdist-build-constraints.txt" --quiet'
    ) in sdist
    assert '--require-hashes -r "$RUNNER_TEMP/sdist-core-runtime-requirements.txt"' in sdist
    assert '--require-hashes --constraints "$RUNNER_TEMP/sdist-build-constraints.txt"' in sdist
    lock = _read("uv.lock")
    hatchling = re.search(r'(?ms)^\[\[package\]\]\nname = "hatchling"\nversion = "([^"]+)"', lock)
    assert hatchling is not None
    assert f"hatchling=={hatchling.group(1)}" in sdist
    assert "--no-deps --no-build-isolation dist/*.tar.gz" in sdist
    assert '"$RUNNER_TEMP/sdist-venv/bin/python" -I scripts/run_without_sockets.py' in sdist
    assert "scripts/smoke_package_imports.py" in sdist
    assert sdist.count("scripts/run_without_sockets.py") >= 19
    import_smoke = PACKAGE_IMPORT_SMOKE.read_text(encoding="utf-8")
    assert "import saliencegate.shadow" in import_smoke
    assert "find_spec(optional_module)" in import_smoke
    assert "optional_module in sys.modules" in import_smoke
    for optional_module in FORBIDDEN_CORE_MODULES:
        assert f'"{optional_module}"' in import_smoke
    for command in (
        '"$RUNNER_TEMP/sdist-venv/bin/saliencegate" doctor',
        '"$RUNNER_TEMP/sdist-venv/bin/saliencegate" replay',
        '"$RUNNER_TEMP/sdist-venv/bin/saliencegate" validate',
        '"$RUNNER_TEMP/sdist-venv/bin/saliencegate" benchmark state-decay-smoke',
        '"$RUNNER_TEMP/sdist-venv/bin/saliencegate" algorithm replay',
    ):
        assert command in sdist
    assert "tests/fixtures/runs/basic.jsonl" in sdist
    assert "tests/fixtures/runs/paper_two_phase_basic.jsonl" in sdist
    assert "tests/fixtures/models/paper_two_phase_fixed_step_responses.jsonl" in sdist
    assert "--condition fixed_step" in sdist
    assert '"$RUNNER_TEMP/sdist-output/algorithm/manifest.json"' in sdist
    assert 'chmod 700 "$RUNNER_TEMP/sdist-home" "$RUNNER_TEMP/sdist-output"' in sdist
    assert (
        '"$RUNNER_TEMP/sdist-venv/bin/saliencegate" demo --json '
        '> "$RUNNER_TEMP/sdist-output/launch-demo.json"'
    ) in sdist
    assert (
        '"$RUNNER_TEMP/sdist-venv/bin/saliencegate-review" --help '
        '> "$RUNNER_TEMP/sdist-output/review-help.txt"'
    ) in sdist
    for command in ("build-pack", "review", "status", "build-envelope"):
        assert (
            f'"$RUNNER_TEMP/sdist-venv/bin/saliencegate-review" {command} --help '
            '>> "$RUNNER_TEMP/sdist-output/review-help.txt"'
        ) in sdist
    assert (
        "scripts/smoke_launch_contracts.py "
        '"$RUNNER_TEMP/sdist-output/launch-demo.json" '
        '"$RUNNER_TEMP/sdist-output/review-help.txt"'
    ) in sdist
    _assert_installed_shadow_smoke(sdist, prefix="sdist")


def test_installed_shadow_smoke_is_private_core_only_and_self_validating() -> None:
    source = SHADOW_INSTALLED_SMOKE.read_text(encoding="utf-8")

    for optional_module in FORBIDDEN_CORE_MODULES:
        assert f'"{optional_module}"' in source
    assert "find_spec(optional_module)" in source
    assert "optional_module in sys.modules" in source
    assert "os.O_EXCL" in source
    assert "0o600" in source
    assert "os.fsync" in source
    assert "StableReadPolicy.PRIVATE_OWNER" in source
    assert "decode_shadow_run_report" in source
    assert "decode_shadow_trace_report" in source
    assert "canonical_json(report" in source
    assert "canonical_json(command_report)" in source
    assert "_EnvironmentReadGuard" in source
    assert "_PROVIDER_CREDENTIAL_KEYS" in source
    assert "_assert_network_is_denied" in source
    assert '"exercise-atif"' in source
    assert '"harbor-codex-v1"' in source
    assert '"harbor-terminus-2-v1"' in source
    for zero_field in (
        "model_calls",
        "budget_reservations",
        "cycles_created",
        "memory_revisions",
        "interventions",
        "delivery_authorizations",
        "deliveries",
        "intervention_outcomes",
    ):
        assert f'"{zero_field}"' in source


def test_ci_keeps_the_installed_model_runtime_smoke_config_only() -> None:
    text = _read(".github/workflows/ci.yml")
    runtime = _job_block(text, "installed-model-runtime")

    assert "needs:\n      - build" in runtime
    assert 'PYTHONPATH: ""' in runtime
    assert "actions/download-artifact@" in runtime
    assert (
        "uv export --locked --all-extras --no-dev --no-emit-project "
        '--output-file "$RUNNER_TEMP/model-runtime-requirements.txt" --quiet'
    ) in runtime
    assert '--require-hashes -r "$RUNNER_TEMP/model-runtime-requirements.txt"' in runtime
    assert "--no-deps dist/*.whl" in runtime
    assert (
        '"$RUNNER_TEMP/model-runtime-venv/bin/python" -I scripts/run_without_sockets.py' in runtime
    )
    assert runtime.count("scripts/run_without_sockets.py") == 1
    assert "scripts/smoke_model_runtime.py" in runtime
    smoke = MODEL_RUNTIME_SMOKE.read_text(encoding="utf-8")
    assert "from saliencegate.models.openai_compatible import OpenAICompatibleConfig" in smoke
    assert 'base_url="http://127.0.0.1:11434/v1"' in smoke
    assert 'model="gpt-oss:20b"' in smoke
    assert "import saliencegate.commands.pilot" in smoke
    assert "OpenAICompatibleClient" not in smoke
    assert "saliencegate pilot" not in runtime
    assert "/chat/completions" not in runtime


def test_ci_installed_jobs_share_only_the_built_distributions() -> None:
    text = _read(".github/workflows/ci.yml")
    installed = tuple(
        _job_block(text, job_id)
        for job_id in ("installed-core-wheel", "installed-sdist", "installed-model-runtime")
    )

    assert text.count("actions/download-artifact@") == 4
    for job in installed:
        assert "name: python-distributions" in job
        assert "path: dist/" in job
        assert "persist-credentials: false" in job
        for line in job.splitlines():
            if '/bin/saliencegate"' in line or '/bin/saliencegate-review"' in line:
                assert "scripts/run_without_sockets.py" in line


def test_socket_guard_allows_only_local_socket_pairs(tmp_path: Path) -> None:
    harmless = tmp_path / "harmless.py"
    harmless.write_text(
        "import _socket\nimport asyncio\nimport socket\n"
        "left, right = socket.socketpair()\nleft.close()\nright.close()\n"
        "left, right = _socket.socketpair()\nleft.close()\nright.close()\n"
        "asyncio.run(asyncio.sleep(0))\n"
        "print('offline-ok')\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        (sys.executable, "-I", str(SOCKET_GUARD), str(harmless)),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0
    assert completed.stdout == "offline-ok\n"
    assert completed.stderr == ""

    blocked_sources = {
        "public-ipv4": "import socket\nsocket.socket(socket.AF_INET)\n",
        "public-ipv6": "import socket\nsocket.socket(socket.AF_INET6)\n",
        "private-ipv4": "import _socket\n_socket.socket(_socket.AF_INET)\n",
        "private-ipv6": "import _socket\n_socket.socket(_socket.AF_INET6)\n",
        "public-resolver": "import socket\nsocket.getaddrinfo('localhost', 80)\n",
        "private-resolver": "import _socket\n_socket.getaddrinfo('localhost', 80)\n",
        "public-unix": "import socket\nsocket.socket(socket.AF_UNIX)\n",
        "private-unix": "import _socket\n_socket.socket(_socket.AF_UNIX)\n",
        "public-inherited-ipv4": (
            "import os\nimport socket\nread, write = os.pipe()\n"
            "socket.socket(socket.AF_INET, fileno=read)\n"
        ),
        "private-inherited-ipv4": (
            "import _socket\nimport os\nread, write = os.pipe()\n"
            "_socket.socket(_socket.AF_INET, fileno=read)\n"
        ),
    }
    for name, source in blocked_sources.items():
        blocked = tmp_path / f"blocked-{name}.py"
        blocked.write_text(source, encoding="utf-8")
        refused = subprocess.run(
            (sys.executable, "-I", str(SOCKET_GUARD), str(blocked)),
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert refused.returncode != 0, name
        assert "SocketAccessError" in refused.stderr, name
        assert "socket or resolver access is disabled for this smoke" in refused.stderr, name


def test_dependabot_and_collaboration_templates_require_reproducible_evidence() -> None:
    dependabot = _read(".github/dependabot.yml")
    assert 'package-ecosystem: "uv"' in dependabot
    assert 'package-ecosystem: "github-actions"' in dependabot
    assert dependabot.count('interval: "weekly"') == 2
    assert dependabot.count('directory: "/"') == 2

    bug = _read(".github/ISSUE_TEMPLATE/bug.yml")
    for field in ("reproduction", "expected", "actual", "environment"):
        assert field in bug.lower()
    assert "secret" in bug.lower()

    benchmark = _read(".github/ISSUE_TEMPLATE/benchmark.yml")
    required_benchmark_fields = (
        "evidence-kind",
        "model-runtime",
        "hardware",
        "configuration-digest",
        "seed",
        "artifact-digest",
        "reproduction-command",
        "results",
    )
    for field_id in required_benchmark_fields:
        field = _issue_field(benchmark, field_id)
        assert re.search(
            r"(?m)^    validations:\n      required: true$",
            field,
        )
    assert "external" in benchmark.lower() and "synthetic" in benchmark.lower()

    for field_id in (
        "reproduction",
        "expected",
        "actual",
        "environment",
    ):
        field = _issue_field(bug, field_id)
        assert re.search(r"(?m)^    validations:\n      required: true$", field)

    pull_request = _read(".github/pull_request_template.md").lower()
    assert "make check" in pull_request
    assert "secret" in pull_request
    assert "claim" in pull_request


def test_contributing_documents_the_same_exact_gate_order() -> None:
    text = CONTRIBUTING.read_text(encoding="utf-8")
    heading = "## Required gate order"
    assert heading in text
    section = text.split(heading, maxsplit=1)[1]
    positions = [section.index(f"`make {target}`") for target in REQUIRED_TARGETS[:-1]]
    assert positions == sorted(positions)
    assert "`make check` runs this sequence" in section
    first_sync = "uv sync --locked --all-extras --dev --no-install-project"
    second_sync = "uv sync --locked --all-extras --dev --no-build-isolation"
    assert first_sync in text
    assert second_sync in text
    assert text.index(first_sync) < text.index(second_sync)

    readme = README.read_text(encoding="utf-8")
    assert first_sync in readme
    assert second_sync in readme
