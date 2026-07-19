from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from scripts.check_public_tree import (
    REQUIRED_IGNORE_PATTERNS,
    check_public_tree,
    find_path_violations,
    has_required_ignore_suffix,
)

ROOT = Path(__file__).parents[1]
CHECKER = ROOT / "scripts" / "check_public_tree.py"


def test_real_repository_public_tree_is_clean() -> None:
    assert check_public_tree(ROOT) == ()


@pytest.mark.parametrize(
    "tracked_path",
    (
        ".agents/settings.json",
        ".CODEX/config.toml",
        ".claude",
        ".cursor/rules/public.mdc",
        ".gemini/settings.json",
        ".continue/config.yaml",
        ".windsurf/rules.md",
        ".cline/config.json",
        ".roo/rules.md",
        "DOCS/SUPERPOWERS/spec.md",
        "AGENTS.md",
        "claude.MD",
        "CODEX.md",
        "gemini.md",
        ".Aider.conf.yml",
        ".aider/cache/state.json",
        ".github/COPILOT-INSTRUCTIONS.md",
        ".github/copilot-instructions.md/archive",
        ".GITHUB/agents/reviewer.md",
        ".github/PROMPTS/release.md",
        ".github/instructions/python.instructions.md",
    ),
)
def test_pure_validator_rejects_only_targeted_workspace_paths(tracked_path: str) -> None:
    assert find_path_violations((tracked_path,)) == (tracked_path,)


def test_pure_validator_allows_legitimate_product_and_fixture_paths() -> None:
    paths = (
        "src/saliencegate/prompts/paper_two_phase_v1.py",
        "tests/fixtures/prompts/paper_two_phase_v1.json",
        "tests/fixtures/pilots/stage_2_cases.json",
        "tests/fixtures/shadow/atif/codex-bundled-synthetic.trajectory.json",
        "docs/reference/agents.md",
        "examples/AGENTS.md",
        ".github/workflows/ci.yml",
    )

    assert find_path_violations(paths) == ()


def test_pure_validator_reports_every_nfc_casefold_collision_deterministically() -> None:
    paths = (
        "docs/Caf\N{LATIN SMALL LETTER E WITH ACUTE}.md",
        "src/Thing.py",
        "docs/Cafe\N{COMBINING ACUTE ACCENT}.md",
        "SRC/thing.py",
        "README.md",
    )

    assert find_path_violations(reversed(paths)) == (
        "SRC/thing.py",
        "docs/Cafe\N{COMBINING ACUTE ACCENT}.md",
        "docs/Caf\N{LATIN SMALL LETTER E WITH ACUTE}.md",
        "src/Thing.py",
    )


@pytest.mark.parametrize(
    "tracked_path",
    (
        "tests/test_" + "task_" + "6a_authority.py",
        "tests/test_" + "task" + "8_executor.py",
        "tests/test_" + "stage" + "1_compatibility.py",
        "tests/test_" + "stage" + "3a_compatibility.py",
    ),
)
def test_pure_validator_rejects_internal_milestones_in_paths(tracked_path: str) -> None:
    assert find_path_violations((tracked_path,)) == (tracked_path,)


@pytest.mark.parametrize(
    "tracked_path",
    ("/absolute.md", "docs\\windows.md", "docs/../private.md", "docs//empty.md"),
)
def test_pure_validator_rejects_nonportable_or_unsafe_paths(tracked_path: str) -> None:
    assert find_path_violations((tracked_path,)) == (tracked_path,)


def test_ignore_contract_requires_the_exact_final_non_comment_block() -> None:
    required = "\n".join(REQUIRED_IGNORE_PATTERNS)

    assert has_required_ignore_suffix(f"# Existing rules\n*.pyc\n\n{required}\n")
    assert has_required_ignore_suffix(f"{required}\n\n# This trailing comment is harmless.\n")
    assert not has_required_ignore_suffix(f"{required}\n!/.agents/keep.json\n")
    assert not has_required_ignore_suffix(required.replace("/.agents/", ".agents/", 1))
    assert not has_required_ignore_suffix(required.replace("/.codex/\n", "", 1))


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ("git", *arguments),
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _write_ignore_contract(repo: Path, *earlier_patterns: str) -> None:
    lines = (*earlier_patterns, "", "# Local agent workspaces", *REQUIRED_IGNORE_PATTERNS)
    (repo / ".gitignore").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _initialize_repo(repo: Path) -> None:
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Public Tree Tests")


def _run_checker(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, str(CHECKER), str(repo)),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_uses_nul_delimited_git_paths_and_prints_stable_path_only_findings(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    forbidden = repo / ".agents" / "line\nbreak.json"
    forbidden.parent.mkdir()
    forbidden.write_text("private\n", encoding="utf-8")
    _git(repo, "add", ".agents")
    _write_ignore_contract(repo)
    _git(repo, "add", ".gitignore")

    result = _run_checker(repo)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == '".agents/line\\nbreak.json"\n'


def test_cli_ignored_tracked_check_uses_only_versioned_gitignore_files(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    for name in ("repo-only.txt", "info-only.txt", "global-only.txt"):
        (repo / name).write_text(name, encoding="utf-8")
    _git(repo, "add", "repo-only.txt", "info-only.txt", "global-only.txt")
    nested = repo / "nested"
    nested.mkdir()
    (nested / "per-directory.txt").write_text("nested", encoding="utf-8")
    _git(repo, "add", "nested/per-directory.txt")

    global_ignore = tmp_path / "global-ignore"
    global_ignore.write_text("/global-only.txt\n", encoding="utf-8")
    _git(repo, "config", "core.excludesFile", str(global_ignore))
    (repo / ".git" / "info" / "exclude").write_text("/info-only.txt\n", encoding="utf-8")
    _write_ignore_contract(repo, "/repo-only.txt")
    (nested / ".gitignore").write_text("/per-directory.txt\n", encoding="utf-8")
    _git(repo, "add", "-f", ".gitignore")
    _git(repo, "add", "nested/.gitignore")

    result = _run_checker(repo)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == '"nested/per-directory.txt"\n"repo-only.txt"\n'


def test_cli_rejects_an_untracked_per_directory_ignore_that_changes_git_semantics(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    nested = repo / "nested"
    nested.mkdir()
    (nested / "tracked.txt").write_text("tracked", encoding="utf-8")
    _write_ignore_contract(repo, "/nested/tracked.txt")
    _git(repo, "add", "-f", ".gitignore", "nested/tracked.txt")
    (nested / ".gitignore").write_text("!/tracked.txt\n", encoding="utf-8")

    result = _run_checker(repo)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == '"nested/.gitignore"\n'


def test_cli_rejects_a_versioned_gitignore_changed_after_it_entered_the_index(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    _write_ignore_contract(repo)
    _git(repo, "add", ".gitignore")
    (repo / ".gitignore").write_text("*.tmp\n", encoding="utf-8")

    result = _run_checker(repo)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == '".gitignore"\n'


def test_cli_passes_a_synthetic_public_tree(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    (repo / "README.md").write_text("# Public\n", encoding="utf-8")
    _write_ignore_contract(repo)
    _git(repo, "add", "README.md", ".gitignore")

    result = _run_checker(repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "Public tree check passed\n"
    assert result.stderr == ""


@pytest.mark.parametrize(
    "milestone",
    (
        "Internal " + "Stage " + "7 plan.\n",
        "Internal " + "Task_" + "6A plan.\n",
        "Internal " + "task" + "8 plan.\n",
        "Internal " + "stage" + "3a plan.\n",
    ),
)
def test_cli_rejects_numbered_internal_milestone_language(
    tmp_path: Path,
    milestone: str,
) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    (repo / "README.md").write_text(f"# Public\n\n{milestone}", encoding="utf-8")
    _write_ignore_contract(repo)
    _git(repo, "add", "README.md", ".gitignore")

    result = _run_checker(repo)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == '"README.md"\n'


def test_cli_propagates_git_failures(tmp_path: Path) -> None:
    result = _run_checker(tmp_path)

    assert result.returncode != 0
    assert "CalledProcessError" in result.stderr
