from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from scripts.check_public_tree import check_public_tree, find_path_violations

ROOT = Path(__file__).parents[1]
CHECKER = ROOT / "scripts" / "check_public_tree.py"


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ("git", *arguments),
        cwd=repo,
        check=True,
        capture_output=True,
    )


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


def test_real_repository_public_tree_is_clean() -> None:
    assert check_public_tree(ROOT) == ()


def test_safe_paths_are_accepted() -> None:
    assert (
        find_path_violations(
            (
                ".github/workflows/ci.yml",
                "docs/reference/cli.md",
                "src/saliencegate/cli.py",
                "tests/fixtures/café.json",
            )
        )
        == ()
    )


def test_unicode_and_case_collisions_are_reported_deterministically() -> None:
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
        "",
        "/absolute.md",
        "docs\\windows.md",
        "docs/../private.md",
        "docs/./private.md",
        "docs//empty.md",
        "docs/\ud800.md",
        ".hidden/state.json",
        ".GitHub/workflows/ci.yml",
    ),
)
def test_nonportable_paths_are_reported(tracked_path: str) -> None:
    assert find_path_violations((tracked_path,)) == (tracked_path,)


def test_tracked_files_ignored_by_versioned_rules_are_reported(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)

    for name in ("ignored.txt", "info-only.txt", "global-only.txt"):
        (repo / name).write_text(name, encoding="utf-8")
    _git(repo, "add", "ignored.txt", "info-only.txt", "global-only.txt")

    global_ignore = tmp_path / "global-ignore"
    global_ignore.write_text("/global-only.txt\n", encoding="utf-8")
    _git(repo, "config", "core.excludesFile", str(global_ignore))
    (repo / ".git" / "info" / "exclude").write_text("/info-only.txt\n", encoding="utf-8")
    (repo / ".gitignore").write_text("/ignored.txt\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")

    assert check_public_tree(repo) == ("ignored.txt",)

    result = _run_checker(repo)
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == '"ignored.txt"\n'


def test_non_regular_index_entries_are_reported(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    (repo / "README.md").write_text("# Project\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "--quiet", "-m", "Initial")

    blob = _git(repo, "rev-parse", "HEAD:README.md").stdout.strip().decode("ascii")
    commit = _git(repo, "rev-parse", "HEAD").stdout.strip().decode("ascii")
    _git(repo, "update-index", "--add", "--cacheinfo", f"120000,{blob},link.txt")
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{commit},nested-repo")

    assert check_public_tree(repo) == ("link.txt", "nested-repo")


def test_hidden_root_state_is_ignored_without_hiding_github() -> None:
    hidden = subprocess.run(
        ("git", "check-ignore", "--no-index", "--quiet", ".workspace/state.json"),
        cwd=ROOT,
        check=False,
    )
    github = subprocess.run(
        ("git", "check-ignore", "--no-index", "--quiet", ".github/workflows/ci.yml"),
        cwd=ROOT,
        check=False,
    )

    assert hidden.returncode == 0
    assert github.returncode == 1


def test_cli_passes_a_clean_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    (repo / "README.md").write_text("# Project\n", encoding="utf-8")
    _git(repo, "add", "README.md")

    result = _run_checker(repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "Public tree check passed\n"
    assert result.stderr == ""


def test_cli_propagates_git_failures(tmp_path: Path) -> None:
    result = _run_checker(tmp_path)

    assert result.returncode != 0
    assert "CalledProcessError" in result.stderr
