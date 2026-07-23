from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from scripts.check_public_tree import (
    MAX_PUBLIC_TEXT_BLOB_BYTES,
    PUBLIC_HISTORY_BASELINE,
    REQUIRED_IGNORE_PATTERNS,
    _parse_git_blob_batch,
    check_public_tree,
    find_path_violations,
    git_head_history_violations,
    has_required_ignore_suffix,
)

ROOT = Path(__file__).parents[1]
CHECKER = ROOT / "scripts" / "check_public_tree.py"


def test_real_repository_public_tree_is_clean() -> None:
    assert check_public_tree(ROOT, history_baseline=PUBLIC_HISTORY_BASELINE) == ()


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
        "tests/test_" + "M" + "11_coverage.py",
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


def _initialize_repo(repo: Path) -> str:
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Public Tree Tests")
    _git(repo, "commit", "--quiet", "--allow-empty", "-m", "Establish public root")
    return _git(repo, "rev-parse", "HEAD").stdout.decode("ascii").strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "commit", "--quiet", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.decode("ascii").strip()


def _history_baseline(repo: Path) -> str:
    commits = (
        _git(repo, "rev-list", "--max-parents=0", "--reverse", "HEAD")
        .stdout.decode("ascii")
        .splitlines()
    )
    if not commits:
        raise ValueError("Synthetic repository has no root commit")
    return commits[0]


def _synthetic_public_tree(repo: Path) -> tuple[str, ...]:
    return check_public_tree(repo, history_baseline=_history_baseline(repo))


def _synthetic_history_violations(repo: Path) -> tuple[str, ...]:
    return git_head_history_violations(repo, baseline=_history_baseline(repo))


def _run_checker(
    repo: Path,
    *,
    history_baseline: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if history_baseline is None:
        try:
            history_baseline = _history_baseline(repo)
        except subprocess.CalledProcessError:
            history_baseline = PUBLIC_HISTORY_BASELINE
    return subprocess.run(
        (
            sys.executable,
            str(CHECKER),
            str(repo),
            "--history-baseline",
            history_baseline,
        ),
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


def test_current_index_rejects_private_plan_content(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    _write_ignore_contract(repo)
    private_marker = ".saliencegate-" + "private/universal-shadow-capture-" + "goal-plan.md"
    (repo / "notes.md").write_text(f"Private source: {private_marker}\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "notes.md")

    assert _synthetic_public_tree(repo) == ("notes.md",)
    result = _run_checker(repo)
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == '"notes.md"\n'


@pytest.mark.parametrize(
    "attribution",
    (
        "AI-" + "generated release notes",
        "Co-" + "authored-by: Synthetic Author <author@example.invalid>",
    ),
)
def test_current_index_rejects_attribution_content(
    tmp_path: Path,
    attribution: str,
) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    _write_ignore_contract(repo)
    (repo / "notes.md").write_text(f"{attribution}\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "notes.md")

    assert _synthetic_public_tree(repo) == ("notes.md",)


@pytest.mark.parametrize(
    "payload",
    (
        pytest.param(b"text before NUL\0text after NUL", id="nul"),
        pytest.param(b"invalid UTF-8: \xff", id="invalid-utf8"),
        pytest.param(b"x" * (MAX_PUBLIC_TEXT_BLOB_BYTES + 1), id="oversize"),
    ),
)
def test_current_index_rejects_unscannable_or_oversize_blobs(
    tmp_path: Path,
    payload: bytes,
) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    _write_ignore_contract(repo)
    (repo / "opaque.bin").write_bytes(payload)
    _git(repo, "add", ".gitignore", "opaque.bin")

    assert _synthetic_public_tree(repo) == ("opaque.bin",)


def test_current_index_rejects_a_symlink_entry_without_following_its_target(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    _write_ignore_contract(repo)
    target = repo / "symlink-target-blob"
    target.write_text(".codex/private.md", encoding="utf-8")
    object_id = _git(repo, "hash-object", "-w", "symlink-target-blob").stdout.decode().strip()
    target.unlink()
    _git(repo, "add", ".gitignore")
    _git(repo, "update-index", "--add", "--cacheinfo", "120000", object_id, "notes.md")

    assert _synthetic_public_tree(repo) == ("notes.md",)


def test_current_index_rejects_a_gitlink_entry(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    commit = _initialize_repo(repo)
    _write_ignore_contract(repo)
    _git(repo, "add", ".gitignore")
    _git(repo, "update-index", "--add", "--cacheinfo", "160000", commit, "dependency")

    assert _synthetic_public_tree(repo) == ("dependency",)


@pytest.mark.parametrize(
    "forbidden_content",
    (
        "AI-" + "generated rejection fixture",
        "Co-" + "authored-by: Synthetic Author <author@example.invalid>",
        ".saliencegate-" + "private/internal-state.json",
        "Internal " + "M" + "12 plan",
    ),
)
def test_rejection_fixture_paths_receive_the_full_content_scan(
    tmp_path: Path,
    forbidden_content: str,
) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    _write_ignore_contract(repo)
    fixture = repo / "tests" / "test_public_tree.py"
    fixture.parent.mkdir()
    fixture.write_text(f"{forbidden_content}\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "tests/test_public_tree.py")

    assert _synthetic_public_tree(repo) == ("tests/test_public_tree.py",)


@pytest.mark.parametrize(
    "payload",
    (
        b"a" * 40 + b" blob 4\ntext",
        b"a" * 40 + b" blob 4\ntext\ntrailing",
        b"b" * 40 + b" blob 4\ntext\n",
        b"a" * 40 + b" tree 4\ntext\n",
        b"a" * 40 + b" blob 3\ntext\n",
    ),
)
def test_git_blob_batch_parser_fails_closed_on_protocol_mismatch(payload: bytes) -> None:
    object_id = "a" * 40
    with pytest.raises(ValueError, match="Git blob batch response"):
        _parse_git_blob_batch(payload, (object_id,), {object_id: 4})


def test_git_blob_batch_parser_accepts_exact_binary_framing() -> None:
    object_id = "a" * 40
    payload = object_id.encode("ascii") + b" blob 4\ntext\n"

    assert _parse_git_blob_batch(payload, (object_id,), {object_id: 4}) == {object_id: b"text"}


def test_head_history_rejects_a_forbidden_path_after_it_was_deleted(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    _write_ignore_contract(repo)
    (repo / "README.md").write_text("# Public\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "README.md")
    _commit(repo, "Start public tree")

    forbidden = repo / ".agents" / "review.md"
    forbidden.parent.mkdir()
    forbidden.write_text("private\n", encoding="utf-8")
    _git(repo, "add", "-f", ".agents/review.md")
    _commit(repo, "Add temporary review")
    forbidden.unlink()
    forbidden.parent.rmdir()
    _git(repo, "add", "-u")
    _commit(repo, "Remove temporary review")

    assert _synthetic_history_violations(repo) == ("history:path:.agents/review.md",)
    result = _run_checker(repo)
    assert result.returncode == 1
    assert result.stderr == '"history:path:.agents/review.md"\n'


@pytest.mark.parametrize(
    "forbidden_content",
    (
        ".saliencegate-" + "private/universal-shadow-capture-" + "progress.json",
        "Internal " + "M" + "12 plan",
        "AI-" + "generated release notes",
        "Co-" + "authored-by: Synthetic Author <author@example.invalid>",
    ),
)
def test_head_history_rejects_forbidden_content_after_the_blob_was_deleted(
    tmp_path: Path,
    forbidden_content: str,
) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    _write_ignore_contract(repo)
    (repo / "README.md").write_text("# Public\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "README.md")
    _commit(repo, "Start public tree")

    notes = repo / "notes.md"
    notes.write_text(f"{forbidden_content}\n", encoding="utf-8")
    _git(repo, "add", "notes.md")
    _commit(repo, "Add temporary notes")
    notes.unlink()
    _git(repo, "add", "-u")
    _commit(repo, "Remove temporary notes")

    assert _synthetic_history_violations(repo) == ("history:content:notes.md",)


def test_head_history_reaudits_a_baseline_blob_copied_to_a_new_path(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    _write_ignore_contract(repo)
    fixture = repo / "tests" / "fixtures" / "rejection.txt"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("Co-" + "authored-by: Synthetic Author\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "tests/fixtures/rejection.txt")
    baseline = _commit(repo, "Establish synthetic rejection fixture")
    object_id = (
        _git(repo, "rev-parse", f"{baseline}:tests/fixtures/rejection.txt")
        .stdout.decode("ascii")
        .strip()
    )

    assert git_head_history_violations(repo, baseline=baseline) == ()

    _git(repo, "update-index", "--add", "--cacheinfo", "100644", object_id, "public-copy.txt")
    _commit(repo, "Copy fixture bytes")

    assert git_head_history_violations(repo, baseline=baseline) == (
        "history:content:public-copy.txt",
    )


def test_head_history_rejects_a_deleted_symlink_entry(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    _write_ignore_contract(repo)
    _git(repo, "add", ".gitignore")
    _commit(repo, "Start public tree")

    target = repo / "symlink-target-blob"
    target.write_text(".codex/private.md", encoding="utf-8")
    object_id = _git(repo, "hash-object", "-w", "symlink-target-blob").stdout.decode().strip()
    target.unlink()
    _git(repo, "update-index", "--add", "--cacheinfo", "120000", object_id, "notes.md")
    _commit(repo, "Add temporary link")
    _git(repo, "update-index", "--force-remove", "notes.md")
    _commit(repo, "Remove temporary link")

    assert _synthetic_history_violations(repo) == ("history:content:notes.md",)


def test_head_history_rejects_a_deleted_gitlink_entry(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    dependency_commit = _initialize_repo(repo)
    _write_ignore_contract(repo)
    _git(repo, "add", ".gitignore")
    _commit(repo, "Add publication ignore contract")

    _git(
        repo,
        "update-index",
        "--add",
        "--cacheinfo",
        "160000",
        dependency_commit,
        "dependency",
    )
    _commit(repo, "Add temporary dependency")
    _git(repo, "update-index", "--force-remove", "dependency")
    _commit(repo, "Remove temporary dependency")

    assert _synthetic_history_violations(repo) == ("history:content:dependency",)


def test_head_history_rejects_a_collision_with_a_preexisting_path(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    _write_ignore_contract(repo)
    (repo / "Report.md").write_text("first\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "Report.md")
    _commit(repo, "Add first report")
    blob = repo / "second-report-blob"
    blob.write_text("second\n", encoding="utf-8")
    object_id = _git(repo, "hash-object", "-w", "second-report-blob").stdout.decode().strip()
    blob.unlink()
    _git(repo, "update-index", "--add", "--cacheinfo", "100644", object_id, "report.md")
    _commit(repo, "Add colliding report")

    assert _synthetic_history_violations(repo) == (
        "history:path:Report.md",
        "history:path:report.md",
    )


def test_head_history_audits_private_content_reachable_only_through_second_parent(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    _write_ignore_contract(repo)
    _git(repo, "add", ".gitignore")
    base = _commit(repo, "Start public tree")
    base_tree = _git(repo, "rev-parse", f"{base}^{{tree}}").stdout.decode().strip()

    private_marker = ".saliencegate-" + "private/internal-state.json\n"
    blob = repo / "private-notes-blob"
    blob.write_text(private_marker, encoding="utf-8")
    object_id = _git(repo, "hash-object", "-w", "private-notes-blob").stdout.decode().strip()
    blob.unlink()

    _git(repo, "read-tree", base_tree)
    _git(repo, "update-index", "--add", "--cacheinfo", "100644", object_id, "notes.md")
    private_tree = _git(repo, "write-tree").stdout.decode().strip()
    private_parent = (
        _git(
            repo,
            "commit-tree",
            private_tree,
            "-p",
            base,
            "-m",
            "Add temporary notes",
        )
        .stdout.decode()
        .strip()
    )
    clean_parent = (
        _git(
            repo,
            "commit-tree",
            base_tree,
            "-p",
            base,
            "-m",
            "Keep public tree clean",
        )
        .stdout.decode()
        .strip()
    )
    merge = (
        _git(
            repo,
            "commit-tree",
            base_tree,
            "-p",
            clean_parent,
            "-p",
            private_parent,
            "-m",
            "Merge public branches",
        )
        .stdout.decode()
        .strip()
    )
    _git(repo, "update-ref", "HEAD", merge)

    assert _synthetic_history_violations(repo) == ("history:content:notes.md",)


def test_head_history_audits_case_collisions_reachable_through_a_merge_parent(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    _write_ignore_contract(repo)
    _git(repo, "add", ".gitignore")
    base = _commit(repo, "Start public tree")
    base_tree = _git(repo, "rev-parse", f"{base}^{{tree}}").stdout.decode().strip()

    object_ids: list[str] = []
    for name, content in (("first-report-blob", "first\n"), ("second-report-blob", "second\n")):
        blob = repo / name
        blob.write_text(content, encoding="utf-8")
        object_ids.append(_git(repo, "hash-object", "-w", name).stdout.decode().strip())
        blob.unlink()

    _git(repo, "read-tree", base_tree)
    _git(repo, "update-index", "--add", "--cacheinfo", "100644", object_ids[0], "Report.md")
    _git(repo, "update-index", "--add", "--cacheinfo", "100644", object_ids[1], "report.md")
    collision_tree = _git(repo, "write-tree").stdout.decode().strip()
    collision_parent = (
        _git(
            repo,
            "commit-tree",
            collision_tree,
            "-p",
            base,
            "-m",
            "Add colliding reports",
        )
        .stdout.decode()
        .strip()
    )
    clean_parent = (
        _git(
            repo,
            "commit-tree",
            base_tree,
            "-p",
            base,
            "-m",
            "Keep public tree clean",
        )
        .stdout.decode()
        .strip()
    )
    merge = (
        _git(
            repo,
            "commit-tree",
            base_tree,
            "-p",
            clean_parent,
            "-p",
            collision_parent,
            "-m",
            "Merge public branches",
        )
        .stdout.decode()
        .strip()
    )
    _git(repo, "update-ref", "HEAD", merge)

    assert _synthetic_history_violations(repo) == (
        "history:path:Report.md",
        "history:path:report.md",
    )


@pytest.mark.parametrize(
    "message",
    (
        "Public change\n\nCo-" + "authored-by: Synthetic Author <author@example.invalid>",
        "Internal " + "M" + "12 preparation",
        "AI-" + "generated release notes",
    ),
)
def test_head_history_rejects_forbidden_commit_metadata(
    tmp_path: Path,
    message: str,
) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    _write_ignore_contract(repo)
    _git(repo, "add", ".gitignore")
    commit = _commit(repo, message)

    assert _synthetic_history_violations(repo) == (f"history:commit:{commit}",)


def test_head_history_ignores_unrelated_refs_and_accepts_clean_ancestry(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    _write_ignore_contract(repo)
    (repo / "README.md").write_text("# Public\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "README.md")
    _commit(repo, "Start public tree")
    _git(repo, "branch", "clean")
    _git(repo, "switch", "--quiet", "-c", "unrelated")
    forbidden = repo / ".codex" / "state.json"
    forbidden.parent.mkdir()
    forbidden.write_text("{}\n", encoding="utf-8")
    _git(repo, "add", "-f", ".codex/state.json")
    _commit(repo, "Temporary off-branch state")
    _git(repo, "switch", "--quiet", "clean")

    assert _synthetic_history_violations(repo) == ()
    result = _run_checker(repo)
    assert result.returncode == 0, result.stderr


def test_head_history_rejects_a_missing_baseline(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    _initialize_repo(repo)
    _write_ignore_contract(repo)
    _git(repo, "add", ".gitignore")
    _commit(repo, "Add publication ignore contract")
    missing = "f" * 40

    assert git_head_history_violations(repo, baseline=missing) == ("history:missing-baseline",)
    result = _run_checker(repo, history_baseline=missing)
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == '"history:missing-baseline"\n'


def test_head_history_rejects_a_baseline_outside_head_ancestry(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    baseline = _initialize_repo(repo)
    _write_ignore_contract(repo)
    _git(repo, "add", ".gitignore")
    _commit(repo, "Add publication ignore contract")
    tree = _git(repo, "rev-parse", "HEAD^{tree}").stdout.decode("ascii").strip()
    unrelated = (
        _git(repo, "commit-tree", tree, "-m", "Create unrelated public root")
        .stdout.decode("ascii")
        .strip()
    )
    _git(repo, "update-ref", "HEAD", unrelated)

    assert git_head_history_violations(repo, baseline=baseline) == (
        "history:baseline-not-ancestor",
    )
    result = _run_checker(repo, history_baseline=baseline)
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == '"history:baseline-not-ancestor"\n'


def test_head_history_rejects_a_shallow_clone(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    baseline = _initialize_repo(source)
    _write_ignore_contract(source)
    (source / "README.md").write_text("# First\n", encoding="utf-8")
    _git(source, "add", ".gitignore", "README.md")
    _commit(source, "First public revision")
    (source / "README.md").write_text("# Second\n", encoding="utf-8")
    _git(source, "add", "README.md")
    _commit(source, "Second public revision")

    clone = tmp_path / "clone"
    subprocess.run(
        ("git", "clone", "--quiet", "--depth", "1", source.as_uri(), str(clone)),
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    assert git_head_history_violations(clone, baseline=baseline) == ("history:shallow-clone",)
    result = _run_checker(clone, history_baseline=baseline)
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == '"history:shallow-clone"\n'


@pytest.mark.parametrize(
    "milestone",
    (
        "Internal " + "Stage " + "7 plan.\n",
        "Internal " + "Task_" + "6A plan.\n",
        "Internal " + "task" + "8 plan.\n",
        "Internal " + "stage" + "3a plan.\n",
        "Internal " + "M" + "11 plan.\n",
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
