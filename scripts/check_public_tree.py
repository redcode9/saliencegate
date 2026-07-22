from __future__ import annotations

import json
import re
import subprocess
import sys
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_IGNORE_PATTERNS = (
    "/.agents/",
    "/.codex/",
    "/.claude/",
    "/.cursor/",
    "/.gemini/",
    "/.continue/",
    "/.windsurf/",
    "/.cline/",
    "/.roo/",
    "/.aider*",
    "/docs/superpowers/",
    "/AGENTS.md",
    "/CLAUDE.md",
    "/CODEX.md",
    "/GEMINI.md",
    "/.github/copilot-instructions.md",
    "/.github/agents/",
    "/.github/prompts/",
    "/.github/instructions/",
)

_WORKSPACE_ROOTS = frozenset(
    {
        ".agents",
        ".codex",
        ".claude",
        ".cursor",
        ".gemini",
        ".continue",
        ".windsurf",
        ".cline",
        ".roo",
    }
)
_ROOT_INSTRUCTION_FILES = frozenset({"agents.md", "claude.md", "codex.md", "gemini.md"})
_GITHUB_WORKSPACE_ROOTS = frozenset({"agents", "prompts", "instructions"})
_INTERNAL_MILESTONE_PATTERN = (
    r"(^|[^[:alnum:]])((stage|task)[[:space:]_-]*[0-9]+[[:alpha:]]?|m[0-9]{2})"
    r"([^[:alnum:]]|$)"
)
_INTERNAL_MILESTONE_TEXT_PATTERN = re.compile(
    r"(?i)(?<![a-z0-9])(?:"
    r"(?P<label>task|stage)[ _-]*(?P<number>[0-9]+)(?P<suffix>[a-z]?)"
    r"|(?P<milestone>m[0-9]{2})"
    r")(?![a-z0-9])"
)


def _normalize_component(component: str) -> str:
    return unicodedata.normalize("NFC", component).casefold()


def _normalize_path(path: str) -> tuple[str, ...]:
    return tuple(_normalize_component(component) for component in path.split("/"))


def _is_unsafe_public_path(path: str, normalized_parts: tuple[str, ...]) -> bool:
    return (
        not path
        or path.startswith("/")
        or "\\" in path
        or any(component in {"", ".", ".."} for component in normalized_parts)
        or any(0xD800 <= ord(character) <= 0xDFFF for character in path)
    )


def _is_forbidden_path(normalized_parts: tuple[str, ...]) -> bool:
    if not normalized_parts:
        return False

    root = normalized_parts[0]
    if root in _WORKSPACE_ROOTS:
        return True
    if root.startswith(".aider"):
        return True
    if len(normalized_parts) >= 2 and normalized_parts[:2] == ("docs", "superpowers"):
        return True
    if len(normalized_parts) == 1:
        return root in _ROOT_INSTRUCTION_FILES
    if root != ".github":
        return False

    github_name = normalized_parts[1]
    if github_name == "copilot-instructions.md":
        return True
    return github_name in _GITHUB_WORKSPACE_ROOTS


def _has_internal_milestone(text: str) -> bool:
    for match in _INTERNAL_MILESTONE_TEXT_PATTERN.finditer(text):
        if match.group("milestone") is not None:
            return True
        if (
            match.group("label").casefold() == "stage"
            and match.group("number") == "2"
            and not match.group("suffix")
        ):
            continue
        return True
    return False


def find_path_violations(tracked_paths: Iterable[str]) -> tuple[str, ...]:
    """Return forbidden and normalized-collision paths without consulting the filesystem."""
    normalized_paths: dict[tuple[str, ...], set[str]] = {}
    violations: set[str] = set()

    for path in tracked_paths:
        normalized = _normalize_path(path)
        normalized_paths.setdefault(normalized, set()).add(path)
        if (
            _is_unsafe_public_path(path, normalized)
            or _is_forbidden_path(normalized)
            or _has_internal_milestone(path)
        ):
            violations.add(path)

    for paths in normalized_paths.values():
        if len(paths) > 1:
            violations.update(paths)

    return tuple(sorted(violations))


def has_required_ignore_suffix(text: str) -> bool:
    """Return whether exact required patterns are the final effective ignore lines."""
    effective_lines = tuple(
        line for line in text.splitlines() if line.strip() and not line.startswith("#")
    )
    suffix_length = len(REQUIRED_IGNORE_PATTERNS)
    return effective_lines[-suffix_length:] == REQUIRED_IGNORE_PATTERNS


def _git_paths(root: Path, arguments: tuple[str, ...]) -> tuple[str, ...]:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
    )
    output = completed.stdout
    if output and not output.endswith(b"\0"):
        raise ValueError("Git path output was not NUL-terminated")
    return tuple(item.decode("utf-8") for item in output.split(b"\0") if item)


def git_tracked_paths(root: Path) -> tuple[str, ...]:
    return _git_paths(root, ("ls-files", "-z"))


def git_ignored_tracked_paths(root: Path) -> tuple[str, ...]:
    return _git_paths(
        root,
        ("ls-files", "-z", "-ci", "--exclude-per-directory=.gitignore"),
    )


def git_internal_milestone_paths(root: Path) -> tuple[str, ...]:
    """Return tracked text paths containing numbered internal milestone language."""
    completed = subprocess.run(
        (
            "git",
            "grep",
            "--cached",
            "-I",
            "-i",
            "-l",
            "-z",
            "-E",
            _INTERNAL_MILESTONE_PATTERN,
            "--",
            ".",
        ),
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode not in {0, 1}:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    if completed.stdout and not completed.stdout.endswith(b"\0"):
        raise ValueError("Git grep path output was not NUL-terminated")
    candidates = tuple(item.decode("utf-8") for item in completed.stdout.split(b"\0") if item)
    findings: list[str] = []
    for path in candidates:
        indexed = subprocess.run(
            ("git", "show", f":{path}"),
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        try:
            text = indexed.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if _has_internal_milestone(text):
            findings.append(path)
    return tuple(findings)


def _gitignore_candidates(tracked_paths: Iterable[str]) -> tuple[str, ...]:
    candidates = {".gitignore"}
    for tracked_path in tracked_paths:
        parts = PurePosixPath(tracked_path).parts
        for depth in range(1, len(parts)):
            candidates.add(PurePosixPath(*parts[:depth], ".gitignore").as_posix())
    return tuple(sorted(candidates))


def _gitignore_worktree_violations(root: Path, tracked_paths: tuple[str, ...]) -> tuple[str, ...]:
    tracked = frozenset(tracked_paths)
    violations: list[str] = []
    for relative_path in _gitignore_candidates(tracked_paths):
        path = root / relative_path
        if not path.exists() and not path.is_symlink():
            continue
        if relative_path not in tracked or path.is_symlink() or not path.is_file():
            violations.append(relative_path)
            continue
        indexed = subprocess.run(
            ("git", "show", f":{relative_path}"),
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        if path.read_bytes() != indexed:
            violations.append(relative_path)
    return tuple(violations)


def check_public_tree(root: Path = ROOT) -> tuple[str, ...]:
    tracked_paths = git_tracked_paths(root)
    violations = set(find_path_violations(tracked_paths))
    violations.update(_gitignore_worktree_violations(root, tracked_paths))
    violations.update(git_ignored_tracked_paths(root))
    violations.update(git_internal_milestone_paths(root))

    ignore_path = root / ".gitignore"
    if (
        ".gitignore" not in tracked_paths
        or not ignore_path.is_file()
        or not has_required_ignore_suffix(ignore_path.read_text(encoding="utf-8"))
    ):
        violations.add(".gitignore")

    return tuple(sorted(violations))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if len(arguments) > 1:
        print("usage: check_public_tree.py [ROOT]", file=sys.stderr)
        return 2

    root = Path(arguments[0]).resolve() if arguments else ROOT
    findings = check_public_tree(root)
    if findings:
        for finding in findings:
            print(json.dumps(finding, ensure_ascii=False), file=sys.stderr)
        return 1
    print("Public tree check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
