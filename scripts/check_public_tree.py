from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_HISTORY_BASELINE = "e3cfcf2f2c61d3917457d168733908a3adfbda41"

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
_INTERNAL_MILESTONE_TEXT_PATTERN = re.compile(
    r"(?i)(?<![a-z0-9])(?:"
    r"(?P<label>task|stage)[ _-]*(?P<number>[0-9]+)(?P<suffix>[a-z]?)"
    r"|(?P<milestone>m[0-9]{2})"
    r")(?![a-z0-9])"
)
_ATTRIBUTION_TRAILER_LABELS = (
    "ai-" + "assisted-" + "by",
    "co-" + "authored-by",
    "generated-" + "by",
    "prompted-" + "by",
)
_FORBIDDEN_COMMIT_TRAILER_PATTERN = re.compile(
    r"(?im)^(?:"
    + "|".join(re.escape(label) for label in _ATTRIBUTION_TRAILER_LABELS)
    + r"):[ \t]*\S"
)
_FORBIDDEN_COMMIT_PHRASE_PATTERN = re.compile(
    r"(?i)(?<![a-z0-9])(?:"
    r"ai[ -](?:assisted|generated)"
    r"|(?:assisted|generated|prompted)[ -]by[ -]ai"
    r"|co[ -]?author(?:ed|ing|ship)?"
    r")(?![a-z0-9])"
)
_PRIVATE_CONTENT_PATTERN = re.compile(
    r"(?i)(?:"
    r"\.saliencegate[ _-]+private"
    r"|universal[ _-]+shadow[ _-]+capture[ _-]+(?:goal[ _-]+plan|progress)"
    r")"
)
_REGULAR_BLOB_MODES = frozenset({"100644", "100755"})
_OBJECT_ID_PATTERN = re.compile(r"[0-9a-f]{40,64}\Z")
MAX_PUBLIC_TEXT_BLOB_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class _GitBlobReference:
    path: str
    mode: str
    object_id: str
    is_regular_blob: bool


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


def _has_forbidden_attribution(text: str) -> bool:
    return (
        _FORBIDDEN_COMMIT_TRAILER_PATTERN.search(text) is not None
        or _FORBIDDEN_COMMIT_PHRASE_PATTERN.search(text) is not None
    )


def _has_forbidden_commit_metadata(message: str) -> bool:
    return _has_internal_milestone(message) or _has_forbidden_attribution(message)


def _has_forbidden_public_content(text: str) -> bool:
    normalized = unicodedata.normalize("NFC", text)
    return (
        _has_internal_milestone(normalized)
        or _PRIVATE_CONTENT_PATTERN.search(normalized) is not None
        or _has_forbidden_attribution(normalized)
    )


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


def _decode_object_id(value: bytes) -> str | None:
    try:
        object_id = value.decode("ascii")
    except UnicodeDecodeError:
        return None
    return object_id if _OBJECT_ID_PATTERN.fullmatch(object_id) is not None else None


def _parse_tree_blob_references(output: bytes) -> tuple[_GitBlobReference, ...]:
    if output and not output.endswith(b"\0"):
        raise ValueError("Git tree output was not NUL-terminated")
    references: list[_GitBlobReference] = []
    for record in output.split(b"\0"):
        if not record:
            continue
        try:
            header, encoded_path = record.split(b"\t", maxsplit=1)
        except ValueError as error:
            raise ValueError("Git tree entry lacked a path delimiter") from error
        fields = header.split()
        if len(fields) != 3:
            raise ValueError("Git tree entry had an invalid header")
        mode_raw, type_raw, object_id_raw = fields
        path = encoded_path.decode("utf-8")
        try:
            mode = mode_raw.decode("ascii")
            object_type = type_raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("Git tree entry metadata was not ASCII") from error
        object_id = _decode_object_id(object_id_raw)
        if object_id is None:
            raise ValueError("Git tree entry had an invalid object identifier")
        references.append(
            _GitBlobReference(
                path=path,
                mode=mode,
                object_id=object_id,
                is_regular_blob=object_type == "blob" and mode in _REGULAR_BLOB_MODES,
            )
        )
    return tuple(references)


def _git_tree_blob_references(root: Path, commit: str) -> tuple[_GitBlobReference, ...]:
    completed = subprocess.run(
        ("git", "ls-tree", "-r", "--full-tree", "-z", commit),
        cwd=root,
        check=True,
        capture_output=True,
    )
    return _parse_tree_blob_references(completed.stdout)


def _parse_index_blob_references(output: bytes) -> tuple[_GitBlobReference, ...]:
    if output and not output.endswith(b"\0"):
        raise ValueError("Git index output was not NUL-terminated")
    references: list[_GitBlobReference] = []
    for record in output.split(b"\0"):
        if not record:
            continue
        try:
            header, encoded_path = record.split(b"\t", maxsplit=1)
        except ValueError as error:
            raise ValueError("Git index entry lacked a path delimiter") from error
        fields = header.split()
        if len(fields) != 3:
            raise ValueError("Git index entry had an invalid header")
        mode_raw, object_id_raw, stage_raw = fields
        path = encoded_path.decode("utf-8")
        try:
            mode = mode_raw.decode("ascii")
            stage = stage_raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("Git index entry metadata was not ASCII") from error
        object_id = _decode_object_id(object_id_raw)
        if object_id is None:
            raise ValueError("Git index entry had an invalid object identifier")
        references.append(
            _GitBlobReference(
                path=path,
                mode=mode,
                object_id=object_id,
                is_regular_blob=stage == "0" and mode in _REGULAR_BLOB_MODES,
            )
        )
    return tuple(references)


def _git_index_blob_references(root: Path) -> tuple[_GitBlobReference, ...]:
    completed = subprocess.run(
        ("git", "ls-files", "--stage", "-z"),
        cwd=root,
        check=True,
        capture_output=True,
    )
    return _parse_index_blob_references(completed.stdout)


def _git_blob_sizes(root: Path, object_ids: tuple[str, ...]) -> dict[str, int]:
    if not object_ids:
        return {}
    completed = subprocess.run(
        (
            "git",
            "cat-file",
            "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        ),
        cwd=root,
        check=True,
        capture_output=True,
        input=("\n".join(object_ids) + "\n").encode("ascii"),
    )
    lines = completed.stdout.splitlines()
    if len(lines) != len(object_ids):
        raise ValueError("Git batch metadata response count did not match the request")
    sizes: dict[str, int] = {}
    for expected_object_id, line in zip(object_ids, lines, strict=True):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError("Git batch metadata response had an invalid header")
        object_id_raw, type_raw, size_raw = fields
        object_id = _decode_object_id(object_id_raw)
        try:
            object_type = type_raw.decode("ascii")
            size = int(size_raw.decode("ascii"))
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("Git batch metadata response was invalid") from error
        if object_id != expected_object_id or object_type != "blob" or size < 0:
            raise ValueError("Git batch metadata response did not identify the requested blob")
        sizes[object_id] = size
    return sizes


def _parse_git_blob_batch(
    output: bytes,
    object_ids: tuple[str, ...],
    expected_sizes: dict[str, int],
) -> dict[str, bytes]:
    offset = 0
    payloads: dict[str, bytes] = {}
    for expected_object_id in object_ids:
        header_end = output.find(b"\n", offset)
        if header_end < 0:
            raise ValueError("Git blob batch response lacked a complete header")
        fields = output[offset:header_end].split()
        if len(fields) != 3:
            raise ValueError("Git blob batch response had an invalid header")
        object_id_raw, type_raw, size_raw = fields
        object_id = _decode_object_id(object_id_raw)
        try:
            object_type = type_raw.decode("ascii")
            size = int(size_raw.decode("ascii"))
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("Git blob batch response was invalid") from error
        if (
            object_id != expected_object_id
            or object_type != "blob"
            or size != expected_sizes.get(expected_object_id)
        ):
            raise ValueError("Git blob batch response did not match its metadata")
        payload_start = header_end + 1
        payload_end = payload_start + size
        if payload_end >= len(output) or output[payload_end : payload_end + 1] != b"\n":
            raise ValueError("Git blob batch response was truncated")
        payloads[expected_object_id] = output[payload_start:payload_end]
        offset = payload_end + 1
    if offset != len(output):
        raise ValueError("Git blob batch response contained trailing data")
    return payloads


def _git_text_blobs(root: Path, object_ids: Iterable[str]) -> dict[str, str | None]:
    requested = tuple(sorted(set(object_ids)))
    sizes = _git_blob_sizes(root, requested)
    eligible = tuple(
        object_id for object_id in requested if sizes[object_id] <= MAX_PUBLIC_TEXT_BLOB_BYTES
    )
    payloads: dict[str, bytes] = {}
    if eligible:
        completed = subprocess.run(
            ("git", "cat-file", "--batch"),
            cwd=root,
            check=True,
            capture_output=True,
            input=("\n".join(eligible) + "\n").encode("ascii"),
        )
        payloads = _parse_git_blob_batch(completed.stdout, eligible, sizes)

    texts: dict[str, str | None] = {}
    for object_id in requested:
        payload = payloads.get(object_id)
        if payload is None or b"\0" in payload:
            texts[object_id] = None
            continue
        try:
            texts[object_id] = payload.decode("utf-8")
        except UnicodeDecodeError:
            texts[object_id] = None
    return texts


def _content_violating_paths(
    root: Path,
    references: Iterable[_GitBlobReference],
) -> tuple[str, ...]:
    materialized = tuple(references)
    texts = _git_text_blobs(
        root,
        (reference.object_id for reference in materialized if reference.is_regular_blob),
    )
    violations: set[str] = set()
    for reference in materialized:
        if not reference.is_regular_blob:
            violations.add(reference.path)
            continue
        text = texts[reference.object_id]
        if text is None or _has_forbidden_public_content(text):
            violations.add(reference.path)
    return tuple(sorted(violations))


def git_index_content_violations(root: Path) -> tuple[str, ...]:
    """Return index paths whose blob is forbidden or cannot be audited as bounded UTF-8."""
    return _content_violating_paths(root, _git_index_blob_references(root))


def _resolve_commit(root: Path, revision: str) -> str | None:
    completed = subprocess.run(
        ("git", "rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}"),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 1:
        return None
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    object_id = completed.stdout.strip()
    if _OBJECT_ID_PATTERN.fullmatch(object_id) is None:
        raise ValueError("Git resolved an invalid commit object identifier")
    return object_id


def _branch_commits(root: Path, head: str, baseline: str) -> tuple[str, ...]:
    completed = subprocess.run(
        ("git", "rev-list", head, f"^{baseline}"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    commits = tuple(completed.stdout.splitlines())
    if any(re.fullmatch(r"[0-9a-f]{40,64}", commit) is None for commit in commits):
        raise ValueError("Git history returned an invalid object identifier")
    return commits


def _commit_parents(root: Path, commit: str) -> tuple[str, ...]:
    completed = subprocess.run(
        ("git", "rev-list", "--parents", "-1", commit),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    fields = completed.stdout.split()
    if not fields or fields[0] != commit:
        raise ValueError("Git history returned invalid commit parent metadata")
    parents = tuple(fields[1:])
    if any(_OBJECT_ID_PATTERN.fullmatch(parent) is None for parent in parents):
        raise ValueError("Git history returned an invalid parent object identifier")
    return parents


def _references_by_path(
    references: Iterable[_GitBlobReference],
) -> dict[str, _GitBlobReference]:
    by_path: dict[str, _GitBlobReference] = {}
    for reference in references:
        if reference.path in by_path:
            raise ValueError("Git tree returned a duplicate path")
        by_path[reference.path] = reference
    return by_path


def _changed_tree_references(
    references: tuple[_GitBlobReference, ...],
    parent_references: tuple[tuple[_GitBlobReference, ...], ...],
) -> tuple[_GitBlobReference, ...]:
    parent_entries = tuple(_references_by_path(parent) for parent in parent_references)
    if not parent_entries:
        return references
    return tuple(
        reference
        for reference in references
        if all(parent.get(reference.path) != reference for parent in parent_entries)
    )


def git_head_history_violations(
    root: Path,
    *,
    baseline: str = PUBLIC_HISTORY_BASELINE,
) -> tuple[str, ...]:
    """Audit commits reachable from HEAD but outside the immutable public baseline."""
    shallow = subprocess.run(
        ("git", "rev-parse", "--is-shallow-repository"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if shallow not in {"true", "false"}:
        raise ValueError("Git returned an invalid shallow-repository state")
    if shallow == "true":
        return ("history:shallow-clone",)

    resolved_baseline = _resolve_commit(root, baseline)
    if resolved_baseline is None:
        return ("history:missing-baseline",)
    head = _resolve_commit(root, "HEAD")
    if head is None:
        return ("history:baseline-not-ancestor",)
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", resolved_baseline, head),
        cwd=root,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode == 1:
        return ("history:baseline-not-ancestor",)
    if ancestor.returncode != 0:
        raise subprocess.CalledProcessError(
            ancestor.returncode,
            ancestor.args,
            output=ancestor.stdout,
            stderr=ancestor.stderr,
        )

    violations: set[str] = set()
    historical_references: list[_GitBlobReference] = []
    tree_cache: dict[str, tuple[_GitBlobReference, ...]] = {}

    def tree_references(commit: str) -> tuple[_GitBlobReference, ...]:
        if commit not in tree_cache:
            tree_cache[commit] = _git_tree_blob_references(root, commit)
        return tree_cache[commit]

    for commit in _branch_commits(root, head, resolved_baseline):
        references = tree_references(commit)
        paths = tuple(reference.path for reference in references)
        violations.update(f"history:path:{path}" for path in find_path_violations(paths))
        parents = _commit_parents(root, commit)
        historical_references.extend(
            _changed_tree_references(
                references,
                tuple(tree_references(parent) for parent in parents),
            )
        )
        message = subprocess.run(
            ("git", "show", "-s", "--format=%B", commit),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if _has_forbidden_commit_metadata(message):
            violations.add(f"history:commit:{commit}")
    violations.update(
        f"history:content:{path}" for path in _content_violating_paths(root, historical_references)
    )
    return tuple(sorted(violations))


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


def check_public_tree(
    root: Path = ROOT,
    *,
    history_baseline: str = PUBLIC_HISTORY_BASELINE,
) -> tuple[str, ...]:
    tracked_paths = git_tracked_paths(root)
    violations = set(find_path_violations(tracked_paths))
    violations.update(_gitignore_worktree_violations(root, tracked_paths))
    violations.update(git_ignored_tracked_paths(root))
    violations.update(git_index_content_violations(root))
    violations.update(git_head_history_violations(root, baseline=history_baseline))

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
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=ROOT)
    parser.add_argument("--history-baseline", default=PUBLIC_HISTORY_BASELINE)
    namespace = parser.parse_args(arguments)

    root = namespace.root.resolve()
    findings = check_public_tree(root, history_baseline=namespace.history_baseline)
    if findings:
        for finding in findings:
            print(json.dumps(finding, ensure_ascii=False), file=sys.stderr)
        return 1
    print("Public tree check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
