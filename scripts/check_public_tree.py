from __future__ import annotations

import argparse
import json
import subprocess
import sys
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_REGULAR_FILE_MODES = frozenset({b"100644", b"100755"})


def _normalize_path(path: str) -> tuple[str, ...]:
    return tuple(
        unicodedata.normalize("NFC", component).casefold() for component in path.split("/")
    )


def _is_nonportable(path: str, normalized_parts: tuple[str, ...]) -> bool:
    parts = path.split("/")
    return (
        not path
        or path.startswith("/")
        or "\\" in path
        or any(component in {"", ".", ".."} for component in normalized_parts)
        or any(0xD800 <= ord(character) <= 0xDFFF for character in path)
        or (len(parts) > 1 and parts[0].startswith(".") and parts[0] != ".github")
    )


def find_path_violations(tracked_paths: Iterable[str]) -> tuple[str, ...]:
    """Return nonportable paths and Unicode/case collisions."""
    normalized_paths: dict[tuple[str, ...], set[str]] = {}
    violations: set[str] = set()

    for path in tracked_paths:
        normalized = _normalize_path(path)
        normalized_paths.setdefault(normalized, set()).add(path)
        if _is_nonportable(path, normalized):
            violations.add(path)

    for paths in normalized_paths.values():
        if len(paths) > 1:
            violations.update(paths)

    return tuple(sorted(violations))


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


def git_non_regular_tracked_paths(root: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ("git", "ls-files", "--stage", "-z"),
        cwd=root,
        check=True,
        capture_output=True,
    )
    output = completed.stdout
    if output and not output.endswith(b"\0"):
        raise ValueError("Git index output was not NUL-terminated")

    findings: set[str] = set()
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
        mode, _object_id, stage = fields
        if mode not in _REGULAR_FILE_MODES or stage != b"0":
            findings.add(encoded_path.decode("utf-8"))
    return tuple(sorted(findings))


def check_public_tree(root: Path = ROOT) -> tuple[str, ...]:
    tracked_paths = git_tracked_paths(root)
    violations = set(find_path_violations(tracked_paths))
    violations.update(git_ignored_tracked_paths(root))
    violations.update(git_non_regular_tracked_paths(root))
    return tuple(sorted(violations))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=ROOT)
    namespace = parser.parse_args(arguments)

    findings = check_public_tree(namespace.root.resolve())
    if findings:
        for finding in findings:
            print(json.dumps(finding, ensure_ascii=False), file=sys.stderr)
        return 1
    print("Public tree check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
