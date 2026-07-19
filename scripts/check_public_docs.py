from __future__ import annotations

import re
import stat
import subprocess
import sys
import unicodedata
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCUMENTS = (
    Path("README.md"),
    Path("CONTRIBUTING.md"),
    Path("SECURITY.md"),
    Path("CODE_OF_CONDUCT.md"),
    Path("CITATION.cff"),
    Path("CHANGELOG.md"),
    Path("docs/benchmarks/foundation-evidence.md"),
    Path("docs/research-claims.md"),
    Path("docs/reference/artifacts.md"),
    Path("docs/reference/cli.md"),
    Path("docs/reference/shadow-mode.md"),
    Path("docs/reference/state-decay-v2-review.md"),
    Path("docs/security.md"),
    Path("examples/atif-shadow/README.md"),
    Path(".github/pull_request_template.md"),
    Path(".github/ISSUE_TEMPLATE/bug.yml"),
    Path(".github/ISSUE_TEMPLATE/benchmark.yml"),
)
PUBLIC_DOCUMENT_DIRECTORIES = (
    Path("docs/benchmarks"),
    Path("docs/reference"),
    Path(".github/ISSUE_TEMPLATE"),
)
OPTIONAL_PUBLIC_DOCUMENTS = (Path("docs/package-description.md"),)
PUBLIC_DOCUMENT_EXCLUSIONS = (
    Path("docs/superpowers"),
    Path(".artifacts"),
    Path("reports/generated"),
)
README_HEADINGS = (
    "# SalienceGate",
    "## Try it locally",
    "## Analyze a trajectory",
    "## What happens inside",
    "## What the examples show",
    "## Use SalienceGate",
    "## Reproduce the research",
    "## Available today",
    "## Limits",
    "## Development",
    "## Citation",
    "## License",
)
README_IMAGE_TARGETS = (
    "docs/assets/readme/pipeline.svg",
    "docs/assets/readme/atif-example-results.svg",
    "docs/assets/readme/reference-run.svg",
)
FORBIDDEN_PATTERNS = (
    (
        re.compile(
            r"(?im)^(?:co-authored-by|signed-off-by|generated-by|assisted-by|"
            r"pair-programmed-by|reviewed-by|acked-by|tested-by):"
        ),
        "attribution trailer",
    ),
    (re.compile(r"(?i)\bstate[- ]of[- ]the[- ]art\b"), "unqualified comparison claim"),
    (re.compile(r"(?i)\bbest[- ]in[- ]class\b"), "unqualified comparison claim"),
    (re.compile(r"(?i)\bproduction[- ]ready\b"), "unsupported readiness claim"),
    (re.compile(r"(?i)\b(?:stage|task)\s+[0-9]+[a-z]?\b"), "internal milestone narration"),
    (re.compile(r"(?i)\bchatgpt enterprise\b"), "private account detail"),
    (re.compile(r"(?i)\benterprise plan\b"), "private account detail"),
    (re.compile(r"(?i)\benterprise subscription\b"), "private account detail"),
    (re.compile(r"(?i)\bworkspace subscription\b"), "private account detail"),
    (re.compile(r"(?i)\borganization plan\b"), "private account detail"),
    (
        re.compile(
            r"(?i)\b(?:OPENAI_API_KEY|ANTHROPIC_API_KEY|API_KEY|ACCESS_TOKEN)\s*=\s*"
            r"(?:sk-)?[A-Za-z0-9_-]{20,}"
        ),
        "credential-like assignment",
    ),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "credential-like token"),
    (re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), "AWS access-key-like token"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b"), "GitHub token-like value"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,255}\b"), "GitHub token-like value"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "Slack token-like value"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "Google API-key-like value"),
    (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"), "private key"),
)
SHADOW_FORBIDDEN_PATTERNS = (
    (re.compile(r"(?i)\b(?:improves|preserves)\s+task success\b"), "task-success claim"),
    (re.compile(r"(?i)\bsaves?\s+tokens?\b"), "token-savings claim"),
    (re.compile(r"(?i)\btoken savings?\b"), "token-savings claim"),
    (re.compile(r"(?i)\bcausal effect\b"), "causal claim"),
    (re.compile(r"(?i)\bcalibrated trigger\b"), "calibration claim"),
    (re.compile(r"(?i)\bpopulation prevalence\b"), "prevalence claim"),
)
SHADOW_REQUIRED_FRAGMENTS = (
    "any-detected-signal-baseline/v1",
    "descriptive_observational",
    "representativeness_supported=false",
    "task_efficacy_supported=false",
    "counterfactual_effect_supported=false",
    "HMAC integrity is not encryption",
)
_MARKDOWN_LINK = re.compile(
    r"(?P<image>!)?\[[^\]]*\]\(\s*(?:<(?P<angle>[^>]+)>|(?P<plain>[^\s)]+))"
    r"(?:\s+['\"][^'\"]*['\"])?\s*\)"
)


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def scan_text(path: Path, text: str) -> list[str]:
    findings: list[str] = []
    for pattern, description in FORBIDDEN_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(f"{path}:{_line_number(text, match.start())}: {description}")

    if path.name == "README.md":
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(unicodedata.category(character) == "So" for character in line):
                findings.append(f"{path}:{line_number}: decorative symbol")

    if path.as_posix() == "docs/reference/shadow-mode.md":
        for pattern, description in SHADOW_FORBIDDEN_PATTERNS:
            for match in pattern.finditer(text):
                findings.append(f"{path}:{_line_number(text, match.start())}: {description}")

    return findings


def _public_paths(root: Path) -> list[Path]:
    paths = [root / relative_path for relative_path in PUBLIC_DOCUMENTS]
    paths.extend(
        root / relative_path
        for relative_path in OPTIONAL_PUBLIC_DOCUMENTS
        if (root / relative_path).exists()
    )
    for relative_directory in PUBLIC_DOCUMENT_DIRECTORIES:
        directory = root / relative_directory
        if directory.is_dir():
            paths.extend(
                path
                for path in sorted(directory.rglob("*"))
                if path.suffix.casefold() in {".md", ".yml", ".yaml"}
            )
    return sorted(set(paths))


def _tracked_paths(root: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ("git", "ls-files", "-z"),
        cwd=root,
        check=True,
        capture_output=True,
    )
    if completed.stdout and not completed.stdout.endswith(b"\0"):
        raise ValueError("Git path output was not NUL-terminated")
    return tuple(item.decode("utf-8") for item in completed.stdout.split(b"\0") if item)


def _without_fenced_code(text: str) -> str:
    output: list[str] = []
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            marker = "```"
        elif stripped.startswith("~~~"):
            marker = "~~~"
        else:
            marker = None
        if marker is not None:
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            output.append("\n" if line.endswith("\n") else "")
        elif fence is None:
            output.append(line)
        else:
            output.append("\n" if line.endswith("\n") else "")
    return "".join(output)


def _resolve_target(document: Path, target: str) -> str | None:
    decoded = unquote(target)
    if "\0" in decoded or "\\" in decoded:
        return None
    parsed = urlsplit(decoded)
    if parsed.scheme or parsed.netloc:
        return ""
    if not parsed.path:
        return document.as_posix()
    if parsed.path.startswith("/"):
        return None
    parts = list(PurePosixPath(document.parent.as_posix()).parts)
    for part in PurePosixPath(parsed.path).parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(part)
    return PurePosixPath(*parts).as_posix() if parts else None


def validate_markdown_links(
    path: Path,
    text: str,
    *,
    tracked_paths: Iterable[str],
    root: Path,
) -> tuple[str, ...]:
    tracked = frozenset(tracked_paths)
    folded_paths: dict[str, set[str]] = {}
    for tracked_path in tracked:
        folded_paths.setdefault(unicodedata.normalize("NFC", tracked_path).casefold(), set()).add(
            tracked_path
        )

    findings: list[str] = []
    searchable = _without_fenced_code(text)
    for match in _MARKDOWN_LINK.finditer(searchable):
        target = match.group("angle") or match.group("plain")
        parsed = urlsplit(unquote(target))
        is_external = bool(parsed.scheme or parsed.netloc)
        line_number = _line_number(searchable, match.start())
        if is_external:
            if match.group("image"):
                findings.append(f"{path}:{line_number}: external image target")
            continue
        resolved = _resolve_target(path, target)
        if resolved is None:
            findings.append(f"{path}:{line_number}: unsafe relative target")
            continue
        if resolved not in tracked:
            folded = unicodedata.normalize("NFC", resolved).casefold()
            description = "case-mismatched target" if folded in folded_paths else "untracked target"
            findings.append(f"{path}:{line_number}: {description} {resolved}")
            continue
        target_path = root / resolved
        try:
            metadata = target_path.lstat()
        except OSError:
            findings.append(f"{path}:{line_number}: missing target {resolved}")
            continue
        if not stat.S_ISREG(metadata.st_mode):
            findings.append(f"{path}:{line_number}: non-regular target {resolved}")
    return tuple(findings)


def validate_public_docs(root: Path = ROOT) -> list[str]:
    findings: list[str] = []
    tracked_paths = _tracked_paths(root)
    tracked = frozenset(tracked_paths)
    for path in _public_paths(root):
        relative_path = path.relative_to(root)
        try:
            metadata = path.lstat()
        except OSError:
            findings.append(f"{relative_path}: missing public document")
            continue
        if not stat.S_ISREG(metadata.st_mode):
            findings.append(f"{relative_path}: non-regular public document")
            continue
        if relative_path.as_posix() not in tracked:
            findings.append(f"{relative_path}: untracked public document")
            continue
        text = path.read_text(encoding="utf-8")
        findings.extend(scan_text(relative_path, text))
        if path.suffix.casefold() == ".md":
            findings.extend(
                validate_markdown_links(
                    relative_path,
                    text,
                    tracked_paths=tracked_paths,
                    root=root,
                )
            )

    readme = root / "README.md"
    if readme.is_file():
        readme_text = readme.read_text(encoding="utf-8")
        lines = readme_text.splitlines()
        position = -1
        for heading in README_HEADINGS:
            matches = [index for index, line in enumerate(lines) if line == heading]
            if not matches:
                findings.append(f"README.md: missing heading {heading!r}")
                continue
            if len(matches) > 1:
                findings.append(f"README.md: duplicate heading {heading!r}")
            next_position = matches[0]
            if next_position <= position:
                findings.append(f"README.md: heading out of order {heading!r}")
            else:
                position = next_position

        word_count = len(readme_text.split())
        if not 1_000 <= word_count <= 1_200:
            findings.append(f"README.md: expected 1000-1200 whitespace tokens, found {word_count}")
        introduction = readme_text.split("\n## Try it locally\n", maxsplit=1)[0]
        for fragment in ("Build with it:", "Study it:", "It is not a memory database."):
            if fragment not in introduction:
                findings.append(f"README.md: missing introductory route {fragment!r}")
        if readme_text.count("![") != 3:
            findings.append("README.md: expected exactly three result visuals")
        for target in README_IMAGE_TARGETS:
            if readme_text.count(target) != 1:
                findings.append(f"README.md: expected one image reference to {target}")
        for label in ("Artifact-compatible after installation:", "Run from a checkout:"):
            if readme_text.count(label) < 2:
                findings.append(f"README.md: expected at least two command labels {label!r}")
        if "\n## Limits\n" in readme_text and "\n## Development\n" in readme_text:
            limits = readme_text.split("\n## Limits\n", maxsplit=1)[1].split(
                "\n## Development\n", maxsplit=1
            )[0]
            if sum(line.startswith("- ") for line in limits.splitlines()) > 5:
                findings.append("README.md: expected no more than five primary limits")

    shadow_reference = root / "docs/reference/shadow-mode.md"
    if shadow_reference.is_file():
        shadow_text = shadow_reference.read_text(encoding="utf-8")
        for fragment in SHADOW_REQUIRED_FRAGMENTS:
            if fragment not in shadow_text:
                findings.append(
                    f"docs/reference/shadow-mode.md: missing evidence fragment {fragment!r}"
                )

    return findings


def main() -> int:
    findings = validate_public_docs()
    if findings:
        for finding in findings:
            print(finding, file=sys.stderr)
        return 1
    print("Public docs check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
