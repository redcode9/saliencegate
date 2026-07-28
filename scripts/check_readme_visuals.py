from __future__ import annotations

import hashlib
import json
import re
import stat
import struct
import subprocess
import sys
import tomllib
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MANIFEST = Path(
    "benchmarks/shadow_trace/reference-macos-26.5.2-arm64-cpython-3.12.3.manifest.json"
)
REFERENCE_COMMAND = (
    "uv --cache-dir /private/tmp/saliencegate-uv-cache run --python 3.12.3 --locked "
    "python scripts/benchmark_shadow_trace.py --assert-budgets"
)
PROJECT_RUNTIME_METADATA_SELECTION = (
    "project.requires-python",
    "project.dependencies",
    "project.optional-dependencies",
    "dependency-groups.dev",
)
PROJECT_RUNTIME_METADATA_CANONICALIZATION = "UTF-8 JSON with sorted keys and compact separators"
README_ASSETS = (
    (Path("docs/assets/readme/pipeline.svg"), ("capture", "hmac", "bounded report")),
    (
        Path("docs/assets/readme/capture-headlines.svg"),
        (
            "synthetic",
            "memory review suggested",
            "no current evidence",
            "insufficient evidence",
        ),
    ),
    (Path("docs/assets/readme/reference-run.svg"), ("memory", "sqlite", "local reference run")),
)

_SVG_NAMESPACE = "http://www.w3.org/2000/svg"
_XML_NAMESPACE = "http://www.w3.org/XML/1998/namespace"
_FORBIDDEN_ELEMENTS = frozenset(
    {
        "a",
        "animate",
        "animateMotion",
        "animateTransform",
        "audio",
        "discard",
        "filter",
        "foreignObject",
        "iframe",
        "image",
        "linearGradient",
        "metadata",
        "radialGradient",
        "script",
        "set",
        "style",
        "video",
    }
)
_RESOURCE_PREFIXES = ("data:", "file:", "ftp:", "http:", "https:", "javascript:")
_MARKDOWN_IMAGE = re.compile(
    r"!\[([^\]]*)\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+['\"][^'\"]*['\"])?\s*\)"
)
_HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}\Z")
_NUMBER = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
_OBSERVATIONAL_FORBIDDEN_PATTERNS = (
    (re.compile(r"(?i)\b(?:improves|preserves)\s+task success\b"), "task-success claim"),
    (re.compile(r"(?i)\bsaves?\s+tokens?\b"), "token-savings claim"),
    (re.compile(r"(?i)\btoken savings?\b"), "token-savings claim"),
    (re.compile(r"(?i)\bcausal effect\b"), "causal claim"),
    (re.compile(r"(?i)\bcalibrated trigger\b"), "calibration claim"),
    (re.compile(r"(?i)\bpopulation prevalence\b"), "prevalence claim"),
    (re.compile(r"(?i)\bthe agent (?:needs|requires) memory\b"), "memory-need certainty"),
    (
        re.compile(r"(?i)\breminders? (?:will|would) (?:help|improve)\b"),
        "reminder-effect claim",
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked_runtime_paths(root: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ("git", "ls-files", "-z", "--", "src/saliencegate"),
        cwd=root,
        check=True,
        capture_output=True,
    )
    if completed.stdout and not completed.stdout.endswith(b"\0"):
        raise ValueError("Git path output was not NUL-terminated")
    paths = tuple(item.decode("utf-8") for item in completed.stdout.split(b"\0") if item)
    return tuple(sorted(paths))


def runtime_source_digest(root: Path) -> tuple[int, str]:
    """Hash the complete tracked runtime surface with unambiguous path/content framing."""
    digest = hashlib.sha256()
    paths = _tracked_runtime_paths(root)
    for relative_path in paths:
        path = root / relative_path
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"runtime source is not a regular file: {relative_path}")
        encoded_path = relative_path.encode("utf-8")
        content = path.read_bytes()
        digest.update(struct.pack(">Q", len(encoded_path)))
        digest.update(encoded_path)
        digest.update(struct.pack(">Q", len(content)))
        digest.update(content)
    return len(paths), digest.hexdigest()


def project_runtime_metadata_digest(
    root: Path,
    path: Path = Path("pyproject.toml"),
) -> str:
    """Hash only project fields that can change the measured runtime environment."""
    metadata = tomllib.loads((root / path).read_text(encoding="utf-8"))
    project = metadata["project"]
    dependency_groups = metadata["dependency-groups"]
    selected = {
        "dependency-groups": {"dev": dependency_groups["dev"]},
        "project": {
            "dependencies": project["dependencies"],
            "optional-dependencies": project["optional-dependencies"],
            "requires-python": project["requires-python"],
        },
    }
    payload = json.dumps(
        selected,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_file_finding(
    root: Path,
    manifest: dict[str, object],
    key: str,
) -> str | None:
    item = manifest.get(key)
    if not isinstance(item, dict):
        return f"{REFERENCE_MANIFEST}: invalid {key} binding"
    path_value = item.get("path")
    digest_value = item.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest_value, str):
        return f"{REFERENCE_MANIFEST}: invalid {key} binding"
    path = root / path_value
    if not path.is_file() or path.is_symlink():
        return f"{REFERENCE_MANIFEST}: missing regular {key} input"
    if _sha256(path) != digest_value:
        return f"{REFERENCE_MANIFEST}: changed {key} input"
    return None


def validate_evidence_manifest(
    root: Path = ROOT,
    manifest_path: Path = REFERENCE_MANIFEST,
) -> tuple[str, ...]:
    findings: list[str] = []
    absolute_manifest = root / manifest_path
    try:
        manifest = json.loads(absolute_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return (f"{manifest_path}: unreadable evidence manifest",)
    if not isinstance(manifest, dict):
        return (f"{manifest_path}: invalid evidence manifest",)

    if manifest.get("schema_version") != "shadow-trace-benchmark-evidence-manifest/v1":
        findings.append(f"{manifest_path}: invalid schema version")
    for key in (
        "report",
        "benchmark_script",
        "socket_guard",
        "python_version_file",
        "dependency_lock",
    ):
        finding = _manifest_file_finding(root, manifest, key)
        if finding is not None:
            findings.append(finding)

    runtime = manifest.get("runtime_source")
    if not isinstance(runtime, dict):
        findings.append(f"{manifest_path}: invalid runtime source binding")
    else:
        count, digest = runtime_source_digest(root)
        expected = {
            "path": "src/saliencegate",
            "git_tracked_regular_file_count": count,
            "digest_algorithm": "sha256",
            "framing": (
                "uint64be(path_utf8_length) || path_utf8 || uint64be(content_length) || content"
            ),
            "sha256": digest,
        }
        if runtime != expected:
            findings.append(f"{manifest_path}: changed runtime source surface")

    project_metadata = manifest.get("project_runtime_metadata")
    project_path = root / "pyproject.toml"
    if not project_path.is_file() or project_path.is_symlink():
        findings.append(f"{manifest_path}: missing regular project runtime metadata input")
    else:
        try:
            project_digest = project_runtime_metadata_digest(root)
        except (KeyError, OSError, UnicodeError, tomllib.TOMLDecodeError):
            findings.append(f"{manifest_path}: unreadable project runtime metadata input")
        else:
            expected_project_metadata = {
                "path": "pyproject.toml",
                "selection": list(PROJECT_RUNTIME_METADATA_SELECTION),
                "canonicalization": PROJECT_RUNTIME_METADATA_CANONICALIZATION,
                "sha256": project_digest,
            }
            if project_metadata != expected_project_metadata:
                findings.append(f"{manifest_path}: changed project runtime metadata")

    toolchain = manifest.get("toolchain")
    if toolchain != {"python": "CPython 3.12.3", "uv": "uv 0.11.26"}:
        findings.append(f"{manifest_path}: invalid captured toolchain")
    if manifest.get("reproduction_command") != REFERENCE_COMMAND:
        findings.append(f"{manifest_path}: invalid reproduction command")

    report_item = manifest.get("report")
    if isinstance(report_item, dict) and isinstance(report_item.get("path"), str):
        try:
            report = json.loads((root / report_item["path"]).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            report = None
        if not isinstance(report, dict):
            findings.append(f"{manifest_path}: unreadable benchmark report")
        else:
            metadata = report.get("metadata")
            if (
                report.get("schema_version") != "shadow-trace-benchmark/v1"
                or report.get("passed") is not True
                or not isinstance(metadata, dict)
                or metadata.get("python_implementation") != "CPython"
                or metadata.get("python_version") != "3.12.3"
            ):
                findings.append(f"{manifest_path}: invalid benchmark report identity")
    return tuple(findings)


def _local_name(name: str) -> str:
    return name.rsplit("}", maxsplit=1)[-1]


def _normalized_text(element: ET.Element) -> str:
    return " ".join("".join(element.itertext()).split())


def _parse_view_box(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    fields = value.replace(",", " ").split()
    if len(fields) != 4:
        return None
    try:
        numbers = tuple(float(field) for field in fields)
    except ValueError:
        return None
    if numbers[2] <= 0 or numbers[3] <= 0:
        return None
    return numbers  # type: ignore[return-value]


def _channel_luminance(channel: int) -> float:
    value = channel / 255
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _luminance(color: str) -> float:
    if not _HEX_COLOR.fullmatch(color):
        raise ValueError("color must be six-digit hexadecimal text")
    channels = tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
    red, green, blue = (_channel_luminance(channel) for channel in channels)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(foreground: str, background: str) -> float:
    lighter, darker = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _attribute_findings(path: Path, element: ET.Element) -> Iterable[str]:
    for raw_name, value in element.attrib.items():
        name = _local_name(raw_name)
        folded_value = value.casefold()
        if name.casefold().startswith("on"):
            yield f"{path}: event handler attribute"
        if name == "style":
            yield f"{path}: inline style attribute"
        if (
            name in {"href", "src"}
            or "url(" in folded_value
            or folded_value.startswith(_RESOURCE_PREFIXES)
        ):
            yield f"{path}: embedded or external resource"


def validate_svg_text(
    path: Path,
    text: str,
    *,
    required_text: Sequence[str] = (),
) -> tuple[str, ...]:
    findings: list[str] = []
    folded = text.casefold()
    if "<!--" in text:
        findings.append(f"{path}: XML comment")
    if "<!doctype" in folded or "<!entity" in folded:
        findings.append(f"{path}: document type or entity declaration")

    namespace_values = re.findall(r"\bxmlns(?::[\w.-]+)?=['\"]([^'\"]+)['\"]", text)
    if any(value not in {_SVG_NAMESPACE, _XML_NAMESPACE} for value in namespace_values):
        findings.append(f"{path}: unsupported XML namespace")
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return tuple((*findings, f"{path}: invalid XML"))
    if root.tag != f"{{{_SVG_NAMESPACE}}}svg":
        findings.append(f"{path}: root is not SVG")
    if _parse_view_box(root.get("viewBox")) is None:
        findings.append(f"{path}: missing or invalid viewBox")
    if root.get("role") != "img":
        findings.append(f'{path}: role must be "img"')

    ids: dict[str, ET.Element] = {}
    duplicate_ids: set[str] = set()
    titles: list[ET.Element] = []
    descriptions: list[ET.Element] = []
    canvas_fill: str | None = None
    for element in root.iter():
        local_name = _local_name(element.tag)
        namespace = element.tag.partition("}")[0].removeprefix("{") if "}" in element.tag else ""
        if namespace != _SVG_NAMESPACE:
            findings.append(f"{path}: unsupported element namespace")
        if local_name in _FORBIDDEN_ELEMENTS:
            findings.append(f"{path}: forbidden element {local_name}")
        identifier = element.get("id")
        if identifier:
            if identifier in ids:
                duplicate_ids.add(identifier)
            ids[identifier] = element
        if local_name == "title":
            titles.append(element)
        elif local_name == "desc":
            descriptions.append(element)
        elif local_name == "rect" and canvas_fill is None:
            fill = element.get("fill")
            if fill and _HEX_COLOR.fullmatch(fill):
                canvas_fill = fill
        if local_name in {"text", "tspan"}:
            size = element.get("font-size")
            if size is not None and (not _NUMBER.fullmatch(size) or float(size) < 28):
                findings.append(f"{path}: meaning-bearing text below 28 SVG units")
            if local_name == "text" and size is None:
                findings.append(f"{path}: text without an explicit font size")
            fill = element.get("fill")
            if fill is None or not _HEX_COLOR.fullmatch(fill):
                findings.append(f"{path}: text without a fixed hexadecimal color")
            elif canvas_fill is not None and contrast_ratio(fill, canvas_fill) < 4.5:
                findings.append(f"{path}: normal text contrast below 4.5:1")
        findings.extend(_attribute_findings(path, element))

    if duplicate_ids:
        findings.append(f"{path}: duplicate element ID")
    if len(titles) != 1 or len(descriptions) != 1:
        findings.append(f"{path}: requires exactly one title and one desc")
    title_text = _normalized_text(titles[0]) if len(titles) == 1 else ""
    desc_text = _normalized_text(descriptions[0]) if len(descriptions) == 1 else ""
    if not title_text or not desc_text:
        findings.append(f"{path}: title and desc require non-empty text")

    labelledby = root.get("aria-labelledby", "").split()
    if (
        len(labelledby) != 2
        or len(set(labelledby)) != 2
        or any(identifier not in ids for identifier in labelledby)
        or (len(titles) == 1 and ids.get(labelledby[0]) is not titles[0])
        or (len(descriptions) == 1 and ids.get(labelledby[-1]) is not descriptions[0])
    ):
        findings.append(f"{path}: aria-labelledby must bind title then desc")

    accessible_text = unicodedata.normalize("NFC", f"{title_text} {desc_text}").casefold()
    if any(
        unicodedata.normalize("NFC", item).casefold() not in accessible_text
        for item in required_text
    ):
        findings.append(f"{path}: title and desc lack asset-specific text")
    rendered_text = _normalized_text(root)
    for pattern, description in _OBSERVATIONAL_FORBIDDEN_PATTERNS:
        if pattern.search(rendered_text):
            findings.append(f"{path}: {description}")
    return tuple(dict.fromkeys(findings))


def validate_markdown_images(path: Path, text: str) -> tuple[str, ...]:
    findings: list[str] = []
    for match in _MARKDOWN_IMAGE.finditer(text):
        alt_text = " ".join(match.group(1).split())
        target = match.group(2) or match.group(3)
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or target.startswith(("/", "#")):
            findings.append(f"{path}: external image target")
            continue
        target_path = PurePosixPath(parsed.path)
        if (
            not target_path.parts
            or any(part in {"", ".", ".."} for part in target_path.parts)
            or target_path.suffix.casefold() != ".svg"
        ):
            findings.append(f"{path}: unsafe or non-SVG image target")
        folded_alt = alt_text.casefold()
        if len(alt_text) < 12 or folded_alt in {"chart", "diagram", "figure", "image", "visual"}:
            findings.append(f"{path}: image requires meaningful alt text")
    return tuple(findings)


def validate_readme_assets(root: Path = ROOT) -> tuple[str, ...]:
    findings: list[str] = []
    for relative_path, required_text in README_ASSETS:
        path = root / relative_path
        try:
            metadata = path.lstat()
        except OSError:
            findings.append(f"{relative_path}: missing README visual")
            continue
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            findings.append(f"{relative_path}: README visual is not a regular file")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            findings.append(f"{relative_path}: unreadable README visual")
            continue
        findings.extend(validate_svg_text(relative_path, text, required_text=required_text))
    return tuple(findings)


def validate_capture_headline_render(root: Path = ROOT) -> tuple[str, ...]:
    script = root / "scripts/render_capture_headlines.py"
    fixture = root / "examples/capture/headline-results.json"
    output = root / "docs/assets/readme/capture-headlines.svg"
    try:
        completed = subprocess.run(
            (
                sys.executable,
                str(script),
                "--check",
                "--fixture",
                str(fixture),
                "--output",
                str(output),
            ),
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ("docs/assets/readme/capture-headlines.svg: renderer check failed",)
    if completed.returncode != 0:
        return ("docs/assets/readme/capture-headlines.svg: fixture rendering is stale",)
    return ()


def main() -> int:
    findings = list(validate_evidence_manifest())
    findings.extend(validate_readme_assets())
    findings.extend(validate_capture_headline_render())
    readme = ROOT / "README.md"
    if readme.is_file():
        findings.extend(
            validate_markdown_images(Path("README.md"), readme.read_text(encoding="utf-8"))
        )
    if findings:
        for finding in findings:
            print(finding, file=sys.stderr)
        return 1
    print("README visual evidence check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
