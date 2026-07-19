from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import re
import stat
import subprocess
import tarfile
import tomllib
import unicodedata
import zipfile
from configparser import ConfigParser
from email.parser import Parser
from importlib import resources
from pathlib import Path, PurePosixPath

import pytest

import saliencegate

ROOT = Path(__file__).parents[1]
DIST = ROOT / "dist"
PACKAGE_DESCRIPTION = ROOT / "docs" / "package-description.md"
REQUIRE_DISTRIBUTIONS = "SALIENCEGATE_REQUIRE_DISTRIBUTIONS"
EXPECTED_CONSOLE_SCRIPTS = {
    "saliencegate": "saliencegate.cli:entrypoint",
    "saliencegate-review": "saliencegate.benchmarks.state_decay_v2.review_cli:entrypoint",
}
SHADOW_RUNTIME_FILES = {
    "saliencegate/shadow/__init__.py",
    "saliencegate/shadow/adapters.py",
    "saliencegate/shadow/analyzer.py",
    "saliencegate/shadow/atif.py",
    "saliencegate/shadow/config.py",
    "saliencegate/shadow/errors.py",
    "saliencegate/shadow/evaluation.py",
    "saliencegate/shadow/inputs.py",
    "saliencegate/shadow/io.py",
    "saliencegate/shadow/observation.py",
    "saliencegate/shadow/report.py",
    "saliencegate/shadow/session.py",
    "saliencegate/shadow/trace.py",
    "saliencegate/shadow/trace_report.py",
}
SHADOW_RESOURCE_FILES = {
    "saliencegate/shadow/atif_profile_compatibility.json",
}
ROOT_SDIST_FILES = {
    ".gitignore",
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    "docs/research-claims.md",
    "docs/security.md",
    "docs/package-description.md",
    "pyproject.toml",
    "uv.lock",
}
SDIST_PREFIXES = (
    "benchmarks/",
    "docs/assets/",
    "docs/benchmarks/",
    "docs/reference/",
    "examples/",
    "scripts/",
    "src/",
    "tests/",
)
FORBIDDEN_PARTS = {
    "__pycache__",
    ".artifacts",
    ".eggs",
    ".git",
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".saliencegate",
    ".venv",
    "env",
    "generated",
    "htmlcov",
    "node_modules",
    "secrets",
    "venv",
    "wheels",
}
FORBIDDEN_TOP_LEVEL = {
    "artifacts",
    "build",
    "checkpoints",
    "dist",
    "models",
    "reports",
    "runs",
    "traces",
}
FORBIDDEN_NAMES = {
    ".ds_store",
    "coverage.xml",
    "junit.xml",
    "thumbs.db",
}
FORBIDDEN_SUFFIXES = (
    ".ckpt",
    ".code-workspace",
    ".db",
    ".db-journal",
    ".db-shm",
    ".db-wal",
    ".gguf",
    ".key",
    ".onnx",
    ".p12",
    ".pem",
    ".pfx",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
    ".swp",
    ".swo",
)


@pytest.fixture(scope="module")
def built_distributions() -> tuple[Path, Path]:
    required = os.environ.get(REQUIRE_DISTRIBUTIONS)
    if required is None:
        pytest.skip("distribution membership is gated only after the single authoritative build")
    assert required == "1", f"{REQUIRE_DISTRIBUTIONS} must be exactly 1"

    wheels = tuple(DIST.glob("*.whl"))
    sdists = tuple(DIST.glob("*.tar.gz"))
    assert {item.name for item in DIST.iterdir()} == {item.name for item in (*wheels, *sdists)}
    assert len(wheels) == 1
    assert len(sdists) == 1
    return wheels[0], sdists[0]


def _reviewable_files() -> frozenset[str]:
    completed = subprocess.run(
        ("git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=10,
    )
    paths = (item.decode("utf-8") for item in completed.stdout.split(b"\0") if item)
    return frozenset(
        path for path in paths if (ROOT / path).is_file() and not (ROOT / path).is_symlink()
    )


def _assert_safe_unique_names(names: tuple[str, ...]) -> None:
    assert names
    assert len(names) == len(set(names))
    folded = tuple(unicodedata.normalize("NFC", item).casefold() for item in names)
    assert len(folded) == len(set(folded))
    for name in names:
        assert name == unicodedata.normalize("NFC", name)
        assert "\\" not in name
        assert "\0" not in name
        assert not name.startswith("/")
        path = PurePosixPath(name)
        assert path.parts
        assert all(part not in ("", ".", "..") for part in path.parts)
        assert path.as_posix() == name


def _wheel_files(wheel: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(wheel) as archive:
        entries = tuple(archive.infolist())
        assert all(not entry.is_dir() for entry in entries)
        names = tuple(entry.filename for entry in entries)
        _assert_safe_unique_names(names)
        for entry in entries:
            mode = entry.external_attr >> 16
            assert stat.S_IFMT(mode) in (0, stat.S_IFREG)
            assert entry.flag_bits & 0x1 == 0
        return {entry.filename: archive.read(entry) for entry in entries}


def _sdist_files(sdist: Path) -> dict[str, bytes]:
    with tarfile.open(sdist, mode="r:gz") as archive:
        entries = tuple(archive.getmembers())
        assert all(entry.isfile() for entry in entries)
        names = tuple(entry.name for entry in entries)
        _assert_safe_unique_names(names)
        expected_root = f"saliencegate-{saliencegate.__version__}"
        assert {PurePosixPath(item).parts[0] for item in names} == {expected_root}
        normalized = tuple(item.removeprefix(f"{expected_root}/") for item in names)
        _assert_safe_unique_names(normalized)
        payloads: dict[str, bytes] = {}
        for name, entry in zip(normalized, entries, strict=True):
            extracted = archive.extractfile(entry)
            assert extracted is not None
            payloads[name] = extracted.read()
        return payloads


def _assert_no_local_or_generated_state(members: frozenset[str]) -> None:
    for member in members:
        path = PurePosixPath(member)
        folded_parts = tuple(part.casefold() for part in path.parts)
        folded_name = path.name.casefold()
        folded_member = member.casefold()
        assert not FORBIDDEN_PARTS.intersection(folded_parts)
        assert not folded_parts or folded_parts[0] not in FORBIDDEN_TOP_LEVEL
        assert not any(part.endswith(".egg-info") for part in folded_parts)
        assert not folded_member.endswith(FORBIDDEN_SUFFIXES)
        assert not folded_member.startswith("docs/superpowers/")
        assert folded_name not in FORBIDDEN_NAMES
        assert not folded_name.startswith((".coverage", ".env"))
        assert not folded_name.endswith("~")


def _assert_wheel_record(files: dict[str, bytes]) -> None:
    record_path = f"saliencegate-{saliencegate.__version__}.dist-info/RECORD"
    rows = tuple(csv.reader(io.StringIO(files[record_path].decode("utf-8"))))
    assert all(len(row) == 3 for row in rows)
    paths = tuple(row[0] for row in rows)
    _assert_safe_unique_names(paths)
    assert set(paths) == set(files)
    for path, digest, size in rows:
        if path == record_path:
            assert digest == size == ""
            continue
        expected_digest = base64.urlsafe_b64encode(hashlib.sha256(files[path]).digest()).rstrip(
            b"="
        )
        assert digest == f"sha256={expected_digest.decode('ascii')}"
        assert size == str(len(files[path]))


def _console_scripts(payload: bytes) -> dict[str, str]:
    parser = ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_string(payload.decode("utf-8"))
    assert parser.sections() == ["console_scripts"]
    assert not parser.defaults()
    return dict(parser.items("console_scripts", raw=True))


def test_package_exposes_a_semantic_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[.-][0-9A-Za-z.-]+)?", saliencegate.__version__)


def test_package_declares_typing_support() -> None:
    assert resources.files("saliencegate").joinpath("py.typed").is_file()


def test_source_metadata_exposes_exact_public_console_scripts() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["scripts"] == EXPECTED_CONSOLE_SCRIPTS
    assert metadata["project"]["readme"] == "docs/package-description.md"


def test_source_distribution_declares_the_complete_shadow_example_membership() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    includes = metadata["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    source_shadow = {
        path.relative_to(ROOT / "src").as_posix()
        for path in (ROOT / "src/saliencegate/shadow").glob("*.py")
    }

    assert "/examples" in includes
    assert "/docs/assets" in includes
    assert "/docs/package-description.md" in includes
    assert "/docs/research-claims.md" in includes
    assert "/docs/security.md" in includes
    assert "/uv.lock" in includes
    assert source_shadow == SHADOW_RUNTIME_FILES
    assert (ROOT / "examples/shadow_asyncio.py").is_file()
    assert {
        "README.md",
        "codex-minimal.trajectory.json",
        "one_call.py",
        "terminus-minimal.trajectory.json",
    } == {path.name for path in (ROOT / "examples/atif-shadow").iterdir()}


@pytest.mark.parametrize(
    "names",
    (
        ("saliencegate/module.py", "saliencegate/module.py"),
        ("saliencegate/module.py", "saliencegate/MODULE.py"),
        ("../saliencegate/module.py",),
        ("/saliencegate/module.py",),
        ("saliencegate/./module.py",),
        ("saliencegate\\module.py",),
    ),
)
def test_archive_names_must_be_unique_and_canonical(names: tuple[str, ...]) -> None:
    with pytest.raises(AssertionError):
        _assert_safe_unique_names(names)


@pytest.mark.parametrize(
    "member",
    (
        "artifacts/result.json",
        "tests/.saliencegate/state.json",
        "tests/.venv/secret.bin",
        "tests/.env.private",
        "tests/coverage.xml",
        "tests/fixtures/model.GGUF",
        "tests/htmlcov/index.html",
        "tests/secrets/token.txt",
    ),
)
def test_distribution_guard_rejects_local_or_generated_state(member: str) -> None:
    with pytest.raises(AssertionError):
        _assert_no_local_or_generated_state(frozenset({member}))


def test_archive_readers_reject_duplicates_and_links(tmp_path: Path) -> None:
    wheel = tmp_path / "duplicate.whl"
    with zipfile.ZipFile(wheel, mode="w") as archive:
        archive.writestr("saliencegate/__init__.py", b"first")
        with pytest.warns(UserWarning):
            archive.writestr("saliencegate/__init__.py", b"second")
    with pytest.raises(AssertionError):
        _wheel_files(wheel)

    sdist = tmp_path / "linked.tar.gz"
    with tarfile.open(sdist, mode="w:gz") as archive:
        link = tarfile.TarInfo(f"saliencegate-{saliencegate.__version__}/linked")
        link.type = tarfile.SYMTYPE
        link.linkname = "target"
        archive.addfile(link)
    with pytest.raises(AssertionError):
        _sdist_files(sdist)


def test_built_wheel_has_exact_runtime_membership_and_payloads(
    built_distributions: tuple[Path, Path],
) -> None:
    wheel, _sdist = built_distributions
    files = _wheel_files(wheel)
    runtime = {
        item.removeprefix("src/")
        for item in _reviewable_files()
        if item.startswith("src/saliencegate/")
    }
    dist_info = f"saliencegate-{saliencegate.__version__}.dist-info"
    metadata = {
        f"{dist_info}/METADATA",
        f"{dist_info}/RECORD",
        f"{dist_info}/WHEEL",
        f"{dist_info}/entry_points.txt",
        f"{dist_info}/licenses/LICENSE",
    }

    assert set(files) == runtime | metadata
    for member in runtime:
        assert files[member] == (ROOT / "src" / member).read_bytes()
    assert files[f"{dist_info}/licenses/LICENSE"] == (ROOT / "LICENSE").read_bytes()
    assert _console_scripts(files[f"{dist_info}/entry_points.txt"]) == EXPECTED_CONSOLE_SCRIPTS
    _assert_wheel_record(files)
    assert {
        "saliencegate/commands/pilot.py",
        "saliencegate/models/openai_compatible.py",
        "saliencegate/prompts/paper_two_phase_v1.py",
        "saliencegate/py.typed",
        "saliencegate/repository/migrations/0001_initial.sql",
        "saliencegate/repository/migrations/0002_unique_invocation_event.sql",
        *SHADOW_RUNTIME_FILES,
        *SHADOW_RESOURCE_FILES,
    } <= files.keys()


def test_built_sdist_has_exact_reviewable_membership_and_payloads(
    built_distributions: tuple[Path, Path],
) -> None:
    _wheel, sdist = built_distributions
    files = _sdist_files(sdist)
    expected = {
        item
        for item in _reviewable_files()
        if item in ROOT_SDIST_FILES or item.startswith(SDIST_PREFIXES)
    }
    expected.add("PKG-INFO")

    assert set(files) == expected
    for member in expected - {"PKG-INFO"}:
        assert files[member] == (ROOT / member).read_bytes()
    assert {
        "benchmarks/state_decay/smoke.jsonl",
        "benchmarks/state_decay/smoke_manifest.json",
        "benchmarks/shadow_trace/reference-macos-26.5.2-arm64-cpython-3.12.3.json",
        "benchmarks/shadow_trace/reference-macos-26.5.2-arm64-cpython-3.12.3.manifest.json",
        "docs/assets/readme/atif-example-results.svg",
        "docs/assets/readme/pipeline.svg",
        "docs/assets/readme/reference-run.svg",
        "docs/benchmarks/foundation-evidence.md",
        "docs/package-description.md",
        "docs/research-claims.md",
        "docs/reference/artifacts.md",
        "docs/reference/cli.md",
        "docs/reference/shadow-mode.md",
        "docs/security.md",
        "examples/atif-shadow/README.md",
        "examples/atif-shadow/codex-minimal.trajectory.json",
        "examples/atif-shadow/one_call.py",
        "examples/atif-shadow/terminus-minimal.trajectory.json",
        "examples/shadow_asyncio.py",
        "scripts/benchmark_shadow_trace.py",
        "scripts/run_without_sockets.py",
        "scripts/smoke_core_imports.py",
        "scripts/smoke_launch_contracts.py",
        "scripts/smoke_model_runtime.py",
        "scripts/smoke_package_imports.py",
        "scripts/smoke_shadow_installed.py",
        "scripts/verify_built_artifacts.py",
        "uv.lock",
        "tests/test_installed_shadow_atif_smoke.py",
        "tests/test_shadow_trace_benchmark.py",
        "tests/fixtures/models/basic_responses.jsonl",
        "tests/fixtures/models/paper_two_phase_always_inject_responses.jsonl",
        "tests/fixtures/models/paper_two_phase_fixed_step_responses.jsonl",
        "tests/fixtures/models/paper_two_phase_retrieval_responses.jsonl",
        "tests/fixtures/models/two_phase_contract_responses.jsonl",
        "tests/fixtures/pilots/stage_2_cases.json",
        "tests/fixtures/prompts/paper_two_phase_v1.json",
        "tests/fixtures/runs/basic.jsonl",
        "tests/fixtures/runs/paper_two_phase_basic.jsonl",
        "tests/fixtures/shadow/atif/codex-bundled-synthetic.trajectory.json",
        "tests/fixtures/shadow/atif/terminus-context-sanitized.trajectory.json",
        "tests/fixtures/shadow/atif/terminus-timeout-sanitized.trajectory.json",
        *(f"src/{path}" for path in SHADOW_RUNTIME_FILES),
        *(f"src/{path}" for path in SHADOW_RESOURCE_FILES),
    } <= files.keys()


def test_built_distributions_exclude_local_and_generated_state(
    built_distributions: tuple[Path, Path],
) -> None:
    wheel, sdist = built_distributions
    wheel_files = _wheel_files(wheel)
    sdist_files = _sdist_files(sdist)

    _assert_no_local_or_generated_state(frozenset(wheel_files))
    _assert_no_local_or_generated_state(frozenset(sdist_files))
    assert not any(
        item.startswith(("benchmarks/", "docs/", "scripts/", "tests/")) for item in wheel_files
    )


def test_distribution_metadata_is_identical_and_keeps_the_runtime_optional(
    built_distributions: tuple[Path, Path],
) -> None:
    wheel, sdist = built_distributions
    metadata_path = f"saliencegate-{saliencegate.__version__}.dist-info/METADATA"
    wheel_metadata = _wheel_files(wheel)[metadata_path]
    sdist_metadata = _sdist_files(sdist)["PKG-INFO"]
    assert wheel_metadata == sdist_metadata

    metadata = Parser().parsestr(wheel_metadata.decode("utf-8"))

    assert metadata["Metadata-Version"] == "2.4"
    assert metadata["Name"] == "saliencegate"
    assert metadata["Version"] == saliencegate.__version__
    assert metadata["Requires-Python"] == ">=3.11"
    assert metadata.get_all("Provides-Extra") == ["model-runtime"]
    assert metadata.get_all("Requires-Dist") == [
        "pydantic<3,>=2.11",
        "httpx<0.29,>=0.28.1; extra == 'model-runtime'",
        "openai-harmony==0.0.8; extra == 'model-runtime'",
    ]
    description = metadata.get_payload()
    assert description.strip() == PACKAGE_DESCRIPTION.read_text(encoding="utf-8").strip()
    assert not re.search(r"!\[[^\]]*\]\([^)]+\)", description)
    assert not re.search(r"(?<!!)\[[^\]]+\]\([^)]+\)", description)
    assert not re.search(r"(?i)<(?:a|img)\b|\b(?:href|src)\s*=|https?://", description)
