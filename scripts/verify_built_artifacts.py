from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import tomllib
import unicodedata
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

DOCUMENTED_COMMAND_CASES = {
    "offline-demo": "saliencegate demo",
    "shadow-native": (
        "saliencegate shadow analyze .saliencegate-shadow/events.ndjson "
        "--run-id b35f05f3-555b-4f09-8996-a7b3693bb54a "
        "--output .saliencegate-shadow/shadow-report.json --json"
    ),
    "atif-codex": (
        "saliencegate shadow analyze-atif "
        ".saliencegate/atif-shadow/codex.trajectory.json "
        "--profile harbor-codex-v1 "
        "--run-id c0de0000-0000-4000-8000-000000000001 "
        "--working-directory /synthetic/workspace "
        "--environment-digest "
        "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee "
        "--output .saliencegate/atif-shadow/codex.report.json --json"
    ),
    "atif-terminus": (
        "saliencegate shadow analyze-atif "
        ".saliencegate/atif-shadow/terminus.trajectory.json "
        "--profile harbor-terminus-2-v1 "
        "--run-id 7e2a0000-0000-4000-8000-000000000001 "
        "--working-directory /synthetic/workspace "
        "--environment-digest "
        "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee "
        "--output .saliencegate/atif-shadow/terminus.report.json --json"
    ),
}

_ARTIFACT_BLOCK = re.compile(
    r"(?ms)^Artifact-compatible after installation:\s*\n+"
    r"```(?:bash|sh)\s*\n(?P<body>.*?)^```\s*$"
)
_ENVIRONMENT_DIGEST = "e" * 64
_WORKING_DIRECTORY = "/synthetic/workspace"


@dataclass(frozen=True)
class _AtifCase:
    name: str
    source_name: str
    profile_alias: str
    profile_id: str
    run_id: str


_ATIF_CASES = (
    _AtifCase(
        name="codex",
        source_name="codex-minimal.trajectory.json",
        profile_alias="harbor-codex-v1",
        profile_id="harbor-codex/v1",
        run_id="c0de0000-0000-4000-8000-000000000001",
    ),
    _AtifCase(
        name="terminus",
        source_name="terminus-minimal.trajectory.json",
        profile_alias="harbor-terminus-2-v1",
        profile_id="harbor-terminus-2/v1",
        run_id="7e2a0000-0000-4000-8000-000000000001",
    ),
)
EXECUTED_COMMAND_CASES = frozenset(
    (
        "offline-demo",
        "shadow-native",
        "atif-one-call",
        *(f"atif-{case.name}" for case in _ATIF_CASES),
    )
)
SUPPLEMENTAL_ARTIFACT_PROOFS = frozenset({"atif-one-call"})


def artifact_compatible_commands(text: str) -> tuple[str, ...]:
    """Return normalized commands from visibly classified Markdown blocks."""
    commands: list[str] = []
    for match in _ARTIFACT_BLOCK.finditer(text):
        pending = ""
        for raw_line in match.group("body").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith("\\"):
                pending += line[:-1].rstrip() + " "
                continue
            commands.append(" ".join((pending + line).split()))
            pending = ""
        if pending:
            raise ValueError("artifact-compatible command has an incomplete line continuation")
    return tuple(commands)


def _run(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            tuple(os.fspath(item) for item in command),
            cwd=cwd,
            env=env,
            check=True,
            capture_output=capture_output,
            timeout=300,
        )
    except subprocess.CalledProcessError as error:
        if not capture_output:
            raise
        stdout = _diagnostic_excerpt(error.stdout)
        stderr = _diagnostic_excerpt(error.stderr)
        raise RuntimeError(
            f"artifact proof command exited with {error.returncode}; "
            f"stdout={stdout!r}; stderr={stderr!r}"
        ) from None


def _diagnostic_excerpt(payload: bytes | None) -> str:
    text = (payload or b"").decode("utf-8", errors="replace")
    for credential in (
        "provider-credential-read-must-fail",
        os.environ.get("ANTHROPIC_API_KEY", ""),
        os.environ.get("OPENAI_API_KEY", ""),
    ):
        if credential:
            text = text.replace(credential, "<redacted>")
    limit = 4096
    if len(text) > limit:
        text = text[:limit] + "\n[output truncated]"
    return text


def _require_one_distribution(dist_dir: Path, suffix: str) -> Path:
    matches = tuple(dist_dir.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {suffix} distribution")
    path = matches[0]
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise RuntimeError(f"distribution is not a regular file: {path.name}")
    return path


def _canonical_archive_path(name: str) -> PurePosixPath:
    if name != unicodedata.normalize("NFC", name) or "\\" in name or "\0" in name:
        raise RuntimeError("source distribution contains a non-canonical path")
    path = PurePosixPath(name)
    if (
        name.startswith("/")
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RuntimeError("source distribution contains an unsafe path")
    if path.as_posix() != name:
        raise RuntimeError("source distribution contains a non-canonical path")
    return path


def _write_private(path: Path, payload: bytes) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _extract_sdist(sdist: Path, destination: Path) -> None:
    destination.mkdir(mode=0o700)
    with tarfile.open(sdist, mode="r:gz") as archive:
        members = tuple(archive.getmembers())
        if not members or any(not member.isfile() for member in members):
            raise RuntimeError("source distribution must contain regular files only")
        paths = tuple(_canonical_archive_path(member.name) for member in members)
        folded = tuple(path.as_posix().casefold() for path in paths)
        if len(folded) != len(set(folded)):
            raise RuntimeError("source distribution contains duplicate paths")
        roots = {path.parts[0] for path in paths}
        if len(roots) != 1 or any(len(path.parts) == 1 for path in paths):
            raise RuntimeError("source distribution must have one non-empty package root")
        for member, archive_path in zip(members, paths, strict=True):
            relative = Path(*archive_path.parts[1:])
            target = destination / relative
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"cannot read source distribution member: {member.name}")
            _write_private(target, extracted.read())


def _hatchling_version(package_root: Path) -> str:
    lock = tomllib.loads((package_root / "uv.lock").read_text(encoding="utf-8"))
    versions = {
        item["version"]
        for item in lock.get("package", ())
        if isinstance(item, dict) and item.get("name") == "hatchling"
    }
    if len(versions) != 1:
        raise RuntimeError("locked Hatchling version is missing or ambiguous")
    return versions.pop()


def _install_environment(
    *,
    uv: str,
    python_version: str,
    environment_root: Path,
    distribution: Path,
    package_root: Path,
    runtime_requirements: Path,
    build_constraints: Path,
    is_sdist: bool,
) -> Path:
    _run((uv, "venv", "--python", python_version, environment_root))
    python = environment_root / "bin" / "python"
    _run((uv, "pip", "install", "--python", python, "--require-hashes", "-r", runtime_requirements))
    if is_sdist:
        hatchling = _hatchling_version(package_root)
        _run(
            (
                uv,
                "pip",
                "install",
                "--python",
                python,
                "--require-hashes",
                "--constraints",
                build_constraints,
                f"hatchling=={hatchling}",
            )
        )
        _run(
            (
                uv,
                "pip",
                "install",
                "--python",
                python,
                "--no-deps",
                "--no-build-isolation",
                distribution,
            )
        )
    else:
        _run((uv, "pip", "install", "--python", python, "--no-deps", distribution))
    return python


def _private_environment(root: Path) -> dict[str, str]:
    home = root / "home"
    config = root / "config"
    home.mkdir(mode=0o700)
    config.mkdir(mode=0o700)
    environment = os.environ.copy()
    environment.update(
        {
            "ANTHROPIC_API_KEY": "provider-credential-read-must-fail",
            "HOME": os.fspath(home),
            "OPENAI_API_KEY": "provider-credential-read-must-fail",
            "PYTHONPATH": "",
            "XDG_CONFIG_HOME": os.fspath(config),
        }
    )
    return environment


def _prove_documented_cases(
    *,
    label: str,
    python: Path,
    package_root: Path,
    work_root: Path,
) -> None:
    case_root = work_root / label
    case_root.mkdir(mode=0o700)
    shadow_root = case_root / "shadow"
    shadow_root.mkdir(mode=0o700)
    environment = _private_environment(case_root)
    guard = package_root / "scripts" / "run_without_sockets.py"
    validator = package_root / "scripts" / "smoke_shadow_installed.py"
    examples = package_root / "examples" / "atif-shadow"
    saliencegate = python.parent / "saliencegate"

    demo = _run(
        (python, "-I", guard, saliencegate, "demo"),
        cwd=case_root,
        env=environment,
        capture_output=True,
    )
    demo_output = demo.stdout.decode("utf-8")
    if (
        demo.stderr
        or "SalienceGate offline demo" not in demo_output
        or "oracle: 32 passed, 0 failed" not in demo_output
        or "This verifies deterministic mechanics, not agent task efficacy." not in demo_output
    ):
        raise RuntimeError(f"{label} offline demo returned unexpected output")

    native_run_id = "b35f05f3-555b-4f09-8996-a7b3693bb54a"
    native_source = shadow_root / "events.ndjson"
    native_report = shadow_root / "shadow-report.json"
    native_command = shadow_root / "shadow-command.json"
    trace = _run(
        (python, "-I", guard, validator, "write-trace", native_source),
        cwd=case_root,
        env=environment,
        capture_output=True,
    )
    if trace.stdout != b"shadow-trace-ok\n" or trace.stderr:
        raise RuntimeError(f"{label} native trace setup returned unexpected output")
    native = _run(
        (
            python,
            "-I",
            guard,
            saliencegate,
            "shadow",
            "analyze",
            native_source,
            "--run-id",
            native_run_id,
            "--output",
            native_report,
            "--json",
        ),
        cwd=case_root,
        env=environment,
        capture_output=True,
    )
    if native.stderr:
        raise RuntimeError(f"{label} native Shadow command wrote to stderr")
    _write_private(native_command, native.stdout)
    native_validation = _run(
        (
            python,
            "-I",
            guard,
            validator,
            "validate-report",
            native_report,
            native_command,
            native_run_id,
        ),
        cwd=case_root,
        env=environment,
        capture_output=True,
    )
    if native_validation.stdout != b"shadow-installed-ok\n" or native_validation.stderr:
        raise RuntimeError(f"{label} native Shadow validation returned unexpected output")

    one_call = _run(
        (python, "-I", guard, examples / "one_call.py"),
        cwd=case_root,
        env=environment,
        capture_output=True,
    )
    output = one_call.stdout.decode("utf-8")
    if (
        len(output.splitlines()) != 2
        or "Codex: profile=harbor-codex/v1" not in output
        or "Terminus 2: profile=harbor-terminus-2/v1" not in output
    ):
        raise RuntimeError(f"{label} one-call example returned unexpected output")

    for case in _ATIF_CASES:
        source = shadow_root / f"{case.name}.trajectory.json"
        report = shadow_root / f"{case.name}.report.json"
        command_report = shadow_root / f"{case.name}.command.json"
        _write_private(source, (examples / case.source_name).read_bytes())
        completed = _run(
            (
                python,
                "-I",
                guard,
                saliencegate,
                "shadow",
                "analyze-atif",
                source,
                "--profile",
                case.profile_alias,
                "--run-id",
                case.run_id,
                "--working-directory",
                _WORKING_DIRECTORY,
                "--environment-digest",
                _ENVIRONMENT_DIGEST,
                "--output",
                report,
                "--json",
            ),
            cwd=case_root,
            env=environment,
            capture_output=True,
        )
        if completed.stderr:
            raise RuntimeError(f"{label} {case.name} command wrote to stderr")
        _write_private(command_report, completed.stdout)
        _run(
            (
                python,
                "-I",
                guard,
                validator,
                "validate-public-atif",
                source,
                report,
                command_report,
                case.run_id,
                case.profile_id,
            ),
            cwd=case_root,
            env=environment,
        )


def verify_built_artifacts(
    *,
    dist_dir: Path,
    work_root: Path,
    python_version: str,
) -> None:
    if set(DOCUMENTED_COMMAND_CASES) | SUPPLEMENTAL_ARTIFACT_PROOFS != EXECUTED_COMMAND_CASES:
        raise RuntimeError("documented and executed artifact command cases differ")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to verify built artifacts")
    wheel = _require_one_distribution(dist_dir, ".whl")
    sdist = _require_one_distribution(dist_dir, ".tar.gz")
    package_root = work_root / "package"
    _extract_sdist(sdist, package_root)
    required = (
        "pyproject.toml",
        "uv.lock",
        "scripts/run_without_sockets.py",
        "scripts/smoke_shadow_installed.py",
        "examples/atif-shadow/one_call.py",
        "examples/atif-shadow/codex-minimal.trajectory.json",
        "examples/atif-shadow/terminus-minimal.trajectory.json",
    )
    if any(not (package_root / item).is_file() for item in required):
        raise RuntimeError("source distribution is missing the artifact proof kit")

    runtime_requirements = work_root / "core-runtime-requirements.txt"
    build_constraints = work_root / "build-constraints.txt"
    _run(
        (
            uv,
            "export",
            "--project",
            package_root,
            "--locked",
            "--no-dev",
            "--no-emit-project",
            "--output-file",
            runtime_requirements,
            "--quiet",
        )
    )
    _run(
        (
            uv,
            "export",
            "--project",
            package_root,
            "--locked",
            "--only-dev",
            "--no-emit-project",
            "--output-file",
            build_constraints,
            "--quiet",
        )
    )
    wheel_python = _install_environment(
        uv=uv,
        python_version=python_version,
        environment_root=work_root / "wheel-venv",
        distribution=wheel,
        package_root=package_root,
        runtime_requirements=runtime_requirements,
        build_constraints=build_constraints,
        is_sdist=False,
    )
    sdist_python = _install_environment(
        uv=uv,
        python_version=python_version,
        environment_root=work_root / "sdist-venv",
        distribution=sdist,
        package_root=package_root,
        runtime_requirements=runtime_requirements,
        build_constraints=build_constraints,
        is_sdist=True,
    )
    _prove_documented_cases(
        label="wheel",
        python=wheel_python,
        package_root=package_root,
        work_root=work_root,
    )
    _prove_documented_cases(
        label="sdist",
        python=sdist_python,
        package_root=package_root,
        work_root=work_root,
    )


@contextmanager
def _workspace(requested: Path | None) -> Iterator[Path]:
    if requested is not None:
        requested.mkdir(mode=0o700)
        canonical = requested.resolve(strict=True)
        try:
            yield canonical
        finally:
            shutil.rmtree(canonical)
        return
    created = Path(tempfile.mkdtemp(prefix="saliencegate-artifact-smoke-"))
    temporary = created.resolve(strict=True)
    try:
        yield temporary
    finally:
        shutil.rmtree(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify SalienceGate wheel and sdist behavior")
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--python", default="3.12")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    dist_dir = arguments.dist_dir.resolve(strict=True)
    with _workspace(arguments.work_dir) as work_root:
        verify_built_artifacts(
            dist_dir=dist_dir,
            work_root=work_root,
            python_version=arguments.python,
        )
    print("artifact smoke passed for wheel and sdist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
