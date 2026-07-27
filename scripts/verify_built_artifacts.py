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
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

DOCUMENTED_COMMAND_CASES = {
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
    "capture-status": "saliencegate status",
    "capture-sessions": "saliencegate sessions --limit 20",
}

_ARTIFACT_BLOCK = re.compile(
    r"(?ms)^Artifact-compatible after installation:\s*\n+"
    r"```(?:bash|sh)\s*\n(?P<body>.*?)^```\s*$"
)
_ENVIRONMENT_DIGEST = "e" * 64
_WORKING_DIRECTORY = "/synthetic/workspace"
_PROVIDER_CREDENTIAL_ENVIRONMENT_KEYS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_ORGANIZATION",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT",
        "OPENAI_PROJECT_ID",
    }
)
_CAPTURE_PROVIDER_ALIASES = ("codex", "claude-code", "opencode", "pi")
_POISONED_PROVIDER_CREDENTIAL = "provider-credential-read-must-fail"
_CONTROLLED_CAPTURE_SENTINELS = tuple(
    f"artifact-{provider}-{kind}-sentinel"
    for provider in _CAPTURE_PROVIDER_ALIASES
    for kind in ("native-session", "raw-content")
)
_WINDOWS_INVALID_FILENAME_CHARACTERS = frozenset('<>:"|?*')
_EXPECTED_CONNECTOR_NODE_VERSION = "v22.19.0"
_SOCKET_GUARD_PTH = b"import saliencegate_artifact_socket_guard\n"
_SOCKET_CANARY_SOURCE = r"""
import _socket
import socket
import sys

guard = sys.modules.get("saliencegate_artifact_socket_guard")
if guard is None or getattr(guard, "SOCKET_DENIAL_ACTIVE", False) is not True:
    raise RuntimeError("installed artifact socket guard was not activated")
if getattr(guard, "PROVIDER_CREDENTIAL_DENIAL_ACTIVE", False) is not True:
    raise RuntimeError("installed artifact credential guard was not activated")
for operation in (
    lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM),
    lambda: _socket.socket(socket.AF_INET, socket.SOCK_STREAM),
    lambda: socket.getaddrinfo("localhost", 9),
    lambda: socket.create_connection(("127.0.0.1", 9)),
):
    try:
        operation()
    except guard.ArtifactSocketAccessError:
        pass
    else:
        raise RuntimeError("installed artifact child opened a socket or resolver")
print("installed-artifact-child-network-and-credential-denial-ok")
""".strip()


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
SUPPLEMENTAL_ARTIFACT_PROOFS = frozenset(
    {"offline-demo", "atif-one-call", "capture-lifecycle", "capture-quickstart"}
)
EXECUTED_COMMAND_CASES = frozenset((*DOCUMENTED_COMMAND_CASES, *SUPPLEMENTAL_ARTIFACT_PROOFS))
_WINDOWS_CLI_SHIM = b"from saliencegate.cli import entrypoint\nraise SystemExit(entrypoint())\n"


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
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    if input_bytes is not None and type(input_bytes) is not bytes:
        raise RuntimeError("artifact proof stdin is invalid")
    try:
        environment = _environment_without_provider_credentials(os.environ) if env is None else env
        return subprocess.run(
            tuple(os.fspath(item) for item in command),
            cwd=cwd,
            env=environment,
            check=True,
            capture_output=capture_output,
            input=input_bytes,
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
    for sentinel in (
        _POISONED_PROVIDER_CREDENTIAL,
        *_CONTROLLED_CAPTURE_SENTINELS,
    ):
        text = text.replace(sentinel, "<redacted>")
    limit = 4096
    if len(text) > limit:
        text = text[:limit] + "\n[output truncated]"
    return text


def _normalize_transport_stdout(payload: bytes, *, platform: str | None = None) -> bytes:
    if type(payload) is not bytes:
        raise RuntimeError("artifact proof stdout is invalid")
    selected_platform = os.name if platform is None else platform
    if selected_platform == "posix":
        return payload
    if selected_platform != "nt":
        raise RuntimeError("artifact proof platform is unsupported")
    if b"\r" not in payload:
        return payload
    if not payload.endswith(b"\r\n") or payload.count(b"\r\n") != 1 or b"\r" in payload[:-2]:
        raise RuntimeError("artifact proof stdout has a non-canonical Windows transport ending")
    return payload[:-2] + b"\n"


def _normalize_terminal_stdout(payload: bytes, *, platform: str | None = None) -> bytes:
    if type(payload) is not bytes:
        raise RuntimeError("artifact proof stdout is invalid")
    selected_platform = os.name if platform is None else platform
    if selected_platform not in {"nt", "posix"}:
        raise RuntimeError("artifact proof platform is unsupported")
    if selected_platform == "posix" or b"\r" not in payload:
        return payload
    normalized = payload.replace(b"\r\n", b"\n")
    if b"\r" in normalized:
        raise RuntimeError("artifact proof terminal stdout has a non-canonical Windows ending")
    return normalized


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
    for part in path.parts:
        windows_part = PureWindowsPath(part)
        if (
            windows_part.drive
            or windows_part.root
            or windows_part.is_reserved()
            or part.endswith((" ", "."))
            or any(
                character in _WINDOWS_INVALID_FILENAME_CHARACTERS or ord(character) < 32
                for character in part
            )
        ):
            raise RuntimeError("source distribution contains a Windows-unsafe path")
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


def _publish_installed_private_file(
    path: Path,
    payload: bytes,
    *,
    python: Path,
    guard: Path,
    validator: Path,
    cwd: Path,
    environment: dict[str, str],
    platform: str | None = None,
) -> None:
    selected_platform = os.name if platform is None else platform
    if selected_platform == "posix":
        _write_private(path, payload)
        return
    if selected_platform != "nt":
        raise RuntimeError("artifact proof platform is unsupported")
    completed = _run(
        (python, "-I", guard, validator, "publish-private-file", path),
        cwd=cwd,
        env=environment,
        capture_output=True,
        input_bytes=payload,
    )
    if (
        _normalize_transport_stdout(completed.stdout, platform=selected_platform)
        != b"shadow-private-file-ok\n"
        or completed.stderr
    ):
        raise RuntimeError("installed private file publication returned unexpected output")


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
    versions: set[str] = set()
    for item in lock.get("package", ()):
        if isinstance(item, dict) and item.get("name") == "hatchling":
            version = item.get("version")
            if not isinstance(version, str):
                raise RuntimeError("locked Hatchling version is invalid")
            versions.add(version)
    if len(versions) != 1:
        raise RuntimeError("locked Hatchling version is missing or ambiguous")
    return versions.pop()


def _venv_python(environment_root: Path, *, platform: str | None = None) -> Path:
    selected_platform = os.name if platform is None else platform
    if selected_platform == "nt":
        return environment_root / "Scripts" / "python.exe"
    if selected_platform == "posix":
        return environment_root / "bin" / "python"
    raise RuntimeError("artifact proof platform is unsupported")


def _installed_cli_script(
    *,
    python: Path,
    case_root: Path,
    platform: str | None = None,
) -> Path:
    selected_platform = os.name if platform is None else platform
    if selected_platform == "posix":
        return python.parent / "saliencegate"
    if selected_platform != "nt":
        raise RuntimeError("artifact proof platform is unsupported")
    shim = case_root / "saliencegate-command.py"
    _write_private(shim, _WINDOWS_CLI_SHIM)
    return shim


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
    python = _venv_python(environment_root)
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


def _environment_without_provider_credentials(source: Mapping[str, str]) -> dict[str, str]:
    environment: dict[str, str] = {}
    try:
        for key in source:
            if type(key) is not str:
                raise RuntimeError("artifact environment is invalid")
            if key.upper() in _PROVIDER_CREDENTIAL_ENVIRONMENT_KEYS:
                continue
            value = source[key]
            if type(value) is not str:
                raise RuntimeError("artifact environment is invalid")
            environment[key] = value
    except RuntimeError:
        raise
    except Exception:
        raise RuntimeError("artifact environment is invalid") from None
    return environment


def _private_environment(root: Path, *, python: Path) -> dict[str, str]:
    home = root / "home"
    config = root / "config"
    cache = root / "cache"
    data = root / "data"
    state = root / "state"
    appdata = root / "appdata"
    localappdata = root / "localappdata"
    temporary = root / "tmp"
    for directory in (
        home,
        config,
        cache,
        data,
        state,
        appdata,
        localappdata,
        temporary,
    ):
        directory.mkdir(mode=0o700)
    fake_hosts = root / "fake-hosts"
    fake_hosts.mkdir(mode=0o700)
    if os.name == "nt":
        host_payloads = {
            "codex.cmd": b"@echo off\r\necho codex-cli 0.144.6\r\n",
            "claude.cmd": b"@echo off\r\necho 2.1.204 (Claude Code)\r\n",
        }
    else:
        host_payloads = {
            "codex": b"#!/bin/sh\nprintf '%s\\n' 'codex-cli 0.144.6'\n",
            "claude": b"#!/bin/sh\nprintf '%s\\n' '2.1.204 (Claude Code)'\n",
        }
    for name, payload in host_payloads.items():
        destination = fake_hosts / name
        _write_private(destination, payload)
        destination.chmod(0o700)
    environment = _environment_without_provider_credentials(os.environ)
    environment.update(
        {
            "HOME": os.fspath(home),
            "USERPROFILE": os.fspath(home),
            "PYTHONPATH": "",
            "XDG_CONFIG_HOME": os.fspath(config),
            "XDG_CACHE_HOME": os.fspath(cache),
            "XDG_DATA_HOME": os.fspath(data),
            "XDG_STATE_HOME": os.fspath(state),
            "APPDATA": os.fspath(appdata),
            "LOCALAPPDATA": os.fspath(localappdata),
            "TEMP": os.fspath(temporary),
            "TMP": os.fspath(temporary),
            "TMPDIR": os.fspath(temporary),
            "PATH": os.pathsep.join((os.fspath(fake_hosts), os.fspath(python.parent))),
            **{key: _POISONED_PROVIDER_CREDENTIAL for key in _PROVIDER_CREDENTIAL_ENVIRONMENT_KEYS},
        }
    )
    if os.name == "nt":
        environment["PATHEXT"] = ".COM;.EXE;.BAT;.CMD"
    return environment


def _run_capture_cli(
    *,
    label: str,
    python: Path,
    guard: Path,
    saliencegate: Path,
    arguments: Sequence[str | os.PathLike[str]],
    cwd: Path,
    environment: dict[str, str],
) -> str:
    completed = _run(
        (python, "-I", guard, saliencegate, *arguments),
        cwd=cwd,
        env=environment,
        capture_output=True,
    )
    output = _normalize_terminal_stdout(completed.stdout).decode("utf-8")
    if any(
        sentinel in output
        for sentinel in (_POISONED_PROVIDER_CREDENTIAL, *_CONTROLLED_CAPTURE_SENTINELS)
    ):
        raise RuntimeError(f"{label} capture quickstart exposed sensitive native data")
    if completed.stderr:
        raise RuntimeError(f"{label} capture quickstart command wrote to stderr")
    return output


def _validate_capture_connect_output(value: str, *, prefix: str, label: str) -> None:
    git_results = (
        "Git will surface 1 (0 already tracked)",
        "All 1 project-local managed file(s) are ignored by Git",
        "no Git work tree was detected",
        "Git visibility could not be determined",
    )
    if (
        not value.startswith(prefix)
        or not any(
            managed in value
            for managed in (
                "Review 1 project-local managed file(s)",
                "All 1 project-local managed file(s)",
            )
        )
        or sum(result in value for result in git_results) != 1
        or not value.endswith(".gitignore was not changed.\n")
    ):
        raise RuntimeError(f"{label} capture quickstart connect returned unexpected output")


def _prove_capture_quickstart(
    *,
    label: str,
    python: Path,
    guard: Path,
    private_validator: Path,
    capture_validator: Path,
    saliencegate: Path,
    case_root: Path,
    environment: dict[str, str],
) -> None:
    quickstart_root = case_root / "capture-quickstart"
    quickstart_root.mkdir(mode=0o700)
    project = quickstart_root / "project"
    project.mkdir(mode=0o700)

    dry_run = _run_capture_cli(
        label=label,
        python=python,
        guard=guard,
        saliencegate=saliencegate,
        arguments=("connect", "codex", "--project", project, "--dry-run"),
        cwd=project,
        environment=environment,
    )
    _validate_capture_connect_output(
        dry_run,
        prefix="codex capture: would install; disabled during dry-run.",
        label=label,
    )
    if tuple(project.iterdir()):
        raise RuntimeError(f"{label} capture quickstart dry-run mutated the project")

    connected = _run_capture_cli(
        label=label,
        python=python,
        guard=guard,
        saliencegate=saliencegate,
        arguments=("connect", "codex", "--project", project),
        cwd=project,
        environment=environment,
    )
    _validate_capture_connect_output(
        connected,
        prefix="codex capture: installed; enabled.",
        label=label,
    )

    doctor = _run_capture_cli(
        label=label,
        python=python,
        guard=guard,
        saliencegate=saliencegate,
        arguments=("doctor", "--capture"),
        cwd=project,
        environment=environment,
    )
    if (
        not doctor.startswith("SalienceGate doctor: healthy\n")
        or "[PASS] Passive capture (required): "
        "Capture store and local boundaries passed read-only checks.\n"
        not in doctor
    ):
        raise RuntimeError(f"{label} capture quickstart doctor returned unexpected output")

    fixture = _run(
        (
            python,
            "-I",
            guard,
            capture_validator,
            "emit-codex-fixture",
            project,
            case_root,
        ),
        cwd=project,
        env=environment,
        capture_output=True,
    )
    if (
        _normalize_transport_stdout(fixture.stdout) != b"capture-codex-fixture-ok\n"
        or fixture.stderr
    ):
        raise RuntimeError(f"{label} capture quickstart fixture returned unexpected output")

    status = _run_capture_cli(
        label=label,
        python=python,
        guard=guard,
        saliencegate=saliencegate,
        arguments=("status", "codex", "--project", project),
        cwd=project,
        environment=environment,
    )
    if (
        not status.startswith("Passive capture status\n")
        or "codex: active_observed (sessions=1; quarantined=0;" not in status
    ):
        raise RuntimeError(f"{label} capture quickstart status returned unexpected output")

    sessions = _run_capture_cli(
        label=label,
        python=python,
        guard=guard,
        saliencegate=saliencegate,
        arguments=("sessions", "--limit", "20"),
        cwd=project,
        environment=environment,
    )
    if not sessions.startswith("Captured sessions\n") or "  codex  open  events=3" not in sessions:
        raise RuntimeError(f"{label} capture quickstart sessions returned unexpected output")

    report_directory = project / ".saliencegate" / "reports"
    for directory in (report_directory.parent, report_directory):
        preparation = _run(
            (python, "-I", guard, private_validator, "prepare-private-directory", directory),
            cwd=project,
            env=environment,
            capture_output=True,
        )
        if (
            _normalize_transport_stdout(preparation.stdout) != b"shadow-private-directory-ok\n"
            or preparation.stderr
        ):
            raise RuntimeError(f"{label} capture quickstart private report directory setup failed")
    report_path = report_directory / "capture-report.json"
    report = _run_capture_cli(
        label=label,
        python=python,
        guard=guard,
        saliencegate=saliencegate,
        arguments=("report", "--latest", "--output", report_path),
        cwd=project,
        environment=environment,
    )
    if (
        not report.startswith("Insufficient evidence\nSalienceGate capture report\n")
        or "profile: codex-hooks/v1; compatibility: verified; host version: 0.144.6\n" not in report
        or "decision authority: false; model calls: 0; confirmatory: false\n" not in report
    ):
        raise RuntimeError(f"{label} capture quickstart report returned unexpected output")
    report_validation = _run(
        (
            python,
            "-I",
            guard,
            capture_validator,
            "validate-codex-report",
            report_path,
            case_root,
        ),
        cwd=project,
        env=environment,
        capture_output=True,
    )
    if (
        _normalize_transport_stdout(report_validation.stdout) != b"capture-codex-report-ok\n"
        or report_validation.stderr
    ):
        raise RuntimeError(f"{label} capture quickstart report validation failed")

    disconnected = _run_capture_cli(
        label=label,
        python=python,
        guard=guard,
        saliencegate=saliencegate,
        arguments=("disconnect", "codex", "--project", project),
        cwd=project,
        environment=environment,
    )
    if disconnected != (
        "codex capture: uninstalled; disabled. Existing capture data was retained.\n"
    ):
        raise RuntimeError(f"{label} capture quickstart disconnect returned unexpected output")


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
    environment = _private_environment(case_root, python=python)
    guard = package_root / "scripts" / "run_without_sockets.py"
    import_smoke = package_root / "scripts" / "smoke_package_imports.py"
    validator = package_root / "scripts" / "smoke_shadow_installed.py"
    capture_validator = package_root / "scripts" / "smoke_capture_installed.py"
    examples = package_root / "examples" / "atif-shadow"
    saliencegate = _installed_cli_script(python=python, case_root=case_root)

    preparation = _run(
        (python, "-I", guard, validator, "prepare-private-directory", shadow_root),
        cwd=case_root,
        env=environment,
        capture_output=True,
    )
    if (
        _normalize_transport_stdout(preparation.stdout) != b"shadow-private-directory-ok\n"
        or preparation.stderr
    ):
        raise RuntimeError(f"{label} private directory setup returned unexpected output")

    imported = _run(
        (python, "-I", guard, import_smoke),
        cwd=case_root,
        env=environment,
        capture_output=True,
    )
    imported_stdout = _normalize_transport_stdout(imported.stdout)
    if imported.stderr or re.fullmatch(rb"[0-9]+\.[0-9]+\.[0-9]+\n", imported_stdout) is None:
        raise RuntimeError(f"{label} core import smoke returned unexpected output")

    empty_capture_project = case_root / "capture-empty-state"
    empty_capture_project.mkdir(mode=0o700)
    capture_status = _run(
        (python, "-I", guard, saliencegate, "status"),
        cwd=empty_capture_project,
        env=environment,
        capture_output=True,
    )
    status_output = _normalize_terminal_stdout(capture_status.stdout).decode("utf-8")
    if capture_status.stderr or any(
        f"{provider}: not_installed" not in status_output for provider in _CAPTURE_PROVIDER_ALIASES
    ):
        raise RuntimeError(f"{label} capture status returned unexpected output")
    capture_sessions = _run(
        (python, "-I", guard, saliencegate, "sessions", "--limit", "20"),
        cwd=empty_capture_project,
        env=environment,
        capture_output=True,
    )
    if (
        _normalize_terminal_stdout(capture_sessions.stdout) != b"No captured sessions.\n"
        or capture_sessions.stderr
    ):
        raise RuntimeError(f"{label} capture sessions returned unexpected output")

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
    if _normalize_transport_stdout(trace.stdout) != b"shadow-trace-ok\n" or trace.stderr:
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
    _publish_installed_private_file(
        native_command,
        _normalize_transport_stdout(native.stdout),
        python=python,
        guard=guard,
        validator=validator,
        cwd=case_root,
        environment=environment,
    )
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
    if (
        _normalize_transport_stdout(native_validation.stdout) != b"shadow-installed-ok\n"
        or native_validation.stderr
    ):
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
        _publish_installed_private_file(
            source,
            (examples / case.source_name).read_bytes(),
            python=python,
            guard=guard,
            validator=validator,
            cwd=case_root,
            environment=environment,
        )
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
        _publish_installed_private_file(
            command_report,
            _normalize_transport_stdout(completed.stdout),
            python=python,
            guard=guard,
            validator=validator,
            cwd=case_root,
            environment=environment,
        )
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

    _prove_capture_quickstart(
        label=label,
        python=python,
        guard=guard,
        private_validator=validator,
        capture_validator=capture_validator,
        saliencegate=saliencegate,
        case_root=case_root,
        environment=environment,
    )

    capture_lifecycle = _run(
        (
            python,
            "-I",
            guard,
            capture_validator,
            case_root / "capture-lifecycle",
        ),
        cwd=case_root,
        env=environment,
        capture_output=True,
    )
    if (
        _normalize_transport_stdout(capture_lifecycle.stdout) != b"capture-installed-artifact-ok\n"
        or capture_lifecycle.stderr
    ):
        raise RuntimeError(f"{label} capture lifecycle returned unexpected output")


def _resolve_exact_connector_node(command: str) -> Path:
    if type(command) is not str or not command or "\0" in command:
        raise RuntimeError("installed connector proof Node.js command is invalid")
    selected = shutil.which(command)
    if selected is None:
        raise RuntimeError("installed connector proof could not resolve Node.js")
    try:
        resolved = Path(selected).resolve(strict=True)
        metadata = resolved.lstat()
    except OSError:
        raise RuntimeError("installed connector proof could not resolve Node.js") from None
    if not stat.S_ISREG(metadata.st_mode) or (
        os.name == "posix" and not os.access(resolved, os.X_OK)
    ):
        raise RuntimeError("installed connector proof could not resolve Node.js")
    completed = _run((resolved, "--version"), capture_output=True)
    if (
        _normalize_transport_stdout(completed.stdout)
        != f"{_EXPECTED_CONNECTOR_NODE_VERSION}\n".encode()
        or completed.stderr
    ):
        raise RuntimeError("installed connector proof requires exact Node.js 22.19.0")
    return resolved


def _install_artifact_socket_guard(*, python: Path, package_root: Path) -> None:
    source = package_root / "scripts" / "artifact_socket_guard.py"
    try:
        source_metadata = source.lstat()
    except OSError:
        raise RuntimeError("source distribution is missing the child socket guard") from None
    if source.is_symlink() or not stat.S_ISREG(source_metadata.st_mode):
        raise RuntimeError("source distribution child socket guard is invalid")
    location = _run(
        (
            python,
            "-I",
            "-c",
            ("import sys,sysconfig; print(sys.prefix); print(sysconfig.get_path('purelib'))"),
        ),
        capture_output=True,
    )
    if location.stderr:
        raise RuntimeError("installed artifact purelib discovery wrote to stderr")
    try:
        prefix_raw, purelib_raw = location.stdout.decode("utf-8", errors="strict").splitlines()
        prefix = Path(prefix_raw).resolve(strict=True)
        purelib = Path(purelib_raw).resolve(strict=True)
        python_prefix = python.parent.parent.resolve(strict=True)
        purelib.relative_to(prefix)
    except (OSError, UnicodeError, ValueError):
        raise RuntimeError("installed artifact purelib discovery is invalid") from None
    if prefix != python_prefix or purelib.is_symlink() or not purelib.is_dir():
        raise RuntimeError("installed artifact purelib discovery is invalid")
    _write_private(
        purelib / "saliencegate_artifact_socket_guard.py",
        source.read_bytes(),
    )
    _write_private(
        purelib / "saliencegate_artifact_socket_guard.pth",
        _SOCKET_GUARD_PTH,
    )


def _prove_artifact_socket_guard(
    *,
    python: Path,
    cwd: Path,
    environment: dict[str, str],
) -> None:
    canary_environment = _environment_without_provider_credentials(environment)
    canary = _run(
        (python, "-c", _SOCKET_CANARY_SOURCE),
        cwd=cwd,
        env=canary_environment,
        capture_output=True,
    )
    if (
        _normalize_transport_stdout(canary.stdout)
        != b"installed-artifact-child-network-and-credential-denial-ok\n"
        or canary.stderr
    ):
        raise RuntimeError("installed artifact child network and credential denial canary failed")


def _prove_installed_connector_e2e(
    *,
    label: str,
    python: Path,
    package_root: Path,
    work_root: Path,
    node: Path,
) -> None:
    case_root = work_root / label / "launcher path & data"
    case_root.mkdir(mode=0o700, parents=True)
    environment = _private_environment(case_root, python=python)
    environment.update(
        {
            "SALIENCEGATE_ARTIFACT_SOCKET_DENIAL": "1",
            "SALIENCEGATE_ARTIFACT_SOCKET_STARTUP_LOG": os.fspath(
                case_root / "socket-denial-startups.log"
            ),
        }
    )
    _prove_artifact_socket_guard(
        python=python,
        cwd=case_root,
        environment=environment,
    )
    guard = package_root / "scripts" / "run_without_sockets.py"
    network_guard = package_root / "connectors" / "scripts" / "deny-network.mjs"
    capture_validator = package_root / "scripts" / "smoke_capture_installed.py"
    lifecycle_environment = dict(environment)
    del lifecycle_environment["SALIENCEGATE_ARTIFACT_SOCKET_DENIAL"]
    lifecycle = _run(
        (
            python,
            "-I",
            guard,
            capture_validator,
            "installed-e2e",
            case_root / "capture-installed-connectors",
            node,
            network_guard,
        ),
        cwd=case_root,
        env=lifecycle_environment,
        capture_output=True,
    )
    if (
        _normalize_transport_stdout(lifecycle.stdout) != b"capture-installed-connectors-e2e-ok\n"
        or lifecycle.stderr
    ):
        raise RuntimeError(f"{label} installed connector end-to-end proof failed")


def verify_built_artifacts(
    *,
    dist_dir: Path,
    work_root: Path,
    python_version: str,
    connector_node: str | None = None,
    capture_connectors_only: bool = False,
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
        "scripts/artifact_socket_guard.py",
        "scripts/smoke_package_imports.py",
        "scripts/smoke_capture_installed.py",
        "scripts/smoke_shadow_installed.py",
        "connectors/scripts/deny-network.mjs",
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
    if capture_connectors_only:
        if connector_node is None:
            raise RuntimeError("installed connector proof requires Node.js")
        node = _resolve_exact_connector_node(connector_node)
        _install_artifact_socket_guard(python=wheel_python, package_root=package_root)
        _install_artifact_socket_guard(python=sdist_python, package_root=package_root)
        _prove_installed_connector_e2e(
            label="wheel-connectors",
            python=wheel_python,
            package_root=package_root,
            work_root=work_root,
            node=node,
        )
        _prove_installed_connector_e2e(
            label="sdist-connectors",
            python=sdist_python,
            package_root=package_root,
            work_root=work_root,
            node=node,
        )
        return
    if connector_node is not None:
        raise RuntimeError("--node requires --capture-connectors-only")
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
def _workspace(
    requested: Path | None,
    *,
    default_parent: Path | None = None,
) -> Iterator[Path]:
    if requested is not None:
        requested.mkdir(mode=0o700)
        canonical = requested.resolve(strict=True)
        try:
            yield canonical
        finally:
            shutil.rmtree(canonical)
        return
    canonical_parent = None if default_parent is None else default_parent.resolve(strict=True)
    if canonical_parent is not None and not canonical_parent.is_dir():
        raise RuntimeError("artifact smoke workspace parent is unavailable")
    created = Path(
        tempfile.mkdtemp(prefix="saliencegate-artifact-smoke-")
        if canonical_parent is None
        else tempfile.mkdtemp(
            prefix="saliencegate-artifact-smoke-",
            dir=canonical_parent,
        )
    )
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
    parser.add_argument("--node")
    parser.add_argument("--capture-connectors-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    dist_dir = arguments.dist_dir.resolve(strict=True)
    with _workspace(arguments.work_dir, default_parent=dist_dir.parent) as work_root:
        verify_built_artifacts(
            dist_dir=dist_dir,
            work_root=work_root,
            python_version=arguments.python,
            connector_node=arguments.node,
            capture_connectors_only=arguments.capture_connectors_only,
        )
    if arguments.capture_connectors_only:
        print("installed connector smoke passed for wheel and sdist")
    else:
        print("artifact smoke passed for wheel and sdist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
