"""Passive, content-free Claude Code lifecycle capture adapter."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePath, PureWindowsPath
from typing import TYPE_CHECKING, Final

from pydantic import AnyUrl

from saliencegate.capture.adapters import (
    CAPTURE_ADAPTER_PROTOCOL_VERSION,
    CaptureAdapterCapabilities,
)
from saliencegate.capture.capabilities import (
    CaptureEventCapability,
    CaptureProfile,
    CompatibilityStatus,
    capture_capability_digest,
    capture_profile,
    classify_capture_compatibility,
)
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.capture.locations import resolve_capture_store_locations
from saliencegate.capture.publication import authenticate_capture_intake
from saliencegate.capture.schema import (
    CAPTURE_NATIVE_JSON_LIMITS,
    CaptureIntake,
    read_bounded_json,
    validate_capture_intake,
)
from saliencegate.domain import canonical_json
from saliencegate.integrations.config_files import (
    ConfigFileError,
    ConfigSyntax,
    OwnedConfigSpec,
    plan_owned_config_install,
    read_config_bytes,
)
from saliencegate.integrations.environment import environment_without_provider_credentials
from saliencegate.integrations.registry import (
    ProviderInstallationKind,
    ProviderInstallationSpec,
)

if TYPE_CHECKING:
    from saliencegate.integrations.hook import CaptureHookDependencies

CLAUDE_CODE_HOST_VERSION: Final = "2.1.204"
CLAUDE_CODE_PROFILE: Final = CaptureProfile.CLAUDE_CODE_HOOKS_V1
CLAUDE_CODE_CONFIG_MARKER: Final = "saliencegate-owned:claude-code-hooks-v1"
CLAUDE_CODE_HOOK_EVENTS: Final = (
    "SessionStart",
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "PostToolBatch",
    "PermissionDenied",
    "SubagentStart",
    "SubagentStop",
    "Stop",
    "StopFailure",
    "SessionEnd",
)
MAX_CLAUDE_CODE_VERSION_OUTPUT_BYTES: Final = 4_096
CLAUDE_CODE_VERSION_TIMEOUT_SECONDS: Final = 2.0

_CONNECTION_ID: Final = re.compile(r"^[a-z0-9][a-z0-9._:-]{11,127}$")
_HOST_VERSION: Final = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
_VERSION_OUTPUT: Final = re.compile(
    rb"((?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)) "
    rb"\(Claude Code\)(?:\r?\n)?"
)
_ZERO_TAG: Final = "0" * 64
_AUDITED_VERSION: Final = (2, 1, 204)


class ClaudeCodeIntegrationError(ValueError):
    """A Claude Code boundary failed without disclosing provider-owned values."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("Claude Code capture integration is invalid")


def _supported_version_parts(host_version: object) -> tuple[int, int, int]:
    if type(host_version) is not str or _HOST_VERSION.fullmatch(host_version) is None:
        raise ClaudeCodeIntegrationError()
    try:
        parts = tuple(int(part) for part in host_version.split("."))
    except Exception:
        raise ClaudeCodeIntegrationError() from None
    generation = parts[2] - _AUDITED_VERSION[2] + 1
    if (
        len(parts) != 3
        or parts[:2] != _AUDITED_VERSION[:2]
        or parts < _AUDITED_VERSION
        or not 1 <= generation <= 1_000_000
    ):
        raise ClaudeCodeIntegrationError()
    return parts


@dataclass(frozen=True, slots=True)
class ClaudeCodeVersionProbe:
    """Bounded, content-free result of one exact Claude Code version probe."""

    host_version: str
    compatibility: CompatibilityStatus

    def __post_init__(self) -> None:
        parts = _supported_version_parts(self.host_version)
        expected = (
            CompatibilityStatus.VERIFIED
            if parts == _AUDITED_VERSION
            else CompatibilityStatus.SCHEMA_COMPATIBLE_UNVERIFIED_VERSION
        )
        if (
            type(self.compatibility) is not CompatibilityStatus
            or self.compatibility is not expected
        ):
            raise ClaudeCodeIntegrationError()


VersionRunner = Callable[..., subprocess.CompletedProcess[bytes]]


def _read_available_version_bytes(descriptor: int, limit: int) -> bytes | None:
    """Read one available pipe chunk without ever waiting for a producer."""

    if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        import ctypes
        import msvcrt

        available = ctypes.c_ulong()
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        if not kernel32.PeekNamedPipe(
            ctypes.c_void_p(msvcrt.get_osfhandle(descriptor)),  # type: ignore[attr-defined]
            None,
            0,
            None,
            ctypes.byref(available),
            None,
        ):
            return b""
        if available.value == 0:
            return None
        return os.read(descriptor, min(limit, available.value))
    try:
        return os.read(descriptor, limit)
    except (BlockingIOError, InterruptedError):
        return None


def _bounded_version_runner(
    command: tuple[str, ...],
    *,
    input: bytes,
    capture_output: bool,
    check: bool,
    timeout: float,
    env: Mapping[str, str],
    cwd: str,
) -> subprocess.CompletedProcess[bytes]:
    """Run a version command while retaining at most one bounded stream chunk."""

    if (
        type(command) is not tuple
        or not command
        or input != b""
        or capture_output is not True
        or check is not False
        or timeout != CLAUDE_CODE_VERSION_TIMEOUT_SECONDS
        or type(cwd) is not str
        or not Path(cwd).is_absolute()
        or ".." in Path(cwd).parts
        or not isinstance(env, Mapping)
        or any(type(key) is not str or type(value) is not str for key, value in env.items())
    ):
        raise ClaudeCodeIntegrationError()
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(env),
            cwd=cwd,
            start_new_session=os.name == "posix",
        )
        if process.stdout is None or process.stderr is None:
            raise ClaudeCodeIntegrationError()
        streams = (process.stdout, process.stderr)
        descriptors = tuple(stream.fileno() for stream in streams)
        if os.name != "nt":
            for descriptor in descriptors:
                os.set_blocking(descriptor, False)
        output = (bytearray(), bytearray())
        open_streams = [True, True]
        deadline = time.monotonic() + timeout
        while True:
            progressed = False
            for index, descriptor in enumerate(descriptors):
                if not open_streams[index]:
                    continue
                remaining = MAX_CLAUDE_CODE_VERSION_OUTPUT_BYTES + 1 - len(output[index])
                chunk = _read_available_version_bytes(descriptor, remaining)
                if chunk is None:
                    continue
                if chunk:
                    output[index].extend(chunk)
                    progressed = True
                    if len(output[index]) > MAX_CLAUDE_CODE_VERSION_OUTPUT_BYTES:
                        _terminate_version_process(process)
                        raise ClaudeCodeIntegrationError()
                else:
                    open_streams[index] = False
                    progressed = True
            returncode = process.poll()
            if returncode is not None and not any(open_streams):
                return subprocess.CompletedProcess(
                    command,
                    returncode,
                    bytes(output[0]),
                    bytes(output[1]),
                )
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                _terminate_version_process(process)
                raise subprocess.TimeoutExpired(command, timeout)
            if not progressed:
                time.sleep(min(0.005, remaining_time))
    finally:
        if process is not None and process.poll() is None:
            _terminate_version_process(process)
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()


def _terminate_version_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate the isolated probe process group without retaining child output pipes."""

    try:
        if os.name == "posix":
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:  # pragma: no cover - exercised by native Windows R01
            with suppress(ClaudeCodeIntegrationError, OSError, subprocess.SubprocessError):
                subprocess.run(
                    (
                        str(_trusted_windows_taskkill()),
                        "/PID",
                        str(process.pid),
                        "/T",
                        "/F",
                    ),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=0.75,
                )
            if process.poll() is None:
                process.kill()
    except OSError:
        return
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive platform fallback
        if process.poll() is None:
            with suppress(OSError):
                process.kill()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=0.25)
    except OSError:
        return


def _exact_executable(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
        raise ClaudeCodeIntegrationError()
    try:
        if path.is_symlink():
            raise ClaudeCodeIntegrationError()
        resolved = path.resolve(strict=True)
        metadata = resolved.lstat()
        if (
            path != resolved
            or not stat.S_ISREG(metadata.st_mode)
            or (os.name == "posix" and not os.access(resolved, os.X_OK))
        ):
            raise ClaudeCodeIntegrationError()
        return resolved
    except ClaudeCodeIntegrationError:
        raise
    except Exception:
        raise ClaudeCodeIntegrationError() from None


def _windows_shim_version_command(
    executable: PurePath,
    powershell: PurePath,
) -> tuple[str, ...]:
    try:
        shim = PureWindowsPath(os.fspath(executable))
        shell = PureWindowsPath(os.fspath(powershell))
        if (
            not shim.is_absolute()
            or shim.suffix.lower() not in {".bat", ".cmd"}
            or not shell.is_absolute()
            or shell.suffix.lower() != ".exe"
            or shell.name.lower() not in {"powershell.exe", "pwsh.exe"}
            or re.fullmatch(r"[A-Za-z]:\\", shim.anchor) is None
            or re.fullmatch(r"[A-Za-z]:\\", shell.anchor) is None
        ):
            raise ClaudeCodeIntegrationError()
        encoded = base64.b64encode(os.fspath(executable).encode("utf-8")).decode("ascii")
        script = (
            "$ErrorActionPreference='SilentlyContinue';try{& "
            "([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('"
            + encoded
            + "'))) '--version';exit $LASTEXITCODE}catch{exit 1}"
        )
        return (
            os.fspath(powershell),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        )
    except ClaudeCodeIntegrationError:
        raise
    except Exception:
        raise ClaudeCodeIntegrationError() from None


def probe_claude_code_version(
    executable: Path,
    *,
    runner: VersionRunner = _bounded_version_runner,
    environ: Mapping[str, str] | None = None,
) -> ClaudeCodeVersionProbe:
    """Probe one trusted local Claude executable with strict output and time bounds."""

    try:
        if not callable(runner):
            raise ClaudeCodeIntegrationError()
        environment = environment_without_provider_credentials(environ)
        selected = _exact_executable(executable)
        command: tuple[str, ...] = (str(selected), "--version")
        if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
            suffix = selected.suffix.lower()
            if suffix in {".bat", ".cmd"}:
                command = _windows_shim_version_command(
                    selected,
                    _trusted_windows_powershell(),
                )
            elif suffix not in {".com", ".exe"}:
                raise ClaudeCodeIntegrationError()
        completed = runner(
            command,
            input=b"",
            capture_output=True,
            check=False,
            timeout=CLAUDE_CODE_VERSION_TIMEOUT_SECONDS,
            env=environment,
            cwd=str(selected.parent),
        )
        if (
            type(completed.returncode) is not int
            or completed.returncode != 0
            or type(completed.stdout) is not bytes
            or type(completed.stderr) is not bytes
            or len(completed.stdout) > MAX_CLAUDE_CODE_VERSION_OUTPUT_BYTES
            or len(completed.stderr) > MAX_CLAUDE_CODE_VERSION_OUTPUT_BYTES
        ):
            raise ClaudeCodeIntegrationError()
        match = _VERSION_OUTPUT.fullmatch(completed.stdout)
        if match is None:
            raise ClaudeCodeIntegrationError()
        host_version = match.group(1).decode("ascii")
        _supported_version_parts(host_version)
        compatibility = (
            CompatibilityStatus.VERIFIED
            if host_version == CLAUDE_CODE_HOST_VERSION
            else CompatibilityStatus.SCHEMA_COMPATIBLE_UNVERIFIED_VERSION
        )
        return ClaudeCodeVersionProbe(host_version=host_version, compatibility=compatibility)
    except ClaudeCodeIntegrationError:
        raise
    except Exception:
        raise ClaudeCodeIntegrationError() from None


def _claude_executable_candidates(
    configured_path: str,
    *,
    windows_pathext: str | None,
    windows: bool,
    cwd: PurePath,
) -> tuple[PurePath, ...]:
    if (
        type(configured_path) is not str
        or len(configured_path) > 65_536
        or (windows_pathext is not None and type(windows_pathext) is not str)
    ):
        raise ClaudeCodeIntegrationError()
    separator = ";" if windows else ":"
    components = configured_path.split(separator)
    if len(components) > 2_048:
        raise ClaudeCodeIntegrationError()
    path_type = PureWindowsPath if windows else Path
    current = path_type(os.fspath(cwd))
    if not current.is_absolute():
        raise ClaudeCodeIntegrationError()
    if windows:
        extension_source = ".COM;.EXE;.BAT;.CMD" if windows_pathext is None else windows_pathext
        if len(extension_source) > 4_096:
            raise ClaudeCodeIntegrationError()
        extensions = tuple(
            dict.fromkeys(
                extension.upper() for extension in extension_source.split(";") if extension
            )
        )
        if (
            not extensions
            or len(extensions) > 64
            or any(
                re.fullmatch(r"\.[A-Z0-9_+-]{1,16}", extension) is None for extension in extensions
            )
        ):
            raise ClaudeCodeIntegrationError()
    else:
        extensions = ("",)

    candidates: list[PurePath] = []
    for raw_component in components:
        component = raw_component
        if len(component) >= 2 and component.startswith('"') and component.endswith('"'):
            component = component[1:-1]
        if "\x00" in component:
            raise ClaudeCodeIntegrationError()
        directory = current if component == "" else path_type(component)
        if not directory.is_absolute():
            directory = current / directory
        candidates.extend(directory / f"claude{extension}" for extension in extensions)
    if len(candidates) > 8_192:
        raise ClaudeCodeIntegrationError()
    return tuple(candidates)


def _resolve_claude_executable(
    configured_path: str,
    *,
    environment: Mapping[str, str],
) -> Path:
    candidates = _claude_executable_candidates(
        configured_path,
        windows_pathext=environment.get("PATHEXT") if os.name == "nt" else None,
        windows=os.name == "nt",
        cwd=Path.cwd(),
    )
    for candidate_name in candidates:
        candidate = Path(os.fspath(candidate_name))
        try:
            metadata = candidate.stat()
            if not stat.S_ISREG(metadata.st_mode) or (
                os.name == "posix" and not os.access(candidate, os.X_OK)
            ):
                continue
            return candidate.resolve(strict=True)
        except OSError:
            continue
    raise ClaudeCodeIntegrationError()


def probe_claude_code_environment(
    *,
    environ: Mapping[str, str] | None = None,
) -> ClaudeCodeVersionProbe:
    """Resolve and optionally probe the Claude executable from one explicit environment."""

    try:
        environment = environment_without_provider_credentials(environ)
        configured_path = environment.get("PATH")
        if (configured_path is None and environ is not None) or (
            configured_path is not None and type(configured_path) is not str
        ):
            raise ClaudeCodeIntegrationError()
        selected_path = os.defpath if configured_path is None else configured_path
        return probe_claude_code_version(
            _resolve_claude_executable(selected_path, environment=environment),
            environ=environment,
        )
    except ClaudeCodeIntegrationError:
        raise
    except Exception:
        raise ClaudeCodeIntegrationError() from None


def _claude_code_hook_fragment(
    launcher: PurePath,
    *,
    windows_powershell: PurePath | None = None,
) -> bytes:
    """Render exact exec-form handlers without changing permissions or trust."""

    try:
        if (
            not isinstance(launcher, PurePath)
            or not launcher.is_absolute()
            or ".." in launcher.parts
            or not launcher.name
        ):
            raise ClaudeCodeIntegrationError()
        launcher_raw = os.fspath(launcher)
        if not launcher_raw or "\x00" in launcher_raw:
            raise ClaudeCodeIntegrationError()
        command = launcher_raw
        windows_encoded_launcher: str | None = None
        if windows_powershell is not None:
            windows_launcher = PureWindowsPath(launcher_raw)
            windows_shell = PureWindowsPath(os.fspath(windows_powershell))
            if (
                not windows_launcher.is_absolute()
                or windows_launcher.suffix.lower() != ".cmd"
                or not windows_shell.is_absolute()
                or windows_shell.suffix.lower() != ".exe"
                or windows_shell.name.lower() not in {"powershell.exe", "pwsh.exe"}
                or re.fullmatch(r"[A-Za-z]:\\", windows_launcher.anchor) is None
                or re.fullmatch(r"[A-Za-z]:\\", windows_shell.anchor) is None
            ):
                raise ClaudeCodeIntegrationError()
            windows_encoded_launcher = base64.b64encode(launcher_raw.encode("utf-8")).decode(
                "ascii"
            )
            command = os.fspath(windows_powershell)
        hooks: dict[str, object] = {}
        for index, event_name in enumerate(CLAUDE_CODE_HOOK_EVENTS):
            owned_arguments = [f"saliencegate-event:{event_name}"]
            if index == 0:
                owned_arguments.insert(0, CLAUDE_CODE_CONFIG_MARKER)
            handler_arguments = owned_arguments
            if windows_encoded_launcher is not None:
                invocation_token = owned_arguments[0]
                uniqueness = f"$null='{owned_arguments[-1]}';" if len(owned_arguments) > 1 else ""
                script = (
                    "$ErrorActionPreference='SilentlyContinue';"
                    + uniqueness
                    + "try{& ([Text.Encoding]::UTF8.GetString("
                    "[Convert]::FromBase64String('"
                    + windows_encoded_launcher
                    + "'))) '"
                    + invocation_token
                    + "' *> $null}catch{};exit 0"
                )
                handler_arguments = [
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    script,
                ]
            hooks[event_name] = [
                {
                    "hooks": [
                        {
                            "args": handler_arguments,
                            "command": command,
                            "timeout": 3,
                            "type": "command",
                        }
                    ]
                }
            ]
        encoded = canonical_json({"hooks": hooks})
        if encoded[:1] != b"{" or encoded[-1:] != b"}":
            raise ClaudeCodeIntegrationError()
        return encoded[1:-1]
    except ClaudeCodeIntegrationError:
        raise
    except Exception:
        raise ClaudeCodeIntegrationError() from None


def _trusted_windows_powershell() -> Path:
    try:  # pragma: no cover - exercised by native Windows R01
        import ctypes

        buffer = ctypes.create_unicode_buffer(32_768)
        length = ctypes.windll.kernel32.GetSystemDirectoryW(  # type: ignore[attr-defined]
            buffer,
            len(buffer),
        )
        if not 0 < length < len(buffer):
            raise ClaudeCodeIntegrationError()
        candidate = (Path(buffer.value) / "WindowsPowerShell" / "v1.0" / "powershell.exe").resolve(
            strict=True
        )
        metadata = candidate.lstat()
        if not stat.S_ISREG(metadata.st_mode) or candidate.is_symlink():
            raise ClaudeCodeIntegrationError()
        return candidate
    except ClaudeCodeIntegrationError:
        raise
    except Exception:
        raise ClaudeCodeIntegrationError() from None


def _trusted_windows_taskkill() -> Path:
    try:  # pragma: no cover - exercised by native Windows R01
        import ctypes

        buffer = ctypes.create_unicode_buffer(32_768)
        length = ctypes.windll.kernel32.GetSystemDirectoryW(  # type: ignore[attr-defined]
            buffer,
            len(buffer),
        )
        if not 0 < length < len(buffer):
            raise ClaudeCodeIntegrationError()
        candidate = (Path(buffer.value) / "taskkill.exe").resolve(strict=True)
        metadata = candidate.lstat()
        if not stat.S_ISREG(metadata.st_mode) or candidate.is_symlink():
            raise ClaudeCodeIntegrationError()
        return candidate
    except ClaudeCodeIntegrationError:
        raise
    except Exception:
        raise ClaudeCodeIntegrationError() from None


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ClaudeCodeIntegrationError()
        document[key] = value
    return document


def _reject_json_constant(_value: str) -> None:
    raise ClaudeCodeIntegrationError()


def _validate_existing_hook_groups(document: Mapping[str, object]) -> None:
    if "hooks" not in document:
        return
    hooks = document["hooks"]
    if type(hooks) is not dict:
        raise ClaudeCodeIntegrationError()
    required_strings = {
        "command": ("command",),
        "prompt": ("prompt",),
        "agent": ("prompt",),
        "http": ("url",),
        "mcp_tool": ("server", "tool"),
    }
    optional_strings = {
        "command": ("if", "rewakeMessage", "rewakeSummary", "shell", "statusMessage"),
        "prompt": ("if", "model", "statusMessage"),
        "agent": ("if", "model", "statusMessage"),
        "http": ("if", "statusMessage"),
        "mcp_tool": ("if", "statusMessage"),
    }
    optional_booleans = {
        "command": ("async", "asyncRewake", "once"),
        "prompt": ("continueOnBlock", "once"),
        "agent": ("once",),
        "http": ("once",),
        "mcp_tool": ("once",),
    }
    for event_name in CLAUDE_CODE_HOOK_EVENTS:
        if event_name not in hooks:
            continue
        groups = hooks[event_name]
        if type(groups) is not list:
            raise ClaudeCodeIntegrationError()
        for group in groups:
            if type(group) is not dict or (
                "matcher" in group and type(group["matcher"]) is not str
            ):
                raise ClaudeCodeIntegrationError()
            handlers = group.get("hooks")
            if type(handlers) is not list:
                raise ClaudeCodeIntegrationError()
            for handler in handlers:
                if type(handler) is not dict or type(handler.get("type")) is not str:
                    raise ClaudeCodeIntegrationError()
                handler_type = handler["type"]
                fields = required_strings.get(handler_type)
                if fields is None or any(type(handler.get(field)) is not str for field in fields):
                    raise ClaudeCodeIntegrationError()
                if any(
                    field in handler and type(handler[field]) is not str
                    for field in optional_strings[handler_type]
                ) or any(
                    field in handler and type(handler[field]) is not bool
                    for field in optional_booleans[handler_type]
                ):
                    raise ClaudeCodeIntegrationError()
                if handler_type == "command" and any(
                    field in handler and not handler[field]
                    for field in ("rewakeMessage", "rewakeSummary")
                ):
                    raise ClaudeCodeIntegrationError()
                if "timeout" in handler:
                    timeout = handler["timeout"]
                    try:
                        finite_timeout = math.isfinite(float(timeout))
                    except (OverflowError, TypeError, ValueError):
                        raise ClaudeCodeIntegrationError() from None
                    if type(timeout) not in {int, float} or timeout <= 0 or not finite_timeout:
                        raise ClaudeCodeIntegrationError()
                if handler_type == "command" and "args" in handler:
                    arguments = handler["args"]
                    if type(arguments) is not list or any(
                        type(argument) is not str for argument in arguments
                    ):
                        raise ClaudeCodeIntegrationError()
                if (
                    handler_type == "command"
                    and "shell" in handler
                    and handler["shell"] not in {"bash", "powershell"}
                ):
                    raise ClaudeCodeIntegrationError()
                if handler_type == "http" and "headers" in handler:
                    headers = handler["headers"]
                    if type(headers) is not dict or any(
                        type(key) is not str or type(value) is not str
                        for key, value in headers.items()
                    ):
                        raise ClaudeCodeIntegrationError()
                if handler_type == "http" and "allowedEnvVars" in handler:
                    allowed = handler["allowedEnvVars"]
                    if type(allowed) is not list or any(
                        type(value) is not str for value in allowed
                    ):
                        raise ClaudeCodeIntegrationError()
                if (
                    handler_type == "mcp_tool"
                    and "input" in handler
                    and type(handler["input"]) is not dict
                ):
                    raise ClaudeCodeIntegrationError()
                if handler_type == "http":
                    url = handler["url"]
                    assert type(url) is str
                    try:
                        AnyUrl(url)
                    except ValueError:
                        raise ClaudeCodeIntegrationError() from None


def _hooks_explicitly_disabled(source: bytes) -> bool:
    try:
        document = json.loads(
            source,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
        if type(document) is not dict:
            raise ClaudeCodeIntegrationError()
        _validate_existing_hook_groups(document)
        if "disableAllHooks" not in document:
            return False
        value = document["disableAllHooks"]
        if type(value) is not bool:
            raise ClaudeCodeIntegrationError()
        return value
    except ClaudeCodeIntegrationError:
        raise
    except Exception:
        raise ClaudeCodeIntegrationError() from None


def _validate_project_hook_policy(
    spec: ProviderInstallationSpec,
    environment: Mapping[str, str],
) -> None:
    """Reject collisions before probing a host or creating runtime state."""

    try:
        config_path = spec.config_path
        config = spec.config
        if config_path is None or config is None:
            raise ClaudeCodeIntegrationError()
        configured_home = environment.get("HOME")
        if configured_home is not None and type(configured_home) is not str:
            raise ClaudeCodeIntegrationError()
        home = Path.home() if configured_home is None else Path(configured_home)
        policy_paths = tuple(
            dict.fromkeys(
                (
                    home / ".claude" / "settings.json",
                    spec.project_root / ".claude" / "settings.json",
                    config_path,
                )
            )
        )
        source: bytes | None = None
        for policy_path in policy_paths:
            try:
                parent = policy_path.parent.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode):
                raise ClaudeCodeIntegrationError()
            candidate = read_config_bytes(policy_path)
            if candidate is not None and _hooks_explicitly_disabled(candidate):
                raise ClaudeCodeIntegrationError()
            if policy_path == config_path:
                source = candidate
        if source is None:
            return
        marker = config.marker.encode("ascii")
        if marker not in source:
            plan_owned_config_install(source, config)
            return
        if source.count(marker) != 1:
            raise ClaudeCodeIntegrationError()
        try:
            receipt = spec.receipt_path.lstat()
        except FileNotFoundError:
            raise ClaudeCodeIntegrationError() from None
        if stat.S_ISLNK(receipt.st_mode) or not stat.S_ISREG(receipt.st_mode):
            raise ClaudeCodeIntegrationError()
    except ClaudeCodeIntegrationError:
        raise
    except Exception:
        raise ClaudeCodeIntegrationError() from None


def provider_installation_spec(
    project: Path,
    *,
    environ: Mapping[str, str] | None = None,
    host_version: str = CLAUDE_CODE_HOST_VERSION,
    probe_host: bool = False,
) -> ProviderInstallationSpec:
    """Describe one reversible project-local Claude Code command-hook installation."""

    try:
        if (
            not isinstance(project, Path)
            or not project.is_absolute()
            or ".." in project.parts
            or not project.is_dir()
            or project.is_symlink()
            or type(probe_host) is not bool
        ):
            raise ClaudeCodeIntegrationError()
        version_parts = _supported_version_parts(host_version)
        environment = environment_without_provider_credentials(environ)
        configured_home = environment.get("HOME")
        if configured_home is not None and type(configured_home) is not str:
            raise ClaudeCodeIntegrationError()
        config_path = project / ".claude" / "settings.local.json"
        home = Path.home() if configured_home is None else Path(configured_home)
        locations = resolve_capture_store_locations(environ=environment, home=home)
        project_locator = hashlib.sha256(
            canonical_json(
                {
                    "schema_version": "claude-code-installation-location/v1",
                    "project_root": os.fspath(project),
                }
            )
        ).hexdigest()
        operational = locations.state_directory / "integrations" / project_locator / "claude-code"
        launcher = operational / ("capture-hook.cmd" if os.name == "nt" else "capture-hook")
        placeholder = (
            b"@exit /b 0\r\n"
            if os.name == "nt"  # pragma: no cover - exercised by native Windows R01
            else b"#!/bin/sh\nexit 0\n"
        )
        manifest = capture_profile(CLAUDE_CODE_PROFILE)
        windows_powershell = _trusted_windows_powershell() if os.name == "nt" else None

        def build_spec(
            selected_host_version: str,
            selected_version_parts: tuple[int, int, int],
        ) -> ProviderInstallationSpec:
            return ProviderInstallationSpec(
                installation_kind=ProviderInstallationKind.COMMAND_HOOK,
                provider_id="claude-code",
                profile=CLAUDE_CODE_PROFILE,
                host_version=selected_host_version,
                project_root=project,
                config_path=config_path,
                receipt_path=operational / "receipt.json",
                journal_path=operational / "journal.json",
                lock_path=operational / "install.lock",
                launcher_path=launcher,
                capability_digest=capture_capability_digest(manifest),
                generation=selected_version_parts[2] - _AUDITED_VERSION[2] + 1,
                launcher_bytes=placeholder,
                config=OwnedConfigSpec(
                    syntax=ConfigSyntax.JSON_OBJECT,
                    marker=CLAUDE_CODE_CONFIG_MARKER,
                    bind_json_paths=True,
                    owned_fragment=_claude_code_hook_fragment(
                        launcher,
                        windows_powershell=windows_powershell,
                    ),
                ),
            )

        spec = build_spec(host_version, version_parts)
        if probe_host:
            _validate_project_hook_policy(spec, environment)
            probed = probe_claude_code_environment(environ=environment)
            probed_parts = _supported_version_parts(probed.host_version)
            spec = build_spec(probed.host_version, probed_parts)
        return spec
    except ClaudeCodeIntegrationError:
        raise
    except Exception:
        raise ClaudeCodeIntegrationError() from None


def _exact_text(value: object, *, maximum: int = 2_048) -> str | None:
    if type(value) is not str or not 1 <= len(value.encode("utf-8")) <= maximum:
        return None
    return value


def _event_capability(event_name: str) -> CaptureEventCapability | None:
    profile = capture_profile(CLAUDE_CODE_PROFILE)
    return next((event for event in profile.events if event.event_name == event_name), None)


def _correlation_preimage(
    *,
    kind: str,
    session_id: str,
    identifier: str | None = None,
) -> bytes:
    body: dict[str, object] = {
        "schema_version": "claude-code-capture-correlation/v1",
        "kind": kind,
        "session_id": session_id,
    }
    if identifier is not None:
        body["identifier"] = identifier
    return canonical_json(body)


def _producer_digest(
    context: CaptureDigestContext,
    *,
    event_name: str,
    session_id: str,
    identifier: str | None = None,
) -> str:
    producer_kind = event_name
    if event_name in {"PostToolUse", "PostToolUseFailure", "PermissionDenied"}:
        producer_kind = "ToolTerminal"
    return context.producer_event(
        _correlation_preimage(
            kind=producer_kind,
            session_id=session_id,
            identifier=identifier,
        )
    )


def _workspace_digest(
    context: CaptureDigestContext,
    *,
    document: Mapping[str, object],
    session_id: str,
) -> str:
    cwd = _exact_text(document.get("cwd"), maximum=CAPTURE_NATIVE_JSON_LIMITS.max_string_bytes)
    material = (
        cwd.encode("utf-8")
        if cwd is not None
        else _correlation_preimage(kind="workspace_unavailable", session_id=session_id)
    )
    return context.workspace_identity(material)


def _environment_digest(context: CaptureDigestContext, *, host_version: str) -> str:
    return context.environment_identity(
        canonical_json(
            {
                "schema_version": "claude-code-capture-environment/v1",
                "profile": CLAUDE_CODE_PROFILE.value,
                "host_version": host_version,
            }
        )
    )


def _tool_class(tool_name: str | None) -> str:
    if tool_name == "Bash":
        return "shell"
    if tool_name in {"Edit", "NotebookEdit", "Write"}:
        return "file_write"
    if tool_name == "Read":
        return "file_read"
    if tool_name in {"Glob", "Grep"}:
        return "search"
    if tool_name in {"WebFetch", "WebSearch"}:
        return "network"
    if tool_name in {"Agent", "Task"}:
        return "subagent"
    return "other"


def _action_identity(
    context: CaptureDigestContext,
    *,
    document: Mapping[str, object],
    call_material: bytes,
) -> tuple[str, str | None, str]:
    tool_name = _exact_text(document.get("tool_name"), maximum=256)
    if tool_name is None:
        return context.unavailable_action_identity(call_material), None, "unavailable"
    return (
        context.action_identity(
            canonical_json(
                {
                    "schema_version": "claude-code-action-identity/v1",
                    "tool_name": tool_name,
                    "input_authority": "unavailable",
                }
            )
        ),
        tool_name,
        "coarse",
    )


def _validated_batch_fields(document: Mapping[str, object]) -> frozenset[str]:
    if "tool_calls[].tool_use_id" in document:
        return frozenset()
    observed = set(document)
    calls = document.get("tool_calls")
    if type(calls) is not tuple or not calls:
        return frozenset(observed)
    identifiers: list[str] = []
    for call in calls:
        if not isinstance(call, Mapping):
            return frozenset(observed)
        identifier = _exact_text(
            call.get("tool_use_id"),
            maximum=CAPTURE_NATIVE_JSON_LIMITS.max_string_bytes,
        )
        if identifier is None:
            return frozenset(observed)
        identifiers.append(identifier)
    if len(set(identifiers)) != len(identifiers):
        return frozenset(observed)
    observed.add("tool_calls[].tool_use_id")
    return frozenset(observed)


class ClaudeCodeCaptureAdapter:
    """Allowlist one official Claude Code hook payload into authenticated intake."""

    __slots__ = ("_capability_digest", "_connection_id", "_host_version")

    def __init__(
        self,
        *,
        connection_id: str,
        host_version: str = CLAUDE_CODE_HOST_VERSION,
    ) -> None:
        try:
            if type(connection_id) is not str or _CONNECTION_ID.fullmatch(connection_id) is None:
                raise ClaudeCodeIntegrationError()
            _supported_version_parts(host_version)
            profile = capture_profile(CLAUDE_CODE_PROFILE)
            self._connection_id = connection_id
            self._host_version = host_version
            self._capability_digest = capture_capability_digest(profile)
        except ClaudeCodeIntegrationError:
            raise
        except Exception:
            raise ClaudeCodeIntegrationError() from None

    def __repr__(self) -> str:
        return "ClaudeCodeCaptureAdapter(<redacted>)"

    __str__ = __repr__

    def capabilities(self) -> CaptureAdapterCapabilities:
        try:
            return CaptureAdapterCapabilities(
                protocol_version=CAPTURE_ADAPTER_PROTOCOL_VERSION,
                profile_id=CLAUDE_CODE_PROFILE,
                capability_digest=self._capability_digest,
                host_version=self._host_version,
            )
        except Exception:
            raise ClaudeCodeIntegrationError() from None

    def _common(
        self,
        *,
        context: CaptureDigestContext,
        session_native: str,
        event_name: str,
        producer_identifier: str | None = None,
    ) -> dict[str, object]:
        return {
            "schema_version": "capture-intake/v1",
            "adapter_profile": CLAUDE_CODE_PROFILE.value,
            "capability_manifest_digest": self._capability_digest,
            "connection_id": self._connection_id,
            "session_id": context.session_id(session_native.encode("utf-8")),
            "producer_event_digest": _producer_digest(
                context,
                event_name=event_name,
                session_id=session_native,
                identifier=producer_identifier,
            ),
            "intake_tag": _ZERO_TAG,
            "occurred_at": None,
            "timestamp_authority": "unavailable",
            "producer_sequence": None,
            "sequence_authority": "unavailable",
            "capture_disposition": "captured",
        }

    @staticmethod
    def _authenticated(
        values: Mapping[str, object],
        *,
        context: CaptureDigestContext,
    ) -> CaptureIntake:
        return authenticate_capture_intake(
            validate_capture_intake(dict(values)),
            context=context,
        )

    def adapt_bytes(
        self,
        source: bytes,
        *,
        context: CaptureDigestContext,
    ) -> tuple[CaptureIntake, ...]:
        """Reduce exactly one bounded hook document; raw bytes never leave this call."""

        try:
            if type(context) is not CaptureDigestContext:
                raise ClaudeCodeIntegrationError()
            document = read_bounded_json(source, limits=CAPTURE_NATIVE_JSON_LIMITS)
            event_name = _exact_text(document.get("hook_event_name"), maximum=256)
            session_native = _exact_text(
                document.get("session_id"),
                maximum=CAPTURE_NATIVE_JSON_LIMITS.max_string_bytes,
            )
            if event_name is None or session_native is None:
                raise ClaudeCodeIntegrationError()
            event = _event_capability(event_name)
            profile = capture_profile(CLAUDE_CODE_PROFILE)
            observed_fields = (
                _validated_batch_fields(document)
                if event_name == "PostToolBatch"
                else frozenset(document)
            )
            compatibility = classify_capture_compatibility(
                profile,
                host_version=self._host_version,
                observed_event=event_name,
                observed_fields=observed_fields,
            )
            if event is None or compatibility is CompatibilityStatus.INCOMPATIBLE:
                raise ClaudeCodeIntegrationError()

            if event_name == "SessionStart":
                values = self._common(
                    context=context,
                    session_native=session_native,
                    event_name=event_name,
                )
                values["kind"] = "session_started"
                return (self._authenticated(values, context=context),)

            if event_name == "PostToolBatch":
                if "tool_calls[].tool_use_id" not in observed_fields:
                    raise ClaudeCodeIntegrationError()
                return ()

            if event_name in {
                "PreToolUse",
                "PostToolUse",
                "PostToolUseFailure",
                "PermissionDenied",
            }:
                tool_use_id = _exact_text(
                    document.get("tool_use_id"),
                    maximum=CAPTURE_NATIVE_JSON_LIMITS.max_string_bytes,
                )
                if tool_use_id is None:
                    raise ClaudeCodeIntegrationError()
                call_material = _correlation_preimage(
                    kind="tool_call",
                    session_id=session_native,
                    identifier=tool_use_id,
                )
                values = self._common(
                    context=context,
                    session_native=session_native,
                    event_name=event_name,
                    producer_identifier=tool_use_id,
                )
                values["call_ref"] = context.call_ref(call_material)
                if event_name == "PermissionDenied":
                    values["kind"] = "permission_denied"
                    return (self._authenticated(values, context=context),)
                if event_name in {"PostToolUse", "PostToolUseFailure"}:
                    failed = event_name == "PostToolUseFailure"
                    if (
                        failed
                        and "is_interrupt" in document
                        and type(document["is_interrupt"]) is not bool
                    ):
                        raise ClaudeCodeIntegrationError()
                    interrupted = failed and document.get("is_interrupt") is True
                    values.update(
                        kind="action_finished",
                        outcome_status="failed" if failed else "succeeded",
                        outcome_authority="producer_claimed_structured",
                        exit_status=None,
                        error_code=("interrupted" if interrupted else "tool_error")
                        if failed
                        else None,
                        failure_signature=None,
                    )
                    return (self._authenticated(values, context=context),)
                action_digest, tool_name, authority = _action_identity(
                    context,
                    document=document,
                    call_material=call_material,
                )
                values.update(
                    kind="action_started",
                    action_digest=action_digest,
                    workspace_digest=_workspace_digest(
                        context,
                        document=document,
                        session_id=session_native,
                    ),
                    environment_digest=_environment_digest(
                        context,
                        host_version=self._host_version,
                    ),
                    tool_class=_tool_class(tool_name),
                    identity_authority=authority,
                )
                return (self._authenticated(values, context=context),)

            if event_name in {"SubagentStart", "SubagentStop"}:
                agent_id = _exact_text(
                    document.get("agent_id"),
                    maximum=CAPTURE_NATIVE_JSON_LIMITS.max_string_bytes,
                )
                if agent_id is None:
                    raise ClaudeCodeIntegrationError()
                if event_name == "SubagentStop":
                    return ()
                values = self._common(
                    context=context,
                    session_native=session_native,
                    event_name=event_name,
                    producer_identifier=agent_id,
                )
                values.update(
                    kind="subagent_started",
                    subagent_id=context.subagent_id(
                        _correlation_preimage(
                            kind="subagent",
                            session_id=session_native,
                            identifier=agent_id,
                        )
                    ),
                )
                return (self._authenticated(values, context=context),)

            if event_name == "Stop":
                return ()

            if event_name == "StopFailure":
                prompt_id = _exact_text(
                    document.get("prompt_id"),
                    maximum=CAPTURE_NATIVE_JSON_LIMITS.max_string_bytes,
                )
                if prompt_id is None:
                    raise ClaudeCodeIntegrationError()
                values = self._common(
                    context=context,
                    session_native=session_native,
                    event_name=event_name,
                    producer_identifier=prompt_id,
                )
                values.update(
                    kind="controller_failed",
                    error_code="provider_callback_failed",
                    failure_signature=None,
                )
                return (self._authenticated(values, context=context),)

            if event_name == "SessionEnd":
                return ()
            raise ClaudeCodeIntegrationError()
        except ClaudeCodeIntegrationError:
            raise
        except Exception:
            raise ClaudeCodeIntegrationError() from None


@dataclass(frozen=True, slots=True, repr=False)
class _ClaudeCodeHookRuntime:
    key: object
    locations: object
    spec: ProviderInstallationSpec
    registration: object
    installation: object
    connection: object


def _claude_code_project_candidates(document: Mapping[str, object]) -> tuple[Path, ...]:
    try:
        cwd = _exact_text(
            document.get("cwd"),
            maximum=CAPTURE_NATIVE_JSON_LIMITS.max_string_bytes,
        )
        if cwd is None:
            raise ClaudeCodeIntegrationError()
        current = Path(cwd)
        if not current.is_absolute() or ".." in current.parts:
            raise ClaudeCodeIntegrationError()
        current = current.resolve(strict=True)
        if not current.is_dir() or current.is_symlink():
            raise ClaudeCodeIntegrationError()
        candidates: list[Path] = []
        for depth, candidate in enumerate((current, *current.parents)):
            if depth > 128:
                break
            config_directory = candidate / ".claude"
            try:
                metadata = config_directory.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                continue
            try:
                config = read_config_bytes(config_directory / "settings.local.json")
            except ConfigFileError:
                continue
            if config is not None and config.count(CLAUDE_CODE_CONFIG_MARKER.encode("ascii")) == 1:
                candidates.append(candidate)
        if not candidates:
            raise ClaudeCodeIntegrationError()
        return tuple(candidates)
    except ClaudeCodeIntegrationError:
        raise
    except Exception:
        raise ClaudeCodeIntegrationError() from None


def build_capture_hook_dependencies(
    source: bytes,
    *,
    connection_id: str,
    environ: Mapping[str, str] | None = None,
    capture_executable: str | os.PathLike[str] | Path | None = None,
) -> CaptureHookDependencies:
    """Authenticate an installed Claude Code runtime before admitting native bytes."""

    try:
        from saliencegate.capture.connections import CaptureConnectionSummary
        from saliencegate.capture.health import CaptureHealthCode
        from saliencegate.capture.locations import CaptureStoreLocations
        from saliencegate.capture.spool import CaptureSpool
        from saliencegate.capture.store import (
            CaptureConnectionState,
            CaptureStore,
            CaptureStoreMode,
        )
        from saliencegate.integrations.hook import CaptureHookDependencies
        from saliencegate.integrations.installation import (
            InstallationIdentity,
            InstallationState,
            InstallationStatus,
            derive_installation_identity,
            inspect_provider_installation,
        )
        from saliencegate.integrations.launcher_materialization import (
            materialize_provider_launcher,
        )
        from saliencegate.integrations.registry import (
            BUILTIN_PROVIDER_REGISTRY,
            ProviderAlias,
            ProviderRegistration,
        )
        from saliencegate.security import InstallationKey, load_installation_key

        if (
            type(source) is not bytes
            or type(connection_id) is not str
            or _CONNECTION_ID.fullmatch(connection_id) is None
            or (environ is not None and not isinstance(environ, Mapping))
        ):
            raise ClaudeCodeIntegrationError()
        environment = environment_without_provider_credentials(environ)
        document = read_bounded_json(source, limits=CAPTURE_NATIVE_JSON_LIMITS)
        session_native = _exact_text(
            document.get("session_id"),
            maximum=CAPTURE_NATIVE_JSON_LIMITS.max_string_bytes,
        )
        if session_native is None:
            raise ClaudeCodeIntegrationError()
        key = load_installation_key(environ=environment)
        configured_home = environment.get("HOME")
        home = Path.home() if configured_home is None else Path(configured_home)
        locations = resolve_capture_store_locations(environ=environment, home=home)
        with CaptureStore.open(
            locations.database_path,
            installation_key=key,
            mode=CaptureStoreMode.HOOK,
        ) as store:
            connection = store.get_connection(connection_id)
        if connection.profile_id is not CLAUDE_CODE_PROFILE:
            raise ClaudeCodeIntegrationError()
        matches: list[tuple[Path, ProviderInstallationSpec, InstallationIdentity]] = []
        for candidate in _claude_code_project_candidates(document):
            candidate_spec = provider_installation_spec(
                candidate,
                environ=environment,
                host_version=connection.host_version,
            )
            candidate_identity = derive_installation_identity(candidate_spec, key)
            if candidate_identity.connection_id == connection_id:
                matches.append((candidate, candidate_spec, candidate_identity))
        if len(matches) != 1:
            raise ClaudeCodeIntegrationError()
        _project, spec, identity = matches[0]
        if connection.project_digest != identity.project_digest:
            raise ClaudeCodeIntegrationError()
        registration = BUILTIN_PROVIDER_REGISTRY.resolve(
            ProviderAlias.CLAUDE_CODE,
            require_available=True,
        )
        if (
            registration.profile is not CLAUDE_CODE_PROFILE
            or registration.host_version != CLAUDE_CODE_HOST_VERSION
        ):
            raise ClaudeCodeIntegrationError()
        spec = materialize_provider_launcher(
            spec,
            key,
            capture_executable=capture_executable,
        )
        installed_identity = derive_installation_identity(spec, key)
        if (
            installed_identity.project_digest != identity.project_digest
            or installed_identity.connection_id != connection_id
        ):
            raise ClaudeCodeIntegrationError()
        installation = inspect_provider_installation(spec, key)
        if (
            installation.state is not InstallationState.ENABLED
            or not installation.installed
            or installation.drift
            or installation.connection_id != connection_id
        ):
            raise ClaudeCodeIntegrationError()
        if (
            connection.state is not CaptureConnectionState.ENABLED
            or connection.capability_manifest_digest != spec.capability_digest
            or connection.host_version != spec.host_version
        ):
            raise ClaudeCodeIntegrationError()
        runtime = _ClaudeCodeHookRuntime(
            key=key,
            locations=locations,
            spec=spec,
            registration=registration,
            installation=installation,
            connection=connection,
        )

        def checked_runtime(value: object) -> _ClaudeCodeHookRuntime:
            if value is not runtime:
                raise ClaudeCodeIntegrationError()
            return runtime

        def validate_registry(profile: CaptureProfile) -> object:
            if profile is not CLAUDE_CODE_PROFILE:
                raise ClaudeCodeIntegrationError()
            return registration

        def validate_receipt(
            profile: CaptureProfile,
            candidate_connection_id: str,
            candidate_registry: object,
        ) -> object:
            if (
                profile is not CLAUDE_CODE_PROFILE
                or candidate_connection_id != connection_id
                or candidate_registry is not registration
            ):
                raise ClaudeCodeIntegrationError()
            return installation

        def validate_connection(
            profile: CaptureProfile,
            candidate_connection_id: str,
            candidate_registry: object,
            candidate_receipt: object,
        ) -> object:
            if (
                profile is not CLAUDE_CODE_PROFILE
                or candidate_connection_id != connection_id
                or candidate_registry is not registration
                or candidate_receipt is not installation
            ):
                raise ClaudeCodeIntegrationError()
            return runtime

        def load_context(candidate: object) -> CaptureDigestContext:
            selected = checked_runtime(candidate)
            if type(selected.key) is not InstallationKey:
                raise ClaudeCodeIntegrationError()
            return CaptureDigestContext(selected.key)

        def resolve_adapter(candidate: object) -> ClaudeCodeCaptureAdapter:
            selected = checked_runtime(candidate)
            if type(selected.connection) is not CaptureConnectionSummary:
                raise ClaudeCodeIntegrationError()
            return ClaudeCodeCaptureAdapter(
                connection_id=selected.connection.connection_id,
                host_version=selected.connection.host_version,
            )

        def open_store(candidate: object) -> CaptureStore:
            selected = checked_runtime(candidate)
            if (
                type(selected.key) is not InstallationKey
                or type(selected.locations) is not CaptureStoreLocations
            ):
                raise ClaudeCodeIntegrationError()
            return CaptureStore.open(
                selected.locations.database_path,
                installation_key=selected.key,
                mode=CaptureStoreMode.HOOK,
            )

        def open_spool(candidate: object) -> CaptureSpool:
            selected = checked_runtime(candidate)
            if (
                type(selected.key) is not InstallationKey
                or type(selected.locations) is not CaptureStoreLocations
            ):
                raise ClaudeCodeIntegrationError()
            return CaptureSpool.open(selected.locations, selected.key)

        session_id = CaptureDigestContext(key).session_id(session_native.encode("utf-8"))

        def mark_health(candidate: object, code: CaptureHealthCode) -> None:
            selected = checked_runtime(candidate)
            if (
                type(code) is not CaptureHealthCode
                or type(selected.key) is not InstallationKey
                or type(selected.locations) is not CaptureStoreLocations
                or type(selected.connection) is not CaptureConnectionSummary
            ):
                raise ClaudeCodeIntegrationError()
            with CaptureStore.open(
                selected.locations.database_path,
                installation_key=selected.key,
                mode=CaptureStoreMode.HOOK,
            ) as health_store:
                health_store.mark_session_health(
                    selected.connection.connection_id,
                    session_id,
                    code,
                )

        if (
            type(registration) is not ProviderRegistration
            or type(installation) is not InstallationStatus
            or type(connection) is not CaptureConnectionSummary
        ):
            raise ClaudeCodeIntegrationError()
        return CaptureHookDependencies(
            validate_registry=validate_registry,
            validate_receipt=validate_receipt,
            validate_connection=validate_connection,
            load_context=load_context,
            resolve_adapter=resolve_adapter,
            open_store=open_store,
            open_spool=open_spool,
            mark_health=mark_health,
        )
    except ClaudeCodeIntegrationError:
        raise
    except Exception:
        raise ClaudeCodeIntegrationError() from None


__all__ = [
    "CLAUDE_CODE_CONFIG_MARKER",
    "CLAUDE_CODE_HOOK_EVENTS",
    "CLAUDE_CODE_HOST_VERSION",
    "CLAUDE_CODE_PROFILE",
    "CLAUDE_CODE_VERSION_TIMEOUT_SECONDS",
    "MAX_CLAUDE_CODE_VERSION_OUTPUT_BYTES",
    "ClaudeCodeCaptureAdapter",
    "ClaudeCodeIntegrationError",
    "ClaudeCodeVersionProbe",
    "build_capture_hook_dependencies",
    "probe_claude_code_environment",
    "probe_claude_code_version",
    "provider_installation_spec",
]
