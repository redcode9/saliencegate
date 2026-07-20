"""Passive, content-free Codex lifecycle capture adapter."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import threading
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

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
    ConfigSyntax,
    OwnedConfigSpec,
    TomlBooleanConstraint,
    plan_owned_config_install,
    read_config_bytes,
)
from saliencegate.integrations.registry import (
    ProviderInstallationKind,
    ProviderInstallationSpec,
)

if TYPE_CHECKING:
    from saliencegate.integrations.hook import CaptureHookDependencies

CODEX_HOST_VERSION: Final = "0.144.6"
CODEX_PROFILE: Final = CaptureProfile.CODEX_HOOKS_V1
CODEX_CONFIG_MARKER: Final = "saliencegate-owned:codex-hooks-v1"
CODEX_HOOK_EVENTS: Final = (
    "SessionStart",
    "PreToolUse",
    "PermissionRequest",
    "PostToolUse",
    "PreCompact",
    "SubagentStart",
    "SubagentStop",
    "Stop",
)
MAX_CODEX_VERSION_OUTPUT_BYTES: Final = 4_096
CODEX_VERSION_TIMEOUT_SECONDS: Final = 2.0

_CONNECTION_ID: Final = re.compile(r"^[a-z0-9][a-z0-9._:-]{11,127}$")
_HOST_VERSION: Final = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
_VERSION_OUTPUT: Final = re.compile(
    rb"codex-cli ((?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))(?:\r?\n)?"
)
_ZERO_TAG: Final = "0" * 64
_AUDITED_VERSION: Final = (0, 144, 6)


class CodexIntegrationError(ValueError):
    """A Codex boundary failed without disclosing provider-owned values."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("Codex capture integration is invalid")


def _supported_version_parts(host_version: object) -> tuple[int, int, int]:
    if type(host_version) is not str or _HOST_VERSION.fullmatch(host_version) is None:
        raise CodexIntegrationError()
    try:
        parts = tuple(int(part) for part in host_version.split("."))
    except Exception:
        raise CodexIntegrationError() from None
    generation = parts[2] - _AUDITED_VERSION[2] + 1
    if (
        len(parts) != 3
        or parts[:2] != _AUDITED_VERSION[:2]
        or parts < _AUDITED_VERSION
        or not 1 <= generation <= 1_000_000
    ):
        raise CodexIntegrationError()
    return parts


@dataclass(frozen=True, slots=True)
class CodexVersionProbe:
    """Bounded, content-free result of an exact Codex CLI version probe."""

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
            raise CodexIntegrationError()


VersionRunner = Callable[..., subprocess.CompletedProcess[bytes]]


def _bounded_version_runner(
    command: tuple[str, ...],
    *,
    input: bytes,
    capture_output: bool,
    check: bool,
    timeout: float,
    env: Mapping[str, str],
) -> subprocess.CompletedProcess[bytes]:
    """Run a version command while retaining at most one bounded chunk per stream."""

    if (
        type(command) is not tuple
        or not command
        or input != b""
        or capture_output is not True
        or check is not False
        or timeout != CODEX_VERSION_TIMEOUT_SECONDS
        or not isinstance(env, Mapping)
        or any(type(key) is not str or type(value) is not str for key, value in env.items())
    ):
        raise CodexIntegrationError()
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(env),
        )
        if process.stdout is None or process.stderr is None:
            raise CodexIntegrationError()
        output: list[bytes | None] = [None, None]
        failed = [False, False]

        def consume(index: int, stream: object) -> None:
            try:
                read = getattr(stream, "read", None)
                close = getattr(stream, "close", None)
                if not callable(read) or not callable(close):
                    raise OSError
                chunk = read(MAX_CODEX_VERSION_OUTPUT_BYTES + 1)
                if type(chunk) is not bytes:
                    raise OSError
                output[index] = chunk
                close()
            except Exception:
                failed[index] = True

        readers = (
            threading.Thread(target=consume, args=(0, process.stdout), daemon=True),
            threading.Thread(target=consume, args=(1, process.stderr), daemon=True),
        )
        for reader in readers:
            reader.start()
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        for reader in readers:
            reader.join(timeout=0.25)
        if any(reader.is_alive() for reader in readers) or any(failed) or None in output:
            raise CodexIntegrationError()
        return subprocess.CompletedProcess(
            command,
            returncode,
            output[0],
            output[1],
        )
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()


def probe_codex_environment(
    *,
    environ: Mapping[str, str] | None = None,
) -> CodexVersionProbe:
    """Resolve and probe the Codex executable selected by one explicit environment."""

    try:
        environment = os.environ if environ is None else environ
        if not isinstance(environment, Mapping) or any(
            type(key) is not str or type(value) is not str for key, value in environment.items()
        ):
            raise CodexIntegrationError()
        configured_path = environment.get("PATH")
        if (configured_path is None and environ is not None) or (
            configured_path is not None and type(configured_path) is not str
        ):
            raise CodexIntegrationError()
        executable_name = shutil.which("codex", path=configured_path)
        if executable_name is None:
            raise CodexIntegrationError()
        return probe_codex_version(
            Path(executable_name).resolve(strict=True),
            environ=environment,
        )
    except CodexIntegrationError:
        raise
    except Exception:
        raise CodexIntegrationError() from None


def _toml_string(value: str) -> str:
    try:
        if type(value) is not str or not value or "\x00" in value:
            raise CodexIntegrationError()
        return json.dumps(value, ensure_ascii=True)
    except CodexIntegrationError:
        raise
    except Exception:
        raise CodexIntegrationError() from None


def _hook_command(launcher: Path) -> str:
    try:
        raw = os.fspath(launcher)
        if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
            return subprocess.list2cmdline((raw,))
        return shlex.quote(raw)
    except Exception:
        raise CodexIntegrationError() from None


def _codex_hook_fragment(launcher: Path) -> bytes:
    """Render the exact project-local hook tables without changing trust settings."""

    try:
        command = _toml_string(_hook_command(launcher))
        matchers: dict[str, str | None] = {
            "SessionStart": "startup|resume|clear|compact",
            "PreToolUse": "*",
            "PermissionRequest": "*",
            "PostToolUse": "*",
            "PreCompact": "manual|auto",
            "SubagentStart": "*",
            "SubagentStop": "*",
            "Stop": None,
        }
        lines = [f"# {CODEX_CONFIG_MARKER}"]
        for event_name in CODEX_HOOK_EVENTS:
            lines.extend(("", f"[[hooks.{event_name}]]"))
            matcher = matchers[event_name]
            if matcher is not None:
                lines.append(f"matcher = {_toml_string(matcher)}")
            lines.extend(
                (
                    "",
                    f"[[hooks.{event_name}.hooks]]",
                    'type = "command"',
                    f"command = {command}",
                    "timeout = 3",
                )
            )
        return ("\n".join(lines) + "\n").encode("utf-8")
    except CodexIntegrationError:
        raise
    except Exception:
        raise CodexIntegrationError() from None


def _validate_project_hook_policy(spec: ProviderInstallationSpec) -> None:
    """Preflight a project layer before any capture runtime state is created."""

    try:
        config_path = spec.config_path
        config = spec.config
        if config_path is None or config is None:
            raise CodexIntegrationError()
        try:
            parent = config_path.parent.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode):
            raise CodexIntegrationError()
        source = read_config_bytes(config_path)
        if source is None:
            return
        document = tomllib.loads(source.decode("utf-8", errors="strict"))
        for constraint in config.toml_boolean_constraints:
            current: object = document
            missing = False
            for component in constraint.path:
                if type(current) is not dict:
                    raise CodexIntegrationError()
                if component not in current:
                    missing = True
                    break
                current = current[component]
            if missing:
                continue
            if type(current) is not bool or current is not constraint.expected:
                raise CodexIntegrationError()
        marker = config.marker.encode("ascii")
        if marker not in source:
            plan_owned_config_install(source, config)
            return
        if source.count(marker) != 1:
            raise CodexIntegrationError()
        try:
            receipt = spec.receipt_path.lstat()
        except FileNotFoundError:
            raise CodexIntegrationError() from None
        if stat.S_ISLNK(receipt.st_mode) or not stat.S_ISREG(receipt.st_mode):
            raise CodexIntegrationError()
    except CodexIntegrationError:
        raise
    except Exception:
        raise CodexIntegrationError() from None


def provider_installation_spec(
    project: Path,
    *,
    environ: Mapping[str, str] | None = None,
    host_version: str = CODEX_HOST_VERSION,
    probe_host: bool = False,
) -> ProviderInstallationSpec:
    """Describe one reversible command-hook installation for a trusted project layer."""

    try:
        if (
            not isinstance(project, Path)
            or not project.is_absolute()
            or ".." in project.parts
            or not project.is_dir()
            or project.is_symlink()
            or type(probe_host) is not bool
        ):
            raise CodexIntegrationError()
        version_parts = _supported_version_parts(host_version)
        environment = os.environ if environ is None else environ
        if not isinstance(environment, Mapping):
            raise CodexIntegrationError()
        configured_home = environment.get("HOME")
        if configured_home is not None and type(configured_home) is not str:
            raise CodexIntegrationError()
        config_path = project / ".codex" / "config.toml"
        home = Path.home() if configured_home is None else Path(configured_home)
        locations = resolve_capture_store_locations(
            environ=environment,
            home=home,
        )
        project_locator = hashlib.sha256(
            canonical_json(
                {
                    "schema_version": "codex-installation-location/v1",
                    "project_root": os.fspath(project),
                }
            )
        ).hexdigest()
        operational = locations.state_directory / "integrations" / project_locator / "codex"
        launcher = operational / ("capture-hook.cmd" if os.name == "nt" else "capture-hook")
        placeholder = (
            b"@exit /b 0\r\n"
            if os.name == "nt"  # pragma: no cover - exercised by native Windows R01
            else b"#!/bin/sh\nexit 0\n"
        )
        manifest = capture_profile(CODEX_PROFILE)

        def build_spec(
            selected_host_version: str,
            selected_version_parts: tuple[int, int, int],
        ) -> ProviderInstallationSpec:
            return ProviderInstallationSpec(
                installation_kind=ProviderInstallationKind.COMMAND_HOOK,
                provider_id="codex",
                profile=CODEX_PROFILE,
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
                    syntax=ConfigSyntax.TOML_DOCUMENT,
                    marker=CODEX_CONFIG_MARKER,
                    owned_fragment=_codex_hook_fragment(launcher),
                    toml_boolean_constraints=(
                        TomlBooleanConstraint(
                            path=("allow_managed_hooks_only",),
                            expected=False,
                        ),
                        TomlBooleanConstraint(
                            path=("features", "codex_hooks"),
                            expected=True,
                        ),
                        TomlBooleanConstraint(
                            path=("features", "hooks"),
                            expected=True,
                        ),
                    ),
                ),
            )

        spec = build_spec(host_version, version_parts)
        if probe_host:
            _validate_project_hook_policy(spec)
            probed_host_version = probe_codex_environment(environ=environment).host_version
            probed_version_parts = _supported_version_parts(probed_host_version)
            spec = build_spec(probed_host_version, probed_version_parts)
        return spec
    except CodexIntegrationError:
        raise
    except Exception:
        raise CodexIntegrationError() from None


@dataclass(frozen=True, slots=True, repr=False)
class _CodexHookRuntime:
    key: object
    locations: object
    spec: ProviderInstallationSpec
    registration: object
    installation: object
    connection: object


def _discover_codex_project(document: Mapping[str, object]) -> Path:
    try:
        from saliencegate.integrations.config_files import read_config_bytes

        cwd = _exact_text(
            document.get("cwd"),
            maximum=CAPTURE_NATIVE_JSON_LIMITS.max_string_bytes,
        )
        if cwd is None:
            raise CodexIntegrationError()
        current = Path(cwd)
        if not current.is_absolute() or ".." in current.parts:
            raise CodexIntegrationError()
        current = current.resolve(strict=True)
        if not current.is_dir() or current.is_symlink():
            raise CodexIntegrationError()
        for depth, candidate in enumerate((current, *current.parents)):
            if depth > 128:
                break
            config_directory = candidate / ".codex"
            try:
                metadata = config_directory.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise CodexIntegrationError()
            config = read_config_bytes(config_directory / "config.toml")
            if config is not None and config.count(CODEX_CONFIG_MARKER.encode("ascii")) == 1:
                return candidate
        raise CodexIntegrationError()
    except CodexIntegrationError:
        raise
    except Exception:
        raise CodexIntegrationError() from None


def build_capture_hook_dependencies(
    source: bytes,
    *,
    connection_id: str,
    environ: Mapping[str, str] | None = None,
    capture_executable: str | os.PathLike[str] | Path | None = None,
) -> CaptureHookDependencies:
    """Authenticate an installed Codex hook runtime before admitting native bytes."""

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
        from saliencegate.commands.capture.connect import materialize_provider_launcher
        from saliencegate.integrations.hook import CaptureHookDependencies
        from saliencegate.integrations.installation import (
            InstallationState,
            InstallationStatus,
            derive_installation_identity,
            inspect_provider_installation,
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
            raise CodexIntegrationError()
        environment = dict(os.environ if environ is None else environ)
        if any(
            type(key) is not str or type(value) is not str for key, value in environment.items()
        ):
            raise CodexIntegrationError()
        document = read_bounded_json(source, limits=CAPTURE_NATIVE_JSON_LIMITS)
        session_native = _exact_text(
            document.get("session_id"),
            maximum=CAPTURE_NATIVE_JSON_LIMITS.max_string_bytes,
        )
        if session_native is None:
            raise CodexIntegrationError()
        project = _discover_codex_project(document)
        spec = provider_installation_spec(project, environ=environment)
        key = load_installation_key(environ=environment)
        identity = derive_installation_identity(spec, key)
        registration = BUILTIN_PROVIDER_REGISTRY.resolve(
            ProviderAlias.CODEX,
            require_available=True,
        )
        if (
            registration.profile is not CODEX_PROFILE
            or registration.host_version != CODEX_HOST_VERSION
        ):
            raise CodexIntegrationError()
        configured_home = environment.get("HOME")
        home = Path.home() if configured_home is None else Path(configured_home)
        locations = resolve_capture_store_locations(environ=environment, home=home)
        with CaptureStore.open(
            locations.database_path,
            installation_key=key,
            mode=CaptureStoreMode.HOOK,
        ) as store:
            connection = store.get_connection(connection_id)
        if (
            connection.project_digest != identity.project_digest
            or connection.profile_id is not CODEX_PROFILE
        ):
            raise CodexIntegrationError()
        spec = provider_installation_spec(
            project,
            environ=environment,
            host_version=connection.host_version,
        )
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
            raise CodexIntegrationError()
        installation = inspect_provider_installation(spec, key)
        if (
            installation.state is not InstallationState.ENABLED
            or not installation.installed
            or installation.drift
            or installation.connection_id != connection_id
        ):
            raise CodexIntegrationError()
        if (
            connection.state is not CaptureConnectionState.ENABLED
            or connection.capability_manifest_digest != spec.capability_digest
            or connection.host_version != spec.host_version
        ):
            raise CodexIntegrationError()
        runtime = _CodexHookRuntime(
            key=key,
            locations=locations,
            spec=spec,
            registration=registration,
            installation=installation,
            connection=connection,
        )

        def checked_runtime(value: object) -> _CodexHookRuntime:
            if value is not runtime:
                raise CodexIntegrationError()
            return runtime

        def validate_registry(profile: CaptureProfile) -> object:
            if profile is not CODEX_PROFILE:
                raise CodexIntegrationError()
            return registration

        def validate_receipt(
            profile: CaptureProfile,
            candidate_connection_id: str,
            candidate_registry: object,
        ) -> object:
            if (
                profile is not CODEX_PROFILE
                or candidate_connection_id != connection_id
                or candidate_registry is not registration
            ):
                raise CodexIntegrationError()
            return installation

        def validate_connection(
            profile: CaptureProfile,
            candidate_connection_id: str,
            candidate_registry: object,
            candidate_receipt: object,
        ) -> object:
            if (
                profile is not CODEX_PROFILE
                or candidate_connection_id != connection_id
                or candidate_registry is not registration
                or candidate_receipt is not installation
            ):
                raise CodexIntegrationError()
            return runtime

        def load_context(candidate: object) -> CaptureDigestContext:
            selected = checked_runtime(candidate)
            if type(selected.key) is not InstallationKey:
                raise CodexIntegrationError()
            return CaptureDigestContext(selected.key)

        def resolve_adapter(candidate: object) -> CodexCaptureAdapter:
            selected = checked_runtime(candidate)
            if type(selected.connection) is not CaptureConnectionSummary:
                raise CodexIntegrationError()
            return CodexCaptureAdapter(
                connection_id=selected.connection.connection_id,
                host_version=selected.connection.host_version,
            )

        def open_store(candidate: object) -> CaptureStore:
            selected = checked_runtime(candidate)
            if (
                type(selected.key) is not InstallationKey
                or type(selected.locations) is not CaptureStoreLocations
            ):
                raise CodexIntegrationError()
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
                raise CodexIntegrationError()
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
                raise CodexIntegrationError()
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
            raise CodexIntegrationError()
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
    except CodexIntegrationError:
        raise
    except Exception:
        raise CodexIntegrationError() from None


def _exact_executable(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
        raise CodexIntegrationError()
    try:
        if path.is_symlink():
            raise CodexIntegrationError()
        resolved = path.resolve(strict=True)
        metadata = resolved.lstat()
        if (
            path != resolved
            or not stat.S_ISREG(metadata.st_mode)
            or (os.name == "posix" and not os.access(resolved, os.X_OK))
        ):
            raise CodexIntegrationError()
        return resolved
    except CodexIntegrationError:
        raise
    except Exception:
        raise CodexIntegrationError() from None


def probe_codex_version(
    executable: Path,
    *,
    runner: VersionRunner = _bounded_version_runner,
    environ: Mapping[str, str] | None = None,
) -> CodexVersionProbe:
    """Probe one trusted local Codex executable with strict output and time bounds."""

    try:
        if not callable(runner):
            raise CodexIntegrationError()
        environment = dict(os.environ if environ is None else environ)
        if any(
            type(key) is not str or type(value) is not str for key, value in environment.items()
        ):
            raise CodexIntegrationError()
        selected = _exact_executable(executable)
        completed = runner(
            (str(selected), "--version"),
            input=b"",
            capture_output=True,
            check=False,
            timeout=CODEX_VERSION_TIMEOUT_SECONDS,
            env=environment,
        )
        if (
            type(completed.returncode) is not int
            or completed.returncode != 0
            or type(completed.stdout) is not bytes
            or type(completed.stderr) is not bytes
            or len(completed.stdout) > MAX_CODEX_VERSION_OUTPUT_BYTES
            or len(completed.stderr) > MAX_CODEX_VERSION_OUTPUT_BYTES
        ):
            raise CodexIntegrationError()
        match = _VERSION_OUTPUT.fullmatch(completed.stdout)
        if match is None:
            raise CodexIntegrationError()
        host_version = match.group(1).decode("ascii")
        _supported_version_parts(host_version)
        compatibility = (
            CompatibilityStatus.VERIFIED
            if host_version == CODEX_HOST_VERSION
            else CompatibilityStatus.SCHEMA_COMPATIBLE_UNVERIFIED_VERSION
        )
        return CodexVersionProbe(
            host_version=host_version,
            compatibility=compatibility,
        )
    except CodexIntegrationError:
        raise
    except Exception:
        raise CodexIntegrationError() from None


def _exact_text(value: object, *, maximum: int = 2_048) -> str | None:
    if type(value) is not str or not 1 <= len(value.encode("utf-8")) <= maximum:
        return None
    return value


def _event_capability(event_name: str) -> CaptureEventCapability | None:
    profile = capture_profile(CODEX_PROFILE)
    return next((event for event in profile.events if event.event_name == event_name), None)


def _correlation_preimage(
    *,
    kind: str,
    session_id: str,
    identifier: str | None = None,
) -> bytes:
    body: dict[str, object] = {
        "schema_version": "codex-capture-correlation/v1",
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
    return context.producer_event(
        _correlation_preimage(
            kind=event_name,
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
                "schema_version": "codex-capture-environment/v1",
                "profile": CODEX_PROFILE.value,
                "host_version": host_version,
            }
        )
    )


def _tool_class(tool_name: str | None) -> str:
    if tool_name == "Bash":
        return "shell"
    if tool_name in {"apply_patch", "Edit", "Write"}:
        return "file_write"
    if tool_name in {"Read", "view_image"}:
        return "file_read"
    if tool_name in {"Agent", "spawn_agent"}:
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
        return (
            context.unavailable_action_identity(call_material),
            None,
            "unavailable",
        )
    tool_input = document.get("tool_input")
    if tool_input is None:
        return (
            context.action_identity(
                canonical_json(
                    {
                        "schema_version": "codex-action-identity/v1",
                        "tool_name": tool_name,
                        "input_authority": "unavailable",
                    }
                )
            ),
            tool_name,
            "coarse",
        )
    try:
        exact = canonical_json(
            {
                "schema_version": "codex-action-identity/v1",
                "tool_name": tool_name,
                "tool_input": tool_input,
            }
        )
        return context.action_identity(exact), tool_name, "exact"
    except Exception:
        return (
            context.unavailable_action_identity(call_material),
            tool_name,
            "unavailable",
        )


class CodexCaptureAdapter:
    """Allowlist one official Codex hook payload into authenticated intake."""

    __slots__ = ("_capability_digest", "_connection_id", "_host_version")

    def __init__(self, *, connection_id: str, host_version: str = CODEX_HOST_VERSION) -> None:
        try:
            if type(connection_id) is not str or _CONNECTION_ID.fullmatch(connection_id) is None:
                raise CodexIntegrationError()
            _supported_version_parts(host_version)
            profile = capture_profile(CODEX_PROFILE)
            self._connection_id = connection_id
            self._host_version = host_version
            self._capability_digest = capture_capability_digest(profile)
        except CodexIntegrationError:
            raise
        except Exception:
            raise CodexIntegrationError() from None

    def __repr__(self) -> str:
        return "CodexCaptureAdapter(<redacted>)"

    __str__ = __repr__

    def capabilities(self) -> CaptureAdapterCapabilities:
        try:
            return CaptureAdapterCapabilities(
                protocol_version=CAPTURE_ADAPTER_PROTOCOL_VERSION,
                profile_id=CODEX_PROFILE,
                capability_digest=self._capability_digest,
                host_version=self._host_version,
            )
        except Exception:
            raise CodexIntegrationError() from None

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
            "adapter_profile": CODEX_PROFILE.value,
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
                raise CodexIntegrationError()
            document = read_bounded_json(source, limits=CAPTURE_NATIVE_JSON_LIMITS)
            event_name = _exact_text(document.get("hook_event_name"), maximum=256)
            session_native = _exact_text(
                document.get("session_id"),
                maximum=CAPTURE_NATIVE_JSON_LIMITS.max_string_bytes,
            )
            if event_name is None or session_native is None:
                raise CodexIntegrationError()
            event = _event_capability(event_name)
            profile = capture_profile(CODEX_PROFILE)
            compatibility = classify_capture_compatibility(
                profile,
                host_version=self._host_version,
                observed_event=event_name,
                observed_fields=frozenset(document),
            )
            if event is None or compatibility is CompatibilityStatus.INCOMPATIBLE:
                raise CodexIntegrationError()

            if event_name == "SessionStart":
                values = self._common(
                    context=context,
                    session_native=session_native,
                    event_name=event_name,
                )
                values["kind"] = "session_started"
                return (self._authenticated(values, context=context),)

            if event_name in {"PermissionRequest", "PreCompact"}:
                return ()

            if event_name in {"PreToolUse", "PostToolUse"}:
                tool_use_id = _exact_text(
                    document.get("tool_use_id"),
                    maximum=CAPTURE_NATIVE_JSON_LIMITS.max_string_bytes,
                )
                if tool_use_id is None:
                    raise CodexIntegrationError()
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
                if event_name == "PostToolUse":
                    values.update(
                        kind="action_finished",
                        outcome_status=None,
                        outcome_authority="unavailable",
                        exit_status=None,
                        error_code=None,
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
                    raise CodexIntegrationError()
                values = self._common(
                    context=context,
                    session_native=session_native,
                    event_name=event_name,
                    producer_identifier=agent_id,
                )
                values.update(
                    kind=(
                        "subagent_started" if event_name == "SubagentStart" else "subagent_finished"
                    ),
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
                turn_id = _exact_text(
                    document.get("turn_id"),
                    maximum=CAPTURE_NATIVE_JSON_LIMITS.max_string_bytes,
                )
                if turn_id is None:
                    return ()
                values = self._common(
                    context=context,
                    session_native=session_native,
                    event_name=event_name,
                    producer_identifier=turn_id,
                )
                values.update(
                    kind="turn_finished",
                    turn_id=context.turn_id(
                        _correlation_preimage(
                            kind="turn",
                            session_id=session_native,
                            identifier=turn_id,
                        )
                    ),
                )
                return (self._authenticated(values, context=context),)
            raise CodexIntegrationError()
        except CodexIntegrationError:
            raise
        except Exception:
            raise CodexIntegrationError() from None


__all__ = [
    "CODEX_CONFIG_MARKER",
    "CODEX_HOOK_EVENTS",
    "CODEX_HOST_VERSION",
    "CODEX_PROFILE",
    "CODEX_VERSION_TIMEOUT_SECONDS",
    "MAX_CODEX_VERSION_OUTPUT_BYTES",
    "CodexCaptureAdapter",
    "CodexIntegrationError",
    "CodexVersionProbe",
    "build_capture_hook_dependencies",
    "probe_codex_environment",
    "probe_codex_version",
    "provider_installation_spec",
]
