"""User-global provider installation specifications."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from collections.abc import Mapping
from pathlib import Path

from saliencegate.capture.capabilities import capture_capability_digest, capture_profile
from saliencegate.capture.locations import resolve_capture_store_locations
from saliencegate.domain import canonical_json
from saliencegate.integrations import claude_code, codex, opencode, pi
from saliencegate.integrations.config_files import (
    ConfigSyntax,
    OwnedConfigSpec,
    TomlBooleanConstraint,
)
from saliencegate.integrations.environment import environment_without_provider_credentials
from saliencegate.integrations.registry import (
    ProviderAlias,
    ProviderInstallationKind,
    ProviderInstallationSpec,
)


class GlobalInstallationError(ValueError):
    """A user-global provider boundary is unavailable or unsafe."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("global capture integration is unavailable")


def _configured_absolute_directory(
    value: str | None,
    *,
    fallback: Path,
) -> Path:
    try:
        candidate = fallback if value is None else Path(value)
        if not candidate.is_absolute() or ".." in candidate.parts or "\x00" in os.fspath(candidate):
            raise GlobalInstallationError()
        resolved = candidate.resolve(strict=True)
        metadata = resolved.lstat()
        if (
            resolved != candidate
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
        ):
            raise GlobalInstallationError()
        return resolved
    except GlobalInstallationError:
        raise
    except Exception:
        raise GlobalInstallationError() from None


def resolve_global_provider_root(
    provider: ProviderAlias | str,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve one existing provider-user configuration root."""

    try:
        alias = provider if type(provider) is ProviderAlias else ProviderAlias(provider)
        if type(provider) not in (str, ProviderAlias):
            raise GlobalInstallationError()
        environment = environment_without_provider_credentials(environ)
        configured_home = environment.get("HOME")
        if configured_home is not None and type(configured_home) is not str:
            raise GlobalInstallationError()
        home = Path.home() if configured_home is None else Path(configured_home)
        home = _configured_absolute_directory(None, fallback=home)
        if alias is ProviderAlias.CODEX:
            return _configured_absolute_directory(
                environment.get("CODEX_HOME"),
                fallback=home / ".codex",
            )
        if alias is ProviderAlias.CLAUDE_CODE:
            return _configured_absolute_directory(
                environment.get("CLAUDE_CONFIG_DIR"),
                fallback=home / ".claude",
            )
        if alias is ProviderAlias.OPENCODE:
            configured = environment.get("OPENCODE_CONFIG_DIR")
            if configured is not None:
                return _configured_absolute_directory(configured, fallback=home)
            xdg_config = environment.get("XDG_CONFIG_HOME")
            base = home / ".config" if xdg_config is None else Path(xdg_config)
            return _configured_absolute_directory(None, fallback=base / "opencode")
        if alias is ProviderAlias.PI:
            return _configured_absolute_directory(
                environment.get("PI_CODING_AGENT_DIR"),
                fallback=home / ".pi" / "agent",
            )
        raise GlobalInstallationError()
    except GlobalInstallationError:
        raise
    except Exception:
        raise GlobalInstallationError() from None


def _operational_directory(
    alias: ProviderAlias,
    root: Path,
    *,
    environment: Mapping[str, str],
) -> Path:
    configured_home = environment.get("HOME")
    home = Path.home() if configured_home is None else Path(configured_home)
    locations = resolve_capture_store_locations(environ=environment, home=home)
    locator = hashlib.sha256(
        canonical_json(
            {
                "schema_version": "global-installation-location/v1",
                "provider_id": alias.value,
                "configuration_root": os.fspath(root),
            }
        )
    ).hexdigest()
    return locations.state_directory / "integrations" / "global" / locator / alias.value


def _placeholder_launcher() -> bytes:
    return (
        b"@exit /b 0\r\n"
        if os.name == "nt"  # pragma: no cover - exercised by native Windows CI
        else b"#!/bin/sh\nexit 0\n"
    )


def global_provider_installation_spec(
    provider: ProviderAlias | str,
    *,
    environ: Mapping[str, str] | None = None,
    probe_host: bool = False,
    host_version: str | None = None,
) -> ProviderInstallationSpec:
    """Describe one reversible provider integration for the current OS user."""

    try:
        alias = provider if type(provider) is ProviderAlias else ProviderAlias(provider)
        if (
            type(provider) not in (str, ProviderAlias)
            or type(probe_host) is not bool
            or (host_version is not None and type(host_version) is not str)
            or (probe_host and host_version is not None)
        ):
            raise GlobalInstallationError()
        environment = environment_without_provider_credentials(environ)
        root = resolve_global_provider_root(alias, environ=environment)
        operational = _operational_directory(alias, root, environment=environment)
        launcher = operational / ("capture-hook.cmd" if os.name == "nt" else "capture-hook")
        receipt_path = operational / "receipt.json"
        journal_path = operational / "journal.json"
        lock_path = operational / "install.lock"
        launcher_bytes = _placeholder_launcher()

        if alias is ProviderAlias.CODEX:
            selected_version = codex.CODEX_HOST_VERSION if host_version is None else host_version
            if probe_host:
                selected_version = codex.probe_codex_environment(environ=environment).host_version
            parts = codex._supported_version_parts(selected_version)
            return ProviderInstallationSpec(
                installation_kind=ProviderInstallationKind.COMMAND_HOOK,
                provider_id=alias.value,
                profile=codex.CODEX_PROFILE,
                host_version=selected_version,
                project_root=root,
                config_path=root / "config.toml",
                receipt_path=receipt_path,
                journal_path=journal_path,
                lock_path=lock_path,
                launcher_path=launcher,
                capability_digest=capture_capability_digest(capture_profile(codex.CODEX_PROFILE)),
                generation=parts[2] - codex._AUDITED_VERSION[2] + 1,
                launcher_bytes=launcher_bytes,
                config=OwnedConfigSpec(
                    syntax=ConfigSyntax.TOML_DOCUMENT,
                    marker=codex.CODEX_CONFIG_MARKER,
                    owned_fragment=codex._codex_hook_fragment(launcher),
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

        if alias is ProviderAlias.CLAUDE_CODE:
            selected_version = (
                claude_code.CLAUDE_CODE_HOST_VERSION if host_version is None else host_version
            )
            if probe_host:
                selected_version = claude_code.probe_claude_code_environment(
                    environ=environment
                ).host_version
            parts = claude_code._supported_version_parts(selected_version)
            windows_powershell = (
                claude_code._trusted_windows_powershell() if os.name == "nt" else None
            )
            return ProviderInstallationSpec(
                installation_kind=ProviderInstallationKind.COMMAND_HOOK,
                provider_id=alias.value,
                profile=claude_code.CLAUDE_CODE_PROFILE,
                host_version=selected_version,
                project_root=root,
                config_path=root / "settings.json",
                receipt_path=receipt_path,
                journal_path=journal_path,
                lock_path=lock_path,
                launcher_path=launcher,
                capability_digest=capture_capability_digest(
                    capture_profile(claude_code.CLAUDE_CODE_PROFILE)
                ),
                generation=parts[2] - claude_code._AUDITED_VERSION[2] + 1,
                launcher_bytes=launcher_bytes,
                config=OwnedConfigSpec(
                    syntax=ConfigSyntax.JSON_OBJECT,
                    marker=claude_code.CLAUDE_CODE_CONFIG_MARKER,
                    bind_json_paths=True,
                    owned_fragment=claude_code._claude_code_hook_fragment(
                        launcher,
                        windows_powershell=windows_powershell,
                    ),
                ),
            )

        if alias is ProviderAlias.OPENCODE:
            if host_version not in (None, opencode.OPENCODE_HOST_VERSION):
                raise GlobalInstallationError()
            bundle_directory = root / "plugins"
            return ProviderInstallationSpec(
                installation_kind=ProviderInstallationKind.BRIDGE,
                provider_id=alias.value,
                profile=opencode.OPENCODE_PROFILE,
                host_version=opencode.OPENCODE_HOST_VERSION,
                project_root=root,
                config_path=None,
                config=None,
                bundle_path=bundle_directory / "saliencegate.js",
                bootstrap_path=bundle_directory / "saliencegate.bootstrap.json",
                receipt_path=receipt_path,
                journal_path=journal_path,
                lock_path=lock_path,
                launcher_path=launcher,
                capability_digest=capture_capability_digest(
                    capture_profile(opencode.OPENCODE_PROFILE)
                ),
                bundle_bytes=opencode._bundle_bytes(),
                launcher_bytes=launcher_bytes,
                bootstrap_relative_reference=opencode.OPENCODE_BOOTSTRAP_REFERENCE,
                generation=1,
            )

        if alias is ProviderAlias.PI:
            if host_version not in (None, pi.PI_HOST_VERSION):
                raise GlobalInstallationError()
            bundle_directory = root / "extensions"
            return ProviderInstallationSpec(
                installation_kind=ProviderInstallationKind.BRIDGE,
                provider_id=alias.value,
                profile=pi.PI_PROFILE,
                host_version=pi.PI_HOST_VERSION,
                project_root=root,
                config_path=None,
                config=None,
                bundle_path=bundle_directory / "saliencegate.ts",
                bootstrap_path=bundle_directory / "saliencegate.bootstrap.json",
                receipt_path=receipt_path,
                journal_path=journal_path,
                lock_path=lock_path,
                launcher_path=launcher,
                capability_digest=capture_capability_digest(capture_profile(pi.PI_PROFILE)),
                bundle_bytes=pi._bundle_bytes(),
                launcher_bytes=launcher_bytes,
                bootstrap_relative_reference=pi.PI_BOOTSTRAP_REFERENCE,
                generation=1,
            )
        raise GlobalInstallationError()
    except GlobalInstallationError:
        raise
    except Exception:
        raise GlobalInstallationError() from None


def global_provider_is_available(
    provider: ProviderAlias | str,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Report whether an existing global provider root has a usable host CLI."""

    try:
        alias = provider if type(provider) is ProviderAlias else ProviderAlias(provider)
        if type(provider) not in (str, ProviderAlias):
            raise GlobalInstallationError()
        environment = environment_without_provider_credentials(environ)
        resolve_global_provider_root(alias, environ=environment)
        if alias is ProviderAlias.CODEX:
            codex.probe_codex_environment(environ=environment)
            return True
        if alias is ProviderAlias.CLAUDE_CODE:
            claude_code.probe_claude_code_environment(environ=environment)
            return True
        executable_name = {
            ProviderAlias.OPENCODE: "opencode",
            ProviderAlias.PI: "pi",
        }.get(alias)
        configured_path = environment.get("PATH")
        if executable_name is None or type(configured_path) is not str:
            return False
        candidate = shutil.which(executable_name, path=configured_path)
        if candidate is None:
            return False
        executable = Path(candidate).resolve(strict=True)
        metadata = executable.lstat()
        return stat.S_ISREG(metadata.st_mode) and (
            os.name != "posix" or os.access(executable, os.X_OK)
        )
    except Exception:
        return False


__all__ = [
    "GlobalInstallationError",
    "global_provider_installation_spec",
    "global_provider_is_available",
    "resolve_global_provider_root",
]
