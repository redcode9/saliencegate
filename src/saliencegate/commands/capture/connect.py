"""Install one passive, project-local capture integration."""

from __future__ import annotations

import importlib
import os
import re
import stat
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from saliencegate.capture import (
    CaptureConnectionState,
    CaptureConnectionSummary,
    CaptureMigrationError,
    CaptureMigrationIntegrityError,
    CaptureSpool,
    CaptureSpoolError,
    CaptureSpoolIntegrityError,
    CaptureStore,
    CaptureStoreError,
    CaptureStoreIntegrityError,
    CaptureStoreMode,
    CaptureStoreStateError,
    initialize_capture_store,
    resolve_capture_store_locations,
)
from saliencegate.commands.capture.common import (
    CaptureCommandConfigurationError,
    CaptureCommandInputError,
    CaptureCommandIntegrityError,
    CaptureCommandUnavailableError,
    resolve_capture_project,
)
from saliencegate.domain import canonical_json
from saliencegate.integrations.config_files import ConfigFileError, read_config_bytes
from saliencegate.integrations.installation import (
    InstallationDisposition,
    InstallationError,
    InstallationState,
    InstallationStatus,
    _load_receipt_optional,
    derive_installation_identity,
    ensure_private_installation_directory,
    git_tracked_project_files,
    inspect_provider_installation,
    install_provider,
)
from saliencegate.integrations.launcher_materialization import materialize_provider_launcher
from saliencegate.integrations.registry import (
    BUILTIN_PROVIDER_REGISTRY,
    ProviderAlias,
    ProviderInstallationSpec,
    ProviderRegistryError,
)
from saliencegate.security import (
    InsecureKeyFileError,
    InsecureKeyPathError,
    InstallationKey,
    InvalidInstallationKeyError,
    default_installation_key_path,
    generate_installation_key,
    load_installation_key,
    load_or_create_installation_key,
)

ProviderSpecResolver = Callable[[ProviderAlias, Path], ProviderInstallationSpec]

_PROVIDER_MODULES: dict[ProviderAlias, str] = {
    ProviderAlias.CODEX: "saliencegate.integrations.codex",
    ProviderAlias.CLAUDE_CODE: "saliencegate.integrations.claude_code",
    ProviderAlias.OPENCODE: "saliencegate.integrations.opencode",
    ProviderAlias.PI: "saliencegate.integrations.pi",
}
_DYNAMIC_HOST_VERSION = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
_DYNAMIC_HOST_PROVIDERS = frozenset((ProviderAlias.CODEX, ProviderAlias.CLAUDE_CODE))


class CaptureConnectReport(BaseModel):
    """Content-free result safe for terminal and machine-readable output."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    schema_version: Literal["capture-connect/v1"] = "capture-connect/v1"
    provider: ProviderAlias
    disposition: InstallationDisposition
    dry_run: bool
    capture_enabled: bool
    project_local_files: Annotated[int, Field(ge=0, le=3)]
    git_tracked_files: Annotated[int, Field(ge=0, le=3)]

    def __repr__(self) -> str:
        return "CaptureConnectReport(<redacted>)"

    __str__ = __repr__


def _default_spec_resolver(
    alias: ProviderAlias,
    project: Path,
    *,
    environ: Mapping[str, str] | None,
    probe_host: bool,
) -> ProviderInstallationSpec:
    """Load a provider factory lazily when its integration is available."""

    try:
        module = importlib.import_module(_PROVIDER_MODULES[alias])
        factory = module.provider_installation_spec
        if not callable(factory):
            raise CaptureCommandUnavailableError()
        result = factory(
            project,
            environ=environ,
            **({"probe_host": True} if alias in _DYNAMIC_HOST_PROVIDERS and probe_host else {}),
        )
        return ProviderInstallationSpec.model_validate(result)
    except CaptureCommandUnavailableError:
        raise
    except (AttributeError, ImportError, KeyError):
        raise CaptureCommandUnavailableError() from None
    except (TypeError, ValueError):
        raise CaptureCommandConfigurationError() from None


def resolve_provider_installation_spec(
    alias: ProviderAlias,
    project: Path,
    resolver: ProviderSpecResolver | None,
    *,
    environ: Mapping[str, str] | None = None,
    probe_host: bool = False,
) -> ProviderInstallationSpec:
    try:
        registration = BUILTIN_PROVIDER_REGISTRY.resolve(alias, require_available=False)
        if resolver is None:
            spec = _default_spec_resolver(
                alias,
                project,
                environ=environ,
                probe_host=probe_host,
            )
        else:
            if not callable(resolver):
                raise TypeError
            spec = ProviderInstallationSpec.model_validate(resolver(alias, project))
        if (
            spec.provider_id != alias.value
            or spec.profile is not registration.profile
            or (
                spec.host_version != registration.host_version
                and not (
                    alias in _DYNAMIC_HOST_PROVIDERS
                    and _DYNAMIC_HOST_VERSION.fullmatch(spec.host_version) is not None
                )
            )
            or spec.project_root != project
        ):
            raise ValueError
        return spec
    except CaptureCommandUnavailableError:
        raise
    except ProviderRegistryError:
        raise CaptureCommandUnavailableError() from None
    except Exception:
        raise CaptureCommandConfigurationError() from None


def project_provider_artifacts_present(
    alias: ProviderAlias,
    project: Path,
    *,
    resolver: ProviderSpecResolver | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Detect known managed artifacts without trusting an unavailable installation key."""

    spec = resolve_provider_installation_spec(alias, project, resolver, environ=environ)
    if spec.config_path is not None and spec.config is not None:
        try:
            try:
                parent = spec.config_path.parent.lstat()
            except FileNotFoundError:
                config = None
            else:
                if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
                    return True
                config = read_config_bytes(spec.config_path)
        except (ConfigFileError, OSError):
            return True
        if config is not None and spec.config.marker.encode("ascii") in config:
            return True
    for path in (
        *(path for path in spec.project_local_paths if path != spec.config_path),
        spec.launcher_path,
        spec.receipt_path,
        spec.journal_path,
        spec.lock_path,
    ):
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            return True
        return True
    return False


def _home(environment: Mapping[str, str]) -> Path:
    configured = environment.get("HOME")
    return Path.home() if configured is None else Path(configured).expanduser()


def _key_for_connect(
    environment: Mapping[str, str],
    *,
    dry_run: bool,
) -> InstallationKey:
    path = default_installation_key_path(environ=environment)
    if not dry_run:
        return load_or_create_installation_key(path)
    try:
        return load_installation_key(path)
    except FileNotFoundError:
        # Planning needs stable-shape identities but must never publish key material.
        return generate_installation_key()


def inspect_project_provider_installation(
    alias: ProviderAlias,
    project: Path,
    installation_key: InstallationKey,
    *,
    resolver: ProviderSpecResolver | None = None,
    capture_executable: str | os.PathLike[str] | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> InstallationStatus:
    """Inspect one fully materialized provider installation without changing it."""

    spec = resolve_provider_installation_spec(alias, project, resolver, environ=environ)
    if resolver is None and alias in _DYNAMIC_HOST_PROVIDERS:
        receipt = _load_receipt_optional(spec, installation_key)
        if receipt is not None:
            module = importlib.import_module(_PROVIDER_MODULES[alias])
            factory = module.provider_installation_spec
            spec = factory(
                project,
                environ=environ,
                host_version=receipt.host_version,
            )
    try:
        materialized = materialize_provider_launcher(
            spec,
            installation_key,
            capture_executable=capture_executable,
        )
    except CaptureCommandUnavailableError:
        status = inspect_provider_installation(spec, installation_key)
        if status.state is InstallationState.DISABLED and not status.drift:
            raise
        payload = status.model_dump(mode="python", warnings="error")
        payload["drift"] = (
            status.drift if "launcher" in status.drift else (*status.drift, "launcher")
        )
        payload["installed"] = False
        return InstallationStatus.model_validate(payload)
    return inspect_provider_installation(materialized, installation_key)


def _matching_store_connections(
    store: CaptureStore,
    spec: ProviderInstallationSpec,
    *,
    project_digest: str,
) -> tuple[CaptureConnectionSummary, ...]:
    return store.list_connections(
        project_digest=project_digest,
        profile_id=spec.profile,
    )


def _preflight_store_lifecycle(
    store: CaptureStore,
    spec: ProviderInstallationSpec,
    *,
    connection_id: str,
    project_digest: str,
) -> None:
    """Reject an upgrade that would cross an in-progress destructive lifecycle."""

    for connection in _matching_store_connections(
        store,
        spec,
        project_digest=project_digest,
    ):
        state = connection.state
        if state is CaptureConnectionState.DELETING or (
            connection.connection_id == connection_id and state is CaptureConnectionState.DRAINING
        ):
            raise CaptureStoreStateError()


def _activate_store_connection(
    spec: ProviderInstallationSpec,
    key: InstallationKey,
    *,
    environment: Mapping[str, str],
    spool: CaptureSpool,
) -> None:
    """Enable the installed generation, then retire its authenticated predecessors."""

    identity = derive_installation_identity(spec, key)
    locations = resolve_capture_store_locations(
        environ=environment,
        home=_home(environment),
    )
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        registration = store.register_connection(
            connection_id=identity.connection_id,
            project_digest=identity.project_digest,
            profile_id=spec.profile,
            capability_manifest_digest=spec.capability_digest,
            host_version=spec.host_version,
        )
        if registration.state is CaptureConnectionState.ENABLED:
            pass
        elif registration.state in (
            CaptureConnectionState.PENDING,
            CaptureConnectionState.DISABLED,
        ):
            store.transition_connection(
                identity.connection_id,
                expected_state=registration.state,
                target_state=CaptureConnectionState.ENABLED,
            )
        else:
            raise CaptureStoreStateError()

        matching = _matching_store_connections(
            store,
            spec,
            project_digest=identity.project_digest,
        )
        superseded = tuple(
            connection
            for connection in matching
            if connection.connection_id != identity.connection_id
        )
        if any(connection.state is CaptureConnectionState.DELETING for connection in superseded):
            raise CaptureStoreStateError()
        if not any(
            connection.state is not CaptureConnectionState.DISABLED for connection in superseded
        ):
            return

        # The installed launcher already targets the enabled new identity. The
        # spool fence prevents a writer that observed SQLITE_BUSY from queuing
        # behind the retirement boundary: it retries after the old identity is
        # draining and therefore fails closed instead.
        with spool.maintenance() as maintenance:
            matching = _matching_store_connections(
                store,
                spec,
                project_digest=identity.project_digest,
            )
            target = tuple(
                connection
                for connection in matching
                if connection.connection_id == identity.connection_id
            )
            if len(target) != 1 or target[0].state is not CaptureConnectionState.ENABLED:
                raise CaptureStoreStateError()
            superseded = tuple(
                connection
                for connection in matching
                if connection.connection_id != identity.connection_id
            )
            if any(
                connection.state is CaptureConnectionState.DELETING for connection in superseded
            ):
                raise CaptureStoreStateError()
            for connection in superseded:
                if connection.state is CaptureConnectionState.PENDING:
                    store.transition_connection(
                        connection.connection_id,
                        expected_state=CaptureConnectionState.PENDING,
                        target_state=CaptureConnectionState.ENABLED,
                    )
                elif connection.state not in (
                    CaptureConnectionState.ENABLED,
                    CaptureConnectionState.DRAINING,
                    CaptureConnectionState.DISABLED,
                ):
                    raise CaptureStoreStateError()
            drained = maintenance.drain(store)
            if drained.remaining_events != 0:
                raise CaptureStoreStateError()
            matching = _matching_store_connections(
                store,
                spec,
                project_digest=identity.project_digest,
            )
            for connection in matching:
                if connection.connection_id == identity.connection_id:
                    if connection.state is not CaptureConnectionState.ENABLED:
                        raise CaptureStoreStateError()
                elif connection.state is CaptureConnectionState.ENABLED:
                    store.transition_connection(
                        connection.connection_id,
                        expected_state=CaptureConnectionState.ENABLED,
                        target_state=CaptureConnectionState.DRAINING,
                    )
                elif connection.state not in (
                    CaptureConnectionState.DRAINING,
                    CaptureConnectionState.DISABLED,
                ):
                    raise CaptureStoreStateError()

        matching = _matching_store_connections(
            store,
            spec,
            project_digest=identity.project_digest,
        )
        for connection in matching:
            if connection.connection_id == identity.connection_id:
                if connection.state is not CaptureConnectionState.ENABLED:
                    raise CaptureStoreStateError()
            elif connection.state is CaptureConnectionState.DRAINING:
                store.transition_connection(
                    connection.connection_id,
                    expected_state=CaptureConnectionState.DRAINING,
                    target_state=CaptureConnectionState.DISABLED,
                )
            elif connection.state is not CaptureConnectionState.DISABLED:
                raise CaptureStoreStateError()


def run_connect(
    *,
    provider: str,
    project: str | os.PathLike[str] | Path | None = None,
    dry_run: bool = False,
    environ: Mapping[str, str] | None = None,
    spec_resolver: ProviderSpecResolver | None = None,
    capture_executable: str | os.PathLike[str] | Path | None = None,
) -> CaptureConnectReport:
    """Plan or install one authenticated passive capture integration."""

    if type(provider) is not str or type(dry_run) is not bool:
        raise CaptureCommandInputError()
    try:
        alias = ProviderAlias(provider)
    except ValueError:
        raise CaptureCommandInputError() from None
    resolved_project = resolve_capture_project(project)
    environment = os.environ if environ is None else environ
    if not isinstance(environment, Mapping):
        raise CaptureCommandConfigurationError()
    spec = resolve_provider_installation_spec(
        alias,
        resolved_project,
        spec_resolver,
        environ=environment,
        probe_host=spec_resolver is None and not dry_run,
    )
    try:
        key = _key_for_connect(environment, dry_run=dry_run)
        spec = materialize_provider_launcher(
            spec,
            key,
            capture_executable=capture_executable,
        )
        tracked_project_files = git_tracked_project_files(spec)
        if dry_run:
            status = install_provider(spec, key, dry_run=True)
        else:
            # Authenticate and plan the provider side before mutating the store.
            # Provider-specific preflight has already rejected fresh config
            # collisions before installation-key creation.
            install_provider(spec, key, dry_run=True)
            # Register PENDING before publishing the provider integration. A retry
            # completes either side of this boundary and enables admission last.
            identity = derive_installation_identity(spec, key)
            locations = resolve_capture_store_locations(
                environ=environment,
                home=_home(environment),
            )
            ensure_private_installation_directory(locations.state_directory)
            initialize_capture_store(locations.database_path)
            spool = CaptureSpool.open(locations, key)
            with CaptureStore.open(
                locations.database_path,
                installation_key=key,
                mode=CaptureStoreMode.MAINTENANCE,
            ) as store:
                _preflight_store_lifecycle(
                    store,
                    spec,
                    connection_id=identity.connection_id,
                    project_digest=identity.project_digest,
                )
                store.register_connection(
                    connection_id=identity.connection_id,
                    project_digest=identity.project_digest,
                    profile_id=spec.profile,
                    capability_manifest_digest=spec.capability_digest,
                    host_version=spec.host_version,
                )
            status = install_provider(spec, key)
            if status.state is not InstallationState.ENABLED or not status.installed:
                raise InstallationError()
            _activate_store_connection(
                spec,
                key,
                environment=environment,
                spool=spool,
            )
        return CaptureConnectReport(
            provider=alias,
            disposition=status.disposition,
            dry_run=dry_run,
            capture_enabled=not dry_run and status.state is InstallationState.ENABLED,
            project_local_files=len(spec.project_local_paths),
            git_tracked_files=len(tracked_project_files),
        )
    except CaptureCommandIntegrityError:
        raise
    except (
        CaptureMigrationIntegrityError,
        CaptureSpoolIntegrityError,
        CaptureStoreIntegrityError,
        InsecureKeyFileError,
        InvalidInstallationKeyError,
        InstallationError,
    ):
        raise CaptureCommandIntegrityError() from None
    except (
        CaptureMigrationError,
        CaptureSpoolError,
        CaptureStoreError,
        InsecureKeyPathError,
        OSError,
        TypeError,
        ValueError,
    ):
        raise CaptureCommandConfigurationError() from None


def render_connect_json(report: CaptureConnectReport) -> str:
    checked = CaptureConnectReport.model_validate(report)
    return canonical_json(checked.model_dump(mode="json", warnings=False)).decode("utf-8") + "\n"


def render_connect_human(report: CaptureConnectReport) -> str:
    checked = CaptureConnectReport.model_validate(report)
    action = "would install" if checked.dry_run else checked.disposition.value
    state = "disabled during dry-run" if checked.dry_run else "enabled"
    tracked = (
        " Project-local managed files are already Git-tracked." if checked.git_tracked_files else ""
    )
    return f"{checked.provider.value} capture: {action}; {state}.{tracked}\n"


__all__ = [
    "CaptureConnectReport",
    "ProviderSpecResolver",
    "inspect_project_provider_installation",
    "materialize_provider_launcher",
    "project_provider_artifacts_present",
    "render_connect_human",
    "render_connect_json",
    "resolve_provider_installation_spec",
    "run_connect",
]
