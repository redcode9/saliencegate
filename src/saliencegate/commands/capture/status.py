"""Content-free operational status for passive capture."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from saliencegate.capture import (
    CaptureConnectionState,
    CaptureProfile,
    CaptureSpool,
    CaptureSpoolError,
    CaptureStoreError,
    CaptureStoreIntegrityError,
    resolve_capture_store_locations,
)
from saliencegate.capture.sessions import CaptureHumanSessionId
from saliencegate.commands.capture.common import (
    CaptureCommandConfigurationError,
    CaptureCommandInputError,
    CaptureCommandIntegrityError,
    CaptureCommandUnavailableError,
    capture_project_digest,
    resolve_capture_project,
)
from saliencegate.commands.capture.connect import (
    ProviderSpecResolver,
    inspect_project_provider_installation,
    project_provider_artifacts_present,
)
from saliencegate.commands.capture.runtime import CaptureCommandRuntime, open_capture_runtime
from saliencegate.commands.capture.sessions import CaptureProviderAlias
from saliencegate.domain import canonical_json
from saliencegate.integrations.installation import InstallationError, InstallationState
from saliencegate.integrations.registry import ProviderAlias
from saliencegate.security import (
    InsecureKeyFileError,
    InsecureKeyPathError,
    InvalidInstallationKeyError,
    default_installation_key_path,
    load_installation_key,
)

_PROVIDERS: tuple[CaptureProviderAlias, ...] = ("codex", "claude-code", "opencode", "pi")
_PROFILES: dict[CaptureProviderAlias, CaptureProfile] = {
    "codex": CaptureProfile.CODEX_HOOKS_V1,
    "claude-code": CaptureProfile.CLAUDE_CODE_HOOKS_V1,
    "opencode": CaptureProfile.OPENCODE_PLUGIN_V1,
    "pi": CaptureProfile.PI_EXTENSION_V1,
}


class CaptureOperationalStatus(StrEnum):
    NOT_INSTALLED = "not_installed"
    # Reserved for a future host-attested installation state that has no store
    # binding. Current v1 providers cannot honestly distinguish that state from
    # either missing-connection drift or installed-but-not-yet-observed.
    INSTALLED = "installed"
    ACTIVE_OBSERVED = "active_observed"
    INSTALLED_NOT_OBSERVED = "installed_not_observed"
    DRIFTED = "drifted"
    DEGRADED = "degraded"


class CaptureStatusDrift(StrEnum):
    BOOTSTRAP = "bootstrap"
    BUNDLE = "bundle"
    CONFIG = "config"
    CONNECTION_GENERATION = "connection_generation"
    CONNECTION_MISSING = "connection_missing"
    MULTIPLE_CONNECTIONS = "multiple_connections"
    CONNECTION_PENDING = "connection_pending"
    CONNECTION_DRAINING = "connection_draining"
    CONNECTION_DELETING = "connection_deleting"
    INSTALLATION_STATE = "installation_state"
    HOST_VERSION = "host_version"
    LAUNCHER = "launcher"
    LOCK = "lock"
    RECEIPT = "receipt"
    SPOOL_MISSING = "spool_missing"
    SPOOL_DROP = "spool_drop"
    SESSION_DEGRADED = "session_degraded"


class _StatusModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<redacted>)"

    __str__ = __repr__


class CaptureProviderStatus(_StatusModel):
    schema_version: Literal["capture-provider-status/v1"] = "capture-provider-status/v1"
    provider: CaptureProviderAlias
    status: CaptureOperationalStatus
    connector_available: bool
    connection_state: CaptureConnectionState | None
    session_count: Annotated[int, Field(ge=0)]
    quarantined_sessions: Annotated[int, Field(ge=0)]
    queued_spool_events: Annotated[int, Field(ge=0)]
    dropped_spool_events: Annotated[int, Field(ge=0)]
    oldest_session: Annotated[CaptureHumanSessionId | None, Field(repr=False)]
    local_bytes: Annotated[int, Field(ge=0)]
    drift: Annotated[tuple[CaptureStatusDrift, ...], Field(max_length=len(CaptureStatusDrift))]


class CaptureStatusReport(_StatusModel):
    schema_version: Literal["capture-status/v1"] = "capture-status/v1"
    providers: Annotated[tuple[CaptureProviderStatus, ...], Field(min_length=1, max_length=4)]


def _connector_available(alias: CaptureProviderAlias) -> bool:
    try:
        from saliencegate.integrations.registry import BUILTIN_PROVIDER_REGISTRY

        return BUILTIN_PROVIDER_REGISTRY.resolve(alias, require_available=False).available
    except Exception:
        return False


def _local_bytes(runtime: CaptureCommandRuntime) -> int:
    total = 0
    paths = [runtime.locations.database_path]
    paths.extend(Path(f"{runtime.locations.database_path}{suffix}") for suffix in ("-wal", "-shm"))
    if runtime.spool is not None:
        try:
            paths.extend(runtime.locations.spool_directory.iterdir())
        except OSError:
            raise CaptureCommandIntegrityError() from None
    try:
        for path in paths:
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(metadata.st_mode):
                total += metadata.st_size
    except OSError:
        raise CaptureCommandIntegrityError() from None
    return total


def _absent(alias: CaptureProviderAlias) -> CaptureProviderStatus:
    return CaptureProviderStatus(
        provider=alias,
        status=CaptureOperationalStatus.NOT_INSTALLED,
        connector_available=_connector_available(alias),
        connection_state=None,
        session_count=0,
        quarantined_sessions=0,
        queued_spool_events=0,
        dropped_spool_events=0,
        oldest_session=None,
        local_bytes=0,
        drift=(),
    )


def _status_without_runtime(
    aliases: tuple[CaptureProviderAlias, ...],
    *,
    project: Path,
    environ: Mapping[str, str] | None,
    spec_resolver: ProviderSpecResolver | None,
    capture_executable: str | os.PathLike[str] | Path | None,
) -> CaptureStatusReport:
    environment = os.environ if environ is None else environ
    if not isinstance(environment, Mapping):
        raise CaptureCommandConfigurationError()
    configured_home = environment.get("HOME")
    home = Path.home() if configured_home is None else Path(configured_home).expanduser()
    locations = resolve_capture_store_locations(environ=environment, home=home)
    try:
        installation_key = load_installation_key(default_installation_key_path(environ=environment))
    except FileNotFoundError:
        runtime_state_exists = False
        for path in (locations.database_path, locations.spool_directory):
            try:
                path.lstat()
            except FileNotFoundError:
                continue
            runtime_state_exists = True
            break
        if runtime_state_exists:
            raise CaptureCommandIntegrityError() from None
        missing_key_providers: list[CaptureProviderStatus] = []
        for alias in aliases:
            try:
                artifacts_present = project_provider_artifacts_present(
                    ProviderAlias(alias),
                    project,
                    resolver=spec_resolver,
                    environ=environment,
                )
            except CaptureCommandUnavailableError:
                artifacts_present = False
            if not artifacts_present:
                missing_key_providers.append(_absent(alias))
                continue
            missing_key_providers.append(
                CaptureProviderStatus(
                    provider=alias,
                    status=CaptureOperationalStatus.DRIFTED,
                    connector_available=_connector_available(alias),
                    connection_state=None,
                    session_count=0,
                    quarantined_sessions=0,
                    queued_spool_events=0,
                    dropped_spool_events=0,
                    oldest_session=None,
                    local_bytes=0,
                    drift=(
                        CaptureStatusDrift.CONNECTION_MISSING,
                        CaptureStatusDrift.RECEIPT,
                        CaptureStatusDrift.SPOOL_MISSING,
                    ),
                )
            )
        return CaptureStatusReport(providers=tuple(missing_key_providers))
    try:
        locations.database_path.lstat()
        database_exists = True
    except FileNotFoundError:
        database_exists = False
    except OSError:
        raise CaptureCommandIntegrityError() from None
    try:
        locations.spool_directory.lstat()
        spool_exists = True
    except FileNotFoundError:
        spool_exists = False
    except OSError:
        raise CaptureCommandIntegrityError() from None
    try:
        spool_health = (
            CaptureSpool.audit_read_only(locations, installation_key=installation_key)
            if spool_exists
            else None
        )
        local_paths = [locations.database_path] if database_exists else []
        if spool_exists:
            local_paths.extend(locations.spool_directory.iterdir())
        local_bytes = sum(
            metadata.st_size
            for path in local_paths
            if stat.S_ISREG((metadata := path.lstat()).st_mode)
        )
    except (CaptureSpoolError, OSError):
        raise CaptureCommandIntegrityError() from None
    providers: list[CaptureProviderStatus] = []
    for alias in aliases:
        try:
            installation = inspect_project_provider_installation(
                ProviderAlias(alias),
                project,
                installation_key,
                resolver=spec_resolver,
                capture_executable=capture_executable,
                environ=environment,
            )
        except CaptureCommandUnavailableError:
            providers.append(_absent(alias))
            continue
        except InstallationError:
            raise CaptureCommandIntegrityError() from None
        if installation.state is InstallationState.DISABLED and not installation.drift:
            providers.append(_absent(alias))
            continue
        drift = {CaptureStatusDrift.CONNECTION_MISSING}
        if not spool_exists:
            drift.add(CaptureStatusDrift.SPOOL_MISSING)
        if spool_health is not None and spool_health.dropped_events:
            drift.add(CaptureStatusDrift.SPOOL_DROP)
        drift.update(CaptureStatusDrift(item) for item in installation.drift)
        providers.append(
            CaptureProviderStatus(
                provider=alias,
                status=CaptureOperationalStatus.DRIFTED,
                connector_available=_connector_available(alias),
                connection_state=None,
                session_count=0,
                quarantined_sessions=0,
                queued_spool_events=(0 if spool_health is None else spool_health.queued_events),
                dropped_spool_events=(0 if spool_health is None else spool_health.dropped_events),
                oldest_session=None,
                local_bytes=local_bytes,
                drift=tuple(sorted(drift, key=lambda item: item.value)),
            )
        )
    return CaptureStatusReport(providers=tuple(providers))


def run_status(
    *,
    provider: str | None = None,
    project: str | os.PathLike[str] | Path | None = None,
    environ: Mapping[str, str] | None = None,
    spec_resolver: ProviderSpecResolver | None = None,
    capture_executable: str | os.PathLike[str] | Path | None = None,
) -> CaptureStatusReport:
    """Drain and summarize capture health without exposing operational paths."""

    if provider is None:
        aliases = _PROVIDERS
    elif type(provider) is str and provider in _PROVIDERS:
        aliases = (provider,)
    else:
        raise CaptureCommandInputError()
    resolved = resolve_capture_project(project)
    try:
        with open_capture_runtime(project=resolved, environ=environ, drain=True) as runtime:
            project_id = capture_project_digest(
                runtime.project,
                installation_key=runtime.installation_key,
            )
            local_bytes = _local_bytes(runtime)
            spool_health = None if runtime.spool is None else runtime.spool.health()
            providers: list[CaptureProviderStatus] = []
            for alias in aliases:
                try:
                    installation = inspect_project_provider_installation(
                        ProviderAlias(alias),
                        resolved,
                        runtime.installation_key,
                        resolver=spec_resolver,
                        capture_executable=capture_executable,
                        environ=environ,
                    )
                except CaptureCommandUnavailableError:
                    installation = None
                except InstallationError:
                    raise CaptureCommandIntegrityError() from None
                connections = runtime.store.list_connections(
                    project_digest=project_id,
                    profile_id=_PROFILES[alias],
                )
                inventory = runtime.store.session_inventory(
                    project_digest=project_id,
                    profile_id=_PROFILES[alias],
                )
                if not connections and (
                    installation is None
                    or (installation.state is InstallationState.DISABLED and not installation.drift)
                ):
                    providers.append(_absent(alias))
                    continue
                active_connections = tuple(
                    item
                    for item in connections
                    if item.state is not CaptureConnectionState.DISABLED
                )
                matched = (
                    None
                    if installation is None
                    else next(
                        (
                            item
                            for item in connections
                            if item.connection_id == installation.connection_id
                        ),
                        None,
                    )
                )
                selected = (
                    matched
                    if matched is not None
                    else (
                        active_connections[-1]
                        if active_connections
                        else (connections[-1] if connections else None)
                    )
                )
                drift: set[CaptureStatusDrift] = set()
                if len(active_connections) > 1:
                    drift.add(CaptureStatusDrift.MULTIPLE_CONNECTIONS)
                if installation is not None:
                    drift.update(CaptureStatusDrift(item) for item in installation.drift)
                    if installation.state is InstallationState.ENABLED:
                        if matched is None:
                            drift.add(CaptureStatusDrift.CONNECTION_MISSING)
                        elif matched.state is not CaptureConnectionState.ENABLED:
                            drift.add(CaptureStatusDrift.INSTALLATION_STATE)
                    elif (
                        matched is not None and matched.state is not CaptureConnectionState.DISABLED
                    ):
                        drift.add(CaptureStatusDrift.INSTALLATION_STATE)
                    if any(
                        item.connection_id != installation.connection_id
                        for item in active_connections
                    ):
                        drift.add(CaptureStatusDrift.CONNECTION_GENERATION)
                if selected is None:
                    drift.add(CaptureStatusDrift.CONNECTION_MISSING)
                elif selected.state is CaptureConnectionState.PENDING:
                    drift.add(CaptureStatusDrift.CONNECTION_PENDING)
                elif selected.state is CaptureConnectionState.DRAINING:
                    drift.add(CaptureStatusDrift.CONNECTION_DRAINING)
                elif selected.state is CaptureConnectionState.DELETING:
                    drift.add(CaptureStatusDrift.CONNECTION_DELETING)
                if runtime.spool is None and (
                    inventory.session_count
                    or (
                        selected is not None
                        and selected.state is not CaptureConnectionState.DISABLED
                    )
                    or (
                        installation is not None and installation.state is InstallationState.ENABLED
                    )
                ):
                    drift.add(CaptureStatusDrift.SPOOL_MISSING)
                if spool_health is not None and spool_health.dropped_events:
                    drift.add(CaptureStatusDrift.SPOOL_DROP)
                if inventory.degraded_sessions:
                    drift.add(CaptureStatusDrift.SESSION_DEGRADED)
                degradation = drift & {
                    CaptureStatusDrift.SPOOL_DROP,
                    CaptureStatusDrift.SESSION_DEGRADED,
                }
                if drift - degradation:
                    status = CaptureOperationalStatus.DRIFTED
                elif degradation:
                    status = CaptureOperationalStatus.DEGRADED
                elif selected is None:
                    status = CaptureOperationalStatus.INSTALLED
                elif selected.state is CaptureConnectionState.DISABLED:
                    status = CaptureOperationalStatus.NOT_INSTALLED
                elif inventory.session_count:
                    status = CaptureOperationalStatus.ACTIVE_OBSERVED
                else:
                    status = CaptureOperationalStatus.INSTALLED_NOT_OBSERVED
                providers.append(
                    CaptureProviderStatus(
                        provider=alias,
                        status=status,
                        connector_available=_connector_available(alias),
                        connection_state=None if selected is None else selected.state,
                        session_count=inventory.session_count,
                        quarantined_sessions=inventory.quarantined_sessions,
                        queued_spool_events=(
                            0 if spool_health is None else spool_health.queued_events
                        ),
                        dropped_spool_events=(
                            0 if spool_health is None else spool_health.dropped_events
                        ),
                        oldest_session=inventory.oldest_session,
                        local_bytes=local_bytes,
                        drift=tuple(sorted(drift, key=lambda item: item.value)),
                    )
                )
            return CaptureStatusReport(providers=tuple(providers))
    except CaptureCommandUnavailableError:
        try:
            return _status_without_runtime(
                aliases,
                project=resolved,
                environ=environ,
                spec_resolver=spec_resolver,
                capture_executable=capture_executable,
            )
        except (InsecureKeyFileError, InvalidInstallationKeyError):
            raise CaptureCommandIntegrityError() from None
        except (InsecureKeyPathError, OSError, TypeError, ValueError):
            raise CaptureCommandConfigurationError() from None
    except (CaptureStoreIntegrityError, CaptureSpoolError):
        raise CaptureCommandIntegrityError() from None
    except CaptureStoreError:
        raise CaptureCommandConfigurationError() from None


def render_status_json(report: CaptureStatusReport) -> str:
    checked = CaptureStatusReport.model_validate(report)
    return canonical_json(checked.model_dump(mode="json", warnings=False)).decode("utf-8") + "\n"


def render_status_human(report: CaptureStatusReport) -> str:
    checked = CaptureStatusReport.model_validate(report)
    lines = ["Passive capture status"]
    for item in checked.providers:
        oldest = "none" if item.oldest_session is None else item.oldest_session
        detail = (
            f"sessions={item.session_count}; quarantined={item.quarantined_sessions}; "
            f"oldest={oldest}; bytes={item.local_bytes}; queued={item.queued_spool_events}; "
            f"dropped={item.dropped_spool_events}"
        )
        if item.drift:
            detail += "; drift=" + ",".join(value.value for value in item.drift)
        lines.append(f"{item.provider}: {item.status.value} ({detail})")
    return "\n".join(lines) + "\n"


__all__ = [
    "CaptureOperationalStatus",
    "CaptureProviderStatus",
    "CaptureStatusDrift",
    "CaptureStatusReport",
    "render_status_human",
    "render_status_json",
    "run_status",
]
