"""Remove one owned passive-capture integration while retaining capture data."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict

from saliencegate.capture import (
    CaptureConnectionState,
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
    resolve_capture_store_locations,
)
from saliencegate.commands.capture.common import (
    CaptureCommandConfigurationError,
    CaptureCommandInputError,
    CaptureCommandIntegrityError,
    CaptureCommandUnavailableError,
    resolve_capture_project,
)
from saliencegate.commands.capture.connect import (
    ProviderSpecResolver,
    resolve_provider_installation_spec,
)
from saliencegate.domain import canonical_json
from saliencegate.integrations.installation import (
    InstallationDisposition,
    InstallationError,
    InstallationState,
    _load_receipt_optional,
    derive_installation_identity,
    inspect_provider_installation,
    uninstall_provider,
)
from saliencegate.integrations.registry import (
    MAX_INTEGRATION_BUNDLE_BYTES,
    MAX_INTEGRATION_LAUNCHER_BYTES,
    ProviderAlias,
    ProviderInstallationSpec,
)
from saliencegate.security import (
    InsecureKeyFileError,
    InsecureKeyPathError,
    InstallationKey,
    InvalidInstallationKeyError,
    default_installation_key_path,
    load_installation_key,
)
from saliencegate.security.files import StableReadPolicy, read_stable_file
from saliencegate.security.windows import (
    NativeWindowsSecurityOperations,
    WindowsPathKind,
    authorize_windows_managed_path,
)


class CaptureDisconnectReport(BaseModel):
    """Content-free disconnect result."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    schema_version: Literal["capture-disconnect/v1"] = "capture-disconnect/v1"
    provider: ProviderAlias
    disposition: InstallationDisposition
    capture_enabled: Literal[False] = False
    capture_retained: Literal[True] = True

    def __repr__(self) -> str:
        return "CaptureDisconnectReport(<redacted>)"

    __str__ = __repr__


def _home(environment: Mapping[str, str]) -> Path:
    configured = environment.get("HOME")
    return Path.home() if configured is None else Path(configured).expanduser()


def _uninstall_to_disabled(
    spec: ProviderInstallationSpec,
    key: InstallationKey,
) -> InstallationDisposition:
    # The installer can first recover an interrupted install. A bounded second
    # pass then performs the requested uninstall; an interrupted uninstall
    # recovers directly to DISABLED on its first pass.
    status = inspect_provider_installation(spec, key)
    if status.state is InstallationState.DISABLED:
        if status.drift:
            raise InstallationError()
        return InstallationDisposition.NOOP
    for _attempt in range(2):
        status = uninstall_provider(spec, key)
        if status.state is InstallationState.DISABLED:
            return status.disposition
    raise InstallationError()


def _read_installed_private_asset(
    path: Path,
    *,
    maximum_bytes: int,
    policy: StableReadPolicy,
) -> bytes:
    """Read one receipt-bound private asset through the host's native boundary."""

    try:
        if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
            return _read_installed_private_asset_windows(path, maximum_bytes=maximum_bytes)
        return read_stable_file(
            path,
            maximum_bytes=maximum_bytes,
            policy=policy,
        ).data
    except Exception:
        raise InstallationError() from None


def _read_installed_private_asset_windows(path: Path, *, maximum_bytes: int) -> bytes:
    operations = NativeWindowsSecurityOperations()
    windows_path = PureWindowsPath(os.fspath(path))
    parent = authorize_windows_managed_path(
        windows_path.parent,
        kind=WindowsPathKind.DIRECTORY,
        operations=operations,
    )
    stable = operations.read_private_file(
        windows_path,
        maximum_bytes=maximum_bytes,
    )
    parent.revalidate()
    stable.authorization.revalidate()
    return stable.data


def _resolve_installed_spec(
    requested: ProviderInstallationSpec,
    key: InstallationKey,
) -> tuple[ProviderInstallationSpec, str | None]:
    """Bind uninstall to an authenticated receipt from this or a prior generation."""

    try:
        receipt = _load_receipt_optional(requested, key)
        if receipt is None:
            return requested, None
        requested_identity = derive_installation_identity(requested, key)
        if (
            receipt.provider_id != requested.provider_id
            or receipt.profile is not requested.profile
            or receipt.host_version != requested.host_version
            or receipt.generation > requested.generation
            or receipt.project_digest != requested_identity.project_digest
            or receipt.capability_digest != requested.capability_digest
            or receipt.config_path != requested.config_path
            or receipt.bootstrap_path != requested.bootstrap_path
            or receipt.launcher_path != requested.launcher_path
            or receipt.receipt_path != requested.receipt_path
            or receipt.journal_path != requested.journal_path
            or receipt.lock_path != requested.lock_path
            or (
                receipt.generation == requested.generation
                and (
                    receipt.bundle_path != requested.bundle_path
                    or receipt.bundle_digest != requested.bundle_digest
                )
            )
        ):
            raise InstallationError()

        bundle_bytes = requested.bundle_bytes
        launcher_bytes = requested.launcher_bytes
        if receipt.state is not InstallationState.DISABLED:
            bundle_bytes = _read_installed_private_asset(
                receipt.bundle_path,
                maximum_bytes=MAX_INTEGRATION_BUNDLE_BYTES,
                policy=StableReadPolicy.PRIVATE_EXACT,
            )
            launcher_bytes = _read_installed_private_asset(
                receipt.launcher_path,
                maximum_bytes=MAX_INTEGRATION_LAUNCHER_BYTES,
                policy=StableReadPolicy.PRIVATE_EXECUTABLE,
            )

        payload = requested.model_dump(mode="python", warnings="error")
        payload.update(
            bundle_bytes=bundle_bytes,
            bundle_path=receipt.bundle_path,
            generation=receipt.generation,
            launcher_bytes=launcher_bytes,
        )
        installed = ProviderInstallationSpec.model_validate(payload)
        installed_identity = derive_installation_identity(installed, key)
        if installed_identity.connection_id != receipt.connection_id or (
            receipt.state is not InstallationState.DISABLED
            and (
                installed.bundle_digest != receipt.bundle_digest
                or installed.launcher_digest != receipt.launcher_digest
            )
        ):
            raise InstallationError()
        return installed, receipt.connection_id
    except InstallationError:
        raise
    except Exception:
        raise InstallationError() from None


def run_disconnect(
    *,
    provider: str,
    project: str | os.PathLike[str] | Path | None = None,
    environ: Mapping[str, str] | None = None,
    spec_resolver: ProviderSpecResolver | None = None,
    capture_executable: str | os.PathLike[str] | Path | None = None,
) -> CaptureDisconnectReport:
    """Disable admission, reverse owned config, and retain local observations."""

    if type(provider) is not str:
        raise CaptureCommandInputError()
    try:
        alias = ProviderAlias(provider)
    except ValueError:
        raise CaptureCommandInputError() from None
    resolved_project = resolve_capture_project(project)
    environment = os.environ if environ is None else environ
    if not isinstance(environment, Mapping):
        raise CaptureCommandConfigurationError()
    spec = resolve_provider_installation_spec(alias, resolved_project, spec_resolver)
    try:
        key = load_installation_key(default_installation_key_path(environ=environment))
        identity = derive_installation_identity(spec, key)
        locations = resolve_capture_store_locations(
            environ=environment,
            home=_home(environment),
        )
        try:
            locations.database_path.lstat()
        except FileNotFoundError:
            installed, _installed_connection_id = _resolve_installed_spec(spec, key)
            disposition = _uninstall_to_disabled(installed, key)
            return CaptureDisconnectReport(provider=alias, disposition=disposition)
        with CaptureStore.open(
            locations.database_path,
            installation_key=key,
            mode=CaptureStoreMode.MAINTENANCE,
        ) as store:
            connections = store.list_connections(
                project_digest=identity.project_digest,
                profile_id=spec.profile,
            )
            if not connections or any(
                item.state is CaptureConnectionState.DELETING for item in connections
            ):
                raise CaptureStoreStateError()

            if any(item.state is not CaptureConnectionState.DISABLED for item in connections):
                spool = CaptureSpool.open(locations, key)
                with spool.maintenance() as maintenance:
                    connections = store.list_connections(
                        project_digest=identity.project_digest,
                        profile_id=spec.profile,
                    )
                    if not connections or any(
                        item.state is CaptureConnectionState.DELETING for item in connections
                    ):
                        raise CaptureStoreStateError()
                    for connection in connections:
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
                    connections = store.list_connections(
                        project_digest=identity.project_digest,
                        profile_id=spec.profile,
                    )
                    if not connections or any(
                        item.state
                        not in (
                            CaptureConnectionState.ENABLED,
                            CaptureConnectionState.DRAINING,
                            CaptureConnectionState.DISABLED,
                        )
                        for item in connections
                    ):
                        raise CaptureStoreStateError()
                    for connection in connections:
                        if connection.state is CaptureConnectionState.ENABLED:
                            store.transition_connection(
                                connection.connection_id,
                                expected_state=CaptureConnectionState.ENABLED,
                                target_state=CaptureConnectionState.DRAINING,
                            )

            spec, installed_connection_id = _resolve_installed_spec(spec, key)
            if installed_connection_id is not None:
                selected = tuple(
                    item for item in connections if item.connection_id == installed_connection_id
                )
                if len(selected) != 1:
                    raise CaptureStoreStateError()
            disposition = _uninstall_to_disabled(spec, key)
            connections = store.list_connections(
                project_digest=identity.project_digest,
                profile_id=spec.profile,
            )
            selected = tuple(
                item for item in connections if item.connection_id == installed_connection_id
            )
            if (
                not connections
                or (installed_connection_id is not None and len(selected) != 1)
                or any(
                    item.state
                    not in (
                        CaptureConnectionState.DRAINING,
                        CaptureConnectionState.DISABLED,
                    )
                    for item in connections
                )
            ):
                raise CaptureStoreStateError()
            for connection in connections:
                if connection.state is CaptureConnectionState.DRAINING:
                    store.transition_connection(
                        connection.connection_id,
                        expected_state=CaptureConnectionState.DRAINING,
                        target_state=CaptureConnectionState.DISABLED,
                    )
        return CaptureDisconnectReport(provider=alias, disposition=disposition)
    except CaptureCommandUnavailableError:
        raise
    except FileNotFoundError:
        raise CaptureCommandUnavailableError() from None
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


def render_disconnect_json(report: CaptureDisconnectReport) -> str:
    checked = CaptureDisconnectReport.model_validate(report)
    return canonical_json(checked.model_dump(mode="json", warnings=False)).decode("utf-8") + "\n"


def render_disconnect_human(report: CaptureDisconnectReport) -> str:
    checked = CaptureDisconnectReport.model_validate(report)
    return (
        f"{checked.provider.value} capture: {checked.disposition.value}; disabled. "
        "Existing capture data was retained.\n"
    )


__all__ = [
    "CaptureDisconnectReport",
    "render_disconnect_human",
    "render_disconnect_json",
    "run_disconnect",
]
