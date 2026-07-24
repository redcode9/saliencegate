"""Install and manage provider capture for every project of the current user."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.capture.locations import resolve_capture_store_locations
from saliencegate.capture.migrations import initialize_capture_store
from saliencegate.capture.scopes import (
    CaptureGlobalParentState,
    CaptureGlobalProvider,
    derive_global_config_root_digest,
    derive_global_parent_id,
)
from saliencegate.capture.spool import CaptureSpool
from saliencegate.capture.store import (
    CaptureStore,
    CaptureStoreMode,
    CaptureStoreStateError,
)
from saliencegate.commands.capture.common import (
    CaptureCommandConfigurationError,
    CaptureCommandInputError,
    CaptureCommandIntegrityError,
    CaptureCommandUnavailableError,
)
from saliencegate.commands.capture.connect import _home, _key_for_connect
from saliencegate.commands.setup import (
    SetupProviderPlan,
    SetupProviderResult,
    SetupScope,
    SetupScopeRequest,
)
from saliencegate.domain import canonical_json
from saliencegate.integrations.global_installation import (
    GlobalInstallationError,
    global_provider_installation_spec,
)
from saliencegate.integrations.global_runtime import resolve_global_project_root
from saliencegate.integrations.installation import (
    InstallationDisposition,
    InstallationError,
    InstallationReceipt,
    InstallationState,
    _load_receipt_optional,
    ensure_private_installation_directory,
    inspect_provider_installation,
    install_provider,
    uninstall_provider,
)
from saliencegate.integrations.launcher_materialization import materialize_provider_launcher
from saliencegate.integrations.registry import ProviderAlias, ProviderInstallationSpec
from saliencegate.security import (
    InsecureKeyFileError,
    InvalidInstallationKeyError,
    default_installation_key_path,
    load_installation_key,
)


class GlobalCaptureConnectReport(BaseModel):
    """Content-free global setup result."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    schema_version: Literal["global-capture-connect/v1"] = "global-capture-connect/v1"
    provider: ProviderAlias
    disposition: InstallationDisposition
    dry_run: bool
    capture_enabled: bool
    managed_files: Annotated[int, Field(ge=0, le=3)]
    excluded_projects: Annotated[int, Field(ge=0, le=1_000)]

    @model_validator(mode="after")
    def state_matches_disposition(self) -> GlobalCaptureConnectReport:
        if self.dry_run is not (self.disposition is InstallationDisposition.PLANNED):
            raise ValueError("global capture disposition is inconsistent")
        if self.capture_enabled is self.dry_run:
            raise ValueError("global capture state is inconsistent")
        return self

    def __repr__(self) -> str:
        return "GlobalCaptureConnectReport(<redacted>)"

    __str__ = __repr__


class GlobalCaptureDisconnectReport(BaseModel):
    """Content-free global disconnect result."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    schema_version: Literal["global-capture-disconnect/v1"] = "global-capture-disconnect/v1"
    provider: ProviderAlias
    disposition: InstallationDisposition

    def __repr__(self) -> str:
        return "GlobalCaptureDisconnectReport(<redacted>)"

    __str__ = __repr__


class GlobalCaptureStatusValue(StrEnum):
    NOT_INSTALLED = "not_installed"
    ENABLED = "enabled"
    DISABLED = "disabled"
    DRIFTED = "drifted"


class GlobalCaptureProviderStatus(BaseModel):
    """One provider-global status without paths or identifiers."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    provider: ProviderAlias
    status: GlobalCaptureStatusValue
    projects: Annotated[int, Field(ge=0, le=1_000)]
    exclusions: Annotated[int, Field(ge=0, le=1_000)]
    drift: tuple[str, ...] = ()


class GlobalCaptureStatusReport(BaseModel):
    """Bounded status for one or all global provider installations."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    schema_version: Literal["global-capture-status/v1"] = "global-capture-status/v1"
    providers: Annotated[
        tuple[GlobalCaptureProviderStatus, ...],
        Field(min_length=1, max_length=4),
    ]


def render_global_connect_json(report: GlobalCaptureConnectReport) -> str:
    checked = GlobalCaptureConnectReport.model_validate(report)
    return canonical_json(checked.model_dump(mode="json", warnings=False)).decode("utf-8") + "\n"


def render_global_connect_human(report: GlobalCaptureConnectReport) -> str:
    checked = GlobalCaptureConnectReport.model_validate(report)
    action = "would install" if checked.dry_run else checked.disposition.value
    state = "disabled during dry-run" if checked.dry_run else "enabled"
    return (
        f"{checked.provider.value} global capture: {action}; {state}. "
        f"Managed files: {checked.managed_files}; "
        f"excluded projects: {checked.excluded_projects}.\n"
    )


def render_global_disconnect_json(report: GlobalCaptureDisconnectReport) -> str:
    checked = GlobalCaptureDisconnectReport.model_validate(report)
    return canonical_json(checked.model_dump(mode="json", warnings=False)).decode("utf-8") + "\n"


def render_global_disconnect_human(report: GlobalCaptureDisconnectReport) -> str:
    checked = GlobalCaptureDisconnectReport.model_validate(report)
    return (
        f"{checked.provider.value} global capture: {checked.disposition.value}; "
        "local session data retained.\n"
    )


def render_global_status_json(report: GlobalCaptureStatusReport) -> str:
    checked = GlobalCaptureStatusReport.model_validate(report)
    return canonical_json(checked.model_dump(mode="json", warnings=False)).decode("utf-8") + "\n"


def render_global_status_human(report: GlobalCaptureStatusReport) -> str:
    checked = GlobalCaptureStatusReport.model_validate(report)
    lines = ["SalienceGate global capture"]
    lines.extend(
        (
            f"{item.provider.value}: {item.status.value}; "
            f"projects {item.projects}; exclusions {item.exclusions}"
            + (f"; drift {','.join(item.drift)}" if item.drift else "")
        )
        for item in checked.providers
    )
    return "\n".join(lines) + "\n"


def _environment(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    selected = os.environ if environ is None else environ
    if not isinstance(selected, Mapping):
        raise CaptureCommandConfigurationError()
    return selected


def _installed_spec(
    provider: ProviderAlias,
    key: object,
    *,
    environment: Mapping[str, str],
) -> tuple[ProviderInstallationSpec, InstallationReceipt | None]:
    try:
        from saliencegate.security import InstallationKey

        if type(key) is not InstallationKey:
            raise InstallationError()
        base = global_provider_installation_spec(provider, environ=environment)
        receipt = _load_receipt_optional(base, key)
        if receipt is None:
            return base, None
        return (
            global_provider_installation_spec(
                provider,
                environ=environment,
                host_version=receipt.host_version,
            ),
            receipt,
        )
    except GlobalInstallationError:
        raise CaptureCommandUnavailableError() from None
    except InstallationError:
        raise
    except Exception:
        raise CaptureCommandConfigurationError() from None


def _resolved_exclusions(values: Sequence[str | os.PathLike[str] | Path]) -> tuple[Path, ...]:
    try:
        if isinstance(values, (str, bytes, os.PathLike)):
            raise CaptureCommandInputError()
        resolved = tuple(
            resolve_global_project_root(os.fspath(Path(value).expanduser().resolve(strict=True)))
            for value in values
        )
        if len(resolved) > 1_000:
            raise CaptureCommandInputError()
        return tuple(dict.fromkeys(resolved))
    except CaptureCommandInputError:
        raise
    except (OSError, TypeError, ValueError):
        raise CaptureCommandInputError() from None


def _parent_coordinates(
    spec: ProviderInstallationSpec,
    key: object,
) -> tuple[CaptureGlobalProvider, str, str]:
    try:
        from saliencegate.security import InstallationKey

        if type(key) is not InstallationKey:
            raise CaptureCommandIntegrityError()
        provider = CaptureGlobalProvider(spec.provider_id)
        config_digest = derive_global_config_root_digest(
            os.fsencode(spec.project_root),
            key,
        )
        return (
            provider,
            config_digest,
            derive_global_parent_id(
                provider_id=provider,
                config_root_digest=config_digest,
                generation=spec.generation,
                installation_key=key,
            ),
        )
    except CaptureCommandIntegrityError:
        raise
    except Exception:
        raise CaptureCommandIntegrityError() from None


def run_global_connect(
    *,
    provider: str,
    exclusions: Sequence[str | os.PathLike[str] | Path] = (),
    dry_run: bool = False,
    environ: Mapping[str, str] | None = None,
    capture_executable: str | os.PathLike[str] | Path | None = None,
) -> GlobalCaptureConnectReport:
    """Plan or install one user-global provider integration."""

    if type(provider) is not str or type(dry_run) is not bool:
        raise CaptureCommandInputError()
    try:
        alias = ProviderAlias(provider)
    except ValueError:
        raise CaptureCommandInputError() from None
    environment = _environment(environ)
    excluded = _resolved_exclusions(exclusions)
    try:
        key = _key_for_connect(environment, dry_run=dry_run)
        spec = global_provider_installation_spec(
            alias,
            environ=environment,
            probe_host=not dry_run,
        )
        spec = materialize_provider_launcher(
            spec,
            key,
            capture_executable=capture_executable,
        )
        if dry_run:
            status = install_provider(spec, key, dry_run=True)
        else:
            install_provider(spec, key, dry_run=True)
            locations = resolve_capture_store_locations(
                environ=environment,
                home=_home(environment),
            )
            ensure_private_installation_directory(locations.state_directory)
            initialize_capture_store(locations.database_path)
            CaptureSpool.open(locations, key)
            global_provider, config_digest, parent_id = _parent_coordinates(spec, key)
            with CaptureStore.open(
                locations.database_path,
                installation_key=key,
                mode=CaptureStoreMode.MAINTENANCE,
            ) as store:
                registration = store.register_global_parent(
                    provider_id=global_provider,
                    config_root_digest=config_digest,
                    profile_id=spec.profile,
                    capability_manifest_digest=spec.capability_digest,
                    host_version=spec.host_version,
                    generation=spec.generation,
                )
                if registration.state in (
                    CaptureGlobalParentState.DRAINING,
                    CaptureGlobalParentState.DELETING,
                ):
                    raise CaptureStoreStateError()
            status = install_provider(spec, key)
            if status.state is not InstallationState.ENABLED or not status.installed:
                raise InstallationError()
            with CaptureStore.open(
                locations.database_path,
                installation_key=key,
                mode=CaptureStoreMode.MAINTENANCE,
            ) as store:
                parent = store.get_global_parent(parent_id)
                if parent.state in (
                    CaptureGlobalParentState.PENDING,
                    CaptureGlobalParentState.DISABLED,
                ):
                    store.transition_global_parent(
                        parent_id,
                        expected_state=parent.state,
                        target_state=CaptureGlobalParentState.ENABLED,
                    )
                elif parent.state is not CaptureGlobalParentState.ENABLED:
                    raise CaptureStoreStateError()
                store.replace_global_exclusions(
                    parent_id,
                    tuple(os.fsencode(path) for path in excluded),
                )
                for superseded in store.list_global_parents(
                    provider_id=global_provider,
                ):
                    if (
                        superseded.global_parent_id == parent_id
                        or superseded.config_root_digest != config_digest
                    ):
                        continue
                    if superseded.state is CaptureGlobalParentState.ENABLED:
                        store.transition_global_parent(
                            superseded.global_parent_id,
                            expected_state=CaptureGlobalParentState.ENABLED,
                            target_state=CaptureGlobalParentState.DRAINING,
                        )
                        store.transition_global_parent(
                            superseded.global_parent_id,
                            expected_state=CaptureGlobalParentState.DRAINING,
                            target_state=CaptureGlobalParentState.DISABLED,
                        )
                    elif superseded.state is CaptureGlobalParentState.DRAINING:
                        store.transition_global_parent(
                            superseded.global_parent_id,
                            expected_state=CaptureGlobalParentState.DRAINING,
                            target_state=CaptureGlobalParentState.DISABLED,
                        )
        return GlobalCaptureConnectReport(
            provider=alias,
            disposition=status.disposition,
            dry_run=dry_run,
            capture_enabled=not dry_run,
            managed_files=len(spec.project_local_paths),
            excluded_projects=len(excluded),
        )
    except CaptureCommandUnavailableError:
        raise
    except GlobalInstallationError:
        raise CaptureCommandUnavailableError() from None
    except (
        InsecureKeyFileError,
        InvalidInstallationKeyError,
        InstallationError,
        CaptureStoreStateError,
    ):
        raise CaptureCommandIntegrityError() from None
    except CaptureCommandIntegrityError:
        raise
    except Exception:
        raise CaptureCommandConfigurationError() from None


def run_global_disconnect(
    *,
    provider: str,
    environ: Mapping[str, str] | None = None,
    capture_executable: str | os.PathLike[str] | Path | None = None,
) -> GlobalCaptureDisconnectReport:
    """Disable a global parent and remove only its owned provider integration."""

    if type(provider) is not str:
        raise CaptureCommandInputError()
    try:
        alias = ProviderAlias(provider)
    except ValueError:
        raise CaptureCommandInputError() from None
    environment = _environment(environ)
    try:
        key = load_installation_key(default_installation_key_path(environ=environment))
        unresolved, receipt = _installed_spec(alias, key, environment=environment)
        if receipt is None:
            raise CaptureCommandUnavailableError()
        spec = materialize_provider_launcher(
            unresolved,
            key,
            capture_executable=capture_executable,
        )
        _global_provider, _config_digest, parent_id = _parent_coordinates(spec, key)
        locations = resolve_capture_store_locations(
            environ=environment,
            home=_home(environment),
        )
        with CaptureStore.open(
            locations.database_path,
            installation_key=key,
            mode=CaptureStoreMode.MAINTENANCE,
        ) as store:
            parent = store.get_global_parent(parent_id)
            if parent.state is CaptureGlobalParentState.ENABLED:
                store.transition_global_parent(
                    parent_id,
                    expected_state=CaptureGlobalParentState.ENABLED,
                    target_state=CaptureGlobalParentState.DRAINING,
                )
            elif parent.state not in (
                CaptureGlobalParentState.DRAINING,
                CaptureGlobalParentState.DISABLED,
            ):
                raise CaptureStoreStateError()
        disposition = uninstall_provider(spec, key).disposition
        with CaptureStore.open(
            locations.database_path,
            installation_key=key,
            mode=CaptureStoreMode.MAINTENANCE,
        ) as store:
            parent = store.get_global_parent(parent_id)
            if parent.state is CaptureGlobalParentState.DRAINING:
                store.transition_global_parent(
                    parent_id,
                    expected_state=CaptureGlobalParentState.DRAINING,
                    target_state=CaptureGlobalParentState.DISABLED,
                )
            elif parent.state is not CaptureGlobalParentState.DISABLED:
                raise CaptureStoreStateError()
        return GlobalCaptureDisconnectReport(
            provider=alias,
            disposition=disposition,
        )
    except CaptureCommandUnavailableError:
        raise
    except (FileNotFoundError, GlobalInstallationError):
        raise CaptureCommandUnavailableError() from None
    except (
        InsecureKeyFileError,
        InvalidInstallationKeyError,
        InstallationError,
        CaptureStoreStateError,
    ):
        raise CaptureCommandIntegrityError() from None
    except CaptureCommandIntegrityError:
        raise
    except Exception:
        raise CaptureCommandConfigurationError() from None


def run_global_status(
    *,
    provider: str | None = None,
    environ: Mapping[str, str] | None = None,
    capture_executable: str | os.PathLike[str] | Path | None = None,
) -> GlobalCaptureStatusReport:
    """Inspect one or all user-global provider installations."""

    try:
        aliases = tuple(ProviderAlias) if provider is None else (ProviderAlias(provider),)
    except (TypeError, ValueError):
        raise CaptureCommandInputError() from None
    environment = _environment(environ)

    def absent(alias: ProviderAlias) -> GlobalCaptureProviderStatus:
        return GlobalCaptureProviderStatus(
            provider=alias,
            status=GlobalCaptureStatusValue.NOT_INSTALLED,
            projects=0,
            exclusions=0,
        )

    try:
        key = load_installation_key(default_installation_key_path(environ=environment))
    except FileNotFoundError:
        return GlobalCaptureStatusReport(
            providers=tuple(absent(alias) for alias in aliases),
        )
    except (InsecureKeyFileError, InvalidInstallationKeyError):
        raise CaptureCommandIntegrityError() from None

    locations = resolve_capture_store_locations(
        environ=environment,
        home=_home(environment),
    )
    results: list[GlobalCaptureProviderStatus] = []
    for alias in aliases:
        try:
            unresolved, receipt = _installed_spec(alias, key, environment=environment)
        except CaptureCommandUnavailableError:
            results.append(absent(alias))
            continue
        if receipt is None:
            results.append(absent(alias))
            continue
        try:
            spec = materialize_provider_launcher(
                unresolved,
                key,
                capture_executable=capture_executable,
            )
            installation = inspect_provider_installation(spec, key)
            global_provider, _config_digest, parent_id = _parent_coordinates(spec, key)
            drift = set(installation.drift)
            projects = 0
            exclusions = 0
            parent_state: CaptureGlobalParentState | None = None
            if not locations.database_path.exists():
                drift.add("store")
            else:
                with CaptureStore.open(
                    locations.database_path,
                    installation_key=key,
                    mode=CaptureStoreMode.MAINTENANCE,
                ) as store:
                    parents = store.list_global_parents(provider_id=global_provider)
                    parent = next(
                        (
                            candidate
                            for candidate in parents
                            if candidate.global_parent_id == parent_id
                        ),
                        None,
                    )
                    if parent is None:
                        drift.add("parent")
                    else:
                        parent_state = parent.state
                        projects = len(store.list_global_children(parent_id))
                        exclusions = len(store.list_global_exclusions(parent_id))
            if (
                not drift
                and installation.state is InstallationState.ENABLED
                and installation.installed
                and parent_state is CaptureGlobalParentState.ENABLED
            ):
                status = GlobalCaptureStatusValue.ENABLED
            elif (
                not drift
                and installation.state is InstallationState.DISABLED
                and parent_state is CaptureGlobalParentState.DISABLED
            ):
                status = GlobalCaptureStatusValue.DISABLED
            else:
                if parent_state is not None:
                    drift.add(f"parent_{parent_state.value}")
                if installation.state is not InstallationState.ENABLED:
                    drift.add(f"installation_{installation.state.value}")
                status = GlobalCaptureStatusValue.DRIFTED
            results.append(
                GlobalCaptureProviderStatus(
                    provider=alias,
                    status=status,
                    projects=projects,
                    exclusions=exclusions,
                    drift=tuple(sorted(drift)),
                )
            )
        except CaptureCommandIntegrityError:
            raise
        except (InstallationError, CaptureStoreStateError):
            raise CaptureCommandIntegrityError() from None
        except Exception:
            raise CaptureCommandConfigurationError() from None
    return GlobalCaptureStatusReport(providers=tuple(results))


@dataclass(frozen=True, slots=True, repr=False)
class GlobalSetupHandler:
    """Setup handler backed by user-global provider installations."""

    exclusions: tuple[Path, ...] = ()
    environ: Mapping[str, str] | None = None
    capture_executable: str | os.PathLike[str] | Path | None = None

    def __repr__(self) -> str:
        return "GlobalSetupHandler(<redacted>)"

    def plan(self, request: SetupScopeRequest) -> tuple[SetupProviderPlan, ...]:
        if (
            request.scope is not SetupScope.GLOBAL
            or request.project is not None
            or request.project_selection is not None
        ):
            raise CaptureCommandConfigurationError()
        reports = tuple(
            run_global_connect(
                provider=provider.value,
                exclusions=self.exclusions,
                dry_run=True,
                environ=self.environ,
                capture_executable=self.capture_executable,
            )
            for provider in request.providers
        )
        return tuple(
            SetupProviderPlan(
                provider=report.provider,
                disposition=report.disposition,
                managed_files=report.managed_files,
            )
            for report in reports
        )

    def apply(self, request: SetupScopeRequest) -> tuple[SetupProviderResult, ...]:
        if (
            request.scope is not SetupScope.GLOBAL
            or request.project is not None
            or request.project_selection is not None
        ):
            raise CaptureCommandConfigurationError()
        reports = tuple(
            run_global_connect(
                provider=provider.value,
                exclusions=self.exclusions,
                environ=self.environ,
                capture_executable=self.capture_executable,
            )
            for provider in request.providers
        )
        return tuple(
            SetupProviderResult(
                provider=report.provider,
                disposition=report.disposition,
                capture_enabled=report.capture_enabled,
                managed_files=report.managed_files,
            )
            for report in reports
        )


__all__ = [
    "GlobalCaptureConnectReport",
    "GlobalCaptureDisconnectReport",
    "GlobalCaptureProviderStatus",
    "GlobalCaptureStatusReport",
    "GlobalCaptureStatusValue",
    "GlobalSetupHandler",
    "render_global_connect_human",
    "render_global_connect_json",
    "render_global_disconnect_human",
    "render_global_disconnect_json",
    "render_global_status_human",
    "render_global_status_json",
    "run_global_connect",
    "run_global_disconnect",
    "run_global_status",
]
