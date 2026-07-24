"""Authenticated routing from one user-global provider hook to project children."""

from __future__ import annotations

import hmac
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from saliencegate.capture.capabilities import CaptureProfile
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.capture.locations import CaptureStoreLocations, resolve_capture_store_locations
from saliencegate.capture.scopes import (
    CaptureGlobalParentState,
    CaptureGlobalProvider,
    derive_global_config_root_digest,
    derive_global_parent_id,
)
from saliencegate.capture.spool import CaptureSpool
from saliencegate.capture.store import (
    CaptureConnectionState,
    CaptureStore,
    CaptureStoreMode,
    _CaptureHookConnection,
)
from saliencegate.domain import canonical_json
from saliencegate.integrations.bootstrap import IntegrationBootstrap
from saliencegate.integrations.environment import environment_without_provider_credentials
from saliencegate.integrations.global_installation import (
    GlobalInstallationError,
    global_provider_installation_spec,
)
from saliencegate.integrations.installation import (
    InstallationReceipt,
    InstallationState,
    InstallationStatus,
    _load_receipt_optional,
    derive_installation_identity,
    inspect_provider_installation,
)
from saliencegate.integrations.launcher_materialization import materialize_provider_launcher
from saliencegate.integrations.registry import (
    BUILTIN_PROVIDER_REGISTRY,
    ProviderAlias,
    ProviderInstallationSpec,
    ProviderRegistration,
)
from saliencegate.security import InstallationKey, load_installation_key

if TYPE_CHECKING:
    from saliencegate.capture.health import CaptureHealthCode
    from saliencegate.integrations.hook import CaptureHookDependencies


class GlobalCaptureRuntimeError(ValueError):
    """A content-free global routing failure."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("global capture routing failed")


class _BridgeBatch(Protocol):
    @property
    def bootstrap(self) -> IntegrationBootstrap: ...

    @property
    def document(self) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True, repr=False)
class _GlobalHookRuntime:
    key: InstallationKey
    locations: CaptureStoreLocations
    provider: ProviderAlias
    profile: CaptureProfile
    requested_connection_id: str
    registration: ProviderRegistration
    installation: InstallationStatus
    connection: _CaptureHookConnection
    project_root: Path
    source: bytes
    rebound_source: bytes | None
    bootstrap: IntegrationBootstrap | None

    def __repr__(self) -> str:
        return "_GlobalHookRuntime(<redacted>)"


class _ReboundBridgeAdapter:
    """Expose child coordinates while preserving an authenticated parent bootstrap."""

    __slots__ = ("_inner", "_rebound", "_source")

    def __init__(self, inner: object, *, source: bytes, rebound: bytes) -> None:
        self._inner = inner
        self._source = source
        self._rebound = rebound

    def __repr__(self) -> str:
        return "_ReboundBridgeAdapter(<redacted>)"

    __str__ = __repr__

    def capabilities(self) -> object:
        method = getattr(self._inner, "capabilities", None)
        if not callable(method):
            raise GlobalCaptureRuntimeError()
        return method()

    def _checked_source(self, source: bytes) -> bytes:
        if type(source) is not bytes or not hmac.compare_digest(source, self._source):
            raise GlobalCaptureRuntimeError()
        return self._rebound

    def adapt_bytes(
        self,
        source: bytes,
        *,
        context: CaptureDigestContext,
    ) -> tuple[object, ...]:
        method = getattr(self._inner, "adapt_bytes", None)
        if not callable(method):
            raise GlobalCaptureRuntimeError()
        result = method(self._checked_source(source), context=context)
        if type(result) is not tuple:
            raise GlobalCaptureRuntimeError()
        return result

    def transport_chunk(
        self,
        source: bytes,
        *,
        context: CaptureDigestContext,
    ) -> object:
        method = getattr(self._inner, "transport_chunk", None)
        if not callable(method):
            raise GlobalCaptureRuntimeError()
        return method(self._checked_source(source), context=context)


def _home(environment: Mapping[str, str]) -> Path:
    configured = environment.get("HOME")
    return Path.home() if configured is None else Path(configured)


def resolve_global_project_root(workspace: str) -> Path:
    try:
        candidate = Path(workspace)
        if not candidate.is_absolute() or ".." in candidate.parts or "\x00" in workspace:
            raise GlobalCaptureRuntimeError()
        resolved = candidate.resolve(strict=True)
        metadata = resolved.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise GlobalCaptureRuntimeError()
        for depth, current in enumerate((resolved, *resolved.parents)):
            if depth > 128:
                break
            marker = current / ".git"
            try:
                marker_metadata = marker.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(marker_metadata.st_mode):
                raise GlobalCaptureRuntimeError()
            if stat.S_ISDIR(marker_metadata.st_mode) or stat.S_ISREG(marker_metadata.st_mode):
                return current
            raise GlobalCaptureRuntimeError()
        return resolved
    except GlobalCaptureRuntimeError:
        raise
    except Exception:
        raise GlobalCaptureRuntimeError() from None


def _native_coordinates(
    provider: ProviderAlias,
    source: bytes,
) -> tuple[Path, str, _BridgeBatch | None]:
    try:
        if provider is ProviderAlias.OPENCODE:
            from saliencegate.integrations import opencode

            opencode_batch = opencode._parse_batch(source)
            if opencode_batch.workspace_path is None:
                raise GlobalCaptureRuntimeError()
            return (
                resolve_global_project_root(opencode_batch.workspace_path),
                opencode_batch.session_id,
                opencode_batch,
            )
        if provider is ProviderAlias.PI:
            from saliencegate.integrations import pi

            pi_batch = pi._parse_batch(source)
            if pi_batch.workspace_path is None:
                raise GlobalCaptureRuntimeError()
            return (
                resolve_global_project_root(pi_batch.workspace_path),
                pi_batch.session_id,
                pi_batch,
            )

        from saliencegate.capture.schema import CAPTURE_NATIVE_JSON_LIMITS, read_bounded_json

        document = read_bounded_json(source, limits=CAPTURE_NATIVE_JSON_LIMITS)
        workspace = document.get("cwd")
        session_id = document.get("session_id")
        if type(workspace) is not str or type(session_id) is not str:
            raise GlobalCaptureRuntimeError()
        return resolve_global_project_root(workspace), session_id, None
    except GlobalCaptureRuntimeError:
        raise
    except Exception:
        raise GlobalCaptureRuntimeError() from None


def _rebound_bridge(
    batch: _BridgeBatch,
    *,
    connection_id: str,
) -> tuple[IntegrationBootstrap, bytes]:
    try:
        bootstrap = batch.bootstrap
        document = batch.document
        if type(bootstrap) is not IntegrationBootstrap or not isinstance(document, Mapping):
            raise GlobalCaptureRuntimeError()
        bootstrap_payload = bootstrap.model_dump(mode="python", warnings="error")
        bootstrap_payload["connection_id"] = connection_id
        child_bootstrap = IntegrationBootstrap.model_validate(bootstrap_payload)
        rebound_document = dict(document)
        rebound_document["bootstrap"] = child_bootstrap.model_dump(
            mode="json",
            warnings="error",
        )
        return child_bootstrap, canonical_json(rebound_document)
    except GlobalCaptureRuntimeError:
        raise
    except Exception:
        raise GlobalCaptureRuntimeError() from None


def _global_spec_from_receipt(
    provider: ProviderAlias,
    key: InstallationKey,
    *,
    environment: Mapping[str, str],
) -> tuple[ProviderInstallationSpec, InstallationReceipt | None]:
    try:
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
        raise
    except Exception:
        raise GlobalCaptureRuntimeError() from None


def try_build_global_capture_hook_dependencies(
    provider: ProviderAlias,
    source: bytes,
    *,
    connection_id: str,
    environ: Mapping[str, str] | None = None,
    capture_executable: str | os.PathLike[str] | Path | None = None,
) -> CaptureHookDependencies | None:
    """Return dependencies only when the requested ID is a global installation."""

    try:
        if (
            type(provider) is not ProviderAlias
            or type(source) is not bytes
            or type(connection_id) is not str
            or (environ is not None and not isinstance(environ, Mapping))
        ):
            raise GlobalCaptureRuntimeError()
        environment = environment_without_provider_credentials(environ)
        try:
            key = load_installation_key(environ=environment)
            unresolved, receipt = _global_spec_from_receipt(
                provider,
                key,
                environment=environment,
            )
        except (FileNotFoundError, GlobalInstallationError):
            return None
        if receipt is None or receipt.connection_id != connection_id:
            return None
        spec = materialize_provider_launcher(
            unresolved,
            key,
            capture_executable=capture_executable,
        )
        identity = derive_installation_identity(spec, key)
        installation = inspect_provider_installation(spec, key)
        if (
            identity.connection_id != connection_id
            or installation.connection_id != connection_id
            or installation.state is not InstallationState.ENABLED
            or not installation.installed
            or installation.drift
        ):
            raise GlobalCaptureRuntimeError()

        project_root, session_native, batch = _native_coordinates(provider, source)
        from saliencegate.commands.capture.connect import project_provider_artifacts_present

        if project_provider_artifacts_present(
            provider,
            project_root,
            environ=environment,
        ):
            raise GlobalCaptureRuntimeError()

        provider_id = CaptureGlobalProvider(provider.value)
        config_root_digest = derive_global_config_root_digest(
            os.fsencode(spec.project_root),
            key,
        )
        global_parent_id = derive_global_parent_id(
            provider_id=provider_id,
            config_root_digest=config_root_digest,
            generation=spec.generation,
            installation_key=key,
        )
        locations = resolve_capture_store_locations(
            environ=environment,
            home=_home(environment),
        )
        with CaptureStore.open(
            locations.database_path,
            installation_key=key,
            mode=CaptureStoreMode.HOOK,
        ) as store:
            parent = store.get_global_parent(global_parent_id)
            if (
                parent.state is not CaptureGlobalParentState.ENABLED
                or parent.provider_id is not provider_id
                or parent.config_root_digest != config_root_digest
                or parent.profile_id is not spec.profile
                or parent.capability_manifest_digest != spec.capability_digest
                or parent.host_version != spec.host_version
                or parent.generation != spec.generation
            ):
                raise GlobalCaptureRuntimeError()
            if store.global_child_is_excluded(
                global_parent_id,
                os.fsencode(project_root),
            ):
                raise GlobalCaptureRuntimeError()
            binding = store.enroll_global_child(
                global_parent_id,
                os.fsencode(project_root),
            )
            connection = store._get_hook_connection(binding.connection_id)
        if (
            connection.state is not CaptureConnectionState.ENABLED
            or connection.project_digest != binding.project_digest
            or connection.profile_id is not spec.profile
            or connection.capability_manifest_digest != spec.capability_digest
            or connection.host_version != spec.host_version
        ):
            raise GlobalCaptureRuntimeError()

        registration = BUILTIN_PROVIDER_REGISTRY.resolve(
            provider,
            require_available=True,
        )
        if registration.profile is not spec.profile:
            raise GlobalCaptureRuntimeError()
        child_bootstrap: IntegrationBootstrap | None = None
        rebound_source: bytes | None = None
        if batch is not None:
            child_bootstrap, rebound_source = _rebound_bridge(
                batch,
                connection_id=connection.connection_id,
            )
        runtime = _GlobalHookRuntime(
            key=key,
            locations=locations,
            provider=provider,
            profile=spec.profile,
            requested_connection_id=connection_id,
            registration=registration,
            installation=installation,
            connection=connection,
            project_root=project_root,
            source=source,
            rebound_source=rebound_source,
            bootstrap=child_bootstrap,
        )

        from saliencegate.integrations.hook import CaptureHookDependencies

        def checked(value: object) -> _GlobalHookRuntime:
            if value is not runtime:
                raise GlobalCaptureRuntimeError()
            return runtime

        def validate_registry(profile: CaptureProfile) -> object:
            if profile is not runtime.profile:
                raise GlobalCaptureRuntimeError()
            return runtime.registration

        def validate_receipt(
            profile: CaptureProfile,
            requested: str,
            candidate_registry: object,
        ) -> object:
            if (
                profile is not runtime.profile
                or requested != runtime.requested_connection_id
                or candidate_registry is not runtime.registration
            ):
                raise GlobalCaptureRuntimeError()
            return runtime.installation

        def validate_connection(
            profile: CaptureProfile,
            requested: str,
            candidate_registry: object,
            candidate_receipt: object,
        ) -> object:
            if (
                profile is not runtime.profile
                or requested != runtime.requested_connection_id
                or candidate_registry is not runtime.registration
                or candidate_receipt is not runtime.installation
            ):
                raise GlobalCaptureRuntimeError()
            return runtime

        def load_context(candidate: object) -> CaptureDigestContext:
            return CaptureDigestContext(checked(candidate).key)

        def resolve_connection_id(candidate: object, requested: str) -> str:
            selected = checked(candidate)
            if requested != selected.requested_connection_id:
                raise GlobalCaptureRuntimeError()
            return selected.connection.connection_id

        def resolve_adapter(candidate: object) -> object:
            selected = checked(candidate)
            if selected.provider is ProviderAlias.CODEX:
                from saliencegate.integrations.codex import CodexCaptureAdapter

                return CodexCaptureAdapter(
                    connection_id=selected.connection.connection_id,
                    host_version=selected.connection.host_version,
                )
            if selected.provider is ProviderAlias.CLAUDE_CODE:
                from saliencegate.integrations.claude_code import ClaudeCodeCaptureAdapter

                return ClaudeCodeCaptureAdapter(
                    connection_id=selected.connection.connection_id,
                    host_version=selected.connection.host_version,
                )
            if selected.bootstrap is None or selected.rebound_source is None:
                raise GlobalCaptureRuntimeError()
            if selected.provider is ProviderAlias.OPENCODE:
                from saliencegate.integrations.opencode import OpenCodeCaptureAdapter

                inner: object = OpenCodeCaptureAdapter(
                    connection_id=selected.connection.connection_id,
                    bootstrap=selected.bootstrap,
                    project_root=selected.project_root,
                    host_version=selected.connection.host_version,
                )
            elif selected.provider is ProviderAlias.PI:
                from saliencegate.integrations.pi import PiCaptureAdapter

                inner = PiCaptureAdapter(
                    connection_id=selected.connection.connection_id,
                    bootstrap=selected.bootstrap,
                    project_root=selected.project_root,
                    host_version=selected.connection.host_version,
                )
            else:
                raise GlobalCaptureRuntimeError()
            return _ReboundBridgeAdapter(
                inner,
                source=selected.source,
                rebound=selected.rebound_source,
            )

        def open_store(candidate: object) -> CaptureStore:
            selected = checked(candidate)
            return CaptureStore.open(
                selected.locations.database_path,
                installation_key=selected.key,
                mode=CaptureStoreMode.HOOK,
            )

        def open_spool(candidate: object) -> CaptureSpool:
            selected = checked(candidate)
            return CaptureSpool.open(selected.locations, selected.key)

        session_id = CaptureDigestContext(key).session_id(session_native.encode("utf-8"))

        def mark_health(candidate: object, code: CaptureHealthCode) -> None:
            selected = checked(candidate)
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

        return CaptureHookDependencies(
            validate_registry=validate_registry,
            validate_receipt=validate_receipt,
            validate_connection=validate_connection,
            load_context=load_context,
            resolve_adapter=resolve_adapter,
            open_store=open_store,
            open_spool=open_spool,
            mark_health=mark_health,
            resolve_connection_id=resolve_connection_id,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except GlobalCaptureRuntimeError:
        raise
    except Exception:
        raise GlobalCaptureRuntimeError() from None


__all__ = [
    "GlobalCaptureRuntimeError",
    "resolve_global_project_root",
    "try_build_global_capture_hook_dependencies",
]
