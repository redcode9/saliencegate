from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from saliencegate.capture import (
    CaptureGlobalParentState,
    CaptureStore,
    CaptureStoreMode,
    resolve_capture_store_locations,
)
from saliencegate.commands import global_capture as global_capture_module
from saliencegate.commands.capture import (
    CaptureCommandConfigurationError,
    CaptureCommandInputError,
    CaptureCommandIntegrityError,
    CaptureCommandUnavailableError,
)
from saliencegate.commands.global_capture import (
    GlobalCaptureConnectReport,
    GlobalCaptureDisconnectReport,
    GlobalCaptureProviderStatus,
    GlobalCaptureStatusReport,
    GlobalCaptureStatusValue,
    GlobalSetupHandler,
    render_global_connect_human,
    render_global_connect_json,
    render_global_disconnect_human,
    render_global_disconnect_json,
    render_global_status_human,
    render_global_status_json,
    run_global_connect,
    run_global_disconnect,
    run_global_status,
)
from saliencegate.commands.setup import SetupProjectSelection
from saliencegate.domain import canonical_json
from saliencegate.integrations.bootstrap import inspect_integration_bootstrap
from saliencegate.integrations.global_installation import global_provider_installation_spec
from saliencegate.integrations.hook import run_capture_hook
from saliencegate.integrations.installation import (
    InstallationDisposition,
    derive_installation_identity,
    inspect_provider_installation,
)
from saliencegate.integrations.launcher_materialization import materialize_provider_launcher
from saliencegate.integrations.registry import ProviderAlias
from saliencegate.security import default_installation_key_path, load_installation_key


def _environment(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    (home / ".claude").mkdir()
    (home / ".config" / "opencode").mkdir(parents=True)
    (home / ".pi" / "agent").mkdir(parents=True)
    state = tmp_path / "state"
    state.mkdir()
    return {
        "HOME": os.fspath(home),
        "XDG_STATE_HOME": os.fspath(state),
        "PATH": os.environ.get("PATH", ""),
    }


def _project(tmp_path: Path, name: str) -> Path:
    project = tmp_path / name
    (project / ".git").mkdir(parents=True)
    return project


def test_global_connect_is_idempotent_and_disconnect_retains_parent_state(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    excluded = _project(tmp_path, "excluded")

    planned = run_global_connect(
        provider="opencode",
        exclusions=(excluded,),
        dry_run=True,
        environ=environment,
    )
    assert planned.disposition.value == "planned"
    assert planned.capture_enabled is False
    assert planned.excluded_projects == 1
    assert not Path(default_installation_key_path(environ=environment)).exists()

    installed = run_global_connect(
        provider="opencode",
        exclusions=(excluded,),
        environ=environment,
    )
    repeated = run_global_connect(
        provider="opencode",
        exclusions=(excluded,),
        environ=environment,
    )
    assert installed.disposition.value == "installed"
    assert repeated.disposition.value == "noop"
    status = run_global_status(provider="opencode", environ=environment)
    assert status.providers[0].status.value == "enabled"
    assert status.providers[0].exclusions == 1

    key = load_installation_key(default_installation_key_path(environ=environment))
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        parents = store.list_global_parents()
        assert len(parents) == 1
        assert parents[0].state is CaptureGlobalParentState.ENABLED
        assert len(store.list_global_exclusions(parents[0].global_parent_id)) == 1

    disconnected = run_global_disconnect(
        provider="opencode",
        environ=environment,
    )
    assert disconnected.disposition.value == "uninstalled"
    assert (
        run_global_status(provider="opencode", environ=environment).providers[0].status.value
        == "disabled"
    )
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        assert store.list_global_parents()[0].state is CaptureGlobalParentState.DISABLED


def test_global_opencode_hook_enrolls_projects_and_honors_exclusions(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    included = _project(tmp_path, "included")
    excluded = _project(tmp_path, "excluded")
    run_global_connect(
        provider="opencode",
        exclusions=(excluded,),
        environ=environment,
    )
    key = load_installation_key(default_installation_key_path(environ=environment))
    spec = materialize_provider_launcher(
        global_provider_installation_spec("opencode", environ=environment),
        key,
    )
    installation = inspect_provider_installation(spec, key)
    bootstrap = inspect_integration_bootstrap(spec.bootstrap_path)

    def source(project: Path, session: str) -> bytes:
        return canonical_json(
            {
                "schema_version": "capture-batch/v1",
                "bootstrap": bootstrap.model_dump(mode="json", warnings="error"),
                "batch_id": "a" * 64,
                "session_id": session,
                "workspace_path": os.fspath(project),
                "chunk_index": 0,
                "chunk_count": 1,
                "events": [],
            }
        )

    arguments = (
        "--profile",
        spec.profile.value,
        "--connection",
        derive_installation_identity(spec, key).connection_id,
    )
    assert (
        run_capture_hook(
            arguments,
            io.BytesIO(source(included, "included-session")),
            environ=environment,
        )
        == 0
    )
    assert (
        run_capture_hook(
            arguments,
            io.BytesIO(source(excluded, "excluded-session")),
            environ=environment,
        )
        == 0
    )

    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        parents = store.list_global_parents()
        children = store.list_global_children(parents[0].global_parent_id)
        assert len(children) == 1
        assert store.session_inventory(project_digest=children[0].project_digest).session_count == 1
    assert installation.installed is True


def test_global_capture_reports_enforce_state_and_render_all_output_modes() -> None:
    planned = GlobalCaptureConnectReport(
        provider=ProviderAlias.OPENCODE,
        disposition=InstallationDisposition.PLANNED,
        dry_run=True,
        capture_enabled=False,
        managed_files=2,
        excluded_projects=1,
    )
    installed = GlobalCaptureConnectReport(
        provider=ProviderAlias.OPENCODE,
        disposition=InstallationDisposition.INSTALLED,
        dry_run=False,
        capture_enabled=True,
        managed_files=2,
        excluded_projects=0,
    )
    disconnected = GlobalCaptureDisconnectReport(
        provider=ProviderAlias.OPENCODE,
        disposition=InstallationDisposition.UNINSTALLED,
    )
    status = GlobalCaptureStatusReport(
        providers=(
            GlobalCaptureProviderStatus(
                provider=ProviderAlias.OPENCODE,
                status=GlobalCaptureStatusValue.DRIFTED,
                projects=2,
                exclusions=1,
                drift=("bundle", "parent"),
            ),
        )
    )

    assert '"dry_run":true' in render_global_connect_json(planned)
    assert "would install" in render_global_connect_human(planned)
    assert "installed; enabled" in render_global_connect_human(installed)
    assert '"disposition":"uninstalled"' in render_global_disconnect_json(disconnected)
    assert "local session data retained" in render_global_disconnect_human(disconnected)
    assert '"status":"drifted"' in render_global_status_json(status)
    assert "drift bundle,parent" in render_global_status_human(status)

    with pytest.raises(ValidationError):
        GlobalCaptureConnectReport(
            provider=ProviderAlias.OPENCODE,
            disposition=InstallationDisposition.INSTALLED,
            dry_run=True,
            capture_enabled=False,
            managed_files=2,
            excluded_projects=0,
        )
    with pytest.raises(ValidationError):
        GlobalCaptureConnectReport(
            provider=ProviderAlias.OPENCODE,
            disposition=InstallationDisposition.PLANNED,
            dry_run=True,
            capture_enabled=True,
            managed_files=2,
            excluded_projects=0,
        )


def test_global_capture_rejects_invalid_inputs_and_oversized_exclusions(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    project = _project(tmp_path, "project")
    spec = global_provider_installation_spec("opencode", environ=environment)

    with pytest.raises(CaptureCommandConfigurationError):
        global_capture_module._environment([])
    with pytest.raises(CaptureCommandInputError):
        global_capture_module._resolved_exclusions(os.fspath(project))
    with pytest.raises(CaptureCommandInputError):
        global_capture_module._resolved_exclusions((tmp_path / "missing",))
    with pytest.raises(CaptureCommandInputError):
        global_capture_module._resolved_exclusions((project,) * 1_001)
    with pytest.raises(CaptureCommandIntegrityError):
        global_capture_module._parent_coordinates(spec, object())

    with pytest.raises(CaptureCommandInputError):
        run_global_connect(provider=1, environ=environment)  # type: ignore[arg-type]
    with pytest.raises(CaptureCommandInputError):
        run_global_connect(provider="future", environ=environment)
    with pytest.raises(CaptureCommandInputError):
        run_global_connect(provider="opencode", dry_run=1, environ=environment)  # type: ignore[arg-type]
    with pytest.raises(CaptureCommandInputError):
        run_global_disconnect(provider=1, environ=environment)  # type: ignore[arg-type]
    with pytest.raises(CaptureCommandInputError):
        run_global_disconnect(provider="future", environ=environment)
    with pytest.raises(CaptureCommandInputError):
        run_global_status(provider=1, environ=environment)  # type: ignore[arg-type]


def test_global_status_without_state_reports_each_provider_absent(tmp_path: Path) -> None:
    environment = _environment(tmp_path)

    report = run_global_status(environ=environment)

    assert tuple(item.provider for item in report.providers) == tuple(ProviderAlias)
    assert all(item.status is GlobalCaptureStatusValue.NOT_INSTALLED for item in report.providers)
    with pytest.raises(CaptureCommandUnavailableError):
        run_global_disconnect(provider="opencode", environ=environment)


def test_global_status_reports_provider_and_store_drift(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    run_global_connect(provider="opencode", environ=environment)
    key = load_installation_key(default_installation_key_path(environ=environment))
    spec = materialize_provider_launcher(
        global_provider_installation_spec("opencode", environ=environment),
        key,
    )
    assert spec.bundle_path is not None
    spec.bundle_path.write_bytes(b"tampered")

    drifted = run_global_status(provider="opencode", environ=environment)
    assert drifted.providers[0].status is GlobalCaptureStatusValue.DRIFTED
    assert drifted.providers[0].drift

    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    locations.database_path.unlink()
    missing_store = run_global_status(provider="opencode", environ=environment)
    assert missing_store.providers[0].status is GlobalCaptureStatusValue.DRIFTED
    assert "store" in missing_store.providers[0].drift


def test_global_setup_handler_rejects_project_scope_requests(tmp_path: Path) -> None:
    request = global_capture_module.SetupScopeRequest(
        scope=global_capture_module.SetupScope.PROJECT,
        providers=(ProviderAlias.OPENCODE,),
        project=tmp_path,
        project_selection=SetupProjectSelection.MANUAL,
    )
    handler = GlobalSetupHandler(environ=_environment(tmp_path))

    with pytest.raises(CaptureCommandConfigurationError):
        handler.plan(request)
    with pytest.raises(CaptureCommandConfigurationError):
        handler.apply(request)
