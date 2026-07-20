from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from saliencegate.capture import (
    CaptureConnectionState,
    CaptureProfile,
    CaptureSpool,
    CaptureStore,
    CaptureStoreMode,
    CaptureStoreStateError,
    capture_capability_digest,
    capture_profile,
    initialize_capture_store,
    resolve_capture_store_locations,
)
from saliencegate.commands.capture.common import CaptureCommandConfigurationError
from saliencegate.commands.capture.connect import (
    materialize_provider_launcher,
    project_provider_artifacts_present,
    render_connect_human,
    render_connect_json,
    run_connect,
)
from saliencegate.integrations.config_files import ConfigSyntax, OwnedConfigSpec
from saliencegate.integrations.installation import (
    derive_installation_identity,
    ensure_private_installation_directory,
    install_provider,
)
from saliencegate.integrations.registry import (
    ProviderAlias,
    ProviderInstallationKind,
    ProviderInstallationSpec,
)
from saliencegate.security import (
    default_installation_key_path,
    load_installation_key,
    load_or_create_installation_key,
)


def _spec(tmp_path: Path, *, generation: int = 1) -> ProviderInstallationSpec:
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    operational = tmp_path / "operational"
    ensure_private_installation_directory(operational)
    launcher = operational / "capture-hook"
    integration = project / ".synthetic"
    return ProviderInstallationSpec(
        provider_id="codex",
        profile=CaptureProfile.CODEX_HOOKS_V1,
        host_version=capture_profile(CaptureProfile.CODEX_HOOKS_V1).host_version,
        project_root=project,
        config_path=integration / "settings.json",
        bundle_path=integration
        / ("saliencegate.js" if generation == 1 else f"saliencegate-v{generation}.js"),
        bootstrap_path=integration / "saliencegate.bootstrap.json",
        receipt_path=operational / "codex.receipt.json",
        journal_path=operational / "codex.journal.json",
        lock_path=operational / "codex.lock",
        launcher_path=launcher,
        capability_digest=capture_capability_digest(capture_profile(CaptureProfile.CODEX_HOOKS_V1)),
        bundle_bytes=(
            b"export const saliencegateBootstrap = "
            b'new URL("./saliencegate.bootstrap.json", import.meta.url);\n'
            + (b"" if generation == 1 else f"// generation {generation}\n".encode("ascii"))
        ),
        launcher_bytes=b"#!/bin/sh\nexit 0\n",
        bootstrap_relative_reference="./saliencegate.bootstrap.json",
        config=OwnedConfigSpec(
            syntax=ConfigSyntax.JSON_OBJECT,
            marker="saliencegate-owned:connect-test-v1",
            owned_fragment=b'"saliencegate":{"marker":"saliencegate-owned:connect-test-v1"}',
        ),
        generation=generation,
    )


def _command_hook_spec(tmp_path: Path, *, generation: int = 1) -> ProviderInstallationSpec:
    payload = _spec(tmp_path, generation=generation).model_dump(mode="python", warnings="error")
    payload.update(
        installation_kind=ProviderInstallationKind.COMMAND_HOOK,
        bundle_path=None,
        bootstrap_path=None,
        bundle_bytes=None,
        bootstrap_relative_reference=None,
    )
    return ProviderInstallationSpec.model_validate(payload)


def _environment(tmp_path: Path) -> dict[str, str]:
    return {
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "configuration"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }


def test_connect_dry_run_writes_nothing_and_renders_only_safe_summary(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    environment = _environment(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    report = run_connect(
        provider="codex",
        project=spec.project_root,
        dry_run=True,
        environ=environment,
        spec_resolver=lambda alias, project: spec,
    )

    assert report.provider is ProviderAlias.CODEX
    assert report.disposition == "planned"
    assert report.dry_run is True
    assert report.capture_enabled is False
    assert report.project_local_files == 3
    assert before == {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert json.loads(render_connect_json(report)) == report.model_dump(mode="json")
    human = render_connect_human(report)
    assert str(spec.project_root) not in human
    assert spec.capability_digest not in human


def test_command_hook_connect_installs_only_config_and_private_launcher(tmp_path: Path) -> None:
    spec = _command_hook_spec(tmp_path)
    environment = _environment(tmp_path)

    report = run_connect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=lambda alias, project: spec,
    )

    assert report.capture_enabled is True
    assert report.project_local_files == 1
    assert report.git_tracked_files == 0
    assert spec.project_local_paths == (spec.config_path,)
    assert spec.config_path.is_file()
    assert spec.launcher_path.is_file()
    assert tuple(path for path in spec.project_root.rglob("*") if path.is_file()) == (
        spec.config_path,
    )
    assert project_provider_artifacts_present(
        ProviderAlias.CODEX,
        spec.project_root,
        resolver=lambda alias, project: spec,
    )


def test_pristine_missing_provider_directory_has_no_managed_artifacts(tmp_path: Path) -> None:
    spec = _spec(tmp_path)

    assert (
        project_provider_artifacts_present(
            ProviderAlias.CODEX,
            spec.project_root,
            resolver=lambda alias, project: spec,
        )
        is False
    )
    assert not spec.config_path.parent.exists()


def test_connect_installs_and_registers_one_enabled_connection(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    environment = _environment(tmp_path)

    report = run_connect(
        provider="codex",
        project=spec.project_root,
        dry_run=False,
        environ=environment,
        spec_resolver=lambda alias, project: spec,
    )

    assert report.capture_enabled is True
    key = load_installation_key(default_installation_key_path(environ=environment))
    identity = derive_installation_identity(spec, key)
    launcher = spec.launcher_path.read_bytes()
    assert identity.connection_id.encode("ascii") in launcher
    assert spec.profile.value.encode("ascii") in launcher
    assert launcher != spec.launcher_bytes
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    assert locations.spool_directory.is_dir()
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        connections = store.list_connections(profile_id=CaptureProfile.CODEX_HOOKS_V1)
        assert len(connections) == 1
        assert connections[0].state.value == "enabled"


def test_connect_upgrade_retires_only_superseded_provider_generation(tmp_path: Path) -> None:
    first = _spec(tmp_path)
    second = _spec(tmp_path, generation=2)
    environment = _environment(tmp_path)
    run_connect(
        provider="codex",
        project=first.project_root,
        environ=environment,
        spec_resolver=lambda alias, project: first,
    )
    key = load_installation_key(default_installation_key_path(environ=environment))
    first_identity = derive_installation_identity(first, key)
    second_identity = derive_installation_identity(second, key)
    unrelated_profile = capture_profile(CaptureProfile.CLAUDE_CODE_HOOKS_V1)
    unrelated_id = "sg-unrelated-provider"
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        store.register_connection(
            connection_id=unrelated_id,
            project_digest=first_identity.project_digest,
            profile_id=unrelated_profile.profile_id,
            capability_manifest_digest=capture_capability_digest(unrelated_profile),
            host_version=unrelated_profile.host_version,
        )
        store.transition_connection(
            unrelated_id,
            expected_state=CaptureConnectionState.PENDING,
            target_state=CaptureConnectionState.ENABLED,
        )

    upgraded = run_connect(
        provider="codex",
        project=second.project_root,
        environ=environment,
        spec_resolver=lambda alias, project: second,
    )
    repeated = run_connect(
        provider="codex",
        project=second.project_root,
        environ=environment,
        spec_resolver=lambda alias, project: second,
    )

    assert upgraded.disposition == "upgraded"
    assert repeated.disposition == "noop"
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        codex_states = {
            item.connection_id: item.state
            for item in store.list_connections(
                project_digest=first_identity.project_digest,
                profile_id=CaptureProfile.CODEX_HOOKS_V1,
            )
        }
        unrelated = tuple(
            item
            for item in store.list_connections(
                project_digest=first_identity.project_digest,
                profile_id=CaptureProfile.CLAUDE_CODE_HOOKS_V1,
            )
            if item.connection_id == unrelated_id
        )
    assert codex_states == {
        first_identity.connection_id: CaptureConnectionState.DISABLED,
        second_identity.connection_id: CaptureConnectionState.ENABLED,
    }
    assert len(unrelated) == 1
    assert unrelated[0].state is CaptureConnectionState.ENABLED


def test_connect_upgrade_resumes_a_draining_prior_generation(tmp_path: Path) -> None:
    first = _spec(tmp_path)
    second = _spec(tmp_path, generation=2)
    environment = _environment(tmp_path)
    run_connect(
        provider="codex",
        project=first.project_root,
        environ=environment,
        spec_resolver=lambda alias, project: first,
    )
    key = load_installation_key(default_installation_key_path(environ=environment))
    first_identity = derive_installation_identity(first, key)
    second_identity = derive_installation_identity(second, key)
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        store.transition_connection(
            first_identity.connection_id,
            expected_state=CaptureConnectionState.ENABLED,
            target_state=CaptureConnectionState.DRAINING,
        )

    run_connect(
        provider="codex",
        project=second.project_root,
        environ=environment,
        spec_resolver=lambda alias, project: second,
    )

    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        states = {
            item.connection_id: item.state
            for item in store.list_connections(
                project_digest=first_identity.project_digest,
                profile_id=CaptureProfile.CODEX_HOOKS_V1,
            )
        }
    assert states[first_identity.connection_id] is CaptureConnectionState.DISABLED
    assert states[second_identity.connection_id] is CaptureConnectionState.ENABLED


def test_connect_upgrade_recovers_prior_pending_install_boundary(tmp_path: Path) -> None:
    first = _spec(tmp_path)
    second = _spec(tmp_path, generation=2)
    environment = _environment(tmp_path)
    key = load_or_create_installation_key(default_installation_key_path(environ=environment))
    executable = Path(sys.executable)
    installed_first = materialize_provider_launcher(
        first,
        key,
        capture_executable=executable,
    )
    first_identity = derive_installation_identity(first, key)
    second_identity = derive_installation_identity(second, key)
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    locations.state_directory.mkdir(mode=0o700, parents=True)
    initialize_capture_store(locations.database_path)
    CaptureSpool.open(locations, key)
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        registration = store.register_connection(
            connection_id=first_identity.connection_id,
            project_digest=first_identity.project_digest,
            profile_id=first.profile,
            capability_manifest_digest=first.capability_digest,
            host_version=first.host_version,
        )
    assert registration.state is CaptureConnectionState.PENDING
    install_provider(installed_first, key)

    run_connect(
        provider="codex",
        project=second.project_root,
        environ=environment,
        spec_resolver=lambda alias, project: second,
        capture_executable=executable,
    )

    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        states = {
            item.connection_id: item.state
            for item in store.list_connections(
                project_digest=first_identity.project_digest,
                profile_id=CaptureProfile.CODEX_HOOKS_V1,
            )
        }
    assert states == {
        first_identity.connection_id: CaptureConnectionState.DISABLED,
        second_identity.connection_id: CaptureConnectionState.ENABLED,
    }


def test_connect_upgrade_retry_retires_prior_after_new_identity_was_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _spec(tmp_path)
    second = _spec(tmp_path, generation=2)
    environment = _environment(tmp_path)
    executable = Path(sys.executable)
    run_connect(
        provider="codex",
        project=first.project_root,
        environ=environment,
        spec_resolver=lambda alias, project: first,
        capture_executable=executable,
    )
    key = load_installation_key(default_installation_key_path(environ=environment))
    first_identity = derive_installation_identity(first, key)
    second_identity = derive_installation_identity(second, key)
    original_transition = CaptureStore.transition_connection

    def interrupt_old_retirement(
        store: CaptureStore,
        connection_id: str,
        *,
        expected_state: CaptureConnectionState,
        target_state: CaptureConnectionState,
    ) -> object:
        if (
            connection_id == first_identity.connection_id
            and expected_state is CaptureConnectionState.ENABLED
            and target_state is CaptureConnectionState.DRAINING
        ):
            raise CaptureStoreStateError()
        return original_transition(
            store,
            connection_id,
            expected_state=expected_state,
            target_state=target_state,
        )

    monkeypatch.setattr(CaptureStore, "transition_connection", interrupt_old_retirement)
    with pytest.raises(CaptureCommandConfigurationError):
        run_connect(
            provider="codex",
            project=second.project_root,
            environ=environment,
            spec_resolver=lambda alias, project: second,
            capture_executable=executable,
        )
    monkeypatch.undo()

    repeated = run_connect(
        provider="codex",
        project=second.project_root,
        environ=environment,
        spec_resolver=lambda alias, project: second,
        capture_executable=executable,
    )

    assert repeated.disposition == "noop"
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        states = {
            item.connection_id: item.state
            for item in store.list_connections(
                project_digest=first_identity.project_digest,
                profile_id=CaptureProfile.CODEX_HOOKS_V1,
            )
        }
    assert states == {
        first_identity.connection_id: CaptureConnectionState.DISABLED,
        second_identity.connection_id: CaptureConnectionState.ENABLED,
    }


def test_connect_upgrade_dry_run_leaves_prior_generation_enabled(tmp_path: Path) -> None:
    first = _spec(tmp_path)
    second = _spec(tmp_path, generation=2)
    environment = _environment(tmp_path)
    run_connect(
        provider="codex",
        project=first.project_root,
        environ=environment,
        spec_resolver=lambda alias, project: first,
    )
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    report = run_connect(
        provider="codex",
        project=second.project_root,
        dry_run=True,
        environ=environment,
        spec_resolver=lambda alias, project: second,
    )

    assert report.disposition == "planned"
    assert report.capture_enabled is False
    assert before == {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    key = load_installation_key(default_installation_key_path(environ=environment))
    identity = derive_installation_identity(first, key)
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        connection = tuple(
            item
            for item in store.list_connections()
            if item.connection_id == identity.connection_id
        )
    assert len(connection) == 1
    assert connection[0].state is CaptureConnectionState.ENABLED


def test_connect_upgrade_fails_closed_before_install_when_prior_is_deleting(
    tmp_path: Path,
) -> None:
    first = _spec(tmp_path)
    second = _spec(tmp_path, generation=2)
    environment = _environment(tmp_path)
    run_connect(
        provider="codex",
        project=first.project_root,
        environ=environment,
        spec_resolver=lambda alias, project: first,
    )
    key = load_installation_key(default_installation_key_path(environ=environment))
    first_identity = derive_installation_identity(first, key)
    second_identity = derive_installation_identity(second, key)
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        store.transition_connection(
            first_identity.connection_id,
            expected_state=CaptureConnectionState.ENABLED,
            target_state=CaptureConnectionState.DRAINING,
        )
        store.transition_connection(
            first_identity.connection_id,
            expected_state=CaptureConnectionState.DRAINING,
            target_state=CaptureConnectionState.DISABLED,
        )
        store.transition_connection(
            first_identity.connection_id,
            expected_state=CaptureConnectionState.DISABLED,
            target_state=CaptureConnectionState.DELETING,
        )
    managed_before = {
        path: path.read_bytes()
        for path in (
            first.config_path,
            first.bundle_path,
            first.bootstrap_path,
            first.launcher_path,
            first.receipt_path,
        )
    }

    with pytest.raises(CaptureCommandConfigurationError):
        run_connect(
            provider="codex",
            project=second.project_root,
            environ=environment,
            spec_resolver=lambda alias, project: second,
        )

    assert managed_before == {path: path.read_bytes() for path in managed_before}
    assert not second.bundle_path.exists()
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        connection_ids = {item.connection_id for item in store.list_connections()}
    assert connection_ids == {first_identity.connection_id}
    assert second_identity.connection_id not in connection_ids
