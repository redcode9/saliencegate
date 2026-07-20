from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath

import pytest
from tests.cli.test_capture_connect import (
    _command_hook_spec,
    _configless_bridge_spec,
    _environment,
    _spec,
)

import saliencegate.commands.capture.disconnect as disconnect_module
from saliencegate.capture import (
    CaptureConnectionState,
    CaptureProfile,
    CaptureStore,
    CaptureStoreMode,
    capture_capability_digest,
    capture_profile,
    resolve_capture_store_locations,
)
from saliencegate.commands.capture.common import (
    CaptureCommandConfigurationError,
    CaptureCommandIntegrityError,
)
from saliencegate.commands.capture.connect import materialize_provider_launcher, run_connect
from saliencegate.commands.capture.disconnect import (
    render_disconnect_human,
    render_disconnect_json,
    run_disconnect,
)
from saliencegate.integrations.installation import (
    derive_installation_identity,
    install_provider,
)
from saliencegate.integrations.registry import ProviderAlias, ProviderInstallationSpec
from saliencegate.security import default_installation_key_path, load_installation_key
from saliencegate.security.windows import WindowsPathKind


def test_windows_receipt_asset_reader_uses_native_private_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Authorization:
        revalidated = False

        def revalidate(self) -> None:
            self.revalidated = True

    class _StableRead:
        def __init__(self, data: bytes, authorization: _Authorization) -> None:
            self.data = data
            self.authorization = authorization

    parent_authorization = _Authorization()
    asset_authorization = _Authorization()
    observed: dict[str, object] = {}

    class _Operations:
        def read_private_file(
            self,
            path: PureWindowsPath,
            *,
            maximum_bytes: int,
        ) -> _StableRead:
            observed["target"] = path
            observed["maximum"] = maximum_bytes
            return _StableRead(b"receipt-bound", asset_authorization)

    operations = _Operations()

    def authorize_parent(
        path: PureWindowsPath,
        *,
        kind: WindowsPathKind,
        operations: object,
    ) -> _Authorization:
        observed["parent"] = path
        observed["kind"] = kind
        observed["operations"] = operations
        return parent_authorization

    monkeypatch.setattr(
        disconnect_module,
        "NativeWindowsSecurityOperations",
        lambda: operations,
    )
    monkeypatch.setattr(
        disconnect_module,
        "authorize_windows_managed_path",
        authorize_parent,
    )

    result = disconnect_module._read_installed_private_asset_windows(
        Path(r"C:\project\.provider\bundle.js"),
        maximum_bytes=4096,
    )

    assert result == b"receipt-bound"
    assert observed == {
        "target": PureWindowsPath(r"C:\project\.provider\bundle.js"),
        "maximum": 4096,
        "parent": PureWindowsPath(r"C:\project\.provider"),
        "kind": WindowsPathKind.DIRECTORY,
        "operations": operations,
    }
    assert parent_authorization.revalidated is True
    assert asset_authorization.revalidated is True


def test_disconnect_reverses_owned_installation_and_retains_disabled_registry(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    environment = _environment(tmp_path)

    def resolver(alias: ProviderAlias, project: Path) -> ProviderInstallationSpec:
        del alias, project
        return spec

    run_connect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
    )

    report = run_disconnect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
    )

    assert report.capture_enabled is False
    assert report.capture_retained is True
    assert report.disposition == "uninstalled"
    assert not spec.config_path.exists()
    assert not spec.bundle_path.exists()
    assert not spec.bootstrap_path.exists()
    assert not spec.launcher_path.exists()
    assert spec.receipt_path.is_file()
    assert json.loads(render_disconnect_json(report)) == report.model_dump(mode="json")
    human = render_disconnect_human(report)
    assert str(spec.project_root) not in human
    assert spec.capability_digest not in human

    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    with CaptureStore.open(
        locations.database_path,
        installation_key=load_installation_key(default_installation_key_path(environ=environment)),
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        connections = store.list_connections(profile_id=spec.profile)
        assert len(connections) == 1
        assert connections[0].state is CaptureConnectionState.DISABLED


def test_disconnect_reconstructs_command_hook_without_receipt_bound_bundle(
    tmp_path: Path,
) -> None:
    spec = _command_hook_spec(tmp_path)
    environment = _environment(tmp_path)

    def resolver(alias: ProviderAlias, project: Path) -> ProviderInstallationSpec:
        del alias, project
        return spec

    run_connect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
    )

    report = run_disconnect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
    )

    assert report.disposition == "uninstalled"
    assert not spec.config_path.exists()
    assert not spec.launcher_path.exists()
    assert spec.receipt_path.is_file()
    assert spec.bundle_path is None
    assert spec.bootstrap_path is None


def test_disconnect_reconstructs_configless_bridge_without_owned_config(
    tmp_path: Path,
) -> None:
    spec = _configless_bridge_spec(tmp_path)
    environment = _environment(tmp_path)

    def resolver(alias: ProviderAlias, project: Path) -> ProviderInstallationSpec:
        del alias, project
        return spec

    run_connect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
    )

    report = run_disconnect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
    )

    assert report.disposition == "uninstalled"
    assert spec.config_path is None
    assert spec.config is None
    assert not spec.bundle_path.exists()
    assert not spec.bootstrap_path.exists()
    assert not spec.launcher_path.exists()
    assert spec.receipt_path.is_file()


def test_disconnect_is_idempotent_after_completed_uninstall(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    environment = _environment(tmp_path)

    def resolver(alias: ProviderAlias, project: Path) -> ProviderInstallationSpec:
        del alias, project
        return spec

    run_connect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
    )
    run_disconnect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
    )

    report = run_disconnect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
    )

    assert report.disposition == "noop"
    assert report.capture_enabled is False


def test_disconnect_removes_authenticated_provider_artifacts_when_capture_db_is_missing(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    environment = _environment(tmp_path)

    def resolver(alias: ProviderAlias, project: Path) -> ProviderInstallationSpec:
        del alias, project
        return spec

    run_connect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
    )
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    locations.database_path.unlink()

    report = run_disconnect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
    )

    assert report.disposition == "uninstalled"
    assert report.capture_retained is True
    assert not spec.config_path.exists()
    assert not spec.bundle_path.exists()
    assert not spec.bootstrap_path.exists()
    assert not spec.launcher_path.exists()
    assert spec.receipt_path.exists()


def test_disconnect_uses_receipt_bound_launcher_after_executable_relocation(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    environment = _environment(tmp_path)
    executable_a = tmp_path / "capture-executable-a"
    executable_b = tmp_path / "capture-executable-b"
    for executable in (executable_a, executable_b):
        executable.write_bytes(b"#!/bin/sh\nexit 0\n")
        executable.chmod(0o700)

    def resolver(alias: ProviderAlias, project: Path) -> ProviderInstallationSpec:
        del alias, project
        return spec

    run_connect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=executable_a,
    )
    installed_launcher = spec.launcher_path.read_bytes()
    assert str(executable_a).encode() in installed_launcher
    assert str(executable_b).encode() not in installed_launcher
    with pytest.raises(CaptureCommandIntegrityError):
        run_connect(
            provider="codex",
            project=spec.project_root,
            environ=environment,
            spec_resolver=resolver,
            capture_executable=executable_b,
        )
    assert spec.launcher_path.read_bytes() == installed_launcher

    key = load_installation_key(default_installation_key_path(environ=environment))
    identity = derive_installation_identity(spec, key)
    other_profile = capture_profile(CaptureProfile.CLAUDE_CODE_HOOKS_V1)
    other_connection_id = "relocation-unrelated-profile"
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
            connection_id=other_connection_id,
            project_digest=identity.project_digest,
            profile_id=other_profile.profile_id,
            capability_manifest_digest=capture_capability_digest(other_profile),
            host_version=other_profile.host_version,
        )
        store.transition_connection(
            other_connection_id,
            expected_state=CaptureConnectionState.PENDING,
            target_state=CaptureConnectionState.ENABLED,
        )

    report = run_disconnect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=executable_b,
    )
    repeated = run_disconnect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=executable_b,
    )

    assert report.disposition == "uninstalled"
    assert repeated.disposition == "noop"
    assert executable_a.is_file()
    assert executable_b.is_file()
    assert not spec.launcher_path.exists()
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        provider_connections = store.list_connections(
            project_digest=identity.project_digest,
            profile_id=spec.profile,
        )
        other_connections = store.list_connections(
            project_digest=identity.project_digest,
            profile_id=other_profile.profile_id,
        )
    assert len(provider_connections) == 1
    assert provider_connections[0].state is CaptureConnectionState.DISABLED
    assert len(other_connections) == 1
    assert other_connections[0].connection_id == other_connection_id
    assert other_connections[0].state is CaptureConnectionState.ENABLED


def test_disconnect_uses_receipt_bound_launcher_after_capture_executable_is_deleted(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    environment = _environment(tmp_path)
    executable = tmp_path / "capture-executable"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)

    def resolver(alias: ProviderAlias, project: Path) -> ProviderInstallationSpec:
        del alias, project
        return spec

    run_connect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=executable,
    )
    installed_launcher = spec.launcher_path.read_bytes()
    assert str(executable).encode() in installed_launcher
    executable.unlink()

    report = run_disconnect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=executable,
    )

    assert report.disposition == "uninstalled"
    assert not executable.exists()
    assert not spec.config_path.exists()
    assert not spec.bundle_path.exists()
    assert not spec.bootstrap_path.exists()
    assert not spec.launcher_path.exists()
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    with CaptureStore.open(
        locations.database_path,
        installation_key=load_installation_key(default_installation_key_path(environ=environment)),
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        connections = store.list_connections(profile_id=spec.profile)
    assert len(connections) == 1
    assert connections[0].state is CaptureConnectionState.DISABLED


def test_executable_relocation_does_not_authorize_installed_launcher_tamper(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    environment = _environment(tmp_path)
    executable_a = tmp_path / "capture-executable-a"
    executable_b = tmp_path / "capture-executable-b"
    for executable in (executable_a, executable_b):
        executable.write_bytes(b"#!/bin/sh\nexit 0\n")
        executable.chmod(0o700)

    def resolver(alias: ProviderAlias, project: Path) -> ProviderInstallationSpec:
        del alias, project
        return spec

    run_connect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=executable_a,
    )
    retained_paths = (spec.config_path, spec.bundle_path, spec.bootstrap_path)
    retained = {path: path.read_bytes() for path in retained_paths}
    tampered_launcher = b"#!/bin/sh\nexit 7\n"
    spec.launcher_path.write_bytes(tampered_launcher)

    with pytest.raises(CaptureCommandIntegrityError):
        run_disconnect(
            provider="codex",
            project=spec.project_root,
            environ=environment,
            spec_resolver=resolver,
            capture_executable=executable_b,
        )

    assert {path: path.read_bytes() for path in retained_paths} == retained
    assert spec.launcher_path.read_bytes() == tampered_launcher
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    with CaptureStore.open(
        locations.database_path,
        installation_key=load_installation_key(default_installation_key_path(environ=environment)),
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        connections = store.list_connections(profile_id=spec.profile)
    assert len(connections) == 1
    assert connections[0].state is CaptureConnectionState.DRAINING


def test_newer_package_disconnect_uninstalls_authenticated_prior_generation(
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

    report = run_disconnect(
        provider="codex",
        project=second.project_root,
        environ=environment,
        spec_resolver=lambda alias, project: second,
    )

    assert report.disposition == "uninstalled"
    assert not first.config_path.exists()
    assert not first.bundle_path.exists()
    assert not first.bootstrap_path.exists()
    assert not first.launcher_path.exists()
    assert not second.bundle_path.exists()
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        connections = store.list_connections(
            project_digest=first_identity.project_digest,
            profile_id=first.profile,
        )
    assert len(connections) == 1
    assert connections[0].connection_id == first_identity.connection_id
    assert connections[0].connection_id != second_identity.connection_id
    assert connections[0].state is CaptureConnectionState.DISABLED


def test_disconnect_retires_every_active_generation_without_touching_other_profiles(
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
    installed_second = materialize_provider_launcher(second, key)
    install_provider(installed_second, key)
    first_identity = derive_installation_identity(first, key)
    second_identity = derive_installation_identity(second, key)
    other_profile = capture_profile(CaptureProfile.CLAUDE_CODE_HOOKS_V1)
    other_connection_id = "unrelated-profile-active"
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
            connection_id=second_identity.connection_id,
            project_digest=second_identity.project_digest,
            profile_id=second.profile,
            capability_manifest_digest=second.capability_digest,
            host_version=second.host_version,
        )
        store.transition_connection(
            second_identity.connection_id,
            expected_state=CaptureConnectionState.PENDING,
            target_state=CaptureConnectionState.ENABLED,
        )
        store.register_connection(
            connection_id=other_connection_id,
            project_digest=second_identity.project_digest,
            profile_id=other_profile.profile_id,
            capability_manifest_digest=capture_capability_digest(other_profile),
            host_version=other_profile.host_version,
        )
        store.transition_connection(
            other_connection_id,
            expected_state=CaptureConnectionState.PENDING,
            target_state=CaptureConnectionState.ENABLED,
        )

    report = run_disconnect(
        provider="codex",
        project=second.project_root,
        environ=environment,
        spec_resolver=lambda alias, project: second,
    )

    assert report.disposition == "uninstalled"
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        provider_states = {
            connection.connection_id: connection.state
            for connection in store.list_connections(
                project_digest=second_identity.project_digest,
                profile_id=second.profile,
            )
        }
        other_connections = store.list_connections(
            project_digest=second_identity.project_digest,
            profile_id=other_profile.profile_id,
        )
    assert provider_states == {
        first_identity.connection_id: CaptureConnectionState.DISABLED,
        second_identity.connection_id: CaptureConnectionState.DISABLED,
    }
    assert len(other_connections) == 1
    assert other_connections[0].connection_id == other_connection_id
    assert other_connections[0].state is CaptureConnectionState.ENABLED


def test_disconnect_rejects_deleting_connection_before_provider_mutation(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    environment = _environment(tmp_path)
    run_connect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=lambda alias, project: spec,
    )
    key = load_installation_key(default_installation_key_path(environ=environment))
    identity = derive_installation_identity(spec, key)
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
            identity.connection_id,
            expected_state=CaptureConnectionState.ENABLED,
            target_state=CaptureConnectionState.DRAINING,
        )
        store.transition_connection(
            identity.connection_id,
            expected_state=CaptureConnectionState.DRAINING,
            target_state=CaptureConnectionState.DISABLED,
        )
        store.transition_connection(
            identity.connection_id,
            expected_state=CaptureConnectionState.DISABLED,
            target_state=CaptureConnectionState.DELETING,
        )
    provider_paths = (
        spec.config_path,
        spec.bundle_path,
        spec.bootstrap_path,
        spec.receipt_path,
        spec.launcher_path,
    )
    before = {path: path.read_bytes() for path in provider_paths}

    with pytest.raises(CaptureCommandConfigurationError):
        run_disconnect(
            provider="codex",
            project=spec.project_root,
            environ=environment,
            spec_resolver=lambda alias, project: spec,
        )

    assert {path: path.read_bytes() for path in provider_paths} == before
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        connections = store.list_connections(
            project_digest=identity.project_digest,
            profile_id=spec.profile,
        )
    assert len(connections) == 1
    assert connections[0].state is CaptureConnectionState.DELETING


@pytest.mark.parametrize("receipt_damage", ("missing", "corrupt"))
def test_disconnect_fails_closed_when_enabled_installation_receipt_is_untrusted(
    tmp_path: Path,
    receipt_damage: str,
) -> None:
    spec = _spec(tmp_path)
    environment = _environment(tmp_path)

    def resolver(alias: ProviderAlias, project: Path) -> ProviderInstallationSpec:
        del alias, project
        return spec

    run_connect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
    )
    provider_artifacts = (
        spec.config_path,
        spec.bundle_path,
        spec.bootstrap_path,
        spec.launcher_path,
    )
    before = {path: path.read_bytes() for path in provider_artifacts}
    if receipt_damage == "missing":
        spec.receipt_path.unlink()
    else:
        spec.receipt_path.write_bytes(b'{"receipt":"corrupt"}')
    damaged_receipt = None if receipt_damage == "missing" else spec.receipt_path.read_bytes()

    with pytest.raises(CaptureCommandIntegrityError) as raised:
        run_disconnect(
            provider="codex",
            project=spec.project_root,
            environ=environment,
            spec_resolver=resolver,
        )

    assert str(raised.value) == "capture integrity check failed"
    assert {path: path.read_bytes() for path in provider_artifacts} == before
    if damaged_receipt is None:
        assert not spec.receipt_path.exists()
    else:
        assert spec.receipt_path.read_bytes() == damaged_receipt

    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    with CaptureStore.open(
        locations.database_path,
        installation_key=load_installation_key(default_installation_key_path(environ=environment)),
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        connections = store.list_connections(profile_id=spec.profile)
        assert len(connections) == 1
        assert connections[0].state is CaptureConnectionState.DRAINING
