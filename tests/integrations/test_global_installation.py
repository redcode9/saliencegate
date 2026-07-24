from __future__ import annotations

import io
import json
import os
import sys
from importlib import resources
from pathlib import Path

import pytest

from saliencegate.capture import (
    CaptureGlobalParentState,
    CaptureStore,
    CaptureStoreMode,
    resolve_capture_store_locations,
)
from saliencegate.commands.global_capture import run_global_connect
from saliencegate.domain import canonical_json
from saliencegate.integrations.bootstrap import inspect_integration_bootstrap
from saliencegate.integrations.global_installation import (
    GlobalInstallationError,
    global_provider_installation_spec,
    global_provider_is_available,
    resolve_global_provider_root,
)
from saliencegate.integrations.hook import run_capture_hook
from saliencegate.integrations.installation import (
    InstallationState,
    derive_installation_identity,
    inspect_provider_installation,
)
from saliencegate.integrations.launcher_materialization import materialize_provider_launcher
from saliencegate.integrations.registry import (
    ProviderAlias,
    ProviderInstallationKind,
    ProviderInstallationSpec,
)
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


def _runtime_environment(tmp_path: Path) -> dict[str, str]:
    environment = _environment(tmp_path)
    provider_bin = tmp_path / "provider-bin"
    provider_bin.mkdir()
    providers = {
        "codex": "codex-cli 0.144.6",
        "claude": "2.1.204 (Claude Code)",
    }
    for name, version in providers.items():
        executable = provider_bin / name
        executable.write_bytes(f"#!/bin/sh\nprintf '{version}\\n'\n".encode())
        executable.chmod(0o700)
    environment["PATH"] = os.fspath(provider_bin)
    return environment


def _capture_executable() -> Path:
    return (Path(sys.executable).parent / "saliencegate-capture-hook").resolve(strict=True)


def _global_hook_source(
    provider: ProviderAlias,
    spec: ProviderInstallationSpec,
    project: Path,
) -> bytes:
    if provider in (ProviderAlias.CODEX, ProviderAlias.CLAUDE_CODE):
        fixture_name = (
            "codex-hooks-v1.json"
            if provider is ProviderAlias.CODEX
            else "claude-code-hooks-v1.json"
        )
        document = json.loads(
            resources.files("saliencegate.integrations")
            .joinpath("fixtures")
            .joinpath(fixture_name)
            .read_bytes()
        )
        assert type(document) is dict
        events = document["events"]
        assert type(events) is list
        event = next(item for item in events if item["event_name"] == "SessionStart")
        payload = dict(event["payload"])
        payload["cwd"] = os.fspath(project)
        return canonical_json(payload)

    assert spec.bootstrap_path is not None
    bootstrap = inspect_integration_bootstrap(spec.bootstrap_path)
    document: dict[str, object] = {
        "schema_version": "capture-batch/v1",
        "bootstrap": bootstrap.model_dump(mode="json", warnings="error"),
        "batch_id": "a" * 64,
        "session_id": f"global-{provider.value}-session",
        "workspace_path": os.fspath(project),
        "chunk_index": 0,
        "chunk_count": 1,
        "events": [],
    }
    if provider is ProviderAlias.PI:
        document["window_discriminator"] = "b" * 64
    return canonical_json(document)


@pytest.mark.parametrize(
    ("provider", "relative_root"),
    (
        (ProviderAlias.CODEX, ".codex"),
        (ProviderAlias.CLAUDE_CODE, ".claude"),
        (ProviderAlias.OPENCODE, ".config/opencode"),
        (ProviderAlias.PI, ".pi/agent"),
    ),
)
def test_global_provider_roots_use_provider_user_boundaries(
    tmp_path: Path,
    provider: ProviderAlias,
    relative_root: str,
) -> None:
    environment = _environment(tmp_path)

    assert resolve_global_provider_root(provider, environ=environment) == (
        Path(environment["HOME"]) / relative_root
    )


def test_global_provider_root_honors_explicit_provider_directories(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    roots = {
        "CODEX_HOME": tmp_path / "codex-home",
        "CLAUDE_CONFIG_DIR": tmp_path / "claude-home",
        "OPENCODE_CONFIG_DIR": tmp_path / "opencode-home",
        "PI_CODING_AGENT_DIR": tmp_path / "pi-home",
    }
    for root in roots.values():
        root.mkdir()
    environment.update({name: os.fspath(path) for name, path in roots.items()})

    assert resolve_global_provider_root("codex", environ=environment) == roots["CODEX_HOME"]
    assert (
        resolve_global_provider_root("claude-code", environ=environment)
        == roots["CLAUDE_CONFIG_DIR"]
    )
    assert (
        resolve_global_provider_root("opencode", environ=environment)
        == roots["OPENCODE_CONFIG_DIR"]
    )
    assert resolve_global_provider_root("pi", environ=environment) == roots["PI_CODING_AGENT_DIR"]


@pytest.mark.parametrize("provider", tuple(ProviderAlias))
def test_global_provider_specs_keep_runtime_state_outside_provider_root(
    tmp_path: Path,
    provider: ProviderAlias,
) -> None:
    environment = _environment(tmp_path)
    spec = global_provider_installation_spec(provider, environ=environment)

    assert spec.provider_id == provider.value
    assert spec.project_root == resolve_global_provider_root(provider, environ=environment)
    assert spec.receipt_path.parent == spec.launcher_path.parent
    assert spec.receipt_path.parent.is_relative_to(Path(environment["XDG_STATE_HOME"]))
    assert not spec.receipt_path.is_relative_to(spec.project_root)
    assert all(path.is_relative_to(spec.project_root) for path in spec.project_local_paths)
    if provider in (ProviderAlias.CODEX, ProviderAlias.CLAUDE_CODE):
        assert spec.installation_kind is ProviderInstallationKind.COMMAND_HOOK
        assert spec.config_path is not None
        assert spec.config is not None
        assert os.fspath(spec.launcher_path).encode() in spec.config.owned_fragment
    else:
        assert spec.installation_kind is ProviderInstallationKind.BRIDGE
        assert spec.bundle_path is not None
        assert spec.bootstrap_path is not None


@pytest.mark.parametrize(
    "value",
    ("relative", "", "missing"),
)
def test_global_provider_root_rejects_unsafe_explicit_boundaries(
    tmp_path: Path,
    value: str,
) -> None:
    environment = _environment(tmp_path)
    environment["CODEX_HOME"] = value if value != "missing" else os.fspath(tmp_path / "missing")

    with pytest.raises(GlobalInstallationError, match="global capture integration"):
        resolve_global_provider_root("codex", environ=environment)


def test_global_provider_root_rejects_symlinks(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    environment["CODEX_HOME"] = os.fspath(alias)

    with pytest.raises(GlobalInstallationError):
        resolve_global_provider_root("codex", environ=environment)


def test_global_provider_root_rejects_invalid_provider(tmp_path: Path) -> None:
    environment = _environment(tmp_path)

    with pytest.raises(GlobalInstallationError):
        resolve_global_provider_root("future", environ=environment)


class _ProviderText(str):
    pass


def test_global_installation_rejects_non_exact_boundary_types(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    provider = _ProviderText("codex")

    with pytest.raises(GlobalInstallationError):
        resolve_global_provider_root(provider, environ=environment)
    with pytest.raises(GlobalInstallationError):
        global_provider_installation_spec(provider, environ=environment)
    assert global_provider_is_available(provider, environ=environment) is False


@pytest.mark.parametrize(
    "arguments",
    (
        {"probe_host": 1},
        {"host_version": 1},
        {"probe_host": True, "host_version": "0.144.6"},
    ),
)
def test_global_provider_spec_rejects_ambiguous_probe_configuration(
    tmp_path: Path,
    arguments: dict[str, object],
) -> None:
    environment = _environment(tmp_path)

    with pytest.raises(GlobalInstallationError):
        global_provider_installation_spec(
            "codex",
            environ=environment,
            **arguments,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("provider", (ProviderAlias.OPENCODE, ProviderAlias.PI))
def test_bridge_specs_reject_unknown_host_versions(
    tmp_path: Path,
    provider: ProviderAlias,
) -> None:
    with pytest.raises(GlobalInstallationError):
        global_provider_installation_spec(
            provider,
            environ=_environment(tmp_path),
            host_version="unsupported",
        )


@pytest.mark.skipif(
    os.name != "posix",
    reason="the deterministic fake provider executables are POSIX scripts",
)
def test_global_provider_availability_requires_root_and_executable(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    provider_bin = tmp_path / "provider-bin"
    provider_bin.mkdir()
    executables = {
        "codex": "codex-cli 0.144.6",
        "claude": "2.1.204 (Claude Code)",
        "opencode": "unused",
        "pi": "unused",
    }
    for name, version in executables.items():
        executable = provider_bin / name
        executable.write_bytes(f"#!/bin/sh\nprintf '{version}\\n'\n".encode())
        executable.chmod(0o700)
    environment["PATH"] = os.fspath(provider_bin)

    assert all(
        global_provider_is_available(provider, environ=environment) for provider in ProviderAlias
    )

    environment.pop("PATH")
    assert global_provider_is_available(ProviderAlias.OPENCODE, environ=environment) is False
    environment["PATH"] = os.fspath(provider_bin / "missing")
    assert global_provider_is_available(ProviderAlias.PI, environ=environment) is False


@pytest.mark.skipif(
    os.name != "posix",
    reason="the deterministic fake provider executables are POSIX scripts",
)
@pytest.mark.parametrize("provider", tuple(ProviderAlias))
def test_each_global_provider_installs_and_routes_one_hook_to_a_project_child(
    tmp_path: Path,
    provider: ProviderAlias,
) -> None:
    environment = _runtime_environment(tmp_path)
    project = tmp_path / f"{provider.value}-project"
    (project / ".git").mkdir(parents=True)
    capture_executable = _capture_executable()

    report = run_global_connect(
        provider=provider.value,
        environ=environment,
        capture_executable=capture_executable,
    )

    assert report.capture_enabled is True
    key = load_installation_key(default_installation_key_path(environ=environment))
    spec = materialize_provider_launcher(
        global_provider_installation_spec(provider, environ=environment),
        key,
        capture_executable=capture_executable,
    )
    installation = inspect_provider_installation(spec, key)
    assert installation.installed is True
    assert installation.state is InstallationState.ENABLED
    if spec.installation_kind is ProviderInstallationKind.COMMAND_HOOK:
        assert spec.config_path is not None and spec.config_path.exists()
    else:
        assert spec.bundle_path is not None and spec.bundle_path.exists()
        assert spec.bootstrap_path is not None and spec.bootstrap_path.exists()

    arguments = (
        "--profile",
        spec.profile.value,
        "--connection",
        derive_installation_identity(spec, key).connection_id,
    )
    assert (
        run_capture_hook(
            arguments,
            io.BytesIO(_global_hook_source(provider, spec, project)),
            environ=environment,
            capture_executable=capture_executable,
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
        assert len(parents) == 1
        assert parents[0].state is CaptureGlobalParentState.ENABLED
        children = store.list_global_children(parents[0].global_parent_id)
        assert len(children) == 1
        connection = store.get_connection(children[0].connection_id)
        assert connection.profile_id is spec.profile
        sessions = store.list_sessions(
            project_digest=children[0].project_digest,
            profile_id=spec.profile,
        )
        assert len(sessions) == 1
        assert sessions[0].event_count == 1
