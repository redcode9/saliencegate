"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.cli.test_capture_connect import _environment, _spec

import saliencegate.commands.capture.connect as connect_module
import saliencegate.commands.capture.disconnect as disconnect_module
from saliencegate.capture import CaptureConnectionState
from saliencegate.commands.capture.common import (
    CaptureCommandConfigurationError,
    CaptureCommandInputError,
    CaptureCommandIntegrityError,
    CaptureCommandUnavailableError,
)
from saliencegate.integrations.config_files import ConfigFileError
from saliencegate.integrations.installation import (
    InstallationDisposition,
    InstallationError,
    InstallationState,
)
from saliencegate.integrations.registry import ProviderAlias, ProviderInstallationKind
from saliencegate.security import generate_installation_key
from saliencegate.security.files import StableReadPolicy


class _Context(AbstractContextManager[Any]):
    def __init__(self, value: Any) -> None:
        self.value = value

    def __enter__(self) -> Any:
        return self.value

    def __exit__(self, *args: object) -> None:
        return None


def _connection(connection_id: str, state: CaptureConnectionState) -> SimpleNamespace:
    return SimpleNamespace(connection_id=connection_id, state=state)


class _ActivationStore:
    def __init__(self, registration_state: CaptureConnectionState) -> None:
        self.registration_state = registration_state
        self.transitions: list[tuple[str, CaptureConnectionState, CaptureConnectionState]] = []

    def register_connection(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(state=self.registration_state)

    def list_connections(self, **_kwargs: object) -> tuple[object, ...]:
        return ()

    def transition_connection(
        self,
        connection_id: str,
        *,
        expected_state: CaptureConnectionState,
        target_state: CaptureConnectionState,
    ) -> None:
        self.transitions.append((connection_id, expected_state, target_state))


class _Maintenance:
    def __init__(self, remaining_events: int) -> None:
        self.remaining_events = remaining_events

    def __enter__(self) -> _Maintenance:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def drain(self, _store: object) -> SimpleNamespace:
        return SimpleNamespace(remaining_events=self.remaining_events)


class _Spool:
    def __init__(self, remaining_events: int = 0) -> None:
        self.remaining_events = remaining_events

    def maintenance(self) -> _Maintenance:
        return _Maintenance(self.remaining_events)


def _patch_activation_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    store: _ActivationStore,
    matching: list[list[SimpleNamespace]],
) -> None:
    monkeypatch.setattr(
        connect_module,
        "derive_installation_identity",
        lambda *_args: SimpleNamespace(connection_id="new", project_digest="project"),
    )
    monkeypatch.setattr(
        connect_module,
        "resolve_capture_store_locations",
        lambda **_kwargs: SimpleNamespace(database_path=tmp_path / "capture.db"),
    )
    monkeypatch.setattr(
        connect_module.CaptureStore,
        "open",
        lambda *_args, **_kwargs: _Context(store),
    )
    scripted = iter(matching)
    monkeypatch.setattr(
        connect_module,
        "_matching_store_connections",
        lambda *_args, **_kwargs: tuple(next(scripted)),
    )


@pytest.mark.parametrize("provider", [None, 1, b"codex"])
def test_connect_and_disconnect_reject_non_string_providers(provider: object) -> None:
    with pytest.raises(CaptureCommandInputError):
        connect_module.run_connect(provider=provider)  # type: ignore[arg-type]
    with pytest.raises(CaptureCommandInputError):
        disconnect_module.run_disconnect(provider=provider)  # type: ignore[arg-type]


def test_capture_commands_reject_unknown_provider_and_environment(tmp_path: Path) -> None:
    with pytest.raises(CaptureCommandInputError):
        connect_module.run_connect(provider="unknown")
    with pytest.raises(CaptureCommandInputError):
        disconnect_module.run_disconnect(provider="unknown")

    spec = _spec(tmp_path)

    def resolver(_alias: object, _project: object) -> object:
        return spec

    with pytest.raises(CaptureCommandConfigurationError):
        connect_module.run_connect(
            provider="codex",
            project=spec.project_root,
            environ=object(),  # type: ignore[arg-type]
            spec_resolver=resolver,
        )
    with pytest.raises(CaptureCommandConfigurationError):
        disconnect_module.run_disconnect(
            provider="codex",
            project=spec.project_root,
            environ=object(),  # type: ignore[arg-type]
            spec_resolver=resolver,
        )


def test_default_resolver_maps_missing_noncallable_and_invalid_factories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path.resolve()
    monkeypatch.setattr(
        connect_module.importlib,
        "import_module",
        lambda _name: SimpleNamespace(provider_installation_spec=None),
    )
    with pytest.raises(CaptureCommandUnavailableError):
        connect_module._default_spec_resolver(
            ProviderAlias.CODEX,
            project,
            environ={},
            probe_host=False,
        )

    monkeypatch.setattr(
        connect_module.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError()),
    )
    with pytest.raises(CaptureCommandUnavailableError):
        connect_module._default_spec_resolver(
            ProviderAlias.CODEX,
            project,
            environ={},
            probe_host=False,
        )

    monkeypatch.setattr(
        connect_module.importlib,
        "import_module",
        lambda _name: SimpleNamespace(
            provider_installation_spec=lambda *_args, **_kwargs: (_ for _ in ()).throw(TypeError())
        ),
    )
    with pytest.raises(CaptureCommandConfigurationError):
        connect_module._default_spec_resolver(
            ProviderAlias.CODEX,
            project,
            environ={},
            probe_host=True,
        )


def test_spec_resolution_rejects_noncallable_mismatched_and_unregistered_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    with pytest.raises(CaptureCommandConfigurationError):
        connect_module.resolve_provider_installation_spec(
            ProviderAlias.CODEX,
            spec.project_root,
            object(),  # type: ignore[arg-type]
        )
    with pytest.raises(CaptureCommandConfigurationError):
        connect_module.resolve_provider_installation_spec(
            ProviderAlias.CODEX,
            tmp_path / "other-project",
            lambda _alias, _project: spec,
        )

    monkeypatch.setattr(
        connect_module,
        "BUILTIN_PROVIDER_REGISTRY",
        SimpleNamespace(
            resolve=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                connect_module.ProviderRegistryError()
            )
        ),
    )
    with pytest.raises(CaptureCommandUnavailableError):
        connect_module.resolve_provider_installation_spec(
            ProviderAlias.CODEX,
            spec.project_root,
            lambda _alias, _project: spec,
        )


def test_artifact_probe_fails_closed_on_hostile_parent_and_io_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    spec.config_path.parent.mkdir(parents=True)
    spec.config_path.parent.rmdir()
    spec.config_path.parent.write_bytes(b"not-a-directory")
    assert connect_module.project_provider_artifacts_present(
        ProviderAlias.CODEX,
        spec.project_root,
        resolver=lambda *_args: spec,
    )

    spec.config_path.parent.unlink()
    spec.config_path.parent.mkdir()
    monkeypatch.setattr(
        connect_module,
        "read_config_bytes",
        lambda _path: (_ for _ in ()).throw(ConfigFileError()),
    )
    assert connect_module.project_provider_artifacts_present(
        ProviderAlias.CODEX,
        spec.project_root,
        resolver=lambda *_args: spec,
    )

    configless_root = tmp_path / "configless"
    configless_root.mkdir()
    configless = _spec(configless_root)
    payload = configless.model_dump(mode="python", warnings="error")
    payload.update(config_path=None, config=None)
    configless = type(configless).model_validate(payload)
    original_lstat = Path.lstat

    def fail_launcher(path: Path) -> object:
        if path == configless.launcher_path:
            raise OSError
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_launcher)
    assert connect_module.project_provider_artifacts_present(
        ProviderAlias.CODEX,
        configless.project_root,
        resolver=lambda *_args: configless,
    )


@pytest.mark.parametrize(
    ("registration", "matching", "remaining"),
    [
        (CaptureConnectionState.DRAINING, [], 0),
        (
            CaptureConnectionState.ENABLED,
            [[_connection("old", CaptureConnectionState.DELETING)]],
            0,
        ),
        (
            CaptureConnectionState.ENABLED,
            [
                [
                    _connection("new", CaptureConnectionState.ENABLED),
                    _connection("old", CaptureConnectionState.ENABLED),
                ],
                [_connection("old", CaptureConnectionState.ENABLED)],
            ],
            0,
        ),
        (
            CaptureConnectionState.ENABLED,
            [
                [
                    _connection("new", CaptureConnectionState.ENABLED),
                    _connection("old", CaptureConnectionState.ENABLED),
                ],
                [
                    _connection("new", CaptureConnectionState.ENABLED),
                    _connection("old", CaptureConnectionState.DELETING),
                ],
            ],
            0,
        ),
        (
            CaptureConnectionState.ENABLED,
            [
                [
                    _connection("new", CaptureConnectionState.ENABLED),
                    _connection("old", CaptureConnectionState.ENABLED),
                ],
                [
                    _connection("new", CaptureConnectionState.ENABLED),
                    _connection("old", CaptureConnectionState.ENABLED),
                ],
            ],
            1,
        ),
        (
            CaptureConnectionState.ENABLED,
            [
                [
                    _connection("new", CaptureConnectionState.ENABLED),
                    _connection("old", CaptureConnectionState.ENABLED),
                ],
                [
                    _connection("new", CaptureConnectionState.ENABLED),
                    _connection("old", CaptureConnectionState.ENABLED),
                ],
                [_connection("new", CaptureConnectionState.DRAINING)],
            ],
            0,
        ),
        (
            CaptureConnectionState.ENABLED,
            [
                [
                    _connection("new", CaptureConnectionState.ENABLED),
                    _connection("old", CaptureConnectionState.ENABLED),
                ],
                [
                    _connection("new", CaptureConnectionState.ENABLED),
                    _connection("old", CaptureConnectionState.ENABLED),
                ],
                [
                    _connection("new", CaptureConnectionState.ENABLED),
                    _connection("old", CaptureConnectionState.PENDING),
                ],
            ],
            0,
        ),
        (
            CaptureConnectionState.ENABLED,
            [
                [
                    _connection("new", CaptureConnectionState.ENABLED),
                    _connection("old", CaptureConnectionState.ENABLED),
                ],
                [
                    _connection("new", CaptureConnectionState.ENABLED),
                    _connection("old", CaptureConnectionState.ENABLED),
                ],
                [
                    _connection("new", CaptureConnectionState.ENABLED),
                    _connection("old", CaptureConnectionState.DRAINING),
                ],
                [
                    _connection("new", CaptureConnectionState.DRAINING),
                    _connection("old", CaptureConnectionState.DISABLED),
                ],
            ],
            0,
        ),
        (
            CaptureConnectionState.ENABLED,
            [
                [
                    _connection("new", CaptureConnectionState.ENABLED),
                    _connection("old", CaptureConnectionState.ENABLED),
                ],
                [
                    _connection("new", CaptureConnectionState.ENABLED),
                    _connection("old", CaptureConnectionState.ENABLED),
                ],
                [
                    _connection("new", CaptureConnectionState.ENABLED),
                    _connection("old", CaptureConnectionState.DRAINING),
                ],
                [
                    _connection("new", CaptureConnectionState.ENABLED),
                    _connection("old", CaptureConnectionState.ENABLED),
                ],
            ],
            0,
        ),
    ],
)
def test_activation_rejects_each_impossible_store_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registration: CaptureConnectionState,
    matching: list[list[SimpleNamespace]],
    remaining: int,
) -> None:
    spec = _spec(tmp_path)
    store = _ActivationStore(registration)
    _patch_activation_boundary(monkeypatch, tmp_path, store, matching)
    with pytest.raises(connect_module.CaptureStoreStateError):
        connect_module._activate_store_connection(
            spec,
            generate_installation_key(),
            environment=_environment(tmp_path),
            spool=_Spool(remaining),  # type: ignore[arg-type]
        )


def test_activation_recovers_pending_predecessor_before_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    store = _ActivationStore(CaptureConnectionState.DISABLED)
    matching = [
        [
            _connection("new", CaptureConnectionState.ENABLED),
            _connection("old", CaptureConnectionState.PENDING),
        ],
        [
            _connection("new", CaptureConnectionState.ENABLED),
            _connection("old", CaptureConnectionState.PENDING),
        ],
        [
            _connection("new", CaptureConnectionState.ENABLED),
            _connection("old", CaptureConnectionState.DRAINING),
        ],
        [
            _connection("new", CaptureConnectionState.ENABLED),
            _connection("old", CaptureConnectionState.DRAINING),
        ],
    ]
    _patch_activation_boundary(monkeypatch, tmp_path, store, matching)
    connect_module._activate_store_connection(
        spec,
        generate_installation_key(),
        environment=_environment(tmp_path),
        spool=_Spool(),  # type: ignore[arg-type]
    )
    assert ("old", CaptureConnectionState.PENDING, CaptureConnectionState.ENABLED) in (
        store.transitions
    )
    assert ("old", CaptureConnectionState.DRAINING, CaptureConnectionState.DISABLED) in (
        store.transitions
    )


def test_connect_maps_invalid_install_result_and_preserves_integrity_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    environment = _environment(tmp_path)
    monkeypatch.setattr(
        connect_module,
        "materialize_provider_launcher",
        lambda value, *_a, **_k: value,
    )
    monkeypatch.setattr(connect_module, "git_tracked_project_files", lambda _spec: ())
    monkeypatch.setattr(
        connect_module,
        "_key_for_connect",
        lambda *_a, **_k: generate_installation_key(),
    )
    monkeypatch.setattr(connect_module, "ensure_private_installation_directory", lambda _path: None)
    monkeypatch.setattr(connect_module, "initialize_capture_store", lambda _path: None)
    monkeypatch.setattr(connect_module.CaptureSpool, "open", lambda *_a, **_k: _Spool())
    store = _ActivationStore(CaptureConnectionState.PENDING)
    monkeypatch.setattr(connect_module.CaptureStore, "open", lambda *_a, **_k: _Context(store))
    calls = 0

    def invalid_install(*_args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if kwargs.get("dry_run"):
            return SimpleNamespace(state=InstallationState.ENABLED, installed=True)
        return SimpleNamespace(state=InstallationState.DISABLED, installed=False)

    monkeypatch.setattr(connect_module, "install_provider", invalid_install)
    with pytest.raises(CaptureCommandIntegrityError):
        connect_module.run_connect(
            provider="codex",
            project=spec.project_root,
            environ=environment,
            spec_resolver=lambda *_args: spec,
        )
    assert calls == 2

    monkeypatch.setattr(
        connect_module,
        "_key_for_connect",
        lambda *_a, **_k: (_ for _ in ()).throw(CaptureCommandIntegrityError()),
    )
    with pytest.raises(CaptureCommandIntegrityError):
        connect_module.run_connect(
            provider="codex",
            project=spec.project_root,
            environ=environment,
            spec_resolver=lambda *_args: spec,
        )


def test_uninstall_requires_disabled_result_and_maps_private_read_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    key = generate_installation_key()
    monkeypatch.setattr(
        disconnect_module,
        "inspect_provider_installation",
        lambda *_args: SimpleNamespace(state=InstallationState.ENABLED, drift=()),
    )
    monkeypatch.setattr(
        disconnect_module,
        "uninstall_provider",
        lambda *_args: SimpleNamespace(
            state=InstallationState.ENABLED,
            disposition=InstallationDisposition.NOOP,
        ),
    )
    with pytest.raises(InstallationError):
        disconnect_module._uninstall_to_disabled(spec, key)

    monkeypatch.setattr(
        disconnect_module,
        "read_stable_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()),
    )
    with pytest.raises(InstallationError):
        disconnect_module._read_installed_private_asset(
            spec.launcher_path,
            maximum_bytes=32,
            policy=StableReadPolicy.PRIVATE_EXECUTABLE,
        )


def test_resolve_installed_spec_rejects_mismatched_and_unreadable_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    key = generate_installation_key()
    monkeypatch.setattr(
        disconnect_module,
        "_load_receipt_optional",
        lambda *_args: SimpleNamespace(installation_kind=ProviderInstallationKind.COMMAND_HOOK),
    )
    with pytest.raises(InstallationError):
        disconnect_module._resolve_installed_spec(spec, key)

    monkeypatch.setattr(
        disconnect_module,
        "_load_receipt_optional",
        lambda *_args: (_ for _ in ()).throw(RuntimeError()),
    )
    with pytest.raises(InstallationError):
        disconnect_module._resolve_installed_spec(spec, key)
