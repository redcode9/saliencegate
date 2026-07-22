from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

import saliencegate.commands.capture.status as status_module
from saliencegate.capture import (
    CaptureConnectionState,
    CaptureProfile,
    CaptureSpoolError,
    CaptureStoreError,
    CaptureStoreIntegrityError,
)
from saliencegate.capture.locations import CaptureStoreLocations
from saliencegate.commands.capture.common import (
    CaptureCommandConfigurationError,
    CaptureCommandInputError,
    CaptureCommandIntegrityError,
    CaptureCommandUnavailableError,
)
from saliencegate.commands.capture.status import (
    CaptureOperationalStatus,
    CaptureStatusDrift,
    _connector_available,
    _local_bytes,
    _status_without_runtime,
    render_status_human,
    run_status,
)
from saliencegate.integrations.installation import InstallationError, InstallationState
from saliencegate.security import InsecureKeyFileError, InstallationKey

_KEY = InstallationKey(b"s" * 32)
_CONNECTION_ID = UUID("11111111-1111-4111-8111-111111111111")
_OTHER_CONNECTION_ID = UUID("22222222-2222-4222-8222-222222222222")


def _locations(tmp_path: Path) -> CaptureStoreLocations:
    return CaptureStoreLocations(
        platform="posix",
        state_directory=tmp_path,
        database_path=tmp_path / "capture.sqlite3",
        spool_directory=tmp_path / "capture-spool",
    )


def test_connector_availability_and_provider_input_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_resolve(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("fixture-sensitive-detail")

    registry = __import__(
        "saliencegate.integrations.registry",
        fromlist=["BUILTIN_PROVIDER_REGISTRY"],
    )
    monkeypatch.setattr(
        registry,
        "BUILTIN_PROVIDER_REGISTRY",
        SimpleNamespace(resolve=fail_resolve),
    )
    assert not _connector_available("codex")
    with pytest.raises(CaptureCommandInputError):
        run_status(provider="unknown-provider")
    with pytest.raises(CaptureCommandInputError):
        run_status(provider=object())  # type: ignore[arg-type]


def test_local_byte_accounting_sanitizes_directory_and_stat_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _locations(tmp_path)
    locations.spool_directory.mkdir()
    runtime = SimpleNamespace(locations=locations, spool=object())
    original_iterdir = Path.iterdir

    def fail_spool(self: Path):
        if self == locations.spool_directory:
            raise OSError("fixture-sensitive-path")
        return original_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", fail_spool)
    with pytest.raises(CaptureCommandIntegrityError):
        _local_bytes(runtime)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "iterdir", original_iterdir)
    original_lstat = Path.lstat

    def fail_database(self: Path):
        if self == locations.database_path:
            raise OSError("fixture-sensitive-path")
        return original_lstat(self)

    monkeypatch.setattr(Path, "lstat", fail_database)
    with pytest.raises(CaptureCommandIntegrityError):
        _local_bytes(runtime)  # type: ignore[arg-type]


def test_status_without_runtime_rejects_non_mapping_environment(tmp_path: Path) -> None:
    with pytest.raises(CaptureCommandConfigurationError):
        _status_without_runtime(
            ("codex",),
            project=tmp_path,
            environ=[],  # type: ignore[arg-type]
            spec_resolver=None,
            capture_executable=None,
        )


def test_status_without_key_rejects_surviving_runtime_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _locations(tmp_path)
    locations.database_path.write_bytes(b"state")
    monkeypatch.setattr(
        status_module,
        "resolve_capture_store_locations",
        lambda **_kwargs: locations,
    )
    monkeypatch.setattr(
        status_module,
        "load_installation_key",
        lambda _path: (_ for _ in ()).throw(FileNotFoundError()),
    )

    with pytest.raises(CaptureCommandIntegrityError):
        _status_without_runtime(
            ("codex",),
            project=tmp_path,
            environ={"HOME": str(tmp_path)},
            spec_resolver=None,
            capture_executable=None,
        )


def test_status_without_key_treats_unavailable_artifact_resolver_as_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _locations(tmp_path)
    monkeypatch.setattr(
        status_module,
        "resolve_capture_store_locations",
        lambda **_kwargs: locations,
    )
    monkeypatch.setattr(
        status_module,
        "load_installation_key",
        lambda _path: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setattr(
        status_module,
        "project_provider_artifacts_present",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(CaptureCommandUnavailableError()),
    )

    report = _status_without_runtime(
        ("codex",),
        project=tmp_path,
        environ={"HOME": str(tmp_path)},
        spec_resolver=None,
        capture_executable=None,
    )
    assert report.providers[0].status is CaptureOperationalStatus.NOT_INSTALLED


@pytest.mark.parametrize(
    ("failure", "expected"),
    (
        (InsecureKeyFileError, CaptureCommandIntegrityError),
        (TypeError, CaptureCommandConfigurationError),
    ),
)
def test_runtime_fallback_maps_key_and_configuration_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[Exception],
    expected: type[Exception],
) -> None:
    monkeypatch.setattr(status_module, "resolve_capture_project", lambda _project: tmp_path)

    def unavailable(**_kwargs: object) -> object:
        raise CaptureCommandUnavailableError()

    def fail_fallback(*_args: object, **_kwargs: object) -> object:
        raise failure("fixture-sensitive-detail")

    monkeypatch.setattr(status_module, "open_capture_runtime", unavailable)
    monkeypatch.setattr(status_module, "_status_without_runtime", fail_fallback)
    with pytest.raises(expected):
        run_status(provider="codex", project=tmp_path)


@pytest.mark.parametrize(
    ("failure", "expected"),
    (
        (CaptureStoreIntegrityError, CaptureCommandIntegrityError),
        (CaptureSpoolError, CaptureCommandIntegrityError),
        (CaptureStoreError, CaptureCommandConfigurationError),
    ),
)
def test_runtime_open_maps_store_and_spool_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[Exception],
    expected: type[Exception],
) -> None:
    monkeypatch.setattr(status_module, "resolve_capture_project", lambda _project: tmp_path)

    def fail_open(**_kwargs: object) -> object:
        raise failure()

    monkeypatch.setattr(status_module, "open_capture_runtime", fail_open)
    with pytest.raises(expected):
        run_status(provider="codex", project=tmp_path)


def _run_fake_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    connections: tuple[SimpleNamespace, ...],
    installation: SimpleNamespace | None,
    session_count: int = 0,
    dropped: int = 0,
) -> object:
    inventory = SimpleNamespace(
        session_count=session_count,
        quarantined_sessions=0,
        degraded_sessions=0,
        oldest_session=None,
    )
    store = SimpleNamespace(
        list_connections=lambda **_kwargs: connections,
        session_inventory=lambda **_kwargs: inventory,
    )
    spool = SimpleNamespace(health=lambda: SimpleNamespace(queued_events=0, dropped_events=dropped))
    runtime = SimpleNamespace(
        project=tmp_path,
        installation_key=_KEY,
        store=store,
        spool=spool,
    )

    @contextmanager
    def open_runtime(**_kwargs: object):
        yield runtime

    monkeypatch.setattr(status_module, "resolve_capture_project", lambda _project: tmp_path)
    monkeypatch.setattr(status_module, "open_capture_runtime", open_runtime)
    monkeypatch.setattr(status_module, "capture_project_digest", lambda *_args, **_kwargs: "a" * 64)
    monkeypatch.setattr(status_module, "_local_bytes", lambda _runtime: 0)
    monkeypatch.setattr(status_module, "_connector_available", lambda _alias: True)
    if installation is None:
        monkeypatch.setattr(
            status_module,
            "inspect_project_provider_installation",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(CaptureCommandUnavailableError()),
        )
    else:
        monkeypatch.setattr(
            status_module,
            "inspect_project_provider_installation",
            lambda *_args, **_kwargs: installation,
        )
    return run_status(provider="codex", project=tmp_path)


def _connection(identifier: UUID, state: CaptureConnectionState) -> SimpleNamespace:
    return SimpleNamespace(connection_id=identifier, state=state)


def _installation(identifier: UUID, state: InstallationState) -> SimpleNamespace:
    return SimpleNamespace(connection_id=identifier, state=state, drift=())


def test_runtime_status_covers_absent_multiple_pending_and_generation_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    absent = _run_fake_status(
        monkeypatch,
        tmp_path,
        connections=(),
        installation=None,
    )
    assert absent.providers[0].status is CaptureOperationalStatus.NOT_INSTALLED

    pending = _connection(_CONNECTION_ID, CaptureConnectionState.PENDING)
    extra = _connection(_OTHER_CONNECTION_ID, CaptureConnectionState.ENABLED)
    report = _run_fake_status(
        monkeypatch,
        tmp_path,
        connections=(extra, pending),
        installation=_installation(_CONNECTION_ID, InstallationState.ENABLED),
    )
    assert CaptureStatusDrift.MULTIPLE_CONNECTIONS in report.providers[0].drift
    assert CaptureStatusDrift.CONNECTION_PENDING in report.providers[0].drift
    assert CaptureStatusDrift.CONNECTION_GENERATION in report.providers[0].drift
    assert CaptureStatusDrift.INSTALLATION_STATE in report.providers[0].drift
    assert "drift=" in render_status_human(report)


def test_runtime_status_covers_missing_deleting_and_spool_degradation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enabled_installation = _installation(_CONNECTION_ID, InstallationState.ENABLED)
    missing = _run_fake_status(
        monkeypatch,
        tmp_path,
        connections=(),
        installation=enabled_installation,
    )
    assert CaptureStatusDrift.CONNECTION_MISSING in missing.providers[0].drift

    deleting = _connection(_CONNECTION_ID, CaptureConnectionState.DELETING)
    degraded = _run_fake_status(
        monkeypatch,
        tmp_path,
        connections=(deleting,),
        installation=enabled_installation,
        dropped=1,
    )
    assert CaptureStatusDrift.CONNECTION_DELETING in degraded.providers[0].drift
    assert CaptureStatusDrift.SPOOL_DROP in degraded.providers[0].drift


def test_runtime_status_maps_installation_inspection_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimpleNamespace(
        project=tmp_path,
        installation_key=_KEY,
        store=SimpleNamespace(),
        spool=None,
    )

    @contextmanager
    def open_runtime(**_kwargs: object):
        yield runtime

    monkeypatch.setattr(status_module, "resolve_capture_project", lambda _project: tmp_path)
    monkeypatch.setattr(status_module, "open_capture_runtime", open_runtime)
    monkeypatch.setattr(status_module, "capture_project_digest", lambda *_args, **_kwargs: "a" * 64)
    monkeypatch.setattr(status_module, "_local_bytes", lambda _runtime: 0)
    monkeypatch.setattr(
        status_module,
        "inspect_project_provider_installation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(InstallationError()),
    )
    with pytest.raises(CaptureCommandIntegrityError):
        run_status(provider="codex", project=tmp_path)


def test_profile_map_stays_bound_to_the_expected_capture_profile() -> None:
    assert status_module._PROFILES["codex"] is CaptureProfile.CODEX_HOOKS_V1
