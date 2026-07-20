from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.capture.store_support import INSTALLATION_KEY, authenticated_intake, register_connection

import saliencegate.capture.spool as spool_module
from saliencegate.capture import (
    CaptureConnectionState,
    CaptureSpool,
    CaptureSpoolMaintenance,
    CaptureStore,
    CaptureStoreMode,
    initialize_capture_store,
    resolve_capture_store_locations,
)
from saliencegate.commands.capture.common import (
    CaptureCommandInputError,
    CaptureCommandIntegrityError,
    CaptureCommandRequiresDisconnectError,
    capture_project_digest,
)
from saliencegate.commands.capture.delete import (
    render_delete_human,
    render_delete_json,
    run_delete,
)
from saliencegate.security import default_installation_key_path


def _prepare(tmp_path: Path) -> tuple[Path, dict[str, str], str]:
    project = tmp_path / "project"
    project.mkdir()
    environment = {
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "configuration"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }
    key_path = default_installation_key_path(environ=environment)
    key_path.parent.mkdir(mode=0o700, parents=True)
    key_path.write_bytes(INSTALLATION_KEY._serialized())
    key_path.chmod(0o600)
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    locations.state_directory.mkdir(mode=0o700, parents=True)
    initialize_capture_store(locations.database_path)
    project_digest = capture_project_digest(project, installation_key=INSTALLATION_KEY)
    with CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        register_connection(store, project_digest=project_digest)
        store.append(authenticated_intake("session_started", producer_index=1))
        human_id = store.list_sessions()[0].human_id
    return project, environment, human_id


def _add_other_project_session(
    tmp_path: Path,
    environment: dict[str, str],
) -> tuple[Path, str]:
    project = tmp_path / "other-project"
    project.mkdir()
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    project_digest = capture_project_digest(project, installation_key=INSTALLATION_KEY)
    with CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        register_connection(
            store,
            connection_id="connection-two",
            project_digest=project_digest,
        )
        store.append(
            authenticated_intake(
                "session_started",
                connection_id="connection-two",
                session_native=b"other-project-session",
                producer_index=11,
            )
        )
        human_id = store.list_sessions(project_digest=project_digest)[0].human_id
    return project, human_id


def test_single_session_delete_is_explicit_idempotent_and_canonical(tmp_path: Path) -> None:
    project, environment, human_id = _prepare(tmp_path)

    report = run_delete(
        session_id=human_id,
        delete_all=False,
        confirm=False,
        project=project,
        environ=environment,
    )

    assert report.scope == "session"
    assert report.disposition == "deleted"
    assert report.session_id == human_id
    assert report.secure_delete is True
    assert report.wal_checkpointed is True
    assert json.loads(render_delete_json(report)) == report.model_dump(mode="json")
    assert human_id in render_delete_human(report)

    repeated = run_delete(
        session_id=human_id,
        delete_all=False,
        confirm=False,
        project=project,
        environ=environment,
    )
    assert repeated.disposition == "already_deleted"


def test_delete_maps_authenticated_spool_corruption_to_integrity_error(
    tmp_path: Path,
) -> None:
    project, environment, human_id = _prepare(tmp_path)
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    spool.enqueue(authenticated_intake("action_started", producer_index=2))
    entry = next(locations.spool_directory.glob("*.capture-intake"))
    entry.write_bytes(entry.read_bytes() + b"corrupt")

    with pytest.raises(CaptureCommandIntegrityError):
        run_delete(
            session_id=human_id,
            project=project,
            environ=environment,
        )


def test_single_session_delete_rejects_an_authenticated_other_project_session_before_drain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, environment, own_human_id = _prepare(tmp_path)
    other_project, other_human_id = _add_other_project_session(tmp_path, environment)

    def forbidden_drain(*_args: object, **_kwargs: object) -> object:
        pytest.fail("wrong-project delete reached spool drain")

    monkeypatch.setattr(CaptureSpoolMaintenance, "drain", forbidden_drain)

    with pytest.raises(CaptureCommandInputError) as raised:
        run_delete(
            session_id=other_human_id,
            project=project,
            environ=environment,
        )

    rendered_error = str(raised.value)
    assert other_human_id not in rendered_error
    assert str(project) not in rendered_error
    assert str(other_project) not in rendered_error
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    assert not locations.spool_directory.exists()
    with CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        assert store.session_by_human_id(own_human_id).human_id == own_human_id
        assert store.session_by_human_id(other_human_id).human_id == other_human_id


def test_deleted_session_retry_remains_bound_to_its_authenticated_project(
    tmp_path: Path,
) -> None:
    project, environment, own_human_id = _prepare(tmp_path)
    other_project, other_human_id = _add_other_project_session(tmp_path, environment)

    deleted = run_delete(
        session_id=other_human_id,
        project=other_project,
        environ=environment,
    )
    assert deleted.disposition == "deleted"

    with pytest.raises(CaptureCommandInputError) as raised:
        run_delete(
            session_id=other_human_id,
            project=project,
            environ=environment,
        )
    assert other_human_id not in str(raised.value)

    repeated = run_delete(
        session_id=other_human_id,
        project=other_project,
        environ=environment,
    )
    assert repeated.disposition == "already_deleted"

    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    with CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        assert store.session_by_human_id(own_human_id).human_id == own_human_id


def test_delete_all_requires_confirmation_and_a_disabled_project(tmp_path: Path) -> None:
    project, environment, _human_id = _prepare(tmp_path)

    with pytest.raises(CaptureCommandInputError):
        run_delete(delete_all=True, confirm=False, project=project, environ=environment)
    with pytest.raises(CaptureCommandRequiresDisconnectError):
        run_delete(delete_all=True, confirm=True, project=project, environ=environment)

    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    with CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        connection = store.list_connections()[0]
        store.transition_connection(
            connection.connection_id,
            expected_state=CaptureConnectionState.ENABLED,
            target_state=CaptureConnectionState.DRAINING,
        )
        store.transition_connection(
            connection.connection_id,
            expected_state=CaptureConnectionState.DRAINING,
            target_state=CaptureConnectionState.DISABLED,
        )

    deleted = run_delete(
        delete_all=True,
        confirm=True,
        project=project,
        environ=environment,
    )
    assert deleted.scope == "project"
    assert deleted.deleted_connections == 1
    assert deleted.deleted_sessions == 1


def test_delete_all_clears_global_drop_health_after_an_empty_drain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, environment, _human_id = _prepare(tmp_path)
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    monkeypatch.setattr(spool_module, "MAX_CAPTURE_SPOOL_EVENTS", 0)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    dropped = spool.enqueue(authenticated_intake("session_started", producer_index=2))
    assert dropped.disposition == "dropped_quota"
    assert spool.health().dropped_events == 1
    with CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        connection = store.list_connections()[0]
        store.transition_connection(
            connection.connection_id,
            expected_state=CaptureConnectionState.ENABLED,
            target_state=CaptureConnectionState.DRAINING,
        )
        store.transition_connection(
            connection.connection_id,
            expected_state=CaptureConnectionState.DRAINING,
            target_state=CaptureConnectionState.DISABLED,
        )

    report = run_delete(
        delete_all=True,
        confirm=True,
        project=project,
        environ=environment,
    )

    assert report.disposition == "deleted"
    assert spool.health().dropped_events == 0


def test_delete_all_retains_global_drop_health_while_another_project_survives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, environment, _human_id = _prepare(tmp_path)
    _other_project, _other_human_id = _add_other_project_session(tmp_path, environment)
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    monkeypatch.setattr(spool_module, "MAX_CAPTURE_SPOOL_EVENTS", 0)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    assert (
        spool.enqueue(authenticated_intake("session_started", producer_index=21)).disposition
        == "dropped_quota"
    )
    project_id = capture_project_digest(project, installation_key=INSTALLATION_KEY)
    with CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        for connection in store.list_connections(project_digest=project_id):
            store.transition_connection(
                connection.connection_id,
                expected_state=CaptureConnectionState.ENABLED,
                target_state=CaptureConnectionState.DRAINING,
            )
            store.transition_connection(
                connection.connection_id,
                expected_state=CaptureConnectionState.DRAINING,
                target_state=CaptureConnectionState.DISABLED,
            )

    report = run_delete(
        delete_all=True,
        confirm=True,
        project=project,
        environ=environment,
    )

    assert report.disposition == "deleted"
    assert spool.health().dropped_events == 1
