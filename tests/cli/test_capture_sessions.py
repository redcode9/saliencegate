from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.capture.store_support import (
    INSTALLATION_KEY,
    authenticated_intake,
    register_connection,
)

from saliencegate.capture import (
    CaptureStore,
    CaptureStoreMode,
    initialize_capture_store,
    resolve_capture_store_locations,
)
from saliencegate.commands.capture.common import (
    CaptureCommandIntegrityError,
    capture_project_digest,
)
from saliencegate.commands.capture.sessions import (
    render_sessions_human,
    render_sessions_json,
    run_sessions,
)
from saliencegate.security import default_installation_key_path


def _environment(tmp_path: Path) -> dict[str, str]:
    return {
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "configuration"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }


def _prepare(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    project = tmp_path / "project"
    project.mkdir()
    environment = _environment(tmp_path)
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
        store.append(authenticated_intake("session_finished", producer_index=2))
    return project, environment


def test_sessions_are_project_bound_content_free_and_canonically_rendered(tmp_path: Path) -> None:
    project, environment = _prepare(tmp_path)

    report = run_sessions(
        project=project,
        provider="codex",
        state="closed",
        limit=10,
        environ=environment,
    )

    assert report.schema_version == "capture-sessions/v1"
    assert len(report.sessions) == 1
    item = report.sessions[0]
    assert item.provider == "codex"
    assert item.state.value == "closed"
    assert item.event_count == 2
    assert len(item.session_id) >= 12
    rendered_json = render_sessions_json(report)
    assert json.loads(rendered_json) == report.model_dump(mode="json")
    rendered_human = render_sessions_human(report)
    assert item.session_id in rendered_human
    assert "project" not in rendered_human
    assert "digest" not in rendered_human


def test_sessions_without_capture_state_are_empty_and_do_not_write(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    environment = _environment(tmp_path)

    report = run_sessions(project=project, environ=environment)

    assert report.sessions == ()
    assert render_sessions_human(report) == "No captured sessions.\n"
    assert tuple(tmp_path.rglob("*")) == (project,)


def test_sessions_fail_integrity_when_capture_state_outlives_its_key(tmp_path: Path) -> None:
    project, environment = _prepare(tmp_path)
    default_installation_key_path(environ=environment).unlink()

    with pytest.raises(CaptureCommandIntegrityError):
        run_sessions(project=project, environ=environment)
