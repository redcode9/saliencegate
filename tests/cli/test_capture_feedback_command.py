from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.capture.store_support import INSTALLATION_KEY, authenticated_intake, register_connection

from saliencegate import cli as cli_module
from saliencegate.capture import (
    CaptureStore,
    CaptureStoreError,
    CaptureStoreIntegrityError,
    CaptureStoreMode,
    CaptureStoreStateError,
    initialize_capture_store,
    resolve_capture_store_locations,
)
from saliencegate.commands.capture.common import (
    CaptureCommandConfigurationError,
    CaptureCommandInputError,
    CaptureCommandIntegrityError,
    capture_project_digest,
)
from saliencegate.commands.capture.feedback import (
    render_capture_feedback_human,
    render_capture_feedback_json,
    run_capture_feedback,
)
from saliencegate.domain import canonical_json
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
    project_id = capture_project_digest(project, installation_key=INSTALLATION_KEY)
    with CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        register_connection(store, project_digest=project_id)
        store.append(authenticated_intake("session_started", producer_index=1))
        store.append(authenticated_intake("session_finished", producer_index=2))
        human_id = store.list_sessions(project_digest=project_id)[0].human_id
    return project, environment, human_id


def _add_other_project_session(
    tmp_path: Path,
    environment: dict[str, str],
) -> tuple[Path, str]:
    other_project = tmp_path / "other-project"
    other_project.mkdir()
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    project_id = capture_project_digest(other_project, installation_key=INSTALLATION_KEY)
    with CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        register_connection(
            store,
            connection_id="connection-two",
            project_digest=project_id,
        )
        store.append(
            authenticated_intake(
                "session_started",
                connection_id="connection-two",
                session_native=b"feedback-other-project-session",
                producer_index=11,
            )
        )
        store.append(
            authenticated_intake(
                "session_finished",
                connection_id="connection-two",
                session_native=b"feedback-other-project-session",
                producer_index=12,
            )
        )
        human_id = store.list_sessions(project_digest=project_id)[0].human_id
    return other_project, human_id


def test_feedback_command_records_repeats_and_changes_canonically(tmp_path: Path) -> None:
    project, environment, human_id = _prepare(tmp_path)

    recorded = run_capture_feedback(
        session_id=human_id,
        label="memory-needed",
        project=project,
        environ=environment,
    )
    unchanged = run_capture_feedback(
        session_id=human_id,
        label="memory-needed",
        project=project,
        environ=environment,
    )
    changed = run_capture_feedback(
        session_id=human_id,
        label="uncertain",
        project=project,
        environ=environment,
    )

    assert recorded.disposition.value == "recorded"
    assert recorded.revision_count == 1
    assert unchanged.disposition.value == "unchanged"
    assert unchanged.revision_count == 1
    assert unchanged.labeled_at == recorded.labeled_at
    assert changed.disposition.value == "changed"
    assert changed.revision_count == 2
    assert changed.label.value == "uncertain"
    rendered_json = render_capture_feedback_json(changed)
    assert rendered_json == (
        canonical_json(changed.model_dump(mode="json", warnings=False)).decode("utf-8") + "\n"
    )
    assert json.loads(rendered_json) == changed.model_dump(mode="json")
    assert str(project) not in rendered_json
    assert "project_digest" not in rendered_json
    assert "connection_id" not in rendered_json
    assert "profile_id" not in rendered_json
    rendered = render_capture_feedback_human(changed)
    assert rendered == (f"Capture feedback {human_id}: changed; label=uncertain; revisions=2.\n")
    assert len(rendered.encode("utf-8")) <= 256
    assert str(project) not in rendered
    assert "connection-two" not in rendered


def test_feedback_command_rejects_wrong_project_without_creating_a_spool(
    tmp_path: Path,
) -> None:
    project, environment, own_human_id = _prepare(tmp_path)
    other_project, other_human_id = _add_other_project_session(tmp_path, environment)
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )

    with pytest.raises(CaptureCommandInputError) as raised:
        run_capture_feedback(
            session_id=other_human_id,
            label="not-memory-needed",
            project=project,
            environ=environment,
        )

    assert own_human_id not in str(raised.value)
    assert other_human_id not in str(raised.value)
    assert str(project) not in str(raised.value)
    assert str(other_project) not in str(raised.value)
    assert not locations.spool_directory.exists()


def test_feedback_command_rejects_an_open_session(tmp_path: Path) -> None:
    project, environment, _closed_human_id = _prepare(tmp_path)
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    project_id = capture_project_digest(project, installation_key=INSTALLATION_KEY)
    with CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        started = store.append(
            authenticated_intake(
                "session_started",
                session_native=b"feedback-open-session",
                producer_index=20,
            )
        )
        open_human_id = next(
            item.human_id
            for item in store.list_sessions(project_digest=project_id)
            if item.session_id == started.session_id
        )

    with pytest.raises(CaptureCommandInputError):
        run_capture_feedback(
            session_id=open_human_id,
            label="memory-needed",
            project=project,
            environ=environment,
        )


@pytest.mark.parametrize(
    ("failure", "expected"),
    (
        (CaptureStoreStateError(), CaptureCommandInputError),
        (CaptureStoreIntegrityError(), CaptureCommandIntegrityError),
        (CaptureStoreError(), CaptureCommandConfigurationError),
    ),
)
def test_feedback_command_translates_store_failures_without_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: CaptureStoreError,
    expected: type[Exception],
) -> None:
    project, environment, human_id = _prepare(tmp_path)

    def fail(*_args: object, **_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(CaptureStore, "record_feedback", fail)

    with pytest.raises(expected) as raised:
        run_capture_feedback(
            session_id=human_id,
            label="memory-needed",
            project=project,
            environ=environment,
        )
    assert human_id not in str(raised.value)
    assert str(project) not in str(raised.value)


@pytest.mark.parametrize(
    ("session_id", "label"),
    (
        ("", "memory-needed"),
        ("sgabcdefghijkl", "unknown"),
        ("sgabcdefghijkl", " memory-needed"),
        (None, "memory-needed"),
        ("sgabcdefghijkl", None),
    ),
)
def test_feedback_command_rejects_invalid_direct_inputs(
    tmp_path: Path,
    session_id: object,
    label: object,
) -> None:
    project, environment, _human_id = _prepare(tmp_path)
    with pytest.raises(CaptureCommandInputError):
        run_capture_feedback(
            session_id=session_id,  # type: ignore[arg-type]
            label=label,  # type: ignore[arg-type]
            project=project,
            environ=environment,
        )


def test_feedback_cli_dispatches_canonical_json_and_content_free_human_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project, environment, human_id = _prepare(tmp_path)
    monkeypatch.chdir(project)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    assert (
        cli_module.main(("feedback", human_id, "--label", "not-memory-needed", "--json"))
        == cli_module.ExitCode.SUCCESS
    )
    json_streams = capsys.readouterr()
    payload = json.loads(json_streams.out)
    assert json_streams.err == ""
    assert set(payload) == {
        "schema_version",
        "session_id",
        "label",
        "disposition",
        "revision_count",
        "labeled_at",
    }
    assert payload["schema_version"] == "capture-feedback-receipt/v1"
    assert payload["session_id"] == human_id
    assert payload["label"] == "not-memory-needed"
    assert payload["disposition"] == "recorded"

    assert (
        cli_module.main(("feedback", human_id, "--label", "uncertain"))
        == cli_module.ExitCode.SUCCESS
    )
    human_streams = capsys.readouterr()
    assert human_streams.err == ""
    assert human_streams.out == (
        f"Capture feedback {human_id}: changed; label=uncertain; revisions=2.\n"
    )
    assert str(project) not in human_streams.out
