from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable
from pathlib import Path, PureWindowsPath

import pytest
from tests.capture.store_support import INSTALLATION_KEY, authenticated_intake, register_connection

import saliencegate.commands.capture.report as report_module
from saliencegate.capture import (
    CaptureSpool,
    CaptureStore,
    CaptureStoreMode,
    initialize_capture_store,
    resolve_capture_store_locations,
)
from saliencegate.commands.capture.common import (
    CaptureCommandInputError,
    capture_project_digest,
)
from saliencegate.commands.capture.report import run_capture_report
from saliencegate.security import default_installation_key_path
from saliencegate.security.windows import (
    NativeWindowsSecurityOperations,
    ensure_windows_private_directory,
)


def test_windows_report_publisher_routes_callbacks_through_native_private_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Authorization:
        revalidated = False

        def revalidate(self) -> None:
            self.revalidated = True

    class _Published:
        def __init__(self, data: bytes, authorization: _Authorization) -> None:
            self.data = data
            self.authorization = authorization

    authorization = _Authorization()
    observed: dict[str, object] = {}

    class _Operations:
        def publish_private_file_in_managed_directory(
            self,
            path: PureWindowsPath,
            data: bytes,
            *,
            maximum_bytes: int,
            validate_replacement: Callable[[bytes], bool] | None,
            validate_published: Callable[[bytes], bool] | None,
        ) -> _Published:
            observed["path"] = path
            observed["maximum"] = maximum_bytes
            observed["replacement"] = (
                None if validate_replacement is None else validate_replacement(b"old")
            )
            observed["published"] = None if validate_published is None else validate_published(data)
            return _Published(data, authorization)

    operations = _Operations()
    monkeypatch.setattr(
        report_module,
        "NativeWindowsSecurityOperations",
        lambda: operations,
    )

    published = report_module._publish_report_windows(
        r"C:\Users\current\capture-report.json",
        b"encoded-report",
        validate_replacement=lambda current: current == b"old",
        validate_published=lambda current: current == b"encoded-report",
    )

    assert published == b"encoded-report"
    assert observed == {
        "path": PureWindowsPath(r"C:\Users\current\capture-report.json"),
        "maximum": report_module.MAX_CAPTURE_COMMAND_REPORT_BYTES,
        "replacement": True,
        "published": True,
    }
    assert authorization.revalidated is True


def _prepare(tmp_path: Path) -> tuple[Path, dict[str, str], str]:
    project = tmp_path / "project"
    project.mkdir()
    environment = {
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "configuration"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }
    key_path = default_installation_key_path(environ=environment)
    if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        operations = NativeWindowsSecurityOperations()
        ensure_windows_private_directory(
            PureWindowsPath(os.fspath(key_path.parent)),
            operations=operations,
        )
        operations.publish_private_file(
            PureWindowsPath(os.fspath(key_path)),
            INSTALLATION_KEY._serialized(),
            maximum_bytes=32,
            validate_published=lambda current: current == INSTALLATION_KEY._serialized(),
        )
    else:
        key_path.parent.mkdir(mode=0o700, parents=True)
        key_path.write_bytes(INSTALLATION_KEY._serialized())
        key_path.chmod(0o600)
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        ensure_windows_private_directory(
            PureWindowsPath(os.fspath(locations.state_directory)),
            operations=NativeWindowsSecurityOperations(),
        )
    else:
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
        store.append(authenticated_intake("action_started", producer_index=2))
        store.append(authenticated_intake("action_finished", producer_index=3))
        store.append(authenticated_intake("session_finished", producer_index=4))
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


def test_latest_report_is_project_bound_and_can_be_published_owner_private(tmp_path: Path) -> None:
    project, environment, human_id = _prepare(tmp_path)
    output = tmp_path / "capture-report.json"

    report = run_capture_report(
        latest=True,
        project=project,
        output_path=output,
        environ=environment,
    )

    assert report.session_id == human_id
    assert json.loads(output.read_text(encoding="utf-8")) == report.model_dump(mode="json")
    if os.name == "posix":
        assert stat.S_IMODE(output.stat().st_mode) == 0o600
    elif os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        security = NativeWindowsSecurityOperations().inspect_path(
            PureWindowsPath(os.fspath(output))
        )
        assert security is not None
        assert security.owner_private_dacl is True
        assert security.hardlink_count == 1
        assert security.reparse_tag is None
    assert str(project) not in output.read_text(encoding="utf-8")

    assert (
        run_capture_report(
            latest=False,
            session_id=human_id,
            project=project,
            output_path=output,
            replace=True,
            environ=environment,
        )
        == report
    )


def test_relative_report_output_is_normalized_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, environment, human_id = _prepare(tmp_path)
    output_directory = tmp_path / "relative-output"
    output_directory.mkdir()
    monkeypatch.chdir(output_directory)

    report = run_capture_report(
        latest=True,
        project=project,
        output_path="capture-report.json",
        environ=environment,
    )

    output = output_directory / "capture-report.json"
    assert report.session_id == human_id
    assert json.loads(output.read_text(encoding="utf-8")) == report.model_dump(mode="json")


def test_explicit_report_rejects_an_authenticated_other_project_session_before_drain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, environment, _human_id = _prepare(tmp_path)
    other_project, other_human_id = _add_other_project_session(tmp_path, environment)
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    CaptureSpool.open(locations, INSTALLATION_KEY)
    output = tmp_path / "forbidden-report.json"

    def forbidden_drain(*_args: object, **_kwargs: object) -> object:
        pytest.fail("wrong-project report reached spool drain")

    def forbidden_snapshot(*_args: object, **_kwargs: object) -> object:
        pytest.fail("wrong-project report reached session snapshot")

    monkeypatch.setattr(CaptureSpool, "drain", forbidden_drain)
    monkeypatch.setattr(CaptureStore, "snapshot_session", forbidden_snapshot)

    with pytest.raises(CaptureCommandInputError) as raised:
        run_capture_report(
            latest=False,
            session_id=other_human_id,
            project=project,
            output_path=output,
            environ=environment,
        )

    rendered_error = str(raised.value)
    assert other_human_id not in rendered_error
    assert str(project) not in rendered_error
    assert str(other_project) not in rendered_error
    assert not output.exists()

    with CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        selected = store.session_by_human_id(other_human_id)
        assert selected.project_digest == capture_project_digest(
            other_project,
            installation_key=INSTALLATION_KEY,
        )


def test_explicit_report_drains_a_valid_project_session_before_snapshot(tmp_path: Path) -> None:
    project, environment, human_id = _prepare(tmp_path)
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    queued = authenticated_intake(
        "session_started",
        session_native=b"queued-valid-project-session",
        producer_index=21,
    )
    assert spool.enqueue(queued).disposition == "queued"

    report = run_capture_report(
        latest=False,
        session_id=human_id,
        project=project,
        environ=environment,
    )

    assert report.session_id == human_id
    assert spool.health().queued_events == 0
    with CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        assert len(store.list_sessions()) == 2


@pytest.mark.parametrize(
    "values",
    (
        {"latest": False, "session_id": None},
        {"latest": True, "session_id": "abcdefghijkl"},
        {"latest": False, "session_id": "abcdefghijkl", "replace": True},
    ),
)
def test_report_rejects_ambiguous_target_or_replace_without_output(
    tmp_path: Path,
    values: dict[str, object],
) -> None:
    project = tmp_path / "project"
    project.mkdir()

    with pytest.raises(CaptureCommandInputError):
        run_capture_report(project=project, **values)  # type: ignore[arg-type]
