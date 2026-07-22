from __future__ import annotations

import json
import os
import shutil
import sqlite3
import stat
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError
from tests.capture.store_support import INSTALLATION_KEY, authenticated_intake, register_connection
from tests.cli.conftest import RunCli

import saliencegate.capture.spool as spool_module
import saliencegate.commands.doctor as doctor_module
import saliencegate.integrations.codex as codex_module
from saliencegate.capture import (
    CaptureSpool,
    CaptureStore,
    CaptureStoreMode,
    initialize_capture_store,
    resolve_capture_store_locations,
)
from saliencegate.commands.capture.connect import run_connect
from saliencegate.commands.doctor import (
    CaptureDoctorCheck,
    CaptureDoctorReport,
    CaptureDoctorState,
    DoctorCheck,
    DoctorCheckName,
    DoctorCheckStatus,
    DoctorReport,
    DoctorReportStatus,
    DoctorSeverity,
    PilotEndpointError,
    render_capture_doctor_human,
    render_capture_doctor_json,
    render_doctor_human,
    render_doctor_json,
    run_capture_doctor,
    run_doctor,
    validated_pilot_endpoint,
)
from saliencegate.security import default_installation_key_path


def _check(report: DoctorReport, name: DoctorCheckName) -> DoctorCheck:
    return next(check for check in report.checks if check.name is name)


def _configured_capture(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    project = tmp_path / "capture-project"
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
    CaptureSpool.open(locations, INSTALLATION_KEY)
    with CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        register_connection(store)
    return project, environment, locations.database_path


def test_doctor_succeeds_without_creating_a_key_or_contacting_a_model(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository_path = tmp_path / "runs.sqlite3"
    key_path = tmp_path / "configuration" / "installation.key"

    report = run_doctor(
        repository_path=repository_path,
        installation_key_path=key_path,
    )

    assert report.schema_version == "doctor/v1"
    assert report.status is DoctorReportStatus.HEALTHY
    assert report.ok is True
    assert tuple(check.name for check in report.checks) == tuple(DoctorCheckName)
    assert all(
        check.status is DoctorCheckStatus.PASS
        for check in report.checks
        if check.name is not DoctorCheckName.ENDPOINT
    )
    endpoint = _check(report, DoctorCheckName.ENDPOINT)
    assert endpoint.status is DoctorCheckStatus.SKIP
    assert endpoint.required is False
    assert endpoint.severity is DoctorSeverity.INFO
    assert not repository_path.exists()
    assert not key_path.exists()
    assert capsys.readouterr() == ("", "")


def test_capture_doctor_is_read_only_when_capture_is_not_configured(tmp_path: Path) -> None:
    environment = {
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "configuration"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }

    report = run_capture_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        environ=environment,
    )

    assert type(report) is CaptureDoctorReport
    assert report.capture.state is CaptureDoctorState.NOT_CONFIGURED
    assert report.status is DoctorReportStatus.HEALTHY
    assert tuple(tmp_path.rglob("*")) == ()
    assert json.loads(render_capture_doctor_json(report)) == report.model_dump(mode="json")
    assert "Passive capture (optional): Capture is not configured." in render_capture_doctor_human(
        report
    )


def test_capture_doctor_authenticates_store_rows_without_writing(tmp_path: Path) -> None:
    project, environment, database_path = _configured_capture(tmp_path)
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    ready = run_capture_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        environ=environment,
        capture_project=project,
    )

    assert ready.capture.state is CaptureDoctorState.READY
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    connection = sqlite3.connect(database_path)
    connection.execute("UPDATE connections SET row_tag = ?", ("0" * 64,))
    connection.commit()
    connection.close()

    degraded = run_capture_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        environ=environment,
        capture_project=project,
    )
    assert degraded.capture.state is CaptureDoctorState.DEGRADED


@pytest.mark.skipif(os.name != "posix", reason="native Windows lifecycle is covered by R01")
def test_capture_doctor_never_launches_codex_during_read_only_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "capture-project"
    project.mkdir()
    provider_bin = tmp_path / "provider-bin"
    provider_bin.mkdir()
    codex_executable = provider_bin / "codex"
    codex_executable.write_bytes(
        b'#!/bin/sh\n/bin/mkdir -p "$HOME"\nprintf probed > "$HOME/doctor-probe"\n'
        b"printf 'codex-cli 0.144.6\\n'\n"
    )
    codex_executable.chmod(0o700)
    environment = {
        "HOME": str(tmp_path / "home"),
        "PATH": str(provider_bin),
        "XDG_CONFIG_HOME": str(tmp_path / "configuration"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }
    run_connect(
        provider="codex",
        project=project,
        environ=environment,
        capture_executable=Path(sys.executable),
    )
    probe_marker = Path(environment["HOME"]) / "doctor-probe"
    assert probe_marker.read_bytes() == b"probed"
    probe_marker.unlink()

    def forbidden_probe(**_kwargs: object) -> object:
        raise AssertionError("doctor must not execute Codex")

    monkeypatch.setattr(codex_module, "probe_codex_environment", forbidden_probe)
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    report = run_capture_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        environ=environment,
        capture_project=project,
        capture_executable=Path(sys.executable),
    )

    assert report.capture.state is CaptureDoctorState.READY
    assert not probe_marker.exists()
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }


def test_capture_doctor_requires_the_configured_spool_boundary(tmp_path: Path) -> None:
    project, environment, _database_path = _configured_capture(tmp_path)
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    shutil.rmtree(locations.spool_directory)

    report = run_capture_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        environ=environment,
        capture_project=project,
    )

    assert report.capture.state is CaptureDoctorState.DEGRADED


def test_capture_doctor_treats_dangling_runtime_symlinks_as_degraded(tmp_path: Path) -> None:
    project = tmp_path / "capture-project"
    project.mkdir()
    environment = {
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "configuration"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    locations.state_directory.mkdir(mode=0o700, parents=True)
    locations.database_path.symlink_to(tmp_path / "missing-database")

    report = run_capture_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        environ=environment,
        capture_project=project,
    )

    assert report.capture.state is CaptureDoctorState.DEGRADED


def test_capture_doctor_authenticates_spooled_intakes_without_writing(tmp_path: Path) -> None:
    project, environment, _database_path = _configured_capture(tmp_path)
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    spool.enqueue(authenticated_intake("session_started", producer_index=1))
    entry = next(locations.spool_directory.glob("*.capture-intake"))
    entry.write_bytes(entry.read_bytes()[:-1] + b"x")
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in locations.spool_directory.iterdir()
        if path.is_file()
    }

    report = run_capture_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        environ=environment,
        capture_project=project,
    )

    assert report.capture.state is CaptureDoctorState.DEGRADED
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in locations.spool_directory.iterdir()
        if path.is_file()
    }


def test_capture_doctor_reports_authenticated_drop_health_as_degraded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, environment, _database_path = _configured_capture(tmp_path)
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    monkeypatch.setattr(spool_module, "MAX_CAPTURE_SPOOL_EVENTS", 0)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    dropped = spool.enqueue(authenticated_intake("session_started", producer_index=1))
    assert dropped.disposition == "dropped_quota"

    report = run_capture_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        environ=environment,
        capture_project=project,
    )

    assert report.capture.state is CaptureDoctorState.DEGRADED


def test_doctor_renderers_are_deterministic_and_do_not_write_streams(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = run_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        installation_key_path=tmp_path / "installation.key",
        endpoint="http://127.0.0.1:11434/v1",
    )

    rendered_json = render_doctor_json(report)
    rendered_human = render_doctor_human(report)

    assert rendered_json.endswith("\n")
    assert json.loads(rendered_json) == report.model_dump(mode="json")
    assert render_doctor_json(report) == rendered_json
    assert rendered_human.startswith("SalienceGate doctor: healthy\n")
    assert "[PASS] Python runtime (required):" in rendered_human
    assert "[PASS] Model endpoint (required):" in rendered_human
    assert "127.0.0.1" not in rendered_human
    assert rendered_human.endswith("\n")
    assert capsys.readouterr() == ("", "")


def test_unsupported_python_is_a_required_runtime_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(doctor_module, "_runtime_python_version", lambda: (3, 10, 14))

    report = run_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        installation_key_path=tmp_path / "installation.key",
    )

    check = _check(report, DoctorCheckName.PYTHON)
    assert report.status is DoctorReportStatus.UNHEALTHY
    assert report.ok is False
    assert check.status is DoctorCheckStatus.FAIL
    assert check.required is True
    assert check.severity is DoctorSeverity.ERROR
    assert "3.10.14" in check.message


def test_too_old_sqlite_is_a_required_dependency_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(doctor_module, "_runtime_sqlite_version", lambda: (3, 23, 1))

    report = run_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        installation_key_path=tmp_path / "installation.key",
    )

    check = _check(report, DoctorCheckName.SQLITE)
    assert check.status is DoctorCheckStatus.FAIL
    assert check.severity is DoctorSeverity.ERROR
    assert "3.23.1" in check.message
    assert report.ok is False


def test_fts5_is_probed_with_an_ephemeral_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_database: str) -> None:
        raise doctor_module.sqlite3.OperationalError("fixture detail must not escape")

    monkeypatch.setattr(doctor_module.sqlite3, "connect", unavailable)

    report = run_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        installation_key_path=tmp_path / "installation.key",
    )

    check = _check(report, DoctorCheckName.FTS5)
    assert check.status is DoctorCheckStatus.FAIL
    assert check.severity is DoctorSeverity.ERROR
    assert "fixture detail" not in check.message
    assert report.status is DoctorReportStatus.UNHEALTHY


def test_existing_repository_file_must_be_writable_with_a_writable_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_path = tmp_path / "runs.sqlite3"
    repository_path.touch()
    real_access = doctor_module._path_access

    def deny_repository(path: Path, mode: int) -> bool:
        if path == repository_path:
            return False
        return real_access(path, mode)

    monkeypatch.setattr(doctor_module, "_path_access", deny_repository)

    report = run_doctor(
        repository_path=repository_path,
        installation_key_path=tmp_path / "installation.key",
    )

    check = _check(report, DoctorCheckName.REPOSITORY_PATH)
    assert check.status is DoctorCheckStatus.FAIL
    assert "readable and writable" in check.message


def test_repository_memory_target_and_writable_directory_are_supported(tmp_path: Path) -> None:
    memory_report = run_doctor(
        repository_path=":memory:",
        installation_key_path=tmp_path / "installation.key",
    )
    directory_report = run_doctor(
        repository_path=tmp_path,
        installation_key_path=tmp_path / "installation.key",
    )

    assert _check(memory_report, DoctorCheckName.REPOSITORY_PATH).status is DoctorCheckStatus.PASS
    assert (
        _check(directory_report, DoctorCheckName.REPOSITORY_PATH).status is DoctorCheckStatus.PASS
    )


def test_repository_rejects_symlinks_and_non_directory_ancestors(tmp_path: Path) -> None:
    target = tmp_path / "target.sqlite3"
    target.touch()
    link = tmp_path / "linked.sqlite3"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("symbolic links are unavailable")

    linked_report = run_doctor(
        repository_path=link,
        installation_key_path=tmp_path / "installation.key",
    )
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.touch()
    blocked_report = run_doctor(
        repository_path=blocking_file / "runs.sqlite3",
        installation_key_path=tmp_path / "installation.key",
    )

    assert _check(linked_report, DoctorCheckName.REPOSITORY_PATH).status is DoctorCheckStatus.FAIL
    assert _check(blocked_report, DoctorCheckName.REPOSITORY_PATH).status is DoctorCheckStatus.FAIL


def test_existing_private_installation_key_passes_without_reading_material(tmp_path: Path) -> None:
    key_path = tmp_path / "installation.key"
    key_path.write_bytes(b"k" * 32)
    key_path.chmod(0o600)

    report = run_doctor(repository_path=tmp_path / "runs.sqlite3", installation_key_path=key_path)

    check = _check(report, DoctorCheckName.INSTALLATION_KEY)
    assert check.status is DoctorCheckStatus.PASS
    assert key_path.read_bytes() == b"k" * 32
    assert "kkkk" not in check.message


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits are platform-specific")
def test_installation_key_rejects_group_or_other_permissions(tmp_path: Path) -> None:
    key_path = tmp_path / "installation.key"
    key_path.write_bytes(b"k" * 32)
    key_path.chmod(0o640)

    report = run_doctor(repository_path=tmp_path / "runs.sqlite3", installation_key_path=key_path)

    check = _check(report, DoctorCheckName.INSTALLATION_KEY)
    assert check.status is DoctorCheckStatus.FAIL
    assert "owner-only" in check.message


def test_installation_key_rejects_symlinks_and_invalid_size(tmp_path: Path) -> None:
    short_key = tmp_path / "short.key"
    short_key.write_bytes(b"short")
    short_key.chmod(0o600)
    target = tmp_path / "target.key"
    target.write_bytes(b"k" * 32)
    target.chmod(0o600)
    linked_key = tmp_path / "linked.key"
    try:
        linked_key.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("symbolic links are unavailable")

    short_report = run_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        installation_key_path=short_key,
    )
    linked_report = run_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        installation_key_path=linked_key,
    )

    assert _check(short_report, DoctorCheckName.INSTALLATION_KEY).status is DoctorCheckStatus.FAIL
    assert _check(linked_report, DoctorCheckName.INSTALLATION_KEY).status is DoctorCheckStatus.FAIL


def test_installation_key_path_must_be_absolute(tmp_path: Path) -> None:
    report = run_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        installation_key_path=Path("relative.key"),
    )

    check = _check(report, DoctorCheckName.INSTALLATION_KEY)
    assert check.status is DoctorCheckStatus.FAIL
    assert "absolute" in check.message


def test_invalid_default_key_configuration_is_reported_without_raising(tmp_path: Path) -> None:
    report = run_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        environ={"XDG_CONFIG_HOME": "relative-configuration"},
    )

    assert _check(report, DoctorCheckName.INSTALLATION_KEY).status is DoctorCheckStatus.FAIL
    assert report.ok is False


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        " https://example.test/v1",
        "ftp://example.test/v1",
        "https:///v1",
        "https://user:password@example.test/v1",
        "https://example.test/v1?token=secret",
        "https://example.test/v1#fragment",
        "http://example.test:99999/v1",
        "https://[invalid/v1",
    ],
)
def test_configured_endpoint_is_validated_syntactically_without_leaking_it(
    tmp_path: Path,
    endpoint: str,
) -> None:
    report = run_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        installation_key_path=tmp_path / "installation.key",
        endpoint=endpoint,
    )

    check = _check(report, DoctorCheckName.ENDPOINT)
    assert check.status is DoctorCheckStatus.FAIL
    assert check.required is True
    assert check.severity is DoctorSeverity.ERROR
    if endpoint:
        assert endpoint not in check.message
    assert report.status is DoctorReportStatus.UNHEALTHY


@pytest.mark.parametrize(
    "endpoint",
    ["http://localhost:11434/v1", "https://models.example.test/v1"],
)
def test_valid_endpoint_configuration_is_not_contacted(tmp_path: Path, endpoint: str) -> None:
    report = run_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        installation_key_path=tmp_path / "installation.key",
        endpoint=endpoint,
    )

    check = _check(report, DoctorCheckName.ENDPOINT)
    assert check.status is DoctorCheckStatus.PASS
    assert check.required is True
    assert "connectivity was not attempted" in check.message


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    (
        ("http://127.0.0.1/v1", "http://127.0.0.1/v1"),
        ("http://127.0.0.1/v1/", "http://127.0.0.1/v1"),
        ("http://127.0.0.2:1/v1", "http://127.0.0.2:1/v1"),
        ("http://127.255.255.254:80/v1", "http://127.255.255.254:80/v1"),
        ("http://127.0.0.1:11434/v1/", "http://127.0.0.1:11434/v1"),
        ("http://127.0.0.1:65535/v1", "http://127.0.0.1:65535/v1"),
        ("http://[::1]/v1", "http://[::1]/v1"),
        ("http://[::1]/v1/", "http://[::1]/v1"),
        ("http://[::1]:11434/v1", "http://[::1]:11434/v1"),
        ("http://[::1]:65535/v1/", "http://[::1]:65535/v1"),
    ),
)
def test_pilot_endpoint_accepts_only_canonical_numeric_loopback_urls(
    endpoint: str,
    expected: str,
) -> None:
    assert validated_pilot_endpoint(endpoint) == expected


@pytest.mark.parametrize(
    "endpoint",
    (
        None,
        b"http://127.0.0.1/v1",
        "",
        " http://127.0.0.1/v1",
        "http://127.0.0.1/v1 ",
        "http://127.0.0.1/\x00v1",
        "http://127.0.0.1/v1\n",
        "http://127.0.0.1/v1\x7f",
        "http://127.0.0.1/v1/é",
        "HTTP://127.0.0.1/v1",
        "https://127.0.0.1/v1",
        "ftp://127.0.0.1/v1",
        "http:/127.0.0.1/v1",
        "http:///127.0.0.1/v1",
        "http://localhost/v1",
        "http://localhost:11434/v1",
        "http://models.example.test/v1",
        "http://0.0.0.0/v1",
        "http://126.255.255.255/v1",
        "http://128.0.0.1/v1",
        "http://192.168.1.1/v1",
        "http://[::]/v1",
        "http://[::2]/v1",
        "http://[::ffff:127.0.0.1]/v1",
        "http://127.1/v1",
        "http://127.000.000.001/v1",
        "http://2130706433/v1",
        "http://127.0.0.1./v1",
        "http://[0:0:0:0:0:0:0:1]/v1",
        "http://user@127.0.0.1/v1",
        "http://user:password@127.0.0.1/v1",
        "http://@127.0.0.1/v1",
        "http://127.0.0.1/v1?model=gpt-oss",
        "http://127.0.0.1/v1?",
        "http://127.0.0.1/v1#fragment",
        "http://127.0.0.1/v1#",
        "http://127.0.0.1/%76%31",
        "http://127.0.0.1/v1%2f",
        "http://127.0.0.1\\v1",
        "http:\\127.0.0.1/v1",
        "http://127.0.0.1",
        "http://127.0.0.1/",
        "http://127.0.0.1/V1",
        "http://127.0.0.1/v1//",
        "http://127.0.0.1/v1/.",
        "http://127.0.0.1/v1/resource",
        "http://127.0.0.1/v1;parameter",
        "http://127.0.0.1:/v1",
        "http://127.0.0.1:0/v1",
        "http://127.0.0.1:01/v1",
        "http://127.0.0.1:080/v1",
        "http://127.0.0.1:+80/v1",
        "http://127.0.0.1:-1/v1",
        "http://127.0.0.1:65536/v1",
        "http://127.0.0.1:99999/v1",
        "http://127.0.0.1:port/v1",
        "http://127.0.0.1:\uff18\uff10/v1",
        "http://::1/v1",
        "http://[::1/v1",
        "http://[::1]]/v1",
        "http://[::1]:/v1",
        "http://[::1]:011434/v1",
        "http://[::1%25lo0]/v1",
        "http://127.0.0.1:80:90/v1",
        "http://127.0.0.1/v1/" + "a" * 2_048,
    ),
)
def test_pilot_endpoint_rejects_noncanonical_unsafe_or_remote_urls(endpoint: Any) -> None:
    with pytest.raises(PilotEndpointError, match=r"^pilot endpoint is invalid$"):
        validated_pilot_endpoint(endpoint)


def test_pilot_endpoint_failure_is_value_free_and_has_no_exception_chain() -> None:
    endpoint = "http://fixture-user:fixture-secret@127.0.0.1:99999/v1?token=secret"

    with pytest.raises(PilotEndpointError) as raised:
        validated_pilot_endpoint(endpoint)

    assert str(raised.value) == "pilot endpoint is invalid"
    assert endpoint not in str(raised.value)
    assert endpoint not in repr(raised.value)
    assert "fixture-secret" not in str(raised.value)
    assert "fixture-secret" not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_doctor_models_reject_inconsistent_status_and_unsafe_messages(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="required check cannot be skipped"):
        DoctorCheck(
            name=DoctorCheckName.PYTHON,
            status=DoctorCheckStatus.SKIP,
            severity=DoctorSeverity.INFO,
            required=True,
            message="Not checked.",
        )
    with pytest.raises(ValidationError, match="control characters"):
        DoctorCheck(
            name=DoctorCheckName.PYTHON,
            status=DoctorCheckStatus.PASS,
            severity=DoctorSeverity.INFO,
            required=True,
            message="unsafe\nmessage",
        )

    invalid_checks = (
        {
            "name": DoctorCheckName.PYTHON,
            "status": DoctorCheckStatus.PASS,
            "severity": DoctorSeverity.WARNING,
            "required": True,
            "message": "Invalid passing severity.",
        },
        {
            "name": DoctorCheckName.ENDPOINT,
            "status": DoctorCheckStatus.SKIP,
            "severity": DoctorSeverity.WARNING,
            "required": False,
            "message": "Invalid skipped severity.",
        },
        {
            "name": DoctorCheckName.PYTHON,
            "status": DoctorCheckStatus.FAIL,
            "severity": DoctorSeverity.WARNING,
            "required": True,
            "message": "Invalid failure severity.",
        },
    )
    for invalid in invalid_checks:
        with pytest.raises(ValidationError):
            DoctorCheck.model_validate(invalid)

    report = run_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        installation_key_path=tmp_path / "installation.key",
    )
    payload = report.model_dump(mode="python")
    payload["checks"] = tuple(reversed(report.checks))
    with pytest.raises(ValidationError, match="canonical order"):
        DoctorReport.model_validate(payload)
    with pytest.raises(ValidationError, match="summary does not match"):
        DoctorReport(
            status=DoctorReportStatus.HEALTHY,
            ok=False,
            checks=report.checks,
        )

    with pytest.raises(ValidationError, match="capture doctor state is inconsistent"):
        CaptureDoctorCheck(
            state=CaptureDoctorState.READY,
            status=DoctorCheckStatus.SKIP,
            required=False,
            message="Capture status is inconsistent.",
        )
    with pytest.raises(ValidationError, match="control characters"):
        CaptureDoctorCheck(
            state=CaptureDoctorState.READY,
            status=DoctorCheckStatus.PASS,
            required=True,
            message="unsafe\nmessage",
        )


def test_doctor_report_supports_optional_degraded_checks(tmp_path: Path) -> None:
    healthy = run_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        installation_key_path=tmp_path / "installation.key",
    )
    checks = list(healthy.checks)
    checks[-1] = DoctorCheck(
        name=DoctorCheckName.ENDPOINT,
        status=DoctorCheckStatus.FAIL,
        severity=DoctorSeverity.WARNING,
        required=False,
        message="Optional endpoint configuration is unavailable.",
    )

    degraded = DoctorReport(
        status=DoctorReportStatus.DEGRADED,
        ok=True,
        checks=tuple(checks),
    )

    assert degraded.status is DoctorReportStatus.DEGRADED
    capture = CaptureDoctorCheck(
        state=CaptureDoctorState.READY,
        status=DoctorCheckStatus.PASS,
        required=True,
        message="Capture is ready.",
    )
    assert (
        CaptureDoctorReport(
            status=DoctorReportStatus.DEGRADED,
            ok=True,
            environment=degraded,
            capture=capture,
        ).status
        is DoctorReportStatus.DEGRADED
    )
    with pytest.raises(ValidationError, match="capture doctor report summary is inconsistent"):
        CaptureDoctorReport(
            status=DoctorReportStatus.HEALTHY,
            ok=True,
            environment=degraded,
            capture=capture,
        )
    with pytest.raises(ValidationError, match="summary does not match"):
        DoctorReport(
            status=DoctorReportStatus.HEALTHY,
            ok=True,
            checks=tuple(checks),
        )


def test_doctor_cli_supports_machine_and_human_output_without_writing_configuration(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    repository = tmp_path / "runs.sqlite3"
    key = tmp_path / "configuration" / "installation.key"

    machine = run_cli(
        "doctor",
        "--repository",
        str(repository),
        "--key",
        str(key),
        "--json",
    )
    human = run_cli(
        "doctor",
        "--repository",
        str(repository),
        "--key",
        str(key),
    )

    assert machine.returncode == human.returncode == 0
    assert machine.stderr == human.stderr == ""
    payload = json.loads(machine.stdout)
    assert payload["schema_version"] == "doctor/v1"
    assert payload["status"] == "healthy"
    assert payload["ok"] is True
    assert human.stdout.startswith("SalienceGate doctor: healthy\n")
    assert not repository.exists()
    assert not key.exists()


def test_doctor_cli_maps_invalid_configuration_without_disclosing_endpoint(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    endpoint = "https://user:fixture-secret@example.test/v1"

    completed = run_cli(
        "doctor",
        "--repository",
        str(tmp_path / "runs.sqlite3"),
        "--key",
        str(tmp_path / "installation.key"),
        "--endpoint",
        endpoint,
        "--json",
    )

    assert completed.returncode == 3
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["status"] == "unhealthy"
    assert endpoint not in completed.stdout
    assert "fixture-secret" not in completed.stdout


def test_non_regular_repository_target_is_rejected(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO files are unavailable")
    fifo = tmp_path / "runs.fifo"
    os.mkfifo(fifo, mode=stat.S_IRUSR | stat.S_IWUSR)

    report = run_doctor(
        repository_path=fifo,
        installation_key_path=tmp_path / "installation.key",
    )

    assert _check(report, DoctorCheckName.REPOSITORY_PATH).status is DoctorCheckStatus.FAIL


def test_runtime_probe_exceptions_are_reported_as_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> tuple[int, int, int]:
        raise RuntimeError("private fixture detail")

    monkeypatch.setattr(doctor_module, "_runtime_python_version", unavailable)
    monkeypatch.setattr(doctor_module, "_runtime_sqlite_version", unavailable)

    report = run_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        installation_key_path=tmp_path / "installation.key",
    )

    assert _check(report, DoctorCheckName.PYTHON).status is DoctorCheckStatus.FAIL
    assert _check(report, DoctorCheckName.SQLITE).status is DoctorCheckStatus.FAIL
    assert "private fixture" not in render_doctor_human(report)


class _FakeFtsConnection:
    def __init__(self, *, row: tuple[int] = (1,), execute_error: bool = False) -> None:
        self.row = row
        self.execute_error = execute_error
        self.closed = False
        self.fail_close = False

    def execute(self, _statement: str) -> _FakeFtsConnection:
        if self.execute_error:
            raise sqlite3.OperationalError("private failure")
        return self

    def fetchone(self) -> tuple[int]:
        return self.row

    def close(self) -> None:
        self.closed = True
        if self.fail_close:
            raise sqlite3.OperationalError("private close failure")


def test_fts5_probe_rejects_unexpected_results_and_closes_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrong_result = _FakeFtsConnection(row=(0,))
    monkeypatch.setattr(doctor_module.sqlite3, "connect", lambda _database: wrong_result)

    check = doctor_module._check_fts5()

    assert check.status is DoctorCheckStatus.FAIL
    assert wrong_result.closed is True

    execution_failure = _FakeFtsConnection(execute_error=True)
    monkeypatch.setattr(doctor_module.sqlite3, "connect", lambda _database: execution_failure)
    assert doctor_module._check_fts5().status is DoctorCheckStatus.FAIL
    assert execution_failure.closed is True


def test_fts5_probe_reports_close_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _FakeFtsConnection()
    connection.fail_close = True
    monkeypatch.setattr(doctor_module.sqlite3, "connect", lambda _database: connection)

    check = doctor_module._check_fts5()

    assert check.status is DoctorCheckStatus.FAIL
    assert "finalized" in check.message


def test_repository_path_failure_modes_are_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = run_doctor(
        repository_path=cast(Any, object()),
        installation_key_path=tmp_path / "installation.key",
    )
    assert _check(invalid, DoctorCheckName.REPOSITORY_PATH).status is DoctorCheckStatus.FAIL

    monkeypatch.chdir(tmp_path)
    relative = run_doctor(
        repository_path=Path("relative.sqlite3"),
        installation_key_path=tmp_path / "installation.key",
    )
    assert _check(relative, DoctorCheckName.REPOSITORY_PATH).status is DoctorCheckStatus.PASS

    directory = tmp_path / "repository"
    directory.mkdir()
    real_access = doctor_module._path_access
    monkeypatch.setattr(
        doctor_module,
        "_path_access",
        lambda path, mode: False if path == directory else real_access(path, mode),
    )
    inaccessible = run_doctor(
        repository_path=directory,
        installation_key_path=tmp_path / "installation.key",
    )
    assert _check(inaccessible, DoctorCheckName.REPOSITORY_PATH).status is DoctorCheckStatus.FAIL


def test_missing_repository_requires_an_accessible_directory_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "new" / "runs.sqlite3"
    real_nearest = doctor_module._nearest_existing_ancestor
    monkeypatch.setattr(
        doctor_module,
        "_nearest_existing_ancestor",
        lambda _path: (_ for _ in ()).throw(PermissionError()),
    )
    ancestry_failure = doctor_module._check_repository_path(target)
    assert ancestry_failure.status is DoctorCheckStatus.FAIL
    assert "ancestry" in ancestry_failure.message

    blocker = tmp_path / "blocker"
    blocker.touch()
    monkeypatch.setattr(
        doctor_module,
        "_nearest_existing_ancestor",
        lambda _path: (blocker, blocker.lstat()),
    )
    blocked = doctor_module._check_repository_path(target)
    assert blocked.status is DoctorCheckStatus.FAIL
    assert "non-directory" in blocked.message

    monkeypatch.setattr(doctor_module, "_nearest_existing_ancestor", real_nearest)
    monkeypatch.setattr(doctor_module, "_path_access", lambda _path, _mode: False)
    denied = doctor_module._check_repository_path(target)
    assert denied.status is DoctorCheckStatus.FAIL
    assert "cannot be created" in denied.message


def test_existing_repository_requires_parent_sidecar_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "runs.sqlite3"
    target.touch()
    real_access = doctor_module._path_access

    def deny_parent(path: Path, mode: int) -> bool:
        if path == tmp_path:
            return False
        return real_access(path, mode)

    monkeypatch.setattr(doctor_module, "_path_access", deny_parent)

    check = doctor_module._check_repository_path(target)

    assert check.status is DoctorCheckStatus.FAIL
    assert "sidecar" in check.message


def test_installation_key_failure_modes_are_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_directory = tmp_path / "key-as-directory"
    key_directory.mkdir()
    directory_check = doctor_module._check_installation_key(key_directory, None)
    assert directory_check.status is DoctorCheckStatus.FAIL
    assert "regular file" in directory_check.message

    missing = tmp_path / "missing" / "installation.key"
    real_nearest = doctor_module._nearest_existing_ancestor
    monkeypatch.setattr(
        doctor_module,
        "_nearest_existing_ancestor",
        lambda _path: (_ for _ in ()).throw(PermissionError()),
    )
    ancestry_failure = doctor_module._check_installation_key(missing, None)
    assert ancestry_failure.status is DoctorCheckStatus.FAIL
    assert "ancestry" in ancestry_failure.message

    monkeypatch.setattr(doctor_module, "_nearest_existing_ancestor", real_nearest)
    monkeypatch.setattr(doctor_module, "_path_access", lambda _path, _mode: False)
    denied = doctor_module._check_installation_key(missing, None)
    assert denied.status is DoctorCheckStatus.FAIL
    assert "cannot be created" in denied.message


@pytest.mark.skipif(os.name != "posix", reason="ownership is POSIX-specific")
def test_installation_key_requires_current_owner_and_read_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_path = tmp_path / "installation.key"
    key_path.write_bytes(b"k" * 32)
    key_path.chmod(0o600)
    real_getuid = os.getuid
    monkeypatch.setattr(doctor_module.os, "getuid", lambda: key_path.stat().st_uid + 1)

    wrong_owner = doctor_module._check_installation_key(key_path, None)

    assert wrong_owner.status is DoctorCheckStatus.FAIL
    assert "owned" in wrong_owner.message

    monkeypatch.setattr(doctor_module.os, "getuid", real_getuid)
    monkeypatch.setattr(doctor_module, "_path_access", lambda _path, _mode: False)
    unreadable = doctor_module._check_installation_key(key_path, None)
    assert unreadable.status is DoctorCheckStatus.FAIL
    assert "readable" in unreadable.message


def test_configured_endpoint_rejects_bounded_and_typed_edge_cases(tmp_path: Path) -> None:
    endpoints = (
        "https://example.test:" + "8" * 2_048,
        "https://example.test/\x7f",
        "http://example.test:0/v1",
        cast(Any, b"https://example.test/v1"),
    )

    for endpoint in endpoints:
        report = run_doctor(
            repository_path=tmp_path / "runs.sqlite3",
            installation_key_path=tmp_path / "installation.key",
            endpoint=endpoint,
        )
        assert _check(report, DoctorCheckName.ENDPOINT).status is DoctorCheckStatus.FAIL


def test_run_doctor_can_represent_an_optional_degraded_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        doctor_module,
        "_check_endpoint",
        lambda _endpoint: doctor_module._fail(
            DoctorCheckName.ENDPOINT,
            "Optional endpoint warning.",
            required=False,
        ),
    )

    report = run_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        installation_key_path=tmp_path / "installation.key",
    )

    assert report.status is DoctorReportStatus.DEGRADED
    assert report.ok is True


def test_nearest_existing_ancestor_stops_at_the_filesystem_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always_missing(_path: Path) -> os.stat_result:
        raise FileNotFoundError

    monkeypatch.setattr(Path, "lstat", always_missing)
    with pytest.raises(FileNotFoundError):
        doctor_module._nearest_existing_ancestor(Path("/"))
