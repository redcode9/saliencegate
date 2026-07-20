from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
from tests.capture.store_support import INSTALLATION_KEY, authenticated_intake, register_connection
from tests.cli.test_capture_connect import _command_hook_spec, _configless_bridge_spec

from saliencegate.capture import (
    CaptureConnectionState,
    CaptureProfile,
    CaptureSpool,
    CaptureStore,
    CaptureStoreMode,
    capture_capability_digest,
    capture_profile,
    initialize_capture_store,
    resolve_capture_store_locations,
)
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.commands.capture.common import (
    CaptureCommandIntegrityError,
    CaptureCommandUnavailableError,
    capture_project_digest,
)
from saliencegate.commands.capture.connect import run_connect
from saliencegate.commands.capture.status import (
    CaptureOperationalStatus,
    CaptureStatusDrift,
    render_status_human,
    render_status_json,
    run_status,
)
from saliencegate.commands.doctor import CaptureDoctorState, run_capture_doctor
from saliencegate.integrations.config_files import ConfigSyntax, OwnedConfigSpec
from saliencegate.integrations.registry import ProviderAlias, ProviderInstallationSpec
from saliencegate.security import default_installation_key_path, load_installation_key


def _environment(tmp_path: Path) -> dict[str, str]:
    return {
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "configuration"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }


def _unavailable_spec(_alias: ProviderAlias, _project: Path) -> ProviderInstallationSpec:
    raise CaptureCommandUnavailableError()


def _installation_spec(tmp_path: Path) -> ProviderInstallationSpec:
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    operational = tmp_path / "operational"
    operational.mkdir(mode=0o700, exist_ok=True)
    integration = project / ".synthetic"
    profile = capture_profile(CaptureProfile.CODEX_HOOKS_V1)
    return ProviderInstallationSpec(
        provider_id="codex",
        profile=profile.profile_id,
        host_version=profile.host_version,
        project_root=project,
        config_path=integration / "settings.json",
        bundle_path=integration / "saliencegate.js",
        bootstrap_path=integration / "saliencegate.bootstrap.json",
        receipt_path=operational / "codex.receipt.json",
        journal_path=operational / "codex.journal.json",
        lock_path=operational / "codex.lock",
        launcher_path=operational / "capture-hook",
        capability_digest=capture_capability_digest(profile),
        bundle_bytes=(
            b"export const saliencegateBootstrap = "
            b'new URL("./saliencegate.bootstrap.json", import.meta.url);\n'
        ),
        launcher_bytes=b"#!/bin/sh\nexit 0\n",
        bootstrap_relative_reference="./saliencegate.bootstrap.json",
        config=OwnedConfigSpec(
            syntax=ConfigSyntax.JSON_OBJECT,
            marker="saliencegate-owned:status-test-v1",
            owned_fragment=b'"saliencegate":{"marker":"saliencegate-owned:status-test-v1"}',
        ),
    )


def test_status_surfaces_observation_health_and_storage_without_paths(tmp_path: Path) -> None:
    project = tmp_path / "project-secret"
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
    CaptureSpool.open(locations, INSTALLATION_KEY)
    with CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        register_connection(
            store,
            project_digest=capture_project_digest(project, installation_key=INSTALLATION_KEY),
        )
        store.append(authenticated_intake("session_started", producer_index=1))

    report = run_status(
        provider="codex",
        project=project,
        environ=environment,
        spec_resolver=_unavailable_spec,
    )

    assert len(report.providers) == 1
    item = report.providers[0]
    assert item.provider == "codex"
    assert item.status is CaptureOperationalStatus.ACTIVE_OBSERVED
    assert item.session_count == 1
    assert item.queued_spool_events == item.dropped_spool_events == 0
    assert item.local_bytes > 0
    assert item.oldest_session is not None
    rendered = render_status_human(report)
    assert "quarantined=0" in rendered
    assert f"oldest={item.oldest_session}" in rendered
    assert "bytes=" in rendered
    assert "project-secret" not in rendered
    assert str(locations.database_path) not in rendered
    assert json.loads(render_status_json(report)) == report.model_dump(mode="json")


def test_status_reports_healthy_installation_without_events_as_not_observed(
    tmp_path: Path,
) -> None:
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
    CaptureSpool.open(locations, INSTALLATION_KEY)
    with CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        register_connection(
            store,
            project_digest=capture_project_digest(project, installation_key=INSTALLATION_KEY),
        )

    report = run_status(
        provider="codex",
        project=project,
        environ=environment,
        spec_resolver=_unavailable_spec,
    )

    assert report.providers[0].status is CaptureOperationalStatus.INSTALLED_NOT_OBSERVED
    assert report.providers[0].drift == ()


def test_status_does_not_require_bridge_assets_for_command_hook_installation(
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

    report = run_status(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
    )

    assert report.providers[0].status is CaptureOperationalStatus.INSTALLED_NOT_OBSERVED
    assert report.providers[0].drift == ()
    assert CaptureStatusDrift.BUNDLE not in report.providers[0].drift
    assert CaptureStatusDrift.BOOTSTRAP not in report.providers[0].drift


def test_status_accepts_configless_bridge_and_reports_only_asset_drift(tmp_path: Path) -> None:
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

    healthy = run_status(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
    )
    assert healthy.providers[0].status is CaptureOperationalStatus.INSTALLED_NOT_OBSERVED
    assert healthy.providers[0].drift == ()

    spec.bundle_path.unlink()
    drifted = run_status(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
    )

    assert drifted.providers[0].status is CaptureOperationalStatus.DRIFTED
    assert drifted.providers[0].drift == (CaptureStatusDrift.BUNDLE,)
    assert CaptureStatusDrift.CONFIG not in drifted.providers[0].drift


def test_status_without_capture_is_read_only_and_reports_all_providers(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    environment = _environment(tmp_path)

    report = run_status(project=project, environ=environment)

    assert tuple(item.provider for item in report.providers) == (
        "codex",
        "claude-code",
        "opencode",
        "pi",
    )
    assert all(item.status is CaptureOperationalStatus.NOT_INSTALLED for item in report.providers)
    assert tuple(tmp_path.rglob("*")) == (project,)


def test_status_distinguishes_lifecycle_drift_from_observation_degradation(
    tmp_path: Path,
) -> None:
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
    CaptureSpool.open(locations, INSTALLATION_KEY)
    with CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        register_connection(
            store,
            project_digest=capture_project_digest(project, installation_key=INSTALLATION_KEY),
        )
        store.transition_connection(
            store.list_connections()[0].connection_id,
            expected_state=CaptureConnectionState.ENABLED,
            target_state=CaptureConnectionState.DRAINING,
        )

    report = run_status(provider="codex", project=project, environ=environment)

    assert report.providers[0].status is CaptureOperationalStatus.DRIFTED


def test_status_inventory_is_not_truncated_at_the_session_list_limit(tmp_path: Path) -> None:
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
    CaptureSpool.open(locations, INSTALLATION_KEY)
    with CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        register_connection(
            store,
            project_digest=capture_project_digest(project, installation_key=INSTALLATION_KEY),
        )
        store.append(
            authenticated_intake(
                "session_started",
                session_native=b"inventory-session-1",
                producer_index=1,
            )
        )
        oldest = store.list_sessions()[0].human_id
        for index in range(2, 102):
            store.append(
                authenticated_intake(
                    "session_started",
                    session_native=f"inventory-session-{index}".encode(),
                    producer_index=index,
                )
            )

    report = run_status(provider="codex", project=project, environ=environment)

    item = report.providers[0]
    assert item.session_count == 101
    assert item.oldest_session == oldest


def test_status_detects_deleted_provider_launcher_as_installation_drift(tmp_path: Path) -> None:
    spec = _installation_spec(tmp_path)
    environment = _environment(tmp_path)

    def resolver(alias: ProviderAlias, _project: Path) -> ProviderInstallationSpec:
        if alias is not ProviderAlias.CODEX:
            raise CaptureCommandUnavailableError()
        return spec

    run_connect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=sys.executable,
    )
    spec.launcher_path.unlink()

    report = run_status(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=sys.executable,
    )

    item = report.providers[0]
    assert item.status is CaptureOperationalStatus.DRIFTED
    assert CaptureStatusDrift.LAUNCHER in item.drift

    doctor = run_capture_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        environ=environment,
        capture_project=spec.project_root,
        capture_spec_resolver=resolver,
        capture_executable=sys.executable,
    )
    assert doctor.capture.state is CaptureDoctorState.DEGRADED


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits are unavailable")
def test_status_reports_an_unsafe_operational_lock_as_drift(tmp_path: Path) -> None:
    spec = _installation_spec(tmp_path)
    environment = _environment(tmp_path)

    def resolver(alias: ProviderAlias, _project: Path) -> ProviderInstallationSpec:
        if alias is not ProviderAlias.CODEX:
            raise CaptureCommandUnavailableError()
        return spec

    run_connect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=sys.executable,
    )
    spec.lock_path.chmod(0o644)

    report = run_status(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=sys.executable,
    )

    item = report.providers[0]
    assert item.status is CaptureOperationalStatus.DRIFTED
    assert CaptureStatusDrift.LOCK in item.drift


def test_status_and_doctor_detect_deleted_pinned_capture_executable(tmp_path: Path) -> None:
    spec = _installation_spec(tmp_path)
    environment = _environment(tmp_path)
    executable = tmp_path / "capture-executable"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)

    def resolver(alias: ProviderAlias, _project: Path) -> ProviderInstallationSpec:
        if alias is not ProviderAlias.CODEX:
            raise CaptureCommandUnavailableError()
        return spec

    run_connect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=executable,
    )
    executable.unlink()

    report = run_status(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=executable,
    )
    replacement = tmp_path / "replacement-capture-executable"
    replacement.write_bytes(b"#!/bin/sh\nexit 0\n")
    replacement.chmod(0o700)
    moved_report = run_status(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=replacement,
    )
    doctor = run_capture_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        environ=environment,
        capture_project=spec.project_root,
        capture_spec_resolver=resolver,
        capture_executable=replacement,
    )

    assert report.providers[0].status is CaptureOperationalStatus.DRIFTED
    assert CaptureStatusDrift.LAUNCHER in report.providers[0].drift
    assert CaptureStatusDrift.LAUNCHER in moved_report.providers[0].drift
    assert doctor.capture.state is CaptureDoctorState.DEGRADED


def test_status_detects_installed_provider_when_runtime_database_is_missing(
    tmp_path: Path,
) -> None:
    spec = _installation_spec(tmp_path)
    environment = _environment(tmp_path)

    def resolver(alias: ProviderAlias, _project: Path) -> ProviderInstallationSpec:
        if alias is not ProviderAlias.CODEX:
            raise CaptureCommandUnavailableError()
        return spec

    run_connect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=sys.executable,
    )
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    locations.database_path.unlink()
    shutil.rmtree(locations.spool_directory)

    report = run_status(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=sys.executable,
    )

    item = report.providers[0]
    assert item.status is CaptureOperationalStatus.DRIFTED
    assert CaptureStatusDrift.CONNECTION_MISSING in item.drift
    assert CaptureStatusDrift.SPOOL_MISSING in item.drift


def test_status_authenticates_an_existing_spool_when_runtime_database_is_missing(
    tmp_path: Path,
) -> None:
    spec = _installation_spec(tmp_path)
    environment = _environment(tmp_path)

    def resolver(alias: ProviderAlias, _project: Path) -> ProviderInstallationSpec:
        if alias is not ProviderAlias.CODEX:
            raise CaptureCommandUnavailableError()
        return spec

    run_connect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=sys.executable,
    )
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    installation_key = load_installation_key(default_installation_key_path(environ=environment))
    spool = CaptureSpool.open(locations, installation_key)
    spool.enqueue(
        authenticated_intake(
            "session_started",
            producer_index=9,
            context=CaptureDigestContext(installation_key),
        )
    )
    locations.database_path.unlink()

    report = run_status(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=sys.executable,
    )

    item = report.providers[0]
    assert item.status is CaptureOperationalStatus.DRIFTED
    assert CaptureStatusDrift.CONNECTION_MISSING in item.drift
    assert CaptureStatusDrift.SPOOL_MISSING not in item.drift
    assert item.queued_spool_events == 1
    assert item.local_bytes > 0


def test_status_detects_missing_spool_before_any_session_is_observed(tmp_path: Path) -> None:
    spec = _installation_spec(tmp_path)
    environment = _environment(tmp_path)

    def resolver(_alias: ProviderAlias, _project: Path) -> ProviderInstallationSpec:
        return spec

    run_connect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=sys.executable,
    )
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    shutil.rmtree(locations.spool_directory)

    report = run_status(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=sys.executable,
    )

    assert report.providers[0].status is CaptureOperationalStatus.DRIFTED
    assert CaptureStatusDrift.SPOOL_MISSING in report.providers[0].drift


def test_status_fails_integrity_when_runtime_state_outlives_its_key(tmp_path: Path) -> None:
    spec = _installation_spec(tmp_path)
    environment = _environment(tmp_path)

    def resolver(_alias: ProviderAlias, _project: Path) -> ProviderInstallationSpec:
        return spec

    run_connect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=sys.executable,
    )
    default_installation_key_path(environ=environment).unlink()

    with pytest.raises(CaptureCommandIntegrityError):
        run_status(
            provider="codex",
            project=spec.project_root,
            environ=environment,
            spec_resolver=resolver,
            capture_executable=sys.executable,
        )


def test_status_and_doctor_detect_installed_artifacts_without_runtime_or_key(
    tmp_path: Path,
) -> None:
    spec = _installation_spec(tmp_path)
    environment = _environment(tmp_path)

    def resolver(_alias: ProviderAlias, _project: Path) -> ProviderInstallationSpec:
        return spec

    run_connect(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=sys.executable,
    )
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    locations.database_path.unlink()
    shutil.rmtree(locations.spool_directory)
    default_installation_key_path(environ=environment).unlink()

    report = run_status(
        provider="codex",
        project=spec.project_root,
        environ=environment,
        spec_resolver=resolver,
        capture_executable=sys.executable,
    )
    doctor = run_capture_doctor(
        repository_path=tmp_path / "runs.sqlite3",
        environ=environment,
        capture_project=spec.project_root,
        capture_spec_resolver=resolver,
        capture_executable=sys.executable,
    )

    item = report.providers[0]
    assert item.status is CaptureOperationalStatus.DRIFTED
    assert item.drift == (
        CaptureStatusDrift.CONNECTION_MISSING,
        CaptureStatusDrift.RECEIPT,
        CaptureStatusDrift.SPOOL_MISSING,
    )
    assert doctor.capture.state is CaptureDoctorState.DEGRADED
