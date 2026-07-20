from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import time
from io import BytesIO
from pathlib import Path

import pytest

from saliencegate.capture.capabilities import (
    CaptureProfile,
    capture_capability_digest,
    capture_profile,
)
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.capture.locations import resolve_capture_store_locations
from saliencegate.capture.publication import verify_capture_intake_authentication
from saliencegate.capture.schema import canonical_capture_intake
from saliencegate.capture.spool import CaptureSpool
from saliencegate.capture.store import CaptureSessionState, CaptureStore, CaptureStoreMode
from saliencegate.commands.capture.connect import run_connect
from saliencegate.commands.capture.disconnect import run_disconnect
from saliencegate.domain import canonical_json
from saliencegate.integrations.bootstrap import IntegrationBootstrap, inspect_integration_bootstrap
from saliencegate.integrations.hook import run_capture_hook
from saliencegate.integrations.installation import derive_installation_identity
from saliencegate.integrations.opencode import (
    OPENCODE_HOST_VERSION,
    OPENCODE_PROFILE,
    OpenCodeCaptureAdapter,
    OpenCodeIntegrationError,
    provider_installation_spec,
)
from saliencegate.integrations.registry import ProviderInstallationKind
from saliencegate.security import InstallationKey, load_installation_key

KEY = InstallationKey(b"o" * 32)
CONTEXT = CaptureDigestContext(KEY)
CONNECTION_ID = "sg-" + "1" * 48
SESSION_ID = "synthetic-opencode-session"
BATCH_ID = "2" * 64
ZERO = "0" * 64


def _bootstrap() -> IntegrationBootstrap:
    return IntegrationBootstrap(
        profile=CaptureProfile.OPENCODE_PLUGIN_V1,
        connection_id=CONNECTION_ID,
        launcher_path=Path("/private/tmp/saliencegate-opencode-hook"),
        capability_digest=capture_capability_digest(
            capture_profile(CaptureProfile.OPENCODE_PLUGIN_V1)
        ),
        bundle_digest="3" * 64,
        receipt_mac="4" * 64,
    )


def _adapter() -> OpenCodeCaptureAdapter:
    return OpenCodeCaptureAdapter(
        connection_id=CONNECTION_ID,
        bootstrap=_bootstrap(),
        project_root=Path("/synthetic/opencode/project"),
    )


def _record(kind: str, **values: object) -> dict[str, object]:
    return {"kind": kind, "session_id": SESSION_ID, **values}


def _batch(
    events: list[dict[str, object]],
    *,
    session_id: str = SESSION_ID,
    batch_id: str = BATCH_ID,
    chunk_index: int = 0,
    chunk_count: int = 1,
    bootstrap: dict[str, object] | None = None,
) -> bytes:
    sidecar = _bootstrap().model_dump(mode="json", warnings="error")
    return canonical_json(
        {
            "schema_version": "capture-batch/v1",
            "bootstrap": sidecar if bootstrap is None else bootstrap,
            "batch_id": batch_id,
            "session_id": session_id,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "events": events,
        }
    )


def _adapt(events: list[dict[str, object]]):
    return _adapter().adapt_bytes(_batch(events), context=CONTEXT)


def test_capabilities_and_configless_project_installation_are_exact(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    environment = {
        "HOME": str(tmp_path / "home"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }

    declaration = _adapter().capabilities()
    spec = provider_installation_spec(project, environ=environment)

    assert declaration.profile_id is OPENCODE_PROFILE
    assert declaration.host_version == OPENCODE_HOST_VERSION == "1.18.3"
    assert declaration.capability_digest == capture_capability_digest(
        capture_profile(OPENCODE_PROFILE)
    )
    assert spec.installation_kind is ProviderInstallationKind.BRIDGE
    assert spec.config_path is spec.config is None
    assert spec.bundle_path == project / ".opencode" / "plugins" / "saliencegate.js"
    assert spec.bootstrap_path == (
        project / ".opencode" / "plugins" / "saliencegate.bootstrap.json"
    )
    assert spec.project_local_paths == (spec.bundle_path, spec.bootstrap_path)
    assert spec.bundle_bytes is not None
    assert spec.bundle_bytes.startswith(
        b"export const saliencegateBootstrap = "
        b'new URL("./saliencegate.bootstrap.json", import.meta.url);\n'
    )
    assert spec.receipt_path.parent == spec.launcher_path.parent
    assert spec.receipt_path.parent.is_relative_to(
        Path(environment["XDG_STATE_HOME"]) / "saliencegate" / "integrations"
    )
    assert not spec.receipt_path.is_relative_to(project)


def test_each_chunk_opens_idempotently_and_maps_closed_reduced_records() -> None:
    intakes = _adapt(
        [
            _record(
                "tool_started",
                event_id="event-start",
                call_id="call-1",
                tool="read",
                input={"path": "/provider/secret-sentinel"},
                identity_authority="exact",
            ),
            _record(
                "tool_finished",
                event_id="event-finish",
                call_id="call-1",
                outcome="failed",
            ),
            _record("turn_finished", event_id="event-idle"),
            _record("controller_failed", event_id="event-error"),
            _record("coverage_boundary", event_id="event-compact"),
            _record("session_finished", event_id="event-delete"),
        ]
    )

    assert tuple(item.kind for item in intakes) == (
        "session_started",
        "action_started",
        "action_finished",
        "turn_finished",
        "controller_failed",
        "turn_finished",
        "session_finished",
    )
    assert intakes[0].producer_event_digest == _adapt([])[0].producer_event_digest
    action = intakes[1]
    assert action.kind == "action_started"
    assert action.tool_class == "file_read"
    assert action.identity_authority == "exact"
    outcome = intakes[2]
    assert outcome.kind == "action_finished"
    assert outcome.outcome_status == "failed"
    assert outcome.outcome_authority == "producer_claimed_structured"
    assert outcome.error_code == "tool_error"
    assert intakes[5].capture_disposition == "coverage_boundary"
    for intake in intakes:
        assert verify_capture_intake_authentication(intake, context=CONTEXT) == intake
        assert intake.occurred_at is None
        assert intake.timestamp_authority == "unavailable"
        assert intake.producer_sequence is None
        assert intake.sequence_authority == "unavailable"

    persisted = b"\n".join(canonical_capture_intake(item) for item in intakes)
    assert b"provider/secret-sentinel" not in persisted
    assert b"call-1" not in persisted
    assert SESSION_ID.encode() not in persisted


def test_session_start_digest_is_stable_across_chunks_and_batches() -> None:
    first = _adapter().adapt_bytes(
        _batch([], batch_id="a" * 64, chunk_index=0, chunk_count=2),
        context=CONTEXT,
    )[0]
    second = _adapter().adapt_bytes(
        _batch([], batch_id="a" * 64, chunk_index=1, chunk_count=2),
        context=CONTEXT,
    )[0]
    later_batch = _adapter().adapt_bytes(
        _batch([], batch_id="b" * 64),
        context=CONTEXT,
    )[0]

    assert first.producer_event_digest == second.producer_event_digest
    assert first.producer_event_digest == later_batch.producer_event_digest
    assert canonical_capture_intake(first) == canonical_capture_intake(second)
    assert canonical_capture_intake(first) == canonical_capture_intake(later_batch)


def test_exact_and_unavailable_action_authority_cannot_collapse() -> None:
    exact = _adapt(
        [
            _record(
                "tool_started",
                call_id="call-a",
                tool="bash",
                input={"command": "printf secret-sentinel"},
                identity_authority="exact",
            )
        ]
    )[1]
    unavailable_a = _adapt(
        [
            _record(
                "tool_started",
                call_id="call-a",
                tool="bash",
                identity_authority="unavailable",
            )
        ]
    )[1]
    unavailable_b = _adapt(
        [
            _record(
                "tool_started",
                call_id="call-b",
                tool="bash",
                identity_authority="unavailable",
            )
        ]
    )[1]

    assert exact.kind == unavailable_a.kind == unavailable_b.kind == "action_started"
    assert exact.identity_authority == "exact"
    assert unavailable_a.identity_authority == unavailable_b.identity_authority == "unavailable"
    assert len({exact.action_digest, unavailable_a.action_digest, unavailable_b.action_digest}) == 3


@pytest.mark.parametrize(
    "record",
    (
        _record("tool_started", call_id="", tool="read", identity_authority="unavailable"),
        _record(
            "tool_started",
            call_id="call",
            tool="read",
            input={},
            identity_authority="unavailable",
        ),
        _record("tool_finished", call_id="call", outcome="unknown"),
        _record("coverage_degraded", reason="unknown"),
        _record("oversize", reason="wrong"),
        _record("turn_finished", extra="forbidden"),
        _record("unknown"),
    ),
)
def test_reduced_records_are_closed_and_fail_content_free(record: dict[str, object]) -> None:
    with pytest.raises(OpenCodeIntegrationError, match="OpenCode capture integration is invalid"):
        _adapt([record])


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.update(schema_version="capture-batch/v2"),
        lambda value: value.update(batch_id="not-a-digest"),
        lambda value: value.update(session_id="other-session"),
        lambda value: value.update(chunk_index=True),
        lambda value: value.update(chunk_index=1),
        lambda value: value.update(extra="forbidden"),
        lambda value: value["bootstrap"].update(receipt_mac="5" * 64),
    ),
)
def test_batch_and_bootstrap_binding_are_closed(mutate) -> None:
    document = json.loads(_batch([_record("turn_finished")]))
    mutate(document)

    with pytest.raises(OpenCodeIntegrationError):
        _adapter().adapt_bytes(canonical_json(document), context=CONTEXT)


def test_session_identifier_cap_matches_the_node_bridge_contract() -> None:
    accepted = "s" * (256 * 1_024)

    intakes = _adapter().adapt_bytes(
        _batch([], session_id=accepted),
        context=CONTEXT,
    )

    assert tuple(item.kind for item in intakes) == ("session_started",)
    with pytest.raises(OpenCodeIntegrationError):
        _adapter().adapt_bytes(
            _batch([], session_id=accepted + "s"),
            context=CONTEXT,
        )


@pytest.mark.parametrize(
    ("field", "maximum", "record"),
    (
        (
            "call_id",
            16 * 1_024,
            _record(
                "tool_started",
                call_id="placeholder",
                tool="read",
                input={},
                identity_authority="exact",
            ),
        ),
        (
            "event_id",
            16 * 1_024,
            _record("turn_finished", event_id="placeholder"),
        ),
        (
            "tool",
            1 * 1_024,
            _record(
                "tool_started",
                call_id="call",
                tool="placeholder",
                input={},
                identity_authority="exact",
            ),
        ),
    ),
)
def test_reduced_identifier_caps_match_the_node_bridge_contract(
    field: str,
    maximum: int,
    record: dict[str, object],
) -> None:
    accepted = dict(record)
    accepted[field] = "x" * maximum
    rejected = dict(record)
    rejected[field] = "x" * (maximum + 1)

    assert len(_adapter().adapt_bytes(_batch([accepted]), context=CONTEXT)) >= 2
    with pytest.raises(OpenCodeIntegrationError):
        _adapter().adapt_bytes(_batch([rejected]), context=CONTEXT)


def test_transport_descriptor_is_receiver_keyed_and_contains_no_native_values() -> None:
    source = _batch(
        [_record("turn_finished")],
        chunk_index=2,
        chunk_count=4,
    )
    descriptor = _adapter().transport_chunk(source, context=CONTEXT)

    assert descriptor.connection_id == CONNECTION_ID
    assert descriptor.session_id == CONTEXT.session_id(SESSION_ID.encode())
    assert descriptor.batch_ref == CONTEXT.transport_batch_ref(BATCH_ID.encode())
    assert descriptor.chunk_index == 2
    assert descriptor.chunk_count == 4
    assert descriptor.chunk_digest == CONTEXT.transport_chunk_digest(source)
    assert SESSION_ID not in repr(descriptor)
    assert BATCH_ID not in repr(descriptor)


def test_exact_source_bytes_are_keyed_without_cross_runtime_reserialization() -> None:
    document = json.loads(_batch([_record("turn_finished")]))
    source = json.dumps(document, indent=2).encode()

    assert tuple(item.kind for item in _adapter().adapt_bytes(source, context=CONTEXT)) == (
        "session_started",
        "turn_finished",
    )
    descriptor = _adapter().transport_chunk(source, context=CONTEXT)
    canonical_descriptor = _adapter().transport_chunk(
        canonical_json(document),
        context=CONTEXT,
    )
    assert descriptor.chunk_digest == CONTEXT.transport_chunk_digest(source)
    assert descriptor.chunk_digest != canonical_descriptor.chunk_digest


@pytest.mark.parametrize("node_number", (b"1e-7", b"0.000001", b"100000000000000000000"))
def test_node_json_number_spellings_cross_the_python_boundary(node_number: bytes) -> None:
    # These spellings are frozen JSON.stringify outputs that differ from Python's
    # encoder for at least one corresponding numeric value.
    source = _batch(
        [
            _record(
                "tool_started",
                call_id="numeric-call",
                tool="synthetic",
                input={"number": "NODE_NUMBER_SENTINEL"},
                identity_authority="exact",
            )
        ]
    ).replace(b'"NODE_NUMBER_SENTINEL"', node_number)

    intakes = _adapter().adapt_bytes(source, context=CONTEXT)
    descriptor = _adapter().transport_chunk(source, context=CONTEXT)

    assert tuple(item.kind for item in intakes) == ("session_started", "action_started")
    assert descriptor.chunk_digest == CONTEXT.transport_chunk_digest(source)


def test_degraded_and_oversize_records_mark_coverage_without_native_content() -> None:
    intakes = _adapt(
        [
            _record("coverage_degraded", reason="invalid_transition"),
            _record("coverage_degraded", reason="missing_field"),
            _record("coverage_degraded", reason="overflow"),
            _record("coverage_degraded", reason="transport_gap"),
            _record("oversize", reason="event_limit"),
        ]
    )

    assert tuple(item.capture_disposition for item in intakes) == (
        "captured",
        "degraded",
        "degraded",
        "degraded",
        "degraded",
        "degraded",
    )
    assert tuple(item.error_code for item in intakes[1:]) == (
        "invalid_transition",
        "invalid_transition",
        "overflow",
        "gap_detected",
        "overflow",
    )


def test_transport_gap_intake_is_session_stable_across_failed_batches() -> None:
    first = _adapter().adapt_bytes(
        _batch(
            [_record("coverage_degraded", reason="transport_gap")],
            batch_id="a" * 64,
        ),
        context=CONTEXT,
    )[1]
    second = _adapter().adapt_bytes(
        _batch(
            [_record("coverage_degraded", reason="transport_gap")],
            batch_id="b" * 64,
        ),
        context=CONTEXT,
    )[1]

    assert canonical_capture_intake(first) == canonical_capture_intake(second)


def test_default_hook_rejects_a_plausible_batch_when_bootstrap_is_not_installed() -> None:
    source = _batch([_record("turn_finished")])
    assert (
        run_capture_hook(
            (
                "--profile",
                CaptureProfile.OPENCODE_PLUGIN_V1.value,
                "--connection",
                CONNECTION_ID,
            ),
            BytesIO(source),
            environ={"HOME": "/nonexistent", "XDG_STATE_HOME": "/nonexistent"},
            capture_executable=sys.executable,
        )
        == 0
    )


def test_bundle_bytes_are_stable_package_input(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    environment = {"HOME": str(tmp_path / "home"), "XDG_STATE_HOME": str(tmp_path / "state")}

    first = provider_installation_spec(project, environ=environment)
    second = provider_installation_spec(project, environ=environment)

    assert first.bundle_bytes == second.bundle_bytes
    assert first.bundle_digest == hashlib.sha256(first.bundle_bytes).hexdigest()
    assert b"sourceMappingURL" not in first.bundle_bytes
    assert b"session.messages" not in first.bundle_bytes
    assert b"session.get(" not in first.bundle_bytes
    assert os.fspath(project).encode() not in first.bundle_bytes


@pytest.mark.skipif(os.name != "posix", reason="native Windows lifecycle is covered by R01")
def test_real_sqlite_writer_contention_spools_bridge_controls_before_launcher_timeout(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    environment = {
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "configuration"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }
    capture_executable = Path(sys.executable).resolve(strict=True)
    run_connect(
        provider="opencode",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    )
    key = load_installation_key(environ=environment)
    spec = provider_installation_spec(project, environ=environment)
    identity = derive_installation_identity(spec, key)
    bootstrap = inspect_integration_bootstrap(spec.bootstrap_path)
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    source = _batch(
        [_record("turn_finished", event_id="contended-turn")],
        bootstrap=bootstrap.model_dump(mode="json", warnings="error"),
    )

    blocker = sqlite3.connect(locations.database_path, isolation_level=None, timeout=0.1)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        result = run_capture_hook(
            (
                "--profile",
                OPENCODE_PROFILE.value,
                "--connection",
                identity.connection_id,
            ),
            BytesIO(source),
            environ=environment,
            capture_executable=capture_executable,
        )
        elapsed = time.monotonic() - started
    finally:
        blocker.rollback()
        blocker.close()

    spool = CaptureSpool.open(locations, key)
    health = spool.health()
    session_id = CaptureDigestContext(key).session_id(SESSION_ID.encode())
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        drained = spool.drain(store)
        snapshot = store.snapshot_session(identity.connection_id, session_id)

    assert result == 0
    assert elapsed < 1.5
    assert health.queued_events == 2
    assert drained.remaining_events == 0
    assert snapshot.event_count == 2
    assert snapshot.coverage_degraded is True


@pytest.mark.skipif(os.name != "posix", reason="native Windows lifecycle is covered by R01")
def test_held_spool_lock_fails_open_before_launcher_timeout(tmp_path: Path) -> None:
    import fcntl

    project = tmp_path / "project"
    project.mkdir()
    environment = {
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "configuration"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }
    capture_executable = Path(sys.executable).resolve(strict=True)
    run_connect(
        provider="opencode",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    )
    key = load_installation_key(environ=environment)
    spec = provider_installation_spec(project, environ=environment)
    identity = derive_installation_identity(spec, key)
    bootstrap = inspect_integration_bootstrap(spec.bootstrap_path)
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    spool = CaptureSpool.open(locations, key)
    spool.health()
    source = _batch(
        [_record("turn_finished", event_id="locked-spool-turn")],
        bootstrap=bootstrap.model_dump(mode="json", warnings="error"),
    )

    descriptor = os.open(locations.spool_directory / ".capture-spool-lock", os.O_RDONLY)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        started = time.monotonic()
        result = run_capture_hook(
            (
                "--profile",
                OPENCODE_PROFILE.value,
                "--connection",
                identity.connection_id,
            ),
            BytesIO(source),
            environ=environment,
            capture_executable=capture_executable,
        )
        elapsed = time.monotonic() - started
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    assert result == 0
    assert elapsed < 1.5


@pytest.mark.skipif(os.name != "posix", reason="native Windows lifecycle is covered by R01")
def test_default_install_capture_transport_and_disconnect_round_trip(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    environment = {
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "configuration"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }
    capture_executable = Path(sys.executable).resolve(strict=True)

    connected = run_connect(
        provider="opencode",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    )
    key = load_installation_key(environ=environment)
    spec = provider_installation_spec(project, environ=environment)
    identity = derive_installation_identity(spec, key)
    installed_bootstrap = inspect_integration_bootstrap(spec.bootstrap_path)

    assert connected.capture_enabled is True
    assert connected.project_local_files == 2
    assert spec.bundle_path.read_bytes() == spec.bundle_bytes
    first_source = _batch(
        [
            _record(
                "tool_started",
                event_id="runtime-event-start",
                call_id="runtime-call",
                tool="bash",
                input={"command": "printf provider-native-secret-sentinel"},
                identity_authority="exact",
            ),
        ],
        chunk_index=0,
        chunk_count=2,
        bootstrap=installed_bootstrap.model_dump(mode="json", warnings="error"),
    )
    second_source = _batch(
        [
            _record(
                "tool_finished",
                event_id="runtime-event-finish",
                call_id="runtime-call",
                outcome="succeeded",
            ),
            _record("session_finished", event_id="runtime-event-delete"),
        ],
        chunk_index=1,
        chunk_count=2,
        bootstrap=installed_bootstrap.model_dump(mode="json", warnings="error"),
    )
    assert (
        run_capture_hook(
            (
                "--profile",
                OPENCODE_PROFILE.value,
                "--connection",
                identity.connection_id,
            ),
            BytesIO(first_source),
            environ=environment,
            capture_executable=capture_executable,
        )
        == 0
    )

    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    session_id = CaptureDigestContext(key).session_id(SESSION_ID.encode())
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        incomplete = store.snapshot_session(identity.connection_id, session_id)
    assert incomplete.state is CaptureSessionState.OPEN
    assert incomplete.transport_receipt_count == 1
    assert incomplete.incomplete_transport_batch_count == 1
    assert incomplete.coverage_degraded is True

    assert (
        run_capture_hook(
            (
                "--profile",
                OPENCODE_PROFILE.value,
                "--connection",
                identity.connection_id,
            ),
            BytesIO(second_source),
            environ=environment,
            capture_executable=capture_executable,
        )
        == 0
    )
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        snapshot = store.snapshot_session(identity.connection_id, session_id)
    assert snapshot.state is CaptureSessionState.CLOSED
    assert snapshot.event_count == 4
    assert snapshot.transport_receipt_count == 2
    assert snapshot.incomplete_transport_batch_count == 0
    assert snapshot.coverage_degraded is False
    assert b"provider-native-secret-sentinel" not in locations.database_path.read_bytes()

    disconnected = run_disconnect(
        provider="opencode",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    )
    assert disconnected.disposition == "uninstalled"
    assert not spec.bundle_path.exists()
    assert not spec.bootstrap_path.exists()
