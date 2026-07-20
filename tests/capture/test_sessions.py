from __future__ import annotations

from pathlib import Path
from threading import Event, Thread

import pytest
from pydantic import ValidationError
from tests.capture.store_support import (
    CONNECTION_ID,
    INSTALLATION_KEY,
    OTHER_CONNECTION_ID,
    WRONG_INSTALLATION_KEY,
    authenticated_intake,
    initialized_store,
    register_connection,
)

from saliencegate.capture.capabilities import CompatibilityStatus
from saliencegate.capture.health import CaptureHealthCode
from saliencegate.capture.migrations import initialize_capture_store
from saliencegate.capture.sessions import (
    CaptureSessionSnapshot,
    CaptureSessionSnapshotError,
    CaptureSnapshotEvent,
    CaptureSnapshotHealth,
    verify_capture_session_snapshot,
)
from saliencegate.capture.store import (
    CaptureAdmissionSource,
    CaptureConnectionState,
    CaptureSessionState,
    CaptureStore,
    CaptureStoreClosedError,
    CaptureStoreError,
    CaptureStoreIntegrityError,
    CaptureStoreMode,
    CaptureStoreStateError,
)
from saliencegate.domain import canonical_json


def _started_snapshot(path: Path) -> tuple[CaptureSessionSnapshot, str]:
    with initialized_store(path) as store:
        register_connection(store)
        intake = authenticated_intake("session_started")
        store.append(intake, source=CaptureAdmissionSource.SPOOL_DRAIN)
        return store.snapshot_session(CONNECTION_ID, intake.session_id), intake.session_id


def test_snapshot_contains_only_verified_store_evidence_and_is_key_bound(
    tmp_path: Path,
) -> None:
    snapshot, session_id = _started_snapshot(tmp_path / "snapshot.sqlite3")

    assert type(snapshot) is CaptureSessionSnapshot
    assert snapshot.schema_version == "capture-session-snapshot/v1"
    assert snapshot.connection_id == CONNECTION_ID
    assert snapshot.session_id == session_id
    assert snapshot.compatibility_status is CompatibilityStatus.VERIFIED
    assert snapshot.connection_state is CaptureConnectionState.ENABLED
    assert snapshot.state is CaptureSessionState.OPEN
    assert snapshot.event_count == 1
    assert snapshot.coverage_degraded is False
    assert snapshot.unattributed_drop is False
    assert snapshot.closed_at is None
    assert snapshot.health == ()
    assert snapshot.spool_observation == "not_observed_by_store_snapshot"
    assert snapshot.spool_boundary_digest is None
    assert snapshot.at_rest_integrity == "hmac_sha256_local_mutation_detection"
    assert snapshot.rollback_detection == "none"
    assert len(snapshot.snapshot_digest) == 64
    assert not hasattr(snapshot, "head_tag")
    assert not hasattr(snapshot, "head_event_tag")

    item = snapshot.events[0]
    assert type(item) is CaptureSnapshotEvent
    assert item.receipt_ordinal == 1
    assert item.admission_source is CaptureAdmissionSource.SPOOL_DRAIN
    assert item.event.intake.kind == "session_started"
    assert item.event.intake.session_id == session_id
    assert item.admitted_at.tzinfo is not None

    verified = verify_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )
    assert verified == snapshot
    assert verified is not snapshot
    assert verified.events[0] is not snapshot.events[0]
    assert verified.events[0].event is not snapshot.events[0].event
    assert "<redacted>" in repr(snapshot)
    for sensitive in (snapshot.connection_id, snapshot.session_id, snapshot.snapshot_digest):
        assert sensitive not in repr(snapshot)


def test_snapshot_is_byte_stable_until_the_authenticated_mvcc_view_changes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stable.sqlite3"
    with initialized_store(path) as store:
        register_connection(store)
        start = authenticated_intake("session_started", producer_index=1)
        store.append(start)

        first = store.snapshot_session(CONNECTION_ID, start.session_id)
        second = store.snapshot_session(CONNECTION_ID, start.session_id)
        assert canonical_json(first) == canonical_json(second)

        store.append(authenticated_intake("turn_finished", producer_index=2))
        changed = store.snapshot_session(CONNECTION_ID, start.session_id)
        assert changed.event_count == 2
        assert changed.snapshot_digest != first.snapshot_digest
        assert canonical_json(changed) != canonical_json(first)


def test_snapshot_uses_receipt_order_when_the_wall_clock_moves_backward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "backward-clock.sqlite3"
    with initialized_store(path) as store:
        register_connection(store)
        start = authenticated_intake("session_started", producer_index=1)
        store.append(start)

        # Establish an authenticated health row before the clock rolls back.
        store.append(authenticated_intake("turn_finished", producer_index=1))
        monkeypatch.setattr(
            "saliencegate.capture.store._now",
            lambda: "2000-01-01T00:00:00.000000Z",
        )
        store.append(authenticated_intake("action_started", producer_index=1))

        verification = store.verify_session(CONNECTION_ID, start.session_id)
        snapshot = store.snapshot_session(CONNECTION_ID, start.session_id)

    assert verification.event_count == snapshot.event_count == 1
    assert snapshot.updated_at < snapshot.opened_at
    assert len(snapshot.health) == 1
    assert snapshot.health[0].updated_at < snapshot.health[0].created_at
    assert verify_capture_session_snapshot(snapshot, installation_key=INSTALLATION_KEY) == snapshot


def test_snapshot_preserves_closed_lifecycle_and_authenticated_health(
    tmp_path: Path,
) -> None:
    path = tmp_path / "closed-and-health.sqlite3"
    with initialized_store(path) as store:
        register_connection(store)
        closed_start = authenticated_intake(
            "session_started",
            session_native=b"closed-session",
            producer_index=10,
        )
        store.append(closed_start)
        store.append(
            authenticated_intake(
                "session_finished",
                session_native=b"closed-session",
                producer_index=11,
            )
        )
        closed = store.snapshot_session(CONNECTION_ID, closed_start.session_id)

        original = authenticated_intake(
            "session_started",
            session_native=b"collision-original",
            producer_index=20,
        )
        collision = authenticated_intake(
            "session_started",
            session_native=b"collision-incoming",
            producer_index=20,
        )
        store.append(original)
        store.append(collision)
        quarantined = store.snapshot_session(CONNECTION_ID, original.session_id)
        empty_quarantined = store.snapshot_session(CONNECTION_ID, collision.session_id)

    assert closed.state is CaptureSessionState.CLOSED
    assert closed.closed_at == closed.updated_at
    assert closed.events[-1].event.intake.kind == "session_finished"

    for snapshot in (quarantined, empty_quarantined):
        assert snapshot.state is CaptureSessionState.QUARANTINED
        assert snapshot.coverage_degraded is True
        assert snapshot.closed_at is None
        assert len(snapshot.health) == 1
        marker = snapshot.health[0]
        assert type(marker) is CaptureSnapshotHealth
        assert marker.code is CaptureHealthCode.PRODUCER_COLLISION
        assert marker.count == 1
        assert marker.lower_bound == 0
        assert marker.created_at <= marker.updated_at
    assert quarantined.event_count == 1
    assert empty_quarantined.event_count == 0


def test_snapshot_verification_fails_closed_for_wrong_key_or_tampering(
    tmp_path: Path,
) -> None:
    snapshot, _ = _started_snapshot(tmp_path / "verification.sqlite3")
    marker = "provider-native-secret-marker"
    tampered = snapshot.model_copy(update={"project_digest": "f" * 64})
    malformed = snapshot.model_copy(update={"event_count": 2})

    for value, key in (
        (snapshot, WRONG_INSTALLATION_KEY),
        (tampered, INSTALLATION_KEY),
        (malformed, INSTALLATION_KEY),
        ({"snapshot_digest": marker}, INSTALLATION_KEY),
    ):
        with pytest.raises(CaptureSessionSnapshotError) as raised:
            verify_capture_session_snapshot(value, installation_key=key)
        assert str(raised.value) == "capture session snapshot is invalid"
        assert marker not in str(raised.value)
        assert marker not in repr(raised.value)
        assert raised.value.__cause__ is None

    with pytest.raises(CaptureSessionSnapshotError):
        verify_capture_session_snapshot(  # type: ignore[arg-type]
            snapshot,
            installation_key=b"k" * 32,
        )


def test_snapshot_models_reject_inconsistent_or_deleting_views(tmp_path: Path) -> None:
    snapshot, _ = _started_snapshot(tmp_path / "model-validation.sqlite3")

    invalid_values = (
        {"connection_state": CaptureConnectionState.DELETING},
        {"state": CaptureSessionState.DELETING},
        {"state": CaptureSessionState.CLOSED, "closed_at": None},
        {"coverage_degraded": False, "unattributed_drop": True},
        {"event_count": 0},
    )
    for changes in invalid_values:
        body = snapshot.model_dump(mode="python")
        body.update(changes)
        with pytest.raises(ValidationError):
            CaptureSessionSnapshot.model_validate(body)


def test_store_snapshot_rejects_invalid_targets_deleting_and_closed_store(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.sqlite3"
    initialize_capture_store(path)
    store = CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    )
    register_connection(store)
    intake = authenticated_intake("session_started")
    store.append(intake)

    with pytest.raises(CaptureStoreError):
        store.snapshot_session(1, intake.session_id)  # type: ignore[arg-type]
    with pytest.raises(CaptureStoreError):
        store.snapshot_session(CONNECTION_ID, None)  # type: ignore[arg-type]
    with pytest.raises(CaptureStoreStateError):
        store.snapshot_session(OTHER_CONNECTION_ID, intake.session_id)
    with pytest.raises(CaptureStoreStateError):
        store.snapshot_session(CONNECTION_ID, "f" * 64)

    store.transition_connection(
        CONNECTION_ID,
        expected_state=CaptureConnectionState.ENABLED,
        target_state=CaptureConnectionState.DRAINING,
    )
    store.transition_connection(
        CONNECTION_ID,
        expected_state=CaptureConnectionState.DRAINING,
        target_state=CaptureConnectionState.DISABLED,
    )
    store.transition_connection(
        CONNECTION_ID,
        expected_state=CaptureConnectionState.DISABLED,
        target_state=CaptureConnectionState.DELETING,
    )
    with pytest.raises(CaptureStoreStateError):
        store.snapshot_session(CONNECTION_ID, intake.session_id)

    store.close()
    with pytest.raises(CaptureStoreClosedError):
        store.snapshot_session(CONNECTION_ID, intake.session_id)


def test_snapshot_is_one_coherent_old_revision_when_a_peer_appends_mid_read(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mvcc.sqlite3"
    initialize_capture_store(path)
    snapshot_ready = Event()
    release_snapshot = Event()

    def pause_after_verification(stage: str) -> None:
        if stage == "snapshot_after_verification":
            snapshot_ready.set()
            assert release_snapshot.wait(timeout=10)

    with (
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            busy_timeout_ms=2_000,
            mode=CaptureStoreMode.MAINTENANCE,
            _fault_injector=pause_after_verification,
        ) as reader,
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            busy_timeout_ms=2_000,
            mode=CaptureStoreMode.MAINTENANCE,
        ) as writer,
    ):
        register_connection(reader)
        start = authenticated_intake("session_started", producer_index=1)
        reader.append(start)
        outcomes: list[CaptureSessionSnapshot | BaseException] = []

        def read_snapshot() -> None:
            try:
                outcomes.append(reader.snapshot_session(CONNECTION_ID, start.session_id))
            except BaseException as error:  # pragma: no cover - asserted below
                outcomes.append(error)

        thread = Thread(target=read_snapshot)
        thread.start()
        assert snapshot_ready.wait(timeout=10)
        writer.append(authenticated_intake("turn_finished", producer_index=2))
        release_snapshot.set()
        thread.join(timeout=10)
        assert not thread.is_alive()

        assert len(outcomes) == 1
        old = outcomes[0]
        assert type(old) is CaptureSessionSnapshot
        assert old.event_count == 1
        assert tuple(item.event.intake.kind for item in old.events) == ("session_started",)
        assert (
            verify_capture_session_snapshot(
                old,
                installation_key=INSTALLATION_KEY,
            )
            == old
        )
        assert writer.snapshot_session(CONNECTION_ID, start.session_id).event_count == 2


def test_store_snapshot_maps_internal_failures_to_a_content_free_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "content-free.sqlite3"
    marker = "provider-native-secret-marker"
    initialize_capture_store(path)

    def fail_snapshot(stage: str) -> None:
        if stage == "snapshot_after_verification":
            raise RuntimeError(marker)

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
        _fault_injector=fail_snapshot,
    ) as store:
        register_connection(store)
        start = authenticated_intake("session_started")
        store.append(start)
        with pytest.raises(CaptureStoreIntegrityError) as raised:
            store.snapshot_session(CONNECTION_ID, start.session_id)

    assert str(raised.value) == "capture store integrity failed"
    assert marker not in str(raised.value)
    assert marker not in repr(raised.value)
    assert raised.value.__cause__ is None
