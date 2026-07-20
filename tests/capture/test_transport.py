from __future__ import annotations

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from tests.capture.store_support import (
    CONNECTION_ID,
    INSTALLATION_KEY,
    PROJECT_DIGEST,
    authenticated_intake,
    capture_context,
)

import saliencegate.capture.migrations as migration_module
from saliencegate.capture.capabilities import (
    CaptureProfile,
    capture_capability_digest,
    capture_profile,
)
from saliencegate.capture.delete import delete_capture_session
from saliencegate.capture.health import CaptureHealthCode
from saliencegate.capture.locations import CaptureStoreLocations
from saliencegate.capture.migrations import (
    CaptureMigrationIntegrityError,
    discover_capture_migrations,
    initialize_capture_store,
)
from saliencegate.capture.normalization import normalize_capture_session_snapshot
from saliencegate.capture.report import (
    CaptureReportHeadline,
    CaptureReportLimit,
    build_capture_session_report,
)
from saliencegate.capture.spool import CaptureSpool
from saliencegate.capture.store import (
    CaptureAdmissionSource,
    CaptureConnectionState,
    CaptureSessionState,
    CaptureStore,
    CaptureStoreIntegrityError,
    CaptureStoreMode,
    CaptureStoreStateError,
    _session_material,
)
from saliencegate.capture.transport import (
    MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION,
    CaptureTransportChunk,
    CaptureTransportDisposition,
    CaptureTransportError,
    validate_capture_transport_chunk,
)

PROFILE = CaptureProfile.OPENCODE_PLUGIN_V1
HOST_VERSION = "1.18.3"
CAPABILITY_DIGEST = capture_capability_digest(capture_profile(PROFILE))
SESSION_NATIVE = b"synthetic-opencode-window"


def _locations(state_directory: Path) -> CaptureStoreLocations:
    return CaptureStoreLocations(
        platform="windows" if os.name == "nt" else "posix",
        state_directory=state_directory,
        database_path=state_directory / "capture.sqlite3",
        spool_directory=state_directory / "capture-spool",
    )


def _register(store: CaptureStore) -> None:
    registration = store.register_connection(
        connection_id=CONNECTION_ID,
        project_digest=PROJECT_DIGEST,
        profile_id=PROFILE,
        capability_manifest_digest=CAPABILITY_DIGEST,
        host_version=HOST_VERSION,
    )
    assert registration.state is CaptureConnectionState.PENDING
    store.transition_connection(
        CONNECTION_ID,
        expected_state=CaptureConnectionState.PENDING,
        target_state=CaptureConnectionState.ENABLED,
    )


def _intake(
    kind: str,
    producer_index: int,
    *,
    session_native: bytes = SESSION_NATIVE,
):
    return authenticated_intake(
        kind,
        session_native=session_native,
        producer_index=producer_index,
        changes={
            "adapter_profile": PROFILE.value,
            "capability_manifest_digest": CAPABILITY_DIGEST,
        },
    )


def _chunk(
    *,
    batch_native: bytes = b"synthetic-bridge-batch",
    chunk_native: bytes = b'{"synthetic":"bridge-chunk"}',
    chunk_index: int = 0,
    chunk_count: int = 1,
    session_native: bytes = SESSION_NATIVE,
) -> CaptureTransportChunk:
    context = capture_context()
    return CaptureTransportChunk(
        connection_id=CONNECTION_ID,
        session_id=context.session_id(session_native),
        batch_ref=context.transport_batch_ref(batch_native),
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        chunk_digest=context.transport_chunk_digest(chunk_native),
    )


def _initialize(path: Path) -> None:
    initialize_capture_store(path)
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        _register(store)


def test_transport_descriptor_is_strict_defensively_copied_and_redacted() -> None:
    descriptor = _chunk()

    validated = validate_capture_transport_chunk(descriptor)

    assert validated == descriptor
    assert validated is not descriptor
    assert repr(descriptor) == "CaptureTransportChunk(<redacted>)"
    invalid = descriptor.model_dump(mode="python", warnings="error")
    invalid["chunk_index"] = invalid["chunk_count"]
    invalid["native_sentinel"] = "must-not-render"
    with pytest.raises(CaptureTransportError) as captured:
        validate_capture_transport_chunk(invalid)
    assert str(captured.value) == "capture transport is invalid"
    assert "must-not-render" not in repr(captured.value)


def test_transport_chunk_admission_is_atomic_and_exact_replay_is_a_noop(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.sqlite3"
    _initialize(path)
    descriptor = _chunk()
    intakes = (_intake("session_started", 1), _intake("action_started", 2))

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        admitted = store.append_transport_chunk(descriptor, intakes)
        before = store.snapshot_session(CONNECTION_ID, descriptor.session_id)
        replayed = store.append_transport_chunk(descriptor, intakes)
        after = store.snapshot_session(CONNECTION_ID, descriptor.session_id)

    assert admitted.disposition is CaptureTransportDisposition.ADMITTED
    assert admitted.transport_ordinal == 1
    assert admitted.intake_count == 2
    assert admitted.incomplete_batch_count == 0
    assert replayed.disposition is CaptureTransportDisposition.REPLAYED
    assert replayed.transport_ordinal == admitted.transport_ordinal
    assert replayed.receipt_tag == admitted.receipt_tag
    assert replayed.incomplete_batch_count == 0
    assert before == after
    assert before.event_count == 2
    assert before.transport_receipt_count == 1
    assert before.incomplete_transport_batch_count == 0


def test_transport_profiles_reject_generic_direct_admission_and_degrade_spool_fallback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.sqlite3"
    _initialize(path)
    start = _intake("session_started", 1)

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        with pytest.raises(CaptureStoreStateError):
            store.append(start)
        store.append(start, source=CaptureAdmissionSource.SPOOL_DRAIN)
        snapshot = store.snapshot_session(CONNECTION_ID, start.session_id)

    assert snapshot.event_count == 1
    assert snapshot.transport_receipt_count == 0
    assert snapshot.coverage_degraded is True


def test_exact_transport_replay_refuses_a_missing_event_row(tmp_path: Path) -> None:
    path = tmp_path / "capture.sqlite3"
    _initialize(path)
    descriptor = _chunk()
    intakes = (_intake("session_started", 1), _intake("action_started", 2))
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        store.append_transport_chunk(descriptor, intakes)

    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM capture_events WHERE receipt_ordinal = 2")
        connection.commit()

    with (
        pytest.raises(CaptureStoreIntegrityError),
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            mode=CaptureStoreMode.HOOK,
        ) as store,
    ):
        store.append_transport_chunk(descriptor, intakes)


def test_transport_receipt_rejects_a_valid_old_event_ledger_splice(tmp_path: Path) -> None:
    path = tmp_path / "capture.sqlite3"
    _initialize(path)
    first = _chunk(chunk_index=0, chunk_count=2)
    second = _chunk(chunk_index=1, chunk_count=2, chunk_native=b"second")
    start = _intake("session_started", 1)
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        store.append_transport_chunk(first, (start,))

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        old_session = connection.execute(
            "SELECT * FROM capture_sessions WHERE connection_id = ? AND session_id = ?",
            (CONNECTION_ID, first.session_id),
        ).fetchone()
        old_head = connection.execute(
            "SELECT * FROM capture_heads WHERE connection_id = ? AND session_id = ?",
            (CONNECTION_ID, first.session_id),
        ).fetchone()
    assert old_session is not None
    assert old_head is not None

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        store.append_transport_chunk(second, (start, _intake("action_started", 2)))

    with sqlite3.connect(path) as connection:
        connection.execute(
            "DELETE FROM capture_events WHERE connection_id = ? AND session_id = ? "
            "AND receipt_ordinal > 1",
            (CONNECTION_ID, first.session_id),
        )
        connection.execute(
            """
            UPDATE capture_sessions
            SET event_count = ?, updated_at = ?, row_tag = ?
            WHERE connection_id = ? AND session_id = ?
            """,
            (
                old_session["event_count"],
                old_session["updated_at"],
                old_session["row_tag"],
                CONNECTION_ID,
                first.session_id,
            ),
        )
        connection.execute(
            """
            UPDATE capture_heads
            SET receipt_count = ?, head_event_tag = ?, head_tag = ?
            WHERE connection_id = ? AND session_id = ?
            """,
            (
                old_head["receipt_count"],
                old_head["head_event_tag"],
                old_head["head_tag"],
                CONNECTION_ID,
                first.session_id,
            ),
        )
        connection.commit()

    with (
        pytest.raises(CaptureStoreIntegrityError),
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            mode=CaptureStoreMode.MAINTENANCE,
        ) as store,
    ):
        store.snapshot_session(CONNECTION_ID, first.session_id)


def test_transport_verification_rejects_a_valid_old_receipt_prefix_with_a_direct_event_tail(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.sqlite3"
    _initialize(path)
    first = _chunk(batch_native=b"first-receipt-prefix")
    second = _chunk(
        batch_native=b"second-receipt-prefix",
        chunk_native=b"second-receipt-prefix-chunk",
    )
    start = _intake("session_started", 1)
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        store.append_transport_chunk(first, (start,))

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        old_transport_head = connection.execute(
            "SELECT * FROM capture_transport_heads WHERE connection_id = ? AND session_id = ?",
            (CONNECTION_ID, first.session_id),
        ).fetchone()
    assert old_transport_head is not None

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        store.append_transport_chunk(second, (start, _intake("action_started", 2)))

    with sqlite3.connect(path) as connection:
        connection.execute(
            "DELETE FROM capture_transport_receipts "
            "WHERE connection_id = ? AND session_id = ? AND transport_ordinal > 1",
            (CONNECTION_ID, first.session_id),
        )
        connection.execute(
            """
            UPDATE capture_transport_heads
            SET receipt_count = ?, head_receipt_tag = ?, head_tag = ?
            WHERE connection_id = ? AND session_id = ?
            """,
            (
                old_transport_head["receipt_count"],
                old_transport_head["head_receipt_tag"],
                old_transport_head["head_tag"],
                CONNECTION_ID,
                first.session_id,
            ),
        )
        connection.commit()

    with (
        pytest.raises(CaptureStoreIntegrityError),
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            mode=CaptureStoreMode.MAINTENANCE,
        ) as store,
    ):
        store.snapshot_session(CONNECTION_ID, first.session_id)


def test_session_commitment_rejects_rollback_of_an_empty_transport_receipt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.sqlite3"
    _initialize(path)
    first = _chunk(batch_native=b"empty-prefix-first")
    second = _chunk(
        batch_native=b"empty-prefix-second",
        chunk_native=b"empty-prefix-second-chunk",
    )
    start = _intake("session_started", 1)
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        store.append_transport_chunk(first, (start,))

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        old_transport_head = connection.execute(
            "SELECT * FROM capture_transport_heads WHERE connection_id = ? AND session_id = ?",
            (CONNECTION_ID, first.session_id),
        ).fetchone()
    assert old_transport_head is not None

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        store.append_transport_chunk(second, (start,))

    with sqlite3.connect(path) as connection:
        connection.execute(
            "DELETE FROM capture_transport_receipts "
            "WHERE connection_id = ? AND session_id = ? AND transport_ordinal > 1",
            (CONNECTION_ID, first.session_id),
        )
        connection.execute(
            """
            UPDATE capture_transport_heads
            SET receipt_count = ?, head_receipt_tag = ?, head_tag = ?
            WHERE connection_id = ? AND session_id = ?
            """,
            (
                old_transport_head["receipt_count"],
                old_transport_head["head_receipt_tag"],
                old_transport_head["head_tag"],
                CONNECTION_ID,
                first.session_id,
            ),
        )
        connection.commit()

    with (
        pytest.raises(CaptureStoreIntegrityError),
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            mode=CaptureStoreMode.MAINTENANCE,
        ) as store,
    ):
        store.snapshot_session(CONNECTION_ID, first.session_id)


@pytest.mark.parametrize(
    ("chunk_index", "chunk_count"),
    ((1, 2), (0, 3), (0, 2)),
    ids=("missing-first", "missing-middle-and-tail", "missing-tail"),
)
def test_any_incomplete_chunk_index_set_degrades_the_verified_snapshot(
    tmp_path: Path,
    chunk_index: int,
    chunk_count: int,
) -> None:
    path = tmp_path / "capture.sqlite3"
    _initialize(path)
    descriptor = _chunk(chunk_index=chunk_index, chunk_count=chunk_count)

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        receipt = store.append_transport_chunk(
            descriptor,
            (_intake("session_started", 1),),
        )
        snapshot = store.snapshot_session(CONNECTION_ID, descriptor.session_id)

    assert receipt.incomplete_batch_count == 1
    assert snapshot.transport_receipt_count == 1
    assert snapshot.incomplete_transport_batch_count == 1
    assert snapshot.coverage_degraded is True
    assert snapshot.health == ()


def test_delayed_middle_chunk_clears_only_the_dynamic_transport_gap(tmp_path: Path) -> None:
    path = tmp_path / "capture.sqlite3"
    _initialize(path)
    first = _chunk(chunk_index=0, chunk_count=3)
    last = _chunk(chunk_index=2, chunk_count=3, chunk_native=b"last")
    middle = _chunk(chunk_index=1, chunk_count=3, chunk_native=b"middle")
    start = _intake("session_started", 1)

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        store.append_transport_chunk(first, (start,))
        store.append_transport_chunk(last, (start, _intake("action_started", 3)))
        incomplete = store.snapshot_session(CONNECTION_ID, first.session_id)
        completed_receipt = store.append_transport_chunk(
            middle,
            (start, _intake("action_started", 2)),
        )
        complete = store.snapshot_session(CONNECTION_ID, first.session_id)

    assert incomplete.incomplete_transport_batch_count == 1
    assert incomplete.coverage_degraded is True
    assert completed_receipt.incomplete_batch_count == 0
    assert complete.transport_receipt_count == 3
    assert complete.incomplete_transport_batch_count == 0
    assert complete.coverage_degraded is False


def test_conflicting_chunk_reuse_quarantines_without_persisting_native_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.sqlite3"
    _initialize(path)
    batch_native = b"native-transport-batch-sentinel"
    original_native = b"native-transport-chunk-sentinel-original"
    conflicting_native = b"native-transport-chunk-sentinel-conflict"
    original = _chunk(
        batch_native=batch_native,
        chunk_native=original_native,
    )
    conflicting = _chunk(
        batch_native=batch_native,
        chunk_native=conflicting_native,
    )
    start = _intake("session_started", 1)

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        store.append_transport_chunk(original, (start,))
        collision = store.append_transport_chunk(conflicting, (start,))
        first_snapshot = store.snapshot_session(CONNECTION_ID, original.session_id)
        replayed_collision = store.append_transport_chunk(conflicting, (start,))
        snapshot = store.snapshot_session(CONNECTION_ID, original.session_id)

    assert collision.disposition is CaptureTransportDisposition.QUARANTINED
    assert replayed_collision.disposition is CaptureTransportDisposition.QUARANTINED
    assert snapshot.state is CaptureSessionState.QUARANTINED
    assert snapshot.coverage_degraded is True
    assert tuple(item.code for item in snapshot.health) == (CaptureHealthCode.PRODUCER_COLLISION,)
    assert snapshot.health[0].count == 1
    assert snapshot.updated_at == first_snapshot.updated_at
    assert snapshot.snapshot_digest == first_snapshot.snapshot_digest
    persisted = b"".join(
        candidate.read_bytes()
        for candidate in (
            path,
            path.with_name(path.name + "-wal"),
            path.with_name(path.name + "-shm"),
        )
        if candidate.exists()
    )
    for native_secret in (
        batch_native,
        original_native,
        conflicting_native,
        SESSION_NATIVE,
    ):
        assert native_secret not in persisted


def test_batch_reuse_across_sessions_quarantines_both_affected_sessions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.sqlite3"
    _initialize(path)
    other_session = b"synthetic-other-opencode-window"
    first = _chunk(chunk_index=0, chunk_count=2)
    conflicting = _chunk(
        chunk_index=1,
        chunk_count=2,
        chunk_native=b"other-session-chunk",
        session_native=other_session,
    )

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        store.append_transport_chunk(first, (_intake("session_started", 1),))
        collision = store.append_transport_chunk(
            conflicting,
            (
                _intake(
                    "session_started",
                    1,
                    session_native=other_session,
                ),
            ),
        )
        first_snapshot = store.snapshot_session(CONNECTION_ID, first.session_id)
        other_snapshot = store.snapshot_session(CONNECTION_ID, conflicting.session_id)

    assert collision.disposition is CaptureTransportDisposition.QUARANTINED
    assert first_snapshot.state is CaptureSessionState.QUARANTINED
    assert other_snapshot.state is CaptureSessionState.QUARANTINED
    assert {item.code for item in first_snapshot.health} == {CaptureHealthCode.PRODUCER_COLLISION}
    assert {item.code for item in other_snapshot.health} == {CaptureHealthCode.PRODUCER_COLLISION}


def test_intake_collision_inside_a_transport_savepoint_quarantines_both_sessions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.sqlite3"
    _initialize(path)
    other_session = b"synthetic-other-opencode-window"
    first = _chunk(batch_native=b"first-independent-batch")
    conflicting = _chunk(
        batch_native=b"second-independent-batch",
        chunk_native=b"second-independent-chunk",
        session_native=other_session,
    )

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        store.append_transport_chunk(first, (_intake("session_started", 1),))
        collision = store.append_transport_chunk(
            conflicting,
            (_intake("session_started", 1, session_native=other_session),),
        )
        first_snapshot = store.snapshot_session(CONNECTION_ID, first.session_id)
        other_snapshot = store.snapshot_session(CONNECTION_ID, conflicting.session_id)

    assert collision.disposition is CaptureTransportDisposition.QUARANTINED
    assert first_snapshot.state is CaptureSessionState.QUARANTINED
    assert other_snapshot.state is CaptureSessionState.QUARANTINED
    assert {item.code for item in first_snapshot.health} == {CaptureHealthCode.PRODUCER_COLLISION}
    assert {item.code for item in other_snapshot.health} == {CaptureHealthCode.PRODUCER_COLLISION}


@pytest.mark.parametrize(
    "stage",
    (
        "transport_after_intake_admission",
        "transport_after_receipt_insert",
        "transport_after_head_write",
        "transport_before_commit",
    ),
)
def test_transport_fault_before_commit_rolls_back_session_events_receipt_and_head(
    tmp_path: Path,
    stage: str,
) -> None:
    path = tmp_path / "capture.sqlite3"
    _initialize(path)

    def fail_at(observed: str) -> None:
        if observed == stage:
            raise RuntimeError("synthetic transport crash")

    with (
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            mode=CaptureStoreMode.HOOK,
            _fault_injector=fail_at,
        ) as store,
        pytest.raises(RuntimeError, match="synthetic transport crash"),
    ):
        store.append_transport_chunk(
            _chunk(),
            (_intake("session_started", 1), _intake("action_started", 2)),
        )

    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM capture_sessions").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM capture_events").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM capture_transport_receipts").fetchone() == (
            0,
        )
        assert connection.execute("SELECT count(*) FROM capture_transport_heads").fetchone() == (0,)


def test_transport_fault_after_commit_retries_as_an_exact_noop(tmp_path: Path) -> None:
    path = tmp_path / "capture.sqlite3"
    _initialize(path)
    descriptor = _chunk()
    intakes = (_intake("session_started", 1), _intake("action_started", 2))

    def fail_after_commit(stage: str) -> None:
        if stage == "transport_after_commit":
            raise RuntimeError("synthetic ambiguous transport outcome")

    with (
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            mode=CaptureStoreMode.HOOK,
            _fault_injector=fail_after_commit,
        ) as store,
        pytest.raises(RuntimeError, match="ambiguous transport outcome"),
    ):
        store.append_transport_chunk(descriptor, intakes)

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        replay = store.append_transport_chunk(descriptor, intakes)
        snapshot = store.snapshot_session(CONNECTION_ID, descriptor.session_id)

    assert replay.disposition is CaptureTransportDisposition.REPLAYED
    assert snapshot.event_count == 2
    assert snapshot.transport_receipt_count == 1


def test_transport_receipt_saturation_cannot_admit_intakes_without_a_receipt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.sqlite3"
    _initialize(path)
    start = _intake("session_started", 1)

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        for index in range(MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION):
            store.append_transport_chunk(
                _chunk(
                    batch_native=f"saturation-batch-{index}".encode(),
                    chunk_native=f"saturation-chunk-{index}".encode(),
                ),
                (start,) if index == 0 else (),
            )
        before = store.snapshot_session(CONNECTION_ID, start.session_id)
        overflow = store.append_transport_chunk(
            _chunk(
                batch_native=b"saturation-overflow-batch",
                chunk_native=b"saturation-overflow-chunk",
            ),
            (start, _intake("action_started", 2)),
        )
        replayed_overflow = store.append_transport_chunk(
            _chunk(
                batch_native=b"saturation-overflow-batch",
                chunk_native=b"saturation-overflow-chunk",
            ),
            (start, _intake("action_started", 2)),
        )
        after = store.snapshot_session(CONNECTION_ID, start.session_id)

    assert before.event_count == 1
    assert before.transport_receipt_count == MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION
    assert overflow.disposition is CaptureTransportDisposition.OVERFLOW
    assert replayed_overflow.disposition is CaptureTransportDisposition.OVERFLOW
    assert overflow.event_count == 1
    assert after.event_count == before.event_count
    assert after.transport_receipt_count == before.transport_receipt_count
    assert tuple(item.event.intake.kind for item in after.events) == ("session_started",)
    assert after.state is CaptureSessionState.QUARANTINED
    assert {item.code for item in after.health} == {CaptureHealthCode.SESSION_OVERFLOW}
    assert after.health[0].count == 1


def test_legacy_bridge_session_lazily_starts_a_verified_transport_chain(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.sqlite3"
    _initialize(path)
    start = _intake("session_started", 1)
    descriptor = _chunk()
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        store.append_transport_chunk(descriptor, (start,))
        session = store._session_row(CONNECTION_ID, descriptor.session_id)
        assert session is not None
        legacy_material = dict(session)
        legacy_material["transport_required"] = 0
        legacy_material["transport_head_tag"] = None
        legacy_tag = store._integrity.tag("session", _session_material(legacy_material))
        store._connection.execute("DELETE FROM capture_transport_receipts")
        store._connection.execute("DELETE FROM capture_transport_heads")
        store._connection.execute(
            """
            UPDATE capture_sessions
            SET transport_required = 0, transport_head_tag = NULL, row_tag = ?
            WHERE connection_id = ? AND session_id = ?
            """,
            (legacy_tag, CONNECTION_ID, descriptor.session_id),
        )
        store._connection.commit()

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        legacy = store.snapshot_session(CONNECTION_ID, descriptor.session_id)
        admitted = store.append_transport_chunk(descriptor, (start,))
        upgraded = store.snapshot_session(CONNECTION_ID, descriptor.session_id)

    assert legacy.transport_receipt_count == 0
    assert legacy.incomplete_transport_batch_count == 0
    assert legacy.coverage_degraded is True
    assert admitted.disposition is CaptureTransportDisposition.ADMITTED
    assert upgraded.event_count == 1
    assert upgraded.transport_receipt_count == 1


def test_populated_version_one_bridge_session_migrates_as_authenticated_legacy(
    tmp_path: Path,
) -> None:
    current_path = tmp_path / "current.sqlite3"
    legacy_path = tmp_path / "legacy.sqlite3"
    _initialize(current_path)
    descriptor = _chunk()
    start = _intake("session_started", 1)
    with CaptureStore.open(
        current_path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        store.append_transport_chunk(descriptor, (start,))
        session = store._session_row(CONNECTION_ID, descriptor.session_id)
        assert session is not None
        legacy_material = dict(session)
        legacy_material["transport_required"] = 0
        legacy_material["transport_head_tag"] = None
        legacy_tag = store._integrity.tag("session", _session_material(legacy_material))
        store._connection.execute("DELETE FROM capture_transport_receipts")
        store._connection.execute("DELETE FROM capture_transport_heads")
        store._connection.execute(
            """
            UPDATE capture_sessions
            SET transport_required = 0, transport_head_tag = NULL, row_tag = ?
            WHERE connection_id = ? AND session_id = ?
            """,
            (legacy_tag, CONNECTION_ID, descriptor.session_id),
        )
        store._connection.commit()

    first = discover_capture_migrations()[0]
    table_columns = {
        "connections": (
            "connection_id",
            "project_digest",
            "profile_id",
            "capability_manifest_digest",
            "host_version",
            "compatibility_status",
            "state",
            "created_at",
            "updated_at",
            "row_tag",
        ),
        "capture_sessions": (
            "connection_id",
            "session_id",
            "human_id",
            "state",
            "event_count",
            "coverage_degraded",
            "unattributed_drop",
            "health_marker_count",
            "health_set_digest",
            "opened_at",
            "updated_at",
            "closed_at",
            "row_tag",
        ),
        "capture_heads": (
            "connection_id",
            "session_id",
            "receipt_count",
            "head_event_tag",
            "head_tag",
        ),
        "capture_events": (
            "connection_id",
            "session_id",
            "receipt_ordinal",
            "producer_event_digest",
            "event_kind",
            "event_json",
            "previous_event_tag",
            "event_tag",
            "admission_source",
            "admitted_at",
        ),
    }
    with (
        sqlite3.connect(current_path) as current,
        sqlite3.connect(legacy_path, isolation_level=None) as legacy,
    ):
        for statement in migration_module._parse_statements(first.sql):
            legacy.execute(statement)
        legacy.execute(
            "INSERT INTO schema_migrations(version, name, checksum) VALUES (?, ?, ?)",
            (first.version, first.name, first.checksum),
        )
        legacy.execute("PRAGMA user_version = 1")
        for table, columns in table_columns.items():
            names = ", ".join(columns)
            values = current.execute(f"SELECT {names} FROM {table}").fetchall()
            placeholders = ", ".join("?" for _column in columns)
            legacy.executemany(
                f"INSERT INTO {table}({names}) VALUES ({placeholders})",
                values,
            )
    legacy_path.chmod(0o600)

    migrated = initialize_capture_store(legacy_path)
    with CaptureStore.open(
        legacy_path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        snapshot = store.snapshot_session(CONNECTION_ID, descriptor.session_id)

    assert migrated.applied_versions == (2,)
    assert snapshot.event_count == 1
    assert snapshot.transport_receipt_count == 0
    assert snapshot.coverage_degraded is True


def test_two_local_stores_handle_concurrent_exact_chunk_retry(tmp_path: Path) -> None:
    path = tmp_path / "capture.sqlite3"
    _initialize(path)
    descriptor = _chunk()
    intakes = (_intake("session_started", 1), _intake("action_started", 2))

    stores = tuple(
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            busy_timeout_ms=5_000,
            mode=CaptureStoreMode.HOOK,
        )
        for _index in range(2)
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            dispositions = tuple(
                executor.map(
                    lambda store: (
                        store.append_transport_chunk(
                            descriptor,
                            intakes,
                        ).disposition
                    ),
                    stores,
                )
            )
    finally:
        for store in stores:
            store.close()

    assert sorted(dispositions) == sorted(
        (CaptureTransportDisposition.ADMITTED, CaptureTransportDisposition.REPLAYED)
    )
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        snapshot = store.snapshot_session(CONNECTION_ID, descriptor.session_id)
    assert snapshot.event_count == 2
    assert snapshot.transport_receipt_count == 1


@pytest.mark.parametrize(
    "statement",
    (
        "DELETE FROM capture_transport_receipts",
        "DELETE FROM capture_transport_heads",
        "UPDATE capture_transport_receipts SET chunk_digest = '0' || substr(chunk_digest, 2)",
    ),
)
def test_transport_receipt_and_head_tampering_fails_closed(
    tmp_path: Path,
    statement: str,
) -> None:
    path = tmp_path / "capture.sqlite3"
    _initialize(path)
    descriptor = _chunk()
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        store.append_transport_chunk(descriptor, (_intake("session_started", 1),))

    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(statement)
        connection.commit()

    with pytest.raises((CaptureMigrationIntegrityError, CaptureStoreIntegrityError)):
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            mode=CaptureStoreMode.MAINTENANCE,
        )


def test_paired_transport_receipt_and_head_deletion_is_bound_by_the_session_tag(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.sqlite3"
    _initialize(path)
    descriptor = _chunk()
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        store.append_transport_chunk(descriptor, (_intake("session_started", 1),))

    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM capture_transport_receipts")
        connection.execute("DELETE FROM capture_transport_heads")
        connection.commit()

    with (
        pytest.raises(CaptureStoreIntegrityError),
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            mode=CaptureStoreMode.MAINTENANCE,
        ) as store,
    ):
        store.snapshot_session(CONNECTION_ID, descriptor.session_id)


def test_verified_session_deletion_cascades_transport_receipts_and_head(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.sqlite3"
    _initialize(path)
    descriptor = _chunk()
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        store.append_transport_chunk(descriptor, (_intake("session_started", 1),))
        human_id = store.snapshot_session(CONNECTION_ID, descriptor.session_id).human_id

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        delete_capture_session(store, human_id, drain=lambda: None)
        assert (
            store._connection.execute("SELECT count(*) FROM capture_transport_receipts").fetchone()[
                0
            ]
            == 0
        )
        assert (
            store._connection.execute("SELECT count(*) FROM capture_transport_heads").fetchone()[0]
            == 0
        )


def test_incomplete_transport_batch_is_a_gap_in_the_verified_report(tmp_path: Path) -> None:
    locations = _locations(tmp_path)
    CaptureSpool.open(locations, INSTALLATION_KEY)
    _initialize(locations.database_path)
    descriptor = _chunk(chunk_index=0, chunk_count=2)

    with CaptureStore.open(
        locations.database_path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        store.append_transport_chunk(
            descriptor,
            (_intake("session_started", 1), _intake("action_started", 2)),
        )
        snapshot = store.snapshot_session(CONNECTION_ID, descriptor.session_id)
    normalization = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )
    report = build_capture_session_report(
        snapshot,
        normalization,
        installation_key=INSTALLATION_KEY,
        spool=CaptureSpool.open(locations, INSTALLATION_KEY),
    )

    assert report.headline is CaptureReportHeadline.INSUFFICIENT_EVIDENCE
    assert report.coverage.gap_count == 1
    assert CaptureReportLimit.GAP_DETECTED in report.coverage.limits
    assert report.coverage.coverage_degraded is True
