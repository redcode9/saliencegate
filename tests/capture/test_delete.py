from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread

import pytest
from tests.capture.store_support import (
    CONNECTION_ID,
    INSTALLATION_KEY,
    OTHER_CONNECTION_ID,
    OTHER_PROJECT_DIGEST,
    PROJECT_DIGEST,
    authenticated_intake,
    register_connection,
)

from saliencegate.capture.delete import (
    CaptureDeleteDisposition,
    CaptureProjectDeleteReceipt,
    CaptureSessionDeleteReceipt,
    delete_capture_project,
    delete_capture_session,
)
from saliencegate.capture.feedback import CaptureFeedbackLabel
from saliencegate.capture.health import CaptureHealthCode
from saliencegate.capture.migrations import initialize_capture_store
from saliencegate.capture.store import (
    CaptureConnectionState,
    CaptureSessionState,
    CaptureStore,
    CaptureStoreIntegrityError,
    CaptureStoreMode,
    CaptureStoreStateError,
)


def _open(
    path: Path,
    *,
    mode: CaptureStoreMode = CaptureStoreMode.MAINTENANCE,
    fault: Callable[[str], None] | None = None,
) -> CaptureStore:
    return CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=mode,
        _fault_injector=fault,
    )


def _session(store: CaptureStore, *, native: bytes = b"delete-session", index: int = 1) -> str:
    started = store.append(
        authenticated_intake(
            "session_started",
            session_native=native,
            producer_index=index,
        )
    )
    store.append(
        authenticated_intake(
            "action_started",
            session_native=native,
            producer_index=index + 1,
        )
    )
    store.append(
        authenticated_intake(
            "session_finished",
            session_native=native,
            producer_index=index + 2,
        )
    )
    return next(
        item.human_id for item in store.list_sessions() if item.session_id == started.session_id
    )


def _disable(store: CaptureStore, connection_id: str) -> None:
    store.transition_connection(
        connection_id,
        expected_state=CaptureConnectionState.ENABLED,
        target_state=CaptureConnectionState.DRAINING,
    )
    store.transition_connection(
        connection_id,
        expected_state=CaptureConnectionState.DRAINING,
        target_state=CaptureConnectionState.DISABLED,
    )


def _count(store: CaptureStore, table: str) -> int:
    row = store._connection.execute(f"SELECT count(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def _add_authenticated_auxiliary_rows(store: CaptureStore, human_id: str) -> None:
    session = store.session_by_human_id(human_id)
    store.mark_session_health(
        session.connection_id,
        session.session_id,
        CaptureHealthCode.COVERAGE_DEGRADED,
    )
    store.record_feedback(
        human_id,
        CaptureFeedbackLabel.MEMORY_NEEDED,
        project_digest=session.project_digest,
    )


def test_single_delete_drains_marks_deletes_tombstones_and_rejects_late_events(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.sqlite3"
    initialize_capture_store(path)
    with _open(path) as store:
        register_connection(store)
        human_id = _session(store)
        _add_authenticated_auxiliary_rows(store, human_id)
        assert _count(store, "capture_health") == 1
        assert _count(store, "feedback_labels") == 2
        calls: list[str] = []

        def drain() -> None:
            session = store.session_by_human_id(human_id)
            assert session.state is CaptureSessionState.CLOSED
            assert session.closed_at is not None
            calls.append("drained")

        receipt = delete_capture_session(store, human_id, drain=drain)

        assert type(receipt) is CaptureSessionDeleteReceipt
        assert receipt.disposition is CaptureDeleteDisposition.DELETED
        assert receipt.human_id == human_id
        assert receipt.secure_delete is True
        assert receipt.wal_checkpointed is True
        assert calls == ["drained"]
        assert store.list_sessions() == ()
        assert _count(store, "capture_sessions") == 0
        assert _count(store, "capture_events") == 0
        assert _count(store, "capture_heads") == 0
        assert _count(store, "capture_health") == 0
        assert _count(store, "feedback_labels") == 0
        assert _count(store, "deleted_sessions") == 1
        secure_delete = store._connection.execute("PRAGMA secure_delete").fetchone()
        assert secure_delete is not None
        assert tuple(secure_delete) == (1,)

        with pytest.raises(CaptureStoreStateError):
            store.append(
                authenticated_intake(
                    "session_started",
                    session_native=b"delete-session",
                    producer_index=99,
                )
            )

        repeated = delete_capture_session(
            store,
            human_id,
            drain=lambda: pytest.fail("durably deleted session must not drain again"),
        )
        assert repeated.disposition is CaptureDeleteDisposition.ALREADY_DELETED
        assert _count(store, "deleted_sessions") == 1
        assert repr(repeated) == "CaptureSessionDeleteReceipt(<redacted>)"
        assert human_id not in repr(repeated)


def test_single_delete_retry_authenticates_the_retained_tombstone(tmp_path: Path) -> None:
    path = tmp_path / "capture.sqlite3"
    initialize_capture_store(path)
    with _open(path) as store:
        register_connection(store)
        human_id = _session(store)
        delete_capture_session(store, human_id, drain=lambda: None)
        store._connection.execute(
            "UPDATE deleted_sessions SET deleted_at = '2000-01-01T00:00:00+00:00'"
        )

        with pytest.raises(CaptureStoreIntegrityError):
            delete_capture_session(
                store,
                human_id,
                drain=lambda: pytest.fail("unverified tombstone must not reach drain"),
            )


@pytest.mark.parametrize(
    "fault_stage",
    (
        "delete_after_mark_commit",
        "delete_after_tombstone_write",
        "delete_before_purge_commit",
        "delete_after_purge_commit",
    ),
)
def test_single_delete_recovers_idempotently_from_each_durable_fault_state(
    tmp_path: Path,
    fault_stage: str,
) -> None:
    path = tmp_path / f"{fault_stage}.sqlite3"
    initialize_capture_store(path)
    with _open(path) as store:
        register_connection(store)
        human_id = _session(store)

    def fault(stage: str) -> None:
        if stage == fault_stage:
            raise RuntimeError("provider-native-crash-secret")

    with _open(path, fault=fault) as store, pytest.raises(RuntimeError):
        delete_capture_session(store, human_id, drain=lambda: None)

    with _open(path) as store:
        remaining = store.list_sessions()
        if remaining:
            assert remaining[0].state is CaptureSessionState.DELETING
            with pytest.raises(CaptureStoreStateError):
                store.append(
                    authenticated_intake(
                        "session_finished",
                        session_native=b"delete-session",
                        producer_index=50,
                    )
                )
        receipt = delete_capture_session(
            store,
            human_id,
            drain=lambda: pytest.fail("durably marked session must not drain again"),
        )
        assert receipt.disposition in {
            CaptureDeleteDisposition.DELETED,
            CaptureDeleteDisposition.ALREADY_DELETED,
        }
        assert store.list_sessions() == ()
        assert _count(store, "deleted_sessions") == 1


@pytest.mark.parametrize("attempt", ("identical_replay", "digest_collision"))
def test_durable_delete_mark_blocks_replay_and_collision_before_phase_two(
    tmp_path: Path,
    attempt: str,
) -> None:
    path = tmp_path / f"delete-race-{attempt}.sqlite3"
    initialize_capture_store(path)
    original = authenticated_intake(
        "session_started",
        session_native=b"delete-race-session",
        producer_index=1,
    )
    with _open(path) as store:
        register_connection(store)
        store.append(original)
        human_id = store.list_sessions()[0].human_id

    candidate = original
    if attempt == "digest_collision":
        candidate = authenticated_intake(
            "turn_finished",
            session_native=b"delete-race-colliding-session",
            producer_index=2,
            changes={"producer_event_digest": original.producer_event_digest},
        )

    delete_marked = Event()
    admission_attempted = Event()
    delete_receipts: list[CaptureSessionDeleteReceipt] = []
    delete_errors: list[BaseException] = []

    def pause_after_delete_mark(stage: str) -> None:
        if stage == "delete_after_mark_commit":
            delete_marked.set()
            if not admission_attempted.wait(timeout=5):
                raise RuntimeError("timed out waiting for the concurrent admission attempt")

    with _open(path, fault=pause_after_delete_mark) as deleting_store:

        def delete_in_background() -> None:
            try:
                delete_receipts.append(
                    delete_capture_session(deleting_store, human_id, drain=lambda: None)
                )
            except BaseException as error:
                delete_errors.append(error)

        worker = Thread(target=delete_in_background)
        worker.start()
        admission_errors: list[BaseException] = []
        observed_state: CaptureSessionState | None = None
        observed_health_count = -1
        observed_event_count = -1
        try:
            assert delete_marked.wait(timeout=5)
            with _open(path, mode=CaptureStoreMode.HOOK) as hook_store:
                try:
                    hook_store.append(candidate)
                except BaseException as error:
                    admission_errors.append(error)
                session = hook_store._session_row(CONNECTION_ID, original.session_id)
                assert session is not None
                observed_state = CaptureSessionState(session["state"])
                observed_health_count = _count(hook_store, "capture_health")
                observed_event_count = _count(hook_store, "capture_events")
        finally:
            admission_attempted.set()
            worker.join(timeout=5)

        assert not worker.is_alive()
        assert [type(error) for error in admission_errors] == [CaptureStoreStateError]
        assert observed_state is CaptureSessionState.DELETING
        assert observed_health_count == 0
        assert observed_event_count == 1
        assert delete_errors == []
        assert len(delete_receipts) == 1
        assert delete_receipts[0].disposition is CaptureDeleteDisposition.DELETED

    with _open(path) as store:
        with pytest.raises(CaptureStoreStateError):
            store.append(original)
        if attempt == "digest_collision":
            with pytest.raises(CaptureStoreStateError):
                store.append(candidate)
        assert store.list_sessions() == ()
        assert _count(store, "capture_events") == 0
        assert _count(store, "capture_health") == 0
        assert _count(store, "deleted_sessions") == 1


def test_delete_all_requires_confirmation_and_disabled_project_connections(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.sqlite3"
    initialize_capture_store(path)
    with _open(path) as store:
        register_connection(store)
        _session(store)
        drain_calls: list[str] = []

        with pytest.raises(CaptureStoreStateError):
            delete_capture_project(
                store,
                PROJECT_DIGEST,
                confirm=False,
                drain=lambda: drain_calls.append("unexpected"),
            )
        with pytest.raises(CaptureStoreStateError):
            delete_capture_project(
                store,
                PROJECT_DIGEST,
                confirm=True,
                drain=lambda: drain_calls.append("unexpected"),
            )
        assert drain_calls == []
        assert len(store.list_connections(project_digest=PROJECT_DIGEST)) == 1
        assert len(store.list_sessions(project_digest=PROJECT_DIGEST)) == 1


def test_delete_all_removes_only_the_disabled_project_and_is_retry_safe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.sqlite3"
    initialize_capture_store(path)
    with _open(path) as store:
        register_connection(store)
        first_human = _session(store, native=b"project-one-first", index=1)
        delete_capture_session(store, first_human, drain=lambda: None)
        store.append(
            authenticated_intake(
                "session_started",
                session_native=b"project-one-second",
                producer_index=20,
            )
        )
        _disable(store, CONNECTION_ID)

        register_connection(
            store,
            connection_id=OTHER_CONNECTION_ID,
            project_digest=OTHER_PROJECT_DIGEST,
        )
        store.append(
            authenticated_intake(
                "session_started",
                connection_id=OTHER_CONNECTION_ID,
                session_native=b"project-two",
                producer_index=30,
            )
        )

        calls: list[str] = []
        receipt = delete_capture_project(
            store,
            PROJECT_DIGEST,
            confirm=True,
            drain=lambda: calls.append("drained"),
        )

        assert type(receipt) is CaptureProjectDeleteReceipt
        assert receipt.disposition is CaptureDeleteDisposition.DELETED
        assert receipt.deleted_connections == 1
        assert receipt.deleted_sessions == 1
        assert receipt.deleted_tombstones == 1
        assert receipt.secure_delete is True
        assert receipt.wal_checkpointed is True
        assert calls == ["drained"]
        assert store.list_connections(project_digest=PROJECT_DIGEST) == ()
        assert store.list_sessions(project_digest=PROJECT_DIGEST) == ()
        assert len(store.list_connections(project_digest=OTHER_PROJECT_DIGEST)) == 1
        assert len(store.list_sessions(project_digest=OTHER_PROJECT_DIGEST)) == 1
        assert _count(store, "deleted_sessions") == 0

        repeated = delete_capture_project(
            store,
            PROJECT_DIGEST,
            confirm=True,
            drain=lambda: calls.append("repeat-drain"),
        )
        assert repeated.disposition is CaptureDeleteDisposition.ALREADY_DELETED
        assert repeated.deleted_connections == repeated.deleted_sessions == 0
        assert calls == ["drained"]


@pytest.mark.parametrize(
    "fault_stage",
    (
        "delete_project_after_mark_commit",
        "delete_project_before_purge_commit",
        "delete_project_after_purge_commit",
    ),
)
def test_delete_all_recovers_idempotently_from_each_durable_fault_state(
    tmp_path: Path,
    fault_stage: str,
) -> None:
    path = tmp_path / f"{fault_stage}.sqlite3"
    initialize_capture_store(path)
    with _open(path) as store:
        register_connection(store)
        _session(store)
        _disable(store, CONNECTION_ID)

    def fault(stage: str) -> None:
        if stage == fault_stage:
            raise RuntimeError("provider-native-project-crash-secret")

    with _open(path, fault=fault) as store, pytest.raises(RuntimeError):
        delete_capture_project(
            store,
            PROJECT_DIGEST,
            confirm=True,
            drain=lambda: None,
        )

    with _open(path) as store:
        remaining = store.list_connections(project_digest=PROJECT_DIGEST)
        if remaining:
            assert tuple(item.state for item in remaining) == (CaptureConnectionState.DELETING,)
            assert len(store.list_sessions(project_digest=PROJECT_DIGEST)) == 1
        receipt = delete_capture_project(
            store,
            PROJECT_DIGEST,
            confirm=True,
            drain=lambda: pytest.fail("durably marked deletion must not drain again"),
        )
        assert receipt.disposition in {
            CaptureDeleteDisposition.DELETED,
            CaptureDeleteDisposition.ALREADY_DELETED,
        }
        assert store.list_connections(project_digest=PROJECT_DIGEST) == ()
        assert store.list_sessions(project_digest=PROJECT_DIGEST) == ()


def test_delete_operations_require_maintenance_mode(tmp_path: Path) -> None:
    path = tmp_path / "capture.sqlite3"
    initialize_capture_store(path)
    with _open(path, mode=CaptureStoreMode.HOOK) as store:
        with pytest.raises(CaptureStoreStateError):
            delete_capture_session(store, "abcdefghijkl", drain=lambda: None)
        with pytest.raises(CaptureStoreStateError):
            delete_capture_project(
                store,
                PROJECT_DIGEST,
                confirm=True,
                drain=lambda: None,
            )


def test_root_exports_delete_contracts() -> None:
    import saliencegate.capture as capture

    expected = {
        "CaptureDeleteDisposition": CaptureDeleteDisposition,
        "CaptureProjectDeleteReceipt": CaptureProjectDeleteReceipt,
        "CaptureSessionDeleteReceipt": CaptureSessionDeleteReceipt,
        "delete_capture_project": delete_capture_project,
        "delete_capture_session": delete_capture_session,
    }
    assert expected.keys() <= set(capture.__all__)
    for name, value in expected.items():
        assert getattr(capture, name) is value
