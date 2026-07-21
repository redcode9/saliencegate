from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.capture.store_support import (
    INSTALLATION_KEY,
    OTHER_PROJECT_DIGEST,
    PROJECT_DIGEST,
    authenticated_intake,
    capture_context,
    initialized_store,
    register_connection,
)

import saliencegate.capture.store as store_module
from saliencegate.capture.capabilities import CaptureProfile
from saliencegate.capture.feedback import (
    CaptureFeedbackBlindingStatus,
    CaptureFeedbackError,
    CaptureFeedbackEvidenceSource,
    CaptureFeedbackExportRecord,
    CaptureFeedbackLabel,
    CaptureFeedbackPartition,
    CaptureFeedbackPrediction,
    CaptureFeedbackRecord,
    CaptureFeedbackRecordOrigin,
    CaptureFeedbackStudyAttestation,
    CaptureFeedbackWriteDisposition,
    build_capture_feedback_dataset,
    build_capture_feedback_export_record,
    build_synthetic_capture_feedback_export_record,
    decode_capture_feedback_dataset,
    encode_capture_feedback_dataset,
)
from saliencegate.capture.health import CaptureHealthCode
from saliencegate.capture.locations import CaptureStoreLocations
from saliencegate.capture.normalization import (
    CaptureNormalization,
    normalize_capture_session_snapshot,
)
from saliencegate.capture.report import (
    CaptureReportHeadline,
    CaptureSessionReport,
    build_capture_session_report,
)
from saliencegate.capture.sessions import CaptureSessionSnapshot
from saliencegate.capture.spool import CaptureSpool
from saliencegate.capture.store import (
    CaptureSessionState,
    CaptureStore,
    CaptureStoreIntegrityError,
    CaptureStoreMode,
    CaptureStoreStateError,
)
from saliencegate.domain import canonical_json, length_prefixed_sha256
from saliencegate.security import InstallationKey

NOW = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)


def _session(
    store: CaptureStore,
    *,
    native: bytes = b"feedback-session",
    index: int = 1,
) -> str:
    intake = authenticated_intake(
        "session_started",
        session_native=native,
        producer_index=index,
    )
    started = store.append(intake)
    store.append(
        authenticated_intake(
            "session_finished",
            session_native=native,
            producer_index=index + 1,
        )
    )
    return next(
        item.human_id
        for item in store.list_sessions(project_digest=PROJECT_DIGEST)
        if item.session_id == started.session_id
    )


def _record(
    *,
    project: str = PROJECT_DIGEST,
    session: str = "1" * 64,
    human: str = "a" * 12,
    label: CaptureFeedbackLabel = CaptureFeedbackLabel.MEMORY_NEEDED,
) -> CaptureFeedbackRecord:
    return CaptureFeedbackRecord(
        project_digest=project,
        profile_id=CaptureProfile.CODEX_HOOKS_V1,
        session_id=session,
        human_id=human,
        label=label,
        revision_count=1,
        labeled_at=NOW,
        record_tag="0" * 64,
    )


def _export_record(
    index: int,
    *,
    project: str = PROJECT_DIGEST,
    label: CaptureFeedbackLabel = CaptureFeedbackLabel.MEMORY_NEEDED,
    prediction: CaptureFeedbackPrediction = CaptureFeedbackPrediction.MEMORY_NEEDED,
    partition: CaptureFeedbackPartition = CaptureFeedbackPartition.FINAL_TEST,
) -> CaptureFeedbackExportRecord:
    return build_synthetic_capture_feedback_export_record(
        project_digest=project,
        profile_id=CaptureProfile.CODEX_HOOKS_V1,
        session_id=f"{index:064x}",
        label=label,
        prediction=prediction,
        partition=partition,
        installation_key=INSTALLATION_KEY,
    )


def _report_bound_feedback(
    tmp_path: Path,
    name: str,
    *,
    action_count: int,
    repeated: bool,
    close: bool = True,
) -> tuple[
    CaptureFeedbackRecord,
    CaptureSessionReport,
    CaptureSessionSnapshot,
    CaptureNormalization,
    CaptureSpool,
]:
    state_directory = tmp_path / name
    locations = CaptureStoreLocations(
        platform="windows" if os.name == "nt" else "posix",
        state_directory=state_directory,
        database_path=state_directory / "capture.sqlite3",
        spool_directory=state_directory / "capture-spool",
    )
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    context = capture_context()
    session_native = name.encode()
    shared_action = context.action_identity(b"feedback-repeated-action")
    with initialized_store(locations.database_path) as store:
        register_connection(store)
        started = store.append(
            authenticated_intake(
                "session_started",
                session_native=session_native,
                producer_index=1,
            )
        )
        for offset in range(action_count):
            changes = {"identity_authority": "exact"}
            if repeated:
                changes["action_digest"] = shared_action
            store.append(
                authenticated_intake(
                    "action_started",
                    session_native=session_native,
                    producer_index=offset + 2,
                    changes=changes,
                )
            )
        if close:
            store.append(
                authenticated_intake(
                    "session_finished",
                    session_native=session_native,
                    producer_index=action_count + 2,
                )
            )
        snapshot = store.snapshot_session(started.connection_id, started.session_id)
        if not close:
            store.append(
                authenticated_intake(
                    "session_finished",
                    session_native=session_native,
                    producer_index=action_count + 2,
                )
            )
        store.record_feedback(
            snapshot.human_id,
            CaptureFeedbackLabel.MEMORY_NEEDED,
            project_digest=PROJECT_DIGEST,
        )
        record = store.list_feedback(project_digest=PROJECT_DIGEST, limit=1)[0]

    normalization = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )
    report = build_capture_session_report(
        snapshot,
        normalization,
        installation_key=INSTALLATION_KEY,
        spool=spool,
    )
    return record, report, snapshot, normalization, spool


def _study_attestation() -> CaptureFeedbackStudyAttestation:
    return CaptureFeedbackStudyAttestation(
        consent_protocol_digest="1" * 64,
        preregistration_digest="2" * 64,
        temporal_split_digest="3" * 64,
        evaluator_freeze_digest="4" * 64,
        label_freeze_digest="5" * 64,
        report_selection_policy_digest="7" * 64,
        no_test_tuning_digest="6" * 64,
        blinding_status=CaptureFeedbackBlindingStatus.EXTERNALLY_ATTESTED,
    )


def test_feedback_label_contract_is_closed_strict_and_redacted() -> None:
    assert tuple(CaptureFeedbackLabel) == (
        CaptureFeedbackLabel.MEMORY_NEEDED,
        CaptureFeedbackLabel.NOT_MEMORY_NEEDED,
        CaptureFeedbackLabel.UNCERTAIN,
    )
    assert tuple(item.value for item in CaptureFeedbackLabel) == (
        "memory-needed",
        "not-memory-needed",
        "uncertain",
    )

    record = _record()
    assert repr(record) == "CaptureFeedbackRecord(<redacted>)"
    assert str(record) == repr(record)
    with pytest.raises(ValidationError):
        CaptureFeedbackRecord.model_validate(
            {**record.model_dump(mode="python"), "label": "helpful"}
        )
    with pytest.raises(ValidationError):
        record.label = CaptureFeedbackLabel.UNCERTAIN  # type: ignore[misc]


def test_first_label_is_recorded_and_history_is_authenticated(tmp_path: Path) -> None:
    path = tmp_path / "capture.sqlite3"
    with initialized_store(path) as store:
        register_connection(store)
        human_id = _session(store)

        receipt = store.record_feedback(
            human_id,
            CaptureFeedbackLabel.MEMORY_NEEDED,
            project_digest=PROJECT_DIGEST,
        )
        history = store.feedback_history(human_id, project_digest=PROJECT_DIGEST)

        assert receipt.disposition is CaptureFeedbackWriteDisposition.RECORDED
        assert receipt.session_id == human_id
        assert receipt.label is CaptureFeedbackLabel.MEMORY_NEEDED
        assert receipt.revision_count == 1
        assert receipt.labeled_at == history[0].created_at
        assert tuple((item.revision, item.label) for item in history) == (
            (1, CaptureFeedbackLabel.MEMORY_NEEDED),
        )
        assert repr(receipt) == "CaptureFeedbackReceipt(<redacted>)"
        assert repr(history[0]) == "CaptureFeedbackRevision(<redacted>)"


def test_identical_label_is_a_true_idempotent_no_op(tmp_path: Path) -> None:
    path = tmp_path / "capture.sqlite3"
    with initialized_store(path) as store:
        register_connection(store)
        human_id = _session(store)
        first = store.record_feedback(
            human_id,
            CaptureFeedbackLabel.NOT_MEMORY_NEEDED,
            project_digest=PROJECT_DIGEST,
        )
        before = store._connection.total_changes

        replay = store.record_feedback(
            human_id,
            CaptureFeedbackLabel.NOT_MEMORY_NEEDED,
            project_digest=PROJECT_DIGEST,
        )

        assert replay.disposition is CaptureFeedbackWriteDisposition.UNCHANGED
        assert replay.revision_count == 1
        assert replay.labeled_at == first.labeled_at
        assert store._connection.total_changes == before
        assert len(store.feedback_history(human_id, project_digest=PROJECT_DIGEST)) == 1


@pytest.mark.parametrize(
    ("fault_stage", "committed"),
    (
        ("feedback_before_commit", False),
        ("feedback_after_commit", True),
    ),
)
def test_feedback_anchor_and_revision_share_one_durable_transaction(
    tmp_path: Path,
    fault_stage: str,
    committed: bool,
) -> None:
    path = tmp_path / "capture.sqlite3"
    with initialized_store(path) as store:
        register_connection(store)
        human_id = _session(store)

    def fault(stage: str) -> None:
        if stage == fault_stage:
            raise RuntimeError("feedback-fault-sentinel")

    with (
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            mode=CaptureStoreMode.MAINTENANCE,
            _fault_injector=fault,
        ) as store,
        pytest.raises(CaptureStoreIntegrityError),
    ):
        store.record_feedback(
            human_id,
            CaptureFeedbackLabel.MEMORY_NEEDED,
            project_digest=PROJECT_DIGEST,
        )

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        records = store.list_feedback(project_digest=PROJECT_DIGEST)
        assert bool(records) is committed
        if committed:
            retry = store.record_feedback(
                human_id,
                CaptureFeedbackLabel.MEMORY_NEEDED,
                project_digest=PROJECT_DIGEST,
            )
            assert retry.disposition is CaptureFeedbackWriteDisposition.UNCHANGED


def test_label_changes_append_a_content_free_monotone_audit(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "capture.sqlite3"
    monkeypatch.setattr(store_module, "_now", lambda: NOW.isoformat())
    with initialized_store(path) as store:
        register_connection(store)
        human_id = _session(store)

        first = store.record_feedback(
            human_id,
            CaptureFeedbackLabel.MEMORY_NEEDED,
            project_digest=PROJECT_DIGEST,
        )
        second = store.record_feedback(
            human_id,
            CaptureFeedbackLabel.UNCERTAIN,
            project_digest=PROJECT_DIGEST,
        )
        third = store.record_feedback(
            human_id,
            CaptureFeedbackLabel.MEMORY_NEEDED,
            project_digest=PROJECT_DIGEST,
        )
        history = store.feedback_history(human_id, project_digest=PROJECT_DIGEST)

        assert first.disposition is CaptureFeedbackWriteDisposition.RECORDED
        assert second.disposition is third.disposition is CaptureFeedbackWriteDisposition.CHANGED
        assert (first.revision_count, second.revision_count, third.revision_count) == (1, 2, 3)
        assert tuple(item.label for item in history) == (
            CaptureFeedbackLabel.MEMORY_NEEDED,
            CaptureFeedbackLabel.UNCERTAIN,
            CaptureFeedbackLabel.MEMORY_NEEDED,
        )
        assert tuple(item.revision for item in history) == (1, 2, 3)
        assert history[0].created_at < history[1].created_at < history[2].created_at

        persisted = store._connection.execute(
            "SELECT label_id, label, created_at, row_tag FROM feedback_labels ORDER BY created_at"
        ).fetchall()
        encoded = json.dumps([dict(row) for row in persisted], sort_keys=True)
        assert len(persisted) == 4
        assert "prompt-sentinel" not in encoded
        assert "output-sentinel" not in encoded
        assert "event_json" not in encoded


def test_first_feedback_timestamp_cannot_precede_authenticated_session_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "capture.sqlite3"
    with initialized_store(path) as store:
        register_connection(store)
        human_id = _session(store)
        closed = next(item for item in store.list_sessions() if item.human_id == human_id)
        assert closed.closed_at is not None
        monkeypatch.setattr(
            store_module,
            "_now",
            lambda: "2000-01-01T00:00:00+00:00",
        )

        receipt = store.record_feedback(
            human_id,
            CaptureFeedbackLabel.MEMORY_NEEDED,
            project_digest=PROJECT_DIGEST,
        )

        assert receipt.labeled_at == closed.closed_at
        assert store.list_feedback(project_digest=PROJECT_DIGEST)[0].labeled_at == closed.closed_at


def test_closed_session_can_be_labeled_after_local_clock_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "capture.sqlite3"
    monkeypatch.setattr(store_module, "_now", lambda: NOW.isoformat())
    with initialized_store(path) as store:
        register_connection(store)
        started = store.append(
            authenticated_intake(
                "session_started",
                session_native=b"feedback-clock-rollback",
                producer_index=1,
            )
        )
        rolled_back = NOW - timedelta(days=1)
        monkeypatch.setattr(store_module, "_now", lambda: rolled_back.isoformat())
        store.append(
            authenticated_intake(
                "session_finished",
                session_native=b"feedback-clock-rollback",
                producer_index=2,
            )
        )
        snapshot = store.snapshot_session(started.connection_id, started.session_id)

        receipt = store.record_feedback(
            snapshot.human_id,
            CaptureFeedbackLabel.MEMORY_NEEDED,
            project_digest=PROJECT_DIGEST,
        )
        summary = store.session_by_human_id(snapshot.human_id)

        assert snapshot.updated_at == snapshot.closed_at == rolled_back
        assert snapshot.updated_at < snapshot.opened_at
        assert summary.updated_at == summary.closed_at == rolled_back
        assert receipt.labeled_at == rolled_back


def test_late_health_preserves_closure_and_feedback_causality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "capture.sqlite3"
    monkeypatch.setattr(store_module, "_now", lambda: NOW.isoformat())
    with initialized_store(path) as store:
        register_connection(store)
        human_id = _session(store)
        closed = store.session_by_human_id(human_id)
        assert closed.closed_at == NOW

        labeled_at = NOW + timedelta(seconds=1)
        monkeypatch.setattr(store_module, "_now", lambda: labeled_at.isoformat())
        store.record_feedback(
            human_id,
            CaptureFeedbackLabel.MEMORY_NEEDED,
            project_digest=PROJECT_DIGEST,
        )

        health_at = NOW + timedelta(seconds=2)
        monkeypatch.setattr(store_module, "_now", lambda: health_at.isoformat())
        store.mark_session_health(
            closed.connection_id,
            closed.session_id,
            CaptureHealthCode.COVERAGE_DEGRADED,
        )

        updated = store.session_by_human_id(human_id)
        snapshot = store.snapshot_session(closed.connection_id, closed.session_id)
        record = store.list_feedback(project_digest=PROJECT_DIGEST, limit=1)[0]

        assert updated.closed_at == snapshot.closed_at == NOW
        assert updated.updated_at == snapshot.updated_at == health_at
        assert record.labeled_at == labeled_at
        assert updated.closed_at <= record.labeled_at < updated.updated_at

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as reopened:
        persisted = reopened.session_by_human_id(human_id)
        persisted_snapshot = reopened.snapshot_session(
            persisted.connection_id,
            persisted.session_id,
        )

        assert persisted.closed_at == persisted_snapshot.closed_at == NOW
        assert persisted.updated_at == persisted_snapshot.updated_at == health_at
        assert len(persisted_snapshot.health) == 1
        assert persisted_snapshot.health[0].code is CaptureHealthCode.COVERAGE_DEGRADED


def test_feedback_mutation_is_project_bound_inside_the_transaction(tmp_path: Path) -> None:
    path = tmp_path / "capture.sqlite3"
    with initialized_store(path) as store:
        register_connection(store)
        human_id = _session(store)

        with pytest.raises(CaptureStoreStateError) as captured:
            store.record_feedback(
                human_id,
                CaptureFeedbackLabel.MEMORY_NEEDED,
                project_digest=OTHER_PROJECT_DIGEST,
            )

        assert str(captured.value) == "capture store state transition failed"
        assert human_id not in repr(captured.value)
        assert store._connection.execute("SELECT COUNT(*) FROM feedback_labels").fetchone()[0] == 0


@pytest.mark.parametrize(
    ("human_id", "label", "project_digest"),
    (
        ("short", CaptureFeedbackLabel.MEMORY_NEEDED, PROJECT_DIGEST),
        ("a" * 12, "memory-needed", PROJECT_DIGEST),
        ("a" * 12, CaptureFeedbackLabel.MEMORY_NEEDED, "not-a-digest"),
    ),
)
def test_feedback_rejects_non_exact_or_unbounded_inputs(
    tmp_path: Path,
    human_id: object,
    label: object,
    project_digest: object,
) -> None:
    path = tmp_path / "capture.sqlite3"
    with initialized_store(path) as store, pytest.raises(CaptureStoreStateError):
        store.record_feedback(  # type: ignore[arg-type]
            human_id,
            label,
            project_digest=project_digest,
        )


def test_feedback_is_maintenance_only(tmp_path: Path) -> None:
    path = tmp_path / "capture.sqlite3"
    with initialized_store(path) as store:
        register_connection(store)
        human_id = _session(store)
    with (
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            mode=CaptureStoreMode.HOOK,
        ) as store,
        pytest.raises(CaptureStoreStateError),
    ):
        store.record_feedback(
            human_id,
            CaptureFeedbackLabel.MEMORY_NEEDED,
            project_digest=PROJECT_DIGEST,
        )


def test_feedback_history_detects_row_and_middle_revision_tampering(tmp_path: Path) -> None:
    path = tmp_path / "capture.sqlite3"
    with initialized_store(path) as store:
        register_connection(store)
        human_id = _session(store)
        for label in (
            CaptureFeedbackLabel.MEMORY_NEEDED,
            CaptureFeedbackLabel.UNCERTAIN,
            CaptureFeedbackLabel.NOT_MEMORY_NEEDED,
        ):
            store.record_feedback(human_id, label, project_digest=PROJECT_DIGEST)

    with sqlite3.connect(path) as connection:
        connection.execute(
            "DELETE FROM feedback_labels WHERE label_id = "
            "(SELECT label_id FROM feedback_labels ORDER BY created_at LIMIT 1 OFFSET 1)"
        )
        connection.commit()

    with pytest.raises(CaptureStoreIntegrityError):
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            mode=CaptureStoreMode.MAINTENANCE,
        )


@pytest.mark.parametrize("deleted_tail_rows", (1, 2))
def test_feedback_anchor_detects_tail_history_deletion(
    tmp_path: Path,
    deleted_tail_rows: int,
) -> None:
    path = tmp_path / "capture.sqlite3"
    with initialized_store(path) as store:
        register_connection(store)
        human_id = _session(store)
        for label in (
            CaptureFeedbackLabel.MEMORY_NEEDED,
            CaptureFeedbackLabel.UNCERTAIN,
            CaptureFeedbackLabel.NOT_MEMORY_NEEDED,
        ):
            store.record_feedback(human_id, label, project_digest=PROJECT_DIGEST)

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            DELETE FROM feedback_labels
            WHERE label_id IN (
                SELECT label_id FROM feedback_labels
                ORDER BY created_at DESC, label_id DESC
                LIMIT ?
            )
            """,
            (deleted_tail_rows,),
        )
        connection.commit()

    with pytest.raises(CaptureStoreIntegrityError):
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            mode=CaptureStoreMode.MAINTENANCE,
        )


def test_complete_feedback_history_removal_matches_documented_rollback_limit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.sqlite3"
    with initialized_store(path) as store:
        register_connection(store)
        human_id = _session(store)
        store.record_feedback(
            human_id,
            CaptureFeedbackLabel.MEMORY_NEEDED,
            project_digest=PROJECT_DIGEST,
        )

    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM feedback_labels")
        connection.commit()

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        assert store.list_feedback(project_digest=PROJECT_DIGEST) == ()


def test_list_feedback_returns_only_latest_project_bound_records(tmp_path: Path) -> None:
    path = tmp_path / "capture.sqlite3"
    with initialized_store(path) as store:
        register_connection(store)
        first_human = _session(store, native=b"first-feedback-session")
        second_human = _session(store, native=b"second-feedback-session", index=3)
        store.record_feedback(
            first_human,
            CaptureFeedbackLabel.MEMORY_NEEDED,
            project_digest=PROJECT_DIGEST,
        )
        store.record_feedback(
            first_human,
            CaptureFeedbackLabel.UNCERTAIN,
            project_digest=PROJECT_DIGEST,
        )
        store.record_feedback(
            second_human,
            CaptureFeedbackLabel.NOT_MEMORY_NEEDED,
            project_digest=PROJECT_DIGEST,
        )

        records = store.list_feedback(project_digest=PROJECT_DIGEST, limit=10)

        assert tuple(item.human_id for item in records) == tuple(
            sorted((first_human, second_human))
        )
        by_human = {item.human_id: item for item in records}
        assert by_human[first_human].label is CaptureFeedbackLabel.UNCERTAIN
        assert by_human[first_human].revision_count == 2
        assert by_human[second_human].revision_count == 1
        assert all(item.project_digest == PROJECT_DIGEST for item in records)
        assert store.list_feedback(project_digest=OTHER_PROJECT_DIGEST, limit=10) == ()
        assert all(repr(item) == "CaptureFeedbackRecord(<redacted>)" for item in records)
        with pytest.raises(CaptureStoreStateError):
            store.list_feedback(project_digest=PROJECT_DIGEST, limit=1)


def test_list_feedback_selects_the_last_revision_strictly_before_the_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "capture.sqlite3"
    freeze = NOW + timedelta(seconds=1)
    monkeypatch.setattr(store_module, "_now", lambda: NOW.isoformat())
    with initialized_store(path) as store:
        register_connection(store)
        human_id = _session(store)
        store.record_feedback(
            human_id,
            CaptureFeedbackLabel.MEMORY_NEEDED,
            project_digest=PROJECT_DIGEST,
        )
        monkeypatch.setattr(
            store_module,
            "_now",
            lambda: (NOW + timedelta(seconds=2)).isoformat(),
        )
        store.record_feedback(
            human_id,
            CaptureFeedbackLabel.NOT_MEMORY_NEEDED,
            project_digest=PROJECT_DIGEST,
        )

        current = store.list_feedback(project_digest=PROJECT_DIGEST, limit=1)[0]
        frozen = store.list_feedback(
            project_digest=PROJECT_DIGEST,
            label_freeze=freeze,
            limit=1,
        )[0]

        assert current.label is CaptureFeedbackLabel.NOT_MEMORY_NEEDED
        assert current.revision_count == 2
        assert frozen.label is CaptureFeedbackLabel.MEMORY_NEEDED
        assert frozen.revision_count == 1
        assert frozen.labeled_at == NOW
        assert frozen.record_tag != current.record_tag
        assert (
            store.list_feedback(
                project_digest=PROJECT_DIGEST,
                label_freeze=NOW,
                limit=1,
            )
            == ()
        )


@pytest.mark.parametrize(
    ("name", "action_count", "repeated", "headline", "prediction"),
    (
        (
            "report-bound-positive",
            2,
            True,
            CaptureReportHeadline.MEMORY_REVIEW_SUGGESTED,
            CaptureFeedbackPrediction.MEMORY_NEEDED,
        ),
        (
            "report-bound-negative",
            2,
            False,
            CaptureReportHeadline.NO_CURRENT_EVIDENCE,
            CaptureFeedbackPrediction.NOT_MEMORY_NEEDED,
        ),
        (
            "report-bound-abstain",
            1,
            False,
            CaptureReportHeadline.INSUFFICIENT_EVIDENCE,
            CaptureFeedbackPrediction.ABSTAIN,
        ),
    ),
)
def test_report_bound_export_derives_prediction_only_from_a_real_closed_report(
    tmp_path: Path,
    name: str,
    action_count: int,
    repeated: bool,
    headline: CaptureReportHeadline,
    prediction: CaptureFeedbackPrediction,
) -> None:
    record, report, snapshot, normalization, spool = _report_bound_feedback(
        tmp_path,
        name,
        action_count=action_count,
        repeated=repeated,
    )

    exported = build_capture_feedback_export_record(
        record,
        report,
        snapshot=snapshot,
        normalization=normalization,
        spool=spool,
        partition=CaptureFeedbackPartition.FINAL_TEST,
        label_freeze=record.labeled_at + timedelta(microseconds=1),
        installation_key=INSTALLATION_KEY,
    )

    assert report.session_state is CaptureSessionState.CLOSED
    assert report.headline is headline
    assert exported.prediction is prediction
    assert exported.origin is CaptureFeedbackRecordOrigin.LOCAL_CAPTURE_REPORT
    assert exported.session_id == record.session_id
    assert exported.source_report_digest == report.report_digest


def test_report_bound_export_rejects_profile_and_session_mismatches(
    tmp_path: Path,
) -> None:
    record, report, snapshot, normalization, spool = _report_bound_feedback(
        tmp_path,
        "report-bound-mismatch",
        action_count=2,
        repeated=False,
    )
    mismatches = (
        record.model_copy(update={"human_id": "b" * 12}),
        record.model_copy(update={"profile_id": CaptureProfile.CLAUDE_CODE_HOOKS_V1}),
    )

    for mismatch in mismatches:
        with pytest.raises(CaptureFeedbackError):
            build_capture_feedback_export_record(
                mismatch,
                report,
                snapshot=snapshot,
                normalization=normalization,
                spool=spool,
                partition=CaptureFeedbackPartition.FINAL_TEST,
                label_freeze=record.labeled_at + timedelta(microseconds=1),
                installation_key=INSTALLATION_KEY,
            )

    with pytest.raises(CaptureFeedbackError):
        build_capture_feedback_export_record(
            record,
            report,
            snapshot=snapshot.model_copy(update={"snapshot_digest": "0" * 64}),
            normalization=normalization,
            spool=spool,
            partition=CaptureFeedbackPartition.FINAL_TEST,
            label_freeze=record.labeled_at + timedelta(microseconds=1),
            installation_key=INSTALLATION_KEY,
        )


def test_report_bound_export_requires_the_authenticated_spool(tmp_path: Path) -> None:
    record, report, snapshot, normalization, _spool = _report_bound_feedback(
        tmp_path,
        "report-bound-missing-spool",
        action_count=2,
        repeated=False,
    )

    with pytest.raises(CaptureFeedbackError):
        build_capture_feedback_export_record(
            record,
            report,
            snapshot=snapshot,
            normalization=normalization,
            spool=None,  # type: ignore[arg-type]
            partition=CaptureFeedbackPartition.FINAL_TEST,
            label_freeze=record.labeled_at + timedelta(microseconds=1),
            installation_key=INSTALLATION_KEY,
        )


def test_report_bound_export_rejects_an_open_session(tmp_path: Path) -> None:
    record, report, snapshot, normalization, spool = _report_bound_feedback(
        tmp_path,
        "report-bound-open",
        action_count=2,
        repeated=True,
        close=False,
    )

    assert report.session_state is CaptureSessionState.OPEN
    with pytest.raises(CaptureFeedbackError):
        build_capture_feedback_export_record(
            record,
            report,
            snapshot=snapshot,
            normalization=normalization,
            spool=spool,
            partition=CaptureFeedbackPartition.FINAL_TEST,
            label_freeze=record.labeled_at + timedelta(microseconds=1),
            installation_key=INSTALLATION_KEY,
        )


@pytest.mark.parametrize("freeze_offset", (timedelta(), -timedelta(microseconds=1)))
def test_report_bound_export_rejects_labels_at_or_after_the_freeze(
    tmp_path: Path,
    freeze_offset: timedelta,
) -> None:
    record, report, snapshot, normalization, spool = _report_bound_feedback(
        tmp_path,
        f"report-bound-freeze-{freeze_offset.total_seconds()}",
        action_count=2,
        repeated=False,
    )

    with pytest.raises(CaptureFeedbackError):
        build_capture_feedback_export_record(
            record,
            report,
            snapshot=snapshot,
            normalization=normalization,
            spool=spool,
            partition=CaptureFeedbackPartition.FINAL_TEST,
            label_freeze=record.labeled_at + freeze_offset,
            installation_key=INSTALLATION_KEY,
        )


def test_dataset_rejects_a_tampered_report_bound_export_tag(tmp_path: Path) -> None:
    record, report, snapshot, normalization, spool = _report_bound_feedback(
        tmp_path,
        "report-bound-tag-tamper",
        action_count=2,
        repeated=False,
    )
    exported = build_capture_feedback_export_record(
        record,
        report,
        snapshot=snapshot,
        normalization=normalization,
        spool=spool,
        partition=CaptureFeedbackPartition.FINAL_TEST,
        label_freeze=record.labeled_at + timedelta(microseconds=1),
        installation_key=INSTALLATION_KEY,
    )
    tampered = exported.model_copy(update={"record_tag": "0" * 64})

    with pytest.raises(CaptureFeedbackError):
        build_capture_feedback_dataset(
            (tampered,),
            installation_key=INSTALLATION_KEY,
            export_nonce=b"t" * 32,
            evidence_source=CaptureFeedbackEvidenceSource.LOCAL_FEEDBACK,
            opt_in=True,
        )


def test_declared_e01_requires_attestation_and_refuses_synthetic_records(
    tmp_path: Path,
) -> None:
    record, report, snapshot, normalization, spool = _report_bound_feedback(
        tmp_path,
        "report-bound-e01",
        action_count=2,
        repeated=False,
    )
    exported = build_capture_feedback_export_record(
        record,
        report,
        snapshot=snapshot,
        normalization=normalization,
        spool=spool,
        partition=CaptureFeedbackPartition.FINAL_TEST,
        label_freeze=record.labeled_at + timedelta(microseconds=1),
        installation_key=INSTALLATION_KEY,
    )
    attestation = _study_attestation()

    with pytest.raises(CaptureFeedbackError):
        build_capture_feedback_dataset(
            (exported,),
            installation_key=INSTALLATION_KEY,
            export_nonce=b"e" * 32,
            evidence_source=CaptureFeedbackEvidenceSource.DECLARED_E01,
            opt_in=True,
        )
    with pytest.raises(CaptureFeedbackError):
        build_capture_feedback_dataset(
            (_export_record(999),),
            installation_key=INSTALLATION_KEY,
            export_nonce=b"s" * 32,
            evidence_source=CaptureFeedbackEvidenceSource.DECLARED_E01,
            opt_in=True,
            study_attestation=attestation,
        )

    dataset = build_capture_feedback_dataset(
        (exported,),
        installation_key=INSTALLATION_KEY,
        export_nonce=b"e" * 32,
        evidence_source=CaptureFeedbackEvidenceSource.DECLARED_E01,
        opt_in=True,
        study_attestation=attestation,
    )

    assert dataset.evidence_source is CaptureFeedbackEvidenceSource.DECLARED_E01
    assert dataset.study_attestation == attestation


def test_explicit_dataset_export_is_pseudonymized_canonical_and_unlinkable() -> None:
    records = (
        _export_record(2, project=PROJECT_DIGEST),
        _export_record(
            1,
            project=OTHER_PROJECT_DIGEST,
            label=CaptureFeedbackLabel.UNCERTAIN,
            prediction=CaptureFeedbackPrediction.ABSTAIN,
        ),
    )

    first = build_capture_feedback_dataset(
        records,
        installation_key=INSTALLATION_KEY,
        export_nonce=b"a" * 32,
        evidence_source=CaptureFeedbackEvidenceSource.SYNTHETIC,
        opt_in=True,
    )
    replay = build_capture_feedback_dataset(
        tuple(reversed(records)),
        installation_key=INSTALLATION_KEY,
        export_nonce=b"a" * 32,
        evidence_source=CaptureFeedbackEvidenceSource.SYNTHETIC,
        opt_in=True,
    )
    unlinked = build_capture_feedback_dataset(
        records,
        installation_key=INSTALLATION_KEY,
        export_nonce=b"b" * 32,
        evidence_source=CaptureFeedbackEvidenceSource.SYNTHETIC,
        opt_in=True,
    )
    encoded = encode_capture_feedback_dataset(first)

    assert (
        first
        == replay
        == decode_capture_feedback_dataset(
            encoded,
            installation_key=INSTALLATION_KEY,
        )
    )
    assert first.dataset_digest == replay.dataset_digest
    assert first.export_id != unlinked.export_id
    assert {item.example_id for item in first.examples}.isdisjoint(
        item.example_id for item in unlinked.examples
    )
    assert {item.project_id for item in first.examples}.isdisjoint(
        item.project_id for item in unlinked.examples
    )
    assert tuple(item.example_id for item in first.examples) == tuple(
        sorted(item.example_id for item in first.examples)
    )
    assert first.explicit_opt_in is True
    assert first.raw_content_included is False
    assert first.direct_identifiers_included is False
    assert first.identifier_scope == "export_specific_hmac"
    for forbidden in (
        PROJECT_DIGEST,
        OTHER_PROJECT_DIGEST,
        f"{1:064x}",
        f"{2:064x}",
        "prompt-sentinel",
        "output-sentinel",
    ):
        assert forbidden.encode() not in encoded


@pytest.mark.parametrize(
    ("opt_in", "nonce", "records"),
    (
        (False, b"a" * 32, (_export_record(1),)),
        (True, b"short", (_export_record(1),)),
        (True, b"a" * 32, (_export_record(1), _export_record(1))),
        (True, b"a" * 32, [_export_record(1)]),
    ),
)
def test_dataset_export_fails_closed_without_exact_opt_in_and_unique_inputs(
    opt_in: object,
    nonce: object,
    records: object,
) -> None:
    with pytest.raises(CaptureFeedbackError) as captured:
        build_capture_feedback_dataset(  # type: ignore[arg-type]
            records,
            installation_key=INSTALLATION_KEY,
            export_nonce=nonce,
            evidence_source=CaptureFeedbackEvidenceSource.LOCAL_FEEDBACK,
            opt_in=opt_in,
        )

    assert str(captured.value) == "capture feedback data is invalid"
    assert "sentinel" not in repr(captured.value)


def test_dataset_decoder_rejects_noncanonical_duplicate_and_digest_tampering() -> None:
    dataset = build_capture_feedback_dataset(
        (_export_record(1),),
        installation_key=INSTALLATION_KEY,
        export_nonce=b"a" * 32,
        evidence_source=CaptureFeedbackEvidenceSource.SYNTHETIC,
        opt_in=True,
    )
    encoded = encode_capture_feedback_dataset(dataset)
    payload = json.loads(encoded)
    payload["evidence_source"] = "local_feedback"
    tampered = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    duplicate = encoded[:-1] + b',"schema_version":"capture-feedback-dataset/v1"}'

    for invalid in (encoded + b"\n", tampered, duplicate, b"[]", b"{", b"\xff"):
        with pytest.raises(CaptureFeedbackError):
            decode_capture_feedback_dataset(
                invalid,
                installation_key=INSTALLATION_KEY,
            )

    forged = json.loads(encoded)
    forged["examples"][0]["label"] = CaptureFeedbackLabel.NOT_MEMORY_NEEDED.value
    body = {
        key: value for key, value in forged.items() if key not in {"dataset_digest", "dataset_tag"}
    }
    forged["dataset_digest"] = length_prefixed_sha256(
        canonical_json(body),
        domain="saliencegate:capture:feedback-dataset:v1",
    )
    with pytest.raises(CaptureFeedbackError):
        decode_capture_feedback_dataset(
            canonical_json(forged),
            installation_key=INSTALLATION_KEY,
        )
    with pytest.raises(CaptureFeedbackError):
        decode_capture_feedback_dataset(
            encoded,
            installation_key=InstallationKey(b"z" * 32),
        )
