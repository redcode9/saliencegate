"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.capture.store_support import (
    CONNECTION_ID,
    INSTALLATION_KEY,
    PROJECT_DIGEST,
    authenticated_intake,
    initialized_store,
    register_connection,
)

import saliencegate.capture.store as store_module
from saliencegate.capture.capabilities import CaptureProfile
from saliencegate.capture.feedback import (
    MAX_CAPTURE_FEEDBACK_REVISIONS_PER_SESSION,
    CaptureFeedbackLabel,
)
from saliencegate.capture.health import CaptureHealthCode
from saliencegate.capture.store import (
    CaptureStore,
    CaptureStoreBusyError,
    CaptureStoreError,
    CaptureStoreIntegrityError,
    CaptureStoreStateError,
)


class _FailingConnection:
    def __init__(self, failure: BaseException) -> None:
        self.failure = failure
        self.rolled_back = False

    def execute(self, _statement: str, _values: object = ()) -> Any:
        raise self.failure

    def rollback(self) -> None:
        self.rolled_back = True


class _ResultConnection:
    def __init__(self, result: object) -> None:
        self.result = result

    def execute(self, _statement: str, _values: object = ()) -> _ResultConnection:
        return self

    def fetchone(self) -> object:
        return self.result


class _AuditConnection:
    def __init__(self, quick_check: object) -> None:
        self.quick_check = quick_check
        self.row_factory: object = None
        self.closed = False

    def execute(self, _statement: str, _values: object = ()) -> _ResultConnection:
        return _ResultConnection(self.quick_check)

    def close(self) -> None:
        self.closed = True


class _AuditAuthorization:
    def __init__(self, *, target_exists: bool, size: int | None = None) -> None:
        self.target_exists = target_exists
        self._target_complete_identity = None if size is None else SimpleNamespace(size=size)

    def revalidate(self) -> None:
        return None


class _RowCountResult:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class _StatementOverrideConnection:
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        marker: str,
        result: object,
    ) -> None:
        self.connection = connection
        self.marker = marker
        self.result = result

    def execute(self, statement: str, values: object = ()) -> Any:
        if self.marker in statement:
            return self.result
        return self.connection.execute(statement, values)

    def commit(self) -> None:
        self.connection.commit()

    def rollback(self) -> None:
        self.connection.rollback()


class _RowsConnection(_ResultConnection):
    def fetchall(self) -> object:
        return self.result


class _StatementRowsConnection:
    def __init__(self, rows_by_marker: dict[str, object]) -> None:
        self.rows_by_marker = rows_by_marker
        self.result: object = ()

    def execute(self, statement: str, _values: object = ()) -> _StatementRowsConnection:
        self.result = next(
            (result for marker, result in self.rows_by_marker.items() if marker in statement),
            (),
        )
        return self

    def fetchall(self) -> object:
        return self.result

    def fetchone(self) -> object:
        return self.result


class _SequenceConnection:
    def __init__(self, results: tuple[object, ...]) -> None:
        self.results = iter(results)
        self.result: object = None

    def execute(self, _statement: str, _values: object = ()) -> _SequenceConnection:
        self.result = next(self.results)
        return self

    def fetchone(self) -> object:
        return self.result


def _closed_session(
    store: CaptureStore,
    *,
    native: bytes = b"coverage-query-session",
    index: int = 1,
) -> str:
    started = store.append(
        authenticated_intake(
            "session_started",
            session_native=native,
            producer_index=index,
        )
    )
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


def _invoke_query(store: CaptureStore, name: str) -> object:
    human_id = "a" * 12
    calls: dict[str, Callable[[], object]] = {
        "list_connections": lambda: store.list_connections(),
        "get_connection": lambda: store.get_connection(CONNECTION_ID),
        "get_hook_connection": lambda: store._get_hook_connection(CONNECTION_ID),
        "list_sessions": lambda: store.list_sessions(),
        "session_inventory": lambda: store.session_inventory(),
        "session_by_human_id": lambda: store.session_by_human_id(human_id),
        "session_belongs_to_project": lambda: store.session_belongs_to_project(
            human_id,
            PROJECT_DIGEST,
        ),
        "feedback_history": lambda: store.feedback_history(
            human_id,
            project_digest=PROJECT_DIGEST,
        ),
        "list_feedback": lambda: store.list_feedback(),
        "latest_session": lambda: store.latest_session(project_digest=PROJECT_DIGEST),
        "project_connections_disabled": lambda: store._require_project_connections_disabled(
            PROJECT_DIGEST
        ),
        "session_delete_requires_drain": lambda: store._session_delete_requires_drain(human_id),
    }
    return calls[name]()


_QUERY_NAMES = (
    "list_connections",
    "get_connection",
    "get_hook_connection",
    "list_sessions",
    "session_inventory",
    "session_by_human_id",
    "session_belongs_to_project",
    "feedback_history",
    "list_feedback",
    "latest_session",
    "project_connections_disabled",
    "session_delete_requires_drain",
)


@pytest.mark.parametrize("name", _QUERY_NAMES)
def test_query_unexpected_failures_rollback_and_fail_as_integrity_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        failing = _FailingConnection(RuntimeError("query secret"))
        real_connection = store._connection
        monkeypatch.setattr(CaptureStore, "_begin_immediate", lambda _self: None)
        store._connection = failing  # type: ignore[assignment]
        try:
            with pytest.raises(CaptureStoreIntegrityError):
                _invoke_query(store, name)
            assert failing.rolled_back is True
        finally:
            store._connection = real_connection


@pytest.mark.parametrize("name", _QUERY_NAMES)
def test_query_base_exceptions_rollback_without_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        failing = _FailingConnection(KeyboardInterrupt())
        real_connection = store._connection
        monkeypatch.setattr(CaptureStore, "_begin_immediate", lambda _self: None)
        store._connection = failing  # type: ignore[assignment]
        try:
            with pytest.raises(KeyboardInterrupt):
                _invoke_query(store, name)
            assert failing.rolled_back is True
        finally:
            store._connection = real_connection


@pytest.mark.parametrize(
    "name",
    (
        "session_belongs_to_project",
        "feedback_history",
        "list_feedback",
        "project_connections_disabled",
        "session_delete_requires_drain",
    ),
)
def test_query_sqlite_failures_are_normalized_as_store_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        failing = _FailingConnection(sqlite3.OperationalError("sqlite secret"))
        real_connection = store._connection
        monkeypatch.setattr(CaptureStore, "_begin_immediate", lambda _self: None)
        store._connection = failing  # type: ignore[assignment]
        try:
            with pytest.raises(CaptureStoreError):
                _invoke_query(store, name)
            assert failing.rolled_back is True
        finally:
            store._connection = real_connection


def test_feedback_history_covers_missing_session_project_and_state_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        register_connection(store)
        human_id = _closed_session(store)

        with pytest.raises(CaptureStoreStateError):
            store.feedback_history("a" * 12, project_digest=PROJECT_DIGEST)
        with pytest.raises(CaptureStoreStateError):
            store.feedback_history(human_id, project_digest="9" * 64)

        with monkeypatch.context() as patch:
            patch.setattr(CaptureStore, "_session_row", lambda *_args: None)
            with pytest.raises(CaptureStoreIntegrityError):
                store.feedback_history(human_id, project_digest=PROJECT_DIGEST)

        real_connection_row = CaptureStore._connection_row

        def deleting_connection(instance: CaptureStore, connection_id: str) -> dict[str, object]:
            row = dict(real_connection_row(instance, connection_id))
            row["state"] = "deleting"
            return row

        with monkeypatch.context() as patch:
            patch.setattr(CaptureStore, "_connection_row", deleting_connection)
            with pytest.raises(CaptureStoreStateError):
                store.feedback_history(human_id, project_digest=PROJECT_DIGEST)


@pytest.mark.parametrize(
    "freeze",
    (
        "not-a-datetime",
        datetime(2026, 1, 1),
        datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=1))),
    ),
)
def test_list_feedback_rejects_noncanonical_freeze_values(
    tmp_path: Path,
    freeze: object,
) -> None:
    with (
        initialized_store(tmp_path / "capture.sqlite3") as store,
        pytest.raises(CaptureStoreStateError),
    ):
        store.list_feedback(label_freeze=freeze)  # type: ignore[arg-type]


def test_list_feedback_covers_integrity_guards_for_authenticated_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        register_connection(store)
        human_id = _closed_session(store)
        store.record_feedback(
            human_id,
            CaptureFeedbackLabel.MEMORY_NEEDED,
            project_digest=PROJECT_DIGEST,
        )

        with monkeypatch.context() as patch:
            patch.setattr(CaptureStore, "_session_row", lambda *_args: None)
            with pytest.raises(CaptureStoreIntegrityError):
                store.list_feedback(project_digest=PROJECT_DIGEST)

        with monkeypatch.context() as patch:
            patch.setattr(CaptureStore, "_verified_feedback_rows", lambda *_args: ())
            with pytest.raises(CaptureStoreIntegrityError):
                store.list_feedback(project_digest=PROJECT_DIGEST)

        assert store.list_feedback(project_digest=PROJECT_DIGEST)[0].human_id == human_id


def test_session_query_integrity_guards_reject_disappearing_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        register_connection(store)
        human_id = _closed_session(store)

        with monkeypatch.context() as patch:
            patch.setattr(CaptureStore, "_session_row", lambda *_args: None)
            with pytest.raises(CaptureStoreIntegrityError):
                store.session_inventory()
            with pytest.raises(CaptureStoreIntegrityError):
                store.session_belongs_to_project(human_id, PROJECT_DIGEST)

        with pytest.raises(CaptureStoreStateError):
            store.session_inventory(profile_id="codex")  # type: ignore[arg-type]
        with pytest.raises(CaptureStoreStateError):
            store.session_belongs_to_project(
                human_id,
                PROJECT_DIGEST,
                include_deleted=1,  # type: ignore[arg-type]
            )


@pytest.mark.parametrize(
    ("result", "error"),
    (
        (None, CaptureStoreError),
        ((0, 0), CaptureStoreError),
        (("0", 0, 0), CaptureStoreError),
        ((1, 0, 0), CaptureStoreBusyError),
    ),
)
def test_checkpoint_rejects_malformed_or_busy_results(
    tmp_path: Path,
    result: object,
    error: type[Exception],
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        real_connection = store._connection
        store._connection = _ResultConnection(result)  # type: ignore[assignment]
        try:
            with pytest.raises(error):
                store._checkpoint_wal()
        finally:
            store._connection = real_connection


@pytest.mark.parametrize(
    ("code", "expected"),
    (
        (sqlite3.SQLITE_BUSY, CaptureStoreBusyError),
        (sqlite3.SQLITE_IOERR, CaptureStoreError),
        (None, CaptureStoreError),
    ),
)
def test_checkpoint_classifies_sqlite_failures(
    tmp_path: Path,
    code: int | None,
    expected: type[Exception],
) -> None:
    error = sqlite3.OperationalError("checkpoint secret")
    if code is not None:
        error.sqlite_errorcode = code
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        real_connection = store._connection
        store._connection = _FailingConnection(error)  # type: ignore[assignment]
        try:
            with pytest.raises(expected):
                store._checkpoint_wal()
        finally:
            store._connection = real_connection


@pytest.mark.parametrize("value", (None, (0,), ("1",), (0,)))
def test_secure_delete_requires_the_exact_sqlite_setting(
    tmp_path: Path,
    value: object,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        real_connection = store._connection
        store._connection = _ResultConnection(value)  # type: ignore[assignment]
        try:
            with pytest.raises(CaptureStoreError):
                store._enable_secure_delete()
        finally:
            store._connection = real_connection


@pytest.mark.parametrize("value", (None, (0,), ("1",)))
def test_data_version_requires_one_positive_integer(
    tmp_path: Path,
    value: object,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        real_connection = store._connection
        store._connection = _ResultConnection(value)  # type: ignore[assignment]
        try:
            with pytest.raises(CaptureStoreIntegrityError):
                store._database_data_version()
        finally:
            store._connection = real_connection


@pytest.mark.parametrize(
    "value",
    (
        None,
        "not-a-timestamp",
        datetime(2026, 1, 1),
    ),
)
def test_stored_timestamp_rejects_noncanonical_values(value: object) -> None:
    with pytest.raises(CaptureStoreIntegrityError):
        store_module._stored_timestamp(value)


def test_integrity_helper_is_immutable_and_rejects_invalid_material() -> None:
    integrity = store_module._CaptureStoreIntegrity(INSTALLATION_KEY)
    with pytest.raises(AttributeError):
        integrity.anything = object()  # type: ignore[attr-defined]
    with pytest.raises(CaptureStoreIntegrityError):
        integrity.tag("purpose", {"unsupported": object()})


def test_integrity_helper_requires_an_exact_installation_key() -> None:
    with pytest.raises(CaptureStoreError):
        store_module._CaptureStoreIntegrity(object())  # type: ignore[arg-type]


def test_hook_connection_rejects_a_non_profile_enum() -> None:
    with pytest.raises(CaptureStoreIntegrityError):
        store_module._CaptureHookConnection(
            connection_id=CONNECTION_ID,
            project_digest=PROJECT_DIGEST,
            profile_id="codex-hooks-v1",  # type: ignore[arg-type]
            capability_manifest_digest="0" * 64,
            host_version="0.1.0",
            state=store_module.CaptureConnectionState.ENABLED,
        )


def test_audit_read_only_rejects_inexact_key_path_and_pathlike_values(
    tmp_path: Path,
) -> None:
    with pytest.raises(CaptureStoreError):
        CaptureStore.audit_read_only(
            tmp_path / "capture.sqlite3",
            installation_key=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(CaptureStoreError):
        CaptureStore.audit_read_only("", installation_key=INSTALLATION_KEY)
    with pytest.raises(CaptureStoreError):
        CaptureStore.audit_read_only(object(), installation_key=INSTALLATION_KEY)  # type: ignore[arg-type]


def test_audit_read_only_requires_an_existing_database_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        store_module,
        "inspect_private_file_location",
        lambda _path: _AuditAuthorization(target_exists=False),
    )
    with pytest.raises(CaptureStoreError):
        CaptureStore.audit_read_only(
            tmp_path / "capture.sqlite3",
            installation_key=INSTALLATION_KEY,
        )


def test_audit_read_only_rejects_nonempty_durable_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def inspect(_path: object) -> _AuditAuthorization:
        nonlocal calls
        calls += 1
        if calls == 2:
            return _AuditAuthorization(target_exists=True, size=1)
        return _AuditAuthorization(target_exists=True, size=0)

    monkeypatch.setattr(store_module, "inspect_private_file_location", inspect)
    with pytest.raises(CaptureStoreError):
        CaptureStore.audit_read_only(
            tmp_path / "capture.sqlite3",
            installation_key=INSTALLATION_KEY,
        )


def test_audit_read_only_rejects_a_failed_sqlite_quick_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _AuditConnection(None)
    monkeypatch.setattr(
        store_module,
        "inspect_private_file_location",
        lambda _path: _AuditAuthorization(target_exists=True, size=0),
    )
    monkeypatch.setattr(store_module.sqlite3, "connect", lambda *_args, **_kwargs: connection)
    monkeypatch.setattr(store_module, "validate_capture_store_schema", lambda _value: None)
    with pytest.raises(CaptureStoreIntegrityError):
        CaptureStore.audit_read_only(
            tmp_path / "capture.sqlite3",
            installation_key=INSTALLATION_KEY,
        )
    assert connection.closed is True


def test_connection_configuration_rejects_non_wal_journal_mode() -> None:
    with pytest.raises(CaptureStoreError):
        CaptureStore._configure_connection(
            _ResultConnection(("delete",)),  # type: ignore[arg-type]
            busy_timeout_ms=100,
        )


def test_low_level_store_operations_normalize_sqlite_failures(tmp_path: Path) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        real_connection = store._connection
        try:
            store._connection = _FailingConnection(  # type: ignore[assignment]
                sqlite3.OperationalError("sqlite secret")
            )
            with pytest.raises(CaptureStoreIntegrityError):
                store._database_data_version()
            with pytest.raises(CaptureStoreError):
                store._begin_immediate()
        finally:
            store._connection = real_connection


def test_connection_lookup_reports_a_missing_row(tmp_path: Path) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        real_connection = store._connection
        store._connection = _ResultConnection(None)  # type: ignore[assignment]
        try:
            with pytest.raises(CaptureStoreStateError):
                store._connection_row(CONNECTION_ID)
        finally:
            store._connection = real_connection


def test_session_summary_rejects_non_boolean_storage_flags(tmp_path: Path) -> None:
    with (
        initialized_store(tmp_path / "capture.sqlite3") as store,
        pytest.raises(CaptureStoreIntegrityError),
    ):
        store._session_summary(  # type: ignore[arg-type]
            {},
            {"coverage_degraded": "0", "unattributed_drop": 0},
        )


def test_connection_queries_reject_non_string_identifiers(tmp_path: Path) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        with pytest.raises(CaptureStoreError):
            store.get_connection(object())  # type: ignore[arg-type]
        with pytest.raises(CaptureStoreError):
            store._get_hook_connection(object())  # type: ignore[arg-type]


def test_session_queries_reject_rows_that_disappear_mid_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        register_connection(store)
        human_id = _closed_session(store)
        monkeypatch.setattr(CaptureStore, "_session_row", lambda *_args: None)

        with pytest.raises(CaptureStoreIntegrityError):
            store.list_sessions(project_digest=PROJECT_DIGEST)
        with pytest.raises(CaptureStoreIntegrityError):
            store.session_by_human_id(human_id)


def test_latest_session_rejects_an_inexact_profile_type(tmp_path: Path) -> None:
    with (
        initialized_store(tmp_path / "capture.sqlite3") as store,
        pytest.raises(CaptureStoreStateError),
    ):
        store.latest_session(
            project_digest=PROJECT_DIGEST,
            profile_id="codex-hooks-v1",  # type: ignore[arg-type]
        )


def test_session_row_rejects_non_boolean_transport_storage() -> None:
    store = CaptureStore.__new__(CaptureStore)
    store._connection = _ResultConnection(  # type: ignore[assignment]
        {"transport_required": "0", "transport_head_tag": None}
    )
    with pytest.raises(CaptureStoreIntegrityError):
        store._session_row(CONNECTION_ID, "session")


def test_record_feedback_covers_missing_race_and_revision_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        register_connection(store)
        human_id = _closed_session(store)

        with pytest.raises(CaptureStoreStateError):
            store.record_feedback(
                "a" * 12,
                CaptureFeedbackLabel.MEMORY_NEEDED,
                project_digest=PROJECT_DIGEST,
            )

        with monkeypatch.context() as patch:
            patch.setattr(CaptureStore, "_session_row", lambda *_args: None)
            with pytest.raises(CaptureStoreIntegrityError):
                store.record_feedback(
                    human_id,
                    CaptureFeedbackLabel.MEMORY_NEEDED,
                    project_digest=PROJECT_DIGEST,
                )

        maximum_rows = tuple(
            {"label": CaptureFeedbackLabel.NOT_MEMORY_NEEDED.value}
            for _index in range(MAX_CAPTURE_FEEDBACK_REVISIONS_PER_SESSION)
        )
        with monkeypatch.context() as patch:
            patch.setattr(CaptureStore, "_verified_feedback_rows", lambda *_args: maximum_rows)
            with pytest.raises(CaptureStoreStateError):
                store.record_feedback(
                    human_id,
                    CaptureFeedbackLabel.MEMORY_NEEDED,
                    project_digest=PROJECT_DIGEST,
                )

        created = datetime(2026, 1, 1, tzinfo=UTC)
        real_stored_timestamp = store_module._stored_timestamp

        def selective_timestamp(value: object) -> datetime:
            if value is created:
                return created.replace(tzinfo=None)
            return real_stored_timestamp(value)

        with monkeypatch.context() as patch:
            patch.setattr(store_module, "_now", lambda: created)
            patch.setattr(store_module, "_stored_timestamp", selective_timestamp)
            with pytest.raises(CaptureStoreIntegrityError):
                store.record_feedback(
                    human_id,
                    CaptureFeedbackLabel.MEMORY_NEEDED,
                    project_digest=PROJECT_DIGEST,
                )


def test_record_feedback_rejects_a_disappearing_previous_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        register_connection(store)
        human_id = _closed_session(store)
        rows = (
            {
                "label": CaptureFeedbackLabel.MEMORY_NEEDED.value,
                "label_id": "0" * 64,
            },
        )
        monkeypatch.setattr(CaptureStore, "_verified_feedback_rows", lambda *_args: rows)
        real_connection = store._connection
        store._connection = _StatementOverrideConnection(  # type: ignore[assignment]
            real_connection,
            marker="SELECT * FROM feedback_labels",
            result=_ResultConnection(None),
        )
        try:
            with pytest.raises(CaptureStoreIntegrityError):
                store.record_feedback(
                    human_id,
                    CaptureFeedbackLabel.NOT_MEMORY_NEEDED,
                    project_digest=PROJECT_DIGEST,
                )
        finally:
            store._connection = real_connection


def test_record_feedback_rejects_a_lost_anchor_delete(
    tmp_path: Path,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        register_connection(store)
        human_id = _closed_session(store)
        store.record_feedback(
            human_id,
            CaptureFeedbackLabel.MEMORY_NEEDED,
            project_digest=PROJECT_DIGEST,
        )
        real_connection = store._connection
        store._connection = _StatementOverrideConnection(  # type: ignore[assignment]
            real_connection,
            marker="DELETE FROM feedback_labels",
            result=_RowCountResult(0),
        )
        try:
            with pytest.raises(CaptureStoreIntegrityError):
                store.record_feedback(
                    human_id,
                    CaptureFeedbackLabel.NOT_MEMORY_NEEDED,
                    project_digest=PROJECT_DIGEST,
                )
        finally:
            store._connection = real_connection


@pytest.mark.parametrize(
    ("failure", "expected"),
    (
        (sqlite3.OperationalError("feedback sqlite secret"), CaptureStoreError),
        (KeyboardInterrupt(), KeyboardInterrupt),
    ),
)
def test_record_feedback_classifies_storage_and_base_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    expected: type[BaseException],
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        register_connection(store)
        human_id = _closed_session(store)

        def fail(*_args: object) -> tuple[sqlite3.Row, ...]:
            raise failure

        monkeypatch.setattr(CaptureStore, "_verified_feedback_rows", fail)
        with pytest.raises(expected):
            store.record_feedback(
                human_id,
                CaptureFeedbackLabel.MEMORY_NEEDED,
                project_digest=PROJECT_DIGEST,
            )


def test_human_session_id_reports_exhausted_collision_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CaptureStore.__new__(CaptureStore)
    store._connection = _ResultConnection(  # type: ignore[assignment]
        {"connection_id": "other", "session_id": "other"}
    )
    monkeypatch.setattr(
        CaptureStore,
        "_encoded_human_session_id",
        lambda *_args: "a" * 12,
    )
    with pytest.raises(CaptureStoreIntegrityError):
        store._human_session_id(CONNECTION_ID, "session")


def test_connection_transition_rejects_a_lost_compare_and_swap(
    tmp_path: Path,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        register_connection(store)
        real_connection = store._connection
        store._connection = _StatementOverrideConnection(  # type: ignore[assignment]
            real_connection,
            marker="UPDATE connections SET state",
            result=_RowCountResult(0),
        )
        try:
            with pytest.raises(CaptureStoreStateError):
                store.transition_connection(
                    CONNECTION_ID,
                    expected_state=store_module.CaptureConnectionState.PENDING,
                    target_state=store_module.CaptureConnectionState.ENABLED,
                )
        finally:
            store._connection = real_connection


def test_missing_session_does_not_belong_to_a_project(tmp_path: Path) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        assert store.session_belongs_to_project("a" * 12, PROJECT_DIGEST) is False


def test_transport_head_rejects_an_inexact_authenticated_tag(tmp_path: Path) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        real_connection = store._connection
        store._connection = _ResultConnection(  # type: ignore[assignment]
            {"receipt_count": 0, "head_receipt_tag": None, "head_tag": object()}
        )
        try:
            with pytest.raises(CaptureStoreIntegrityError):
                store._transport_head_row(CONNECTION_ID, "session")
        finally:
            store._connection = real_connection


def test_transport_chain_rejects_inexact_flag_and_disappearing_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        with pytest.raises(CaptureStoreError):
            store._verify_transport_chain(
                CONNECTION_ID,
                "session",
                allow_pending_event_tail=1,  # type: ignore[arg-type]
            )
        monkeypatch.setattr(
            CaptureStore,
            "_connection_row",
            lambda *_args: {"profile_id": CaptureProfile.CODEX_HOOKS_V1.value},
        )
        monkeypatch.setattr(CaptureStore, "_session_row", lambda *_args: None)
        with pytest.raises(CaptureStoreIntegrityError):
            store._verify_transport_chain(CONNECTION_ID, "session")


@pytest.mark.parametrize(
    ("profile", "session", "rows", "head"),
    (
        (
            CaptureProfile.CODEX_HOOKS_V1.value,
            {"transport_required": 1},
            (),
            None,
        ),
        (
            CaptureProfile.OPENCODE_PLUGIN_V1.value,
            {"transport_required": 0},
            ({"unexpected": True},),
            None,
        ),
    ),
)
def test_transport_chain_rejects_profile_ledger_inconsistency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    session: dict[str, object],
    rows: tuple[object, ...],
    head: object,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        monkeypatch.setattr(CaptureStore, "_connection_row", lambda *_args: {"profile_id": profile})
        monkeypatch.setattr(CaptureStore, "_session_row", lambda *_args: session)
        monkeypatch.setattr(CaptureStore, "_transport_head_row", lambda *_args: head)
        real_connection = store._connection
        store._connection = _RowsConnection(rows)  # type: ignore[assignment]
        try:
            with pytest.raises(CaptureStoreIntegrityError):
                store._verify_transport_chain(CONNECTION_ID, "session")
        finally:
            store._connection = real_connection


def test_transport_chain_rejects_event_ordinal_and_duplicate_chunk_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        session = {"transport_required": 1, "transport_head_tag": "head", "coverage_degraded": 1}
        monkeypatch.setattr(
            CaptureStore,
            "_connection_row",
            lambda *_args: {"profile_id": CaptureProfile.OPENCODE_PLUGIN_V1.value},
        )
        monkeypatch.setattr(CaptureStore, "_session_row", lambda *_args: session)
        monkeypatch.setattr(
            CaptureStore,
            "_transport_head_row",
            lambda *_args: {"head_tag": "head", "receipt_count": 0, "head_receipt_tag": None},
        )
        real_connection = store._connection
        store._connection = _StatementRowsConnection(  # type: ignore[assignment]
            {
                "capture_transport_receipts": (),
                "capture_events": ({"event_tag": "event", "admission_source": "direct"},),
            }
        )
        monkeypatch.setattr(
            CaptureStore,
            "_load_verified_event",
            lambda *_args: SimpleNamespace(receipt_ordinal=2, event_tag="event"),
        )
        try:
            with pytest.raises(CaptureStoreIntegrityError):
                store._verify_transport_chain(CONNECTION_ID, "session")
        finally:
            store._connection = real_connection

        rows = (
            {
                "connection_id": CONNECTION_ID,
                "session_id": "session",
                "transport_ordinal": 1,
                "previous_receipt_tag": None,
                "batch_ref": "batch",
                "chunk_index": 0,
                "chunk_count": 1,
                "post_event_count": 0,
                "post_head_event_tag": None,
                "receipt_tag": "first",
            },
            {
                "connection_id": CONNECTION_ID,
                "session_id": "session",
                "transport_ordinal": 2,
                "previous_receipt_tag": "first",
                "batch_ref": "batch",
                "chunk_index": 0,
                "chunk_count": 1,
                "post_event_count": 0,
                "post_head_event_tag": None,
                "receipt_tag": "second",
            },
        )
        store._connection = _StatementRowsConnection(  # type: ignore[assignment]
            {"capture_transport_receipts": rows, "capture_events": ()}
        )
        monkeypatch.setattr(
            CaptureStore,
            "_load_verified_transport_receipt",
            lambda _self, row: row,
        )
        try:
            with pytest.raises(CaptureStoreIntegrityError):
                store._verify_transport_chain(CONNECTION_ID, "session")
        finally:
            store._connection = real_connection


def test_transport_chain_rejects_unreceipted_nonspool_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        session = {"transport_required": 1, "transport_head_tag": "head", "coverage_degraded": 0}
        monkeypatch.setattr(
            CaptureStore,
            "_connection_row",
            lambda *_args: {"profile_id": CaptureProfile.OPENCODE_PLUGIN_V1.value},
        )
        monkeypatch.setattr(CaptureStore, "_session_row", lambda *_args: session)
        monkeypatch.setattr(
            CaptureStore,
            "_transport_head_row",
            lambda *_args: {"head_tag": "head", "receipt_count": 0, "head_receipt_tag": None},
        )
        monkeypatch.setattr(
            CaptureStore,
            "_load_verified_event",
            lambda *_args: SimpleNamespace(receipt_ordinal=1, event_tag="event"),
        )
        real_connection = store._connection
        store._connection = _StatementRowsConnection(  # type: ignore[assignment]
            {
                "capture_transport_receipts": (),
                "capture_events": ({"event_tag": "event", "admission_source": "direct"},),
            }
        )
        try:
            with pytest.raises(CaptureStoreIntegrityError):
                store._verify_transport_chain(CONNECTION_ID, "session")
        finally:
            store._connection = real_connection


def test_feedback_verifier_rejects_label_and_metadata_corruption(tmp_path: Path) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        cases = (
            ({"label": "invalid"},),
            (
                {
                    "label": CaptureFeedbackLabel.MEMORY_NEEDED.value,
                    "created_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                    "connection_id": "wrong",
                },
            ),
        )
        real_connection = store._connection
        try:
            for revision_rows in cases:
                rows = (*revision_rows, {"label": CaptureFeedbackLabel.MEMORY_NEEDED.value})
                store._connection = _RowsConnection(rows)  # type: ignore[assignment]
                with pytest.raises(CaptureStoreIntegrityError):
                    store._verified_feedback_rows(CONNECTION_ID, "session")
        finally:
            store._connection = real_connection


def test_event_loader_rejects_nonbyte_payloads(tmp_path: Path) -> None:
    with (
        initialized_store(tmp_path / "capture.sqlite3") as store,
        pytest.raises(CaptureStoreIntegrityError),
    ):
        store._load_verified_event({"event_json": "invalid"})  # type: ignore[arg-type]


def test_chain_row_verifier_rejects_inventory_mismatch(tmp_path: Path) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        real_connection = store._connection
        store._connection = _RowsConnection(())  # type: ignore[assignment]
        try:
            with pytest.raises(CaptureStoreIntegrityError):
                store._verify_chain_rows(
                    CONNECTION_ID,
                    "session",
                    {"event_count": 1},  # type: ignore[arg-type]
                    {"receipt_count": 1, "head_event_tag": None},  # type: ignore[arg-type]
                )
        finally:
            store._connection = real_connection


def test_append_commitment_rejects_count_latest_and_event_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        with pytest.raises(CaptureStoreIntegrityError):
            store._verify_append_commitment(
                CONNECTION_ID,
                "session",
                {"event_count": 1},  # type: ignore[arg-type]
                {"receipt_count": 0},  # type: ignore[arg-type]
            )

        key = (CONNECTION_ID, "session")
        store._mode = store_module.CaptureStoreMode.HOOK
        monkeypatch.setattr(
            CaptureStore,
            "_database_data_version",
            lambda self: self._verified_data_version,
        )
        monkeypatch.setattr(CaptureStore, "_verify_health_set", lambda *_args: ())
        real_connection = store._connection

        store._verified_append_heads[key] = (0, None)
        store._connection = _SequenceConnection(  # type: ignore[assignment]
            ({"event_count": 0}, {"receipt_ordinal": 1})
        )
        try:
            with pytest.raises(CaptureStoreIntegrityError):
                store._verify_append_commitment(
                    *key,
                    {"event_count": 0},  # type: ignore[arg-type]
                    {"receipt_count": 0, "head_event_tag": None},  # type: ignore[arg-type]
                )
        finally:
            store._connection = real_connection

        store._verified_append_heads[key] = (1, "head")
        store._connection = _SequenceConnection(  # type: ignore[assignment]
            ({"event_count": 1}, None)
        )
        try:
            with pytest.raises(CaptureStoreIntegrityError):
                store._verify_append_commitment(
                    *key,
                    {"event_count": 1},  # type: ignore[arg-type]
                    {"receipt_count": 1, "head_event_tag": "head"},  # type: ignore[arg-type]
                )
        finally:
            store._connection = real_connection

        store._verified_append_heads[key] = (1, "head")
        store._connection = _SequenceConnection(  # type: ignore[assignment]
            ({"event_count": 1}, {"receipt_ordinal": 1, "event_tag": "head"})
        )
        monkeypatch.setattr(
            CaptureStore,
            "_load_verified_event",
            lambda *_args: SimpleNamespace(
                intake=SimpleNamespace(connection_id="wrong", session_id="session"),
                receipt_ordinal=1,
                event_tag="head",
            ),
        )
        try:
            with pytest.raises(CaptureStoreIntegrityError):
                store._verify_append_commitment(
                    *key,
                    {"event_count": 1},  # type: ignore[arg-type]
                    {"receipt_count": 1, "head_event_tag": "head"},  # type: ignore[arg-type]
                )
        finally:
            store._connection = real_connection


def test_full_state_verification_requires_an_exact_immutable_flag(tmp_path: Path) -> None:
    with (
        initialized_store(tmp_path / "capture.sqlite3") as store,
        pytest.raises(CaptureStoreError),
    ):
        store._verify_all_state(immutable=1)  # type: ignore[arg-type]


def test_deleted_session_lookup_rejects_missing_and_ambiguous_tombstones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = (
        {"connection_id": "one", "session_id": "first"},
        {"connection_id": "two", "session_id": "second"},
    )
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        real_connection = store._connection
        store._connection = _RowsConnection(identities[:1])  # type: ignore[assignment]
        monkeypatch.setattr(CaptureStore, "_tombstone_row", lambda *_args: None)
        try:
            with pytest.raises(CaptureStoreIntegrityError):
                store._deleted_session_for_human_id("a" * 12)
        finally:
            store._connection = real_connection

        store._connection = _RowsConnection(identities)  # type: ignore[assignment]
        monkeypatch.setattr(
            CaptureStore,
            "_tombstone_row",
            lambda _self, connection_id, session_id: {
                "connection_id": connection_id,
                "session_id": session_id,
            },
        )
        monkeypatch.setattr(CaptureStore, "_encoded_human_session_id", lambda *_args: "a" * 52)
        try:
            with pytest.raises(CaptureStoreIntegrityError):
                store._deleted_session_for_human_id("a" * 12)
        finally:
            store._connection = real_connection


def test_secure_delete_normalizes_sqlite_failures(tmp_path: Path) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        real_connection = store._connection
        store._connection = _FailingConnection(  # type: ignore[assignment]
            sqlite3.OperationalError("secure delete secret")
        )
        try:
            with pytest.raises(CaptureStoreError):
                store._enable_secure_delete()
        finally:
            store._connection = real_connection


def test_session_delete_queries_cover_missing_and_disappearing_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        with pytest.raises(CaptureStoreStateError):
            store._session_delete_requires_drain("a" * 12)
        with pytest.raises(CaptureStoreStateError):
            store._delete_session("a" * 12)

        register_connection(store)
        human_id = _closed_session(store)
        monkeypatch.setattr(CaptureStore, "_session_row", lambda *_args: None)
        with pytest.raises(CaptureStoreIntegrityError):
            store._session_delete_requires_drain(human_id)
        with pytest.raises(CaptureStoreIntegrityError):
            store._delete_session(human_id)


def test_session_creation_and_update_reject_disappearing_or_lost_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        register_connection(store)
        intake = authenticated_intake("session_started", session_native=b"create-race")
        with monkeypatch.context() as patch:
            patch.setattr(CaptureStore, "_session_row", lambda *_args: None)
            with pytest.raises(CaptureStoreIntegrityError):
                store._create_session(intake)
            store._rollback()

        human_id = _closed_session(store, native=b"update-race")
        summary = store.session_by_human_id(human_id)
        row = store._session_row(CONNECTION_ID, summary.session_id)
        assert row is not None
        real_connection = store._connection
        store._connection = _StatementOverrideConnection(  # type: ignore[assignment]
            real_connection,
            marker="UPDATE capture_sessions",
            result=_RowCountResult(0),
        )
        try:
            with pytest.raises(CaptureStoreIntegrityError):
                store._update_session(
                    row,
                    state=store_module.CaptureSessionState.CLOSED,
                    event_count=row["event_count"],
                    coverage_degraded=False,
                )
        finally:
            store._connection = real_connection


def test_transport_fallback_health_and_public_health_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        register_connection(store)
        human_id = _closed_session(store)
        summary = store.session_by_human_id(human_id)
        row = store._session_row(CONNECTION_ID, summary.session_id)
        assert row is not None
        with pytest.raises(CaptureStoreIntegrityError):
            store._mark_transport_fallback_session(row)

        monkeypatch.setattr(CaptureStore, "_session_row", lambda *_args: None)
        with pytest.raises(CaptureStoreIntegrityError):
            store._record_health(
                connection_id=CONNECTION_ID,
                session_id=summary.session_id,
                code=CaptureHealthCode.COVERAGE_DEGRADED,
            )
        with pytest.raises(CaptureStoreError):
            store.mark_session_health(
                object(),  # type: ignore[arg-type]
                summary.session_id,
                CaptureHealthCode.COVERAGE_DEGRADED,
            )


def test_mark_session_health_requires_an_enabled_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialized_store(tmp_path / "capture.sqlite3") as store:
        monkeypatch.setattr(
            CaptureStore,
            "_connection_row",
            lambda *_args: {"state": store_module.CaptureConnectionState.PENDING.value},
        )
        with pytest.raises(CaptureStoreStateError):
            store.mark_session_health(
                CONNECTION_ID,
                "a" * 64,
                CaptureHealthCode.COVERAGE_DEGRADED,
            )
