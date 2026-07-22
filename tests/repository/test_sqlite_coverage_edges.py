"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from tests.repository.conformance import (
    CONDITIONAL_EVENT_ID_A,
    RUN_A,
    RUN_B,
    event_draft,
)
from tests.repository.sqlite_support import repository as sqlite_repository
from tests.repository.test_memory_repository import create_repository

from saliencegate.ports.repository import (
    ConditionalEventAppend,
    DigestVerificationError,
    InvalidRecordError,
    InvalidRecordTypeError,
)
from saliencegate.repository import sqlite as sqlite_module
from saliencegate.repository.memory import MemoryRunRepository, _RunSlot, _VerifiedRunState
from saliencegate.repository.projector import empty_projection
from saliencegate.repository.sqlite import SQLiteRepositoryError, SQLiteRunRepository
from saliencegate.security import SecureFileError


class _Cursor:
    def __init__(self, row: object) -> None:
        self._row = row

    def fetchone(self) -> object:
        return self._row


class _SequenceConnection:
    def __init__(self, *rows: object, error: sqlite3.Error | None = None) -> None:
        self._rows = list(rows)
        self._error = error
        self.closed = False

    def execute(self, _statement: str, _parameters: object = ()) -> _Cursor:
        if self._error is not None:
            raise self._error
        return _Cursor(self._rows.pop(0) if self._rows else None)

    def close(self) -> None:
        self.closed = True


def _shell(connection: object) -> SQLiteRunRepository:
    repository = SQLiteRunRepository.__new__(SQLiteRunRepository)
    repository._connection = connection  # type: ignore[assignment]
    return repository


def test_authorized_constructor_requires_exact_authorization_type() -> None:
    with pytest.raises(TypeError, match="exactly StableFileAuthorization"):
        SQLiteRunRepository._from_file_authorization(object())  # type: ignore[arg-type]


async def test_close_rejects_a_locked_repository(tmp_path: Path) -> None:
    repository = sqlite_repository(tmp_path / "locked-close.sqlite3")
    await repository._lock.acquire()
    try:
        with pytest.raises(SQLiteRepositoryError):
            repository.close()
    finally:
        repository._lock.release()
        repository.close()


async def test_aclose_is_idempotent_after_sync_close(tmp_path: Path) -> None:
    repository = sqlite_repository(tmp_path / "idempotent-close.sqlite3")
    repository.close()

    await repository.aclose()


def test_close_connection_reports_revalidation_failure(tmp_path: Path) -> None:
    repository = sqlite_repository(tmp_path / "boundary-close.sqlite3")

    class Authorization:
        cleaned = False

        def revalidate(self) -> None:
            raise SecureFileError()

        def _cleanup_created_sqlite_sidecars(self) -> None:
            self.cleaned = True

    authorization = Authorization()
    repository._file_authorization = authorization  # type: ignore[assignment]

    with pytest.raises(SecureFileError):
        repository._close_connection()
    assert authorization.cleaned
    assert repository._closed


def test_close_connection_normalizes_sqlite_error() -> None:
    class Connection:
        def close(self) -> None:
            raise sqlite3.OperationalError("close failed")

    repository = _shell(Connection())
    repository._file_authorization = None

    with pytest.raises(SQLiteRepositoryError):
        repository._close_connection()


async def test_run_database_preserves_non_cancellation_failure() -> None:
    repository = SQLiteRunRepository.__new__(SQLiteRunRepository)

    def fail() -> None:
        raise RuntimeError("database failed")

    with pytest.raises(RuntimeError, match="database failed"):
        await repository._run_database(fail)


async def test_run_database_restores_pending_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = SQLiteRunRepository.__new__(SQLiteRunRepository)
    calls = 0

    def fail() -> None:
        raise RuntimeError("database failed")

    async def interrupt_once(task: asyncio.Task[None]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise asyncio.CancelledError
        await task

    monkeypatch.setattr(sqlite_module.asyncio, "shield", interrupt_once)

    with pytest.raises(asyncio.CancelledError):
        await repository._run_database(fail)
    assert calls == 2


def test_configure_connection_requires_wal_for_disk_database() -> None:
    connection = _SequenceConnection(None, None, None)
    repository = _shell(connection)
    repository._database = "coverage.sqlite3"
    repository._busy_timeout_ms = 5_000

    with pytest.raises(SQLiteRepositoryError):
        repository._configure_connection()


@pytest.mark.parametrize(
    "rows",
    [
        (None, (5_000,), (2,)),
        ((0,), (5_000,), (2,)),
        ((1,), None, (2,)),
        ((1,), (4_999,), (2,)),
        ((1,), (5_000,), None),
        ((1,), (5_000,), (1,)),
    ],
)
def test_configure_connection_rejects_incorrect_pragma_state(rows: tuple[object, ...]) -> None:
    connection = _SequenceConnection(None, None, None, *rows)
    repository = _shell(connection)
    repository._database = ":memory:"
    repository._busy_timeout_ms = 5_000

    with pytest.raises(SQLiteRepositoryError):
        repository._configure_connection()


def test_canonical_run_id_normalizes_invalid_uuid() -> None:
    with pytest.raises(DigestVerificationError, match="durable run catalog"):
        SQLiteRunRepository._canonical_run_id({"run_id": "not-a-uuid"})  # type: ignore[arg-type]


async def _state_with_events(
    count: int,
    repository: MemoryRunRepository | None = None,
) -> tuple[MemoryRunRepository, _VerifiedRunState]:
    repository = create_repository() if repository is None else repository
    await repository.append(event_draft(source_event_id="sqlite-state-one"))
    if count > 1:
        await repository.append(
            event_draft(
                source_event_id="sqlite-state-two",
                timestamp=event_draft().timestamp.replace(microsecond=1),
            )
        )
    return repository, await repository._verified_state(RUN_A)


async def test_install_replayed_runs_requires_empty_engine() -> None:
    source, state = await _state_with_events(1)
    replayed = source._replay_run(state.ledger, state.ledger_head)
    target = create_repository()
    target._slots[RUN_A] = _RunSlot(run_id=RUN_A)

    with pytest.raises(DigestVerificationError, match="batch replay target"):
        SQLiteRunRepository._install_replayed_runs(target, {RUN_A: replayed})


async def test_install_replayed_runs_rejects_mapping_key_mismatch() -> None:
    source, state = await _state_with_events(1)
    replayed = source._replay_run(state.ledger, state.ledger_head)

    with pytest.raises(DigestVerificationError, match="batch replay target"):
        SQLiteRunRepository._install_replayed_runs(create_repository(), {RUN_B: replayed})


async def test_anchor_recheck_rejects_same_length_changed_head(tmp_path: Path) -> None:
    repository = sqlite_repository(tmp_path / "anchor-check.sqlite3")
    _source, state = await _state_with_events(1)
    repository._anchors[RUN_A] = state
    changed = replace(
        state,
        ledger_head=state.ledger_head.model_copy(
            update={"head_tag": state.ledger_head.head_tag.model_copy(update={"value": "0" * 64})}
        ),
    )
    try:
        with pytest.raises(DigestVerificationError, match="durable ledger rollback"):
            repository._check_and_remember_anchors({RUN_A: changed})
    finally:
        repository.close()


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (DigestVerificationError("catalog"), DigestVerificationError),
        (sqlite3.OperationalError("catalog"), SQLiteRepositoryError),
    ],
)
def test_read_batch_transaction_normalizes_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    expected: type[BaseException],
) -> None:
    repository = sqlite_repository(tmp_path / f"batch-{type(failure).__name__}.sqlite3")

    def fail(*_args: object, **_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(repository, "_read_durable_catalog", fail)
    try:
        with pytest.raises(expected):
            repository._read_batch_transaction()
    finally:
        repository.close()


def test_database_version_normalizes_sqlite_error() -> None:
    repository = _shell(_SequenceConnection(error=sqlite3.OperationalError("read failed")))

    with pytest.raises(SQLiteRepositoryError):
        repository._database_version(RUN_A)


def test_database_version_rejects_run_without_head() -> None:
    repository = _shell(_SequenceConnection(None, (1,)))

    with pytest.raises(DigestVerificationError, match="durable run catalog"):
        repository._database_version(RUN_A)


def test_database_version_rejects_head_without_run() -> None:
    row = {"entry_count": 1, "algorithm": "hmac_sha256", "head_tag": "0" * 64}
    repository = _shell(_SequenceConnection(row, None))

    with pytest.raises(DigestVerificationError, match="durable run catalog"):
        repository._database_version(RUN_A)


def test_database_version_rejects_invalid_head_fields() -> None:
    row = {"entry_count": 1, "algorithm": "invalid", "head_tag": "0" * 64}
    repository = _shell(_SequenceConnection(row, (1,)))

    with pytest.raises(DigestVerificationError, match="durable ledger head"):
        repository._database_version(RUN_A)


async def test_authenticate_batch_candidate_rejects_prefix_regression(tmp_path: Path) -> None:
    repository = sqlite_repository(tmp_path / "candidate-prefix.sqlite3")
    _source, state = await _state_with_events(1)
    candidate = replace(state, ledger=())
    try:
        with pytest.raises(DigestVerificationError, match="candidate ledger fork"):
            repository._authenticate_batch_candidate(candidate, state)
    finally:
        repository.close()


async def test_authenticate_batch_candidate_rejects_cross_run_base(tmp_path: Path) -> None:
    repository = sqlite_repository(tmp_path / "candidate-run.sqlite3")
    _source, state = await _state_with_events(1)
    candidate = replace(state, run_id=RUN_B)
    try:
        with pytest.raises(DigestVerificationError, match="candidate ledger fork"):
            repository._authenticate_batch_candidate(candidate, state)
    finally:
        repository.close()


async def test_authenticate_batch_candidate_rejects_empty_new_run(tmp_path: Path) -> None:
    repository = sqlite_repository(tmp_path / "candidate-empty.sqlite3")
    _source, state = await _state_with_events(1)
    candidate = replace(
        state,
        ledger=(),
        projection=empty_projection(RUN_A),
    )
    try:
        with pytest.raises(DigestVerificationError, match="batch candidate"):
            repository._authenticate_batch_candidate(candidate, None)
    finally:
        repository.close()


async def test_authenticate_batch_candidate_rejects_cross_run_suffix(tmp_path: Path) -> None:
    repository = sqlite_repository(tmp_path / "candidate-suffix.sqlite3")
    _base_source, base = await _state_with_events(1)
    _candidate_source, candidate = await _state_with_events(2)
    suffix = candidate.ledger[-1].model_copy(update={"run_id": RUN_B})
    candidate = replace(candidate, ledger=(*candidate.ledger[:-1], suffix))
    try:
        with pytest.raises(DigestVerificationError, match="batch candidate"):
            repository._authenticate_batch_candidate(candidate, base)
    finally:
        repository.close()


async def test_authenticate_batch_candidate_rejects_mismatched_head(tmp_path: Path) -> None:
    repository = sqlite_repository(tmp_path / "candidate-head.sqlite3")
    _source, candidate = await _state_with_events(1, repository._new_engine())
    candidate = replace(
        candidate,
        ledger_head=candidate.ledger_head.model_copy(update={"entry_count": 2}),
    )
    try:
        with pytest.raises(DigestVerificationError, match="batch candidate"):
            repository._authenticate_batch_candidate(candidate, None)
    finally:
        repository.close()


async def test_authenticate_batch_candidate_rejects_projection_mismatch(tmp_path: Path) -> None:
    repository = sqlite_repository(tmp_path / "candidate-projection.sqlite3")
    _source, candidate = await _state_with_events(1, repository._new_engine())
    candidate = replace(candidate, projection=empty_projection(RUN_A))
    try:
        with pytest.raises(DigestVerificationError, match="batch candidate"):
            repository._authenticate_batch_candidate(candidate, None)
    finally:
        repository.close()


def test_copy_conditional_operations_rejects_unknown_operation() -> None:
    with pytest.raises(InvalidRecordTypeError, match="conditional_batch"):
        SQLiteRunRepository._copy_conditional_operations((object(),))


def test_copy_conditional_operations_normalizes_detachment_failure() -> None:
    valid = ConditionalEventAppend(
        event=event_draft(source_event_id="sqlite-copy"),
        event_id=CONDITIONAL_EVENT_ID_A,
    )
    forged = valid.model_copy(update={"event": object()})

    with pytest.raises(InvalidRecordError, match="conditional_batch"):
        SQLiteRunRepository._copy_conditional_operations((forged,))
