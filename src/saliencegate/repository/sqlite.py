from __future__ import annotations

import asyncio
import os
import sqlite3
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import TypeVar
from uuid import UUID

from pydantic import ValidationError

from saliencegate.domain import (
    BudgetSnapshot,
    CycleRecord,
    DeliveryRecord,
    InterventionOutcome,
    InvocationDecision,
    NormalizedTraceEventDraft,
    PayloadDigest,
    PayloadDigestAlgorithm,
    Signal,
    canonical_json,
    new_repository_id,
)
from saliencegate.ports.repository import (
    MAX_CONDITIONAL_BATCH_OPERATIONS,
    AppendReceipt,
    BeginCycle,
    BeginDeliveryAttempt,
    ClaimDelivery,
    CommitCycle,
    CompleteDelivery,
    ConditionalAppendOperation,
    ConditionalBatchReceipt,
    ConditionalEventAppend,
    ConditionalSignalAppend,
    CycleReceipt,
    CycleRecoveryReceipt,
    DeliveryAttemptReceipt,
    DeliveryRecoveryReceipt,
    DeliveryTransitionReceipt,
    DigestVerificationError,
    FailCycle,
    InvalidRecordError,
    InvalidRecordTypeError,
    LedgerEntry,
    LedgerHead,
    LedgerHeadConflictError,
    LedgerReceipt,
    MarkDeliveryUnknown,
    MemoryDeltaPreview,
    MemoryHit,
    MemoryQuery,
    MemorySnapshot,
    PreviewMemoryDelta,
    RebuildReceipt,
    RejectDelivery,
    RepositoryError,
    ReserveCycle,
    StartCycle,
)
from saliencegate.repository.integrity import IntegrityContext
from saliencegate.repository.memory import (
    MemoryRunRepository,
    _ReplayedRun,
    _RunSlot,
    _VerifiedRunState,
)
from saliencegate.repository.migrations import MigrationError, apply_migrations
from saliencegate.repository.projector import (
    apply_entry,
    empty_projection,
    projection_digests,
    validate_complete_projection,
)
from saliencegate.security import (
    InstallationKey,
    RedactionPolicy,
    SecureFileError,
    StableFileAuthorization,
    load_or_create_installation_key,
)

ResultT = TypeVar("ResultT")

_STATE_SCHEMA_VERSION = 1
_STATE_ROOT_DOMAIN = "saliencegate:sqlite-state-root:v1"
_DEFAULT_BUSY_TIMEOUT_MS = 5_000
_MAX_CAS_ATTEMPTS = 8


class SQLiteRepositoryError(RepositoryError):
    """A sanitized SQLite boundary failure."""

    def __init__(self) -> None:
        super().__init__("SQLite repository operation failed")


class ConcurrentWriteError(SQLiteRepositoryError):
    """The bounded optimistic retry loop could not publish a candidate."""

    def __init__(self) -> None:
        RepositoryError.__init__(self, "concurrent SQLite update could not be committed")


class ClosedSQLiteRepositoryError(SQLiteRepositoryError):
    def __init__(self) -> None:
        RepositoryError.__init__(self, "SQLite repository is closed")


class _ConcurrentMutationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _RunVersion:
    entry_count: int
    algorithm: PayloadDigestAlgorithm
    head_tag: str

    @classmethod
    def from_head(cls, head: LedgerHead) -> _RunVersion:
        return cls(
            entry_count=head.entry_count,
            algorithm=head.head_tag.algorithm,
            head_tag=head.head_tag.value,
        )


@dataclass(frozen=True, slots=True)
class _ProjectionSources:
    signal_positions: Mapping[UUID, int]
    decision_positions: Mapping[UUID, int]
    cycle_positions: Mapping[tuple[str, int], int]
    memory_cycles: Mapping[tuple[UUID, int], tuple[str, int]]
    intervention_cycles: Mapping[UUID, tuple[str, int]]
    outcome_positions: Mapping[UUID, int]
    delivery_positions: Mapping[tuple[UUID, int], int]


class SQLiteRunRepository:
    """Ledger-first SQLite repository with replayable, sacrificial projections.

    The authenticated ledger and its head are authoritative. All projection tables,
    including FTS5, are rewritten from verified replay and may safely be discarded.
    Instances are intended for one event loop; separate instances coordinate through
    an optimistic, cross-process compare-and-swap on the ledger head.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        redaction_policy: RedactionPolicy | None = None,
        installation_key: InstallationKey | None = None,
        synthetic_benchmark: bool = False,
        id_factory: Callable[[], UUID] = new_repository_id,
        busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        self._initialize(
            path,
            redaction_policy=redaction_policy,
            installation_key=installation_key,
            synthetic_benchmark=synthetic_benchmark,
            id_factory=id_factory,
            busy_timeout_ms=busy_timeout_ms,
            file_authorization=None,
        )

    @classmethod
    def _from_file_authorization(
        cls,
        authorization: StableFileAuthorization,
        *,
        redaction_policy: RedactionPolicy | None = None,
        installation_key: InstallationKey | None = None,
        synthetic_benchmark: bool = False,
        id_factory: Callable[[], UUID] = new_repository_id,
        busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS,
    ) -> SQLiteRunRepository:
        """Open an already authorized on-disk database without create fallback."""

        if type(authorization) is not StableFileAuthorization:
            raise TypeError("authorization must be exactly StableFileAuthorization")
        repository = cls.__new__(cls)
        repository._initialize(
            authorization.path,
            redaction_policy=redaction_policy,
            installation_key=installation_key,
            synthetic_benchmark=synthetic_benchmark,
            id_factory=id_factory,
            busy_timeout_ms=busy_timeout_ms,
            file_authorization=authorization,
        )
        return repository

    def _initialize(
        self,
        path: str | os.PathLike[str],
        *,
        redaction_policy: RedactionPolicy | None,
        installation_key: InstallationKey | None,
        synthetic_benchmark: bool,
        id_factory: Callable[[], UUID],
        busy_timeout_ms: int,
        file_authorization: StableFileAuthorization | None,
    ) -> None:
        if redaction_policy is not None and type(redaction_policy) is not RedactionPolicy:
            raise TypeError("redaction_policy must be exactly RedactionPolicy")
        if type(busy_timeout_ms) is not int or not 1 <= busy_timeout_ms <= 60_000:
            raise ValueError("busy_timeout_ms must be an integer from 1 through 60000")
        if installation_key is None and not synthetic_benchmark:
            installation_key = load_or_create_installation_key()
        self._integrity = IntegrityContext(
            key=installation_key,
            synthetic_benchmark=synthetic_benchmark,
        )
        self._policy = (
            RedactionPolicy()
            if redaction_policy is None
            else RedactionPolicy(
                literal_secrets=redaction_policy.literal_secrets,
                structured_field_names=redaction_policy.structured_field_names,
            )
        )
        self._id_factory = id_factory
        self._busy_timeout_ms = busy_timeout_ms
        try:
            self._database = os.fspath(path)
        except TypeError:
            raise TypeError("path must be a string or path-like value") from None
        if not isinstance(self._database, str) or not self._database:
            raise ValueError("path must identify a SQLite database")
        self._lock = asyncio.Lock()
        self._closed = False
        self._anchors: dict[UUID, _VerifiedRunState] = {}
        self._file_authorization = file_authorization
        try:
            if file_authorization is None:
                self._connection = sqlite3.connect(
                    self._database,
                    timeout=busy_timeout_ms / 1_000,
                    isolation_level=None,
                    check_same_thread=False,
                )
            else:
                file_authorization._revalidate_before_sqlite_statements()
                database_uri = f"{Path(file_authorization.path).as_uri()}?mode=rw"
                self._connection = sqlite3.connect(
                    database_uri,
                    timeout=busy_timeout_ms / 1_000,
                    isolation_level=None,
                    check_same_thread=False,
                    uri=True,
                )
                file_authorization._revalidate_mutable_sqlite()
            self._connection.row_factory = sqlite3.Row
            self._configure_connection()
            apply_migrations(self._connection)
            self._recover_derived_state()
            if file_authorization is not None:
                file_authorization._revalidate_mutable_sqlite()
        except (MigrationError, RepositoryError, ValidationError):
            self._abort_initialization()
            raise
        except sqlite3.Error:
            self._abort_initialization()
            raise SQLiteRepositoryError() from None
        except BaseException:
            self._abort_initialization()
            raise

    def _abort_initialization(self) -> None:
        connection = getattr(self, "_connection", None)
        connection_closed = True
        if isinstance(connection, sqlite3.Connection):
            try:
                connection.close()
            except sqlite3.Error:
                connection_closed = False
        authorization = getattr(self, "_file_authorization", None)
        if connection_closed and isinstance(authorization, StableFileAuthorization):
            authorization._cleanup_created_sqlite_sidecars()

    def __repr__(self) -> str:
        return f"SQLiteRunRepository(closed={self._closed})"

    def __enter__(self) -> SQLiteRunRepository:
        self._ensure_open()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self) -> SQLiteRunRepository:
        self._ensure_open()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    def close(self) -> None:
        if self._closed:
            return
        if self._lock.locked():
            raise SQLiteRepositoryError()
        self._close_connection()

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._close_connection()

    def _close_connection(self) -> None:
        authorization = self._file_authorization
        boundary_failed = False
        if authorization is not None:
            try:
                authorization._revalidate_mutable_sqlite()
            except SecureFileError:
                boundary_failed = True
        try:
            self._connection.close()
        except sqlite3.Error:
            raise SQLiteRepositoryError() from None
        self._closed = True
        if authorization is not None:
            authorization._cleanup_created_sqlite_sidecars()
        if boundary_failed:
            raise SecureFileError()

    def _ensure_open(self) -> None:
        if self._closed:
            raise ClosedSQLiteRepositoryError()

    def _rollback_best_effort(self) -> None:
        with suppress(sqlite3.Error):
            self._connection.rollback()

    async def _run_database(self, operation: Callable[[], ResultT]) -> ResultT:
        task = asyncio.create_task(asyncio.to_thread(operation))
        cancellation_received = False
        while True:
            try:
                result = await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                cancellation_received = True
                continue
            except BaseException:
                if cancellation_received:
                    raise asyncio.CancelledError from None
                raise
            if cancellation_received:
                raise asyncio.CancelledError
            return result

    def _configure_connection(self) -> None:
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        if self._database != ":memory:":
            journal_row = self._connection.execute("PRAGMA journal_mode = WAL").fetchone()
            if journal_row is None or str(journal_row[0]).casefold() != "wal":
                raise SQLiteRepositoryError()
        self._connection.execute("PRAGMA synchronous = FULL")
        foreign_keys = self._connection.execute("PRAGMA foreign_keys").fetchone()
        busy_timeout = self._connection.execute("PRAGMA busy_timeout").fetchone()
        synchronous = self._connection.execute("PRAGMA synchronous").fetchone()
        if (
            foreign_keys is None
            or foreign_keys[0] != 1
            or busy_timeout is None
            or busy_timeout[0] != self._busy_timeout_ms
            or synchronous is None
            or synchronous[0] != 2
        ):
            raise SQLiteRepositoryError()

    def _new_engine(self) -> MemoryRunRepository:
        return MemoryRunRepository(
            redaction_policy=self._policy,
            installation_key=self._integrity.key,
            synthetic_benchmark=self._integrity.synthetic_benchmark,
            id_factory=self._id_factory,
        )

    @staticmethod
    def _blob(row: sqlite3.Row, column: str) -> bytes:
        value = row[column]
        if type(value) is not bytes:
            raise DigestVerificationError("durable ledger encoding")
        return value

    @staticmethod
    def _text(row: sqlite3.Row, column: str) -> str:
        value = row[column]
        if type(value) is not str:
            raise DigestVerificationError("durable ledger encoding")
        return value

    @classmethod
    def _canonical_run_id(cls, row: sqlite3.Row) -> UUID:
        raw = cls._text(row, "run_id")
        try:
            run_id = UUID(raw)
        except ValueError:
            raise DigestVerificationError("durable run catalog") from None
        if raw != str(run_id):
            raise DigestVerificationError("durable run catalog")
        return run_id

    def _head_from_row(self, row: sqlite3.Row) -> LedgerHead:
        try:
            algorithm = PayloadDigestAlgorithm(self._text(row, "algorithm"))
            run_id = UUID(self._text(row, "run_id"))
            return LedgerHead(
                run_id=run_id,
                entry_count=row["entry_count"],
                chain_tag=PayloadDigest(
                    algorithm=algorithm,
                    value=self._text(row, "chain_tag"),
                ),
                projection_tag=PayloadDigest(
                    algorithm=algorithm,
                    value=self._text(row, "projection_tag"),
                ),
                head_tag=PayloadDigest(
                    algorithm=algorithm,
                    value=self._text(row, "head_tag"),
                ),
            )
        except (TypeError, ValueError, ValidationError):
            raise DigestVerificationError("durable ledger head") from None

    def _entry_from_row(self, row: sqlite3.Row) -> LedgerEntry:
        encoded = self._blob(row, "entry_json")
        try:
            entry = LedgerEntry.model_validate_json(encoded)
            if canonical_json(entry) != encoded:
                raise ValueError
            previous_algorithm = (
                None
                if row["previous_chain_algorithm"] is None
                else PayloadDigestAlgorithm(self._text(row, "previous_chain_algorithm"))
            )
            previous_tag = entry.previous_chain_tag
            expected_previous_algorithm = None if previous_tag is None else previous_tag.algorithm
            expected_previous_value = None if previous_tag is None else previous_tag.value
            sidecars_match = (
                self._text(row, "run_id") == str(entry.run_id)
                and row["position"] == entry.position
                and self._text(row, "record_key") == entry.record_key
                and self._text(row, "record_type") == entry.record.record_type
                and PayloadDigestAlgorithm(self._text(row, "record_algorithm"))
                is entry.record_tag.algorithm
                and self._text(row, "record_tag") == entry.record_tag.value
                and previous_algorithm is expected_previous_algorithm
                and row["previous_chain_tag"] == expected_previous_value
                and PayloadDigestAlgorithm(self._text(row, "chain_algorithm"))
                is entry.chain_tag.algorithm
                and self._text(row, "chain_tag") == entry.chain_tag.value
            )
            if not sidecars_match:
                raise ValueError
            return entry
        except (TypeError, ValueError, ValidationError):
            raise DigestVerificationError("durable ledger entry") from None

    def _read_durable_catalog(
        self,
        engine: MemoryRunRepository,
        *,
        retain_replays: bool = False,
        trusted_states: Mapping[UUID, _VerifiedRunState] | None = None,
    ) -> tuple[dict[UUID, _VerifiedRunState], dict[UUID, _ReplayedRun]]:
        try:
            run_rows = self._connection.execute(
                "SELECT run_id FROM runs ORDER BY run_id"
            ).fetchall()
            head_rows = self._connection.execute(
                """
                SELECT run_id, entry_count, algorithm, chain_tag, projection_tag, head_tag
                FROM ledger_heads
                ORDER BY run_id
                """
            ).fetchall()
            entry_run_rows = self._connection.execute(
                "SELECT DISTINCT run_id FROM ledger_entries ORDER BY run_id"
            ).fetchall()
            run_ids = {self._canonical_run_id(row) for row in run_rows}
            head_ids = {self._canonical_run_id(row) for row in head_rows}
            entry_ids = {self._canonical_run_id(row) for row in entry_run_rows}
        except (TypeError, ValueError, sqlite3.Error):
            raise DigestVerificationError("durable run catalog") from None
        if run_ids != head_ids or run_ids != entry_ids:
            raise DigestVerificationError("durable run catalog")

        heads = {self._canonical_run_id(row): self._head_from_row(row) for row in head_rows}
        states: dict[UUID, _VerifiedRunState] = {}
        replays: dict[UUID, _ReplayedRun] = {}
        for run_id in sorted(run_ids, key=str):
            try:
                rows = self._connection.execute(
                    """
                    SELECT
                        run_id,
                        position,
                        record_key,
                        record_type,
                        entry_json,
                        record_algorithm,
                        record_tag,
                        previous_chain_algorithm,
                        previous_chain_tag,
                        chain_algorithm,
                        chain_tag
                    FROM ledger_entries
                    WHERE run_id = ?
                    ORDER BY position
                    """,
                    (str(run_id),),
                ).fetchall()
            except sqlite3.Error:
                raise SQLiteRepositoryError() from None
            entries = tuple(self._entry_from_row(row) for row in rows)
            head = heads[run_id]
            trusted = None if trusted_states is None else trusted_states.get(run_id)
            if (
                not retain_replays
                and trusted is not None
                and entries == trusted.ledger
                and head == trusted.ledger_head
            ):
                states[run_id] = trusted
                continue
            replayed = engine._replay_run(entries, head)
            states[run_id] = replayed.state
            if retain_replays:
                replays[run_id] = replayed
        return states, replays

    def _read_durable_runs(self, engine: MemoryRunRepository) -> dict[UUID, _VerifiedRunState]:
        states, _ = self._read_durable_catalog(engine)
        return states

    @staticmethod
    def _install_replayed_runs(
        engine: MemoryRunRepository,
        replays: Mapping[UUID, _ReplayedRun],
    ) -> None:
        """Install already authenticated replay results into one fresh batch engine."""

        if engine._slots or engine._trusted_heads or engine._trusted_projections:
            raise DigestVerificationError("batch replay target")
        for run_id, replayed in replays.items():
            state = replayed.state
            if run_id != state.run_id:
                raise DigestVerificationError("batch replay target")
            engine._slots[run_id] = _RunSlot(
                run_id=run_id,
                ledger=state.ledger,
                ledger_head=state.ledger_head,
                projection=state.projection,
                direct_records=dict(replayed.direct_records),
                cycle_records=dict(replayed.cycle_records),
                delivery_records=dict(replayed.delivery_records),
                collision_receipts=dict(replayed.collision_receipts),
            )
            engine._trusted_heads[run_id] = state.ledger_head
            engine._trusted_projections[run_id] = state.projection

    def _check_and_remember_anchors(
        self,
        states: Mapping[UUID, _VerifiedRunState],
    ) -> None:
        missing = set(self._anchors).difference(states)
        if missing:
            raise DigestVerificationError("durable ledger rollback")
        for run_id, state in states.items():
            trusted = self._anchors.get(run_id)
            if trusted is not None:
                if len(state.ledger) < len(trusted.ledger):
                    raise DigestVerificationError("durable ledger rollback")
                for trusted_entry, current_entry in zip(
                    trusted.ledger,
                    state.ledger,
                    strict=False,
                ):
                    if canonical_json(trusted_entry) != canonical_json(current_entry):
                        raise DigestVerificationError("durable ledger fork")
                if (
                    len(state.ledger) == len(trusted.ledger)
                    and state.ledger_head != trusted.ledger_head
                ):
                    raise DigestVerificationError("durable ledger rollback")
            self._anchors[run_id] = state

    def _read_transaction(self) -> tuple[MemoryRunRepository, dict[UUID, _VerifiedRunState]]:
        self._ensure_open()
        engine = self._new_engine()
        try:
            self._connection.execute("BEGIN")
            states = self._read_durable_runs(engine)
            self._connection.commit()
        except (RepositoryError, ValidationError):
            self._rollback_best_effort()
            raise
        except sqlite3.Error:
            self._rollback_best_effort()
            raise SQLiteRepositoryError() from None
        self._check_and_remember_anchors(states)
        return engine, states

    async def _load_engine(self) -> tuple[MemoryRunRepository, dict[UUID, _VerifiedRunState]]:
        engine, states = await self._run_database(self._read_transaction)
        for state in states.values():
            await engine._restore_run(state.ledger, state.ledger_head)
        return engine, states

    def _read_batch_transaction(
        self,
    ) -> tuple[MemoryRunRepository, dict[UUID, _VerifiedRunState]]:
        self._ensure_open()
        engine = self._new_engine()
        try:
            self._connection.execute("BEGIN")
            states, replays = self._read_durable_catalog(engine, retain_replays=True)
            self._connection.commit()
        except (RepositoryError, ValidationError):
            self._rollback_best_effort()
            raise
        except sqlite3.Error:
            self._rollback_best_effort()
            raise SQLiteRepositoryError() from None
        self._check_and_remember_anchors(states)
        self._install_replayed_runs(engine, replays)
        return engine, states

    async def _load_batch_engine(
        self,
    ) -> tuple[MemoryRunRepository, dict[UUID, _VerifiedRunState]]:
        return await self._run_database(self._read_batch_transaction)

    async def _batch_candidate_state(
        self,
        engine: MemoryRunRepository,
        run_id: UUID,
    ) -> _VerifiedRunState:
        """Export a fully staged batch without replaying its authenticated prefix."""

        slot = await engine._slot(run_id)
        async with slot.lock:
            engine._verify_head(slot)
            if slot.ledger_head is None or slot.projection is None:  # pragma: no cover
                raise DigestVerificationError("batch candidate")
            validate_complete_projection(slot.projection)
            digests = projection_digests(
                slot.projection,
                engine._integrity,
                ledger_position=len(slot.ledger),
            )
            return _VerifiedRunState(
                run_id=run_id,
                ledger=slot.ledger,
                ledger_head=slot.ledger_head,
                projection=slot.projection,
                digests=digests,
            )

    @staticmethod
    def _state_root_value(state: _VerifiedRunState) -> dict[str, object]:
        projection = state.projection
        return {
            "state_schema_version": _STATE_SCHEMA_VERSION,
            "run_id": str(state.run_id),
            "ledger_head": state.ledger_head.model_dump(mode="json"),
            "projection_digests": state.digests.model_dump(mode="json"),
            "ledger_position": len(state.ledger),
            "ingestion_cursor": projection.ingestion_cursor,
            "memory_cursor": projection.memory_cursor,
            "current_private_status_id": (
                None
                if projection.current_private_status_id is None
                else str(projection.current_private_status_id)
            ),
        }

    def _projection_sources(self, state: _VerifiedRunState) -> _ProjectionSources:
        projected = empty_projection(state.run_id)
        signal_positions: dict[UUID, int] = {}
        decision_positions: dict[UUID, int] = {}
        cycle_positions: dict[tuple[str, int], int] = {}
        memory_cycles: dict[tuple[UUID, int], tuple[str, int]] = {}
        intervention_cycles: dict[UUID, tuple[str, int]] = {}
        outcome_positions: dict[UUID, int] = {}
        delivery_positions: dict[tuple[UUID, int], int] = {}

        for entry in state.ledger:
            before_memory = set(projected.memory_history)
            before_interventions = set(projected.interventions)
            projected = apply_entry(projected, entry)
            record = entry.record
            if isinstance(record, Signal):
                signal_positions[record.signal_id] = entry.position
            elif isinstance(record, InvocationDecision):
                decision_positions[record.decision_id] = entry.position
            elif isinstance(record, CycleRecord):
                source = (record.cycle_id, record.revision)
                cycle_positions[source] = entry.position
                for key in set(projected.memory_history).difference(before_memory):
                    memory_cycles[key] = source
                for intervention_id in set(projected.interventions).difference(
                    before_interventions
                ):
                    intervention_cycles[intervention_id] = source
            elif isinstance(record, InterventionOutcome):
                outcome_positions[record.outcome_id] = entry.position
            elif isinstance(record, DeliveryRecord):
                delivery_positions[(record.delivery_id, record.revision)] = entry.position

        if projected != state.projection:
            raise DigestVerificationError("durable projection materialization")
        return _ProjectionSources(
            signal_positions=signal_positions,
            decision_positions=decision_positions,
            cycle_positions=cycle_positions,
            memory_cycles=memory_cycles,
            intervention_cycles=intervention_cycles,
            outcome_positions=outcome_positions,
            delivery_positions=delivery_positions,
        )

    def _delete_projection(self, run_id: UUID) -> None:
        parameters = (str(run_id),)
        for table in (
            "projection_outcomes",
            "projection_delivery_revisions",
            "projection_budgets",
            "projection_memories",
            "projection_interventions",
            "projection_cycle_revisions",
            "projection_decisions",
            "projection_signals",
            "projection_events",
        ):
            self._connection.execute(f"DELETE FROM {table} WHERE run_id = ?", parameters)

    def _delete_all_projections(self) -> None:
        for table in (
            "projection_outcomes",
            "projection_delivery_revisions",
            "projection_budgets",
            "projection_memories",
            "projection_interventions",
            "projection_cycle_revisions",
            "projection_decisions",
            "projection_signals",
            "projection_events",
            "projection_state",
        ):
            self._connection.execute(f"DELETE FROM {table}")

    def _write_projection(self, state: _VerifiedRunState) -> None:
        projection = state.projection
        sources = self._projection_sources(state)
        run_id = str(state.run_id)
        self._delete_projection(state.run_id)

        self._connection.executemany(
            """
            INSERT INTO projection_events(
                run_id, event_id, sequence, source_event_id, ledger_position, record_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    run_id,
                    str(event.event_id),
                    event.sequence,
                    event.source_event_id,
                    projection.event_positions[event.event_id],
                    canonical_json(event),
                )
                for _, event in sorted(projection.events_by_sequence.items())
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO projection_signals(
                run_id, signal_id, ledger_position, created_at, record_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                (
                    run_id,
                    str(signal.signal_id),
                    sources.signal_positions[signal.signal_id],
                    signal.created_at.isoformat(),
                    canonical_json(signal),
                )
                for _, signal in sorted(projection.signals.items(), key=lambda item: str(item[0]))
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO projection_decisions(
                run_id, decision_id, event_sequence, ledger_position, created_at, record_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    run_id,
                    str(decision.decision_id),
                    decision.event_sequence,
                    sources.decision_positions[decision.decision_id],
                    decision.created_at.isoformat(),
                    canonical_json(decision),
                )
                for _, decision in sorted(
                    projection.decisions.items(),
                    key=lambda item: str(item[0]),
                )
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO projection_cycle_revisions(
                run_id,
                cycle_id,
                revision,
                state,
                is_latest,
                ledger_position,
                invocation_decision_id,
                created_at,
                updated_at,
                record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    run_id,
                    cycle.cycle_id,
                    cycle.revision,
                    cycle.state.value,
                    int(projection.cycles[cycle.cycle_id].revision == cycle.revision),
                    sources.cycle_positions[(cycle.cycle_id, cycle.revision)],
                    str(cycle.invocation_decision_id),
                    cycle.created_at.isoformat(),
                    cycle.updated_at.isoformat(),
                    canonical_json(cycle),
                )
                for _, cycle in sorted(projection.cycle_history.items())
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO projection_memories(
                run_id,
                memory_id,
                revision,
                kind,
                validity,
                trust_label,
                content,
                is_latest,
                source_cycle_id,
                source_cycle_revision,
                record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    run_id,
                    str(memory.memory_id),
                    memory.revision,
                    memory.kind.value,
                    memory.validity.value,
                    memory.trust_label.value,
                    memory.content,
                    int(projection.memories[memory.memory_id].revision == memory.revision),
                    sources.memory_cycles[(memory.memory_id, memory.revision)][0],
                    sources.memory_cycles[(memory.memory_id, memory.revision)][1],
                    canonical_json(memory),
                )
                for _, memory in sorted(
                    projection.memory_history.items(),
                    key=lambda item: (str(item[0][0]), item[0][1]),
                )
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO projection_interventions(
                run_id,
                intervention_id,
                cycle_id,
                cycle_revision,
                action,
                created_at,
                record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    run_id,
                    str(intervention.intervention_id),
                    sources.intervention_cycles[intervention.intervention_id][0],
                    sources.intervention_cycles[intervention.intervention_id][1],
                    intervention.action.value,
                    intervention.created_at.isoformat(),
                    canonical_json(intervention),
                )
                for _, intervention in sorted(
                    projection.interventions.items(),
                    key=lambda item: str(item[0]),
                )
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO projection_outcomes(
                run_id, outcome_id, intervention_id, ledger_position, created_at, record_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    run_id,
                    str(outcome.outcome_id),
                    str(outcome.intervention_id),
                    sources.outcome_positions[outcome.outcome_id],
                    outcome.created_at.isoformat(),
                    canonical_json(outcome),
                )
                for _, outcome in sorted(
                    projection.outcomes.items(),
                    key=lambda item: str(item[0]),
                )
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO projection_delivery_revisions(
                run_id,
                delivery_id,
                revision,
                intervention_id,
                state,
                attempt_count,
                is_latest,
                ledger_position,
                created_at,
                updated_at,
                record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    run_id,
                    str(delivery.delivery_id),
                    delivery.revision,
                    str(delivery.intervention_id),
                    delivery.state.value,
                    delivery.attempt_count,
                    int(projection.deliveries[delivery.delivery_id].revision == delivery.revision),
                    sources.delivery_positions[(delivery.delivery_id, delivery.revision)],
                    delivery.created_at.isoformat(),
                    delivery.updated_at.isoformat(),
                    canonical_json(delivery),
                )
                for _, delivery in sorted(
                    projection.delivery_history.items(),
                    key=lambda item: (str(item[0][0]), item[0][1]),
                )
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO projection_budgets(
                run_id,
                cycle_id,
                cycle_revision,
                state,
                reservation_json,
                settlement_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    run_id,
                    cycle.cycle_id,
                    cycle.revision,
                    cycle.state.value,
                    (
                        None
                        if cycle.budget_reservation is None
                        else canonical_json(cycle.budget_reservation)
                    ),
                    (
                        None
                        if cycle.budget_settlement is None
                        else canonical_json(cycle.budget_settlement)
                    ),
                )
                for _, cycle in sorted(projection.cycles.items())
            ),
        )

        state_tag = self._integrity.tag(
            self._state_root_value(state),
            domain=_STATE_ROOT_DOMAIN,
        )
        self._connection.execute(
            """
            INSERT INTO projection_state(
                run_id,
                state_schema_version,
                ledger_position,
                ingestion_cursor,
                memory_cursor,
                current_private_status_id,
                projection_digests_json,
                state_algorithm,
                state_tag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                state_schema_version = excluded.state_schema_version,
                ledger_position = excluded.ledger_position,
                ingestion_cursor = excluded.ingestion_cursor,
                memory_cursor = excluded.memory_cursor,
                current_private_status_id = excluded.current_private_status_id,
                projection_digests_json = excluded.projection_digests_json,
                state_algorithm = excluded.state_algorithm,
                state_tag = excluded.state_tag
            """,
            (
                run_id,
                _STATE_SCHEMA_VERSION,
                len(state.ledger),
                projection.ingestion_cursor,
                projection.memory_cursor,
                (
                    None
                    if projection.current_private_status_id is None
                    else str(projection.current_private_status_id)
                ),
                canonical_json(state.digests),
                state_tag.algorithm.value,
                state_tag.value,
            ),
        )

    def _recover_derived_state(self) -> None:
        engine = self._new_engine()
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            states = self._read_durable_runs(engine)
            self._connection.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild')")
            self._delete_all_projections()
            for state in states.values():
                expected_created_at = state.projection.events_by_sequence[1].timestamp.isoformat()
                self._connection.execute(
                    "UPDATE runs SET created_at = ? WHERE run_id = ?",
                    (expected_created_at, str(state.run_id)),
                )
                self._write_projection(state)
            self._connection.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild')")
            self._connection.commit()
        except (RepositoryError, ValidationError):
            self._rollback_best_effort()
            raise
        except (KeyError, sqlite3.Error):
            self._rollback_best_effort()
            raise SQLiteRepositoryError() from None
        self._check_and_remember_anchors(states)

    def _database_version(self, run_id: UUID) -> _RunVersion | None:
        try:
            row = self._connection.execute(
                "SELECT entry_count, algorithm, head_tag FROM ledger_heads WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
            run_exists = self._connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
        except sqlite3.Error:
            raise SQLiteRepositoryError() from None
        if row is None:
            if run_exists is not None:
                raise DigestVerificationError("durable run catalog")
            return None
        if run_exists is None:
            raise DigestVerificationError("durable run catalog")
        try:
            return _RunVersion(
                entry_count=row["entry_count"],
                algorithm=PayloadDigestAlgorithm(self._text(row, "algorithm")),
                head_tag=self._text(row, "head_tag"),
            )
        except (TypeError, ValueError):
            raise DigestVerificationError("durable ledger head") from None

    @staticmethod
    def _prefix_matches(base: _VerifiedRunState, candidate: _VerifiedRunState) -> bool:
        if len(candidate.ledger) < len(base.ledger):
            return False
        return all(
            canonical_json(old) == canonical_json(new)
            for old, new in zip(base.ledger, candidate.ledger, strict=False)
        )

    def _insert_ledger_entry(self, entry: LedgerEntry) -> None:
        previous = entry.previous_chain_tag
        self._connection.execute(
            """
            INSERT INTO ledger_entries(
                run_id,
                position,
                record_key,
                record_type,
                entry_json,
                record_algorithm,
                record_tag,
                previous_chain_algorithm,
                previous_chain_tag,
                chain_algorithm,
                chain_tag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(entry.run_id),
                entry.position,
                entry.record_key,
                entry.record.record_type,
                canonical_json(entry),
                entry.record_tag.algorithm.value,
                entry.record_tag.value,
                None if previous is None else previous.algorithm.value,
                None if previous is None else previous.value,
                entry.chain_tag.algorithm.value,
                entry.chain_tag.value,
            ),
        )

    def _write_head(self, head: LedgerHead, *, new_run: bool) -> None:
        values = (
            str(head.run_id),
            head.entry_count,
            head.head_tag.algorithm.value,
            head.chain_tag.value,
            head.projection_tag.value,
            head.head_tag.value,
        )
        if new_run:
            self._connection.execute(
                """
                INSERT INTO ledger_heads(
                    run_id, entry_count, algorithm, chain_tag, projection_tag, head_tag
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            return
        self._connection.execute(
            """
            UPDATE ledger_heads
            SET entry_count = ?, algorithm = ?, chain_tag = ?, projection_tag = ?, head_tag = ?
            WHERE run_id = ?
            """,
            (
                head.entry_count,
                head.head_tag.algorithm.value,
                head.chain_tag.value,
                head.projection_tag.value,
                head.head_tag.value,
                str(head.run_id),
            ),
        )

    def _commit_candidate(
        self,
        candidate: _VerifiedRunState,
        base: _VerifiedRunState | None,
        *,
        rebuild_fts: bool = False,
    ) -> bool:
        if base is not None and not self._prefix_matches(base, candidate):
            raise DigestVerificationError("candidate ledger fork")
        base_version = None if base is None else _RunVersion.from_head(base.ledger_head)
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            current_states = self._read_durable_runs(self._new_engine())
            self._check_and_remember_anchors(current_states)
            current = current_states.get(candidate.run_id)
            current_version = (
                None if current is None else _RunVersion.from_head(current.ledger_head)
            )
            if current_version != base_version:
                raise _ConcurrentMutationError
            if base is not None and (
                current is None
                or current.ledger_head != base.ledger_head
                or not self._prefix_matches(base, current)
                or len(current.ledger) != len(base.ledger)
            ):
                raise DigestVerificationError("concurrent ledger fork")

            if base is None:
                first_event = candidate.projection.events_by_sequence.get(1)
                if first_event is None:
                    raise DigestVerificationError("candidate run origin")
                self._connection.execute(
                    "INSERT INTO runs(run_id, created_at) VALUES (?, ?)",
                    (str(candidate.run_id), first_event.timestamp.isoformat()),
                )
                suffix = candidate.ledger
            else:
                suffix = candidate.ledger[len(base.ledger) :]
            for entry in suffix:
                self._insert_ledger_entry(entry)
            self._write_head(candidate.ledger_head, new_run=base is None)
            self._write_projection(candidate)
            if rebuild_fts:
                self._connection.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild')")
            expected_states = dict(current_states)
            expected_states[candidate.run_id] = candidate
            persisted_states = self._read_durable_runs(self._new_engine())
            if set(persisted_states) != set(expected_states) or any(
                persisted.ledger != expected_states[run_id].ledger
                or persisted.ledger_head != expected_states[run_id].ledger_head
                or persisted.digests != expected_states[run_id].digests
                for run_id, persisted in persisted_states.items()
            ):
                raise DigestVerificationError("post-write durable ledger")
            self._connection.commit()
        except _ConcurrentMutationError:
            self._rollback_best_effort()
            return False
        except (RepositoryError, ValidationError):
            self._rollback_best_effort()
            raise
        except (KeyError, sqlite3.Error):
            self._rollback_best_effort()
            raise SQLiteRepositoryError() from None
        self._check_and_remember_anchors(persisted_states)
        return True

    def _commit_batch_candidate(
        self,
        candidate: _VerifiedRunState,
        base: _VerifiedRunState | None,
    ) -> bool:
        """Publish one preflighted conditional batch in one write transaction."""

        self._authenticate_batch_candidate(candidate, base)
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            current_states, _ = self._read_durable_catalog(
                self._new_engine(),
                trusted_states=self._anchors,
            )
            self._check_and_remember_anchors(current_states)
            current = current_states.get(candidate.run_id)
            if base is None:
                if current is not None:
                    raise _ConcurrentMutationError
            elif (
                current is None
                or current.ledger_head != base.ledger_head
                or current.ledger != base.ledger
            ):
                raise _ConcurrentMutationError

            if base is None:
                first_event = candidate.projection.events_by_sequence.get(1)
                if first_event is None:
                    raise DigestVerificationError("candidate run origin")
                self._connection.execute(
                    "INSERT INTO runs(run_id, created_at) VALUES (?, ?)",
                    (str(candidate.run_id), first_event.timestamp.isoformat()),
                )
                suffix = candidate.ledger
            else:
                suffix = candidate.ledger[len(base.ledger) :]
            for entry in suffix:
                self._insert_ledger_entry(entry)
            self._write_head(candidate.ledger_head, new_run=base is None)
            self._write_projection(candidate)

            expected_states = dict(current_states)
            expected_states[candidate.run_id] = candidate
            persisted_states, _ = self._read_durable_catalog(
                self._new_engine(),
                trusted_states=expected_states,
            )
            if set(persisted_states) != set(expected_states) or any(
                persisted.ledger != expected_states[run_id].ledger
                or persisted.ledger_head != expected_states[run_id].ledger_head
                or persisted.digests != expected_states[run_id].digests
                for run_id, persisted in persisted_states.items()
            ):
                raise DigestVerificationError("post-write durable ledger")
            self._connection.commit()
        except _ConcurrentMutationError:
            self._rollback_best_effort()
            return False
        except (RepositoryError, ValidationError):
            self._rollback_best_effort()
            raise
        except (KeyError, sqlite3.Error):
            self._rollback_best_effort()
            raise SQLiteRepositoryError() from None
        except BaseException:
            self._rollback_best_effort()
            raise
        self._check_and_remember_anchors(persisted_states)
        return True

    def _authenticate_batch_candidate(
        self,
        candidate: _VerifiedRunState,
        base: _VerifiedRunState | None,
    ) -> None:
        """Verify a candidate suffix against its already authenticated base."""

        if base is not None and not self._prefix_matches(base, candidate):
            raise DigestVerificationError("candidate ledger fork")
        engine = self._new_engine()
        if base is None:
            base_count = 0
            projected = empty_projection(candidate.run_id)
            previous_chain: PayloadDigest | None = None
            projection_tag: PayloadDigest | None = None
        else:
            if candidate.run_id != base.run_id:
                raise DigestVerificationError("candidate ledger fork")
            base_count = len(base.ledger)
            projected = base.projection
            previous_chain = base.ledger_head.chain_tag
            projection_tag = base.ledger_head.projection_tag

        suffix = candidate.ledger[base_count:]
        if not suffix:
            if base is None or (
                candidate.ledger != base.ledger
                or candidate.ledger_head != base.ledger_head
                or candidate.projection != base.projection
                or candidate.digests != base.digests
            ):
                raise DigestVerificationError("batch candidate")
            return

        for expected_position, entry in enumerate(suffix, start=base_count + 1):
            if entry.run_id != candidate.run_id:
                raise DigestVerificationError("batch candidate")
            engine._verify_entry(
                entry,
                expected_position=expected_position,
                previous=previous_chain,
            )
            projected = apply_entry(projected, entry)
            projection_tag = engine._projection_checkpoint_tag(
                entry,
                projected,
                previous=projection_tag,
            )
            previous_chain = entry.chain_tag

        head = candidate.ledger_head
        if (
            head.run_id != candidate.run_id
            or head.entry_count != len(candidate.ledger)
            or head.chain_tag != previous_chain
            or head.projection_tag != projection_tag
            or not self._integrity.verify(
                engine._head_value(
                    head.run_id,
                    head.entry_count,
                    head.chain_tag,
                    head.projection_tag,
                ),
                head.head_tag,
                domain="saliencegate:ledger-head:v1",
            )
        ):
            raise DigestVerificationError("batch candidate")
        validate_complete_projection(projected)
        digests = projection_digests(
            projected,
            self._integrity,
            ledger_position=len(candidate.ledger),
        )
        if projected != candidate.projection or digests != candidate.digests:
            raise DigestVerificationError("batch candidate")

    @staticmethod
    def _retry_run_id(value: object) -> UUID | None:
        run_id = getattr(value, "run_id", None)
        return SQLiteRunRepository._valid_run_id(run_id)

    @staticmethod
    def _valid_run_id(run_id: object) -> UUID | None:
        if type(run_id) is UUID and run_id.version == 4:
            return UUID(int=run_id.int)
        return None

    async def _mutate(
        self,
        operation: Callable[[MemoryRunRepository], Awaitable[ResultT]],
        result_run_id: Callable[[ResultT], UUID],
        *,
        retry_run_id: UUID | None,
        rebuild_fts: bool = False,
    ) -> ResultT:
        async with self._lock:
            self._ensure_open()
            for _ in range(_MAX_CAS_ATTEMPTS):
                engine, loaded = await self._load_engine()
                before_version = (
                    None
                    if retry_run_id is None or retry_run_id not in loaded
                    else _RunVersion.from_head(loaded[retry_run_id].ledger_head)
                )
                try:
                    result = await operation(engine)
                except RepositoryError:
                    if retry_run_id is None:
                        raise
                    current_version = await self._run_database(
                        lambda: self._database_version(retry_run_id)
                    )
                    if current_version != before_version:
                        continue
                    raise
                run_id = result_run_id(result)
                candidate = await engine._verified_state(run_id)
                base = loaded.get(run_id)
                committed = await self._run_database(
                    partial(
                        self._commit_candidate,
                        candidate,
                        base,
                        rebuild_fts=rebuild_fts,
                    )
                )
                if committed:
                    return result
            raise ConcurrentWriteError()

    async def _mutate_if_head(
        self,
        operation: Callable[[MemoryRunRepository], Awaitable[ResultT]],
        result_run_id: Callable[[ResultT], UUID],
    ) -> ResultT:
        async with self._lock:
            self._ensure_open()
            engine, loaded = await self._load_engine()
            result = await operation(engine)
            run_id = result_run_id(result)
            candidate = await engine._verified_state(run_id)
            base = loaded.get(run_id)
            committed = await self._run_database(partial(self._commit_candidate, candidate, base))
            if not committed:
                raise LedgerHeadConflictError()
            return result

    @staticmethod
    def _copy_conditional_operations(
        operations: object,
    ) -> tuple[ConditionalAppendOperation, ...]:
        operation_name = "conditional_batch"
        if type(operations) is not tuple:
            raise InvalidRecordTypeError(operation_name, type(operations))
        if not 1 <= len(operations) <= MAX_CONDITIONAL_BATCH_OPERATIONS:
            raise InvalidRecordError(operation_name)
        copied: list[ConditionalAppendOperation] = []
        for operation in operations:
            detached: ConditionalAppendOperation
            try:
                if type(operation) is ConditionalEventAppend:
                    detached = ConditionalEventAppend.model_validate_json(
                        ConditionalEventAppend.model_dump_json(operation, warnings=False)
                    )
                elif type(operation) is ConditionalSignalAppend:
                    detached = ConditionalSignalAppend.model_validate_json(
                        ConditionalSignalAppend.model_dump_json(operation, warnings=False)
                    )
                else:
                    raise InvalidRecordTypeError(operation_name, type(operation))
            except InvalidRecordTypeError:
                raise
            except Exception:
                raise InvalidRecordError(operation_name) from None
            copied.append(detached)
        return tuple(copied)

    async def _read(
        self,
        operation: Callable[[MemoryRunRepository], Awaitable[ResultT]],
    ) -> ResultT:
        async with self._lock:
            self._ensure_open()
            engine, _ = await self._load_engine()
            return await operation(engine)

    async def append(
        self,
        event: NormalizedTraceEventDraft,
        *,
        event_id: UUID | None = None,
    ) -> AppendReceipt:
        return await self._mutate(
            lambda engine: engine.append(event, event_id=event_id),
            lambda receipt: receipt.event.run_id,
            retry_run_id=self._retry_run_id(event),
        )

    async def append_event_if_head(
        self,
        event: NormalizedTraceEventDraft,
        *,
        event_id: UUID,
        expected_head: LedgerHead | None,
    ) -> AppendReceipt:
        return await self._mutate_if_head(
            lambda engine: engine.append_event_if_head(
                event,
                event_id=event_id,
                expected_head=expected_head,
            ),
            lambda receipt: receipt.event.run_id,
        )

    async def append_records_if_head(
        self,
        operations: tuple[ConditionalAppendOperation, ...],
        *,
        expected_head: LedgerHead | None,
    ) -> ConditionalBatchReceipt:
        copied_operations = self._copy_conditional_operations(operations)
        self._new_engine()._prepare_conditional_batch(copied_operations)
        copied_head = (
            None
            if expected_head is None
            else MemoryRunRepository._copy_expected_head(expected_head)
        )
        async with self._lock:
            self._ensure_open()
            engine, loaded = await self._load_batch_engine()
            receipt = await engine.append_records_if_head(
                copied_operations,
                expected_head=copied_head,
            )
            run_id = receipt.final_head.run_id
            candidate = await self._batch_candidate_state(engine, run_id)
            base = loaded.get(run_id)
            committed = await self._run_database(
                partial(self._commit_batch_candidate, candidate, base)
            )
            if not committed:
                raise LedgerHeadConflictError()
            return receipt

    async def record_signal(self, signal: Signal) -> LedgerReceipt:
        return await self._mutate(
            lambda engine: engine.record_signal(signal),
            lambda _receipt: signal.run_id,
            retry_run_id=self._retry_run_id(signal),
        )

    async def record_signal_if_head(
        self,
        signal: Signal,
        *,
        expected_head: LedgerHead,
    ) -> LedgerReceipt:
        validated_signal = MemoryRunRepository._validate_direct_record(signal, Signal, "signal")
        copied_head = MemoryRunRepository._copy_expected_head(expected_head)
        return await self._mutate_if_head(
            lambda engine: engine.record_signal_if_head(
                validated_signal,
                expected_head=copied_head,
            ),
            lambda _receipt: copied_head.run_id,
        )

    async def record_invocation_decision(
        self,
        decision: InvocationDecision,
    ) -> LedgerReceipt:
        return await self._mutate(
            lambda engine: engine.record_invocation_decision(decision),
            lambda _receipt: decision.run_id,
            retry_run_id=self._retry_run_id(decision),
        )

    async def record_outcome(self, outcome: InterventionOutcome) -> LedgerReceipt:
        return await self._mutate(
            lambda engine: engine.record_outcome(outcome),
            lambda _receipt: outcome.run_id,
            retry_run_id=self._retry_run_id(outcome),
        )

    async def begin_cycle(self, command: BeginCycle) -> CycleReceipt:
        return await self._mutate(
            lambda engine: engine.begin_cycle(command),
            lambda receipt: receipt.cycle.run_id,
            retry_run_id=self._retry_run_id(command),
        )

    async def reserve_cycle(self, command: ReserveCycle) -> CycleReceipt:
        return await self._mutate(
            lambda engine: engine.reserve_cycle(command),
            lambda receipt: receipt.cycle.run_id,
            retry_run_id=self._retry_run_id(command),
        )

    async def mark_cycle_running(self, command: StartCycle) -> CycleReceipt:
        return await self._mutate(
            lambda engine: engine.mark_cycle_running(command),
            lambda receipt: receipt.cycle.run_id,
            retry_run_id=self._retry_run_id(command),
        )

    async def commit_cycle(self, command: CommitCycle) -> CycleReceipt:
        return await self._mutate(
            lambda engine: engine.commit_cycle(command),
            lambda receipt: receipt.cycle.run_id,
            retry_run_id=self._retry_run_id(command),
        )

    async def fail_cycle(self, command: FailCycle) -> CycleReceipt:
        return await self._mutate(
            lambda engine: engine.fail_cycle(command),
            lambda receipt: receipt.cycle.run_id,
            retry_run_id=self._retry_run_id(command),
        )

    async def preview_memory_delta(self, command: PreviewMemoryDelta) -> MemoryDeltaPreview:
        return await self._read(lambda engine: engine.preview_memory_delta(command))

    async def budget_snapshot(self, run_id: UUID) -> BudgetSnapshot:
        return await self._read(lambda engine: engine.budget_snapshot(run_id))

    async def recover_cycles(
        self,
        run_id: UUID,
        *,
        recovered_at: datetime,
    ) -> CycleRecoveryReceipt:
        return await self._mutate(
            lambda engine: engine.recover_cycles(run_id, recovered_at=recovered_at),
            lambda receipt: receipt.run_id,
            retry_run_id=self._valid_run_id(run_id),
        )

    async def delivery(self, run_id: UUID, delivery_id: UUID) -> DeliveryRecord:
        return await self._read(lambda engine: engine.delivery(run_id, delivery_id))

    async def claim_delivery(
        self,
        command: ClaimDelivery,
    ) -> DeliveryTransitionReceipt:
        return await self._mutate(
            lambda engine: engine.claim_delivery(command),
            lambda receipt: receipt.delivery.run_id,
            retry_run_id=self._retry_run_id(command),
        )

    async def begin_delivery_attempt(
        self,
        command: BeginDeliveryAttempt,
    ) -> DeliveryAttemptReceipt:
        return await self._mutate(
            lambda engine: engine.begin_delivery_attempt(command),
            lambda receipt: receipt.delivery.run_id,
            retry_run_id=self._retry_run_id(command),
        )

    async def complete_delivery(
        self,
        command: CompleteDelivery,
    ) -> DeliveryTransitionReceipt:
        return await self._mutate(
            lambda engine: engine.complete_delivery(command),
            lambda receipt: receipt.delivery.run_id,
            retry_run_id=self._retry_run_id(command),
        )

    async def mark_delivery_unknown(
        self,
        command: MarkDeliveryUnknown,
    ) -> DeliveryTransitionReceipt:
        return await self._mutate(
            lambda engine: engine.mark_delivery_unknown(command),
            lambda receipt: receipt.delivery.run_id,
            retry_run_id=self._retry_run_id(command),
        )

    async def reject_delivery(
        self,
        command: RejectDelivery,
    ) -> DeliveryTransitionReceipt:
        return await self._mutate(
            lambda engine: engine.reject_delivery(command),
            lambda receipt: receipt.delivery.run_id,
            retry_run_id=self._retry_run_id(command),
        )

    async def recover_deliveries(
        self,
        run_id: UUID,
        *,
        recovered_at: datetime,
    ) -> DeliveryRecoveryReceipt:
        return await self._mutate(
            lambda engine: engine.recover_deliveries(run_id, recovered_at=recovered_at),
            lambda receipt: receipt.run_id,
            retry_run_id=self._valid_run_id(run_id),
        )

    async def ledger(self, run_id: UUID) -> tuple[LedgerEntry, ...]:
        return await self._read(lambda engine: engine.ledger(run_id))

    async def ledger_head(self, run_id: UUID) -> LedgerHead:
        return await self._read(lambda engine: engine.ledger_head(run_id))

    async def search(self, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        return await self._read(lambda engine: engine.search(query))

    async def snapshot(self, run_id: UUID) -> MemorySnapshot:
        return await self._read(lambda engine: engine.snapshot(run_id))

    async def rebuild(self, run_id: UUID) -> RebuildReceipt:
        return await self._mutate(
            lambda engine: engine.rebuild(run_id),
            lambda receipt: receipt.run_id,
            retry_run_id=self._valid_run_id(run_id),
            rebuild_fts=True,
        )


__all__ = [
    "ClosedSQLiteRepositoryError",
    "ConcurrentWriteError",
    "SQLiteRepositoryError",
    "SQLiteRunRepository",
]
