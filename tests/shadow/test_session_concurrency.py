from __future__ import annotations

import asyncio
import socket
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from saliencegate.shadow.session import ShadowSession
from tests.shadow.conftest import NOW, RUN_ID

from saliencegate.domain import (
    BudgetAmounts,
    BudgetLimits,
    BudgetSnapshot,
    EventType,
    InvocationDecision,
    ReasonCode,
    Signal,
    SignalType,
    TraceEvent,
)
from saliencegate.ports.repository import (
    AppendReceipt,
    LedgerHead,
    LedgerHeadConflictError,
    LedgerReceipt,
    RepositoryError,
)
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.repository.sqlite import SQLiteRunRepository
from saliencegate.security import InstallationKey
from saliencegate.shadow.errors import (
    ShadowInputError,
    ShadowInvariantError,
    ShadowStateError,
)
from saliencegate.shadow.inputs import ShadowObservationSource
from saliencegate.shadow.observation import derive_shadow_extraction_report_digest
from saliencegate.signals import (
    DetectionContext,
    DetectorContractError,
    DeterministicSignalExtractor,
    ExtractionReport,
)

_KEY_MATERIAL = b"s" * 32
_ENVIRONMENT_DIGEST = "e" * 64
_FOREIGN_DECISION_ID = UUID("00000000-0000-4000-8000-000000000901")
_FORGED_SIGNAL_ID = UUID("00000000-0000-4000-8000-000000000902")


def _key() -> InstallationKey:
    return InstallationKey(_KEY_MATERIAL)


def _memory_session() -> ShadowSession:
    return ShadowSession.in_memory(run_id=RUN_ID, installation_key=_key())


def _sqlite_session(path: Path) -> ShadowSession:
    return ShadowSession.sqlite(path, run_id=RUN_ID, installation_key=_key())


async def _start(session: ShadowSession) -> None:
    await session.start(source_event_id="run-start", occurred_at=NOW)


async def _action(
    session: ShadowSession,
    *,
    source_event_id: str,
    seconds: int,
    argv: tuple[str, ...] = ("pytest", "-q"),
) -> Any:
    return await session.action(
        source_event_id=source_event_id,
        occurred_at=NOW + timedelta(seconds=seconds),
        argv=argv,
        working_directory="/project",
        environment_digest=_ENVIRONMENT_DIGEST,
    )


async def _failed_tool_result(
    session: ShadowSession,
    *,
    source_event_id: str,
    seconds: int,
    action: Any,
) -> Any:
    return await session.tool_result(
        source_event_id=source_event_id,
        occurred_at=NOW + timedelta(seconds=seconds),
        action=action.ref,
        status="failed",
        exit_status=1,
        exception_type="AssertionError",
    )


async def _records(session: ShadowSession) -> tuple[object, ...]:
    entries = await session._repository.ledger(RUN_ID)
    return tuple(entry.record for entry in entries)


def _foreign_decision() -> InvocationDecision:
    limits = BudgetLimits(
        model_calls=1,
        input_tokens=1,
        output_tokens=1,
        canonical_token_equivalents=2,
        latency_us=1,
        max_call_latency_us=1,
        interventions=1,
        schema_repairs=1,
    )
    return InvocationDecision(
        decision_id=_FOREIGN_DECISION_ID,
        run_id=RUN_ID,
        event_sequence=1,
        invoke=False,
        risk_score=0.1,
        reason_codes=(ReasonCode.RISK_BELOW_THRESHOLD,),
        policy_version="foreign-controller/v1",
        configuration_digest="f" * 64,
        budget_snapshot=BudgetSnapshot(
            limits=limits,
            reserved=BudgetAmounts(),
            consumed=BudgetAmounts(),
        ),
        cooldown_active=False,
        created_at=NOW,
    )


async def test_complete_head_conflicts_stop_after_eight_full_reloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _memory_session() as session:
        await _start(session)
        original_append = MemoryRunRepository.append_event_if_head
        original_ledger = MemoryRunRepository.ledger
        original_head = MemoryRunRepository.ledger_head
        append_attempts = 0
        ledger_loads = 0
        head_loads = 0

        async def conflicting_append(
            repository: MemoryRunRepository,
            event: Any,
            *,
            event_id: UUID,
            expected_head: LedgerHead | None,
        ) -> AppendReceipt:
            nonlocal append_attempts
            if event.source_event_id == "conflict-exhaustion":
                append_attempts += 1
                raise LedgerHeadConflictError()
            return await original_append(
                repository,
                event,
                event_id=event_id,
                expected_head=expected_head,
            )

        async def counted_ledger(
            repository: MemoryRunRepository,
            run_id: UUID,
        ) -> tuple[Any, ...]:
            nonlocal ledger_loads
            ledger_loads += 1
            return await original_ledger(repository, run_id)

        async def counted_head(repository: MemoryRunRepository, run_id: UUID) -> LedgerHead:
            nonlocal head_loads
            head_loads += 1
            return await original_head(repository, run_id)

        monkeypatch.setattr(MemoryRunRepository, "append_event_if_head", conflicting_append)
        monkeypatch.setattr(MemoryRunRepository, "ledger", counted_ledger)
        monkeypatch.setattr(MemoryRunRepository, "ledger_head", counted_head)

        with pytest.raises(ShadowStateError) as captured_error:
            await session.observation(
                source_event_id="conflict-exhaustion",
                occurred_at=NOW + timedelta(seconds=1),
                source=ShadowObservationSource.TASK_INPUT,
                payload={"bounded": True},
            )

        assert append_attempts == 8
        assert captured_error.value.__cause__ is None
        assert captured_error.value.__context__ is None
        assert ledger_loads >= append_attempts
        assert head_loads >= append_attempts
        assert all(
            not isinstance(record, TraceEvent) or record.source_event_id != "conflict-exhaustion"
            for record in await _records(session)
        )


async def test_head_conflict_reloads_and_rejects_a_new_mixed_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _memory_session() as session:
        await _start(session)
        original_append = MemoryRunRepository.append_event_if_head
        attempts = 0

        async def interleaving_append(
            repository: MemoryRunRepository,
            event: Any,
            *,
            event_id: UUID,
            expected_head: LedgerHead | None,
        ) -> AppendReceipt:
            nonlocal attempts
            if event.source_event_id != "mixed-after-conflict":
                return await original_append(
                    repository,
                    event,
                    event_id=event_id,
                    expected_head=expected_head,
                )
            attempts += 1
            if attempts > 1:
                pytest.fail("the session retried without rejecting the mixed ledger")
            await repository.record_invocation_decision(_foreign_decision())
            raise LedgerHeadConflictError()

        monkeypatch.setattr(MemoryRunRepository, "append_event_if_head", interleaving_append)

        with pytest.raises(ShadowStateError):
            await session.observation(
                source_event_id="mixed-after-conflict",
                occurred_at=NOW + timedelta(seconds=1),
                source=ShadowObservationSource.TASK_INPUT,
                payload={"bounded": True},
            )

        records = await _records(session)
        assert attempts == 1
        assert records[-1] == _foreign_decision()
        assert all(
            not isinstance(record, TraceEvent) or record.source_event_id != "mixed-after-conflict"
            for record in records
        )


async def test_two_sqlite_sessions_cannot_append_after_a_raced_finish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "terminal-race.sqlite3"
    session_a = _sqlite_session(path)
    session_b = _sqlite_session(path)
    async with session_a, session_b:
        await _start(session_a)
        original_append = SQLiteRunRepository.append_event_if_head
        late_is_stale = asyncio.Event()
        finish_committed = asyncio.Event()

        async def raced_append(
            repository: SQLiteRunRepository,
            event: Any,
            *,
            event_id: UUID,
            expected_head: LedgerHead | None,
        ) -> AppendReceipt:
            if event.source_event_id == "late-event":
                late_is_stale.set()
                await finish_committed.wait()
            elif event.source_event_id == "run-end":
                await late_is_stale.wait()
            try:
                return await original_append(
                    repository,
                    event,
                    event_id=event_id,
                    expected_head=expected_head,
                )
            finally:
                if event.source_event_id == "run-end":
                    finish_committed.set()

        monkeypatch.setattr(SQLiteRunRepository, "append_event_if_head", raced_append)
        late_task = asyncio.create_task(
            session_b.observation(
                source_event_id="late-event",
                occurred_at=NOW + timedelta(seconds=2),
                source=ShadowObservationSource.TASK_INPUT,
                payload={"late": True},
            )
        )
        await asyncio.wait_for(late_is_stale.wait(), timeout=5)
        await session_a.finish(
            source_event_id="run-end",
            occurred_at=NOW + timedelta(seconds=3),
        )

        with pytest.raises(ShadowInputError):
            await asyncio.wait_for(late_task, timeout=5)

        trace_events = tuple(
            record for record in await _records(session_a) if isinstance(record, TraceEvent)
        )
        assert tuple(event.event_type for event in trace_events) == (
            EventType.RUN_START,
            EventType.RUN_END,
        )
        assert all(event.source_event_id != "late-event" for event in trace_events)


async def test_two_sqlite_sessions_revalidate_timestamp_after_a_peer_advance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "timestamp-race.sqlite3"
    session_a = _sqlite_session(path)
    session_b = _sqlite_session(path)
    async with session_a, session_b:
        await _start(session_a)
        original_append = SQLiteRunRepository.append_event_if_head
        stale_ready = asyncio.Event()
        newer_committed = asyncio.Event()

        async def raced_append(
            repository: SQLiteRunRepository,
            event: Any,
            *,
            event_id: UUID,
            expected_head: LedgerHead | None,
        ) -> AppendReceipt:
            if event.source_event_id == "stale-timestamp":
                stale_ready.set()
                await newer_committed.wait()
            elif event.source_event_id == "newer-timestamp":
                await stale_ready.wait()
            try:
                return await original_append(
                    repository,
                    event,
                    event_id=event_id,
                    expected_head=expected_head,
                )
            finally:
                if event.source_event_id == "newer-timestamp":
                    newer_committed.set()

        monkeypatch.setattr(SQLiteRunRepository, "append_event_if_head", raced_append)
        stale_task = asyncio.create_task(
            session_b.observation(
                source_event_id="stale-timestamp",
                occurred_at=NOW + timedelta(seconds=2),
                source=ShadowObservationSource.TASK_INPUT,
                payload={"order": "stale"},
            )
        )
        await asyncio.wait_for(stale_ready.wait(), timeout=5)
        await session_a.observation(
            source_event_id="newer-timestamp",
            occurred_at=NOW + timedelta(seconds=3),
            source=ShadowObservationSource.TASK_INPUT,
            payload={"order": "newer"},
        )

        with pytest.raises(ShadowInputError):
            await asyncio.wait_for(stale_task, timeout=5)

        trace_events = tuple(
            record for record in await _records(session_a) if isinstance(record, TraceEvent)
        )
        assert tuple(event.source_event_id for event in trace_events) == (
            "run-start",
            "newer-timestamp",
        )


async def test_source_collision_before_and_after_finish_never_writes_an_audit_event() -> None:
    async with _memory_session() as session:
        await _start(session)
        original = await _action(session, source_event_id="one-action", seconds=1)
        before_collision = await _records(session)

        with pytest.raises(ShadowInputError):
            await _action(
                session,
                source_event_id="one-action",
                seconds=1,
                argv=("pytest", "tests/unit"),
            )
        assert await _records(session) == before_collision

        await session.finish(
            source_event_id="run-end",
            occurred_at=NOW + timedelta(seconds=2),
        )
        terminal = await _records(session)
        assert await _action(session, source_event_id="one-action", seconds=1) == original

        with pytest.raises(ShadowInputError):
            await _action(
                session,
                source_event_id="one-action",
                seconds=1,
                argv=("pytest", "tests/integration"),
            )

        assert await _records(session) == terminal
        assert all(
            not isinstance(record, TraceEvent)
            or record.event_type is not EventType.CONTROLLER_ERROR
            for record in terminal
        )


async def test_retry_after_crash_post_append_recomputes_the_same_extraction_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _memory_session() as session:
        await _start(session)
        action = await _action(session, source_event_id="action-before-crash", seconds=1)
        original_extract = DeterministicSignalExtractor.extract_report
        captured: list[ExtractionReport] = []

        def extract_then_crash(
            extractor: DeterministicSignalExtractor,
            context: DetectionContext,
        ) -> ExtractionReport:
            report = original_extract(extractor, context)
            if context.current.source_event_id == "result-before-crash":
                captured.append(report)
                raise DetectorContractError()
            return report

        monkeypatch.setattr(
            DeterministicSignalExtractor,
            "extract_report",
            extract_then_crash,
        )
        with pytest.raises(ShadowInvariantError):
            await _failed_tool_result(
                session,
                source_event_id="result-before-crash",
                seconds=2,
                action=action,
            )

        records_after_crash = await _records(session)
        assert captured
        assert any(
            isinstance(record, TraceEvent) and record.source_event_id == "result-before-crash"
            for record in records_after_crash
        )
        assert all(not isinstance(record, Signal) for record in records_after_crash)

        monkeypatch.setattr(
            DeterministicSignalExtractor,
            "extract_report",
            original_extract,
        )
        recovered = await _failed_tool_result(
            session,
            source_event_id="result-before-crash",
            seconds=2,
            action=action,
        )

        assert recovered.observation.extraction_report_digest == (
            derive_shadow_extraction_report_digest(captured[-1])
        )
        assert (
            await _failed_tool_result(
                session,
                source_event_id="result-before-crash",
                seconds=2,
                action=action,
            )
            == recovered
        )


@pytest.mark.parametrize("crash_after_signal", [1, 2])
async def test_retry_after_each_partial_signal_write_records_only_the_missing_signals(
    crash_after_signal: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _memory_session() as session:
        await _start(session)
        first_action = await _action(session, source_event_id="repeat-action-1", seconds=1)
        await _failed_tool_result(
            session,
            source_event_id="repeat-failure-1",
            seconds=2,
            action=first_action,
        )
        second_action = await _action(session, source_event_id="repeat-action-2", seconds=3)
        original_record = MemoryRunRepository.record_signal_if_head
        written: list[Signal] = []

        async def write_then_crash(
            repository: MemoryRunRepository,
            signal: Signal,
            *,
            expected_head: LedgerHead,
        ) -> LedgerReceipt:
            receipt = await original_record(
                repository,
                signal,
                expected_head=expected_head,
            )
            written.append(signal)
            if len(written) == crash_after_signal:
                raise RepositoryError("injected repository crash")
            return receipt

        monkeypatch.setattr(MemoryRunRepository, "record_signal_if_head", write_then_crash)
        with pytest.raises(ShadowStateError) as captured_error:
            await _failed_tool_result(
                session,
                source_event_id="repeat-failure-2",
                seconds=4,
                action=second_action,
            )

        assert captured_error.value.__cause__ is None
        assert captured_error.value.__context__ is None
        assert "injected" not in str(captured_error.value)

        target_ids = {signal.signal_id for signal in written}
        persisted_before_retry = tuple(
            record
            for record in await _records(session)
            if isinstance(record, Signal) and record.signal_id in target_ids
        )
        assert len(persisted_before_retry) == crash_after_signal

        monkeypatch.setattr(
            MemoryRunRepository,
            "record_signal_if_head",
            original_record,
        )
        recovered = await _failed_tool_result(
            session,
            source_event_id="repeat-failure-2",
            seconds=4,
            action=second_action,
        )
        recovered_ids = {signal.signal_id for signal in recovered.observation.detected_signals}
        recovered_types = {signal.signal_type for signal in recovered.observation.detected_signals}
        persisted = tuple(
            record
            for record in await _records(session)
            if isinstance(record, Signal) and record.signal_id in recovered_ids
        )

        assert recovered_types == {SignalType.REPEATED_FAILURE, SignalType.TOOL_ERROR}
        assert len(persisted) == 2
        assert len({signal.signal_id for signal in persisted}) == 2
        assert (
            await _failed_tool_result(
                session,
                source_event_id="repeat-failure-2",
                seconds=4,
                action=second_action,
            )
            == recovered
        )


async def test_terminal_run_allows_only_signal_recovery_for_an_immutable_prior_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "terminal-recovery.sqlite3"
    session_a = _sqlite_session(path)
    async with session_a:
        await _start(session_a)
        action = await _action(session_a, source_event_id="recover-action", seconds=1)
        original_extract = DeterministicSignalExtractor.extract_report

        def crash_after_append(
            extractor: DeterministicSignalExtractor,
            context: DetectionContext,
        ) -> ExtractionReport:
            report = original_extract(extractor, context)
            if context.current.source_event_id == "recover-result":
                raise DetectorContractError()
            return report

        monkeypatch.setattr(
            DeterministicSignalExtractor,
            "extract_report",
            crash_after_append,
        )
        with pytest.raises(ShadowInvariantError):
            await _failed_tool_result(
                session_a,
                source_event_id="recover-result",
                seconds=2,
                action=action,
            )
        monkeypatch.setattr(
            DeterministicSignalExtractor,
            "extract_report",
            original_extract,
        )

        async with _sqlite_session(path) as session_b:
            await session_b.finish(
                source_event_id="run-end",
                occurred_at=NOW + timedelta(seconds=3),
            )

        recovered = await _failed_tool_result(
            session_a,
            source_event_id="recover-result",
            seconds=2,
            action=action,
        )
        records = await _records(session_a)
        run_end_index = next(
            index
            for index, record in enumerate(records)
            if isinstance(record, TraceEvent) and record.event_type is EventType.RUN_END
        )
        suffix = records[run_end_index + 1 :]

        assert len(recovered.observation.detected_signals) == 1
        assert recovered.observation.detected_signals[0].signal_type is SignalType.TOOL_ERROR
        assert suffix == recovered.observation.detected_signals
        assert all(isinstance(record, Signal) for record in suffix)
        assert recovered.observation.sequence < next(
            record.sequence
            for record in records
            if isinstance(record, TraceEvent) and record.event_type is EventType.RUN_END
        )


async def test_a_preexisting_mixed_record_fails_closed_before_the_next_shadow_write() -> None:
    async with _memory_session() as session:
        await _start(session)
        await session._repository.record_invocation_decision(_foreign_decision())
        before = await _records(session)

        with pytest.raises(ShadowStateError):
            await session.observation(
                source_event_id="blocked-by-mixed-ledger",
                occurred_at=NOW + timedelta(seconds=1),
                source=ShadowObservationSource.TASK_INPUT,
                payload={"bounded": True},
            )

        assert await _records(session) == before


@pytest.mark.parametrize("forgery", ["signal_id", "future_evidence", "detector_version"])
async def test_forged_future_or_mismatched_preexisting_signal_fails_closed(
    forgery: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _memory_session() as session:
        await _start(session)
        action = await _action(session, source_event_id="forged-signal-action", seconds=1)
        original_extract = DeterministicSignalExtractor.extract_report
        captured: list[ExtractionReport] = []

        def capture_then_crash(
            extractor: DeterministicSignalExtractor,
            context: DetectionContext,
        ) -> ExtractionReport:
            report = original_extract(extractor, context)
            if context.current.source_event_id == "forged-signal-result":
                captured.append(report)
                raise DetectorContractError()
            return report

        monkeypatch.setattr(
            DeterministicSignalExtractor,
            "extract_report",
            capture_then_crash,
        )
        with pytest.raises(ShadowInvariantError):
            await _failed_tool_result(
                session,
                source_event_id="forged-signal-result",
                seconds=2,
                action=action,
            )
        monkeypatch.setattr(
            DeterministicSignalExtractor,
            "extract_report",
            original_extract,
        )
        expected = captured[-1].signals[0]
        updates: dict[str, object]
        if forgery == "signal_id":
            updates = {"signal_id": _FORGED_SIGNAL_ID}
        elif forgery == "detector_version":
            updates = {"detector_version": "forged-detector/v1"}
        else:
            future = await session.observation(
                source_event_id="future-of-forged-signal",
                occurred_at=NOW + timedelta(seconds=3),
                source=ShadowObservationSource.TASK_INPUT,
                payload={"future": True},
            )
            updates = {"evidence_event_ids": (future.ref.event_id,)}
        forged_values = expected.model_dump(mode="python", warnings=False)
        forged_values.update(updates)
        forged = Signal.model_validate(forged_values)
        await session._repository.record_signal(forged)
        before = await _records(session)

        with pytest.raises(ShadowStateError):
            await _failed_tool_result(
                session,
                source_event_id="forged-signal-result",
                seconds=2,
                action=action,
            )

        assert await _records(session) == before


async def test_duplicate_older_observation_is_immune_to_future_events() -> None:
    async with _memory_session() as session:
        await _start(session)
        earlier = await _action(session, source_event_id="stable-old-action", seconds=1)
        later = await _action(session, source_event_id="future-action", seconds=2)
        await _failed_tool_result(
            session,
            source_event_id="future-result",
            seconds=3,
            action=later,
        )

        duplicate = await _action(session, source_event_id="stable-old-action", seconds=1)

        assert duplicate == earlier
        assert duplicate.observation.observation_digest == earlier.observation.observation_digest
        assert duplicate.observation.event_prefix_digest == earlier.observation.event_prefix_digest
        assert (
            duplicate.observation.feature_snapshot_digest
            == earlier.observation.feature_snapshot_digest
        )


async def test_cancellation_after_committed_append_propagates_and_prefix_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _memory_session() as session:
        await _start(session)
        action = await _action(session, source_event_id="cancel-action", seconds=1)
        original_append = MemoryRunRepository.append_event_if_head
        committed = asyncio.Event()
        never = asyncio.Event()

        async def append_then_wait_for_cancellation(
            repository: MemoryRunRepository,
            event: Any,
            *,
            event_id: UUID,
            expected_head: LedgerHead | None,
        ) -> AppendReceipt:
            receipt = await original_append(
                repository,
                event,
                event_id=event_id,
                expected_head=expected_head,
            )
            if event.source_event_id == "cancel-result":
                committed.set()
                await never.wait()
            return receipt

        monkeypatch.setattr(
            MemoryRunRepository,
            "append_event_if_head",
            append_then_wait_for_cancellation,
        )
        task = asyncio.create_task(
            _failed_tool_result(
                session,
                source_event_id="cancel-result",
                seconds=2,
                action=action,
            )
        )
        await asyncio.wait_for(committed.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        records = await _records(session)
        assert any(
            isinstance(record, TraceEvent) and record.source_event_id == "cancel-result"
            for record in records
        )
        assert all(not isinstance(record, Signal) for record in records)

        monkeypatch.setattr(
            MemoryRunRepository,
            "append_event_if_head",
            original_append,
        )
        recovered = await _failed_tool_result(
            session,
            source_event_id="cancel-result",
            seconds=2,
            action=action,
        )
        assert tuple(signal.signal_type for signal in recovered.observation.detected_signals) == (
            SignalType.TOOL_ERROR,
        )


async def test_shadow_vertical_slice_touches_no_socket_provider_memory_or_runtime_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def forbidden_socket(*_args: object, **_kwargs: object) -> Any:
        calls.append("socket")
        raise AssertionError("Shadow Mode attempted to open a socket")

    async def forbidden_repository_path(*_args: object, **_kwargs: object) -> Any:
        calls.append("active-runtime")
        raise AssertionError("Shadow Mode attempted an active runtime operation")

    for method_name in (
        "record_invocation_decision",
        "record_outcome",
        "begin_cycle",
        "reserve_cycle",
        "mark_cycle_running",
        "commit_cycle",
        "fail_cycle",
        "preview_memory_delta",
        "budget_snapshot",
        "recover_cycles",
        "search",
        "snapshot",
        "claim_delivery",
        "begin_delivery_attempt",
        "complete_delivery",
        "mark_delivery_unknown",
        "reject_delivery",
        "recover_deliveries",
    ):
        monkeypatch.setattr(MemoryRunRepository, method_name, forbidden_repository_path)
    monkeypatch.setattr(socket, "socket", forbidden_socket)
    monkeypatch.setattr(socket, "create_connection", forbidden_socket)
    modules_before = set(sys.modules)

    async with _memory_session() as session:
        await _start(session)
        action = await _action(session, source_event_id="guarded-action", seconds=1)
        result = await _failed_tool_result(
            session,
            source_event_id="guarded-result",
            seconds=2,
            action=action,
        )

    imported = set(sys.modules).difference(modules_before)
    forbidden_imports = {
        name
        for name in imported
        if name == "httpx"
        or name.startswith("httpx.")
        or name == "openai_harmony"
        or name.startswith("openai_harmony.")
        or name.startswith("saliencegate.models")
        or name.startswith("saliencegate.memory")
        or name.startswith("saliencegate.runtime")
    }
    assert result.observation.model_calls == 0
    assert result.observation.memory_revisions == 0
    assert result.observation.cycles_created == 0
    assert result.observation.deliveries == 0
    assert calls == []
    assert forbidden_imports == set()
