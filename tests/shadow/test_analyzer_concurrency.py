from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.shadow.test_trace import ENVIRONMENT_DIGEST, build_trace

import saliencegate.shadow.analyzer as analyzer_module
from saliencegate.domain import TraceEvent
from saliencegate.ports.repository import (
    ConditionalAppendOperation,
    ConditionalBatchReceipt,
    ConditionalEventAppend,
    LedgerHead,
    LedgerHeadConflictError,
    RunNotFoundError,
)
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.security import InstallationKey
from saliencegate.shadow import ShadowAnalyzer, ShadowSession, ShadowStateError
from saliencegate.shadow.inputs import (
    ShadowActionInput,
    derive_shadow_event_id,
    project_shadow_input,
)
from saliencegate.shadow.trace import ShadowTrace

_KEY = InstallationKey(b"c" * 32)
_ACTION_AT = datetime(2026, 7, 17, 9, 0, 1, tzinfo=UTC)


def _memory_session(trace: ShadowTrace) -> ShadowSession:
    return ShadowSession.in_memory_for_trace(
        run_id=trace.run_id,
        trace_binding=trace.binding,
        installation_key=_KEY,
    )


def _memory_repository(session: ShadowSession) -> MemoryRunRepository:
    repository = session._repository
    assert type(repository) is MemoryRunRepository
    return repository


def _large_trace() -> ShadowTrace:
    started_at = datetime(2026, 7, 17, 9, 0, tzinfo=UTC)

    def timestamp(offset: int) -> str:
        return (started_at + timedelta(seconds=offset)).isoformat().replace("+00:00", "Z")

    records: list[dict[str, object]] = [
        {
            "schema_version": "shadow-input/v1",
            "kind": "run_start",
            "source_event_id": "large-start",
            "occurred_at": timestamp(0),
        }
    ]
    records.extend(
        {
            "schema_version": "shadow-input/v1",
            "kind": "observation",
            "source_event_id": f"large-observation-{index}",
            "occurred_at": timestamp(index),
            "source": "task_input",
            "payload": {"sequence": index},
        }
        for index in range(1, 129)
    )
    records.append(
        {
            "schema_version": "shadow-input/v1",
            "kind": "run_end",
            "source_event_id": "large-finish",
            "occurred_at": timestamp(129),
        }
    )
    return build_trace(records)


@pytest.mark.asyncio
async def test_one_analyzer_serializes_concurrent_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    analyzer = ShadowAnalyzer(session)
    first_analysis_started = asyncio.Event()
    release_first_analysis = asyncio.Event()
    second_call_started = asyncio.Event()
    analysis_tasks: set[object] = set()
    batch_calls: list[tuple[ConditionalAppendOperation, ...]] = []
    original_analyze_prepared = analyzer_module._analyze_prepared
    original_append = MemoryRunRepository.append_records_if_head

    async def paused_first_analysis(current: ShadowSession, prepared: object) -> object:
        task = asyncio.current_task()
        assert task is not None
        if task not in analysis_tasks:
            analysis_tasks.add(task)
            if len(analysis_tasks) == 1:
                first_analysis_started.set()
                await release_first_analysis.wait()
        return await original_analyze_prepared(current, prepared)  # type: ignore[arg-type]

    async def counted_append(
        repository: MemoryRunRepository,
        operations: tuple[ConditionalAppendOperation, ...],
        *,
        expected_head: LedgerHead | None,
    ) -> ConditionalBatchReceipt:
        batch_calls.append(operations)
        return await original_append(repository, operations, expected_head=expected_head)

    async def run_second() -> object:
        second_call_started.set()
        return await analyzer.analyze(trace)

    monkeypatch.setattr(analyzer_module, "_analyze_prepared", paused_first_analysis)
    monkeypatch.setattr(MemoryRunRepository, "append_records_if_head", counted_append)

    async with session:
        first = asyncio.create_task(analyzer.analyze(trace))
        await asyncio.wait_for(first_analysis_started.wait(), timeout=5)
        second = asyncio.create_task(run_second())
        await asyncio.wait_for(second_call_started.wait(), timeout=5)
        await asyncio.sleep(0)

        assert len(analysis_tasks) == 1

        release_first_analysis.set()
        first_report, second_report = await asyncio.wait_for(
            asyncio.gather(first, second),
            timeout=10,
        )
        entries = await _memory_repository(session).ledger(trace.run_id)

    assert first_report.run_id == second_report.run_id == trace.run_id
    assert len(analysis_tasks) == 2
    assert len(batch_calls) == 1
    assert sum(type(entry.record) is TraceEvent for entry in entries) == len(trace.records)


@pytest.mark.asyncio
async def test_distinct_analyzers_share_session_serialization_without_duplicate_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    both_analyses_prepared = asyncio.Event()
    analysis_tasks: set[object] = set()
    batch_calls: list[tuple[ConditionalAppendOperation, ...]] = []
    original_analyze_prepared = analyzer_module._analyze_prepared
    original_append = MemoryRunRepository.append_records_if_head

    async def aligned_analyses(current: ShadowSession, prepared: object) -> object:
        task = asyncio.current_task()
        assert task is not None
        if task not in analysis_tasks:
            analysis_tasks.add(task)
            if len(analysis_tasks) == 2:
                both_analyses_prepared.set()
            await both_analyses_prepared.wait()
        return await original_analyze_prepared(current, prepared)  # type: ignore[arg-type]

    async def counted_append(
        repository: MemoryRunRepository,
        operations: tuple[ConditionalAppendOperation, ...],
        *,
        expected_head: LedgerHead | None,
    ) -> ConditionalBatchReceipt:
        batch_calls.append(operations)
        return await original_append(repository, operations, expected_head=expected_head)

    monkeypatch.setattr(analyzer_module, "_analyze_prepared", aligned_analyses)
    monkeypatch.setattr(MemoryRunRepository, "append_records_if_head", counted_append)

    async with session:
        reports = await asyncio.wait_for(
            asyncio.gather(
                ShadowAnalyzer(session).analyze(trace),
                ShadowAnalyzer(session).analyze(trace),
            ),
            timeout=10,
        )
        entries = await _memory_repository(session).ledger(trace.run_id)

    assert len(analysis_tasks) == 2
    assert len(batch_calls) == 1
    assert all(report.run_id == trace.run_id for report in reports)
    assert sum(type(entry.record) is TraceEvent for entry in entries) == len(trace.records)


@pytest.mark.asyncio
async def test_cas_conflict_resumes_only_from_an_authenticated_exact_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    baseline_session = _memory_session(trace)
    async with baseline_session:
        await ShadowAnalyzer(baseline_session).analyze(trace)
        baseline = await _memory_repository(baseline_session).ledger(trace.run_id)

    raced_session = _memory_session(trace)
    calls: list[tuple[ConditionalAppendOperation, ...]] = []
    expected_heads: list[LedgerHead | None] = []
    original_append = MemoryRunRepository.append_records_if_head
    extension_injected = False

    async def inject_exact_extension(
        repository: MemoryRunRepository,
        operations: tuple[ConditionalAppendOperation, ...],
        *,
        expected_head: LedgerHead | None,
    ) -> ConditionalBatchReceipt:
        nonlocal extension_injected
        calls.append(operations)
        expected_heads.append(expected_head)
        if not extension_injected:
            extension_injected = True
            assert type(operations[0]) is ConditionalEventAppend
            await original_append(repository, (operations[0],), expected_head=None)
            raise LedgerHeadConflictError()
        return await original_append(repository, operations, expected_head=expected_head)

    monkeypatch.setattr(
        MemoryRunRepository,
        "append_records_if_head",
        inject_exact_extension,
    )

    async with raced_session:
        report = await ShadowAnalyzer(raced_session).analyze(trace)
        raced = await _memory_repository(raced_session).ledger(trace.run_id)

    assert report.run_id == trace.run_id
    assert report.shadow_report.initial_ledger_entry_count == 0
    assert report.shadow_report.appended_event_count == len(trace.records)
    assert report.shadow_report.preexisting_event_count == 0
    assert raced == baseline
    assert len(calls) == 2
    assert expected_heads[0] is None
    assert expected_heads[1] is not None
    assert expected_heads[1].entry_count == 1
    second_events = tuple(
        operation for operation in calls[1] if type(operation) is ConditionalEventAppend
    )
    assert all(operation.event.source_event_id != "start-1" for operation in second_events)


@pytest.mark.asyncio
async def test_last_allowed_conflict_accepts_an_exactly_completed_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    repository = _memory_repository(session)
    original_append = MemoryRunRepository.append_records_if_head
    calls = 0

    async def commit_then_report_conflict(
        current_repository: MemoryRunRepository,
        operations: tuple[ConditionalAppendOperation, ...],
        *,
        expected_head: LedgerHead | None,
    ) -> ConditionalBatchReceipt:
        nonlocal calls
        calls += 1
        await original_append(
            current_repository,
            operations,
            expected_head=expected_head,
        )
        raise LedgerHeadConflictError()

    monkeypatch.setattr(analyzer_module, "_MAX_CAS_ATTEMPTS", 1)
    monkeypatch.setattr(
        MemoryRunRepository,
        "append_records_if_head",
        commit_then_report_conflict,
    )

    async with session:
        report = await ShadowAnalyzer(session).analyze(trace)
        entries = await repository.ledger(trace.run_id)

    assert calls == 1
    assert report.shadow_report.initial_ledger_entry_count == 0
    assert report.shadow_report.preexisting_event_count == 0
    assert report.shadow_report.appended_event_count == len(trace.records)
    assert sum(type(entry.record) is TraceEvent for entry in entries) == len(trace.records)


@pytest.mark.asyncio
async def test_last_allowed_conflict_rejects_an_incomplete_exact_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    repository = _memory_repository(session)
    original_append = MemoryRunRepository.append_records_if_head
    calls = 0

    async def commit_prefix_then_report_conflict(
        current_repository: MemoryRunRepository,
        operations: tuple[ConditionalAppendOperation, ...],
        *,
        expected_head: LedgerHead | None,
    ) -> ConditionalBatchReceipt:
        nonlocal calls
        calls += 1
        await original_append(
            current_repository,
            operations[:1],
            expected_head=expected_head,
        )
        raise LedgerHeadConflictError()

    monkeypatch.setattr(analyzer_module, "_MAX_CAS_ATTEMPTS", 1)
    monkeypatch.setattr(
        MemoryRunRepository,
        "append_records_if_head",
        commit_prefix_then_report_conflict,
    )

    async with session:
        with pytest.raises(ShadowStateError):
            await ShadowAnalyzer(session).analyze(trace)
        entries = await repository.ledger(trace.run_id)

    assert calls == 1
    assert len(entries) == 1
    assert type(entries[0].record) is TraceEvent


@pytest.mark.asyncio
async def test_cas_conflict_rejects_a_divergent_authenticated_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    repository = _memory_repository(session)
    secret = "fixture-secret-concurrent-divergence"
    divergent_action = ConditionalEventAppend(
        event=project_shadow_input(
            ShadowActionInput(
                source_event_id="action-1",
                occurred_at=_ACTION_AT,
                command=secret,
                working_directory="/private/project",
                environment_digest=ENVIRONMENT_DIGEST,
            ),
            run_id=trace.run_id,
            source_adapter=trace.binding.source_adapter,
        ),
        event_id=derive_shadow_event_id(trace.run_id, "action-1"),
    )
    original_append = MemoryRunRepository.append_records_if_head
    calls = 0

    async def inject_divergence(
        current_repository: MemoryRunRepository,
        operations: tuple[ConditionalAppendOperation, ...],
        *,
        expected_head: LedgerHead | None,
    ) -> ConditionalBatchReceipt:
        nonlocal calls
        calls += 1
        assert calls == 1
        assert expected_head is None
        assert type(operations[0]) is ConditionalEventAppend
        await original_append(
            current_repository,
            (operations[0], divergent_action),
            expected_head=None,
        )
        raise LedgerHeadConflictError()

    monkeypatch.setattr(MemoryRunRepository, "append_records_if_head", inject_divergence)

    async with session:
        with pytest.raises(ShadowStateError) as captured:
            await ShadowAnalyzer(session).analyze(trace)
        entries = await repository.ledger(trace.run_id)

    assert calls == 1
    assert len(entries) == 2
    assert all(type(entry.record) is TraceEvent for entry in entries)
    assert secret not in str(captured.value)
    assert secret not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.asyncio
async def test_cas_conflict_without_authenticated_progress_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    repository = _memory_repository(session)
    calls = 0

    async def no_progress(
        _repository: MemoryRunRepository,
        _operations: tuple[ConditionalAppendOperation, ...],
        *,
        expected_head: LedgerHead | None,
    ) -> ConditionalBatchReceipt:
        nonlocal calls
        calls += 1
        assert expected_head is None
        raise LedgerHeadConflictError()

    monkeypatch.setattr(MemoryRunRepository, "append_records_if_head", no_progress)

    async with session:
        with pytest.raises(ShadowStateError):
            await ShadowAnalyzer(session).analyze(trace)
        with pytest.raises(RunNotFoundError):
            await repository.ledger(trace.run_id)

    assert calls == 1


@pytest.mark.asyncio
async def test_cancellation_before_batch_mutation_leaves_ledger_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    repository = _memory_repository(session)
    batch_entered = asyncio.Event()
    allow_batch = asyncio.Event()
    original_append = MemoryRunRepository.append_records_if_head

    async def paused_before_mutation(
        current_repository: MemoryRunRepository,
        operations: tuple[ConditionalAppendOperation, ...],
        *,
        expected_head: LedgerHead | None,
    ) -> ConditionalBatchReceipt:
        batch_entered.set()
        await allow_batch.wait()
        return await original_append(
            current_repository,
            operations,
            expected_head=expected_head,
        )

    monkeypatch.setattr(
        MemoryRunRepository,
        "append_records_if_head",
        paused_before_mutation,
    )

    async with session:
        task = asyncio.create_task(ShadowAnalyzer(session).analyze(trace))
        await asyncio.wait_for(batch_entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(RunNotFoundError):
            await repository.ledger(trace.run_id)

    assert not allow_batch.is_set()


@pytest.mark.asyncio
async def test_cancellation_during_deep_batch_staging_discards_the_whole_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _large_trace()
    session = _memory_session(trace)
    repository = _memory_repository(session)
    original_append = repository._append_conditional_to_slot
    calls = 0

    def cancelling_append(*args: object, **kwargs: object) -> object:
        nonlocal calls
        receipt = original_append(*args, **kwargs)
        calls += 1
        if calls == 2:
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
        return receipt

    monkeypatch.setattr(repository, "_append_conditional_to_slot", cancelling_append)

    async with session:
        task = asyncio.create_task(ShadowAnalyzer(session).analyze(trace))
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(RunNotFoundError):
            await repository.ledger(trace.run_id)

    assert calls == 128


@pytest.mark.asyncio
async def test_cancellation_after_atomic_publish_keeps_full_idempotent_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    repository = _memory_repository(session)
    release_started = asyncio.Event()
    allow_release = asyncio.Event()
    original_release = repository._release_append_slot

    async def delayed_release(slot: object) -> None:
        release_started.set()
        await allow_release.wait()
        await original_release(slot)  # type: ignore[arg-type]

    monkeypatch.setattr(repository, "_release_append_slot", delayed_release)

    async with session:
        task = asyncio.create_task(ShadowAnalyzer(session).analyze(trace))
        await asyncio.wait_for(release_started.wait(), timeout=5)
        task.cancel()
        await asyncio.sleep(0)
        allow_release.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        published = await repository.ledger(trace.run_id)
        rerun = await ShadowAnalyzer(session).analyze(trace)
        after_rerun = await repository.ledger(trace.run_id)

    assert sum(type(entry.record) is TraceEvent for entry in published) == len(trace.records)
    assert after_rerun == published
    assert rerun.shadow_report.appended_event_count == 0
