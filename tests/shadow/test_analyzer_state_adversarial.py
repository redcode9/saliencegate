from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from tests.shadow.test_analyzer import (
    _count_memory_batches,
    _memory_session,
    _seed_action,
    _seed_start,
)
from tests.shadow.test_trace import build_trace, complete_records

import saliencegate.shadow.analyzer as analyzer_module
from saliencegate.domain import Signal, TraceEvent
from saliencegate.ports.repository import (
    AppendReceipt,
    ConditionalAppendOperation,
    ConditionalBatchReceipt,
    ConditionalSignalAppend,
    LedgerHead,
    LedgerReceipt,
)
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.shadow import (
    ShadowAnalyzer,
    ShadowInputError,
    ShadowInvariantError,
    ShadowSession,
    ShadowStateError,
)
from saliencegate.shadow.inputs import ShadowObservationSource
from saliencegate.shadow.trace import ShadowTrace
from saliencegate.signals import DeterministicSignalExtractor
from saliencegate.signals.base import _TrustedDetectionContext, _TrustedExtraction

_EXTRA_SIGNAL_ID = UUID("44444444-4444-4444-8444-444444444444")
_EXTRA_EVENT_AT = datetime(2026, 7, 17, 9, 0, 2, tzinfo=UTC)


def _memory_repository(session: ShadowSession) -> MemoryRunRepository:
    repository = session._repository
    assert type(repository) is MemoryRunRepository
    return repository


def _prepare(
    session: ShadowSession,
    trace: ShadowTrace,
) -> analyzer_module._PreparedAnalysis:
    prepared = analyzer_module._prepare_analysis(session, trace)
    assert type(prepared) is analyzer_module._PreparedAnalysis
    return prepared


async def _seed_complete_events_without_signals(
    repository: MemoryRunRepository,
    prepared: analyzer_module._PreparedAnalysis,
) -> None:
    operations = tuple(item.operation for item in prepared.events)
    receipt = await repository.append_records_if_head(operations, expected_head=None)
    assert len(receipt.receipts) == len(prepared.events)


def _extra_signal(
    prepared: analyzer_module._PreparedAnalysis,
    *,
    signal_id: UUID = _EXTRA_SIGNAL_ID,
) -> Signal:
    reference = prepared.signals[0].signal
    return Signal(
        signal_id=signal_id,
        run_id=reference.run_id,
        created_at=reference.created_at,
        signal_type=reference.signal_type,
        strength=0.5,
        evidence_event_ids=(prepared.events[-1].event.event_id,),
        detector_version="fixture/1",
        reason_code=reference.reason_code,
    )


@pytest.mark.asyncio
async def test_last_record_preflight_failure_happens_before_any_repository_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    parsed_records = 0
    repository_accesses = 0
    original_parse = analyzer_module._parse_json_object

    def counted_parse(value: bytes) -> dict[str, object]:
        nonlocal parsed_records
        parsed_records += 1
        parsed = original_parse(value)
        if parsed_records == len(trace.records):
            raise ShadowInputError()
        return parsed

    def forbidden_repository_access(_session: ShadowSession) -> MemoryRunRepository:
        nonlocal repository_accesses
        repository_accesses += 1
        raise AssertionError("preflight failure reached repository access")

    monkeypatch.setattr(analyzer_module, "_parse_json_object", counted_parse)
    monkeypatch.setattr(
        ShadowSession,
        "_repository_for_operation",
        forbidden_repository_access,
    )

    async with session:
        with pytest.raises(ShadowInputError):
            await ShadowAnalyzer(session).analyze(trace)

    assert parsed_records == len(trace.records)
    assert repository_accesses == 0


@pytest.mark.asyncio
async def test_last_record_detector_failure_happens_before_any_repository_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    detector_calls = 0
    repository_accesses = 0
    original_extract = analyzer_module._extract_trusted_report

    def fail_on_last_detector_call(
        extractor: DeterministicSignalExtractor,
        context: _TrustedDetectionContext,
    ) -> _TrustedExtraction:
        nonlocal detector_calls
        detector_calls += 1
        if detector_calls == len(trace.records):
            raise ShadowInvariantError()
        return original_extract(extractor, context)

    def forbidden_repository_access(_session: ShadowSession) -> MemoryRunRepository:
        nonlocal repository_accesses
        repository_accesses += 1
        raise AssertionError("detector failure reached repository access")

    monkeypatch.setattr(
        analyzer_module,
        "_extract_trusted_report",
        fail_on_last_detector_call,
    )
    monkeypatch.setattr(
        ShadowSession,
        "_repository_for_operation",
        forbidden_repository_access,
    )

    async with session:
        with pytest.raises(ShadowInvariantError):
            await ShadowAnalyzer(session).analyze(trace)

    assert detector_calls == len(trace.records)
    assert repository_accesses == 0


@pytest.mark.asyncio
async def test_authenticated_extra_event_fails_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace(complete_records()[:2], capture_scope="unknown")
    session = _memory_session(trace)

    async with session:
        await _seed_start(session)
        await _seed_action(session)
        await session.observation(
            source_event_id="extra-observation",
            occurred_at=_EXTRA_EVENT_AT,
            source=ShadowObservationSource.TASK_INPUT,
            payload={"kind": "extra"},
        )
        repository = _memory_repository(session)
        before = await repository.ledger(trace.run_id)
        batch_calls = _count_memory_batches(monkeypatch)

        with pytest.raises(ShadowStateError):
            await ShadowAnalyzer(session).analyze(trace)

        after = await repository.ledger(trace.run_id)

    assert before == after
    assert len(batch_calls) == 0
    assert sum(type(entry.record) is TraceEvent for entry in after) == len(trace.records) + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ("altered", "extra", "future_evidence"))
async def test_invalid_existing_signal_fails_before_batch_mutation(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    prepared = _prepare(session, trace)
    repository = _memory_repository(session)
    await _seed_complete_events_without_signals(repository, prepared)
    expected = prepared.signals[0].signal

    if mutation == "altered":
        signal = expected.model_copy(
            update={"strength": 0.125 if expected.strength != 0.125 else 0.875}
        )
    elif mutation == "extra":
        signal = _extra_signal(prepared)
    else:
        signal = expected.model_copy(
            update={"evidence_event_ids": (prepared.events[-1].event.event_id,)}
        )
    await repository.record_signal(signal)
    before = await repository.ledger(trace.run_id)
    batch_calls = _count_memory_batches(monkeypatch)

    async with session:
        with pytest.raises(ShadowStateError):
            await ShadowAnalyzer(session).analyze(trace)
        after = await repository.ledger(trace.run_id)

    assert before == after
    assert len(batch_calls) == 0


@pytest.mark.asyncio
async def test_complete_event_trace_repairs_only_the_missing_signal_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    prepared = _prepare(session, trace)
    repository = _memory_repository(session)
    await _seed_complete_events_without_signals(repository, prepared)
    expected_signals = tuple(item.signal for item in prepared.signals)
    assert len(expected_signals) >= 3
    preexisting = expected_signals[::2]
    missing = expected_signals[1::2]
    for signal in preexisting:
        await repository.record_signal(signal)
    initial_head = await repository.ledger_head(trace.run_id)
    batch_calls = _count_memory_batches(monkeypatch)

    async with session:
        report = await ShadowAnalyzer(session).analyze(trace)
        final_entries = await repository.ledger(trace.run_id)

    assert len(batch_calls) == 1
    assert all(type(operation) is ConditionalSignalAppend for operation in batch_calls[0])
    assert tuple(operation.signal.signal_id for operation in batch_calls[0]) == tuple(
        signal.signal_id for signal in missing
    )
    final_events = tuple(
        entry.record for entry in final_entries if type(entry.record) is TraceEvent
    )
    final_signals = tuple(entry.record for entry in final_entries if type(entry.record) is Signal)
    assert final_events == prepared.expected_events
    assert {signal.signal_id: signal for signal in final_signals} == {
        signal.signal_id: signal for signal in expected_signals
    }
    assert report.shadow_report.initial_ledger_entry_count == initial_head.entry_count
    assert report.shadow_report.preexisting_event_count == len(prepared.events)
    assert report.shadow_report.appended_event_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "inconsistency",
    (
        "initial_head",
        "receipt_count",
        "final_head",
        "gross_final_head",
        "event_cursor",
        "signal_record_tag",
        "signal_chain_tag",
    ),
)
async def test_incoherent_batch_receipt_fails_closed_after_atomic_commit(
    monkeypatch: pytest.MonkeyPatch,
    inconsistency: str,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    repository = _memory_repository(session)
    original_append = MemoryRunRepository.append_records_if_head
    calls = 0

    async def incoherent_receipt(
        current_repository: MemoryRunRepository,
        operations: tuple[ConditionalAppendOperation, ...],
        *,
        expected_head: LedgerHead | None,
    ) -> ConditionalBatchReceipt:
        nonlocal calls
        calls += 1
        receipt = await original_append(
            current_repository,
            operations,
            expected_head=expected_head,
        )
        if inconsistency == "initial_head":
            return receipt.model_copy(update={"initial_head": receipt.final_head})
        if inconsistency == "receipt_count":
            return receipt.model_copy(update={"receipts": receipt.receipts[:-1]})
        if inconsistency == "gross_final_head":
            return receipt.model_copy(update={"final_head": object()})
        if inconsistency == "event_cursor":
            changed_receipts = list(receipt.receipts)
            event_index = next(
                index
                for index, operation_receipt in enumerate(changed_receipts)
                if type(operation_receipt) is AppendReceipt
            )
            event_receipt = changed_receipts[event_index]
            assert type(event_receipt) is AppendReceipt
            changed_receipts[event_index] = event_receipt.model_copy(
                update={"ingestion_cursor": event_receipt.ingestion_cursor + 1}
            )
            return receipt.model_copy(update={"receipts": tuple(changed_receipts)})
        if inconsistency in ("signal_record_tag", "signal_chain_tag"):
            changed_receipts = list(receipt.receipts)
            signal_index = next(
                index
                for index, operation_receipt in enumerate(changed_receipts)
                if type(operation_receipt) is LedgerReceipt
            )
            signal_receipt = changed_receipts[signal_index]
            assert type(signal_receipt) is LedgerReceipt
            field_name = "record_tag" if inconsistency == "signal_record_tag" else "chain_tag"
            tag = getattr(signal_receipt, field_name)
            changed_receipts[signal_index] = signal_receipt.model_copy(
                update={field_name: tag.model_copy(update={"value": "0" * 64})}
            )
            return receipt.model_copy(update={"receipts": tuple(changed_receipts)})
        changed_head = receipt.final_head.model_copy(
            update={"entry_count": receipt.final_head.entry_count - 1}
        )
        return receipt.model_copy(update={"final_head": changed_head})

    monkeypatch.setattr(
        MemoryRunRepository,
        "append_records_if_head",
        incoherent_receipt,
    )

    async with session:
        with pytest.raises(ShadowStateError) as captured:
            await ShadowAnalyzer(session).analyze(trace)
        entries = await repository.ledger(trace.run_id)

    assert calls == 1
    assert sum(type(entry.record) is TraceEvent for entry in entries) == len(trace.records)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.asyncio
async def test_report_is_not_built_until_the_post_batch_state_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    repository = _memory_repository(session)
    prepared = _prepare(session, trace)
    extra = _extra_signal(prepared)
    original_append = MemoryRunRepository.append_records_if_head
    batch_calls = 0
    report_builder_calls = 0

    async def append_then_diverge(
        current_repository: MemoryRunRepository,
        operations: tuple[ConditionalAppendOperation, ...],
        *,
        expected_head: LedgerHead | None,
    ) -> ConditionalBatchReceipt:
        nonlocal batch_calls
        batch_calls += 1
        receipt = await original_append(
            current_repository,
            operations,
            expected_head=expected_head,
        )
        await current_repository.record_signal(extra)
        return receipt

    def forbidden_report_builder(*_args: object, **_kwargs: object) -> object:
        nonlocal report_builder_calls
        report_builder_calls += 1
        raise AssertionError("an inexact final state reached report construction")

    monkeypatch.setattr(
        MemoryRunRepository,
        "append_records_if_head",
        append_then_diverge,
    )
    monkeypatch.setattr(
        analyzer_module,
        "_build_shadow_trace_report_trusted",
        forbidden_report_builder,
    )

    async with session:
        with pytest.raises(ShadowStateError):
            await ShadowAnalyzer(session).analyze(trace)
        entries = await repository.ledger(trace.run_id)

    assert batch_calls == 1
    assert report_builder_calls == 0
    assert any(
        type(entry.record) is Signal and entry.record.signal_id == _EXTRA_SIGNAL_ID
        for entry in entries
    )
