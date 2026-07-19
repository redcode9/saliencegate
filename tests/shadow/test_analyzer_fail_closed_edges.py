from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest
from tests.shadow.test_analyzer import _memory_session
from tests.shadow.test_trace import build_trace

import saliencegate.shadow.analyzer as analyzer_module
from saliencegate.domain import TraceEvent
from saliencegate.ports.repository import (
    ConditionalBatchReceipt,
    ConditionalEventAppend,
    ConditionalSignalAppend,
)
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.shadow import (
    ShadowAnalyzer,
    ShadowConfigurationError,
    ShadowInputError,
    ShadowInvariantError,
    ShadowSession,
    ShadowStateError,
)
from saliencegate.shadow.io import PreflightedShadowTrace
from saliencegate.shadow.trace import ShadowTrace


def _prepared_pair() -> tuple[
    ShadowSession,
    analyzer_module._PreparedAnalysis,
]:
    trace = build_trace()
    session = _memory_session(trace)
    return session, analyzer_module._prepare_analysis(session, trace)


async def _persist_prepared(
    session: ShadowSession,
    prepared: analyzer_module._PreparedAnalysis,
) -> tuple[ConditionalBatchReceipt, analyzer_module._RunState]:
    repository = session._repository
    assert type(repository) is MemoryRunRepository
    receipt = await repository.append_records_if_head(
        prepared.full_operations,
        expected_head=None,
    )
    state = await session._load_state()
    assert type(state) is analyzer_module._RunState
    return receipt, state


def test_binding_comparison_fails_closed_but_preserves_process_interrupts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)

    def fail_encoding(_value: object) -> bytes:
        raise RuntimeError("untrusted binding encoder failure")

    monkeypatch.setattr(analyzer_module, "canonical_json", fail_encoding)
    assert not analyzer_module._bindings_match(session, trace)

    def interrupt_encoding(_value: object) -> bytes:
        raise SystemExit(17)

    monkeypatch.setattr(analyzer_module, "canonical_json", interrupt_encoding)
    with pytest.raises(SystemExit, match="17"):
        analyzer_module._bindings_match(session, trace)


def test_analysis_preflight_rejects_non_trace_and_legacy_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)

    with pytest.raises(ShadowInputError):
        analyzer_module._prepare_analysis(session, object())

    object.__setattr__(trace.binding, "identity_mode", "legacy_ordered")
    monkeypatch.setattr(ShadowTrace, "_copy_exact", lambda self: self)
    with pytest.raises(ShadowInputError):
        analyzer_module._prepare_analysis(session, trace)


@pytest.mark.parametrize(
    ("limit_name", "limit_value"),
    (
        ("MAX_CONDITIONAL_BATCH_EVENTS", 0),
        ("MAX_CONDITIONAL_BATCH_SIGNALS", 0),
        ("MAX_CONDITIONAL_BATCH_OPERATIONS", 0),
    ),
)
def test_analysis_preflight_enforces_each_atomic_batch_bound(
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit_value: int,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    monkeypatch.setattr(analyzer_module, limit_name, limit_value)

    with pytest.raises(ShadowInvariantError):
        analyzer_module._prepare_analysis(session, trace)


def test_analysis_preflight_rejects_a_projected_event_sequence_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    original = ShadowSession._preflight_event

    def corrupt_sequence(
        current: ShadowSession,
        *args: object,
        **kwargs: object,
    ) -> TraceEvent:
        event = original(current, *args, **kwargs)  # type: ignore[arg-type]
        return event.model_copy(update={"sequence": event.sequence + 1})

    monkeypatch.setattr(ShadowSession, "_preflight_event", corrupt_sequence)
    with pytest.raises(ShadowInvariantError):
        analyzer_module._prepare_analysis(session, trace)


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ("too_many", "unknown", "early_position"))
async def test_preexisting_signal_state_must_be_an_exact_causally_ordered_subset(
    corruption: str,
) -> None:
    session, prepared = _prepared_pair()
    _receipt, state = await _persist_prepared(session, prepared)
    signal = state.signals[0]

    if corruption == "too_many":
        damaged = replace(state, signals=state.signals + state.signals)
    elif corruption == "unknown":
        unknown = signal.model_copy(update={"signal_id": uuid4()})
        damaged = replace(state, signals=(unknown,))
    else:
        positions = dict(state.signal_positions)
        positions[signal.signal_id] = 0
        damaged = replace(state, signal_positions=positions)

    with pytest.raises(ShadowStateError):
        analyzer_module._validate_state(damaged, prepared)


@pytest.mark.asyncio
async def test_trace_state_loader_exhausts_only_bounded_snapshot_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _prepared = _prepared_pair()
    calls = 0

    async def always_racing(_session: ShadowSession) -> None:
        nonlocal calls
        calls += 1
        raise analyzer_module._RetryableSnapshotRaceError()

    monkeypatch.setattr(ShadowSession, "_load_state", always_racing)
    with pytest.raises(ShadowStateError):
        await analyzer_module._load_trace_state(session)
    assert calls == analyzer_module._MAX_CAS_ATTEMPTS


@pytest.mark.asyncio
async def test_exact_extension_classifier_rejects_absence_and_non_growth() -> None:
    session, prepared = _prepared_pair()
    _receipt, state = await _persist_prepared(session, prepared)

    assert not analyzer_module._is_strict_exact_extension(None, None)
    assert analyzer_module._is_strict_exact_extension(None, state)
    assert not analyzer_module._is_strict_exact_extension(state, state)


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ("receipt_count", "event_position", "signal_position"))
async def test_atomic_batch_receipt_must_match_every_requested_operation(
    corruption: str,
) -> None:
    session, prepared = _prepared_pair()
    receipt, _state = await _persist_prepared(session, prepared)

    if corruption == "receipt_count":
        operations = prepared.full_operations[:-1]
        damaged_prepared = prepared
    elif corruption == "event_position":
        first = prepared.events[0]
        changed = first.event.model_copy(update={"sequence": first.event.sequence + 99})
        damaged_prepared = replace(
            prepared,
            events=(replace(first, event=changed), *prepared.events[1:]),
        )
        operations = prepared.full_operations
    else:
        operations_list = list(prepared.full_operations)
        index = next(
            index
            for index, item in enumerate(operations_list)
            if type(item) is ConditionalSignalAppend
        )
        operation = operations_list[index]
        assert type(operation) is ConditionalSignalAppend
        signal = operation.signal.model_copy(update={"signal_id": uuid4()})
        operations_list[index] = operation.model_copy(update={"signal": signal})
        operations = tuple(operations_list)
        damaged_prepared = prepared

    with pytest.raises(ShadowStateError):
        analyzer_module._validate_batch_receipt(
            operations,
            damaged_prepared,
            initial_head=None,
            receipt=receipt,
        )


@pytest.mark.asyncio
async def test_receipt_to_ledger_validation_rejects_shape_position_and_unknown_receipts() -> None:
    session, prepared = _prepared_pair()
    receipt, state = await _persist_prepared(session, prepared)

    with pytest.raises(ShadowStateError):
        analyzer_module._validate_receipt_ledger_links(receipt, state.entries[:-1])

    changed_entry = state.entries[0].model_copy(update={"position": 99})
    with pytest.raises(ShadowStateError):
        analyzer_module._validate_receipt_ledger_links(
            receipt,
            (changed_entry, *state.entries[1:]),
        )

    unknown = SimpleNamespace(
        receipts=(SimpleNamespace(ledger_position=state.entries[0].position),)
    )
    with pytest.raises(ShadowStateError):
        analyzer_module._validate_receipt_ledger_links(  # type: ignore[arg-type]
            unknown,
            state.entries[:1],
        )


@pytest.mark.asyncio
async def test_receipt_to_ledger_validation_preserves_process_interrupts() -> None:
    session, prepared = _prepared_pair()
    _receipt, state = await _persist_prepared(session, prepared)

    class InterruptingReceipt:
        @property
        def ledger_position(self) -> int:
            raise KeyboardInterrupt()

    unknown = SimpleNamespace(receipts=(InterruptingReceipt(),))
    with pytest.raises(KeyboardInterrupt):
        analyzer_module._validate_receipt_ledger_links(  # type: ignore[arg-type]
            unknown,
            state.entries[:1],
        )


def test_operation_projection_rejects_an_unmapped_event_identity() -> None:
    _session, prepared = _prepared_pair()
    operation = prepared.events[0].operation.model_copy(update={"event_id": uuid4()})
    assert type(operation) is ConditionalEventAppend

    with pytest.raises(ShadowInvariantError):
        analyzer_module._operation_records((operation,), prepared)


def test_report_construction_rejects_rows_without_preflighted_observations() -> None:
    session, prepared = _prepared_pair()
    damaged = replace(prepared, events=())

    with pytest.raises(ShadowInvariantError):
        analyzer_module._build_run_report(
            session,
            damaged,
            initial_head=None,
            initial_event_count=0,
        )


@pytest.mark.asyncio
async def test_preview_and_analysis_reject_invalid_or_closed_session_state() -> None:
    session, prepared = _prepared_pair()
    with pytest.raises(ShadowInputError):
        await analyzer_module._preview_prepared(  # type: ignore[arg-type]
            session,
            object(),
            assume_empty=True,
        )

    await session.aclose()
    with pytest.raises(ShadowInputError):
        await analyzer_module._preview_prepared(session, prepared, assume_empty=True)
    with pytest.raises(ShadowInputError):
        await analyzer_module._analyze_prepared(session, prepared)


@pytest.mark.asyncio
async def test_public_analyzer_sanitizes_unexpected_internal_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)

    with pytest.raises(ShadowConfigurationError):
        ShadowAnalyzer(object())  # type: ignore[arg-type]

    def fail_preflight(_session: ShadowSession, _trace: object) -> None:
        raise RuntimeError("fixture-secret-internal-detail")

    monkeypatch.setattr(analyzer_module, "_prepare_analysis", fail_preflight)
    with pytest.raises(ShadowInvariantError) as captured:
        await ShadowAnalyzer(session).analyze(trace)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert "fixture-secret" not in repr(captured.value)


@pytest.mark.asyncio
async def test_legacy_analyzer_requires_exact_preflighted_input() -> None:
    trace = build_trace()
    session = _memory_session(trace)

    with pytest.raises(ShadowInputError):
        await analyzer_module._analyze_legacy_preflighted(
            session,
            object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ShadowInputError):
        await analyzer_module._analyze_legacy_preflighted(
            object(),  # type: ignore[arg-type]
            PreflightedShadowTrace,  # type: ignore[arg-type]
        )
