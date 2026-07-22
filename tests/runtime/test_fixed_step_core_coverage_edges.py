"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from tests.runtime.test_fixed_step_runtime import (
    _make_repository,
    _message_event,
    _run,
    _run_start,
)

import saliencegate.runtime.fixed_step_core as fixed_step_core
from saliencegate.domain import InvocationDecision
from saliencegate.ports.repository import RunRepository
from saliencegate.ports.trajectory import LogicalMessageRole
from saliencegate.runtime.fixed_step_core import (
    FixedStepTraceBoundary,
    FixedStepTraceDriver,
    FixedStepTraceInput,
    FixedStepTraceInputError,
    FixedStepTraceInvariantError,
    _append_reconciled_trace_event,
    _authoritative_invocation_decision,
    _authoritative_trace_event,
    record_reconciled_invocation_decision,
)


def _repository() -> RunRepository:
    return _make_repository("memory", Path("unused-fixed-step-core.sqlite3"))


def _trace_input(ordinal: int = 1) -> FixedStepTraceInput:
    item = (
        _run_start()
        if ordinal == 1
        else _message_event(ordinal, step=1, role=LogicalMessageRole.ASSISTANT)
    )
    return FixedStepTraceInput(
        draft=item.draft,
        expected_event_id=item.expected_event_id,
        task_description=item.task_description,
        logical_messages=item.logical_messages,
        action_step=item.action_step,
        target_request_id=item.target_request_id,
    )


class _RepositoryProxy:
    def __init__(self, inner: RunRepository) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class _NonEquivalentRebuildRepository(_RepositoryProxy):
    async def rebuild(self, run_id: UUID):
        rebuilt = await self.inner.rebuild(run_id)
        return rebuilt.model_copy(update={"equivalent": False})


class _WrongLedgerCountRepository(_RepositoryProxy):
    async def ledger_head(self, run_id: UUID):
        head = await self.inner.ledger_head(run_id)
        return head.model_copy(update={"entry_count": head.entry_count + 1})


async def _noop_boundary(_boundary: FixedStepTraceBoundary) -> None:
    return None


@pytest.mark.asyncio
async def test_authoritative_trace_event_distinguishes_conflict_from_absence() -> None:
    conflicting = _repository()
    item = _trace_input()
    await conflicting.append(
        item.draft,
        event_id=UUID("00000000-0000-4000-8000-00000000b001"),
    )

    with pytest.raises(FixedStepTraceInvariantError):
        await _authoritative_trace_event(conflicting, item, expected_sequence=1)

    absent = _repository()
    first = _trace_input()
    await absent.append(first.draft, event_id=first.expected_event_id)

    assert (
        await _authoritative_trace_event(
            absent,
            _trace_input(2),
            expected_sequence=2,
        )
        is None
    )


@pytest.mark.asyncio
async def test_authoritative_decision_rejects_an_identifier_conflict() -> None:
    repository = _repository()
    result, _client = await _run(
        repository,
        (_run_start(),),
        mode="silence",
        cycle_capacity=1,
    )
    decision = result.decisions[0].model_copy(
        update={"decision_id": UUID("00000000-0000-4000-8000-00000000b002")}
    )

    with pytest.raises(FixedStepTraceInvariantError):
        await _authoritative_invocation_decision(repository, decision)

    mismatched = result.decisions[0].model_copy(
        update={"created_at": result.decisions[0].created_at + timedelta(microseconds=1)}
    )
    with pytest.raises(FixedStepTraceInvariantError):
        await _authoritative_invocation_decision(repository, mismatched)


class _FailingAppendRepository:
    async def append(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("append failed")


class _FailingDecisionRepository:
    async def record_invocation_decision(self, _decision: InvocationDecision) -> None:
        raise RuntimeError("decision failed")


class _ReceiptRepository:
    def __init__(self, receipt: Any) -> None:
        self.receipt = receipt

    async def record_invocation_decision(self, _decision: InvocationDecision) -> Any:
        return self.receipt


def _cancelled_reconciliation() -> tuple[
    Any,
    Any,
]:
    calls = 0

    async def complete(operation: Awaitable[Any]) -> tuple[Any, bool]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return await operation, False
        close = getattr(operation, "close", None)
        if close is not None:
            close()
        return None, True

    return complete, lambda: calls


@pytest.mark.asyncio
async def test_append_error_preserves_cancellation_during_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete, call_count = _cancelled_reconciliation()
    monkeypatch.setattr(fixed_step_core, "_complete_boundary", complete)

    with pytest.raises(asyncio.CancelledError):
        await _append_reconciled_trace_event(
            _FailingAppendRepository(),  # type: ignore[arg-type]
            _trace_input(),
            expected_sequence=1,
        )

    assert call_count() == 2


@pytest.mark.asyncio
async def test_decision_error_preserves_cancellation_during_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository()
    result, _client = await _run(
        repository,
        (_run_start(),),
        mode="silence",
        cycle_capacity=1,
    )
    complete, call_count = _cancelled_reconciliation()
    monkeypatch.setattr(fixed_step_core, "_complete_boundary", complete)

    with pytest.raises(asyncio.CancelledError):
        await record_reconciled_invocation_decision(
            _FailingDecisionRepository(),  # type: ignore[arg-type]
            result.decisions[0],
        )

    assert call_count() == 2


@pytest.mark.asyncio
async def test_decision_write_requires_matching_authoritative_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository()
    result, _client = await _run(
        repository,
        (_run_start(),),
        mode="silence",
        cycle_capacity=1,
    )
    receipt = await repository.record_invocation_decision(result.decisions[0])
    appended = receipt.model_copy(update={"appended": True})

    async def missing_authority(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        fixed_step_core,
        "_authoritative_invocation_decision",
        missing_authority,
    )

    with pytest.raises(FixedStepTraceInvariantError):
        await record_reconciled_invocation_decision(
            _ReceiptRepository(appended),  # type: ignore[arg-type]
            result.decisions[0],
        )


@pytest.mark.asyncio
async def test_trace_driver_rejects_empty_input() -> None:
    with pytest.raises(FixedStepTraceInputError):
        await FixedStepTraceDriver(_repository()).run((), _noop_boundary)


@pytest.mark.asyncio
async def test_trace_driver_wraps_binding_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_binding(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("binding failed")

    monkeypatch.setattr(fixed_step_core, "bind_persisted_trajectory_event", fail_binding)

    with pytest.raises(FixedStepTraceInvariantError):
        await FixedStepTraceDriver(_repository()).run((_trace_input(),), _noop_boundary)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "repository_type",
    (_NonEquivalentRebuildRepository, _WrongLedgerCountRepository),
)
async def test_trace_driver_rejects_invalid_final_repository_attestation(
    repository_type: type[_RepositoryProxy],
) -> None:
    repository = repository_type(_repository())

    with pytest.raises(FixedStepTraceInvariantError):
        await FixedStepTraceDriver(repository).run(  # type: ignore[arg-type]
            (_trace_input(),),
            _noop_boundary,
        )
