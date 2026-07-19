from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar
from uuid import UUID

from saliencegate.domain import (
    InvocationDecision,
    NormalizedTraceEventDraft,
    TraceEvent,
    canonical_digest,
)
from saliencegate.domain.records import ComponentIdentifier
from saliencegate.ports.repository import (
    AppendDisposition,
    LedgerEntry,
    LedgerHead,
    LedgerReceipt,
    ProjectionDigests,
    RunNotFoundError,
    RunRepository,
)
from saliencegate.ports.trajectory import (
    ActionStepBinding,
    AttestedTrajectoryPrefix,
    EventTextSelector,
    LogicalMessageBinding,
    TrajectoryBinding,
    TrajectoryPrefixRequest,
    bind_persisted_trajectory_event,
    resolve_trajectory_prefix,
)
from saliencegate.runtime.algorithm_result import algorithm_trace_digest
from saliencegate.runtime.message_window import MessageWindow, project_message_window
from saliencegate.runtime.scheduling import (
    FixedStepDecision,
    FixedStepSchedule,
    project_fixed_step_schedule,
)

_BoundaryProjection = TypeVar("_BoundaryProjection")
_BoundaryResult = TypeVar("_BoundaryResult")


class FixedStepTraceInputError(RuntimeError):
    """The trace cannot begin or one event cannot be appended exactly once."""

    def __init__(self) -> None:
        super().__init__("fixed-step trace input failed validation")


class FixedStepTraceInvariantError(RuntimeError):
    """The repository diverged from the trace projection contract."""

    def __init__(self) -> None:
        super().__init__("fixed-step trace authoritative state diverged")


@dataclass(frozen=True, slots=True)
class FixedStepTraceInput:
    """Validated event data required by the fixed-step trace driver."""

    draft: NormalizedTraceEventDraft
    expected_event_id: UUID
    task_description: EventTextSelector | None = None
    logical_messages: tuple[LogicalMessageBinding, ...] = ()
    action_step: ActionStepBinding | None = None
    target_request_id: ComponentIdentifier | None = None


@dataclass(frozen=True, slots=True)
class FixedStepTraceBoundary:
    """One authoritative event boundary, ready for an algorithm callback."""

    ordinal: int
    trace_digest: str
    trace_input: FixedStepTraceInput
    event: TraceEvent
    binding: TrajectoryBinding
    prefix: AttestedTrajectoryPrefix
    schedule: FixedStepSchedule
    scheduled: FixedStepDecision
    window: MessageWindow | None


@dataclass(frozen=True, slots=True)
class FixedStepTraceSpine:
    """Content-addressed trace projections shared by fixed-step conditions."""

    run_id: UUID
    trace_digest: str
    normalized_draft_digests: tuple[str, ...]
    persisted_events: tuple[TraceEvent, ...]
    persisted_event_draft_digests: tuple[str, ...]
    bindings: tuple[TrajectoryBinding, ...]
    trajectory_prefix: AttestedTrajectoryPrefix
    schedule: FixedStepSchedule
    windows: tuple[MessageWindow, ...]


@dataclass(frozen=True, slots=True)
class FixedStepTraceResult(Generic[_BoundaryProjection]):
    """Trace spine, callback projections, and rebuilt repository evidence."""

    spine: FixedStepTraceSpine
    boundary_projections: tuple[_BoundaryProjection, ...]
    projection_digests: ProjectionDigests
    ledger: tuple[LedgerEntry, ...]
    ledger_head: LedgerHead
    rebuild_equivalent: bool


def _persisted_draft_digest(event: TraceEvent) -> str:
    return canonical_digest(
        NormalizedTraceEventDraft(
            run_id=event.run_id,
            source_event_id=event.source_event_id,
            timestamp=event.timestamp,
            event_type=event.event_type,
            phase=event.phase,
            payload=event.payload,
            parent_ids=event.parent_ids,
            source_adapter=event.source_adapter,
            trust_label=event.trust_label,
        )
    )


async def _complete_boundary(
    operation: Awaitable[_BoundaryResult],
) -> tuple[_BoundaryResult, bool]:
    """Finish one repository boundary and remember caller cancellation."""

    task = asyncio.ensure_future(operation)
    cancellation_requested = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.cancelled():
                cancellation_requested = True
        except Exception:
            break
    try:
        result = task.result()
    except asyncio.CancelledError:
        raise
    except Exception:
        if cancellation_requested:
            raise asyncio.CancelledError() from None
        raise
    return result, cancellation_requested


@dataclass(frozen=True, slots=True)
class _PersistedTraceEvent:
    event: TraceEvent
    entry: LedgerEntry


def _entry_has_exact_envelope(
    ledger: tuple[LedgerEntry, ...],
    entry: LedgerEntry,
    *,
    expected_record_key: str,
) -> bool:
    return (
        type(entry) is LedgerEntry
        and entry.record_key == expected_record_key
        and 0 < entry.position <= len(ledger)
        and ledger[entry.position - 1] == entry
    )


def _event_matches_input(
    event: TraceEvent,
    trace_input: FixedStepTraceInput,
    *,
    expected_sequence: int,
) -> bool:
    draft = trace_input.draft
    return (
        event.event_id == trace_input.expected_event_id
        and event.run_id == draft.run_id
        and event.sequence == expected_sequence
        and event.source_event_id == draft.source_event_id
        and event.timestamp == draft.timestamp
        and event.event_type is draft.event_type
        and event.phase is draft.phase
        and event.parent_ids == draft.parent_ids
        and event.source_adapter == draft.source_adapter
        and event.trust_label is draft.trust_label
    )


async def _authoritative_trace_event(
    repository: RunRepository,
    trace_input: FixedStepTraceInput,
    *,
    expected_sequence: int,
) -> _PersistedTraceEvent | None:
    try:
        ledger = await repository.ledger(trace_input.draft.run_id)
    except RunNotFoundError:
        return None
    events = tuple((entry, entry.record) for entry in ledger if type(entry.record) is TraceEvent)
    by_identifier = tuple(
        item for item in events if item[1].event_id == trace_input.expected_event_id
    )
    by_sequence = tuple(item for item in events if item[1].sequence == expected_sequence)
    by_source = tuple(
        item for item in events if item[1].source_event_id == trace_input.draft.source_event_id
    )
    if not by_identifier:
        if by_sequence or by_source:
            raise FixedStepTraceInvariantError
        return None
    expected_sequences = tuple(range(1, expected_sequence + 1))
    if (
        len(by_identifier) != 1
        or by_sequence != by_identifier
        or by_source != by_identifier
        or tuple(event.sequence for _, event in events) != expected_sequences
        or events[-1] != by_identifier[0]
    ):
        raise FixedStepTraceInvariantError
    entry, event = by_identifier[0]
    if not _entry_has_exact_envelope(
        ledger,
        entry,
        expected_record_key=f"trace_event:{trace_input.expected_event_id}",
    ) or not _event_matches_input(
        event,
        trace_input,
        expected_sequence=expected_sequence,
    ):
        raise FixedStepTraceInvariantError
    return _PersistedTraceEvent(event=event, entry=entry)


async def _append_reconciled_trace_event(
    repository: RunRepository,
    trace_input: FixedStepTraceInput,
    *,
    expected_sequence: int,
) -> tuple[_PersistedTraceEvent, bool]:
    try:
        append, write_cancelled = await _complete_boundary(
            repository.append(
                trace_input.draft,
                event_id=trace_input.expected_event_id,
            )
        )
    except asyncio.CancelledError:
        persisted, _ = await _complete_boundary(
            _authoritative_trace_event(
                repository,
                trace_input,
                expected_sequence=expected_sequence,
            )
        )
        if persisted is None:
            raise
        return persisted, True
    except Exception:
        persisted, reconcile_cancelled = await _complete_boundary(
            _authoritative_trace_event(
                repository,
                trace_input,
                expected_sequence=expected_sequence,
            )
        )
        if persisted is None:
            if reconcile_cancelled:
                raise asyncio.CancelledError() from None
            raise FixedStepTraceInputError from None
        return persisted, reconcile_cancelled

    if (
        append.disposition is not AppendDisposition.APPENDED
        or append.event.event_id != trace_input.expected_event_id
        or append.event.sequence != expected_sequence
        or append.ingestion_cursor != expected_sequence
    ):
        raise FixedStepTraceInputError
    persisted, read_cancelled = await _complete_boundary(
        _authoritative_trace_event(
            repository,
            trace_input,
            expected_sequence=expected_sequence,
        )
    )
    if (
        persisted is None
        or persisted.event != append.event
        or persisted.entry.position != append.ledger_position
    ):
        raise FixedStepTraceInvariantError
    return persisted, write_cancelled or read_cancelled


async def _authoritative_invocation_decision(
    repository: RunRepository,
    decision: InvocationDecision,
) -> LedgerReceipt | None:
    ledger = await repository.ledger(decision.run_id)
    decisions = tuple(
        (entry, entry.record)
        for entry in ledger
        if type(entry.record) is InvocationDecision
        and entry.record.decision_id == decision.decision_id
    )
    event_decisions = tuple(
        (entry, entry.record)
        for entry in ledger
        if type(entry.record) is InvocationDecision
        and entry.record.event_sequence == decision.event_sequence
    )
    if not decisions:
        if event_decisions:
            raise FixedStepTraceInvariantError
        return None
    if len(decisions) != 1 or event_decisions != decisions or decisions[0][1] != decision:
        raise FixedStepTraceInvariantError
    entry = decisions[0][0]
    if not _entry_has_exact_envelope(
        ledger,
        entry,
        expected_record_key=f"invocation_decision:{decision.decision_id}",
    ):
        raise FixedStepTraceInvariantError
    return LedgerReceipt(
        appended=False,
        record_id=decision.decision_id,
        record_tag=entry.record_tag,
        ledger_position=entry.position,
        chain_tag=entry.chain_tag,
    )


async def record_reconciled_invocation_decision(
    repository: RunRepository,
    decision: InvocationDecision,
) -> tuple[LedgerReceipt, bool]:
    """Record one deterministic decision, reconciling a lost acknowledgement."""

    try:
        receipt, write_cancelled = await _complete_boundary(
            repository.record_invocation_decision(decision)
        )
    except asyncio.CancelledError:
        reconciled, _ = await _complete_boundary(
            _authoritative_invocation_decision(repository, decision)
        )
        if reconciled is None:
            raise
        return reconciled, True
    except Exception:
        reconciled, reconcile_cancelled = await _complete_boundary(
            _authoritative_invocation_decision(repository, decision)
        )
        if reconciled is None:
            if reconcile_cancelled:
                raise asyncio.CancelledError() from None
            raise FixedStepTraceInvariantError from None
        return reconciled, reconcile_cancelled

    if not receipt.appended:
        raise FixedStepTraceInvariantError
    reconciled, read_cancelled = await _complete_boundary(
        _authoritative_invocation_decision(repository, decision)
    )
    if reconciled is None or receipt != reconciled.model_copy(update={"appended": True}):
        raise FixedStepTraceInvariantError
    return receipt, write_cancelled or read_cancelled


class FixedStepTraceDriver:
    """Stream one validated trace through shared fixed-step projections.

    The callback owns algorithm policy and lifecycle transitions. The driver owns
    only the ordering that every condition must share: append, bind, resolve the
    prefix, project the schedule and optional message window, invoke the callback,
    then rebuild and attest the final repository projections.
    """

    __slots__ = ("_repository",)

    def __init__(self, repository: RunRepository) -> None:
        self._repository = repository

    async def run(
        self,
        inputs: tuple[FixedStepTraceInput, ...],
        callback: Callable[[FixedStepTraceBoundary], Awaitable[_BoundaryProjection]],
    ) -> FixedStepTraceResult[_BoundaryProjection]:
        if not inputs:
            raise FixedStepTraceInputError
        drafts = tuple(item.draft for item in inputs)
        normalized_draft_digests = tuple(canonical_digest(draft) for draft in drafts)
        trace_digest = algorithm_trace_digest(normalized_draft_digests)
        run_id = drafts[0].run_id
        try:
            await self._repository.ledger(run_id)
        except RunNotFoundError:
            pass
        except Exception:
            raise FixedStepTraceInvariantError from None
        else:
            raise FixedStepTraceInputError

        bindings: list[TrajectoryBinding] = []
        persisted_events: list[TraceEvent] = []
        persisted_event_draft_digests: list[str] = []
        windows: list[MessageWindow] = []
        boundary_projections: list[_BoundaryProjection] = []
        final_prefix: AttestedTrajectoryPrefix | None = None
        final_schedule: FixedStepSchedule | None = None

        for ordinal, item in enumerate(inputs, start=1):
            persisted, append_cancelled = await _append_reconciled_trace_event(
                self._repository,
                item,
                expected_sequence=ordinal,
            )
            event = persisted.event
            try:
                binding = bind_persisted_trajectory_event(
                    persisted.entry,
                    task_description=item.task_description,
                    logical_messages=item.logical_messages,
                    action_step=item.action_step,
                )
            except Exception:
                raise FixedStepTraceInvariantError from None
            bindings.append(binding)
            persisted_events.append(event)
            persisted_event_draft_digests.append(_persisted_draft_digest(event))
            prefix = await resolve_trajectory_prefix(
                self._repository,
                TrajectoryPrefixRequest(
                    schema_version="trajectory-prefix-request/v1",
                    run_id=run_id,
                    boundary_event_sequence=event.sequence,
                    bindings=tuple(bindings),
                ),
            )
            schedule = await project_fixed_step_schedule(self._repository, prefix)
            scheduled = schedule.decisions[-1]
            window = (
                await project_message_window(self._repository, prefix) if scheduled.invoke else None
            )
            if window is not None:
                windows.append(window)
            boundary = FixedStepTraceBoundary(
                ordinal=ordinal,
                trace_digest=trace_digest,
                trace_input=item,
                event=event,
                binding=binding,
                prefix=prefix,
                schedule=schedule,
                scheduled=scheduled,
                window=window,
            )
            if append_cancelled:
                completed_projection = await callback(boundary)
                boundary_projections.append(completed_projection)
                raise asyncio.CancelledError
            boundary_projections.append(await callback(boundary))
            final_prefix = prefix
            final_schedule = schedule

        if final_prefix is None or final_schedule is None:  # pragma: no cover
            raise FixedStepTraceInvariantError
        rebuild = await self._repository.rebuild(run_id)
        if not rebuild.equivalent:
            raise FixedStepTraceInvariantError
        ledger = await self._repository.ledger(run_id)
        ledger_head = await self._repository.ledger_head(run_id)
        if ledger_head.entry_count != len(ledger):
            raise FixedStepTraceInvariantError
        return FixedStepTraceResult(
            spine=FixedStepTraceSpine(
                run_id=run_id,
                trace_digest=trace_digest,
                normalized_draft_digests=normalized_draft_digests,
                persisted_events=tuple(persisted_events),
                persisted_event_draft_digests=tuple(persisted_event_draft_digests),
                bindings=tuple(bindings),
                trajectory_prefix=final_prefix,
                schedule=final_schedule,
                windows=tuple(windows),
            ),
            boundary_projections=tuple(boundary_projections),
            projection_digests=rebuild.after,
            ledger=ledger,
            ledger_head=ledger_head,
            rebuild_equivalent=rebuild.equivalent,
        )


__all__ = [
    "FixedStepTraceBoundary",
    "FixedStepTraceDriver",
    "FixedStepTraceInput",
    "FixedStepTraceInputError",
    "FixedStepTraceInvariantError",
    "FixedStepTraceResult",
    "FixedStepTraceSpine",
    "record_reconciled_invocation_decision",
]
