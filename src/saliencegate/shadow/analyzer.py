"""Atomic, resumable whole-trace Shadow analysis."""

from __future__ import annotations

import asyncio
import hmac
from dataclasses import dataclass, field
from typing import Literal, cast, final
from uuid import UUID

from saliencegate.domain import (
    NormalizedTraceEventDraft,
    Signal,
    TraceEvent,
    canonical_json,
)
from saliencegate.ports.repository import (
    MAX_CONDITIONAL_BATCH_EVENTS,
    MAX_CONDITIONAL_BATCH_OPERATIONS,
    MAX_CONDITIONAL_BATCH_REQUEST_BYTES,
    MAX_CONDITIONAL_BATCH_SIGNALS,
    AppendDisposition,
    AppendReceipt,
    ConditionalAppendOperation,
    ConditionalBatchReceipt,
    ConditionalEventAppend,
    ConditionalSignalAppend,
    LedgerEntry,
    LedgerHead,
    LedgerHeadConflictError,
    LedgerReceipt,
    RepositoryError,
)
from saliencegate.security.keys import InstallationKey
from saliencegate.security.redaction import RedactionPolicy
from saliencegate.shadow.atif import (
    ATIFProfile,
    ATIFShadowAdapter,
    ShadowEnvironmentBinding,
)
from saliencegate.shadow.errors import (
    ShadowConfigurationError,
    ShadowInputError,
    ShadowInvariantError,
    ShadowStateError,
    ShadowTraceInputError,
)
from saliencegate.shadow.inputs import (
    SHADOW_PROJECTION_MATRIX,
    ShadowInputKind,
    project_shadow_input,
)
from saliencegate.shadow.io import (
    PreflightedShadowRow,
    PreflightedShadowTrace,
    _normalized_input_digest,
    _parse_json_object,
    _preflight_record_values,
    _prepare_options,
)
from saliencegate.shadow.observation import (
    ShadowObservation,
    _admit_shadow_observation_sequence,
    _build_shadow_observation_trusted,
)
from saliencegate.shadow.report import (
    ShadowReportRow,
    ShadowRunReport,
    _build_shadow_run_report_trusted,
    _require_trusted_shadow_run_report,
    _TrustedShadowRunReport,
    build_shadow_run_report,
)
from saliencegate.shadow.session import (
    _MAX_CAS_ATTEMPTS,
    ShadowSession,
    _RetryableSnapshotRaceError,
    _RunState,
)
from saliencegate.shadow.trace import ShadowTrace
from saliencegate.shadow.trace_report import (
    ShadowTraceReport,
    _build_shadow_trace_report_trusted,
    _require_profile_diagnostic_links,
)
from saliencegate.signals import ExtractionReport
from saliencegate.signals.base import (
    _admit_detection_sequence,
    _extract_trusted_report,
    _longest_trusted_detection_context,
)


@dataclass(frozen=True, slots=True, repr=False)
class _PreparedSignal:
    signal: Signal = field(repr=False)
    detection_cutoff: int


@dataclass(frozen=True, slots=True, repr=False)
class _PreparedEvent:
    row: PreflightedShadowRow = field(repr=False)
    draft: NormalizedTraceEventDraft = field(repr=False)
    event: TraceEvent = field(repr=False)
    extraction_report: ExtractionReport = field(repr=False)
    observation: ShadowObservation = field(repr=False)
    operation: ConditionalEventAppend = field(repr=False)


@dataclass(frozen=True, slots=True, repr=False)
class _PreparedAnalysis:
    trace: ShadowTrace = field(repr=False)
    rows: tuple[PreflightedShadowRow, ...] = field(repr=False)
    events: tuple[_PreparedEvent, ...] = field(repr=False)
    signals: tuple[_PreparedSignal, ...] = field(repr=False)
    normalized_input_digest: str
    full_operations: tuple[ConditionalAppendOperation, ...] = field(repr=False)

    @property
    def expected_events(self) -> tuple[TraceEvent, ...]:
        return tuple(item.event for item in self.events)


@dataclass(frozen=True, slots=True, repr=False)
class _PreparedTracePreview:
    initial_state: _RunState | None = field(repr=False)
    shadow_report: ShadowRunReport = field(repr=False)


_UNSPECIFIED_INITIAL_STATE = object()


def _bindings_match(session: ShadowSession, trace: ShadowTrace) -> bool:
    binding = session._trace_binding
    try:
        return binding is not None and hmac.compare_digest(
            canonical_json(binding),
            canonical_json(trace.binding),
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return False


def _prepare_analysis(session: ShadowSession, value: object) -> _PreparedAnalysis:
    if type(value) is not ShadowTrace:
        raise ShadowInputError()
    trace = value._copy_exact()
    if trace.binding.identity_mode != "profile_content_addressed":
        raise ShadowInputError()
    if trace.run_id != session._run_id or not _bindings_match(session, trace):
        raise ShadowConfigurationError()
    _require_profile_diagnostic_links(trace.binding, trace.diagnostics)

    options = _prepare_options(
        run_id=session._run_id,
        config=session._config,
        installation_key=session._installation_key,
        redaction_policy=session._redaction_policy,
        redaction_policy_tag=session._redaction_policy_tag,
        capture_scope=session._options.capture_scope,
        task_scope_digest=session._task_scope_digest,
        lineage_scope_digest=session._lineage_scope_digest,
        capture_manifest_digest=session._manifest_digest,
        source_adapter=session._source_adapter,
    )
    wire_records = tuple(_parse_json_object(item) for item in trace._wire_record_bytes())
    rows = _preflight_record_values(wire_records, options)
    normalized_digest = _normalized_input_digest(
        options,
        tuple(row.input_record for row in rows),
    )

    unique_rows: list[PreflightedShadowRow] = []
    drafts: list[NormalizedTraceEventDraft] = []
    events: list[TraceEvent] = []
    start_event: TraceEvent | None = None
    for row in rows:
        if row.is_retry:
            continue
        kind = row.input_kind
        start_payload = session._start_payload() if kind is ShadowInputKind.START else None
        finish_payload = (
            session._finish_payload(start_event)
            if kind is ShadowInputKind.FINISH and start_event is not None
            else None
        )
        draft = project_shadow_input(
            row.input_record,
            run_id=session._run_id,
            source_adapter=session._source_adapter,
            start_payload=start_payload,
            finish_payload=finish_payload,
        )
        event = session._preflight_event(
            draft,
            kind=kind,
            event_id=row.event_ref.event_id,
            sequence=row.event_sequence,
        )
        if event.run_id != row.event_ref.run_id or event.sequence != len(events) + 1:
            raise ShadowInvariantError()
        if kind is ShadowInputKind.START:
            start_event = event
        unique_rows.append(row)
        drafts.append(draft)
        events.append(event)

    detection_sequence = _admit_detection_sequence(tuple(events))
    event_tuple = detection_sequence.events
    observation_admission = _admit_shadow_observation_sequence(
        detection_sequence,
        config=session._config,
        redaction_policy_tag=session._redaction_policy_tag,
    )
    prepared_events: list[_PreparedEvent] = []
    prepared_signals: list[_PreparedSignal] = []
    signals_by_id: dict[object, _PreparedSignal] = {}
    prefix_ids: set[object] = set()
    for index, (row, draft, event) in enumerate(zip(unique_rows, drafts, event_tuple, strict=True)):
        if event.event_id != row.event_ref.event_id or event.sequence != row.event_sequence:
            raise ShadowInvariantError()
        prefix_ids.add(event.event_id)
        trusted_context = _longest_trusted_detection_context(
            detection_sequence,
            index + 1,
        )
        extraction = _extract_trusted_report(session._extractor, trusted_context)
        report = extraction.report
        observation = _build_shadow_observation_trusted(
            observation_admission,
            extraction,
            input_kind=row.input_kind,
            source_event_digest=row.source_event_digest,
            cli_input_ordinal=row.input_ordinal,
        )
        prepared_events.append(
            _PreparedEvent(
                row=row,
                draft=draft,
                event=event,
                extraction_report=report,
                observation=observation,
                operation=ConditionalEventAppend(event=draft, event_id=event.event_id),
            )
        )
        for signal in report.signals:
            if (
                signal.run_id != session._run_id
                or not signal.evidence_event_ids
                or any(item not in prefix_ids for item in signal.evidence_event_ids)
            ):
                raise ShadowInvariantError()
            prepared = _PreparedSignal(
                signal=signal,
                detection_cutoff=event.sequence,
            )
            existing = signals_by_id.get(signal.signal_id)
            if existing is None:
                signals_by_id[signal.signal_id] = prepared
                prepared_signals.append(prepared)
            elif existing.signal != signal:
                raise ShadowInvariantError()

    if start_event is None or len(prepared_events) > MAX_CONDITIONAL_BATCH_EVENTS:
        raise ShadowInvariantError()
    if len(prepared_signals) > MAX_CONDITIONAL_BATCH_SIGNALS:
        raise ShadowInvariantError()

    signals_at_cutoff: dict[int, list[Signal]] = {}
    for item in prepared_signals:
        signals_at_cutoff.setdefault(item.detection_cutoff, []).append(item.signal)
    full_operations: list[ConditionalAppendOperation] = []
    for prepared_event in prepared_events:
        full_operations.append(prepared_event.operation)
        full_operations.extend(
            ConditionalSignalAppend(signal=signal)
            for signal in signals_at_cutoff.get(prepared_event.event.sequence, ())
        )
    if (
        not full_operations
        or len(full_operations) > MAX_CONDITIONAL_BATCH_OPERATIONS
        or len(canonical_json(tuple(full_operations))) > MAX_CONDITIONAL_BATCH_REQUEST_BYTES
    ):
        raise ShadowInvariantError()
    return _PreparedAnalysis(
        trace=trace,
        rows=rows,
        events=tuple(prepared_events),
        signals=tuple(prepared_signals),
        normalized_input_digest=normalized_digest,
        full_operations=tuple(full_operations),
    )


def _validate_state(state: _RunState | None, prepared: _PreparedAnalysis) -> None:
    if state is None:
        return
    expected_events = prepared.expected_events
    if (
        len(state.events) > len(expected_events)
        or state.events != expected_events[: len(state.events)]
    ):
        raise ShadowStateError()

    event_sequences = {event.event_id: event.sequence for event in state.events}
    expected_signals = {item.signal.signal_id: item for item in prepared.signals}
    if len(state.signals) > len(expected_signals):
        raise ShadowStateError()
    for signal in state.signals:
        expected = expected_signals.get(signal.signal_id)
        evidence_sequences = tuple(event_sequences.get(item) for item in signal.evidence_event_ids)
        if (
            expected is None
            or expected.signal != signal
            or expected.detection_cutoff > len(state.events)
            or not evidence_sequences
            or any(item is None for item in evidence_sequences)
        ):
            raise ShadowStateError()
        last_evidence_sequence = max(item for item in evidence_sequences if item is not None)
        evidence_event = state.events[last_evidence_sequence - 1]
        if state.signal_positions.get(signal.signal_id, 0) <= state.event_positions.get(
            evidence_event.event_id,
            0,
        ):
            raise ShadowStateError()


def _missing_operations(
    state: _RunState | None,
    prepared: _PreparedAnalysis,
) -> tuple[ConditionalAppendOperation, ...]:
    prefix_count = 0 if state is None else len(state.events)
    existing_signal_ids = set() if state is None else {item.signal_id for item in state.signals}
    signals_by_cutoff: dict[int, list[Signal]] = {}
    for item in prepared.signals:
        if item.signal.signal_id not in existing_signal_ids:
            signals_by_cutoff.setdefault(item.detection_cutoff, []).append(item.signal)

    operations: list[ConditionalAppendOperation] = []
    for cutoff in range(1, prefix_count + 1):
        operations.extend(
            ConditionalSignalAppend(signal=signal) for signal in signals_by_cutoff.get(cutoff, ())
        )
    for event in prepared.events[prefix_count:]:
        operations.append(event.operation)
        operations.extend(
            ConditionalSignalAppend(signal=signal)
            for signal in signals_by_cutoff.get(event.event.sequence, ())
        )
    return tuple(operations)


async def _load_trace_state(session: ShadowSession) -> _RunState | None:
    for _attempt in range(_MAX_CAS_ATTEMPTS):
        try:
            return await session._load_state()
        except _RetryableSnapshotRaceError:
            continue
    raise ShadowStateError()


def _is_strict_exact_extension(
    previous: _RunState | None,
    current: _RunState | None,
) -> bool:
    if current is None:
        return False
    if previous is None:
        return bool(current.entries)
    return (
        len(current.entries) > len(previous.entries)
        and current.entries[: len(previous.entries)] == previous.entries
    )


def _validate_batch_receipt(
    operations: tuple[ConditionalAppendOperation, ...],
    prepared: _PreparedAnalysis,
    *,
    initial_head: LedgerHead | None,
    receipt: ConditionalBatchReceipt,
) -> ConditionalBatchReceipt:
    copied: ConditionalBatchReceipt | None = None
    try:
        if type(receipt) is ConditionalBatchReceipt:
            encoded = ConditionalBatchReceipt.__pydantic_serializer__.to_json(
                receipt,
                warnings=False,
            )
            candidate = ConditionalBatchReceipt.model_validate_json(encoded)
            if candidate == receipt:
                copied = candidate
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        copied = None
    if copied is None or copied.initial_head != initial_head:
        raise ShadowStateError()
    receipt = copied
    initial_count = 0 if initial_head is None else initial_head.entry_count
    if len(receipt.receipts) != len(
        operations
    ) or receipt.final_head.entry_count != initial_count + len(operations):
        raise ShadowStateError()
    events_by_id = {item.event.event_id: item.event for item in prepared.events}
    for offset, (operation, operation_receipt) in enumerate(
        zip(operations, receipt.receipts, strict=True),
        start=1,
    ):
        expected_position = initial_count + offset
        if type(operation) is ConditionalEventAppend:
            expected_event = events_by_id.get(operation.event_id)
            if (
                type(operation_receipt) is not AppendReceipt
                or operation_receipt.disposition is not AppendDisposition.APPENDED
                or operation_receipt.ledger_position != expected_position
                or operation_receipt.event != expected_event
            ):
                raise ShadowStateError()
        elif (
            type(operation) is not ConditionalSignalAppend
            or type(operation_receipt) is not LedgerReceipt
            or not operation_receipt.appended
            or operation_receipt.record_id != operation.signal.signal_id
            or operation_receipt.ledger_position != expected_position
        ):
            raise ShadowStateError()
    return receipt


def _validate_receipt_ledger_links(
    receipt: ConditionalBatchReceipt,
    entries: tuple[LedgerEntry, ...],
) -> None:
    if len(receipt.receipts) != len(entries):
        raise ShadowStateError()
    for operation_receipt, entry in zip(receipt.receipts, entries, strict=True):
        try:
            if operation_receipt.ledger_position != entry.position:
                raise ValueError("receipt position does not match the ledger")
            if type(operation_receipt) is AppendReceipt:
                if (
                    operation_receipt.ingestion_cursor != operation_receipt.event.sequence
                    or operation_receipt.event != entry.record
                ):
                    raise ValueError("event receipt does not match the ledger")
            elif type(operation_receipt) is LedgerReceipt:
                if (
                    operation_receipt.record_tag != entry.record_tag
                    or operation_receipt.chain_tag != entry.chain_tag
                ):
                    raise ValueError("record receipt does not match the ledger")
            else:
                raise ValueError("unsupported receipt type")
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise ShadowStateError() from None


def _operation_records(
    operations: tuple[ConditionalAppendOperation, ...],
    prepared: _PreparedAnalysis,
) -> tuple[TraceEvent | Signal, ...]:
    events_by_id = {item.event.event_id: item.event for item in prepared.events}
    records: list[TraceEvent | Signal] = []
    for operation in operations:
        if type(operation) is ConditionalEventAppend:
            event = events_by_id.get(operation.event_id)
            if event is None:
                raise ShadowInvariantError()
            records.append(event)
        elif type(operation) is ConditionalSignalAppend:
            records.append(operation.signal)
        else:  # pragma: no cover - the discriminated operation constructors guard this
            raise ShadowInvariantError()
    return tuple(records)


def _report_row(
    row: PreflightedShadowRow,
    *,
    observation_digest: str,
    persistence_disposition: Literal["appended", "preexisting"],
) -> ShadowReportRow:
    projection = SHADOW_PROJECTION_MATRIX[row.input_kind]
    return ShadowReportRow(
        input_ordinal=row.input_ordinal,
        source_event_digest=row.source_event_digest,
        first_occurrence_ordinal=None if row.is_retry else row.input_ordinal,
        retry_target_ordinal=row.retry_target_ordinal,
        event_type=projection.event_type,
        phase=projection.phase,
        input_kind=row.input_kind,
        persistence_disposition=persistence_disposition,
        observation_digest=observation_digest,
    )


def _legacy_prefix_matches(
    events: tuple[object, ...],
    unique_rows: tuple[PreflightedShadowRow, ...],
) -> bool:
    if len(events) > len(unique_rows):
        return False
    return all(
        getattr(event, "event_id", None) == row.event_ref.event_id
        and getattr(event, "sequence", None) == row.event_sequence
        for event, row in zip(events, unique_rows, strict=False)
    )


async def _analyze_legacy_preflighted(
    session: ShadowSession,
    trace: PreflightedShadowTrace,
) -> ShadowRunReport:
    """Preserve the 10,000-row incremental NDJSON path behind the shared report core."""

    if type(session) is not ShadowSession or type(trace) is not PreflightedShadowTrace:
        raise ShadowInputError()
    initial_head, initial_events = await session._snapshot_for_cli()
    unique_rows = tuple(row for row in trace.rows if not row.is_retry)
    if not _legacy_prefix_matches(initial_events, unique_rows):
        raise ShadowStateError()
    initial_event_count = len(initial_events)
    observations_by_ordinal: dict[int, ShadowObservation] = {}
    report_rows: list[ShadowReportRow] = []
    observations: list[ShadowObservation] = []
    for row in trace.rows:
        if row.is_retry:
            observation = observations_by_ordinal.get(row.retry_target_ordinal or 0)
            if observation is None:
                raise ShadowInvariantError()
            report_rows.append(
                _report_row(
                    row,
                    observation_digest=observation.observation_digest,
                    persistence_disposition="preexisting",
                )
            )
            continue
        result = await session._submit(
            row.input_record,
            cli_input_ordinal=row.input_ordinal,
        )
        if result.ref != row.event_ref:
            raise ShadowStateError()
        observation = result.observation
        observations_by_ordinal[row.input_ordinal] = observation
        observations.append(observation)
        report_rows.append(
            _report_row(
                row,
                observation_digest=observation.observation_digest,
                persistence_disposition=(
                    "preexisting" if row.event_sequence <= initial_event_count else "appended"
                ),
            )
        )
    _final_head, final_events = await session._snapshot_for_cli()
    if not _legacy_prefix_matches(final_events, unique_rows) or len(final_events) != len(
        unique_rows
    ):
        raise ShadowStateError()
    return build_shadow_run_report(
        run_id=trace.run_id,
        initial_ledger_entry_count=0 if initial_head is None else initial_head.entry_count,
        initial_ledger_chain_tag=None if initial_head is None else initial_head.chain_tag,
        initial_ledger_projection_tag=(
            None if initial_head is None else initial_head.projection_tag
        ),
        initial_ledger_head_tag=None if initial_head is None else initial_head.head_tag,
        input_byte_digest=trace.input_byte_digest,
        normalized_input_digest=trace.normalized_input_digest,
        redaction_policy_tag=session._redaction_policy_tag,
        detector_profile_digest=session._config.detector_profile_digest,
        capture_scope=session._options.capture_scope,
        task_scope_digest=session._task_scope_digest,
        lineage_scope_digest=session._lineage_scope_digest,
        capture_manifest_digest=session._manifest_digest,
        rows=tuple(report_rows),
        observations=tuple(observations),
    )


def _build_run_report(
    session: ShadowSession,
    prepared: _PreparedAnalysis,
    *,
    initial_head: LedgerHead | None,
    initial_event_count: int,
) -> _TrustedShadowRunReport:
    observations_by_ordinal = {item.row.input_ordinal: item.observation for item in prepared.events}
    rows: list[ShadowReportRow] = []
    for row in prepared.rows:
        target_ordinal = row.retry_target_ordinal if row.is_retry else row.input_ordinal
        observation = observations_by_ordinal.get(target_ordinal or 0)
        if observation is None:
            raise ShadowInvariantError()
        rows.append(
            _report_row(
                row,
                observation_digest=observation.observation_digest,
                persistence_disposition=(
                    "preexisting"
                    if row.is_retry or row.event_sequence <= initial_event_count
                    else "appended"
                ),
            )
        )
    return _build_shadow_run_report_trusted(
        run_id=prepared.trace.run_id,
        initial_ledger_entry_count=0 if initial_head is None else initial_head.entry_count,
        initial_ledger_chain_tag=None if initial_head is None else initial_head.chain_tag,
        initial_ledger_projection_tag=None if initial_head is None else initial_head.projection_tag,
        initial_ledger_head_tag=None if initial_head is None else initial_head.head_tag,
        input_byte_digest=prepared.trace.binding.source_byte_digest,
        normalized_input_digest=prepared.normalized_input_digest,
        redaction_policy_tag=session._redaction_policy_tag,
        detector_profile_digest=session._config.detector_profile_digest,
        capture_scope=session._options.capture_scope,
        task_scope_digest=session._task_scope_digest,
        lineage_scope_digest=session._lineage_scope_digest,
        capture_manifest_digest=session._manifest_digest,
        rows=tuple(rows),
        observations=tuple(item.observation for item in prepared.events),
    )


async def _preview_prepared(
    session: ShadowSession,
    prepared: _PreparedAnalysis,
    *,
    assume_empty: bool,
) -> _PreparedTracePreview:
    """Predict the exact report without appending any ledger record."""

    if (
        type(session) is not ShadowSession
        or type(prepared) is not _PreparedAnalysis
        or type(assume_empty) is not bool
    ):
        raise ShadowInputError()
    async with session._lock:
        if session._closed or session._trace_binding is None:
            raise ShadowInputError()
        state = None if assume_empty else await _load_trace_state(session)
        _validate_state(state, prepared)
        nested = _build_run_report(
            session,
            prepared,
            initial_head=None if state is None else state.head,
            initial_event_count=0 if state is None else len(state.events),
        )
        return _PreparedTracePreview(
            initial_state=state,
            shadow_report=_require_trusted_shadow_run_report(nested),
        )


async def _analyze_prepared(
    session: ShadowSession,
    prepared: _PreparedAnalysis,
    *,
    expected_initial_state: _RunState | object | None = _UNSPECIFIED_INITIAL_STATE,
) -> ShadowTraceReport:
    async with session._lock:
        if session._closed or session._trace_binding is None:
            raise ShadowInputError()
        state = await _load_trace_state(session)
        if expected_initial_state is not _UNSPECIFIED_INITIAL_STATE:
            expected = cast(_RunState | None, expected_initial_state)
            if state != expected:
                raise ShadowStateError()
        report_initial_head = None if state is None else state.head
        report_initial_event_count = 0 if state is None else len(state.events)
        for attempt in range(_MAX_CAS_ATTEMPTS):
            _validate_state(state, prepared)
            operations = _missing_operations(state, prepared)
            attempt_head = None if state is None else state.head
            if not operations:
                final_state = state
                break
            try:
                receipt = await session._append_trace_batch_locked(
                    operations,
                    expected_head=attempt_head,
                )
            except LedgerHeadConflictError:
                advanced = await _load_trace_state(session)
                if advanced is None or not _is_strict_exact_extension(state, advanced):
                    raise ShadowStateError() from None
                _validate_state(advanced, prepared)
                state = advanced
                if not _missing_operations(state, prepared):
                    final_state = state
                    break
                if attempt + 1 == _MAX_CAS_ATTEMPTS:
                    raise ShadowStateError() from None
                continue
            receipt = _validate_batch_receipt(
                operations,
                prepared,
                initial_head=attempt_head,
                receipt=receipt,
            )
            final_state = await _load_trace_state(session)
            if final_state is None or final_state.head != receipt.final_head:
                raise ShadowStateError()
            initial_entry_count = 0 if attempt_head is None else attempt_head.entry_count
            suffix_entries = final_state.entries[initial_entry_count:]
            if tuple(entry.record for entry in suffix_entries) != _operation_records(
                operations,
                prepared,
            ):
                raise ShadowStateError()
            _validate_receipt_ledger_links(receipt, suffix_entries)
            break
        else:  # pragma: no cover - loop exits or raises on the final attempt
            raise ShadowStateError()

        if final_state is None:
            raise ShadowStateError()
        _validate_state(final_state, prepared)
        if final_state.events != prepared.expected_events or {
            signal.signal_id: signal for signal in final_state.signals
        } != {item.signal.signal_id: item.signal for item in prepared.signals}:
            raise ShadowStateError()
        nested = _build_run_report(
            session,
            prepared,
            initial_head=report_initial_head,
            initial_event_count=report_initial_event_count,
        )
        assert session._trace_binding is not None
        return _build_shadow_trace_report_trusted(
            trace=prepared.trace,
            shadow_report=nested,
            session_binding=session._trace_binding,
            authenticated_start_source_adapter=final_state.start.source_adapter,
        )


@final
class ShadowAnalyzer:
    """Analyze one immutable trace without owning or closing its session."""

    __slots__ = ("_lock", "_session")

    def __init__(self, session: ShadowSession) -> None:
        if type(session) is not ShadowSession:
            raise ShadowConfigurationError()
        self._session = session
        self._lock = asyncio.Lock()

    def __repr__(self) -> str:
        return "ShadowAnalyzer(<bound>)"

    async def analyze(self, trace: ShadowTrace) -> ShadowTraceReport:
        """Preflight completely, then atomically persist and report one trace."""

        result: ShadowTraceReport | None = None
        failure: Exception | None = None
        try:
            async with self._lock:
                prepared = _prepare_analysis(self._session, trace)
                result = await _analyze_prepared(self._session, prepared)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            failure = error
        if result is not None:
            return result
        if isinstance(failure, ShadowConfigurationError):
            raise ShadowConfigurationError()
        if isinstance(failure, ShadowInputError | ShadowTraceInputError):
            raise ShadowInputError()
        if isinstance(
            failure,
            ShadowStateError
            | LedgerHeadConflictError
            | RepositoryError
            | _RetryableSnapshotRaceError,
        ):
            raise ShadowStateError()
        if isinstance(failure, ShadowInvariantError):
            raise ShadowInvariantError()
        raise ShadowInvariantError()


async def analyze_atif_bytes(
    source_bytes: bytes,
    *,
    run_id: UUID,
    profile: ATIFProfile,
    environment: ShadowEnvironmentBinding,
    installation_key: InstallationKey | None = None,
    redaction_policy: RedactionPolicy | None = None,
    task_scope_digest: str | None = None,
    lineage_scope_digest: str | None = None,
    capture_manifest_digest: str | None = None,
) -> ShadowTraceReport:
    """Adapt and analyze one ATIF source in an owned in-memory session."""

    if type(profile) is not ATIFProfile or type(environment) is not ShadowEnvironmentBinding:
        raise ShadowConfigurationError()
    adapter = ATIFShadowAdapter(
        profile=profile,
        environment=environment,
    )
    trace = adapter.adapt_bytes(
        source_bytes,
        run_id=run_id,
        task_scope_digest=task_scope_digest,
        lineage_scope_digest=lineage_scope_digest,
        capture_manifest_digest=capture_manifest_digest,
    )
    async with ShadowSession.in_memory_for_trace(
        run_id=run_id,
        trace_binding=trace.binding,
        installation_key=installation_key,
        redaction_policy=redaction_policy,
    ) as session:
        return await ShadowAnalyzer(session).analyze(trace)


__all__ = ["ShadowAnalyzer", "analyze_atif_bytes"]
