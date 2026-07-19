from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from saliencegate.domain import (
    EventPhase,
    EventType,
    MemoryRecord,
    ReasonCode,
    Signal,
    SignalType,
    TraceEvent,
    TrustLabel,
    ValidityState,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.domain.errors import CanonicalJSONError
from saliencegate.domain.records import UUID4
from saliencegate.runtime.token_counting import DeterministicTokenCounter, TextSize

_BATCH_SCHEMA_VERSION: Literal["1.0"] = "1.0"
_BATCHER_VERSION: Literal["deterministic-batcher-v1"] = "deterministic-batcher-v1"
_MAX_SEQUENCE = (1 << 63) - 1


class BatchInputError(ValueError):
    """A sanitized batching-boundary validation failure."""

    def __init__(self) -> None:
        super().__init__("memory-cycle batch input failed validation")


class BatchIntegrityError(ValueError):
    """A built manifest no longer matches its canonical attestation."""

    def __init__(self) -> None:
        super().__init__("memory-cycle batch integrity verification failed")


class BatchStatus(StrEnum):
    READY = "ready"
    MANDATORY_INPUT_OVERFLOW = "mandatory_input_overflow"


class BatchMemoryRole(StrEnum):
    TASK_REQUIREMENT = "task_requirement"
    UNRESOLVED_STATE = "unresolved_state"


class BatchPriorityKind(StrEnum):
    CONTROLLER_ERROR = "controller_error"
    ACTION_PROPOSAL = "action_proposal"
    TOOL_ERROR = "tool_error"
    TEST_FAILURE = "test_failure"
    CONFLICT = "conflict"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
        validate_default=True,
    )


class BatchConfig(_FrozenModel):
    """Fully resolved, artifact-safe limits for one deterministic batch."""

    max_utf8_bytes: Annotated[int, Field(ge=1, le=10_000_000)]
    max_approximate_tokens: Annotated[int, Field(ge=1, le=2_500_000)]
    recent_event_count: Annotated[int, Field(ge=0, le=10_000)]
    max_controller_errors: Annotated[int, Field(ge=0, le=10_000)]
    max_action_proposals: Annotated[int, Field(ge=0, le=10_000)]
    max_tool_errors: Annotated[int, Field(ge=0, le=10_000)]
    max_test_failures: Annotated[int, Field(ge=0, le=10_000)]
    max_conflicts: Annotated[int, Field(ge=0, le=10_000)]
    batcher_version: Literal["deterministic-batcher-v1"] = _BATCHER_VERSION


class BatchRequest(_FrozenModel):
    """Internal post-repository input, never an adapter-ingress type.

    The runtime must populate this model only with records returned by a verified
    ``RunRepository`` ledger and memory snapshot. Its strict validation prevents
    structural confusion; it does not turn caller-created records into trusted data.
    """

    run_id: UUID4
    memory_cursor: Annotated[int, Field(ge=0, lt=_MAX_SEQUENCE)]
    events: Annotated[tuple[TraceEvent, ...], Field(min_length=1)]
    historical_event_ids: tuple[UUID4, ...] = ()
    signals: tuple[Signal, ...] = ()
    task_requirements: tuple[MemoryRecord, ...] = ()
    unresolved_state: tuple[MemoryRecord, ...] = ()

    @field_validator("events")
    @classmethod
    def order_events(cls, value: tuple[TraceEvent, ...]) -> tuple[TraceEvent, ...]:
        return tuple(sorted(value, key=lambda event: event.sequence))

    @field_validator("historical_event_ids")
    @classmethod
    def order_historical_event_ids(cls, value: tuple[UUID, ...]) -> tuple[UUID, ...]:
        return tuple(sorted(value, key=str))

    @field_validator("signals")
    @classmethod
    def order_signals(cls, value: tuple[Signal, ...]) -> tuple[Signal, ...]:
        return tuple(sorted(value, key=lambda signal: (signal.created_at, str(signal.signal_id))))

    @field_validator("task_requirements", "unresolved_state")
    @classmethod
    def order_memories(cls, value: tuple[MemoryRecord, ...]) -> tuple[MemoryRecord, ...]:
        return tuple(
            sorted(
                value,
                key=lambda memory: (memory.kind.value, str(memory.memory_id), memory.revision),
            )
        )

    @model_validator(mode="after")
    def records_form_one_unprocessed_slice(self) -> Self:
        if any(event.run_id != self.run_id for event in self.events):
            raise ValueError("events must belong to the requested run")
        sequences = tuple(event.sequence for event in self.events)
        if sequences[-1] > _MAX_SEQUENCE:
            raise ValueError("event sequence exceeds the signed 64-bit ledger range")
        expected = tuple(range(self.memory_cursor + 1, self.memory_cursor + len(self.events) + 1))
        if sequences != expected:
            raise ValueError("events must be one contiguous unprocessed sequence")
        event_ids = tuple(event.event_id for event in self.events)
        source_ids = tuple(event.source_event_id for event in self.events)
        if len(set(event_ids)) != len(event_ids) or len(set(source_ids)) != len(source_ids):
            raise ValueError("event identities must be unique")

        current_event_ids = set(event_ids)
        historical_event_ids = set(self.historical_event_ids)
        if len(historical_event_ids) != len(self.historical_event_ids):
            raise ValueError("historical event identities must be unique")
        if current_event_ids.intersection(historical_event_ids):
            raise ValueError("historical event identities must be disjoint from the run slice")
        known_event_ids = current_event_ids | historical_event_ids
        signal_ids = tuple(signal.signal_id for signal in self.signals)
        if len(set(signal_ids)) != len(signal_ids):
            raise ValueError("signal identities must be unique")
        if any(
            signal.run_id != self.run_id
            or any(event_id not in known_event_ids for event_id in signal.evidence_event_ids)
            or current_event_ids.isdisjoint(signal.evidence_event_ids)
            for signal in self.signals
        ):
            raise ValueError(
                "signals must cite known evidence and intersect the requested run slice"
            )

        memories = self.task_requirements + self.unresolved_state
        memory_ids = tuple(memory.memory_id for memory in memories)
        if len(set(memory_ids)) != len(memory_ids):
            raise ValueError("mandatory memories must be unique")
        if any(
            memory.run_id != self.run_id or memory.validity is not ValidityState.ACTIVE
            for memory in memories
        ):
            raise ValueError("mandatory memories must be active records from the requested run")
        return self


class SequenceRange(_FrozenModel):
    first_sequence: Annotated[int, Field(ge=1)]
    last_sequence: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def range_is_ordered(self) -> Self:
        if self.last_sequence < self.first_sequence:
            raise ValueError("sequence range is reversed")
        return self


class BatchMemory(_FrozenModel):
    role: BatchMemoryRole
    record: MemoryRecord


class VerbatimEvent(_FrozenModel):
    event: TraceEvent
    priority_kinds: tuple[BatchPriorityKind, ...] = ()

    @field_validator("priority_kinds")
    @classmethod
    def canonicalize_priority_kinds(
        cls,
        value: tuple[BatchPriorityKind, ...],
    ) -> tuple[BatchPriorityKind, ...]:
        ordered = tuple(sorted(set(value), key=lambda kind: kind.value))
        if len(ordered) != len(value):
            raise ValueError("priority kinds must be unique")
        return ordered


class EventAggregate(_FrozenModel):
    event_type: EventType
    phase: EventPhase
    trust_label: TrustLabel
    structural_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    priority_kinds: tuple[BatchPriorityKind, ...] = ()
    count: Annotated[int, Field(ge=1)]
    first_sequence: Annotated[int, Field(ge=1)]
    last_sequence: Annotated[int, Field(ge=1)]
    provenance_ranges: Annotated[tuple[SequenceRange, ...], Field(min_length=1)]
    source_event_ids: Annotated[tuple[UUID4, ...], Field(min_length=1)]
    source_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @field_validator("priority_kinds")
    @classmethod
    def canonicalize_priority_kinds(
        cls,
        value: tuple[BatchPriorityKind, ...],
    ) -> tuple[BatchPriorityKind, ...]:
        ordered = tuple(sorted(set(value), key=lambda kind: kind.value))
        if len(ordered) != len(value):
            raise ValueError("priority kinds must be unique")
        return ordered

    @model_validator(mode="after")
    def provenance_matches_count(self) -> Self:
        previous_last = 0
        for item in self.provenance_ranges:
            if item.first_sequence <= previous_last:
                raise ValueError("aggregate provenance ranges must be ordered and disjoint")
            previous_last = item.last_sequence
        represented = sum(
            item.last_sequence - item.first_sequence + 1 for item in self.provenance_ranges
        )
        if represented != self.count or len(self.source_event_ids) != self.count:
            raise ValueError("aggregate provenance does not match its count")
        if (
            self.first_sequence != self.provenance_ranges[0].first_sequence
            or self.last_sequence != self.provenance_ranges[-1].last_sequence
        ):
            raise ValueError("aggregate bounds do not match its provenance")
        if len(set(self.source_event_ids)) != len(self.source_event_ids):
            raise ValueError("aggregate source event IDs must be unique")
        return self


class BatchPayload(_FrozenModel):
    schema_version: Literal["1.0"] = _BATCH_SCHEMA_VERSION
    batcher_version: Literal["deterministic-batcher-v1"] = _BATCHER_VERSION
    run_id: UUID4
    memory_cursor: Annotated[int, Field(ge=0)]
    first_event_sequence: Annotated[int, Field(ge=1)]
    last_event_sequence: Annotated[int, Field(ge=1)]
    configuration_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    mandatory_memories: tuple[BatchMemory, ...]
    verbatim_events: tuple[VerbatimEvent, ...]
    aggregates: tuple[EventAggregate, ...]
    represented_event_count: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def event_partition_is_exact(self) -> Self:
        verbatim_sequences = [item.event.sequence for item in self.verbatim_events]
        aggregate_sequences = [
            sequence
            for aggregate in self.aggregates
            for provenance in aggregate.provenance_ranges
            for sequence in range(provenance.first_sequence, provenance.last_sequence + 1)
        ]
        represented = verbatim_sequences + aggregate_sequences
        expected = list(range(self.first_event_sequence, self.last_event_sequence + 1))
        if self.first_event_sequence != self.memory_cursor + 1:
            raise ValueError("batch does not continue the memory cursor")
        if sorted(represented) != expected or len(represented) != len(set(represented)):
            raise ValueError("batch events are not represented exactly once")
        if self.represented_event_count != len(expected):
            raise ValueError("represented event count is inconsistent")
        event_ids = [item.event.event_id for item in self.verbatim_events] + [
            event_id for aggregate in self.aggregates for event_id in aggregate.source_event_ids
        ]
        if len(event_ids) != len(expected) or len(event_ids) != len(set(event_ids)):
            raise ValueError("batch event identities are not represented exactly once")
        if any(item.event.run_id != self.run_id for item in self.verbatim_events):
            raise ValueError("verbatim event belongs to another run")
        memory_ids = [item.record.memory_id for item in self.mandatory_memories]
        if len(memory_ids) != len(set(memory_ids)) or any(
            item.record.run_id != self.run_id or item.record.validity is not ValidityState.ACTIVE
            for item in self.mandatory_memories
        ):
            raise ValueError("batch mandatory memory is invalid")
        return self


class BatchManifest(_FrozenModel):
    payload: BatchPayload
    payload_size: TextSize
    batch_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def metrics_and_digest_match_payload(self) -> Self:
        if self._attested_payload() is None:
            raise ValueError("batch manifest attestation is inconsistent")
        return self

    def _attested_payload(self) -> bytes | None:
        try:
            encoded = canonical_json(self.payload)
            measured = DeterministicTokenCounter().measure(encoded.decode("utf-8"))
            expected_digest = length_prefixed_sha256(
                encoded,
                domain="saliencegate:batch:manifest:v1",
            )
        except (CanonicalJSONError, UnicodeError, ValueError):
            return None
        if self.payload_size != measured or self.batch_digest != expected_digest:
            return None
        return encoded

    def canonical_payload(self) -> bytes:
        encoded = self._attested_payload()
        if encoded is None:
            raise BatchIntegrityError()
        return encoded


class BatchBuildResult(_FrozenModel):
    status: BatchStatus
    reason_code: ReasonCode | None = None
    manifest: BatchManifest | None = None
    required_size: TextSize

    @model_validator(mode="after")
    def status_matches_payload(self) -> Self:
        overflow = self.status is BatchStatus.MANDATORY_INPUT_OVERFLOW
        if overflow:
            if self.reason_code is not ReasonCode.MANDATORY_INPUT_OVERFLOW:
                raise ValueError("overflow requires its stable reason code")
            if self.manifest is not None:
                raise ValueError("overflow cannot expose a partial batch")
        elif self.reason_code is not None or self.manifest is None:
            raise ValueError("ready batch requires a manifest and no silence reason")
        if self.manifest is not None and self.required_size != self.manifest.payload_size:
            raise ValueError("ready batch required size must match its manifest")
        return self


def _copy_request(request: object) -> BatchRequest:
    if type(request) is not BatchRequest:
        raise BatchInputError()
    try:
        return BatchRequest.model_validate_json(request.model_dump_json(warnings=False))
    except (AttributeError, TypeError, ValueError, ValidationError):
        raise BatchInputError() from None


def _copy_config(config: object) -> BatchConfig:
    if type(config) is not BatchConfig:
        raise BatchInputError()
    try:
        return BatchConfig.model_validate_json(config.model_dump_json(warnings=False))
    except (AttributeError, TypeError, ValueError, ValidationError):
        raise BatchInputError() from None


def _digest(value: object, *, domain: str) -> str:
    return length_prefixed_sha256(canonical_json(value), domain=domain)


def _json_shape(value: object) -> object:
    """Describe JSON structure without retaining caller-controlled keys or values."""

    if isinstance(value, Mapping):
        children = tuple(
            sorted(
                (_json_shape(item) for item in value.values()),
                key=canonical_json,
            )
        )
        return {
            "field_count": len(value),
            "fields": children,
            "kind": "object",
        }
    if isinstance(value, tuple):
        return {
            "item_count": len(value),
            "items": tuple(_json_shape(item) for item in value),
            "kind": "array",
        }
    if value is None:
        return {"kind": "null"}
    if type(value) is bool:
        return {"kind": "boolean"}
    if type(value) is int:
        return {"kind": "integer"}
    if type(value) is float:
        return {"kind": "number"}
    if type(value) is str:
        return {"kind": "string"}
    raise BatchInputError()


def _event_structural_fingerprint(event: TraceEvent) -> str:
    return _digest(
        {
            "event_type": event.event_type.value,
            "phase": event.phase.value,
            "payload_digest_algorithm": event.payload_digest.algorithm.value,
            "payload_shape": _json_shape(event.payload),
            "trust_label": event.trust_label.value,
        },
        domain="saliencegate:batch:event-structural-fingerprint:v1",
    )


def _ranges(sequences: Iterable[int]) -> tuple[SequenceRange, ...]:
    ordered = tuple(sorted(sequences))
    ranges: list[SequenceRange] = []
    start = ordered[0]
    end = start
    for sequence in ordered[1:]:
        if sequence == end + 1:
            end = sequence
            continue
        ranges.append(SequenceRange(first_sequence=start, last_sequence=end))
        start = end = sequence
    ranges.append(SequenceRange(first_sequence=start, last_sequence=end))
    return tuple(ranges)


def _aggregate(
    events: tuple[TraceEvent, ...],
    priority_kinds: tuple[BatchPriorityKind, ...],
) -> EventAggregate:
    first = events[0]
    provenance = _ranges(event.sequence for event in events)
    sources = [
        {
            "sequence": event.sequence,
            "event_id": str(event.event_id),
            "event_type": event.event_type.value,
            "phase": event.phase.value,
            "payload_digest": event.payload_digest.model_dump(mode="json"),
        }
        for event in events
    ]
    return EventAggregate(
        event_type=first.event_type,
        phase=first.phase,
        trust_label=first.trust_label,
        structural_fingerprint=_event_structural_fingerprint(first),
        priority_kinds=priority_kinds,
        count=len(events),
        first_sequence=events[0].sequence,
        last_sequence=events[-1].sequence,
        provenance_ranges=provenance,
        source_event_ids=tuple(event.event_id for event in events),
        source_digest=_digest(sources, domain="saliencegate:batch:aggregate-sources:v1"),
    )


def _aggregate_key(
    event: TraceEvent,
    reasons: Mapping[UUID, set[BatchPriorityKind]],
) -> tuple[str, str, str, str, tuple[str, ...]]:
    return (
        event.event_type.value,
        event.phase.value,
        event.trust_label.value,
        _event_structural_fingerprint(event),
        tuple(
            kind.value
            for kind in sorted(reasons.get(event.event_id, set()), key=lambda kind: kind.value)
        ),
    )


def _aggregate_model_key(
    aggregate: EventAggregate,
) -> tuple[str, str, str, str, tuple[str, ...]]:
    return (
        aggregate.event_type.value,
        aggregate.phase.value,
        aggregate.trust_label.value,
        aggregate.structural_fingerprint,
        tuple(kind.value for kind in aggregate.priority_kinds),
    )


def _aggregates(
    events: Iterable[TraceEvent],
    reasons: Mapping[UUID, set[BatchPriorityKind]],
) -> tuple[EventAggregate, ...]:
    groups: dict[tuple[str, str, str, str, tuple[str, ...]], list[TraceEvent]] = defaultdict(list)
    for event in events:
        groups[_aggregate_key(event, reasons)].append(event)
    aggregated = tuple(
        _aggregate(
            tuple(sorted(group, key=lambda event: event.sequence)),
            tuple(BatchPriorityKind(value) for value in key[-1]),
        )
        for key, group in sorted(groups.items())
    )
    return tuple(
        sorted(
            aggregated,
            key=lambda item: (
                item.first_sequence,
                item.event_type.value,
                item.structural_fingerprint,
            ),
        )
    )


def _priority_reasons(request: BatchRequest) -> dict[UUID, set[BatchPriorityKind]]:
    reasons: dict[UUID, set[BatchPriorityKind]] = defaultdict(set)
    for event in request.events:
        if event.event_type is EventType.CONTROLLER_ERROR:
            reasons[event.event_id].add(BatchPriorityKind.CONTROLLER_ERROR)
        if event.event_type is EventType.ACTION_PROPOSAL:
            reasons[event.event_id].add(BatchPriorityKind.ACTION_PROPOSAL)
    signal_kinds = {
        SignalType.TOOL_ERROR: BatchPriorityKind.TOOL_ERROR,
        SignalType.TEST_FAILURE: BatchPriorityKind.TEST_FAILURE,
        SignalType.CONFLICT: BatchPriorityKind.CONFLICT,
    }
    for signal in request.signals:
        kind = signal_kinds.get(signal.signal_type)
        if kind is not None:
            for event_id in signal.evidence_event_ids:
                reasons[event_id].add(kind)
    return reasons


_PRIORITY_ORDER = (
    BatchPriorityKind.CONTROLLER_ERROR,
    BatchPriorityKind.TEST_FAILURE,
    BatchPriorityKind.CONFLICT,
    BatchPriorityKind.TOOL_ERROR,
    BatchPriorityKind.ACTION_PROPOSAL,
)


def _priority_limits(config: BatchConfig) -> dict[BatchPriorityKind, int]:
    return {
        BatchPriorityKind.CONTROLLER_ERROR: config.max_controller_errors,
        BatchPriorityKind.ACTION_PROPOSAL: config.max_action_proposals,
        BatchPriorityKind.TOOL_ERROR: config.max_tool_errors,
        BatchPriorityKind.TEST_FAILURE: config.max_test_failures,
        BatchPriorityKind.CONFLICT: config.max_conflicts,
    }


def _priority_candidates(
    request: BatchRequest,
    mandatory_ids: set[UUID],
    reasons: Mapping[UUID, set[BatchPriorityKind]],
) -> tuple[TraceEvent, ...]:
    rank = {kind: index for index, kind in enumerate(_PRIORITY_ORDER)}
    return tuple(
        sorted(
            (
                event
                for event in request.events
                if event.event_id not in mandatory_ids and reasons.get(event.event_id)
            ),
            key=lambda event: (
                min(rank[reason] for reason in reasons[event.event_id]),
                -event.sequence,
                str(event.event_id),
            ),
        )
    )


def _mandatory_memories(request: BatchRequest) -> tuple[BatchMemory, ...]:
    return tuple(
        BatchMemory(role=BatchMemoryRole.TASK_REQUIREMENT, record=record)
        for record in request.task_requirements
    ) + tuple(
        BatchMemory(role=BatchMemoryRole.UNRESOLVED_STATE, record=record)
        for record in request.unresolved_state
    )


def _verbatim_event(
    event: TraceEvent,
    reasons: Mapping[UUID, set[BatchPriorityKind]],
) -> VerbatimEvent:
    return VerbatimEvent(
        event=event,
        priority_kinds=tuple(
            sorted(reasons.get(event.event_id, set()), key=lambda kind: kind.value)
        ),
    )


def _payload(
    request: BatchRequest,
    config_digest: str,
    verbatim_ids: set[UUID],
    reasons: Mapping[UUID, set[BatchPriorityKind]],
) -> BatchPayload:
    verbatim = tuple(
        _verbatim_event(event, reasons)
        for event in request.events
        if event.event_id in verbatim_ids
    )
    compacted = _aggregates(
        (event for event in request.events if event.event_id not in verbatim_ids),
        reasons,
    )
    return BatchPayload(
        run_id=request.run_id,
        memory_cursor=request.memory_cursor,
        first_event_sequence=request.events[0].sequence,
        last_event_sequence=request.events[-1].sequence,
        configuration_digest=config_digest,
        mandatory_memories=_mandatory_memories(request),
        verbatim_events=verbatim,
        aggregates=compacted,
        represented_event_count=len(request.events),
    )


def _array_inner_size(count: int, item_bytes: int) -> int:
    if count < 0 or item_bytes < 0 or ((count == 0) != (item_bytes == 0)):
        raise BatchIntegrityError()
    return item_bytes + max(count - 1, 0)


_RANGE_FIXED_BYTES = len(
    canonical_json(SequenceRange(first_sequence=1, last_sequence=1))
) - 2 * len(canonical_json(1))


@dataclass(frozen=True, slots=True)
class _AggregateRemoval:
    sequence: int
    previous_sequence: int | None
    next_sequence: int | None
    count: int
    head_sequence: int | None
    tail_sequence: int | None
    range_count: int
    range_endpoint_bytes: int
    source_id_bytes: int
    serialized_size: int


@dataclass(slots=True)
class _AggregateSizeState:
    aggregate_shell: dict[str, object]
    remaining_sequences: set[int]
    previous_sequences: dict[int, int | None]
    next_sequences: dict[int, int | None]
    event_sequences: dict[UUID, int]
    source_id_sizes: dict[UUID, int]
    count: int
    head_sequence: int | None
    tail_sequence: int | None
    range_count: int
    range_endpoint_bytes: int
    source_id_bytes: int
    current_size: int

    @classmethod
    def from_aggregate(
        cls,
        aggregate: EventAggregate,
        events: tuple[TraceEvent, ...],
    ) -> Self:
        ordered = tuple(sorted(events, key=lambda event: event.sequence))
        if tuple(event.event_id for event in ordered) != aggregate.source_event_ids:
            raise BatchIntegrityError()
        sequences = tuple(event.sequence for event in ordered)
        previous_sequences: dict[int, int | None] = {
            sequence: sequences[index - 1] if index else None
            for index, sequence in enumerate(sequences)
        }
        next_sequences: dict[int, int | None] = {
            sequence: sequences[index + 1] if index + 1 < len(sequences) else None
            for index, sequence in enumerate(sequences)
        }
        source_id_sizes = {
            event.event_id: len(canonical_json(str(event.event_id))) for event in ordered
        }
        aggregate_shell = aggregate.model_dump(mode="json")
        aggregate_shell["provenance_ranges"] = []
        aggregate_shell["source_event_ids"] = []
        state = cls(
            aggregate_shell=aggregate_shell,
            remaining_sequences=set(sequences),
            previous_sequences=previous_sequences,
            next_sequences=next_sequences,
            event_sequences={event.event_id: event.sequence for event in ordered},
            source_id_sizes=source_id_sizes,
            count=len(ordered),
            head_sequence=sequences[0],
            tail_sequence=sequences[-1],
            range_count=len(aggregate.provenance_ranges),
            range_endpoint_bytes=sum(
                len(canonical_json(item.first_sequence)) + len(canonical_json(item.last_sequence))
                for item in aggregate.provenance_ranges
            ),
            source_id_bytes=sum(source_id_sizes.values()),
            current_size=0,
        )
        state.current_size = state._serialized_size(
            count=state.count,
            head_sequence=state.head_sequence,
            tail_sequence=state.tail_sequence,
            range_count=state.range_count,
            range_endpoint_bytes=state.range_endpoint_bytes,
            source_id_bytes=state.source_id_bytes,
        )
        if state.current_size != len(canonical_json(aggregate)):
            raise BatchIntegrityError()
        return state

    def _serialized_size(
        self,
        *,
        count: int,
        head_sequence: int | None,
        tail_sequence: int | None,
        range_count: int,
        range_endpoint_bytes: int,
        source_id_bytes: int,
    ) -> int:
        if count == 0:
            return 0
        if head_sequence is None or tail_sequence is None or range_count < 1:
            raise BatchIntegrityError()
        shell = dict(self.aggregate_shell)
        shell["count"] = count
        shell["first_sequence"] = head_sequence
        shell["last_sequence"] = tail_sequence
        range_item_bytes = range_count * _RANGE_FIXED_BYTES + range_endpoint_bytes
        return (
            len(canonical_json(shell))
            + _array_inner_size(range_count, range_item_bytes)
            + _array_inner_size(count, source_id_bytes)
        )

    def preview_removal(self, event_id: UUID) -> _AggregateRemoval:
        try:
            sequence = self.event_sequences[event_id]
            previous_sequence = self.previous_sequences[sequence]
            next_sequence = self.next_sequences[sequence]
            source_size = self.source_id_sizes[event_id]
        except KeyError:
            raise BatchIntegrityError() from None

        count = self.count - 1
        head_sequence = next_sequence if sequence == self.head_sequence else self.head_sequence
        tail_sequence = previous_sequence if sequence == self.tail_sequence else self.tail_sequence
        if count == 0:
            head_sequence = tail_sequence = None

        previous_is_adjacent = sequence - 1 in self.remaining_sequences
        next_is_adjacent = sequence + 1 in self.remaining_sequences
        range_count = self.range_count
        range_endpoint_bytes = self.range_endpoint_bytes
        sequence_size = len(canonical_json(sequence))
        if previous_is_adjacent and next_is_adjacent:
            range_count += 1
            range_endpoint_bytes += len(canonical_json(sequence - 1)) + len(
                canonical_json(sequence + 1)
            )
        elif previous_is_adjacent:
            range_endpoint_bytes += len(canonical_json(sequence - 1)) - sequence_size
        elif next_is_adjacent:
            range_endpoint_bytes += len(canonical_json(sequence + 1)) - sequence_size
        else:
            range_count -= 1
            range_endpoint_bytes -= 2 * sequence_size

        source_id_bytes = self.source_id_bytes - source_size
        serialized_size = self._serialized_size(
            count=count,
            head_sequence=head_sequence,
            tail_sequence=tail_sequence,
            range_count=range_count,
            range_endpoint_bytes=range_endpoint_bytes,
            source_id_bytes=source_id_bytes,
        )
        return _AggregateRemoval(
            sequence=sequence,
            previous_sequence=previous_sequence,
            next_sequence=next_sequence,
            count=count,
            head_sequence=head_sequence,
            tail_sequence=tail_sequence,
            range_count=range_count,
            range_endpoint_bytes=range_endpoint_bytes,
            source_id_bytes=source_id_bytes,
            serialized_size=serialized_size,
        )

    def apply_removal(self, event_id: UUID, removal: _AggregateRemoval) -> None:
        if self.event_sequences.get(event_id) != removal.sequence:
            raise BatchIntegrityError()
        if removal.previous_sequence is not None:
            self.next_sequences[removal.previous_sequence] = removal.next_sequence
        if removal.next_sequence is not None:
            self.previous_sequences[removal.next_sequence] = removal.previous_sequence
        self.remaining_sequences.remove(removal.sequence)
        del self.previous_sequences[removal.sequence]
        del self.next_sequences[removal.sequence]
        del self.event_sequences[event_id]
        del self.source_id_sizes[event_id]
        self.count = removal.count
        self.head_sequence = removal.head_sequence
        self.tail_sequence = removal.tail_sequence
        self.range_count = removal.range_count
        self.range_endpoint_bytes = removal.range_endpoint_bytes
        self.source_id_bytes = removal.source_id_bytes
        self.current_size = removal.serialized_size


@dataclass(frozen=True, slots=True)
class _PromotionPreview:
    event_id: UUID
    aggregate_state: _AggregateSizeState
    aggregate_removal: _AggregateRemoval
    verbatim_count: int
    verbatim_item_bytes: int
    aggregate_count: int
    aggregate_item_bytes: int
    total_size: int


@dataclass(slots=True)
class _BatchSizePlanner:
    shell_size: int
    verbatim_count: int
    verbatim_item_bytes: int
    aggregate_count: int
    aggregate_item_bytes: int
    total_size: int
    aggregate_by_event: dict[UUID, _AggregateSizeState]

    @classmethod
    def from_payload(
        cls,
        request: BatchRequest,
        payload: BatchPayload,
        payload_size: TextSize,
        verbatim_ids: set[UUID],
        reasons: Mapping[UUID, set[BatchPriorityKind]],
    ) -> Self:
        groups: dict[tuple[str, str, str, str, tuple[str, ...]], list[TraceEvent]] = defaultdict(
            list
        )
        for event in request.events:
            if event.event_id not in verbatim_ids:
                groups[_aggregate_key(event, reasons)].append(event)
        aggregates_by_key = {
            _aggregate_model_key(aggregate): aggregate for aggregate in payload.aggregates
        }
        if len(aggregates_by_key) != len(payload.aggregates) or set(groups) != set(
            aggregates_by_key
        ):
            raise BatchIntegrityError()

        aggregate_by_event: dict[UUID, _AggregateSizeState] = {}
        states: list[_AggregateSizeState] = []
        for key, grouped_events in groups.items():
            state = _AggregateSizeState.from_aggregate(
                aggregates_by_key[key],
                tuple(grouped_events),
            )
            states.append(state)
            for event in grouped_events:
                aggregate_by_event[event.event_id] = state

        shell = payload.model_dump(mode="json")
        shell["aggregates"] = []
        shell["verbatim_events"] = []
        verbatim_item_bytes = sum(len(canonical_json(item)) for item in payload.verbatim_events)
        aggregate_item_bytes = sum(state.current_size for state in states)
        total_size = (
            len(canonical_json(shell))
            + _array_inner_size(len(payload.verbatim_events), verbatim_item_bytes)
            + _array_inner_size(len(states), aggregate_item_bytes)
        )
        if total_size != payload_size.utf8_bytes:
            raise BatchIntegrityError()
        return cls(
            shell_size=len(canonical_json(shell)),
            verbatim_count=len(payload.verbatim_events),
            verbatim_item_bytes=verbatim_item_bytes,
            aggregate_count=len(states),
            aggregate_item_bytes=aggregate_item_bytes,
            total_size=total_size,
            aggregate_by_event=aggregate_by_event,
        )

    def preview(self, event_id: UUID, verbatim_size: int) -> _PromotionPreview:
        try:
            aggregate_state = self.aggregate_by_event[event_id]
        except KeyError:
            raise BatchIntegrityError() from None
        removal = aggregate_state.preview_removal(event_id)
        aggregate_count = self.aggregate_count - int(removal.count == 0)
        aggregate_item_bytes = (
            self.aggregate_item_bytes - aggregate_state.current_size + removal.serialized_size
        )
        verbatim_count = self.verbatim_count + 1
        verbatim_item_bytes = self.verbatim_item_bytes + verbatim_size
        total_size = (
            self.shell_size
            + _array_inner_size(verbatim_count, verbatim_item_bytes)
            + _array_inner_size(aggregate_count, aggregate_item_bytes)
        )
        return _PromotionPreview(
            event_id=event_id,
            aggregate_state=aggregate_state,
            aggregate_removal=removal,
            verbatim_count=verbatim_count,
            verbatim_item_bytes=verbatim_item_bytes,
            aggregate_count=aggregate_count,
            aggregate_item_bytes=aggregate_item_bytes,
            total_size=total_size,
        )

    def apply(self, preview: _PromotionPreview) -> None:
        if self.aggregate_by_event.get(preview.event_id) is not preview.aggregate_state:
            raise BatchIntegrityError()
        preview.aggregate_state.apply_removal(preview.event_id, preview.aggregate_removal)
        del self.aggregate_by_event[preview.event_id]
        self.verbatim_count = preview.verbatim_count
        self.verbatim_item_bytes = preview.verbatim_item_bytes
        self.aggregate_count = preview.aggregate_count
        self.aggregate_item_bytes = preview.aggregate_item_bytes
        self.total_size = preview.total_size


def _measure(payload: BatchPayload, counter: DeterministicTokenCounter) -> TextSize:
    try:
        serialized = canonical_json(payload).decode("utf-8", errors="strict")
        return counter.measure(serialized)
    except (CanonicalJSONError, UnicodeError, ValueError):
        raise BatchInputError() from None


def _fits(size: TextSize, config: BatchConfig) -> bool:
    return (
        size.utf8_bytes <= config.max_utf8_bytes
        and size.approximate_tokens <= config.max_approximate_tokens
    )


class DeterministicBatcher:
    """Represent one contiguous event slice without generated natural-language summaries."""

    __slots__ = ("_counter",)

    def __init__(self, *, counter: DeterministicTokenCounter | None = None) -> None:
        if counter is not None and type(counter) is not DeterministicTokenCounter:
            raise TypeError("counter must be exactly DeterministicTokenCounter")
        self._counter = DeterministicTokenCounter() if counter is None else counter

    def build(self, request: BatchRequest, config: BatchConfig) -> BatchBuildResult:
        validated = _copy_request(request)
        resolved = _copy_config(config)
        try:
            config_digest = _digest(
                resolved.model_dump(mode="json"),
                domain="saliencegate:batch:configuration:v1",
            )
            reasons = _priority_reasons(validated)
            recent = (
                validated.events[-resolved.recent_event_count :]
                if resolved.recent_event_count
                else ()
            )
            verbatim_ids = {event.event_id for event in recent}
            current = _payload(validated, config_digest, verbatim_ids, reasons)
            current_size = _measure(current, self._counter)
        except (CanonicalJSONError, KeyError, TypeError, ValueError, ValidationError):
            raise BatchInputError() from None

        if not _fits(current_size, resolved):
            return BatchBuildResult(
                status=BatchStatus.MANDATORY_INPUT_OVERFLOW,
                reason_code=ReasonCode.MANDATORY_INPUT_OVERFLOW,
                required_size=current_size,
            )

        try:
            planner = _BatchSizePlanner.from_payload(
                validated,
                current,
                current_size,
                verbatim_ids,
                reasons,
            )
            limits = _priority_limits(resolved)
            used = {
                kind: sum(kind in reasons.get(event_id, set()) for event_id in verbatim_ids)
                for kind in _PRIORITY_ORDER
            }
            candidates = _priority_candidates(
                validated,
                verbatim_ids,
                reasons,
            )
            effective_byte_budget = min(
                resolved.max_utf8_bytes,
                resolved.max_approximate_tokens * 4,
            )
            for candidate in candidates:
                candidate_reasons = reasons[candidate.event_id]
                if any(used[reason] >= limits[reason] for reason in candidate_reasons):
                    continue
                wrapper = _verbatim_event(candidate, reasons)
                preview = planner.preview(candidate.event_id, len(canonical_json(wrapper)))
                if (
                    preview.total_size < planner.total_size
                    or preview.total_size > effective_byte_budget
                ):
                    continue
                planner.apply(preview)
                verbatim_ids.add(candidate.event_id)
                for reason in candidate_reasons:
                    used[reason] += 1

            if len(verbatim_ids) != len(recent):
                current = _payload(validated, config_digest, verbatim_ids, reasons)
                current_size = _measure(current, self._counter)
            if current_size.utf8_bytes != planner.total_size or not _fits(
                current_size,
                resolved,
            ):
                raise BatchIntegrityError()

            payload_bytes = canonical_json(current)
            manifest = BatchManifest(
                payload=current,
                payload_size=current_size,
                batch_digest=length_prefixed_sha256(
                    payload_bytes,
                    domain="saliencegate:batch:manifest:v1",
                ),
            )
            return BatchBuildResult(
                status=BatchStatus.READY,
                manifest=manifest,
                required_size=current_size,
            )
        except BatchIntegrityError:
            raise
        except (CanonicalJSONError, KeyError, TypeError, ValueError, ValidationError):
            raise BatchInputError() from None


__all__ = [
    "BatchBuildResult",
    "BatchConfig",
    "BatchInputError",
    "BatchIntegrityError",
    "BatchManifest",
    "BatchMemory",
    "BatchMemoryRole",
    "BatchPayload",
    "BatchPriorityKind",
    "BatchRequest",
    "BatchStatus",
    "DeterministicBatcher",
    "EventAggregate",
    "SequenceRange",
    "VerbatimEvent",
]
