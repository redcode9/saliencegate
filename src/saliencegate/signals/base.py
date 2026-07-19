from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from itertools import pairwise
from types import MappingProxyType
from typing import Annotated, Protocol, Self
from uuid import UUID
from weakref import WeakKeyDictionary

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from saliencegate.domain import (
    EventPhase,
    EventType,
    PayloadDigest,
    PayloadDigestAlgorithm,
    ReasonCode,
    Signal,
    SignalType,
    TraceEvent,
    TrustLabel,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.domain.records import UUID4, UtcDatetime

_MAX_SEQUENCE = (1 << 63) - 1
_MAX_CONTEXT_SIZE_UPPER_BOUND = 10_000_000
_MAX_CONTEXT_PAYLOAD_NODES = 200_000
_MAX_EVENT_PARENTS = 256
_DETECTOR_VERSION = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._:/+\-]{0,255}$")
_DETECTION_SEQUENCE_TOKEN = object()
_TRUSTED_CONTEXT_TOKEN = object()
_TRUSTED_EXTRACTION_TOKEN = object()
_EXTRACTOR_TOKEN = object()


class _TrustedSeal:
    __slots__ = ("__weakref__",)


_DETECTION_SEQUENCE_SEALS: WeakKeyDictionary[_TrustedSeal, tuple[object, ...]] = WeakKeyDictionary()
_TRUSTED_CONTEXT_SEALS: WeakKeyDictionary[_TrustedSeal, tuple[object, ...]] = WeakKeyDictionary()
_TRUSTED_EXTRACTION_SEALS: WeakKeyDictionary[_TrustedSeal, tuple[object, ...]] = WeakKeyDictionary()
_TRUSTED_EXTRACTOR_SEALS: WeakKeyDictionary[_TrustedSeal, tuple[object, ...]] = WeakKeyDictionary()


class DetectionInputError(ValueError):
    """A sanitized detector-boundary validation failure."""

    def __init__(self) -> None:
        super().__init__("signal detection input failed validation")


class DetectorContractError(ValueError):
    """A detector plugin violated the deterministic output contract."""

    def __init__(self) -> None:
        super().__init__("signal detector output failed contract validation")


class DetectionStatus(StrEnum):
    DETECTED = "detected"
    NO_MATCH = "no_match"
    ABSTAINED = "abstained"


class AbstentionReason(StrEnum):
    EVENT_NOT_APPLICABLE = "event_not_applicable"
    STRUCTURED_EVIDENCE_MISSING = "structured_evidence_missing"
    STRUCTURED_EVIDENCE_INVALID = "structured_evidence_invalid"
    INSUFFICIENT_HISTORY = "insufficient_history"
    PRE_ACTION_INTERCEPTION_UNAVAILABLE = "pre_action_interception_unavailable"
    PARENT_ACTION_MISSING = "parent_action_missing"
    AMBIGUOUS_PARENT_ACTION = "ambiguous_parent_action"
    REDACTED_EQUIVALENCE_INPUT = "redacted_equivalence_input"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
        validate_default=True,
    )


class DetectionContext(_FrozenModel):
    """A verified contiguous ledger window ending at the event under evaluation."""

    run_id: UUID4
    events: Annotated[
        tuple[TraceEvent, ...],
        Field(min_length=1, max_length=10_000, repr=False),
    ]

    @field_validator("events")
    @classmethod
    def order_events(cls, value: tuple[TraceEvent, ...]) -> tuple[TraceEvent, ...]:
        return tuple(sorted(value, key=lambda event: event.sequence))

    @model_validator(mode="after")
    def events_form_one_run_window(self) -> Self:
        if any(event.run_id != self.run_id for event in self.events):
            raise ValueError("detection events must belong to one run")
        sequences = tuple(event.sequence for event in self.events)
        if sequences[-1] > _MAX_SEQUENCE or any(
            right != left + 1 for left, right in pairwise(sequences)
        ):
            raise ValueError("detection events must be a contiguous signed-64-bit sequence")
        event_ids = tuple(event.event_id for event in self.events)
        source_ids = tuple(event.source_event_id for event in self.events)
        if len(set(event_ids)) != len(event_ids) or len(set(source_ids)) != len(source_ids):
            raise ValueError("detection event identities must be unique")
        if not _context_is_bounded(self):
            raise ValueError("detection context exceeds its local size bound")
        return self

    @property
    def current(self) -> TraceEvent:
        return self.events[-1]


class DetectionOutcome(_FrozenModel):
    signal_type: SignalType
    status: DetectionStatus
    strength: Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)] | None = None
    reason_code: ReasonCode | None = None
    evidence_event_ids: Annotated[tuple[UUID4, ...], Field(max_length=10_000)] = ()
    related_event_ids: Annotated[tuple[UUID4, ...], Field(max_length=10_000)] = ()
    abstention_reason: AbstentionReason | None = None

    @model_validator(mode="after")
    def fields_match_status(self) -> Self:
        detected = self.status is DetectionStatus.DETECTED
        abstained = self.status is DetectionStatus.ABSTAINED
        expected_reason = ReasonCode(self.signal_type.value)
        if detected:
            if (
                self.strength is None
                or self.reason_code is not expected_reason
                or not self.evidence_event_ids
                or self.related_event_ids
                or self.abstention_reason is not None
            ):
                raise ValueError("detected outcome fields are inconsistent")
        elif (
            self.strength is not None
            or self.reason_code is not None
            or self.evidence_event_ids
            or not self.related_event_ids
            or (abstained != (self.abstention_reason is not None))
        ):
            raise ValueError("non-detected outcome fields are inconsistent")
        if len(set(self.evidence_event_ids)) != len(self.evidence_event_ids) or len(
            set(self.related_event_ids)
        ) != len(self.related_event_ids):
            raise ValueError("detection evidence IDs must be unique")
        return self

    @classmethod
    def detected(
        cls,
        signal_type: SignalType,
        evidence_event_ids: tuple[UUID, ...],
        *,
        strength: float = 1.0,
    ) -> DetectionOutcome:
        return cls(
            signal_type=signal_type,
            status=DetectionStatus.DETECTED,
            strength=strength,
            reason_code=ReasonCode(signal_type.value),
            evidence_event_ids=evidence_event_ids,
        )

    @classmethod
    def no_match(
        cls,
        signal_type: SignalType,
        related_event_ids: tuple[UUID, ...],
    ) -> DetectionOutcome:
        return cls(
            signal_type=signal_type,
            status=DetectionStatus.NO_MATCH,
            related_event_ids=related_event_ids,
        )

    @classmethod
    def abstained(
        cls,
        signal_type: SignalType,
        reason: AbstentionReason,
        related_event_ids: tuple[UUID, ...],
    ) -> DetectionOutcome:
        return cls(
            signal_type=signal_type,
            status=DetectionStatus.ABSTAINED,
            abstention_reason=reason,
            related_event_ids=related_event_ids,
        )


class DetectorEvaluation(_FrozenModel):
    """One sanitized, version-attributed detector result."""

    signal_type: SignalType
    detector_version: str = Field(pattern=_DETECTOR_VERSION.pattern)
    outcome: DetectionOutcome

    @field_validator("detector_version", mode="before")
    @classmethod
    def exact_detector_version(cls, value: str) -> str:
        if type(value) is not str:
            raise ValueError("detector version must be an exact string")
        return value

    @model_validator(mode="after")
    def outcome_matches_detector(self) -> Self:
        if self.outcome.signal_type is not self.signal_type:
            raise ValueError("detector evaluation signal types disagree")
        return self


def _uuid4_tuple_is_bounded(value: object) -> bool:
    return (
        type(value) is tuple
        and len(value) <= 10_000
        and all(type(item) is UUID and item.version == 4 for item in value)
    )


def _outcome_is_preflight_safe(value: object) -> bool:
    if type(value) is not DetectionOutcome:
        return False
    return (
        type(value.signal_type) is SignalType
        and type(value.status) is DetectionStatus
        and (
            value.strength is None
            or (type(value.strength) is float and math.isfinite(value.strength))
        )
        and (value.reason_code is None or type(value.reason_code) is ReasonCode)
        and (value.abstention_reason is None or type(value.abstention_reason) is AbstentionReason)
        and _uuid4_tuple_is_bounded(value.evidence_event_ids)
        and _uuid4_tuple_is_bounded(value.related_event_ids)
    )


def _evaluation_is_preflight_safe(value: object) -> bool:
    return (
        type(value) is DetectorEvaluation
        and type(value.signal_type) is SignalType
        and type(value.detector_version) is str
        and len(value.detector_version) <= 256
        and _outcome_is_preflight_safe(value.outcome)
    )


def _signal_is_preflight_safe(value: object) -> bool:
    return (
        type(value) is Signal
        and value.schema_version == "1.0"
        and type(value.schema_version) is str
        and value.record_type == "signal"
        and type(value.record_type) is str
        and type(value.signal_id) is UUID
        and value.signal_id.version == 4
        and type(value.run_id) is UUID
        and value.run_id.version == 4
        and type(value.created_at) is datetime
        and type(value.signal_type) is SignalType
        and type(value.strength) is float
        and math.isfinite(value.strength)
        and _uuid4_tuple_is_bounded(value.evidence_event_ids)
        and type(value.detector_version) is str
        and len(value.detector_version) <= 256
        and type(value.reason_code) is ReasonCode
    )


class ExtractionReport(_FrozenModel):
    """Auditable extraction result, including no-match and abstention outcomes."""

    run_id: UUID4
    current_event_id: UUID4
    current_event_timestamp: UtcDatetime
    evaluations: Annotated[tuple[DetectorEvaluation, ...], Field(max_length=len(SignalType))]
    signals: Annotated[tuple[Signal, ...], Field(max_length=len(SignalType))]

    @model_validator(mode="after")
    def contents_match_run(self) -> Self:
        invalid_records = False
        try:
            invalid_records = any(
                not _evaluation_is_preflight_safe(item)
                or DetectorEvaluation.model_validate_json(item.model_dump_json(warnings=False))
                != item
                for item in self.evaluations
            ) or any(
                not _signal_is_preflight_safe(signal)
                or Signal.model_validate_json(signal.model_dump_json(warnings=False)) != signal
                for signal in self.signals
            )
        except (AttributeError, TypeError, ValueError, ValidationError):
            invalid_records = True
        if invalid_records:
            raise ValueError("extraction report contains unvalidated records")
        if any(
            (
                item.outcome.status is DetectionStatus.DETECTED
                and item.outcome.evidence_event_ids[-1] != self.current_event_id
            )
            or (
                item.outcome.status is not DetectionStatus.DETECTED
                and item.outcome.related_event_ids[-1] != self.current_event_id
            )
            for item in self.evaluations
        ):
            raise ValueError("extraction report outcomes do not end at the current event")
        evaluation_types = tuple(item.signal_type for item in self.evaluations)
        if len(set(evaluation_types)) != len(evaluation_types) or evaluation_types != tuple(
            sorted(evaluation_types, key=lambda item: item.value)
        ):
            raise ValueError("extraction report signal types must be unique")
        detected = {
            item.signal_type: item
            for item in self.evaluations
            if item.outcome.status is DetectionStatus.DETECTED
        }
        signals = {signal.signal_type: signal for signal in self.signals}
        if (
            any(signal.run_id != self.run_id for signal in self.signals)
            or len(signals) != len(self.signals)
            or signals.keys() != detected.keys()
            or tuple(signal.signal_type for signal in self.signals)
            != tuple(sorted(signals, key=lambda item: item.value))
        ):
            raise ValueError("extraction report signals disagree with evaluations")
        for signal_type, signal in signals.items():
            evaluation = detected[signal_type]
            outcome = evaluation.outcome
            if (
                signal.detector_version != evaluation.detector_version
                or signal.strength != outcome.strength
                or signal.reason_code is not outcome.reason_code
                or signal.evidence_event_ids != outcome.evidence_event_ids
                or self.current_event_id not in signal.evidence_event_ids
                or signal.created_at != self.current_event_timestamp
                or signal.signal_id
                != _deterministic_signal_id(
                    self.run_id,
                    outcome,
                    evaluation.detector_version,
                )
            ):
                raise ValueError("extraction report signal attribution is inconsistent")
        return self


class SignalDetector(Protocol):
    @property
    def signal_type(self) -> SignalType: ...

    @property
    def detector_version(self) -> str: ...

    def evaluate(self, context: DetectionContext) -> DetectionOutcome: ...


class _ValidatedSignalDetector(ABC):
    """Nominal marker for built-ins that accept an extractor-validated context."""

    __slots__ = ()

    @abstractmethod
    def _evaluate_validated(self, context: DetectionContext) -> DetectionOutcome: ...


@dataclass(frozen=True, slots=True)
class _DetectorEntry:
    evaluator: Callable[[DetectionContext], object]
    signal_type: SignalType
    detector_version: str
    accepts_trusted_context: bool


def validate_detection_context(value: object) -> DetectionContext:
    if type(value) is not DetectionContext or not _context_is_bounded(value):
        raise DetectionInputError()
    validated: DetectionContext | None = None
    try:
        validated = DetectionContext.model_validate_json(value.model_dump_json(warnings=False))
    except Exception:
        validated = None
    if validated is None or validated != value:
        raise DetectionInputError()
    return validated


def _json_size_upper_bound(value: object, *, limit: int) -> int | None:
    total = 0
    nodes = 0
    active_containers: set[int] = set()
    stack = [(value, False)]
    while stack:
        current, exiting = stack.pop()
        if exiting:
            active_containers.remove(id(current))
            continue
        nodes += 1
        if nodes > _MAX_CONTEXT_PAYLOAD_NODES:
            return None
        if type(current) in (dict, MappingProxyType):
            assert isinstance(current, Mapping)
            identity = id(current)
            if identity in active_containers:
                return None
            active_containers.add(identity)
            total += 2 + max(len(current) - 1, 0) + len(current)
            stack.append((current, True))
            stack.extend((item, False) for item in current)
            stack.extend((item, False) for item in current.values())
        elif type(current) is tuple:
            identity = id(current)
            if identity in active_containers:
                return None
            active_containers.add(identity)
            total += 2 + max(len(current) - 1, 0)
            stack.append((current, True))
            stack.extend((item, False) for item in current)
        elif type(current) is str:
            total += 2 + 6 * len(current)
        elif current is None:
            total += 4
        elif type(current) is bool:
            total += 5
        elif type(current) is int:
            total += max(1, (current.bit_length() * 30_103) // 100_000 + 2)
        elif type(current) is float:
            if not math.isfinite(current):
                return None
            total += 32
        else:
            return None
        if total > limit:
            return None
    return total


def _context_is_bounded(context: DetectionContext) -> bool:
    try:
        if (
            type(context.run_id) is not UUID
            or context.run_id.version != 4
            or type(context.events) is not tuple
        ):
            return False
        remaining = _MAX_CONTEXT_SIZE_UPPER_BOUND
        for event in context.events:
            remaining -= _context_event_cost(event)
            if remaining < 0:
                return False
        return True
    except Exception:
        return False


def _event_metadata_is_structurally_bounded(event: object) -> bool:
    if type(event) is not TraceEvent:
        return False
    digest = event.payload_digest
    return (
        event.schema_version == "1.0"
        and type(event.schema_version) is str
        and event.record_type == "trace_event"
        and type(event.record_type) is str
        and type(event.event_id) is UUID
        and event.event_id.version == 4
        and type(event.run_id) is UUID
        and event.run_id.version == 4
        and type(event.sequence) is int
        and 1 <= event.sequence <= _MAX_SEQUENCE
        and type(event.source_event_id) is str
        and type(event.timestamp) is datetime
        and type(event.event_type) is EventType
        and type(event.phase) is EventPhase
        and type(event.payload) in (dict, MappingProxyType)
        and type(digest) is PayloadDigest
        and type(digest.algorithm) is PayloadDigestAlgorithm
        and type(digest.value) is str
        and len(digest.value) == 64
        and type(event.parent_ids) is tuple
        and len(event.parent_ids) <= _MAX_EVENT_PARENTS
        and all(
            type(parent_id) is UUID and parent_id.version == 4 for parent_id in event.parent_ids
        )
        and type(event.source_adapter) is str
        and type(event.trust_label) is TrustLabel
    )


@dataclass(frozen=True, slots=True, repr=False)
class _DetectionSequenceProof:
    run_id: UUID
    events: tuple[TraceEvent, ...] = field(repr=False)
    event_bytes: tuple[bytes, ...] = field(repr=False)
    _event_costs: tuple[int, ...] = field(repr=False)
    _prefix_costs: tuple[int, ...] = field(repr=False)
    _context_starts: tuple[int, ...] = field(repr=False)
    _maximum_context_cost: int = field(repr=False)
    _token: object = field(repr=False, compare=False)
    _seal: _TrustedSeal = field(repr=False, compare=False)

    def __repr__(self) -> str:
        return f"_DetectionSequenceProof(event_count={len(self.events)})"


@dataclass(frozen=True, slots=True, repr=False)
class _TrustedDetectionContext:
    sequence: _DetectionSequenceProof = field(repr=False)
    context: DetectionContext = field(repr=False)
    start_index: int
    end_ordinal: int
    _events: tuple[TraceEvent, ...] = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)
    _seal: _TrustedSeal = field(repr=False, compare=False)

    def __repr__(self) -> str:
        return (
            "_TrustedDetectionContext("
            f"start_index={self.start_index}, end_ordinal={self.end_ordinal})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class _TrustedExtraction:
    trusted_context: _TrustedDetectionContext = field(repr=False)
    report: ExtractionReport = field(repr=False)
    _token: object = field(repr=False, compare=False)
    _seal: _TrustedSeal = field(repr=False, compare=False)

    @property
    def context(self) -> DetectionContext:
        return self.trusted_context.context

    def __repr__(self) -> str:
        return "_TrustedExtraction(<validated>)"


def _context_event_cost(event: TraceEvent) -> int:
    if not _event_metadata_is_structurally_bounded(event):
        raise ValueError("detection event metadata is invalid")
    metadata_cost = (
        1_024
        + 6 * len(event.source_event_id)
        + 6 * len(event.source_adapter)
        + 40 * len(event.parent_ids)
    )
    payload_cost = _json_size_upper_bound(
        event.payload,
        limit=_MAX_CONTEXT_SIZE_UPPER_BOUND,
    )
    if payload_cost is None:
        raise ValueError("detection event payload is invalid")
    return metadata_cost + payload_cost


def _detection_sequence_proof_is_exact(value: object) -> bool:
    try:
        if type(value) is not _DetectionSequenceProof or type(value._seal) is not _TrustedSeal:
            return False
        sealed = _DETECTION_SEQUENCE_SEALS.get(value._seal)
        return (
            sealed is not None
            and len(sealed) == 7
            and value._token is _DETECTION_SEQUENCE_TOKEN
            and value.run_id is sealed[0]
            and value.events is sealed[1]
            and value.event_bytes is sealed[2]
            and value._event_costs is sealed[3]
            and value._prefix_costs is sealed[4]
            and value._context_starts is sealed[5]
            and value._maximum_context_cost == sealed[6]
            and type(value.run_id) is UUID
            and value.run_id.version == 4
            and type(value.events) is tuple
            and 1 <= len(value.events) <= 10_000
            and type(value.event_bytes) is tuple
            and type(value._event_costs) is tuple
            and type(value._prefix_costs) is tuple
            and type(value._context_starts) is tuple
            and len(value.event_bytes) == len(value.events)
            and len(value._event_costs) == len(value.events)
            and len(value._prefix_costs) == len(value.events) + 1
            and len(value._context_starts) == len(value.events)
            and value._prefix_costs[0] == 0
            and type(value._maximum_context_cost) is int
            and value._maximum_context_cost == _MAX_CONTEXT_SIZE_UPPER_BOUND
            and type(value.events[0]) is TraceEvent
            and type(value.events[-1]) is TraceEvent
            and type(value.event_bytes[0]) is bytes
            and type(value.event_bytes[-1]) is bytes
            and type(value._event_costs[0]) is int
            and value._event_costs[0] > 0
            and type(value._event_costs[-1]) is int
            and value._event_costs[-1] > 0
            and type(value._prefix_costs[-1]) is int
            and value._prefix_costs[-1] > 0
            and type(value._context_starts[0]) is int
            and value._context_starts[0] == 0
            and type(value._context_starts[-1]) is int
            and 0 <= value._context_starts[-1] < len(value.events)
        )
    except Exception:
        return False


def _admit_detection_sequence(events: tuple[TraceEvent, ...]) -> _DetectionSequenceProof:
    result: _DetectionSequenceProof | None = None
    try:
        if type(events) is not tuple or not 1 <= len(events) <= 10_000:
            raise ValueError("detection sequence is invalid")
        snapshots: list[tuple[TraceEvent, bytes]] = []
        for event in events:
            if type(event) is not TraceEvent:
                raise ValueError("detection sequence event is invalid")
            encoded = canonical_json(event)
            copied = TraceEvent.model_validate_json(encoded)
            if copied != event or canonical_json(copied) != encoded:
                raise ValueError("detection sequence event is inexact")
            snapshots.append((copied, encoded))
        snapshots.sort(key=lambda item: item[0].sequence)
        copied_events = tuple(item[0] for item in snapshots)
        event_bytes = tuple(item[1] for item in snapshots)
        run_id = copied_events[0].run_id
        sequences = tuple(event.sequence for event in copied_events)
        event_ids = tuple(event.event_id for event in copied_events)
        source_ids = tuple(event.source_event_id for event in copied_events)
        if (
            type(run_id) is not UUID
            or run_id.version != 4
            or sequences[-1] > _MAX_SEQUENCE
            or any(event.run_id != run_id for event in copied_events)
            or any(right != left + 1 for left, right in pairwise(sequences))
            or len(set(event_ids)) != len(event_ids)
            or len(set(source_ids)) != len(source_ids)
        ):
            raise ValueError("detection sequence is not one contiguous run")
        costs = tuple(_context_event_cost(event) for event in copied_events)
        prefix_costs = [0]
        context_starts: list[int] = []
        first = 0
        window_cost = 0
        for index, cost in enumerate(costs):
            prefix_costs.append(prefix_costs[-1] + cost)
            window_cost += cost
            while window_cost > _MAX_CONTEXT_SIZE_UPPER_BOUND and first < index:
                window_cost -= costs[first]
                first += 1
            if window_cost > _MAX_CONTEXT_SIZE_UPPER_BOUND:
                raise ValueError("detection event exceeds the context bound")
            context_starts.append(first)
        seal = _TrustedSeal()
        result = _DetectionSequenceProof(
            run_id=UUID(int=run_id.int),
            events=copied_events,
            event_bytes=event_bytes,
            _event_costs=costs,
            _prefix_costs=tuple(prefix_costs),
            _context_starts=tuple(context_starts),
            _maximum_context_cost=_MAX_CONTEXT_SIZE_UPPER_BOUND,
            _token=_DETECTION_SEQUENCE_TOKEN,
            _seal=seal,
        )
        _DETECTION_SEQUENCE_SEALS[seal] = (
            result.run_id,
            result.events,
            result.event_bytes,
            result._event_costs,
            result._prefix_costs,
            result._context_starts,
            result._maximum_context_cost,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        result = None
    if result is None or not _detection_sequence_proof_is_exact(result):
        raise DetectionInputError()
    return result


def _trusted_detection_context_is_exact(value: object) -> bool:
    try:
        if type(value) is not _TrustedDetectionContext or type(value._seal) is not _TrustedSeal:
            return False
        sealed = _TRUSTED_CONTEXT_SEALS.get(value._seal)
        if (
            sealed is None
            or len(sealed) != 5
            or value._token is not _TRUSTED_CONTEXT_TOKEN
            or value.sequence is not sealed[0]
            or value.context is not sealed[1]
            or value._events is not sealed[2]
            or value.start_index != sealed[3]
            or value.end_ordinal != sealed[4]
            or not _detection_sequence_proof_is_exact(value.sequence)
            or type(value.start_index) is not int
            or type(value.end_ordinal) is not int
            or not 0 <= value.start_index < value.end_ordinal <= len(value.sequence.events)
            or type(value._events) is not tuple
            or type(value.context) is not DetectionContext
            or value.context.run_id != value.sequence.run_id
            or value.context.events is not value._events
            or value._events is not value.context.events
            or len(value._events) != value.end_ordinal - value.start_index
            or value._events[0] is not value.sequence.events[value.start_index]
            or value._events[-1] is not value.sequence.events[value.end_ordinal - 1]
        ):
            return False
        prefix_costs = value.sequence._prefix_costs
        cost = prefix_costs[value.end_ordinal] - prefix_costs[value.start_index]
        if cost > value.sequence._maximum_context_cost:
            return False
        return value.start_index == 0 or (
            prefix_costs[value.end_ordinal] - prefix_costs[value.start_index - 1]
            > value.sequence._maximum_context_cost
        )
    except Exception:
        return False


def _longest_trusted_detection_context(
    sequence: _DetectionSequenceProof,
    end_ordinal: int,
) -> _TrustedDetectionContext:
    result: _TrustedDetectionContext | None = None
    try:
        if (
            not _detection_sequence_proof_is_exact(sequence)
            or type(end_ordinal) is not int
            or not 1 <= end_ordinal <= len(sequence.events)
        ):
            raise ValueError("trusted detection context coordinate is invalid")
        start_index = sequence._context_starts[end_ordinal - 1]
        window_events = sequence.events[start_index:end_ordinal]
        context = DetectionContext.model_construct(
            run_id=sequence.run_id,
            events=window_events,
        )
        seal = _TrustedSeal()
        result = _TrustedDetectionContext(
            sequence=sequence,
            context=context,
            start_index=start_index,
            end_ordinal=end_ordinal,
            _events=window_events,
            _token=_TRUSTED_CONTEXT_TOKEN,
            _seal=seal,
        )
        _TRUSTED_CONTEXT_SEALS[seal] = (
            result.sequence,
            result.context,
            result._events,
            result.start_index,
            result.end_ordinal,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        result = None
    if result is None or not _trusted_detection_context_is_exact(result):
        raise DetectionInputError()
    return result


def _deterministic_signal_id(
    run_id: UUID,
    outcome: DetectionOutcome,
    detector_version: str,
) -> UUID:
    identity = canonical_json(
        {
            "detector_version": detector_version,
            "evidence_event_ids": tuple(str(value) for value in outcome.evidence_event_ids),
            "reason_code": outcome.reason_code.value if outcome.reason_code is not None else None,
            "run_id": str(run_id),
            "signal_type": outcome.signal_type.value,
            "strength": outcome.strength,
        }
    )
    raw = bytearray(
        bytes.fromhex(length_prefixed_sha256(identity, domain="saliencegate:signal:identity:v1"))[
            :16
        ]
    )
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(raw))


def _materialize_extraction_report(
    context: DetectionContext,
    evaluations: tuple[DetectorEvaluation, ...],
) -> ExtractionReport:
    signals = tuple(
        Signal(
            signal_id=_deterministic_signal_id(
                context.run_id,
                evaluation.outcome,
                evaluation.detector_version,
            ),
            run_id=context.run_id,
            created_at=context.current.timestamp,
            signal_type=evaluation.signal_type,
            strength=evaluation.outcome.strength,
            evidence_event_ids=evaluation.outcome.evidence_event_ids,
            detector_version=evaluation.detector_version,
            reason_code=evaluation.outcome.reason_code,
        )
        for evaluation in evaluations
        if evaluation.outcome.status is DetectionStatus.DETECTED
        and evaluation.outcome.strength is not None
        and evaluation.outcome.reason_code is not None
    )
    return ExtractionReport(
        run_id=context.run_id,
        current_event_id=context.current.event_id,
        current_event_timestamp=context.current.timestamp,
        evaluations=evaluations,
        signals=tuple(sorted(signals, key=lambda signal: signal.signal_type.value)),
    )


@dataclass(frozen=True, slots=True, init=False, repr=False)
class DeterministicSignalExtractor:
    """Validate detector outcomes and materialize replay-stable domain signals."""

    _entries: tuple[_DetectorEntry, ...]
    _factory_token: object = field(repr=False, compare=False)
    _seal: _TrustedSeal = field(repr=False, compare=False)

    def __init__(self, detectors: tuple[SignalDetector, ...]) -> None:
        if type(detectors) is not tuple or len(detectors) > len(SignalType):
            raise DetectorContractError()
        entries: list[_DetectorEntry] = []
        for detector in detectors:
            detector_entry: _DetectorEntry | None = None
            try:
                public_evaluator = getattr(detector, "evaluate", None)
                if callable(public_evaluator):
                    if isinstance(detector, _ValidatedSignalDetector):
                        accepts_trusted_context = True
                        evaluator = detector._evaluate_validated
                    else:
                        accepts_trusted_context = False
                        evaluator = public_evaluator
                    signal_type = detector.signal_type
                    detector_version = detector.detector_version
                    if (
                        type(signal_type) is SignalType
                        and type(detector_version) is str
                        and _DETECTOR_VERSION.fullmatch(detector_version) is not None
                    ):
                        detector_entry = _DetectorEntry(
                            evaluator=evaluator,
                            signal_type=signal_type,
                            detector_version=detector_version,
                            accepts_trusted_context=accepts_trusted_context,
                        )
            except Exception:
                detector_entry = None
            if detector_entry is None:
                raise DetectorContractError()
            entries.append(detector_entry)
        signal_types = tuple(entry.signal_type for entry in entries)
        if len(set(signal_types)) != len(signal_types):
            raise DetectorContractError()
        object.__setattr__(
            self,
            "_entries",
            tuple(
                sorted(
                    entries,
                    key=lambda entry: (entry.signal_type.value, entry.detector_version),
                )
            ),
        )
        object.__setattr__(self, "_factory_token", _EXTRACTOR_TOKEN)
        seal = _TrustedSeal()
        object.__setattr__(self, "_seal", seal)
        _TRUSTED_EXTRACTOR_SEALS[seal] = (self._entries,)

    def _evaluate_validated(
        self,
        validated: DetectionContext,
    ) -> tuple[DetectorEvaluation, ...]:
        known_ids = {event.event_id for event in validated.events}
        order = {event.event_id: index for index, event in enumerate(validated.events)}
        evaluations: list[DetectorEvaluation] = []
        for entry in self._entries:
            outcome: DetectionOutcome | None = None
            try:
                candidate = entry.evaluator(validated)
                if _outcome_is_preflight_safe(candidate):
                    assert type(candidate) is DetectionOutcome
                    parsed = DetectionOutcome.model_validate(
                        candidate.model_dump(mode="python", warnings=False)
                    )
                    if parsed == candidate:
                        outcome = parsed
            except Exception:
                outcome = None
            if outcome is None:
                raise DetectorContractError()
            if outcome.signal_type is not entry.signal_type:
                raise DetectorContractError()
            if outcome.status is DetectionStatus.DETECTED and (
                validated.current.event_id not in outcome.evidence_event_ids
                or any(event_id not in known_ids for event_id in outcome.evidence_event_ids)
                or tuple(sorted(outcome.evidence_event_ids, key=order.__getitem__))
                != outcome.evidence_event_ids
            ):
                raise DetectorContractError()
            if outcome.status is not DetectionStatus.DETECTED and (
                validated.current.event_id not in outcome.related_event_ids
                or any(event_id not in known_ids for event_id in outcome.related_event_ids)
                or tuple(sorted(outcome.related_event_ids, key=order.__getitem__))
                != outcome.related_event_ids
            ):
                raise DetectorContractError()
            evaluations.append(
                DetectorEvaluation(
                    signal_type=entry.signal_type,
                    detector_version=entry.detector_version,
                    outcome=outcome,
                )
            )
        return tuple(evaluations)

    def evaluate(self, context: DetectionContext) -> tuple[DetectorEvaluation, ...]:
        """Return every version-attributed outcome, including abstentions."""

        return self._evaluate_validated(validate_detection_context(context))

    def extract(self, context: DetectionContext) -> tuple[Signal, ...]:
        """Materialize only detected signals; use ``extract_report`` for audit data."""

        return self.extract_report(context).signals

    def extract_report(self, context: DetectionContext) -> ExtractionReport:
        """Evaluate once and retain every detector outcome beside emitted signals."""

        validated = validate_detection_context(context)
        evaluations = self._evaluate_validated(validated)
        return _materialize_extraction_report(validated, evaluations)


def _trusted_extractor_is_exact(value: object) -> bool:
    try:
        if type(value) is not DeterministicSignalExtractor or type(value._seal) is not _TrustedSeal:
            return False
        sealed = _TRUSTED_EXTRACTOR_SEALS.get(value._seal)
        return (
            sealed is not None
            and len(sealed) == 1
            and value._entries is sealed[0]
            and value._factory_token is _EXTRACTOR_TOKEN
            and type(value._entries) is tuple
            and all(
                type(entry) is _DetectorEntry
                and entry.accepts_trusted_context is True
                and callable(entry.evaluator)
                and type(entry.signal_type) is SignalType
                and type(entry.detector_version) is str
                and _DETECTOR_VERSION.fullmatch(entry.detector_version) is not None
                for entry in value._entries
            )
        )
    except Exception:
        return False


def _trusted_extraction_is_exact(value: object) -> bool:
    try:
        if type(value) is not _TrustedExtraction or type(value._seal) is not _TrustedSeal:
            return False
        sealed = _TRUSTED_EXTRACTION_SEALS.get(value._seal)
        context = value.trusted_context.context
        report = value.report
        return (
            sealed is not None
            and len(sealed) == 2
            and value.trusted_context is sealed[0]
            and value.report is sealed[1]
            and value._token is _TRUSTED_EXTRACTION_TOKEN
            and _trusted_detection_context_is_exact(value.trusted_context)
            and type(report) is ExtractionReport
            and report.run_id == context.run_id
            and report.current_event_id == context.current.event_id
            and report.current_event_timestamp == context.current.timestamp
        )
    except Exception:
        return False


def _extract_trusted_report(
    extractor: DeterministicSignalExtractor,
    trusted_context: _TrustedDetectionContext,
) -> _TrustedExtraction:
    result: _TrustedExtraction | None = None
    try:
        if not _trusted_extractor_is_exact(extractor) or not _trusted_detection_context_is_exact(
            trusted_context
        ):
            raise ValueError("trusted extraction admission is invalid")
        context = trusted_context.context
        evaluations = extractor._evaluate_validated(context)
        report = _materialize_extraction_report(context, evaluations)
        seal = _TrustedSeal()
        result = _TrustedExtraction(
            trusted_context=trusted_context,
            report=report,
            _token=_TRUSTED_EXTRACTION_TOKEN,
            _seal=seal,
        )
        _TRUSTED_EXTRACTION_SEALS[seal] = (
            result.trusted_context,
            result.report,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        result = None
    if result is None or not _trusted_extraction_is_exact(result):
        raise DetectorContractError()
    return result


__all__ = [
    "AbstentionReason",
    "DetectionContext",
    "DetectionInputError",
    "DetectionOutcome",
    "DetectionStatus",
    "DetectorContractError",
    "DetectorEvaluation",
    "DeterministicSignalExtractor",
    "ExtractionReport",
    "SignalDetector",
    "validate_detection_context",
]
