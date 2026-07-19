from __future__ import annotations

import math
import re
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from threading import Lock, RLock
from typing import Final, TextIO, cast

from saliencegate.domain import canonical_json
from saliencegate.security import Redactor

OBSERVABILITY_SCHEMA_VERSION: Final = "observability-event/v1"

_METRIC_NAME = re.compile(r"^[a-z][a-z0-9]*(?:[._][a-z0-9]+)*$")
_MAX_ATTRIBUTES_DEPTH = 32
_MAX_ATTRIBUTE_ITEMS = 10_000
_MAX_LOG_BYTES = 1024 * 1024
_MAX_METRICS = 256
_DESTINATION_REGISTRY_LOCK = Lock()


@dataclass(slots=True)
class _DestinationState:
    lock: RLock = field(default_factory=RLock)
    active: bool = False
    users: int = 0


_DESTINATION_STATES: dict[int, _DestinationState] = {}


class ObservabilityError(RuntimeError):
    """A value-free failure at the structured-observability boundary."""


class ObservabilityEvent(StrEnum):
    """Versioned event names forming the public structured-log vocabulary."""

    RUN_STARTED = "saliencegate.run.started"
    RUN_FINISHED = "saliencegate.run.finished"
    TRACE_EVENT_RECORDED = "saliencegate.trace_event.recorded"
    SIGNAL_DETECTED = "saliencegate.signal.detected"
    INVOCATION_DECIDED = "saliencegate.invocation.decided"
    CYCLE_STARTED = "saliencegate.cycle.started"
    CYCLE_FINISHED = "saliencegate.cycle.finished"
    MODEL_STARTED = "saliencegate.model.started"
    MODEL_FINISHED = "saliencegate.model.finished"
    BUDGET_RESERVED = "saliencegate.budget.reserved"
    BUDGET_SETTLED = "saliencegate.budget.settled"
    MEMORY_REVISION_COMMITTED = "saliencegate.memory_revision.committed"
    INTERVENTION_ACCEPTED = "saliencegate.intervention.accepted"
    INTERVENTION_REJECTED = "saliencegate.intervention.rejected"
    DELIVERY_UPDATED = "saliencegate.delivery.updated"
    OUTCOME_RECORDED = "saliencegate.outcome.recorded"


class LogLevel(StrEnum):
    """Stable severity labels for structured events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class MetricUnit(StrEnum):
    """Stable, unambiguous units allowed in a structured metric."""

    COUNT = "count"
    BYTES = "bytes"
    MICROSECONDS = "microseconds"
    MILLISECONDS = "milliseconds"
    SECONDS = "seconds"
    TOKENS = "tokens"
    TOKEN_EQUIVALENTS = "token_equivalents"
    RATIO = "ratio"
    SCORE = "score"


class MetricName(StrEnum):
    """Closed metric vocabulary; every name has exactly one canonical unit."""

    ARTIFACT_BYTES = "artifact.bytes"
    BUDGET_CANONICAL_TOKEN_EQUIVALENTS = "budget.canonical_token_equivalents"
    BUDGET_INPUT_TOKENS = "budget.input_tokens"
    BUDGET_LATENCY_US = "budget.latency_us"
    BUDGET_MODEL_CALLS = "budget.model_calls"
    BUDGET_OUTPUT_TOKENS = "budget.output_tokens"
    CYCLES = "run.cycles"
    DELIVERIES = "run.deliveries"
    EVENTS = "run.events"
    INTERVENTIONS = "run.interventions"
    MODEL_CALLS = "model.calls"
    MODEL_CANONICAL_TOKEN_EQUIVALENTS = "model.canonical_token_equivalents"
    MODEL_INPUT_TOKENS = "model.input_tokens"
    MODEL_LATENCY_US = "model.latency_us"
    MODEL_OUTPUT_TOKENS = "model.output_tokens"
    OUTCOMES = "run.outcomes"
    RUN_BYTES = "run.bytes"
    SIGNALS = "run.signals"


_METRIC_UNITS: dict[MetricName, MetricUnit] = {
    MetricName.ARTIFACT_BYTES: MetricUnit.BYTES,
    MetricName.BUDGET_CANONICAL_TOKEN_EQUIVALENTS: MetricUnit.TOKEN_EQUIVALENTS,
    MetricName.BUDGET_INPUT_TOKENS: MetricUnit.TOKENS,
    MetricName.BUDGET_LATENCY_US: MetricUnit.MICROSECONDS,
    MetricName.BUDGET_MODEL_CALLS: MetricUnit.COUNT,
    MetricName.BUDGET_OUTPUT_TOKENS: MetricUnit.TOKENS,
    MetricName.CYCLES: MetricUnit.COUNT,
    MetricName.DELIVERIES: MetricUnit.COUNT,
    MetricName.EVENTS: MetricUnit.COUNT,
    MetricName.INTERVENTIONS: MetricUnit.COUNT,
    MetricName.MODEL_CALLS: MetricUnit.COUNT,
    MetricName.MODEL_CANONICAL_TOKEN_EQUIVALENTS: MetricUnit.TOKEN_EQUIVALENTS,
    MetricName.MODEL_INPUT_TOKENS: MetricUnit.TOKENS,
    MetricName.MODEL_LATENCY_US: MetricUnit.MICROSECONDS,
    MetricName.MODEL_OUTPUT_TOKENS: MetricUnit.TOKENS,
    MetricName.OUTCOMES: MetricUnit.COUNT,
    MetricName.RUN_BYTES: MetricUnit.BYTES,
    MetricName.SIGNALS: MetricUnit.COUNT,
}


@dataclass(frozen=True, slots=True)
class Metric:
    """A finite numeric observation whose unit is explicit in the log record."""

    name: MetricName
    value: int | float
    unit: MetricUnit

    def __post_init__(self) -> None:
        if type(self.name) is not MetricName or _METRIC_NAME.fullmatch(self.name.value) is None:
            raise ValueError("metric name must be a registered stable metric")
        if type(self.value) not in (int, float):
            raise TypeError("metric value must be an int or float, excluding bool")
        if type(self.value) is int and self.value.bit_length() > 63:
            raise ValueError("metric integer exceeds the signed 64-bit bound")
        if type(self.value) is float and not math.isfinite(self.value):
            raise ValueError("metric value must be finite")
        if type(self.unit) is not MetricUnit:
            raise TypeError("metric unit must be a MetricUnit member")
        if self.unit is not _METRIC_UNITS[self.name]:
            raise ValueError("metric unit does not match the stable metric definition")

    def as_json(self) -> dict[str, str | int | float]:
        return {
            "name": self.name.value,
            "unit": self.unit.value,
            "value": self.value,
        }


def _system_utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_timestamp(value: datetime) -> str:
    if type(value) is not datetime:
        raise TypeError("observability clock must return a datetime")
    if value.tzinfo is not UTC:
        raise ValueError("observability timestamp must use the exact UTC timezone")
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _consume_text(value: str, byte_budget: list[int]) -> None:
    if len(value) > byte_budget[0]:
        raise ObservabilityError("observability attributes exceed their byte bound")
    encoding_failed = False
    encoded_length = 0
    try:
        encoded_length = len(value.encode("utf-8", errors="strict"))
    except Exception:
        encoding_failed = True
    if encoding_failed:
        raise ObservabilityError("observability attributes require valid UTF-8 text")
    byte_budget[0] -= encoded_length
    if byte_budget[0] < 0:
        raise ObservabilityError("observability attributes exceed their byte bound")


def _acquire_destination_state(destination: TextIO) -> tuple[int, _DestinationState]:
    destination_id = id(destination)
    with _DESTINATION_REGISTRY_LOCK:
        state = _DESTINATION_STATES.get(destination_id)
        if state is None:
            state = _DestinationState()
            _DESTINATION_STATES[destination_id] = state
        state.users += 1
    return destination_id, state


def _release_destination_state(destination_id: int, state: _DestinationState) -> None:
    with _DESTINATION_REGISTRY_LOCK:
        state.users -= 1
        if state.users == 0 and _DESTINATION_STATES.get(destination_id) is state:
            del _DESTINATION_STATES[destination_id]


def _copy_json(
    value: object,
    *,
    depth: int = 0,
    item_budget: list[int] | None = None,
    byte_budget: list[int] | None = None,
) -> object:
    if item_budget is None:
        item_budget = [_MAX_ATTRIBUTE_ITEMS]
    if byte_budget is None:
        byte_budget = [_MAX_LOG_BYTES]
    if depth > _MAX_ATTRIBUTES_DEPTH or item_budget[0] < 0:
        raise ObservabilityError("observability attributes exceed their structural bound")
    if value is None or type(value) is bool:
        return value
    if type(value) is str:
        _consume_text(value, byte_budget)
        return value
    if type(value) is int:
        if value.bit_length() > 63:
            raise ObservabilityError("observability attribute integer exceeds its bound")
        byte_budget[0] -= 8
        if byte_budget[0] < 0:
            raise ObservabilityError("observability attributes exceed their byte bound")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ObservabilityError("observability attributes require finite JSON numbers")
        byte_budget[0] -= 8
        if byte_budget[0] < 0:
            raise ObservabilityError("observability attributes exceed their byte bound")
        return value
    if type(value) is list or type(value) is tuple:
        item_budget[0] -= len(value)
        return [
            _copy_json(
                item,
                depth=depth + 1,
                item_budget=item_budget,
                byte_budget=byte_budget,
            )
            for item in value
        ]
    if type(value) is dict:
        item_budget[0] -= len(value)
        copied: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ObservabilityError("observability attribute keys must be exact strings")
            _consume_text(key, byte_budget)
            copied[key] = _copy_json(
                item,
                depth=depth + 1,
                item_budget=item_budget,
                byte_budget=byte_budget,
            )
        return copied
    raise ObservabilityError("observability attributes must be bounded exact JSON values")


class StructuredLogger:
    """Write redacted, deterministic JSON events to stderr or an injected stream.

    Each call performs one locked write, so concurrent callers cannot interleave JSON lines.
    Caller-provided data is kept under ``attributes`` and is redacted as part of the complete
    event before canonical JSON formatting.
    """

    __slots__ = ("_clock", "_flush", "_redactor", "_stream")

    def __init__(
        self,
        *,
        redactor: Redactor | None = None,
        stream: TextIO | None = None,
        clock: Callable[[], datetime] | None = None,
        flush: bool = True,
    ) -> None:
        if redactor is not None and type(redactor) is not Redactor:
            raise TypeError("redactor must be a Redactor")
        stream_probe_failed = False
        writer: object | None = None
        if stream is not None:
            try:
                writer = stream.write
            except Exception:
                stream_probe_failed = True
        if stream_probe_failed:
            raise ObservabilityError("observability stream failed validation")
        if stream is not None and not callable(writer):
            raise TypeError("stream must be a text stream")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        if type(flush) is not bool:
            raise TypeError("flush must be a bool")
        self._redactor = redactor if redactor is not None else Redactor()
        self._stream = stream
        self._clock = clock if clock is not None else _system_utc_now
        self._flush = flush

    def __repr__(self) -> str:
        destination = "stderr" if self._stream is None else "injected"
        return (
            f"StructuredLogger(redactor={self._redactor!r}, "
            f"stream={destination}, flush={self._flush!r})"
        )

    def emit(
        self,
        event: ObservabilityEvent,
        *,
        level: LogLevel = LogLevel.INFO,
        attributes: dict[str, object] | None = None,
        metrics: Iterable[Metric] = (),
    ) -> None:
        """Validate, redact, canonically format, and write one structured event."""

        if type(event) is not ObservabilityEvent:
            raise TypeError("event must be an ObservabilityEvent member")
        if type(level) is not LogLevel:
            raise TypeError("level must be a LogLevel member")
        if attributes is not None and type(attributes) is not dict:
            raise TypeError("attributes must be an exact dictionary")

        ordered_metrics: list[Metric] = []
        names: set[MetricName] = set()
        iterator_failed = False
        iterator: Iterator[Metric] | None = None
        try:
            iterator = iter(metrics)
        except Exception:
            iterator_failed = True
        if iterator_failed or iterator is None:
            raise ObservabilityError("observability metrics failed validation")
        while True:
            next_failed = False
            stopped = False
            metric: object | None = None
            try:
                metric = next(iterator)
            except StopIteration:
                stopped = True
            except Exception:
                next_failed = True
            if next_failed:
                raise ObservabilityError("observability metrics failed validation")
            if stopped:
                break
            if len(ordered_metrics) >= _MAX_METRICS:
                raise ObservabilityError("observability metric count exceeds its bound")
            if type(metric) is not Metric:
                raise TypeError("metrics must contain Metric values")
            if metric.name in names:
                raise ValueError("metric names must be unique")
            names.add(metric.name)
            ordered_metrics.append(metric)
        ordered_metrics.sort(key=lambda item: item.name.value)

        safe_attributes = {} if attributes is None else _copy_json(attributes)
        clock_failed = False
        clock_value: object | None = None
        try:
            clock_value = self._clock()
        except Exception:
            clock_failed = True
        if clock_failed:
            raise ObservabilityError("observability clock failed")
        timestamp = _utc_timestamp(cast(datetime, clock_value))

        record: dict[str, object] = {
            "attributes": safe_attributes,
            "event_name": event.value,
            "level": level.value,
            "metrics": tuple(metric.as_json() for metric in ordered_metrics),
            "schema_version": OBSERVABILITY_SCHEMA_VERSION,
            "timestamp": timestamp,
        }

        formatting_failed = False
        encoded_bytes = b""
        try:
            redacted = self._redactor.redact_payload(record)
            encoded_bytes = canonical_json(redacted.payload.root)
        except Exception:
            formatting_failed = True
        if formatting_failed:
            raise ObservabilityError("observability redaction or formatting failed")
        if len(encoded_bytes) > _MAX_LOG_BYTES:
            raise ObservabilityError("observability event exceeds its byte bound")
        encoded = encoded_bytes.decode("utf-8", errors="strict") + "\n"

        destination = sys.stderr if self._stream is None else self._stream
        destination_id, destination_state = _acquire_destination_state(destination)
        write_failed = False
        written: object | None = None
        reentrant = False
        try:
            with destination_state.lock:
                if destination_state.active:
                    reentrant = True
                else:
                    destination_state.active = True
                    try:
                        written = destination.write(encoded)
                        if self._flush:
                            destination.flush()
                    except Exception:
                        write_failed = True
                    finally:
                        destination_state.active = False
        finally:
            _release_destination_state(destination_id, destination_state)
        if reentrant:
            raise ObservabilityError("observability stream rejected reentrant emission")
        if write_failed:
            raise ObservabilityError("observability stream write failed")
        if type(written) is not int or written != len(encoded):
            raise ObservabilityError("observability stream performed an incomplete write")


__all__ = [
    "OBSERVABILITY_SCHEMA_VERSION",
    "LogLevel",
    "Metric",
    "MetricName",
    "MetricUnit",
    "ObservabilityError",
    "ObservabilityEvent",
    "StructuredLogger",
]
