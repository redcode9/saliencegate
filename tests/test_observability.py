from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from io import StringIO
from threading import Event, Thread

import pytest

from saliencegate.domain import canonical_json
from saliencegate.observability import (
    OBSERVABILITY_SCHEMA_VERSION,
    LogLevel,
    Metric,
    MetricName,
    MetricUnit,
    ObservabilityError,
    ObservabilityEvent,
    StructuredLogger,
)
from saliencegate.security import Redactor

NOW = datetime(2026, 7, 12, 10, 30, 45, 123456, tzinfo=UTC)


def fixed_clock() -> datetime:
    return NOW


def test_emit_writes_one_canonical_json_line_with_stable_names_and_units() -> None:
    stream = StringIO()
    logger = StructuredLogger(stream=stream, clock=fixed_clock)

    logger.emit(
        ObservabilityEvent.BUDGET_SETTLED,
        level=LogLevel.INFO,
        attributes={"run_id": "00000000-0000-4000-8000-000000000001"},
        metrics=(
            Metric(
                MetricName.MODEL_CANONICAL_TOKEN_EQUIVALENTS,
                12,
                MetricUnit.TOKEN_EQUIVALENTS,
            ),
            Metric(MetricName.MODEL_LATENCY_US, 4500, MetricUnit.MICROSECONDS),
        ),
    )

    assert stream.getvalue() == (
        '{"attributes":{"run_id":"00000000-0000-4000-8000-000000000001"},'
        '"event_name":"saliencegate.budget.settled","level":"info","metrics":['
        '{"name":"model.canonical_token_equivalents","unit":"token_equivalents",'
        '"value":12},{"name":"model.latency_us","unit":"microseconds","value":4500}],'
        f'"schema_version":"{OBSERVABILITY_SCHEMA_VERSION}",'
        '"timestamp":"2026-07-12T10:30:45.123456Z"}\n'
    )


def test_emit_is_deterministic_for_metric_and_attribute_input_order() -> None:
    first = StringIO()
    second = StringIO()
    first_logger = StructuredLogger(stream=first, clock=fixed_clock)
    second_logger = StructuredLogger(stream=second, clock=fixed_clock)

    first_logger.emit(
        ObservabilityEvent.RUN_FINISHED,
        attributes={"z": 2, "a": 1},
        metrics=(
            Metric(MetricName.EVENTS, 8, MetricUnit.COUNT),
            Metric(MetricName.RUN_BYTES, 256, MetricUnit.BYTES),
        ),
    )
    second_logger.emit(
        ObservabilityEvent.RUN_FINISHED,
        attributes={"a": 1, "z": 2},
        metrics=(
            Metric(MetricName.RUN_BYTES, 256, MetricUnit.BYTES),
            Metric(MetricName.EVENTS, 8, MetricUnit.COUNT),
        ),
    )

    assert first.getvalue() == second.getvalue()


def test_redaction_happens_before_canonical_formatting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "ordinary-fixture-secret"
    stream = StringIO()
    logger = StructuredLogger(
        stream=stream,
        clock=fixed_clock,
        redactor=Redactor(literal_secrets=(secret,)),
    )
    real_canonical_json = canonical_json
    formatted_inputs: list[object] = []

    def observe_canonical_input(value: object) -> bytes:
        formatted_inputs.append(value)
        return real_canonical_json(value)

    monkeypatch.setattr("saliencegate.observability.canonical_json", observe_canonical_input)

    logger.emit(
        ObservabilityEvent.MODEL_FINISHED,
        attributes={
            "message": f"model returned {secret}",
            "headers": {"X-API-Key": "fixture-api-key"},
            "authorization_note": "Bearer fixture_token_1234567890",
        },
    )

    assert len(formatted_inputs) == 1
    formatted = real_canonical_json(formatted_inputs[0]).decode("utf-8")
    output = stream.getvalue()
    for forbidden in (secret, "fixture-api-key", "fixture_token_1234567890"):
        assert forbidden not in formatted
        assert forbidden not in output
    assert output.count("[REDACTED]") == 3


def test_secret_in_attribute_key_fails_closed_without_writing() -> None:
    secret = "sk-proj-fixture1234567890abcdef"
    stream = StringIO()
    logger = StructuredLogger(
        stream=stream,
        clock=fixed_clock,
        redactor=Redactor(literal_secrets=(secret,)),
    )

    with pytest.raises(ObservabilityError, match="redaction"):
        logger.emit(
            ObservabilityEvent.RUN_STARTED,
            attributes={f"prefix-{secret}": "value"},
        )

    assert stream.getvalue() == ""


@pytest.mark.parametrize(
    ("name", "value", "unit", "message"),
    [
        ("Run Duration", 1, MetricUnit.MICROSECONDS, "metric name"),
        (MetricName.MODEL_LATENCY_US, True, MetricUnit.MICROSECONDS, "metric value"),
        (MetricName.MODEL_LATENCY_US, "1", MetricUnit.MICROSECONDS, "metric value"),
        (MetricName.MODEL_LATENCY_US, 1, "microseconds", "metric unit"),
        (MetricName.MODEL_LATENCY_US, 1, MetricUnit.MILLISECONDS, "stable metric"),
    ],
)
def test_metric_rejects_noncanonical_fields(
    name: object,
    value: object,
    unit: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        Metric(name, value, unit)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_metric_rejects_non_finite_value_at_construction(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        Metric(MetricName.MODEL_LATENCY_US, value, MetricUnit.MICROSECONDS)


def test_emit_rejects_duplicate_metric_names_before_writing() -> None:
    stream = StringIO()
    logger = StructuredLogger(stream=stream, clock=fixed_clock)

    with pytest.raises(ValueError, match="unique"):
        logger.emit(
            ObservabilityEvent.RUN_FINISHED,
            metrics=(
                Metric(MetricName.MODEL_LATENCY_US, 2, MetricUnit.MICROSECONDS),
                Metric(MetricName.MODEL_LATENCY_US, 3, MetricUnit.MICROSECONDS),
            ),
        )

    assert stream.getvalue() == ""


@pytest.mark.parametrize(
    ("event", "level", "message"),
    [
        ("saliencegate.run.started", LogLevel.INFO, "event"),
        (ObservabilityEvent.RUN_STARTED, "info", "level"),
    ],
)
def test_emit_rejects_unversioned_event_or_level_values(
    event: object,
    level: object,
    message: str,
) -> None:
    stream = StringIO()
    logger = StructuredLogger(stream=stream, clock=fixed_clock)

    with pytest.raises(TypeError, match=message):
        logger.emit(event, level=level)  # type: ignore[arg-type]

    assert stream.getvalue() == ""


def test_emit_rejects_non_mapping_attributes_and_non_metric_items() -> None:
    stream = StringIO()
    logger = StructuredLogger(stream=stream, clock=fixed_clock)

    with pytest.raises(TypeError, match="attributes"):
        logger.emit(ObservabilityEvent.RUN_STARTED, attributes=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Metric"):
        logger.emit(ObservabilityEvent.RUN_STARTED, metrics=(object(),))  # type: ignore[arg-type]

    assert stream.getvalue() == ""


@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 7, 12, 10, 30),
        datetime(2026, 7, 12, 12, 30, tzinfo=timezone(timedelta(hours=2))),
    ],
)
def test_emit_requires_an_aware_utc_clock(now: datetime) -> None:
    stream = StringIO()
    logger = StructuredLogger(stream=stream, clock=lambda: now)

    with pytest.raises(ValueError, match="UTC"):
        logger.emit(ObservabilityEvent.RUN_STARTED)

    assert stream.getvalue() == ""


def test_emit_rejects_a_clock_returning_the_wrong_type() -> None:
    stream = StringIO()

    def wrong_clock() -> datetime:
        return "now"  # type: ignore[return-value]

    logger = StructuredLogger(stream=stream, clock=wrong_clock)

    with pytest.raises(TypeError, match="datetime"):
        logger.emit(ObservabilityEvent.RUN_STARTED)

    assert stream.getvalue() == ""


def test_default_stream_is_resolved_at_emit_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = StringIO()
    logger = StructuredLogger(clock=fixed_clock)
    monkeypatch.setattr(sys, "stderr", stream)

    logger.emit(ObservabilityEvent.RUN_STARTED)

    assert '"event_name":"saliencegate.run.started"' in stream.getvalue()


def test_default_clock_emits_utc_and_flush_can_be_disabled() -> None:
    stream = StringIO()
    logger = StructuredLogger(stream=stream, flush=False)

    logger.emit(ObservabilityEvent.RUN_STARTED)

    timestamp = json.loads(stream.getvalue())["timestamp"]
    assert timestamp.endswith("Z")
    assert datetime.fromisoformat(timestamp.replace("Z", "+00:00")).tzinfo is not None


def test_constructor_rejects_invalid_collaborators() -> None:
    with pytest.raises(TypeError, match="redactor"):
        StructuredLogger(redactor=object())  # type: ignore[arg-type]
    with pytest.raises(ObservabilityError, match="stream"):
        StructuredLogger(stream=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="clock"):
        StructuredLogger(clock=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="flush"):
        StructuredLogger(flush=1)  # type: ignore[arg-type]


def test_logger_repr_does_not_expose_redactor_literals() -> None:
    secret = "ordinary-fixture-secret"
    logger = StructuredLogger(redactor=Redactor(literal_secrets=(secret,)))

    assert secret not in repr(logger)
    assert "StructuredLogger" in repr(logger)


def test_microseconds_are_an_explicit_runtime_metric_unit() -> None:
    metric = Metric(MetricName.MODEL_LATENCY_US, 125, MetricUnit.MICROSECONDS)

    assert metric.as_json()["unit"] == "microseconds"


class _ExplodingMapping(Mapping[str, object]):
    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("fixture-secret-from-mapping")

    def __len__(self) -> int:
        raise RuntimeError("fixture-secret-from-mapping")

    def __getitem__(self, key: str) -> object:
        raise RuntimeError("fixture-secret-from-mapping")


def test_hostile_mapping_is_rejected_without_iteration_or_secret_leakage() -> None:
    stream = StringIO()
    logger = StructuredLogger(stream=stream, clock=fixed_clock)

    with pytest.raises(TypeError) as error:
        logger.emit(
            ObservabilityEvent.RUN_STARTED,
            attributes=_ExplodingMapping(),  # type: ignore[arg-type]
        )

    assert "fixture-secret" not in str(error.value)
    assert stream.getvalue() == ""


def test_hostile_metric_iterator_and_clock_errors_are_sanitized() -> None:
    stream = StringIO()
    logger = StructuredLogger(stream=stream, clock=fixed_clock)

    def metrics() -> Iterator[Metric]:
        raise ValueError("fixture-secret-from-iterator")
        yield Metric(MetricName.EVENTS, 1, MetricUnit.COUNT)

    with pytest.raises(ObservabilityError) as iterator_error:
        logger.emit(ObservabilityEvent.RUN_STARTED, metrics=metrics())
    assert "fixture-secret" not in str(iterator_error.value)
    assert iterator_error.value.__context__ is None
    assert iterator_error.value.__cause__ is None

    def bad_clock() -> datetime:
        raise TypeError("fixture-secret-from-clock")

    logger = StructuredLogger(stream=stream, clock=bad_clock)
    with pytest.raises(ObservabilityError) as clock_error:
        logger.emit(ObservabilityEvent.RUN_STARTED)
    assert "fixture-secret" not in str(clock_error.value)
    assert clock_error.value.__context__ is None
    assert clock_error.value.__cause__ is None
    assert stream.getvalue() == ""


@pytest.mark.parametrize(
    "attributes",
    [
        {"value": float("nan")},
        {1: "non-string key"},
        {"value": object()},
        {"value": 1 << 64},
        {"value": "unpaired-surrogate-\ud800"},
    ],
)
def test_attributes_accept_only_bounded_exact_json(attributes: dict[object, object]) -> None:
    stream = StringIO()
    logger = StructuredLogger(stream=stream, clock=fixed_clock)

    with pytest.raises(ObservabilityError):
        logger.emit(
            ObservabilityEvent.RUN_STARTED,
            attributes=attributes,  # type: ignore[arg-type]
        )

    assert stream.getvalue() == ""


def test_attribute_depth_and_encoded_size_are_bounded() -> None:
    stream = StringIO()
    logger = StructuredLogger(stream=stream, clock=fixed_clock)
    nested: object = "leaf"
    for _ in range(34):
        nested = [nested]

    with pytest.raises(ObservabilityError, match="structural bound"):
        logger.emit(ObservabilityEvent.RUN_STARTED, attributes={"nested": nested})
    with pytest.raises(ObservabilityError, match="byte bound"):
        logger.emit(
            ObservabilityEvent.RUN_STARTED,
            attributes={"message": "x" * (1024 * 1024)},
        )

    assert stream.getvalue() == ""


class _FailingStream(StringIO):
    def write(self, value: str) -> int:
        del value
        raise RuntimeError("fixture-secret-from-stream")


def test_formatting_and_stream_failures_are_value_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = StringIO()
    logger = StructuredLogger(stream=stream, clock=fixed_clock)

    def fail_format(value: object) -> bytes:
        del value
        raise RuntimeError("fixture-secret-from-formatter")

    monkeypatch.setattr("saliencegate.observability.canonical_json", fail_format)
    with pytest.raises(ObservabilityError) as formatting_error:
        logger.emit(ObservabilityEvent.RUN_STARTED)
    assert "fixture-secret" not in str(formatting_error.value)
    assert formatting_error.value.__context__ is None
    assert formatting_error.value.__cause__ is None

    monkeypatch.setattr("saliencegate.observability.canonical_json", canonical_json)
    logger = StructuredLogger(stream=_FailingStream(), clock=fixed_clock)
    with pytest.raises(ObservabilityError) as stream_error:
        logger.emit(ObservabilityEvent.RUN_STARTED)
    assert "fixture-secret" not in str(stream_error.value)
    assert stream_error.value.__context__ is None
    assert stream_error.value.__cause__ is None
    assert stream.getvalue() == ""


class _HostileTimezone(tzinfo):
    def utcoffset(self, value: datetime | None) -> timedelta:
        del value
        raise ValueError("fixture-secret-from-timezone")

    def dst(self, value: datetime | None) -> timedelta:
        del value
        return timedelta(0)

    def tzname(self, value: datetime | None) -> str:
        del value
        return "hostile"


def test_timezone_callbacks_are_never_invoked() -> None:
    stream = StringIO()
    hostile = datetime(2026, 7, 12, tzinfo=_HostileTimezone())
    logger = StructuredLogger(stream=stream, clock=lambda: hostile)

    with pytest.raises(ValueError) as error:
        logger.emit(ObservabilityEvent.RUN_STARTED)

    assert "fixture-secret" not in str(error.value)
    assert error.value.__context__ is None
    assert stream.getvalue() == ""


class _HostileStreamProbe:
    def __getattribute__(self, name: str) -> object:
        del name
        raise ValueError("fixture-secret-from-stream-probe")


def test_constructor_stream_probe_failure_is_value_free() -> None:
    with pytest.raises(ObservabilityError) as error:
        StructuredLogger(stream=_HostileStreamProbe())  # type: ignore[arg-type]

    assert "fixture-secret" not in str(error.value)
    assert error.value.__context__ is None
    assert error.value.__cause__ is None


class _ShortWriteStream(StringIO):
    def write(self, value: str) -> int:
        del value
        return 0


def test_short_write_is_rejected() -> None:
    logger = StructuredLogger(stream=_ShortWriteStream(), clock=fixed_clock)

    with pytest.raises(ObservabilityError, match="incomplete write"):
        logger.emit(ObservabilityEvent.RUN_STARTED)


class _ReentrantStream(StringIO):
    logger: StructuredLogger | None = None

    def write(self, value: str) -> int:
        assert self.logger is not None
        self.logger.emit(ObservabilityEvent.RUN_FINISHED)
        return super().write(value)


def test_reentrant_stream_fails_without_deadlock_or_exception_context() -> None:
    stream = _ReentrantStream()
    logger = StructuredLogger(stream=stream, clock=fixed_clock)
    stream.logger = logger

    with pytest.raises(ObservabilityError) as error:
        logger.emit(ObservabilityEvent.RUN_STARTED)

    assert error.value.__context__ is None
    assert error.value.__cause__ is None
    assert stream.getvalue() == ""


def test_metric_integer_bound_and_name_unit_registry_are_closed() -> None:
    with pytest.raises(ValueError, match="signed 64-bit"):
        Metric(MetricName.EVENTS, 1 << 64, MetricUnit.COUNT)
    with pytest.raises(ValueError, match="stable metric"):
        Metric(MetricName.EVENTS, 1, MetricUnit.BYTES)


class _BlockingStream(StringIO):
    def __init__(self, started: Event, release: Event) -> None:
        super().__init__()
        self.started = started
        self.release = release

    def write(self, value: str) -> int:
        self.started.set()
        if not self.release.wait(timeout=2):
            raise RuntimeError("blocking stream timed out")
        return super().write(value)


def test_blocked_destination_does_not_block_an_independent_logger() -> None:
    started = Event()
    release = Event()
    first_stream = _BlockingStream(started, release)
    second_stream = StringIO()
    first = StructuredLogger(stream=first_stream, clock=fixed_clock)
    second = StructuredLogger(stream=second_stream, clock=fixed_clock)
    second_finished = Event()

    first_thread = Thread(target=lambda: first.emit(ObservabilityEvent.RUN_STARTED))

    def emit_second() -> None:
        second.emit(ObservabilityEvent.RUN_FINISHED)
        second_finished.set()

    second_thread = Thread(target=emit_second)
    first_thread.start()
    assert started.wait(timeout=1)
    second_thread.start()
    assert second_finished.wait(timeout=1)
    release.set()
    first_thread.join(timeout=1)
    second_thread.join(timeout=1)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert '"event_name":"saliencegate.run.finished"' in second_stream.getvalue()
