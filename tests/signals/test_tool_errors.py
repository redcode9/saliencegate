from __future__ import annotations

from collections.abc import Callable
from typing import cast
from uuid import UUID

import pytest

from saliencegate.domain import EventType, ReasonCode, SignalType, TraceEvent
from saliencegate.signals.base import (
    AbstentionReason,
    DetectionContext,
    DetectionInputError,
    DetectionStatus,
    DeterministicSignalExtractor,
)
from saliencegate.signals.tool_errors import ToolErrorDetector

EventFactory = Callable[..., TraceEvent]
RUN_ID = UUID("00000000-0000-4000-8000-000000002001")


def context(event: TraceEvent) -> DetectionContext:
    return DetectionContext(run_id=event.run_id, events=(event,))


def test_controller_error_is_detected_without_payload(event_factory: EventFactory) -> None:
    event = event_factory(1, event_type=EventType.CONTROLLER_ERROR)

    outcome = ToolErrorDetector().evaluate(context(event))

    assert outcome.status is DetectionStatus.DETECTED
    assert outcome.signal_type is SignalType.TOOL_ERROR
    assert outcome.strength == 1.0
    assert outcome.reason_code is ReasonCode.TOOL_ERROR
    assert outcome.evidence_event_ids == (event.event_id,)
    assert outcome.abstention_reason is None


@pytest.mark.parametrize(
    "tool_outcome",
    [
        {"status": "failed"},
        {"status": "failed", "exit_status": 7},
        {"status": "failed", "exception_type": "TimeoutError"},
        {"status": "failed", "error_code": "EPIPE"},
        {"status": "failed", "failure_signature": "connection closed"},
        {"exit_status": 7},
        {"exception_type": "TimeoutError"},
        {"error_code": "EPIPE"},
        {"failure_signature": "connection closed"},
    ],
)
def test_structured_tool_failure_is_detected(
    event_factory: EventFactory,
    tool_outcome: dict[str, object],
) -> None:
    event = event_factory(
        1,
        event_type=EventType.TOOL_COMPLETION,
        payload={"tool_outcome": {"schema_version": "1.0", **tool_outcome}},
    )

    outcome = ToolErrorDetector().evaluate(context(event))

    assert outcome.status is DetectionStatus.DETECTED
    assert outcome.evidence_event_ids == (event.event_id,)


@pytest.mark.parametrize(
    "tool_outcome",
    [
        {"status": "succeeded"},
        {"status": "succeeded", "exit_status": 0},
        {"exit_status": 0},
    ],
)
def test_consistent_structured_success_is_no_match(
    event_factory: EventFactory,
    tool_outcome: dict[str, object],
) -> None:
    event = event_factory(
        1,
        event_type=EventType.TOOL_COMPLETION,
        payload={"tool_outcome": {"schema_version": "1.0", **tool_outcome}},
    )

    outcome = ToolErrorDetector().evaluate(context(event))

    assert outcome.status is DetectionStatus.NO_MATCH
    assert outcome.strength is None
    assert outcome.evidence_event_ids == ()
    assert outcome.abstention_reason is None


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({}, AbstentionReason.STRUCTURED_EVIDENCE_MISSING),
        (
            {"tool_outcome": {"schema_version": "1.0"}},
            AbstentionReason.STRUCTURED_EVIDENCE_INVALID,
        ),
        (
            {
                "tool_outcome": {
                    "schema_version": "1.0",
                    "status": "succeeded",
                    "exit_status": 1,
                }
            },
            AbstentionReason.STRUCTURED_EVIDENCE_INVALID,
        ),
        (
            {
                "tool_outcome": {
                    "schema_version": "1.0",
                    "status": "succeeded",
                    "exception_type": "TimeoutError",
                }
            },
            AbstentionReason.STRUCTURED_EVIDENCE_INVALID,
        ),
        (
            {
                "tool_outcome": {
                    "schema_version": "1.0",
                    "status": "failed",
                    "error_code": "[REDACTED]",
                }
            },
            None,
        ),
    ],
)
def test_tool_evidence_uses_minimal_failure_classification(
    event_factory: EventFactory,
    payload: dict[str, object],
    reason: AbstentionReason | None,
) -> None:
    event = event_factory(1, event_type=EventType.TOOL_COMPLETION, payload=payload)

    outcome = ToolErrorDetector().evaluate(context(event))

    if reason is None:
        assert outcome.status is DetectionStatus.DETECTED
        assert outcome.evidence_event_ids == (event.event_id,)
    else:
        assert outcome.status is DetectionStatus.ABSTAINED
        assert outcome.abstention_reason is reason
        assert outcome.evidence_event_ids == ()


def test_failed_status_is_not_suppressed_by_redacted_optional_details(
    event_factory: EventFactory,
) -> None:
    event = event_factory(
        1,
        event_type=EventType.TOOL_COMPLETION,
        payload={
            "tool_outcome": {
                "schema_version": "1.0",
                "status": "failed",
                "exit_status": 2,
                "failure_signature": "[REDACTED]",
            }
        },
    )

    outcome = ToolErrorDetector().evaluate(context(event))

    assert outcome.status is DetectionStatus.DETECTED
    assert outcome.evidence_event_ids == (event.event_id,)


def test_unrelated_event_is_not_applicable(event_factory: EventFactory) -> None:
    event = event_factory(
        1,
        event_type=EventType.OBSERVATION,
        payload={"tool_outcome": {"schema_version": "1.0", "status": "failed"}},
    )

    outcome = ToolErrorDetector().evaluate(context(event))

    assert outcome.status is DetectionStatus.ABSTAINED
    assert outcome.abstention_reason is AbstentionReason.EVENT_NOT_APPLICABLE


def test_detector_revalidates_its_exact_context_boundary(event_factory: EventFactory) -> None:
    event = event_factory(1, event_type=EventType.CONTROLLER_ERROR)

    with pytest.raises(DetectionInputError, match="input failed validation"):
        ToolErrorDetector().evaluate(cast(DetectionContext, {"run_id": RUN_ID, "events": (event,)}))


def test_signal_extractor_materializes_a_replay_stable_signal(
    event_factory: EventFactory,
) -> None:
    event = event_factory(
        1,
        event_type=EventType.TOOL_COMPLETION,
        payload={
            "tool_outcome": {
                "schema_version": "1.0",
                "status": "failed",
                "exit_status": 2,
            }
        },
    )
    detection_context = context(event)
    extractor = DeterministicSignalExtractor((ToolErrorDetector(),))

    first = extractor.extract(detection_context)
    second = extractor.extract(detection_context)

    assert first == second
    assert len(first) == 1
    signal = first[0]
    assert signal.signal_id.version == 4
    assert signal.run_id == event.run_id
    assert signal.created_at == event.timestamp
    assert signal.signal_type is SignalType.TOOL_ERROR
    assert signal.strength == 1.0
    assert signal.evidence_event_ids == (event.event_id,)
    assert signal.detector_version == "tool-error/v1"
    assert signal.reason_code is ReasonCode.TOOL_ERROR


def test_signal_extractor_emits_nothing_for_success_or_abstention(
    event_factory: EventFactory,
) -> None:
    succeeded = event_factory(
        1,
        event_type=EventType.TOOL_COMPLETION,
        payload={
            "tool_outcome": {
                "schema_version": "1.0",
                "status": "succeeded",
                "exit_status": 0,
            }
        },
    )
    unrelated = event_factory(2, event_type=EventType.OBSERVATION)
    extractor = DeterministicSignalExtractor((ToolErrorDetector(),))

    assert extractor.extract(context(succeeded)) == ()
    assert extractor.extract(context(unrelated)) == ()
