from __future__ import annotations

import pytest

from saliencegate.domain import EventType, ReasonCode, SignalType
from saliencegate.signals.base import (
    AbstentionReason,
    DetectionContext,
    DetectionStatus,
    DeterministicSignalExtractor,
)
from saliencegate.signals.test_failures import TestFailureDetector


def report(
    status: str,
    *,
    failures: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "test_report": {
            "schema_version": "1.0",
            "framework": "pytest",
            "status": status,
            "failures": (
                []
                if failures is None
                else [{"schema_version": "1.0", **failure} for failure in failures]
            ),
        }
    }


def test_failed_report_emits_one_evidence_backed_signal(event_factory: object) -> None:
    event = event_factory(
        1,
        event_type=EventType.TOOL_COMPLETION,
        payload=report(
            "failed",
            failures=[
                {
                    "test_id": "./tests/test_api.py::test_timeout",
                    "failure_type": "AssertionError",
                    "signature": "expected 200, got 504",
                }
            ],
        ),
    )
    context = DetectionContext(run_id=event.run_id, events=(event,))
    detector = TestFailureDetector()

    outcome = detector.evaluate(context)
    first = DeterministicSignalExtractor((detector,)).extract(context)
    second = DeterministicSignalExtractor((detector,)).extract(context)

    assert outcome.status is DetectionStatus.DETECTED
    assert outcome.evidence_event_ids == (event.event_id,)
    assert outcome.strength == 1.0
    assert first == second
    assert len(first) == 1
    assert first[0].signal_id.version == 4
    assert first[0].created_at == event.timestamp
    assert first[0].signal_type is SignalType.TEST_FAILURE
    assert first[0].reason_code is ReasonCode.TEST_FAILURE
    assert first[0].detector_version == "test-failure/v1"


def test_passing_report_is_an_explicit_no_match(event_factory: object) -> None:
    event = event_factory(
        1,
        event_type=EventType.OBSERVATION,
        payload=report("passed"),
    )
    outcome = TestFailureDetector().evaluate(DetectionContext(run_id=event.run_id, events=(event,)))

    assert outcome.status is DetectionStatus.NO_MATCH
    assert (
        DeterministicSignalExtractor((TestFailureDetector(),)).extract(
            DetectionContext(run_id=event.run_id, events=(event,))
        )
        == ()
    )


@pytest.mark.parametrize(
    "failures",
    (
        [
            {
                "test_id": "tests/test_api.py::test_timeout",
                "failure_type": "[REDACTED]",
                "signature": "[REDACTED]",
            }
        ],
        [
            {"test_id": "tests/test_api.py::test_timeout"},
            {"test_id": "tests/test_api.py::test_timeout"},
        ],
    ),
)
def test_base_signal_ignores_optional_equivalence_details(
    event_factory: object,
    failures: list[dict[str, object]],
) -> None:
    event = event_factory(
        1,
        event_type=EventType.TOOL_COMPLETION,
        payload=report("failed", failures=failures),
    )

    outcome = TestFailureDetector().evaluate(DetectionContext(run_id=event.run_id, events=(event,)))

    assert outcome.status is DetectionStatus.DETECTED
    assert outcome.evidence_event_ids == (event.event_id,)


@pytest.mark.parametrize(
    ("payload", "reason"),
    (
        ({}, AbstentionReason.STRUCTURED_EVIDENCE_MISSING),
        (
            report("failed"),
            AbstentionReason.STRUCTURED_EVIDENCE_INVALID,
        ),
        (
            report("passed", failures=[{"test_id": "tests/a.py::test_a"}]),
            AbstentionReason.STRUCTURED_EVIDENCE_INVALID,
        ),
        (
            report(
                "failed",
                failures=[{"test_id": "tests/a.py::test_a", "unexpected": True}],
            ),
            AbstentionReason.STRUCTURED_EVIDENCE_INVALID,
        ),
    ),
)
def test_missing_or_inconsistent_test_evidence_abstains(
    event_factory: object,
    payload: dict[str, object],
    reason: AbstentionReason,
) -> None:
    event = event_factory(1, event_type=EventType.TOOL_COMPLETION, payload=payload)
    outcome = TestFailureDetector().evaluate(DetectionContext(run_id=event.run_id, events=(event,)))

    assert outcome.status is DetectionStatus.ABSTAINED
    assert outcome.abstention_reason is reason


def test_unrelated_event_type_abstains_without_reading_free_text(event_factory: object) -> None:
    event = event_factory(
        1,
        event_type=EventType.MODEL_OUTPUT,
        payload={"message": "FAILED tests/test_api.py::test_timeout"},
    )
    outcome = TestFailureDetector().evaluate(DetectionContext(run_id=event.run_id, events=(event,)))

    assert outcome.status is DetectionStatus.ABSTAINED
    assert outcome.abstention_reason is AbstentionReason.EVENT_NOT_APPLICABLE
