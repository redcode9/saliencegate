from __future__ import annotations

from saliencegate.domain import EventType, SignalType
from saliencegate.signals.base import (
    AbstentionReason,
    DetectionContext,
    DetectionOutcome,
    _ValidatedSignalDetector,
    validate_detection_context,
)
from saliencegate.signals.fingerprints import (
    FingerprintUnavailableError,
    TestReportStatus,
    classify_test_report,
)

_DETECTOR_VERSION = "test-failure/v1"


class TestFailureDetector(_ValidatedSignalDetector):
    __slots__ = ()

    @property
    def signal_type(self) -> SignalType:
        return SignalType.TEST_FAILURE

    @property
    def detector_version(self) -> str:
        return _DETECTOR_VERSION

    def evaluate(self, context: DetectionContext) -> DetectionOutcome:
        return self._evaluate_validated(validate_detection_context(context))

    def _evaluate_validated(self, context: DetectionContext) -> DetectionOutcome:
        current = context.current
        if current.event_type not in (EventType.TOOL_COMPLETION, EventType.OBSERVATION):
            return DetectionOutcome.abstained(
                self.signal_type,
                AbstentionReason.EVENT_NOT_APPLICABLE,
                (current.event_id,),
            )
        try:
            status = classify_test_report(current)
        except FingerprintUnavailableError as error:
            return DetectionOutcome.abstained(
                self.signal_type,
                error.reason,
                (current.event_id,),
            )
        if status is TestReportStatus.PASSED:
            return DetectionOutcome.no_match(self.signal_type, (current.event_id,))
        return DetectionOutcome.detected(
            self.signal_type,
            (current.event_id,),
        )


__all__ = ["TestFailureDetector"]
