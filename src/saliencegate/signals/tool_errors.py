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
    ToolOutcomeStatus,
    classify_tool_outcome,
)

_DETECTOR_VERSION = "tool-error/v1"


class ToolErrorDetector(_ValidatedSignalDetector):
    """Detect explicit controller and structured tool failures without text inference."""

    __slots__ = ()

    @property
    def signal_type(self) -> SignalType:
        return SignalType.TOOL_ERROR

    @property
    def detector_version(self) -> str:
        return _DETECTOR_VERSION

    def evaluate(self, context: DetectionContext) -> DetectionOutcome:
        return self._evaluate_validated(validate_detection_context(context))

    def _evaluate_validated(self, context: DetectionContext) -> DetectionOutcome:
        current = context.current

        if current.event_type is EventType.CONTROLLER_ERROR:
            return DetectionOutcome.detected(
                self.signal_type,
                (current.event_id,),
            )
        if current.event_type is not EventType.TOOL_COMPLETION:
            return DetectionOutcome.abstained(
                self.signal_type,
                AbstentionReason.EVENT_NOT_APPLICABLE,
                (current.event_id,),
            )

        try:
            status = classify_tool_outcome(current)
        except FingerprintUnavailableError as error:
            return DetectionOutcome.abstained(
                self.signal_type,
                error.reason,
                (current.event_id,),
            )

        if status is ToolOutcomeStatus.FAILED:
            return DetectionOutcome.detected(
                self.signal_type,
                (current.event_id,),
            )
        return DetectionOutcome.no_match(self.signal_type, (current.event_id,))


__all__ = ["ToolErrorDetector"]
