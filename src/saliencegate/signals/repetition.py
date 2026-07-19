from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from saliencegate.domain import (
    EventPhase,
    EventType,
    SignalType,
    TraceEvent,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.signals.base import (
    AbstentionReason,
    DetectionContext,
    DetectionInputError,
    DetectionOutcome,
    _ValidatedSignalDetector,
    validate_detection_context,
)
from saliencegate.signals.fingerprints import (
    ActionFingerprint,
    FailureFingerprint,
    FingerprintUnavailableError,
    TestReportStatus,
    ToolOutcomeStatus,
    action_fingerprint,
    failure_fingerprint,
    parse_test_report,
    parse_tool_outcome,
)


class RepetitionConfig(BaseModel):
    """Fully resolved bounds for deterministic repetition detectors."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
        validate_default=True,
    )

    window_events: Annotated[int, Field(ge=2, le=10_000)]


def _validated_config(value: RepetitionConfig) -> RepetitionConfig:
    if (
        type(value) is not RepetitionConfig
        or type(value.window_events) is not int
        or not 2 <= value.window_events <= 10_000
    ):
        raise DetectionInputError()
    validated: RepetitionConfig | None = None
    try:
        validated = RepetitionConfig.model_validate_json(value.model_dump_json(warnings=False))
    except Exception:
        validated = None
    if validated is None:
        raise DetectionInputError()
    return validated


def _detector_version(name: str, config: RepetitionConfig) -> str:
    digest = length_prefixed_sha256(
        canonical_json(config),
        domain="saliencegate:signals:repetition-config:v1",
    )
    return f"{name}/v1+{digest}"


def _abstained(
    signal_type: SignalType,
    error: FingerprintUnavailableError,
    related_event_ids: tuple[UUID, ...],
) -> DetectionOutcome:
    return DetectionOutcome.abstained(signal_type, error.reason, related_event_ids)


@dataclass(frozen=True, slots=True, init=False)
class RepeatedActionDetector(_ValidatedSignalDetector):
    """Detect a deterministic action fingerprint repeated in a bounded event suffix."""

    _config: RepetitionConfig
    _detector_version: str

    def __init__(self, config: RepetitionConfig) -> None:
        validated = _validated_config(config)
        object.__setattr__(self, "_config", validated)
        object.__setattr__(
            self,
            "_detector_version",
            _detector_version("repeated-action", validated),
        )

    @property
    def signal_type(self) -> SignalType:
        return SignalType.REPEATED_ACTION

    @property
    def detector_version(self) -> str:
        return self._detector_version

    @property
    def config(self) -> RepetitionConfig:
        return self._config

    def evaluate(self, context: DetectionContext) -> DetectionOutcome:
        return self._evaluate_validated(validate_detection_context(context))

    def _evaluate_validated(self, context: DetectionContext) -> DetectionOutcome:
        current = context.current
        if current.event_type is not EventType.ACTION_PROPOSAL:
            return DetectionOutcome.abstained(
                self.signal_type,
                AbstentionReason.EVENT_NOT_APPLICABLE,
                (current.event_id,),
            )
        if current.phase is not EventPhase.PRE_ACTION:
            return DetectionOutcome.abstained(
                self.signal_type,
                AbstentionReason.PRE_ACTION_INTERCEPTION_UNAVAILABLE,
                (current.event_id,),
            )

        try:
            current_fingerprint = action_fingerprint(current)
        except FingerprintUnavailableError as error:
            return _abstained(self.signal_type, error, (current.event_id,))

        window = context.events[-self._config.window_events :]
        if len(window) < 2:
            return DetectionOutcome.abstained(
                self.signal_type,
                AbstentionReason.INSUFFICIENT_HISTORY,
                (current.event_id,),
            )

        unavailable: tuple[FingerprintUnavailableError, UUID] | None = None
        for candidate in reversed(window[:-1]):
            if candidate.event_type is not EventType.ACTION_PROPOSAL:
                continue
            try:
                candidate_fingerprint = action_fingerprint(candidate)
            except FingerprintUnavailableError as error:
                if unavailable is None:
                    unavailable = error, candidate.event_id
                continue
            if candidate_fingerprint == current_fingerprint:
                return DetectionOutcome.detected(
                    self.signal_type,
                    (candidate.event_id, current.event_id),
                )

        if unavailable is not None:
            unavailable_error, event_id = unavailable
            return _abstained(
                self.signal_type,
                unavailable_error,
                (event_id, current.event_id),
            )
        return DetectionOutcome.no_match(self.signal_type, (current.event_id,))


class _OutcomeKind(StrEnum):
    ABSENT = "absent"
    SUCCESS = "success"
    FAILURE = "failure"


class _RelatedFingerprintUnavailableError(FingerprintUnavailableError):
    __slots__ = ("event_id",)

    def __init__(self, reason: AbstentionReason, event_id: UUID) -> None:
        self.event_id = event_id
        super().__init__(reason)


def _structured_outcome(
    event: TraceEvent,
) -> tuple[_OutcomeKind, FailureFingerprint | None]:
    has_test_report = "test_report" in event.payload
    has_tool_outcome = "tool_outcome" in event.payload
    if has_test_report and has_tool_outcome:
        raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_INVALID)
    if has_test_report:
        report = parse_test_report(event)
        if report.status is TestReportStatus.PASSED:
            return _OutcomeKind.SUCCESS, None
        return _OutcomeKind.FAILURE, failure_fingerprint(event)
    if has_tool_outcome:
        outcome = parse_tool_outcome(event)
        if outcome.status is ToolOutcomeStatus.SUCCEEDED:
            return _OutcomeKind.SUCCESS, None
        return _OutcomeKind.FAILURE, failure_fingerprint(event)
    return _OutcomeKind.ABSENT, None


def _parent_action(
    event: TraceEvent,
    events_by_id: Mapping[UUID, TraceEvent],
) -> TraceEvent:
    if not event.parent_ids:
        raise FingerprintUnavailableError(AbstentionReason.PARENT_ACTION_MISSING)
    resolved: list[TraceEvent] = []
    for parent_id in event.parent_ids:
        parent = events_by_id.get(parent_id)
        if parent is None or parent.sequence >= event.sequence:
            raise FingerprintUnavailableError(AbstentionReason.PARENT_ACTION_MISSING)
        if parent.event_type is EventType.ACTION_PROPOSAL:
            resolved.append(parent)
    if not resolved:
        raise FingerprintUnavailableError(AbstentionReason.PARENT_ACTION_MISSING)
    if len(resolved) != 1:
        raise FingerprintUnavailableError(AbstentionReason.AMBIGUOUS_PARENT_ACTION)
    return resolved[0]


def _action_for_outcome(
    event: TraceEvent,
    events_by_id: Mapping[UUID, TraceEvent],
) -> tuple[TraceEvent, ActionFingerprint]:
    action = _parent_action(event, events_by_id)
    return action, action_fingerprint(action)


def _matching_success_between(
    events: tuple[TraceEvent, ...],
    *,
    after_sequence: int,
    before_sequence: int,
    action: ActionFingerprint,
    events_by_id: Mapping[UUID, TraceEvent],
) -> tuple[TraceEvent, TraceEvent] | None:
    for event in events:
        if not after_sequence < event.sequence < before_sequence:
            continue
        try:
            kind, _ = _structured_outcome(event)
        except FingerprintUnavailableError as error:
            if "test_report" in event.payload or "tool_outcome" in event.payload:
                raise _RelatedFingerprintUnavailableError(error.reason, event.event_id) from None
            continue
        if kind is not _OutcomeKind.SUCCESS:
            continue
        try:
            candidate_action_event, candidate_action = _action_for_outcome(event, events_by_id)
        except FingerprintUnavailableError as error:
            raise _RelatedFingerprintUnavailableError(error.reason, event.event_id) from None
        if candidate_action == action:
            if candidate_action_event.sequence <= after_sequence:
                raise _RelatedFingerprintUnavailableError(
                    AbstentionReason.STRUCTURED_EVIDENCE_INVALID,
                    event.event_id,
                )
            return candidate_action_event, event
    return None


def _success_for_action_event_between(
    events: tuple[TraceEvent, ...],
    *,
    after_sequence: int,
    before_sequence: int,
    action_event_id: UUID,
    events_by_id: Mapping[UUID, TraceEvent],
) -> TraceEvent | None:
    for event in events:
        if not after_sequence < event.sequence < before_sequence:
            continue
        try:
            kind, _ = _structured_outcome(event)
        except FingerprintUnavailableError as error:
            raise _RelatedFingerprintUnavailableError(error.reason, event.event_id) from None
        if kind is not _OutcomeKind.SUCCESS:
            continue
        try:
            parent = _parent_action(event, events_by_id)
        except FingerprintUnavailableError as error:
            raise _RelatedFingerprintUnavailableError(error.reason, event.event_id) from None
        if parent.event_id == action_event_id:
            return event
    return None


@dataclass(frozen=True, slots=True, init=False)
class RepeatedFailureDetector(_ValidatedSignalDetector):
    """Detect a repeated structured failure while its action diagnosis is unresolved."""

    _config: RepetitionConfig
    _detector_version: str

    def __init__(self, config: RepetitionConfig) -> None:
        validated = _validated_config(config)
        object.__setattr__(self, "_config", validated)
        object.__setattr__(
            self,
            "_detector_version",
            _detector_version("repeated-failure", validated),
        )

    @property
    def signal_type(self) -> SignalType:
        return SignalType.REPEATED_FAILURE

    @property
    def detector_version(self) -> str:
        return self._detector_version

    @property
    def config(self) -> RepetitionConfig:
        return self._config

    def evaluate(self, context: DetectionContext) -> DetectionOutcome:
        return self._evaluate_validated(validate_detection_context(context))

    def _evaluate_validated(self, context: DetectionContext) -> DetectionOutcome:
        current = context.current
        window = context.events[-self._config.window_events :]
        events_by_id = {event.event_id: event for event in window}

        try:
            current_kind, current_failure = _structured_outcome(current)
        except FingerprintUnavailableError as error:
            return _abstained(self.signal_type, error, (current.event_id,))

        if current_kind is _OutcomeKind.ABSENT:
            reason = (
                AbstentionReason.STRUCTURED_EVIDENCE_MISSING
                if current.event_type in (EventType.TOOL_COMPLETION, EventType.OBSERVATION)
                else AbstentionReason.EVENT_NOT_APPLICABLE
            )
            return DetectionOutcome.abstained(
                self.signal_type,
                reason,
                (current.event_id,),
            )
        if current_kind is _OutcomeKind.SUCCESS:
            return DetectionOutcome.no_match(self.signal_type, (current.event_id,))
        assert current_failure is not None

        if len(window) < 2:
            return DetectionOutcome.abstained(
                self.signal_type,
                AbstentionReason.INSUFFICIENT_HISTORY,
                (current.event_id,),
            )

        try:
            current_action, current_action_fingerprint = _action_for_outcome(
                current,
                events_by_id,
            )
        except FingerprintUnavailableError as error:
            return _abstained(self.signal_type, error, (current.event_id,))

        try:
            current_action_success = _success_for_action_event_between(
                window,
                after_sequence=current_action.sequence,
                before_sequence=current.sequence,
                action_event_id=current_action.event_id,
                events_by_id=events_by_id,
            )
        except _RelatedFingerprintUnavailableError as error:
            return _abstained(
                self.signal_type,
                error,
                (error.event_id, current.event_id),
            )
        if current_action_success is not None:
            return DetectionOutcome.abstained(
                self.signal_type,
                AbstentionReason.STRUCTURED_EVIDENCE_INVALID,
                (current_action_success.event_id, current.event_id),
            )

        unavailable: tuple[FingerprintUnavailableError, UUID] | None = None
        for candidate in reversed(window[:-1]):
            try:
                candidate_kind, candidate_failure = _structured_outcome(candidate)
            except FingerprintUnavailableError as error:
                if unavailable is None:
                    unavailable = error, candidate.event_id
                continue
            if candidate_kind is not _OutcomeKind.FAILURE or candidate_failure != current_failure:
                continue
            try:
                prior_action, prior_action_fingerprint = _action_for_outcome(
                    candidate,
                    events_by_id,
                )
            except FingerprintUnavailableError as error:
                return _abstained(
                    self.signal_type,
                    error,
                    (candidate.event_id, current.event_id),
                )
            if prior_action_fingerprint != current_action_fingerprint:
                continue
            if prior_action.event_id == current_action.event_id:
                continue
            if not (
                prior_action.sequence
                < candidate.sequence
                < current_action.sequence
                < current.sequence
            ):
                continue
            try:
                resolving_success = _matching_success_between(
                    window,
                    after_sequence=candidate.sequence,
                    before_sequence=current_action.sequence,
                    action=current_action_fingerprint,
                    events_by_id=events_by_id,
                )
            except _RelatedFingerprintUnavailableError as error:
                return _abstained(
                    self.signal_type,
                    error,
                    (error.event_id, current.event_id),
                )
            if resolving_success is not None:
                resolution_action, resolution_outcome = resolving_success
                return DetectionOutcome.no_match(
                    self.signal_type,
                    (
                        prior_action.event_id,
                        candidate.event_id,
                        resolution_action.event_id,
                        resolution_outcome.event_id,
                        current_action.event_id,
                        current.event_id,
                    ),
                )
            return DetectionOutcome.detected(
                self.signal_type,
                (
                    prior_action.event_id,
                    candidate.event_id,
                    current_action.event_id,
                    current.event_id,
                ),
            )

        if unavailable is not None:
            unavailable_error, event_id = unavailable
            return _abstained(
                self.signal_type,
                unavailable_error,
                (event_id, current.event_id),
            )
        return DetectionOutcome.no_match(self.signal_type, (current.event_id,))


__all__ = [
    "RepeatedActionDetector",
    "RepeatedFailureDetector",
    "RepetitionConfig",
]
