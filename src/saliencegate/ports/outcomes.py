from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from saliencegate.domain import (
    ConstraintStatus,
    InterventionOutcome,
    OutcomeEvidenceMode,
    RepeatedErrorStatus,
)


class OutcomeRecordingError(ValueError):
    """A value-free failure at the outcome recorder boundary."""

    def __init__(self) -> None:
        super().__init__("outcome failed policy-replay validation")


@runtime_checkable
class OutcomeRecorder(Protocol):
    async def record(self, outcome: InterventionOutcome) -> None: ...


def _validated_policy_outcome(value: object) -> InterventionOutcome:
    if type(value) is not InterventionOutcome:
        raise OutcomeRecordingError()
    try:
        outcome = InterventionOutcome.model_validate_json(value.model_dump_json(warnings=False))
    except Exception:
        raise OutcomeRecordingError() from None
    if (
        outcome != value
        or outcome.evidence_mode is not OutcomeEvidenceMode.POLICY_REPLAY
        or outcome.next_action_fingerprint is not None
        or outcome.repeated_error_status is not RepeatedErrorStatus.UNKNOWN
        or outcome.constraint_status is not ConstraintStatus.UNKNOWN
        or outcome.utility is not None
        or outcome.action_changed is not None
        or outcome.task_reward is not None
        or outcome.task_passed is not None
    ):
        raise OutcomeRecordingError()
    return outcome


class PolicyReplayOutcomeRecorder:
    """An in-memory recorder that permits only explicitly non-causal replay evidence."""

    __slots__ = ("_lock", "_outcomes")

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._outcomes: list[InterventionOutcome] = []

    @property
    def outcomes(self) -> tuple[InterventionOutcome, ...]:
        return tuple(
            InterventionOutcome.model_validate_json(outcome.model_dump_json(warnings=False))
            for outcome in self._outcomes
        )

    async def record(self, outcome: InterventionOutcome) -> None:
        validated = _validated_policy_outcome(outcome)
        async with self._lock:
            self._outcomes.append(validated)


__all__ = [
    "OutcomeRecorder",
    "OutcomeRecordingError",
    "PolicyReplayOutcomeRecorder",
]
