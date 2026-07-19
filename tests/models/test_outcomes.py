from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID

import pytest

from saliencegate.domain import (
    ConstraintStatus,
    InterventionOutcome,
    OutcomeEvidenceMode,
    RepeatedErrorStatus,
    UtilityLabel,
    canonical_digest,
)
from saliencegate.ports.outcomes import (
    OutcomeRecorder,
    OutcomeRecordingError,
    PolicyReplayOutcomeRecorder,
)

RUN_ID = UUID("00000000-0000-4000-8000-00000000c001")
INTERVENTION_ID = UUID("00000000-0000-4000-8000-00000000c002")
NOW = datetime(2026, 7, 11, 20, 0, tzinfo=UTC)


def outcome(
    *,
    mode: OutcomeEvidenceMode = OutcomeEvidenceMode.POLICY_REPLAY,
) -> InterventionOutcome:
    return InterventionOutcome(
        outcome_id=UUID("00000000-0000-4000-8000-00000000c003"),
        run_id=RUN_ID,
        intervention_id=INTERVENTION_ID,
        repeated_error_status=RepeatedErrorStatus.UNKNOWN,
        constraint_status=ConstraintStatus.UNKNOWN,
        evidence_mode=mode,
        created_at=NOW,
    )


async def test_policy_recorder_records_only_revalidated_noncausal_outcomes() -> None:
    recorder = PolicyReplayOutcomeRecorder()
    first = outcome()
    second = first.model_copy(update={"outcome_id": UUID("00000000-0000-4000-8000-00000000c004")})

    await asyncio.gather(recorder.record(first), recorder.record(second))

    assert isinstance(recorder, OutcomeRecorder)
    assert set(recorder.outcomes) == {first, second}
    assert all(recorded is not first for recorded in recorder.outcomes)
    snapshot = recorder.outcomes
    assert snapshot is not recorder.outcomes


async def test_policy_recorder_rejects_non_policy_evidence_mode() -> None:
    recorder = PolicyReplayOutcomeRecorder()

    with pytest.raises(OutcomeRecordingError) as captured:
        await recorder.record(outcome(mode=OutcomeEvidenceMode.LIVE_OBSERVATION))

    assert str(captured.value) == "outcome failed policy-replay validation"
    assert recorder.outcomes == ()


async def test_policy_recorder_rejects_forged_causal_utility_without_disclosure() -> None:
    recorder = PolicyReplayOutcomeRecorder()
    secret = "helpful-because-private-secret"
    forged = outcome().model_copy(update={"utility": UtilityLabel.HELPFUL})
    forged.__dict__["next_action_fingerprint"] = secret

    with pytest.raises(OutcomeRecordingError) as captured:
        await recorder.record(forged)

    assert secret not in str(captured.value)
    assert recorder.outcomes == ()


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("action_changed", True),
        ("task_reward", 1.0),
        ("task_passed", True),
    ),
)
async def test_policy_recorder_rejects_other_causal_outcome_fields(
    field_name: str,
    value: object,
) -> None:
    recorder = PolicyReplayOutcomeRecorder()
    causal = outcome().model_copy(update={field_name: value})

    with pytest.raises(OutcomeRecordingError):
        await recorder.record(causal)

    assert recorder.outcomes == ()


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("next_action_fingerprint", canonical_digest("observed action")),
        ("repeated_error_status", RepeatedErrorStatus.AVOIDED),
        ("constraint_status", ConstraintStatus.RESPECTED),
    ),
)
async def test_policy_recorder_rejects_observed_effect_claims(
    field_name: str,
    value: object,
) -> None:
    recorder = PolicyReplayOutcomeRecorder()
    causal = outcome().model_copy(update={field_name: value})

    with pytest.raises(OutcomeRecordingError):
        await recorder.record(causal)

    assert recorder.outcomes == ()


async def test_policy_recorder_rejects_wrong_runtime_type() -> None:
    recorder = PolicyReplayOutcomeRecorder()

    with pytest.raises(OutcomeRecordingError):
        await recorder.record(object())  # type: ignore[arg-type]
