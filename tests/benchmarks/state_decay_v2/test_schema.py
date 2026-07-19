from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from saliencegate.benchmarks.state_decay_v2.schema import (
    AdapterMetadata,
    ArtifactRole,
    BenchmarkSplit,
    BranchResult,
    ContinuationBranch,
    PolicyAllowedAction,
    PolicyCandidateMemory,
    PolicyEvent,
    PolicyEvidenceReference,
    PolicyPivot,
    PolicyViewScenario,
    ScenarioFamily,
    ScenarioOutcome,
)
from saliencegate.domain import ValidityState, canonical_json


def _event(
    event_id: str,
    sequence: int,
    action_step: int,
    *,
    statement: str | None = None,
) -> PolicyEvent:
    return PolicyEvent(
        event_id=event_id,
        sequence=sequence,
        action_step=action_step,
        statement=statement or f"Observed {event_id}",
    )


def _memory(
    *,
    memory_id: str = "memory-01",
    evidence_refs: tuple[PolicyEvidenceReference, ...] | None = None,
    recorded_sequence: int = 2,
    recorded_action_step: int = 1,
    validity: ValidityState = ValidityState.ACTIVE,
    validity_sequence: int | None = None,
    validity_action_step: int | None = None,
) -> PolicyCandidateMemory:
    return PolicyCandidateMemory(
        memory_id=memory_id,
        revision=1,
        statement="Retain the reviewed constraint.",
        evidence_refs=evidence_refs
        or (PolicyEvidenceReference(event_id="event-01", event_sequence=1),),
        recorded_sequence=recorded_sequence,
        recorded_action_step=recorded_action_step,
        validity=validity,
        validity_sequence=validity_sequence,
        validity_action_step=validity_action_step,
    )


def _scenario(**updates: object) -> PolicyViewScenario:
    values: dict[str, object] = {
        "split": BenchmarkSplit.TRAIN,
        "scenario_id": "1" * 64,
        "trajectory": (
            _event("event-01", 1, 0),
            _event("event-02", 2, 1),
        ),
        "candidate_memories": (_memory(),),
        "pivot": PolicyPivot(
            event_id="pivot-01",
            sequence=3,
            action_step=2,
            statement="Choose the next bounded action.",
        ),
        "allowed_actions": (
            PolicyAllowedAction(action_id="action-01", statement="Use the constraint."),
            PolicyAllowedAction(action_id="action-02", statement="Continue without it."),
        ),
        "adapter": AdapterMetadata(
            adapter_id="state-machine",
            adapter_version="v2",
            response_profile_id="canonical-proposal",
            response_profile_digest="2" * 64,
            capabilities=("typed-proposal", "paired-continuation"),
        ),
    }
    values.update(updates)
    return PolicyViewScenario.model_validate(values)


def test_closed_enums_freeze_the_v2_vocabulary() -> None:
    assert tuple(BenchmarkSplit) == (
        BenchmarkSplit.TRAIN,
        BenchmarkSplit.DEVELOPMENT,
        BenchmarkSplit.LOCKED,
        BenchmarkSplit.DIAGNOSTIC,
    )
    assert tuple(ArtifactRole) == (
        ArtifactRole.POLICY_VIEW,
        ArtifactRole.ORACLE_VAULT,
        ArtifactRole.ANALYSIS_CLUSTER_MAP,
    )
    assert tuple(ScenarioFamily) == (
        ScenarioFamily.FORGOTTEN_REQUIREMENT,
        ScenarioFamily.FAILED_PRIOR_ATTEMPT,
        ScenarioFamily.NEGLECTED_SUBGOAL,
        ScenarioFamily.STALE_MEMORY,
        ScenarioFamily.STABLE_ENVIRONMENT_FACT,
        ScenarioFamily.RETAINED_DIAGNOSIS,
        ScenarioFamily.CONFLICTING_EVIDENCE,
        ScenarioFamily.IRREVERSIBLE_ACTION,
    )
    assert tuple(ScenarioOutcome) == (
        ScenarioOutcome.HELPFUL,
        ScenarioOutcome.HARMFUL,
        ScenarioOutcome.REDUNDANT,
        ScenarioOutcome.UNRESOLVED,
    )
    assert tuple(ContinuationBranch) == (
        ContinuationBranch.REMINDER,
        ContinuationBranch.SILENCE,
    )
    assert tuple(BranchResult) == (BranchResult.SUCCESS, BranchResult.FAILURE)


def test_policy_view_has_an_exact_hidden_incapable_shape_and_canonical_round_trip() -> None:
    scenario = _scenario()
    forbidden = {
        "family",
        "template_lineage_id",
        "outcome",
        "label",
        "allocation_slot",
        "generator_slot",
        "oracle",
        "paired_continuations",
        "decisive_action_id",
        "treatment_outcome",
        "hidden_role_digest",
    }

    assert set(PolicyViewScenario.model_fields) == {
        "schema_version",
        "suite_id",
        "suite_version",
        "split",
        "scenario_id",
        "trajectory",
        "candidate_memories",
        "pivot",
        "allowed_actions",
        "adapter",
    }
    assert forbidden.isdisjoint(PolicyViewScenario.model_fields)
    assert PolicyViewScenario.model_validate_json(canonical_json(scenario)) == scenario

    for hidden_name in sorted(forbidden):
        payload = scenario.model_dump(mode="json")
        payload[hidden_name] = "fixture-secret"
        with pytest.raises(ValidationError):
            PolicyViewScenario.model_validate(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda values: values.update(
            trajectory=(_event("event-01", 1, 0), _event("event-01", 2, 1))
        ),
        lambda values: values.update(
            trajectory=(_event("event-01", 1, 1), _event("event-02", 2, 0))
        ),
        lambda values: values.update(
            trajectory=(_event("event-01", 2, 0), _event("event-02", 1, 1))
        ),
        lambda values: values.update(
            pivot=PolicyPivot(
                event_id="pivot-01",
                sequence=2,
                action_step=2,
                statement="Not after the prefix.",
            )
        ),
        lambda values: values.update(candidate_memories=(_memory(), _memory())),
        lambda values: values.update(
            candidate_memories=(
                _memory(
                    evidence_refs=(
                        PolicyEvidenceReference(
                            event_id="missing-event",
                            event_sequence=1,
                        ),
                    )
                ),
            )
        ),
        lambda values: values.update(
            candidate_memories=(
                _memory(
                    evidence_refs=(PolicyEvidenceReference(event_id="event-01", event_sequence=2),)
                ),
            )
        ),
        lambda values: values.update(candidate_memories=(_memory(recorded_sequence=4),)),
        lambda values: values.update(
            allowed_actions=(
                PolicyAllowedAction(action_id="action-01", statement="One."),
                PolicyAllowedAction(action_id="action-01", statement="Duplicate."),
            )
        ),
    ],
)
def test_policy_view_rejects_duplicate_unresolved_future_and_non_monotone_data(
    mutate: Callable[[dict[str, object]], None],
) -> None:
    values = _scenario().model_dump(mode="python")
    mutate(values)

    with pytest.raises(ValidationError):
        PolicyViewScenario.model_validate(values)


def test_memory_validity_transition_is_explicit_and_bounded_by_the_pivot() -> None:
    with pytest.raises(ValidationError):
        _memory(validity_sequence=2, validity_action_step=1)
    with pytest.raises(ValidationError):
        _memory(validity=ValidityState.INVALIDATED)
    with pytest.raises(ValidationError):
        _memory(
            validity=ValidityState.INVALIDATED,
            validity_sequence=1,
            validity_action_step=0,
        )

    invalidated = _memory(
        validity=ValidityState.INVALIDATED,
        validity_sequence=3,
        validity_action_step=2,
    )
    assert _scenario(candidate_memories=(invalidated,)).candidate_memories == (invalidated,)

    after_pivot = _memory(
        validity=ValidityState.INVALIDATED,
        validity_sequence=4,
        validity_action_step=3,
    )
    with pytest.raises(ValidationError):
        _scenario(candidate_memories=(after_pivot,))


def test_policy_models_are_strict_frozen_and_bounded() -> None:
    scenario = _scenario()

    with pytest.raises(ValidationError):
        PolicyEvent(
            event_id="event-01",
            sequence=True,
            action_step=0,
            statement="Strict integer.",
        )
    with pytest.raises(ValidationError):
        PolicyEvent(
            event_id="event-01",
            sequence=1,
            action_step=0,
            statement="😀" * 1_500,
        )
    with pytest.raises(ValidationError):
        _scenario(
            allowed_actions=tuple(
                PolicyAllowedAction(action_id=f"action-{index:02d}", statement="Bounded.")
                for index in range(17)
            )
        )
    with pytest.raises(ValidationError):
        scenario.scenario_id = "3" * 64  # type: ignore[misc]
