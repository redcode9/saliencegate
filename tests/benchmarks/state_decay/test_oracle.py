from __future__ import annotations

from itertools import product
from typing import Any

import pytest

import saliencegate.benchmarks.state_decay.oracle as oracle_module
from saliencegate.benchmarks.state_decay.oracle import (
    ORACLE_RESULT_SCHEMA_VERSION,
    OracleEvaluationError,
    OracleResult,
    evaluate_scenario,
)
from saliencegate.benchmarks.state_decay.schema import (
    AllowedAction,
    CandidateMemory,
    ContinuationBranch,
    ContinuationOutcome,
    EvidenceCriteria,
    InterventionLabel,
    MemorySourceRef,
    OracleCriteria,
    PairedContinuation,
    Pivot,
    ScenarioFamily,
    StateDecayScenario,
    TrajectoryEvent,
)
from saliencegate.domain import ValidityState

SOURCE_ID = "source-1"
PIVOT_ID = "pivot-1"
MEMORY_ID = "memory-1"
REQUIRED_ACTION_ID = "action-required"
OTHER_ACTION_ID = "action-other"


def _continuation(
    branch: ContinuationBranch,
    *,
    succeeds: bool,
) -> PairedContinuation:
    return PairedContinuation(
        branch=branch,
        selected_action_id=REQUIRED_ACTION_ID if succeeds else OTHER_ACTION_ID,
        outcome=(ContinuationOutcome.SUCCESS if succeeds else ContinuationOutcome.FAILURE),
        explanation="The paired branch follows its deterministic transition.",
        evidence_source_ids=(SOURCE_ID,) if succeeds else (),
        evidence_memory_ids=(MEMORY_ID,) if succeeds else (),
    )


def scenario(
    family: ScenarioFamily = ScenarioFamily.FORGOTTEN_REQUIREMENT,
    label: InterventionLabel = InterventionLabel.INTERVENE,
) -> StateDecayScenario:
    reminder_succeeds = label is InterventionLabel.INTERVENE
    return StateDecayScenario(
        generator_version="v1",
        seed=17,
        scenario_id="a" * 64,
        template_lineage_id="lineage-1",
        family=family,
        trajectory_prefix=(
            TrajectoryEvent(
                step=1,
                source_id=SOURCE_ID,
                statement="The decisive requirement was recorded.",
            ),
        ),
        candidate_memories=(
            CandidateMemory(
                memory_id=MEMORY_ID,
                statement="Retain the decisive requirement.",
                source_refs=(MemorySourceRef(source_id=SOURCE_ID, source_step=1),),
                revision=1,
                validity=ValidityState.ACTIVE,
                validity_step=None,
                recorded_step=2,
            ),
        ),
        pivot=Pivot(
            step=3,
            source_id=PIVOT_ID,
            statement="Choose the next deterministic action.",
        ),
        allowed_actions=(
            AllowedAction(
                action_id=REQUIRED_ACTION_ID,
                statement="Respect the decisive evidence.",
            ),
            AllowedAction(
                action_id=OTHER_ACTION_ID,
                statement="Ignore the decisive evidence.",
            ),
        ),
        label=label,
        oracle=OracleCriteria(
            required_action_id=REQUIRED_ACTION_ID,
            success_condition="The required action is selected.",
            failure_condition="A different action is selected.",
        ),
        evidence_criteria=EvidenceCriteria(
            decisive_source_ids=(SOURCE_ID,),
            decisive_memory_ids=(MEMORY_ID,),
        ),
        paired_continuations=(
            _continuation(
                ContinuationBranch.REMINDER,
                succeeds=reminder_succeeds,
            ),
            _continuation(
                ContinuationBranch.SILENCE,
                succeeds=not reminder_succeeds,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("family", "label"),
    tuple(product(tuple(ScenarioFamily), tuple(InterventionLabel))),
)
def test_oracle_derives_both_labels_for_all_eight_families(
    family: ScenarioFamily,
    label: InterventionLabel,
) -> None:
    assert len(tuple(ScenarioFamily)) == 8

    result = evaluate_scenario(scenario(family, label))

    assert isinstance(result, OracleResult)
    assert result.schema_version == ORACLE_RESULT_SCHEMA_VERSION
    assert result.expected_label is label
    assert result.observed_label is label
    assert result.matched
    assert result.reminder_success is (label is InterventionLabel.INTERVENE)
    assert result.silence_success is (label is InterventionLabel.SILENCE)
    assert result.decisive_action_id == REQUIRED_ACTION_ID
    assert result.decisive_source_ids == (SOURCE_ID,)
    assert result.decisive_memory_ids == (MEMORY_ID,)
    assert result.reminder_evidence_supported is result.reminder_success
    assert result.silence_evidence_supported is result.silence_success


def test_declared_label_is_observed_instead_of_trusted_as_expected() -> None:
    original = scenario(label=InterventionLabel.INTERVENE)
    altered = original.model_copy(update={"label": InterventionLabel.SILENCE})

    result = evaluate_scenario(altered)

    assert result.expected_label is InterventionLabel.INTERVENE
    assert result.observed_label is InterventionLabel.SILENCE
    assert not result.matched


def test_task_success_and_evidence_support_are_independent_axes() -> None:
    original = scenario()
    reminder, silence = original.paired_continuations
    unsupported = reminder.model_copy(update={"evidence_memory_ids": ()})
    altered = original.model_copy(update={"paired_continuations": (unsupported, silence)})

    result = evaluate_scenario(altered)

    assert result.reminder_success is True
    assert result.reminder_evidence_supported is False
    assert result.expected_label is InterventionLabel.INTERVENE


def test_failure_outcome_with_all_success_criteria_is_incoherent() -> None:
    original = scenario()
    reminder, silence = original.paired_continuations
    false_failure = reminder.model_copy(update={"outcome": ContinuationOutcome.FAILURE})
    altered = original.model_copy(update={"paired_continuations": (false_failure, silence)})

    with pytest.raises(OracleEvaluationError):
        evaluate_scenario(altered)


def test_success_with_a_disallowed_action_is_rejected() -> None:
    original = scenario()
    reminder, silence = original.paired_continuations
    disallowed = reminder.model_copy(update={"selected_action_id": "action-forged"})
    altered = original.model_copy(update={"paired_continuations": (disallowed, silence)})

    with pytest.raises(OracleEvaluationError):
        evaluate_scenario(altered)


def test_stale_decisive_memory_cannot_support_a_branch() -> None:
    original = scenario()
    stale = original.candidate_memories[0].model_copy(
        update={"validity": ValidityState.INVALIDATED, "validity_step": 2}
    )
    altered = original.model_copy(update={"candidate_memories": (stale,)})

    with pytest.raises(OracleEvaluationError):
        evaluate_scenario(altered)


def test_unknown_or_future_evidence_source_is_rejected() -> None:
    original = scenario()
    reminder, silence = original.paired_continuations
    future = reminder.model_copy(update={"evidence_source_ids": ("source-future",)})
    altered = original.model_copy(update={"paired_continuations": (future, silence)})

    with pytest.raises(OracleEvaluationError):
        evaluate_scenario(altered)


def test_future_memory_source_reference_is_rejected() -> None:
    original = scenario()
    future_ref = MemorySourceRef(source_id=SOURCE_ID, source_step=4)
    future_memory = original.candidate_memories[0].model_copy(update={"source_refs": (future_ref,)})
    altered = original.model_copy(update={"candidate_memories": (future_memory,)})

    with pytest.raises(OracleEvaluationError):
        evaluate_scenario(altered)


def test_duplicate_or_missing_paired_branch_is_rejected() -> None:
    original = scenario()
    reminder = original.paired_continuations[0]
    altered = original.model_copy(update={"paired_continuations": (reminder, reminder)})

    with pytest.raises(OracleEvaluationError):
        evaluate_scenario(altered)


def test_forged_scenario_is_rejected_without_disclosing_values() -> None:
    secret = "private-forged-oracle-value"
    forged = scenario().model_copy(update={"seed": secret})

    with pytest.raises(OracleEvaluationError) as captured:
        evaluate_scenario(forged)

    assert str(captured.value) == "scenario failed deterministic oracle validation"
    assert secret not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_wrong_runtime_type_is_rejected_value_free() -> None:
    with pytest.raises(OracleEvaluationError):
        evaluate_scenario(object())  # type: ignore[arg-type]


def _reject_without_schema_boundary(
    monkeypatch: pytest.MonkeyPatch,
    forged: StateDecayScenario,
) -> None:
    monkeypatch.setattr(oracle_module, "_validated_scenario", lambda _: forged)
    with pytest.raises(OracleEvaluationError):
        evaluate_scenario(forged)


def test_oracle_rechecks_action_and_evidence_criteria_after_the_schema_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = scenario()
    unknown_action = original.oracle.model_copy(update={"required_action_id": "action-unknown"})
    no_evidence = original.evidence_criteria.model_copy(
        update={"decisive_source_ids": (), "decisive_memory_ids": ()}
    )

    for forged in (
        original.model_copy(update={"oracle": unknown_action}),
        original.model_copy(update={"evidence_criteria": no_evidence}),
    ):
        _reject_without_schema_boundary(monkeypatch, forged)


def test_oracle_rechecks_decisive_references_after_the_schema_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = scenario()
    unknown_source = original.evidence_criteria.model_copy(
        update={"decisive_source_ids": ("source-unknown",)}
    )
    unknown_memory = original.evidence_criteria.model_copy(
        update={"decisive_memory_ids": ("memory-unknown",)}
    )
    stale = original.candidate_memories[0].model_copy(
        update={"validity": ValidityState.INVALIDATED, "validity_step": 2}
    )

    for forged in (
        original.model_copy(update={"evidence_criteria": unknown_source}),
        original.model_copy(update={"evidence_criteria": unknown_memory}),
        original.model_copy(update={"candidate_memories": (stale,)}),
    ):
        _reject_without_schema_boundary(monkeypatch, forged)


def test_oracle_rechecks_each_continuation_reference_after_the_schema_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = scenario()
    reminder, silence = original.paired_continuations
    forged_continuations = (
        reminder.model_copy(update={"branch": ContinuationBranch.SILENCE}),
        reminder.model_copy(update={"selected_action_id": "action-unknown"}),
        reminder.model_copy(update={"evidence_source_ids": ("source-unknown",)}),
        reminder.model_copy(update={"evidence_memory_ids": ("memory-unknown",)}),
    )

    for forged_continuation in forged_continuations:
        forged = original.model_copy(
            update={"paired_continuations": (forged_continuation, silence)}
        )
        _reject_without_schema_boundary(monkeypatch, forged)


@pytest.mark.parametrize(
    "successful_branches", ((), (ContinuationBranch.REMINDER, ContinuationBranch.SILENCE))
)
def test_oracle_rejects_pairs_without_one_decisive_branch(
    monkeypatch: pytest.MonkeyPatch,
    successful_branches: tuple[ContinuationBranch, ...],
) -> None:
    original = scenario()
    paired = tuple(
        _continuation(branch, succeeds=branch in successful_branches)
        for branch in (ContinuationBranch.REMINDER, ContinuationBranch.SILENCE)
    )
    forged = original.model_copy(update={"paired_continuations": paired})

    _reject_without_schema_boundary(monkeypatch, forged)


def test_oracle_rejects_an_incomplete_pair_after_the_schema_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = scenario()
    forged = original.model_copy(
        update={"paired_continuations": (original.paired_continuations[0],)}
    )

    _reject_without_schema_boundary(monkeypatch, forged)


def test_unexpected_internal_failure_is_value_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "private-unexpected-oracle-value"

    def explode(**_: Any) -> OracleResult:
        raise RuntimeError(secret)

    monkeypatch.setattr(oracle_module, "OracleResult", explode)

    with pytest.raises(OracleEvaluationError) as captured:
        evaluate_scenario(scenario())

    assert secret not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
