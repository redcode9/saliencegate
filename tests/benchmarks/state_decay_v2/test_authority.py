from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from saliencegate.benchmarks.state_decay_v2.authority import (
    AdmissibleTreatmentBinding,
    AnalysisClusterEntry,
    DecisiveEvidence,
    MemoryRevisionEvidence,
    OracleVaultEntry,
    PairedBranchOutcome,
    ResolvedProposalLabel,
    TerminalVerifierInstruction,
    revalidate_analysis_cluster_entry,
    revalidate_oracle_vault_entry,
)
from saliencegate.benchmarks.state_decay_v2.schema import (
    BenchmarkSplit,
    BranchResult,
    ContinuationBranch,
    PolicyViewScenario,
    ScenarioFamily,
    ScenarioOutcome,
)


def _treatment() -> AdmissibleTreatmentBinding:
    return AdmissibleTreatmentBinding(
        fixture_digest="1" * 64,
        proposal_digest="2" * 64,
        grounding_receipt_digest="3" * 64,
        renderer_id="fixed-ascii",
        renderer_version="v1",
        renderer_digest="4" * 64,
        rendered_text_digest="5" * 64,
        evidence_id_set_digest="6" * 64,
        evidence_revision_set_digest="7" * 64,
    )


def _verifier() -> TerminalVerifierInstruction:
    return TerminalVerifierInstruction(
        verifier_id="bounded-state-machine",
        verifier_version="v2",
        maximum_steps=8,
        success_condition="The required terminal state is reached.",
        failure_condition="The bounded continuation ends without that state.",
        instruction_digest="8" * 64,
    )


def _evidence() -> DecisiveEvidence:
    return DecisiveEvidence(
        event_ids=("event-01",),
        memory_revisions=(MemoryRevisionEvidence(memory_id="memory-01", revision=1),),
        decisive_action_id="action-01",
    )


def _branches(
    reminder: BranchResult,
    silence: BranchResult,
) -> tuple[PairedBranchOutcome, ...]:
    return (
        PairedBranchOutcome(
            branch=ContinuationBranch.REMINDER,
            selected_action_id="action-01",
            result=reminder,
            terminal_state_digest="9" * 64,
            verifier_receipt_digest="a" * 64,
            action_count=2,
            repeated_action_count=0,
            failure_loop_count=0 if reminder is BranchResult.SUCCESS else 1,
        ),
        PairedBranchOutcome(
            branch=ContinuationBranch.SILENCE,
            selected_action_id="action-02",
            result=silence,
            terminal_state_digest="b" * 64,
            verifier_receipt_digest="c" * 64,
            action_count=2,
            repeated_action_count=1,
            failure_loop_count=0 if silence is BranchResult.SUCCESS else 1,
        ),
    )


_TRUTH_TABLE = {
    ScenarioOutcome.HELPFUL: (
        BranchResult.SUCCESS,
        BranchResult.FAILURE,
        True,
        ResolvedProposalLabel.INTERVENE,
    ),
    ScenarioOutcome.HARMFUL: (
        BranchResult.FAILURE,
        BranchResult.SUCCESS,
        False,
        ResolvedProposalLabel.SILENCE,
    ),
    ScenarioOutcome.REDUNDANT: (
        BranchResult.SUCCESS,
        BranchResult.SUCCESS,
        False,
        ResolvedProposalLabel.SILENCE,
    ),
    ScenarioOutcome.UNRESOLVED: (
        BranchResult.FAILURE,
        BranchResult.FAILURE,
        False,
        None,
    ),
}


def _vault(
    outcome: ScenarioOutcome = ScenarioOutcome.HELPFUL,
    **updates: object,
) -> OracleVaultEntry:
    reminder, silence, positive, resolved = _TRUTH_TABLE[outcome]
    values: dict[str, object] = {
        "split": BenchmarkSplit.TRAIN,
        "scenario_id": "d" * 64,
        "outcome": outcome,
        "opportunity_positive": positive,
        "resolved_label": resolved,
        "admissible_treatment": _treatment(),
        "verifier": _verifier(),
        "decisive_evidence": _evidence(),
        "branches": _branches(reminder, silence),
    }
    values.update(updates)
    return OracleVaultEntry.model_validate(values)


@pytest.mark.parametrize("outcome", tuple(ScenarioOutcome))
def test_oracle_vault_enforces_the_exact_causal_truth_table(
    outcome: ScenarioOutcome,
) -> None:
    vault = _vault(outcome)
    reminder, silence, positive, resolved = _TRUTH_TABLE[outcome]

    assert {branch.branch: branch.result for branch in vault.branches} == {
        ContinuationBranch.REMINDER: reminder,
        ContinuationBranch.SILENCE: silence,
    }
    assert vault.opportunity_positive is positive
    assert vault.resolved_label is resolved
    assert revalidate_oracle_vault_entry(vault) == vault


@pytest.mark.parametrize(
    "mutate",
    [
        lambda values: values.update(opportunity_positive=False),
        lambda values: values.update(resolved_label=ResolvedProposalLabel.SILENCE),
        lambda values: values.update(
            branches=_branches(BranchResult.FAILURE, BranchResult.FAILURE)
        ),
        lambda values: values.update(
            branches=(
                _branches(BranchResult.SUCCESS, BranchResult.FAILURE)[0],
                _branches(BranchResult.SUCCESS, BranchResult.FAILURE)[0],
            )
        ),
    ],
)
def test_oracle_vault_rejects_inconsistent_or_incomplete_authority(
    mutate: Callable[[dict[str, object]], None],
) -> None:
    values = _vault().model_dump(mode="python")
    mutate(values)

    with pytest.raises(ValidationError):
        OracleVaultEntry.model_validate(values)


def test_role_models_have_exact_non_overlapping_authority_shapes() -> None:
    assert set(OracleVaultEntry.model_fields) == {
        "schema_version",
        "suite_id",
        "suite_version",
        "split",
        "scenario_id",
        "outcome",
        "opportunity_positive",
        "resolved_label",
        "admissible_treatment",
        "verifier",
        "decisive_evidence",
        "branches",
    }
    assert set(AnalysisClusterEntry.model_fields) == {
        "schema_version",
        "suite_id",
        "suite_version",
        "split",
        "scenario_id",
        "family",
        "template_lineage_id",
        "generator_slot",
        "cluster_order",
        "review_digest",
    }
    assert {
        "family",
        "template_lineage_id",
        "generator_slot",
        "cluster_order",
        "outcome",
        "opportunity_positive",
        "resolved_label",
        "admissible_treatment",
        "verifier",
        "decisive_evidence",
        "branches",
    }.isdisjoint(PolicyViewScenario.model_fields)
    assert {
        "outcome",
        "opportunity_positive",
        "resolved_label",
        "admissible_treatment",
        "verifier",
        "decisive_evidence",
        "branches",
    }.isdisjoint(AnalysisClusterEntry.model_fields)
    assert {
        "family",
        "template_lineage_id",
        "generator_slot",
        "cluster_order",
        "review_digest",
    }.isdisjoint(OracleVaultEntry.model_fields)


def test_treatment_and_decisive_evidence_are_complete_unique_and_bounded() -> None:
    assert set(AdmissibleTreatmentBinding.model_fields) == {
        "schema_version",
        "fixture_digest",
        "proposal_digest",
        "grounding_receipt_digest",
        "renderer_id",
        "renderer_version",
        "renderer_digest",
        "rendered_text_digest",
        "evidence_id_set_digest",
        "evidence_revision_set_digest",
    }
    with pytest.raises(ValidationError):
        DecisiveEvidence(
            event_ids=(),
            memory_revisions=(),
            decisive_action_id="action-01",
        )
    duplicate = MemoryRevisionEvidence(memory_id="memory-01", revision=1)
    with pytest.raises(ValidationError):
        DecisiveEvidence(
            event_ids=("event-01", "event-01"),
            memory_revisions=(duplicate,),
            decisive_action_id="action-01",
        )
    with pytest.raises(ValidationError):
        DecisiveEvidence(
            event_ids=("event-01",),
            memory_revisions=(duplicate, duplicate),
            decisive_action_id="action-01",
        )
    with pytest.raises(ValidationError):
        TerminalVerifierInstruction(
            verifier_id="bounded-state-machine",
            verifier_version="v2",
            maximum_steps=65,
            success_condition="Success.",
            failure_condition="Failure.",
            instruction_digest="8" * 64,
        )


def test_branch_loop_annotations_are_non_negative_and_bounded() -> None:
    values = _branches(BranchResult.SUCCESS, BranchResult.FAILURE)[0].model_dump(mode="python")
    values["failure_loop_count"] = -1
    with pytest.raises(ValidationError):
        PairedBranchOutcome.model_validate(values)

    values["failure_loop_count"] = 1_000_001
    with pytest.raises(ValidationError):
        PairedBranchOutcome.model_validate(values)


def test_analysis_cluster_entry_is_strict_canonical_and_hidden_from_the_policy() -> None:
    entry = AnalysisClusterEntry(
        split=BenchmarkSplit.TRAIN,
        scenario_id="d" * 64,
        family=ScenarioFamily.FORGOTTEN_REQUIREMENT,
        template_lineage_id="lineage-001",
        generator_slot=0,
        cluster_order=0,
        review_digest="e" * 64,
    )

    assert revalidate_analysis_cluster_entry(entry) == entry
    with pytest.raises(ValidationError):
        AnalysisClusterEntry.model_validate({**entry.model_dump(mode="json"), "outcome": "helpful"})
    with pytest.raises(ValidationError):
        AnalysisClusterEntry.model_validate(
            {**entry.model_dump(mode="python"), "generator_slot": 5}
        )
    with pytest.raises(ValidationError):
        entry.cluster_order = 1  # type: ignore[misc]
