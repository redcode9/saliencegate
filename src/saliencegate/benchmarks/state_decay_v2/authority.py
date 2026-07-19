from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.benchmarks.state_decay_v2.schema import (
    SUITE_ID,
    SUITE_VERSION,
    BenchmarkSplit,
    BranchResult,
    ContinuationBranch,
    PolicyText,
    ScenarioFamily,
    ScenarioOutcome,
)
from saliencegate.domain import canonical_json
from saliencegate.domain.records import (
    ComponentIdentifier,
    PositiveInt,
    Sha256Digest,
)

ORACLE_VAULT_ENTRY_SCHEMA_VERSION: Literal["state-decay-oracle-vault-entry/v2"] = (
    "state-decay-oracle-vault-entry/v2"
)
ANALYSIS_CLUSTER_ENTRY_SCHEMA_VERSION: Literal["state-decay-analysis-cluster-entry/v2"] = (
    "state-decay-analysis-cluster-entry/v2"
)

BoundedLoopCount = Annotated[int, Field(ge=0, le=1_000_000)]
BoundedContinuationSteps = Annotated[int, Field(ge=1, le=64)]
GeneratorSlot = Annotated[int, Field(ge=0, le=4)]
ClusterOrder = Annotated[int, Field(ge=0, le=1_199)]


class ResolvedProposalLabel(StrEnum):
    INTERVENE = "intervene"
    SILENCE = "silence"


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class AdmissibleTreatmentBinding(_StrictModel):
    schema_version: Literal["admissible-treatment-binding/v1"] = "admissible-treatment-binding/v1"
    fixture_digest: Sha256Digest
    proposal_digest: Sha256Digest
    grounding_receipt_digest: Sha256Digest
    renderer_id: ComponentIdentifier
    renderer_version: ComponentIdentifier
    renderer_digest: Sha256Digest
    rendered_text_digest: Sha256Digest
    evidence_id_set_digest: Sha256Digest
    evidence_revision_set_digest: Sha256Digest


class MemoryRevisionEvidence(_StrictModel):
    memory_id: ComponentIdentifier
    revision: PositiveInt


class DecisiveEvidence(_StrictModel):
    event_ids: Annotated[tuple[ComponentIdentifier, ...], Field(max_length=16)] = ()
    memory_revisions: Annotated[
        tuple[MemoryRevisionEvidence, ...],
        Field(max_length=16),
    ] = ()
    decisive_action_id: ComponentIdentifier

    @model_validator(mode="after")
    def evidence_is_nonempty_and_unique(self) -> Self:
        if not self.event_ids and not self.memory_revisions:
            raise ValueError("decisive evidence cannot be empty")
        if len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("decisive event identifiers must be unique")
        revision_keys = tuple((item.memory_id, item.revision) for item in self.memory_revisions)
        if len(set(revision_keys)) != len(revision_keys):
            raise ValueError("decisive memory revisions must be unique")
        return self


class TerminalVerifierInstruction(_StrictModel):
    schema_version: Literal["terminal-verifier-instruction/v1"] = "terminal-verifier-instruction/v1"
    verifier_id: ComponentIdentifier
    verifier_version: ComponentIdentifier
    maximum_steps: BoundedContinuationSteps
    success_condition: PolicyText
    failure_condition: PolicyText
    instruction_digest: Sha256Digest


class PairedBranchOutcome(_StrictModel):
    schema_version: Literal["paired-branch-outcome/v1"] = "paired-branch-outcome/v1"
    branch: ContinuationBranch
    selected_action_id: ComponentIdentifier
    result: BranchResult
    terminal_state_digest: Sha256Digest
    verifier_receipt_digest: Sha256Digest
    action_count: BoundedContinuationSteps
    repeated_action_count: BoundedLoopCount
    failure_loop_count: BoundedLoopCount

    @model_validator(mode="after")
    def loop_counts_fit_the_bounded_continuation(self) -> Self:
        if (
            self.repeated_action_count > self.action_count
            or self.failure_loop_count > self.action_count
        ):
            raise ValueError("loop annotations exceed the continuation")
        return self


_OUTCOME_TRUTH = {
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


class OracleVaultEntry(_StrictModel):
    schema_version: Literal["state-decay-oracle-vault-entry/v2"] = ORACLE_VAULT_ENTRY_SCHEMA_VERSION
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    split: BenchmarkSplit
    scenario_id: Sha256Digest
    outcome: ScenarioOutcome
    opportunity_positive: bool
    resolved_label: ResolvedProposalLabel | None
    admissible_treatment: AdmissibleTreatmentBinding
    verifier: TerminalVerifierInstruction
    decisive_evidence: DecisiveEvidence
    branches: Annotated[tuple[PairedBranchOutcome, ...], Field(min_length=2, max_length=2)]

    @model_validator(mode="after")
    def paired_branches_match_the_declared_outcome(self) -> Self:
        by_branch = {branch.branch: branch for branch in self.branches}
        if len(by_branch) != 2 or set(by_branch) != {
            ContinuationBranch.REMINDER,
            ContinuationBranch.SILENCE,
        }:
            raise ValueError("paired continuation branches must be complete and unique")
        reminder, silence, positive, resolved = _OUTCOME_TRUTH[self.outcome]
        if (
            by_branch[ContinuationBranch.REMINDER].result is not reminder
            or by_branch[ContinuationBranch.SILENCE].result is not silence
            or self.opportunity_positive is not positive
            or self.resolved_label is not resolved
        ):
            raise ValueError("oracle outcome does not match its causal truth table")
        if any(branch.action_count > self.verifier.maximum_steps for branch in self.branches):
            raise ValueError("branch exceeds the terminal verifier step bound")
        return self


class AnalysisClusterEntry(_StrictModel):
    schema_version: Literal["state-decay-analysis-cluster-entry/v2"] = (
        ANALYSIS_CLUSTER_ENTRY_SCHEMA_VERSION
    )
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    split: BenchmarkSplit
    scenario_id: Sha256Digest
    family: ScenarioFamily
    template_lineage_id: ComponentIdentifier
    generator_slot: GeneratorSlot
    cluster_order: ClusterOrder
    review_digest: Sha256Digest


def revalidate_oracle_vault_entry(value: OracleVaultEntry) -> OracleVaultEntry:
    """Return a detached authority only after canonical byte validation."""

    if type(value) is not OracleVaultEntry:
        raise ValueError("oracle vault entry has an invalid type")
    return OracleVaultEntry.model_validate_json(canonical_json(value))


def revalidate_analysis_cluster_entry(value: AnalysisClusterEntry) -> AnalysisClusterEntry:
    """Return a detached cluster entry only after canonical byte validation."""

    if type(value) is not AnalysisClusterEntry:
        raise ValueError("analysis cluster entry has an invalid type")
    return AnalysisClusterEntry.model_validate_json(canonical_json(value))


__all__ = [
    "ANALYSIS_CLUSTER_ENTRY_SCHEMA_VERSION",
    "ORACLE_VAULT_ENTRY_SCHEMA_VERSION",
    "AdmissibleTreatmentBinding",
    "AnalysisClusterEntry",
    "BoundedContinuationSteps",
    "BoundedLoopCount",
    "ClusterOrder",
    "DecisiveEvidence",
    "GeneratorSlot",
    "MemoryRevisionEvidence",
    "OracleVaultEntry",
    "PairedBranchOutcome",
    "ResolvedProposalLabel",
    "TerminalVerifierInstruction",
    "revalidate_analysis_cluster_entry",
    "revalidate_oracle_vault_entry",
]
