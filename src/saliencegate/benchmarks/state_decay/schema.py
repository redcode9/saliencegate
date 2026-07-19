from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from saliencegate.domain import ValidityState
from saliencegate.domain.records import (
    ComponentIdentifier,
    NonNegativeInt,
    PositiveInt,
    Sha256Digest,
)

STATE_DECAY_SCENARIO_SCHEMA_VERSION: Literal["state-decay-scenario/v1"] = "state-decay-scenario/v1"
GENERATOR_VERSION: Literal["v1"] = "v1"

BenchmarkText = Annotated[str, StringConstraints(min_length=1, max_length=2_048)]


class ScenarioFamily(StrEnum):
    FORGOTTEN_REQUIREMENT = "forgotten_requirement"
    STABLE_ENVIRONMENT_FACT = "stable_environment_fact"
    FAILED_PRIOR_ATTEMPT = "failed_prior_attempt"
    RETAINED_DIAGNOSIS = "retained_diagnosis"
    NEGLECTED_SUBGOAL = "neglected_subgoal"
    STALE_MEMORY = "stale_memory"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    IRREVERSIBLE_ACTION = "irreversible_action"


class InterventionLabel(StrEnum):
    INTERVENE = "intervene"
    SILENCE = "silence"


class ContinuationBranch(StrEnum):
    REMINDER = "reminder"
    SILENCE = "silence"


class ContinuationOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class TrajectoryEvent(_StrictModel):
    step: PositiveInt
    source_id: ComponentIdentifier
    statement: BenchmarkText


class MemorySourceRef(_StrictModel):
    source_id: ComponentIdentifier
    source_step: PositiveInt


class CandidateMemory(_StrictModel):
    memory_id: ComponentIdentifier
    statement: BenchmarkText
    source_refs: Annotated[tuple[MemorySourceRef, ...], Field(min_length=1, max_length=8)]
    revision: PositiveInt
    validity: ValidityState
    validity_step: PositiveInt | None
    recorded_step: PositiveInt

    @model_validator(mode="after")
    def validity_transition_is_explicit(self) -> Self:
        if self.validity is ValidityState.ACTIVE:
            if self.validity_step is not None:
                raise ValueError("active memory cannot have a validity transition step")
        elif self.validity_step is None:
            raise ValueError("inactive memory requires a validity transition step")
        return self


class Pivot(_StrictModel):
    step: PositiveInt
    source_id: ComponentIdentifier
    statement: BenchmarkText


class AllowedAction(_StrictModel):
    action_id: ComponentIdentifier
    statement: BenchmarkText


class OracleCriteria(_StrictModel):
    required_action_id: ComponentIdentifier
    success_condition: BenchmarkText
    failure_condition: BenchmarkText


class EvidenceCriteria(_StrictModel):
    decisive_source_ids: Annotated[tuple[ComponentIdentifier, ...], Field(max_length=16)] = ()
    decisive_memory_ids: Annotated[tuple[ComponentIdentifier, ...], Field(max_length=16)] = ()

    @model_validator(mode="after")
    def decisive_evidence_is_nonempty_and_unique(self) -> Self:
        if not self.decisive_source_ids and not self.decisive_memory_ids:
            raise ValueError("decisive evidence cannot be empty")
        if len(set(self.decisive_source_ids)) != len(self.decisive_source_ids) or len(
            set(self.decisive_memory_ids)
        ) != len(self.decisive_memory_ids):
            raise ValueError("decisive evidence identifiers must be unique")
        return self


class PairedContinuation(_StrictModel):
    branch: ContinuationBranch
    selected_action_id: ComponentIdentifier
    outcome: ContinuationOutcome
    explanation: BenchmarkText
    evidence_source_ids: Annotated[tuple[ComponentIdentifier, ...], Field(max_length=16)] = ()
    evidence_memory_ids: Annotated[tuple[ComponentIdentifier, ...], Field(max_length=16)] = ()

    @model_validator(mode="after")
    def evidence_identifiers_are_unique(self) -> Self:
        if len(set(self.evidence_source_ids)) != len(self.evidence_source_ids) or len(
            set(self.evidence_memory_ids)
        ) != len(self.evidence_memory_ids):
            raise ValueError("continuation evidence identifiers must be unique")
        return self


class StateDecayScenario(_StrictModel):
    schema_version: Literal["state-decay-scenario/v1"] = STATE_DECAY_SCENARIO_SCHEMA_VERSION
    generator_version: Literal["v1"]
    seed: NonNegativeInt
    scenario_id: Sha256Digest
    template_lineage_id: ComponentIdentifier
    family: ScenarioFamily
    trajectory_prefix: Annotated[tuple[TrajectoryEvent, ...], Field(min_length=1, max_length=32)]
    candidate_memories: Annotated[tuple[CandidateMemory, ...], Field(min_length=1, max_length=16)]
    pivot: Pivot
    allowed_actions: Annotated[tuple[AllowedAction, ...], Field(min_length=2, max_length=16)]
    label: InterventionLabel
    oracle: OracleCriteria
    evidence_criteria: EvidenceCriteria
    paired_continuations: Annotated[
        tuple[PairedContinuation, ...], Field(min_length=2, max_length=2)
    ]

    @model_validator(mode="after")
    def references_are_unique_resolved_and_not_from_the_future(self) -> Self:
        prefix_steps = tuple(event.step for event in self.trajectory_prefix)
        if tuple(sorted(prefix_steps)) != prefix_steps or len(set(prefix_steps)) != len(
            prefix_steps
        ):
            raise ValueError("trajectory steps must be strictly increasing")
        if prefix_steps[-1] >= self.pivot.step:
            raise ValueError("trajectory prefix must precede the pivot")

        source_steps = {event.source_id: event.step for event in self.trajectory_prefix}
        if len(source_steps) != len(self.trajectory_prefix) or self.pivot.source_id in source_steps:
            raise ValueError("scenario source identifiers must be unique")
        source_steps[self.pivot.source_id] = self.pivot.step

        memories = {memory.memory_id: memory for memory in self.candidate_memories}
        if len(memories) != len(self.candidate_memories):
            raise ValueError("memory identifiers must be unique")
        for memory in self.candidate_memories:
            refs = {(ref.source_id, ref.source_step) for ref in memory.source_refs}
            if len(refs) != len(memory.source_refs):
                raise ValueError("memory source references must be unique")
            for reference in memory.source_refs:
                if source_steps.get(reference.source_id) != reference.source_step:
                    raise ValueError("memory source reference does not resolve")
                if reference.source_step > memory.recorded_step:
                    raise ValueError("memory source reference is from the future")
            if memory.recorded_step > self.pivot.step:
                raise ValueError("candidate memory is from the future")
            if memory.validity_step is not None and not (
                memory.recorded_step <= memory.validity_step <= self.pivot.step
            ):
                raise ValueError("memory validity transition is out of range")

        action_ids = {action.action_id for action in self.allowed_actions}
        if len(action_ids) != len(self.allowed_actions):
            raise ValueError("allowed action identifiers must be unique")
        if self.oracle.required_action_id not in action_ids:
            raise ValueError("oracle action must be allowed")

        for source_id in self.evidence_criteria.decisive_source_ids:
            if source_id not in source_steps or source_steps[source_id] > self.pivot.step:
                raise ValueError("decisive source does not resolve before the pivot")
        for memory_id in self.evidence_criteria.decisive_memory_ids:
            decisive_memory = memories.get(memory_id)
            if (
                decisive_memory is None
                or decisive_memory.recorded_step > self.pivot.step
                or decisive_memory.validity is not ValidityState.ACTIVE
            ):
                raise ValueError("decisive memory is unavailable at the pivot")

        branches = {continuation.branch for continuation in self.paired_continuations}
        if branches != {ContinuationBranch.REMINDER, ContinuationBranch.SILENCE}:
            raise ValueError("paired continuation branches must be complete and unique")
        for continuation in self.paired_continuations:
            if continuation.selected_action_id not in action_ids:
                raise ValueError("continuation selected an unavailable action")
            for source_id in continuation.evidence_source_ids:
                if source_id not in source_steps or source_steps[source_id] > self.pivot.step:
                    raise ValueError("continuation source is unavailable at the pivot")
            for memory_id in continuation.evidence_memory_ids:
                evidence_memory = memories.get(memory_id)
                if (
                    evidence_memory is None
                    or evidence_memory.recorded_step > self.pivot.step
                    or evidence_memory.validity is not ValidityState.ACTIVE
                ):
                    raise ValueError("continuation memory is unavailable at the pivot")
        return self


__all__ = [
    "GENERATOR_VERSION",
    "STATE_DECAY_SCENARIO_SCHEMA_VERSION",
    "AllowedAction",
    "CandidateMemory",
    "ContinuationBranch",
    "ContinuationOutcome",
    "EvidenceCriteria",
    "InterventionLabel",
    "MemorySourceRef",
    "OracleCriteria",
    "PairedContinuation",
    "Pivot",
    "ScenarioFamily",
    "StateDecayScenario",
    "TrajectoryEvent",
]
