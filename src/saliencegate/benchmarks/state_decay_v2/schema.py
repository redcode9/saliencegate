from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from saliencegate.domain import ValidityState
from saliencegate.domain.records import (
    ComponentIdentifier,
    PositiveInt,
    PositiveSigned64Offset,
    Sha256Digest,
    Signed64Offset,
)

POLICY_VIEW_SCHEMA_VERSION: Literal["state-decay-policy-view/v2"] = "state-decay-policy-view/v2"
SUITE_ID: Literal["state-decay-v2"] = "state-decay-v2"
SUITE_VERSION: Literal["v2"] = "v2"
MAX_POLICY_TEXT_CHARACTERS = 2_048
MAX_POLICY_TEXT_UTF8_BYTES = 4_096


def _bounded_policy_text(value: str) -> str:
    if type(value) is not str:
        raise ValueError("text subclasses are not accepted")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError("text must be valid UTF-8") from error
    if len(encoded) > MAX_POLICY_TEXT_UTF8_BYTES:
        raise ValueError("text exceeds its UTF-8 byte bound")
    return value


PolicyText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_POLICY_TEXT_CHARACTERS),
    AfterValidator(_bounded_policy_text),
]


class BenchmarkSplit(StrEnum):
    TRAIN = "train"
    DEVELOPMENT = "development"
    LOCKED = "locked"
    DIAGNOSTIC = "diagnostic"


class ArtifactRole(StrEnum):
    POLICY_VIEW = "policy-view"
    ORACLE_VAULT = "oracle-vault"
    ANALYSIS_CLUSTER_MAP = "analysis-cluster-map"


class ScenarioFamily(StrEnum):
    FORGOTTEN_REQUIREMENT = "forgotten_requirement"
    FAILED_PRIOR_ATTEMPT = "failed_prior_attempt"
    NEGLECTED_SUBGOAL = "neglected_subgoal"
    STALE_MEMORY = "stale_memory"
    STABLE_ENVIRONMENT_FACT = "stable_environment_fact"
    RETAINED_DIAGNOSIS = "retained_diagnosis"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    IRREVERSIBLE_ACTION = "irreversible_action"


class ScenarioOutcome(StrEnum):
    HELPFUL = "helpful"
    HARMFUL = "harmful"
    REDUNDANT = "redundant"
    UNRESOLVED = "unresolved"


class ContinuationBranch(StrEnum):
    REMINDER = "reminder"
    SILENCE = "silence"


class BranchResult(StrEnum):
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


class PolicyEvent(_StrictModel):
    event_id: ComponentIdentifier
    sequence: PositiveSigned64Offset
    action_step: Signed64Offset
    statement: PolicyText


class PolicyEvidenceReference(_StrictModel):
    event_id: ComponentIdentifier
    event_sequence: PositiveSigned64Offset


class PolicyCandidateMemory(_StrictModel):
    memory_id: ComponentIdentifier
    revision: PositiveInt
    statement: PolicyText
    evidence_refs: Annotated[
        tuple[PolicyEvidenceReference, ...],
        Field(min_length=1, max_length=16),
    ]
    recorded_sequence: PositiveSigned64Offset
    recorded_action_step: Signed64Offset
    validity: ValidityState
    validity_sequence: PositiveSigned64Offset | None = None
    validity_action_step: Signed64Offset | None = None

    @model_validator(mode="after")
    def validity_and_evidence_are_explicit(self) -> Self:
        reference_keys = tuple(
            (reference.event_id, reference.event_sequence) for reference in self.evidence_refs
        )
        if len(set(reference_keys)) != len(reference_keys):
            raise ValueError("memory evidence references must be unique")
        if self.validity is ValidityState.ACTIVE:
            if self.validity_sequence is not None or self.validity_action_step is not None:
                raise ValueError("active memory cannot carry a validity transition")
            return self
        if self.validity_sequence is None or self.validity_action_step is None:
            raise ValueError("inactive memory requires a complete validity transition")
        if (
            self.validity_sequence < self.recorded_sequence
            or self.validity_action_step < self.recorded_action_step
        ):
            raise ValueError("memory validity transition predates the memory")
        return self


class PolicyPivot(_StrictModel):
    event_id: ComponentIdentifier
    sequence: PositiveSigned64Offset
    action_step: Signed64Offset
    statement: PolicyText


class PolicyAllowedAction(_StrictModel):
    action_id: ComponentIdentifier
    statement: PolicyText


class AdapterMetadata(_StrictModel):
    schema_version: Literal["state-decay-adapter-metadata/v1"] = "state-decay-adapter-metadata/v1"
    adapter_id: ComponentIdentifier
    adapter_version: ComponentIdentifier
    response_profile_id: ComponentIdentifier
    response_profile_digest: Sha256Digest
    capabilities: Annotated[tuple[ComponentIdentifier, ...], Field(max_length=16)] = ()

    @model_validator(mode="after")
    def capabilities_are_unique(self) -> Self:
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("adapter capabilities must be unique")
        return self


class PolicyViewScenario(_StrictModel):
    schema_version: Literal["state-decay-policy-view/v2"] = POLICY_VIEW_SCHEMA_VERSION
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    split: BenchmarkSplit
    scenario_id: Sha256Digest
    trajectory: Annotated[tuple[PolicyEvent, ...], Field(min_length=1, max_length=64)]
    candidate_memories: Annotated[
        tuple[PolicyCandidateMemory, ...],
        Field(min_length=1, max_length=32),
    ]
    pivot: PolicyPivot
    allowed_actions: Annotated[
        tuple[PolicyAllowedAction, ...],
        Field(min_length=2, max_length=16),
    ]
    adapter: AdapterMetadata

    @model_validator(mode="after")
    def references_are_unique_resolved_and_monotone(self) -> Self:
        sequences = tuple(event.sequence for event in self.trajectory)
        action_steps = tuple(event.action_step for event in self.trajectory)
        if sequences != tuple(sorted(sequences)) or len(set(sequences)) != len(sequences):
            raise ValueError("trajectory sequence values must be strictly increasing")
        if action_steps != tuple(sorted(action_steps)):
            raise ValueError("trajectory action steps must be monotone")
        if self.pivot.sequence <= sequences[-1] or self.pivot.action_step < action_steps[-1]:
            raise ValueError("pivot must follow the visible trajectory")

        events = {event.event_id: (event.sequence, event.action_step) for event in self.trajectory}
        if len(events) != len(self.trajectory) or self.pivot.event_id in events:
            raise ValueError("policy event identifiers must be unique")

        memories = {memory.memory_id: memory for memory in self.candidate_memories}
        if len(memories) != len(self.candidate_memories):
            raise ValueError("candidate memory identifiers must be unique")
        for memory in self.candidate_memories:
            if (
                memory.recorded_sequence > sequences[-1]
                or memory.recorded_action_step > action_steps[-1]
            ):
                raise ValueError("candidate memory was recorded outside the visible prefix")
            for reference in memory.evidence_refs:
                resolved = events.get(reference.event_id)
                if resolved is None or resolved[0] != reference.event_sequence:
                    raise ValueError("memory evidence reference does not resolve")
                if (
                    resolved[0] > memory.recorded_sequence
                    or resolved[1] > memory.recorded_action_step
                ):
                    raise ValueError("memory evidence reference is from the future")
            if memory.validity_sequence is not None:
                assert memory.validity_action_step is not None
                if (
                    memory.validity_sequence > self.pivot.sequence
                    or memory.validity_action_step > self.pivot.action_step
                ):
                    raise ValueError("memory validity transition is after the pivot")

        action_ids = tuple(action.action_id for action in self.allowed_actions)
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("allowed action identifiers must be unique")
        return self


__all__ = [
    "MAX_POLICY_TEXT_CHARACTERS",
    "MAX_POLICY_TEXT_UTF8_BYTES",
    "POLICY_VIEW_SCHEMA_VERSION",
    "SUITE_ID",
    "SUITE_VERSION",
    "AdapterMetadata",
    "ArtifactRole",
    "BenchmarkSplit",
    "BranchResult",
    "ContinuationBranch",
    "PolicyAllowedAction",
    "PolicyCandidateMemory",
    "PolicyEvent",
    "PolicyEvidenceReference",
    "PolicyPivot",
    "PolicyText",
    "PolicyViewScenario",
    "ScenarioFamily",
    "ScenarioOutcome",
]
