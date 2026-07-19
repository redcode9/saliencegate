from __future__ import annotations

import hmac
import re
import unicodedata
from collections.abc import Mapping
from enum import StrEnum
from itertools import pairwise, product
from types import MappingProxyType
from typing import Annotated, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from saliencegate.benchmarks.state_decay_v2.authority import (
    ANALYSIS_CLUSTER_ENTRY_SCHEMA_VERSION,
    ORACLE_VAULT_ENTRY_SCHEMA_VERSION,
)
from saliencegate.benchmarks.state_decay_v2.config import (
    GENERATED_INTEGER_MAX,
    GENERATION_CONTRACT,
    PUBLIC_GENERATION_SEED,
    CounterbalanceAxis,
    SeedPurpose,
    derive_seed,
)
from saliencegate.benchmarks.state_decay_v2.protocol import (
    LINEAGE_REVIEW_PROTOCOL,
    NUISANCE_FEATURE_INVENTORY,
    derive_independent_lineage_seed,
    independent_lineage_seed_commitment,
)
from saliencegate.benchmarks.state_decay_v2.schema import (
    POLICY_VIEW_SCHEMA_VERSION,
    SUITE_ID,
    SUITE_VERSION,
    AdapterMetadata,
    BenchmarkSplit,
    ScenarioFamily,
    ScenarioOutcome,
)
from saliencegate.domain import (
    ClaimKind,
    EventPhase,
    EventType,
    JsonObject,
    SignalType,
    ValidityState,
    canonical_json,
    length_prefixed_sha256,
    trace_event_payload_is_bounded,
)
from saliencegate.domain.records import (
    ComponentIdentifier,
    JsonPointer,
    PositiveInt,
    PositiveSigned64Offset,
    Sha256Digest,
    Signed64Offset,
)

TRANSITION_GRAPH_DIGEST_DOMAIN: Literal[
    "saliencegate:state-decay-v2:public-review:transition-graph:v1"
] = "saliencegate:state-decay-v2:public-review:transition-graph:v1"
EVIDENCE_TOPOLOGY_DIGEST_DOMAIN: Literal[
    "saliencegate:state-decay-v2:public-review:evidence-topology:v1"
] = "saliencegate:state-decay-v2:public-review:evidence-topology:v1"
SEMANTIC_SIGNATURE_DIGEST_DOMAIN: Literal[
    "saliencegate:state-decay-v2:public-review:semantic-signature:v1"
] = "saliencegate:state-decay-v2:public-review:semantic-signature:v1"
CAUSAL_DELTA_DIGEST_DOMAIN: Literal["saliencegate:state-decay-v2:public-review:causal-delta:v1"] = (
    "saliencegate:state-decay-v2:public-review:causal-delta:v1"
)
GENERATOR_CONFIGURATION_DIGEST_DOMAIN: Literal[
    "saliencegate:state-decay-v2:public-review:generator-configuration:v1"
] = "saliencegate:state-decay-v2:public-review:generator-configuration:v1"
GENERATOR_ALGORITHM_DIGEST_DOMAIN: Literal[
    "saliencegate:state-decay-v2:public-review:generator-algorithm:v1"
] = "saliencegate:state-decay-v2:public-review:generator-algorithm:v1"
PROFILE_CATALOG_DIGEST_DOMAIN: Literal[
    "saliencegate:state-decay-v2:public-review:profile-catalog:v1"
] = "saliencegate:state-decay-v2:public-review:profile-catalog:v1"
SKELETON_PREVIEW_DIGEST_DOMAIN: Literal[
    "saliencegate:state-decay-v2:public-review:skeleton-preview:v1"
] = "saliencegate:state-decay-v2:public-review:skeleton-preview:v1"
CANDIDATE_PACKET_DIGEST_DOMAIN: Literal[
    "saliencegate:state-decay-v2:public-review:candidate-packet:v1"
] = "saliencegate:state-decay-v2:public-review:candidate-packet:v1"
CANDIDATE_REGISTRY_DIGEST_DOMAIN: Literal[
    "saliencegate:state-decay-v2:public-review:candidate-registry:v1"
] = "saliencegate:state-decay-v2:public-review:candidate-registry:v1"
RENDERED_POLICY_DIGEST_DOMAIN: Literal[
    "saliencegate:state-decay-v2:public-review:rendered-policy:v1"
] = "saliencegate:state-decay-v2:public-review:rendered-policy:v1"
SIGNAL_PROFILE_DIGEST_DOMAIN: Literal[
    "saliencegate:state-decay-v2:public-review:signal-profile:v1"
] = "saliencegate:state-decay-v2:public-review:signal-profile:v1"
TRACE_FIXTURE_DIGEST_DOMAIN: Literal[
    "saliencegate:state-decay-v2:public-review:trace-fixture:v1"
] = "saliencegate:state-decay-v2:public-review:trace-fixture:v1"
PUBLIC_ASSERTION_FIXTURE_DIGEST_DOMAIN: Literal[
    "saliencegate:state-decay-v2:public-review:assertion-fixture:v1"
] = "saliencegate:state-decay-v2:public-review:assertion-fixture:v1"

MAX_REVIEW_SAFE_TEXT_UTF8_BYTES: Literal[4_096] = 4_096
PUBLIC_LINEAGE_KEY_PATTERN: Literal[r"^pub-(fr|fp|ns|sm|sf|rd)-([0-2][0-9])$"] = (
    r"^pub-(fr|fp|ns|sm|sf|rd)-([0-2][0-9])$"
)

_PUBLIC_LINEAGE_KEY = re.compile(PUBLIC_LINEAGE_KEY_PATTERN)
_PUBLIC_FAMILY_CODES: Mapping[ScenarioFamily, str] = MappingProxyType(
    {
        ScenarioFamily.FORGOTTEN_REQUIREMENT: "fr",
        ScenarioFamily.FAILED_PRIOR_ATTEMPT: "fp",
        ScenarioFamily.NEGLECTED_SUBGOAL: "ns",
        ScenarioFamily.STALE_MEMORY: "sm",
        ScenarioFamily.STABLE_ENVIRONMENT_FACT: "sf",
        ScenarioFamily.RETAINED_DIAGNOSIS: "rd",
    }
)
_PUBLIC_CODE_FAMILIES: Mapping[str, ScenarioFamily] = MappingProxyType(
    {code: family for family, code in _PUBLIC_FAMILY_CODES.items()}
)
_PUBLIC_FAMILY_SPLITS: Mapping[ScenarioFamily, BenchmarkSplit] = MappingProxyType(
    {
        ScenarioFamily.FORGOTTEN_REQUIREMENT: BenchmarkSplit.TRAIN,
        ScenarioFamily.FAILED_PRIOR_ATTEMPT: BenchmarkSplit.TRAIN,
        ScenarioFamily.NEGLECTED_SUBGOAL: BenchmarkSplit.TRAIN,
        ScenarioFamily.STALE_MEMORY: BenchmarkSplit.TRAIN,
        ScenarioFamily.STABLE_ENVIRONMENT_FACT: BenchmarkSplit.DEVELOPMENT,
        ScenarioFamily.RETAINED_DIAGNOSIS: BenchmarkSplit.DEVELOPMENT,
    }
)
_OUTCOME_LABELS: tuple[str, ...] = tuple(outcome.value for outcome in ScenarioOutcome)
_SEMANTIC_CAUSAL_POINTERS = frozenset(
    {
        "/event_pool/0/statement",
        "/event_pool/1/statement",
        "/event_pool/2/statement",
        "/pivot/statement",
        "/action_pool/0/statement",
        "/action_pool/1/statement",
    }
)
_EVIDENCE_CAUSAL_POINTERS = frozenset({"/memory_pool/0/statement"})
_RENDERED_SEMANTIC_CAUSAL_POINTERS = frozenset(
    {
        "/trajectory/0/statement",
        "/trajectory/1/statement",
        "/trajectory/2/statement",
        "/pivot/statement",
        "/allowed_actions/0/statement",
        "/allowed_actions/1/statement",
    }
)
_RENDERED_EVIDENCE_CAUSAL_POINTERS = frozenset({"/candidate_memories/0/statement"})


def _reject_outcome_label_content(value: object, *, context: str) -> None:
    serialized = canonical_json(value).decode("utf-8").casefold()
    if any(label in serialized for label in _OUTCOME_LABELS):
        raise ValueError(f"{context} contains an outcome label")


def _is_forbidden_code_point(code_point: int) -> bool:
    return (
        0 <= code_point <= 0x1F
        or 0x7F <= code_point <= 0x9F
        or code_point in (0x061C, 0x200E, 0x200F)
        or 0x2028 <= code_point <= 0x2029
        or 0x202A <= code_point <= 0x202E
        or 0x2066 <= code_point <= 0x2069
        or 0xD800 <= code_point <= 0xDFFF
        or 0xFDD0 <= code_point <= 0xFDEF
        or code_point & 0xFFFF in (0xFFFE, 0xFFFF)
    )


def _review_safe_text(value: object) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError("review-safe text is invalid")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("review-safe text is invalid")
    if any(_is_forbidden_code_point(ord(character)) for character in value):
        raise ValueError("review-safe text is invalid")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError:
        raise ValueError("review-safe text is invalid") from None
    if len(encoded) > MAX_REVIEW_SAFE_TEXT_UTF8_BYTES:
        raise ValueError("review-safe text is invalid")
    return value


def validate_review_safe_text(value: object) -> str:
    return _review_safe_text(value)


ReviewSafeText = Annotated[str, BeforeValidator(validate_review_safe_text)]


def _outcome_free_policy_text(value: object) -> str:
    checked = _review_safe_text(value)
    if len(checked) > 2_048:
        raise ValueError("outcome-free policy text is invalid")
    return checked


OutcomeFreePolicyText = Annotated[str, BeforeValidator(_outcome_free_policy_text)]


def _causal_replacement_text(value: object) -> str:
    checked = _outcome_free_policy_text(value)
    if len(checked.encode("utf-8")) > 256:
        raise ValueError("causal replacement text is invalid")
    if any(label in checked.casefold() for label in _OUTCOME_LABELS):
        raise ValueError("causal replacement contains an outcome label")
    return checked


CausalReplacementText = Annotated[str, BeforeValidator(_causal_replacement_text)]


def _rendered_causal_replacement_text(value: object) -> str:
    checked = _outcome_free_policy_text(value)
    if any(label in checked.casefold() for label in _OUTCOME_LABELS):
        raise ValueError("rendered causal replacement contains an outcome label")
    return checked


RenderedCausalReplacementText = Annotated[
    str,
    BeforeValidator(_rendered_causal_replacement_text),
]
PublicGeneratorSlot = Annotated[int, Field(ge=0, le=4)]
GeneratedParameterValue = Annotated[int, Field(ge=0, le=GENERATED_INTEGER_MAX)]
PublicEventPoolIndex = Annotated[int, Field(ge=0, le=7)]
PublicMemoryPoolIndex = Annotated[int, Field(ge=0, le=3)]
PublicAssertionIndex = Annotated[int, Field(ge=0, le=15)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def public_lineage_key(family: ScenarioFamily, index: int) -> str:
    if type(family) is not ScenarioFamily or type(index) is not int or not 0 <= index <= 29:
        raise ValueError("public lineage coordinate is invalid")
    code = _PUBLIC_FAMILY_CODES.get(family)
    if code is None:
        raise ValueError("public lineage coordinate is invalid")
    return f"pub-{code}-{index:02d}"


def parse_public_lineage_key(value: str) -> tuple[ScenarioFamily, int]:
    if type(value) is not str:
        raise ValueError("public lineage key is invalid")
    match = _PUBLIC_LINEAGE_KEY.fullmatch(value)
    if match is None:
        raise ValueError("public lineage key is invalid")
    family = _PUBLIC_CODE_FAMILIES[match.group(1)]
    index = int(match.group(2))
    if public_lineage_key(family, index) != value:
        raise ValueError("public lineage key is invalid")
    return family, index


def _public_lineage_key(value: object) -> str:
    if type(value) is not str:
        raise ValueError("public lineage key is invalid")
    parse_public_lineage_key(value)
    return value


PublicLineageKey = Annotated[str, BeforeValidator(_public_lineage_key)]


def _self_digest(
    value: BaseModel | Mapping[str, object],
    *,
    self_field: str,
    domain: str,
) -> str:
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", exclude={self_field}, warnings=False)
    else:
        try:
            payload = {key: item for key, item in value.items() if key != self_field}
        except Exception:
            raise ValueError("public digest payload is invalid") from None
    return length_prefixed_sha256(canonical_json(payload), domain=domain)


def transition_graph_digest(value: BaseModel | Mapping[str, object]) -> str:
    return _self_digest(
        value,
        self_field="transition_graph_digest",
        domain=TRANSITION_GRAPH_DIGEST_DOMAIN,
    )


def evidence_topology_digest(value: BaseModel | Mapping[str, object]) -> str:
    return _self_digest(
        value,
        self_field="evidence_topology_digest",
        domain=EVIDENCE_TOPOLOGY_DIGEST_DOMAIN,
    )


def semantic_signature_digest(value: BaseModel | Mapping[str, object]) -> str:
    return _self_digest(
        value,
        self_field="semantic_signature_digest",
        domain=SEMANTIC_SIGNATURE_DIGEST_DOMAIN,
    )


def causal_delta_digest(value: BaseModel | Mapping[str, object]) -> str:
    return _self_digest(
        value,
        self_field="causal_delta_digest",
        domain=CAUSAL_DELTA_DIGEST_DOMAIN,
    )


def generator_configuration_digest(value: BaseModel | Mapping[str, object]) -> str:
    return _self_digest(
        value,
        self_field="configuration_digest",
        domain=GENERATOR_CONFIGURATION_DIGEST_DOMAIN,
    )


def generator_algorithm_digest(value: BaseModel | Mapping[str, object]) -> str:
    return _self_digest(
        value,
        self_field="algorithm_digest",
        domain=GENERATOR_ALGORITHM_DIGEST_DOMAIN,
    )


def profile_catalog_digest(value: BaseModel | Mapping[str, object]) -> str:
    return _self_digest(
        value,
        self_field="catalog_digest",
        domain=PROFILE_CATALOG_DIGEST_DOMAIN,
    )


def skeleton_preview_digest(value: BaseModel | Mapping[str, object]) -> str:
    return _self_digest(
        value,
        self_field="preview_digest",
        domain=SKELETON_PREVIEW_DIGEST_DOMAIN,
    )


def candidate_packet_digest(value: BaseModel | Mapping[str, object]) -> str:
    return _self_digest(
        value,
        self_field="candidate_packet_digest",
        domain=CANDIDATE_PACKET_DIGEST_DOMAIN,
    )


def candidate_registry_digest(value: BaseModel | Mapping[str, object]) -> str:
    return _self_digest(
        value,
        self_field="registry_digest",
        domain=CANDIDATE_REGISTRY_DIGEST_DOMAIN,
    )


def rendered_policy_digest(
    value: BaseModel | Mapping[str, object] | bytes,
) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, BaseModel):
        payload = canonical_json(value.model_dump(mode="json", warnings=False))
    else:
        payload = canonical_json(value)
    return length_prefixed_sha256(
        payload,
        domain=RENDERED_POLICY_DIGEST_DOMAIN,
    )


def signal_profile_digest(value: BaseModel | Mapping[str, object]) -> str:
    return _self_digest(
        value,
        self_field="profile_digest",
        domain=SIGNAL_PROFILE_DIGEST_DOMAIN,
    )


def trace_fixture_digest(value: BaseModel | Mapping[str, object]) -> str:
    return _self_digest(
        value,
        self_field="trace_fixture_digest",
        domain=TRACE_FIXTURE_DIGEST_DOMAIN,
    )


class OutcomeFreeEvent(_StrictModel):
    event_id: ComponentIdentifier
    sequence: PositiveSigned64Offset
    action_step: Signed64Offset
    statement: OutcomeFreePolicyText


class OutcomeFreeEvidenceReference(_StrictModel):
    event_id: ComponentIdentifier
    event_sequence: PositiveSigned64Offset


class OutcomeFreeTemplateEvent(_StrictModel):
    event_id: ComponentIdentifier
    statement: OutcomeFreePolicyText


class OutcomeFreeTemplateMemory(_StrictModel):
    memory_id: ComponentIdentifier
    statement: OutcomeFreePolicyText
    evidence_event_ids: Annotated[
        tuple[ComponentIdentifier, ...],
        Field(min_length=3, max_length=3),
    ]
    recorded_event_id: ComponentIdentifier


class OutcomeFreeTemplatePivot(_StrictModel):
    event_id: ComponentIdentifier
    statement: OutcomeFreePolicyText


class OutcomeFreeTemplateAction(_StrictModel):
    action_id: ComponentIdentifier
    statement: OutcomeFreePolicyText


class OutcomeFreeTaskTemplate(_StrictModel):
    event_pool: Annotated[
        tuple[OutcomeFreeTemplateEvent, ...],
        Field(min_length=8, max_length=8),
    ]
    memory_pool: Annotated[
        tuple[OutcomeFreeTemplateMemory, ...],
        Field(min_length=4, max_length=4),
    ]
    pivot: OutcomeFreeTemplatePivot
    action_pool: Annotated[
        tuple[OutcomeFreeTemplateAction, ...],
        Field(min_length=2, max_length=2),
    ]
    adapter: AdapterMetadata

    @model_validator(mode="after")
    def pools_are_fixed_resolved_and_disjoint(self) -> Self:
        event_ids = tuple(event.event_id for event in self.event_pool)
        memory_ids = tuple(memory.memory_id for memory in self.memory_pool)
        action_ids = tuple(action.action_id for action in self.action_pool)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("task template event identifiers must be unique")
        if len(set(memory_ids)) != len(memory_ids):
            raise ValueError("task template memory identifiers must be unique")
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("task template action identifiers must be unique")

        all_ids = (*event_ids, *memory_ids, self.pivot.event_id, *action_ids)
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("task template identifiers must be disjoint")

        first_three_event_ids = event_ids[:3]
        for memory in self.memory_pool:
            if (
                memory.evidence_event_ids != first_three_event_ids
                or memory.recorded_event_id != first_three_event_ids[2]
            ):
                if memory.evidence_event_ids != first_three_event_ids:
                    raise ValueError(
                        "task template memories must resolve to the first three events in order"
                    )
                raise ValueError("task template memory must be recorded at the third event")
        return self


class OutcomeFreeCandidateMemory(_StrictModel):
    memory_id: ComponentIdentifier
    revision: PositiveInt
    statement: OutcomeFreePolicyText
    evidence_refs: Annotated[
        tuple[OutcomeFreeEvidenceReference, ...],
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


class OutcomeFreePivot(_StrictModel):
    event_id: ComponentIdentifier
    sequence: PositiveSigned64Offset
    action_step: Signed64Offset
    statement: OutcomeFreePolicyText


class OutcomeFreeAllowedAction(_StrictModel):
    action_id: ComponentIdentifier
    statement: OutcomeFreePolicyText


class OutcomeFreeTaskSkeleton(_StrictModel):
    trajectory: Annotated[tuple[OutcomeFreeEvent, ...], Field(min_length=1, max_length=64)]
    candidate_memories: Annotated[
        tuple[OutcomeFreeCandidateMemory, ...],
        Field(min_length=1, max_length=32),
    ]
    pivot: OutcomeFreePivot
    allowed_actions: Annotated[
        tuple[OutcomeFreeAllowedAction, ...],
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


class CausalTextReplacement(_StrictModel):
    template_pointer: JsonPointer
    replacement: CausalReplacementText


class PublicCausalExposure(StrEnum):
    GUIDANCE_APPLIED = "guidance_applied"
    BASELINE_CONTINUED = "baseline_continued"


class PublicTerminalState(StrEnum):
    GOAL_REACHED = "goal_reached"
    GOAL_NOT_REACHED = "goal_not_reached"


class PublicCausalFactor(_StrictModel):
    factor_id: ComponentIdentifier
    true_description: ReviewSafeText
    false_description: ReviewSafeText

    @model_validator(mode="after")
    def factor_is_outcome_free(self) -> Self:
        serialized = " ".join(
            (self.factor_id, self.true_description, self.false_description)
        ).casefold()
        if any(label in serialized for label in _OUTCOME_LABELS):
            raise ValueError("causal factor contains an outcome label")
        if self.true_description == self.false_description:
            raise ValueError("causal factor descriptions must differ")
        return self


class PublicCausalFactorValue(_StrictModel):
    factor_id: ComponentIdentifier
    value: bool


class CausalSemanticDelta(_StrictModel):
    schema_version: Literal["state-decay-v2-public-causal-semantic-delta/v1"] = (
        "state-decay-v2-public-causal-semantic-delta/v1"
    )
    delta_index: Annotated[int, Field(ge=0, le=3)]
    delta_id: ComponentIdentifier
    family: ScenarioFamily
    lineage_registry_key: PublicLineageKey
    factor_values: Annotated[
        tuple[PublicCausalFactorValue, ...],
        Field(min_length=2, max_length=2),
    ]
    semantic_replacements: Annotated[
        tuple[CausalTextReplacement, ...],
        Field(max_length=4),
    ] = ()
    evidence_replacements: Annotated[
        tuple[CausalTextReplacement, ...],
        Field(max_length=4),
    ] = ()
    causal_delta_digest: Sha256Digest

    @model_validator(mode="after")
    def delta_is_local_outcome_free_and_self_attesting(self) -> Self:
        family, _ = parse_public_lineage_key(self.lineage_registry_key)
        if family is not self.family:
            raise ValueError("causal delta family and public lineage key do not agree")
        if any(label in self.delta_id.casefold() for label in _OUTCOME_LABELS):
            raise ValueError("causal delta identifier contains an outcome label")
        factor_ids = tuple(item.factor_id for item in self.factor_values)
        if len(set(factor_ids)) != 2:
            raise ValueError("causal delta factor assignments must be unique")
        replacements = (*self.semantic_replacements, *self.evidence_replacements)
        if not 1 <= len(replacements) <= 4:
            raise ValueError("causal delta must contain between one and four replacements")
        targets = tuple(replacement.template_pointer for replacement in replacements)
        if len(set(targets)) != len(targets):
            raise ValueError("causal delta replacement targets must be unique and disjoint")
        if any(
            replacement.template_pointer not in _SEMANTIC_CAUSAL_POINTERS
            for replacement in self.semantic_replacements
        ):
            raise ValueError("causal delta semantic replacement pointer is not allowed")
        if any(
            replacement.template_pointer not in _EVIDENCE_CAUSAL_POINTERS
            for replacement in self.evidence_replacements
        ):
            raise ValueError("causal delta evidence replacement pointer is not allowed")
        if not hmac.compare_digest(self.causal_delta_digest, causal_delta_digest(self)):
            raise ValueError("causal delta digest does not match")
        return self


class PublicGeneratorOperation(StrEnum):
    CANDIDATE_TO_PREVIEW = "candidate_to_preview"
    PREVIEW_TO_SCENARIO_ID = "preview_to_scenario_id"
    BALANCED_OUTCOME_ALLOCATION = "balanced_outcome_allocation"
    CAUSAL_DELTA_SELECTION = "causal_delta_selection"
    ROLE_NEUTRAL_COMPOSITION = "role_neutral_composition"


class PublicGeneratorStep(_StrictModel):
    position: Annotated[int, Field(ge=0, le=4)]
    operation: PublicGeneratorOperation
    operator_id: ComponentIdentifier
    operator_version: ComponentIdentifier


class PublicGeneratorAlgorithm(_StrictModel):
    schema_version: Literal["state-decay-v2-public-generator-algorithm/v1"] = (
        "state-decay-v2-public-generator-algorithm/v1"
    )
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    generator_version: Literal["state-decay-v2-public-generator/v1"] = (
        "state-decay-v2-public-generator/v1"
    )
    generation_contract_digest: Sha256Digest
    generator_configuration_digest: Sha256Digest
    profile_catalog_digest: Sha256Digest
    steps: Annotated[tuple[PublicGeneratorStep, ...], Field(min_length=5, max_length=5)]
    semantic_pointer_allowlist: Annotated[
        tuple[JsonPointer, ...],
        Field(min_length=1, max_length=32),
    ]
    evidence_pointer_allowlist: Annotated[
        tuple[JsonPointer, ...],
        Field(min_length=1, max_length=32),
    ]
    causal_delta_digest_domain: Literal[
        "saliencegate:state-decay-v2:public-review:causal-delta:v1"
    ] = CAUSAL_DELTA_DIGEST_DOMAIN
    slot_materialization_rule: Literal["prefix-pools-global-profile-and-ascii-padding/v1"] = (
        "prefix-pools-global-profile-and-ascii-padding/v1"
    )
    parameter_rendering_rule: Literal["pivot-ordered-ascii-parameter-clause-before-padding/v1"] = (
        "pivot-ordered-ascii-parameter-clause-before-padding/v1"
    )
    decisive_evidence_rule: Literal["selected-prefixes-and-logical-action-zero/v1"] = (
        "selected-prefixes-and-logical-action-zero/v1"
    )
    delta_rendering_rule: Literal[
        "stable-template-pointer-equal-utf8-replacement-plus-global-padding/v1"
    ] = "stable-template-pointer-equal-utf8-replacement-plus-global-padding/v1"
    signal_materialization_rule: Literal["closed-composite-to-attested-trace-fixture/v1"] = (
        "closed-composite-to-attested-trace-fixture/v1"
    )
    signal_evaluation_rule: Literal[
        "repository-normalized-final-boundary-four-real-five-reference/v1"
    ] = "repository-normalized-final-boundary-four-real-five-reference/v1"
    outcome_derivation_rule: Literal["rendered-policy-digest-bounded-state-machine/v1"] = (
        "rendered-policy-digest-bounded-state-machine/v1"
    )
    algorithm_digest: Sha256Digest

    @model_validator(mode="after")
    def algorithm_is_global_ordered_and_self_attesting(self) -> Self:
        if self.generation_contract_digest != GENERATION_CONTRACT.contract_digest:
            raise ValueError("generator algorithm generation contract does not match")
        expected_operations = tuple(PublicGeneratorOperation)
        if (
            tuple(step.position for step in self.steps) != tuple(range(5))
            or tuple(step.operation for step in self.steps) != expected_operations
        ):
            raise ValueError("generator steps are not canonical")
        operator_ids = tuple(step.operator_id for step in self.steps)
        if len(set(operator_ids)) != len(operator_ids):
            raise ValueError("generator step operator identifiers must be unique")

        semantic_pointers = self.semantic_pointer_allowlist
        evidence_pointers = self.evidence_pointer_allowlist
        if (
            len(set(semantic_pointers)) != len(semantic_pointers)
            or len(set(evidence_pointers)) != len(evidence_pointers)
            or set(semantic_pointers) & set(evidence_pointers)
        ):
            raise ValueError("generator pointer allowlists must be unique and disjoint")
        if any(pointer not in _SEMANTIC_CAUSAL_POINTERS for pointer in semantic_pointers):
            raise ValueError("generator semantic pointer allowlist is not stable and closed")
        if any(pointer not in _EVIDENCE_CAUSAL_POINTERS for pointer in evidence_pointers):
            raise ValueError("generator evidence pointer allowlist is not stable and closed")
        if not hmac.compare_digest(self.algorithm_digest, generator_algorithm_digest(self)):
            raise ValueError("generator algorithm digest does not match")
        return self


class PublicGeneratorConfiguration(_StrictModel):
    schema_version: Literal["state-decay-v2-public-generator-configuration/v1"] = (
        "state-decay-v2-public-generator-configuration/v1"
    )
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    generator_version: Literal["state-decay-v2-public-generator/v1"] = (
        "state-decay-v2-public-generator/v1"
    )
    generation_contract_digest: Sha256Digest
    visible_splits: Annotated[tuple[BenchmarkSplit, ...], Field(min_length=2, max_length=2)]
    visible_families: Annotated[tuple[ScenarioFamily, ...], Field(min_length=6, max_length=6)]
    lineages_per_family: Literal[30] = 30
    generator_slots_per_lineage: Literal[5] = 5
    candidate_count: Literal[180] = 180
    preview_count: Literal[900] = 900
    maximum_review_text_utf8_bytes: Literal[4_096] = MAX_REVIEW_SAFE_TEXT_UTF8_BYTES
    legacy_repetition_window_events: Literal[8] = 8
    policy_schema_version: Literal["state-decay-policy-view/v2"] = POLICY_VIEW_SCHEMA_VERSION
    oracle_schema_version: Literal["state-decay-oracle-vault-entry/v2"] = (
        ORACLE_VAULT_ENTRY_SCHEMA_VERSION
    )
    analysis_schema_version: Literal["state-decay-analysis-cluster-entry/v2"] = (
        ANALYSIS_CLUSTER_ENTRY_SCHEMA_VERSION
    )
    nuisance_inventory_digest: Sha256Digest
    configuration_digest: Sha256Digest

    @model_validator(mode="after")
    def configuration_is_complete_and_self_attesting(self) -> Self:
        if self.generation_contract_digest != GENERATION_CONTRACT.contract_digest:
            raise ValueError("generator configuration generation contract does not match")
        if self.visible_splits != (BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT):
            raise ValueError("generator configuration public splits are not canonical")
        if self.visible_families != tuple(_PUBLIC_FAMILY_CODES):
            raise ValueError("generator configuration visible families are not canonical")
        if self.nuisance_inventory_digest != NUISANCE_FEATURE_INVENTORY.inventory_digest:
            raise ValueError("generator configuration nuisance inventory does not match")
        if not hmac.compare_digest(
            self.configuration_digest,
            generator_configuration_digest(self),
        ):
            raise ValueError("generator configuration digest does not match")
        return self


class PublicCounterbalanceProfile(_StrictModel):
    profile_id: ComponentIdentifier
    allowed_action_order: tuple[Literal[0, 1], Literal[0, 1]]
    decisive_action_position: Literal[0, 1]
    memory_validity: ValidityState
    include_validity_transition: bool

    @model_validator(mode="after")
    def action_order_and_validity_are_canonical(self) -> Self:
        if self.allowed_action_order not in ((0, 1), (1, 0)):
            raise ValueError("counterbalance action order is invalid")
        if self.allowed_action_order[self.decisive_action_position] != 0:
            raise ValueError("counterbalance decisive action position is inconsistent")
        transition_required = self.memory_validity is not ValidityState.ACTIVE
        if self.include_validity_transition is not transition_required:
            raise ValueError("counterbalance validity transition is inconsistent")
        return self


class PublicParameterValue(_StrictModel):
    parameter_id: ComponentIdentifier
    value: GeneratedParameterValue


class PublicParameterProfile(_StrictModel):
    profile_id: ComponentIdentifier
    allowed_values: Annotated[tuple[PublicParameterValue, ...], Field(min_length=1, max_length=16)]

    @model_validator(mode="after")
    def parameter_ids_are_unique(self) -> Self:
        identifiers = tuple(item.parameter_id for item in self.allowed_values)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("parameter profile identifiers must be unique")
        return self


class PublicStructuralProfile(_StrictModel):
    profile_id: ComponentIdentifier
    trajectory_event_count: Annotated[int, Field(ge=3, le=8)]
    candidate_memory_count: Annotated[int, Field(ge=1, le=4)]
    allowed_action_count: Literal[2] = 2


class PublicIntegerProfile(_StrictModel):
    profile_id: ComponentIdentifier
    sequence_start: Annotated[int, Field(ge=1, le=GENERATED_INTEGER_MAX)]
    sequence_stride: Annotated[int, Field(ge=1, le=GENERATED_INTEGER_MAX)]
    action_step_start: Annotated[int, Field(ge=0, le=GENERATED_INTEGER_MAX)]
    action_step_stride: Annotated[int, Field(ge=1, le=GENERATED_INTEGER_MAX)]
    memory_revision: Annotated[int, Field(ge=1, le=GENERATED_INTEGER_MAX)]


class PublicEvidenceProfile(_StrictModel):
    profile_id: ComponentIdentifier
    evidence_reference_count: Annotated[int, Field(ge=1, le=12)]
    decisive_event_count: Annotated[int, Field(ge=0, le=16)]
    decisive_memory_count: Annotated[int, Field(ge=0, le=16)]

    @model_validator(mode="after")
    def decisive_evidence_is_nonempty(self) -> Self:
        if self.decisive_event_count + self.decisive_memory_count == 0:
            raise ValueError("evidence profile decisive evidence is empty")
        return self


class PublicSignalFixtureVariant(StrEnum):
    FAILED_TEST_CONFLICT_MISSING_CONSTRAINT = "failed_test_conflict_missing_constraint"
    REPEATED_ACTION_SCOPE_SHIFT_IRREVERSIBLE = "repeated_action_scope_shift_irreversible"
    STAGNANT_CONFLICTING_ASSERTIONS = "stagnant_conflicting_assertions"
    REPEATED_FAILURE_SUPERSEDED_CONSTRAINT = "repeated_failure_superseded_constraint"
    REPEATED_ACTION_SCOPE_SHIFT_STAGNATION = "repeated_action_scope_shift_stagnation"


_SIGNAL_PROFILE_SHAPES: Mapping[
    PublicSignalFixtureVariant,
    tuple[tuple[SignalType, int], ...],
] = MappingProxyType(
    {
        PublicSignalFixtureVariant.FAILED_TEST_CONFLICT_MISSING_CONSTRAINT: (
            (SignalType.CONFLICT, 1_000_000),
            (SignalType.STALE_CONSTRAINT, 1_000_000),
            (SignalType.TEST_FAILURE, 1_000_000),
        ),
        PublicSignalFixtureVariant.REPEATED_ACTION_SCOPE_SHIFT_IRREVERSIBLE: (
            (SignalType.CONTEXT_SHIFT, 500_000),
            (SignalType.IRREVERSIBLE_ACTION, 1_000_000),
            (SignalType.REPEATED_ACTION, 1_000_000),
        ),
        PublicSignalFixtureVariant.STAGNANT_CONFLICTING_ASSERTIONS: (
            (SignalType.CONFLICT, 1_000_000),
            (SignalType.STAGNATION, 500_000),
        ),
        PublicSignalFixtureVariant.REPEATED_FAILURE_SUPERSEDED_CONSTRAINT: (
            (SignalType.REPEATED_FAILURE, 1_000_000),
            (SignalType.STALE_CONSTRAINT, 750_000),
            (SignalType.TOOL_ERROR, 1_000_000),
        ),
        PublicSignalFixtureVariant.REPEATED_ACTION_SCOPE_SHIFT_STAGNATION: (
            (SignalType.CONTEXT_SHIFT, 500_000),
            (SignalType.REPEATED_ACTION, 1_000_000),
            (SignalType.STAGNATION, 625_000),
        ),
    }
)


class PublicExpectedMemoryEvidence(_StrictModel):
    memory_pool_index: PublicMemoryPoolIndex
    revision: PositiveInt


class PublicExpectedAssertionEvidence(_StrictModel):
    binding_event_pool_index: PublicEventPoolIndex
    assertion_index: PublicAssertionIndex


class PublicExpectedDetectorEvidence(_StrictModel):
    event_pool_indices: Annotated[
        tuple[PublicEventPoolIndex, ...],
        Field(max_length=16),
    ]
    binding_event_pool_indices: Annotated[
        tuple[PublicEventPoolIndex, ...],
        Field(max_length=16),
    ]
    memory_references: Annotated[
        tuple[PublicExpectedMemoryEvidence, ...],
        Field(max_length=16),
    ]
    assertion_references: Annotated[
        tuple[PublicExpectedAssertionEvidence, ...],
        Field(max_length=16),
    ]

    @model_validator(mode="after")
    def evidence_is_nonempty_unique_and_canonical(self) -> Self:
        if not (
            self.event_pool_indices
            or self.binding_event_pool_indices
            or self.memory_references
            or self.assertion_references
        ):
            raise ValueError("expected detector evidence is empty")
        if self.event_pool_indices != tuple(
            sorted(set(self.event_pool_indices))
        ) or self.binding_event_pool_indices != tuple(sorted(set(self.binding_event_pool_indices))):
            raise ValueError("expected detector evidence indices are not canonical and unique")
        memory_keys = tuple(
            (item.memory_pool_index, item.revision) for item in self.memory_references
        )
        assertion_keys = tuple(
            (item.binding_event_pool_index, item.assertion_index)
            for item in self.assertion_references
        )
        if memory_keys != tuple(sorted(set(memory_keys))) or assertion_keys != tuple(
            sorted(set(assertion_keys))
        ):
            raise ValueError("expected detector evidence references are not canonical and unique")
        if not set(self.binding_event_pool_indices).issubset(self.event_pool_indices):
            raise ValueError("expected binding evidence must also cite its event")
        if any(
            item.binding_event_pool_index not in self.binding_event_pool_indices
            for item in self.assertion_references
        ):
            raise ValueError("expected assertion evidence does not cite its binding")
        return self


class PublicExpectedSignal(_StrictModel):
    signal_type: SignalType
    strength_ppm: Annotated[int, Field(ge=1, le=1_000_000)]
    evidence: PublicExpectedDetectorEvidence

    @model_validator(mode="after")
    def strength_matches_the_frozen_reference_predicate(self) -> Self:
        if self.signal_type is SignalType.CONTEXT_SHIFT:
            valid = (
                500_000 <= self.strength_ppm <= 1_000_000
                and (self.strength_ppm - 500_000) % 100_000 == 0
            )
        elif self.signal_type is SignalType.STALE_CONSTRAINT:
            valid = self.strength_ppm in (750_000, 1_000_000)
        elif self.signal_type is SignalType.STAGNATION:
            valid = self.strength_ppm in (
                500_000,
                625_000,
                750_000,
                875_000,
                1_000_000,
            )
        else:
            valid = self.strength_ppm == 1_000_000
        if not valid:
            raise ValueError("expected signal strength does not match its reference predicate")
        return self

    @model_validator(mode="after")
    def evidence_matches_the_frozen_reference_predicate(self) -> Self:
        events = self.evidence.event_pool_indices
        bindings = self.evidence.binding_event_pool_indices
        memories = self.evidence.memory_references
        assertions = self.evidence.assertion_references
        if self.signal_type in (SignalType.TOOL_ERROR, SignalType.TEST_FAILURE):
            valid = len(events) == 1 and not bindings and not memories and not assertions
        elif self.signal_type is SignalType.REPEATED_ACTION:
            valid = len(events) == 2 and not bindings and not memories and not assertions
        elif self.signal_type is SignalType.REPEATED_FAILURE:
            valid = len(events) == 4 and not bindings and not memories and not assertions
        elif self.signal_type is SignalType.CONTEXT_SHIFT:
            valid = len(events) >= 2 and len(bindings) >= 2 and bool(memories) and not assertions
        elif self.signal_type is SignalType.STALE_CONSTRAINT:
            valid = len(events) >= 1 and len(bindings) >= 1 and bool(memories) and not assertions
        elif self.signal_type is SignalType.STAGNATION:
            valid = 4 <= len(events) <= 8 and bindings == events and not memories and not assertions
        elif self.signal_type is SignalType.IRREVERSIBLE_ACTION:
            valid = len(events) == 1 and bindings == events and not memories and not assertions
        else:
            valid = (
                self.signal_type is SignalType.CONFLICT
                and len(events) == 1
                and bindings == events
                and not memories
                and len(assertions) >= 2
            )
        if not valid:
            raise ValueError("expected signal evidence does not match its reference predicate")
        return self


class PublicSignalProfile(_StrictModel):
    profile_id: ComponentIdentifier
    fixture_variant: PublicSignalFixtureVariant
    expected_signals: Annotated[
        tuple[PublicExpectedSignal, ...],
        Field(min_length=1, max_length=9),
    ]
    profile_digest: Sha256Digest

    @model_validator(mode="after")
    def signals_are_canonical_unique_and_self_attesting(self) -> Self:
        signal_types = tuple(item.signal_type for item in self.expected_signals)
        expected_order = tuple(sorted(signal_types, key=lambda item: item.value))
        if len(set(signal_types)) != len(signal_types) or signal_types != expected_order:
            raise ValueError("signal profile types must be unique and canonical")
        actual_shape = tuple(
            (item.signal_type, item.strength_ppm) for item in self.expected_signals
        )
        if actual_shape != _SIGNAL_PROFILE_SHAPES[self.fixture_variant]:
            raise ValueError("signal profile mask does not match its fixture variant")
        if not hmac.compare_digest(self.profile_digest, signal_profile_digest(self)):
            raise ValueError("signal profile digest does not match")
        return self


class PublicImpactClass(StrEnum):
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"
    HIGH_IMPACT = "high_impact"
    UNKNOWN = "unknown"


class PublicConstraintReferenceFixture(_StrictModel):
    memory_pool_index: PublicMemoryPoolIndex
    revision: PositiveInt


class PublicAssertionFixture(_StrictModel):
    subject_id: ComponentIdentifier
    predicate_id: ComponentIdentifier
    value_digest: Sha256Digest
    precedence: GeneratedParameterValue
    revision: PositiveInt
    supersedes_assertion_digest: Sha256Digest | None


def public_assertion_fixture_digest(assertion: PublicAssertionFixture) -> str:
    """Return the frozen canonical identity used by assertion supersession references."""

    if type(assertion) is not PublicAssertionFixture:
        raise ValueError("public assertion fixture has an invalid type")
    try:
        checked = PublicAssertionFixture.model_validate_json(canonical_json(assertion))
    except Exception:
        raise ValueError("public assertion fixture is invalid") from None
    return length_prefixed_sha256(
        canonical_json(checked),
        domain=PUBLIC_ASSERTION_FIXTURE_DIGEST_DOMAIN,
    )


class PublicFixtureEvent(_StrictModel):
    event_pool_index: PublicEventPoolIndex
    event_type: EventType
    phase: EventPhase
    payload: JsonObject
    parent_event_pool_indices: Annotated[
        tuple[PublicEventPoolIndex, ...],
        Field(max_length=16),
    ]

    @model_validator(mode="after")
    def event_payload_and_parents_are_bounded_and_canonical(self) -> Self:
        if not trace_event_payload_is_bounded(self.payload):
            raise ValueError("trace fixture event payload is not bounded")
        if self.parent_event_pool_indices != tuple(sorted(set(self.parent_event_pool_indices))):
            raise ValueError("trace fixture event parents are not canonical and unique")
        if any(index >= self.event_pool_index for index in self.parent_event_pool_indices):
            raise ValueError("trace fixture event parent must precede its child")
        return self


class PublicBindingFixture(_StrictModel):
    event_pool_index: PublicEventPoolIndex
    action_step: Signed64Offset | None
    scope_id: ComponentIdentifier | None
    progress_marker_digest: Sha256Digest | None
    constraint_references: (
        Annotated[
            tuple[PublicConstraintReferenceFixture, ...],
            Field(max_length=16),
        ]
        | None
    )
    impact: PublicImpactClass | None
    authorization_event_pool_indices: (
        Annotated[
            tuple[PublicEventPoolIndex, ...],
            Field(max_length=16),
        ]
        | None
    )
    safeguard_event_pool_indices: (
        Annotated[
            tuple[PublicEventPoolIndex, ...],
            Field(max_length=16),
        ]
        | None
    )
    assertions: (
        Annotated[
            tuple[PublicAssertionFixture, ...],
            Field(max_length=16),
        ]
        | None
    )

    @model_validator(mode="after")
    def binding_collections_are_unique_and_canonical(self) -> Self:
        if self.constraint_references is not None:
            keys = tuple(
                (item.memory_pool_index, item.revision) for item in self.constraint_references
            )
            if keys != tuple(sorted(set(keys))):
                raise ValueError("trace fixture constraint references are not canonical and unique")
        for name, values in (
            ("authorization", self.authorization_event_pool_indices),
            ("safeguard", self.safeguard_event_pool_indices),
        ):
            if values is not None and values != tuple(sorted(set(values))):
                raise ValueError(f"trace fixture {name} references are not canonical and unique")
        if self.assertions is not None:
            encoded = tuple(canonical_json(item) for item in self.assertions)
            if len(set(encoded)) != len(encoded):
                raise ValueError("trace fixture assertions must be unique")
        return self


class PublicDetectorMemoryFixture(_StrictModel):
    memory_pool_index: PublicMemoryPoolIndex
    kind: ClaimKind
    current_revision: PositiveInt
    validity: ValidityState
    provenance_event_pool_indices: Annotated[
        tuple[PublicEventPoolIndex, ...],
        Field(min_length=1, max_length=16),
    ]
    expires_at_event_pool_index: PublicEventPoolIndex | None

    @model_validator(mode="after")
    def memory_provenance_is_unique_and_canonical(self) -> Self:
        if self.provenance_event_pool_indices != tuple(
            sorted(set(self.provenance_event_pool_indices))
        ):
            raise ValueError("trace fixture memory provenance is not canonical and unique")
        if (
            self.expires_at_event_pool_index is not None
            and self.expires_at_event_pool_index < self.provenance_event_pool_indices[-1]
        ):
            raise ValueError("trace fixture memory expires before its provenance")
        return self


class OutcomeFreeTraceFixture(_StrictModel):
    schema_version: Literal["state-decay-v2-outcome-free-trace-fixture/v1"] = (
        "state-decay-v2-outcome-free-trace-fixture/v1"
    )
    events: Annotated[tuple[PublicFixtureEvent, ...], Field(min_length=3, max_length=8)]
    bindings: Annotated[tuple[PublicBindingFixture, ...], Field(min_length=3, max_length=8)]
    memories: Annotated[tuple[PublicDetectorMemoryFixture, ...], Field(max_length=4)]
    trace_fixture_digest: Sha256Digest

    @model_validator(mode="after")
    def fixture_is_resolved_outcome_free_and_self_attesting(self) -> Self:
        event_indices = tuple(item.event_pool_index for item in self.events)
        if event_indices != tuple(range(len(self.events))):
            raise ValueError("trace fixture events are not a canonical prefix")
        binding_indices = tuple(item.event_pool_index for item in self.bindings)
        if binding_indices != event_indices:
            raise ValueError("trace fixture bindings do not match its events")
        memory_indices = tuple(item.memory_pool_index for item in self.memories)
        if memory_indices != tuple(sorted(set(memory_indices))):
            raise ValueError("trace fixture memories are not canonical and unique")

        for binding in self.bindings:
            for references in (
                binding.authorization_event_pool_indices,
                binding.safeguard_event_pool_indices,
            ):
                if references is not None and any(
                    index > binding.event_pool_index for index in references
                ):
                    raise ValueError("trace fixture binding evidence comes from the future")
        for memory in self.memories:
            if any(index not in event_indices for index in memory.provenance_event_pool_indices):
                raise ValueError("trace fixture memory provenance does not resolve")
            if (
                memory.expires_at_event_pool_index is not None
                and memory.expires_at_event_pool_index not in event_indices
            ):
                raise ValueError("trace fixture memory expiry does not resolve")

        encoded = canonical_json(self.model_dump(mode="json", warnings=False))
        forbidden_markers = (
            b'"profile_id"',
            b'"fixture_variant"',
            b'"expected_signals"',
            b'"signal_type"',
            b'"strength_ppm"',
        )
        if any(marker in encoded for marker in forbidden_markers):
            raise ValueError("trace fixture contains a policy shortcut marker")
        _reject_outcome_label_content(self, context="trace fixture")
        if not hmac.compare_digest(
            self.trace_fixture_digest,
            trace_fixture_digest(self),
        ):
            raise ValueError("trace fixture digest does not match")
        return self


class PublicTextLengthProfile(_StrictModel):
    profile_id: ComponentIdentifier
    event_padding_spaces: Annotated[int, Field(ge=0, le=64)]
    memory_padding_spaces: Annotated[int, Field(ge=0, le=64)]
    pivot_padding_spaces: Annotated[int, Field(ge=0, le=64)]
    action_padding_spaces: Annotated[int, Field(ge=0, le=64)]


class PublicSlotProfile(_StrictModel):
    generator_slot: PublicGeneratorSlot
    counterbalance: PublicCounterbalanceProfile
    parameters: PublicParameterProfile
    structure: PublicStructuralProfile
    integers: PublicIntegerProfile
    evidence: PublicEvidenceProfile
    text_lengths: PublicTextLengthProfile
    signals: PublicSignalProfile

    @model_validator(mode="after")
    def profile_dimensions_fit_the_slot(self) -> Self:
        if not (
            self.structure.candidate_memory_count
            <= self.evidence.evidence_reference_count
            <= self.structure.candidate_memory_count * 3
        ):
            raise ValueError("slot evidence reference total does not fit its candidates")
        if self.evidence.decisive_event_count > self.structure.trajectory_event_count:
            raise ValueError("slot decisive events exceed its trajectory")
        if self.evidence.decisive_memory_count > self.structure.candidate_memory_count:
            raise ValueError("slot decisive memories exceed its candidates")
        if (
            self.integers.sequence_start
            + self.structure.trajectory_event_count * self.integers.sequence_stride
            > GENERATED_INTEGER_MAX
            or self.integers.action_step_start
            + (self.structure.trajectory_event_count - 1) * self.integers.action_step_stride
            > GENERATED_INTEGER_MAX
            or self.integers.memory_revision + self.structure.candidate_memory_count - 1
            > GENERATED_INTEGER_MAX
        ):
            raise ValueError("slot derived integer exceeds the generation bound")
        trajectory_count = self.structure.trajectory_event_count
        memory_count = self.structure.candidate_memory_count
        for expected in self.signals.expected_signals:
            evidence = expected.evidence
            event_indices = (
                *evidence.event_pool_indices,
                *evidence.binding_event_pool_indices,
                *(item.binding_event_pool_index for item in evidence.assertion_references),
            )
            if any(index >= trajectory_count for index in event_indices):
                raise ValueError("slot signal evidence exceeds its trajectory")
            if (
                any(item.memory_pool_index >= memory_count for item in evidence.memory_references)
                and expected.signal_type is not SignalType.STALE_CONSTRAINT
            ):
                raise ValueError("slot signal memory evidence does not resolve")
        return self


class PublicProfileCatalog(_StrictModel):
    schema_version: Literal["state-decay-v2-public-profile-catalog/v1"] = (
        "state-decay-v2-public-profile-catalog/v1"
    )
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    generation_contract_digest: Sha256Digest
    generator_configuration_digest: Sha256Digest
    counterbalance_axes: Annotated[
        tuple[CounterbalanceAxis, ...],
        Field(min_length=10, max_length=10),
    ]
    slot_profiles: Annotated[tuple[PublicSlotProfile, ...], Field(min_length=5, max_length=5)]
    catalog_digest: Sha256Digest

    @model_validator(mode="after")
    def catalog_is_complete_ordered_and_self_attesting(self) -> Self:
        if self.generation_contract_digest != GENERATION_CONTRACT.contract_digest:
            raise ValueError("profile catalog generation contract does not match")
        if self.counterbalance_axes != tuple(CounterbalanceAxis):
            raise ValueError("profile catalog counterbalance axes are not canonical")
        if tuple(profile.generator_slot for profile in self.slot_profiles) != tuple(range(5)):
            raise ValueError("profile catalog slot profiles are not canonical")
        profile_ids = tuple(
            profile_id
            for profile in self.slot_profiles
            for profile_id in (
                profile.counterbalance.profile_id,
                profile.parameters.profile_id,
                profile.structure.profile_id,
                profile.integers.profile_id,
                profile.evidence.profile_id,
                profile.text_lengths.profile_id,
                profile.signals.profile_id,
            )
        )
        parameter_ids = tuple(
            value.parameter_id
            for profile in self.slot_profiles
            for value in profile.parameters.allowed_values
        )
        if len(set(profile_ids)) != len(profile_ids) or len(set(parameter_ids)) != len(
            parameter_ids
        ):
            raise ValueError("profile catalog identifiers must be globally unique")
        variants = tuple(profile.signals.fixture_variant for profile in self.slot_profiles)
        masks = tuple(
            tuple(item.signal_type for item in profile.signals.expected_signals)
            for profile in self.slot_profiles
        )
        signal_types = {signal_type for mask in masks for signal_type in mask}
        strengths = {
            item.strength_ppm
            for profile in self.slot_profiles
            for item in profile.signals.expected_signals
        }
        if variants != tuple(PublicSignalFixtureVariant):
            raise ValueError("profile catalog signal fixture variants are not canonical")
        if len(set(masks)) != 5 or signal_types != set(SignalType):
            raise ValueError("profile catalog signal masks do not provide complete coverage")
        if len(strengths) < 3:
            raise ValueError("profile catalog signal strengths do not provide multiple levels")
        if not hmac.compare_digest(self.catalog_digest, profile_catalog_digest(self)):
            raise ValueError("profile catalog digest does not match")
        return self


class PublicTransitionState(_StrictModel):
    state_id: ComponentIdentifier
    description: ReviewSafeText
    terminal: PublicTerminalState | None = None

    @model_validator(mode="after")
    def state_is_outcome_free(self) -> Self:
        serialized = f"{self.state_id} {self.description}".casefold()
        if any(label in serialized for label in _OUTCOME_LABELS):
            raise ValueError("transition state contains an outcome label")
        return self


class PublicTransition(_StrictModel):
    source_state_id: ComponentIdentifier
    target_state_id: ComponentIdentifier
    exposure: PublicCausalExposure
    factor_values: Annotated[
        tuple[PublicCausalFactorValue, ...],
        Field(min_length=2, max_length=2),
    ]
    action_fingerprint_id: ComponentIdentifier
    failure_fingerprint_id: ComponentIdentifier | None = None
    trigger: ReviewSafeText

    @model_validator(mode="after")
    def transition_is_outcome_free(self) -> Self:
        serialized = " ".join(
            (
                self.source_state_id,
                self.target_state_id,
                self.action_fingerprint_id,
                self.failure_fingerprint_id or "",
                self.trigger,
            )
        ).casefold()
        if any(label in serialized for label in _OUTCOME_LABELS):
            raise ValueError("transition contains an outcome label")
        return self


class PublicTransitionExecution(_StrictModel):
    exposure: PublicCausalExposure
    factor_values: Annotated[
        tuple[PublicCausalFactorValue, ...],
        Field(min_length=2, max_length=2),
    ]
    visited_state_ids: Annotated[
        tuple[ComponentIdentifier, ...],
        Field(min_length=2, max_length=65),
    ]
    action_fingerprint_ids: Annotated[
        tuple[ComponentIdentifier, ...],
        Field(min_length=1, max_length=64),
    ]
    failure_fingerprint_ids: Annotated[
        tuple[ComponentIdentifier | None, ...],
        Field(min_length=1, max_length=64),
    ]
    terminal: PublicTerminalState
    repeated_action_count: Annotated[int, Field(ge=0, le=64)]
    failure_loop_count: Annotated[int, Field(ge=0, le=64)]

    @model_validator(mode="after")
    def execution_vectors_have_the_same_length(self) -> Self:
        action_count = len(self.action_fingerprint_ids)
        if (
            len(self.failure_fingerprint_ids) != action_count
            or len(self.visited_state_ids) != action_count + 1
            or self.repeated_action_count >= action_count + 1
            or self.failure_loop_count >= action_count + 1
        ):
            raise ValueError("transition execution vectors do not agree")
        return self


class PublicTransitionGraph(_StrictModel):
    schema_version: Literal["state-decay-v2-public-transition-graph/v1"] = (
        "state-decay-v2-public-transition-graph/v1"
    )
    initial_state_id: ComponentIdentifier
    factors: Annotated[tuple[PublicCausalFactor, ...], Field(min_length=2, max_length=2)]
    states: Annotated[tuple[PublicTransitionState, ...], Field(min_length=3, max_length=16)]
    transitions: Annotated[tuple[PublicTransition, ...], Field(min_length=8, max_length=32)]
    transition_graph_digest: Sha256Digest

    @model_validator(mode="after")
    def graph_is_executable_resolved_and_self_attesting(self) -> Self:
        factor_ids = tuple(factor.factor_id for factor in self.factors)
        if len(set(factor_ids)) != 2:
            raise ValueError("transition graph factors must be unique")

        state_ids = tuple(state.state_id for state in self.states)
        if len(set(state_ids)) != len(state_ids):
            raise ValueError("transition graph states must be unique")
        states = {state.state_id: state for state in self.states}
        initial = states.get(self.initial_state_id)
        if initial is None:
            raise ValueError("transition graph initial state does not resolve")
        if initial.terminal is not None:
            raise ValueError("transition graph initial state must be nonterminal")
        terminal_values = {state.terminal for state in self.states if state.terminal is not None}
        if terminal_values != set(PublicTerminalState):
            raise ValueError("transition graph terminal states are incomplete")

        guards: list[tuple[object, ...]] = []
        for transition in self.transitions:
            if transition.source_state_id not in states or transition.target_state_id not in states:
                raise ValueError("transition endpoint does not resolve")
            if states[transition.source_state_id].terminal is not None:
                raise ValueError("terminal transition state cannot have outgoing edges")
            transition_factor_ids = tuple(item.factor_id for item in transition.factor_values)
            if transition_factor_ids != factor_ids:
                raise ValueError("transition guard factors are not canonical and complete")
            guard = (
                transition.source_state_id,
                transition.exposure,
                *(item.value for item in transition.factor_values),
            )
            guards.append(guard)
        if len(set(guards)) != len(guards):
            raise ValueError("transition graph contains an ambiguous guard")

        visited_states: set[ComponentIdentifier] = set()
        visited_transitions: set[int] = set()
        for values in product((False, True), repeat=2):
            assignments = tuple(
                PublicCausalFactorValue(factor_id=factor_id, value=value)
                for factor_id, value in zip(factor_ids, values, strict=True)
            )
            for exposure in PublicCausalExposure:
                execution, transition_indices = _execute_public_transition_graph_unchecked(
                    self,
                    exposure,
                    assignments,
                )
                visited_states.update(execution.visited_state_ids)
                visited_transitions.update(transition_indices)
        if visited_states != set(state_ids):
            raise ValueError("transition graph contains an unreachable state")
        if visited_transitions != set(range(len(self.transitions))):
            raise ValueError("transition graph contains an unreachable transition")

        if not hmac.compare_digest(
            self.transition_graph_digest,
            transition_graph_digest(self),
        ):
            raise ValueError("transition graph digest does not match")
        return self


def _execute_public_transition_graph_unchecked(
    graph: PublicTransitionGraph,
    exposure: PublicCausalExposure,
    factor_values: tuple[PublicCausalFactorValue, ...],
) -> tuple[PublicTransitionExecution, frozenset[int]]:
    expected_factor_ids = tuple(factor.factor_id for factor in graph.factors)
    if tuple(item.factor_id for item in factor_values) != expected_factor_ids:
        raise ValueError("transition execution factors are not canonical and complete")

    states = {state.state_id: state for state in graph.states}
    current_state_id = graph.initial_state_id
    visited_state_ids = [current_state_id]
    action_fingerprint_ids: list[ComponentIdentifier] = []
    failure_fingerprint_ids: list[ComponentIdentifier | None] = []
    visited_transition_indices: set[int] = set()

    for _ in range(64):
        state = states[current_state_id]
        if state.terminal is not None:
            break
        matches = tuple(
            (index, transition)
            for index, transition in enumerate(graph.transitions)
            if transition.source_state_id == current_state_id
            and transition.exposure is exposure
            and transition.factor_values == factor_values
        )
        if len(matches) != 1:
            raise ValueError("transition execution requires exactly one matching guard")
        transition_index, transition = matches[0]
        visited_transition_indices.add(transition_index)
        action_fingerprint_ids.append(transition.action_fingerprint_id)
        failure_fingerprint_ids.append(transition.failure_fingerprint_id)
        current_state_id = transition.target_state_id
        visited_state_ids.append(current_state_id)
    else:
        raise ValueError("transition execution exceeds the 64-step bound")

    terminal = states[current_state_id].terminal
    if terminal is None:
        raise ValueError("transition execution did not reach a terminal state")
    repeated_action_count = sum(left == right for left, right in pairwise(action_fingerprint_ids))
    failure_loop_count = sum(
        left is not None and left == right for left, right in pairwise(failure_fingerprint_ids)
    )
    return (
        PublicTransitionExecution(
            exposure=exposure,
            factor_values=factor_values,
            visited_state_ids=tuple(visited_state_ids),
            action_fingerprint_ids=tuple(action_fingerprint_ids),
            failure_fingerprint_ids=tuple(failure_fingerprint_ids),
            terminal=terminal,
            repeated_action_count=repeated_action_count,
            failure_loop_count=failure_loop_count,
        ),
        frozenset(visited_transition_indices),
    )


def execute_public_transition_graph(
    graph: PublicTransitionGraph,
    exposure: PublicCausalExposure,
    factor_values: tuple[PublicCausalFactorValue, ...],
) -> PublicTransitionExecution:
    checked_graph = PublicTransitionGraph.model_validate(graph)
    if type(exposure) is not PublicCausalExposure or type(factor_values) is not tuple:
        raise ValueError("transition execution input is invalid")
    checked_factor_values = tuple(
        PublicCausalFactorValue.model_validate(item) for item in factor_values
    )
    execution, _ = _execute_public_transition_graph_unchecked(
        checked_graph,
        exposure,
        checked_factor_values,
    )
    return execution


def derive_causal_outcome(
    graph: PublicTransitionGraph,
    factor_values: tuple[PublicCausalFactorValue, ...],
) -> ScenarioOutcome:
    guidance = execute_public_transition_graph(
        graph,
        PublicCausalExposure.GUIDANCE_APPLIED,
        factor_values,
    )
    baseline = execute_public_transition_graph(
        graph,
        PublicCausalExposure.BASELINE_CONTINUED,
        factor_values,
    )
    truth = (
        guidance.terminal is PublicTerminalState.GOAL_REACHED,
        baseline.terminal is PublicTerminalState.GOAL_REACHED,
    )
    return {
        (True, False): ScenarioOutcome.HELPFUL,
        (False, True): ScenarioOutcome.HARMFUL,
        (True, True): ScenarioOutcome.REDUNDANT,
        (False, False): ScenarioOutcome.UNRESOLVED,
    }[truth]


class PublicEvidenceRelation(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    CONTEXTUALIZES = "contextualizes"


class PublicEvidenceNode(_StrictModel):
    evidence_id: ComponentIdentifier
    statement: ReviewSafeText


class PublicEvidenceEdge(_StrictModel):
    source_evidence_id: ComponentIdentifier
    target_evidence_id: ComponentIdentifier
    relation: PublicEvidenceRelation


class PublicEvidenceTopology(_StrictModel):
    schema_version: Literal["state-decay-v2-public-evidence-topology/v1"] = (
        "state-decay-v2-public-evidence-topology/v1"
    )
    nodes: Annotated[tuple[PublicEvidenceNode, ...], Field(min_length=1, max_length=32)]
    edges: Annotated[tuple[PublicEvidenceEdge, ...], Field(max_length=64)] = ()
    evidence_topology_digest: Sha256Digest

    @model_validator(mode="after")
    def topology_is_resolved_unique_and_self_attesting(self) -> Self:
        node_ids = tuple(node.evidence_id for node in self.nodes)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("evidence topology nodes must be unique")
        edges = tuple(
            (edge.source_evidence_id, edge.target_evidence_id, edge.relation) for edge in self.edges
        )
        if len(set(edges)) != len(edges):
            raise ValueError("evidence topology edges must be unique")
        if any(
            edge.source_evidence_id not in node_ids or edge.target_evidence_id not in node_ids
            for edge in self.edges
        ):
            raise ValueError("evidence topology endpoint does not resolve")
        if not hmac.compare_digest(
            self.evidence_topology_digest,
            evidence_topology_digest(self),
        ):
            raise ValueError("evidence topology digest does not match")
        return self


class PublicFailureMechanism(_StrictModel):
    failure_mechanism_id: ComponentIdentifier
    description: ReviewSafeText


class PublicSemanticSignature(_StrictModel):
    schema_version: Literal["state-decay-v2-public-semantic-signature/v1"] = (
        "state-decay-v2-public-semantic-signature/v1"
    )
    concept_ids: Annotated[tuple[ComponentIdentifier, ...], Field(min_length=1, max_length=16)]
    canonical_claims: Annotated[tuple[ReviewSafeText, ...], Field(min_length=1, max_length=16)]
    semantic_signature_digest: Sha256Digest

    @model_validator(mode="after")
    def signature_is_unique_and_self_attesting(self) -> Self:
        if len(set(self.concept_ids)) != len(self.concept_ids):
            raise ValueError("semantic signature concepts must be unique")
        if not hmac.compare_digest(
            self.semantic_signature_digest,
            semantic_signature_digest(self),
        ):
            raise ValueError("semantic signature digest does not match")
        return self


class RenderedCausalTextReplacement(_StrictModel):
    policy_pointer: JsonPointer
    replacement: RenderedCausalReplacementText


class RenderedCausalSemanticDelta(_StrictModel):
    delta_index: Annotated[int, Field(ge=0, le=3)]
    delta_id: ComponentIdentifier
    causal_delta_digest: Sha256Digest
    factor_values: Annotated[
        tuple[PublicCausalFactorValue, ...],
        Field(min_length=2, max_length=2),
    ]
    semantic_replacements: Annotated[
        tuple[RenderedCausalTextReplacement, ...],
        Field(max_length=4),
    ] = ()
    evidence_replacements: Annotated[
        tuple[RenderedCausalTextReplacement, ...],
        Field(max_length=4),
    ] = ()
    rendered_policy_digest: Sha256Digest

    @model_validator(mode="after")
    def rendered_delta_is_outcome_free_and_unambiguous(self) -> Self:
        if any(label in self.delta_id.casefold() for label in _OUTCOME_LABELS):
            raise ValueError("rendered causal delta identifier contains an outcome label")
        factor_ids = tuple(item.factor_id for item in self.factor_values)
        if len(set(factor_ids)) != 2:
            raise ValueError("rendered causal delta factor assignments must be unique")
        replacements = (*self.semantic_replacements, *self.evidence_replacements)
        if not 1 <= len(replacements) <= 4:
            raise ValueError("rendered causal delta must contain between one and four replacements")
        targets = tuple(replacement.policy_pointer for replacement in replacements)
        if len(set(targets)) != len(targets):
            raise ValueError("rendered causal delta targets must be unique and disjoint")
        if any(
            replacement.policy_pointer not in _RENDERED_SEMANTIC_CAUSAL_POINTERS
            for replacement in self.semantic_replacements
        ):
            raise ValueError("rendered causal delta semantic replacement pointer is not allowed")
        if any(
            replacement.policy_pointer not in _RENDERED_EVIDENCE_CAUSAL_POINTERS
            for replacement in self.evidence_replacements
        ):
            raise ValueError("rendered causal delta evidence replacement pointer is not allowed")
        return self


def _validate_controlled_replacement(
    source: str,
    replacement: str,
    *,
    context: str,
) -> None:
    if source == replacement:
        raise ValueError(f"{context} is a no-op")
    if any(
        character.isspace() and character != " "
        for text in (source, replacement)
        for character in text
    ):
        raise ValueError(f"{context} contains non-U+0020 whitespace")
    source_bytes = source.encode("utf-8")
    replacement_bytes = replacement.encode("utf-8")
    if len(source_bytes) != len(replacement_bytes):
        raise ValueError(f"{context} must preserve equal UTF-8 length")
    source_space_positions = tuple(
        index for index, character in enumerate(source) if character == " "
    )
    replacement_space_positions = tuple(
        index for index, character in enumerate(replacement) if character == " "
    )
    if source_space_positions != replacement_space_positions:
        raise ValueError(f"{context} must preserve U+0020 positions; whitespace-only drift")
    if source.replace(" ", "").casefold() == replacement.replace(" ", "").casefold():
        raise ValueError(f"{context} is case-only or whitespace-only")


def _template_policy_statement(
    template: OutcomeFreeTaskTemplate,
    pointer: str,
) -> str:
    if pointer.startswith("/event_pool/"):
        return template.event_pool[int(pointer.split("/", maxsplit=3)[2])].statement
    if pointer.startswith("/memory_pool/"):
        return template.memory_pool[int(pointer.split("/", maxsplit=3)[2])].statement
    if pointer == "/pivot/statement":
        return template.pivot.statement
    if pointer.startswith("/action_pool/"):
        return template.action_pool[int(pointer.split("/", maxsplit=3)[2])].statement
    raise ValueError("causal delta template pointer is not allowed")


def _rendered_policy_statement(
    skeleton: OutcomeFreeTaskSkeleton,
    pointer: str,
) -> str:
    if pointer.startswith("/trajectory/"):
        index = int(pointer.split("/", maxsplit=3)[2])
        if index < len(skeleton.trajectory):
            return skeleton.trajectory[index].statement
    elif pointer.startswith("/candidate_memories/"):
        index = int(pointer.split("/", maxsplit=3)[2])
        if index < len(skeleton.candidate_memories):
            return skeleton.candidate_memories[index].statement
    elif pointer == "/pivot/statement":
        return skeleton.pivot.statement
    elif pointer.startswith("/allowed_actions/"):
        index = int(pointer.split("/", maxsplit=3)[2])
        if index < len(skeleton.allowed_actions):
            return skeleton.allowed_actions[index].statement
    raise ValueError("rendered causal delta policy pointer does not resolve")


def _replace_rendered_policy_statement(
    skeleton: OutcomeFreeTaskSkeleton,
    replacement: RenderedCausalTextReplacement,
) -> OutcomeFreeTaskSkeleton:
    pointer = replacement.policy_pointer
    _validate_controlled_replacement(
        _rendered_policy_statement(skeleton, pointer),
        replacement.replacement,
        context="rendered causal replacement",
    )
    if pointer.startswith("/trajectory/"):
        index = int(pointer.split("/", maxsplit=3)[2])
        trajectory = list(skeleton.trajectory)
        if index >= len(trajectory):
            raise ValueError("rendered causal delta policy pointer does not resolve")
        trajectory[index] = trajectory[index].model_copy(
            update={"statement": replacement.replacement}
        )
        return skeleton.model_copy(update={"trajectory": tuple(trajectory)})
    if pointer.startswith("/candidate_memories/"):
        index = int(pointer.split("/", maxsplit=3)[2])
        memories = list(skeleton.candidate_memories)
        if index >= len(memories):
            raise ValueError("rendered causal delta policy pointer does not resolve")
        memories[index] = memories[index].model_copy(update={"statement": replacement.replacement})
        return skeleton.model_copy(update={"candidate_memories": tuple(memories)})
    if pointer == "/pivot/statement":
        return skeleton.model_copy(
            update={
                "pivot": skeleton.pivot.model_copy(update={"statement": replacement.replacement})
            }
        )
    if pointer.startswith("/allowed_actions/"):
        index = int(pointer.split("/", maxsplit=3)[2])
        actions = list(skeleton.allowed_actions)
        if index >= len(actions):
            raise ValueError("rendered causal delta policy pointer does not resolve")
        actions[index] = actions[index].model_copy(update={"statement": replacement.replacement})
        return skeleton.model_copy(update={"allowed_actions": tuple(actions)})
    raise ValueError("rendered causal delta policy pointer is not allowed")


def _rendered_pointer_for_template_pointer(
    template_pointer: str,
    allowed_action_order: tuple[int, int],
) -> str:
    if template_pointer.startswith("/event_pool/"):
        index = int(template_pointer.split("/", maxsplit=3)[2])
        return f"/trajectory/{index}/statement"
    if template_pointer.startswith("/memory_pool/"):
        index = int(template_pointer.split("/", maxsplit=3)[2])
        return f"/candidate_memories/{index}/statement"
    if template_pointer == "/pivot/statement":
        return template_pointer
    if template_pointer.startswith("/action_pool/"):
        logical_index = int(template_pointer.split("/", maxsplit=3)[2])
        return f"/allowed_actions/{allowed_action_order.index(logical_index)}/statement"
    raise ValueError("causal delta template pointer is not allowed")


def _replacement_materialization_suffix(
    template_pointer: str,
    slot_profile: PublicSlotProfile,
) -> str:
    profile = slot_profile.text_lengths
    if template_pointer.startswith("/event_pool/"):
        return " " * profile.event_padding_spaces
    elif template_pointer.startswith("/memory_pool/"):
        return " " * profile.memory_padding_spaces
    elif template_pointer == "/pivot/statement":
        parameters = ",".join(
            f"{item.parameter_id}={item.value}" for item in slot_profile.parameters.allowed_values
        )
        return f" [parameters:{parameters}]" + " " * profile.pivot_padding_spaces
    elif template_pointer.startswith("/action_pool/"):
        return " " * profile.action_padding_spaces
    else:
        raise ValueError("causal delta template pointer is not allowed")


def _rendered_policy_bytes(
    skeleton: OutcomeFreeTaskSkeleton,
    delta: RenderedCausalSemanticDelta,
) -> bytes:
    rendered = skeleton
    for replacement in (*delta.semantic_replacements, *delta.evidence_replacements):
        rendered = _replace_rendered_policy_statement(rendered, replacement)
    result = canonical_json(rendered)
    if result == canonical_json(skeleton):
        raise ValueError("candidate rendered causal delta is a no-op")
    return result


def materialize_rendered_policy_skeleton(
    skeleton: OutcomeFreeTaskSkeleton,
    delta: RenderedCausalSemanticDelta,
) -> OutcomeFreeTaskSkeleton:
    """Apply one closed rendered delta and verify the complete resulting policy digest."""

    if (
        type(skeleton) is not OutcomeFreeTaskSkeleton
        or type(delta) is not RenderedCausalSemanticDelta
    ):
        raise ValueError("rendered policy materialization input is invalid")
    try:
        checked_skeleton = OutcomeFreeTaskSkeleton.model_validate_json(canonical_json(skeleton))
        checked_delta = RenderedCausalSemanticDelta.model_validate_json(canonical_json(delta))
        policy_bytes = _rendered_policy_bytes(checked_skeleton, checked_delta)
    except Exception:
        raise ValueError("rendered policy materialization input is invalid") from None
    if not hmac.compare_digest(
        checked_delta.rendered_policy_digest,
        rendered_policy_digest(policy_bytes),
    ):
        raise ValueError("rendered policy materialization digest does not match")
    return OutcomeFreeTaskSkeleton.model_validate_json(policy_bytes)


class PreAllocationSkeletonPreview(_StrictModel):
    schema_version: Literal["state-decay-v2-pre-allocation-skeleton-preview/v1"] = (
        "state-decay-v2-pre-allocation-skeleton-preview/v1"
    )
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    split: BenchmarkSplit
    family: ScenarioFamily
    lineage_registry_key: PublicLineageKey
    generator_slot: PublicGeneratorSlot
    generator_version: Literal["state-decay-v2-public-generator/v1"] = (
        "state-decay-v2-public-generator/v1"
    )
    generation_contract_digest: Sha256Digest
    generator_configuration_digest: Sha256Digest
    generator_algorithm_digest: Sha256Digest
    profile_catalog_digest: Sha256Digest
    transition_graph_digest: Sha256Digest
    evidence_topology_digest: Sha256Digest
    failure_mechanism_id: ComponentIdentifier
    semantic_signature_digest: Sha256Digest
    slot_profile: PublicSlotProfile
    allowed_parameter_values: Annotated[
        tuple[PublicParameterValue, ...],
        Field(min_length=1, max_length=16),
    ]
    task_skeleton: OutcomeFreeTaskSkeleton
    trace_fixture: OutcomeFreeTraceFixture
    rendered_causal_deltas: Annotated[
        tuple[RenderedCausalSemanticDelta, ...],
        Field(min_length=4, max_length=4),
    ]
    preview_digest: Sha256Digest

    @model_validator(mode="after")
    def preview_is_public_bound_ordered_and_self_attesting(self) -> Self:
        expected_split = _PUBLIC_FAMILY_SPLITS.get(self.family)
        family, _ = parse_public_lineage_key(self.lineage_registry_key)
        if expected_split is None or self.split is not expected_split or family is not self.family:
            raise ValueError("preview public lineage coordinates do not agree")
        if self.generation_contract_digest != GENERATION_CONTRACT.contract_digest:
            raise ValueError("preview generation contract does not match")
        if self.slot_profile.generator_slot != self.generator_slot:
            raise ValueError("preview slot profile does not match its generator slot")
        if self.allowed_parameter_values != self.slot_profile.parameters.allowed_values:
            raise ValueError("preview allowed parameter values do not match its slot profile")
        if (
            len(self.task_skeleton.trajectory) != self.slot_profile.structure.trajectory_event_count
            or len(self.task_skeleton.candidate_memories)
            != self.slot_profile.structure.candidate_memory_count
            or len(self.task_skeleton.allowed_actions)
            != self.slot_profile.structure.allowed_action_count
        ):
            raise ValueError("preview task skeleton does not match its structural profile")
        if len(self.trace_fixture.events) != len(self.task_skeleton.trajectory):
            raise ValueError("preview trace fixture does not match its trajectory")
        for binding, event in zip(
            self.trace_fixture.bindings,
            self.task_skeleton.trajectory,
            strict=True,
        ):
            if binding.action_step is not None and binding.action_step != event.action_step:
                raise ValueError("preview trace fixture action step does not match its trajectory")
        fixture_memory_indices = tuple(
            item.memory_pool_index for item in self.trace_fixture.memories
        )
        if fixture_memory_indices != tuple(range(len(self.task_skeleton.candidate_memories))):
            raise ValueError("preview trace fixture memories do not match its candidates")
        for fixture_memory, candidate_memory in zip(
            self.trace_fixture.memories,
            self.task_skeleton.candidate_memories,
            strict=True,
        ):
            if (
                fixture_memory.current_revision != candidate_memory.revision
                or fixture_memory.validity is not candidate_memory.validity
            ):
                raise ValueError("preview trace fixture memory state does not match its candidate")
        fixture_memories = {item.memory_pool_index: item for item in self.trace_fixture.memories}
        for expected in self.slot_profile.signals.expected_signals:
            for memory in expected.evidence.memory_references:
                resolved_memory = fixture_memories.get(memory.memory_pool_index)
                if resolved_memory is None:
                    if expected.signal_type is not SignalType.STALE_CONSTRAINT:
                        raise ValueError("preview expected signal memory evidence does not resolve")
                elif (
                    expected.signal_type is not SignalType.STALE_CONSTRAINT
                    and resolved_memory.current_revision != memory.revision
                ):
                    raise ValueError("preview expected signal memory revision does not resolve")
            for assertion in expected.evidence.assertion_references:
                binding = self.trace_fixture.bindings[assertion.binding_event_pool_index]
                if binding.assertions is None or assertion.assertion_index >= len(
                    binding.assertions
                ):
                    raise ValueError("preview expected assertion evidence does not resolve")
        if tuple(delta.delta_index for delta in self.rendered_causal_deltas) != tuple(range(4)):
            raise ValueError("preview rendered causal deltas are not canonical")
        delta_ids = tuple(delta.delta_id for delta in self.rendered_causal_deltas)
        delta_digests = tuple(delta.causal_delta_digest for delta in self.rendered_causal_deltas)
        if len(set(delta_ids)) != 4 or len(set(delta_digests)) != 4:
            raise ValueError("preview rendered causal delta bindings must be unique")
        for delta in self.rendered_causal_deltas:
            policy_bytes = _rendered_policy_bytes(self.task_skeleton, delta)
            if not hmac.compare_digest(
                delta.rendered_policy_digest,
                rendered_policy_digest(policy_bytes),
            ):
                raise ValueError("preview rendered policy digest does not match")
        if not hmac.compare_digest(self.preview_digest, skeleton_preview_digest(self)):
            raise ValueError("preview digest does not match")
        return self


class PublicLineageCandidate(_StrictModel):
    schema_version: Literal["state-decay-v2-public-lineage-candidate/v1"] = (
        "state-decay-v2-public-lineage-candidate/v1"
    )
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    split: BenchmarkSplit
    family: ScenarioFamily
    lineage_registry_key: PublicLineageKey
    generator_version: Literal["state-decay-v2-public-generator/v1"] = (
        "state-decay-v2-public-generator/v1"
    )
    generation_contract_digest: Sha256Digest
    generator_configuration_digest: Sha256Digest
    generator_algorithm_digest: Sha256Digest
    profile_catalog_digest: Sha256Digest
    independent_seed_commitment_digest: Sha256Digest
    task_template: OutcomeFreeTaskTemplate
    transition_graph: PublicTransitionGraph
    evidence_topology: PublicEvidenceTopology
    failure_mechanism: PublicFailureMechanism
    semantic_signature: PublicSemanticSignature
    causal_deltas: Annotated[
        tuple[CausalSemanticDelta, ...],
        Field(min_length=4, max_length=4),
    ]
    semantic_rationale: ReviewSafeText
    derivation_parent_keys: Annotated[
        tuple[PublicLineageKey, ...],
        Field(max_length=0),
    ] = ()
    previews: Annotated[
        tuple[PreAllocationSkeletonPreview, ...],
        Field(min_length=5, max_length=5),
    ]
    candidate_packet_digest: Sha256Digest

    @model_validator(mode="after")
    def candidate_is_public_bound_ordered_and_self_attesting(self) -> Self:
        _reject_outcome_label_content(self, context="candidate-owned content")
        expected_split = _PUBLIC_FAMILY_SPLITS.get(self.family)
        family, _ = parse_public_lineage_key(self.lineage_registry_key)
        if expected_split is None or self.split is not expected_split or family is not self.family:
            raise ValueError("candidate public lineage coordinates do not agree")
        if self.generation_contract_digest != GENERATION_CONTRACT.contract_digest:
            raise ValueError("candidate generation contract does not match")

        public_leaf = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.PUBLIC)
        expected_seed_commitment = independent_lineage_seed_commitment(
            derive_independent_lineage_seed(
                public_leaf,
                split=self.split,
                family=self.family,
                lineage_registry_key=self.lineage_registry_key,
            )
        )
        if not hmac.compare_digest(
            self.independent_seed_commitment_digest,
            expected_seed_commitment,
        ):
            raise ValueError("candidate public lineage seed commitment does not match")

        if tuple(delta.delta_index for delta in self.causal_deltas) != tuple(range(4)):
            raise ValueError("candidate causal deltas are not canonical")
        delta_ids = tuple(delta.delta_id for delta in self.causal_deltas)
        delta_digests = tuple(delta.causal_delta_digest for delta in self.causal_deltas)
        if len(set(delta_ids)) != 4 or len(set(delta_digests)) != 4:
            raise ValueError("candidate causal delta bindings must be unique")
        if any(
            delta.family is not self.family
            or delta.lineage_registry_key != self.lineage_registry_key
            for delta in self.causal_deltas
        ):
            raise ValueError("candidate causal delta coordinates do not agree")
        for delta in self.causal_deltas:
            for replacement in (*delta.semantic_replacements, *delta.evidence_replacements):
                _validate_controlled_replacement(
                    _template_policy_statement(
                        self.task_template,
                        replacement.template_pointer,
                    ),
                    replacement.replacement,
                    context="candidate source causal replacement",
                )
        graph_factor_ids = tuple(factor.factor_id for factor in self.transition_graph.factors)
        factor_vectors = tuple(
            tuple(item.value for item in delta.factor_values) for delta in self.causal_deltas
        )
        if any(
            tuple(item.factor_id for item in delta.factor_values) != graph_factor_ids
            for delta in self.causal_deltas
        ):
            raise ValueError("candidate causal delta factors do not match the transition graph")
        if set(factor_vectors) != set(product((False, True), repeat=2)):
            raise ValueError("candidate causal delta factor vectors are not complete and unique")
        derived_outcomes = tuple(
            derive_causal_outcome(self.transition_graph, delta.factor_values)
            for delta in self.causal_deltas
        )
        if set(derived_outcomes) != set(ScenarioOutcome):
            raise ValueError("candidate causal graph does not derive all paired result classes")

        if tuple(preview.generator_slot for preview in self.previews) != tuple(range(5)):
            raise ValueError("candidate previews are not in canonical slot order")
        source_delta_bindings = tuple(
            (
                delta.delta_index,
                delta.delta_id,
                delta.causal_delta_digest,
                delta.factor_values,
            )
            for delta in self.causal_deltas
        )
        for preview in self.previews:
            if preview.task_skeleton.adapter != self.task_template.adapter:
                raise ValueError("candidate preview adapter does not match its task template")
            if (
                preview.split is not self.split
                or preview.family is not self.family
                or preview.lineage_registry_key != self.lineage_registry_key
                or preview.generator_version != self.generator_version
                or preview.generation_contract_digest != self.generation_contract_digest
                or preview.generator_configuration_digest != self.generator_configuration_digest
                or preview.generator_algorithm_digest != self.generator_algorithm_digest
                or preview.profile_catalog_digest != self.profile_catalog_digest
                or preview.transition_graph_digest != self.transition_graph.transition_graph_digest
                or preview.evidence_topology_digest
                != self.evidence_topology.evidence_topology_digest
                or preview.failure_mechanism_id != self.failure_mechanism.failure_mechanism_id
                or preview.semantic_signature_digest
                != self.semantic_signature.semantic_signature_digest
            ):
                raise ValueError("candidate preview source bindings do not agree")
            rendered_delta_bindings = tuple(
                (
                    delta.delta_index,
                    delta.delta_id,
                    delta.causal_delta_digest,
                    delta.factor_values,
                )
                for delta in preview.rendered_causal_deltas
            )
            if rendered_delta_bindings != source_delta_bindings:
                raise ValueError("candidate preview causal delta bindings do not agree")
            for source_delta, rendered_delta in zip(
                self.causal_deltas,
                preview.rendered_causal_deltas,
                strict=True,
            ):
                for source_replacements, rendered_replacements in (
                    (
                        source_delta.semantic_replacements,
                        rendered_delta.semantic_replacements,
                    ),
                    (
                        source_delta.evidence_replacements,
                        rendered_delta.evidence_replacements,
                    ),
                ):
                    if len(source_replacements) != len(rendered_replacements):
                        raise ValueError(
                            "candidate preview causal replacement materialization does not agree"
                        )
                    for source_replacement, rendered_replacement in zip(
                        source_replacements,
                        rendered_replacements,
                        strict=True,
                    ):
                        expected_pointer = _rendered_pointer_for_template_pointer(
                            source_replacement.template_pointer,
                            preview.slot_profile.counterbalance.allowed_action_order,
                        )
                        if rendered_replacement.policy_pointer != expected_pointer:
                            raise ValueError(
                                "candidate preview causal materialization pointer does not agree"
                            )
                        suffix = _replacement_materialization_suffix(
                            source_replacement.template_pointer,
                            preview.slot_profile,
                        )
                        if _rendered_policy_statement(
                            preview.task_skeleton,
                            expected_pointer,
                        ) != (
                            _template_policy_statement(
                                self.task_template,
                                source_replacement.template_pointer,
                            )
                            + suffix
                        ):
                            raise ValueError(
                                "candidate preview source statement materialization does not agree"
                            )
                        if rendered_replacement.replacement != (
                            source_replacement.replacement + suffix
                        ):
                            raise ValueError(
                                "candidate preview causal replacement "
                                "materialization does not agree"
                            )
            rendered_policies = tuple(
                _rendered_policy_bytes(preview.task_skeleton, delta)
                for delta in preview.rendered_causal_deltas
            )
            if len(set(rendered_policies)) != 4:
                raise ValueError("candidate rendered causal policies must be pairwise distinct")

        if not hmac.compare_digest(
            self.candidate_packet_digest,
            candidate_packet_digest(self),
        ):
            raise ValueError("candidate packet digest does not match")
        return self


class PublicLineageRegistry(_StrictModel):
    schema_version: Literal["state-decay-v2-public-lineage-registry/v1"] = (
        "state-decay-v2-public-lineage-registry/v1"
    )
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    generation_contract_digest: Sha256Digest
    lineage_review_protocol_digest: Sha256Digest
    generator_configuration_digest: Sha256Digest
    generator_algorithm_digest: Sha256Digest
    profile_catalog: PublicProfileCatalog
    candidates: Annotated[
        tuple[PublicLineageCandidate, ...],
        Field(min_length=180, max_length=180),
    ]
    registry_digest: Sha256Digest

    @model_validator(mode="after")
    def registry_is_complete_unique_and_self_attesting(self) -> Self:
        if self.generation_contract_digest != GENERATION_CONTRACT.contract_digest:
            raise ValueError("public lineage registry generation contract does not match")
        if self.lineage_review_protocol_digest != LINEAGE_REVIEW_PROTOCOL.protocol_digest:
            raise ValueError("public lineage registry review protocol does not match")
        if (
            self.profile_catalog.generation_contract_digest != self.generation_contract_digest
            or self.profile_catalog.generator_configuration_digest
            != self.generator_configuration_digest
        ):
            raise ValueError("public lineage registry profile catalog bindings do not agree")

        expected_coordinates = tuple(
            (family, public_lineage_key(family, index))
            for family in _PUBLIC_FAMILY_CODES
            for index in range(30)
        )
        actual_coordinates = tuple(
            (candidate.family, candidate.lineage_registry_key) for candidate in self.candidates
        )
        if actual_coordinates != expected_coordinates:
            raise ValueError("public lineage registry candidates are not canonical and complete")

        for candidate in self.candidates:
            if (
                candidate.generation_contract_digest != self.generation_contract_digest
                or candidate.generator_configuration_digest != self.generator_configuration_digest
                or candidate.generator_algorithm_digest != self.generator_algorithm_digest
                or candidate.profile_catalog_digest != self.profile_catalog.catalog_digest
            ):
                raise ValueError("public lineage registry candidate bindings do not agree")
            if tuple(preview.slot_profile for preview in candidate.previews) != (
                self.profile_catalog.slot_profiles
            ):
                raise ValueError(
                    "public lineage registry preview does not use its canonical profile catalog"
                )

        candidate_digests = tuple(
            candidate.candidate_packet_digest for candidate in self.candidates
        )
        preview_digests = tuple(
            preview.preview_digest
            for candidate in self.candidates
            for preview in candidate.previews
        )
        if len(set(candidate_digests)) != 180 or len(set(preview_digests)) != 900:
            raise ValueError("public lineage registry candidate or preview digests collide")
        rendered_policy_digests = tuple(
            delta.rendered_policy_digest
            for candidate in self.candidates
            for preview in candidate.previews
            for delta in preview.rendered_causal_deltas
        )
        if len(set(rendered_policy_digests)) != 3_600:
            raise ValueError("public lineage registry rendered policy digests collide")

        for family in _PUBLIC_FAMILY_CODES:
            family_candidates = tuple(
                candidate for candidate in self.candidates if candidate.family is family
            )
            unique_columns = (
                tuple(candidate.candidate_packet_digest for candidate in family_candidates),
                tuple(candidate.lineage_registry_key for candidate in family_candidates),
                tuple(
                    candidate.independent_seed_commitment_digest for candidate in family_candidates
                ),
                tuple(
                    candidate.transition_graph.transition_graph_digest
                    for candidate in family_candidates
                ),
                tuple(
                    candidate.evidence_topology.evidence_topology_digest
                    for candidate in family_candidates
                ),
                tuple(
                    candidate.failure_mechanism.failure_mechanism_id
                    for candidate in family_candidates
                ),
                tuple(
                    candidate.semantic_signature.semantic_signature_digest
                    for candidate in family_candidates
                ),
            )
            if any(len(set(column)) != 30 for column in unique_columns):
                raise ValueError("public lineage registry family-local fields are not unique")

        if not hmac.compare_digest(self.registry_digest, candidate_registry_digest(self)):
            raise ValueError("public lineage registry digest does not match")
        return self


class RoleNeutralGeneratedParts(_StrictModel):
    schema_version: Literal["state-decay-v2-public-role-neutral-parts/v1"] = (
        "state-decay-v2-public-role-neutral-parts/v1"
    )
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    split: BenchmarkSplit
    scenario_id: Sha256Digest
    trajectory: Annotated[tuple[OutcomeFreeEvent, ...], Field(min_length=1, max_length=64)]
    candidate_memories: Annotated[
        tuple[OutcomeFreeCandidateMemory, ...],
        Field(min_length=1, max_length=32),
    ]
    pivot: OutcomeFreePivot
    allowed_actions: Annotated[
        tuple[OutcomeFreeAllowedAction, ...],
        Field(min_length=2, max_length=16),
    ]
    adapter: AdapterMetadata

    @model_validator(mode="after")
    def policy_fields_repeat_the_skeleton_invariants(self) -> Self:
        OutcomeFreeTaskSkeleton(
            trajectory=self.trajectory,
            candidate_memories=self.candidate_memories,
            pivot=self.pivot,
            allowed_actions=self.allowed_actions,
            adapter=self.adapter,
        )
        return self


__all__ = [
    "CANDIDATE_PACKET_DIGEST_DOMAIN",
    "CANDIDATE_REGISTRY_DIGEST_DOMAIN",
    "CAUSAL_DELTA_DIGEST_DOMAIN",
    "EVIDENCE_TOPOLOGY_DIGEST_DOMAIN",
    "GENERATOR_ALGORITHM_DIGEST_DOMAIN",
    "GENERATOR_CONFIGURATION_DIGEST_DOMAIN",
    "MAX_REVIEW_SAFE_TEXT_UTF8_BYTES",
    "PROFILE_CATALOG_DIGEST_DOMAIN",
    "PUBLIC_ASSERTION_FIXTURE_DIGEST_DOMAIN",
    "PUBLIC_LINEAGE_KEY_PATTERN",
    "RENDERED_POLICY_DIGEST_DOMAIN",
    "SEMANTIC_SIGNATURE_DIGEST_DOMAIN",
    "SIGNAL_PROFILE_DIGEST_DOMAIN",
    "SKELETON_PREVIEW_DIGEST_DOMAIN",
    "TRACE_FIXTURE_DIGEST_DOMAIN",
    "TRANSITION_GRAPH_DIGEST_DOMAIN",
    "CausalReplacementText",
    "CausalSemanticDelta",
    "CausalTextReplacement",
    "GeneratedParameterValue",
    "OutcomeFreeAllowedAction",
    "OutcomeFreeCandidateMemory",
    "OutcomeFreeEvent",
    "OutcomeFreeEvidenceReference",
    "OutcomeFreePivot",
    "OutcomeFreePolicyText",
    "OutcomeFreeTaskSkeleton",
    "OutcomeFreeTaskTemplate",
    "OutcomeFreeTemplateAction",
    "OutcomeFreeTemplateEvent",
    "OutcomeFreeTemplateMemory",
    "OutcomeFreeTemplatePivot",
    "OutcomeFreeTraceFixture",
    "PreAllocationSkeletonPreview",
    "PublicAssertionFixture",
    "PublicAssertionIndex",
    "PublicBindingFixture",
    "PublicCausalExposure",
    "PublicCausalFactor",
    "PublicCausalFactorValue",
    "PublicConstraintReferenceFixture",
    "PublicCounterbalanceProfile",
    "PublicDetectorMemoryFixture",
    "PublicEventPoolIndex",
    "PublicEvidenceEdge",
    "PublicEvidenceNode",
    "PublicEvidenceProfile",
    "PublicEvidenceRelation",
    "PublicEvidenceTopology",
    "PublicExpectedAssertionEvidence",
    "PublicExpectedDetectorEvidence",
    "PublicExpectedMemoryEvidence",
    "PublicExpectedSignal",
    "PublicFailureMechanism",
    "PublicFixtureEvent",
    "PublicGeneratorAlgorithm",
    "PublicGeneratorConfiguration",
    "PublicGeneratorOperation",
    "PublicGeneratorSlot",
    "PublicGeneratorStep",
    "PublicImpactClass",
    "PublicIntegerProfile",
    "PublicLineageCandidate",
    "PublicLineageKey",
    "PublicLineageRegistry",
    "PublicMemoryPoolIndex",
    "PublicParameterProfile",
    "PublicParameterValue",
    "PublicProfileCatalog",
    "PublicSemanticSignature",
    "PublicSignalFixtureVariant",
    "PublicSignalProfile",
    "PublicSlotProfile",
    "PublicStructuralProfile",
    "PublicTerminalState",
    "PublicTextLengthProfile",
    "PublicTransition",
    "PublicTransitionExecution",
    "PublicTransitionGraph",
    "PublicTransitionState",
    "RenderedCausalReplacementText",
    "RenderedCausalSemanticDelta",
    "RenderedCausalTextReplacement",
    "ReviewSafeText",
    "RoleNeutralGeneratedParts",
    "candidate_packet_digest",
    "candidate_registry_digest",
    "causal_delta_digest",
    "derive_causal_outcome",
    "evidence_topology_digest",
    "execute_public_transition_graph",
    "generator_algorithm_digest",
    "generator_configuration_digest",
    "materialize_rendered_policy_skeleton",
    "parse_public_lineage_key",
    "profile_catalog_digest",
    "public_assertion_fixture_digest",
    "public_lineage_key",
    "rendered_policy_digest",
    "semantic_signature_digest",
    "signal_profile_digest",
    "skeleton_preview_digest",
    "trace_fixture_digest",
    "transition_graph_digest",
    "validate_review_safe_text",
]
