from __future__ import annotations

import hmac
from collections.abc import Mapping, Sequence
from enum import StrEnum
from math import gcd
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from saliencegate.benchmarks.state_decay_v2.config import (
    BOOTSTRAP_INDEX_DOMAIN,
    FOLD_KEY_DOMAIN,
    GENERATED_INTEGER_MAX,
    GENERATION_CONTRACT,
    LINEAGES_PER_FAMILY,
    PERMUTATION_INDEX_DOMAIN,
    PROPOSAL_FIXTURE_SEED_COMMITMENT_DOMAIN,
    PROPOSAL_FIXTURE_SEED_DOMAIN,
    PUBLIC_GENERATION_SEED,
    TEMPLATE_REGISTRY_UNIQUE_FIELDS,
    SeedPurpose,
    SeedSourceBoundary,
    derive_proposal_fixture_seed,
    derive_seed,
    proposal_fixture_seed_commitment,
    seed_commitment,
    u64be,
)
from saliencegate.benchmarks.state_decay_v2.manifest import ValidationAudit
from saliencegate.benchmarks.state_decay_v2.schema import (
    SUITE_ID,
    SUITE_VERSION,
    BenchmarkSplit,
    ScenarioFamily,
    ScenarioOutcome,
)
from saliencegate.domain import ValidityState, canonical_json, length_prefixed_sha256
from saliencegate.domain.records import ComponentIdentifier, Sha256Digest

NUISANCE_INVENTORY_SCHEMA_VERSION: Literal["state-decay-v2-nuisance-inventory/v1"] = (
    "state-decay-v2-nuisance-inventory/v1"
)
LEAKAGE_PROTOCOL_SCHEMA_VERSION: Literal["state-decay-v2-leakage-protocol/v1"] = (
    "state-decay-v2-leakage-protocol/v1"
)
LINEAGE_REVIEW_RECORD_SCHEMA_VERSION: Literal["state-decay-v2-lineage-review-record/v1"] = (
    "state-decay-v2-lineage-review-record/v1"
)
LINEAGE_REVIEW_PROTOCOL_SCHEMA_VERSION: Literal["state-decay-v2-lineage-review-protocol/v1"] = (
    "state-decay-v2-lineage-review-protocol/v1"
)
TREATMENT_COVERAGE_PROTOCOL_SCHEMA_VERSION: Literal[
    "state-decay-v2-treatment-coverage-protocol/v1"
] = "state-decay-v2-treatment-coverage-protocol/v1"
FINITE_SAMPLE_PROTOCOL_SCHEMA_VERSION: Literal["state-decay-v2-finite-sample-protocol/v1"] = (
    "state-decay-v2-finite-sample-protocol/v1"
)

NUISANCE_INVENTORY_DIGEST_DOMAIN = "saliencegate:state-decay-v2:nuisance-inventory:v1"
LEAKAGE_PROTOCOL_DIGEST_DOMAIN = "saliencegate:state-decay-v2:leakage-protocol:v1"
LINEAGE_REVIEW_RECORD_DIGEST_DOMAIN: Literal[
    "saliencegate:state-decay-v2:lineage-review-record:v1"
] = "saliencegate:state-decay-v2:lineage-review-record:v1"
LINEAGE_REVIEW_PROTOCOL_DIGEST_DOMAIN = "saliencegate:state-decay-v2:lineage-review-protocol:v1"
TREATMENT_PROTOCOL_DIGEST_DOMAIN = "saliencegate:state-decay-v2:exact-treatment-protocol:v1"
FINITE_SAMPLE_PROTOCOL_DIGEST_DOMAIN = "saliencegate:state-decay-v2:finite-sample-protocol:v1"
INDEPENDENT_LINEAGE_SEED_DOMAIN: Literal["saliencegate:state-decay-v2:lineage-seed:v1"] = (
    "saliencegate:state-decay-v2:lineage-seed:v1"
)
INDEPENDENT_LINEAGE_SEED_COMMITMENT_DOMAIN: Literal[
    "saliencegate:state-decay-v2:lineage-seed-commitment:v1"
] = "saliencegate:state-decay-v2:lineage-seed-commitment:v1"
ESTIMATOR_RANDOM_STATE_DOMAIN: Literal["saliencegate:state-decay-v2:estimator-random-state:v1"] = (
    "saliencegate:state-decay-v2:estimator-random-state:v1"
)
BOOTSTRAP_GOLDEN_SOURCE_DOMAIN: Literal[
    "saliencegate:state-decay-v2:bootstrap-golden-source:v1"
] = "saliencegate:state-decay-v2:bootstrap-golden-source:v1"
BOOTSTRAP_GOLDEN_VECTORS_DOMAIN: Literal[
    "saliencegate:state-decay-v2:bootstrap-golden-vectors:v1"
] = "saliencegate:state-decay-v2:bootstrap-golden-vectors:v1"
FOLD_GOLDEN_DOMAIN: Literal["saliencegate:state-decay-v2:fold-golden:v1"] = (
    "saliencegate:state-decay-v2:fold-golden:v1"
)

NANOSCALE: Literal[1_000_000_000_000] = 1_000_000_000_000
PPM_SCALE: Literal[1_000_000] = 1_000_000
NUISANCE_VECTOR_WIDTH: Literal[1_020] = 1_020


def _exact_bounded_text(value: str) -> str:
    if type(value) is not str:
        raise ValueError("string subclasses are not accepted")
    if not value.strip():
        raise ValueError("text cannot be blank or whitespace-only")
    return value


ProtocolText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=4_096),
    AfterValidator(_exact_bounded_text),
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class ExactRatio(_StrictModel):
    numerator: Annotated[int, Field(ge=-1_000_000, le=1_000_000)]
    denominator: Annotated[int, Field(ge=1, le=1_000_000)]

    @model_validator(mode="after")
    def ratio_is_reduced_and_canonical(self) -> Self:
        if self.numerator == 0 and self.denominator != 1:
            raise ValueError("zero ratio must be canonical zero over one")
        if self.numerator != 0 and gcd(abs(self.numerator), self.denominator) != 1:
            raise ValueError("ratio must be reduced")
        return self


def _ratio(numerator: int, denominator: int) -> ExactRatio:
    return ExactRatio(numerator=numerator, denominator=denominator)


class NuisanceFeatureBlock(StrEnum):
    FIRST_ACTION_INDEX = "first_action_index"
    TRAJECTORY_EVENT_COUNT = "trajectory_event_count"
    CANDIDATE_MEMORY_COUNT = "candidate_memory_count"
    ALLOWED_ACTION_COUNT = "allowed_action_count"
    EVIDENCE_REFERENCE_COUNT = "evidence_reference_count"
    PIVOT_SEQUENCE = "pivot_sequence"
    PIVOT_ACTION_STEP = "pivot_action_step"
    VALIDITY_STATE_COUNTS = "validity_state_counts"
    OPTIONAL_FIELD_PRESENCE = "optional_field_presence"
    IDENTIFIER_BYTE_HISTOGRAM = "identifier_byte_histogram"
    IDENTIFIER_NIBBLE_HISTOGRAM = "identifier_nibble_histogram"
    EVENT_SEQUENCE_SUMMARY = "event_sequence_summary"
    ACTION_STEP_SUMMARY = "action_step_summary"
    MEMORY_RECORDED_SEQUENCE_SUMMARY = "memory_recorded_sequence_summary"
    MEMORY_RECORDED_ACTION_STEP_SUMMARY = "memory_recorded_action_step_summary"
    VALIDITY_SEQUENCE_SUMMARY = "validity_sequence_summary"
    VALIDITY_ACTION_STEP_SUMMARY = "validity_action_step_summary"
    EVIDENCE_EVENT_SEQUENCE_SUMMARY = "evidence_event_sequence_summary"
    MEMORY_REVISION_SUMMARY = "memory_revision_summary"
    EVENT_TEXT_LENGTHS = "event_text_lengths"
    MEMORY_TEXT_LENGTHS = "memory_text_lengths"
    EVIDENCE_TEXT_LENGTHS = "evidence_text_lengths"
    ACTION_TEXT_LENGTHS = "action_text_lengths"


class FeaturePadding(StrEnum):
    NONE = "none"
    ZERO = "zero"
    MINUS_ONE = "minus_one"
    EMPTY_SUMMARY = "empty_summary"


class IdentifierOccurrenceSource(StrEnum):
    SCHEMA_VERSION = "schema_version"
    SUITE_ID = "suite_id"
    SUITE_VERSION = "suite_version"
    SPLIT = "split"
    SCENARIO_ID = "scenario_id"
    TRAJECTORY_EVENT_IDS = "trajectory_event_ids"
    CANDIDATE_MEMORY_IDS = "candidate_memory_ids"
    EVIDENCE_REFERENCE_EVENT_IDS = "evidence_reference_event_ids"
    PIVOT_EVENT_ID = "pivot_event_id"
    ALLOWED_ACTION_IDS = "allowed_action_ids"
    ADAPTER_ID = "adapter_id"
    ADAPTER_VERSION = "adapter_version"
    RESPONSE_PROFILE_ID = "response_profile_id"
    RESPONSE_PROFILE_DIGEST = "response_profile_digest"
    ADAPTER_CAPABILITIES = "adapter_capabilities"


_NUISANCE_LAYOUT: tuple[tuple[NuisanceFeatureBlock, int, int, FeaturePadding], ...] = (
    (NuisanceFeatureBlock.FIRST_ACTION_INDEX, 0, 1, FeaturePadding.NONE),
    (NuisanceFeatureBlock.TRAJECTORY_EVENT_COUNT, 1, 1, FeaturePadding.NONE),
    (NuisanceFeatureBlock.CANDIDATE_MEMORY_COUNT, 2, 1, FeaturePadding.NONE),
    (NuisanceFeatureBlock.ALLOWED_ACTION_COUNT, 3, 1, FeaturePadding.NONE),
    (NuisanceFeatureBlock.EVIDENCE_REFERENCE_COUNT, 4, 1, FeaturePadding.NONE),
    (NuisanceFeatureBlock.PIVOT_SEQUENCE, 5, 1, FeaturePadding.NONE),
    (NuisanceFeatureBlock.PIVOT_ACTION_STEP, 6, 1, FeaturePadding.NONE),
    (NuisanceFeatureBlock.VALIDITY_STATE_COUNTS, 7, 4, FeaturePadding.NONE),
    (NuisanceFeatureBlock.OPTIONAL_FIELD_PRESENCE, 11, 64, FeaturePadding.ZERO),
    (NuisanceFeatureBlock.IDENTIFIER_BYTE_HISTOGRAM, 75, 256, FeaturePadding.NONE),
    (NuisanceFeatureBlock.IDENTIFIER_NIBBLE_HISTOGRAM, 331, 16, FeaturePadding.NONE),
    (NuisanceFeatureBlock.EVENT_SEQUENCE_SUMMARY, 347, 6, FeaturePadding.EMPTY_SUMMARY),
    (NuisanceFeatureBlock.ACTION_STEP_SUMMARY, 353, 6, FeaturePadding.EMPTY_SUMMARY),
    (
        NuisanceFeatureBlock.MEMORY_RECORDED_SEQUENCE_SUMMARY,
        359,
        6,
        FeaturePadding.EMPTY_SUMMARY,
    ),
    (
        NuisanceFeatureBlock.MEMORY_RECORDED_ACTION_STEP_SUMMARY,
        365,
        6,
        FeaturePadding.EMPTY_SUMMARY,
    ),
    (NuisanceFeatureBlock.VALIDITY_SEQUENCE_SUMMARY, 371, 6, FeaturePadding.EMPTY_SUMMARY),
    (NuisanceFeatureBlock.VALIDITY_ACTION_STEP_SUMMARY, 377, 6, FeaturePadding.EMPTY_SUMMARY),
    (
        NuisanceFeatureBlock.EVIDENCE_EVENT_SEQUENCE_SUMMARY,
        383,
        6,
        FeaturePadding.EMPTY_SUMMARY,
    ),
    (NuisanceFeatureBlock.MEMORY_REVISION_SUMMARY, 389, 6, FeaturePadding.EMPTY_SUMMARY),
    (NuisanceFeatureBlock.EVENT_TEXT_LENGTHS, 395, 65, FeaturePadding.MINUS_ONE),
    (NuisanceFeatureBlock.MEMORY_TEXT_LENGTHS, 460, 32, FeaturePadding.MINUS_ONE),
    (NuisanceFeatureBlock.EVIDENCE_TEXT_LENGTHS, 492, 512, FeaturePadding.MINUS_ONE),
    (NuisanceFeatureBlock.ACTION_TEXT_LENGTHS, 1_004, 16, FeaturePadding.MINUS_ONE),
)


class FeatureBlockSpec(_StrictModel):
    block: NuisanceFeatureBlock
    offset: Annotated[int, Field(ge=0, le=1_019)]
    width: Annotated[int, Field(ge=1, le=512)]
    padding: FeaturePadding

    @model_validator(mode="after")
    def block_matches_the_frozen_layout(self) -> Self:
        expected = next(item for item in _NUISANCE_LAYOUT if item[0] is self.block)
        if (self.block, self.offset, self.width, self.padding) != expected:
            raise ValueError("feature block does not match the frozen layout")
        return self


class FeatureSourceRule(_StrictModel):
    block: NuisanceFeatureBlock
    source_rule: ProtocolText


class NuisanceFeatureInventory(_StrictModel):
    schema_version: Literal["state-decay-v2-nuisance-inventory/v1"] = (
        NUISANCE_INVENTORY_SCHEMA_VERSION
    )
    value_type: Literal["signed-int64"] = "signed-int64"
    blocks: Annotated[tuple[FeatureBlockSpec, ...], Field(min_length=23, max_length=23)]
    vector_width: Literal[1_020] = NUISANCE_VECTOR_WIDTH
    first_action_index_rule: Literal[
        "zero-based-rank-of-first-action-id-in-all-action-ids-sorted-canonical-utf8/v1"
    ] = "zero-based-rank-of-first-action-id-in-all-action-ids-sorted-canonical-utf8/v1"
    trajectory_event_count_rule: Literal["len-trajectory-pivot-excluded/v1"] = (
        "len-trajectory-pivot-excluded/v1"
    )
    count_source_rules: Annotated[
        tuple[FeatureSourceRule, ...],
        Field(min_length=4, max_length=4),
    ]
    validity_order: Annotated[tuple[ValidityState, ...], Field(min_length=4, max_length=4)]
    temporal_summary_order: Annotated[tuple[str, ...], Field(min_length=6, max_length=6)]
    empty_summary: Annotated[tuple[int, ...], Field(min_length=6, max_length=6)]
    optional_field_order: Annotated[tuple[str, ...], Field(min_length=2, max_length=2)]
    optional_presence_width: Literal[64] = 64
    optional_presence_rule: Literal[
        "candidate-memory-order-two-presence-bits-right-pad-zero-to-64/v1"
    ] = "candidate-memory-order-two-presence-bits-right-pad-zero-to-64/v1"
    identifier_sources: Annotated[
        tuple[IdentifierOccurrenceSource, ...],
        Field(min_length=15, max_length=15),
    ]
    identifier_rule: Literal["utf8-byte-and-both-nibbles-with-occurrence-multiplicity/v1"] = (
        "utf8-byte-and-both-nibbles-with-occurrence-multiplicity/v1"
    )
    text_length_rule: Literal["utf8-bytes-sorted-ascending-right-pad-minus-one/v1"] = (
        "utf8-bytes-sorted-ascending-right-pad-minus-one/v1"
    )
    evidence_text_rule: Literal["resolved-event-text-reference-multiplicity/v1"] = (
        "resolved-event-text-reference-multiplicity/v1"
    )
    summary_source_rules: Annotated[
        tuple[FeatureSourceRule, ...],
        Field(min_length=8, max_length=8),
    ]
    text_source_rules: Annotated[
        tuple[FeatureSourceRule, ...],
        Field(min_length=4, max_length=4),
    ]
    raw_identifier_numeric_features_allowed: Literal[False] = False
    timestamp_feature_allowed: Literal[False] = False
    signed_int64_overflow_rule: Literal[
        "check-every-feature-and-intermediate-sum-reject-overflow/v1"
    ] = "check-every-feature-and-intermediate-sum-reject-overflow/v1"
    generated_value_min: Literal[0] = 0
    generated_value_max: Literal[1_000_000] = GENERATED_INTEGER_MAX
    ppm_scale: Literal[1_000_000] = PPM_SCALE
    inventory_digest: Sha256Digest

    @model_validator(mode="after")
    def inventory_is_complete_ordered_and_self_attesting(self) -> Self:
        if (
            tuple((item.block, item.offset, item.width, item.padding) for item in self.blocks)
            != _NUISANCE_LAYOUT
        ):
            raise ValueError("nuisance feature blocks are not complete and ordered")
        if sum(item.width for item in self.blocks) != self.vector_width:
            raise ValueError("nuisance vector width does not match its blocks")
        if self.validity_order != tuple(ValidityState):
            raise ValueError("validity order is not canonical")
        if self.temporal_summary_order != (
            "count",
            "min",
            "max",
            "range",
            "sum",
            "unique_count",
        ):
            raise ValueError("temporal summary order is not canonical")
        if self.empty_summary != (0, -1, -1, -1, 0, 0):
            raise ValueError("empty summary sentinel is not canonical")
        if self.optional_field_order != ("validity_sequence", "validity_action_step"):
            raise ValueError("optional field order is not canonical")
        if tuple((item.block, item.source_rule) for item in self.count_source_rules) != (
            (NuisanceFeatureBlock.TRAJECTORY_EVENT_COUNT, "len-trajectory-pivot-excluded"),
            (NuisanceFeatureBlock.CANDIDATE_MEMORY_COUNT, "len-candidate-memories"),
            (NuisanceFeatureBlock.ALLOWED_ACTION_COUNT, "len-allowed-actions"),
            (
                NuisanceFeatureBlock.EVIDENCE_REFERENCE_COUNT,
                "sum-memory-evidence-references-with-multiplicity",
            ),
        ):
            raise ValueError("nuisance count sources are not canonical")
        if self.identifier_sources != tuple(IdentifierOccurrenceSource):
            raise ValueError("identifier source order is not canonical")
        if tuple((item.block, item.source_rule) for item in self.summary_source_rules) != (
            (
                NuisanceFeatureBlock.EVENT_SEQUENCE_SUMMARY,
                "trajectory-sequences-in-order-then-pivot-sequence",
            ),
            (
                NuisanceFeatureBlock.ACTION_STEP_SUMMARY,
                "trajectory-action-steps-in-order-then-pivot-action-step",
            ),
            (
                NuisanceFeatureBlock.MEMORY_RECORDED_SEQUENCE_SUMMARY,
                "candidate-memory-recorded-sequences-in-order",
            ),
            (
                NuisanceFeatureBlock.MEMORY_RECORDED_ACTION_STEP_SUMMARY,
                "candidate-memory-recorded-action-steps-in-order",
            ),
            (
                NuisanceFeatureBlock.VALIDITY_SEQUENCE_SUMMARY,
                "present-validity-sequences-in-candidate-memory-order",
            ),
            (
                NuisanceFeatureBlock.VALIDITY_ACTION_STEP_SUMMARY,
                "present-validity-action-steps-in-candidate-memory-order",
            ),
            (
                NuisanceFeatureBlock.EVIDENCE_EVENT_SEQUENCE_SUMMARY,
                "candidate-memory-then-reference-order-event-sequences-with-multiplicity",
            ),
            (
                NuisanceFeatureBlock.MEMORY_REVISION_SUMMARY,
                "candidate-memory-revisions-once-in-order",
            ),
        ):
            raise ValueError("nuisance summary sources are not canonical")
        if tuple((item.block, item.source_rule) for item in self.text_source_rules) != (
            (
                NuisanceFeatureBlock.EVENT_TEXT_LENGTHS,
                "trajectory-statements-in-order-then-pivot-statement",
            ),
            (
                NuisanceFeatureBlock.MEMORY_TEXT_LENGTHS,
                "candidate-memory-statements-in-order",
            ),
            (
                NuisanceFeatureBlock.EVIDENCE_TEXT_LENGTHS,
                "resolved-event-statements-in-memory-reference-order-with-multiplicity",
            ),
            (
                NuisanceFeatureBlock.ACTION_TEXT_LENGTHS,
                "allowed-action-statements-in-order",
            ),
        ):
            raise ValueError("nuisance text sources are not canonical")
        if not hmac.compare_digest(self.inventory_digest, nuisance_inventory_digest(self)):
            raise ValueError("nuisance inventory digest does not match")
        return self


def nuisance_inventory_digest(value: BaseModel | Mapping[str, object]) -> str:
    payload = (
        value.model_dump(mode="json", exclude={"inventory_digest"}, warnings=False)
        if isinstance(value, BaseModel)
        else {key: item for key, item in value.items() if key != "inventory_digest"}
    )
    return length_prefixed_sha256(
        canonical_json(payload),
        domain=NUISANCE_INVENTORY_DIGEST_DOMAIN,
    )


def build_nuisance_feature_inventory() -> NuisanceFeatureInventory:
    values: dict[str, object] = {
        "schema_version": NUISANCE_INVENTORY_SCHEMA_VERSION,
        "value_type": "signed-int64",
        "blocks": tuple(
            FeatureBlockSpec(block=block, offset=offset, width=width, padding=padding)
            for block, offset, width, padding in _NUISANCE_LAYOUT
        ),
        "vector_width": NUISANCE_VECTOR_WIDTH,
        "first_action_index_rule": (
            "zero-based-rank-of-first-action-id-in-all-action-ids-sorted-canonical-utf8/v1"
        ),
        "trajectory_event_count_rule": "len-trajectory-pivot-excluded/v1",
        "count_source_rules": (
            FeatureSourceRule(
                block=NuisanceFeatureBlock.TRAJECTORY_EVENT_COUNT,
                source_rule="len-trajectory-pivot-excluded",
            ),
            FeatureSourceRule(
                block=NuisanceFeatureBlock.CANDIDATE_MEMORY_COUNT,
                source_rule="len-candidate-memories",
            ),
            FeatureSourceRule(
                block=NuisanceFeatureBlock.ALLOWED_ACTION_COUNT,
                source_rule="len-allowed-actions",
            ),
            FeatureSourceRule(
                block=NuisanceFeatureBlock.EVIDENCE_REFERENCE_COUNT,
                source_rule="sum-memory-evidence-references-with-multiplicity",
            ),
        ),
        "validity_order": tuple(ValidityState),
        "temporal_summary_order": ("count", "min", "max", "range", "sum", "unique_count"),
        "empty_summary": (0, -1, -1, -1, 0, 0),
        "optional_field_order": ("validity_sequence", "validity_action_step"),
        "optional_presence_width": 64,
        "optional_presence_rule": (
            "candidate-memory-order-two-presence-bits-right-pad-zero-to-64/v1"
        ),
        "identifier_sources": tuple(IdentifierOccurrenceSource),
        "identifier_rule": "utf8-byte-and-both-nibbles-with-occurrence-multiplicity/v1",
        "text_length_rule": "utf8-bytes-sorted-ascending-right-pad-minus-one/v1",
        "evidence_text_rule": "resolved-event-text-reference-multiplicity/v1",
        "summary_source_rules": (
            FeatureSourceRule(
                block=NuisanceFeatureBlock.EVENT_SEQUENCE_SUMMARY,
                source_rule="trajectory-sequences-in-order-then-pivot-sequence",
            ),
            FeatureSourceRule(
                block=NuisanceFeatureBlock.ACTION_STEP_SUMMARY,
                source_rule="trajectory-action-steps-in-order-then-pivot-action-step",
            ),
            FeatureSourceRule(
                block=NuisanceFeatureBlock.MEMORY_RECORDED_SEQUENCE_SUMMARY,
                source_rule="candidate-memory-recorded-sequences-in-order",
            ),
            FeatureSourceRule(
                block=NuisanceFeatureBlock.MEMORY_RECORDED_ACTION_STEP_SUMMARY,
                source_rule="candidate-memory-recorded-action-steps-in-order",
            ),
            FeatureSourceRule(
                block=NuisanceFeatureBlock.VALIDITY_SEQUENCE_SUMMARY,
                source_rule="present-validity-sequences-in-candidate-memory-order",
            ),
            FeatureSourceRule(
                block=NuisanceFeatureBlock.VALIDITY_ACTION_STEP_SUMMARY,
                source_rule="present-validity-action-steps-in-candidate-memory-order",
            ),
            FeatureSourceRule(
                block=NuisanceFeatureBlock.EVIDENCE_EVENT_SEQUENCE_SUMMARY,
                source_rule=(
                    "candidate-memory-then-reference-order-event-sequences-with-multiplicity"
                ),
            ),
            FeatureSourceRule(
                block=NuisanceFeatureBlock.MEMORY_REVISION_SUMMARY,
                source_rule="candidate-memory-revisions-once-in-order",
            ),
        ),
        "text_source_rules": (
            FeatureSourceRule(
                block=NuisanceFeatureBlock.EVENT_TEXT_LENGTHS,
                source_rule="trajectory-statements-in-order-then-pivot-statement",
            ),
            FeatureSourceRule(
                block=NuisanceFeatureBlock.MEMORY_TEXT_LENGTHS,
                source_rule="candidate-memory-statements-in-order",
            ),
            FeatureSourceRule(
                block=NuisanceFeatureBlock.EVIDENCE_TEXT_LENGTHS,
                source_rule=(
                    "resolved-event-statements-in-memory-reference-order-with-multiplicity"
                ),
            ),
            FeatureSourceRule(
                block=NuisanceFeatureBlock.ACTION_TEXT_LENGTHS,
                source_rule="allowed-action-statements-in-order",
            ),
        ),
        "raw_identifier_numeric_features_allowed": False,
        "timestamp_feature_allowed": False,
        "signed_int64_overflow_rule": (
            "check-every-feature-and-intermediate-sum-reject-overflow/v1"
        ),
        "generated_value_min": 0,
        "generated_value_max": GENERATED_INTEGER_MAX,
        "ppm_scale": PPM_SCALE,
    }
    values["inventory_digest"] = nuisance_inventory_digest(values)
    return NuisanceFeatureInventory.model_validate(values)


NUISANCE_FEATURE_INVENTORY = build_nuisance_feature_inventory()


class LeakageEstimand(StrEnum):
    HELPFUL_VS_REST = "helpful_vs_rest"
    FOUR_OUTCOME = "four_outcome"
    RESOLVED_ONLY = "resolved_only"


class ShortcutBaseline(StrEnum):
    MAJORITY = "majority"
    FIRST_ACTION = "first_action"
    MEMORY_VALIDITY = "memory_validity"
    LENGTH_ONLY = "length_only"
    STRUCTURAL_ONLY = "structural_only"
    IDENTIFIER_ONLY = "identifier_only"
    SINGLE_FIELD_LOOKUP = "single_field_lookup"
    LOGISTIC = "logistic"
    TREE = "tree"


class LeakageNullFixture(StrEnum):
    CONSTANT_EMPIRICAL_PRIOR = "constant_empirical_prior"
    UNIFORM_CLASS_PROBABILITY = "uniform_class_probability"


class ReviewBoundary(StrEnum):
    PUBLIC = "public"
    CUSTODY = "custody"


class ReviewDecision(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class PermutationGoldenDraw(_StrictModel):
    step: Annotated[int, Field(ge=1, le=4)]
    attempt: Annotated[int, Field(ge=0, le=(1 << 64) - 1)]
    draw_digest: Sha256Digest
    selected_index: Annotated[int, Field(ge=0, le=4)]


class PermutationGoldenFixture(_StrictModel):
    replicate: Annotated[int, Field(ge=0, le=9_999)]
    family: ScenarioFamily
    lineage_registry_key: ComponentIdentifier
    draws: Annotated[tuple[PermutationGoldenDraw, ...], Field(min_length=4, max_length=4)]
    final_outcomes: Annotated[tuple[ScenarioOutcome, ...], Field(min_length=5, max_length=5)]


class FoldProtocol(_StrictModel):
    fold_count: Literal[5] = 5
    lineages_per_family: Literal[30] = LINEAGES_PER_FAMILY
    lineages_per_fold: Literal[6] = 6
    hash_domain: Literal["saliencegate:state-decay-v2:fold-key:v1"] = FOLD_KEY_DOMAIN
    hash_primitive: Literal["length-prefixed-sha256/v1"] = "length-prefixed-sha256/v1"
    coordinate_order: Annotated[tuple[str, ...], Field(min_length=2, max_length=2)]
    coordinate_encoding: Annotated[tuple[str, ...], Field(min_length=2, max_length=2)]
    tie_break: Literal["canonical-lineage-registry-key-utf8/v1"] = (
        "canonical-lineage-registry-key-utf8/v1"
    )
    assignment: Literal["sort-hash-then-six-consecutive-lineages/v1"] = (
        "sort-hash-then-six-consecutive-lineages/v1"
    )
    seed_free: Literal[True] = True
    indivisible_unit: Literal["family-lineage"] = "family-lineage"
    golden_domain: Literal["saliencegate:state-decay-v2:fold-golden:v1"] = FOLD_GOLDEN_DOMAIN
    golden_family: Literal[ScenarioFamily.FORGOTTEN_REQUIREMENT] = (
        ScenarioFamily.FORGOTTEN_REQUIREMENT
    )
    golden_key_format: Literal["golden-lineage-{zero-based-index:02d}"] = (
        "golden-lineage-{zero-based-index:02d}"
    )
    golden_first_key_digest: Literal[
        "0a8a9ec3b2479683006cf2a2816c624ec2a7a526abdb2f98173f38e86bdb56bb"
    ] = "0a8a9ec3b2479683006cf2a2816c624ec2a7a526abdb2f98173f38e86bdb56bb"
    golden_fixture_representation: Literal[
        "sorted-fold-key-digest-lineage-key-fold-index-canonical-json/v1"
    ] = "sorted-fold-key-digest-lineage-key-fold-index-canonical-json/v1"
    golden_fixture_digest: Literal[
        "58006ccf81dcd15c8081a30fcdf19d0456c332e2ad4fcbee543cbfb1929bac68"
    ] = "58006ccf81dcd15c8081a30fcdf19d0456c332e2ad4fcbee543cbfb1929bac68"

    @model_validator(mode="after")
    def fold_coordinates_are_frozen(self) -> Self:
        if self.coordinate_order != ("family", "pre_id_lineage_registry_key"):
            raise ValueError("fold coordinate order is not canonical")
        if self.coordinate_encoding != ("utf8", "utf8"):
            raise ValueError("fold coordinate encoding is not canonical")
        keys = tuple(f"golden-lineage-{index:02d}" for index in range(LINEAGES_PER_FAMILY))
        rows = tuple(
            sorted(
                (
                    (
                        key,
                        length_prefixed_sha256(
                            self.golden_family.value,
                            key,
                            domain=FOLD_KEY_DOMAIN,
                        ),
                    )
                    for key in keys
                ),
                key=lambda item: (item[1], item[0].encode("utf-8")),
            )
        )
        if rows[0][1] != self.golden_first_key_digest:
            raise ValueError("fold first golden digest does not match")
        fixture = tuple(
            {
                "lineage_registry_key": key,
                "fold_key_digest": digest,
                "fold_index": rank // self.lineages_per_fold,
            }
            for rank, (key, digest) in enumerate(rows)
        )
        fixture_digest = length_prefixed_sha256(
            canonical_json(fixture),
            domain=FOLD_GOLDEN_DOMAIN,
        )
        if not hmac.compare_digest(fixture_digest, self.golden_fixture_digest):
            raise ValueError("fold golden fixture digest does not match")
        return self


class PermutationProtocol(_StrictModel):
    draw_count: Literal[10_000] = 10_000
    seed_purpose: Literal[SeedPurpose.PERMUTATION] = SeedPurpose.PERMUTATION
    public_seed_commitment_digest: Sha256Digest
    custody_seed_binding: Literal["suite-generation-commitment-required/v1"] = (
        "suite-generation-commitment-required/v1"
    )
    hash_domain: Literal["saliencegate:state-decay-v2:permutation-index:v1"] = (
        PERMUTATION_INDEX_DOMAIN
    )
    hash_primitive: Literal["length-prefixed-sha256/v1"] = "length-prefixed-sha256/v1"
    label_multiset: Annotated[tuple[ScenarioOutcome, ...], Field(min_length=5, max_length=5)]
    fisher_yates_steps: Annotated[tuple[int, ...], Field(min_length=4, max_length=4)]
    coordinate_order: Annotated[tuple[str, ...], Field(min_length=5, max_length=5)]
    integer_coordinate_encoding: Literal["unsigned-64-bit-big-endian"] = (
        "unsigned-64-bit-big-endian"
    )
    replicate_origin: Literal[0] = 0
    initial_attempt: Literal[0] = 0
    digest_integer: Literal["unsigned-big-endian-256"] = "unsigned-big-endian-256"
    sampling: Literal["rejection-below-2^256-modulo-width/v1"] = (
        "rejection-below-2^256-modulo-width/v1"
    )
    rejection_attempt_rule: Literal["increment-attempt"] = "increment-attempt"
    exceedance_rule: Literal["statistic-greater-than-or-equal-observed"] = (
        "statistic-greater-than-or-equal-observed"
    )
    p_value_rule: Literal["plus-one/v1"] = "plus-one/v1"
    interval_quantiles: Annotated[tuple[ExactRatio, ...], Field(min_length=2, max_length=2)]
    interval_rule: Literal["one-indexed-nearest-rank/v1"] = "one-indexed-nearest-rank/v1"
    golden_seed_binding: Literal["public-generation-permutation-leaf/v1"] = (
        "public-generation-permutation-leaf/v1"
    )
    golden_fixtures: Annotated[
        tuple[PermutationGoldenFixture, ...],
        Field(min_length=2, max_length=2),
    ]

    @model_validator(mode="after")
    def permutation_is_complete_and_canonical(self) -> Self:
        if self.label_multiset != (
            ScenarioOutcome.HELPFUL,
            ScenarioOutcome.HELPFUL,
            ScenarioOutcome.HARMFUL,
            ScenarioOutcome.REDUNDANT,
            ScenarioOutcome.UNRESOLVED,
        ):
            raise ValueError("permutation label multiset is not canonical")
        if self.fisher_yates_steps != (4, 3, 2, 1):
            raise ValueError("Fisher-Yates steps are not canonical")
        if self.coordinate_order != (
            "replicate_u64be",
            "family",
            "lineage_registry_key",
            "step_u64be",
            "attempt_u64be",
        ):
            raise ValueError("permutation coordinate order is not canonical")
        if self.interval_quantiles != (_ratio(1, 40), _ratio(39, 40)):
            raise ValueError("permutation interval quantiles are not canonical")
        expected_commitment = seed_commitment(
            derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.PERMUTATION),
            SeedPurpose.PERMUTATION,
        )
        if not hmac.compare_digest(self.public_seed_commitment_digest, expected_commitment):
            raise ValueError("public permutation seed commitment does not match")
        permutation_leaf = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.PERMUTATION)
        for fixture in self.golden_fixtures:
            outcomes = list(self.label_multiset)
            if tuple(item.step for item in fixture.draws) != self.fisher_yates_steps:
                raise ValueError("permutation golden steps are not canonical")
            for draw in fixture.draws:
                digest = length_prefixed_sha256(
                    permutation_leaf,
                    u64be(fixture.replicate),
                    fixture.family.value,
                    fixture.lineage_registry_key,
                    u64be(draw.step),
                    u64be(draw.attempt),
                    domain=PERMUTATION_INDEX_DOMAIN,
                )
                width = draw.step + 1
                limit = (1 << 256) - ((1 << 256) % width)
                integer = int(digest, 16)
                if integer >= limit:
                    raise ValueError("permutation golden unexpectedly requires rejection")
                if digest != draw.draw_digest or integer % width != draw.selected_index:
                    raise ValueError("permutation golden draw does not match")
                outcomes[draw.step], outcomes[draw.selected_index] = (
                    outcomes[draw.selected_index],
                    outcomes[draw.step],
                )
            if tuple(outcomes) != fixture.final_outcomes:
                raise ValueError("permutation golden result does not match")
        return self


class FrozenEstimatorArgument(_StrictModel):
    name: ComponentIdentifier
    json_literal: ProtocolText


_SCALER_ARGUMENTS = (
    ("copy", "true"),
    ("with_mean", "true"),
    ("with_std", "true"),
)
_LOGISTIC_ARGUMENTS = (
    ("l1_ratio", "0.0"),
    ("C", "1.0"),
    ("dual", "false"),
    ("tol", "1e-10"),
    ("fit_intercept", "true"),
    ("intercept_scaling", "1"),
    ("class_weight", "null"),
    ("solver", '"lbfgs"'),
    ("max_iter", "2000"),
    ("verbose", "0"),
    ("warm_start", "false"),
    ("random_state", '"seed-derived-u32"'),
)
_TREE_ARGUMENTS = (
    ("criterion", '"gini"'),
    ("splitter", '"best"'),
    ("max_depth", "3"),
    ("min_samples_split", "2"),
    ("min_samples_leaf", "10"),
    ("min_weight_fraction_leaf", "0.0"),
    ("max_features", "null"),
    ("random_state", '"seed-derived-u32"'),
    ("max_leaf_nodes", "null"),
    ("min_impurity_decrease", "0.0"),
    ("class_weight", "null"),
    ("ccp_alpha", "0.0"),
    ("monotonic_cst", "null"),
)


def _frozen_arguments(values: tuple[tuple[str, str], ...]) -> tuple[FrozenEstimatorArgument, ...]:
    return tuple(FrozenEstimatorArgument(name=name, json_literal=value) for name, value in values)


class ProbabilityConversionProtocol(_StrictModel):
    input_type: Literal["numpy.float64"] = "numpy.float64"
    input_requirements: Annotated[tuple[str, ...], Field(min_length=2, max_length=2)]
    conversion_order: Annotated[tuple[str, ...], Field(min_length=4, max_length=4)]
    ppm_scale: Literal[1_000_000] = PPM_SCALE
    rounding: Literal["ROUND_HALF_UP"] = "ROUND_HALF_UP"

    @model_validator(mode="after")
    def probability_conversion_is_exact(self) -> Self:
        if self.input_requirements != ("finite", "closed-interval-zero-one"):
            raise ValueError("probability input requirements are not canonical")
        if self.conversion_order != (
            "numpy-float64-item",
            "python-float-repr",
            "decimal-from-repr",
            "ppm-round-half-up",
        ):
            raise ValueError("probability conversion order is not canonical")
        return self


class OutcomeIntegerEncoding(_StrictModel):
    outcome: ScenarioOutcome
    integer_label: Annotated[int, Field(ge=0, le=3)]


class ResearchAdapterProtocol(_StrictModel):
    adapter_id: Literal["state-decay-v2-sklearn"] = "state-decay-v2-sklearn"
    adapter_version: Literal["state-decay-v2-sklearn-1.9/v1"] = "state-decay-v2-sklearn-1.9/v1"
    dependencies: Annotated[tuple[str, ...], Field(min_length=2, max_length=2)]
    scaler_arguments: Annotated[
        tuple[FrozenEstimatorArgument, ...],
        Field(min_length=3, max_length=3),
    ]
    logistic_arguments: Annotated[
        tuple[FrozenEstimatorArgument, ...],
        Field(min_length=12, max_length=12),
    ]
    logistic_omitted_arguments: Annotated[tuple[str, ...], Field(min_length=2, max_length=2)]
    tree_arguments: Annotated[
        tuple[FrozenEstimatorArgument, ...],
        Field(min_length=13, max_length=13),
    ]
    random_state_domain: Literal["saliencegate:state-decay-v2:estimator-random-state:v1"] = (
        ESTIMATOR_RANDOM_STATE_DOMAIN
    )
    random_state_seed_purpose: Literal[SeedPurpose.PERMUTATION] = SeedPurpose.PERMUTATION
    random_state_hash_primitive: Literal["length-prefixed-sha256/v1"] = "length-prefixed-sha256/v1"
    random_state_coordinates: Annotated[tuple[str, ...], Field(min_length=4, max_length=4)]
    random_state_integer_coordinate_encoding: Literal["unsigned-64-bit-big-endian"] = (
        "unsigned-64-bit-big-endian"
    )
    random_state_digest_slice: Literal["first-four-bytes"] = "first-four-bytes"
    random_state_bytes: Literal[4] = 4
    random_state_byte_order: Literal["unsigned-big-endian"] = "unsigned-big-endian"
    random_state_golden_estimator: Literal["logistic"] = "logistic"
    random_state_golden_fold: Literal[0] = 0
    random_state_golden_estimand: Literal[LeakageEstimand.HELPFUL_VS_REST] = (
        LeakageEstimand.HELPFUL_VS_REST
    )
    random_state_golden_digest: Literal[
        "051609cc385ac68e861577a2da29705117e5058a1910616b74f8e8273540f72c"
    ] = "051609cc385ac68e861577a2da29705117e5058a1910616b74f8e8273540f72c"
    random_state_golden_u32: Literal[85_330_380] = 85_330_380
    class_order: Annotated[tuple[ScenarioOutcome, ...], Field(min_length=4, max_length=4)]
    label_integer_encoding: Annotated[
        tuple[OutcomeIntegerEncoding, ...],
        Field(min_length=4, max_length=4),
    ]
    logistic_pipeline: Annotated[tuple[str, ...], Field(min_length=2, max_length=2)]
    tree_pipeline: Annotated[tuple[str, ...], Field(min_length=1, max_length=1)]
    logistic_feature_blocks: Annotated[
        tuple[NuisanceFeatureBlock, ...],
        Field(min_length=23, max_length=23),
    ]
    tree_feature_blocks: Annotated[
        tuple[NuisanceFeatureBlock, ...],
        Field(min_length=23, max_length=23),
    ]
    preprocessing_fit_scope: Literal["training-fold-only"] = "training-fold-only"
    estimator_refit_scope: Literal["each-fold-and-each-permutation"] = (
        "each-fold-and-each-permutation"
    )
    probability_conversion: ProbabilityConversionProtocol
    thread_limit: Literal[1] = 1
    thread_environment_variables: Annotated[
        tuple[str, ...],
        Field(min_length=6, max_length=6),
    ]
    threadpoolctl_required: Literal[True] = True
    fallback_allowed: Literal[False] = False
    warning_or_convergence_mismatch: Literal["blocking"] = "blocking"

    @model_validator(mode="after")
    def adapter_is_fully_preregistered(self) -> Self:
        if self.dependencies != ("numpy==2.3.5", "scikit-learn==1.9.0"):
            raise ValueError("research dependency versions are not canonical")
        if tuple((item.name, item.json_literal) for item in self.scaler_arguments) != (
            _SCALER_ARGUMENTS
        ):
            raise ValueError("scaler arguments are not canonical")
        if tuple((item.name, item.json_literal) for item in self.logistic_arguments) != (
            _LOGISTIC_ARGUMENTS
        ):
            raise ValueError("logistic arguments are not canonical")
        if self.logistic_omitted_arguments != ("penalty", "n_jobs"):
            raise ValueError("logistic omitted arguments are not canonical")
        if tuple((item.name, item.json_literal) for item in self.tree_arguments) != _TREE_ARGUMENTS:
            raise ValueError("tree arguments are not canonical")
        if self.random_state_coordinates != (
            "permutation_leaf",
            "estimator",
            "fold_u64be",
            "estimand",
        ):
            raise ValueError("estimator random-state coordinates are not canonical")
        random_state_digest = length_prefixed_sha256(
            derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.PERMUTATION),
            self.random_state_golden_estimator,
            u64be(self.random_state_golden_fold),
            self.random_state_golden_estimand.value,
            domain=ESTIMATOR_RANDOM_STATE_DOMAIN,
        )
        random_state = int.from_bytes(bytes.fromhex(random_state_digest)[:4], byteorder="big")
        if (
            random_state_digest != self.random_state_golden_digest
            or random_state != self.random_state_golden_u32
        ):
            raise ValueError("estimator random-state golden does not match")
        if self.class_order != tuple(ScenarioOutcome):
            raise ValueError("estimator class order is not canonical")
        if tuple(
            (item.outcome, item.integer_label) for item in self.label_integer_encoding
        ) != tuple((outcome, index) for index, outcome in enumerate(ScenarioOutcome)):
            raise ValueError("estimator label encoding is not canonical")
        if self.logistic_pipeline != ("standard_scaler", "logistic_regression"):
            raise ValueError("logistic pipeline is not canonical")
        if self.tree_pipeline != ("decision_tree_unscaled",):
            raise ValueError("tree pipeline is not canonical")
        if self.logistic_feature_blocks != tuple(NuisanceFeatureBlock):
            raise ValueError("logistic feature blocks are not complete and ordered")
        if self.tree_feature_blocks != tuple(NuisanceFeatureBlock):
            raise ValueError("tree feature blocks are not complete and ordered")
        if self.thread_environment_variables != (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            raise ValueError("thread environment is not canonical")
        return self


class BaselineFeatureSubset(_StrictModel):
    baseline: ShortcutBaseline
    blocks: Annotated[tuple[NuisanceFeatureBlock, ...], Field(min_length=1, max_length=23)]


class LeakageCeiling(_StrictModel):
    estimand: LeakageEstimand
    balanced_accuracy_max_ppm: Annotated[int | None, Field(ge=0, le=PPM_SCALE)]
    average_precision_max_ppm: Annotated[int | None, Field(ge=0, le=PPM_SCALE)]
    governed: bool


class LeakageNullGolden(_StrictModel):
    fixture: LeakageNullFixture
    estimand: LeakageEstimand
    class_order: Annotated[tuple[str, ...], Field(min_length=2, max_length=4)]
    probability_numerators: Annotated[tuple[int, ...], Field(min_length=2, max_length=4)]
    probability_denominator: Annotated[int, Field(ge=1, le=5)]
    hard_label_rule: Literal["argmax-then-class-order"] = "argmax-then-class-order"
    hard_label: ProtocolText
    balanced_accuracy_ppm: Annotated[int, Field(ge=0, le=PPM_SCALE)]
    average_precision_ppm: Annotated[int, Field(ge=0, le=PPM_SCALE)]

    @model_validator(mode="after")
    def probability_vector_is_complete(self) -> Self:
        if len(self.class_order) != len(self.probability_numerators):
            raise ValueError("null golden class and probability widths differ")
        if len(set(self.class_order)) != len(self.class_order):
            raise ValueError("null golden class order contains duplicates")
        if any(type(item) is not int or item < 0 for item in self.probability_numerators):
            raise ValueError("null golden probability numerator is invalid")
        if sum(self.probability_numerators) != self.probability_denominator:
            raise ValueError("null golden probabilities do not sum to one")
        if self.hard_label not in self.class_order:
            raise ValueError("null golden hard label is outside its class order")
        return self


class LeakageClassLabel(StrEnum):
    NEGATIVE = "negative"
    HELPFUL = "helpful"
    HARMFUL = "harmful"
    REDUNDANT = "redundant"
    UNRESOLVED = "unresolved"


class LeakageOutcomeClass(_StrictModel):
    outcome: ScenarioOutcome
    assigned_class: LeakageClassLabel | None


class LeakageClassIntegerEncoding(_StrictModel):
    class_label: LeakageClassLabel
    integer_label: Annotated[int, Field(ge=0, le=3)]


class LeakageEstimandClasses(_StrictModel):
    estimand: LeakageEstimand
    included_outcomes: Annotated[tuple[ScenarioOutcome, ...], Field(min_length=3, max_length=4)]
    outcome_classes: Annotated[tuple[LeakageOutcomeClass, ...], Field(min_length=4, max_length=4)]
    class_order: Annotated[tuple[LeakageClassLabel, ...], Field(min_length=2, max_length=4)]
    class_integer_encoding: Annotated[
        tuple[LeakageClassIntegerEncoding, ...],
        Field(min_length=2, max_length=4),
    ]
    positive_class: LeakageClassLabel | None
    majority_tie_order: Annotated[tuple[LeakageClassLabel, ...], Field(min_length=2, max_length=4)]
    lookup_tie_order: Annotated[tuple[LeakageClassLabel, ...], Field(min_length=2, max_length=4)]
    row_inclusion_rule: ProtocolText

    @model_validator(mode="after")
    def estimand_classes_are_closed_and_ordered(self) -> Self:
        if tuple(item.outcome for item in self.outcome_classes) != tuple(ScenarioOutcome):
            raise ValueError("leakage outcome class map is not complete and ordered")
        included_from_map = tuple(
            item.outcome for item in self.outcome_classes if item.assigned_class is not None
        )
        if included_from_map != self.included_outcomes:
            raise ValueError("leakage included outcomes diverge from the class map")
        mapped_classes = {item.assigned_class for item in self.outcome_classes} - {None}
        if mapped_classes != set(self.class_order) or len(set(self.class_order)) != len(
            self.class_order
        ):
            raise ValueError("leakage class order does not cover the mapped classes")
        if tuple(
            (item.class_label, item.integer_label) for item in self.class_integer_encoding
        ) != tuple((label, index) for index, label in enumerate(self.class_order)):
            raise ValueError("leakage class integer encoding is not canonical")
        if self.majority_tie_order != self.class_order or self.lookup_tie_order != self.class_order:
            raise ValueError("leakage baseline tie order diverges from its class order")
        if self.positive_class is not None and self.positive_class not in self.class_order:
            raise ValueError("leakage positive class is outside its class order")
        return self


class LeakageAuditScope(_StrictModel):
    scope_id: ComponentIdentifier
    splits: Annotated[tuple[BenchmarkSplit, ...], Field(min_length=1, max_length=3)]
    boundary: ReviewBoundary
    analysis_seed_source: SeedSourceBoundary
    analysis_seed_purpose: Literal[SeedPurpose.PERMUTATION] = SeedPurpose.PERMUTATION
    analysis_seed_consumers: Annotated[tuple[str, ...], Field(min_length=3, max_length=3)]
    analysis_seed_scope_rule: Literal["one-purpose-leaf-for-entire-audit-scope/v1"] = (
        "one-purpose-leaf-for-entire-audit-scope/v1"
    )
    row_count: Annotated[int, Field(ge=900, le=1_200)]
    family_count: Annotated[int, Field(ge=6, le=8)]
    reuse_frozen_methods_and_ceilings: Literal[True] = True
    tuning_allowed: Literal[False] = False


_LEAKAGE_NULL_GOLDENS: tuple[
    tuple[
        LeakageNullFixture,
        LeakageEstimand,
        tuple[str, ...],
        tuple[int, ...],
        int,
        str,
        int,
        int,
    ],
    ...,
] = (
    (
        LeakageNullFixture.CONSTANT_EMPIRICAL_PRIOR,
        LeakageEstimand.HELPFUL_VS_REST,
        ("negative", "helpful"),
        (3, 2),
        5,
        "negative",
        500_000,
        400_000,
    ),
    (
        LeakageNullFixture.CONSTANT_EMPIRICAL_PRIOR,
        LeakageEstimand.FOUR_OUTCOME,
        ("helpful", "harmful", "redundant", "unresolved"),
        (2, 1, 1, 1),
        5,
        "helpful",
        250_000,
        250_000,
    ),
    (
        LeakageNullFixture.CONSTANT_EMPIRICAL_PRIOR,
        LeakageEstimand.RESOLVED_ONLY,
        ("negative", "helpful"),
        (1, 1),
        2,
        "negative",
        500_000,
        500_000,
    ),
    (
        LeakageNullFixture.UNIFORM_CLASS_PROBABILITY,
        LeakageEstimand.HELPFUL_VS_REST,
        ("negative", "helpful"),
        (1, 1),
        2,
        "negative",
        500_000,
        400_000,
    ),
    (
        LeakageNullFixture.UNIFORM_CLASS_PROBABILITY,
        LeakageEstimand.FOUR_OUTCOME,
        ("helpful", "harmful", "redundant", "unresolved"),
        (1, 1, 1, 1),
        4,
        "helpful",
        250_000,
        250_000,
    ),
    (
        LeakageNullFixture.UNIFORM_CLASS_PROBABILITY,
        LeakageEstimand.RESOLVED_ONLY,
        ("negative", "helpful"),
        (1, 1),
        2,
        "negative",
        500_000,
        500_000,
    ),
)

_LEAKAGE_CLASS_CONTRACTS: tuple[
    tuple[
        LeakageEstimand,
        tuple[ScenarioOutcome, ...],
        tuple[LeakageClassLabel | None, ...],
        tuple[LeakageClassLabel, ...],
        LeakageClassLabel | None,
        str,
    ],
    ...,
] = (
    (
        LeakageEstimand.HELPFUL_VS_REST,
        tuple(ScenarioOutcome),
        (
            LeakageClassLabel.HELPFUL,
            LeakageClassLabel.NEGATIVE,
            LeakageClassLabel.NEGATIVE,
            LeakageClassLabel.NEGATIVE,
        ),
        (LeakageClassLabel.NEGATIVE, LeakageClassLabel.HELPFUL),
        LeakageClassLabel.HELPFUL,
        "include-all-outcome-labels-after-each-permutation",
    ),
    (
        LeakageEstimand.FOUR_OUTCOME,
        tuple(ScenarioOutcome),
        (
            LeakageClassLabel.HELPFUL,
            LeakageClassLabel.HARMFUL,
            LeakageClassLabel.REDUNDANT,
            LeakageClassLabel.UNRESOLVED,
        ),
        (
            LeakageClassLabel.HELPFUL,
            LeakageClassLabel.HARMFUL,
            LeakageClassLabel.REDUNDANT,
            LeakageClassLabel.UNRESOLVED,
        ),
        None,
        "include-all-outcome-labels-after-each-permutation",
    ),
    (
        LeakageEstimand.RESOLVED_ONLY,
        (
            ScenarioOutcome.HELPFUL,
            ScenarioOutcome.HARMFUL,
            ScenarioOutcome.REDUNDANT,
        ),
        (
            LeakageClassLabel.HELPFUL,
            LeakageClassLabel.NEGATIVE,
            LeakageClassLabel.NEGATIVE,
            None,
        ),
        (LeakageClassLabel.NEGATIVE, LeakageClassLabel.HELPFUL),
        LeakageClassLabel.HELPFUL,
        "exclude-unresolved-outcome-label-after-each-permutation",
    ),
)


class LeakageProtocol(_StrictModel):
    schema_version: Literal["state-decay-v2-leakage-protocol/v1"] = LEAKAGE_PROTOCOL_SCHEMA_VERSION
    nuisance_inventory_digest: Sha256Digest
    audit_scopes: Annotated[tuple[LeakageAuditScope, ...], Field(min_length=3, max_length=3)]
    row_scope: Literal["all-pivots-in-audit-scope"] = "all-pivots-in-audit-scope"
    fold: FoldProtocol
    permutation: PermutationProtocol
    baseline_order: Annotated[tuple[ShortcutBaseline, ...], Field(min_length=9, max_length=9)]
    feature_subsets: Annotated[tuple[BaselineFeatureSubset, ...], Field(min_length=5, max_length=5)]
    single_field_lookup_universe: Annotated[
        tuple[NuisanceFeatureBlock, ...],
        Field(min_length=23, max_length=23),
    ]
    single_field_lookup_key_unit: Literal["one-flat-signed-int64-coordinate/v1"] = (
        "one-flat-signed-int64-coordinate/v1"
    )
    single_field_lookup_candidate_order: Literal[
        "nuisance-block-order-then-zero-based-coordinate/v1"
    ] = "nuisance-block-order-then-zero-based-coordinate/v1"
    single_field_lookup_candidate_count: Literal[1_020] = NUISANCE_VECTOR_WIDTH
    unseen_lookup_rule: Literal["training-prior"] = "training-prior"
    single_field_selection: Literal[
        "maximum-preregistered-held-out-statistic-across-candidate-coordinates/v1"
    ] = "maximum-preregistered-held-out-statistic-across-candidate-coordinates/v1"
    estimand_classes: Annotated[
        tuple[LeakageEstimandClasses, ...],
        Field(min_length=3, max_length=3),
    ]
    heldout_aggregation: Literal[
        "concatenate-oof-predictions-in-canonical-scenario-order-then-one-global-metric"
    ] = "concatenate-oof-predictions-in-canonical-scenario-order-then-one-global-metric"
    fold_metric_average_allowed: Literal[False] = False
    ceiling_scope: Literal["every-baseline-every-governed-estimand"] = (
        "every-baseline-every-governed-estimand"
    )
    permutation_reuse_scope: Literal["same-permutations-all-baselines-and-statistics"] = (
        "same-permutations-all-baselines-and-statistics"
    )
    balanced_accuracy_rule: Literal["unadjusted-arithmetic-mean-class-recall/v1"] = (
        "unadjusted-arithmetic-mean-class-recall/v1"
    )
    average_precision_rule: Literal["full-tie-group-average-precision/v1"] = (
        "full-tie-group-average-precision/v1"
    )
    macro_average_precision_rule: Literal["mean-four-full-tie-one-vs-rest/v1"] = (
        "mean-four-full-tie-one-vs-rest/v1"
    )
    adapter: ResearchAdapterProtocol
    ceilings: Annotated[tuple[LeakageCeiling, ...], Field(min_length=3, max_length=3)]
    null_goldens: Annotated[tuple[LeakageNullGolden, ...], Field(min_length=6, max_length=6)]
    protocol_digest: Sha256Digest

    @model_validator(mode="after")
    def leakage_protocol_is_complete_and_self_attesting(self) -> Self:
        if self.nuisance_inventory_digest != NUISANCE_FEATURE_INVENTORY.inventory_digest:
            raise ValueError("leakage protocol does not bind the nuisance inventory")
        if tuple(
            (
                item.scope_id,
                item.splits,
                item.boundary,
                item.analysis_seed_source,
                item.row_count,
                item.family_count,
            )
            for item in self.audit_scopes
        ) != (
            (
                "public-train-development",
                (BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT),
                ReviewBoundary.PUBLIC,
                SeedSourceBoundary.TRACKED_PUBLIC,
                900,
                6,
            ),
            (
                "custody-primary-pre-lock",
                (
                    BenchmarkSplit.TRAIN,
                    BenchmarkSplit.DEVELOPMENT,
                    BenchmarkSplit.LOCKED,
                ),
                ReviewBoundary.CUSTODY,
                SeedSourceBoundary.CUSTODY_ONLY,
                1_200,
                8,
            ),
            (
                "custody-diagnostic-pre-lock",
                (BenchmarkSplit.DIAGNOSTIC,),
                ReviewBoundary.CUSTODY,
                SeedSourceBoundary.CUSTODY_ONLY,
                1_200,
                8,
            ),
        ):
            raise ValueError("leakage audit scopes are not canonical")
        if any(
            item.analysis_seed_consumers
            != ("label_permutation", "logistic_random_state", "tree_random_state")
            for item in self.audit_scopes
        ):
            raise ValueError("leakage audit seed consumers are not canonical")
        if self.baseline_order != tuple(ShortcutBaseline):
            raise ValueError("shortcut baseline order is not canonical")
        if tuple(item.baseline for item in self.feature_subsets) != (
            ShortcutBaseline.FIRST_ACTION,
            ShortcutBaseline.MEMORY_VALIDITY,
            ShortcutBaseline.LENGTH_ONLY,
            ShortcutBaseline.STRUCTURAL_ONLY,
            ShortcutBaseline.IDENTIFIER_ONLY,
        ):
            raise ValueError("baseline feature subsets are not canonical")
        expected_subsets = (
            (ShortcutBaseline.FIRST_ACTION, (NuisanceFeatureBlock.FIRST_ACTION_INDEX,)),
            (
                ShortcutBaseline.MEMORY_VALIDITY,
                (
                    NuisanceFeatureBlock.VALIDITY_STATE_COUNTS,
                    NuisanceFeatureBlock.OPTIONAL_FIELD_PRESENCE,
                ),
            ),
            (
                ShortcutBaseline.LENGTH_ONLY,
                (
                    NuisanceFeatureBlock.EVENT_TEXT_LENGTHS,
                    NuisanceFeatureBlock.MEMORY_TEXT_LENGTHS,
                    NuisanceFeatureBlock.EVIDENCE_TEXT_LENGTHS,
                    NuisanceFeatureBlock.ACTION_TEXT_LENGTHS,
                ),
            ),
            (
                ShortcutBaseline.STRUCTURAL_ONLY,
                (
                    NuisanceFeatureBlock.TRAJECTORY_EVENT_COUNT,
                    NuisanceFeatureBlock.CANDIDATE_MEMORY_COUNT,
                    NuisanceFeatureBlock.ALLOWED_ACTION_COUNT,
                    NuisanceFeatureBlock.EVIDENCE_REFERENCE_COUNT,
                    NuisanceFeatureBlock.PIVOT_SEQUENCE,
                    NuisanceFeatureBlock.PIVOT_ACTION_STEP,
                    NuisanceFeatureBlock.EVENT_SEQUENCE_SUMMARY,
                    NuisanceFeatureBlock.ACTION_STEP_SUMMARY,
                    NuisanceFeatureBlock.MEMORY_RECORDED_SEQUENCE_SUMMARY,
                    NuisanceFeatureBlock.MEMORY_RECORDED_ACTION_STEP_SUMMARY,
                    NuisanceFeatureBlock.VALIDITY_SEQUENCE_SUMMARY,
                    NuisanceFeatureBlock.VALIDITY_ACTION_STEP_SUMMARY,
                    NuisanceFeatureBlock.EVIDENCE_EVENT_SEQUENCE_SUMMARY,
                    NuisanceFeatureBlock.MEMORY_REVISION_SUMMARY,
                ),
            ),
            (
                ShortcutBaseline.IDENTIFIER_ONLY,
                (
                    NuisanceFeatureBlock.IDENTIFIER_BYTE_HISTOGRAM,
                    NuisanceFeatureBlock.IDENTIFIER_NIBBLE_HISTOGRAM,
                ),
            ),
        )
        if tuple((item.baseline, item.blocks) for item in self.feature_subsets) != expected_subsets:
            raise ValueError("baseline feature block membership is not canonical")
        if self.single_field_lookup_universe != tuple(NuisanceFeatureBlock):
            raise ValueError("single-field lookup universe is not canonical")
        if tuple(item.estimand for item in self.estimand_classes) != tuple(LeakageEstimand):
            raise ValueError("leakage estimand class contracts are not complete and ordered")
        if (
            tuple(
                (
                    item.estimand,
                    item.included_outcomes,
                    tuple(mapping.assigned_class for mapping in item.outcome_classes),
                    item.class_order,
                    item.positive_class,
                    item.row_inclusion_rule,
                )
                for item in self.estimand_classes
            )
            != _LEAKAGE_CLASS_CONTRACTS
        ):
            raise ValueError("leakage estimand class contracts are not canonical")
        four_outcome = self.estimand_classes[1]
        if tuple(
            (item.outcome.value, item.integer_label) for item in self.adapter.label_integer_encoding
        ) != tuple(
            (item.class_label.value, item.integer_label)
            for item in four_outcome.class_integer_encoding
        ):
            raise ValueError("adapter label encoding diverges from the four-outcome estimand")
        if tuple(item.estimand for item in self.ceilings) != tuple(LeakageEstimand):
            raise ValueError("leakage ceilings are not complete and ordered")
        if tuple(
            (
                item.estimand,
                item.balanced_accuracy_max_ppm,
                item.average_precision_max_ppm,
                item.governed,
            )
            for item in self.ceilings
        ) != (
            (LeakageEstimand.HELPFUL_VS_REST, 550_000, 550_000, True),
            (LeakageEstimand.FOUR_OUTCOME, 350_000, 350_000, True),
            (LeakageEstimand.RESOLVED_ONLY, None, None, False),
        ):
            raise ValueError("leakage ceiling values are not canonical")
        expected_nulls = _LEAKAGE_NULL_GOLDENS
        if (
            tuple(
                (
                    item.fixture,
                    item.estimand,
                    item.class_order,
                    item.probability_numerators,
                    item.probability_denominator,
                    item.hard_label,
                    item.balanced_accuracy_ppm,
                    item.average_precision_ppm,
                )
                for item in self.null_goldens
            )
            != expected_nulls
        ):
            raise ValueError("leakage null goldens are not canonical")
        for null in self.null_goldens:
            classes = next(item for item in self.estimand_classes if item.estimand is null.estimand)
            if null.class_order != tuple(item.value for item in classes.class_order):
                raise ValueError("leakage null golden class order diverges from its estimand")
        if not hmac.compare_digest(self.protocol_digest, leakage_protocol_digest(self)):
            raise ValueError("leakage protocol digest does not match")
        return self


def leakage_protocol_digest(value: BaseModel | Mapping[str, object]) -> str:
    payload = (
        value.model_dump(mode="json", exclude={"protocol_digest"}, warnings=False)
        if isinstance(value, BaseModel)
        else {key: item for key, item in value.items() if key != "protocol_digest"}
    )
    return length_prefixed_sha256(canonical_json(payload), domain=LEAKAGE_PROTOCOL_DIGEST_DOMAIN)


def build_leakage_protocol() -> LeakageProtocol:
    structural_blocks = (
        NuisanceFeatureBlock.TRAJECTORY_EVENT_COUNT,
        NuisanceFeatureBlock.CANDIDATE_MEMORY_COUNT,
        NuisanceFeatureBlock.ALLOWED_ACTION_COUNT,
        NuisanceFeatureBlock.EVIDENCE_REFERENCE_COUNT,
        NuisanceFeatureBlock.PIVOT_SEQUENCE,
        NuisanceFeatureBlock.PIVOT_ACTION_STEP,
        NuisanceFeatureBlock.EVENT_SEQUENCE_SUMMARY,
        NuisanceFeatureBlock.ACTION_STEP_SUMMARY,
        NuisanceFeatureBlock.MEMORY_RECORDED_SEQUENCE_SUMMARY,
        NuisanceFeatureBlock.MEMORY_RECORDED_ACTION_STEP_SUMMARY,
        NuisanceFeatureBlock.VALIDITY_SEQUENCE_SUMMARY,
        NuisanceFeatureBlock.VALIDITY_ACTION_STEP_SUMMARY,
        NuisanceFeatureBlock.EVIDENCE_EVENT_SEQUENCE_SUMMARY,
        NuisanceFeatureBlock.MEMORY_REVISION_SUMMARY,
    )
    values: dict[str, object] = {
        "schema_version": LEAKAGE_PROTOCOL_SCHEMA_VERSION,
        "nuisance_inventory_digest": NUISANCE_FEATURE_INVENTORY.inventory_digest,
        "audit_scopes": (
            LeakageAuditScope(
                scope_id="public-train-development",
                splits=(BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT),
                boundary=ReviewBoundary.PUBLIC,
                analysis_seed_source=SeedSourceBoundary.TRACKED_PUBLIC,
                analysis_seed_consumers=(
                    "label_permutation",
                    "logistic_random_state",
                    "tree_random_state",
                ),
                row_count=900,
                family_count=6,
            ),
            LeakageAuditScope(
                scope_id="custody-primary-pre-lock",
                splits=(
                    BenchmarkSplit.TRAIN,
                    BenchmarkSplit.DEVELOPMENT,
                    BenchmarkSplit.LOCKED,
                ),
                boundary=ReviewBoundary.CUSTODY,
                analysis_seed_source=SeedSourceBoundary.CUSTODY_ONLY,
                analysis_seed_consumers=(
                    "label_permutation",
                    "logistic_random_state",
                    "tree_random_state",
                ),
                row_count=1_200,
                family_count=8,
            ),
            LeakageAuditScope(
                scope_id="custody-diagnostic-pre-lock",
                splits=(BenchmarkSplit.DIAGNOSTIC,),
                boundary=ReviewBoundary.CUSTODY,
                analysis_seed_source=SeedSourceBoundary.CUSTODY_ONLY,
                analysis_seed_consumers=(
                    "label_permutation",
                    "logistic_random_state",
                    "tree_random_state",
                ),
                row_count=1_200,
                family_count=8,
            ),
        ),
        "row_scope": "all-pivots-in-audit-scope",
        "fold": FoldProtocol(
            coordinate_order=("family", "pre_id_lineage_registry_key"),
            coordinate_encoding=("utf8", "utf8"),
        ),
        "permutation": PermutationProtocol(
            public_seed_commitment_digest=seed_commitment(
                derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.PERMUTATION),
                SeedPurpose.PERMUTATION,
            ),
            label_multiset=(
                ScenarioOutcome.HELPFUL,
                ScenarioOutcome.HELPFUL,
                ScenarioOutcome.HARMFUL,
                ScenarioOutcome.REDUNDANT,
                ScenarioOutcome.UNRESOLVED,
            ),
            fisher_yates_steps=(4, 3, 2, 1),
            coordinate_order=(
                "replicate_u64be",
                "family",
                "lineage_registry_key",
                "step_u64be",
                "attempt_u64be",
            ),
            interval_quantiles=(_ratio(1, 40), _ratio(39, 40)),
            golden_fixtures=(
                PermutationGoldenFixture(
                    replicate=0,
                    family=ScenarioFamily.FORGOTTEN_REQUIREMENT,
                    lineage_registry_key="golden-lineage-00",
                    draws=(
                        PermutationGoldenDraw(
                            step=4,
                            attempt=0,
                            draw_digest=(
                                "1a0b36ec8763ea9bc1d2ecdc397b93f16e3f8711658302845d6578bd228f1d6d"
                            ),
                            selected_index=1,
                        ),
                        PermutationGoldenDraw(
                            step=3,
                            attempt=0,
                            draw_digest=(
                                "e61877f0667544121fb3e8f85dcd7c5092b054c0771099cdee8445b5fdf0de3d"
                            ),
                            selected_index=1,
                        ),
                        PermutationGoldenDraw(
                            step=2,
                            attempt=0,
                            draw_digest=(
                                "38e7b7e5eadc9989822430d37bf249c08c4e41a0a76108866e876a666f6615fe"
                            ),
                            selected_index=1,
                        ),
                        PermutationGoldenDraw(
                            step=1,
                            attempt=0,
                            draw_digest=(
                                "ea135267d27b880cebd1c6111cce135de79b887610a5d8b4901d537efeaf198a"
                            ),
                            selected_index=0,
                        ),
                    ),
                    final_outcomes=(
                        ScenarioOutcome.HARMFUL,
                        ScenarioOutcome.HELPFUL,
                        ScenarioOutcome.REDUNDANT,
                        ScenarioOutcome.UNRESOLVED,
                        ScenarioOutcome.HELPFUL,
                    ),
                ),
                PermutationGoldenFixture(
                    replicate=9_999,
                    family=ScenarioFamily.IRREVERSIBLE_ACTION,
                    lineage_registry_key="golden-lineage-29",
                    draws=(
                        PermutationGoldenDraw(
                            step=4,
                            attempt=0,
                            draw_digest=(
                                "f911d23bfe96e3b1cca21319436d80a015530af614d24a7596a2850d554b9c41"
                            ),
                            selected_index=2,
                        ),
                        PermutationGoldenDraw(
                            step=3,
                            attempt=0,
                            draw_digest=(
                                "8cc52ad7008e0bd56c72fab42b624353b8fd019588007ef45d6f34e32d178614"
                            ),
                            selected_index=0,
                        ),
                        PermutationGoldenDraw(
                            step=2,
                            attempt=0,
                            draw_digest=(
                                "b2bc5a56b631d57ee1526d95eee33e93ff5066469fedbc21d2d0688a7985403f"
                            ),
                            selected_index=1,
                        ),
                        PermutationGoldenDraw(
                            step=1,
                            attempt=0,
                            draw_digest=(
                                "12b6c4d6e473081a8d3662978bc44705e02c97f50b3f81ce7b26e9bb76fdd179"
                            ),
                            selected_index=1,
                        ),
                    ),
                    final_outcomes=(
                        ScenarioOutcome.REDUNDANT,
                        ScenarioOutcome.UNRESOLVED,
                        ScenarioOutcome.HELPFUL,
                        ScenarioOutcome.HELPFUL,
                        ScenarioOutcome.HARMFUL,
                    ),
                ),
            ),
        ),
        "baseline_order": tuple(ShortcutBaseline),
        "feature_subsets": (
            BaselineFeatureSubset(
                baseline=ShortcutBaseline.FIRST_ACTION,
                blocks=(NuisanceFeatureBlock.FIRST_ACTION_INDEX,),
            ),
            BaselineFeatureSubset(
                baseline=ShortcutBaseline.MEMORY_VALIDITY,
                blocks=(
                    NuisanceFeatureBlock.VALIDITY_STATE_COUNTS,
                    NuisanceFeatureBlock.OPTIONAL_FIELD_PRESENCE,
                ),
            ),
            BaselineFeatureSubset(
                baseline=ShortcutBaseline.LENGTH_ONLY,
                blocks=(
                    NuisanceFeatureBlock.EVENT_TEXT_LENGTHS,
                    NuisanceFeatureBlock.MEMORY_TEXT_LENGTHS,
                    NuisanceFeatureBlock.EVIDENCE_TEXT_LENGTHS,
                    NuisanceFeatureBlock.ACTION_TEXT_LENGTHS,
                ),
            ),
            BaselineFeatureSubset(
                baseline=ShortcutBaseline.STRUCTURAL_ONLY,
                blocks=structural_blocks,
            ),
            BaselineFeatureSubset(
                baseline=ShortcutBaseline.IDENTIFIER_ONLY,
                blocks=(
                    NuisanceFeatureBlock.IDENTIFIER_BYTE_HISTOGRAM,
                    NuisanceFeatureBlock.IDENTIFIER_NIBBLE_HISTOGRAM,
                ),
            ),
        ),
        "single_field_lookup_universe": tuple(NuisanceFeatureBlock),
        "single_field_lookup_key_unit": "one-flat-signed-int64-coordinate/v1",
        "single_field_lookup_candidate_order": (
            "nuisance-block-order-then-zero-based-coordinate/v1"
        ),
        "single_field_lookup_candidate_count": NUISANCE_VECTOR_WIDTH,
        "unseen_lookup_rule": "training-prior",
        "single_field_selection": (
            "maximum-preregistered-held-out-statistic-across-candidate-coordinates/v1"
        ),
        "estimand_classes": tuple(
            LeakageEstimandClasses(
                estimand=estimand,
                included_outcomes=included_outcomes,
                outcome_classes=tuple(
                    LeakageOutcomeClass(outcome=outcome, assigned_class=assigned_class)
                    for outcome, assigned_class in zip(
                        ScenarioOutcome,
                        outcome_classes,
                        strict=True,
                    )
                ),
                class_order=class_order,
                class_integer_encoding=tuple(
                    LeakageClassIntegerEncoding(class_label=label, integer_label=index)
                    for index, label in enumerate(class_order)
                ),
                positive_class=positive_class,
                majority_tie_order=class_order,
                lookup_tie_order=class_order,
                row_inclusion_rule=row_inclusion_rule,
            )
            for (
                estimand,
                included_outcomes,
                outcome_classes,
                class_order,
                positive_class,
                row_inclusion_rule,
            ) in _LEAKAGE_CLASS_CONTRACTS
        ),
        "heldout_aggregation": (
            "concatenate-oof-predictions-in-canonical-scenario-order-then-one-global-metric"
        ),
        "fold_metric_average_allowed": False,
        "ceiling_scope": "every-baseline-every-governed-estimand",
        "permutation_reuse_scope": "same-permutations-all-baselines-and-statistics",
        "balanced_accuracy_rule": "unadjusted-arithmetic-mean-class-recall/v1",
        "average_precision_rule": "full-tie-group-average-precision/v1",
        "macro_average_precision_rule": "mean-four-full-tie-one-vs-rest/v1",
        "adapter": ResearchAdapterProtocol(
            dependencies=("numpy==2.3.5", "scikit-learn==1.9.0"),
            scaler_arguments=_frozen_arguments(_SCALER_ARGUMENTS),
            logistic_arguments=_frozen_arguments(_LOGISTIC_ARGUMENTS),
            logistic_omitted_arguments=("penalty", "n_jobs"),
            tree_arguments=_frozen_arguments(_TREE_ARGUMENTS),
            random_state_coordinates=(
                "permutation_leaf",
                "estimator",
                "fold_u64be",
                "estimand",
            ),
            class_order=tuple(ScenarioOutcome),
            label_integer_encoding=tuple(
                OutcomeIntegerEncoding(outcome=outcome, integer_label=index)
                for index, outcome in enumerate(ScenarioOutcome)
            ),
            logistic_pipeline=("standard_scaler", "logistic_regression"),
            tree_pipeline=("decision_tree_unscaled",),
            logistic_feature_blocks=tuple(NuisanceFeatureBlock),
            tree_feature_blocks=tuple(NuisanceFeatureBlock),
            probability_conversion=ProbabilityConversionProtocol(
                input_requirements=("finite", "closed-interval-zero-one"),
                conversion_order=(
                    "numpy-float64-item",
                    "python-float-repr",
                    "decimal-from-repr",
                    "ppm-round-half-up",
                ),
            ),
            thread_environment_variables=(
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ),
        ),
        "ceilings": (
            LeakageCeiling(
                estimand=LeakageEstimand.HELPFUL_VS_REST,
                balanced_accuracy_max_ppm=550_000,
                average_precision_max_ppm=550_000,
                governed=True,
            ),
            LeakageCeiling(
                estimand=LeakageEstimand.FOUR_OUTCOME,
                balanced_accuracy_max_ppm=350_000,
                average_precision_max_ppm=350_000,
                governed=True,
            ),
            LeakageCeiling(
                estimand=LeakageEstimand.RESOLVED_ONLY,
                balanced_accuracy_max_ppm=None,
                average_precision_max_ppm=None,
                governed=False,
            ),
        ),
        "null_goldens": tuple(
            LeakageNullGolden(
                fixture=fixture,
                estimand=estimand,
                class_order=class_order,
                probability_numerators=probability_numerators,
                probability_denominator=probability_denominator,
                hard_label=hard_label,
                balanced_accuracy_ppm=balanced_accuracy,
                average_precision_ppm=average_precision,
            )
            for (
                fixture,
                estimand,
                class_order,
                probability_numerators,
                probability_denominator,
                hard_label,
                balanced_accuracy,
                average_precision,
            ) in _LEAKAGE_NULL_GOLDENS
        ),
    }
    values["protocol_digest"] = leakage_protocol_digest(values)
    return LeakageProtocol.model_validate(values)


LEAKAGE_PROTOCOL = build_leakage_protocol()


class LineageReviewRecord(_StrictModel):
    schema_version: Literal["state-decay-v2-lineage-review-record/v1"] = (
        LINEAGE_REVIEW_RECORD_SCHEMA_VERSION
    )
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    split: BenchmarkSplit
    family: ScenarioFamily
    boundary: ReviewBoundary
    lineage_registry_key: ComponentIdentifier
    candidate_packet_digest: Sha256Digest
    independent_seed_commitment_digest: Sha256Digest
    transition_graph_digest: Sha256Digest
    evidence_topology_digest: Sha256Digest
    failure_mechanism_id: ComponentIdentifier
    semantic_signature_digest: Sha256Digest
    derivation_parent_keys: Annotated[
        tuple[ComponentIdentifier, ...],
        Field(max_length=16),
    ]
    semantic_rationale: ProtocolText
    reviewer_id: ComponentIdentifier
    review_rationale: ProtocolText
    decision: ReviewDecision
    review_digest: Sha256Digest

    @model_validator(mode="after")
    def review_is_role_bound_and_self_attesting(self) -> Self:
        expected_boundary = (
            ReviewBoundary.PUBLIC
            if self.split in (BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT)
            else ReviewBoundary.CUSTODY
        )
        if self.boundary is not expected_boundary:
            raise ValueError("lineage review boundary does not match its split")
        split_families = next(
            geometry.families
            for geometry in GENERATION_CONTRACT.splits
            if geometry.split is self.split
        )
        if self.family not in split_families:
            raise ValueError("lineage review family does not belong to its split")
        if self.boundary is ReviewBoundary.PUBLIC:
            public_leaf = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.PUBLIC)
            expected_commitment = independent_lineage_seed_commitment(
                derive_independent_lineage_seed(
                    public_leaf,
                    split=self.split,
                    family=self.family,
                    lineage_registry_key=self.lineage_registry_key,
                )
            )
            if not hmac.compare_digest(
                self.independent_seed_commitment_digest,
                expected_commitment,
            ):
                raise ValueError("public lineage seed commitment does not match")
        if len(set(self.derivation_parent_keys)) != len(self.derivation_parent_keys):
            raise ValueError("lineage review derivation parents must be unique")
        if not hmac.compare_digest(self.review_digest, lineage_review_record_digest(self)):
            raise ValueError("lineage review record digest does not match")
        return self


def derive_independent_lineage_seed(
    source_leaf: bytes,
    *,
    split: BenchmarkSplit,
    family: ScenarioFamily,
    lineage_registry_key: str,
) -> bytes:
    if type(source_leaf) is not bytes or len(source_leaf) != 32:
        raise ValueError("lineage source leaf is invalid")
    if type(split) is not BenchmarkSplit or type(family) is not ScenarioFamily:
        raise ValueError("lineage seed coordinates are invalid")
    if type(lineage_registry_key) is not str or not lineage_registry_key:
        raise ValueError("lineage registry key is invalid")
    return bytes.fromhex(
        length_prefixed_sha256(
            source_leaf,
            split.value,
            family.value,
            lineage_registry_key,
            domain=INDEPENDENT_LINEAGE_SEED_DOMAIN,
        )
    )


def independent_lineage_seed_commitment(seed: bytes) -> str:
    if type(seed) is not bytes or len(seed) != 32:
        raise ValueError("independent lineage seed is invalid")
    return length_prefixed_sha256(
        seed,
        domain=INDEPENDENT_LINEAGE_SEED_COMMITMENT_DOMAIN,
    )


def lineage_review_record_digest(value: BaseModel | Mapping[str, object]) -> str:
    payload = (
        value.model_dump(mode="json", exclude={"review_digest"}, warnings=False)
        if isinstance(value, BaseModel)
        else {key: item for key, item in value.items() if key != "review_digest"}
    )
    return length_prefixed_sha256(
        canonical_json(payload),
        domain=LINEAGE_REVIEW_RECORD_DIGEST_DOMAIN,
    )


def validate_lineage_review_registry(
    records: Sequence[LineageReviewRecord],
    *,
    expected_splits: Sequence[BenchmarkSplit],
) -> tuple[LineageReviewRecord, ...]:
    if not isinstance(expected_splits, Sequence) or isinstance(
        expected_splits, (str, bytes, bytearray)
    ):
        raise ValueError("expected review splits are invalid")
    try:
        splits = tuple(expected_splits)
    except Exception:
        raise ValueError("expected review splits are invalid") from None
    accepted_scopes = (
        (BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT),
        (BenchmarkSplit.LOCKED, BenchmarkSplit.DIAGNOSTIC),
        tuple(BenchmarkSplit),
    )
    if any(type(split) is not BenchmarkSplit for split in splits) or splits not in accepted_scopes:
        raise ValueError("expected review splits are invalid")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        raise ValueError("lineage review registry is invalid")
    try:
        snapshot = tuple(records)
    except Exception:
        raise ValueError("lineage review registry is invalid") from None
    if not snapshot or any(type(item) is not LineageReviewRecord for item in snapshot):
        raise ValueError("lineage review registry is invalid")
    try:
        reviewed = tuple(
            LineageReviewRecord.model_validate_json(canonical_json(item)) for item in snapshot
        )
    except Exception:
        raise ValueError("lineage review registry is invalid") from None
    if any(item.split not in splits for item in reviewed):
        raise ValueError("lineage review registry contains an unexpected split")
    if any(
        item.decision is not ReviewDecision.ACCEPTED or item.derivation_parent_keys
        for item in reviewed
    ):
        raise ValueError("lineage review registry contains an unaccepted record")
    if len({item.review_digest for item in reviewed}) != len(reviewed):
        raise ValueError("lineage review registry contains duplicate attestations")
    lineage_keys = tuple((item.split, item.family, item.lineage_registry_key) for item in reviewed)
    if len(set(lineage_keys)) != len(lineage_keys):
        raise ValueError("lineage review registry contains duplicate lineages")
    for family in ScenarioFamily:
        family_records = tuple(item for item in reviewed if item.family is family)
        for field in LINEAGE_REVIEW_PROTOCOL.family_local_unique_fields:
            values = tuple(getattr(item, field) for item in family_records)
            if len(set(values)) != len(values):
                raise ValueError("lineage review registry violates family-local uniqueness")
    expected_groups = tuple(
        (split, family)
        for split in splits
        for family in next(
            geometry.families for geometry in GENERATION_CONTRACT.splits if geometry.split is split
        )
    )
    actual_groups = {(item.split, item.family) for item in reviewed}
    if actual_groups != set(expected_groups) or any(
        sum(item.split is split and item.family is family for item in reviewed)
        != LINEAGES_PER_FAMILY
        for split, family in expected_groups
    ):
        raise ValueError("lineage review registry geometry is incomplete")
    return reviewed


class LineageSeedSourceRule(_StrictModel):
    split: BenchmarkSplit
    source_purpose: SeedPurpose


class ReviewBoundaryRule(_StrictModel):
    split: BenchmarkSplit
    boundary: ReviewBoundary


class LineageReviewProtocol(_StrictModel):
    schema_version: Literal["state-decay-v2-lineage-review-protocol/v1"] = (
        LINEAGE_REVIEW_PROTOCOL_SCHEMA_VERSION
    )
    record_schema_version: Literal["state-decay-v2-lineage-review-record/v1"] = (
        LINEAGE_REVIEW_RECORD_SCHEMA_VERSION
    )
    record_digest_domain: Literal["saliencegate:state-decay-v2:lineage-review-record:v1"] = (
        LINEAGE_REVIEW_RECORD_DIGEST_DOMAIN
    )
    lineage_seed_domain: Literal["saliencegate:state-decay-v2:lineage-seed:v1"] = (
        INDEPENDENT_LINEAGE_SEED_DOMAIN
    )
    lineage_seed_commitment_domain: Literal[
        "saliencegate:state-decay-v2:lineage-seed-commitment:v1"
    ] = INDEPENDENT_LINEAGE_SEED_COMMITMENT_DOMAIN
    lineage_seed_hash_primitive: Literal["length-prefixed-sha256/v1"] = "length-prefixed-sha256/v1"
    lineage_seed_source_rule: Literal["purpose-leaf-of-split-generation-source/v1"] = (
        "purpose-leaf-of-split-generation-source/v1"
    )
    lineage_seed_coordinates: Annotated[tuple[str, ...], Field(min_length=3, max_length=3)]
    lineage_seed_sources: Annotated[
        tuple[LineageSeedSourceRule, ...],
        Field(min_length=4, max_length=4),
    ]
    boundary_rules: Annotated[tuple[ReviewBoundaryRule, ...], Field(min_length=4, max_length=4)]
    public_seed_golden_split: Literal[BenchmarkSplit.TRAIN] = BenchmarkSplit.TRAIN
    public_seed_golden_family: Literal[ScenarioFamily.FORGOTTEN_REQUIREMENT] = (
        ScenarioFamily.FORGOTTEN_REQUIREMENT
    )
    public_seed_golden_lineage_key: Literal["golden-lineage-00"] = "golden-lineage-00"
    public_seed_golden_commitment: Literal[
        "bd876f155dd3293526a972bfef073f661022876698199c1babdf86806993615c"
    ] = "bd876f155dd3293526a972bfef073f661022876698199c1babdf86806993615c"
    required_record_fields: Annotated[tuple[str, ...], Field(min_length=19, max_length=19)]
    forbidden_semantic_fields: Annotated[tuple[str, ...], Field(min_length=6, max_length=6)]
    accepted_decision: Literal[ReviewDecision.ACCEPTED] = ReviewDecision.ACCEPTED
    accepted_parent_count: Literal[0] = 0
    family_local_unique_fields: Annotated[tuple[str, ...], Field(min_length=7, max_length=7)]
    reviewer_reuse_allowed: Literal[True] = True
    duplicate_attestations_allowed: Literal[False] = False
    human_attestation_required: Literal[True] = True
    generated_attestations_forbidden: Literal[True] = True
    acceptance_scope: Literal["family-global-across-splits/v1"] = "family-global-across-splits/v1"
    accepted_registry_split_scopes: Annotated[
        tuple[str, ...],
        Field(min_length=3, max_length=3),
    ]
    protocol_digest: Sha256Digest

    @model_validator(mode="after")
    def review_protocol_is_complete_and_self_attesting(self) -> Self:
        if self.lineage_seed_coordinates != ("split", "family", "lineage_registry_key"):
            raise ValueError("lineage seed coordinates are not canonical")
        if tuple(item.split for item in self.lineage_seed_sources) != tuple(BenchmarkSplit):
            raise ValueError("lineage seed sources are not complete and ordered")
        if tuple(item.source_purpose for item in self.lineage_seed_sources) != (
            SeedPurpose.PUBLIC,
            SeedPurpose.PUBLIC,
            SeedPurpose.LOCKED,
            SeedPurpose.DIAGNOSTIC,
        ):
            raise ValueError("lineage seed source purposes are not canonical")
        if tuple(item.split for item in self.boundary_rules) != tuple(BenchmarkSplit):
            raise ValueError("review boundary rules are not complete and ordered")
        if tuple(item.boundary for item in self.boundary_rules) != (
            ReviewBoundary.PUBLIC,
            ReviewBoundary.PUBLIC,
            ReviewBoundary.CUSTODY,
            ReviewBoundary.CUSTODY,
        ):
            raise ValueError("review boundaries are not canonical")
        public_golden = independent_lineage_seed_commitment(
            derive_independent_lineage_seed(
                derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.PUBLIC),
                split=self.public_seed_golden_split,
                family=self.public_seed_golden_family,
                lineage_registry_key=self.public_seed_golden_lineage_key,
            )
        )
        if public_golden != self.public_seed_golden_commitment:
            raise ValueError("public lineage seed golden does not match")
        if self.family_local_unique_fields != (
            "candidate_packet_digest",
            "lineage_registry_key",
            "independent_seed_commitment_digest",
            "transition_graph_digest",
            "evidence_topology_digest",
            "failure_mechanism_id",
            "semantic_signature_digest",
        ):
            raise ValueError("lineage review uniqueness fields are not canonical")
        if self.family_local_unique_fields != TEMPLATE_REGISTRY_UNIQUE_FIELDS:
            raise ValueError("lineage review uniqueness diverges from the template registry")
        if self.required_record_fields != tuple(LineageReviewRecord.model_fields):
            raise ValueError("lineage review required fields are not canonical")
        if self.forbidden_semantic_fields != (
            "outcome",
            "allocation_rank",
            "generator_slot",
            "scenario_id",
            "oracle_branch",
            "treatment_outcome",
        ):
            raise ValueError("lineage review forbidden fields are not canonical")
        if self.accepted_registry_split_scopes != (
            "public-train-development",
            "custody-locked-diagnostic",
            "full-suite",
        ):
            raise ValueError("lineage review registry scopes are not canonical")
        if not hmac.compare_digest(self.protocol_digest, lineage_review_protocol_digest(self)):
            raise ValueError("lineage review protocol digest does not match")
        return self


def lineage_review_protocol_digest(value: BaseModel | Mapping[str, object]) -> str:
    payload = (
        value.model_dump(mode="json", exclude={"protocol_digest"}, warnings=False)
        if isinstance(value, BaseModel)
        else {key: item for key, item in value.items() if key != "protocol_digest"}
    )
    return length_prefixed_sha256(
        canonical_json(payload),
        domain=LINEAGE_REVIEW_PROTOCOL_DIGEST_DOMAIN,
    )


def build_lineage_review_protocol() -> LineageReviewProtocol:
    values: dict[str, object] = {
        "schema_version": LINEAGE_REVIEW_PROTOCOL_SCHEMA_VERSION,
        "record_schema_version": LINEAGE_REVIEW_RECORD_SCHEMA_VERSION,
        "record_digest_domain": LINEAGE_REVIEW_RECORD_DIGEST_DOMAIN,
        "lineage_seed_domain": INDEPENDENT_LINEAGE_SEED_DOMAIN,
        "lineage_seed_commitment_domain": INDEPENDENT_LINEAGE_SEED_COMMITMENT_DOMAIN,
        "lineage_seed_hash_primitive": "length-prefixed-sha256/v1",
        "lineage_seed_source_rule": "purpose-leaf-of-split-generation-source/v1",
        "lineage_seed_coordinates": ("split", "family", "lineage_registry_key"),
        "lineage_seed_sources": tuple(
            LineageSeedSourceRule(split=split, source_purpose=purpose)
            for split, purpose in zip(
                BenchmarkSplit,
                (
                    SeedPurpose.PUBLIC,
                    SeedPurpose.PUBLIC,
                    SeedPurpose.LOCKED,
                    SeedPurpose.DIAGNOSTIC,
                ),
                strict=True,
            )
        ),
        "boundary_rules": tuple(
            ReviewBoundaryRule(split=split, boundary=boundary)
            for split, boundary in zip(
                BenchmarkSplit,
                (
                    ReviewBoundary.PUBLIC,
                    ReviewBoundary.PUBLIC,
                    ReviewBoundary.CUSTODY,
                    ReviewBoundary.CUSTODY,
                ),
                strict=True,
            )
        ),
        "public_seed_golden_split": BenchmarkSplit.TRAIN,
        "public_seed_golden_family": ScenarioFamily.FORGOTTEN_REQUIREMENT,
        "public_seed_golden_lineage_key": "golden-lineage-00",
        "public_seed_golden_commitment": (
            "bd876f155dd3293526a972bfef073f661022876698199c1babdf86806993615c"
        ),
        "required_record_fields": (
            "schema_version",
            "suite_id",
            "suite_version",
            "split",
            "family",
            "boundary",
            "lineage_registry_key",
            "candidate_packet_digest",
            "independent_seed_commitment_digest",
            "transition_graph_digest",
            "evidence_topology_digest",
            "failure_mechanism_id",
            "semantic_signature_digest",
            "derivation_parent_keys",
            "semantic_rationale",
            "reviewer_id",
            "review_rationale",
            "decision",
            "review_digest",
        ),
        "forbidden_semantic_fields": (
            "outcome",
            "allocation_rank",
            "generator_slot",
            "scenario_id",
            "oracle_branch",
            "treatment_outcome",
        ),
        "accepted_decision": ReviewDecision.ACCEPTED,
        "accepted_parent_count": 0,
        "family_local_unique_fields": (
            "candidate_packet_digest",
            "lineage_registry_key",
            "independent_seed_commitment_digest",
            "transition_graph_digest",
            "evidence_topology_digest",
            "failure_mechanism_id",
            "semantic_signature_digest",
        ),
        "reviewer_reuse_allowed": True,
        "duplicate_attestations_allowed": False,
        "human_attestation_required": True,
        "generated_attestations_forbidden": True,
        "acceptance_scope": "family-global-across-splits/v1",
        "accepted_registry_split_scopes": (
            "public-train-development",
            "custody-locked-diagnostic",
            "full-suite",
        ),
    }
    values["protocol_digest"] = lineage_review_protocol_digest(values)
    return LineageReviewProtocol.model_validate(values)


LINEAGE_REVIEW_PROTOCOL = build_lineage_review_protocol()


class TreatmentBindingField(StrEnum):
    FIXTURE_DIGEST = "fixture_digest"
    PROPOSAL_DIGEST = "proposal_digest"
    GROUNDING_RECEIPT_DIGEST = "grounding_receipt_digest"
    RENDERER_ID = "renderer_id"
    RENDERER_VERSION = "renderer_version"
    RENDERER_DIGEST = "renderer_digest"
    RENDERED_TEXT_DIGEST = "rendered_text_digest"
    EVIDENCE_ID_SET_DIGEST = "evidence_id_set_digest"
    EVIDENCE_REVISION_SET_DIGEST = "evidence_revision_set_digest"


class TreatmentCoverageFailure(StrEnum):
    MISSING_FIXTURE_KEY = "missing_fixture_key"
    DUPLICATE_FIXTURE_KEY = "duplicate_fixture_key"
    BINDING_MISMATCH = "binding_mismatch"
    INCOMPLETE_OUTCOME_COVERAGE = "incomplete_outcome_coverage"
    POLICY_VIEW_DIGEST_MISMATCH = "policy_view_digest_mismatch"
    NONEXACT_DELIVERY = "nonexact_delivery"
    ZERO_DENOMINATOR = "zero_denominator"


class ProposalSeedSourceRule(_StrictModel):
    split: BenchmarkSplit
    boundary: ReviewBoundary
    purpose: Literal[SeedPurpose.PROPOSAL] = SeedPurpose.PROPOSAL


class PublicProposalSeedCommitment(_StrictModel):
    split: BenchmarkSplit
    commitment_digest: Sha256Digest

    @model_validator(mode="after")
    def public_commitment_matches_split_derivation(self) -> Self:
        if self.split not in (BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT):
            raise ValueError("public proposal commitment split is invalid")
        proposal_leaf = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.PROPOSAL)
        expected = proposal_fixture_seed_commitment(
            derive_proposal_fixture_seed(proposal_leaf, self.split)
        )
        if not hmac.compare_digest(self.commitment_digest, expected):
            raise ValueError("public proposal fixture seed commitment does not match")
        return self


class TreatmentCoverageProtocol(_StrictModel):
    schema_version: Literal["state-decay-v2-treatment-coverage-protocol/v1"] = (
        TREATMENT_COVERAGE_PROTOCOL_SCHEMA_VERSION
    )
    binding_schema_version: Literal["admissible-treatment-binding/v1"] = (
        "admissible-treatment-binding/v1"
    )
    binding_fields: Annotated[tuple[TreatmentBindingField, ...], Field(min_length=9, max_length=9)]
    fixture_key: Literal["scenario_id"] = "scenario_id"
    fixture_cardinality: Literal["exactly-one-per-scenario"] = "exactly-one-per-scenario"
    comparison: Literal["canonical-full-typed-equality/v1"] = "canonical-full-typed-equality/v1"
    outcome_access_during_derivation: Literal[False] = False
    fixture_inputs: Annotated[tuple[str, ...], Field(min_length=3, max_length=3)]
    fixture_seed_purpose: Literal[SeedPurpose.PROPOSAL] = SeedPurpose.PROPOSAL
    fixture_seed_derivation_domain: Literal[
        "saliencegate:state-decay-v2:proposal-fixture-seed:v1"
    ] = PROPOSAL_FIXTURE_SEED_DOMAIN
    fixture_seed_commitment_domain: Literal[
        "saliencegate:state-decay-v2:proposal-fixture-seed-commitment:v1"
    ] = PROPOSAL_FIXTURE_SEED_COMMITMENT_DOMAIN
    fixture_seed_hash_primitive: Literal["length-prefixed-sha256/v1"] = "length-prefixed-sha256/v1"
    fixture_seed_coordinates: Annotated[tuple[str, ...], Field(min_length=2, max_length=2)]
    fixture_seed_cardinality: Literal["one-distinct-commitment-per-split"] = (
        "one-distinct-commitment-per-split"
    )
    fixture_seed_sources: Annotated[
        tuple[ProposalSeedSourceRule, ...],
        Field(min_length=4, max_length=4),
    ]
    public_fixture_seed_commitments: Annotated[
        tuple[PublicProposalSeedCommitment, ...],
        Field(min_length=2, max_length=2),
    ]
    custody_seed_binding: Literal["suite-generation-commitment-required/v1"] = (
        "suite-generation-commitment-required/v1"
    )
    coverage_partition: Annotated[tuple[ScenarioOutcome, ...], Field(min_length=4, max_length=4)]
    proposal_coverage_minimum: ExactRatio
    delivered_exact_coverage_minimum: ExactRatio
    zero_denominator_rule: Literal["not-evaluable-and-invalid"] = "not-evaluable-and-invalid"
    nonexact_delivery_rule: Literal["invalidate-report-and-exclude-from-D_j"] = (
        "invalidate-report-and-exclude-from-D_j"
    )
    failures: Annotated[
        tuple[TreatmentCoverageFailure, ...],
        Field(min_length=7, max_length=7),
    ]
    protocol_digest: Sha256Digest

    @model_validator(mode="after")
    def treatment_protocol_is_complete_and_self_attesting(self) -> Self:
        if self.binding_fields != tuple(TreatmentBindingField):
            raise ValueError("treatment binding fields are not complete and ordered")
        if self.fixture_inputs != (
            "policy_view_projection",
            "public_response_profile",
            "pre_allocation_fixture_seed",
        ):
            raise ValueError("treatment fixture inputs are not canonical")
        if self.fixture_seed_coordinates != ("proposal_leaf", "split"):
            raise ValueError("proposal fixture seed coordinates are not canonical")
        if tuple(item.split for item in self.fixture_seed_sources) != tuple(BenchmarkSplit):
            raise ValueError("proposal fixture seed sources are not complete and ordered")
        if tuple(item.boundary for item in self.fixture_seed_sources) != (
            ReviewBoundary.PUBLIC,
            ReviewBoundary.PUBLIC,
            ReviewBoundary.CUSTODY,
            ReviewBoundary.CUSTODY,
        ):
            raise ValueError("proposal fixture seed boundaries are not canonical")
        if tuple(item.split for item in self.public_fixture_seed_commitments) != (
            BenchmarkSplit.TRAIN,
            BenchmarkSplit.DEVELOPMENT,
        ):
            raise ValueError("public proposal fixture commitments are not canonical")
        if len({item.commitment_digest for item in self.public_fixture_seed_commitments}) != 2:
            raise ValueError("public proposal fixture commitments must be split-distinct")
        if self.coverage_partition != tuple(ScenarioOutcome):
            raise ValueError("treatment coverage outcomes are not canonical")
        if self.proposal_coverage_minimum != _ratio(99, 100):
            raise ValueError("proposal coverage threshold is not canonical")
        if self.delivered_exact_coverage_minimum != _ratio(1, 1):
            raise ValueError("delivery coverage threshold is not canonical")
        if self.failures != tuple(TreatmentCoverageFailure):
            raise ValueError("treatment failure inventory is not canonical")
        if not hmac.compare_digest(self.protocol_digest, treatment_protocol_digest(self)):
            raise ValueError("treatment coverage protocol digest does not match")
        return self


def treatment_protocol_digest(value: BaseModel | Mapping[str, object]) -> str:
    payload = (
        value.model_dump(mode="json", exclude={"protocol_digest"}, warnings=False)
        if isinstance(value, BaseModel)
        else {key: item for key, item in value.items() if key != "protocol_digest"}
    )
    return length_prefixed_sha256(canonical_json(payload), domain=TREATMENT_PROTOCOL_DIGEST_DOMAIN)


def build_treatment_coverage_protocol() -> TreatmentCoverageProtocol:
    values: dict[str, object] = {
        "schema_version": TREATMENT_COVERAGE_PROTOCOL_SCHEMA_VERSION,
        "binding_schema_version": "admissible-treatment-binding/v1",
        "binding_fields": tuple(TreatmentBindingField),
        "fixture_key": "scenario_id",
        "fixture_cardinality": "exactly-one-per-scenario",
        "comparison": "canonical-full-typed-equality/v1",
        "outcome_access_during_derivation": False,
        "fixture_inputs": (
            "policy_view_projection",
            "public_response_profile",
            "pre_allocation_fixture_seed",
        ),
        "fixture_seed_purpose": SeedPurpose.PROPOSAL,
        "fixture_seed_derivation_domain": PROPOSAL_FIXTURE_SEED_DOMAIN,
        "fixture_seed_commitment_domain": PROPOSAL_FIXTURE_SEED_COMMITMENT_DOMAIN,
        "fixture_seed_hash_primitive": "length-prefixed-sha256/v1",
        "fixture_seed_coordinates": ("proposal_leaf", "split"),
        "fixture_seed_cardinality": "one-distinct-commitment-per-split",
        "fixture_seed_sources": tuple(
            ProposalSeedSourceRule(split=split, boundary=boundary)
            for split, boundary in zip(
                BenchmarkSplit,
                (
                    ReviewBoundary.PUBLIC,
                    ReviewBoundary.PUBLIC,
                    ReviewBoundary.CUSTODY,
                    ReviewBoundary.CUSTODY,
                ),
                strict=True,
            )
        ),
        "public_fixture_seed_commitments": tuple(
            PublicProposalSeedCommitment(
                split=split,
                commitment_digest=proposal_fixture_seed_commitment(
                    derive_proposal_fixture_seed(
                        derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.PROPOSAL),
                        split,
                    )
                ),
            )
            for split in (BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT)
        ),
        "custody_seed_binding": "suite-generation-commitment-required/v1",
        "coverage_partition": tuple(ScenarioOutcome),
        "proposal_coverage_minimum": _ratio(99, 100),
        "delivered_exact_coverage_minimum": _ratio(1, 1),
        "zero_denominator_rule": "not-evaluable-and-invalid",
        "nonexact_delivery_rule": "invalidate-report-and-exclude-from-D_j",
        "failures": tuple(TreatmentCoverageFailure),
    }
    values["protocol_digest"] = treatment_protocol_digest(values)
    return TreatmentCoverageProtocol.model_validate(values)


TREATMENT_COVERAGE_PROTOCOL = build_treatment_coverage_protocol()


class MetricStatus(StrEnum):
    DEFINED = "defined"
    NOT_EVALUABLE = "not_evaluable"
    NOT_MEASURED = "not_measured"


class BootstrapMetricFamily(StrEnum):
    TRIGGER = "trigger"
    INTERVENTION = "intervention"
    SUCCESS = "success"
    CALL = "call"
    TOKEN = "token"
    FAILURE_LOOP = "failure_loop"


class FiniteSampleValidity(StrEnum):
    FEASIBLE = "feasible"
    NOT_FEASIBLE = "not_feasible"


class AssuranceAxis(StrEnum):
    TRIGGER = "trigger"
    HARMFUL_INCIDENCE = "harmful_incidence"
    SUCCESS_DELTA = "success_delta"
    CALL_REDUCTION = "call_reduction"
    TOKEN_REDUCTION = "token_reduction"


class AssuranceReason(StrEnum):
    PAIRED_BASELINE_DISCORDANCE_UNIDENTIFIED = "paired_baseline_discordance_unidentified"
    CALL_OPPORTUNITY_DISTRIBUTION_UNIDENTIFIED = "call_opportunity_distribution_unidentified"
    TOKENS_ARE_NOT_BERNOULLI_TRIALS = "tokens_are_not_bernoulli_trials"


class LockedMetricGeometry(_StrictModel):
    rows: Literal[300] = 300
    independent_lineages: Literal[60] = 60
    family_strata: Literal[2] = 2
    rows_per_lineage: Literal[5] = 5
    positives: Literal[120] = 120
    negatives: Literal[180] = 180
    resolved: Literal[240] = 240
    helpful: Literal[120] = 120
    harmful: Literal[60] = 60
    redundant: Literal[60] = 60
    unresolved: Literal[60] = 60


class ECEBin(_StrictModel):
    index: Annotated[int, Field(ge=0, le=9)]
    lower_ppm: Annotated[int, Field(ge=0, le=900_000)]
    upper_ppm: Annotated[int, Field(ge=100_000, le=1_000_000)]
    lower_inclusive: Literal[True] = True
    upper_inclusive: bool

    @model_validator(mode="after")
    def bin_is_the_canonical_equal_width_interval(self) -> Self:
        if self.lower_ppm != self.index * 100_000:
            raise ValueError("ECE bin lower bound is not canonical")
        if self.upper_ppm != (self.index + 1) * 100_000:
            raise ValueError("ECE bin upper bound is not canonical")
        if self.upper_inclusive is not (self.index == 9):
            raise ValueError("only the last ECE bin may include its upper bound")
        return self


class MetricGateRule(_StrictModel):
    gate_id: ComponentIdentifier
    canonical_integer_inequality: ProtocolText
    zero_denominator_fails: bool


class MetricDefinition(_StrictModel):
    metric_id: ComponentIdentifier
    numerator: ProtocolText
    denominator: ProtocolText
    governed: bool
    zero_denominator_status: MetricStatus


class MetricProtocol(_StrictModel):
    geometry: LockedMetricGeometry
    primary_score: Literal["opportunity_score_10"] = "opportunity_score_10"
    positive_label: Literal["exact-adjudicable-helpful-opportunity"] = (
        "exact-adjudicable-helpful-opportunity"
    )
    estimand: Literal["all-locked-pivots-helpful-vs-rest"] = "all-locked-pivots-helpful-vs-rest"
    score_order: Literal["descending"] = "descending"
    tie_rule: Literal["consume-complete-score-group"] = "consume-complete-score-group"
    area_method: Literal["average-precision-not-trapezoidal"] = "average-precision-not-trapezoidal"
    average_precision_gate: ExactRatio
    zero_positive_rule: Literal["not_evaluable-and-fail"] = "not_evaluable-and-fail"
    positive_importance_weight: ExactRatio
    negative_importance_weight: ExactRatio
    score_probability_rule: Literal["opportunity-score-10-over-10-exact-rational/v1"] = (
        "opportunity-score-10-over-10-exact-rational/v1"
    )
    brier_rule: Literal["sum-weight-times-squared-probability-error-over-sum-weights/v1"] = (
        "sum-weight-times-squared-probability-error-over-sum-weights/v1"
    )
    importance_weights_apply_to: Annotated[tuple[str, ...], Field(min_length=2, max_length=2)]
    weighted_loss_normalization: Literal["sum-weighted-loss-over-sum-weights/v1"] = (
        "sum-weighted-loss-over-sum-weights/v1"
    )
    ece_bins: Annotated[tuple[ECEBin, ...], Field(min_length=10, max_length=10)]
    ece_rule: Literal[
        "weighted-bin-means-then-sum-bin-weight-absolute-gap-over-total-weight/v1"
    ] = "weighted-bin-means-then-sum-bin-weight-absolute-gap-over-total-weight/v1"
    empty_ece_bin_rule: Literal["ignore"] = "ignore"
    operational_prevalences: Annotated[tuple[ExactRatio, ...], Field(min_length=4, max_length=4)]
    projection_rule: Literal["exact-tpr-fpr-alert-rate-and-precision/v1"] = (
        "exact-tpr-fpr-alert-rate-and-precision/v1"
    )
    projection_formulas: Annotated[tuple[str, ...], Field(min_length=4, max_length=4)]
    zero_alert_rate_rule: Literal["precision-not_evaluable"] = "precision-not_evaluable"
    rate_record_operands: Annotated[tuple[str, ...], Field(min_length=6, max_length=6)]
    statuses: Annotated[tuple[MetricStatus, ...], Field(min_length=3, max_length=3)]
    metric_definitions: Annotated[tuple[MetricDefinition, ...], Field(min_length=22, max_length=22)]
    comparison_conditions: Annotated[tuple[str, ...], Field(min_length=4, max_length=4)]
    primary_inclusion_rules: Annotated[tuple[str, ...], Field(min_length=6, max_length=6)]
    call_token_accounting_rules: Annotated[tuple[str, ...], Field(min_length=6, max_length=6)]
    governed_gate_rules: Annotated[tuple[MetricGateRule, ...], Field(min_length=17, max_length=17)]

    @model_validator(mode="after")
    def metric_contract_is_exact(self) -> Self:
        if self.average_precision_gate != _ratio(4, 5):
            raise ValueError("average precision gate is not canonical")
        if self.positive_importance_weight != _ratio(1, 4):
            raise ValueError("positive importance weight is not canonical")
        if self.negative_importance_weight != _ratio(3, 2):
            raise ValueError("negative importance weight is not canonical")
        if self.importance_weights_apply_to != ("brier", "ece"):
            raise ValueError("importance-weight metric scope is not canonical")
        if tuple(item.index for item in self.ece_bins) != tuple(range(10)):
            raise ValueError("ECE bins are not complete and ordered")
        if self.operational_prevalences != (
            _ratio(1, 100),
            _ratio(1, 20),
            _ratio(1, 10),
            _ratio(1, 4),
        ):
            raise ValueError("operational prevalence projections are not canonical")
        if self.projection_formulas != (
            "TPR=TP/120",
            "FPR=FP/180",
            "alert_rate_pi=pi*TPR+(1-pi)*FPR",
            "precision_pi=pi*TPR/alert_rate_pi",
        ):
            raise ValueError("prevalence projection formulas are not canonical")
        if self.rate_record_operands != (
            "numerator",
            "denominator",
            "comparison_left",
            "comparison_right",
            "value_ppm_or_null",
            "status_and_typed_reason",
        ):
            raise ValueError("rate record operands are not canonical")
        if self.statuses != tuple(MetricStatus):
            raise ValueError("metric statuses are not complete and ordered")
        if tuple(
            (
                item.metric_id,
                item.numerator,
                item.denominator,
                item.governed,
                item.zero_denominator_status,
            )
            for item in self.metric_definitions
        ) != tuple((*definition, MetricStatus.NOT_EVALUABLE) for definition in _METRIC_DEFINITIONS):
            raise ValueError("metric definitions and denominators are not canonical")
        if self.comparison_conditions != (
            "no_memory",
            "canonical_fixed_step_schedule",
            "event_risk_only",
            "saliencegate_event",
        ):
            raise ValueError("metric comparison conditions are not canonical")
        if self.primary_inclusion_rules != _PRIMARY_INCLUSION_RULES:
            raise ValueError("metric inclusion rules are not canonical")
        if self.call_token_accounting_rules != _CALL_TOKEN_ACCOUNTING_RULES:
            raise ValueError("call and token accounting rules are not canonical")
        expected_gate_ids = (
            "proposal_coverage",
            "delivery_treatment_coverage",
            "trigger_average_precision",
            "trigger_recall",
            "trigger_precision",
            "false_intervention",
            "nonbeneficial_delivery",
            "delivered_provenance",
            "proposed_provenance",
            "evidence_adjudication",
            "evidence_support",
            "harmful_point_incidence",
            "harmful_upper_bound",
            "failure_loop_reduction",
            "success_point_margin",
            "call_reduction",
            "token_reduction",
        )
        if tuple(item.gate_id for item in self.governed_gate_rules) != expected_gate_ids:
            raise ValueError("metric gate rules are not complete and ordered")
        if tuple(
            (
                item.gate_id,
                item.canonical_integer_inequality,
                item.zero_denominator_fails,
            )
            for item in self.governed_gate_rules
        ) != tuple((gate_id, inequality, True) for gate_id, inequality in _METRIC_GATE_RULES):
            raise ValueError("metric gate inequalities are not canonical")
        return self


class ClopperPearsonPairGolden(_StrictModel):
    left_harmful_count: Annotated[int, Field(ge=0, le=30)]
    right_harmful_count: Annotated[int, Field(ge=0, le=30)]
    combined_upper_scaled: Annotated[int, Field(ge=0, le=NANOSCALE)]
    passes_upper_gate: bool


class ClopperPearsonProtocol(_StrictModel):
    trials_per_family: Literal[30] = 30
    family_tail: ExactRatio
    simultaneous_coverage_minimum: ExactRatio
    decimal_precision: Literal[100] = 100
    internal_rounding: Literal["ROUND_HALF_EVEN"] = "ROUND_HALF_EVEN"
    bisection_count: Literal[256] = 256
    initial_bracket: Annotated[tuple[int, ...], Field(min_length=2, max_length=2)]
    cdf_order: Literal["ascending-count"] = "ascending-count"
    bracket_update: Literal["cdf-mid-greater-tail-moves-lower-else-upper"] = (
        "cdf-mid-greater-tail-moves-lower-else-upper"
    )
    all_events_rule: Literal["h-equals-30-is-exactly-one"] = "h-equals-30-is-exactly-one"
    pair_composition: Literal["unrounded-U-left-plus-U-right-over-eight"] = (
        "unrounded-U-left-plus-U-right-over-eight"
    )
    final_rounding: Literal["ROUND_CEILING"] = "ROUND_CEILING"
    final_scale: Literal[1_000_000_000_000] = NANOSCALE
    gate_upper_scaled: Literal[50_000_000_000] = 50_000_000_000
    family_display_goldens_scaled: Annotated[tuple[int, ...], Field(min_length=31, max_length=31)]
    family_display_values_feed_gate: Literal[False] = False
    zero_pair_checkpoint_scaled: Literal[28_925_827_056] = 28_925_827_056
    threshold_adjacent_pair_goldens: Annotated[
        tuple[ClopperPearsonPairGolden, ...],
        Field(min_length=10, max_length=10),
    ]
    pooled_sixty_role: Literal["diagnostic-only"] = "diagnostic-only"
    pooled_sixty_method: Literal["one-sided-clopper-pearson/v1"] = "one-sided-clopper-pearson/v1"
    pooled_sixty_trials: Literal[60] = 60
    pooled_sixty_tail: ExactRatio
    pooled_sixty_composition: Literal["unrounded-upper-divided-by-four"] = (
        "unrounded-upper-divided-by-four"
    )
    bootstrap_substitution_allowed: Literal[False] = False

    @model_validator(mode="after")
    def clopper_pearson_contract_is_exact(self) -> Self:
        if self.family_tail != _ratio(1, 40):
            raise ValueError("Clopper-Pearson family tail is not canonical")
        if self.simultaneous_coverage_minimum != _ratio(19, 20):
            raise ValueError("simultaneous coverage is not canonical")
        if self.pooled_sixty_tail != _ratio(1, 20):
            raise ValueError("pooled Clopper-Pearson tail is not canonical")
        if self.initial_bracket != (0, 1):
            raise ValueError("Clopper-Pearson bracket is not canonical")
        if self.family_display_goldens_scaled != _CP_FAMILY_GOLDENS:
            raise ValueError("Clopper-Pearson family goldens are not canonical")
        if (
            tuple(
                (item.left_harmful_count, item.right_harmful_count)
                for item in self.threshold_adjacent_pair_goldens
            )
            != _CP_ADJACENT_PAIRS
        ):
            raise ValueError("Clopper-Pearson adjacent pairs are not canonical")
        expected_passes = (True, True, True, True, False, False, False, False, False, True)
        if tuple(
            (item.combined_upper_scaled, item.passes_upper_gate)
            for item in self.threshold_adjacent_pair_goldens
        ) != tuple(zip(_CP_ADJACENT_VALUES, expected_passes, strict=True)):
            raise ValueError("Clopper-Pearson adjacent pair values are not canonical")
        return self


_CP_FAMILY_GOLDENS = (
    115_703_308_223,
    172_169_455_634,
    220_735_401_523,
    265_288_450_475,
    307_218_350_277,
    347_211_698_835,
    385_666_510_997,
    422_836_522_979,
    458_893_651_395,
    493_959_041_463,
    528_120_044_790,
    561_440_150_989,
    593_965_069_949,
    625_726_549_391,
    656_744_761_840,
    687_029_714_132,
    716_581_920_817,
    745_392_450_097,
    773_442_351_172,
    800_701_374_988,
    827_125_778_474,
    852_654_815_246,
    877_205_190_128,
    900_662_135_042,
    922_864_487_999,
    943_578_303_532,
    962_446_503_662,
    978_882_862_971,
    991_821_865_540,
    999_156_429_074,
    1_000_000_000_000,
)
_CP_ADJACENT_PAIRS = (
    (0, 3),
    (1, 2),
    (2, 1),
    (3, 0),
    (0, 4),
    (1, 3),
    (2, 2),
    (3, 1),
    (4, 0),
    (0, 0),
)
_CP_ADJACENT_VALUES = (
    47_623_969_838,
    49_113_107_145,
    49_113_107_145,
    47_623_969_838,
    52_865_207_313,
    54_682_238_264,
    55_183_850_381,
    54_682_238_264,
    52_865_207_313,
    28_925_827_056,
)


class BootstrapVectorCheckpoint(_StrictModel):
    metric_family: BootstrapMetricFamily
    scenario_family: ScenarioFamily
    replicate: Annotated[int, Field(ge=0, le=2)]
    indexes: Annotated[tuple[int, ...], Field(min_length=30, max_length=30)]

    @model_validator(mode="after")
    def indexes_are_valid_lineage_coordinates(self) -> Self:
        if any(type(index) is not int or not 0 <= index < 30 for index in self.indexes):
            raise ValueError("bootstrap checkpoint contains an invalid lineage index")
        return self


class BootstrapProtocol(_StrictModel):
    method: Literal["lineage-cluster-bootstrap/v1"] = "lineage-cluster-bootstrap/v1"
    seed_purpose: Literal[SeedPurpose.BOOTSTRAP] = SeedPurpose.BOOTSTRAP
    public_seed_commitment_digest: Sha256Digest
    custody_seed_binding: Literal["suite-generation-commitment-required/v1"] = (
        "suite-generation-commitment-required/v1"
    )
    hash_domain: Literal["saliencegate:state-decay-v2:bootstrap-index:v1"] = BOOTSTRAP_INDEX_DOMAIN
    hash_primitive: Literal["length-prefixed-sha256/v1"] = "length-prefixed-sha256/v1"
    coordinate_order: Annotated[tuple[str, ...], Field(min_length=6, max_length=6)]
    metric_family_order: Annotated[
        tuple[BootstrapMetricFamily, ...],
        Field(min_length=6, max_length=6),
    ]
    stratum_order: Annotated[tuple[ScenarioFamily, ...], Field(min_length=8, max_length=8)]
    lineage_indexes_per_stratum: Literal[30] = 30
    sampling: Literal["within-family-with-replacement"] = "within-family-with-replacement"
    digest_integer: Literal["unsigned-big-endian-256"] = "unsigned-big-endian-256"
    rejection_rule: Literal["accept-below-2^256-minus-mod-30-then-mod-30"] = (
        "accept-below-2^256-minus-mod-30-then-mod-30"
    )
    rejection_attempt_rule: Literal["increment-attempt"] = "increment-attempt"
    development_replicates: Literal[10_000] = 10_000
    final_replicates: Literal[100_000] = 100_000
    percentiles: Annotated[tuple[ExactRatio, ...], Field(min_length=3, max_length=3)]
    percentile_rule: Literal["one-indexed-nearest-rank-ceil-q-times-B"] = (
        "one-indexed-nearest-rank-ceil-q-times-B"
    )
    sort_value_type: Literal["exact-rational"] = "exact-rational"
    duplicate_sort_rule: Literal["canonical-rational-order"] = "canonical-rational-order"
    paired_condition_rule: Literal["reuse-indexes-within-metric-family"] = (
        "reuse-indexes-within-metric-family"
    )
    golden_source_hex: Sha256Digest
    golden_source_domain: Literal["saliencegate:state-decay-v2:bootstrap-golden-source:v1"] = (
        BOOTSTRAP_GOLDEN_SOURCE_DOMAIN
    )
    golden_derived_seed_hex: Sha256Digest
    golden_replicates: Annotated[tuple[int, ...], Field(min_length=3, max_length=3)]
    golden_vector_count: Literal[144] = 144
    golden_vector_width: Literal[30] = 30
    golden_vector_representation: Literal[
        "metric-family-scenario-family-replicate-indexes-canonical-json/v1"
    ] = "metric-family-scenario-family-replicate-indexes-canonical-json/v1"
    golden_vectors_domain: Literal["saliencegate:state-decay-v2:bootstrap-golden-vectors:v1"] = (
        BOOTSTRAP_GOLDEN_VECTORS_DOMAIN
    )
    golden_vectors_digest: Literal[
        "f20b1efded7d96f0648add2cfa0cb52fa1e28dce019651e731d0876c375635fc"
    ] = "f20b1efded7d96f0648add2cfa0cb52fa1e28dce019651e731d0876c375635fc"
    vector_checkpoints: Annotated[
        tuple[BootstrapVectorCheckpoint, ...],
        Field(min_length=2, max_length=2),
    ]

    @model_validator(mode="after")
    def bootstrap_contract_is_exact(self) -> Self:
        expected_commitment = seed_commitment(
            derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.BOOTSTRAP),
            SeedPurpose.BOOTSTRAP,
        )
        if not hmac.compare_digest(self.public_seed_commitment_digest, expected_commitment):
            raise ValueError("public bootstrap seed commitment does not match")
        if self.coordinate_order != (
            "seed",
            "metric_family",
            "replicate_u64be",
            "family_stratum",
            "draw_ordinal_u64be",
            "attempt_u64be",
        ):
            raise ValueError("bootstrap coordinate order is not canonical")
        if self.metric_family_order != tuple(BootstrapMetricFamily):
            raise ValueError("bootstrap metric family order is not canonical")
        if self.stratum_order != tuple(ScenarioFamily):
            raise ValueError("bootstrap stratum order is not canonical")
        if self.percentiles != (_ratio(1, 40), _ratio(1, 2), _ratio(39, 40)):
            raise ValueError("bootstrap percentiles are not canonical")
        if self.golden_source_hex != bytes(range(32)).hex():
            raise ValueError("bootstrap golden source is not canonical")
        expected_derived = length_prefixed_sha256(
            bytes(range(32)),
            domain=BOOTSTRAP_GOLDEN_SOURCE_DOMAIN,
        )
        if self.golden_derived_seed_hex != expected_derived:
            raise ValueError("bootstrap golden derived seed is not canonical")
        if self.golden_replicates != (0, 1, 2):
            raise ValueError("bootstrap golden replicates are not canonical")
        if tuple(
            (item.metric_family, item.scenario_family, item.replicate)
            for item in self.vector_checkpoints
        ) != (
            (BootstrapMetricFamily.TRIGGER, ScenarioFamily.FORGOTTEN_REQUIREMENT, 0),
            (BootstrapMetricFamily.FAILURE_LOOP, ScenarioFamily.IRREVERSIBLE_ACTION, 2),
        ):
            raise ValueError("bootstrap vector checkpoints are not canonical")
        if tuple(item.indexes for item in self.vector_checkpoints) != (
            _BOOTSTRAP_FIRST_CHECKPOINT,
            _BOOTSTRAP_LAST_CHECKPOINT,
        ):
            raise ValueError("bootstrap vector checkpoint indexes are not canonical")
        return self


class NotEvaluableAssurance(_StrictModel):
    axis: AssuranceAxis
    reason: AssuranceReason


class AssuranceProtocol(_StrictModel):
    tpr_grid: Annotated[tuple[ExactRatio, ...], Field(min_length=4, max_length=4)]
    fpr_grid: Annotated[tuple[ExactRatio, ...], Field(min_length=3, max_length=3)]
    harmful_incidence_grid: Annotated[tuple[ExactRatio, ...], Field(min_length=3, max_length=3)]
    success_delta_grid: Annotated[tuple[ExactRatio, ...], Field(min_length=3, max_length=3)]
    call_reduction_grid: Annotated[tuple[ExactRatio, ...], Field(min_length=3, max_length=3)]
    token_reduction_grid: Annotated[tuple[ExactRatio, ...], Field(min_length=3, max_length=3)]
    rho_grid: Annotated[tuple[ExactRatio, ...], Field(min_length=3, max_length=3)]
    trigger_lineages: Literal[60] = 60
    positive_trials_per_lineage: Literal[2] = 2
    negative_trials_per_lineage: Literal[3] = 3
    trigger_kernel: Literal[
        "independent-positive-negative-lineage-beta-binomial-with-rho-zero-binomial/v1"
    ] = "independent-positive-negative-lineage-beta-binomial-with-rho-zero-binomial/v1"
    beta_parameters: Literal["alpha=q*(1/rho-1);beta=(1-q)*(1/rho-1)"] = (
        "alpha=q*(1/rho-1);beta=(1-q)*(1/rho-1)"
    )
    degenerate_probability_rule: Literal["q-zero-or-one-degenerate"] = "q-zero-or-one-degenerate"
    trigger_acceptance_event: Literal["TP>=90-and-TP>=9*FP"] = "TP>=90-and-TP>=9*FP"
    harmful_kernel: Literal["two-independent-binomial-30-with-q-harm-equals-4p"] = (
        "two-independent-binomial-30-with-q-harm-equals-4p"
    )
    harmful_correlation: Literal["not_applicable"] = "not_applicable"
    harmful_acceptance_event: Literal["combined-clopper-pearson-upper<=5/100"] = (
        "combined-clopper-pearson-upper<=5/100"
    )
    not_evaluable_axes: Annotated[
        tuple[NotEvaluableAssurance, ...], Field(min_length=3, max_length=3)
    ]
    observed_integer_regions: Annotated[tuple[str, ...], Field(min_length=4, max_length=4)]
    assurance_role: Literal["descriptive-only"] = "descriptive-only"
    assurance_changes_feasibility: Literal[False] = False
    assurance_changes_gate: Literal[False] = False
    assurance_changes_claims: Literal[False] = False

    @model_validator(mode="after")
    def assurance_contract_is_identifiable_only(self) -> Self:
        if self.tpr_grid != (_ratio(7, 10), _ratio(3, 4), _ratio(4, 5), _ratio(9, 10)):
            raise ValueError("TPR assurance grid is not canonical")
        if self.fpr_grid != (_ratio(1, 50), _ratio(1, 20), _ratio(1, 10)):
            raise ValueError("FPR assurance grid is not canonical")
        if self.harmful_incidence_grid != (_ratio(0, 1), _ratio(1, 100), _ratio(3, 100)):
            raise ValueError("harmful assurance grid is not canonical")
        if self.success_delta_grid != (_ratio(-1, 50), _ratio(0, 1), _ratio(1, 50)):
            raise ValueError("success assurance grid is not canonical")
        if self.call_reduction_grid != (_ratio(1, 2), _ratio(3, 5), _ratio(7, 10)):
            raise ValueError("call assurance grid is not canonical")
        if self.token_reduction_grid != (_ratio(2, 5), _ratio(1, 2), _ratio(3, 5)):
            raise ValueError("token assurance grid is not canonical")
        if self.rho_grid != (_ratio(0, 1), _ratio(1, 4), _ratio(1, 2)):
            raise ValueError("rho assurance grid is not canonical")
        if tuple(item.axis for item in self.not_evaluable_axes) != (
            AssuranceAxis.SUCCESS_DELTA,
            AssuranceAxis.CALL_REDUCTION,
            AssuranceAxis.TOKEN_REDUCTION,
        ):
            raise ValueError("not-evaluable assurance axes are not canonical")
        if tuple((item.axis, item.reason) for item in self.not_evaluable_axes) != (
            (
                AssuranceAxis.SUCCESS_DELTA,
                AssuranceReason.PAIRED_BASELINE_DISCORDANCE_UNIDENTIFIED,
            ),
            (
                AssuranceAxis.CALL_REDUCTION,
                AssuranceReason.CALL_OPPORTUNITY_DISTRIBUTION_UNIDENTIFIED,
            ),
            (
                AssuranceAxis.TOKEN_REDUCTION,
                AssuranceReason.TOKENS_ARE_NOT_BERNOULLI_TRIALS,
            ),
        ):
            raise ValueError("not-evaluable assurance reasons are not canonical")
        if self.observed_integer_regions != (
            "S_event-S_schedule>-6",
            "5*C_event<=2*C_schedule",
            "2*T_event<=T_schedule",
            "10*L_event<=7*L_no_memory-and-L_no_memory>0",
        ):
            raise ValueError("observed integer regions are not canonical")
        return self


class FiniteSampleProtocol(_StrictModel):
    schema_version: Literal["state-decay-v2-finite-sample-protocol/v1"] = (
        FINITE_SAMPLE_PROTOCOL_SCHEMA_VERSION
    )
    metric: MetricProtocol
    clopper_pearson: ClopperPearsonProtocol
    bootstrap: BootstrapProtocol
    assurance: AssuranceProtocol
    validity_outcomes: Annotated[
        tuple[FiniteSampleValidity, ...],
        Field(min_length=2, max_length=2),
    ]
    not_feasible_conditions: Annotated[tuple[str, ...], Field(min_length=3, max_length=3)]
    protocol_digest: Sha256Digest

    @model_validator(mode="after")
    def finite_sample_protocol_is_complete_and_self_attesting(self) -> Self:
        if self.validity_outcomes != tuple(FiniteSampleValidity):
            raise ValueError("finite-sample validity outcomes are not canonical")
        if self.not_feasible_conditions != (
            "planned-denominator-zero",
            "integer-acceptance-region-empty",
            "harmful-bound-cannot-attain-gate",
        ):
            raise ValueError("finite-sample failure conditions are not canonical")
        if not hmac.compare_digest(self.protocol_digest, finite_sample_protocol_digest(self)):
            raise ValueError("finite-sample protocol digest does not match")
        return self


def finite_sample_protocol_digest(value: BaseModel | Mapping[str, object]) -> str:
    payload = (
        value.model_dump(mode="json", exclude={"protocol_digest"}, warnings=False)
        if isinstance(value, BaseModel)
        else {key: item for key, item in value.items() if key != "protocol_digest"}
    )
    return length_prefixed_sha256(
        canonical_json(payload),
        domain=FINITE_SAMPLE_PROTOCOL_DIGEST_DOMAIN,
    )


_METRIC_DEFINITIONS: tuple[tuple[str, str, str, bool], ...] = (
    ("trigger_precision", "TP", "TP+FP", True),
    ("trigger_recall", "TP", "120", True),
    (
        "trigger_average_precision",
        "sum-score-groups(delta_TP*TP/(TP+FP))",
        "120",
        True,
    ),
    (
        "false_intervention_rate",
        "delivered_harmful_or_redundant",
        "120",
        True,
    ),
    (
        "nonbeneficial_delivery_rate",
        "delivered_harmful_redundant_or_unresolved",
        "180",
        True,
    ),
    ("helpful_intervention_rate", "delivered_helpful", "240", False),
    ("harmful_intervention_rate", "delivered_harmful", "240", True),
    ("redundant_intervention_rate", "delivered_redundant", "240", False),
    ("helpful_delivery_recall", "delivered_helpful", "120", False),
    ("harmful_selection_rate", "delivered_harmful", "60", False),
    ("unresolved_delivery_rate", "delivered_unresolved", "60", False),
    (
        "conditional_harm_rate",
        "delivered_harmful",
        "delivered_helpful+delivered_harmful+delivered_redundant",
        False,
    ),
    (
        "delivered_provenance_valid_rate",
        "delivered_provenance_valid",
        "delivered_reminders",
        True,
    ),
    (
        "proposed_provenance_valid_rate",
        "proposed_provenance_valid",
        "proposed_reminders",
        True,
    ),
    (
        "evidence_adjudication_coverage",
        "authorized_claims_adjudicated",
        "authorized_claims",
        True,
    ),
    (
        "evidence_supported_rate",
        "supported_authorized_claims",
        "authorized_claims",
        True,
    ),
    (
        "proposal_adjudicability",
        "exact_canonical_proposals",
        "canonical_fixture_invocations",
        True,
    ),
    ("success_rate", "successful_terminal_scenarios", "300", False),
    ("success_point_margin", "S_event-S_schedule", "300", True),
    (
        "failure_loop_reduction",
        "loops_no_memory-loops_event",
        "loops_no_memory",
        True,
    ),
    ("call_reduction", "C_schedule-C_event", "C_schedule", True),
    ("token_reduction", "T_schedule-T_event", "T_schedule", True),
)
_PRIMARY_INCLUSION_RULES = (
    "scored-pivot-bootstrap-hard-watchdog-ordinary-all-in-primary-trigger",
    "nonpivot-bootstrap-and-watchdog-excluded-from-pivot-classification",
    "nonpivot-bootstrap-and-watchdog-included-in-cost-failure-delivery-accounting",
    "only-exact-reminder-versus-silence-at-pivot-receives-causal-label",
    "diagnostic-macro-average-excludes-undefined-and-reports-count",
    "missing-support-label-counts-unsupported-never-excluded",
)
_CALL_TOKEN_ACCOUNTING_RULES = (
    "C-condition-counts-distinct-durable-STARTED-call-identities",
    "STARTED-counts-once-regardless-of-COMPLETED-FAILED-UNKNOWN",
    "T-condition-sums-authoritative-cycle-settlement-native-input-output-tokens",
    "known-terminal-settlement-reconciles-observed-native-tokens-shared-tokenizer",
    "any-UNKNOWN-means-FAILED_UNKNOWN_COST-full-reservation-no-double-count",
    "crash-cannot-improve-a-compute-gate",
)

_METRIC_GATE_RULES: tuple[tuple[str, str], ...] = (
    (
        "proposal_coverage",
        "100*exact_canonical_proposals>=99*canonical_fixture_invocations",
    ),
    (
        "delivery_treatment_coverage",
        "exact_deliveries==delivered_reminders",
    ),
    (
        "trigger_average_precision",
        "5*trigger_average_precision_numerator>=4*trigger_average_precision_denominator",
    ),
    ("trigger_recall", "TP>=90"),
    ("trigger_precision", "TP>=9*FP"),
    ("false_intervention", "10*delivered_harmful_or_redundant<=120"),
    (
        "nonbeneficial_delivery",
        "10*delivered_harmful_redundant_or_unresolved<=180",
    ),
    (
        "delivered_provenance",
        "delivered_provenance_valid==delivered_reminders",
    ),
    (
        "proposed_provenance",
        "100*proposed_provenance_valid>=99*proposed_reminders",
    ),
    (
        "evidence_adjudication",
        "authorized_claims_adjudicated==authorized_claims",
    ),
    (
        "evidence_support",
        "100*supported_authorized_claims>=98*authorized_claims",
    ),
    ("harmful_point_incidence", "100*delivered_harmful<=3*240"),
    ("harmful_upper_bound", "harmful_upper_scaled<=50000000000"),
    ("failure_loop_reduction", "10*loops_event<=7*loops_no_memory"),
    ("success_point_margin", "S_event-S_schedule>-6"),
    ("call_reduction", "5*C_event<=2*C_schedule"),
    ("token_reduction", "2*T_event<=T_schedule"),
)

_BOOTSTRAP_FIRST_CHECKPOINT = (
    0,
    9,
    6,
    11,
    15,
    28,
    19,
    4,
    5,
    23,
    29,
    10,
    9,
    3,
    28,
    5,
    10,
    4,
    27,
    11,
    12,
    9,
    5,
    2,
    2,
    24,
    16,
    11,
    28,
    16,
)
_BOOTSTRAP_LAST_CHECKPOINT = (
    18,
    21,
    27,
    12,
    15,
    25,
    20,
    23,
    23,
    18,
    21,
    12,
    15,
    4,
    11,
    4,
    28,
    19,
    19,
    15,
    5,
    9,
    5,
    14,
    21,
    28,
    23,
    29,
    23,
    17,
)


def build_metric_protocol() -> MetricProtocol:
    return MetricProtocol(
        geometry=LockedMetricGeometry(),
        average_precision_gate=_ratio(4, 5),
        positive_importance_weight=_ratio(1, 4),
        negative_importance_weight=_ratio(3, 2),
        importance_weights_apply_to=("brier", "ece"),
        ece_bins=tuple(
            ECEBin(
                index=index,
                lower_ppm=index * 100_000,
                upper_ppm=(index + 1) * 100_000,
                lower_inclusive=True,
                upper_inclusive=index == 9,
            )
            for index in range(10)
        ),
        operational_prevalences=(
            _ratio(1, 100),
            _ratio(1, 20),
            _ratio(1, 10),
            _ratio(1, 4),
        ),
        projection_formulas=(
            "TPR=TP/120",
            "FPR=FP/180",
            "alert_rate_pi=pi*TPR+(1-pi)*FPR",
            "precision_pi=pi*TPR/alert_rate_pi",
        ),
        rate_record_operands=(
            "numerator",
            "denominator",
            "comparison_left",
            "comparison_right",
            "value_ppm_or_null",
            "status_and_typed_reason",
        ),
        statuses=tuple(MetricStatus),
        metric_definitions=tuple(
            MetricDefinition(
                metric_id=metric_id,
                numerator=numerator,
                denominator=denominator,
                governed=governed,
                zero_denominator_status=MetricStatus.NOT_EVALUABLE,
            )
            for metric_id, numerator, denominator, governed in _METRIC_DEFINITIONS
        ),
        comparison_conditions=(
            "no_memory",
            "canonical_fixed_step_schedule",
            "event_risk_only",
            "saliencegate_event",
        ),
        primary_inclusion_rules=_PRIMARY_INCLUSION_RULES,
        call_token_accounting_rules=_CALL_TOKEN_ACCOUNTING_RULES,
        governed_gate_rules=tuple(
            MetricGateRule(
                gate_id=gate_id,
                canonical_integer_inequality=inequality,
                zero_denominator_fails=True,
            )
            for gate_id, inequality in _METRIC_GATE_RULES
        ),
    )


def build_clopper_pearson_protocol() -> ClopperPearsonProtocol:
    passes = (True, True, True, True, False, False, False, False, False, True)
    return ClopperPearsonProtocol(
        family_tail=_ratio(1, 40),
        simultaneous_coverage_minimum=_ratio(19, 20),
        pooled_sixty_tail=_ratio(1, 20),
        initial_bracket=(0, 1),
        family_display_goldens_scaled=_CP_FAMILY_GOLDENS,
        threshold_adjacent_pair_goldens=tuple(
            ClopperPearsonPairGolden(
                left_harmful_count=pair[0],
                right_harmful_count=pair[1],
                combined_upper_scaled=value,
                passes_upper_gate=passed,
            )
            for pair, value, passed in zip(
                _CP_ADJACENT_PAIRS,
                _CP_ADJACENT_VALUES,
                passes,
                strict=True,
            )
        ),
    )


def build_bootstrap_protocol() -> BootstrapProtocol:
    return BootstrapProtocol(
        public_seed_commitment_digest=seed_commitment(
            derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.BOOTSTRAP),
            SeedPurpose.BOOTSTRAP,
        ),
        coordinate_order=(
            "seed",
            "metric_family",
            "replicate_u64be",
            "family_stratum",
            "draw_ordinal_u64be",
            "attempt_u64be",
        ),
        metric_family_order=tuple(BootstrapMetricFamily),
        stratum_order=tuple(ScenarioFamily),
        percentiles=(_ratio(1, 40), _ratio(1, 2), _ratio(39, 40)),
        golden_source_hex=bytes(range(32)).hex(),
        golden_derived_seed_hex=length_prefixed_sha256(
            bytes(range(32)),
            domain=BOOTSTRAP_GOLDEN_SOURCE_DOMAIN,
        ),
        golden_replicates=(0, 1, 2),
        vector_checkpoints=(
            BootstrapVectorCheckpoint(
                metric_family=BootstrapMetricFamily.TRIGGER,
                scenario_family=ScenarioFamily.FORGOTTEN_REQUIREMENT,
                replicate=0,
                indexes=_BOOTSTRAP_FIRST_CHECKPOINT,
            ),
            BootstrapVectorCheckpoint(
                metric_family=BootstrapMetricFamily.FAILURE_LOOP,
                scenario_family=ScenarioFamily.IRREVERSIBLE_ACTION,
                replicate=2,
                indexes=_BOOTSTRAP_LAST_CHECKPOINT,
            ),
        ),
    )


def build_assurance_protocol() -> AssuranceProtocol:
    return AssuranceProtocol(
        tpr_grid=(_ratio(7, 10), _ratio(3, 4), _ratio(4, 5), _ratio(9, 10)),
        fpr_grid=(_ratio(1, 50), _ratio(1, 20), _ratio(1, 10)),
        harmful_incidence_grid=(_ratio(0, 1), _ratio(1, 100), _ratio(3, 100)),
        success_delta_grid=(_ratio(-1, 50), _ratio(0, 1), _ratio(1, 50)),
        call_reduction_grid=(_ratio(1, 2), _ratio(3, 5), _ratio(7, 10)),
        token_reduction_grid=(_ratio(2, 5), _ratio(1, 2), _ratio(3, 5)),
        rho_grid=(_ratio(0, 1), _ratio(1, 4), _ratio(1, 2)),
        not_evaluable_axes=(
            NotEvaluableAssurance(
                axis=AssuranceAxis.SUCCESS_DELTA,
                reason=AssuranceReason.PAIRED_BASELINE_DISCORDANCE_UNIDENTIFIED,
            ),
            NotEvaluableAssurance(
                axis=AssuranceAxis.CALL_REDUCTION,
                reason=AssuranceReason.CALL_OPPORTUNITY_DISTRIBUTION_UNIDENTIFIED,
            ),
            NotEvaluableAssurance(
                axis=AssuranceAxis.TOKEN_REDUCTION,
                reason=AssuranceReason.TOKENS_ARE_NOT_BERNOULLI_TRIALS,
            ),
        ),
        observed_integer_regions=(
            "S_event-S_schedule>-6",
            "5*C_event<=2*C_schedule",
            "2*T_event<=T_schedule",
            "10*L_event<=7*L_no_memory-and-L_no_memory>0",
        ),
    )


def build_finite_sample_protocol() -> FiniteSampleProtocol:
    values: dict[str, object] = {
        "schema_version": FINITE_SAMPLE_PROTOCOL_SCHEMA_VERSION,
        "metric": build_metric_protocol(),
        "clopper_pearson": build_clopper_pearson_protocol(),
        "bootstrap": build_bootstrap_protocol(),
        "assurance": build_assurance_protocol(),
        "validity_outcomes": tuple(FiniteSampleValidity),
        "not_feasible_conditions": (
            "planned-denominator-zero",
            "integer-acceptance-region-empty",
            "harmful-bound-cannot-attain-gate",
        ),
    }
    values["protocol_digest"] = finite_sample_protocol_digest(values)
    return FiniteSampleProtocol.model_validate(values)


FINITE_SAMPLE_PROTOCOL = build_finite_sample_protocol()


class ValidationProtocolBinding(_StrictModel):
    audit: ValidationAudit
    protocol_digest: Sha256Digest


def validation_protocol_bindings() -> tuple[ValidationProtocolBinding, ...]:
    bindings = (
        ValidationProtocolBinding(
            audit=ValidationAudit.GEOMETRY,
            protocol_digest=GENERATION_CONTRACT.contract_digest,
        ),
        ValidationProtocolBinding(
            audit=ValidationAudit.LINEAGE_REVIEW,
            protocol_digest=LINEAGE_REVIEW_PROTOCOL.protocol_digest,
        ),
        ValidationProtocolBinding(
            audit=ValidationAudit.TREATMENT_COVERAGE,
            protocol_digest=TREATMENT_COVERAGE_PROTOCOL.protocol_digest,
        ),
        ValidationProtocolBinding(
            audit=ValidationAudit.LEAKAGE,
            protocol_digest=LEAKAGE_PROTOCOL.protocol_digest,
        ),
        ValidationProtocolBinding(
            audit=ValidationAudit.FINITE_SAMPLE,
            protocol_digest=FINITE_SAMPLE_PROTOCOL.protocol_digest,
        ),
    )
    if tuple(item.audit for item in bindings) != tuple(ValidationAudit):
        raise RuntimeError("validation protocol bindings are not canonically ordered")
    if len({item.protocol_digest for item in bindings}) != len(bindings):
        raise RuntimeError("validation protocol digests must be role-distinct")
    return bindings


def validation_protocol_digests() -> dict[ValidationAudit, str]:
    return {item.audit: item.protocol_digest for item in validation_protocol_bindings()}


__all__ = [
    "BOOTSTRAP_GOLDEN_SOURCE_DOMAIN",
    "BOOTSTRAP_GOLDEN_VECTORS_DOMAIN",
    "ESTIMATOR_RANDOM_STATE_DOMAIN",
    "FINITE_SAMPLE_PROTOCOL",
    "FINITE_SAMPLE_PROTOCOL_SCHEMA_VERSION",
    "FOLD_GOLDEN_DOMAIN",
    "INDEPENDENT_LINEAGE_SEED_COMMITMENT_DOMAIN",
    "INDEPENDENT_LINEAGE_SEED_DOMAIN",
    "LEAKAGE_PROTOCOL",
    "LEAKAGE_PROTOCOL_SCHEMA_VERSION",
    "LINEAGE_REVIEW_PROTOCOL",
    "LINEAGE_REVIEW_PROTOCOL_SCHEMA_VERSION",
    "LINEAGE_REVIEW_RECORD_SCHEMA_VERSION",
    "NANOSCALE",
    "NUISANCE_FEATURE_INVENTORY",
    "NUISANCE_INVENTORY_SCHEMA_VERSION",
    "NUISANCE_VECTOR_WIDTH",
    "PPM_SCALE",
    "TREATMENT_COVERAGE_PROTOCOL",
    "TREATMENT_COVERAGE_PROTOCOL_SCHEMA_VERSION",
    "AssuranceAxis",
    "AssuranceProtocol",
    "AssuranceReason",
    "BaselineFeatureSubset",
    "BootstrapMetricFamily",
    "BootstrapProtocol",
    "BootstrapVectorCheckpoint",
    "ClopperPearsonPairGolden",
    "ClopperPearsonProtocol",
    "ECEBin",
    "ExactRatio",
    "FeatureBlockSpec",
    "FeaturePadding",
    "FiniteSampleProtocol",
    "FiniteSampleValidity",
    "FoldProtocol",
    "FrozenEstimatorArgument",
    "IdentifierOccurrenceSource",
    "LeakageAuditScope",
    "LeakageCeiling",
    "LeakageClassIntegerEncoding",
    "LeakageClassLabel",
    "LeakageEstimand",
    "LeakageEstimandClasses",
    "LeakageNullFixture",
    "LeakageNullGolden",
    "LeakageOutcomeClass",
    "LeakageProtocol",
    "LineageReviewProtocol",
    "LineageReviewRecord",
    "LineageSeedSourceRule",
    "LockedMetricGeometry",
    "MetricDefinition",
    "MetricGateRule",
    "MetricProtocol",
    "MetricStatus",
    "NotEvaluableAssurance",
    "NuisanceFeatureBlock",
    "NuisanceFeatureInventory",
    "OutcomeIntegerEncoding",
    "PermutationGoldenDraw",
    "PermutationGoldenFixture",
    "ProbabilityConversionProtocol",
    "ProposalSeedSourceRule",
    "PublicProposalSeedCommitment",
    "ResearchAdapterProtocol",
    "ReviewBoundary",
    "ReviewBoundaryRule",
    "ReviewDecision",
    "ShortcutBaseline",
    "TreatmentBindingField",
    "TreatmentCoverageFailure",
    "TreatmentCoverageProtocol",
    "ValidationProtocolBinding",
    "build_assurance_protocol",
    "build_bootstrap_protocol",
    "build_clopper_pearson_protocol",
    "build_finite_sample_protocol",
    "build_leakage_protocol",
    "build_lineage_review_protocol",
    "build_metric_protocol",
    "build_nuisance_feature_inventory",
    "build_treatment_coverage_protocol",
    "derive_independent_lineage_seed",
    "finite_sample_protocol_digest",
    "independent_lineage_seed_commitment",
    "leakage_protocol_digest",
    "lineage_review_protocol_digest",
    "lineage_review_record_digest",
    "nuisance_inventory_digest",
    "treatment_protocol_digest",
    "validate_lineage_review_registry",
    "validation_protocol_bindings",
    "validation_protocol_digests",
]
