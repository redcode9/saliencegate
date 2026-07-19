from __future__ import annotations

import hmac
from collections.abc import Mapping, Sequence
from enum import StrEnum
from hashlib import sha256
from types import MappingProxyType
from typing import Annotated, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.benchmarks.state_decay_v2.schema import (
    SUITE_ID,
    SUITE_VERSION,
    BenchmarkSplit,
    ScenarioFamily,
    ScenarioOutcome,
)
from saliencegate.domain import canonical_json, length_prefixed_sha256
from saliencegate.domain.records import ComponentIdentifier, Sha256Digest

GENERATION_CONTRACT_SCHEMA_VERSION: Literal["state-decay-v2-generation-contract/v1"] = (
    "state-decay-v2-generation-contract/v1"
)
SEED_COMMITMENT_SET_SCHEMA_VERSION: Literal["state-decay-v2-seed-commitments/v1"] = (
    "state-decay-v2-seed-commitments/v1"
)

PUBLIC_SEED_DOMAIN: Literal["saliencegate:state-decay-v2:public-seed:v1"] = (
    "saliencegate:state-decay-v2:public-seed:v1"
)
SCENARIO_ID_DOMAIN: Literal["saliencegate:state-decay-v2:scenario-id:v1"] = (
    "saliencegate:state-decay-v2:scenario-id:v1"
)
ALLOCATION_ORDER_DOMAIN: Literal["saliencegate:state-decay-v2:allocation-order:v1"] = (
    "saliencegate:state-decay-v2:allocation-order:v1"
)
ALLOCATION_GOLDEN_DOMAIN: Literal["saliencegate:state-decay-v2:allocation-golden:v1"] = (
    "saliencegate:state-decay-v2:allocation-golden:v1"
)
FOLD_KEY_DOMAIN: Literal["saliencegate:state-decay-v2:fold-key:v1"] = (
    "saliencegate:state-decay-v2:fold-key:v1"
)
PERMUTATION_INDEX_DOMAIN: Literal["saliencegate:state-decay-v2:permutation-index:v1"] = (
    "saliencegate:state-decay-v2:permutation-index:v1"
)
BOOTSTRAP_INDEX_DOMAIN: Literal["saliencegate:state-decay-v2:bootstrap-index:v1"] = (
    "saliencegate:state-decay-v2:bootstrap-index:v1"
)
PROPOSAL_FIXTURE_SEED_DOMAIN: Literal["saliencegate:state-decay-v2:proposal-fixture-seed:v1"] = (
    "saliencegate:state-decay-v2:proposal-fixture-seed:v1"
)
PROPOSAL_FIXTURE_SEED_COMMITMENT_DOMAIN: Literal[
    "saliencegate:state-decay-v2:proposal-fixture-seed-commitment:v1"
] = "saliencegate:state-decay-v2:proposal-fixture-seed-commitment:v1"
GENERATION_CONTRACT_DIGEST_DOMAIN = "saliencegate:state-decay-v2:generation-contract:v1"
SEED_COMMITMENT_SET_DIGEST_DOMAIN = "saliencegate:state-decay-v2:seed-commitment-set:v1"
TEMPLATE_REGISTRY_CONTRACT_DOMAIN = "saliencegate:state-decay-v2:template-registry-contract:v1"

PUBLIC_SEED_LITERAL: Literal["state-decay-v2-public-seed-v1"] = "state-decay-v2-public-seed-v1"
PUBLIC_GENERATION_SEED = bytes.fromhex(
    length_prefixed_sha256(PUBLIC_SEED_LITERAL, domain=PUBLIC_SEED_DOMAIN)
)

LINEAGES_PER_FAMILY: Literal[30] = 30
GENERATOR_SLOTS_PER_LINEAGE: Literal[5] = 5
GENERATED_INTEGER_MAX: Literal[1_000_000] = 1_000_000
TEMPLATE_REGISTRY_UNIQUE_FIELDS = (
    "candidate_packet_digest",
    "lineage_registry_key",
    "independent_seed_commitment_digest",
    "transition_graph_digest",
    "evidence_topology_digest",
    "failure_mechanism_id",
    "semantic_signature_digest",
)
_U64_MAX = (1 << 64) - 1


class SeedPurpose(StrEnum):
    ID = "id"
    ALLOCATION = "allocation"
    PUBLIC = "public"
    LOCKED = "locked"
    DIAGNOSTIC = "diagnostic"
    PROPOSAL = "proposal"
    PERMUTATION = "permutation"
    BOOTSTRAP = "bootstrap"


class CounterbalanceAxis(StrEnum):
    ALLOWED_ACTION_ORDER = "allowed_action_order"
    CANDIDATE_COUNT = "candidate_count"
    TRAJECTORY_LENGTH = "trajectory_length"
    TEXT_LENGTH_PROFILE = "text_length_profile"
    ACTION_POSITION = "action_position"
    EVIDENCE_COUNT = "evidence_count"
    MEMORY_VALIDITY_PROFILE = "memory_validity_profile"
    OPTIONAL_FIELD_PROFILE = "optional_field_profile"
    SEQUENCE_REVISION_PROFILE = "sequence_revision_profile"
    STRUCTURAL_SHAPE = "structural_shape"


class SeedSourceBoundary(StrEnum):
    TRACKED_PUBLIC = "tracked_public"
    CUSTODY_ONLY = "custody_only"


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class SeedDomainContract(_StrictModel):
    purpose: SeedPurpose
    derivation_domain: ComponentIdentifier
    commitment_domain: ComponentIdentifier

    @model_validator(mode="after")
    def domains_match_the_purpose(self) -> Self:
        if self.derivation_domain != seed_derivation_domain(self.purpose):
            raise ValueError("seed derivation domain does not match its purpose")
        if self.commitment_domain != seed_commitment_domain(self.purpose):
            raise ValueError("seed commitment domain does not match its purpose")
        return self


class PreGenerationHashDomains(_StrictModel):
    scenario_id: Literal["saliencegate:state-decay-v2:scenario-id:v1"] = SCENARIO_ID_DOMAIN
    allocation_order: Literal["saliencegate:state-decay-v2:allocation-order:v1"] = (
        ALLOCATION_ORDER_DOMAIN
    )
    fold_key: Literal["saliencegate:state-decay-v2:fold-key:v1"] = FOLD_KEY_DOMAIN
    permutation_index: Literal["saliencegate:state-decay-v2:permutation-index:v1"] = (
        PERMUTATION_INDEX_DOMAIN
    )
    bootstrap_index: Literal["saliencegate:state-decay-v2:bootstrap-index:v1"] = (
        BOOTSTRAP_INDEX_DOMAIN
    )

    @model_validator(mode="after")
    def domains_match_the_runtime_primitives(self) -> Self:
        if (
            self.scenario_id,
            self.allocation_order,
            self.fold_key,
            self.permutation_index,
            self.bootstrap_index,
        ) != (
            SCENARIO_ID_DOMAIN,
            ALLOCATION_ORDER_DOMAIN,
            FOLD_KEY_DOMAIN,
            PERMUTATION_INDEX_DOMAIN,
            BOOTSTRAP_INDEX_DOMAIN,
        ):
            raise ValueError("pre-generation hash domains do not match the runtime primitives")
        return self


class OutcomeMultiplicity(_StrictModel):
    outcome: ScenarioOutcome
    count_per_lineage: Annotated[int, Field(ge=1, le=2)]


class SplitGeometry(_StrictModel):
    split: BenchmarkSplit
    families: Annotated[tuple[ScenarioFamily, ...], Field(min_length=2, max_length=8)]
    lineages_per_family: Literal[30] = LINEAGES_PER_FAMILY
    generator_slots_per_lineage: Literal[5] = GENERATOR_SLOTS_PER_LINEAGE
    scenario_count: Annotated[int, Field(ge=300, le=1_200)]

    @model_validator(mode="after")
    def geometry_matches_the_frozen_split(self) -> Self:
        expected_families = _SPLIT_FAMILIES[self.split]
        if self.families != expected_families:
            raise ValueError("split families do not match the generation contract")
        expected_count = len(expected_families) * LINEAGES_PER_FAMILY * GENERATOR_SLOTS_PER_LINEAGE
        if self.scenario_count != expected_count:
            raise ValueError("split scenario count does not match its geometry")
        return self


class SeedCommitment(_StrictModel):
    purpose: SeedPurpose
    source_boundary: SeedSourceBoundary
    commitment_digest: Sha256Digest


class SeedCommitmentSet(_StrictModel):
    schema_version: Literal["state-decay-v2-seed-commitments/v1"] = (
        SEED_COMMITMENT_SET_SCHEMA_VERSION
    )
    commitments: Annotated[tuple[SeedCommitment, ...], Field(min_length=8, max_length=8)]
    commitment_set_digest: Sha256Digest

    @model_validator(mode="after")
    def commitments_are_complete_ordered_and_self_attesting(self) -> Self:
        if tuple(item.purpose for item in self.commitments) != tuple(SeedPurpose):
            raise ValueError("seed commitments must be complete and canonically ordered")
        boundaries = {item.source_boundary for item in self.commitments}
        if len(boundaries) != 1:
            raise ValueError("seed commitments must share one source boundary")
        if len({item.commitment_digest for item in self.commitments}) != len(self.commitments):
            raise ValueError("seed commitments must be distinct")
        public_commitments = tuple(
            seed_commitment(derive_seed(PUBLIC_GENERATION_SEED, purpose), purpose)
            for purpose in SeedPurpose
        )
        actual_commitments = tuple(item.commitment_digest for item in self.commitments)
        boundary = self.commitments[0].source_boundary
        if (
            boundary is SeedSourceBoundary.TRACKED_PUBLIC
            and actual_commitments != public_commitments
        ):
            raise ValueError("tracked public commitments do not match the canonical public source")
        if boundary is SeedSourceBoundary.CUSTODY_ONLY and not set(actual_commitments).isdisjoint(
            public_commitments
        ):
            raise ValueError("custody commitments must be disjoint from the tracked public source")
        expected = seed_commitment_set_digest(self)
        if not hmac.compare_digest(self.commitment_set_digest, expected):
            raise ValueError("seed commitment set digest does not match")
        return self


class GenerationContract(_StrictModel):
    schema_version: Literal["state-decay-v2-generation-contract/v1"] = (
        GENERATION_CONTRACT_SCHEMA_VERSION
    )
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    public_source_literal: Literal["state-decay-v2-public-seed-v1"] = PUBLIC_SEED_LITERAL
    public_source_domain: Literal["saliencegate:state-decay-v2:public-seed:v1"] = PUBLIC_SEED_DOMAIN
    public_source_digest: Sha256Digest
    hash_primitive: Literal["length-prefixed-sha256/v1"] = "length-prefixed-sha256/v1"
    digest_to_seed_conversion: Literal["bytes-from-lowercase-hex/v1"] = (
        "bytes-from-lowercase-hex/v1"
    )
    seed_width_bytes: Literal[32] = 32
    integer_coordinate_encoding: Literal["unsigned-64-bit-big-endian"] = (
        "unsigned-64-bit-big-endian"
    )
    seed_domains: Annotated[tuple[SeedDomainContract, ...], Field(min_length=8, max_length=8)]
    hash_domains: PreGenerationHashDomains
    splits: Annotated[tuple[SplitGeometry, ...], Field(min_length=4, max_length=4)]
    outcome_multiplicities: Annotated[
        tuple[OutcomeMultiplicity, ...],
        Field(min_length=4, max_length=4),
    ]
    counterbalance_axes: Annotated[
        tuple[CounterbalanceAxis, ...],
        Field(min_length=10, max_length=10),
    ]
    generated_integer_max: Literal[1_000_000] = GENERATED_INTEGER_MAX
    id_method: Literal["pre-allocation-skeleton-length-prefixed-sha256/v1"] = (
        "pre-allocation-skeleton-length-prefixed-sha256/v1"
    )
    allocation_method: Literal["pre-id-hash-ranked-balanced-rotation/v1"] = (
        "pre-id-hash-ranked-balanced-rotation/v1"
    )
    allocation_coordinate_order: Annotated[tuple[str, ...], Field(min_length=3, max_length=3)]
    allocation_coordinate_encoding: Annotated[
        tuple[str, ...],
        Field(min_length=3, max_length=3),
    ]
    allocation_sort_key: Annotated[tuple[str, ...], Field(min_length=2, max_length=2)]
    allocation_golden_domain: Literal["saliencegate:state-decay-v2:allocation-golden:v1"] = (
        ALLOCATION_GOLDEN_DOMAIN
    )
    allocation_golden_family: Literal[ScenarioFamily.FORGOTTEN_REQUIREMENT] = (
        ScenarioFamily.FORGOTTEN_REQUIREMENT
    )
    allocation_golden_registry_rule: Literal[
        "lineage-NN-with-sha256-ascii-candidate-packet-NN/v1"
    ] = "lineage-NN-with-sha256-ascii-candidate-packet-NN/v1"
    allocation_golden_first_packet_digest: Sha256Digest
    allocation_golden_first_order_digest: Sha256Digest
    allocation_golden_first_rank: Literal[0] = 0
    allocation_golden_fixture_digest: Sha256Digest
    template_registry_contract_digest: Sha256Digest
    contract_digest: Sha256Digest

    @model_validator(mode="after")
    def contract_is_complete_and_self_attesting(self) -> Self:
        if self.public_source_digest != PUBLIC_GENERATION_SEED.hex():
            raise ValueError("public seed digest does not match the tracked source")
        if tuple(item.purpose for item in self.seed_domains) != tuple(SeedPurpose):
            raise ValueError("seed domains must be complete and canonically ordered")
        if (
            len(
                {
                    domain
                    for item in self.seed_domains
                    for domain in (item.derivation_domain, item.commitment_domain)
                }
            )
            != len(self.seed_domains) * 2
        ):
            raise ValueError("seed domains must be distinct")
        if tuple(item.split for item in self.splits) != tuple(BenchmarkSplit):
            raise ValueError("split geometry must be complete and canonically ordered")
        if tuple(item.outcome for item in self.outcome_multiplicities) != tuple(ScenarioOutcome):
            raise ValueError("outcome multiplicities must be complete and canonically ordered")
        if tuple(item.count_per_lineage for item in self.outcome_multiplicities) != (2, 1, 1, 1):
            raise ValueError("outcome multiplicities must be two-one-one-one")
        if self.counterbalance_axes != tuple(CounterbalanceAxis):
            raise ValueError("counterbalance axes must be complete and canonically ordered")
        if self.allocation_coordinate_order != (
            "allocation_leaf",
            "family",
            "candidate_packet_digest",
        ):
            raise ValueError("allocation coordinate order is not canonical")
        if self.allocation_coordinate_encoding != (
            "bytes32",
            "utf8",
            "lowercase-hex-utf8",
        ):
            raise ValueError("allocation coordinate encoding is not canonical")
        if self.allocation_sort_key != (
            "allocation_order_digest-lowercase-hex",
            "candidate_packet_digest-lowercase-hex-utf8",
        ):
            raise ValueError("allocation sort key is not canonical")
        golden_rows = _allocation_golden_rows()
        first = next(item for item in golden_rows if item["lineage_registry_key"] == "lineage-00")
        if (
            self.allocation_golden_first_packet_digest != first["candidate_packet_digest"]
            or self.allocation_golden_first_order_digest != first["allocation_order_digest"]
            or self.allocation_golden_first_rank != first["allocation_rank"]
        ):
            raise ValueError("allocation first golden does not match")
        fixture_digest = length_prefixed_sha256(
            canonical_json(golden_rows),
            domain=ALLOCATION_GOLDEN_DOMAIN,
        )
        if not hmac.compare_digest(
            self.allocation_golden_fixture_digest,
            fixture_digest,
        ):
            raise ValueError("allocation golden fixture digest does not match")
        if self.template_registry_contract_digest != template_registry_contract_digest():
            raise ValueError("template registry contract digest does not match")
        if not hmac.compare_digest(self.contract_digest, generation_contract_digest(self)):
            raise ValueError("generation contract digest does not match")
        return self


GeneratorSlot = Annotated[int, Field(ge=0, le=4)]
AllocationRank = Annotated[int, Field(ge=0, le=29)]


def _outcomes_for_allocation_rank(rank: int) -> tuple[ScenarioOutcome, ...]:
    if type(rank) is not int or not 0 <= rank < LINEAGES_PER_FAMILY:
        raise ValueError("allocation rank is invalid")
    rotation = rank % GENERATOR_SLOTS_PER_LINEAGE
    outcomes = [ScenarioOutcome.HELPFUL] * GENERATOR_SLOTS_PER_LINEAGE
    outcomes[rotation] = ScenarioOutcome.HARMFUL
    outcomes[(rotation + 1) % GENERATOR_SLOTS_PER_LINEAGE] = ScenarioOutcome.REDUNDANT
    outcomes[(rotation + 2) % GENERATOR_SLOTS_PER_LINEAGE] = ScenarioOutcome.UNRESOLVED
    return tuple(outcomes)


class LineageAllocation(_StrictModel):
    schema_version: Literal["state-decay-v2-lineage-allocation/v1"] = (
        "state-decay-v2-lineage-allocation/v1"
    )
    family: ScenarioFamily
    lineage_registry_key: ComponentIdentifier
    candidate_packet_digest: Sha256Digest
    allocation_order_digest: Sha256Digest
    allocation_rank: AllocationRank
    outcomes_by_slot: Annotated[tuple[ScenarioOutcome, ...], Field(min_length=5, max_length=5)]

    @model_validator(mode="after")
    def allocation_has_the_frozen_quota(self) -> Self:
        counts = tuple(self.outcomes_by_slot.count(outcome) for outcome in ScenarioOutcome)
        if counts != (2, 1, 1, 1):
            raise ValueError("lineage allocation must preserve two-one-one-one")
        if self.outcomes_by_slot != _outcomes_for_allocation_rank(self.allocation_rank):
            raise ValueError("lineage allocation does not match its deterministic rank rotation")
        return self


_SPLIT_FAMILIES: Mapping[BenchmarkSplit, tuple[ScenarioFamily, ...]] = MappingProxyType(
    {
        BenchmarkSplit.TRAIN: (
            ScenarioFamily.FORGOTTEN_REQUIREMENT,
            ScenarioFamily.FAILED_PRIOR_ATTEMPT,
            ScenarioFamily.NEGLECTED_SUBGOAL,
            ScenarioFamily.STALE_MEMORY,
        ),
        BenchmarkSplit.DEVELOPMENT: (
            ScenarioFamily.STABLE_ENVIRONMENT_FACT,
            ScenarioFamily.RETAINED_DIAGNOSIS,
        ),
        BenchmarkSplit.LOCKED: (
            ScenarioFamily.CONFLICTING_EVIDENCE,
            ScenarioFamily.IRREVERSIBLE_ACTION,
        ),
        BenchmarkSplit.DIAGNOSTIC: tuple(ScenarioFamily),
    }
)


def seed_derivation_domain(purpose: SeedPurpose) -> str:
    if type(purpose) is not SeedPurpose:
        raise ValueError("seed purpose is invalid")
    return f"saliencegate:state-decay-v2:seed:{purpose.value}:v1"


def seed_commitment_domain(purpose: SeedPurpose) -> str:
    if type(purpose) is not SeedPurpose:
        raise ValueError("seed purpose is invalid")
    return f"saliencegate:state-decay-v2:seed-commitment:{purpose.value}:v1"


def u64be(value: int) -> bytes:
    if type(value) is not int or not 0 <= value <= _U64_MAX:
        raise ValueError("unsigned 64-bit coordinate is invalid")
    return value.to_bytes(8, byteorder="big", signed=False)


def _checked_seed(value: bytes) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise ValueError("seed must be exactly 32 bytes")
    return value


def derive_seed(
    source_seed: bytes,
    purpose: SeedPurpose,
    *coordinates: bytes,
) -> bytes:
    checked_source = _checked_seed(source_seed)
    if type(purpose) is not SeedPurpose:
        raise ValueError("seed purpose is invalid")
    if any(type(item) is not bytes for item in coordinates):
        raise ValueError("seed derivation coordinates are invalid")
    return bytes.fromhex(
        length_prefixed_sha256(
            checked_source,
            *coordinates,
            domain=seed_derivation_domain(purpose),
        )
    )


def seed_commitment(seed: bytes, purpose: SeedPurpose) -> str:
    return length_prefixed_sha256(
        _checked_seed(seed),
        domain=seed_commitment_domain(purpose),
    )


def derive_proposal_fixture_seed(
    proposal_leaf: bytes,
    split: BenchmarkSplit,
) -> bytes:
    checked_leaf = _checked_seed(proposal_leaf)
    if type(split) is not BenchmarkSplit:
        raise ValueError("proposal fixture split is invalid")
    return bytes.fromhex(
        length_prefixed_sha256(
            checked_leaf,
            split.value,
            domain=PROPOSAL_FIXTURE_SEED_DOMAIN,
        )
    )


def proposal_fixture_seed_commitment(seed: bytes) -> str:
    return length_prefixed_sha256(
        _checked_seed(seed),
        domain=PROPOSAL_FIXTURE_SEED_COMMITMENT_DOMAIN,
    )


def seed_commitment_set_digest(value: BaseModel | Mapping[str, object]) -> str:
    payload = (
        value.model_dump(mode="json", exclude={"commitment_set_digest"}, warnings=False)
        if isinstance(value, BaseModel)
        else {key: item for key, item in value.items() if key != "commitment_set_digest"}
    )
    return length_prefixed_sha256(
        canonical_json(payload),
        domain=SEED_COMMITMENT_SET_DIGEST_DOMAIN,
    )


def _build_seed_commitment_set(
    leaves: tuple[bytes, ...],
    *,
    source_boundary: SeedSourceBoundary,
) -> SeedCommitmentSet:
    if type(source_boundary) is not SeedSourceBoundary:
        raise ValueError("seed commitment inputs are invalid")
    if len(leaves) != len(SeedPurpose):
        raise ValueError("seed leaves must be complete")
    if any(type(leaf) is not bytes or len(leaf) != 32 for leaf in leaves):
        raise ValueError("seed leaves must each be exactly 32 bytes")
    if len(set(leaves)) != len(leaves):
        raise ValueError("seed leaves must be distinct")
    public_leaves = tuple(derive_seed(PUBLIC_GENERATION_SEED, purpose) for purpose in SeedPurpose)
    if source_boundary is SeedSourceBoundary.TRACKED_PUBLIC and leaves != public_leaves:
        raise ValueError("tracked public leaves do not match the canonical public source")
    if source_boundary is SeedSourceBoundary.CUSTODY_ONLY and not set(leaves).isdisjoint(
        public_leaves
    ):
        raise ValueError("custody leaves must be disjoint from the tracked public source")
    commitments = tuple(
        SeedCommitment(
            purpose=purpose,
            source_boundary=source_boundary,
            commitment_digest=seed_commitment(leaf, purpose),
        )
        for purpose, leaf in zip(SeedPurpose, leaves, strict=True)
    )
    values: dict[str, object] = {
        "schema_version": SEED_COMMITMENT_SET_SCHEMA_VERSION,
        "commitments": commitments,
    }
    values["commitment_set_digest"] = seed_commitment_set_digest(values)
    return SeedCommitmentSet.model_validate(values)


def derive_seed_commitment_set(
    source_seed: bytes,
    *,
    source_boundary: SeedSourceBoundary,
) -> SeedCommitmentSet:
    checked_source = _checked_seed(source_seed)
    if type(source_boundary) is not SeedSourceBoundary:
        raise ValueError("seed source boundary is invalid")
    if (
        source_boundary is SeedSourceBoundary.TRACKED_PUBLIC
        and checked_source != PUBLIC_GENERATION_SEED
    ):
        raise ValueError("tracked public source is not canonical")
    if (
        source_boundary is SeedSourceBoundary.CUSTODY_ONLY
        and checked_source == PUBLIC_GENERATION_SEED
    ):
        raise ValueError("custody source cannot be the tracked public source")
    leaves = tuple(derive_seed(checked_source, purpose) for purpose in SeedPurpose)
    return _build_seed_commitment_set(leaves, source_boundary=source_boundary)


def template_registry_contract_digest() -> str:
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": "state-decay-v2-template-registry-contract/v1",
                "families": tuple(item.value for item in ScenarioFamily),
                "lineages_per_family": LINEAGES_PER_FAMILY,
                "generator_slots_per_lineage": GENERATOR_SLOTS_PER_LINEAGE,
                "unique_fields": TEMPLATE_REGISTRY_UNIQUE_FIELDS,
            }
        ),
        domain=TEMPLATE_REGISTRY_CONTRACT_DOMAIN,
    )


def generation_contract_digest(value: BaseModel | Mapping[str, object]) -> str:
    payload = (
        value.model_dump(mode="json", exclude={"contract_digest"}, warnings=False)
        if isinstance(value, BaseModel)
        else {key: item for key, item in value.items() if key != "contract_digest"}
    )
    return length_prefixed_sha256(canonical_json(payload), domain=GENERATION_CONTRACT_DIGEST_DOMAIN)


def _allocation_golden_rows() -> tuple[dict[str, object], ...]:
    allocation_leaf = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.ALLOCATION)
    candidates = tuple(
        (
            f"lineage-{index:02d}",
            sha256(f"candidate-packet-{index:02d}".encode("ascii")).hexdigest(),
        )
        for index in range(LINEAGES_PER_FAMILY)
    )
    ranked = tuple(
        sorted(
            (
                (
                    length_prefixed_sha256(
                        allocation_leaf,
                        ScenarioFamily.FORGOTTEN_REQUIREMENT.value,
                        packet_digest,
                        domain=ALLOCATION_ORDER_DOMAIN,
                    ),
                    packet_digest,
                    lineage_registry_key,
                )
                for lineage_registry_key, packet_digest in candidates
            ),
            key=lambda item: (item[0], item[1].encode("utf-8")),
        )
    )
    return tuple(
        {
            "allocation_order_digest": allocation_digest,
            "candidate_packet_digest": packet_digest,
            "lineage_registry_key": lineage_registry_key,
            "allocation_rank": rank,
        }
        for rank, (allocation_digest, packet_digest, lineage_registry_key) in enumerate(ranked)
    )


def build_generation_contract() -> GenerationContract:
    seed_domains = tuple(
        SeedDomainContract(
            purpose=purpose,
            derivation_domain=seed_derivation_domain(purpose),
            commitment_domain=seed_commitment_domain(purpose),
        )
        for purpose in SeedPurpose
    )
    splits = tuple(
        SplitGeometry(
            split=split,
            families=_SPLIT_FAMILIES[split],
            scenario_count=(
                len(_SPLIT_FAMILIES[split]) * LINEAGES_PER_FAMILY * GENERATOR_SLOTS_PER_LINEAGE
            ),
        )
        for split in BenchmarkSplit
    )
    outcomes = tuple(
        OutcomeMultiplicity(
            outcome=outcome,
            count_per_lineage=2 if outcome is ScenarioOutcome.HELPFUL else 1,
        )
        for outcome in ScenarioOutcome
    )
    allocation_golden_rows = _allocation_golden_rows()
    allocation_golden_first = next(
        item for item in allocation_golden_rows if item["lineage_registry_key"] == "lineage-00"
    )
    values: dict[str, object] = {
        "schema_version": GENERATION_CONTRACT_SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "public_source_literal": PUBLIC_SEED_LITERAL,
        "public_source_domain": PUBLIC_SEED_DOMAIN,
        "public_source_digest": PUBLIC_GENERATION_SEED.hex(),
        "hash_primitive": "length-prefixed-sha256/v1",
        "digest_to_seed_conversion": "bytes-from-lowercase-hex/v1",
        "seed_width_bytes": 32,
        "integer_coordinate_encoding": "unsigned-64-bit-big-endian",
        "seed_domains": seed_domains,
        "hash_domains": PreGenerationHashDomains(
            scenario_id=SCENARIO_ID_DOMAIN,
            allocation_order=ALLOCATION_ORDER_DOMAIN,
            fold_key=FOLD_KEY_DOMAIN,
            permutation_index=PERMUTATION_INDEX_DOMAIN,
            bootstrap_index=BOOTSTRAP_INDEX_DOMAIN,
        ),
        "splits": splits,
        "outcome_multiplicities": outcomes,
        "counterbalance_axes": tuple(CounterbalanceAxis),
        "generated_integer_max": GENERATED_INTEGER_MAX,
        "id_method": "pre-allocation-skeleton-length-prefixed-sha256/v1",
        "allocation_method": "pre-id-hash-ranked-balanced-rotation/v1",
        "allocation_coordinate_order": (
            "allocation_leaf",
            "family",
            "candidate_packet_digest",
        ),
        "allocation_coordinate_encoding": (
            "bytes32",
            "utf8",
            "lowercase-hex-utf8",
        ),
        "allocation_sort_key": (
            "allocation_order_digest-lowercase-hex",
            "candidate_packet_digest-lowercase-hex-utf8",
        ),
        "allocation_golden_domain": ALLOCATION_GOLDEN_DOMAIN,
        "allocation_golden_family": ScenarioFamily.FORGOTTEN_REQUIREMENT,
        "allocation_golden_registry_rule": ("lineage-NN-with-sha256-ascii-candidate-packet-NN/v1"),
        "allocation_golden_first_packet_digest": allocation_golden_first["candidate_packet_digest"],
        "allocation_golden_first_order_digest": allocation_golden_first["allocation_order_digest"],
        "allocation_golden_first_rank": allocation_golden_first["allocation_rank"],
        "allocation_golden_fixture_digest": length_prefixed_sha256(
            canonical_json(allocation_golden_rows),
            domain=ALLOCATION_GOLDEN_DOMAIN,
        ),
        "template_registry_contract_digest": template_registry_contract_digest(),
    }
    values["contract_digest"] = generation_contract_digest(values)
    return GenerationContract.model_validate(values)


def _checked_lineages(
    lineages: Mapping[str, str] | Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    if isinstance(lineages, Mapping):
        items: tuple[object, ...] = tuple(lineages.items())
    elif isinstance(lineages, Sequence) and not isinstance(lineages, (str, bytes, bytearray)):
        items = tuple(lineages)
    else:
        raise ValueError("reviewed lineages are invalid")
    if len(items) != LINEAGES_PER_FAMILY:
        raise ValueError("allocation requires exactly thirty reviewed lineages")
    checked_items: list[tuple[str, str]] = []
    for item in items:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("reviewed lineage identities are invalid")
        key, digest = cast(tuple[object, object], item)
        if (
            type(key) is not str
            or not key
            or type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("reviewed lineage identities are invalid")
        checked_items.append((key, digest))
    keys = tuple(item[0] for item in checked_items)
    digests = tuple(item[1] for item in checked_items)
    if len(set(keys)) != len(keys) or len(set(digests)) != len(digests):
        raise ValueError("reviewed lineage identities are invalid")
    return tuple(checked_items)


def allocate_balanced_outcomes(
    allocation_seed: bytes,
    family: ScenarioFamily,
    lineages: Mapping[str, str] | Sequence[tuple[str, str]],
) -> tuple[LineageAllocation, ...]:
    checked_seed = _checked_seed(allocation_seed)
    if type(family) is not ScenarioFamily:
        raise ValueError("allocation family is invalid")
    items = _checked_lineages(lineages)
    ranked = sorted(
        (
            length_prefixed_sha256(
                checked_seed,
                family.value,
                packet_digest,
                domain=ALLOCATION_ORDER_DOMAIN,
            ),
            packet_digest,
            registry_key,
        )
        for registry_key, packet_digest in items
    )
    allocations: list[LineageAllocation] = []
    for rank, (order_digest, packet_digest, registry_key) in enumerate(ranked):
        allocations.append(
            LineageAllocation(
                family=family,
                lineage_registry_key=registry_key,
                candidate_packet_digest=packet_digest,
                allocation_order_digest=order_digest,
                allocation_rank=rank,
                outcomes_by_slot=_outcomes_for_allocation_rank(rank),
            )
        )
    return tuple(allocations)


def validate_balanced_allocations(
    allocation_seed: bytes,
    family: ScenarioFamily,
    lineages: Mapping[str, str] | Sequence[tuple[str, str]],
    allocations: Sequence[LineageAllocation],
) -> tuple[LineageAllocation, ...]:
    if not isinstance(allocations, Sequence) or isinstance(allocations, (str, bytes, bytearray)):
        raise ValueError("lineage allocations are invalid")
    supplied = tuple(allocations)
    if any(type(item) is not LineageAllocation for item in supplied):
        raise ValueError("lineage allocations are invalid")
    expected = allocate_balanced_outcomes(allocation_seed, family, lineages)
    if supplied != expected:
        raise ValueError("lineage allocations do not match the committed seed and registry")
    return supplied


def derive_scenario_id(
    id_seed: bytes,
    *,
    skeleton_digest: str,
) -> str:
    checked_seed = _checked_seed(id_seed)
    if (
        type(skeleton_digest) is not str
        or len(skeleton_digest) != 64
        or any(character not in "0123456789abcdef" for character in skeleton_digest)
    ):
        raise ValueError("scenario skeleton digest is invalid")
    return length_prefixed_sha256(
        checked_seed,
        skeleton_digest,
        domain=SCENARIO_ID_DOMAIN,
    )


GENERATION_CONTRACT = build_generation_contract()


__all__ = [
    "ALLOCATION_GOLDEN_DOMAIN",
    "ALLOCATION_ORDER_DOMAIN",
    "BOOTSTRAP_INDEX_DOMAIN",
    "FOLD_KEY_DOMAIN",
    "GENERATED_INTEGER_MAX",
    "GENERATION_CONTRACT",
    "GENERATION_CONTRACT_SCHEMA_VERSION",
    "GENERATOR_SLOTS_PER_LINEAGE",
    "LINEAGES_PER_FAMILY",
    "PERMUTATION_INDEX_DOMAIN",
    "PROPOSAL_FIXTURE_SEED_COMMITMENT_DOMAIN",
    "PROPOSAL_FIXTURE_SEED_DOMAIN",
    "PUBLIC_GENERATION_SEED",
    "PUBLIC_SEED_DOMAIN",
    "PUBLIC_SEED_LITERAL",
    "SCENARIO_ID_DOMAIN",
    "SEED_COMMITMENT_SET_SCHEMA_VERSION",
    "TEMPLATE_REGISTRY_UNIQUE_FIELDS",
    "CounterbalanceAxis",
    "GenerationContract",
    "LineageAllocation",
    "OutcomeMultiplicity",
    "PreGenerationHashDomains",
    "SeedCommitment",
    "SeedCommitmentSet",
    "SeedDomainContract",
    "SeedPurpose",
    "SeedSourceBoundary",
    "SplitGeometry",
    "allocate_balanced_outcomes",
    "build_generation_contract",
    "derive_proposal_fixture_seed",
    "derive_scenario_id",
    "derive_seed",
    "derive_seed_commitment_set",
    "generation_contract_digest",
    "proposal_fixture_seed_commitment",
    "seed_commitment",
    "seed_commitment_domain",
    "seed_commitment_set_digest",
    "seed_derivation_domain",
    "template_registry_contract_digest",
    "u64be",
    "validate_balanced_allocations",
]
