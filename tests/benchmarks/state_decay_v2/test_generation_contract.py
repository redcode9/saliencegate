from __future__ import annotations

from hashlib import sha256

import pytest
from pydantic import ValidationError

import saliencegate.benchmarks.state_decay_v2.config as generation_config
from saliencegate.benchmarks.state_decay_v2.config import (
    ALLOCATION_ORDER_DOMAIN,
    BOOTSTRAP_INDEX_DOMAIN,
    FOLD_KEY_DOMAIN,
    GENERATED_INTEGER_MAX,
    GENERATION_CONTRACT,
    GENERATOR_SLOTS_PER_LINEAGE,
    LINEAGES_PER_FAMILY,
    PERMUTATION_INDEX_DOMAIN,
    PUBLIC_GENERATION_SEED,
    PUBLIC_SEED_DOMAIN,
    PUBLIC_SEED_LITERAL,
    SCENARIO_ID_DOMAIN,
    CounterbalanceAxis,
    GenerationContract,
    LineageAllocation,
    SeedDomainContract,
    SeedPurpose,
    SeedSourceBoundary,
    SplitGeometry,
    allocate_balanced_outcomes,
    derive_proposal_fixture_seed,
    derive_scenario_id,
    derive_seed,
    derive_seed_commitment_set,
    proposal_fixture_seed_commitment,
    seed_commitment,
    seed_commitment_domain,
    seed_commitment_set_digest,
    seed_derivation_domain,
    u64be,
    validate_balanced_allocations,
)
from saliencegate.benchmarks.state_decay_v2.schema import (
    BenchmarkSplit,
    ScenarioFamily,
    ScenarioOutcome,
)
from saliencegate.domain import canonical_json


def _digest(value: str) -> str:
    return sha256(value.encode("ascii")).hexdigest()


def _lineages() -> tuple[tuple[str, str], ...]:
    return tuple(
        (f"lineage-{index:02d}", _digest(f"candidate-packet-{index:02d}"))
        for index in range(LINEAGES_PER_FAMILY)
    )


def _purpose_leaves(source: bytes = PUBLIC_GENERATION_SEED) -> dict[SeedPurpose, bytes]:
    return {purpose: derive_seed(source, purpose) for purpose in SeedPurpose}


def test_generation_contract_is_complete_canonical_and_golden() -> None:
    assert GENERATION_CONTRACT.contract_digest == (
        "4a713f38074f220abc287345b73bd70ec0cf33a86006dcaa1162b57350dda7fa"
    )
    assert GenerationContract.model_validate_json(canonical_json(GENERATION_CONTRACT)) == (
        GENERATION_CONTRACT
    )
    assert GENERATION_CONTRACT.public_source_literal == PUBLIC_SEED_LITERAL
    assert GENERATION_CONTRACT.public_source_domain == PUBLIC_SEED_DOMAIN
    assert GENERATION_CONTRACT.public_source_digest == PUBLIC_GENERATION_SEED.hex()
    assert GENERATION_CONTRACT.hash_primitive == "length-prefixed-sha256/v1"
    assert GENERATION_CONTRACT.digest_to_seed_conversion == "bytes-from-lowercase-hex/v1"
    assert GENERATION_CONTRACT.seed_width_bytes == 32
    assert GENERATION_CONTRACT.integer_coordinate_encoding == "unsigned-64-bit-big-endian"
    assert GENERATION_CONTRACT.generated_integer_max == GENERATED_INTEGER_MAX
    assert tuple(item.purpose for item in GENERATION_CONTRACT.seed_domains) == tuple(SeedPurpose)
    assert GENERATION_CONTRACT.hash_domains.model_dump(mode="python") == {
        "scenario_id": SCENARIO_ID_DOMAIN,
        "allocation_order": ALLOCATION_ORDER_DOMAIN,
        "fold_key": FOLD_KEY_DOMAIN,
        "permutation_index": PERMUTATION_INDEX_DOMAIN,
        "bootstrap_index": BOOTSTRAP_INDEX_DOMAIN,
    }
    assert GENERATION_CONTRACT.counterbalance_axes == tuple(CounterbalanceAxis)
    assert GENERATION_CONTRACT.allocation_coordinate_order == (
        "allocation_leaf",
        "family",
        "candidate_packet_digest",
    )
    assert GENERATION_CONTRACT.allocation_coordinate_encoding == (
        "bytes32",
        "utf8",
        "lowercase-hex-utf8",
    )
    assert GENERATION_CONTRACT.allocation_golden_first_packet_digest == (
        "b457b8ca35e3922e20d4fa67aa041a702d94b3915e5b4f5af4444b62b37beaf3"
    )
    assert GENERATION_CONTRACT.allocation_golden_first_order_digest == (
        "0391ffff0035b07216cc2849d411ee96a2a6fc6a2e70be03a8260c0c4059ebfd"
    )
    assert GENERATION_CONTRACT.allocation_golden_first_rank == 0
    assert GENERATION_CONTRACT.allocation_golden_fixture_digest == (
        "d6c1acf1d4606bf72276a5c2272cbe036c0d389c6dd66a746e1c9cd2325961a7"
    )

    split_counts = {item.split: item.scenario_count for item in GENERATION_CONTRACT.splits}
    assert split_counts == {
        BenchmarkSplit.TRAIN: 600,
        BenchmarkSplit.DEVELOPMENT: 300,
        BenchmarkSplit.LOCKED: 300,
        BenchmarkSplit.DIAGNOSTIC: 1_200,
    }
    assert tuple(item.count_per_lineage for item in GENERATION_CONTRACT.outcome_multiplicities) == (
        2,
        1,
        1,
        1,
    )


def test_generation_contract_rejects_tampering_and_non_strict_inputs() -> None:
    for field, value in (
        ("contract_digest", "0" * 64),
        ("generated_integer_max", True),
        ("public_source_literal", "replacement"),
    ):
        payload = GENERATION_CONTRACT.model_dump(mode="python")
        payload[field] = value
        with pytest.raises(ValidationError):
            GenerationContract.model_validate(payload)

    payload = GENERATION_CONTRACT.model_dump(mode="python")
    payload["undeclared"] = "value"
    with pytest.raises(ValidationError):
        GenerationContract.model_validate(payload)

    with pytest.raises(ValidationError):
        GENERATION_CONTRACT.splits[0].scenario_count = 601  # type: ignore[misc]


def test_component_contract_validators_reject_internal_mismatches() -> None:
    seed_domain = GENERATION_CONTRACT.seed_domains[0]
    for field, value, message in (
        (
            "derivation_domain",
            seed_derivation_domain(SeedPurpose.ALLOCATION),
            "seed derivation domain",
        ),
        (
            "commitment_domain",
            seed_commitment_domain(SeedPurpose.ALLOCATION),
            "seed commitment domain",
        ),
    ):
        payload = seed_domain.model_dump(mode="python")
        payload[field] = value
        with pytest.raises(ValidationError, match=message):
            SeedDomainContract.model_validate(payload)

    mismatched_hash_domains = GENERATION_CONTRACT.hash_domains.model_copy(
        update={"scenario_id": ALLOCATION_ORDER_DOMAIN}
    )
    with pytest.raises(ValueError, match="pre-generation hash domains"):
        mismatched_hash_domains.domains_match_the_runtime_primitives()

    split = GENERATION_CONTRACT.splits[0]
    wrong_families = split.model_dump(mode="python")
    wrong_families["families"] = GENERATION_CONTRACT.splits[1].families
    with pytest.raises(ValidationError, match="split families"):
        SplitGeometry.model_validate(wrong_families)

    wrong_count = split.model_dump(mode="python")
    wrong_count["scenario_count"] = split.scenario_count + 1
    with pytest.raises(ValidationError, match="split scenario count"):
        SplitGeometry.model_validate(wrong_count)


def test_generation_contract_rejects_each_cross_field_invariant_mismatch() -> None:
    outcome_multiplicities = list(GENERATION_CONTRACT.outcome_multiplicities)
    outcome_multiplicities[0] = outcome_multiplicities[0].model_copy(
        update={"count_per_lineage": 1}
    )
    cases = (
        ("public_source_digest", "0" * 64, "public seed digest"),
        ("seed_domains", tuple(reversed(GENERATION_CONTRACT.seed_domains)), "seed domains"),
        ("splits", tuple(reversed(GENERATION_CONTRACT.splits)), "split geometry"),
        (
            "outcome_multiplicities",
            tuple(reversed(GENERATION_CONTRACT.outcome_multiplicities)),
            "outcome multiplicities",
        ),
        (
            "outcome_multiplicities",
            tuple(outcome_multiplicities),
            "two-one-one-one",
        ),
        (
            "counterbalance_axes",
            tuple(reversed(GENERATION_CONTRACT.counterbalance_axes)),
            "counterbalance axes",
        ),
        (
            "allocation_coordinate_order",
            tuple(reversed(GENERATION_CONTRACT.allocation_coordinate_order)),
            "allocation coordinate order",
        ),
        (
            "allocation_coordinate_encoding",
            tuple(reversed(GENERATION_CONTRACT.allocation_coordinate_encoding)),
            "allocation coordinate encoding",
        ),
        (
            "allocation_sort_key",
            tuple(reversed(GENERATION_CONTRACT.allocation_sort_key)),
            "allocation sort key",
        ),
        ("allocation_golden_first_packet_digest", "0" * 64, "allocation first golden"),
        ("allocation_golden_fixture_digest", "0" * 64, "allocation golden fixture"),
        ("template_registry_contract_digest", "0" * 64, "template registry contract"),
    )
    for field, value, message in cases:
        payload = GENERATION_CONTRACT.model_dump(mode="python")
        payload[field] = value
        with pytest.raises(ValidationError, match=message):
            GenerationContract.model_validate(payload)

    duplicate_domain = GENERATION_CONTRACT.seed_domains[1].model_copy(
        update={
            "derivation_domain": GENERATION_CONTRACT.seed_domains[0].derivation_domain,
        }
    )
    duplicate_domains = GENERATION_CONTRACT.model_copy(
        update={
            "seed_domains": (
                GENERATION_CONTRACT.seed_domains[0],
                duplicate_domain,
                *GENERATION_CONTRACT.seed_domains[2:],
            )
        }
    )
    with pytest.raises(ValueError, match="seed domains must be distinct"):
        duplicate_domains.contract_is_complete_and_self_attesting()


def test_seed_domains_and_derivation_are_separate_reproducible_and_bounded() -> None:
    assert PUBLIC_GENERATION_SEED.hex() == (
        "78014e2210de533a51c5c26f0fcda82a792a622a151de6c7cac2dec876387f95"
    )
    assert (SCENARIO_ID_DOMAIN, ALLOCATION_ORDER_DOMAIN, FOLD_KEY_DOMAIN) == (
        "saliencegate:state-decay-v2:scenario-id:v1",
        "saliencegate:state-decay-v2:allocation-order:v1",
        "saliencegate:state-decay-v2:fold-key:v1",
    )
    assert (PERMUTATION_INDEX_DOMAIN, BOOTSTRAP_INDEX_DOMAIN) == (
        "saliencegate:state-decay-v2:permutation-index:v1",
        "saliencegate:state-decay-v2:bootstrap-index:v1",
    )

    leaves = _purpose_leaves()
    assert len(set(leaves.values())) == len(SeedPurpose)
    assert leaves == _purpose_leaves()
    assert (
        len(
            {
                domain
                for purpose in SeedPurpose
                for domain in (seed_derivation_domain(purpose), seed_commitment_domain(purpose))
            }
        )
        == 16
    )
    assert derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.ID, u64be(7)) != leaves[SeedPurpose.ID]
    assert u64be(0) == b"\x00" * 8
    assert u64be((1 << 64) - 1) == b"\xff" * 8

    for invalid in (-1, 1 << 64, True, 1.0):
        with pytest.raises(ValueError, match="unsigned 64-bit"):
            u64be(invalid)  # type: ignore[arg-type]
    for invalid_seed in (b"", b"x" * 31, b"x" * 33, bytearray(32)):
        with pytest.raises(ValueError, match="exactly 32 bytes"):
            derive_seed(invalid_seed, SeedPurpose.ID)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="purpose"):
        derive_seed(PUBLIC_GENERATION_SEED, "id")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="coordinates"):
        derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.ID, "coordinate")  # type: ignore[arg-type]


def test_seed_and_proposal_helpers_reject_wrong_coordinate_types() -> None:
    proposal_leaf = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.PROPOSAL)
    for helper in (seed_derivation_domain, seed_commitment_domain):
        with pytest.raises(ValueError, match="seed purpose"):
            helper("id")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="seed purpose"):
        seed_commitment(PUBLIC_GENERATION_SEED, "id")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        derive_proposal_fixture_seed(b"short", BenchmarkSplit.TRAIN)
    with pytest.raises(ValueError, match="proposal fixture split"):
        derive_proposal_fixture_seed(
            proposal_leaf,
            "train",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        proposal_fixture_seed_commitment(b"short")

    for rank in (-1, LINEAGES_PER_FAMILY, True):
        with pytest.raises(ValueError, match="allocation rank"):
            generation_config._outcomes_for_allocation_rank(rank)  # type: ignore[arg-type]


@pytest.mark.parametrize("changed_purpose", tuple(SeedPurpose))
def test_each_purpose_leaf_can_change_without_changing_any_sibling(
    changed_purpose: SeedPurpose,
) -> None:
    original = _purpose_leaves()
    mutated = {
        purpose: (
            derive_seed(PUBLIC_GENERATION_SEED, purpose, b"isolated-metamorphic-change")
            if purpose is changed_purpose
            else leaf
        )
        for purpose, leaf in original.items()
    }
    changed_commitments = tuple(
        purpose
        for purpose in SeedPurpose
        if seed_commitment(original[purpose], purpose) != seed_commitment(mutated[purpose], purpose)
    )
    assert changed_commitments == (changed_purpose,)


def test_proposal_fixture_seed_is_split_derived_from_only_the_proposal_leaf() -> None:
    proposal_leaf = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.PROPOSAL)
    commitments = tuple(
        proposal_fixture_seed_commitment(derive_proposal_fixture_seed(proposal_leaf, split))
        for split in BenchmarkSplit
    )
    assert len(set(commitments)) == len(BenchmarkSplit)

    changed_proposal_leaf = derive_seed(
        PUBLIC_GENERATION_SEED,
        SeedPurpose.PROPOSAL,
        b"isolated-metamorphic-change",
    )
    assert (
        tuple(
            proposal_fixture_seed_commitment(
                derive_proposal_fixture_seed(changed_proposal_leaf, split)
            )
            for split in BenchmarkSplit
        )
        != commitments
    )


def test_seed_commitment_sets_bind_every_leaf_without_serializing_seeds() -> None:
    commitment_set = derive_seed_commitment_set(
        PUBLIC_GENERATION_SEED,
        source_boundary=SeedSourceBoundary.TRACKED_PUBLIC,
    )
    assert tuple(item.purpose for item in commitment_set.commitments) == tuple(SeedPurpose)
    assert {item.source_boundary for item in commitment_set.commitments} == {
        SeedSourceBoundary.TRACKED_PUBLIC
    }
    assert commitment_set.commitment_set_digest == (
        "de25d5d7bd4ab3e9c353287db358666942f01faa47314ea2443beb033b129218"
    )
    serialized = canonical_json(commitment_set)
    assert PUBLIC_GENERATION_SEED.hex().encode("ascii") not in serialized
    for leaf in _purpose_leaves().values():
        assert leaf.hex().encode("ascii") not in serialized

    payload = commitment_set.model_dump(mode="python")
    payload["commitment_set_digest"] = "0" * 64
    with pytest.raises(ValidationError):
        type(commitment_set).model_validate(payload)


def test_seed_commitment_set_rejects_reordering_and_public_digest_substitution() -> None:
    public_set = derive_seed_commitment_set(
        PUBLIC_GENERATION_SEED,
        source_boundary=SeedSourceBoundary.TRACKED_PUBLIC,
    )
    reordered = public_set.model_dump(mode="python")
    reordered["commitments"] = tuple(reversed(reordered["commitments"]))
    with pytest.raises(ValidationError, match="canonically ordered"):
        type(public_set).model_validate(reordered)

    substituted = public_set.model_dump(mode="python")
    first = dict(substituted["commitments"][0])
    first["commitment_digest"] = "0" * 64
    substituted["commitments"] = (first, *substituted["commitments"][1:])
    with pytest.raises(ValidationError, match="canonical public source"):
        type(public_set).model_validate(substituted)


def test_seed_commitment_building_rejects_malformed_leaf_sets_and_boundaries() -> None:
    public_leaves = tuple(_purpose_leaves().values())
    custody_leaves = tuple(
        _purpose_leaves(bytes.fromhex(_digest("independent-custody-source"))).values()
    )

    with pytest.raises(ValueError, match="seed source boundary"):
        derive_seed_commitment_set(
            PUBLIC_GENERATION_SEED,
            source_boundary="tracked_public",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="seed commitment inputs"):
        generation_config._build_seed_commitment_set(
            public_leaves,
            source_boundary="tracked_public",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="complete"):
        generation_config._build_seed_commitment_set(
            public_leaves[:-1],
            source_boundary=SeedSourceBoundary.TRACKED_PUBLIC,
        )
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        generation_config._build_seed_commitment_set(
            (*public_leaves[:-1], b"short"),
            source_boundary=SeedSourceBoundary.TRACKED_PUBLIC,
        )
    with pytest.raises(ValueError, match="distinct"):
        generation_config._build_seed_commitment_set(
            (*public_leaves[:-1], public_leaves[0]),
            source_boundary=SeedSourceBoundary.TRACKED_PUBLIC,
        )
    with pytest.raises(ValueError, match="canonical public source"):
        generation_config._build_seed_commitment_set(
            custody_leaves,
            source_boundary=SeedSourceBoundary.TRACKED_PUBLIC,
        )
    with pytest.raises(ValueError, match="disjoint"):
        generation_config._build_seed_commitment_set(
            (public_leaves[0], *custody_leaves[1:]),
            source_boundary=SeedSourceBoundary.CUSTODY_ONLY,
        )


def test_custody_source_mutation_changes_every_leaf_and_the_root() -> None:
    original = derive_seed_commitment_set(
        bytes.fromhex(_digest("custody-source-a")),
        source_boundary=SeedSourceBoundary.CUSTODY_ONLY,
    )
    mutated = derive_seed_commitment_set(
        bytes.fromhex(_digest("custody-source-b")),
        source_boundary=SeedSourceBoundary.CUSTODY_ONLY,
    )

    changed = tuple(
        before.purpose
        for before, after in zip(original.commitments, mutated.commitments, strict=True)
        if before.commitment_digest != after.commitment_digest
    )
    assert changed == tuple(SeedPurpose)
    assert original.commitment_set_digest != mutated.commitment_set_digest

    duplicate_payload = original.model_dump(mode="python")
    duplicate = dict(duplicate_payload["commitments"][-1])
    duplicate["commitment_digest"] = duplicate_payload["commitments"][-2]["commitment_digest"]
    duplicate_payload["commitments"] = (*duplicate_payload["commitments"][:-1], duplicate)
    with pytest.raises(ValidationError, match="distinct"):
        type(original).model_validate(duplicate_payload)


def test_seed_source_boundaries_cannot_be_relabelled() -> None:
    custody_source = bytes.fromhex(_digest("custody-source"))
    with pytest.raises(ValueError, match="tracked public source"):
        derive_seed_commitment_set(
            custody_source,
            source_boundary=SeedSourceBoundary.TRACKED_PUBLIC,
        )
    with pytest.raises(ValueError, match="custody source"):
        derive_seed_commitment_set(
            PUBLIC_GENERATION_SEED,
            source_boundary=SeedSourceBoundary.CUSTODY_ONLY,
        )

    public_set = derive_seed_commitment_set(
        PUBLIC_GENERATION_SEED,
        source_boundary=SeedSourceBoundary.TRACKED_PUBLIC,
    )
    payload = public_set.model_dump(mode="python")
    first = dict(payload["commitments"][0])
    first["source_boundary"] = SeedSourceBoundary.CUSTODY_ONLY
    payload["commitments"] = (first, *payload["commitments"][1:])
    with pytest.raises(ValidationError, match="one source boundary"):
        type(public_set).model_validate(payload)

    custody_set = derive_seed_commitment_set(
        custody_source,
        source_boundary=SeedSourceBoundary.CUSTODY_ONLY,
    )
    partially_relabelled = custody_set.model_dump(mode="python")
    public_commitment = public_set.commitments[0].commitment_digest
    first_custody = dict(partially_relabelled["commitments"][0])
    first_custody["commitment_digest"] = public_commitment
    partially_relabelled["commitments"] = (
        first_custody,
        *partially_relabelled["commitments"][1:],
    )
    with pytest.raises(ValidationError, match="disjoint"):
        type(custody_set).model_validate(partially_relabelled)

    rotated_public = public_set.model_dump(mode="python")
    public_digests = tuple(item["commitment_digest"] for item in rotated_public["commitments"])
    rotated_commitments = []
    for index, commitment in enumerate(rotated_public["commitments"]):
        rotated = dict(commitment)
        rotated["source_boundary"] = SeedSourceBoundary.CUSTODY_ONLY
        rotated["commitment_digest"] = public_digests[(index + 1) % len(public_digests)]
        rotated_commitments.append(rotated)
    rotated_public["commitments"] = tuple(rotated_commitments)
    rotated_public["commitment_set_digest"] = seed_commitment_set_digest(rotated_public)
    with pytest.raises(ValidationError, match="disjoint"):
        type(public_set).model_validate(rotated_public)


def test_balanced_allocation_is_deterministic_and_exact_per_slot() -> None:
    allocation_seed = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.ALLOCATION)
    allocations = allocate_balanced_outcomes(
        allocation_seed,
        ScenarioFamily.FORGOTTEN_REQUIREMENT,
        _lineages(),
    )

    assert allocations == allocate_balanced_outcomes(
        allocation_seed,
        ScenarioFamily.FORGOTTEN_REQUIREMENT,
        tuple(reversed(_lineages())),
    )
    assert (
        validate_balanced_allocations(
            allocation_seed,
            ScenarioFamily.FORGOTTEN_REQUIREMENT,
            _lineages(),
            allocations,
        )
        == allocations
    )
    assert len(allocations) == LINEAGES_PER_FAMILY
    assert tuple(item.allocation_rank for item in allocations) == tuple(range(LINEAGES_PER_FAMILY))
    assert allocations[0].allocation_order_digest == (
        GENERATION_CONTRACT.allocation_golden_first_order_digest
    )
    assert allocations[0].candidate_packet_digest == (
        GENERATION_CONTRACT.allocation_golden_first_packet_digest
    )
    assert tuple(item.lineage_registry_key[-2:] for item in allocations) == (
        "00",
        "14",
        "08",
        "03",
        "29",
        "21",
        "20",
        "15",
        "13",
        "05",
        "12",
        "01",
        "23",
        "04",
        "16",
        "19",
        "26",
        "18",
        "28",
        "02",
        "27",
        "07",
        "09",
        "22",
        "24",
        "17",
        "06",
        "25",
        "11",
        "10",
    )
    assert all(
        tuple(item.outcomes_by_slot.count(outcome) for outcome in ScenarioOutcome) == (2, 1, 1, 1)
        for item in allocations
    )
    for slot in range(GENERATOR_SLOTS_PER_LINEAGE):
        counts = tuple(
            sum(item.outcomes_by_slot[slot] is outcome for item in allocations)
            for outcome in ScenarioOutcome
        )
        assert counts == (12, 6, 6, 6)


def test_id_and_allocation_seed_mutations_are_independent() -> None:
    lineages = _lineages()
    allocation_seed = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.ALLOCATION)
    changed_allocation_seed = derive_seed(
        PUBLIC_GENERATION_SEED,
        SeedPurpose.ALLOCATION,
        b"metamorphic-change",
    )
    original_allocations = allocate_balanced_outcomes(
        allocation_seed,
        ScenarioFamily.FORGOTTEN_REQUIREMENT,
        lineages,
    )
    changed_allocations = allocate_balanced_outcomes(
        changed_allocation_seed,
        ScenarioFamily.FORGOTTEN_REQUIREMENT,
        lineages,
    )
    original_by_lineage = {
        item.lineage_registry_key: item.outcomes_by_slot for item in original_allocations
    }
    changed_by_lineage = {
        item.lineage_registry_key: item.outcomes_by_slot for item in changed_allocations
    }
    assert original_by_lineage != changed_by_lineage

    skeletons = tuple(_digest(f"pre-allocation-skeleton-{index}") for index in range(5))
    id_seed = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.ID)
    changed_id_seed = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.ID, b"metamorphic-change")
    ids = tuple(derive_scenario_id(id_seed, skeleton_digest=item) for item in skeletons)
    changed_ids = tuple(
        derive_scenario_id(changed_id_seed, skeleton_digest=item) for item in skeletons
    )
    assert ids != changed_ids
    assert ids == tuple(derive_scenario_id(id_seed, skeleton_digest=item) for item in skeletons)
    assert original_allocations == allocate_balanced_outcomes(
        allocation_seed,
        ScenarioFamily.FORGOTTEN_REQUIREMENT,
        lineages,
    )


def test_scenario_id_is_bound_only_to_the_id_leaf_and_preallocation_skeleton() -> None:
    id_seed = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.ID)
    skeleton = _digest("pre-allocation-skeleton")
    scenario_id = derive_scenario_id(id_seed, skeleton_digest=skeleton)
    assert scenario_id == "ba0e82458a95dbed4b72778710e228334ada5a9cdfe2b9993ab3918af1792faf"
    assert derive_scenario_id(id_seed, skeleton_digest=skeleton) == scenario_id
    assert derive_scenario_id(id_seed, skeleton_digest=_digest("other-skeleton")) != scenario_id

    with pytest.raises(ValueError, match="skeleton digest"):
        derive_scenario_id(id_seed, skeleton_digest="0" * 63)
    with pytest.raises(TypeError):
        derive_scenario_id(  # type: ignore[call-arg]
            id_seed,
            skeleton_digest=skeleton,
            outcome=ScenarioOutcome.HELPFUL,
        )


@pytest.mark.parametrize(
    "lineages",
    [
        _lineages()[:-1],
        (*_lineages()[:-1], _lineages()[0]),
        (*_lineages()[:-1], ("different-key", _lineages()[0][1])),
        (*_lineages()[:-1], ("different-key", "0" * 63)),
        (*_lineages()[:-1], ("different-key", "G" * 64)),
        (*_lineages()[:-1], ("", _digest("valid"))),
        (*_lineages()[:-1], ("malformed",)),
    ],
)
def test_allocation_rejects_incomplete_duplicate_or_malformed_lineages(
    lineages: object,
) -> None:
    with pytest.raises(ValueError):
        allocate_balanced_outcomes(
            derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.ALLOCATION),
            ScenarioFamily.FORGOTTEN_REQUIREMENT,
            lineages,  # type: ignore[arg-type]
        )


def test_allocation_model_and_api_fail_closed() -> None:
    allocation = allocate_balanced_outcomes(
        derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.ALLOCATION),
        ScenarioFamily.FORGOTTEN_REQUIREMENT,
        _lineages(),
    )[0]
    payload = allocation.model_dump(mode="python")
    payload["outcomes_by_slot"] = (ScenarioOutcome.HELPFUL,) * 5
    with pytest.raises(ValidationError, match="two-one-one-one"):
        LineageAllocation.model_validate(payload)

    wrong_rotation = allocation.model_dump(mode="python")
    wrong_rotation["allocation_rank"] = (allocation.allocation_rank + 1) % LINEAGES_PER_FAMILY
    with pytest.raises(ValidationError, match="rank rotation"):
        LineageAllocation.model_validate(wrong_rotation)

    allocations = allocate_balanced_outcomes(
        derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.ALLOCATION),
        ScenarioFamily.FORGOTTEN_REQUIREMENT,
        _lineages(),
    )
    with pytest.raises(ValueError, match="committed seed and registry"):
        validate_balanced_allocations(
            derive_seed(
                PUBLIC_GENERATION_SEED,
                SeedPurpose.ALLOCATION,
                b"wrong-seed",
            ),
            ScenarioFamily.FORGOTTEN_REQUIREMENT,
            _lineages(),
            allocations,
        )

    with pytest.raises(ValueError, match="seed"):
        allocate_balanced_outcomes(
            b"too-short",
            ScenarioFamily.FORGOTTEN_REQUIREMENT,
            _lineages(),
        )
    with pytest.raises(ValueError, match="family"):
        allocate_balanced_outcomes(
            derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.ALLOCATION),
            "forgotten_requirement",  # type: ignore[arg-type]
            _lineages(),
        )

    public_leaf_commitment = seed_commitment(
        derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.PUBLIC),
        SeedPurpose.PUBLIC,
    )
    assert len(public_leaf_commitment) == 64
    assert public_leaf_commitment != GENERATION_CONTRACT.public_source_digest


def test_allocation_container_boundaries_accept_mappings_and_reject_malformed_sequences() -> None:
    allocation_seed = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.ALLOCATION)
    family = ScenarioFamily.FORGOTTEN_REQUIREMENT
    expected = allocate_balanced_outcomes(allocation_seed, family, _lineages())

    assert allocate_balanced_outcomes(allocation_seed, family, dict(_lineages())) == expected

    malformed_mapping = dict(_lineages())
    malformed_mapping["lineage-00"] = None  # type: ignore[assignment]
    for malformed in (
        object(),
        malformed_mapping,
        [list(item) for item in _lineages()],
    ):
        with pytest.raises(ValueError, match="reviewed lineage"):
            allocate_balanced_outcomes(
                allocation_seed,
                family,
                malformed,  # type: ignore[arg-type]
            )

    with pytest.raises(ValueError, match="lineage allocations are invalid"):
        validate_balanced_allocations(
            allocation_seed,
            family,
            _lineages(),
            "not-allocations",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="lineage allocations are invalid"):
        validate_balanced_allocations(
            allocation_seed,
            family,
            _lineages(),
            (*expected[:-1], object()),  # type: ignore[arg-type]
        )
