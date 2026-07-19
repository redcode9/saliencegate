from __future__ import annotations

from collections.abc import Sequence
from decimal import ROUND_CEILING, ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from hashlib import sha256
from math import comb

import pytest
from pydantic import ValidationError

from saliencegate.benchmarks.state_decay_v2.authority import AdmissibleTreatmentBinding
from saliencegate.benchmarks.state_decay_v2.config import (
    BOOTSTRAP_INDEX_DOMAIN,
    GENERATION_CONTRACT,
    PUBLIC_GENERATION_SEED,
    TEMPLATE_REGISTRY_UNIQUE_FIELDS,
    SeedPurpose,
    SeedSourceBoundary,
    derive_seed,
    seed_commitment,
    u64be,
)
from saliencegate.benchmarks.state_decay_v2.manifest import ValidationAudit
from saliencegate.benchmarks.state_decay_v2.protocol import (
    BOOTSTRAP_GOLDEN_SOURCE_DOMAIN,
    BOOTSTRAP_GOLDEN_VECTORS_DOMAIN,
    FINITE_SAMPLE_PROTOCOL,
    LEAKAGE_PROTOCOL,
    LINEAGE_REVIEW_PROTOCOL,
    NUISANCE_FEATURE_INVENTORY,
    NUISANCE_VECTOR_WIDTH,
    TREATMENT_COVERAGE_PROTOCOL,
    AssuranceAxis,
    AssuranceReason,
    BootstrapMetricFamily,
    ExactRatio,
    FeaturePadding,
    FiniteSampleProtocol,
    IdentifierOccurrenceSource,
    LeakageEstimand,
    LeakageNullFixture,
    LeakageProtocol,
    LineageReviewProtocol,
    LineageReviewRecord,
    MetricStatus,
    NuisanceFeatureBlock,
    NuisanceFeatureInventory,
    ReviewBoundary,
    ReviewDecision,
    ShortcutBaseline,
    TreatmentBindingField,
    TreatmentCoverageProtocol,
    derive_independent_lineage_seed,
    independent_lineage_seed_commitment,
    lineage_review_record_digest,
    validate_lineage_review_registry,
    validation_protocol_bindings,
    validation_protocol_digests,
)
from saliencegate.benchmarks.state_decay_v2.schema import (
    BenchmarkSplit,
    ScenarioFamily,
    ScenarioOutcome,
)
from saliencegate.domain import canonical_json, length_prefixed_sha256


def _digest(value: str) -> str:
    return sha256(value.encode("ascii")).hexdigest()


def test_protocol_digests_are_canonical_distinct_and_golden() -> None:
    expected = {
        "nuisance": "8c4819451365264ed29ab5fb843f12a6f7e26d3f65d7aaa87547ec6df5a3c2f3",
        "leakage": "79d14debc79b5f37221f552f32f9337da3ac7d4da9587de3d4c59f9571efcebf",
        "review": "afa9e9c7904c855e0b94aa2ae4b6a8eff660916c66988d07db1570bfa5decc8d",
        "treatment": "ef7a0087c1a36890c6890a7bfff60431e37083991f295d00c5fb49d4baf40447",
        "finite": "a61a0858f1fed0ef09210742912ee5d873134152233e73590c82a6079a979809",
    }
    assert NUISANCE_FEATURE_INVENTORY.inventory_digest == expected["nuisance"]
    assert LEAKAGE_PROTOCOL.protocol_digest == expected["leakage"]
    assert LINEAGE_REVIEW_PROTOCOL.protocol_digest == expected["review"]
    assert TREATMENT_COVERAGE_PROTOCOL.protocol_digest == expected["treatment"]
    assert FINITE_SAMPLE_PROTOCOL.protocol_digest == expected["finite"]
    assert len(set(expected.values())) == len(expected)

    for value in (
        NUISANCE_FEATURE_INVENTORY,
        LEAKAGE_PROTOCOL,
        LINEAGE_REVIEW_PROTOCOL,
        TREATMENT_COVERAGE_PROTOCOL,
        FINITE_SAMPLE_PROTOCOL,
    ):
        assert type(value).model_validate_json(canonical_json(value)) == value


def test_nuisance_inventory_is_contiguous_complete_and_exactly_1020_wide() -> None:
    inventory = NUISANCE_FEATURE_INVENTORY
    assert inventory.vector_width == NUISANCE_VECTOR_WIDTH == 1_020
    assert tuple(item.block for item in inventory.blocks) == tuple(NuisanceFeatureBlock)
    assert inventory.validity_order == (
        "active",
        "invalidated",
        "expired",
        "superseded",
    )
    assert inventory.identifier_sources == tuple(IdentifierOccurrenceSource)
    assert inventory.empty_summary == (0, -1, -1, -1, 0, 0)

    cursor = 0
    for block in inventory.blocks:
        assert block.offset == cursor
        cursor += block.width
    assert cursor == NUISANCE_VECTOR_WIDTH
    assert inventory.blocks[8].padding is FeaturePadding.ZERO
    assert all(item.padding is FeaturePadding.MINUS_ONE for item in inventory.blocks[-4:])

    payload = inventory.model_dump(mode="python")
    first = dict(payload["blocks"][0])
    first["offset"] = 1
    payload["blocks"] = (first, *payload["blocks"][1:])
    with pytest.raises(ValidationError, match="frozen layout"):
        NuisanceFeatureInventory.model_validate(payload)


@pytest.mark.parametrize(
    ("numerator", "denominator"),
    [(2, 4), (0, 2), (1, 0), (True, 1), (1.0, 2)],
)
def test_exact_ratio_rejects_noncanonical_or_ill_typed_values(
    numerator: object,
    denominator: object,
) -> None:
    with pytest.raises(ValidationError):
        ExactRatio.model_validate(
            {
                "numerator": numerator,
                "denominator": denominator,
            }
        )


def test_leakage_protocol_freezes_folds_permutations_subsets_and_estimators() -> None:
    protocol = LEAKAGE_PROTOCOL
    assert tuple(
        (
            item.scope_id,
            item.splits,
            item.analysis_seed_source,
            item.row_count,
            item.family_count,
        )
        for item in protocol.audit_scopes
    ) == (
        (
            "public-train-development",
            (BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT),
            SeedSourceBoundary.TRACKED_PUBLIC,
            900,
            6,
        ),
        (
            "custody-primary-pre-lock",
            (BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT, BenchmarkSplit.LOCKED),
            SeedSourceBoundary.CUSTODY_ONLY,
            1_200,
            8,
        ),
        (
            "custody-diagnostic-pre-lock",
            (BenchmarkSplit.DIAGNOSTIC,),
            SeedSourceBoundary.CUSTODY_ONLY,
            1_200,
            8,
        ),
    )
    assert all(item.tuning_allowed is False for item in protocol.audit_scopes)
    assert all(
        item.analysis_seed_purpose is SeedPurpose.PERMUTATION
        and item.analysis_seed_consumers
        == ("label_permutation", "logistic_random_state", "tree_random_state")
        for item in protocol.audit_scopes
    )
    assert protocol.row_scope == "all-pivots-in-audit-scope"
    assert protocol.baseline_order == tuple(ShortcutBaseline)
    assert protocol.fold.fold_count == 5
    assert protocol.fold.lineages_per_fold == 6
    assert protocol.fold.seed_free is True
    assert protocol.fold.hash_primitive == "length-prefixed-sha256/v1"
    assert protocol.fold.coordinate_encoding == ("utf8", "utf8")
    assert protocol.permutation.draw_count == 10_000
    assert protocol.permutation.coordinate_order == (
        "replicate_u64be",
        "family",
        "lineage_registry_key",
        "step_u64be",
        "attempt_u64be",
    )
    assert protocol.permutation.label_multiset == (
        ScenarioOutcome.HELPFUL,
        ScenarioOutcome.HELPFUL,
        ScenarioOutcome.HARMFUL,
        ScenarioOutcome.REDUNDANT,
        ScenarioOutcome.UNRESOLVED,
    )
    assert protocol.permutation.fisher_yates_steps == (4, 3, 2, 1)
    assert protocol.permutation.rejection_attempt_rule == "increment-attempt"
    assert protocol.adapter.dependencies == ("numpy==2.3.5", "scikit-learn==1.9.0")
    assert tuple(item.name for item in protocol.adapter.logistic_arguments) == (
        "l1_ratio",
        "C",
        "dual",
        "tol",
        "fit_intercept",
        "intercept_scaling",
        "class_weight",
        "solver",
        "max_iter",
        "verbose",
        "warm_start",
        "random_state",
    )
    assert protocol.adapter.logistic_omitted_arguments == ("penalty", "n_jobs")
    assert protocol.adapter.random_state_seed_purpose is SeedPurpose.PERMUTATION
    assert protocol.adapter.random_state_hash_primitive == "length-prefixed-sha256/v1"
    assert protocol.adapter.random_state_integer_coordinate_encoding == "unsigned-64-bit-big-endian"
    assert protocol.adapter.random_state_digest_slice == "first-four-bytes"
    assert protocol.adapter.random_state_golden_digest == (
        "051609cc385ac68e861577a2da29705117e5058a1910616b74f8e8273540f72c"
    )
    assert protocol.adapter.random_state_golden_u32 == 85_330_380
    assert protocol.adapter.logistic_pipeline == ("standard_scaler", "logistic_regression")
    assert protocol.adapter.tree_pipeline == ("decision_tree_unscaled",)
    assert protocol.adapter.preprocessing_fit_scope == "training-fold-only"
    assert protocol.adapter.estimator_refit_scope == "each-fold-and-each-permutation"
    assert protocol.adapter.logistic_feature_blocks == tuple(NuisanceFeatureBlock)
    assert protocol.adapter.tree_feature_blocks == tuple(NuisanceFeatureBlock)
    assert tuple(
        (item.outcome, item.integer_label) for item in protocol.adapter.label_integer_encoding
    ) == tuple((outcome, index) for index, outcome in enumerate(ScenarioOutcome))
    assert protocol.single_field_lookup_universe == tuple(NuisanceFeatureBlock)
    assert protocol.single_field_lookup_key_unit == "one-flat-signed-int64-coordinate/v1"
    assert (
        protocol.single_field_lookup_candidate_order
        == "nuisance-block-order-then-zero-based-coordinate/v1"
    )
    assert protocol.single_field_lookup_candidate_count == 1_020
    assert protocol.single_field_selection == (
        "maximum-preregistered-held-out-statistic-across-candidate-coordinates/v1"
    )
    class_contracts = {item.estimand: item for item in protocol.estimand_classes}
    assert class_contracts[LeakageEstimand.HELPFUL_VS_REST].class_order == (
        "negative",
        "helpful",
    )
    assert class_contracts[LeakageEstimand.RESOLVED_ONLY].included_outcomes == (
        ScenarioOutcome.HELPFUL,
        ScenarioOutcome.HARMFUL,
        ScenarioOutcome.REDUNDANT,
    )
    assert class_contracts[LeakageEstimand.RESOLVED_ONLY].outcome_classes[-1].assigned_class is None
    assert (
        class_contracts[LeakageEstimand.RESOLVED_ONLY].row_inclusion_rule
        == "exclude-unresolved-outcome-label-after-each-permutation"
    )
    assert all(
        item.majority_tie_order == item.class_order and item.lookup_tie_order == item.class_order
        for item in protocol.estimand_classes
    )
    assert protocol.fold_metric_average_allowed is False
    assert protocol.ceiling_scope == "every-baseline-every-governed-estimand"
    assert protocol.permutation_reuse_scope == "same-permutations-all-baselines-and-statistics"

    ceilings = {item.estimand: item for item in protocol.ceilings}
    assert ceilings[LeakageEstimand.HELPFUL_VS_REST].balanced_accuracy_max_ppm == 550_000
    assert ceilings[LeakageEstimand.FOUR_OUTCOME].average_precision_max_ppm == 350_000
    assert ceilings[LeakageEstimand.RESOLVED_ONLY].governed is False
    assert len(protocol.null_goldens) == 6
    assert tuple(item.fixture for item in protocol.null_goldens) == (
        LeakageNullFixture.CONSTANT_EMPIRICAL_PRIOR,
        LeakageNullFixture.CONSTANT_EMPIRICAL_PRIOR,
        LeakageNullFixture.CONSTANT_EMPIRICAL_PRIOR,
        LeakageNullFixture.UNIFORM_CLASS_PROBABILITY,
        LeakageNullFixture.UNIFORM_CLASS_PROBABILITY,
        LeakageNullFixture.UNIFORM_CLASS_PROBABILITY,
    )
    assert all(item.hard_label_rule == "argmax-then-class-order" for item in protocol.null_goldens)


def test_fold_and_permutation_goldens_pin_framing_and_integer_encoding() -> None:
    fold = LEAKAGE_PROTOCOL.fold
    keys = tuple(f"golden-lineage-{index:02d}" for index in range(30))
    ordered = tuple(
        sorted(
            (
                (
                    key,
                    length_prefixed_sha256(
                        ScenarioFamily.FORGOTTEN_REQUIREMENT.value,
                        key,
                        domain=fold.hash_domain,
                    ),
                )
                for key in keys
            ),
            key=lambda item: (item[1], item[0].encode("utf-8")),
        )
    )
    assert ordered[0] == (
        "golden-lineage-00",
        fold.golden_first_key_digest,
    )
    assert tuple(key[-2:] for key, _ in ordered[:6]) == ("00", "10", "23", "12", "22", "27")

    permutation = LEAKAGE_PROTOCOL.permutation
    assert permutation.integer_coordinate_encoding == "unsigned-64-bit-big-endian"
    assert tuple(item.final_outcomes for item in permutation.golden_fixtures) == (
        (
            ScenarioOutcome.HARMFUL,
            ScenarioOutcome.HELPFUL,
            ScenarioOutcome.REDUNDANT,
            ScenarioOutcome.UNRESOLVED,
            ScenarioOutcome.HELPFUL,
        ),
        (
            ScenarioOutcome.REDUNDANT,
            ScenarioOutcome.UNRESOLVED,
            ScenarioOutcome.HELPFUL,
            ScenarioOutcome.HELPFUL,
            ScenarioOutcome.HARMFUL,
        ),
    )


def test_rejection_attempt_contracts_freeze_only_the_declared_bindings() -> None:
    permutation = LEAKAGE_PROTOCOL.permutation
    bootstrap = FINITE_SAMPLE_PROTOCOL.bootstrap

    assert permutation.initial_attempt == 0
    assert permutation.coordinate_order[-1] == "attempt_u64be"
    assert permutation.rejection_attempt_rule == "increment-attempt"
    assert bootstrap.coordinate_order[-1] == "attempt_u64be"
    assert bootstrap.rejection_attempt_rule == "increment-attempt"


def test_leakage_null_goldens_follow_their_declared_probability_geometry() -> None:
    for golden in LEAKAGE_PROTOCOL.null_goldens:
        maximum = max(golden.probability_numerators)
        first_maximum = golden.probability_numerators.index(maximum)
        assert golden.hard_label == golden.class_order[first_maximum]

        if golden.estimand is LeakageEstimand.HELPFUL_VS_REST:
            supports = (180, 120)
            expected_balanced_accuracy = Fraction(1, 2)
            expected_average_precision = Fraction(2, 5)
        elif golden.estimand is LeakageEstimand.FOUR_OUTCOME:
            supports = (120, 60, 60, 60)
            expected_balanced_accuracy = Fraction(1, 4)
            expected_average_precision = (
                sum(
                    (Fraction(count, 300) for count in supports),
                    start=Fraction(0, 1),
                )
                / 4
            )
        else:
            supports = (120, 120)
            expected_balanced_accuracy = Fraction(1, 2)
            expected_average_precision = Fraction(1, 2)

        predicted_class = first_maximum
        recalls = tuple(
            Fraction(1, 1) if index == predicted_class else Fraction(0, 1)
            for index, _ in enumerate(supports)
        )
        assert sum(recalls, start=Fraction(0, 1)) / len(recalls) == (expected_balanced_accuracy)
        assert golden.balanced_accuracy_ppm == (
            expected_balanced_accuracy.numerator
            * 1_000_000
            // expected_balanced_accuracy.denominator
        )
        assert golden.average_precision_ppm == (
            expected_average_precision.numerator
            * 1_000_000
            // expected_average_precision.denominator
        )


def test_leakage_protocol_rejects_parameter_inventory_and_digest_tampering() -> None:
    payload = LEAKAGE_PROTOCOL.model_dump(mode="python")
    adapter = dict(payload["adapter"])
    adapter["dependencies"] = ("numpy==2.3.4", "scikit-learn==1.9.0")
    payload["adapter"] = adapter
    with pytest.raises(ValidationError, match="dependency versions"):
        LeakageProtocol.model_validate(payload)

    payload = LEAKAGE_PROTOCOL.model_dump(mode="python")
    custody_scope = dict(payload["audit_scopes"][1])
    custody_scope["splits"] = (BenchmarkSplit.LOCKED,)
    payload["audit_scopes"] = (
        payload["audit_scopes"][0],
        custody_scope,
        payload["audit_scopes"][2],
    )
    with pytest.raises(ValidationError, match="audit scopes"):
        LeakageProtocol.model_validate(payload)

    payload = LEAKAGE_PROTOCOL.model_dump(mode="python")
    resolved = dict(payload["estimand_classes"][2])
    mappings = list(resolved["outcome_classes"])
    unresolved = dict(mappings[-1])
    unresolved["assigned_class"] = "negative"
    mappings[-1] = unresolved
    resolved["outcome_classes"] = tuple(mappings)
    payload["estimand_classes"] = (*payload["estimand_classes"][:2], resolved)
    with pytest.raises(ValidationError):
        LeakageProtocol.model_validate(payload)

    payload = LEAKAGE_PROTOCOL.model_dump(mode="python")
    payload["protocol_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="digest"):
        LeakageProtocol.model_validate(payload)


def _rejected_review_payload() -> dict[str, object]:
    public_leaf = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.PUBLIC)
    lineage_key = "schema-test-lineage"
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-lineage-review-record/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "split": BenchmarkSplit.TRAIN,
        "family": ScenarioFamily.FORGOTTEN_REQUIREMENT,
        "boundary": ReviewBoundary.PUBLIC,
        "lineage_registry_key": lineage_key,
        "candidate_packet_digest": _digest("candidate-packet"),
        "independent_seed_commitment_digest": independent_lineage_seed_commitment(
            derive_independent_lineage_seed(
                public_leaf,
                split=BenchmarkSplit.TRAIN,
                family=ScenarioFamily.FORGOTTEN_REQUIREMENT,
                lineage_registry_key=lineage_key,
            )
        ),
        "transition_graph_digest": _digest("transition-graph"),
        "evidence_topology_digest": _digest("evidence-topology"),
        "failure_mechanism_id": "schema-test-failure-mechanism",
        "semantic_signature_digest": _digest("semantic-signature"),
        "derivation_parent_keys": ("schema-test-parent",),
        "semantic_rationale": "Schema fixture only; this is not benchmark research data.",
        "reviewer_id": "test-only-nonattestation",
        "review_rationale": "Rejected schema fixture, not a human attestation.",
        "decision": ReviewDecision.REJECTED,
    }
    values["review_digest"] = lineage_review_record_digest(values)
    return values


def test_lineage_review_schema_is_blind_boundary_bound_and_nonfabricating() -> None:
    record = LineageReviewRecord.model_validate(_rejected_review_payload())
    with pytest.raises(ValueError, match="unaccepted"):
        validate_lineage_review_registry(
            (record,),
            expected_splits=(BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT),
        )
    assert LineageReviewRecord.model_validate_json(canonical_json(record)) == record
    assert LINEAGE_REVIEW_PROTOCOL.human_attestation_required is True
    assert LINEAGE_REVIEW_PROTOCOL.generated_attestations_forbidden is True
    assert LINEAGE_REVIEW_PROTOCOL.accepted_parent_count == 0
    assert LINEAGE_REVIEW_PROTOCOL.lineage_seed_hash_primitive == "length-prefixed-sha256/v1"
    assert LINEAGE_REVIEW_PROTOCOL.public_seed_golden_commitment == (
        "bd876f155dd3293526a972bfef073f661022876698199c1babdf86806993615c"
    )
    assert LINEAGE_REVIEW_PROTOCOL.family_local_unique_fields == (TEMPLATE_REGISTRY_UNIQUE_FIELDS)
    assert all(
        field in LineageReviewRecord.model_fields for field in TEMPLATE_REGISTRY_UNIQUE_FIELDS
    )
    for forbidden in LINEAGE_REVIEW_PROTOCOL.forbidden_semantic_fields:
        assert forbidden not in LineageReviewRecord.model_fields

    wrong_boundary = _rejected_review_payload()
    wrong_boundary["boundary"] = ReviewBoundary.CUSTODY
    wrong_boundary["review_digest"] = lineage_review_record_digest(wrong_boundary)
    with pytest.raises(ValidationError, match="boundary"):
        LineageReviewRecord.model_validate(wrong_boundary)

    extra = _rejected_review_payload()
    extra["outcome"] = ScenarioOutcome.HELPFUL
    with pytest.raises(ValidationError):
        LineageReviewRecord.model_validate(extra)


def _accepted_review_record(
    split: BenchmarkSplit,
    family: ScenarioFamily,
    index: int,
    **overrides: object,
) -> LineageReviewRecord:
    lineage_key = f"test-{split.value}-{family.value}-{index:02d}"
    token = f"{split.value}:{family.value}:{index:02d}"
    if split in (BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT):
        source_leaf = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.PUBLIC)
        lineage_commitment = independent_lineage_seed_commitment(
            derive_independent_lineage_seed(
                source_leaf,
                split=split,
                family=family,
                lineage_registry_key=lineage_key,
            )
        )
        boundary = ReviewBoundary.PUBLIC
    else:
        lineage_commitment = _digest(f"custody-lineage-seed:{token}")
        boundary = ReviewBoundary.CUSTODY
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-lineage-review-record/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "split": split,
        "family": family,
        "boundary": boundary,
        "lineage_registry_key": lineage_key,
        "candidate_packet_digest": _digest(f"candidate:{token}"),
        "independent_seed_commitment_digest": lineage_commitment,
        "transition_graph_digest": _digest(f"transition:{token}"),
        "evidence_topology_digest": _digest(f"evidence:{token}"),
        "failure_mechanism_id": f"test-failure-{split.value}-{family.value}-{index:02d}",
        "semantic_signature_digest": _digest(f"semantic:{token}"),
        "derivation_parent_keys": (),
        "semantic_rationale": "Synthetic validator fixture; never benchmark research data.",
        "reviewer_id": "test-only-validator",
        "review_rationale": "Synthetic acceptance exercises registry invariants only.",
        "decision": ReviewDecision.ACCEPTED,
    }
    values.update(overrides)
    values["review_digest"] = lineage_review_record_digest(values)
    return LineageReviewRecord.model_validate(values)


@pytest.mark.parametrize(
    "expected_splits",
    (
        (BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT),
        (BenchmarkSplit.LOCKED, BenchmarkSplit.DIAGNOSTIC),
        tuple(BenchmarkSplit),
    ),
    ids=("public", "custody", "full-suite"),
)
def test_lineage_review_registry_accepts_every_complete_scope(
    expected_splits: tuple[BenchmarkSplit, ...],
) -> None:
    records = tuple(
        _accepted_review_record(split, family, index)
        for split in expected_splits
        for family in next(
            geometry.families for geometry in GENERATION_CONTRACT.splits if geometry.split is split
        )
        for index in range(30)
    )
    assert validate_lineage_review_registry(records, expected_splits=expected_splits) == records


def test_lineage_review_registry_rejects_incomplete_or_unsupported_scopes() -> None:
    records = tuple(
        _accepted_review_record(BenchmarkSplit.TRAIN, family, index)
        for family in next(
            geometry.families
            for geometry in GENERATION_CONTRACT.splits
            if geometry.split is BenchmarkSplit.TRAIN
        )
        for index in range(30)
    )

    with pytest.raises(ValueError, match="expected review splits"):
        validate_lineage_review_registry(
            records[:30],
            expected_splits=(BenchmarkSplit.TRAIN,),
        )
    with pytest.raises(ValueError, match="geometry is incomplete"):
        validate_lineage_review_registry(
            records[:1],
            expected_splits=(BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT),
        )


def test_lineage_review_registry_rejects_duplicate_attestations_and_lineages() -> None:
    record = _accepted_review_record(
        BenchmarkSplit.LOCKED,
        ScenarioFamily.CONFLICTING_EVIDENCE,
        0,
    )
    with pytest.raises(ValueError, match="duplicate attestations"):
        validate_lineage_review_registry(
            (record, record),
            expected_splits=(BenchmarkSplit.LOCKED, BenchmarkSplit.DIAGNOSTIC),
        )

    changed = record.model_dump(mode="python")
    changed["review_rationale"] = "Distinct synthetic review over the same lineage."
    changed["review_digest"] = lineage_review_record_digest(changed)
    second = LineageReviewRecord.model_validate(changed)
    with pytest.raises(ValueError, match="duplicate lineages"):
        validate_lineage_review_registry(
            (record, second),
            expected_splits=(BenchmarkSplit.LOCKED, BenchmarkSplit.DIAGNOSTIC),
        )


@pytest.mark.parametrize("field", TEMPLATE_REGISTRY_UNIQUE_FIELDS)
def test_lineage_review_registry_rejects_every_cross_split_family_local_collision(
    field: str,
) -> None:
    first = _accepted_review_record(
        BenchmarkSplit.LOCKED,
        ScenarioFamily.CONFLICTING_EVIDENCE,
        0,
    )
    second_values = _accepted_review_record(
        BenchmarkSplit.DIAGNOSTIC,
        ScenarioFamily.CONFLICTING_EVIDENCE,
        1,
    ).model_dump(mode="python")
    second_values[field] = getattr(first, field)
    second_values["review_digest"] = lineage_review_record_digest(second_values)
    second = LineageReviewRecord.model_validate(second_values)
    with pytest.raises(ValueError, match="family-local uniqueness"):
        validate_lineage_review_registry(
            (first, second),
            expected_splits=(BenchmarkSplit.LOCKED, BenchmarkSplit.DIAGNOSTIC),
        )


def test_lineage_review_registry_rejects_parent_derived_acceptance_as_unaccepted() -> None:
    record = _accepted_review_record(
        BenchmarkSplit.LOCKED,
        ScenarioFamily.CONFLICTING_EVIDENCE,
        0,
        derivation_parent_keys=("test-parent-lineage",),
    )
    with pytest.raises(ValueError, match="unaccepted record"):
        validate_lineage_review_registry(
            (record,),
            expected_splits=(BenchmarkSplit.LOCKED, BenchmarkSplit.DIAGNOSTIC),
        )


@pytest.mark.parametrize("field", ("semantic_rationale", "review_rationale"))
def test_lineage_review_record_rejects_whitespace_only_rationales(field: str) -> None:
    payload = _rejected_review_payload()
    payload[field] = " \t "
    payload["review_digest"] = lineage_review_record_digest(payload)
    with pytest.raises(ValidationError, match="blank or whitespace-only"):
        LineageReviewRecord.model_validate(payload)


class _ExplodingSequence(Sequence[object]):
    def __getitem__(self, index: int) -> object:
        raise RuntimeError("hostile input detail")

    def __len__(self) -> int:
        raise RuntimeError("hostile input detail")


def test_lineage_review_registry_normalizes_hostile_sequence_failures() -> None:
    with pytest.raises(ValueError, match="lineage review registry is invalid") as captured:
        validate_lineage_review_registry(
            _ExplodingSequence(),  # type: ignore[arg-type]
            expected_splits=(BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT),
        )
    assert "hostile input detail" not in str(captured.value)

    with pytest.raises(ValueError, match="expected review splits are invalid") as captured:
        validate_lineage_review_registry(
            (),
            expected_splits=_ExplodingSequence(),  # type: ignore[arg-type]
        )
    assert "hostile input detail" not in str(captured.value)


def test_review_protocol_rejects_field_inventory_and_digest_tampering() -> None:
    payload = LINEAGE_REVIEW_PROTOCOL.model_dump(mode="python")
    payload["required_record_fields"] = (*payload["required_record_fields"][:-1], "outcome")
    with pytest.raises(ValidationError, match="required fields"):
        LineageReviewProtocol.model_validate(payload)

    payload = LINEAGE_REVIEW_PROTOCOL.model_dump(mode="python")
    payload["protocol_digest"] = "f" * 64
    with pytest.raises(ValidationError, match="digest"):
        LineageReviewProtocol.model_validate(payload)


def test_treatment_protocol_matches_the_typed_authority_and_fails_zero_denominators() -> None:
    authority_fields = tuple(
        field for field in AdmissibleTreatmentBinding.model_fields if field != "schema_version"
    )
    assert tuple(item.value for item in TreatmentBindingField) == authority_fields
    assert TREATMENT_COVERAGE_PROTOCOL.binding_fields == tuple(TreatmentBindingField)
    assert TREATMENT_COVERAGE_PROTOCOL.proposal_coverage_minimum == ExactRatio(
        numerator=99,
        denominator=100,
    )
    assert TREATMENT_COVERAGE_PROTOCOL.delivered_exact_coverage_minimum == ExactRatio(
        numerator=1,
        denominator=1,
    )
    assert TREATMENT_COVERAGE_PROTOCOL.zero_denominator_rule == "not-evaluable-and-invalid"
    assert TREATMENT_COVERAGE_PROTOCOL.outcome_access_during_derivation is False
    assert TREATMENT_COVERAGE_PROTOCOL.fixture_seed_purpose is SeedPurpose.PROPOSAL
    assert TREATMENT_COVERAGE_PROTOCOL.fixture_seed_hash_primitive == ("length-prefixed-sha256/v1")
    assert TREATMENT_COVERAGE_PROTOCOL.fixture_seed_coordinates == ("proposal_leaf", "split")
    assert tuple(item.split for item in TREATMENT_COVERAGE_PROTOCOL.fixture_seed_sources) == tuple(
        BenchmarkSplit
    )
    assert (
        len(
            {
                item.commitment_digest
                for item in TREATMENT_COVERAGE_PROTOCOL.public_fixture_seed_commitments
            }
        )
        == 2
    )
    assert (
        TREATMENT_COVERAGE_PROTOCOL.custody_seed_binding
        == "suite-generation-commitment-required/v1"
    )

    payload = TREATMENT_COVERAGE_PROTOCOL.model_dump(mode="python")
    payload["binding_fields"] = tuple(reversed(payload["binding_fields"]))
    with pytest.raises(ValidationError, match="binding fields"):
        TreatmentCoverageProtocol.model_validate(payload)


def test_metric_protocol_freezes_geometry_weights_bins_statuses_and_integer_gates() -> None:
    metric = FINITE_SAMPLE_PROTOCOL.metric
    assert metric.geometry.rows == 300
    assert metric.geometry.independent_lineages == 60
    assert (metric.geometry.positives, metric.geometry.negatives) == (120, 180)
    assert (
        metric.geometry.helpful,
        metric.geometry.harmful,
        metric.geometry.redundant,
        metric.geometry.unresolved,
    ) == (120, 60, 60, 60)
    assert metric.positive_importance_weight == ExactRatio(numerator=1, denominator=4)
    assert metric.negative_importance_weight == ExactRatio(numerator=3, denominator=2)
    assert metric.score_probability_rule == "opportunity-score-10-over-10-exact-rational/v1"
    assert metric.importance_weights_apply_to == ("brier", "ece")
    assert tuple((item.lower_ppm, item.upper_ppm) for item in metric.ece_bins) == tuple(
        (index * 100_000, (index + 1) * 100_000) for index in range(10)
    )
    assert metric.ece_bins[-1].upper_inclusive is True
    assert all(item.lower_inclusive is True for item in metric.ece_bins)
    assert metric.comparison_conditions == (
        "no_memory",
        "canonical_fixed_step_schedule",
        "event_risk_only",
        "saliencegate_event",
    )
    definitions = {item.metric_id: item for item in metric.metric_definitions}
    assert definitions["proposal_adjudicability"].denominator == "canonical_fixture_invocations"
    assert definitions["evidence_adjudication_coverage"].denominator == "authorized_claims"
    assert definitions["call_reduction"].denominator == "C_schedule"
    assert metric.statuses == tuple(MetricStatus)
    gates = {item.gate_id: item.canonical_integer_inequality for item in metric.governed_gate_rules}
    assert gates["trigger_recall"] == "TP>=90"
    assert gates["trigger_precision"] == "TP>=9*FP"
    assert gates["success_point_margin"] == "S_event-S_schedule>-6"
    assert gates["call_reduction"] == "5*C_event<=2*C_schedule"
    assert gates["token_reduction"] == "2*T_event<=T_schedule"
    assert all(item.zero_denominator_fails for item in metric.governed_gate_rules)


def _clopper_pearson_upper(harmful_count: int) -> Decimal:
    if harmful_count == 30:
        return Decimal(1)
    with localcontext() as context:
        context.prec = 100
        context.rounding = ROUND_HALF_EVEN
        lower = Decimal(0)
        upper = Decimal(1)
        tail = Decimal(1) / Decimal(40)
        for _ in range(256):
            midpoint = (lower + upper) / 2
            complement = Decimal(1) - midpoint
            cdf = sum(
                Decimal(comb(30, count)) * midpoint**count * complement ** (30 - count)
                for count in range(harmful_count + 1)
            )
            if cdf > tail:
                lower = midpoint
            else:
                upper = midpoint
        return +upper


def _combined_upper_scaled(left: Decimal, right: Decimal) -> int:
    with localcontext() as context:
        context.prec = 100
        context.rounding = ROUND_HALF_EVEN
        combined = (left + right) / Decimal(8)
        return int(
            (combined * Decimal(1_000_000_000_000)).to_integral_value(rounding=ROUND_CEILING)
        )


def test_clopper_pearson_goldens_and_pair_composition_are_independent_and_exact() -> None:
    protocol = FINITE_SAMPLE_PROTOCOL.clopper_pearson
    bounds = tuple(_clopper_pearson_upper(count) for count in range(31))
    display_goldens = tuple(
        int((value * Decimal(1_000_000_000_000)).to_integral_value(rounding=ROUND_CEILING))
        for value in bounds
    )
    assert display_goldens == protocol.family_display_goldens_scaled
    assert _combined_upper_scaled(bounds[0], bounds[0]) == 28_925_827_056

    for left in range(31):
        row = tuple(_combined_upper_scaled(bounds[left], bounds[right]) for right in range(31))
        assert row == tuple(sorted(row))
        for right, value in enumerate(row):
            assert value == _combined_upper_scaled(bounds[right], bounds[left])

    for golden in protocol.threshold_adjacent_pair_goldens:
        value = _combined_upper_scaled(
            bounds[golden.left_harmful_count],
            bounds[golden.right_harmful_count],
        )
        assert value == golden.combined_upper_scaled
        assert golden.passes_upper_gate is (value <= protocol.gate_upper_scaled)
    assert protocol.family_display_values_feed_gate is False
    assert protocol.bootstrap_substitution_allowed is False
    assert protocol.pooled_sixty_role == "diagnostic-only"
    assert protocol.pooled_sixty_method == "one-sided-clopper-pearson/v1"
    assert protocol.pooled_sixty_trials == 60
    assert protocol.pooled_sixty_tail == ExactRatio(numerator=1, denominator=20)
    assert protocol.pooled_sixty_composition == "unrounded-upper-divided-by-four"


def _bootstrap_golden_vectors() -> tuple[dict[str, object], ...]:
    protocol = FINITE_SAMPLE_PROTOCOL.bootstrap
    seed = bytes.fromhex(protocol.golden_derived_seed_hex)
    limit = (1 << 256) - ((1 << 256) % 30)
    vectors: list[dict[str, object]] = []
    for metric_family in BootstrapMetricFamily:
        for scenario_family in ScenarioFamily:
            for replicate in protocol.golden_replicates:
                indexes: list[int] = []
                for draw_ordinal in range(30):
                    attempt = 0
                    while True:
                        value = int(
                            length_prefixed_sha256(
                                seed,
                                metric_family.value,
                                u64be(replicate),
                                scenario_family.value,
                                u64be(draw_ordinal),
                                u64be(attempt),
                                domain=BOOTSTRAP_INDEX_DOMAIN,
                            ),
                            16,
                        )
                        if value < limit:
                            indexes.append(value % 30)
                            break
                        attempt += 1
                vectors.append(
                    {
                        "metric_family": metric_family.value,
                        "scenario_family": scenario_family.value,
                        "replicate": replicate,
                        "indexes": tuple(indexes),
                    }
                )
    return tuple(vectors)


def test_bootstrap_golden_vectors_cover_every_metric_stratum_and_replicate() -> None:
    protocol = FINITE_SAMPLE_PROTOCOL.bootstrap
    assert protocol.seed_purpose is SeedPurpose.BOOTSTRAP
    assert protocol.public_seed_commitment_digest == seed_commitment(
        derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.BOOTSTRAP),
        SeedPurpose.BOOTSTRAP,
    )
    assert protocol.custody_seed_binding == "suite-generation-commitment-required/v1"
    assert protocol.golden_source_hex == bytes(range(32)).hex()
    assert protocol.golden_derived_seed_hex == length_prefixed_sha256(
        bytes(range(32)),
        domain=BOOTSTRAP_GOLDEN_SOURCE_DOMAIN,
    )
    assert (
        protocol.golden_derived_seed_hex
        != derive_seed(
            PUBLIC_GENERATION_SEED,
            SeedPurpose.BOOTSTRAP,
        ).hex()
    )
    vectors = _bootstrap_golden_vectors()
    assert len(vectors) == 6 * 8 * 3 == protocol.golden_vector_count
    for value in vectors:
        indexes = value["indexes"]
        assert isinstance(indexes, tuple)
        assert len(indexes) == 30
    assert (
        length_prefixed_sha256(
            canonical_json(vectors),
            domain=BOOTSTRAP_GOLDEN_VECTORS_DOMAIN,
        )
        == protocol.golden_vectors_digest
    )
    assert vectors[0]["indexes"] == protocol.vector_checkpoints[0].indexes
    assert vectors[-1]["indexes"] == protocol.vector_checkpoints[-1].indexes


def test_assurance_is_numeric_only_for_identified_models() -> None:
    assurance = FINITE_SAMPLE_PROTOCOL.assurance
    assert assurance.trigger_acceptance_event == "TP>=90-and-TP>=9*FP"
    assert assurance.harmful_acceptance_event == "combined-clopper-pearson-upper<=5/100"
    assert assurance.harmful_correlation == "not_applicable"
    assert tuple(item.axis for item in assurance.not_evaluable_axes) == (
        AssuranceAxis.SUCCESS_DELTA,
        AssuranceAxis.CALL_REDUCTION,
        AssuranceAxis.TOKEN_REDUCTION,
    )
    assert tuple(item.reason for item in assurance.not_evaluable_axes) == (
        AssuranceReason.PAIRED_BASELINE_DISCORDANCE_UNIDENTIFIED,
        AssuranceReason.CALL_OPPORTUNITY_DISTRIBUTION_UNIDENTIFIED,
        AssuranceReason.TOKENS_ARE_NOT_BERNOULLI_TRIALS,
    )
    assert assurance.assurance_role == "descriptive-only"
    assert assurance.assurance_changes_feasibility is False


def test_finite_sample_protocol_rejects_nested_and_root_tampering() -> None:
    payload = FINITE_SAMPLE_PROTOCOL.model_dump(mode="python")
    bootstrap = dict(payload["bootstrap"])
    bootstrap["final_replicates"] = 99_999
    payload["bootstrap"] = bootstrap
    with pytest.raises(ValidationError):
        FiniteSampleProtocol.model_validate(payload)

    payload = FINITE_SAMPLE_PROTOCOL.model_dump(mode="python")
    payload["protocol_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="digest"):
        FiniteSampleProtocol.model_validate(payload)


def test_validation_audits_bind_one_noninterchangeable_protocol_each() -> None:
    bindings = validation_protocol_bindings()
    digests = validation_protocol_digests()
    assert tuple(item.audit for item in bindings) == tuple(ValidationAudit)
    assert tuple(digests) == tuple(ValidationAudit)
    assert len(set(digests.values())) == len(ValidationAudit)
    assert digests[ValidationAudit.GEOMETRY] == GENERATION_CONTRACT.contract_digest
    assert digests[ValidationAudit.LINEAGE_REVIEW] == LINEAGE_REVIEW_PROTOCOL.protocol_digest
    assert (
        digests[ValidationAudit.TREATMENT_COVERAGE] == TREATMENT_COVERAGE_PROTOCOL.protocol_digest
    )
    assert digests[ValidationAudit.LEAKAGE] == LEAKAGE_PROTOCOL.protocol_digest
    assert digests[ValidationAudit.FINITE_SAMPLE] == FINITE_SAMPLE_PROTOCOL.protocol_digest

    digests[ValidationAudit.GEOMETRY] = "0" * 64
    assert (
        validation_protocol_digests()[ValidationAudit.GEOMETRY]
        == GENERATION_CONTRACT.contract_digest
    )
