from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from hashlib import sha256
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from saliencegate.benchmarks.state_decay_v2.config import (
    GENERATION_CONTRACT,
    PUBLIC_GENERATION_SEED,
    SeedPurpose,
    derive_seed,
)
from saliencegate.benchmarks.state_decay_v2.protocol import (
    FINITE_SAMPLE_PROTOCOL,
    LEAKAGE_PROTOCOL,
    LINEAGE_REVIEW_PROTOCOL,
    NUISANCE_FEATURE_INVENTORY,
    TREATMENT_COVERAGE_PROTOCOL,
    AssuranceProtocol,
    AssuranceReason,
    BootstrapProtocol,
    BootstrapVectorCheckpoint,
    ClopperPearsonProtocol,
    ECEBin,
    FiniteSampleProtocol,
    FoldProtocol,
    LeakageClassLabel,
    LeakageEstimandClasses,
    LeakageNullGolden,
    LeakageProtocol,
    LineageReviewProtocol,
    LineageReviewRecord,
    MetricProtocol,
    NuisanceFeatureBlock,
    NuisanceFeatureInventory,
    PermutationProtocol,
    ProbabilityConversionProtocol,
    PublicProposalSeedCommitment,
    ResearchAdapterProtocol,
    ReviewBoundary,
    ReviewDecision,
    TreatmentCoverageProtocol,
    derive_independent_lineage_seed,
    independent_lineage_seed_commitment,
    lineage_review_record_digest,
    validate_lineage_review_registry,
)
from saliencegate.benchmarks.state_decay_v2.schema import (
    BenchmarkSplit,
    ScenarioFamily,
)

Payload = dict[str, Any]
Path = tuple[str | int, ...]
Mutation = Callable[[Payload], None]


def _value_at(value: Any, path: Path) -> Any:
    for part in path:
        value = value[part]
    return value


def _replaced(value: Any, path: Path, replacement: Any) -> Any:
    if not path:
        return deepcopy(replacement)
    head, *tail = path
    if isinstance(head, str):
        updated = dict(value)
        updated[head] = _replaced(updated[head], tuple(tail), replacement)
        return updated
    updated_items = list(value)
    updated_items[head] = _replaced(updated_items[head], tuple(tail), replacement)
    return tuple(updated_items) if isinstance(value, tuple) else updated_items


def _set(path: Path, replacement: Any) -> Mutation:
    def mutate(payload: Payload) -> None:
        updated = _replaced(payload, path, replacement)
        payload.clear()
        payload.update(updated)

    return mutate


def _swap(path: Path, left: int = 0, right: int = 1) -> Mutation:
    def mutate(payload: Payload) -> None:
        items = list(_value_at(payload, path))
        items[left], items[right] = items[right], items[left]
        updated = _replaced(payload, path, tuple(items))
        payload.clear()
        payload.update(updated)

    return mutate


def _copy_value(source: Path, destination: Path) -> Mutation:
    def mutate(payload: Payload) -> None:
        updated = _replaced(payload, destination, _value_at(payload, source))
        payload.clear()
        payload.update(updated)

    return mutate


def _assert_model_rejects(
    model_type: type[BaseModel],
    canonical: BaseModel,
    mutate: Mutation,
    expected_error: str,
) -> None:
    payload = canonical.model_dump(mode="python")
    mutate(payload)
    with pytest.raises(ValidationError, match=expected_error):
        model_type.model_validate(payload)


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        pytest.param(_swap(("blocks",)), "blocks are not complete", id="block-order"),
        pytest.param(_swap(("validity_order",)), "validity order", id="validity-order"),
        pytest.param(
            _swap(("temporal_summary_order",)),
            "temporal summary order",
            id="summary-order",
        ),
        pytest.param(
            _set(("empty_summary",), (0, -1, -1, 0, -1, 0)),
            "empty summary sentinel",
            id="empty-summary",
        ),
        pytest.param(
            _swap(("optional_field_order",)),
            "optional field order",
            id="optional-fields",
        ),
        pytest.param(
            _swap(("count_source_rules",)),
            "count sources",
            id="count-sources",
        ),
        pytest.param(
            _swap(("identifier_sources",)),
            "identifier source order",
            id="identifier-sources",
        ),
        pytest.param(
            _swap(("summary_source_rules",)),
            "summary sources",
            id="summary-sources",
        ),
        pytest.param(
            _swap(("text_source_rules",)),
            "text sources",
            id="text-sources",
        ),
        pytest.param(
            _set(("inventory_digest",), "0" * 64),
            "inventory digest",
            id="inventory-digest",
        ),
    ),
)
def test_nuisance_inventory_rejects_independent_contract_tampering(
    mutate: Mutation,
    expected_error: str,
) -> None:
    _assert_model_rejects(
        NuisanceFeatureInventory,
        NUISANCE_FEATURE_INVENTORY,
        mutate,
        expected_error,
    )


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        pytest.param(
            _swap(("coordinate_order",)),
            "coordinate order",
            id="coordinate-order",
        ),
        pytest.param(
            _set(("coordinate_encoding",), ("utf8", "lowercase-hex-utf8")),
            "coordinate encoding",
            id="coordinate-encoding",
        ),
    ),
)
def test_fold_contract_rejects_coordinate_tampering(
    mutate: Mutation,
    expected_error: str,
) -> None:
    _assert_model_rejects(FoldProtocol, LEAKAGE_PROTOCOL.fold, mutate, expected_error)


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        pytest.param(
            _swap(("label_multiset",), 1, 2),
            "label multiset",
            id="label-multiset",
        ),
        pytest.param(
            _swap(("fisher_yates_steps",)),
            "Fisher-Yates steps",
            id="fisher-yates-steps",
        ),
        pytest.param(
            _swap(("coordinate_order",)),
            "coordinate order",
            id="coordinate-order",
        ),
        pytest.param(
            _swap(("interval_quantiles",)),
            "interval quantiles",
            id="interval-quantiles",
        ),
        pytest.param(
            _set(("public_seed_commitment_digest",), "0" * 64),
            "public permutation seed commitment",
            id="public-seed-commitment",
        ),
        pytest.param(
            _swap(("golden_fixtures", 0, "draws")),
            "golden steps",
            id="golden-step-order",
        ),
        pytest.param(
            _set(("golden_fixtures", 0, "draws", 0, "selected_index"), 2),
            "golden draw",
            id="golden-selected-index",
        ),
        pytest.param(
            _swap(("golden_fixtures", 0, "final_outcomes")),
            "golden result",
            id="golden-result",
        ),
    ),
)
def test_permutation_contract_rejects_framing_and_golden_tampering(
    mutate: Mutation,
    expected_error: str,
) -> None:
    _assert_model_rejects(
        PermutationProtocol,
        LEAKAGE_PROTOCOL.permutation,
        mutate,
        expected_error,
    )


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        pytest.param(
            _swap(("input_requirements",)),
            "input requirements",
            id="input-requirements",
        ),
        pytest.param(
            _swap(("conversion_order",)),
            "conversion order",
            id="conversion-order",
        ),
    ),
)
def test_probability_conversion_rejects_order_tampering(
    mutate: Mutation,
    expected_error: str,
) -> None:
    _assert_model_rejects(
        ProbabilityConversionProtocol,
        LEAKAGE_PROTOCOL.adapter.probability_conversion,
        mutate,
        expected_error,
    )


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        pytest.param(_swap(("scaler_arguments",)), "scaler arguments", id="scaler"),
        pytest.param(
            _swap(("logistic_arguments",)),
            "logistic arguments",
            id="logistic",
        ),
        pytest.param(
            _swap(("logistic_omitted_arguments",)),
            "omitted arguments",
            id="omitted",
        ),
        pytest.param(_swap(("tree_arguments",)), "tree arguments", id="tree"),
        pytest.param(
            _swap(("random_state_coordinates",)),
            "random-state coordinates",
            id="random-state-coordinates",
        ),
        pytest.param(_swap(("class_order",)), "class order", id="class-order"),
        pytest.param(
            _swap(("label_integer_encoding",)),
            "label encoding",
            id="label-encoding",
        ),
        pytest.param(
            _swap(("logistic_pipeline",)),
            "logistic pipeline",
            id="logistic-pipeline",
        ),
        pytest.param(
            _set(("tree_pipeline",), ("standard_scaler",)),
            "tree pipeline",
            id="tree-pipeline",
        ),
        pytest.param(
            _swap(("logistic_feature_blocks",)),
            "logistic feature blocks",
            id="logistic-features",
        ),
        pytest.param(
            _swap(("tree_feature_blocks",)),
            "tree feature blocks",
            id="tree-features",
        ),
        pytest.param(
            _swap(("thread_environment_variables",)),
            "thread environment",
            id="thread-environment",
        ),
    ),
)
def test_research_adapter_rejects_independent_preregistration_tampering(
    mutate: Mutation,
    expected_error: str,
) -> None:
    _assert_model_rejects(
        ResearchAdapterProtocol,
        LEAKAGE_PROTOCOL.adapter,
        mutate,
        expected_error,
    )


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        pytest.param(
            _set(("probability_numerators",), (3, 1, 1)),
            "widths differ",
            id="probability-width",
        ),
        pytest.param(
            _set(("class_order",), ("negative", "negative")),
            "contains duplicates",
            id="duplicate-class",
        ),
        pytest.param(
            _set(("probability_numerators",), (-1, 6)),
            "numerator is invalid",
            id="negative-probability",
        ),
        pytest.param(
            _set(("probability_numerators",), (3, 1)),
            "do not sum",
            id="probability-sum",
        ),
        pytest.param(
            _set(("hard_label",), "unresolved"),
            "outside its class order",
            id="hard-label",
        ),
    ),
)
def test_null_golden_rejects_invalid_probability_geometry(
    mutate: Mutation,
    expected_error: str,
) -> None:
    _assert_model_rejects(
        LeakageNullGolden,
        LEAKAGE_PROTOCOL.null_goldens[0],
        mutate,
        expected_error,
    )


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        pytest.param(
            _swap(("outcome_classes",)),
            "class map is not complete and ordered",
            id="outcome-map-order",
        ),
        pytest.param(
            _swap(("included_outcomes",)),
            "included outcomes diverge",
            id="included-outcomes",
        ),
        pytest.param(
            _set(
                ("class_order",),
                (LeakageClassLabel.NEGATIVE, LeakageClassLabel.HARMFUL),
            ),
            "class order does not cover",
            id="class-coverage",
        ),
        pytest.param(
            _swap(("class_integer_encoding",)),
            "integer encoding",
            id="integer-encoding",
        ),
        pytest.param(
            _swap(("majority_tie_order",)),
            "tie order",
            id="majority-tie-order",
        ),
        pytest.param(
            _set(("positive_class",), LeakageClassLabel.HARMFUL),
            "positive class is outside",
            id="positive-class",
        ),
    ),
)
def test_estimand_class_contract_rejects_mapping_and_tie_tampering(
    mutate: Mutation,
    expected_error: str,
) -> None:
    _assert_model_rejects(
        LeakageEstimandClasses,
        LEAKAGE_PROTOCOL.estimand_classes[0],
        mutate,
        expected_error,
    )


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        pytest.param(
            _set(("nuisance_inventory_digest",), "0" * 64),
            "does not bind the nuisance inventory",
            id="nuisance-digest",
        ),
        pytest.param(_swap(("audit_scopes",)), "audit scopes", id="scope-order"),
        pytest.param(
            _swap(("audit_scopes", 0, "analysis_seed_consumers")),
            "seed consumers",
            id="scope-seed-consumers",
        ),
        pytest.param(_swap(("baseline_order",)), "baseline order", id="baseline-order"),
        pytest.param(
            _swap(("feature_subsets",)),
            "feature subsets",
            id="feature-subset-order",
        ),
        pytest.param(
            _set(
                ("feature_subsets", 0, "blocks"),
                (NuisanceFeatureBlock.PIVOT_SEQUENCE,),
            ),
            "feature block membership",
            id="feature-membership",
        ),
        pytest.param(
            _swap(("single_field_lookup_universe",)),
            "lookup universe",
            id="lookup-universe",
        ),
        pytest.param(
            _swap(("estimand_classes",)),
            "class contracts are not complete",
            id="estimand-order",
        ),
        pytest.param(
            _set(("estimand_classes", 0, "row_inclusion_rule"), "include-no-labels"),
            "class contracts are not canonical",
            id="estimand-rule",
        ),
        pytest.param(_swap(("ceilings",)), "ceilings are not complete", id="ceiling-order"),
        pytest.param(
            _copy_value(
                ("ceilings", 1, "balanced_accuracy_max_ppm"),
                ("ceilings", 0, "balanced_accuracy_max_ppm"),
            ),
            "ceiling values",
            id="ceiling-value",
        ),
        pytest.param(
            _swap(("null_goldens",)),
            "null goldens",
            id="null-golden-order",
        ),
        pytest.param(
            _set(("protocol_digest",), "0" * 64),
            "protocol digest",
            id="protocol-digest",
        ),
    ),
)
def test_leakage_protocol_rejects_cross_component_tampering(
    mutate: Mutation,
    expected_error: str,
) -> None:
    _assert_model_rejects(LeakageProtocol, LEAKAGE_PROTOCOL, mutate, expected_error)


def _digest(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _review_payload(
    *,
    split: BenchmarkSplit = BenchmarkSplit.TRAIN,
    decision: ReviewDecision = ReviewDecision.ACCEPTED,
    parents: tuple[str, ...] = (),
) -> Payload:
    family = next(
        geometry.families[0] for geometry in GENERATION_CONTRACT.splits if geometry.split is split
    )
    lineage_key = f"validation-edge-{split.value}-00"
    if split in (BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT):
        lineage_seed = derive_independent_lineage_seed(
            derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.PUBLIC),
            split=split,
            family=family,
            lineage_registry_key=lineage_key,
        )
        seed_commitment = independent_lineage_seed_commitment(lineage_seed)
        boundary = ReviewBoundary.PUBLIC
    else:
        seed_commitment = _digest(f"custody-seed:{split.value}")
        boundary = ReviewBoundary.CUSTODY
    payload: Payload = {
        "schema_version": "state-decay-v2-lineage-review-record/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "split": split,
        "family": family,
        "boundary": boundary,
        "lineage_registry_key": lineage_key,
        "candidate_packet_digest": _digest(f"candidate:{split.value}"),
        "independent_seed_commitment_digest": seed_commitment,
        "transition_graph_digest": _digest(f"transition:{split.value}"),
        "evidence_topology_digest": _digest(f"evidence:{split.value}"),
        "failure_mechanism_id": f"validation-edge-failure-{split.value}",
        "semantic_signature_digest": _digest(f"semantic:{split.value}"),
        "derivation_parent_keys": parents,
        "semantic_rationale": "Synthetic validator fixture; not benchmark research data.",
        "reviewer_id": "validation-edge-reviewer",
        "review_rationale": "Synthetic record used only to test fail-closed invariants.",
        "decision": decision,
    }
    payload["review_digest"] = lineage_review_record_digest(payload)
    return payload


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        pytest.param(
            _set(("family",), ScenarioFamily.CONFLICTING_EVIDENCE),
            "family does not belong",
            id="split-family",
        ),
        pytest.param(
            _set(("independent_seed_commitment_digest",), "0" * 64),
            "public lineage seed commitment",
            id="public-seed-commitment",
        ),
        pytest.param(
            _set(("derivation_parent_keys",), ("same-parent", "same-parent")),
            "parents must be unique",
            id="duplicate-parents",
        ),
        pytest.param(
            _set(("review_digest",), "0" * 64),
            "review record digest",
            id="record-digest",
        ),
    ),
)
def test_lineage_review_record_rejects_role_and_attestation_tampering(
    mutate: Mutation,
    expected_error: str,
) -> None:
    payload = _review_payload()
    mutate(payload)
    if payload["review_digest"] != "0" * 64:
        payload["review_digest"] = lineage_review_record_digest(payload)
    with pytest.raises(ValidationError, match=expected_error):
        LineageReviewRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("source_leaf", "split", "family", "lineage_key", "expected_error"),
    (
        pytest.param(
            b"short",
            BenchmarkSplit.TRAIN,
            ScenarioFamily.FORGOTTEN_REQUIREMENT,
            "lineage",
            "source leaf",
            id="source-width",
        ),
        pytest.param(
            b"0" * 32,
            "train",
            ScenarioFamily.FORGOTTEN_REQUIREMENT,
            "lineage",
            "coordinates",
            id="split-type",
        ),
        pytest.param(
            b"0" * 32,
            BenchmarkSplit.TRAIN,
            "forgotten_requirement",
            "lineage",
            "coordinates",
            id="family-type",
        ),
        pytest.param(
            b"0" * 32,
            BenchmarkSplit.TRAIN,
            ScenarioFamily.FORGOTTEN_REQUIREMENT,
            "",
            "registry key",
            id="empty-key",
        ),
    ),
)
def test_lineage_seed_derivation_rejects_malformed_coordinates(
    source_leaf: Any,
    split: Any,
    family: Any,
    lineage_key: Any,
    expected_error: str,
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        derive_independent_lineage_seed(
            source_leaf,
            split=split,
            family=family,
            lineage_registry_key=lineage_key,
        )


def test_lineage_seed_commitment_rejects_non_seed_bytes() -> None:
    with pytest.raises(ValueError, match="independent lineage seed"):
        independent_lineage_seed_commitment(b"short")


@pytest.mark.parametrize(
    ("records", "expected_splits", "expected_error"),
    (
        pytest.param((), "train", "expected review splits", id="split-string"),
        pytest.param((), (BenchmarkSplit.TRAIN,), "expected review splits", id="partial-scope"),
        pytest.param("records", tuple(BenchmarkSplit), "registry is invalid", id="record-string"),
        pytest.param((), tuple(BenchmarkSplit), "registry is invalid", id="empty-registry"),
        pytest.param((object(),), tuple(BenchmarkSplit), "registry is invalid", id="wrong-record"),
    ),
)
def test_review_registry_rejects_invalid_container_boundaries(
    records: Any,
    expected_splits: Any,
    expected_error: str,
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        validate_lineage_review_registry(records, expected_splits=expected_splits)


def test_review_registry_rejects_unexpected_and_unaccepted_records() -> None:
    locked = LineageReviewRecord.model_validate(_review_payload(split=BenchmarkSplit.LOCKED))
    with pytest.raises(ValueError, match="unexpected split"):
        validate_lineage_review_registry(
            (locked,),
            expected_splits=(BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT),
        )

    rejected = LineageReviewRecord.model_validate(
        _review_payload(split=BenchmarkSplit.LOCKED, decision=ReviewDecision.REJECTED)
    )
    with pytest.raises(ValueError, match="unaccepted record"):
        validate_lineage_review_registry(
            (rejected,),
            expected_splits=(BenchmarkSplit.LOCKED, BenchmarkSplit.DIAGNOSTIC),
        )


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        pytest.param(
            _swap(("lineage_seed_coordinates",)),
            "seed coordinates",
            id="seed-coordinates",
        ),
        pytest.param(
            _swap(("lineage_seed_sources",)),
            "sources are not complete",
            id="source-order",
        ),
        pytest.param(
            _set(("lineage_seed_sources", 2, "source_purpose"), SeedPurpose.PUBLIC),
            "source purposes",
            id="source-purpose",
        ),
        pytest.param(
            _swap(("boundary_rules",)),
            "boundary rules are not complete",
            id="boundary-order",
        ),
        pytest.param(
            _set(("boundary_rules", 0, "boundary"), ReviewBoundary.CUSTODY),
            "review boundaries",
            id="boundary-value",
        ),
        pytest.param(
            _swap(("family_local_unique_fields",)),
            "uniqueness fields",
            id="unique-fields",
        ),
        pytest.param(
            _swap(("required_record_fields",)),
            "required fields",
            id="required-fields",
        ),
        pytest.param(
            _swap(("forbidden_semantic_fields",)),
            "forbidden fields",
            id="forbidden-fields",
        ),
        pytest.param(
            _swap(("accepted_registry_split_scopes",)),
            "registry scopes",
            id="registry-scopes",
        ),
        pytest.param(
            _set(("protocol_digest",), "0" * 64),
            "protocol digest",
            id="protocol-digest",
        ),
    ),
)
def test_lineage_review_protocol_rejects_contract_tampering(
    mutate: Mutation,
    expected_error: str,
) -> None:
    _assert_model_rejects(
        LineageReviewProtocol,
        LINEAGE_REVIEW_PROTOCOL,
        mutate,
        expected_error,
    )


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        pytest.param(
            _set(("split",), BenchmarkSplit.LOCKED),
            "commitment split",
            id="private-split",
        ),
        pytest.param(
            _set(("commitment_digest",), "0" * 64),
            "fixture seed commitment",
            id="commitment",
        ),
    ),
)
def test_public_proposal_seed_commitment_rejects_substitution(
    mutate: Mutation,
    expected_error: str,
) -> None:
    _assert_model_rejects(
        PublicProposalSeedCommitment,
        TREATMENT_COVERAGE_PROTOCOL.public_fixture_seed_commitments[0],
        mutate,
        expected_error,
    )


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        pytest.param(_swap(("binding_fields",)), "binding fields", id="binding-fields"),
        pytest.param(_swap(("fixture_inputs",)), "fixture inputs", id="fixture-inputs"),
        pytest.param(
            _swap(("fixture_seed_coordinates",)),
            "seed coordinates",
            id="seed-coordinates",
        ),
        pytest.param(
            _swap(("fixture_seed_sources",)),
            "seed sources are not complete",
            id="source-order",
        ),
        pytest.param(
            _set(("fixture_seed_sources", 0, "boundary"), ReviewBoundary.CUSTODY),
            "seed boundaries",
            id="source-boundary",
        ),
        pytest.param(
            _swap(("public_fixture_seed_commitments",)),
            "commitments are not canonical",
            id="public-commitment-order",
        ),
        pytest.param(
            _swap(("coverage_partition",)),
            "coverage outcomes",
            id="outcome-partition",
        ),
        pytest.param(
            _copy_value(("delivered_exact_coverage_minimum",), ("proposal_coverage_minimum",)),
            "proposal coverage threshold",
            id="proposal-threshold",
        ),
        pytest.param(
            _copy_value(("proposal_coverage_minimum",), ("delivered_exact_coverage_minimum",)),
            "delivery coverage threshold",
            id="delivery-threshold",
        ),
        pytest.param(_swap(("failures",)), "failure inventory", id="failure-inventory"),
        pytest.param(
            _set(("protocol_digest",), "0" * 64),
            "protocol digest",
            id="protocol-digest",
        ),
    ),
)
def test_treatment_protocol_rejects_coverage_and_seed_tampering(
    mutate: Mutation,
    expected_error: str,
) -> None:
    _assert_model_rejects(
        TreatmentCoverageProtocol,
        TREATMENT_COVERAGE_PROTOCOL,
        mutate,
        expected_error,
    )


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        pytest.param(_set(("lower_ppm",), 0), "lower bound", id="lower-bound"),
        pytest.param(_set(("upper_ppm",), 100_000), "upper bound", id="upper-bound"),
        pytest.param(
            _set(("upper_inclusive",), True),
            "only the last ECE bin",
            id="upper-inclusion",
        ),
    ),
)
def test_ece_bin_rejects_noncanonical_intervals(
    mutate: Mutation,
    expected_error: str,
) -> None:
    _assert_model_rejects(
        ECEBin,
        FINITE_SAMPLE_PROTOCOL.metric.ece_bins[1],
        mutate,
        expected_error,
    )


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        pytest.param(
            _copy_value(("positive_importance_weight",), ("average_precision_gate",)),
            "average precision gate",
            id="average-precision-gate",
        ),
        pytest.param(
            _copy_value(("negative_importance_weight",), ("positive_importance_weight",)),
            "positive importance weight",
            id="positive-weight",
        ),
        pytest.param(
            _copy_value(("positive_importance_weight",), ("negative_importance_weight",)),
            "negative importance weight",
            id="negative-weight",
        ),
        pytest.param(
            _swap(("importance_weights_apply_to",)),
            "metric scope",
            id="weight-scope",
        ),
        pytest.param(_swap(("ece_bins",)), "ECE bins", id="ece-bin-order"),
        pytest.param(
            _swap(("operational_prevalences",)),
            "prevalence projections",
            id="prevalence-order",
        ),
        pytest.param(
            _swap(("projection_formulas",)),
            "projection formulas",
            id="projection-formulas",
        ),
        pytest.param(
            _swap(("rate_record_operands",)),
            "rate record operands",
            id="rate-operands",
        ),
        pytest.param(_swap(("statuses",)), "metric statuses", id="statuses"),
        pytest.param(
            _swap(("metric_definitions",)),
            "metric definitions",
            id="metric-definitions",
        ),
        pytest.param(
            _swap(("comparison_conditions",)),
            "comparison conditions",
            id="comparison-conditions",
        ),
        pytest.param(
            _swap(("primary_inclusion_rules",)),
            "inclusion rules",
            id="inclusion-rules",
        ),
        pytest.param(
            _swap(("call_token_accounting_rules",)),
            "accounting rules",
            id="accounting-rules",
        ),
        pytest.param(
            _swap(("governed_gate_rules",)),
            "gate rules are not complete",
            id="gate-order",
        ),
        pytest.param(
            _copy_value(
                ("governed_gate_rules", 1, "canonical_integer_inequality"),
                ("governed_gate_rules", 0, "canonical_integer_inequality"),
            ),
            "gate inequalities",
            id="gate-inequality",
        ),
    ),
)
def test_metric_protocol_rejects_geometry_and_gate_tampering(
    mutate: Mutation,
    expected_error: str,
) -> None:
    _assert_model_rejects(
        MetricProtocol,
        FINITE_SAMPLE_PROTOCOL.metric,
        mutate,
        expected_error,
    )


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        pytest.param(
            _copy_value(("pooled_sixty_tail",), ("family_tail",)),
            "family tail",
            id="family-tail",
        ),
        pytest.param(
            _copy_value(("family_tail",), ("simultaneous_coverage_minimum",)),
            "simultaneous coverage",
            id="simultaneous-coverage",
        ),
        pytest.param(
            _copy_value(("family_tail",), ("pooled_sixty_tail",)),
            "pooled Clopper-Pearson tail",
            id="pooled-tail",
        ),
        pytest.param(_swap(("initial_bracket",)), "bracket", id="initial-bracket"),
        pytest.param(
            _swap(("family_display_goldens_scaled",)),
            "family goldens",
            id="family-goldens",
        ),
        pytest.param(
            _swap(("threshold_adjacent_pair_goldens",)),
            "adjacent pairs",
            id="adjacent-pair-order",
        ),
        pytest.param(
            _copy_value(
                ("threshold_adjacent_pair_goldens", 1, "combined_upper_scaled"),
                ("threshold_adjacent_pair_goldens", 0, "combined_upper_scaled"),
            ),
            "adjacent pair values",
            id="adjacent-pair-value",
        ),
    ),
)
def test_clopper_pearson_rejects_exactness_tampering(
    mutate: Mutation,
    expected_error: str,
) -> None:
    _assert_model_rejects(
        ClopperPearsonProtocol,
        FINITE_SAMPLE_PROTOCOL.clopper_pearson,
        mutate,
        expected_error,
    )


def test_bootstrap_checkpoint_rejects_out_of_stratum_indexes() -> None:
    payload = FINITE_SAMPLE_PROTOCOL.bootstrap.vector_checkpoints[0].model_dump(mode="python")
    payload["indexes"] = (-1, *payload["indexes"][1:])
    with pytest.raises(ValidationError, match="invalid lineage index"):
        BootstrapVectorCheckpoint.model_validate(payload)


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        pytest.param(
            _set(("public_seed_commitment_digest",), "0" * 64),
            "public bootstrap seed commitment",
            id="public-seed-commitment",
        ),
        pytest.param(
            _swap(("coordinate_order",)),
            "coordinate order",
            id="coordinate-order",
        ),
        pytest.param(
            _swap(("metric_family_order",)),
            "metric family order",
            id="metric-order",
        ),
        pytest.param(_swap(("stratum_order",)), "stratum order", id="stratum-order"),
        pytest.param(_swap(("percentiles",)), "percentiles", id="percentiles"),
        pytest.param(
            _set(("golden_source_hex",), "0" * 64),
            "golden source",
            id="golden-source",
        ),
        pytest.param(
            _set(("golden_derived_seed_hex",), "0" * 64),
            "golden derived seed",
            id="golden-derived-seed",
        ),
        pytest.param(
            _swap(("golden_replicates",)),
            "golden replicates",
            id="golden-replicates",
        ),
        pytest.param(
            _swap(("vector_checkpoints",)),
            "vector checkpoints are not canonical",
            id="checkpoint-order",
        ),
        pytest.param(
            _swap(("vector_checkpoints", 0, "indexes")),
            "checkpoint indexes",
            id="checkpoint-indexes",
        ),
    ),
)
def test_bootstrap_protocol_rejects_seed_and_vector_tampering(
    mutate: Mutation,
    expected_error: str,
) -> None:
    _assert_model_rejects(
        BootstrapProtocol,
        FINITE_SAMPLE_PROTOCOL.bootstrap,
        mutate,
        expected_error,
    )


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        pytest.param(_swap(("tpr_grid",)), "TPR assurance grid", id="tpr-grid"),
        pytest.param(_swap(("fpr_grid",)), "FPR assurance grid", id="fpr-grid"),
        pytest.param(
            _swap(("harmful_incidence_grid",)),
            "harmful assurance grid",
            id="harmful-grid",
        ),
        pytest.param(
            _swap(("success_delta_grid",)),
            "success assurance grid",
            id="success-grid",
        ),
        pytest.param(
            _swap(("call_reduction_grid",)),
            "call assurance grid",
            id="call-grid",
        ),
        pytest.param(
            _swap(("token_reduction_grid",)),
            "token assurance grid",
            id="token-grid",
        ),
        pytest.param(_swap(("rho_grid",)), "rho assurance grid", id="rho-grid"),
        pytest.param(
            _swap(("not_evaluable_axes",)),
            "not-evaluable assurance axes",
            id="not-evaluable-axis-order",
        ),
        pytest.param(
            _set(
                ("not_evaluable_axes", 0, "reason"),
                AssuranceReason.CALL_OPPORTUNITY_DISTRIBUTION_UNIDENTIFIED,
            ),
            "not-evaluable assurance reasons",
            id="not-evaluable-reason",
        ),
        pytest.param(
            _swap(("observed_integer_regions",)),
            "observed integer regions",
            id="integer-regions",
        ),
    ),
)
def test_assurance_protocol_rejects_grid_and_identifiability_tampering(
    mutate: Mutation,
    expected_error: str,
) -> None:
    _assert_model_rejects(
        AssuranceProtocol,
        FINITE_SAMPLE_PROTOCOL.assurance,
        mutate,
        expected_error,
    )


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        pytest.param(
            _swap(("validity_outcomes",)),
            "validity outcomes",
            id="validity-order",
        ),
        pytest.param(
            _swap(("not_feasible_conditions",)),
            "failure conditions",
            id="failure-conditions",
        ),
        pytest.param(
            _set(("protocol_digest",), "0" * 64),
            "protocol digest",
            id="protocol-digest",
        ),
    ),
)
def test_finite_sample_protocol_rejects_root_contract_tampering(
    mutate: Mutation,
    expected_error: str,
) -> None:
    _assert_model_rejects(
        FiniteSampleProtocol,
        FINITE_SAMPLE_PROTOCOL,
        mutate,
        expected_error,
    )
