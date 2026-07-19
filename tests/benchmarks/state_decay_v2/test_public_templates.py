from __future__ import annotations

from dataclasses import replace
from itertools import product

import pytest
from pydantic import ValidationError

from saliencegate.benchmarks.state_decay_v2.config import CounterbalanceAxis
from saliencegate.benchmarks.state_decay_v2.preallocation import preview_to_scenario_id
from saliencegate.benchmarks.state_decay_v2.public_catalog import PUBLIC_LINEAGE_DEFINITIONS
from saliencegate.benchmarks.state_decay_v2.public_contract import (
    PublicCausalExposure,
    PublicGeneratorOperation,
    PublicLineageRegistry,
    candidate_packet_digest,
    candidate_registry_digest,
    derive_causal_outcome,
    execute_public_transition_graph,
    skeleton_preview_digest,
)
from saliencegate.benchmarks.state_decay_v2.schema import (
    BenchmarkSplit,
    ScenarioFamily,
    ScenarioOutcome,
)
from saliencegate.benchmarks.state_decay_v2.templates import (
    PUBLIC_GENERATOR_ALGORITHM,
    PUBLIC_GENERATOR_CONFIGURATION,
    PUBLIC_LINEAGE_REGISTRY,
    PUBLIC_PROFILE_CATALOG,
    build_public_lineage_candidate,
    build_public_lineage_registry,
    validate_public_lineage_registry_materialization,
)
from saliencegate.domain import SignalType, ValidityState, canonical_json

_PUBLIC_FAMILIES = (
    ScenarioFamily.FORGOTTEN_REQUIREMENT,
    ScenarioFamily.FAILED_PRIOR_ATTEMPT,
    ScenarioFamily.NEGLECTED_SUBGOAL,
    ScenarioFamily.STALE_MEMORY,
    ScenarioFamily.STABLE_ENVIRONMENT_FACT,
    ScenarioFamily.RETAINED_DIAGNOSIS,
)


def test_global_public_generator_contracts_are_complete_and_canonical() -> None:
    configuration = PUBLIC_GENERATOR_CONFIGURATION
    catalog = PUBLIC_PROFILE_CATALOG
    algorithm = PUBLIC_GENERATOR_ALGORITHM

    assert configuration.visible_splits == (
        BenchmarkSplit.TRAIN,
        BenchmarkSplit.DEVELOPMENT,
    )
    assert configuration.visible_families == _PUBLIC_FAMILIES
    assert (configuration.candidate_count, configuration.preview_count) == (180, 900)
    assert catalog.generator_configuration_digest == configuration.configuration_digest
    assert catalog.counterbalance_axes == tuple(CounterbalanceAxis)
    assert tuple(profile.generator_slot for profile in catalog.slot_profiles) == tuple(range(5))
    assert algorithm.generator_configuration_digest == configuration.configuration_digest
    assert algorithm.profile_catalog_digest == catalog.catalog_digest
    assert tuple(step.operation for step in algorithm.steps) == tuple(PublicGeneratorOperation)


def test_profile_catalog_is_complete_closed_and_signal_covering() -> None:
    profiles = PUBLIC_PROFILE_CATALOG.slot_profiles

    assert tuple(profile.structure.trajectory_event_count for profile in profiles) == (
        3,
        4,
        5,
        6,
        7,
    )
    assert tuple(profile.structure.candidate_memory_count for profile in profiles) == (
        1,
        2,
        3,
        4,
        2,
    )
    assert tuple(profile.counterbalance.memory_validity for profile in profiles) == (
        ValidityState.ACTIVE,
        ValidityState.ACTIVE,
        ValidityState.INVALIDATED,
        ValidityState.SUPERSEDED,
        ValidityState.ACTIVE,
    )
    masks = tuple(
        tuple(signal.signal_type for signal in profile.signals.expected_signals)
        for profile in profiles
    )
    assert len(set(masks)) == 5
    assert {signal for mask in masks for signal in mask} == set(SignalType)
    assert {
        signal.strength_ppm for profile in profiles for signal in profile.signals.expected_signals
    } == {500_000, 625_000, 750_000, 1_000_000}


def test_registry_has_exact_public_geometry_and_empty_parentage() -> None:
    registry = PUBLIC_LINEAGE_REGISTRY

    assert len(registry.candidates) == 180
    assert tuple(candidate.family for candidate in registry.candidates[::30]) == _PUBLIC_FAMILIES
    assert tuple(candidate.split for candidate in registry.candidates[::30]) == (
        BenchmarkSplit.TRAIN,
        BenchmarkSplit.TRAIN,
        BenchmarkSplit.TRAIN,
        BenchmarkSplit.TRAIN,
        BenchmarkSplit.DEVELOPMENT,
        BenchmarkSplit.DEVELOPMENT,
    )
    assert all(candidate.derivation_parent_keys == () for candidate in registry.candidates)
    assert all(len(candidate.previews) == 5 for candidate in registry.candidates)
    assert all(
        tuple(preview.generator_slot for preview in candidate.previews) == tuple(range(5))
        for candidate in registry.candidates
    )


def test_semantic_catalog_has_thirty_explicit_loci_per_visible_family() -> None:
    assert len(PUBLIC_LINEAGE_DEFINITIONS) == 180
    for family in _PUBLIC_FAMILIES:
        definitions = tuple(item for item in PUBLIC_LINEAGE_DEFINITIONS if item.family is family)
        assert tuple(item.lineage_index for item in definitions) == tuple(range(30))
        assert len({item.lineage_registry_key for item in definitions}) == 30
        assert len({item.locus_id for item in definitions}) == 30
        assert all(item.semantic_rationale for item in definitions)


def test_all_public_identifiers_and_rendered_policies_are_globally_unique() -> None:
    candidates = PUBLIC_LINEAGE_REGISTRY.candidates
    previews = tuple(preview for candidate in candidates for preview in candidate.previews)
    scenario_ids = tuple(preview_to_scenario_id(preview) for preview in previews)
    rendered_policy_digests = tuple(
        delta.rendered_policy_digest
        for preview in previews
        for delta in preview.rendered_causal_deltas
    )

    assert len(previews) == len({preview.preview_digest for preview in previews}) == 900
    assert len(scenario_ids) == len(set(scenario_ids)) == 900
    assert len(rendered_policy_digests) == len(set(rendered_policy_digests)) == 3_600

    for family in _PUBLIC_FAMILIES:
        family_candidates = tuple(item for item in candidates if item.family is family)
        columns = (
            tuple(item.candidate_packet_digest for item in family_candidates),
            tuple(item.lineage_registry_key for item in family_candidates),
            tuple(item.independent_seed_commitment_digest for item in family_candidates),
            tuple(item.transition_graph.transition_graph_digest for item in family_candidates),
            tuple(item.evidence_topology.evidence_topology_digest for item in family_candidates),
            tuple(item.failure_mechanism.failure_mechanism_id for item in family_candidates),
            tuple(item.semantic_signature.semantic_signature_digest for item in family_candidates),
        )
        assert all(len(column) == len(set(column)) == 30 for column in columns)


def test_every_preview_projects_only_its_canonical_slot_profile() -> None:
    catalog_profiles = PUBLIC_PROFILE_CATALOG.slot_profiles

    for candidate in PUBLIC_LINEAGE_REGISTRY.candidates:
        for preview in candidate.previews:
            profile = catalog_profiles[preview.generator_slot]
            assert preview.slot_profile == profile
            assert preview.allowed_parameter_values == profile.parameters.allowed_values
            assert len(preview.task_skeleton.trajectory) == profile.structure.trajectory_event_count
            assert (
                len(preview.task_skeleton.candidate_memories)
                == profile.structure.candidate_memory_count
            )
            assert preview.task_skeleton.adapter == candidate.task_template.adapter


def test_slot_materialization_preserves_exact_prefix_coordinates_and_text_rules() -> None:
    candidate = PUBLIC_LINEAGE_REGISTRY.candidates[0]

    for preview in candidate.previews:
        profile = preview.slot_profile
        skeleton = preview.task_skeleton
        integers = profile.integers
        assert tuple(event.event_id for event in skeleton.trajectory) == tuple(
            event.event_id
            for event in candidate.task_template.event_pool[
                : profile.structure.trajectory_event_count
            ]
        )
        assert tuple(event.sequence for event in skeleton.trajectory) == tuple(
            integers.sequence_start + index * integers.sequence_stride
            for index in range(profile.structure.trajectory_event_count)
        )
        assert tuple(event.action_step for event in skeleton.trajectory) == tuple(
            integers.action_step_start + index * integers.action_step_stride
            for index in range(profile.structure.trajectory_event_count)
        )
        assert tuple(action.action_id for action in skeleton.allowed_actions) == tuple(
            candidate.task_template.action_pool[index].action_id
            for index in profile.counterbalance.allowed_action_order
        )
        expected_clause = (
            " [parameters:"
            + ",".join(
                f"{item.parameter_id}={item.value}" for item in profile.parameters.allowed_values
            )
            + "]"
        )
        assert skeleton.pivot.statement == (
            candidate.task_template.pivot.statement
            + expected_clause
            + " " * profile.text_lengths.pivot_padding_spaces
        )
        assert (
            sum(len(memory.evidence_refs) for memory in skeleton.candidate_memories)
            == profile.evidence.evidence_reference_count
        )


def test_every_candidate_has_a_complete_executable_two_factor_machine() -> None:
    for candidate in PUBLIC_LINEAGE_REGISTRY.candidates:
        graph = candidate.transition_graph
        factor_ids = tuple(factor.factor_id for factor in graph.factors)
        assert len(factor_ids) == 2
        assert {
            tuple(value.value for value in delta.factor_values) for delta in candidate.causal_deltas
        } == set(product((False, True), repeat=2))
        assert {
            derive_causal_outcome(graph, delta.factor_values) for delta in candidate.causal_deltas
        } == set(ScenarioOutcome)
        for delta in candidate.causal_deltas:
            for exposure in PublicCausalExposure:
                execution = execute_public_transition_graph(graph, exposure, delta.factor_values)
                assert execution.terminal is not None
                assert len(execution.action_fingerprint_ids) <= 64


def test_causal_policy_text_encodes_the_two_reviewable_factors() -> None:
    for candidate in PUBLIC_LINEAGE_REGISTRY.candidates:
        source = candidate.task_template.event_pool[0].statement
        assert source.startswith("Guidance=latent baseline=reserved for ")
        for delta in candidate.causal_deltas:
            left, right = (item.value for item in delta.factor_values)
            replacement = delta.semantic_replacements[0].replacement
            assert ("Guidance=potent" in replacement) is left
            assert ("Guidance=static" in replacement) is (not left)
            assert ("baseline=recovers" in replacement) is right
            assert ("baseline=declines" in replacement) is (not right)
            assert len(source.encode("utf-8")) == len(replacement.encode("utf-8"))
            assert tuple(index for index, item in enumerate(source) if item == " ") == tuple(
                index for index, item in enumerate(replacement) if item == " "
            )


def test_public_catalog_and_registry_are_byte_deterministic() -> None:
    rebuilt = build_public_lineage_registry(
        PUBLIC_GENERATOR_CONFIGURATION,
        PUBLIC_PROFILE_CATALOG,
        PUBLIC_GENERATOR_ALGORITHM,
        PUBLIC_LINEAGE_DEFINITIONS,
    )

    assert canonical_json(rebuilt) == canonical_json(PUBLIC_LINEAGE_REGISTRY)
    assert rebuilt.registry_digest == PUBLIC_LINEAGE_REGISTRY.registry_digest


def test_registry_rejects_a_reminted_preview_with_a_noncanonical_slot_profile() -> None:
    payload = PUBLIC_LINEAGE_REGISTRY.model_dump(mode="python")
    candidate = payload["candidates"][0]
    preview = candidate["previews"][0]
    preview["slot_profile"]["counterbalance"]["profile_id"] = "reminted-profile"
    preview["preview_digest"] = skeleton_preview_digest(preview)
    candidate["candidate_packet_digest"] = candidate_packet_digest(candidate)
    payload["registry_digest"] = candidate_registry_digest(payload)

    with pytest.raises(ValidationError, match="canonical profile catalog"):
        PublicLineageRegistry.model_validate(payload)


def test_registry_materialization_validator_rebuilds_and_byte_compares_every_candidate() -> None:
    payload = PUBLIC_LINEAGE_REGISTRY.model_dump(mode="python")
    candidate = payload["candidates"][0]
    candidate["semantic_rationale"] = (
        "A locally valid but noncanonical rationale must not survive rematerialization."
    )
    candidate["candidate_packet_digest"] = candidate_packet_digest(candidate)
    payload["registry_digest"] = candidate_registry_digest(payload)
    reminted = PublicLineageRegistry.model_validate(payload)

    with pytest.raises(ValueError, match="does not match its canonical materialization"):
        validate_public_lineage_registry_materialization(
            reminted,
            PUBLIC_GENERATOR_CONFIGURATION,
            PUBLIC_PROFILE_CATALOG,
            PUBLIC_GENERATOR_ALGORITHM,
            PUBLIC_LINEAGE_DEFINITIONS,
        )


def test_registry_builder_rejects_disagreeing_global_inputs() -> None:
    mismatched_algorithm = PUBLIC_GENERATOR_ALGORITHM.model_copy(
        update={"profile_catalog_digest": "0" * 64}
    )

    with pytest.raises(ValueError, match="global inputs do not agree"):
        build_public_lineage_registry(
            PUBLIC_GENERATOR_CONFIGURATION,
            PUBLIC_PROFILE_CATALOG,
            mismatched_algorithm,
            PUBLIC_LINEAGE_DEFINITIONS,
        )


def test_registry_materialization_validator_rejects_non_registry_exact_type() -> None:
    assert (
        validate_public_lineage_registry_materialization(
            PUBLIC_LINEAGE_REGISTRY,
            PUBLIC_GENERATOR_CONFIGURATION,
            PUBLIC_PROFILE_CATALOG,
            PUBLIC_GENERATOR_ALGORITHM,
            PUBLIC_LINEAGE_DEFINITIONS,
        )
        is PUBLIC_LINEAGE_REGISTRY
    )
    with pytest.raises(ValueError, match="invalid type"):
        validate_public_lineage_registry_materialization(
            object(),  # type: ignore[arg-type]
            PUBLIC_GENERATOR_CONFIGURATION,
            PUBLIC_PROFILE_CATALOG,
            PUBLIC_GENERATOR_ALGORITHM,
            PUBLIC_LINEAGE_DEFINITIONS,
        )


def test_candidate_semantic_change_propagates_without_changing_global_algorithm() -> None:
    original = PUBLIC_LINEAGE_REGISTRY.candidates[0]
    definition = replace(
        PUBLIC_LINEAGE_DEFINITIONS[0],
        retained_subject="the retained boundary",
    )

    changed = build_public_lineage_candidate(
        definition,
        PUBLIC_GENERATOR_CONFIGURATION,
        PUBLIC_PROFILE_CATALOG,
        PUBLIC_GENERATOR_ALGORITHM,
    )

    assert changed.generator_algorithm_digest == original.generator_algorithm_digest
    assert changed.candidate_packet_digest != original.candidate_packet_digest
    assert all(
        changed_preview.preview_digest != original_preview.preview_digest
        for changed_preview, original_preview in zip(
            changed.previews,
            original.previews,
            strict=True,
        )
    )
    assert all(
        changed_preview.slot_profile == original_preview.slot_profile
        for changed_preview, original_preview in zip(
            changed.previews,
            original.previews,
            strict=True,
        )
    )


def test_public_registry_literal_boundary_goldens() -> None:
    first_candidate = PUBLIC_LINEAGE_REGISTRY.candidates[0]
    last_candidate = PUBLIC_LINEAGE_REGISTRY.candidates[-1]

    assert (
        PUBLIC_GENERATOR_CONFIGURATION.configuration_digest
        == "837338e8463b79b38fcd2f4e3fa3b547c01ad143a8e48507f19da427d64c1785"
    )
    assert (
        PUBLIC_PROFILE_CATALOG.catalog_digest
        == "10f0bd934824ced5bab760db3eeb10eb3afcbc3191afcbf5e24cdcfe9416a6e7"
    )
    assert (
        PUBLIC_GENERATOR_ALGORITHM.algorithm_digest
        == "9108650981f11119326ce704e3a4c943f5bb3a3f42af3a65f04c88d5901f2ce4"
    )
    assert (
        first_candidate.candidate_packet_digest
        == "e69128c8fbdd0d2a2332a95e28d5e27d9f9b133aac9609169108d51f1257708a"
    )
    assert (
        last_candidate.candidate_packet_digest
        == "63f74a950aafd4c9bfc1e158433036e96d3e41014c38ee87ffb7349a41f72af4"
    )
    assert (
        first_candidate.previews[0].preview_digest
        == "ce138229ac384abdb1f2cbe1d42fb1ee6d94f291e219b40443cb3c3980669379"
    )
    assert (
        last_candidate.previews[-1].preview_digest
        == "8eacaf590c0c3b7b39b75750c0c2ac377c49c1f6e4a10ce6f31ee96d6d72602a"
    )
    assert (
        preview_to_scenario_id(first_candidate.previews[0])
        == "e6ccc48d0dd78f66b20b66683bc34b421c6d37d5e0317cb051521ce7af4abb5f"
    )
    assert (
        preview_to_scenario_id(last_candidate.previews[-1])
        == "492cbccfe9fbf49e0f4bac10874a1dc0164b141264a785d91f584edaa84aca2c"
    )
    assert (
        PUBLIC_LINEAGE_REGISTRY.registry_digest
        == "1b396e5fbee6e7a95ffb2739a47ddd97807cb76c0398ffd20363a26b9f076372"
    )
