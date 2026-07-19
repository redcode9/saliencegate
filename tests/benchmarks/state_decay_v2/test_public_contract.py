from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

import saliencegate.benchmarks.state_decay_v2.public_contract as public_contract_module
from saliencegate.benchmarks.state_decay_v2.authority import (
    ANALYSIS_CLUSTER_ENTRY_SCHEMA_VERSION,
    ORACLE_VAULT_ENTRY_SCHEMA_VERSION,
)
from saliencegate.benchmarks.state_decay_v2.config import (
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
from saliencegate.benchmarks.state_decay_v2.public_contract import (
    CANDIDATE_PACKET_DIGEST_DOMAIN,
    CANDIDATE_REGISTRY_DIGEST_DOMAIN,
    CAUSAL_DELTA_DIGEST_DOMAIN,
    EVIDENCE_TOPOLOGY_DIGEST_DOMAIN,
    GENERATOR_ALGORITHM_DIGEST_DOMAIN,
    GENERATOR_CONFIGURATION_DIGEST_DOMAIN,
    PROFILE_CATALOG_DIGEST_DOMAIN,
    RENDERED_POLICY_DIGEST_DOMAIN,
    SEMANTIC_SIGNATURE_DIGEST_DOMAIN,
    SIGNAL_PROFILE_DIGEST_DOMAIN,
    SKELETON_PREVIEW_DIGEST_DOMAIN,
    TRACE_FIXTURE_DIGEST_DOMAIN,
    TRANSITION_GRAPH_DIGEST_DOMAIN,
    CausalSemanticDelta,
    CausalTextReplacement,
    OutcomeFreeAllowedAction,
    OutcomeFreeCandidateMemory,
    OutcomeFreeEvent,
    OutcomeFreeEvidenceReference,
    OutcomeFreePivot,
    OutcomeFreeTaskSkeleton,
    OutcomeFreeTaskTemplate,
    OutcomeFreeTemplateAction,
    OutcomeFreeTemplateEvent,
    OutcomeFreeTemplateMemory,
    OutcomeFreeTemplatePivot,
    OutcomeFreeTraceFixture,
    PreAllocationSkeletonPreview,
    PublicAssertionFixture,
    PublicBindingFixture,
    PublicCausalExposure,
    PublicCausalFactor,
    PublicCausalFactorValue,
    PublicConstraintReferenceFixture,
    PublicCounterbalanceProfile,
    PublicDetectorMemoryFixture,
    PublicEvidenceEdge,
    PublicEvidenceNode,
    PublicEvidenceProfile,
    PublicEvidenceRelation,
    PublicEvidenceTopology,
    PublicExpectedAssertionEvidence,
    PublicExpectedDetectorEvidence,
    PublicExpectedMemoryEvidence,
    PublicExpectedSignal,
    PublicFailureMechanism,
    PublicFixtureEvent,
    PublicGeneratorAlgorithm,
    PublicGeneratorConfiguration,
    PublicGeneratorOperation,
    PublicGeneratorStep,
    PublicImpactClass,
    PublicIntegerProfile,
    PublicLineageCandidate,
    PublicLineageRegistry,
    PublicParameterProfile,
    PublicParameterValue,
    PublicProfileCatalog,
    PublicSemanticSignature,
    PublicSignalFixtureVariant,
    PublicSignalProfile,
    PublicSlotProfile,
    PublicStructuralProfile,
    PublicTerminalState,
    PublicTextLengthProfile,
    PublicTransition,
    PublicTransitionGraph,
    PublicTransitionState,
    RenderedCausalSemanticDelta,
    RenderedCausalTextReplacement,
    ReviewSafeText,
    RoleNeutralGeneratedParts,
    candidate_packet_digest,
    candidate_registry_digest,
    causal_delta_digest,
    evidence_topology_digest,
    generator_algorithm_digest,
    generator_configuration_digest,
    parse_public_lineage_key,
    profile_catalog_digest,
    public_lineage_key,
    rendered_policy_digest,
    semantic_signature_digest,
    signal_profile_digest,
    skeleton_preview_digest,
    trace_fixture_digest,
    transition_graph_digest,
)
from saliencegate.benchmarks.state_decay_v2.schema import (
    POLICY_VIEW_SCHEMA_VERSION,
    AdapterMetadata,
    BenchmarkSplit,
    ScenarioFamily,
    ScenarioOutcome,
)
from saliencegate.domain import (
    ClaimKind,
    EventPhase,
    EventType,
    SignalType,
    ValidityState,
    canonical_json,
)


class _TextProbe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    value: ReviewSafeText


_CAUSAL_FACTOR_IDS = ("guidance-relevant", "baseline-can-recover")
_CAUSAL_FACTOR_VECTORS = (
    (False, False),
    (False, True),
    (True, False),
    (True, True),
)
_PAIRED_TERMINALS = (
    (PublicTerminalState.GOAL_REACHED, PublicTerminalState.GOAL_NOT_REACHED),
    (PublicTerminalState.GOAL_NOT_REACHED, PublicTerminalState.GOAL_REACHED),
    (PublicTerminalState.GOAL_REACHED, PublicTerminalState.GOAL_REACHED),
    (PublicTerminalState.GOAL_NOT_REACHED, PublicTerminalState.GOAL_NOT_REACHED),
)


def _causal_factor_values(
    vector: tuple[bool, bool],
) -> tuple[PublicCausalFactorValue, ...]:
    return tuple(
        PublicCausalFactorValue(factor_id=factor_id, value=value)
        for factor_id, value in zip(_CAUSAL_FACTOR_IDS, vector, strict=True)
    )


def _causal_factor_vector_key(vector: tuple[bool, bool]) -> str:
    return "".join("1" if value else "0" for value in vector)


def test_review_safe_text_accepts_exact_nfc_utf8_boundary() -> None:
    assert _TextProbe(value="a" * 4_096).value == "a" * 4_096
    assert _TextProbe(value="é" * 2_048).value == "é" * 2_048
    assert _TextProbe(value="Café: [literal](not-rendered) \\ path").value == (
        "Café: [literal](not-rendered) \\ path"
    )


@pytest.mark.parametrize(
    "value",
    (
        "",
        "   ",
        "e\u0301",
        "before\0after",
        "before\x1fafter",
        "before\x7fafter",
        "before\x80after",
        "before\x9fafter",
        "before\nafter",
        "before\tafter",
        "before\u061cafter",
        "before\u200eafter",
        "before\u2028after",
        "before\u2029after",
        "before\u202eafter",
        "before\u2066after",
        "before\ufdd0after",
        "before\ufffeafter",
        "before\U0001ffffafter",
        "before\ud800after",
        "a" * 4_097,
        "é" * 2_049,
    ),
)
def test_review_safe_text_rejects_every_forbidden_class(value: str) -> None:
    with pytest.raises(ValidationError):
        _TextProbe(value=value)


def test_public_lineage_key_grammar_and_family_agreement_are_exact() -> None:
    expected = (
        (ScenarioFamily.FORGOTTEN_REQUIREMENT, "fr"),
        (ScenarioFamily.FAILED_PRIOR_ATTEMPT, "fp"),
        (ScenarioFamily.NEGLECTED_SUBGOAL, "ns"),
        (ScenarioFamily.STALE_MEMORY, "sm"),
        (ScenarioFamily.STABLE_ENVIRONMENT_FACT, "sf"),
        (ScenarioFamily.RETAINED_DIAGNOSIS, "rd"),
    )
    for family, code in expected:
        assert public_lineage_key(family, 0) == f"pub-{code}-00"
        key = public_lineage_key(family, 29)
        assert key == f"pub-{code}-29"
        assert parse_public_lineage_key(key) == (family, 29)


@pytest.mark.parametrize(
    "value",
    (
        "pub-fr-30",
        "pub-fr-3",
        "pub-fr-030",
        "pub-FR-00",
        "PUB-fr-00",
        "pub-ce-00",
        "pub-ia-00",
        "pub-fr-00.json",
        "pub/fr/00",
        " pub-fr-00",
    ),
)
def test_public_lineage_parser_rejects_aliases_and_out_of_range_keys(value: str) -> None:
    with pytest.raises(ValueError, match="public lineage key"):
        parse_public_lineage_key(value)


def test_public_lineage_builder_rejects_non_public_or_non_exact_coordinates() -> None:
    for family, index in (
        (ScenarioFamily.CONFLICTING_EVIDENCE, 0),
        (ScenarioFamily.IRREVERSIBLE_ACTION, 0),
        (ScenarioFamily.FORGOTTEN_REQUIREMENT, -1),
        (ScenarioFamily.FORGOTTEN_REQUIREMENT, 30),
        (ScenarioFamily.FORGOTTEN_REQUIREMENT, True),
    ):
        with pytest.raises(ValueError, match="public lineage coordinate"):
            public_lineage_key(family, index)


def test_public_digest_domains_and_literal_payload_goldens_are_frozen() -> None:
    payload = {
        "schema_version": "digest-probe/v1",
        "value": "alpha",
        "other_digest": "1" * 64,
    }
    cases: tuple[tuple[str, str, Callable[[Mapping[str, object]], str], str], ...] = (
        (
            "transition_graph_digest",
            TRANSITION_GRAPH_DIGEST_DOMAIN,
            transition_graph_digest,
            "65e9d67e792ba8ea658ed0da80863226f0853c19235b263c90d7629487688be4",
        ),
        (
            "evidence_topology_digest",
            EVIDENCE_TOPOLOGY_DIGEST_DOMAIN,
            evidence_topology_digest,
            "0bc7241c059f205933f8f07fd221ad563cc58ed393a32ea7b81be6416dfcc2ff",
        ),
        (
            "semantic_signature_digest",
            SEMANTIC_SIGNATURE_DIGEST_DOMAIN,
            semantic_signature_digest,
            "6fc27125476270c0983da9118e9db2e29f24ccae48fca55661b2489d7955e310",
        ),
        (
            "causal_delta_digest",
            CAUSAL_DELTA_DIGEST_DOMAIN,
            causal_delta_digest,
            "d246664a5cf79c555135788cfe44f715065aea4e856f38a677fe302c4394d406",
        ),
        (
            "configuration_digest",
            GENERATOR_CONFIGURATION_DIGEST_DOMAIN,
            generator_configuration_digest,
            "1016f4607b447e5b686c96b3342804284a0b65bbbef2577c20f873c31bde338c",
        ),
        (
            "algorithm_digest",
            GENERATOR_ALGORITHM_DIGEST_DOMAIN,
            generator_algorithm_digest,
            "c38c5ad025e3b46d44d2f78985709290b1d5b6f7a41c6124e4ec20b44b3e10c1",
        ),
        (
            "catalog_digest",
            PROFILE_CATALOG_DIGEST_DOMAIN,
            profile_catalog_digest,
            "488e5eb068e4b068d92819a7e16868def99180541ff6e3973a934cbde0380180",
        ),
        (
            "preview_digest",
            SKELETON_PREVIEW_DIGEST_DOMAIN,
            skeleton_preview_digest,
            "21a26a1ed6b4e197561e1bebd37b1472cd5dd71b93d5f572be5a0fb02e20ea40",
        ),
        (
            "candidate_packet_digest",
            CANDIDATE_PACKET_DIGEST_DOMAIN,
            candidate_packet_digest,
            "33267564d055bc7785d4093d7789056b1c6e3f366e74ea9dca362c70444d4602",
        ),
        (
            "registry_digest",
            CANDIDATE_REGISTRY_DIGEST_DOMAIN,
            candidate_registry_digest,
            "427749131fa92e2c852f083fbf7bffd8c8cc41ca571a51afc273775fd4ef46eb",
        ),
        (
            "profile_digest",
            SIGNAL_PROFILE_DIGEST_DOMAIN,
            signal_profile_digest,
            "b7ee774b620074dfd1c85efd6ae80cac66441c098cd913865189fe792e8dcf14",
        ),
        (
            "trace_fixture_digest",
            TRACE_FIXTURE_DIGEST_DOMAIN,
            trace_fixture_digest,
            "362a33f76db1c839b94e17ec00e91ab6ba80f5e632bf93983e75892200316c77",
        ),
    )
    assert tuple(domain for _, domain, _, _ in cases) == (
        "saliencegate:state-decay-v2:public-review:transition-graph:v1",
        "saliencegate:state-decay-v2:public-review:evidence-topology:v1",
        "saliencegate:state-decay-v2:public-review:semantic-signature:v1",
        "saliencegate:state-decay-v2:public-review:causal-delta:v1",
        "saliencegate:state-decay-v2:public-review:generator-configuration:v1",
        "saliencegate:state-decay-v2:public-review:generator-algorithm:v1",
        "saliencegate:state-decay-v2:public-review:profile-catalog:v1",
        "saliencegate:state-decay-v2:public-review:skeleton-preview:v1",
        "saliencegate:state-decay-v2:public-review:candidate-packet:v1",
        "saliencegate:state-decay-v2:public-review:candidate-registry:v1",
        "saliencegate:state-decay-v2:public-review:signal-profile:v1",
        "saliencegate:state-decay-v2:public-review:trace-fixture:v1",
    )
    for self_field, _, digest, golden in cases:
        value = {**payload, self_field: "0" * 64}
        assert digest(value) == golden
        assert digest({**value, self_field: "f" * 64}) == golden
        assert digest({**value, "other_digest": "2" * 64}) != golden

    assert RENDERED_POLICY_DIGEST_DOMAIN == (
        "saliencegate:state-decay-v2:public-review:rendered-policy:v1"
    )
    assert rendered_policy_digest(payload) == (
        "e79377c63620c16ee3729c4073c2cee19c349d8b311721f6ef08e1122ca64f1a"
    )


def _outcome_free_skeleton() -> OutcomeFreeTaskSkeleton:
    return OutcomeFreeTaskSkeleton(
        trajectory=(
            OutcomeFreeEvent(
                event_id="event-1",
                sequence=1,
                action_step=0,
                statement="The retained constraint was recorded.",
            ),
            OutcomeFreeEvent(
                event_id="event-2",
                sequence=2,
                action_step=1,
                statement="A routine operation completed.",
            ),
            OutcomeFreeEvent(
                event_id="event-3",
                sequence=3,
                action_step=2,
                statement="Current evidence was recorded.",
            ),
        ),
        candidate_memories=(
            OutcomeFreeCandidateMemory(
                memory_id="memory-1",
                revision=1,
                statement="The retained constraint remains available.",
                evidence_refs=(OutcomeFreeEvidenceReference(event_id="event-1", event_sequence=1),),
                recorded_sequence=2,
                recorded_action_step=1,
                validity=ValidityState.ACTIVE,
            ),
        ),
        pivot=OutcomeFreePivot(
            event_id="pivot-1",
            sequence=4,
            action_step=2,
            statement="Choose exactly one next action.",
        ),
        allowed_actions=(
            OutcomeFreeAllowedAction(
                action_id="action-primary",
                statement="Apply the retained constraint.",
            ),
            OutcomeFreeAllowedAction(
                action_id="action-alternate",
                statement="Apply the current alternative.",
            ),
        ),
        adapter=AdapterMetadata(
            adapter_id="state-decay-public-test",
            adapter_version="v1",
            response_profile_id="two-action-choice",
            response_profile_digest="1" * 64,
        ),
    )


def _slot_outcome_free_skeleton(profile: PublicSlotProfile) -> OutcomeFreeTaskSkeleton:
    integers = profile.integers
    event_count = profile.structure.trajectory_event_count
    trajectory = tuple(
        OutcomeFreeEvent(
            event_id=f"event-{index + 1}",
            sequence=integers.sequence_start + index * integers.sequence_stride,
            action_step=integers.action_step_start + index * integers.action_step_stride,
            statement=(
                f"Repository-authored event statement number {index}."
                + " " * profile.text_lengths.event_padding_spaces
            ),
        )
        for index in range(event_count)
    )
    last_event = trajectory[-1]
    validity = profile.counterbalance.memory_validity
    candidate_memories = tuple(
        OutcomeFreeCandidateMemory(
            memory_id=f"memory-{index + 1}",
            revision=integers.memory_revision + index,
            statement=f"Retained constraint {index + 1} remains available.",
            evidence_refs=(
                OutcomeFreeEvidenceReference(
                    event_id=trajectory[0].event_id,
                    event_sequence=trajectory[0].sequence,
                ),
            ),
            recorded_sequence=trajectory[0].sequence,
            recorded_action_step=trajectory[0].action_step,
            validity=validity,
            validity_sequence=(None if validity is ValidityState.ACTIVE else last_event.sequence),
            validity_action_step=(
                None if validity is ValidityState.ACTIVE else last_event.action_step
            ),
        )
        for index in range(profile.structure.candidate_memory_count)
    )
    logical_actions = (
        OutcomeFreeAllowedAction(
            action_id="action-primary",
            statement="Apply the retained constraint.",
        ),
        OutcomeFreeAllowedAction(
            action_id="action-alternate",
            statement="Apply the current alternative.",
        ),
    )
    return OutcomeFreeTaskSkeleton(
        trajectory=trajectory,
        candidate_memories=candidate_memories,
        pivot=OutcomeFreePivot(
            event_id="pivot-1",
            sequence=last_event.sequence + integers.sequence_stride,
            action_step=last_event.action_step,
            statement="Choose exactly one next action.",
        ),
        allowed_actions=tuple(
            logical_actions[index] for index in profile.counterbalance.allowed_action_order
        ),
        adapter=AdapterMetadata(
            adapter_id="state-decay-public-test",
            adapter_version="v1",
            response_profile_id="two-action-choice",
            response_profile_digest="1" * 64,
        ),
    )


def _task_template() -> OutcomeFreeTaskTemplate:
    events = tuple(
        OutcomeFreeTemplateEvent(
            event_id=f"event-{index}",
            statement=f"Repository-authored event statement number {index}.",
        )
        for index in range(8)
    )
    memories = tuple(
        OutcomeFreeTemplateMemory(
            memory_id=f"memory-{index}",
            statement=f"Repository-authored memory statement number {index}.",
            evidence_event_ids=("event-0", "event-1", "event-2"),
            recorded_event_id="event-2",
        )
        for index in range(4)
    )
    return OutcomeFreeTaskTemplate(
        event_pool=events,
        memory_pool=memories,
        pivot=OutcomeFreeTemplatePivot(
            event_id="pivot-1",
            statement="Choose exactly one next action.",
        ),
        action_pool=(
            OutcomeFreeTemplateAction(
                action_id="action-primary",
                statement="Apply the retained constraint.",
            ),
            OutcomeFreeTemplateAction(
                action_id="action-alternate",
                statement="Apply the current alternative.",
            ),
        ),
        adapter=AdapterMetadata(
            adapter_id="state-decay-public-test",
            adapter_version="v1",
            response_profile_id="two-action-choice",
            response_profile_digest="1" * 64,
        ),
    )


def test_outcome_free_task_template_has_fixed_resolved_disjoint_pools() -> None:
    template = _task_template()

    assert OutcomeFreeTaskTemplate.model_validate_json(canonical_json(template)) == template
    assert len(template.event_pool) == 8
    assert len(template.memory_pool) == 4
    assert len(template.action_pool) == 2

    payload = template.model_dump(mode="python")
    payload["memory_pool"][0]["evidence_event_ids"] = ("event-0", "event-1", "event-7")
    with pytest.raises(ValidationError, match="first three"):
        OutcomeFreeTaskTemplate.model_validate(payload)

    payload = template.model_dump(mode="python")
    payload["pivot"]["event_id"] = "event-0"
    with pytest.raises(ValidationError, match="disjoint"):
        OutcomeFreeTaskTemplate.model_validate(payload)


def test_outcome_free_task_template_freezes_nonfuture_evidence_order() -> None:
    template = _task_template()
    payload = template.model_dump(mode="python")
    payload["memory_pool"][0]["evidence_event_ids"] = (
        "event-2",
        "event-1",
        "event-0",
    )
    with pytest.raises(ValidationError, match="first three"):
        OutcomeFreeTaskTemplate.model_validate(payload)

    payload = template.model_dump(mode="python")
    payload["memory_pool"][0]["recorded_event_id"] = "event-0"
    with pytest.raises(ValidationError, match="third event"):
        OutcomeFreeTaskTemplate.model_validate(payload)


def test_outcome_free_skeleton_is_safe_strict_frozen_and_canonical() -> None:
    skeleton = _outcome_free_skeleton()

    assert OutcomeFreeTaskSkeleton.model_validate_json(canonical_json(skeleton)) == skeleton
    assert tuple(OutcomeFreeTaskSkeleton.model_fields) == (
        "trajectory",
        "candidate_memories",
        "pivot",
        "allowed_actions",
        "adapter",
    )
    with pytest.raises(ValidationError):
        OutcomeFreeTaskSkeleton.model_validate(
            {**skeleton.model_dump(mode="python"), "scenario_id": "0" * 64}
        )
    with pytest.raises(ValidationError):
        skeleton.trajectory = ()  # type: ignore[misc]

    payload = skeleton.model_dump(mode="python")
    payload["trajectory"][0]["statement"] = "spoof\u202e"
    with pytest.raises(ValidationError):
        OutcomeFreeTaskSkeleton.model_validate(payload)

    payload = skeleton.model_dump(mode="python")
    payload["trajectory"][0]["statement"] = "a" * 2_049
    with pytest.raises(ValidationError):
        OutcomeFreeTaskSkeleton.model_validate(payload)


def test_outcome_free_skeleton_rejects_broken_policy_references() -> None:
    skeleton = _outcome_free_skeleton()
    payload = skeleton.model_dump(mode="python")
    payload["candidate_memories"][0]["evidence_refs"][0]["event_id"] = "missing"
    with pytest.raises(ValidationError, match="evidence reference"):
        OutcomeFreeTaskSkeleton.model_validate(payload)

    payload = skeleton.model_dump(mode="python")
    payload["pivot"]["sequence"] = 3
    with pytest.raises(ValidationError, match="pivot"):
        OutcomeFreeTaskSkeleton.model_validate(payload)


def _generator_configuration() -> PublicGeneratorConfiguration:
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-public-generator-configuration/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "generator_version": "state-decay-v2-public-generator/v1",
        "generation_contract_digest": GENERATION_CONTRACT.contract_digest,
        "visible_splits": (BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT),
        "visible_families": (
            ScenarioFamily.FORGOTTEN_REQUIREMENT,
            ScenarioFamily.FAILED_PRIOR_ATTEMPT,
            ScenarioFamily.NEGLECTED_SUBGOAL,
            ScenarioFamily.STALE_MEMORY,
            ScenarioFamily.STABLE_ENVIRONMENT_FACT,
            ScenarioFamily.RETAINED_DIAGNOSIS,
        ),
        "lineages_per_family": 30,
        "generator_slots_per_lineage": 5,
        "candidate_count": 180,
        "preview_count": 900,
        "maximum_review_text_utf8_bytes": 4_096,
        "legacy_repetition_window_events": 8,
        "policy_schema_version": POLICY_VIEW_SCHEMA_VERSION,
        "oracle_schema_version": ORACLE_VAULT_ENTRY_SCHEMA_VERSION,
        "analysis_schema_version": ANALYSIS_CLUSTER_ENTRY_SCHEMA_VERSION,
        "nuisance_inventory_digest": NUISANCE_FEATURE_INVENTORY.inventory_digest,
    }
    values["configuration_digest"] = generator_configuration_digest(values)
    return PublicGeneratorConfiguration.model_validate(values)


def test_public_generator_configuration_is_complete_and_self_attesting() -> None:
    configuration = _generator_configuration()

    assert PublicGeneratorConfiguration.model_validate_json(canonical_json(configuration)) == (
        configuration
    )
    assert tuple(PublicGeneratorConfiguration.model_fields) == (
        "schema_version",
        "suite_id",
        "suite_version",
        "generator_version",
        "generation_contract_digest",
        "visible_splits",
        "visible_families",
        "lineages_per_family",
        "generator_slots_per_lineage",
        "candidate_count",
        "preview_count",
        "maximum_review_text_utf8_bytes",
        "legacy_repetition_window_events",
        "policy_schema_version",
        "oracle_schema_version",
        "analysis_schema_version",
        "nuisance_inventory_digest",
        "configuration_digest",
    )
    assert configuration.legacy_repetition_window_events == 8
    assert configuration.candidate_count == 180
    assert configuration.preview_count == 900
    for field, replacement in (
        ("configuration_digest", "0" * 64),
        ("candidate_count", 181),
        ("visible_splits", tuple(reversed(configuration.visible_splits))),
    ):
        payload = configuration.model_dump(mode="python")
        payload[field] = replacement
        with pytest.raises(ValidationError):
            PublicGeneratorConfiguration.model_validate(payload)


def _expected_signal_evidence(
    *event_indices: int,
    binding_indices: tuple[int, ...] = (),
    memory_references: tuple[PublicExpectedMemoryEvidence, ...] = (),
    assertion_references: tuple[PublicExpectedAssertionEvidence, ...] = (),
) -> PublicExpectedDetectorEvidence:
    return PublicExpectedDetectorEvidence(
        event_pool_indices=event_indices,
        binding_event_pool_indices=binding_indices,
        memory_references=memory_references,
        assertion_references=assertion_references,
    )


def _signal_profile(
    *,
    slot: int,
    variant: PublicSignalFixtureVariant,
    signals: tuple[PublicExpectedSignal, ...],
) -> PublicSignalProfile:
    values: dict[str, object] = {
        "profile_id": f"signals-slot-{slot}",
        "fixture_variant": variant,
        "expected_signals": signals,
    }
    values["profile_digest"] = signal_profile_digest(values)
    return PublicSignalProfile.model_validate(values)


def _signal_profiles() -> tuple[PublicSignalProfile, ...]:
    return (
        _signal_profile(
            slot=0,
            variant=PublicSignalFixtureVariant.FAILED_TEST_CONFLICT_MISSING_CONSTRAINT,
            signals=(
                PublicExpectedSignal(
                    signal_type=SignalType.CONFLICT,
                    strength_ppm=1_000_000,
                    evidence=_expected_signal_evidence(
                        2,
                        binding_indices=(2,),
                        assertion_references=(
                            PublicExpectedAssertionEvidence(
                                binding_event_pool_index=2,
                                assertion_index=0,
                            ),
                            PublicExpectedAssertionEvidence(
                                binding_event_pool_index=2,
                                assertion_index=1,
                            ),
                        ),
                    ),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.STALE_CONSTRAINT,
                    strength_ppm=1_000_000,
                    evidence=_expected_signal_evidence(
                        2,
                        binding_indices=(2,),
                        memory_references=(
                            PublicExpectedMemoryEvidence(memory_pool_index=1, revision=1),
                        ),
                    ),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.TEST_FAILURE,
                    strength_ppm=1_000_000,
                    evidence=_expected_signal_evidence(2),
                ),
            ),
        ),
        _signal_profile(
            slot=1,
            variant=PublicSignalFixtureVariant.REPEATED_ACTION_SCOPE_SHIFT_IRREVERSIBLE,
            signals=(
                PublicExpectedSignal(
                    signal_type=SignalType.CONTEXT_SHIFT,
                    strength_ppm=500_000,
                    evidence=_expected_signal_evidence(
                        2,
                        3,
                        binding_indices=(2, 3),
                        memory_references=(
                            PublicExpectedMemoryEvidence(memory_pool_index=0, revision=2),
                        ),
                    ),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.IRREVERSIBLE_ACTION,
                    strength_ppm=1_000_000,
                    evidence=_expected_signal_evidence(3, binding_indices=(3,)),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.REPEATED_ACTION,
                    strength_ppm=1_000_000,
                    evidence=_expected_signal_evidence(2, 3),
                ),
            ),
        ),
        _signal_profile(
            slot=2,
            variant=PublicSignalFixtureVariant.STAGNANT_CONFLICTING_ASSERTIONS,
            signals=(
                PublicExpectedSignal(
                    signal_type=SignalType.CONFLICT,
                    strength_ppm=1_000_000,
                    evidence=_expected_signal_evidence(
                        4,
                        binding_indices=(4,),
                        assertion_references=(
                            PublicExpectedAssertionEvidence(
                                binding_event_pool_index=4,
                                assertion_index=0,
                            ),
                            PublicExpectedAssertionEvidence(
                                binding_event_pool_index=4,
                                assertion_index=1,
                            ),
                        ),
                    ),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.STAGNATION,
                    strength_ppm=500_000,
                    evidence=_expected_signal_evidence(
                        1,
                        2,
                        3,
                        4,
                        binding_indices=(1, 2, 3, 4),
                    ),
                ),
            ),
        ),
        _signal_profile(
            slot=3,
            variant=PublicSignalFixtureVariant.REPEATED_FAILURE_SUPERSEDED_CONSTRAINT,
            signals=(
                PublicExpectedSignal(
                    signal_type=SignalType.REPEATED_FAILURE,
                    strength_ppm=1_000_000,
                    evidence=_expected_signal_evidence(2, 3, 4, 5),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.STALE_CONSTRAINT,
                    strength_ppm=750_000,
                    evidence=_expected_signal_evidence(
                        5,
                        binding_indices=(5,),
                        memory_references=(
                            PublicExpectedMemoryEvidence(memory_pool_index=0, revision=4),
                        ),
                    ),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.TOOL_ERROR,
                    strength_ppm=1_000_000,
                    evidence=_expected_signal_evidence(5),
                ),
            ),
        ),
        _signal_profile(
            slot=4,
            variant=PublicSignalFixtureVariant.REPEATED_ACTION_SCOPE_SHIFT_STAGNATION,
            signals=(
                PublicExpectedSignal(
                    signal_type=SignalType.CONTEXT_SHIFT,
                    strength_ppm=500_000,
                    evidence=_expected_signal_evidence(
                        5,
                        6,
                        binding_indices=(5, 6),
                        memory_references=(
                            PublicExpectedMemoryEvidence(memory_pool_index=0, revision=5),
                        ),
                    ),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.REPEATED_ACTION,
                    strength_ppm=1_000_000,
                    evidence=_expected_signal_evidence(2, 6),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.STAGNATION,
                    strength_ppm=625_000,
                    evidence=_expected_signal_evidence(
                        2,
                        3,
                        4,
                        5,
                        6,
                        binding_indices=(2, 3, 4, 5, 6),
                    ),
                ),
            ),
        ),
    )


def _trace_fixture(
    profile: PublicSlotProfile,
    skeleton: OutcomeFreeTaskSkeleton,
) -> OutcomeFreeTraceFixture:
    expected_by_type = {
        expected.signal_type: expected for expected in profile.signals.expected_signals
    }
    action_indices = set(
        expected_by_type[SignalType.REPEATED_ACTION].evidence.event_pool_indices
        if SignalType.REPEATED_ACTION in expected_by_type
        else ()
    )
    failure_indices = {
        index
        for signal_type in (
            SignalType.REPEATED_FAILURE,
            SignalType.TEST_FAILURE,
            SignalType.TOOL_ERROR,
        )
        for index in (
            expected_by_type[signal_type].evidence.event_pool_indices
            if signal_type in expected_by_type
            else ()
        )
    }
    context_indices = (
        expected_by_type[SignalType.CONTEXT_SHIFT].evidence.binding_event_pool_indices
        if SignalType.CONTEXT_SHIFT in expected_by_type
        else ()
    )
    irreversible_indices = set(
        expected_by_type[SignalType.IRREVERSIBLE_ACTION].evidence.binding_event_pool_indices
        if SignalType.IRREVERSIBLE_ACTION in expected_by_type
        else ()
    )
    conflict_indices = set(
        expected_by_type[SignalType.CONFLICT].evidence.binding_event_pool_indices
        if SignalType.CONFLICT in expected_by_type
        else ()
    )

    events: list[PublicFixtureEvent] = []
    bindings: list[PublicBindingFixture] = []
    for index, policy_event in enumerate(skeleton.trajectory):
        if index == 0:
            event_type = EventType.RUN_START
            phase = EventPhase.INITIALIZATION
            payload: dict[str, object] = {}
        elif index in failure_indices:
            event_type = EventType.TOOL_COMPLETION
            phase = EventPhase.POST_ACTION
            payload = {
                "tool_outcome": {
                    "schema_version": "1.0",
                    "status": "failed",
                    "exit_status": 1,
                },
                "test_report": {
                    "schema_version": "1.0",
                    "framework": "pytest",
                    "status": "failed",
                    "failures": (),
                },
            }
        elif index in action_indices:
            event_type = EventType.ACTION_PROPOSAL
            phase = EventPhase.PRE_ACTION
            payload = {
                "action": {
                    "schema_version": "1.0",
                    "kind": "shell",
                    "command": "pytest -q",
                    "working_directory": "/workspace",
                    "environment_digest": "a" * 64,
                }
            }
        else:
            event_type = EventType.OBSERVATION
            phase = EventPhase.POST_ACTION
            payload = {"observation_index": index}
        events.append(
            PublicFixtureEvent(
                event_pool_index=index,
                event_type=event_type,
                phase=phase,
                payload=payload,
                parent_event_pool_indices=(() if index == 0 else (index - 1,)),
            )
        )

        constraint_references = tuple(
            PublicConstraintReferenceFixture(
                memory_pool_index=memory.memory_pool_index,
                revision=memory.revision,
            )
            for expected in profile.signals.expected_signals
            if index in expected.evidence.binding_event_pool_indices
            for memory in expected.evidence.memory_references
        )
        assertions = (
            (
                PublicAssertionFixture(
                    subject_id="build",
                    predicate_id="status",
                    value_digest="c" * 64,
                    precedence=1,
                    revision=1,
                    supersedes_assertion_digest=None,
                ),
                PublicAssertionFixture(
                    subject_id="build",
                    predicate_id="status",
                    value_digest="d" * 64,
                    precedence=1,
                    revision=1,
                    supersedes_assertion_digest=None,
                ),
            )
            if index in conflict_indices
            else ()
        )
        scope_id = (
            f"scope-{context_indices.index(index)}" if index in context_indices else "scope-0"
        )
        bindings.append(
            PublicBindingFixture(
                event_pool_index=index,
                action_step=policy_event.action_step,
                scope_id=scope_id,
                progress_marker_digest="b" * 64,
                constraint_references=constraint_references,
                impact=(
                    PublicImpactClass.IRREVERSIBLE
                    if index in irreversible_indices
                    else PublicImpactClass.REVERSIBLE
                ),
                authorization_event_pool_indices=(),
                safeguard_event_pool_indices=(),
                assertions=assertions,
            )
        )

    memories = tuple(
        PublicDetectorMemoryFixture(
            memory_pool_index=index,
            kind=ClaimKind.REQUIREMENT,
            current_revision=memory.revision,
            validity=memory.validity,
            provenance_event_pool_indices=(0,),
            expires_at_event_pool_index=None,
        )
        for index, memory in enumerate(skeleton.candidate_memories)
    )
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-outcome-free-trace-fixture/v1",
        "events": tuple(events),
        "bindings": tuple(bindings),
        "memories": memories,
    }
    values["trace_fixture_digest"] = trace_fixture_digest(values)
    return OutcomeFreeTraceFixture.model_validate(values)


def _slot_profile(slot: int) -> PublicSlotProfile:
    inactive = slot % 2 == 1
    return PublicSlotProfile(
        generator_slot=slot,
        counterbalance=PublicCounterbalanceProfile(
            profile_id=f"counterbalance-slot-{slot}",
            allowed_action_order=(0, 1) if slot % 2 == 0 else (1, 0),
            decisive_action_position=slot % 2,  # type: ignore[arg-type]
            memory_validity=(ValidityState.SUPERSEDED if inactive else ValidityState.ACTIVE),
            include_validity_transition=inactive,
        ),
        parameters=PublicParameterProfile(
            profile_id=f"parameters-slot-{slot}",
            allowed_values=(
                PublicParameterValue(parameter_id=f"parameter-slot-{slot}", value=slot),
            ),
        ),
        structure=PublicStructuralProfile(
            profile_id=f"structure-slot-{slot}",
            trajectory_event_count=3 + slot,
            candidate_memory_count=1 + (slot % 2),
        ),
        integers=PublicIntegerProfile(
            profile_id=f"integers-slot-{slot}",
            sequence_start=1 + slot,
            sequence_stride=1,
            action_step_start=slot,
            action_step_stride=1,
            memory_revision=1 + slot,
        ),
        evidence=PublicEvidenceProfile(
            profile_id=f"evidence-slot-{slot}",
            evidence_reference_count=1 + (slot % 2),
            decisive_event_count=1,
            decisive_memory_count=1,
        ),
        text_lengths=PublicTextLengthProfile(
            profile_id=f"text-slot-{slot}",
            event_padding_spaces=slot,
            memory_padding_spaces=slot + 1,
            pivot_padding_spaces=slot + 2,
            action_padding_spaces=slot + 3,
        ),
        signals=_signal_profiles()[slot],
    )


def _profile_catalog() -> PublicProfileCatalog:
    configuration = _generator_configuration()
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-public-profile-catalog/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "generation_contract_digest": GENERATION_CONTRACT.contract_digest,
        "generator_configuration_digest": configuration.configuration_digest,
        "counterbalance_axes": tuple(CounterbalanceAxis),
        "slot_profiles": tuple(_slot_profile(slot) for slot in range(5)),
    }
    values["catalog_digest"] = profile_catalog_digest(values)
    return PublicProfileCatalog.model_validate(values)


def test_public_profile_catalog_contains_five_complete_global_slots() -> None:
    catalog = _profile_catalog()

    assert PublicProfileCatalog.model_validate_json(canonical_json(catalog)) == catalog
    assert tuple(PublicSlotProfile.model_fields)[-1] == "signals"
    assert tuple(profile.generator_slot for profile in catalog.slot_profiles) == tuple(range(5))
    assert catalog.counterbalance_axes == tuple(CounterbalanceAxis)
    assert tuple(profile.signals.fixture_variant for profile in catalog.slot_profiles) == tuple(
        PublicSignalFixtureVariant
    )

    payload = catalog.model_dump(mode="python")
    payload["slot_profiles"][1]["generator_slot"] = 0
    payload["catalog_digest"] = profile_catalog_digest(payload)
    with pytest.raises(ValidationError, match="slot profiles"):
        PublicProfileCatalog.model_validate(payload)

    payload = catalog.model_dump(mode="python")
    payload["slot_profiles"][0]["signals"]["profile_digest"] = "0" * 64
    payload["catalog_digest"] = profile_catalog_digest(payload)
    with pytest.raises(ValidationError, match="profile digest"):
        PublicProfileCatalog.model_validate(payload)


def test_profile_components_reject_internal_mismatches() -> None:
    profile = _slot_profile(0)
    payload = profile.counterbalance.model_dump(mode="python")
    payload["allowed_action_order"] = (0, 0)
    with pytest.raises(ValidationError, match="action order"):
        PublicCounterbalanceProfile.model_validate(payload)

    payload = profile.counterbalance.model_dump(mode="python")
    payload["include_validity_transition"] = True
    with pytest.raises(ValidationError, match="validity transition"):
        PublicCounterbalanceProfile.model_validate(payload)

    payload = profile.counterbalance.model_dump(mode="python")
    payload["decisive_action_position"] = 1
    with pytest.raises(ValidationError, match="decisive action"):
        PublicCounterbalanceProfile.model_validate(payload)

    overflowing = profile.model_dump(mode="python")
    overflowing["integers"]["sequence_start"] = 999_999
    overflowing["integers"]["sequence_stride"] = 2
    with pytest.raises(ValidationError, match="derived integer"):
        PublicSlotProfile.model_validate(overflowing)


def _semantic_artifacts() -> tuple[
    PublicTransitionGraph,
    PublicEvidenceTopology,
    PublicFailureMechanism,
    PublicSemanticSignature,
]:
    transitions = tuple(
        PublicTransition(
            source_state_id="initial",
            target_state_id=(
                "goal-reached"
                if terminal is PublicTerminalState.GOAL_REACHED
                else "goal-not-reached"
            ),
            exposure=exposure,
            factor_values=_causal_factor_values(vector),
            action_fingerprint_id=(f"action-{exposure.value}-{_causal_factor_vector_key(vector)}"),
            failure_fingerprint_id=(
                None
                if terminal is PublicTerminalState.GOAL_REACHED
                else f"failure-{exposure.value}-{_causal_factor_vector_key(vector)}"
            ),
            trigger=f"Execute the {exposure.value} path for this factor vector.",
        )
        for vector, paired_terminals in zip(
            _CAUSAL_FACTOR_VECTORS,
            _PAIRED_TERMINALS,
            strict=True,
        )
        for exposure, terminal in zip(
            (
                PublicCausalExposure.GUIDANCE_APPLIED,
                PublicCausalExposure.BASELINE_CONTINUED,
            ),
            paired_terminals,
            strict=True,
        )
    )
    graph_values: dict[str, object] = {
        "schema_version": "state-decay-v2-public-transition-graph/v1",
        "initial_state_id": "initial",
        "factors": (
            PublicCausalFactor(
                factor_id=_CAUSAL_FACTOR_IDS[0],
                true_description="The guidance changes the decisive action path.",
                false_description="The decisive action path remains available.",
            ),
            PublicCausalFactor(
                factor_id=_CAUSAL_FACTOR_IDS[1],
                true_description="The uninterrupted baseline can recover the task goal.",
                false_description="The uninterrupted baseline cannot recover the task goal.",
            ),
        ),
        "states": (
            PublicTransitionState(
                state_id="initial",
                description="The action path has not terminated.",
            ),
            PublicTransitionState(
                state_id="goal-reached",
                description="The task goal is reached.",
                terminal=PublicTerminalState.GOAL_REACHED,
            ),
            PublicTransitionState(
                state_id="goal-not-reached",
                description="The task goal is not reached.",
                terminal=PublicTerminalState.GOAL_NOT_REACHED,
            ),
        ),
        "transitions": transitions,
    }
    graph_values["transition_graph_digest"] = transition_graph_digest(graph_values)
    graph = PublicTransitionGraph.model_validate(graph_values)

    topology_values: dict[str, object] = {
        "schema_version": "state-decay-v2-public-evidence-topology/v1",
        "nodes": (
            PublicEvidenceNode(evidence_id="retained", statement="The retained fact was observed."),
            PublicEvidenceNode(evidence_id="current", statement="The current fact was observed."),
        ),
        "edges": (
            PublicEvidenceEdge(
                source_evidence_id="current",
                target_evidence_id="retained",
                relation=PublicEvidenceRelation.CONTEXTUALIZES,
            ),
        ),
    }
    topology_values["evidence_topology_digest"] = evidence_topology_digest(topology_values)
    topology = PublicEvidenceTopology.model_validate(topology_values)

    signature_values: dict[str, object] = {
        "schema_version": "state-decay-v2-public-semantic-signature/v1",
        "concept_ids": ("retained-fact", "current-evidence"),
        "canonical_claims": (
            "The retained fact is visible before the pivot.",
            "Current evidence is visible before the action choice.",
        ),
    }
    signature_values["semantic_signature_digest"] = semantic_signature_digest(signature_values)
    signature = PublicSemanticSignature.model_validate(signature_values)
    failure = PublicFailureMechanism(
        failure_mechanism_id="ignored-decisive-evidence",
        description="The wrong action ignores decisive evidence visible before the pivot.",
    )
    return graph, topology, failure, signature


def test_semantic_artifacts_are_resolved_self_attesting_and_canonical() -> None:
    graph, topology, failure, signature = _semantic_artifacts()

    for model in (graph, topology, failure, signature):
        assert type(model).model_validate_json(canonical_json(model)) == model

    graph_payload = graph.model_dump(mode="python")
    graph_payload["transitions"][0]["target_state_id"] = "missing"
    graph_payload["transition_graph_digest"] = transition_graph_digest(graph_payload)
    with pytest.raises(ValidationError, match="transition endpoint"):
        PublicTransitionGraph.model_validate(graph_payload)

    topology_payload = topology.model_dump(mode="python")
    topology_payload["evidence_topology_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="evidence topology digest"):
        PublicEvidenceTopology.model_validate(topology_payload)


def _causal_delta(*, index: int = 0) -> CausalSemanticDelta:
    replacement_word = ("adjusted", "modified", "reframed", "restated")[index]
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-public-causal-semantic-delta/v1",
        "delta_index": index,
        "delta_id": f"delta-pub-fr-00-{index}",
        "family": ScenarioFamily.FORGOTTEN_REQUIREMENT,
        "lineage_registry_key": "pub-fr-00",
        "factor_values": _causal_factor_values(_CAUSAL_FACTOR_VECTORS[index]),
        "semantic_replacements": (
            CausalTextReplacement(
                template_pointer="/event_pool/0/statement",
                replacement=f"Repository-{replacement_word} event statement number 0.",
            ),
        ),
        "evidence_replacements": (),
    }
    values["causal_delta_digest"] = causal_delta_digest(values)
    return CausalSemanticDelta.model_validate(values)


def test_causal_delta_is_candidate_local_outcome_free_and_self_attesting() -> None:
    delta = _causal_delta()

    assert CausalSemanticDelta.model_validate_json(canonical_json(delta)) == delta
    assert tuple(CausalTextReplacement.model_fields) == ("template_pointer", "replacement")
    assert tuple(CausalSemanticDelta.model_fields) == (
        "schema_version",
        "delta_index",
        "delta_id",
        "family",
        "lineage_registry_key",
        "factor_values",
        "semantic_replacements",
        "evidence_replacements",
        "causal_delta_digest",
    )
    assert "outcome" not in canonical_json(delta).decode("utf-8")

    payload = delta.model_dump(mode="python")
    payload["causal_delta_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="causal delta digest"):
        CausalSemanticDelta.model_validate(payload)


def test_causal_delta_rejects_duplicate_targets_and_outcome_labels() -> None:
    delta = _causal_delta()
    payload = delta.model_dump(mode="python")
    payload["evidence_replacements"] = payload["semantic_replacements"]
    payload["causal_delta_digest"] = causal_delta_digest(payload)
    with pytest.raises(ValidationError, match="replacement targets"):
        CausalSemanticDelta.model_validate(payload)

    payload = delta.model_dump(mode="python")
    payload["delta_id"] = "helpful-delta"
    payload["causal_delta_digest"] = causal_delta_digest(payload)
    with pytest.raises(ValidationError, match="outcome label"):
        CausalSemanticDelta.model_validate(payload)

    payload = delta.model_dump(mode="python")
    payload["family"] = ScenarioFamily.FAILED_PRIOR_ATTEMPT
    payload["causal_delta_digest"] = causal_delta_digest(payload)
    with pytest.raises(ValidationError, match="family and public lineage key"):
        CausalSemanticDelta.model_validate(payload)


@pytest.mark.parametrize(
    ("source_field", "template_pointer", "message"),
    (
        (
            "semantic_replacements",
            "/memory_pool/0/statement",
            "semantic replacement pointer",
        ),
        (
            "evidence_replacements",
            "/event_pool/0/statement",
            "evidence replacement pointer",
        ),
    ),
)
def test_causal_delta_rejects_miscategorized_source_pointers(
    source_field: str,
    template_pointer: str,
    message: str,
) -> None:
    payload = _causal_delta().model_dump(mode="python")
    payload["semantic_replacements"] = ()
    payload["evidence_replacements"] = ()
    payload[source_field] = (
        {
            "template_pointer": template_pointer,
            "replacement": "Repository-adjusted event statement number 0.",
        },
    )
    payload["causal_delta_digest"] = causal_delta_digest(payload)
    with pytest.raises(ValidationError, match=message):
        CausalSemanticDelta.model_validate(payload)


@pytest.mark.parametrize(
    "replacement",
    (
        "This text calls the branch helpful.",
        "é" * 129,
    ),
)
def test_causal_replacement_rejects_labels_and_utf8_overflow(replacement: str) -> None:
    with pytest.raises(ValidationError):
        CausalTextReplacement(
            template_pointer="/event_pool/0/statement",
            replacement=replacement,
        )


def _generator_algorithm() -> PublicGeneratorAlgorithm:
    configuration = _generator_configuration()
    catalog = _profile_catalog()
    operations = tuple(PublicGeneratorOperation)
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-public-generator-algorithm/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "generator_version": "state-decay-v2-public-generator/v1",
        "generation_contract_digest": GENERATION_CONTRACT.contract_digest,
        "generator_configuration_digest": configuration.configuration_digest,
        "profile_catalog_digest": catalog.catalog_digest,
        "steps": tuple(
            PublicGeneratorStep(
                position=position,
                operation=operation,
                operator_id=f"public-generator-{operation.value}",
                operator_version="v1",
            )
            for position, operation in enumerate(operations)
        ),
        "semantic_pointer_allowlist": ("/event_pool/0/statement",),
        "evidence_pointer_allowlist": ("/memory_pool/0/statement",),
        "causal_delta_digest_domain": CAUSAL_DELTA_DIGEST_DOMAIN,
        "slot_materialization_rule": "prefix-pools-global-profile-and-ascii-padding/v1",
        "parameter_rendering_rule": "pivot-ordered-ascii-parameter-clause-before-padding/v1",
        "decisive_evidence_rule": "selected-prefixes-and-logical-action-zero/v1",
        "delta_rendering_rule": (
            "stable-template-pointer-equal-utf8-replacement-plus-global-padding/v1"
        ),
        "signal_materialization_rule": "closed-composite-to-attested-trace-fixture/v1",
        "signal_evaluation_rule": (
            "repository-normalized-final-boundary-four-real-five-reference/v1"
        ),
        "outcome_derivation_rule": "rendered-policy-digest-bounded-state-machine/v1",
    }
    values["algorithm_digest"] = generator_algorithm_digest(values)
    return PublicGeneratorAlgorithm.model_validate(values)


def test_generator_algorithm_is_global_generic_ordered_and_self_attesting() -> None:
    algorithm = _generator_algorithm()

    assert PublicGeneratorAlgorithm.model_validate_json(canonical_json(algorithm)) == algorithm
    assert tuple(PublicGeneratorAlgorithm.model_fields) == (
        "schema_version",
        "suite_id",
        "suite_version",
        "generator_version",
        "generation_contract_digest",
        "generator_configuration_digest",
        "profile_catalog_digest",
        "steps",
        "semantic_pointer_allowlist",
        "evidence_pointer_allowlist",
        "causal_delta_digest_domain",
        "slot_materialization_rule",
        "parameter_rendering_rule",
        "decisive_evidence_rule",
        "delta_rendering_rule",
        "signal_materialization_rule",
        "signal_evaluation_rule",
        "outcome_derivation_rule",
        "algorithm_digest",
    )
    assert tuple(step.operation for step in algorithm.steps) == tuple(PublicGeneratorOperation)
    serialized = canonical_json(algorithm).decode("utf-8")
    for forbidden in (
        "task_template",
        "lineage_registry_key",
        "causal_deltas",
        "outcome_delta_order",
    ):
        assert forbidden not in serialized

    payload = algorithm.model_dump(mode="python")
    payload["steps"] = tuple(reversed(payload["steps"]))
    payload["algorithm_digest"] = generator_algorithm_digest(payload)
    with pytest.raises(ValidationError, match="generator steps"):
        PublicGeneratorAlgorithm.model_validate(payload)

    payload = algorithm.model_dump(mode="python")
    payload["evidence_pointer_allowlist"] = payload["semantic_pointer_allowlist"]
    payload["algorithm_digest"] = generator_algorithm_digest(payload)
    with pytest.raises(ValidationError, match="pointer allowlists"):
        PublicGeneratorAlgorithm.model_validate(payload)


def test_generator_algorithm_rejects_unstable_or_miscategorized_pointers() -> None:
    algorithm = _generator_algorithm()
    payload = algorithm.model_dump(mode="python")
    payload["semantic_pointer_allowlist"] = ("/trajectory/0/statement",)
    payload["algorithm_digest"] = generator_algorithm_digest(payload)
    with pytest.raises(ValidationError, match="semantic pointer allowlist"):
        PublicGeneratorAlgorithm.model_validate(payload)

    payload = algorithm.model_dump(mode="python")
    payload["evidence_pointer_allowlist"] = ("/event_pool/1/statement",)
    payload["algorithm_digest"] = generator_algorithm_digest(payload)
    with pytest.raises(ValidationError, match="evidence pointer allowlist"):
        PublicGeneratorAlgorithm.model_validate(payload)


def _rendered_policy_bytes(
    skeleton: OutcomeFreeTaskSkeleton,
    replacements: tuple[RenderedCausalTextReplacement, ...],
) -> bytes:
    rendered = skeleton.model_dump(mode="python")
    for replacement in replacements:
        pointer = replacement.policy_pointer
        if pointer.startswith("/trajectory/"):
            index = int(pointer.split("/")[2])
            rendered["trajectory"][index]["statement"] = replacement.replacement
        elif pointer.startswith("/candidate_memories/"):
            index = int(pointer.split("/")[2])
            rendered["candidate_memories"][index]["statement"] = replacement.replacement
        elif pointer == "/pivot/statement":
            rendered["pivot"]["statement"] = replacement.replacement
        elif pointer.startswith("/allowed_actions/"):
            index = int(pointer.split("/")[2])
            rendered["allowed_actions"][index]["statement"] = replacement.replacement
        else:
            raise AssertionError("test replacement pointer is unsupported")
    return canonical_json(OutcomeFreeTaskSkeleton.model_validate(rendered))


def _rendered_policy_digest_from_payload(
    skeleton_payload: Mapping[str, object],
    delta_payload: Mapping[str, object],
) -> str:
    skeleton = OutcomeFreeTaskSkeleton.model_validate(skeleton_payload)
    delta = RenderedCausalSemanticDelta.model_validate(delta_payload)
    replacements = (*delta.semantic_replacements, *delta.evidence_replacements)
    return rendered_policy_digest(_rendered_policy_bytes(skeleton, replacements))


def _rendered_delta(
    delta: CausalSemanticDelta,
    *,
    slot: int,
    skeleton: OutcomeFreeTaskSkeleton,
) -> RenderedCausalSemanticDelta:
    source = delta.semantic_replacements[0]
    replacements = (
        RenderedCausalTextReplacement(
            policy_pointer="/trajectory/0/statement",
            replacement=source.replacement + " " * slot,
        ),
    )
    return RenderedCausalSemanticDelta(
        delta_index=delta.delta_index,
        delta_id=delta.delta_id,
        causal_delta_digest=delta.causal_delta_digest,
        factor_values=delta.factor_values,
        semantic_replacements=replacements,
        evidence_replacements=(),
        rendered_policy_digest=rendered_policy_digest(
            _rendered_policy_bytes(skeleton, replacements)
        ),
    )


@pytest.mark.parametrize(
    ("source_field", "policy_pointer", "message"),
    (
        (
            "semantic_replacements",
            "/candidate_memories/0/statement",
            "semantic replacement pointer",
        ),
        (
            "evidence_replacements",
            "/trajectory/0/statement",
            "evidence replacement pointer",
        ),
    ),
)
def test_rendered_delta_rejects_miscategorized_policy_pointers(
    source_field: str,
    policy_pointer: str,
    message: str,
) -> None:
    assert tuple(RenderedCausalSemanticDelta.model_fields) == (
        "delta_index",
        "delta_id",
        "causal_delta_digest",
        "factor_values",
        "semantic_replacements",
        "evidence_replacements",
        "rendered_policy_digest",
    )
    values: dict[str, object] = {
        "delta_index": 0,
        "delta_id": "delta-pub-fr-00-0",
        "causal_delta_digest": _causal_delta().causal_delta_digest,
        "factor_values": _causal_factor_values(_CAUSAL_FACTOR_VECTORS[0]),
        "semantic_replacements": (),
        "evidence_replacements": (),
        "rendered_policy_digest": "0" * 64,
    }
    values[source_field] = (
        RenderedCausalTextReplacement(
            policy_pointer=policy_pointer,
            replacement="A distinct rendered statement.",
        ),
    )
    with pytest.raises(ValidationError, match=message):
        RenderedCausalSemanticDelta.model_validate(values)


def _preview(
    *,
    slot: int,
    deltas: tuple[CausalSemanticDelta, ...],
    graph: PublicTransitionGraph,
    topology: PublicEvidenceTopology,
    failure: PublicFailureMechanism,
    signature: PublicSemanticSignature,
    algorithm: PublicGeneratorAlgorithm,
    catalog: PublicProfileCatalog,
) -> PreAllocationSkeletonPreview:
    profile = catalog.slot_profiles[slot]
    skeleton = _slot_outcome_free_skeleton(profile)
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-pre-allocation-skeleton-preview/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "split": BenchmarkSplit.TRAIN,
        "family": ScenarioFamily.FORGOTTEN_REQUIREMENT,
        "lineage_registry_key": "pub-fr-00",
        "generator_slot": slot,
        "generator_version": "state-decay-v2-public-generator/v1",
        "generation_contract_digest": GENERATION_CONTRACT.contract_digest,
        "generator_configuration_digest": algorithm.generator_configuration_digest,
        "generator_algorithm_digest": algorithm.algorithm_digest,
        "profile_catalog_digest": catalog.catalog_digest,
        "transition_graph_digest": graph.transition_graph_digest,
        "evidence_topology_digest": topology.evidence_topology_digest,
        "failure_mechanism_id": failure.failure_mechanism_id,
        "semantic_signature_digest": signature.semantic_signature_digest,
        "slot_profile": profile,
        "allowed_parameter_values": profile.parameters.allowed_values,
        "task_skeleton": skeleton,
        "trace_fixture": _trace_fixture(profile, skeleton),
        "rendered_causal_deltas": tuple(
            _rendered_delta(delta, slot=slot, skeleton=skeleton) for delta in deltas
        ),
    }
    values["preview_digest"] = skeleton_preview_digest(values)
    return PreAllocationSkeletonPreview.model_validate(values)


def _candidate() -> PublicLineageCandidate:
    catalog = _profile_catalog()
    algorithm = _generator_algorithm()
    graph, topology, failure, signature = _semantic_artifacts()
    deltas = tuple(_causal_delta(index=index) for index in range(4))
    lineage_key = "pub-fr-00"
    public_leaf = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.PUBLIC)
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-public-lineage-candidate/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "split": BenchmarkSplit.TRAIN,
        "family": ScenarioFamily.FORGOTTEN_REQUIREMENT,
        "lineage_registry_key": lineage_key,
        "generator_version": "state-decay-v2-public-generator/v1",
        "generation_contract_digest": GENERATION_CONTRACT.contract_digest,
        "generator_configuration_digest": algorithm.generator_configuration_digest,
        "generator_algorithm_digest": algorithm.algorithm_digest,
        "profile_catalog_digest": catalog.catalog_digest,
        "independent_seed_commitment_digest": independent_lineage_seed_commitment(
            derive_independent_lineage_seed(
                public_leaf,
                split=BenchmarkSplit.TRAIN,
                family=ScenarioFamily.FORGOTTEN_REQUIREMENT,
                lineage_registry_key=lineage_key,
            )
        ),
        "task_template": _task_template(),
        "transition_graph": graph,
        "evidence_topology": topology,
        "failure_mechanism": failure,
        "semantic_signature": signature,
        "causal_deltas": deltas,
        "semantic_rationale": "The candidate tests retention of an earlier constraint.",
        "derivation_parent_keys": (),
        "previews": tuple(
            _preview(
                slot=slot,
                deltas=deltas,
                graph=graph,
                topology=topology,
                failure=failure,
                signature=signature,
                algorithm=algorithm,
                catalog=catalog,
            )
            for slot in range(5)
        ),
    }
    values["candidate_packet_digest"] = candidate_packet_digest(values)
    return PublicLineageCandidate.model_validate(values)


def test_preview_and_candidate_are_outcome_free_bound_and_self_attesting() -> None:
    candidate = _candidate()

    assert PublicLineageCandidate.model_validate_json(canonical_json(candidate)) == candidate
    assert tuple(PublicLineageCandidate.model_fields) == (
        "schema_version",
        "suite_id",
        "suite_version",
        "split",
        "family",
        "lineage_registry_key",
        "generator_version",
        "generation_contract_digest",
        "generator_configuration_digest",
        "generator_algorithm_digest",
        "profile_catalog_digest",
        "independent_seed_commitment_digest",
        "task_template",
        "transition_graph",
        "evidence_topology",
        "failure_mechanism",
        "semantic_signature",
        "causal_deltas",
        "semantic_rationale",
        "derivation_parent_keys",
        "previews",
        "candidate_packet_digest",
    )
    assert tuple(PreAllocationSkeletonPreview.model_fields) == (
        "schema_version",
        "suite_id",
        "suite_version",
        "split",
        "family",
        "lineage_registry_key",
        "generator_slot",
        "generator_version",
        "generation_contract_digest",
        "generator_configuration_digest",
        "generator_algorithm_digest",
        "profile_catalog_digest",
        "transition_graph_digest",
        "evidence_topology_digest",
        "failure_mechanism_id",
        "semantic_signature_digest",
        "slot_profile",
        "allowed_parameter_values",
        "task_skeleton",
        "trace_fixture",
        "rendered_causal_deltas",
        "preview_digest",
    )
    assert tuple(candidate.previews[index].generator_slot for index in range(5)) == tuple(range(5))
    for preview in candidate.previews:
        assert len(preview.task_skeleton.trajectory) == 3 + preview.generator_slot
        assert len(preview.trace_fixture.events) == len(preview.task_skeleton.trajectory)
        assert tuple(memory.current_revision for memory in preview.trace_fixture.memories) == tuple(
            memory.revision for memory in preview.task_skeleton.candidate_memories
        )
        assert tuple(memory.validity for memory in preview.trace_fixture.memories) == tuple(
            memory.validity for memory in preview.task_skeleton.candidate_memories
        )
    assert len(candidate.previews[0].trace_fixture.bindings[2].assertions or ()) == 2
    assert len(candidate.previews[2].trace_fixture.bindings[4].assertions or ()) == 2
    assert {
        tuple(item.value for item in delta.factor_values) for delta in candidate.causal_deltas
    } == set(_CAUSAL_FACTOR_VECTORS)
    serialized = canonical_json(candidate).decode("utf-8")
    for forbidden in ('"outcome"', '"allocation', '"scenario_id"', '"reviewer'):
        assert forbidden not in serialized

    payload = candidate.model_dump(mode="python")
    payload["candidate_packet_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="candidate packet digest"):
        PublicLineageCandidate.model_validate(payload)

    payload = candidate.model_dump(mode="python")
    payload["independent_seed_commitment_digest"] = "0" * 64
    payload["candidate_packet_digest"] = candidate_packet_digest(payload)
    with pytest.raises(ValidationError, match="lineage seed commitment"):
        PublicLineageCandidate.model_validate(payload)

    payload = candidate.model_dump(mode="python")
    payload["derivation_parent_keys"] = ("pub-fr-01",)
    payload["candidate_packet_digest"] = candidate_packet_digest(payload)
    with pytest.raises(ValidationError):
        PublicLineageCandidate.model_validate(payload)

    payload = candidate.model_dump(mode="python")
    payload["outcome"] = ScenarioOutcome.HELPFUL
    with pytest.raises(ValidationError):
        PublicLineageCandidate.model_validate(payload)

    preview_payload = candidate.previews[0].model_dump(mode="python")
    preview_payload["scenario_id"] = "3" * 64
    with pytest.raises(ValidationError):
        PreAllocationSkeletonPreview.model_validate(preview_payload)

    payload = candidate.model_dump(mode="python")
    payload["previews"][0]["trace_fixture"]["trace_fixture_digest"] = "0" * 64
    payload["previews"][0]["preview_digest"] = skeleton_preview_digest(payload["previews"][0])
    payload["candidate_packet_digest"] = candidate_packet_digest(payload)
    with pytest.raises(ValidationError, match="trace fixture digest"):
        PublicLineageCandidate.model_validate(payload)


def test_candidate_rejects_recursive_outcome_labels_and_adapter_drift() -> None:
    candidate = _candidate()
    payload = candidate.model_dump(mode="python")
    payload["task_template"]["action_pool"][0]["action_id"] = "helpful-action"
    payload["candidate_packet_digest"] = candidate_packet_digest(payload)
    with pytest.raises(ValidationError, match="outcome label"):
        PublicLineageCandidate.model_validate(payload)

    payload = candidate.model_dump(mode="python")
    preview = payload["previews"][0]
    preview["task_skeleton"]["adapter"]["response_profile_digest"] = "2" * 64
    for delta in preview["rendered_causal_deltas"]:
        delta["rendered_policy_digest"] = _rendered_policy_digest_from_payload(
            preview["task_skeleton"],
            delta,
        )
    preview["preview_digest"] = skeleton_preview_digest(preview)
    payload["candidate_packet_digest"] = candidate_packet_digest(payload)
    with pytest.raises(ValidationError, match="adapter"):
        PublicLineageCandidate.model_validate(payload)


def test_candidate_rejects_uncontrolled_source_replacements() -> None:
    candidate = _candidate()
    payload = candidate.model_dump(mode="python")
    delta = payload["causal_deltas"][0]
    delta["semantic_replacements"][0]["replacement"] += "!"
    delta["causal_delta_digest"] = causal_delta_digest(delta)
    payload["candidate_packet_digest"] = candidate_packet_digest(payload)
    with pytest.raises(ValidationError, match="equal UTF-8 length"):
        PublicLineageCandidate.model_validate(payload)

    payload = candidate.model_dump(mode="python")
    source = "Repository-authored event statement number 0. "
    whitespace_only = "Repository-authored  event statement number 0."
    payload["task_template"]["event_pool"][0]["statement"] = source
    for index, delta in enumerate(payload["causal_deltas"]):
        delta["semantic_replacements"][0]["replacement"] = (
            whitespace_only
            if index == 0
            else delta["semantic_replacements"][0]["replacement"] + " "
        )
        delta["causal_delta_digest"] = causal_delta_digest(delta)
    payload["candidate_packet_digest"] = candidate_packet_digest(payload)
    with pytest.raises(ValidationError, match="whitespace-only"):
        PublicLineageCandidate.model_validate(payload)

    payload = candidate.model_dump(mode="python")
    payload["task_template"]["event_pool"][0]["statement"] = (
        "Repository-Authored event statement number 0."
    )
    payload["causal_deltas"][0]["semantic_replacements"][0]["replacement"] = (
        "repository-Authored event statement number 0."
    )
    payload["causal_deltas"][0]["causal_delta_digest"] = causal_delta_digest(
        payload["causal_deltas"][0]
    )
    payload["candidate_packet_digest"] = candidate_packet_digest(payload)
    with pytest.raises(ValidationError, match="case-only"):
        PublicLineageCandidate.model_validate(payload)

    payload = candidate.model_dump(mode="python")
    payload["task_template"]["event_pool"][0]["statement"] = (
        "Repository-authored\N{NO-BREAK SPACE}event statement number 0."
    )
    payload["causal_deltas"][0]["semantic_replacements"][0]["replacement"] = (
        "Repository-adjusted\N{NO-BREAK SPACE}event statement number 0."
    )
    payload["causal_deltas"][0]["causal_delta_digest"] = causal_delta_digest(
        payload["causal_deltas"][0]
    )
    payload["candidate_packet_digest"] = candidate_packet_digest(payload)
    with pytest.raises(ValidationError, match=r"non-U\+0020 whitespace"):
        PublicLineageCandidate.model_validate(payload)


def test_candidate_rejects_rendered_replacement_drift() -> None:
    candidate = _candidate()
    payload = candidate.model_dump(mode="python")
    preview = payload["previews"][0]
    delta = preview["rendered_causal_deltas"][0]
    delta["semantic_replacements"][0]["replacement"] += "!"
    delta["rendered_policy_digest"] = _rendered_policy_digest_from_payload(
        preview["task_skeleton"],
        delta,
    )
    preview["preview_digest"] = skeleton_preview_digest(preview)
    payload["candidate_packet_digest"] = candidate_packet_digest(payload)
    with pytest.raises(ValidationError, match="equal UTF-8 length"):
        PublicLineageCandidate.model_validate(payload)

    payload = candidate.model_dump(mode="python")
    preview = payload["previews"][1]
    delta = preview["rendered_causal_deltas"][0]
    delta["semantic_replacements"][0]["replacement"] = (
        "Repository-authored  event statement number 0."
    )
    delta["rendered_policy_digest"] = _rendered_policy_digest_from_payload(
        preview["task_skeleton"],
        delta,
    )
    preview["preview_digest"] = skeleton_preview_digest(preview)
    payload["candidate_packet_digest"] = candidate_packet_digest(payload)
    with pytest.raises(ValidationError, match="whitespace-only"):
        PublicLineageCandidate.model_validate(payload)

    payload = candidate.model_dump(mode="python")
    preview = payload["previews"][0]
    delta = preview["rendered_causal_deltas"][0]
    delta["semantic_replacements"][0]["policy_pointer"] = "/trajectory/1/statement"
    delta["rendered_policy_digest"] = _rendered_policy_digest_from_payload(
        preview["task_skeleton"],
        delta,
    )
    preview["preview_digest"] = skeleton_preview_digest(preview)
    payload["candidate_packet_digest"] = candidate_packet_digest(payload)
    with pytest.raises(ValidationError, match="materialization pointer"):
        PublicLineageCandidate.model_validate(payload)


def test_candidate_rejects_stale_preview_source_bindings() -> None:
    candidate = _candidate()
    payload = candidate.model_dump(mode="python")
    preview = payload["previews"][0]
    preview["rendered_causal_deltas"][0]["delta_id"] = "different-delta"
    preview["preview_digest"] = skeleton_preview_digest(preview)
    payload["candidate_packet_digest"] = candidate_packet_digest(payload)
    with pytest.raises(ValidationError, match="preview causal delta bindings"):
        PublicLineageCandidate.model_validate(payload)

    payload = candidate.model_dump(mode="python")
    preview = payload["previews"][0]
    factor_value = preview["rendered_causal_deltas"][0]["factor_values"][0]
    factor_value["value"] = not factor_value["value"]
    preview["preview_digest"] = skeleton_preview_digest(preview)
    payload["candidate_packet_digest"] = candidate_packet_digest(payload)
    with pytest.raises(ValidationError, match="preview causal delta bindings"):
        PublicLineageCandidate.model_validate(payload)


@pytest.mark.parametrize(
    ("replacement_kind", "message"), (("noop", "no-op"), ("shared", "pairwise distinct"))
)
def test_candidate_rejects_noop_or_identical_rendered_policies(
    replacement_kind: str,
    message: str,
) -> None:
    payload = _candidate().model_dump(mode="python")
    if replacement_kind == "shared":
        for delta in payload["causal_deltas"]:
            delta["semantic_replacements"][0]["replacement"] = (
                "Repository-adjusted event statement number 0."
            )
            delta["causal_delta_digest"] = causal_delta_digest(delta)
    for preview in payload["previews"]:
        slot = preview["generator_slot"]
        replacement = (
            f"Repository-authored event statement number 0.{' ' * slot}"
            if replacement_kind == "noop"
            else f"Repository-adjusted event statement number 0.{' ' * slot}"
        )
        for index, delta in enumerate(preview["rendered_causal_deltas"]):
            if replacement_kind == "shared":
                delta["causal_delta_digest"] = payload["causal_deltas"][index][
                    "causal_delta_digest"
                ]
            delta["semantic_replacements"][0]["replacement"] = replacement
            delta["rendered_policy_digest"] = _rendered_policy_digest_from_payload(
                preview["task_skeleton"],
                delta,
            )
        preview["preview_digest"] = skeleton_preview_digest(preview)
    payload["candidate_packet_digest"] = candidate_packet_digest(payload)
    with pytest.raises(ValidationError, match=message):
        PublicLineageCandidate.model_validate(payload)


def test_candidate_rejects_an_unallowlisted_rendered_pointer() -> None:
    payload = _candidate().model_dump(mode="python")
    preview = payload["previews"][0]
    preview["rendered_causal_deltas"][0]["semantic_replacements"][0]["policy_pointer"] = (
        "/trajectory/7/statement"
    )
    preview["preview_digest"] = skeleton_preview_digest(preview)
    payload["candidate_packet_digest"] = candidate_packet_digest(payload)

    with pytest.raises(ValidationError, match="semantic replacement pointer is not allowed"):
        PublicLineageCandidate.model_validate(payload)


def test_role_neutral_parts_repeat_policy_invariants_and_forbid_metadata() -> None:
    skeleton = _outcome_free_skeleton()
    parts = RoleNeutralGeneratedParts(
        split=BenchmarkSplit.TRAIN,
        scenario_id="2" * 64,
        **skeleton.model_dump(mode="python"),
    )
    assert RoleNeutralGeneratedParts.model_validate_json(canonical_json(parts)) == parts
    assert tuple(RoleNeutralGeneratedParts.model_fields) == (
        "schema_version",
        "suite_id",
        "suite_version",
        "split",
        "scenario_id",
        "trajectory",
        "candidate_memories",
        "pivot",
        "allowed_actions",
        "adapter",
    )

    payload = parts.model_dump(mode="python")
    payload["outcome"] = ScenarioOutcome.HELPFUL
    with pytest.raises(ValidationError):
        RoleNeutralGeneratedParts.model_validate(payload)

    payload = parts.model_dump(mode="python")
    payload["candidate_memories"][0]["evidence_refs"][0]["event_id"] = "missing"
    with pytest.raises(ValidationError, match="evidence reference"):
        RoleNeutralGeneratedParts.model_validate(payload)


def test_registry_contract_freezes_complete_geometry_and_review_protocol_binding() -> None:
    assert tuple(PublicLineageRegistry.model_fields) == (
        "schema_version",
        "suite_id",
        "suite_version",
        "generation_contract_digest",
        "lineage_review_protocol_digest",
        "generator_configuration_digest",
        "generator_algorithm_digest",
        "profile_catalog",
        "candidates",
        "registry_digest",
    )
    with pytest.raises(ValidationError):
        PublicLineageRegistry(
            generation_contract_digest=GENERATION_CONTRACT.contract_digest,
            lineage_review_protocol_digest=LINEAGE_REVIEW_PROTOCOL.protocol_digest,
            generator_configuration_digest="1" * 64,
            generator_algorithm_digest="2" * 64,
            profile_catalog=_profile_catalog(),
            candidates=(),
            registry_digest="3" * 64,
        )


def test_public_text_key_and_digest_defenses_cover_noncanonical_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(public_contract_module, "_is_forbidden_code_point", lambda code: False)
        with pytest.raises(ValueError, match="review-safe text is invalid"):
            public_contract_module.validate_review_safe_text("\ud800")

    with pytest.raises(ValueError, match="rendered causal replacement"):
        public_contract_module._rendered_causal_replacement_text("This would be a helpful result.")
    with pytest.raises(ValueError, match="lineage key"):
        parse_public_lineage_key(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lineage key"):
        public_contract_module._public_lineage_key(1)

    with monkeypatch.context() as patch:
        patch.setattr(public_contract_module, "public_lineage_key", lambda family, index: "other")
        with pytest.raises(ValueError, match="lineage key"):
            parse_public_lineage_key("pub-fr-00")

    with pytest.raises(ValueError, match="digest payload is invalid"):
        transition_graph_digest(object())  # type: ignore[arg-type]


def test_task_template_and_memory_validators_reject_every_identity_and_validity_drift() -> None:
    template = _task_template()
    for pool_name, duplicate_index, message in (
        ("event_pool", 1, "event identifiers"),
        ("memory_pool", 1, "memory identifiers"),
        ("action_pool", 1, "action identifiers"),
    ):
        payload = template.model_dump(mode="python")
        identifier_field = {
            "event_pool": "event_id",
            "memory_pool": "memory_id",
            "action_pool": "action_id",
        }[pool_name]
        payload[pool_name][duplicate_index][identifier_field] = payload[pool_name][0][
            identifier_field
        ]
        with pytest.raises(ValidationError, match=message):
            OutcomeFreeTaskTemplate.model_validate(payload)

    memory = _outcome_free_skeleton().candidate_memories[0]
    payload = memory.model_dump(mode="python")
    payload["evidence_refs"] = (*payload["evidence_refs"], payload["evidence_refs"][0])
    with pytest.raises(ValidationError, match="evidence references"):
        OutcomeFreeCandidateMemory.model_validate(payload)

    payload = memory.model_dump(mode="python")
    payload["validity_sequence"] = memory.recorded_sequence
    payload["validity_action_step"] = memory.recorded_action_step
    with pytest.raises(ValidationError, match="active memory"):
        OutcomeFreeCandidateMemory.model_validate(payload)

    payload = memory.model_dump(mode="python")
    payload["validity"] = ValidityState.SUPERSEDED
    with pytest.raises(ValidationError, match="complete validity transition"):
        OutcomeFreeCandidateMemory.model_validate(payload)

    payload = memory.model_dump(mode="python")
    payload["validity"] = ValidityState.SUPERSEDED
    payload["validity_sequence"] = memory.recorded_sequence - 1
    payload["validity_action_step"] = memory.recorded_action_step - 1
    with pytest.raises(ValidationError, match="predates"):
        OutcomeFreeCandidateMemory.model_validate(payload)


def test_task_skeleton_rejects_order_identity_temporal_and_reference_drifts() -> None:
    skeleton = _outcome_free_skeleton()

    payload = skeleton.model_dump(mode="python")
    payload["trajectory"][1]["sequence"] = payload["trajectory"][0]["sequence"]
    with pytest.raises(ValidationError, match="sequence values"):
        OutcomeFreeTaskSkeleton.model_validate(payload)

    payload = skeleton.model_dump(mode="python")
    payload["trajectory"][1]["action_step"] = 3
    with pytest.raises(ValidationError, match="action steps"):
        OutcomeFreeTaskSkeleton.model_validate(payload)

    payload = skeleton.model_dump(mode="python")
    payload["trajectory"][1]["event_id"] = payload["trajectory"][0]["event_id"]
    with pytest.raises(ValidationError, match="event identifiers"):
        OutcomeFreeTaskSkeleton.model_validate(payload)

    payload = skeleton.model_dump(mode="python")
    payload["candidate_memories"] = (
        payload["candidate_memories"][0],
        payload["candidate_memories"][0],
    )
    with pytest.raises(ValidationError, match="memory identifiers"):
        OutcomeFreeTaskSkeleton.model_validate(payload)

    payload = skeleton.model_dump(mode="python")
    payload["candidate_memories"][0]["recorded_sequence"] = 4
    with pytest.raises(ValidationError, match="visible prefix"):
        OutcomeFreeTaskSkeleton.model_validate(payload)

    payload = skeleton.model_dump(mode="python")
    payload["candidate_memories"][0]["evidence_refs"] = (
        {"event_id": "event-3", "event_sequence": 3},
    )
    with pytest.raises(ValidationError, match="from the future"):
        OutcomeFreeTaskSkeleton.model_validate(payload)

    payload = skeleton.model_dump(mode="python")
    payload["candidate_memories"][0].update(
        validity=ValidityState.SUPERSEDED,
        validity_sequence=5,
        validity_action_step=2,
    )
    with pytest.raises(ValidationError, match="after the pivot"):
        OutcomeFreeTaskSkeleton.model_validate(payload)

    payload = skeleton.model_dump(mode="python")
    payload["allowed_actions"] = (
        payload["allowed_actions"][0],
        payload["allowed_actions"][0],
    )
    with pytest.raises(ValidationError, match="action identifiers"):
        OutcomeFreeTaskSkeleton.model_validate(payload)


def test_causal_generator_and_profile_components_reject_closed_contract_drift() -> None:
    with pytest.raises(ValidationError, match="outcome label"):
        PublicCausalFactor(
            factor_id="helpful-factor",
            true_description="The condition applies.",
            false_description="The condition does not apply.",
        )
    with pytest.raises(ValidationError, match="descriptions must differ"):
        PublicCausalFactor(
            factor_id="factor",
            true_description="The condition is unchanged.",
            false_description="The condition is unchanged.",
        )

    delta = _causal_delta()
    payload = delta.model_dump(mode="python")
    payload["factor_values"] = (
        payload["factor_values"][0],
        payload["factor_values"][0],
    )
    with pytest.raises(ValidationError, match="assignments must be unique"):
        CausalSemanticDelta.model_validate(payload)

    payload = delta.model_dump(mode="python")
    payload["semantic_replacements"] = ()
    payload["evidence_replacements"] = ()
    with pytest.raises(ValidationError, match="between one and four"):
        CausalSemanticDelta.model_validate(payload)

    algorithm = _generator_algorithm()
    payload = algorithm.model_dump(mode="python")
    payload["generation_contract_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="generation contract"):
        PublicGeneratorAlgorithm.model_validate(payload)

    payload = algorithm.model_dump(mode="python")
    payload["steps"][1]["operator_id"] = payload["steps"][0]["operator_id"]
    payload["algorithm_digest"] = generator_algorithm_digest(payload)
    with pytest.raises(ValidationError, match="operator identifiers"):
        PublicGeneratorAlgorithm.model_validate(payload)

    configuration = _generator_configuration()
    payload = configuration.model_dump(mode="python")
    payload["generation_contract_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="generation contract"):
        PublicGeneratorConfiguration.model_validate(payload)

    payload = configuration.model_dump(mode="python")
    payload["visible_families"] = tuple(reversed(payload["visible_families"]))
    payload["configuration_digest"] = generator_configuration_digest(payload)
    with pytest.raises(ValidationError, match="visible families"):
        PublicGeneratorConfiguration.model_validate(payload)

    payload = configuration.model_dump(mode="python")
    payload["nuisance_inventory_digest"] = "0" * 64
    payload["configuration_digest"] = generator_configuration_digest(payload)
    with pytest.raises(ValidationError, match="nuisance inventory"):
        PublicGeneratorConfiguration.model_validate(payload)

    parameter = PublicParameterValue(parameter_id="parameter", value=1)
    with pytest.raises(ValidationError, match="identifiers must be unique"):
        PublicParameterProfile(
            profile_id="parameters",
            allowed_values=(parameter, parameter),
        )
    with pytest.raises(ValidationError, match="decisive evidence is empty"):
        PublicEvidenceProfile(
            profile_id="evidence",
            evidence_reference_count=1,
            decisive_event_count=0,
            decisive_memory_count=0,
        )


def test_expected_evidence_and_assertion_helpers_reject_ambiguous_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = PublicExpectedMemoryEvidence(memory_pool_index=0, revision=1)
    with pytest.raises(ValidationError, match="references are not canonical"):
        PublicExpectedDetectorEvidence(
            event_pool_indices=(0,),
            binding_event_pool_indices=(0,),
            memory_references=(memory, memory),
            assertion_references=(),
        )
    with pytest.raises(ValidationError, match="binding evidence"):
        PublicExpectedDetectorEvidence(
            event_pool_indices=(0,),
            binding_event_pool_indices=(1,),
            memory_references=(),
            assertion_references=(),
        )

    with pytest.raises(ValueError, match="invalid type"):
        public_contract_module.public_assertion_fixture_digest(object())  # type: ignore[arg-type]

    assertion = PublicAssertionFixture(
        subject_id="subject",
        predicate_id="predicate",
        value_digest="1" * 64,
        precedence=1,
        revision=1,
        supersedes_assertion_digest=None,
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            public_contract_module,
            "canonical_json",
            lambda value: (_ for _ in ()).throw(RuntimeError("private-json")),
        )
        with pytest.raises(ValueError, match="fixture is invalid"):
            public_contract_module.public_assertion_fixture_digest(assertion)


def test_trace_fixture_components_reject_noncanonical_collections_and_bounds() -> None:
    with pytest.raises(ValidationError, match="payload is not bounded"):
        PublicFixtureEvent(
            event_pool_index=1,
            event_type=EventType.OBSERVATION,
            phase=EventPhase.POST_ACTION,
            payload={"text": "x" * (1024 * 1024 + 1)},
            parent_event_pool_indices=(0,),
        )
    with pytest.raises(ValidationError, match="parents are not canonical"):
        PublicFixtureEvent(
            event_pool_index=2,
            event_type=EventType.OBSERVATION,
            phase=EventPhase.POST_ACTION,
            payload={},
            parent_event_pool_indices=(0, 0),
        )

    constraint = PublicConstraintReferenceFixture(memory_pool_index=0, revision=1)
    assertion = PublicAssertionFixture(
        subject_id="subject",
        predicate_id="predicate",
        value_digest="1" * 64,
        precedence=1,
        revision=1,
        supersedes_assertion_digest=None,
    )
    binding_values: dict[str, object] = {
        "event_pool_index": 2,
        "action_step": 1,
        "scope_id": "scope",
        "progress_marker_digest": "2" * 64,
        "constraint_references": (constraint,),
        "impact": PublicImpactClass.REVERSIBLE,
        "authorization_event_pool_indices": (0,),
        "safeguard_event_pool_indices": (1,),
        "assertions": (assertion,),
    }
    with pytest.raises(ValidationError, match="constraint references"):
        PublicBindingFixture.model_validate(
            {**binding_values, "constraint_references": (constraint, constraint)}
        )
    with pytest.raises(ValidationError, match="authorization references"):
        PublicBindingFixture.model_validate(
            {**binding_values, "authorization_event_pool_indices": (1, 0)}
        )
    with pytest.raises(ValidationError, match="assertions must be unique"):
        PublicBindingFixture.model_validate(
            {**binding_values, "assertions": (assertion, assertion)}
        )

    memory_values: dict[str, object] = {
        "memory_pool_index": 0,
        "kind": ClaimKind.REQUIREMENT,
        "current_revision": 1,
        "validity": ValidityState.ACTIVE,
        "provenance_event_pool_indices": (1,),
        "expires_at_event_pool_index": None,
    }
    with pytest.raises(ValidationError, match="provenance is not canonical"):
        PublicDetectorMemoryFixture.model_validate(
            {**memory_values, "provenance_event_pool_indices": (1, 1)}
        )
    with pytest.raises(ValidationError, match="expires before"):
        PublicDetectorMemoryFixture.model_validate(
            {**memory_values, "expires_at_event_pool_index": 0}
        )


def test_trace_fixture_rejects_unresolved_future_and_shortcut_material() -> None:
    profile = _slot_profile(0)
    skeleton = _slot_outcome_free_skeleton(profile)
    fixture = _trace_fixture(profile, skeleton)

    payload = fixture.model_dump(mode="python")
    payload["events"][0]["event_pool_index"] = 1
    payload["trace_fixture_digest"] = trace_fixture_digest(payload)
    with pytest.raises(ValidationError, match="canonical prefix"):
        OutcomeFreeTraceFixture.model_validate(payload)

    second_profile = _slot_profile(1)
    second_skeleton = _slot_outcome_free_skeleton(second_profile)
    second_fixture = _trace_fixture(second_profile, second_skeleton)
    payload = second_fixture.model_dump(mode="python")
    payload["memories"][1]["memory_pool_index"] = 0
    payload["trace_fixture_digest"] = trace_fixture_digest(payload)
    with pytest.raises(ValidationError, match="memories are not canonical"):
        OutcomeFreeTraceFixture.model_validate(payload)

    payload = fixture.model_dump(mode="python")
    payload["bindings"][0]["authorization_event_pool_indices"] = (1,)
    payload["trace_fixture_digest"] = trace_fixture_digest(payload)
    with pytest.raises(ValidationError, match="evidence comes from the future"):
        OutcomeFreeTraceFixture.model_validate(payload)

    payload = fixture.model_dump(mode="python")
    payload["memories"][0]["provenance_event_pool_indices"] = (7,)
    payload["trace_fixture_digest"] = trace_fixture_digest(payload)
    with pytest.raises(ValidationError, match="provenance does not resolve"):
        OutcomeFreeTraceFixture.model_validate(payload)

    payload = fixture.model_dump(mode="python")
    payload["memories"][0]["expires_at_event_pool_index"] = 7
    payload["trace_fixture_digest"] = trace_fixture_digest(payload)
    with pytest.raises(ValidationError, match="expiry does not resolve"):
        OutcomeFreeTraceFixture.model_validate(payload)

    payload = fixture.model_dump(mode="python")
    payload["events"][0]["payload"] = {"profile_id": "shortcut"}
    payload["trace_fixture_digest"] = trace_fixture_digest(payload)
    with pytest.raises(ValidationError, match="policy shortcut marker"):
        OutcomeFreeTraceFixture.model_validate(payload)


def test_slot_profile_rejects_dimension_and_signal_reference_overflow() -> None:
    profile = _slot_profile(0)

    payload = profile.model_dump(mode="python")
    payload["evidence"]["decisive_event_count"] = 4
    with pytest.raises(ValidationError, match="decisive events"):
        PublicSlotProfile.model_validate(payload)

    payload = profile.model_dump(mode="python")
    payload["evidence"]["decisive_memory_count"] = 2
    with pytest.raises(ValidationError, match="decisive memories"):
        PublicSlotProfile.model_validate(payload)

    second = _slot_profile(1)
    payload = second.model_dump(mode="python")
    payload["evidence"]["evidence_reference_count"] = 1
    with pytest.raises(ValidationError, match="reference total"):
        PublicSlotProfile.model_validate(payload)

    payload = profile.model_dump(mode="python")
    conflict = next(
        item
        for item in payload["signals"]["expected_signals"]
        if item["signal_type"] is SignalType.CONFLICT
    )
    conflict["evidence"]["event_pool_indices"] = (7,)
    conflict["evidence"]["binding_event_pool_indices"] = (7,)
    conflict["evidence"]["assertion_references"] = tuple(
        {
            **assertion,
            "binding_event_pool_index": 7,
        }
        for assertion in conflict["evidence"]["assertion_references"]
    )
    payload["signals"]["profile_digest"] = signal_profile_digest(payload["signals"])
    with pytest.raises(ValidationError, match="signal evidence exceeds"):
        PublicSlotProfile.model_validate(payload)

    payload = second.model_dump(mode="python")
    context_signal = next(
        item
        for item in payload["signals"]["expected_signals"]
        if item["signal_type"] is SignalType.CONTEXT_SHIFT
    )
    context_signal["evidence"]["memory_references"][0]["memory_pool_index"] = 3
    payload["signals"]["profile_digest"] = signal_profile_digest(payload["signals"])
    with pytest.raises(ValidationError, match="memory evidence does not resolve"):
        PublicSlotProfile.model_validate(payload)


def test_profile_catalog_rejects_protocol_global_identity_and_digest_drift() -> None:
    catalog = _profile_catalog()

    payload = catalog.model_dump(mode="python")
    payload["generation_contract_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="generation contract"):
        PublicProfileCatalog.model_validate(payload)

    payload = catalog.model_dump(mode="python")
    payload["counterbalance_axes"] = tuple(reversed(payload["counterbalance_axes"]))
    payload["catalog_digest"] = profile_catalog_digest(payload)
    with pytest.raises(ValidationError, match="counterbalance axes"):
        PublicProfileCatalog.model_validate(payload)

    payload = catalog.model_dump(mode="python")
    payload["slot_profiles"][1]["counterbalance"]["profile_id"] = payload["slot_profiles"][0][
        "counterbalance"
    ]["profile_id"]
    payload["catalog_digest"] = profile_catalog_digest(payload)
    with pytest.raises(ValidationError, match="globally unique"):
        PublicProfileCatalog.model_validate(payload)

    payload = catalog.model_dump(mode="python")
    payload["catalog_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="catalog digest"):
        PublicProfileCatalog.model_validate(payload)


def test_transition_execution_graph_and_semantic_models_reject_closed_world_drift() -> None:
    graph, topology, _, signature = _semantic_artifacts()
    factors = _causal_factor_values((False, False))

    with pytest.raises(ValidationError, match="vectors do not agree"):
        public_contract_module.PublicTransitionExecution(
            exposure=PublicCausalExposure.GUIDANCE_APPLIED,
            factor_values=factors,
            visited_state_ids=("initial", "middle"),
            action_fingerprint_ids=("action-1", "action-2"),
            failure_fingerprint_ids=(None,),
            terminal=PublicTerminalState.GOAL_REACHED,
            repeated_action_count=0,
            failure_loop_count=0,
        )

    payload = graph.model_dump(mode="python")
    payload["states"][1]["state_id"] = payload["states"][0]["state_id"]
    payload["transition_graph_digest"] = transition_graph_digest(payload)
    with pytest.raises(ValidationError, match="states must be unique"):
        PublicTransitionGraph.model_validate(payload)

    payload = graph.model_dump(mode="python")
    payload["initial_state_id"] = "missing-state"
    payload["transition_graph_digest"] = transition_graph_digest(payload)
    with pytest.raises(ValidationError, match="initial state does not resolve"):
        PublicTransitionGraph.model_validate(payload)

    terminal_state = next(state for state in graph.states if state.terminal is not None)
    payload = graph.model_dump(mode="python")
    payload["initial_state_id"] = terminal_state.state_id
    payload["transition_graph_digest"] = transition_graph_digest(payload)
    with pytest.raises(ValidationError, match="initial state must be nonterminal"):
        PublicTransitionGraph.model_validate(payload)

    payload = graph.model_dump(mode="python")
    terminal_index = next(
        index for index, state in enumerate(graph.states) if state.terminal is not None
    )
    payload["states"][terminal_index]["terminal"] = None
    payload["transition_graph_digest"] = transition_graph_digest(payload)
    with pytest.raises(ValidationError, match="terminal states are incomplete"):
        PublicTransitionGraph.model_validate(payload)

    payload = graph.model_dump(mode="python")
    payload["states"] = (
        *payload["states"],
        {
            "state_id": "unreachable-state",
            "description": "This state is intentionally unreachable.",
            "terminal": None,
        },
    )
    payload["transition_graph_digest"] = transition_graph_digest(payload)
    with pytest.raises(ValidationError, match="unreachable state"):
        PublicTransitionGraph.model_validate(payload)

    with pytest.raises(ValueError, match="factors are not canonical"):
        public_contract_module._execute_public_transition_graph_unchecked(
            graph,
            PublicCausalExposure.GUIDANCE_APPLIED,
            (factors[1], factors[0]),
        )
    graph_without_transitions = graph.model_copy(update={"transitions": ()})
    with pytest.raises(ValueError, match="exactly one matching guard"):
        public_contract_module._execute_public_transition_graph_unchecked(
            graph_without_transitions,
            PublicCausalExposure.GUIDANCE_APPLIED,
            factors,
        )
    with pytest.raises(ValueError, match="execution input is invalid"):
        public_contract_module.execute_public_transition_graph(
            graph,
            "guidance_applied",  # type: ignore[arg-type]
            factors,
        )

    topology_payload = topology.model_dump(mode="python")
    topology_payload["nodes"] = (
        topology_payload["nodes"][0],
        topology_payload["nodes"][0],
    )
    topology_payload["evidence_topology_digest"] = evidence_topology_digest(topology_payload)
    with pytest.raises(ValidationError, match="nodes must be unique"):
        PublicEvidenceTopology.model_validate(topology_payload)

    topology_payload = topology.model_dump(mode="python")
    topology_payload["edges"] = (
        topology_payload["edges"][0],
        topology_payload["edges"][0],
    )
    topology_payload["evidence_topology_digest"] = evidence_topology_digest(topology_payload)
    with pytest.raises(ValidationError, match="edges must be unique"):
        PublicEvidenceTopology.model_validate(topology_payload)

    topology_payload = topology.model_dump(mode="python")
    topology_payload["edges"][0]["target_evidence_id"] = "missing-evidence"
    topology_payload["evidence_topology_digest"] = evidence_topology_digest(topology_payload)
    with pytest.raises(ValidationError, match="endpoint does not resolve"):
        PublicEvidenceTopology.model_validate(topology_payload)

    signature_payload = signature.model_dump(mode="python")
    signature_payload["concept_ids"] = (
        signature_payload["concept_ids"][0],
        signature_payload["concept_ids"][0],
    )
    signature_payload["semantic_signature_digest"] = semantic_signature_digest(signature_payload)
    with pytest.raises(ValidationError, match="concepts must be unique"):
        PublicSemanticSignature.model_validate(signature_payload)

    signature_payload = signature.model_dump(mode="python")
    signature_payload["semantic_signature_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="signature digest"):
        PublicSemanticSignature.model_validate(signature_payload)


def test_rendered_delta_and_materialization_helpers_cover_all_pointer_and_failure_edges() -> None:
    candidate = _candidate()
    preview = candidate.previews[0]
    delta = preview.rendered_causal_deltas[0]

    payload = delta.model_dump(mode="python")
    payload["delta_id"] = "helpful-delta"
    with pytest.raises(ValidationError, match="identifier contains an outcome"):
        RenderedCausalSemanticDelta.model_validate(payload)

    payload = delta.model_dump(mode="python")
    payload["factor_values"] = (
        payload["factor_values"][0],
        payload["factor_values"][0],
    )
    with pytest.raises(ValidationError, match="assignments must be unique"):
        RenderedCausalSemanticDelta.model_validate(payload)

    payload = delta.model_dump(mode="python")
    payload["semantic_replacements"] = ()
    payload["evidence_replacements"] = ()
    with pytest.raises(ValidationError, match="between one and four"):
        RenderedCausalSemanticDelta.model_validate(payload)

    payload = delta.model_dump(mode="python")
    payload["evidence_replacements"] = payload["semantic_replacements"]
    with pytest.raises(ValidationError, match="unique and disjoint"):
        RenderedCausalSemanticDelta.model_validate(payload)

    template = candidate.task_template
    assert (
        public_contract_module._template_policy_statement(template, "/memory_pool/0/statement")
        == template.memory_pool[0].statement
    )
    assert (
        public_contract_module._template_policy_statement(template, "/pivot/statement")
        == template.pivot.statement
    )
    assert (
        public_contract_module._template_policy_statement(template, "/action_pool/0/statement")
        == template.action_pool[0].statement
    )
    with pytest.raises(ValueError, match="template pointer is not allowed"):
        public_contract_module._template_policy_statement(template, "/unknown/statement")
    with pytest.raises(ValueError, match="policy pointer does not resolve"):
        public_contract_module._rendered_policy_statement(
            preview.task_skeleton,
            "/trajectory/99/statement",
        )

    assert (
        public_contract_module._rendered_pointer_for_template_pointer(
            "/memory_pool/0/statement", (0, 1)
        )
        == "/candidate_memories/0/statement"
    )
    assert (
        public_contract_module._rendered_pointer_for_template_pointer("/pivot/statement", (0, 1))
        == "/pivot/statement"
    )
    assert (
        public_contract_module._rendered_pointer_for_template_pointer(
            "/action_pool/0/statement", (1, 0)
        )
        == "/allowed_actions/1/statement"
    )
    with pytest.raises(ValueError, match="template pointer is not allowed"):
        public_contract_module._rendered_pointer_for_template_pointer("/unknown/statement", (0, 1))

    profile = preview.slot_profile
    assert (
        public_contract_module._replacement_materialization_suffix(
            "/memory_pool/0/statement", profile
        )
        == " " * profile.text_lengths.memory_padding_spaces
    )
    assert public_contract_module._replacement_materialization_suffix(
        "/pivot/statement", profile
    ).startswith(" [parameters:")
    assert (
        public_contract_module._replacement_materialization_suffix(
            "/action_pool/0/statement", profile
        )
        == " " * profile.text_lengths.action_padding_spaces
    )
    with pytest.raises(ValueError, match="template pointer is not allowed"):
        public_contract_module._replacement_materialization_suffix("/unknown/statement", profile)

    no_op = delta.model_copy(update={"semantic_replacements": (), "evidence_replacements": ()})
    with pytest.raises(ValueError, match="no-op"):
        public_contract_module._rendered_policy_bytes(preview.task_skeleton, no_op)
    with pytest.raises(ValueError, match="materialization input is invalid"):
        public_contract_module.materialize_rendered_policy_skeleton(
            object(),  # type: ignore[arg-type]
            delta,
        )
    with pytest.raises(ValueError, match="materialization input is invalid"):
        public_contract_module.materialize_rendered_policy_skeleton(
            preview.task_skeleton,
            no_op,
        )
    stale_digest = delta.model_copy(update={"rendered_policy_digest": "0" * 64})
    with pytest.raises(ValueError, match="digest does not match"):
        public_contract_module.materialize_rendered_policy_skeleton(
            preview.task_skeleton,
            stale_digest,
        )


def test_preview_rejects_coordinate_profile_fixture_delta_and_digest_drift() -> None:
    candidate = _candidate()
    preview = candidate.previews[0]

    payload = preview.model_dump(mode="python")
    payload["split"] = BenchmarkSplit.DEVELOPMENT
    with pytest.raises(ValidationError, match="coordinates"):
        PreAllocationSkeletonPreview.model_validate(payload)

    payload = preview.model_dump(mode="python")
    payload["generation_contract_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="generation contract"):
        PreAllocationSkeletonPreview.model_validate(payload)

    payload = preview.model_dump(mode="python")
    payload["generator_slot"] = 1
    with pytest.raises(ValidationError, match="slot profile"):
        PreAllocationSkeletonPreview.model_validate(payload)

    payload = preview.model_dump(mode="python")
    payload["allowed_parameter_values"][0]["value"] += 1
    with pytest.raises(ValidationError, match="allowed parameter values"):
        PreAllocationSkeletonPreview.model_validate(payload)

    payload = preview.model_dump(mode="python")
    payload["slot_profile"]["structure"]["trajectory_event_count"] += 1
    with pytest.raises(ValidationError, match="structural profile"):
        PreAllocationSkeletonPreview.model_validate(payload)

    payload = preview.model_dump(mode="python")
    extra_event = {
        **payload["trace_fixture"]["events"][-1],
        "event_pool_index": 3,
        "parent_event_pool_indices": (2,),
    }
    extra_binding = {
        **payload["trace_fixture"]["bindings"][-1],
        "event_pool_index": 3,
    }
    payload["trace_fixture"]["events"] = (
        *payload["trace_fixture"]["events"],
        extra_event,
    )
    payload["trace_fixture"]["bindings"] = (
        *payload["trace_fixture"]["bindings"],
        extra_binding,
    )
    payload["trace_fixture"]["trace_fixture_digest"] = trace_fixture_digest(
        payload["trace_fixture"]
    )
    with pytest.raises(ValidationError, match="trace fixture does not match"):
        PreAllocationSkeletonPreview.model_validate(payload)

    payload = preview.model_dump(mode="python")
    payload["trace_fixture"]["bindings"][0]["action_step"] += 1
    payload["trace_fixture"]["trace_fixture_digest"] = trace_fixture_digest(
        payload["trace_fixture"]
    )
    with pytest.raises(ValidationError, match="action step"):
        PreAllocationSkeletonPreview.model_validate(payload)

    second = candidate.previews[1]
    payload = second.model_dump(mode="python")
    payload["trace_fixture"]["memories"] = payload["trace_fixture"]["memories"][:1]
    payload["trace_fixture"]["trace_fixture_digest"] = trace_fixture_digest(
        payload["trace_fixture"]
    )
    with pytest.raises(ValidationError, match="memories do not match"):
        PreAllocationSkeletonPreview.model_validate(payload)

    payload = preview.model_dump(mode="python")
    payload["trace_fixture"]["memories"][0]["current_revision"] += 1
    payload["trace_fixture"]["trace_fixture_digest"] = trace_fixture_digest(
        payload["trace_fixture"]
    )
    with pytest.raises(ValidationError, match="memory state"):
        PreAllocationSkeletonPreview.model_validate(payload)

    conflict_preview = candidate.previews[2]
    payload = conflict_preview.model_dump(mode="python")
    assertion_index = next(
        index
        for index, binding in enumerate(payload["trace_fixture"]["bindings"])
        if binding["assertions"]
    )
    payload["trace_fixture"]["bindings"][assertion_index]["assertions"] = ()
    payload["trace_fixture"]["trace_fixture_digest"] = trace_fixture_digest(
        payload["trace_fixture"]
    )
    with pytest.raises(ValidationError, match="assertion evidence"):
        PreAllocationSkeletonPreview.model_validate(payload)

    payload = preview.model_dump(mode="python")
    payload["rendered_causal_deltas"] = tuple(reversed(payload["rendered_causal_deltas"]))
    with pytest.raises(ValidationError, match="deltas are not canonical"):
        PreAllocationSkeletonPreview.model_validate(payload)

    payload = preview.model_dump(mode="python")
    payload["rendered_causal_deltas"][1]["delta_id"] = payload["rendered_causal_deltas"][0][
        "delta_id"
    ]
    with pytest.raises(ValidationError, match="bindings must be unique"):
        PreAllocationSkeletonPreview.model_validate(payload)

    payload = preview.model_dump(mode="python")
    payload["rendered_causal_deltas"][0]["rendered_policy_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="rendered policy digest"):
        PreAllocationSkeletonPreview.model_validate(payload)

    payload = preview.model_dump(mode="python")
    payload["preview_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="preview digest"):
        PreAllocationSkeletonPreview.model_validate(payload)


def test_candidate_rejects_coordinate_delta_factor_preview_and_source_binding_drift() -> None:
    candidate = _candidate()

    payload = candidate.model_dump(mode="python")
    payload["split"] = BenchmarkSplit.DEVELOPMENT
    with pytest.raises(ValidationError, match="coordinates"):
        PublicLineageCandidate.model_validate(payload)

    payload = candidate.model_dump(mode="python")
    payload["generation_contract_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="generation contract"):
        PublicLineageCandidate.model_validate(payload)

    payload = candidate.model_dump(mode="python")
    payload["causal_deltas"] = tuple(reversed(payload["causal_deltas"]))
    with pytest.raises(ValidationError, match="deltas are not canonical"):
        PublicLineageCandidate.model_validate(payload)

    payload = candidate.model_dump(mode="python")
    payload["causal_deltas"][1]["delta_id"] = payload["causal_deltas"][0]["delta_id"]
    payload["causal_deltas"][1]["causal_delta_digest"] = causal_delta_digest(
        payload["causal_deltas"][1]
    )
    with pytest.raises(ValidationError, match="bindings must be unique"):
        PublicLineageCandidate.model_validate(payload)

    payload = candidate.model_dump(mode="python")
    payload["causal_deltas"][0]["lineage_registry_key"] = "pub-fr-01"
    payload["causal_deltas"][0]["causal_delta_digest"] = causal_delta_digest(
        payload["causal_deltas"][0]
    )
    with pytest.raises(ValidationError, match="coordinates do not agree"):
        PublicLineageCandidate.model_validate(payload)

    payload = candidate.model_dump(mode="python")
    for delta_payload in payload["causal_deltas"]:
        delta_payload["factor_values"][0]["factor_id"] = "different-factor"
        delta_payload["causal_delta_digest"] = causal_delta_digest(delta_payload)
    with pytest.raises(ValidationError, match="factors do not match"):
        PublicLineageCandidate.model_validate(payload)

    payload = candidate.model_dump(mode="python")
    payload["causal_deltas"][3]["factor_values"] = payload["causal_deltas"][0]["factor_values"]
    payload["causal_deltas"][3]["causal_delta_digest"] = causal_delta_digest(
        payload["causal_deltas"][3]
    )
    with pytest.raises(ValidationError, match="factor vectors"):
        PublicLineageCandidate.model_validate(payload)

    payload = candidate.model_dump(mode="python")
    payload["previews"] = tuple(reversed(payload["previews"]))
    with pytest.raises(ValidationError, match="canonical slot order"):
        PublicLineageCandidate.model_validate(payload)

    payload = candidate.model_dump(mode="python")
    payload["previews"][0]["transition_graph_digest"] = "0" * 64
    payload["previews"][0]["preview_digest"] = skeleton_preview_digest(payload["previews"][0])
    with pytest.raises(ValidationError, match="source bindings"):
        PublicLineageCandidate.model_validate(payload)
