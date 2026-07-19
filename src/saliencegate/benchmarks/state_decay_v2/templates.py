from __future__ import annotations

from itertools import product
from typing import Final

from saliencegate.benchmarks.state_decay_v2.config import (
    GENERATION_CONTRACT,
    PUBLIC_GENERATION_SEED,
    SeedPurpose,
    derive_seed,
)
from saliencegate.benchmarks.state_decay_v2.preallocation import (
    materialize_preallocation_preview,
)
from saliencegate.benchmarks.state_decay_v2.protocol import (
    LINEAGE_REVIEW_PROTOCOL,
    NUISANCE_FEATURE_INVENTORY,
    derive_independent_lineage_seed,
    independent_lineage_seed_commitment,
)
from saliencegate.benchmarks.state_decay_v2.public_catalog import (
    PUBLIC_LINEAGE_DEFINITIONS,
    PublicLineageDefinition,
    build_public_profile_catalog,
)
from saliencegate.benchmarks.state_decay_v2.public_contract import (
    CAUSAL_DELTA_DIGEST_DOMAIN,
    CausalSemanticDelta,
    CausalTextReplacement,
    OutcomeFreeTaskTemplate,
    OutcomeFreeTemplateAction,
    OutcomeFreeTemplateEvent,
    OutcomeFreeTemplateMemory,
    OutcomeFreeTemplatePivot,
    PublicCausalExposure,
    PublicCausalFactor,
    PublicCausalFactorValue,
    PublicEvidenceEdge,
    PublicEvidenceNode,
    PublicEvidenceRelation,
    PublicEvidenceTopology,
    PublicFailureMechanism,
    PublicGeneratorAlgorithm,
    PublicGeneratorConfiguration,
    PublicGeneratorOperation,
    PublicGeneratorStep,
    PublicLineageCandidate,
    PublicLineageRegistry,
    PublicProfileCatalog,
    PublicSemanticSignature,
    PublicTerminalState,
    PublicTransition,
    PublicTransitionGraph,
    PublicTransitionState,
    candidate_packet_digest,
    candidate_registry_digest,
    causal_delta_digest,
    evidence_topology_digest,
    generator_algorithm_digest,
    generator_configuration_digest,
    semantic_signature_digest,
    transition_graph_digest,
)
from saliencegate.benchmarks.state_decay_v2.schema import (
    AdapterMetadata,
    BenchmarkSplit,
    ScenarioFamily,
)
from saliencegate.domain import canonical_json, length_prefixed_sha256

_PUBLIC_FAMILIES: Final[tuple[ScenarioFamily, ...]] = (
    ScenarioFamily.FORGOTTEN_REQUIREMENT,
    ScenarioFamily.FAILED_PRIOR_ATTEMPT,
    ScenarioFamily.NEGLECTED_SUBGOAL,
    ScenarioFamily.STALE_MEMORY,
    ScenarioFamily.STABLE_ENVIRONMENT_FACT,
    ScenarioFamily.RETAINED_DIAGNOSIS,
)
_PUBLIC_SPLIT_BY_FAMILY: Final[dict[ScenarioFamily, BenchmarkSplit]] = {
    ScenarioFamily.FORGOTTEN_REQUIREMENT: BenchmarkSplit.TRAIN,
    ScenarioFamily.FAILED_PRIOR_ATTEMPT: BenchmarkSplit.TRAIN,
    ScenarioFamily.NEGLECTED_SUBGOAL: BenchmarkSplit.TRAIN,
    ScenarioFamily.STALE_MEMORY: BenchmarkSplit.TRAIN,
    ScenarioFamily.STABLE_ENVIRONMENT_FACT: BenchmarkSplit.DEVELOPMENT,
    ScenarioFamily.RETAINED_DIAGNOSIS: BenchmarkSplit.DEVELOPMENT,
}
_SEMANTIC_POINTERS: Final[tuple[str, ...]] = (
    "/event_pool/0/statement",
    "/event_pool/1/statement",
    "/event_pool/2/statement",
    "/pivot/statement",
    "/action_pool/0/statement",
    "/action_pool/1/statement",
)
_EVIDENCE_POINTERS: Final[tuple[str, ...]] = ("/memory_pool/0/statement",)


def build_public_generator_configuration() -> PublicGeneratorConfiguration:
    payload: dict[str, object] = {
        "schema_version": "state-decay-v2-public-generator-configuration/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "generator_version": "state-decay-v2-public-generator/v1",
        "generation_contract_digest": GENERATION_CONTRACT.contract_digest,
        "visible_splits": (BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT),
        "visible_families": _PUBLIC_FAMILIES,
        "lineages_per_family": 30,
        "generator_slots_per_lineage": 5,
        "candidate_count": 180,
        "preview_count": 900,
        "maximum_review_text_utf8_bytes": 4_096,
        "legacy_repetition_window_events": 8,
        "policy_schema_version": "state-decay-policy-view/v2",
        "oracle_schema_version": "state-decay-oracle-vault-entry/v2",
        "analysis_schema_version": "state-decay-analysis-cluster-entry/v2",
        "nuisance_inventory_digest": NUISANCE_FEATURE_INVENTORY.inventory_digest,
        "configuration_digest": "0" * 64,
    }
    payload["configuration_digest"] = generator_configuration_digest(payload)
    return PublicGeneratorConfiguration.model_validate(payload)


def build_public_generator_algorithm(
    configuration: PublicGeneratorConfiguration,
    profile_catalog: PublicProfileCatalog,
) -> PublicGeneratorAlgorithm:
    steps = tuple(
        PublicGeneratorStep(
            position=position,
            operation=operation,
            operator_id=(
                "public-candidate-preview",
                "public-preview-scenario-id",
                "public-balanced-outcome-allocation",
                "public-causal-delta-selection",
                "public-role-neutral-composition",
            )[position],
            operator_version="v1",
        )
        for position, operation in enumerate(PublicGeneratorOperation)
    )
    payload: dict[str, object] = {
        "schema_version": "state-decay-v2-public-generator-algorithm/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "generator_version": configuration.generator_version,
        "generation_contract_digest": configuration.generation_contract_digest,
        "generator_configuration_digest": configuration.configuration_digest,
        "profile_catalog_digest": profile_catalog.catalog_digest,
        "steps": steps,
        "semantic_pointer_allowlist": _SEMANTIC_POINTERS,
        "evidence_pointer_allowlist": _EVIDENCE_POINTERS,
        "causal_delta_digest_domain": CAUSAL_DELTA_DIGEST_DOMAIN,
        "slot_materialization_rule": "prefix-pools-global-profile-and-ascii-padding/v1",
        "parameter_rendering_rule": ("pivot-ordered-ascii-parameter-clause-before-padding/v1"),
        "decisive_evidence_rule": "selected-prefixes-and-logical-action-zero/v1",
        "delta_rendering_rule": (
            "stable-template-pointer-equal-utf8-replacement-plus-global-padding/v1"
        ),
        "signal_materialization_rule": "closed-composite-to-attested-trace-fixture/v1",
        "signal_evaluation_rule": (
            "repository-normalized-final-boundary-four-real-five-reference/v1"
        ),
        "outcome_derivation_rule": "rendered-policy-digest-bounded-state-machine/v1",
        "algorithm_digest": "0" * 64,
    }
    payload["algorithm_digest"] = generator_algorithm_digest(payload)
    return PublicGeneratorAlgorithm.model_validate(payload)


_ADAPTER: Final[AdapterMetadata] = AdapterMetadata(
    adapter_id="state-decay-openai-response-adapter",
    adapter_version="v1",
    response_profile_id="strict-two-action-json",
    response_profile_digest=length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": "state-decay-public-response-profile/v1",
                "choice_field": "action_id",
                "allowed_choice_count": 2,
                "additional_fields": False,
            }
        ),
        domain="saliencegate:state-decay-v2:public-response-profile:v1",
    ),
    capabilities=("strict-json", "two-action-choice"),
)


def _task_template(definition: PublicLineageDefinition) -> OutcomeFreeTaskTemplate:
    key = definition.lineage_registry_key
    locus = definition.locus_id.replace("-", " ")
    events = (
        OutcomeFreeTemplateEvent(
            event_id=f"{key}-event-0",
            statement=f"Guidance=latent baseline=reserved for {key}.",
        ),
        OutcomeFreeTemplateEvent(
            event_id=f"{key}-event-1",
            statement=f"The {locus} record identifies {definition.retained_subject}.",
        ),
        OutcomeFreeTemplateEvent(
            event_id=f"{key}-event-2",
            statement=f"Evidence for {key} records the governing context.",
        ),
        OutcomeFreeTemplateEvent(
            event_id=f"{key}-event-3",
            statement=f"A routine check inspects the {locus} record.",
        ),
        OutcomeFreeTemplateEvent(
            event_id=f"{key}-event-4",
            statement=f"The current step considers {definition.alternate_subject}.",
        ),
        OutcomeFreeTemplateEvent(
            event_id=f"{key}-event-5",
            statement=f"A later observation preserves the {locus} boundary.",
        ),
        OutcomeFreeTemplateEvent(
            event_id=f"{key}-event-6",
            statement=f"The decision point for {key} is now active.",
        ),
        OutcomeFreeTemplateEvent(
            event_id=f"{key}-event-7",
            statement=f"The visible prefix for {key} is complete.",
        ),
    )
    first_three = tuple(event.event_id for event in events[:3])
    memories = tuple(
        OutcomeFreeTemplateMemory(
            memory_id=f"{key}-memory-{index}",
            statement=(f"Memory {index} for {key} retains the {locus} context."),
            evidence_event_ids=first_three,
            recorded_event_id=first_three[2],
        )
        for index in range(4)
    )
    return OutcomeFreeTaskTemplate(
        event_pool=events,
        memory_pool=memories,
        pivot=OutcomeFreeTemplatePivot(
            event_id=f"{key}-pivot",
            statement="Choose exactly one listed next action.",
        ),
        action_pool=(
            OutcomeFreeTemplateAction(
                action_id=f"{key}-action-retained",
                statement=f"Use {definition.retained_subject} for {key}.",
            ),
            OutcomeFreeTemplateAction(
                action_id=f"{key}-action-alternate",
                statement=f"Use {definition.alternate_subject} for {key}.",
            ),
        ),
        adapter=_ADAPTER,
    )


def _transition_graph(definition: PublicLineageDefinition) -> PublicTransitionGraph:
    key = definition.lineage_registry_key
    factor_ids = (f"{key}-guidance-potency", f"{key}-baseline-recovery")
    factors = (
        PublicCausalFactor(
            factor_id=factor_ids[0],
            false_description="The guidance remains static at the decision point.",
            true_description="The guidance is potent at the decision point.",
        ),
        PublicCausalFactor(
            factor_id=factor_ids[1],
            false_description="The baseline path declines at the decision point.",
            true_description="The baseline path recovers at the decision point.",
        ),
    )
    states = (
        PublicTransitionState(
            state_id=f"{key}-decision",
            description="The next action remains to be evaluated.",
        ),
        PublicTransitionState(
            state_id=f"{key}-target-met",
            description="The target condition is met after the action.",
            terminal=PublicTerminalState.GOAL_REACHED,
        ),
        PublicTransitionState(
            state_id=f"{key}-target-missed",
            description="The target condition remains unmet after the action.",
            terminal=PublicTerminalState.GOAL_NOT_REACHED,
        ),
    )
    transitions: list[PublicTransition] = []
    for left, right in product((False, True), repeat=2):
        assignments = (
            PublicCausalFactorValue(factor_id=factor_ids[0], value=left),
            PublicCausalFactorValue(factor_id=factor_ids[1], value=right),
        )
        for exposure in PublicCausalExposure:
            reaches_target = left if exposure is PublicCausalExposure.GUIDANCE_APPLIED else right
            exposure_code = (
                "guided" if exposure is PublicCausalExposure.GUIDANCE_APPLIED else "base"
            )
            vector_code = f"{int(left)}{int(right)}"
            transitions.append(
                PublicTransition(
                    source_state_id=states[0].state_id,
                    target_state_id=(states[1].state_id if reaches_target else states[2].state_id),
                    exposure=exposure,
                    factor_values=assignments,
                    action_fingerprint_id=f"{key}-{exposure_code}-{vector_code}-action",
                    failure_fingerprint_id=(
                        None if reaches_target else f"{key}-{exposure_code}-{vector_code}-failure"
                    ),
                    trigger=(
                        f"Evaluate {exposure_code} path {vector_code} against the recorded context."
                    ),
                )
            )
    payload: dict[str, object] = {
        "schema_version": "state-decay-v2-public-transition-graph/v1",
        "initial_state_id": states[0].state_id,
        "factors": factors,
        "states": states,
        "transitions": tuple(transitions),
        "transition_graph_digest": "0" * 64,
    }
    payload["transition_graph_digest"] = transition_graph_digest(payload)
    return PublicTransitionGraph.model_validate(payload)


def _evidence_topology(definition: PublicLineageDefinition) -> PublicEvidenceTopology:
    key = definition.lineage_registry_key
    nodes = (
        PublicEvidenceNode(
            evidence_id=f"{key}-evidence-context",
            statement=f"The {definition.locus_id.replace('-', ' ')} context was recorded.",
        ),
        PublicEvidenceNode(
            evidence_id=f"{key}-evidence-choice",
            statement=f"The decision for {key} can cite the recorded context.",
        ),
    )
    payload: dict[str, object] = {
        "schema_version": "state-decay-v2-public-evidence-topology/v1",
        "nodes": nodes,
        "edges": (
            PublicEvidenceEdge(
                source_evidence_id=nodes[0].evidence_id,
                target_evidence_id=nodes[1].evidence_id,
                relation=PublicEvidenceRelation.SUPPORTS,
            ),
        ),
        "evidence_topology_digest": "0" * 64,
    }
    payload["evidence_topology_digest"] = evidence_topology_digest(payload)
    return PublicEvidenceTopology.model_validate(payload)


def _semantic_signature(definition: PublicLineageDefinition) -> PublicSemanticSignature:
    key = definition.lineage_registry_key
    payload: dict[str, object] = {
        "schema_version": "state-decay-v2-public-semantic-signature/v1",
        "concept_ids": (
            f"{key}-concept-family",
            f"{key}-concept-locus",
        ),
        "canonical_claims": (
            f"{definition.family_title} governs the {definition.locus_id.replace('-', ' ')} case.",
            f"The decision for {key} must preserve the recorded context boundary.",
        ),
        "semantic_signature_digest": "0" * 64,
    }
    payload["semantic_signature_digest"] = semantic_signature_digest(payload)
    return PublicSemanticSignature.model_validate(payload)


def _causal_deltas(
    definition: PublicLineageDefinition,
    graph: PublicTransitionGraph,
) -> tuple[CausalSemanticDelta, ...]:
    factor_ids = tuple(factor.factor_id for factor in graph.factors)
    deltas: list[CausalSemanticDelta] = []
    for index, (left, right) in enumerate(product((False, True), repeat=2)):
        guidance_state = "potent" if left else "static"
        baseline_state = "recovers" if right else "declines"
        payload: dict[str, object] = {
            "schema_version": "state-decay-v2-public-causal-semantic-delta/v1",
            "delta_index": index,
            "delta_id": f"{definition.lineage_registry_key}-delta-{index}",
            "family": definition.family,
            "lineage_registry_key": definition.lineage_registry_key,
            "factor_values": (
                PublicCausalFactorValue(factor_id=factor_ids[0], value=left),
                PublicCausalFactorValue(factor_id=factor_ids[1], value=right),
            ),
            "semantic_replacements": (
                CausalTextReplacement(
                    template_pointer="/event_pool/0/statement",
                    replacement=(
                        f"Guidance={guidance_state} baseline={baseline_state} "
                        f"for {definition.lineage_registry_key}."
                    ),
                ),
            ),
            "evidence_replacements": (),
            "causal_delta_digest": "0" * 64,
        }
        payload["causal_delta_digest"] = causal_delta_digest(payload)
        deltas.append(CausalSemanticDelta.model_validate(payload))
    return tuple(deltas)


def build_public_lineage_candidate(
    definition: PublicLineageDefinition,
    configuration: PublicGeneratorConfiguration,
    profile_catalog: PublicProfileCatalog,
    algorithm: PublicGeneratorAlgorithm,
) -> PublicLineageCandidate:
    split = _PUBLIC_SPLIT_BY_FAMILY[definition.family]
    task_template = _task_template(definition)
    transition_graph = _transition_graph(definition)
    evidence_topology = _evidence_topology(definition)
    failure_mechanism = PublicFailureMechanism(
        failure_mechanism_id=f"{definition.lineage_registry_key}-failure-mechanism",
        description=(
            f"The {definition.locus_id.replace('-', ' ')} context can be omitted at the decision."
        ),
    )
    semantic_signature = _semantic_signature(definition)
    causal_deltas = _causal_deltas(definition, transition_graph)
    previews = tuple(
        materialize_preallocation_preview(
            split=split,
            family=definition.family,
            lineage_registry_key=definition.lineage_registry_key,
            configuration=configuration,
            algorithm=algorithm,
            profile_catalog=profile_catalog,
            slot_profile=slot_profile,
            task_template=task_template,
            transition_graph=transition_graph,
            evidence_topology=evidence_topology,
            failure_mechanism=failure_mechanism,
            semantic_signature=semantic_signature,
            causal_deltas=causal_deltas,
        )
        for slot_profile in profile_catalog.slot_profiles
    )
    public_leaf = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.PUBLIC)
    independent_seed = derive_independent_lineage_seed(
        public_leaf,
        split=split,
        family=definition.family,
        lineage_registry_key=definition.lineage_registry_key,
    )
    payload: dict[str, object] = {
        "schema_version": "state-decay-v2-public-lineage-candidate/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "split": split,
        "family": definition.family,
        "lineage_registry_key": definition.lineage_registry_key,
        "generator_version": configuration.generator_version,
        "generation_contract_digest": configuration.generation_contract_digest,
        "generator_configuration_digest": configuration.configuration_digest,
        "generator_algorithm_digest": algorithm.algorithm_digest,
        "profile_catalog_digest": profile_catalog.catalog_digest,
        "independent_seed_commitment_digest": independent_lineage_seed_commitment(independent_seed),
        "task_template": task_template,
        "transition_graph": transition_graph,
        "evidence_topology": evidence_topology,
        "failure_mechanism": failure_mechanism,
        "semantic_signature": semantic_signature,
        "causal_deltas": causal_deltas,
        "semantic_rationale": definition.semantic_rationale,
        "derivation_parent_keys": (),
        "previews": previews,
        "candidate_packet_digest": "0" * 64,
    }
    payload["candidate_packet_digest"] = candidate_packet_digest(payload)
    return PublicLineageCandidate.model_validate(payload)


def build_public_lineage_registry(
    configuration: PublicGeneratorConfiguration,
    profile_catalog: PublicProfileCatalog,
    algorithm: PublicGeneratorAlgorithm,
    definitions: tuple[PublicLineageDefinition, ...],
) -> PublicLineageRegistry:
    if (
        profile_catalog.generator_configuration_digest != configuration.configuration_digest
        or algorithm.generator_configuration_digest != configuration.configuration_digest
        or algorithm.profile_catalog_digest != profile_catalog.catalog_digest
    ):
        raise ValueError("public lineage registry global inputs do not agree")
    candidates = tuple(
        build_public_lineage_candidate(
            definition,
            configuration,
            profile_catalog,
            algorithm,
        )
        for definition in definitions
    )
    payload: dict[str, object] = {
        "schema_version": "state-decay-v2-public-lineage-registry/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "generation_contract_digest": GENERATION_CONTRACT.contract_digest,
        "lineage_review_protocol_digest": LINEAGE_REVIEW_PROTOCOL.protocol_digest,
        "generator_configuration_digest": configuration.configuration_digest,
        "generator_algorithm_digest": algorithm.algorithm_digest,
        "profile_catalog": profile_catalog,
        "candidates": candidates,
        "registry_digest": "0" * 64,
    }
    payload["registry_digest"] = candidate_registry_digest(payload)
    return PublicLineageRegistry.model_validate(payload)


def validate_public_lineage_registry_materialization(
    registry: PublicLineageRegistry,
    configuration: PublicGeneratorConfiguration,
    profile_catalog: PublicProfileCatalog,
    algorithm: PublicGeneratorAlgorithm,
    definitions: tuple[PublicLineageDefinition, ...],
) -> PublicLineageRegistry:
    """Rebuild a public registry from its non-self inputs and compare all canonical bytes."""

    if type(registry) is not PublicLineageRegistry:
        raise ValueError("public lineage registry has an invalid type")
    rebuilt = build_public_lineage_registry(
        configuration,
        profile_catalog,
        algorithm,
        definitions,
    )
    if canonical_json(registry) != canonical_json(rebuilt):
        raise ValueError("public lineage registry does not match its canonical materialization")
    return registry


PUBLIC_GENERATOR_CONFIGURATION: Final[PublicGeneratorConfiguration] = (
    build_public_generator_configuration()
)
PUBLIC_PROFILE_CATALOG: Final[PublicProfileCatalog] = build_public_profile_catalog(
    PUBLIC_GENERATOR_CONFIGURATION
)
PUBLIC_GENERATOR_ALGORITHM: Final[PublicGeneratorAlgorithm] = build_public_generator_algorithm(
    PUBLIC_GENERATOR_CONFIGURATION,
    PUBLIC_PROFILE_CATALOG,
)
PUBLIC_LINEAGE_REGISTRY: Final[PublicLineageRegistry] = build_public_lineage_registry(
    PUBLIC_GENERATOR_CONFIGURATION,
    PUBLIC_PROFILE_CATALOG,
    PUBLIC_GENERATOR_ALGORITHM,
    PUBLIC_LINEAGE_DEFINITIONS,
)


__all__ = [
    "PUBLIC_GENERATOR_ALGORITHM",
    "PUBLIC_GENERATOR_CONFIGURATION",
    "PUBLIC_LINEAGE_REGISTRY",
    "PUBLIC_PROFILE_CATALOG",
    "build_public_generator_algorithm",
    "build_public_generator_configuration",
    "build_public_lineage_candidate",
    "build_public_lineage_registry",
    "validate_public_lineage_registry_materialization",
]
