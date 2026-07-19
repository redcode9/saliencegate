from __future__ import annotations

from saliencegate.benchmarks.state_decay_v2.config import (
    PUBLIC_GENERATION_SEED,
    SeedPurpose,
    derive_scenario_id,
    derive_seed,
)
from saliencegate.benchmarks.state_decay_v2.public_contract import (
    CausalSemanticDelta,
    OutcomeFreeAllowedAction,
    OutcomeFreeCandidateMemory,
    OutcomeFreeEvent,
    OutcomeFreeEvidenceReference,
    OutcomeFreePivot,
    OutcomeFreeTaskSkeleton,
    OutcomeFreeTaskTemplate,
    PreAllocationSkeletonPreview,
    PublicEvidenceTopology,
    PublicFailureMechanism,
    PublicGeneratorAlgorithm,
    PublicGeneratorConfiguration,
    PublicLineageKey,
    PublicProfileCatalog,
    PublicSemanticSignature,
    PublicSlotProfile,
    PublicTransitionGraph,
    RenderedCausalSemanticDelta,
    RenderedCausalTextReplacement,
    rendered_policy_digest,
    skeleton_preview_digest,
)
from saliencegate.benchmarks.state_decay_v2.schema import BenchmarkSplit, ScenarioFamily
from saliencegate.domain import ValidityState, canonical_json


def _parameter_clause(profile: PublicSlotProfile) -> str:
    parameters = ",".join(
        f"{item.parameter_id}={item.value}" for item in profile.parameters.allowed_values
    )
    return f" [parameters:{parameters}]"


def materialize_task_skeleton(
    task_template: OutcomeFreeTaskTemplate,
    slot_profile: PublicSlotProfile,
) -> OutcomeFreeTaskSkeleton:
    event_count = slot_profile.structure.trajectory_event_count
    memory_count = slot_profile.structure.candidate_memory_count
    integers = slot_profile.integers
    text_lengths = slot_profile.text_lengths

    selected_events = task_template.event_pool[:event_count]
    trajectory = tuple(
        OutcomeFreeEvent(
            event_id=event.event_id,
            sequence=integers.sequence_start + index * integers.sequence_stride,
            action_step=integers.action_step_start + index * integers.action_step_stride,
            statement=event.statement + " " * text_lengths.event_padding_spaces,
        )
        for index, event in enumerate(selected_events)
    )
    events_by_id = {event.event_id: event for event in trajectory}
    last_event = trajectory[-1]

    reference_ids: list[list[str]] = [
        [memory.evidence_event_ids[0]] for memory in task_template.memory_pool[:memory_count]
    ]
    remaining_references = slot_profile.evidence.evidence_reference_count - memory_count
    for reference_position in (1, 2):
        for memory_index in range(memory_count):
            if remaining_references == 0:
                break
            reference_ids[memory_index].append(
                task_template.memory_pool[memory_index].evidence_event_ids[reference_position]
            )
            remaining_references -= 1
    if remaining_references != 0:
        raise ValueError("slot evidence references cannot be materialized")

    validity = slot_profile.counterbalance.memory_validity
    candidate_memories = tuple(
        OutcomeFreeCandidateMemory(
            memory_id=template_memory.memory_id,
            revision=integers.memory_revision + index,
            statement=template_memory.statement + " " * text_lengths.memory_padding_spaces,
            evidence_refs=tuple(
                OutcomeFreeEvidenceReference(
                    event_id=event_id,
                    event_sequence=events_by_id[event_id].sequence,
                )
                for event_id in reference_ids[index]
            ),
            recorded_sequence=events_by_id[template_memory.recorded_event_id].sequence,
            recorded_action_step=events_by_id[template_memory.recorded_event_id].action_step,
            validity=validity,
            validity_sequence=(None if validity is ValidityState.ACTIVE else last_event.sequence),
            validity_action_step=(
                None if validity is ValidityState.ACTIVE else last_event.action_step
            ),
        )
        for index, template_memory in enumerate(task_template.memory_pool[:memory_count])
    )

    action_order = slot_profile.counterbalance.allowed_action_order
    allowed_actions = tuple(
        OutcomeFreeAllowedAction(
            action_id=task_template.action_pool[logical_index].action_id,
            statement=(
                task_template.action_pool[logical_index].statement
                + " " * text_lengths.action_padding_spaces
            ),
        )
        for logical_index in action_order
    )
    pivot = OutcomeFreePivot(
        event_id=task_template.pivot.event_id,
        sequence=last_event.sequence + integers.sequence_stride,
        action_step=last_event.action_step,
        statement=(
            task_template.pivot.statement
            + _parameter_clause(slot_profile)
            + " " * text_lengths.pivot_padding_spaces
        ),
    )
    return OutcomeFreeTaskSkeleton(
        trajectory=trajectory,
        candidate_memories=candidate_memories,
        pivot=pivot,
        allowed_actions=allowed_actions,
        adapter=task_template.adapter,
    )


def _rendered_pointer(template_pointer: str, slot_profile: PublicSlotProfile) -> str:
    if template_pointer.startswith("/event_pool/"):
        return template_pointer.replace("/event_pool/", "/trajectory/", 1)
    if template_pointer.startswith("/memory_pool/"):
        return template_pointer.replace("/memory_pool/", "/candidate_memories/", 1)
    if template_pointer == "/pivot/statement":
        return template_pointer
    if template_pointer.startswith("/action_pool/"):
        logical_index = int(template_pointer.split("/", maxsplit=3)[2])
        rendered_index = slot_profile.counterbalance.allowed_action_order.index(logical_index)
        return f"/allowed_actions/{rendered_index}/statement"
    raise ValueError("causal delta template pointer is not materializable")


def _rendered_replacement(
    template_pointer: str,
    replacement: str,
    slot_profile: PublicSlotProfile,
) -> str:
    lengths = slot_profile.text_lengths
    if template_pointer.startswith("/event_pool/"):
        suffix = " " * lengths.event_padding_spaces
    elif template_pointer.startswith("/memory_pool/"):
        suffix = " " * lengths.memory_padding_spaces
    elif template_pointer == "/pivot/statement":
        suffix = _parameter_clause(slot_profile) + " " * lengths.pivot_padding_spaces
    elif template_pointer.startswith("/action_pool/"):
        suffix = " " * lengths.action_padding_spaces
    else:
        raise ValueError("causal delta template pointer is not materializable")
    return replacement + suffix


def _replace_statement(
    skeleton: OutcomeFreeTaskSkeleton,
    replacement: RenderedCausalTextReplacement,
) -> OutcomeFreeTaskSkeleton:
    pointer = replacement.policy_pointer
    if pointer.startswith("/trajectory/"):
        index = int(pointer.split("/", maxsplit=3)[2])
        events = list(skeleton.trajectory)
        events[index] = events[index].model_copy(update={"statement": replacement.replacement})
        return OutcomeFreeTaskSkeleton.model_validate(
            skeleton.model_copy(update={"trajectory": tuple(events)})
        )
    if pointer.startswith("/candidate_memories/"):
        index = int(pointer.split("/", maxsplit=3)[2])
        memories = list(skeleton.candidate_memories)
        memories[index] = memories[index].model_copy(update={"statement": replacement.replacement})
        return OutcomeFreeTaskSkeleton.model_validate(
            skeleton.model_copy(update={"candidate_memories": tuple(memories)})
        )
    if pointer == "/pivot/statement":
        return OutcomeFreeTaskSkeleton.model_validate(
            skeleton.model_copy(
                update={
                    "pivot": skeleton.pivot.model_copy(
                        update={"statement": replacement.replacement}
                    )
                }
            )
        )
    if pointer.startswith("/allowed_actions/"):
        index = int(pointer.split("/", maxsplit=3)[2])
        actions = list(skeleton.allowed_actions)
        actions[index] = actions[index].model_copy(update={"statement": replacement.replacement})
        return OutcomeFreeTaskSkeleton.model_validate(
            skeleton.model_copy(update={"allowed_actions": tuple(actions)})
        )
    raise ValueError("rendered causal delta pointer is not materializable")


def render_causal_delta(
    skeleton: OutcomeFreeTaskSkeleton,
    slot_profile: PublicSlotProfile,
    delta: CausalSemanticDelta,
) -> RenderedCausalSemanticDelta:
    semantic_replacements = tuple(
        RenderedCausalTextReplacement(
            policy_pointer=_rendered_pointer(item.template_pointer, slot_profile),
            replacement=_rendered_replacement(
                item.template_pointer,
                item.replacement,
                slot_profile,
            ),
        )
        for item in delta.semantic_replacements
    )
    evidence_replacements = tuple(
        RenderedCausalTextReplacement(
            policy_pointer=_rendered_pointer(item.template_pointer, slot_profile),
            replacement=_rendered_replacement(
                item.template_pointer,
                item.replacement,
                slot_profile,
            ),
        )
        for item in delta.evidence_replacements
    )
    rendered_skeleton = skeleton
    for replacement in (*semantic_replacements, *evidence_replacements):
        rendered_skeleton = _replace_statement(rendered_skeleton, replacement)
    return RenderedCausalSemanticDelta(
        delta_index=delta.delta_index,
        delta_id=delta.delta_id,
        causal_delta_digest=delta.causal_delta_digest,
        factor_values=delta.factor_values,
        semantic_replacements=semantic_replacements,
        evidence_replacements=evidence_replacements,
        rendered_policy_digest=rendered_policy_digest(canonical_json(rendered_skeleton)),
    )


def materialize_preallocation_preview(
    *,
    split: BenchmarkSplit,
    family: ScenarioFamily,
    lineage_registry_key: PublicLineageKey,
    configuration: PublicGeneratorConfiguration,
    algorithm: PublicGeneratorAlgorithm,
    profile_catalog: PublicProfileCatalog,
    slot_profile: PublicSlotProfile,
    task_template: OutcomeFreeTaskTemplate,
    transition_graph: PublicTransitionGraph,
    evidence_topology: PublicEvidenceTopology,
    failure_mechanism: PublicFailureMechanism,
    semantic_signature: PublicSemanticSignature,
    causal_deltas: tuple[CausalSemanticDelta, ...],
) -> PreAllocationSkeletonPreview:
    if profile_catalog.slot_profiles[slot_profile.generator_slot] != slot_profile:
        raise ValueError("pre-allocation slot profile is not the canonical catalog entry")
    if (
        configuration.configuration_digest != profile_catalog.generator_configuration_digest
        or algorithm.generator_configuration_digest != configuration.configuration_digest
        or algorithm.profile_catalog_digest != profile_catalog.catalog_digest
    ):
        raise ValueError("pre-allocation global bindings do not agree")
    semantic_allowlist = set(algorithm.semantic_pointer_allowlist)
    evidence_allowlist = set(algorithm.evidence_pointer_allowlist)
    if any(
        item.template_pointer not in semantic_allowlist
        for delta in causal_deltas
        for item in delta.semantic_replacements
    ) or any(
        item.template_pointer not in evidence_allowlist
        for delta in causal_deltas
        for item in delta.evidence_replacements
    ):
        raise ValueError("pre-allocation causal pointer is outside the generator allowlist")

    task_skeleton = materialize_task_skeleton(task_template, slot_profile)
    # Kept local so the raw-fixture module can import contract types without an import cycle.
    from saliencegate.benchmarks.state_decay_v2.signal_fixtures import (
        materialize_public_trace_fixture,
    )

    trace_fixture = materialize_public_trace_fixture(slot_profile, task_skeleton)
    rendered_deltas = tuple(
        render_causal_delta(task_skeleton, slot_profile, delta) for delta in causal_deltas
    )
    payload: dict[str, object] = {
        "schema_version": "state-decay-v2-pre-allocation-skeleton-preview/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "split": split,
        "family": family,
        "lineage_registry_key": lineage_registry_key,
        "generator_slot": slot_profile.generator_slot,
        "generator_version": configuration.generator_version,
        "generation_contract_digest": configuration.generation_contract_digest,
        "generator_configuration_digest": configuration.configuration_digest,
        "generator_algorithm_digest": algorithm.algorithm_digest,
        "profile_catalog_digest": profile_catalog.catalog_digest,
        "transition_graph_digest": transition_graph.transition_graph_digest,
        "evidence_topology_digest": evidence_topology.evidence_topology_digest,
        "failure_mechanism_id": failure_mechanism.failure_mechanism_id,
        "semantic_signature_digest": semantic_signature.semantic_signature_digest,
        "slot_profile": slot_profile,
        "allowed_parameter_values": slot_profile.parameters.allowed_values,
        "task_skeleton": task_skeleton,
        "trace_fixture": trace_fixture,
        "rendered_causal_deltas": rendered_deltas,
        "preview_digest": "0" * 64,
    }
    payload["preview_digest"] = skeleton_preview_digest(payload)
    return PreAllocationSkeletonPreview.model_validate(payload)


def preview_to_scenario_id(preview: PreAllocationSkeletonPreview) -> str:
    checked = PreAllocationSkeletonPreview.model_validate(preview)
    id_leaf = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.ID)
    return derive_scenario_id(id_leaf, skeleton_digest=checked.preview_digest)


__all__ = [
    "materialize_preallocation_preview",
    "materialize_task_skeleton",
    "preview_to_scenario_id",
    "render_causal_delta",
]
