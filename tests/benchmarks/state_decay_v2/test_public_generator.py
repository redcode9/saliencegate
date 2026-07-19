from __future__ import annotations

import copy
import pickle
import subprocess
import sys
import weakref
from collections import Counter
from collections.abc import Mapping
from dataclasses import fields
from hashlib import sha256

import pytest

import saliencegate.benchmarks.state_decay_v2.config as generation_config
import saliencegate.benchmarks.state_decay_v2.generation_authority as generation_authority
import saliencegate.benchmarks.state_decay_v2.generator as public_generator
import saliencegate.benchmarks.state_decay_v2.templates as public_templates
from saliencegate.benchmarks.state_decay_v2.generation_authority import (
    PublicGenerationAuthority,
    PublicGenerationAuthorityError,
)
from saliencegate.benchmarks.state_decay_v2.generator import (
    PublicCausalResolution,
    PublicCausalResolutionIndex,
    PublicGeneratorInputError,
    PublicSignalValidation,
    SyntheticPublicAssignment,
    build_public_causal_resolution_index,
    compose_role_neutral_parts,
    compose_synthetic_public_scenarios,
    generate_public_scenarios,
    render_public_causal_policy,
    resolve_public_causal_policy,
    select_public_causal_delta,
    verify_public_signal_fixture,
)
from saliencegate.benchmarks.state_decay_v2.preallocation import (
    preview_to_scenario_id,
    render_causal_delta,
)
from saliencegate.benchmarks.state_decay_v2.public_contract import (
    CausalSemanticDelta,
    CausalTextReplacement,
    OutcomeFreeTaskSkeleton,
    PreAllocationSkeletonPreview,
    RoleNeutralGeneratedParts,
    causal_delta_digest,
    derive_causal_outcome,
    skeleton_preview_digest,
)
from saliencegate.benchmarks.state_decay_v2.review_contract import PublicLineageReviewSubreport
from saliencegate.benchmarks.state_decay_v2.schema import BenchmarkSplit, ScenarioOutcome
from saliencegate.benchmarks.state_decay_v2.templates import PUBLIC_LINEAGE_REGISTRY
from saliencegate.domain import ValidityState, canonical_json


@pytest.fixture(scope="module")
def causal_index() -> PublicCausalResolutionIndex:
    return build_public_causal_resolution_index(PUBLIC_LINEAGE_REGISTRY)


def test_generator_import_does_not_materialize_the_template_registry() -> None:
    module = "saliencegate.benchmarks.state_decay_v2.generator"
    templates = "saliencegate.benchmarks.state_decay_v2.templates"
    result = subprocess.run(
        (
            sys.executable,
            "-I",
            "-c",
            f"import sys; import {module}; assert {templates!r} not in sys.modules",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def _skeleton_from_parts(parts: RoleNeutralGeneratedParts) -> OutcomeFreeTaskSkeleton:
    return OutcomeFreeTaskSkeleton(
        trajectory=parts.trajectory,
        candidate_memories=parts.candidate_memories,
        pivot=parts.pivot,
        allowed_actions=parts.allowed_actions,
        adapter=parts.adapter,
    )


def _diff_paths(left: object, right: object, pointer: str = "") -> set[str]:
    if type(left) is not type(right):
        return {pointer or "/"}
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return {pointer or "/"}
        result: set[str] = set()
        for key in left:
            result.update(_diff_paths(left[key], right[key], f"{pointer}/{key}"))
        return result
    if isinstance(left, list):
        if len(left) != len(right):
            return {pointer or "/"}
        result = set()
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            result.update(_diff_paths(left_item, right_item, f"{pointer}/{index}"))
        return result
    return set() if left == right else {pointer or "/"}


def _temporal_summary(values: tuple[int, ...]) -> tuple[int, ...]:
    if not values:
        return (0, -1, -1, -1, 0, 0)
    minimum = min(values)
    maximum = max(values)
    return (
        len(values),
        minimum,
        maximum,
        maximum - minimum,
        sum(values),
        len(set(values)),
    )


def _right_pad(values: tuple[int, ...], width: int, padding: int) -> tuple[int, ...]:
    assert len(values) <= width
    return (*values, *((padding,) * (width - len(values))))


def _identifier_histograms(parts: RoleNeutralGeneratedParts) -> tuple[tuple[int, ...], ...]:
    identifiers = (
        parts.schema_version,
        parts.suite_id,
        parts.suite_version,
        parts.split.value,
        parts.scenario_id,
        *(event.event_id for event in parts.trajectory),
        *(memory.memory_id for memory in parts.candidate_memories),
        *(
            reference.event_id
            for memory in parts.candidate_memories
            for reference in memory.evidence_refs
        ),
        parts.pivot.event_id,
        *(action.action_id for action in parts.allowed_actions),
        parts.adapter.adapter_id,
        parts.adapter.adapter_version,
        parts.adapter.response_profile_id,
        parts.adapter.response_profile_digest,
        *parts.adapter.capabilities,
    )
    byte_counts = Counter(byte for identifier in identifiers for byte in identifier.encode("utf-8"))
    nibble_counts = Counter(
        nibble
        for identifier in identifiers
        for byte in identifier.encode("utf-8")
        for nibble in (byte >> 4, byte & 0x0F)
    )
    return (
        tuple(byte_counts[index] for index in range(256)),
        tuple(nibble_counts[index] for index in range(16)),
    )


def _independent_nuisance_vector(parts: RoleNeutralGeneratedParts) -> tuple[int, ...]:
    event_by_id = {event.event_id: event for event in parts.trajectory}
    action_ids = tuple(action.action_id for action in parts.allowed_actions)
    first_action_index = tuple(sorted(action_ids, key=lambda value: value.encode("utf-8"))).index(
        action_ids[0]
    )
    optional_presence = tuple(
        presence
        for memory in parts.candidate_memories
        for presence in (
            int(memory.validity_sequence is not None),
            int(memory.validity_action_step is not None),
        )
    )
    byte_histogram, nibble_histogram = _identifier_histograms(parts)
    evidence_refs = tuple(
        reference for memory in parts.candidate_memories for reference in memory.evidence_refs
    )
    evidence_text_lengths = tuple(
        len(event_by_id[reference.event_id].statement.encode("utf-8"))
        for reference in evidence_refs
    )
    vector = (
        first_action_index,
        len(parts.trajectory),
        len(parts.candidate_memories),
        len(parts.allowed_actions),
        len(evidence_refs),
        parts.pivot.sequence,
        parts.pivot.action_step,
        *(
            sum(memory.validity is state for memory in parts.candidate_memories)
            for state in ValidityState
        ),
        *_right_pad(optional_presence, 64, 0),
        *byte_histogram,
        *nibble_histogram,
        *_temporal_summary(
            (*tuple(event.sequence for event in parts.trajectory), parts.pivot.sequence)
        ),
        *_temporal_summary(
            (*tuple(event.action_step for event in parts.trajectory), parts.pivot.action_step)
        ),
        *_temporal_summary(tuple(memory.recorded_sequence for memory in parts.candidate_memories)),
        *_temporal_summary(
            tuple(memory.recorded_action_step for memory in parts.candidate_memories)
        ),
        *_temporal_summary(
            tuple(
                memory.validity_sequence
                for memory in parts.candidate_memories
                if memory.validity_sequence is not None
            )
        ),
        *_temporal_summary(
            tuple(
                memory.validity_action_step
                for memory in parts.candidate_memories
                if memory.validity_action_step is not None
            )
        ),
        *_temporal_summary(tuple(reference.event_sequence for reference in evidence_refs)),
        *_temporal_summary(tuple(memory.revision for memory in parts.candidate_memories)),
        *_right_pad(
            tuple(
                sorted(
                    (
                        *(len(event.statement.encode("utf-8")) for event in parts.trajectory),
                        len(parts.pivot.statement.encode("utf-8")),
                    )
                )
            ),
            65,
            -1,
        ),
        *_right_pad(
            tuple(
                sorted(len(memory.statement.encode("utf-8")) for memory in parts.candidate_memories)
            ),
            32,
            -1,
        ),
        *_right_pad(tuple(sorted(evidence_text_lengths)), 512, -1),
        *_right_pad(
            tuple(
                sorted(len(action.statement.encode("utf-8")) for action in parts.allowed_actions)
            ),
            16,
            -1,
        ),
    )
    assert len(vector) == 1_020
    return vector


def _controlled_ascii_replacement(source: str) -> str:
    index = next(index for index, character in enumerate(source) if character != " ")
    replacement = "Z" if source[index] != "Z" else "Y"
    return f"{source[:index]}{replacement}{source[index + 1 :]}"


def _template_statement(candidate: object, pointer: str) -> str:
    if pointer.startswith("/event_pool/"):
        return candidate.task_template.event_pool[int(pointer.split("/")[2])].statement
    if pointer.startswith("/memory_pool/"):
        return candidate.task_template.memory_pool[int(pointer.split("/")[2])].statement
    if pointer == "/pivot/statement":
        return candidate.task_template.pivot.statement
    if pointer.startswith("/action_pool/"):
        return candidate.task_template.action_pool[int(pointer.split("/")[2])].statement
    raise AssertionError(pointer)


def _expected_rendered_pointer(pointer: str, action_order: tuple[int, int]) -> str:
    if pointer.startswith("/event_pool/"):
        return f"/trajectory/{pointer.split('/')[2]}/statement"
    if pointer.startswith("/memory_pool/"):
        return f"/candidate_memories/{pointer.split('/')[2]}/statement"
    if pointer == "/pivot/statement":
        return pointer
    logical_index = int(pointer.split("/")[2])
    return f"/allowed_actions/{action_order.index(logical_index)}/statement"


def test_causal_resolution_index_is_closed_opaque_and_policy_only(
    causal_index: PublicCausalResolutionIndex,
) -> None:
    assert causal_index.binding_count == 3_600
    assert not isinstance(causal_index, Mapping)
    with pytest.raises(PublicGeneratorInputError):
        PublicCausalResolutionIndex()
    with pytest.raises(TypeError):
        causal_index["0" * 64]  # type: ignore[index]

    preview = PUBLIC_LINEAGE_REGISTRY.candidates[0].previews[0]
    delta = preview.rendered_causal_deltas[0]
    rendered = render_public_causal_policy(preview, delta)
    resolution = resolve_public_causal_policy(causal_index, rendered)

    assert type(resolution) is PublicCausalResolution
    assert tuple(field.name for field in fields(resolution)) == (
        "transition_graph",
        "factor_values",
    )
    assert resolution.factor_values == delta.factor_values
    assert not any(
        "digest" in name and callable(getattr(causal_index, name)) for name in dir(causal_index)
    )
    with pytest.raises(PublicGeneratorInputError):
        causal_index._PublicCausalResolutionIndex__bindings = {}  # type: ignore[attr-defined]
    for copier in (copy.copy, copy.deepcopy):
        with pytest.raises(PublicGeneratorInputError):
            copier(causal_index)
    with pytest.raises(PublicGeneratorInputError):
        pickle.dumps(causal_index)

    forged = object.__new__(PublicCausalResolutionIndex)
    object.__setattr__(
        forged,
        "_PublicCausalResolutionIndex__bindings",
        {delta.rendered_policy_digest: resolution},
    )
    with pytest.raises(PublicGeneratorInputError):
        resolve_public_causal_policy(forged, rendered)

    changed_event = rendered.trajectory[0].model_copy(
        update={"statement": rendered.trajectory[0].statement + " "}
    )
    changed_policy = rendered.model_copy(
        update={"trajectory": (changed_event, *rendered.trajectory[1:])}
    )
    with pytest.raises(PublicGeneratorInputError):
        resolve_public_causal_policy(causal_index, changed_policy)
    with pytest.raises(PublicGeneratorInputError):
        resolve_public_causal_policy(causal_index, delta.rendered_policy_digest)  # type: ignore[arg-type]


def test_every_allowlisted_pointer_renders_with_inverse_action_mapping_and_exact_nuisance() -> None:
    candidate = PUBLIC_LINEAGE_REGISTRY.candidates[0]
    pointers = (
        "/event_pool/0/statement",
        "/event_pool/1/statement",
        "/event_pool/2/statement",
        "/pivot/statement",
        "/action_pool/0/statement",
        "/action_pool/1/statement",
        "/memory_pool/0/statement",
    )

    for preview in candidate.previews:
        for pointer in pointers:
            source = _template_statement(candidate, pointer)
            replacement = CausalTextReplacement(
                template_pointer=pointer,
                replacement=_controlled_ascii_replacement(source),
            )
            payload: dict[str, object] = {
                "schema_version": "state-decay-v2-public-causal-semantic-delta/v1",
                "delta_index": 0,
                "delta_id": f"{candidate.lineage_registry_key}-synthetic-pointer",
                "family": candidate.family,
                "lineage_registry_key": candidate.lineage_registry_key,
                "factor_values": candidate.causal_deltas[0].factor_values,
                "semantic_replacements": ()
                if pointer.startswith("/memory_pool/")
                else (replacement,),
                "evidence_replacements": (replacement,)
                if pointer.startswith("/memory_pool/")
                else (),
                "causal_delta_digest": "0" * 64,
            }
            payload["causal_delta_digest"] = causal_delta_digest(payload)
            source_delta = CausalSemanticDelta.model_validate(payload)
            rendered_delta = render_causal_delta(
                preview.task_skeleton,
                preview.slot_profile,
                source_delta,
            )
            expected_pointer = _expected_rendered_pointer(
                pointer,
                preview.slot_profile.counterbalance.allowed_action_order,
            )
            rendered_replacement = (
                rendered_delta.evidence_replacements[0]
                if pointer.startswith("/memory_pool/")
                else rendered_delta.semantic_replacements[0]
            )
            assert rendered_replacement.policy_pointer == expected_pointer

            synthetic_payload = preview.model_dump(mode="python")
            synthetic_payload["rendered_causal_deltas"] = (
                rendered_delta,
                *preview.rendered_causal_deltas[1:],
            )
            synthetic_payload["preview_digest"] = skeleton_preview_digest(synthetic_payload)
            synthetic_preview = PreAllocationSkeletonPreview.model_validate(synthetic_payload)
            scenario_id = preview_to_scenario_id(synthetic_preview)
            parts, _, _ = compose_role_neutral_parts(
                synthetic_preview,
                rendered_delta,
                scenario_id=scenario_id,
            )
            rendered = _skeleton_from_parts(parts)
            assert _diff_paths(
                preview.task_skeleton.model_dump(mode="json"),
                rendered.model_dump(mode="json"),
            ) == {expected_pointer}
            source_parts = RoleNeutralGeneratedParts(
                split=preview.split,
                scenario_id=scenario_id,
                trajectory=preview.task_skeleton.trajectory,
                candidate_memories=preview.task_skeleton.candidate_memories,
                pivot=preview.task_skeleton.pivot,
                allowed_actions=preview.task_skeleton.allowed_actions,
                adapter=preview.task_skeleton.adapter,
            )
            assert _independent_nuisance_vector(parts) == _independent_nuisance_vector(source_parts)

            rendered_source = preview.task_skeleton.model_dump(mode="json")
            source_statement: object = rendered_source
            for token in expected_pointer.strip("/").split("/"):
                source_statement = (
                    source_statement[int(token)] if token.isdigit() else source_statement[token]
                )
            assert type(source_statement) is str
            assert len(source_statement.encode("utf-8")) == len(
                rendered_replacement.replacement.encode("utf-8")
            )
            assert tuple(
                index for index, value in enumerate(source_statement) if value == " "
            ) == tuple(
                index
                for index, value in enumerate(rendered_replacement.replacement)
                if value == " "
            )
            assert (
                source_statement.replace(" ", "").casefold()
                != rendered_replacement.replacement.replace(" ", "").casefold()
            )
            if pointer == "/pivot/statement":
                assert " [parameters:" in rendered_replacement.replacement


def test_every_policy_resolves_through_graph_execution_and_composes_without_nuisance_drift(
    causal_index: PublicCausalResolutionIndex,
) -> None:
    seen_policy_bytes: set[bytes] = set()

    for candidate in PUBLIC_LINEAGE_REGISTRY.candidates:
        for preview in candidate.previews:
            scenario_id = preview_to_scenario_id(preview)
            for source_delta in preview.rendered_causal_deltas:
                assigned_outcome = derive_causal_outcome(
                    candidate.transition_graph,
                    source_delta.factor_values,
                )
                selected = select_public_causal_delta(
                    causal_index,
                    preview,
                    assigned_outcome,
                )
                assert selected == source_delta

                rendered = render_public_causal_policy(preview, selected)
                resolution = resolve_public_causal_policy(causal_index, rendered)
                assert (
                    derive_causal_outcome(
                        resolution.transition_graph,
                        resolution.factor_values,
                    )
                    is assigned_outcome
                )

                parts, fixture, evidence = compose_role_neutral_parts(
                    preview,
                    selected,
                    scenario_id=scenario_id,
                )
                composed = _skeleton_from_parts(parts)
                source_payload = preview.task_skeleton.model_dump(mode="json")
                composed_payload = composed.model_dump(mode="json")
                allowed_paths = {
                    replacement.policy_pointer
                    for replacement in (
                        *selected.semantic_replacements,
                        *selected.evidence_replacements,
                    )
                }
                assert _diff_paths(source_payload, composed_payload) == allowed_paths
                source_parts = RoleNeutralGeneratedParts(
                    split=preview.split,
                    scenario_id=scenario_id,
                    trajectory=preview.task_skeleton.trajectory,
                    candidate_memories=preview.task_skeleton.candidate_memories,
                    pivot=preview.task_skeleton.pivot,
                    allowed_actions=preview.task_skeleton.allowed_actions,
                    adapter=preview.task_skeleton.adapter,
                )
                assert _independent_nuisance_vector(parts) == _independent_nuisance_vector(
                    source_parts
                )
                assert fixture == preview.trace_fixture
                assert canonical_json(fixture) == canonical_json(preview.trace_fixture)
                assert parts.scenario_id == scenario_id
                assert parts.split is preview.split
                assert evidence.event_ids == tuple(
                    event.event_id
                    for event in composed.trajectory[
                        : preview.slot_profile.evidence.decisive_event_count
                    ]
                )
                assert tuple(
                    (item.memory_id, item.revision) for item in evidence.memory_revisions
                ) == tuple(
                    (memory.memory_id, memory.revision)
                    for memory in composed.candidate_memories[
                        : preview.slot_profile.evidence.decisive_memory_count
                    ]
                )
                decisive_position = preview.slot_profile.counterbalance.decisive_action_position
                assert (
                    evidence.decisive_action_id
                    == composed.allowed_actions[decisive_position].action_id
                )
                assert (
                    evidence.decisive_action_id == candidate.task_template.action_pool[0].action_id
                )

                encoded = canonical_json(parts)
                for forbidden in (
                    b'"family"',
                    b'"lineage_registry_key"',
                    b'"generator_slot"',
                    b'"outcome"',
                    b'"allocation',
                    b'"delta_',
                    b'"review',
                    b'"oracle',
                    b'"cluster',
                ):
                    assert forbidden not in encoded
                seen_policy_bytes.add(canonical_json(composed))

    assert len(seen_policy_bytes) == 3_600


@pytest.mark.asyncio
async def test_fixture_validation_keeps_real_and_reference_results_separate() -> None:
    validated = 0
    for candidate in PUBLIC_LINEAGE_REGISTRY.candidates:
        for preview in candidate.previews:
            validation = await verify_public_signal_fixture(
                preview,
                scenario_id=preview_to_scenario_id(preview),
            )
            assert validation.projection == preview.slot_profile.signals.expected_signals
            assert len(validation.legacy.results) == 4
            assert len(validation.reference_results) == 5
            validated += 1
    assert validated == 900


def test_synthetic_public_composition_has_exact_visible_split_geometry(
    causal_index: PublicCausalResolutionIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic_outcomes = (
        ScenarioOutcome.HELPFUL,
        ScenarioOutcome.HELPFUL,
        ScenarioOutcome.HARMFUL,
        ScenarioOutcome.REDUNDANT,
        ScenarioOutcome.UNRESOLVED,
    )
    assignments = []
    for candidate in PUBLIC_LINEAGE_REGISTRY.candidates:
        for preview, outcome in zip(candidate.previews, synthetic_outcomes, strict=True):
            assignments.append(
                SyntheticPublicAssignment(
                    candidate_packet_digest=candidate.candidate_packet_digest,
                    generator_slot=preview.generator_slot,
                    scenario_id=preview_to_scenario_id(preview),
                    assigned_outcome=outcome,
                )
            )

    allocation_calls: list[object] = []

    def allocation_spy(*args: object, **kwargs: object) -> tuple[()]:
        allocation_calls.append((args, kwargs))
        return ()

    monkeypatch.setattr(generation_config, "allocate_balanced_outcomes", allocation_spy)
    generated = compose_synthetic_public_scenarios(
        PUBLIC_LINEAGE_REGISTRY,
        causal_index,
        tuple(assignments),
    )
    parts = tuple(item[0] for item in generated)

    assert allocation_calls == []
    assert len(parts) == len({item.scenario_id for item in parts}) == 900
    assert Counter(item.split for item in parts) == {
        BenchmarkSplit.TRAIN: 600,
        BenchmarkSplit.DEVELOPMENT: 300,
    }

    reordered = list(assignments)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(PublicGeneratorInputError):
        compose_synthetic_public_scenarios(
            PUBLIC_LINEAGE_REGISTRY,
            causal_index,
            tuple(reordered),
        )


@pytest.mark.asyncio
async def test_gated_orchestration_rejects_before_registry_ids_or_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def touched(*args: object, **kwargs: object) -> object:
        del args, kwargs
        calls.append("after-authority")
        raise AssertionError("authority check did not run first")

    monkeypatch.setattr(
        public_templates,
        "validate_public_lineage_registry_materialization",
        touched,
    )
    monkeypatch.setattr(generation_config, "derive_scenario_id", touched)
    monkeypatch.setattr(generation_config, "allocate_balanced_outcomes", touched)
    monkeypatch.setattr(public_generator, "preview_to_scenario_id", touched)

    with pytest.raises(PublicGenerationAuthorityError):
        await generate_public_scenarios(object())
    assert calls == []


def test_generator_rejects_wrong_exact_input_types(
    causal_index: PublicCausalResolutionIndex,
) -> None:
    preview = PUBLIC_LINEAGE_REGISTRY.candidates[0].previews[0]
    delta = preview.rendered_causal_deltas[0]
    assert render_public_causal_policy(preview, delta.model_copy()) == render_public_causal_policy(
        preview, delta
    )

    with pytest.raises(PublicGeneratorInputError):
        build_public_causal_resolution_index(object())  # type: ignore[arg-type]
    with pytest.raises(PublicGeneratorInputError):
        render_public_causal_policy(object(), delta)  # type: ignore[arg-type]
    with pytest.raises(PublicGeneratorInputError):
        render_public_causal_policy(
            preview.model_copy(update={"preview_digest": "0" * 64}),
            delta,
        )
    with pytest.raises(PublicGeneratorInputError):
        select_public_causal_delta(causal_index, preview, "helpful")  # type: ignore[arg-type]
    with pytest.raises(PublicGeneratorInputError):
        compose_role_neutral_parts(preview, delta, scenario_id="0" * 63)
    with pytest.raises(PublicGeneratorInputError):
        compose_role_neutral_parts(preview, delta, scenario_id="0" * 64)


def test_generator_value_objects_and_index_protocol_reject_invalid_state(
    causal_index: PublicCausalResolutionIndex,
) -> None:
    candidate = PUBLIC_LINEAGE_REGISTRY.candidates[0]
    preview = candidate.previews[0]
    resolution = resolve_public_causal_policy(
        causal_index,
        render_public_causal_policy(preview, preview.rendered_causal_deltas[0]),
    )

    with pytest.raises(PublicGeneratorInputError):
        PublicCausalResolution(
            transition_graph=object(),  # type: ignore[arg-type]
            factor_values=resolution.factor_values,
        )
    with pytest.raises(PublicGeneratorInputError):
        PublicCausalResolution(
            transition_graph=resolution.transition_graph,
            factor_values=resolution.factor_values[:1],
        )
    with pytest.raises(PublicGeneratorInputError):
        PublicSignalValidation(
            legacy=object(),  # type: ignore[arg-type]
            reference_results=(),
            projection=(),
        )
    for values in (
        ("z" * 64, 0, "0" * 64, ScenarioOutcome.HELPFUL),
        ("0" * 64, 5, "0" * 64, ScenarioOutcome.HELPFUL),
        ("0" * 64, 0, "0" * 63, ScenarioOutcome.HELPFUL),
        ("0" * 64, 0, "0" * 64, "helpful"),
    ):
        with pytest.raises(PublicGeneratorInputError):
            SyntheticPublicAssignment(
                candidate_packet_digest=values[0],
                generator_slot=values[1],
                scenario_id=values[2],
                assigned_outcome=values[3],  # type: ignore[arg-type]
            )

    with pytest.raises(PublicGeneratorInputError):
        causal_index.__delattr__("_PublicCausalResolutionIndex__bindings")
    with pytest.raises(PublicGeneratorInputError):
        causal_index.__reduce__()
    with pytest.raises(PublicGeneratorInputError):
        resolve_public_causal_policy(
            object(),  # type: ignore[arg-type]
            preview.task_skeleton,
        )
    with pytest.raises(PublicGeneratorInputError):
        public_generator._new_public_causal_resolution_index({})


def test_generator_sanitizes_wrapped_canonicalization_and_rendering_failures(
    causal_index: PublicCausalResolutionIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview = PUBLIC_LINEAGE_REGISTRY.candidates[0].previews[0]
    delta = preview.rendered_causal_deltas[0]
    scenario_id = preview_to_scenario_id(preview)

    with pytest.raises(PublicGeneratorInputError):
        build_public_causal_resolution_index(
            PUBLIC_LINEAGE_REGISTRY.model_copy(update={"candidates": ()})
        )

    def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("fault-injection detail")

    monkeypatch.setattr(public_generator, "preview_to_scenario_id", fail)
    with pytest.raises(PublicGeneratorInputError, match="failed validation"):
        compose_role_neutral_parts(preview, delta, scenario_id=scenario_id)
    monkeypatch.undo()

    monkeypatch.setattr(public_generator, "materialize_rendered_policy_skeleton", fail)
    with pytest.raises(PublicGeneratorInputError, match="failed validation"):
        render_public_causal_policy(preview, delta)
    monkeypatch.undo()

    monkeypatch.setattr(public_generator, "canonical_json", fail)
    with pytest.raises(PublicGeneratorInputError, match="failed validation"):
        compose_role_neutral_parts(preview, delta, scenario_id=scenario_id)


def test_causal_selection_rejects_failed_incomplete_and_ambiguous_execution(
    causal_index: PublicCausalResolutionIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview = PUBLIC_LINEAGE_REGISTRY.candidates[0].previews[0]

    def fail(*args: object, **kwargs: object) -> ScenarioOutcome:
        del args, kwargs
        raise RuntimeError("fault-injection detail")

    monkeypatch.setattr(public_generator, "derive_causal_outcome", fail)
    with pytest.raises(PublicGeneratorInputError, match="failed validation"):
        select_public_causal_delta(causal_index, preview, ScenarioOutcome.HELPFUL)

    monkeypatch.setattr(
        public_generator,
        "derive_causal_outcome",
        lambda *args, **kwargs: ScenarioOutcome.HELPFUL,
    )
    with pytest.raises(PublicGeneratorInputError):
        select_public_causal_delta(causal_index, preview, ScenarioOutcome.HELPFUL)
    monkeypatch.undo()

    duplicate_preview = preview.model_copy(
        update={
            "rendered_causal_deltas": (
                *preview.rendered_causal_deltas,
                preview.rendered_causal_deltas[0],
            )
        }
    )
    duplicate_preview = duplicate_preview.model_copy(
        update={"preview_digest": skeleton_preview_digest(duplicate_preview)}
    )
    duplicated_outcome = derive_causal_outcome(
        PUBLIC_LINEAGE_REGISTRY.candidates[0].transition_graph,
        preview.rendered_causal_deltas[0].factor_values,
    )
    with pytest.raises(PublicGeneratorInputError):
        select_public_causal_delta(causal_index, duplicate_preview, duplicated_outcome)


@pytest.mark.asyncio
async def test_signal_validation_sanitizes_evaluator_failures_and_projection_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview = PUBLIC_LINEAGE_REGISTRY.candidates[0].previews[0]
    scenario_id = preview_to_scenario_id(preview)
    real_evaluator = public_generator.evaluate_legacy_signal_fixture  # type: ignore[attr-defined]

    async def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("fault-injection detail")

    monkeypatch.setattr(public_generator, "evaluate_legacy_signal_fixture", fail)
    with pytest.raises(PublicGeneratorInputError, match="failed validation"):
        await verify_public_signal_fixture(preview, scenario_id=scenario_id)

    monkeypatch.setattr(public_generator, "evaluate_legacy_signal_fixture", real_evaluator)
    monkeypatch.setattr(
        public_generator,
        "detected_signal_projection",
        lambda **kwargs: (),
    )
    with pytest.raises(PublicGeneratorInputError):
        await verify_public_signal_fixture(preview, scenario_id=scenario_id)


@pytest.mark.asyncio
async def test_future_orchestration_executes_from_one_registered_live_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = object.__new__(PublicGenerationAuthority)
    review_subreport = object.__new__(PublicLineageReviewSubreport)
    state = generation_authority._PublicGenerationAuthorityState(
        registry=PUBLIC_LINEAGE_REGISTRY,
        review_subreport=review_subreport,
        capability_kind="ready-public-generation-registry",
    )
    monkeypatch.setitem(
        generation_authority._ISSUED_PUBLIC_GENERATION_AUTHORITIES,
        id(authority),
        (weakref.ref(authority), state),
    )

    generated = await generate_public_scenarios(authority)

    assert len(generated) == 900
    assert len({scenario[0].scenario_id for scenario in generated}) == 900
    assert Counter(scenario[0].split for scenario in generated) == {
        BenchmarkSplit.TRAIN: 600,
        BenchmarkSplit.DEVELOPMENT: 300,
    }


def test_public_delta_vector_literal_goldens() -> None:
    candidate = PUBLIC_LINEAGE_REGISTRY.candidates[0]
    assert tuple(delta.causal_delta_digest for delta in candidate.causal_deltas) == (
        "849db05f521ec542d7ad4f3c6a62ad9b03cf7b0327e9d7d6cf9bd1b399696d36",
        "aca76ca6a6758adc1a8022ad58f89c406f10b0d11561306b3bfa53ca5de4fa8f",
        "e4ceb75a8e9eb22401c121fa5fa271301be5c87c22c0c672d7c13a12bfbbd833",
        "7c499bd54f2e3dbf38d1d6491a95356f1d5a0c1eb52d55b8e6411c0215c69493",
    )
    assert tuple(
        delta.rendered_policy_digest for delta in candidate.previews[0].rendered_causal_deltas
    ) == (
        "8618ff2034b209558ce410c2c2b548ad479836c11acfd9598555646456af4ea0",
        "6e63279d187f40812ed869ecd8b161c07a4ff762602167faab21c80b310d4cbb",
        "6e8bea9387ed1cbcdef1d0a799b11a03dc3ffe65f7a32f11661065a9b00a4574",
        "85ad1f210ad308e10f183a54cf7f72be598d065ae15552082acb1b1bffd750c0",
    )
    complete_matrix = tuple(
        (
            item.lineage_registry_key,
            tuple(delta.causal_delta_digest for delta in item.causal_deltas),
            tuple(
                tuple(delta.rendered_policy_digest for delta in preview.rendered_causal_deltas)
                for preview in item.previews
            ),
        )
        for item in PUBLIC_LINEAGE_REGISTRY.candidates
    )
    assert sha256(canonical_json(complete_matrix)).hexdigest() == (
        "520497562afe4a56b288e8c2b79b6947d29c6f2e628a5196b1b3f605ec3f51e0"
    )
