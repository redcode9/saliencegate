from __future__ import annotations

import weakref
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Never, SupportsIndex, TypeAlias

from saliencegate.benchmarks.state_decay_v2.authority import (
    DecisiveEvidence,
    MemoryRevisionEvidence,
)
from saliencegate.benchmarks.state_decay_v2.generation_authority import (
    require_public_generation_authority,
)
from saliencegate.benchmarks.state_decay_v2.preallocation import preview_to_scenario_id
from saliencegate.benchmarks.state_decay_v2.public_contract import (
    OutcomeFreeTaskSkeleton,
    OutcomeFreeTraceFixture,
    PreAllocationSkeletonPreview,
    PublicCausalFactorValue,
    PublicExpectedSignal,
    PublicLineageRegistry,
    PublicTransitionGraph,
    RenderedCausalSemanticDelta,
    RoleNeutralGeneratedParts,
    derive_causal_outcome,
    materialize_rendered_policy_skeleton,
    rendered_policy_digest,
    skeleton_preview_digest,
)
from saliencegate.benchmarks.state_decay_v2.schema import BenchmarkSplit, ScenarioOutcome
from saliencegate.benchmarks.state_decay_v2.signal_fixtures import (
    LegacyFixtureEvaluation,
    ReferencePredicateResult,
    detected_signal_projection,
    evaluate_legacy_signal_fixture,
    evaluate_reference_predicates,
)
from saliencegate.domain import canonical_json

if TYPE_CHECKING:
    from saliencegate.benchmarks.state_decay_v2.config import LineageAllocation


class PublicGeneratorInputError(ValueError):
    """A value-free failure at the public generator boundary."""

    def __init__(self) -> None:
        super().__init__("public generator input failed validation")


@dataclass(frozen=True, slots=True)
class PublicCausalResolution:
    """The only causal authority recoverable from a complete rendered policy."""

    transition_graph: PublicTransitionGraph
    factor_values: tuple[PublicCausalFactorValue, ...]

    def __post_init__(self) -> None:
        if (
            type(self.transition_graph) is not PublicTransitionGraph
            or type(self.factor_values) is not tuple
            or len(self.factor_values) != 2
            or any(type(value) is not PublicCausalFactorValue for value in self.factor_values)
            or tuple(value.factor_id for value in self.factor_values)
            != tuple(factor.factor_id for factor in self.transition_graph.factors)
        ):
            raise PublicGeneratorInputError()


class PublicCausalResolutionIndex:
    """Opaque complete-policy index; it deliberately exposes no digest lookup API."""

    __slots__ = ("__bindings", "__weakref__")
    __bindings: Mapping[str, PublicCausalResolution]

    def __new__(cls, *args: object, **kwargs: object) -> PublicCausalResolutionIndex:
        del cls, args, kwargs
        raise PublicGeneratorInputError()

    def __setattr__(self, name: str, value: object) -> None:
        del self, name, value
        raise PublicGeneratorInputError()

    def __delattr__(self, name: str) -> None:
        del self, name
        raise PublicGeneratorInputError()

    def __copy__(self) -> Never:
        del self
        raise PublicGeneratorInputError()

    def __deepcopy__(self, memo: dict[int, object]) -> Never:
        del self, memo
        raise PublicGeneratorInputError()

    def __reduce__(self) -> Never:
        del self
        raise PublicGeneratorInputError()

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del self, protocol
        raise PublicGeneratorInputError()

    @property
    def binding_count(self) -> int:
        return len(self.__bindings)

    def _resolve_complete_policy(
        self,
        rendered_policy: OutcomeFreeTaskSkeleton,
    ) -> PublicCausalResolution:
        if type(rendered_policy) is not OutcomeFreeTaskSkeleton:
            raise PublicGeneratorInputError()
        try:
            checked = OutcomeFreeTaskSkeleton.model_validate_json(canonical_json(rendered_policy))
            digest = rendered_policy_digest(checked)
            return self.__bindings[digest]
        except Exception:
            raise PublicGeneratorInputError() from None


_ISSUED_PUBLIC_CAUSAL_INDEXES: dict[
    int,
    weakref.ReferenceType[PublicCausalResolutionIndex],
] = {}


def _require_public_causal_resolution_index(
    value: object,
) -> PublicCausalResolutionIndex:
    if type(value) is not PublicCausalResolutionIndex:
        raise PublicGeneratorInputError()
    reference = _ISSUED_PUBLIC_CAUSAL_INDEXES.get(id(value))
    if reference is None or reference() is not value:
        raise PublicGeneratorInputError()
    return value


def _new_public_causal_resolution_index(
    bindings: dict[str, PublicCausalResolution],
) -> PublicCausalResolutionIndex:
    if (
        type(bindings) is not dict
        or len(bindings) != 3_600
        or any(
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or type(resolution) is not PublicCausalResolution
            for digest, resolution in bindings.items()
        )
    ):
        raise PublicGeneratorInputError()
    index = object.__new__(PublicCausalResolutionIndex)
    object.__setattr__(
        index,
        "_PublicCausalResolutionIndex__bindings",
        MappingProxyType(dict(bindings)),
    )
    _ISSUED_PUBLIC_CAUSAL_INDEXES[id(index)] = weakref.ref(index)
    return index


@dataclass(frozen=True, slots=True)
class PublicSignalValidation:
    legacy: LegacyFixtureEvaluation
    reference_results: tuple[ReferencePredicateResult, ...]
    projection: tuple[PublicExpectedSignal, ...]

    def __post_init__(self) -> None:
        if (
            type(self.legacy) is not LegacyFixtureEvaluation
            or type(self.reference_results) is not tuple
            or len(self.reference_results) != 5
            or any(
                type(result) is not ReferencePredicateResult for result in self.reference_results
            )
            or type(self.projection) is not tuple
            or any(type(signal) is not PublicExpectedSignal for signal in self.projection)
        ):
            raise PublicGeneratorInputError()


@dataclass(frozen=True, slots=True)
class SyntheticPublicAssignment:
    candidate_packet_digest: str
    generator_slot: int
    scenario_id: str
    assigned_outcome: ScenarioOutcome

    def __post_init__(self) -> None:
        if (
            type(self.candidate_packet_digest) is not str
            or len(self.candidate_packet_digest) != 64
            or any(
                character not in "0123456789abcdef" for character in self.candidate_packet_digest
            )
            or type(self.generator_slot) is not int
            or not 0 <= self.generator_slot <= 4
            or type(self.scenario_id) is not str
            or len(self.scenario_id) != 64
            or any(character not in "0123456789abcdef" for character in self.scenario_id)
            or type(self.assigned_outcome) is not ScenarioOutcome
        ):
            raise PublicGeneratorInputError()


GeneratedPublicScenario: TypeAlias = tuple[
    RoleNeutralGeneratedParts,
    OutcomeFreeTraceFixture,
    DecisiveEvidence,
]


def _checked_registry(registry: PublicLineageRegistry) -> PublicLineageRegistry:
    if type(registry) is not PublicLineageRegistry:
        raise PublicGeneratorInputError()
    try:
        return PublicLineageRegistry.model_validate_json(canonical_json(registry))
    except Exception:
        raise PublicGeneratorInputError() from None


def _checked_scenario_id(scenario_id: str) -> str:
    if (
        type(scenario_id) is not str
        or len(scenario_id) != 64
        or any(character not in "0123456789abcdef" for character in scenario_id)
    ):
        raise PublicGeneratorInputError()
    return scenario_id


def _checked_preview(
    preview: PreAllocationSkeletonPreview,
) -> PreAllocationSkeletonPreview:
    if type(
        preview
    ) is not PreAllocationSkeletonPreview or preview.preview_digest != skeleton_preview_digest(
        preview
    ):
        raise PublicGeneratorInputError()
    return preview


def _checked_preview_scenario_id(
    preview: PreAllocationSkeletonPreview,
    scenario_id: str,
) -> str:
    checked_preview = _checked_preview(preview)
    checked = _checked_scenario_id(scenario_id)
    try:
        expected = preview_to_scenario_id(checked_preview)
    except Exception:
        raise PublicGeneratorInputError() from None
    if checked != expected:
        raise PublicGeneratorInputError()
    return checked


def build_public_causal_resolution_index(
    registry: PublicLineageRegistry,
) -> PublicCausalResolutionIndex:
    """Bind every complete rendered policy to graph and factors—and nothing else."""

    checked = _checked_registry(registry)
    bindings: dict[str, PublicCausalResolution] = {}
    for candidate in checked.candidates:
        for preview in candidate.previews:
            for delta in preview.rendered_causal_deltas:
                rendered = _render_public_causal_policy_checked(preview, delta)
                digest = rendered_policy_digest(rendered)
                if digest != delta.rendered_policy_digest or digest in bindings:
                    raise PublicGeneratorInputError()
                bindings[digest] = PublicCausalResolution(
                    transition_graph=candidate.transition_graph,
                    factor_values=delta.factor_values,
                )
    return _new_public_causal_resolution_index(bindings)


def render_public_causal_policy(
    preview: PreAllocationSkeletonPreview,
    delta: RenderedCausalSemanticDelta,
) -> OutcomeFreeTaskSkeleton:
    """Render one prebound causal delta into the complete outcome-free policy skeleton."""

    checked_preview = _checked_preview(preview)
    return _render_public_causal_policy_checked(checked_preview, delta)


def _render_public_causal_policy_checked(
    preview: PreAllocationSkeletonPreview,
    delta: RenderedCausalSemanticDelta,
) -> OutcomeFreeTaskSkeleton:
    if type(delta) is not RenderedCausalSemanticDelta or delta not in (
        preview.rendered_causal_deltas
    ):
        raise PublicGeneratorInputError()
    try:
        return materialize_rendered_policy_skeleton(preview.task_skeleton, delta)
    except Exception:
        raise PublicGeneratorInputError() from None


def resolve_public_causal_policy(
    index: PublicCausalResolutionIndex,
    rendered_policy: OutcomeFreeTaskSkeleton,
) -> PublicCausalResolution:
    """Resolve causal authority by recomputing the digest of the complete policy only."""

    checked_index = _require_public_causal_resolution_index(index)
    return checked_index._resolve_complete_policy(rendered_policy)


def select_public_causal_delta(
    index: PublicCausalResolutionIndex,
    preview: PreAllocationSkeletonPreview,
    assigned_outcome: ScenarioOutcome,
) -> RenderedCausalSemanticDelta:
    """Select the sole delta whose resolved graph derives the typed assignment."""

    checked_index = _require_public_causal_resolution_index(index)
    checked_preview = _checked_preview(preview)
    if type(assigned_outcome) is not ScenarioOutcome:
        raise PublicGeneratorInputError()
    resolved: list[tuple[ScenarioOutcome, RenderedCausalSemanticDelta]] = []
    for delta in checked_preview.rendered_causal_deltas:
        policy = _render_public_causal_policy_checked(checked_preview, delta)
        resolution = resolve_public_causal_policy(checked_index, policy)
        try:
            outcome = derive_causal_outcome(
                resolution.transition_graph,
                resolution.factor_values,
            )
        except Exception:
            raise PublicGeneratorInputError() from None
        resolved.append((outcome, delta))
    if {outcome for outcome, _ in resolved} != set(ScenarioOutcome):
        raise PublicGeneratorInputError()
    matches = tuple(delta for outcome, delta in resolved if outcome is assigned_outcome)
    if len(matches) != 1:
        raise PublicGeneratorInputError()
    return matches[0]


def compose_role_neutral_parts(
    preview: PreAllocationSkeletonPreview,
    delta: RenderedCausalSemanticDelta,
    *,
    scenario_id: str,
) -> GeneratedPublicScenario:
    """Purely compose safe policy fields plus the two typed, non-policy siblings."""

    checked_id = _checked_preview_scenario_id(preview, scenario_id)
    rendered = _render_public_causal_policy_checked(preview, delta)
    profile = preview.slot_profile
    decisive_position = profile.counterbalance.decisive_action_position
    decisive_action_id = rendered.allowed_actions[decisive_position].action_id
    parts = RoleNeutralGeneratedParts(
        split=preview.split,
        scenario_id=checked_id,
        trajectory=rendered.trajectory,
        candidate_memories=rendered.candidate_memories,
        pivot=rendered.pivot,
        allowed_actions=rendered.allowed_actions,
        adapter=rendered.adapter,
    )
    try:
        fixture = OutcomeFreeTraceFixture.model_validate_json(canonical_json(preview.trace_fixture))
        evidence = DecisiveEvidence(
            event_ids=tuple(
                event.event_id
                for event in rendered.trajectory[: profile.evidence.decisive_event_count]
            ),
            memory_revisions=tuple(
                MemoryRevisionEvidence(memory_id=memory.memory_id, revision=memory.revision)
                for memory in rendered.candidate_memories[: profile.evidence.decisive_memory_count]
            ),
            decisive_action_id=decisive_action_id,
        )
    except Exception:
        raise PublicGeneratorInputError() from None
    return parts, fixture, evidence


def compose_synthetic_public_scenarios(
    registry: PublicLineageRegistry,
    index: PublicCausalResolutionIndex,
    assignments: tuple[SyntheticPublicAssignment, ...],
) -> tuple[GeneratedPublicScenario, ...]:
    """Exercise the production composer with a closed canonical synthetic assignment set."""

    checked_registry = _checked_registry(registry)
    _require_public_causal_resolution_index(index)
    if (
        type(assignments) is not tuple
        or len(assignments) != 900
        or any(type(assignment) is not SyntheticPublicAssignment for assignment in assignments)
    ):
        raise PublicGeneratorInputError()
    expected_coordinates = tuple(
        (candidate, preview)
        for candidate in checked_registry.candidates
        for preview in candidate.previews
    )
    generated: list[GeneratedPublicScenario] = []
    for assignment, (candidate, preview) in zip(
        assignments,
        expected_coordinates,
        strict=True,
    ):
        if (
            assignment.candidate_packet_digest != candidate.candidate_packet_digest
            or assignment.generator_slot != preview.generator_slot
        ):
            raise PublicGeneratorInputError()
        delta = select_public_causal_delta(
            index,
            preview,
            assignment.assigned_outcome,
        )
        generated.append(
            compose_role_neutral_parts(
                preview,
                delta,
                scenario_id=assignment.scenario_id,
            )
        )
    scenario_ids = tuple(item[0].scenario_id for item in generated)
    split_counts = Counter(item[0].split for item in generated)
    if len(set(scenario_ids)) != 900 or split_counts != {
        BenchmarkSplit.TRAIN: 600,
        BenchmarkSplit.DEVELOPMENT: 300,
    }:
        raise PublicGeneratorInputError()
    return tuple(generated)


async def verify_public_signal_fixture(
    preview: PreAllocationSkeletonPreview,
    *,
    scenario_id: str,
) -> PublicSignalValidation:
    """Validate real and reference signal paths separately, then compare their common projection."""

    checked_id = _checked_preview_scenario_id(preview, scenario_id)
    try:
        legacy = await evaluate_legacy_signal_fixture(
            preview.trace_fixture,
            scenario_id=checked_id,
        )
        reference_results = evaluate_reference_predicates(
            preview.trace_fixture,
            preview.task_skeleton,
        )
        projection = detected_signal_projection(
            legacy_results=legacy.results,
            reference_results=reference_results,
        )
    except Exception:
        raise PublicGeneratorInputError() from None
    if projection != preview.slot_profile.signals.expected_signals:
        raise PublicGeneratorInputError()
    return PublicSignalValidation(
        legacy=legacy,
        reference_results=reference_results,
        projection=projection,
    )


async def generate_public_scenarios(authority: object) -> tuple[GeneratedPublicScenario, ...]:
    """Capability-gated generation boundary with no caller before review authority exists."""

    authority_state = require_public_generation_authority(authority)

    # These imports materialize the public registry and expose allocation only after authority.
    from saliencegate.benchmarks.state_decay_v2.config import (
        PUBLIC_GENERATION_SEED,
        SeedPurpose,
        allocate_balanced_outcomes,
        derive_seed,
    )
    from saliencegate.benchmarks.state_decay_v2.public_catalog import PUBLIC_LINEAGE_DEFINITIONS
    from saliencegate.benchmarks.state_decay_v2.review_contract import (
        PublicLineageReviewSubreport,
    )
    from saliencegate.benchmarks.state_decay_v2.templates import (
        PUBLIC_GENERATOR_ALGORITHM,
        PUBLIC_GENERATOR_CONFIGURATION,
        PUBLIC_PROFILE_CATALOG,
        validate_public_lineage_registry_materialization,
    )

    if (
        authority_state.capability_kind != "ready-public-generation-registry"
        or type(authority_state.registry) is not PublicLineageRegistry
        or type(authority_state.review_subreport) is not PublicLineageReviewSubreport
    ):
        raise PublicGeneratorInputError()
    registry = validate_public_lineage_registry_materialization(
        authority_state.registry,
        PUBLIC_GENERATOR_CONFIGURATION,
        PUBLIC_PROFILE_CATALOG,
        PUBLIC_GENERATOR_ALGORITHM,
        PUBLIC_LINEAGE_DEFINITIONS,
    )
    scenario_ids = tuple(
        preview_to_scenario_id(preview)
        for candidate in registry.candidates
        for preview in candidate.previews
    )
    if len(scenario_ids) != 900 or len(set(scenario_ids)) != 900:
        raise PublicGeneratorInputError()
    index = build_public_causal_resolution_index(registry)

    allocation_leaf = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.ALLOCATION)
    allocations_by_lineage: dict[str, LineageAllocation] = {}
    for family in PUBLIC_GENERATOR_CONFIGURATION.visible_families:
        family_candidates = tuple(
            candidate for candidate in registry.candidates if candidate.family is family
        )
        lineages = tuple(
            (candidate.lineage_registry_key, candidate.candidate_packet_digest)
            for candidate in family_candidates
        )
        allocations = allocate_balanced_outcomes(allocation_leaf, family, lineages)
        allocations_by_lineage.update(
            (allocation.lineage_registry_key, allocation) for allocation in allocations
        )

    flattened = tuple(
        (candidate, preview) for candidate in registry.candidates for preview in candidate.previews
    )
    assignments: list[SyntheticPublicAssignment] = []
    for scenario_id, (candidate, preview) in zip(scenario_ids, flattened, strict=True):
        allocation = allocations_by_lineage.get(candidate.lineage_registry_key)
        if (
            allocation is None
            or allocation.candidate_packet_digest != candidate.candidate_packet_digest
        ):
            raise PublicGeneratorInputError()
        assignments.append(
            SyntheticPublicAssignment(
                candidate_packet_digest=candidate.candidate_packet_digest,
                generator_slot=preview.generator_slot,
                scenario_id=scenario_id,
                assigned_outcome=allocation.outcomes_by_slot[preview.generator_slot],
            )
        )
    generated = compose_synthetic_public_scenarios(registry, index, tuple(assignments))
    for (_, preview), scenario in zip(flattened, generated, strict=True):
        await verify_public_signal_fixture(
            preview,
            scenario_id=scenario[0].scenario_id,
        )
    return generated


__all__ = [
    "GeneratedPublicScenario",
    "PublicCausalResolution",
    "PublicCausalResolutionIndex",
    "PublicGeneratorInputError",
    "PublicSignalValidation",
    "SyntheticPublicAssignment",
    "build_public_causal_resolution_index",
    "compose_role_neutral_parts",
    "compose_synthetic_public_scenarios",
    "generate_public_scenarios",
    "render_public_causal_policy",
    "resolve_public_causal_policy",
    "select_public_causal_delta",
    "verify_public_signal_fixture",
]
