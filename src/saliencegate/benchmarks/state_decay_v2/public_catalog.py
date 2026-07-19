from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, cast

from saliencegate.benchmarks.state_decay_v2.config import (
    GENERATION_CONTRACT,
    CounterbalanceAxis,
)
from saliencegate.benchmarks.state_decay_v2.public_contract import (
    PublicCounterbalanceProfile,
    PublicEvidenceProfile,
    PublicExpectedAssertionEvidence,
    PublicExpectedDetectorEvidence,
    PublicExpectedMemoryEvidence,
    PublicExpectedSignal,
    PublicGeneratorConfiguration,
    PublicIntegerProfile,
    PublicParameterProfile,
    PublicParameterValue,
    PublicProfileCatalog,
    PublicSignalFixtureVariant,
    PublicSignalProfile,
    PublicSlotProfile,
    PublicStructuralProfile,
    PublicTextLengthProfile,
    profile_catalog_digest,
    public_lineage_key,
    signal_profile_digest,
)
from saliencegate.benchmarks.state_decay_v2.schema import ScenarioFamily
from saliencegate.domain import SignalType, ValidityState


@dataclass(frozen=True, slots=True)
class PublicLineageDefinition:
    family: ScenarioFamily
    lineage_index: int
    lineage_registry_key: str
    locus_id: str
    family_title: str
    retained_subject: str
    alternate_subject: str
    semantic_rationale: str


_PUBLIC_FAMILIES: Final[tuple[ScenarioFamily, ...]] = (
    ScenarioFamily.FORGOTTEN_REQUIREMENT,
    ScenarioFamily.FAILED_PRIOR_ATTEMPT,
    ScenarioFamily.NEGLECTED_SUBGOAL,
    ScenarioFamily.STALE_MEMORY,
    ScenarioFamily.STABLE_ENVIRONMENT_FACT,
    ScenarioFamily.RETAINED_DIAGNOSIS,
)

_SEMANTIC_LOCI: Final[tuple[str, ...]] = (
    "access-boundary",
    "archive-format",
    "artifact-owner",
    "backup-window",
    "cache-generation",
    "change-ticket",
    "client-capability",
    "command-scope",
    "compatibility-level",
    "concurrency-limit",
    "data-retention",
    "dependency-pin",
    "deployment-region",
    "error-budget",
    "feature-flag",
    "file-encoding",
    "identity-scope",
    "index-version",
    "migration-phase",
    "network-boundary",
    "ordering-rule",
    "permission-mode",
    "protocol-version",
    "release-channel",
    "resource-ceiling",
    "retry-policy",
    "schema-revision",
    "service-owner",
    "storage-class",
    "validation-rule",
)

_FAMILY_TITLES: Final[MappingProxyType[ScenarioFamily, str]] = MappingProxyType(
    {
        ScenarioFamily.FORGOTTEN_REQUIREMENT: "Forgotten requirement",
        ScenarioFamily.FAILED_PRIOR_ATTEMPT: "Failed prior attempt",
        ScenarioFamily.NEGLECTED_SUBGOAL: "Neglected subgoal",
        ScenarioFamily.STALE_MEMORY: "Stale memory",
        ScenarioFamily.STABLE_ENVIRONMENT_FACT: "Stable environment fact",
        ScenarioFamily.RETAINED_DIAGNOSIS: "Retained diagnosis",
    }
)

_FAMILY_SUBJECTS: Final[MappingProxyType[ScenarioFamily, tuple[str, str]]] = MappingProxyType(
    {
        ScenarioFamily.FORGOTTEN_REQUIREMENT: (
            "the retained requirement",
            "the recent request",
        ),
        ScenarioFamily.FAILED_PRIOR_ATTEMPT: (
            "the recorded failed approach",
            "the familiar approach",
        ),
        ScenarioFamily.NEGLECTED_SUBGOAL: (
            "the open subgoal",
            "the immediate action",
        ),
        ScenarioFamily.STALE_MEMORY: (
            "the current memory revision",
            "the earlier memory revision",
        ),
        ScenarioFamily.STABLE_ENVIRONMENT_FACT: (
            "the attested environment fact",
            "the assumed environment value",
        ),
        ScenarioFamily.RETAINED_DIAGNOSIS: (
            "the retained diagnosis",
            "the surface symptom",
        ),
    }
)


def _build_lineage_definitions() -> tuple[PublicLineageDefinition, ...]:
    definitions: list[PublicLineageDefinition] = []
    for family in _PUBLIC_FAMILIES:
        title = _FAMILY_TITLES[family]
        retained_subject, alternate_subject = _FAMILY_SUBJECTS[family]
        for index, locus_id in enumerate(_SEMANTIC_LOCI):
            key = public_lineage_key(family, index)
            definitions.append(
                PublicLineageDefinition(
                    family=family,
                    lineage_index=index,
                    lineage_registry_key=key,
                    locus_id=locus_id,
                    family_title=title,
                    retained_subject=retained_subject,
                    alternate_subject=alternate_subject,
                    semantic_rationale=(
                        f"{title} lineage {key} isolates the {locus_id.replace('-', ' ')} "
                        "decision while fixed global profiles control presentation variation."
                    ),
                )
            )
    return tuple(definitions)


PUBLIC_LINEAGE_DEFINITIONS: Final[tuple[PublicLineageDefinition, ...]] = (
    _build_lineage_definitions()
)


def _evidence(
    *,
    events: tuple[int, ...],
    bindings: tuple[int, ...] = (),
    memories: tuple[tuple[int, int], ...] = (),
    assertions: tuple[tuple[int, int], ...] = (),
) -> PublicExpectedDetectorEvidence:
    return PublicExpectedDetectorEvidence(
        event_pool_indices=events,
        binding_event_pool_indices=bindings,
        memory_references=tuple(
            PublicExpectedMemoryEvidence(memory_pool_index=index, revision=revision)
            for index, revision in memories
        ),
        assertion_references=tuple(
            PublicExpectedAssertionEvidence(
                binding_event_pool_index=event_index,
                assertion_index=assertion_index,
            )
            for event_index, assertion_index in assertions
        ),
    )


def _signal_profile(
    slot: int,
    variant: PublicSignalFixtureVariant,
    signals: tuple[PublicExpectedSignal, ...],
) -> PublicSignalProfile:
    payload: dict[str, object] = {
        "profile_id": f"slot-{slot}-signal-profile",
        "fixture_variant": variant,
        "expected_signals": signals,
        "profile_digest": "0" * 64,
    }
    payload["profile_digest"] = signal_profile_digest(payload)
    return PublicSignalProfile.model_validate(payload)


def _signal_profiles() -> tuple[PublicSignalProfile, ...]:
    return (
        _signal_profile(
            0,
            PublicSignalFixtureVariant.FAILED_TEST_CONFLICT_MISSING_CONSTRAINT,
            (
                PublicExpectedSignal(
                    signal_type=SignalType.CONFLICT,
                    strength_ppm=1_000_000,
                    evidence=_evidence(
                        events=(2,),
                        bindings=(2,),
                        assertions=((2, 0), (2, 1)),
                    ),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.STALE_CONSTRAINT,
                    strength_ppm=1_000_000,
                    evidence=_evidence(events=(2,), bindings=(2,), memories=((1, 1),)),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.TEST_FAILURE,
                    strength_ppm=1_000_000,
                    evidence=_evidence(events=(2,)),
                ),
            ),
        ),
        _signal_profile(
            1,
            PublicSignalFixtureVariant.REPEATED_ACTION_SCOPE_SHIFT_IRREVERSIBLE,
            (
                PublicExpectedSignal(
                    signal_type=SignalType.CONTEXT_SHIFT,
                    strength_ppm=500_000,
                    evidence=_evidence(
                        events=(2, 3),
                        bindings=(2, 3),
                        memories=((0, 2),),
                    ),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.IRREVERSIBLE_ACTION,
                    strength_ppm=1_000_000,
                    evidence=_evidence(events=(3,), bindings=(3,)),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.REPEATED_ACTION,
                    strength_ppm=1_000_000,
                    evidence=_evidence(events=(2, 3)),
                ),
            ),
        ),
        _signal_profile(
            2,
            PublicSignalFixtureVariant.STAGNANT_CONFLICTING_ASSERTIONS,
            (
                PublicExpectedSignal(
                    signal_type=SignalType.CONFLICT,
                    strength_ppm=1_000_000,
                    evidence=_evidence(
                        events=(4,),
                        bindings=(4,),
                        assertions=((4, 0), (4, 1)),
                    ),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.STAGNATION,
                    strength_ppm=500_000,
                    evidence=_evidence(events=(1, 2, 3, 4), bindings=(1, 2, 3, 4)),
                ),
            ),
        ),
        _signal_profile(
            3,
            PublicSignalFixtureVariant.REPEATED_FAILURE_SUPERSEDED_CONSTRAINT,
            (
                PublicExpectedSignal(
                    signal_type=SignalType.REPEATED_FAILURE,
                    strength_ppm=1_000_000,
                    evidence=_evidence(events=(2, 3, 4, 5)),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.STALE_CONSTRAINT,
                    strength_ppm=750_000,
                    evidence=_evidence(events=(5,), bindings=(5,), memories=((0, 4),)),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.TOOL_ERROR,
                    strength_ppm=1_000_000,
                    evidence=_evidence(events=(5,)),
                ),
            ),
        ),
        _signal_profile(
            4,
            PublicSignalFixtureVariant.REPEATED_ACTION_SCOPE_SHIFT_STAGNATION,
            (
                PublicExpectedSignal(
                    signal_type=SignalType.CONTEXT_SHIFT,
                    strength_ppm=500_000,
                    evidence=_evidence(
                        events=(5, 6),
                        bindings=(5, 6),
                        memories=((0, 5),),
                    ),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.REPEATED_ACTION,
                    strength_ppm=1_000_000,
                    evidence=_evidence(events=(2, 6)),
                ),
                PublicExpectedSignal(
                    signal_type=SignalType.STAGNATION,
                    strength_ppm=625_000,
                    evidence=_evidence(
                        events=(2, 3, 4, 5, 6),
                        bindings=(2, 3, 4, 5, 6),
                    ),
                ),
            ),
        ),
    )


def build_public_profile_catalog(
    configuration: PublicGeneratorConfiguration,
) -> PublicProfileCatalog:
    signal_profiles = _signal_profiles()
    validities = (
        ValidityState.ACTIVE,
        ValidityState.ACTIVE,
        ValidityState.INVALIDATED,
        ValidityState.SUPERSEDED,
        ValidityState.ACTIVE,
    )
    action_orders: tuple[tuple[Literal[0, 1], Literal[0, 1]], ...] = (
        (0, 1),
        (1, 0),
        (0, 1),
        (1, 0),
        (0, 1),
    )
    memory_counts = (1, 2, 3, 4, 2)
    evidence_counts = (1, 4, 7, 10, 5)
    slots: list[PublicSlotProfile] = []
    for slot in range(5):
        validity = validities[slot]
        action_order = action_orders[slot]
        slots.append(
            PublicSlotProfile(
                generator_slot=slot,
                counterbalance=PublicCounterbalanceProfile(
                    profile_id=f"slot-{slot}-counterbalance",
                    allowed_action_order=action_order,
                    decisive_action_position=cast(Literal[0, 1], action_order.index(0)),
                    memory_validity=validity,
                    include_validity_transition=validity is not ValidityState.ACTIVE,
                ),
                parameters=PublicParameterProfile(
                    profile_id=f"slot-{slot}-parameters",
                    allowed_values=(
                        PublicParameterValue(
                            parameter_id=f"slot-{slot}-budget",
                            value=100 + slot * 17,
                        ),
                        PublicParameterValue(
                            parameter_id=f"slot-{slot}-depth",
                            value=2 + slot,
                        ),
                    ),
                ),
                structure=PublicStructuralProfile(
                    profile_id=f"slot-{slot}-structure",
                    trajectory_event_count=3 + slot,
                    candidate_memory_count=memory_counts[slot],
                ),
                integers=PublicIntegerProfile(
                    profile_id=f"slot-{slot}-integers",
                    sequence_start=10_000 + slot * 10_000,
                    sequence_stride=7 + slot * 2,
                    action_step_start=slot * 11,
                    action_step_stride=2 + slot,
                    memory_revision=(1, 2, 7, 4, 5)[slot],
                ),
                evidence=PublicEvidenceProfile(
                    profile_id=f"slot-{slot}-evidence",
                    evidence_reference_count=evidence_counts[slot],
                    decisive_event_count=(1, 2, 3, 4, 5)[slot],
                    decisive_memory_count=(1, 1, 2, 2, 1)[slot],
                ),
                text_lengths=PublicTextLengthProfile(
                    profile_id=f"slot-{slot}-text-lengths",
                    event_padding_spaces=slot,
                    memory_padding_spaces=(slot + 1) % 5,
                    pivot_padding_spaces=(slot * 2) % 7,
                    action_padding_spaces=(slot * 3) % 6,
                ),
                signals=signal_profiles[slot],
            )
        )

    payload: dict[str, object] = {
        "schema_version": "state-decay-v2-public-profile-catalog/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "generation_contract_digest": GENERATION_CONTRACT.contract_digest,
        "generator_configuration_digest": configuration.configuration_digest,
        "counterbalance_axes": tuple(CounterbalanceAxis),
        "slot_profiles": tuple(slots),
        "catalog_digest": "0" * 64,
    }
    payload["catalog_digest"] = profile_catalog_digest(payload)
    return PublicProfileCatalog.model_validate(payload)


__all__ = [
    "PUBLIC_LINEAGE_DEFINITIONS",
    "PublicLineageDefinition",
    "build_public_profile_catalog",
]
