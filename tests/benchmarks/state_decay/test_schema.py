from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from saliencegate.benchmarks.state_decay.generator import generate_smoke_scenarios
from saliencegate.benchmarks.state_decay.schema import (
    CandidateMemory,
    EvidenceCriteria,
    MemorySourceRef,
    PairedContinuation,
    StateDecayScenario,
)
from saliencegate.domain import ValidityState


def _reject_scenario(scenario: StateDecayScenario) -> None:
    with pytest.raises(ValidationError):
        StateDecayScenario.model_validate_json(scenario.model_dump_json(warnings=False))


def test_memory_validity_and_evidence_collections_fail_closed() -> None:
    scenario = generate_smoke_scenarios()[0]
    memory = scenario.candidate_memories[0]
    reminder = scenario.paired_continuations[0]

    for forged in (
        memory.model_copy(update={"validity_step": 2}),
        memory.model_copy(update={"validity": ValidityState.INVALIDATED, "validity_step": None}),
    ):
        with pytest.raises(ValidationError):
            CandidateMemory.model_validate_json(forged.model_dump_json(warnings=False))

    for decisive in (
        EvidenceCriteria.model_construct(),
        EvidenceCriteria.model_construct(
            decisive_source_ids=("source-duplicate", "source-duplicate"),
            decisive_memory_ids=(),
        ),
    ):
        with pytest.raises(ValidationError):
            EvidenceCriteria.model_validate_json(decisive.model_dump_json(warnings=False))

    duplicated_evidence = reminder.model_copy(
        update={
            "evidence_source_ids": (
                reminder.evidence_source_ids[0],
                reminder.evidence_source_ids[0],
            )
        }
    )
    with pytest.raises(ValidationError):
        PairedContinuation.model_validate_json(duplicated_evidence.model_dump_json(warnings=False))


def test_scenario_rejects_order_identity_reference_and_action_forgery() -> None:
    scenario = generate_smoke_scenarios()[0]
    first, second, third = scenario.trajectory_prefix
    memory = scenario.candidate_memories[0]
    reference = memory.source_refs[0]
    required, alternate = scenario.allowed_actions
    reminder, silence = scenario.paired_continuations

    mutations: tuple[Callable[[], StateDecayScenario], ...] = (
        lambda: scenario.model_copy(
            update={
                "trajectory_prefix": (
                    first,
                    second.model_copy(update={"step": 1}),
                    third,
                )
            }
        ),
        lambda: scenario.model_copy(
            update={
                "trajectory_prefix": (
                    first,
                    second,
                    third.model_copy(update={"step": scenario.pivot.step}),
                )
            }
        ),
        lambda: scenario.model_copy(
            update={
                "trajectory_prefix": (
                    first,
                    second.model_copy(update={"source_id": first.source_id}),
                    third,
                )
            }
        ),
        lambda: scenario.model_copy(update={"candidate_memories": (memory, memory)}),
        lambda: scenario.model_copy(
            update={
                "candidate_memories": (
                    memory.model_copy(update={"source_refs": (reference, reference)}),
                )
            }
        ),
        lambda: scenario.model_copy(
            update={
                "candidate_memories": (
                    memory.model_copy(
                        update={"source_refs": (reference.model_copy(update={"source_step": 2}),)}
                    ),
                )
            }
        ),
        lambda: scenario.model_copy(
            update={
                "candidate_memories": (
                    memory.model_copy(
                        update={
                            "source_refs": (
                                MemorySourceRef(
                                    source_id=third.source_id,
                                    source_step=third.step,
                                ),
                            )
                        }
                    ),
                )
            }
        ),
        lambda: scenario.model_copy(
            update={
                "candidate_memories": (
                    memory.model_copy(update={"recorded_step": scenario.pivot.step + 1}),
                )
            }
        ),
        lambda: scenario.model_copy(
            update={
                "candidate_memories": (
                    memory.model_copy(
                        update={
                            "validity": ValidityState.SUPERSEDED,
                            "validity_step": scenario.pivot.step + 1,
                        }
                    ),
                )
            }
        ),
        lambda: scenario.model_copy(update={"allowed_actions": (required, required)}),
        lambda: scenario.model_copy(
            update={
                "oracle": scenario.oracle.model_copy(
                    update={"required_action_id": "action-not-allowed"}
                )
            }
        ),
        lambda: scenario.model_copy(
            update={
                "evidence_criteria": scenario.evidence_criteria.model_copy(
                    update={"decisive_source_ids": ("source-not-present",)}
                )
            }
        ),
        lambda: scenario.model_copy(
            update={
                "evidence_criteria": scenario.evidence_criteria.model_copy(
                    update={"decisive_memory_ids": ("memory-not-present",)}
                )
            }
        ),
        lambda: scenario.model_copy(update={"paired_continuations": (reminder, reminder)}),
        lambda: scenario.model_copy(
            update={
                "paired_continuations": (
                    reminder.model_copy(update={"selected_action_id": "action-not-allowed"}),
                    silence,
                )
            }
        ),
        lambda: scenario.model_copy(
            update={
                "paired_continuations": (
                    reminder.model_copy(update={"evidence_source_ids": ("source-not-present",)}),
                    silence,
                )
            }
        ),
        lambda: scenario.model_copy(
            update={
                "paired_continuations": (
                    reminder.model_copy(update={"evidence_memory_ids": ("memory-not-present",)}),
                    silence,
                )
            }
        ),
        lambda: scenario.model_copy(
            update={
                "allowed_actions": (
                    required,
                    alternate,
                ),
                "pivot": scenario.pivot.model_copy(update={"source_id": first.source_id}),
            }
        ),
    )

    for mutate in mutations:
        _reject_scenario(mutate())


def test_inactive_or_future_decisive_memory_is_unavailable() -> None:
    scenario = generate_smoke_scenarios()[0]
    memory = scenario.candidate_memories[0]

    inactive = memory.model_copy(update={"validity": ValidityState.SUPERSEDED, "validity_step": 3})
    _reject_scenario(scenario.model_copy(update={"candidate_memories": (inactive,)}))

    future = memory.model_copy(update={"recorded_step": scenario.pivot.step + 1})
    _reject_scenario(scenario.model_copy(update={"candidate_memories": (future,)}))
