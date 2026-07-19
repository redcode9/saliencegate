from __future__ import annotations

from typing import Literal, NoReturn

from pydantic import BaseModel, ConfigDict

from saliencegate.benchmarks.state_decay.schema import (
    ContinuationBranch,
    ContinuationOutcome,
    InterventionLabel,
    PairedContinuation,
    StateDecayScenario,
)
from saliencegate.domain import ValidityState
from saliencegate.domain.records import ComponentIdentifier, Sha256Digest

ORACLE_RESULT_SCHEMA_VERSION: Literal["state-decay-oracle-result/v1"] = (
    "state-decay-oracle-result/v1"
)


class OracleEvaluationError(ValueError):
    """A value-free failure at the deterministic benchmark boundary."""

    def __init__(self) -> None:
        super().__init__("scenario failed deterministic oracle validation")


class OracleResult(BaseModel):
    """A deterministic comparison between paired behavior and the stored label."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    schema_version: Literal["state-decay-oracle-result/v1"] = ORACLE_RESULT_SCHEMA_VERSION
    scenario_id: Sha256Digest
    expected_label: InterventionLabel
    observed_label: InterventionLabel
    matched: bool
    reminder_success: bool
    silence_success: bool
    decisive_action_id: ComponentIdentifier
    decisive_source_ids: tuple[ComponentIdentifier, ...]
    decisive_memory_ids: tuple[ComponentIdentifier, ...]
    reminder_evidence_supported: bool
    silence_evidence_supported: bool


def _fail() -> NoReturn:
    raise OracleEvaluationError() from None


def _validated_scenario(value: object) -> StateDecayScenario:
    if type(value) is not StateDecayScenario:
        _fail()
    scenario = value
    validated: StateDecayScenario | None = None
    valid = False
    try:
        validated = StateDecayScenario.model_validate_json(scenario.model_dump_json(warnings=False))
        valid = validated == scenario
    except Exception:
        pass
    if validated is None or not valid:
        _fail()
    return validated


def _evidence_is_supported(
    continuation: PairedContinuation,
    *,
    decisive_source_ids: frozenset[str],
    decisive_memory_ids: frozenset[str],
) -> bool:
    return decisive_source_ids.issubset(continuation.evidence_source_ids) and (
        decisive_memory_ids.issubset(continuation.evidence_memory_ids)
    )


def evaluate_scenario(scenario: StateDecayScenario) -> OracleResult:
    """Derive the intervention label from one validated paired continuation.

    The stored label is observational metadata. The expected label is reconstructed
    independently from branch actions and recorded task outcomes. Decisive evidence
    support is validated and reported as a separate axis.
    """

    validated = _validated_scenario(scenario)
    try:
        event_steps = {event.source_id: event.step for event in validated.trajectory_prefix}
        event_steps[validated.pivot.source_id] = validated.pivot.step
        memories = {memory.memory_id: memory for memory in validated.candidate_memories}
        allowed_action_ids = {action.action_id for action in validated.allowed_actions}
        required_action_id = validated.oracle.required_action_id
        decisive_source_ids = frozenset(validated.evidence_criteria.decisive_source_ids)
        decisive_memory_ids = frozenset(validated.evidence_criteria.decisive_memory_ids)

        if required_action_id not in allowed_action_ids:
            _fail()
        if not decisive_source_ids and not decisive_memory_ids:
            _fail()

        for source_id in decisive_source_ids:
            if event_steps.get(source_id, validated.pivot.step + 1) > validated.pivot.step:
                _fail()
        for memory_id in decisive_memory_ids:
            memory = memories.get(memory_id)
            if (
                memory is None
                or memory.recorded_step > validated.pivot.step
                or memory.validity is not ValidityState.ACTIVE
            ):
                _fail()

        branches: dict[ContinuationBranch, PairedContinuation] = {}
        branch_success: dict[ContinuationBranch, bool] = {}
        evidence_support: dict[ContinuationBranch, bool] = {}
        for continuation in validated.paired_continuations:
            if continuation.branch in branches:
                _fail()
            branches[continuation.branch] = continuation
            if continuation.selected_action_id not in allowed_action_ids:
                _fail()

            for source_id in continuation.evidence_source_ids:
                if event_steps.get(source_id, validated.pivot.step + 1) > validated.pivot.step:
                    _fail()
            for memory_id in continuation.evidence_memory_ids:
                memory = memories.get(memory_id)
                if (
                    memory is None
                    or memory.recorded_step > validated.pivot.step
                    or memory.validity is not ValidityState.ACTIVE
                ):
                    _fail()

            supported = _evidence_is_supported(
                continuation,
                decisive_source_ids=decisive_source_ids,
                decisive_memory_ids=decisive_memory_ids,
            )
            calculated_success = continuation.selected_action_id == required_action_id
            recorded_success = continuation.outcome is ContinuationOutcome.SUCCESS
            if calculated_success != recorded_success:
                _fail()
            branch_success[continuation.branch] = calculated_success
            evidence_support[continuation.branch] = supported

        expected_branches = {
            ContinuationBranch.REMINDER,
            ContinuationBranch.SILENCE,
        }
        if set(branches) != expected_branches:
            _fail()

        reminder_success = branch_success[ContinuationBranch.REMINDER]
        silence_success = branch_success[ContinuationBranch.SILENCE]
        if reminder_success == silence_success:
            _fail()
        expected_label = (
            InterventionLabel.INTERVENE if reminder_success else InterventionLabel.SILENCE
        )
        observed_label = validated.label
        return OracleResult(
            scenario_id=validated.scenario_id,
            expected_label=expected_label,
            observed_label=observed_label,
            matched=expected_label is observed_label,
            reminder_success=reminder_success,
            silence_success=silence_success,
            decisive_action_id=required_action_id,
            decisive_source_ids=tuple(sorted(decisive_source_ids)),
            decisive_memory_ids=tuple(sorted(decisive_memory_ids)),
            reminder_evidence_supported=evidence_support[ContinuationBranch.REMINDER],
            silence_evidence_supported=evidence_support[ContinuationBranch.SILENCE],
        )
    except OracleEvaluationError:
        raise
    except Exception:
        pass
    _fail()


__all__ = [
    "ORACLE_RESULT_SCHEMA_VERSION",
    "OracleEvaluationError",
    "OracleResult",
    "evaluate_scenario",
]
