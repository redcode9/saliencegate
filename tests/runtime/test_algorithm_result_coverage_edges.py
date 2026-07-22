"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from tests.runtime.test_fixed_step_runtime import (
    _DeliveryAdapter,
    _make_repository,
    _multi_step_events,
    _run,
    _run_start,
)

import saliencegate.ports.two_phase as two_phase_module
import saliencegate.runtime.algorithm_result as algorithm_result_module
from saliencegate.domain import (
    CycleRecord,
    CycleState,
    InterventionAction,
    ReasonCode,
    canonical_json,
)
from saliencegate.intervention import GroundingConfig, GroundingReceipt
from saliencegate.ports.model_calls import CanonicalUsageProvenance, StructuredCallPhase
from saliencegate.ports.two_phase import CallReceipt, TwoPhaseCycleFailure, TwoPhaseFailureReason
from saliencegate.runtime.algorithm_result import (
    AlgorithmRunResult,
    algorithm_result_digest,
    model_token_usage_attestation,
)
from saliencegate.runtime.model_token_counting import DeterministicModelTokenCounter


@pytest.fixture(scope="module")
def algorithm_result() -> AlgorithmRunResult:
    repository = _make_repository("memory", Path("unused-algorithm-result.sqlite3"))
    result, _client = asyncio.run(
        _run(
            repository,
            _multi_step_events(),
            mode="silence",
            cycle_capacity=2,
        )
    )
    return result


@pytest.fixture(scope="module")
def reminder_result() -> AlgorithmRunResult:
    from saliencegate.domain import DeliveryTarget

    events = list(_multi_step_events())
    events[0] = _run_start(target_request_id="algorithm-reminder-request-1")
    events[-1] = events[-1].model_copy(update={"target_request_id": "algorithm-reminder-request-2"})
    repository = _make_repository("memory", Path("unused-algorithm-reminder.sqlite3"))
    result, _client = asyncio.run(
        _run(
            repository,
            tuple(events),
            mode="reminder",
            cycle_capacity=2,
            delivery_adapter=_DeliveryAdapter("deliver"),
            requested_delivery_target=DeliveryTarget.NEXT_MODEL_CALL,
        )
    )
    assert len(result.cycles) == len(result.deliveries) == 2
    return result


def _recalculated(result: AlgorithmRunResult, **updates: object) -> AlgorithmRunResult:
    changed = result.model_copy(update=updates)
    values = changed.model_dump(mode="json", exclude={"result_digest"}, warnings=False)
    return changed.model_copy(update={"result_digest": algorithm_result_digest(values)})


def _explode(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("forced reconstruction failure")


def _failure_from_result(result: AlgorithmRunResult) -> TwoPhaseCycleFailure:
    execution = result.executions[0]
    request = result.cycle_requests[0]
    return TwoPhaseCycleFailure.model_construct(
        outcome="failed",
        schema_version="two-phase-cycle-failure/v1",
        request_digest=execution.request_digest,
        run_id=execution.run_id,
        cycle_id=execution.cycle_id,
        window_digest=execution.window_digest,
        prompt_bundle_digest=execution.prompt_bundle_digest,
        model_id=execution.model_id,
        model_profile_digest=execution.model_profile_digest,
        call_policy_digest=execution.call_policy_digest,
        call_policy=execution.call_policy,
        failed_phase=StructuredCallPhase.INTERVENTION,
        reason=TwoPhaseFailureReason.MODEL_ERROR,
        assigned_memory_id_capacity=len(request.assigned_memory_ids),
        memory_edit_output=execution.memory_edit_output,
        intervention_output=execution.intervention_output,
        materialization_failure_reason=None,
        call_receipts=execution.call_receipts,
        usage=execution.usage,
        cost_certainty="known",
        failure_digest="0" * 64,
    )


def test_token_usage_wrapper_rejects_a_replay_counter_without_configuration(
    algorithm_result: AlgorithmRunResult,
) -> None:
    assert (
        model_token_usage_attestation(
            algorithm_result.configuration,
            algorithm_result.call_receipts,
        )
        == algorithm_result.model_token_usage
    )

    identity = DeterministicModelTokenCounter(
        model_id=algorithm_result.configuration.model_profile.model_id,
        input_token_count=1,
        output_token_count=1,
    ).identity
    usage = algorithm_result.call_receipts[0].usage.model_copy(
        update={
            "canonical_input_tokens": 1,
            "canonical_output_tokens": 1,
            "canonical_usage_provenance": CanonicalUsageProvenance.REPLAY_ATTESTED,
            "local_counter_id": identity.counter_id,
            "local_counter_version": identity.counter_version,
            "local_counter_configuration_digest": identity.configuration_digest,
            "local_counter_model_id": identity.model_id,
        }
    )
    receipt = algorithm_result.call_receipts[0].model_copy(update={"usage": usage})

    with pytest.raises(ValueError, match="counter differs"):
        model_token_usage_attestation(algorithm_result.configuration, (receipt,))


def test_result_rejects_recalculated_token_usage_mismatch(
    algorithm_result: AlgorithmRunResult,
) -> None:
    usage = algorithm_result.model_token_usage.model_copy(update={"provider_input_tokens": 999})
    changed = _recalculated(algorithm_result, model_token_usage=usage)

    with pytest.raises(ValueError, match="model token usage"):
        changed.result_attests_one_complete_execution()


@pytest.mark.parametrize(
    ("attribute", "message"),
    (
        ("TrajectoryPrefixRequest", "trajectory request"),
        ("_project_verified_fixed_step_schedule", "schedule failed"),
        ("_project_verified_message_window", "window failed"),
    ),
)
def test_trace_validator_wraps_projection_reconstruction_failures(
    algorithm_result: AlgorithmRunResult,
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    message: str,
) -> None:
    monkeypatch.setattr(algorithm_result_module, attribute, _explode)

    with pytest.raises(ValueError, match=message):
        algorithm_result._validate_trace_schedule_and_windows()


def test_trace_validator_rejects_causal_and_schedule_forgery(
    algorithm_result: AlgorithmRunResult,
) -> None:
    items = algorithm_result.trajectory_prefix.items
    first_event = items[0].event.model_copy(update={"parent_ids": (items[1].event.event_id,)})
    first_item = items[0].model_copy(update={"event": first_event})
    forged_items = (first_item, *items[1:])
    prefix = algorithm_result.trajectory_prefix.model_copy(update={"items": forged_items})
    persisted = tuple(
        algorithm_result_module._persisted_event_draft_digest(item.event) for item in forged_items
    )
    causal = algorithm_result.model_copy(
        update={
            "trajectory_prefix": prefix,
            "persisted_event_draft_digests": persisted,
        }
    )

    with pytest.raises(ValueError, match="parent graph"):
        causal._validate_trace_schedule_and_windows()

    schedule = algorithm_result.schedule.model_copy(update={"schedule_digest": "0" * 64})
    with pytest.raises(ValueError, match="schedule does not match"):
        algorithm_result.model_copy(
            update={"schedule": schedule}
        )._validate_trace_schedule_and_windows()


@pytest.mark.parametrize(
    ("owner", "attribute", "message"),
    (
        (GroundingConfig, "model_validate_json", "grounding history configuration"),
        (CycleRecord, "model_validate", "running cycle"),
    ),
)
def test_cycle_validator_wraps_configuration_and_cycle_reconstruction(
    algorithm_result: AlgorithmRunResult,
    monkeypatch: pytest.MonkeyPatch,
    owner: type[Any],
    attribute: str,
    message: str,
) -> None:
    monkeypatch.setattr(owner, attribute, staticmethod(_explode))

    with pytest.raises(ValueError, match=message):
        algorithm_result._validate_decisions_and_cycles()


def test_cycle_validator_wraps_reminder_history_reconstruction(
    reminder_result: AlgorithmRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert all(
        cycle.intervention is not None and cycle.intervention.action is InterventionAction.REMIND
        for cycle in reminder_result.cycles
    )
    grounding = GroundingConfig.model_validate_json(
        canonical_json(reminder_result.configuration.grounding_configuration.configuration)
    )
    wider_history = grounding.model_copy(update={"duplicate_window_events": 10})
    monkeypatch.setattr(
        GroundingConfig,
        "model_validate_json",
        classmethod(lambda _cls, *_args, **_kwargs: wider_history),
    )
    monkeypatch.setattr(algorithm_result_module, "ReminderHistory", _explode)

    with pytest.raises(ValueError, match="reminder history"):
        reminder_result._validate_decisions_and_cycles()


def test_call_validator_rejects_duplicate_cycles_and_request_drift(
    algorithm_result: AlgorithmRunResult,
) -> None:
    duplicate = algorithm_result.model_copy(
        update={"cycles": (algorithm_result.cycles[0], algorithm_result.cycles[0])}
    )
    with pytest.raises(ValueError, match="cycles are not unique"):
        duplicate._validate_calls_deliveries_and_outcomes()

    request = algorithm_result.cycle_requests[0].model_copy(
        update={"delta_id": UUID("00000000-0000-4000-8000-00000000b101")}
    )
    drifted = algorithm_result.model_copy(
        update={"cycle_requests": (request, *algorithm_result.cycle_requests[1:])}
    )
    with pytest.raises(ValueError, match="request differs"):
        drifted._validate_calls_deliveries_and_outcomes()


def test_call_validator_rejects_nonbaseline_committed_call_shape(
    algorithm_result: AlgorithmRunResult,
) -> None:
    cycle = algorithm_result.cycles[0]
    execution = algorithm_result.executions[0]
    receipt = algorithm_result.call_receipts[0]
    shortened_cycle = cycle.model_copy(
        update={
            "model_call_digests": (receipt.call_digest,),
            "model_call_latencies_us": (receipt.usage.latency_us,),
        }
    )
    shortened_execution = execution.model_copy(update={"call_receipts": (receipt,)})
    changed = algorithm_result.model_copy(
        update={
            "cycles": (shortened_cycle, *algorithm_result.cycles[1:]),
            "executions": (shortened_execution, *algorithm_result.executions[1:]),
            "call_receipts": (receipt, *algorithm_result.call_receipts[2:]),
        }
    )

    with pytest.raises(ValueError, match="exactly two phase calls"):
        changed._validate_calls_deliveries_and_outcomes()


def test_call_validator_rejects_missing_or_drifted_committed_outputs(
    algorithm_result: AlgorithmRunResult,
) -> None:
    missing = algorithm_result.cycles[0].model_copy(update={"intervention": None})
    with pytest.raises(ValueError, match="lacks its intervention"):
        algorithm_result.model_copy(
            update={"cycles": (missing, *algorithm_result.cycles[1:])}
        )._validate_calls_deliveries_and_outcomes()

    delta = algorithm_result.cycles[0].validated_delta
    assert delta is not None
    drifted_delta = delta.model_copy(
        update={"delta_id": UUID("00000000-0000-4000-8000-00000000b102")}
    )
    drifted = algorithm_result.cycles[0].model_copy(update={"validated_delta": drifted_delta})
    with pytest.raises(ValueError, match="outputs differ"):
        algorithm_result.model_copy(
            update={"cycles": (drifted, *algorithm_result.cycles[1:])}
        )._validate_calls_deliveries_and_outcomes()


def test_call_validator_wraps_and_checks_grounding_receipt(
    algorithm_result: AlgorithmRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(GroundingReceipt, "model_validate_json", staticmethod(_explode))
    with pytest.raises(ValueError, match="grounding receipt failed"):
        algorithm_result._validate_calls_deliveries_and_outcomes()

    monkeypatch.undo()
    cycle = algorithm_result.cycles[0]
    assert cycle.intervention is not None
    receipt = GroundingReceipt.model_validate_json(
        canonical_json(cycle.intervention.grounding_receipt)
    )
    wrong = receipt.model_copy(update={"model_call_index": receipt.model_call_index + 1})
    monkeypatch.setattr(
        GroundingReceipt,
        "model_validate_json",
        classmethod(lambda _cls, *_args, **_kwargs: wrong),
    )
    with pytest.raises(ValueError, match="does not name its final call"):
        algorithm_result._validate_calls_deliveries_and_outcomes()


def test_call_validator_distinguishes_failure_attestation_cases(
    algorithm_result: AlgorithmRunResult,
) -> None:
    failure = _failure_from_result(algorithm_result)
    committed = algorithm_result.model_copy(
        update={"executions": (failure, *algorithm_result.executions[1:])}
    )
    with pytest.raises(ValueError, match="committed cycle lacks"):
        committed._validate_calls_deliveries_and_outcomes()

    failed_cycle = algorithm_result.cycles[0].model_copy(
        update={"state": CycleState.FAILED, "failure_reason": ReasonCode.MODEL_TIMEOUT}
    )
    wrong_failure = algorithm_result.model_copy(
        update={
            "cycles": (failed_cycle, *algorithm_result.cycles[1:]),
            "executions": (failure, *algorithm_result.executions[1:]),
        }
    )
    with pytest.raises(ValueError, match="failed cycle reason"):
        wrong_failure._validate_calls_deliveries_and_outcomes()

    unexplained_cycle = algorithm_result.cycles[0].model_copy(
        update={"state": CycleState.FAILED, "failure_reason": ReasonCode.MODEL_ERROR}
    )
    unexplained = algorithm_result.model_copy(
        update={"cycles": (unexplained_cycle, *algorithm_result.cycles[1:])}
    )
    with pytest.raises(ValueError, match="cannot explain"):
        unexplained._validate_calls_deliveries_and_outcomes()


def test_delivery_validator_rejects_duplicate_identities(
    reminder_result: AlgorithmRunResult,
) -> None:
    duplicate = reminder_result.deliveries[1].model_copy(
        update={"delivery_id": reminder_result.deliveries[0].delivery_id}
    )
    changed = reminder_result.model_copy(
        update={"deliveries": (reminder_result.deliveries[0], duplicate)}
    )

    with pytest.raises(ValueError, match="delivery identities"):
        changed._validate_calls_deliveries_and_outcomes()


def test_call_validator_rejects_duplicate_call_identities(
    algorithm_result: AlgorithmRunResult,
) -> None:
    original = algorithm_result.call_receipts[2]
    values = original.model_dump(mode="json", exclude={"receipt_digest"}, warnings=False)
    values["call_digest"] = algorithm_result.call_receipts[0].call_digest
    values["receipt_digest"] = two_phase_module._call_receipt_digest(values)
    duplicate = CallReceipt.model_validate_json(canonical_json(values))

    second_cycle = algorithm_result.cycles[1]
    call_digests = (duplicate.call_digest, second_cycle.model_call_digests[1])
    changed_cycle = second_cycle.model_copy(update={"model_call_digests": call_digests})
    second_execution = algorithm_result.executions[1]
    changed_execution = second_execution.model_copy(
        update={"call_receipts": (duplicate, second_execution.call_receipts[1])}
    )
    changed = algorithm_result.model_copy(
        update={
            "cycles": (algorithm_result.cycles[0], changed_cycle),
            "executions": (algorithm_result.executions[0], changed_execution),
            "call_receipts": (
                *algorithm_result.call_receipts[:2],
                duplicate,
                algorithm_result.call_receipts[3],
            ),
        }
    )

    with pytest.raises(ValueError, match="call identities"):
        changed._validate_calls_deliveries_and_outcomes()


def test_unknown_cost_cycle_skips_exact_settlement_reconciliation(
    algorithm_result: AlgorithmRunResult,
) -> None:
    unknown = algorithm_result.cycles[0].model_copy(
        update={"failure_reason": ReasonCode.FAILED_UNKNOWN_COST}
    )

    algorithm_result._validate_cycle_settlement(unknown, algorithm_result.call_receipts)
