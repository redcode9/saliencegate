from __future__ import annotations

import asyncio
import hashlib

import pytest
from pydantic import ValidationError
from tests.memory.test_two_phase_executor import (
    CREATED_KNOWLEDGE_ID,
    UNUSED_ASSIGNMENT_ID,
    _operations,
    _prepare,
    _repository_state,
    _running_case,
    _selection,
)
from tests.memory.test_two_phase_failures import (
    _CancellingMaterializer,
    _harness,
    _rejected,
    _repair_policy,
    _valid,
)

from saliencegate.memory.two_phase import TwoPhaseExecutionCancelled
from saliencegate.ports.model_calls import (
    StructuredCallParseStatus,
    StructuredCallPhase,
)
from saliencegate.ports.two_phase import (
    PhaseOneCycleExecutor,
    PhaseOneCycleResult,
    TwoPhaseCycleFailure,
    TwoPhaseCycleResult,
    TwoPhaseFailureReason,
)

_CREATE_RESULT_DIGEST = "31a313667f366032726a7532784ffd4b902732ae5f1124f1e8582e2c4bfebe98"
_CREATE_SERIALIZED_SHA256 = "08de20f1a767bad46350414f342fc6b802a8566e0604c197cd27f6557b401a32"


@pytest.mark.asyncio
async def test_phase_one_success_stops_before_intervention_and_preserves_repository() -> None:
    case = await _running_case()
    prepared = await _prepare(
        case,
        operations=_operations(case, "create"),
        selection=_selection(remind_created_memory=True),
        assignment_pool=(CREATED_KNOWLEDGE_ID, UNUSED_ASSIGNMENT_ID),
    )
    before = await _repository_state(case.repository)

    outcome = await prepared.executor.execute_phase_one(prepared.request)

    assert type(outcome) is PhaseOneCycleResult
    assert outcome.outcome == "phase_one_completed"
    assert outcome.request_digest == prepared.request.request_digest
    assert outcome.run_id == prepared.request.cycle_receipt.cycle.run_id
    assert outcome.cycle_id == prepared.request.cycle_receipt.cycle.cycle_id
    assert outcome.window_digest == prepared.request.window.window_digest
    assert outcome.current_bank_view_digest == prepared.request.current_bank.view_digest
    assert (
        outcome.current_bank_source_projection_digest
        == prepared.request.current_bank.source_projection_digest
    )
    assert outcome.materialization == prepared.expected_materialization
    assert outcome.memory_edit_output == _operations(case, "create")
    assert outcome.validated_delta == prepared.expected_materialization.delta
    assert outcome.memory_id_assignments == prepared.expected_materialization.memory_id_assignments
    assert outcome.call_receipts[0].call_digest == prepared.expected_results[0].call_digest
    assert outcome.usage.model_calls == 1
    assert outcome.usage.provider_input_tokens == 21
    assert outcome.usage.provider_output_tokens == 5
    assert outcome.usage.latency_us == 101
    assert prepared.client.requests == [prepared.expected_calls[0]]
    assert prepared.client.replay.remaining_responses == 1
    assert len(prepared.materializer.requests) == 1
    assert await _repository_state(case.repository) == before
    assert isinstance(prepared.executor, PhaseOneCycleExecutor)
    assert not isinstance(object(), PhaseOneCycleExecutor)


@pytest.mark.asyncio
async def test_phase_one_and_full_execution_share_the_exact_phase_one_projection() -> None:
    phase_case = await _running_case()
    phase_prepared = await _prepare(
        phase_case,
        operations=_operations(phase_case, "create"),
        selection=_selection(remind_created_memory=True),
        assignment_pool=(CREATED_KNOWLEDGE_ID, UNUSED_ASSIGNMENT_ID),
    )
    full_case = await _running_case()
    full_prepared = await _prepare(
        full_case,
        operations=_operations(full_case, "create"),
        selection=_selection(remind_created_memory=True),
        assignment_pool=(CREATED_KNOWLEDGE_ID, UNUSED_ASSIGNMENT_ID),
    )

    phase_outcome = await phase_prepared.executor.execute_phase_one(phase_prepared.request)
    full_outcome = await full_prepared.executor.execute(full_prepared.request)

    assert type(phase_outcome) is PhaseOneCycleResult
    assert type(full_outcome) is TwoPhaseCycleResult
    assert phase_outcome.request_digest == full_outcome.request_digest
    assert phase_outcome.materialization == full_outcome.materialization
    assert phase_outcome.memory_edit_output == full_outcome.memory_edit_output
    assert phase_outcome.candidate_bank_view_digest == full_outcome.candidate_bank_view_digest
    assert phase_outcome.call_receipts == full_outcome.call_receipts[:1]
    assert phase_outcome.usage.model_calls == 1
    assert full_outcome.result_digest == _CREATE_RESULT_DIGEST
    serialized = full_outcome.model_dump_json(warnings=False).encode()
    assert hashlib.sha256(serialized).hexdigest() == _CREATE_SERIALIZED_SHA256


@pytest.mark.asyncio
async def test_phase_one_result_rejects_tampered_content_and_cross_component_links() -> None:
    case = await _running_case()
    prepared = await _prepare(
        case,
        operations=_operations(case, "create"),
        selection=_selection(remind_created_memory=True),
        assignment_pool=(CREATED_KNOWLEDGE_ID,),
    )
    outcome = await prepared.executor.execute_phase_one(prepared.request)
    assert type(outcome) is PhaseOneCycleResult

    with pytest.raises(ValidationError, match="result digest does not match"):
        PhaseOneCycleResult.model_validate(
            outcome.model_dump(mode="python") | {"request_digest": "0" * 64}
        )

    values = outcome.model_dump(mode="python", exclude={"result_digest"})
    for change in (
        {"current_bank_view_digest": "0" * 64},
        {"candidate_bank_view_digest": "0" * 64},
        {
            "current_bank_source_projection_digest": (
                outcome.materialization.preview_projection_digest
            )
        },
        {"memory_edit_output": _operations(case, "noop")},
        {"model_profile_digest": "0" * 64},
        {"call_policy_digest": "0" * 64},
        {"usage": outcome.usage.model_copy(update={"latency_us": 0})},
    ):
        with pytest.raises(ValidationError, match="components do not match"):
            PhaseOneCycleResult.model_validate(values | change)


@pytest.mark.asyncio
async def test_phase_one_repair_returns_only_memory_edit_receipts() -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (
            _rejected(StructuredCallParseStatus.SCHEMA_INVALID),
            _valid(_operations(case, "noop")),
        ),
        policy=_repair_policy(),
    )

    outcome = await harness.executor.execute_phase_one(harness.request)

    assert type(outcome) is PhaseOneCycleResult
    assert tuple(item.model_call_index for item in outcome.call_receipts) == (0, 1)
    assert tuple(item.phase for item in outcome.call_receipts) == (
        StructuredCallPhase.MEMORY_EDIT,
        StructuredCallPhase.MEMORY_EDIT,
    )
    assert tuple(item.attempt for item in outcome.call_receipts) == (0, 1)
    assert outcome.usage.model_calls == 2
    assert outcome.usage.schema_repairs == 1
    assert len(harness.client.requests) == 2
    assert len(harness.repository_materializer.requests) == 1


@pytest.mark.asyncio
async def test_phase_one_failure_keeps_the_existing_typed_failure_contract() -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (_rejected(StructuredCallParseStatus.SCHEMA_INVALID),),
    )

    outcome = await harness.executor.execute_phase_one(harness.request)

    assert type(outcome) is TwoPhaseCycleFailure
    assert outcome.failed_phase is StructuredCallPhase.MEMORY_EDIT
    assert outcome.reason is TwoPhaseFailureReason.SCHEMA_INVALID
    assert outcome.usage.model_calls == 1
    assert len(harness.client.requests) == 1
    assert not harness.repository_materializer.requests


@pytest.mark.asyncio
async def test_phase_one_client_cancellation_has_unknown_cost() -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (asyncio.CancelledError("secret provider cancellation"),),
        policy=_repair_policy(),
    )

    with pytest.raises(TwoPhaseExecutionCancelled) as raised:
        await harness.executor.execute_phase_one(harness.request)

    assert str(raised.value) == ""
    assert raised.value.cost_certainty == "unknown"
    assert not raised.value.call_receipts
    assert raised.value.usage.model_calls == 0
    assert len(harness.client.requests) == 1
    assert not harness.repository_materializer.requests


@pytest.mark.asyncio
async def test_phase_one_materializer_cancellation_preserves_known_call_cost() -> None:
    case = await _running_case()
    materializer = _CancellingMaterializer()
    harness = _harness(
        case,
        (_valid(_operations(case, "noop")),),
        materializer=materializer,
    )

    with pytest.raises(TwoPhaseExecutionCancelled) as raised:
        await harness.executor.execute_phase_one(harness.request)

    assert str(raised.value) == ""
    assert raised.value.cost_certainty == "known"
    assert len(raised.value.call_receipts) == 1
    assert raised.value.call_receipts[0].phase is StructuredCallPhase.MEMORY_EDIT
    assert raised.value.usage.model_calls == 1
    assert len(harness.client.requests) == 1
    assert len(materializer.requests) == 1
    assert not harness.repository_materializer.requests
