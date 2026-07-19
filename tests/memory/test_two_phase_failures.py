from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TypeAlias

import pytest
from pydantic import ValidationError
from tests.memory.test_two_phase_executor import (
    CREATED_KNOWLEDGE_ID,
    UNUSED_ASSIGNMENT_ID,
    _call_policy,
    _Case,
    _cycle_request,
    _invalid_candidate_selection,
    _model_profile,
    _operations,
    _RepositoryMaterializer,
    _running_case,
    _selection,
)
from tests.repository.conformance import cycle_grounding_config

from saliencegate.domain import (
    InterventionAction,
    InterventionDecision,
    PayloadDigest,
    PayloadDigestAlgorithm,
    ReasonCode,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.intervention import GroundingPipeline, ProposalParseStatus
from saliencegate.memory.materialize import (
    MaterializationFailureReason,
    MaterializedBankOperations,
    OperationMaterializationRequest,
)
from saliencegate.memory.two_phase import (
    PaperTwoPhaseCycleExecutor,
    TwoPhaseExecutionCancelled,
    TwoPhaseExecutionError,
)
from saliencegate.ports.model_calls import (
    ProviderUsageProvenance,
    StructuredCallParseStatus,
    StructuredCallPhase,
    StructuredCallRequest,
    StructuredCallResult,
    StructuredCallStatus,
    StructuredCallUsage,
    StructuredPhaseOutput,
)
from saliencegate.ports.two_phase import (
    CallReceipt,
    OperationMaterializer,
    TwoPhaseCallPolicy,
    TwoPhaseCycleFailure,
    TwoPhaseCycleRequest,
    TwoPhaseCycleResult,
    TwoPhaseFailureReason,
    TwoPhaseUsage,
)
from saliencegate.prompts import PAPER_TWO_PHASE_V1

_COMPLETION_DIGEST_DOMAIN = "saliencegate:test:two-phase-failure-completion:v1"


@dataclass(frozen=True, slots=True)
class _ResponseSpec:
    status: StructuredCallStatus
    parse_status: StructuredCallParseStatus
    output: StructuredPhaseOutput | None = None
    provider_input_tokens: int | None = 13
    provider_output_tokens: int | None = 5
    latency_us: int = 101


_ScriptItem: TypeAlias = _ResponseSpec | BaseException


@dataclass(slots=True)
class _ScriptedClient:
    script: tuple[_ScriptItem, ...]
    requests: list[StructuredCallRequest] = field(default_factory=list)

    async def generate(self, request: StructuredCallRequest) -> StructuredCallResult:
        ordinal = len(self.requests)
        self.requests.append(request)
        item = self.script[ordinal]
        if isinstance(item, BaseException):
            raise item
        return _result_for_request(request, item)


@dataclass(slots=True)
class _ExplodingMaterializer:
    requests: list[OperationMaterializationRequest] = field(default_factory=list)

    async def materialize(
        self,
        request: OperationMaterializationRequest,
    ) -> MaterializedBankOperations:
        self.requests.append(request)
        raise RuntimeError("secret materializer payload")


@dataclass(slots=True)
class _CancellingMaterializer:
    requests: list[OperationMaterializationRequest] = field(default_factory=list)

    async def materialize(
        self,
        request: OperationMaterializationRequest,
    ) -> MaterializedBankOperations:
        self.requests.append(request)
        raise asyncio.CancelledError("secret materializer cancellation")


@dataclass(frozen=True, slots=True)
class _Harness:
    executor: PaperTwoPhaseCycleExecutor
    request: TwoPhaseCycleRequest
    client: _ScriptedClient
    repository_materializer: _RepositoryMaterializer


def _valid(output: StructuredPhaseOutput, *, known_usage: bool = True) -> _ResponseSpec:
    return _ResponseSpec(
        status=StructuredCallStatus.COMPLETED,
        parse_status=StructuredCallParseStatus.VALID,
        output=output,
        provider_input_tokens=13 if known_usage else None,
        provider_output_tokens=5 if known_usage else None,
    )


def _rejected(parse_status: StructuredCallParseStatus) -> _ResponseSpec:
    return _ResponseSpec(
        status=StructuredCallStatus.COMPLETED,
        parse_status=parse_status,
        output=None,
    )


def _transport(status: StructuredCallStatus) -> _ResponseSpec:
    return _ResponseSpec(
        status=status,
        parse_status=StructuredCallParseStatus.NOT_ATTEMPTED,
        output=None,
    )


def _result_for_request(
    request: StructuredCallRequest,
    spec: _ResponseSpec,
) -> StructuredCallResult:
    has_completion = spec.status is StructuredCallStatus.COMPLETED
    completion = (
        canonical_json(spec.output)
        if spec.output is not None
        else canonical_json(
            {
                "phase": request.phase.value,
                "parse_status": spec.parse_status.value,
            }
        )
    )
    completion_bytes = completion
    completion_digest = (
        PayloadDigest(
            algorithm=PayloadDigestAlgorithm.SYNTHETIC_SHA256,
            value=length_prefixed_sha256(
                completion_bytes,
                domain=_COMPLETION_DIGEST_DOMAIN,
            ),
        )
        if has_completion
        else None
    )
    return StructuredCallResult(
        schema_version="structured-call-result/v1",
        request_digest=request.request_digest,
        model_call_index=request.model_call_index,
        phase=request.phase,
        attempt=request.attempt,
        response_schema_version=request.response_schema_version,
        status=spec.status,
        parse_status=spec.parse_status,
        output=spec.output,
        completion_digest=completion_digest,
        completion_byte_count=len(completion_bytes) if has_completion else None,
        usage=StructuredCallUsage(
            schema_version="structured-call-usage/v1",
            provider_input_tokens=spec.provider_input_tokens,
            provider_output_tokens=spec.provider_output_tokens,
            provider_usage_provenance=(
                ProviderUsageProvenance.REPLAY_ATTESTED
                if spec.provider_input_tokens is not None
                else ProviderUsageProvenance.UNAVAILABLE
            ),
            latency_us=spec.latency_us,
        ),
    )


def _repair_policy() -> TwoPhaseCallPolicy:
    values = _call_policy().model_dump(mode="python", exclude={"policy_digest"})
    values.update(max_model_calls=3, max_schema_repairs=1)
    return TwoPhaseCallPolicy(**values)


def _harness(
    case: _Case,
    script: tuple[_ScriptItem, ...],
    *,
    policy: TwoPhaseCallPolicy | None = None,
    materializer: OperationMaterializer | None = None,
) -> _Harness:
    client = _ScriptedClient(script)
    repository_materializer = _RepositoryMaterializer(case.repository)
    executor = PaperTwoPhaseCycleExecutor(
        materializer=repository_materializer if materializer is None else materializer,
        client=client,
        prompt_bundle=PAPER_TWO_PHASE_V1,
        grounding_pipeline=GroundingPipeline(cycle_grounding_config()),
        model_profile=_model_profile(),
        call_policy=_call_policy() if policy is None else policy,
    )
    return _Harness(
        executor=executor,
        request=_cycle_request(case, assigned_memory_ids=(UNUSED_ASSIGNMENT_ID,)),
        client=client,
        repository_materializer=repository_materializer,
    )


@pytest.mark.parametrize(
    ("status", "reason"),
    (
        (StructuredCallStatus.MODEL_ERROR, TwoPhaseFailureReason.MODEL_ERROR),
        (StructuredCallStatus.MODEL_TIMEOUT, TwoPhaseFailureReason.MODEL_TIMEOUT),
    ),
)
async def test_phase_one_transport_failure_stops_before_materialization(
    status: StructuredCallStatus,
    reason: TwoPhaseFailureReason,
) -> None:
    case = await _running_case()
    harness = _harness(case, (_transport(status),))

    outcome = await harness.executor.execute(harness.request)

    assert type(outcome) is TwoPhaseCycleFailure
    assert outcome.failed_phase is StructuredCallPhase.MEMORY_EDIT
    assert outcome.reason is reason
    assert outcome.cost_certainty == "known"
    assert outcome.usage.model_calls == 1
    assert len(harness.client.requests) == 1
    assert not harness.repository_materializer.requests


async def test_phase_one_schema_invalid_without_repairs_is_noncommittable() -> None:
    case = await _running_case()
    harness = _harness(case, (_rejected(StructuredCallParseStatus.SCHEMA_INVALID),))

    outcome = await harness.executor.execute(harness.request)

    assert type(outcome) is TwoPhaseCycleFailure
    assert outcome.failed_phase is StructuredCallPhase.MEMORY_EDIT
    assert outcome.reason is TwoPhaseFailureReason.SCHEMA_INVALID
    assert outcome.usage.model_calls == 1
    assert outcome.usage.schema_repairs == 0
    assert tuple(item.attempt for item in outcome.call_receipts) == (0,)
    assert len(harness.client.requests) == 1
    assert not harness.repository_materializer.requests


@pytest.mark.parametrize(
    ("status", "reason"),
    (
        (StructuredCallStatus.MODEL_ERROR, TwoPhaseFailureReason.MODEL_ERROR),
        (StructuredCallStatus.MODEL_TIMEOUT, TwoPhaseFailureReason.MODEL_TIMEOUT),
    ),
)
async def test_phase_two_transport_failure_preserves_both_receipts(
    status: StructuredCallStatus,
    reason: TwoPhaseFailureReason,
) -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (
            _valid(_operations(case, "noop")),
            _transport(status),
        ),
    )

    outcome = await harness.executor.execute(harness.request)

    assert type(outcome) is TwoPhaseCycleFailure
    assert outcome.failed_phase is StructuredCallPhase.INTERVENTION
    assert outcome.reason is reason
    assert outcome.usage.model_calls == 2
    assert tuple(item.model_call_index for item in outcome.call_receipts) == (0, 1)
    assert tuple(item.phase for item in outcome.call_receipts) == (
        StructuredCallPhase.MEMORY_EDIT,
        StructuredCallPhase.INTERVENTION,
    )
    assert len(harness.client.requests) == 2
    assert len(harness.repository_materializer.requests) == 1


@pytest.mark.parametrize(
    ("parse_status", "receipt_status", "reason"),
    (
        (
            StructuredCallParseStatus.SCHEMA_INVALID,
            ProposalParseStatus.SCHEMA_INVALID,
            ReasonCode.SCHEMA_INVALID,
        ),
        (
            StructuredCallParseStatus.EMPTY_REMINDER,
            ProposalParseStatus.EMPTY_REMINDER,
            ReasonCode.NO_GROUNDED_CLAIMS,
        ),
        (
            StructuredCallParseStatus.CLAIM_OVER_LIMIT,
            ProposalParseStatus.CLAIM_OVER_LIMIT,
            ReasonCode.CLAIM_OVER_LIMIT,
        ),
    ),
)
async def test_phase_two_parse_rejection_materializes_canonical_silence(
    parse_status: StructuredCallParseStatus,
    receipt_status: ProposalParseStatus,
    reason: ReasonCode,
) -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (
            _valid(_operations(case, "noop")),
            _rejected(parse_status),
        ),
    )

    outcome = await harness.executor.execute(harness.request)

    assert type(outcome) is TwoPhaseCycleResult
    assert outcome.intervention.action is InterventionAction.SILENCE
    assert outcome.intervention.reason_code is reason
    assert outcome.grounding_receipt.parse_status is receipt_status
    assert outcome.grounding_receipt.model_call_index == 1
    assert outcome.grounding_receipt.model_call_digest == outcome.call_receipts[-1].call_digest
    assert outcome.usage.model_calls == 2
    assert len(harness.repository_materializer.requests) == 1


async def test_phase_one_repair_is_a_visible_successful_call() -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (
            _rejected(StructuredCallParseStatus.SCHEMA_INVALID),
            _valid(_operations(case, "noop")),
            _valid(_selection(remind_created_memory=False)),
        ),
        policy=_repair_policy(),
    )

    outcome = await harness.executor.execute(harness.request)

    assert type(outcome) is TwoPhaseCycleResult
    assert tuple(item.model_call_index for item in outcome.call_receipts) == (0, 1, 2)
    assert tuple(item.phase for item in outcome.call_receipts) == (
        StructuredCallPhase.MEMORY_EDIT,
        StructuredCallPhase.MEMORY_EDIT,
        StructuredCallPhase.INTERVENTION,
    )
    assert tuple(item.attempt for item in outcome.call_receipts) == (0, 1, 0)
    assert outcome.usage.model_calls == 3
    assert outcome.usage.schema_repairs == 1
    assert len(harness.client.requests) == 3
    assert len(harness.repository_materializer.requests) == 1


async def test_phase_one_repair_exhaustion_stops_before_materialization() -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (
            _rejected(StructuredCallParseStatus.SCHEMA_INVALID),
            _rejected(StructuredCallParseStatus.SCHEMA_INVALID),
        ),
        policy=_repair_policy(),
    )

    outcome = await harness.executor.execute(harness.request)

    assert type(outcome) is TwoPhaseCycleFailure
    assert outcome.failed_phase is StructuredCallPhase.MEMORY_EDIT
    assert outcome.reason is TwoPhaseFailureReason.REPAIR_EXHAUSTED
    assert tuple(item.model_call_index for item in outcome.call_receipts) == (0, 1)
    assert tuple(item.attempt for item in outcome.call_receipts) == (0, 1)
    assert outcome.usage.schema_repairs == 1
    assert len(harness.client.requests) == 2
    assert not harness.repository_materializer.requests


async def test_repair_exhaustion_rejects_a_policy_with_remaining_allowance() -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (
            _rejected(StructuredCallParseStatus.SCHEMA_INVALID),
            _rejected(StructuredCallParseStatus.SCHEMA_INVALID),
        ),
        policy=_repair_policy(),
    )
    outcome = await harness.executor.execute(harness.request)
    assert type(outcome) is TwoPhaseCycleFailure
    policy_values = _call_policy().model_dump(mode="python", exclude={"policy_digest"})
    policy_values.update(max_model_calls=4, max_schema_repairs=2)
    unexhausted = TwoPhaseCallPolicy(**policy_values)
    values = outcome.model_dump(mode="python", exclude={"failure_digest"})
    values["call_policy"] = unexhausted
    values["call_policy_digest"] = unexhausted.policy_digest

    with pytest.raises(ValidationError, match="failure receipts do not match"):
        TwoPhaseCycleFailure.model_validate(values)


async def test_phase_two_repair_is_visible_and_binds_the_final_grounding_receipt() -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (
            _valid(_operations(case, "noop")),
            _rejected(StructuredCallParseStatus.EMPTY_REMINDER),
            _valid(_selection(remind_created_memory=False)),
        ),
        policy=_repair_policy(),
    )

    outcome = await harness.executor.execute(harness.request)

    assert type(outcome) is TwoPhaseCycleResult
    assert tuple(item.model_call_index for item in outcome.call_receipts) == (0, 1, 2)
    assert tuple(item.phase for item in outcome.call_receipts) == (
        StructuredCallPhase.MEMORY_EDIT,
        StructuredCallPhase.INTERVENTION,
        StructuredCallPhase.INTERVENTION,
    )
    assert tuple(item.attempt for item in outcome.call_receipts) == (0, 0, 1)
    assert outcome.grounding_receipt.model_call_index == 2
    assert outcome.grounding_receipt.model_call_digest == outcome.call_receipts[-1].call_digest
    assert outcome.usage.schema_repairs == 1
    assert len(harness.client.requests) == 3
    assert len(harness.repository_materializer.requests) == 1


async def test_terminal_phase_two_rejection_requires_exhausted_repair_allowance() -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (
            _valid(_operations(case, "noop")),
            _rejected(StructuredCallParseStatus.SCHEMA_INVALID),
        ),
    )
    outcome = await harness.executor.execute(harness.request)
    assert type(outcome) is TwoPhaseCycleResult
    unexhausted = _repair_policy()
    values = outcome.model_dump(mode="python", exclude={"result_digest"})
    values["call_policy"] = unexhausted
    values["call_policy_digest"] = unexhausted.policy_digest

    with pytest.raises(ValidationError, match="components do not match"):
        TwoPhaseCycleResult.model_validate(values)


async def test_partial_provider_usage_keeps_aggregate_totals_unknown() -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (
            _valid(_operations(case, "noop")),
            _valid(_selection(remind_created_memory=False), known_usage=False),
        ),
    )

    outcome = await harness.executor.execute(harness.request)

    assert type(outcome) is TwoPhaseCycleResult
    assert outcome.call_receipts[0].usage.provider_input_tokens == 13
    assert outcome.call_receipts[1].usage.provider_input_tokens is None
    assert outcome.usage.provider_input_tokens is None
    assert outcome.usage.provider_output_tokens is None
    assert outcome.usage.model_calls == 2
    assert outcome.usage.latency_us == 202


async def test_known_provider_lower_bound_cannot_hide_behind_an_unknown_call() -> None:
    case = await _running_case()
    values = _call_policy().model_dump(mode="python", exclude={"policy_digest"})
    values["max_provider_input_tokens"] = 10
    policy = TwoPhaseCallPolicy(**values)
    harness = _harness(
        case,
        (
            _ResponseSpec(
                status=StructuredCallStatus.COMPLETED,
                parse_status=StructuredCallParseStatus.VALID,
                output=_operations(case, "noop"),
                provider_input_tokens=None,
                provider_output_tokens=None,
            ),
            _ResponseSpec(
                status=StructuredCallStatus.COMPLETED,
                parse_status=StructuredCallParseStatus.VALID,
                output=_selection(remind_created_memory=False),
                provider_input_tokens=11,
                provider_output_tokens=1,
            ),
        ),
        policy=policy,
    )

    outcome = await harness.executor.execute(harness.request)

    assert type(outcome) is TwoPhaseCycleFailure
    assert outcome.reason is TwoPhaseFailureReason.CALL_POLICY_EXCEEDED
    assert outcome.failed_phase is StructuredCallPhase.INTERVENTION
    assert outcome.usage.provider_input_tokens is None
    assert tuple(item.usage.provider_input_tokens for item in outcome.call_receipts) == (None, 11)
    assert len(harness.client.requests) == 2


async def test_client_exception_is_sanitized_and_never_retried() -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (RuntimeError("secret provider payload"),),
        policy=_repair_policy(),
    )

    with pytest.raises(TwoPhaseExecutionError) as raised:
        await harness.executor.execute(harness.request)

    assert str(raised.value) == "two-phase cycle execution failed validation"
    assert "secret provider payload" not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert len(harness.client.requests) == 1
    assert not harness.repository_materializer.requests


async def test_client_cancellation_is_sanitized_and_never_retried() -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (asyncio.CancelledError("secret cancellation payload"),),
        policy=_repair_policy(),
    )

    with pytest.raises(TwoPhaseExecutionCancelled) as raised:
        await harness.executor.execute(harness.request)

    assert str(raised.value) == ""
    assert "secret cancellation payload" not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert raised.value.cost_certainty == "unknown"
    assert not raised.value.call_receipts
    assert raised.value.usage.model_calls == 0
    assert len(harness.client.requests) == 1
    assert not harness.repository_materializer.requests


async def test_phase_two_client_cancellation_preserves_prior_receipts_as_unknown_cost() -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (
            _valid(_operations(case, "noop")),
            asyncio.CancelledError("secret phase-two cancellation"),
        ),
    )

    with pytest.raises(TwoPhaseExecutionCancelled) as raised:
        await harness.executor.execute(harness.request)

    assert str(raised.value) == ""
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert raised.value.cost_certainty == "unknown"
    assert len(raised.value.call_receipts) == 1
    assert raised.value.call_receipts[0].phase is StructuredCallPhase.MEMORY_EDIT
    assert raised.value.usage.model_calls == 1
    assert len(harness.client.requests) == 2
    assert len(harness.repository_materializer.requests) == 1


async def test_materializer_cancellation_preserves_known_phase_one_accounting() -> None:
    case = await _running_case()
    materializer = _CancellingMaterializer()
    harness = _harness(
        case,
        (_valid(_operations(case, "noop")),),
        materializer=materializer,
    )

    with pytest.raises(TwoPhaseExecutionCancelled) as raised:
        await harness.executor.execute(harness.request)

    assert str(raised.value) == ""
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert raised.value.cost_certainty == "known"
    assert len(raised.value.call_receipts) == 1
    assert raised.value.call_receipts[0].phase is StructuredCallPhase.MEMORY_EDIT
    assert raised.value.usage.model_calls == 1
    assert len(harness.client.requests) == 1
    assert len(materializer.requests) == 1
    assert not harness.repository_materializer.requests


async def test_generic_materializer_exception_becomes_a_typed_known_cost_failure() -> None:
    case = await _running_case()
    materializer = _ExplodingMaterializer()
    harness = _harness(
        case,
        (_valid(_operations(case, "noop")),),
        materializer=materializer,
    )

    outcome = await harness.executor.execute(harness.request)

    assert type(outcome) is TwoPhaseCycleFailure
    assert outcome.failed_phase is StructuredCallPhase.MEMORY_EDIT
    assert outcome.reason is TwoPhaseFailureReason.MATERIALIZATION_REJECTED
    assert outcome.cost_certainty == "known"
    assert outcome.usage.model_calls == 1
    assert "secret materializer payload" not in repr(outcome)
    assert len(harness.client.requests) == 1
    assert len(materializer.requests) == 1
    assert not harness.repository_materializer.requests


@pytest.mark.parametrize(
    "forged_reason",
    (
        TwoPhaseFailureReason.INVALID_OPERATION,
        TwoPhaseFailureReason.OPERATION_OVERFLOW,
        TwoPhaseFailureReason.CALL_CONTRACT_INVALID,
    ),
)
async def test_materialization_failure_witness_rejects_an_incompatible_reason(
    forged_reason: TwoPhaseFailureReason,
) -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (_valid(_operations(case, "noop")),),
        materializer=_ExplodingMaterializer(),
    )
    outcome = await harness.executor.execute(harness.request)
    assert type(outcome) is TwoPhaseCycleFailure
    values = outcome.model_dump(mode="python", exclude={"failure_digest"})
    values["reason"] = forged_reason

    with pytest.raises(ValidationError, match="failure receipts do not match"):
        TwoPhaseCycleFailure.model_validate(values)


async def test_stale_delete_returns_invalid_operation_with_exact_detail() -> None:
    case = await _running_case()
    proposal = _operations(case, "delete")
    stale_delete = proposal.operations[0].model_copy(update={"expected_revision": 2})
    stale_proposal = proposal.model_copy(update={"operations": (stale_delete,)})
    harness = _harness(case, (_valid(stale_proposal),))

    outcome = await harness.executor.execute(harness.request)

    assert type(outcome) is TwoPhaseCycleFailure
    assert outcome.reason is TwoPhaseFailureReason.INVALID_OPERATION
    assert outcome.materialization_failure_reason is MaterializationFailureReason.REFERENCE_STALE
    assert outcome.memory_edit_output == stale_proposal
    assert outcome.assigned_memory_id_capacity == 1
    assert len(harness.client.requests) == 1
    assert len(harness.repository_materializer.requests) == 1


async def test_two_writes_with_one_assignment_fail_before_materialization() -> None:
    case = await _running_case()
    proposal = _operations(case, "create")
    second_write = proposal.operations[0].model_copy(
        update={"content": "Keep the second verified deployment requirement."}
    )
    overflow = proposal.model_copy(update={"operations": (proposal.operations[0], second_write)})
    harness = _harness(case, (_valid(overflow),))

    outcome = await harness.executor.execute(harness.request)

    assert type(outcome) is TwoPhaseCycleFailure
    assert outcome.reason is TwoPhaseFailureReason.OPERATION_OVERFLOW
    assert outcome.materialization_failure_reason is None
    assert outcome.memory_edit_output == overflow
    assert outcome.assigned_memory_id_capacity == 1
    assert len(harness.client.requests) == 1
    assert not harness.repository_materializer.requests


def _resigned_receipt(receipt: CallReceipt, **changes: object) -> CallReceipt:
    values = receipt.model_dump(mode="python", exclude={"receipt_digest"})
    values.update(changes)
    return CallReceipt.model_validate(values)


def _resigned_failure_values(
    failure: TwoPhaseCycleFailure,
    *,
    receipts: tuple[CallReceipt, ...] | None = None,
    **changes: object,
) -> dict[str, object]:
    values = failure.model_dump(mode="python", exclude={"failure_digest"})
    if receipts is not None:
        values["call_receipts"] = receipts
        values["usage"] = TwoPhaseUsage.from_receipts(receipts)
    values.update(changes)
    return values


def _intervention_with_reason(
    intervention: InterventionDecision,
    reason: ReasonCode,
) -> InterventionDecision:
    values = intervention.model_dump(mode="python")
    values["reason_code"] = reason
    return InterventionDecision.model_validate(values)


async def _paired_reminder_and_rejection_results() -> tuple[
    TwoPhaseCycleResult,
    TwoPhaseCycleResult,
]:
    case = await _running_case()
    request = _cycle_request(case, assigned_memory_ids=(CREATED_KNOWLEDGE_ID,))
    reminder_harness = _harness(
        case,
        (
            _valid(_operations(case, "create")),
            _valid(_selection(remind_created_memory=True)),
        ),
    )
    rejection_harness = _harness(
        case,
        (
            _valid(_operations(case, "create")),
            _rejected(StructuredCallParseStatus.SCHEMA_INVALID),
        ),
    )

    reminder = await reminder_harness.executor.execute(request)
    rejection = await rejection_harness.executor.execute(request)

    assert type(reminder) is TwoPhaseCycleResult
    assert type(rejection) is TwoPhaseCycleResult
    assert reminder.materialization == rejection.materialization
    assert reminder.intervention.action is InterventionAction.REMIND
    assert rejection.intervention.action is InterventionAction.SILENCE
    return reminder, rejection


async def test_failure_validator_binds_reason_phase_and_terminal_receipt() -> None:
    case = await _running_case()
    harness = _harness(case, (_transport(StructuredCallStatus.MODEL_ERROR),))
    outcome = await harness.executor.execute(harness.request)
    assert type(outcome) is TwoPhaseCycleFailure

    for change in (
        {"reason": TwoPhaseFailureReason.MODEL_TIMEOUT},
        {"reason": TwoPhaseFailureReason.INVALID_OPERATION},
        {"failed_phase": StructuredCallPhase.INTERVENTION},
    ):
        with pytest.raises(ValidationError, match="failure receipts do not match"):
            TwoPhaseCycleFailure.model_validate(_resigned_failure_values(outcome, **change))


async def test_failure_validator_rejects_a_receipt_after_terminal_transport_outcome() -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (
            _rejected(StructuredCallParseStatus.SCHEMA_INVALID),
            _rejected(StructuredCallParseStatus.SCHEMA_INVALID),
        ),
        policy=_repair_policy(),
    )
    outcome = await harness.executor.execute(harness.request)
    assert type(outcome) is TwoPhaseCycleFailure

    terminal_transport = _resigned_receipt(
        outcome.call_receipts[0],
        status=StructuredCallStatus.MODEL_ERROR,
        parse_status=StructuredCallParseStatus.NOT_ATTEMPTED,
        completion_digest=None,
        completion_byte_count=None,
    )
    forged = (terminal_transport, outcome.call_receipts[1])

    with pytest.raises(ValidationError, match="failure receipts do not match"):
        TwoPhaseCycleFailure.model_validate(_resigned_failure_values(outcome, receipts=forged))


@pytest.mark.parametrize(
    "parse_status",
    (
        StructuredCallParseStatus.EMPTY_REMINDER,
        StructuredCallParseStatus.CLAIM_OVER_LIMIT,
    ),
)
async def test_failure_validator_rejects_intervention_only_parse_status_in_memory_phase(
    parse_status: StructuredCallParseStatus,
) -> None:
    case = await _running_case()
    harness = _harness(case, (_rejected(StructuredCallParseStatus.SCHEMA_INVALID),))
    outcome = await harness.executor.execute(harness.request)
    assert type(outcome) is TwoPhaseCycleFailure

    forged_receipt = _resigned_receipt(outcome.call_receipts[0], parse_status=parse_status)

    with pytest.raises(ValidationError, match="failure receipts do not match"):
        TwoPhaseCycleFailure.model_validate(
            _resigned_failure_values(outcome, receipts=(forged_receipt,))
        )


@pytest.mark.parametrize("digest_field", ("request_digest", "call_digest"))
async def test_failure_validator_requires_fresh_repair_digests(digest_field: str) -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (
            _rejected(StructuredCallParseStatus.SCHEMA_INVALID),
            _rejected(StructuredCallParseStatus.SCHEMA_INVALID),
        ),
        policy=_repair_policy(),
    )
    outcome = await harness.executor.execute(harness.request)
    assert type(outcome) is TwoPhaseCycleFailure

    duplicate = _resigned_receipt(
        outcome.call_receipts[1],
        **{digest_field: getattr(outcome.call_receipts[0], digest_field)},
    )

    with pytest.raises(ValidationError, match="failure receipts do not match"):
        TwoPhaseCycleFailure.model_validate(
            _resigned_failure_values(
                outcome,
                receipts=(outcome.call_receipts[0], duplicate),
            )
        )


@pytest.mark.parametrize(
    "parse_status",
    (
        StructuredCallParseStatus.SCHEMA_INVALID,
        StructuredCallParseStatus.EMPTY_REMINDER,
        StructuredCallParseStatus.CLAIM_OVER_LIMIT,
    ),
)
async def test_result_validator_rejects_a_noncanonical_parse_rejection_reason(
    parse_status: StructuredCallParseStatus,
) -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (
            _valid(_operations(case, "noop")),
            _rejected(parse_status),
        ),
    )
    outcome = await harness.executor.execute(harness.request)
    assert type(outcome) is TwoPhaseCycleResult
    forged_intervention = _intervention_with_reason(
        outcome.intervention,
        ReasonCode.UNGROUNDED,
    )
    values = outcome.model_dump(mode="python", exclude={"result_digest"})
    values["intervention"] = forged_intervention

    with pytest.raises(ValidationError, match="components do not match"):
        TwoPhaseCycleResult.model_validate(values)


async def test_result_validator_rejects_canonical_rejection_swapped_to_reminder() -> None:
    reminder, rejection = await _paired_reminder_and_rejection_results()
    forged_values = reminder.intervention.model_dump(mode="python")
    forged_values["grounding_receipt"] = rejection.intervention.grounding_receipt
    forged_intervention = InterventionDecision.model_validate(forged_values)
    values = rejection.model_dump(mode="python", exclude={"result_digest"})
    values["intervention"] = forged_intervention

    with pytest.raises(ValidationError):
        TwoPhaseCycleResult.model_validate(values)


async def test_result_validator_rejects_reminder_swapped_to_canonical_silence() -> None:
    reminder, rejection = await _paired_reminder_and_rejection_results()
    forged_values = rejection.intervention.model_dump(mode="python")
    forged_values["grounding_receipt"] = reminder.intervention.grounding_receipt
    forged_intervention = InterventionDecision.model_validate(forged_values)
    values = reminder.model_dump(mode="python", exclude={"result_digest"})
    values["intervention"] = forged_intervention

    with pytest.raises(ValidationError):
        TwoPhaseCycleResult.model_validate(values)


async def test_result_validator_rejects_altered_rendered_reminder_text() -> None:
    reminder, _ = await _paired_reminder_and_rejection_results()
    forged_values = reminder.intervention.model_dump(mode="python")
    forged_values["rendered_text"] = "attacker-controlled reminder"
    forged_intervention = InterventionDecision.model_validate(forged_values)
    values = reminder.model_dump(mode="python", exclude={"result_digest"})
    values["intervention"] = forged_intervention

    with pytest.raises(ValidationError):
        TwoPhaseCycleResult.model_validate(values)


async def test_missing_candidate_memory_materializes_citation_missing_silence() -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (
            _valid(_operations(case, "noop")),
            _valid(_selection(remind_created_memory=True)),
        ),
    )

    outcome = await harness.executor.execute(harness.request)

    assert type(outcome) is TwoPhaseCycleResult
    assert outcome.intervention.action is InterventionAction.SILENCE
    assert outcome.intervention.reason_code is ReasonCode.CITATION_MISSING
    assert outcome.grounding_receipt.parse_status is ProposalParseStatus.VALID
    assert outcome.grounding_receipt.claims


@pytest.mark.parametrize("violation", ("stale_revision", "wrong_kind"))
async def test_result_validator_rejects_wrong_reason_for_invalid_candidate_provenance(
    violation: str,
) -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (
            _valid(_operations(case, "create")),
            _valid(_invalid_candidate_selection(case, violation)),
        ),
    )
    request = _cycle_request(case, assigned_memory_ids=(CREATED_KNOWLEDGE_ID,))
    outcome = await harness.executor.execute(request)
    assert type(outcome) is TwoPhaseCycleResult
    assert outcome.intervention.reason_code is ReasonCode.INVALID_PROVENANCE

    forged_intervention = _intervention_with_reason(
        outcome.intervention,
        ReasonCode.DUPLICATE_REMINDER,
    )
    values = outcome.model_dump(mode="python", exclude={"result_digest"})
    values["intervention"] = forged_intervention

    with pytest.raises(ValidationError, match="components do not match"):
        TwoPhaseCycleResult.model_validate(values)


async def test_result_validator_binds_phase_one_output_to_its_call_and_materialization() -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (
            _valid(_operations(case, "create")),
            _valid(_selection(remind_created_memory=True)),
        ),
    )
    request = _cycle_request(case, assigned_memory_ids=(CREATED_KNOWLEDGE_ID,))
    outcome = await harness.executor.execute(request)
    assert type(outcome) is TwoPhaseCycleResult
    values = outcome.model_dump(mode="python", exclude={"result_digest"})
    values["memory_edit_output"] = _operations(case, "noop")

    with pytest.raises(ValidationError, match="components do not match"):
        TwoPhaseCycleResult.model_validate(values)


async def test_result_validator_binds_the_exact_grounding_state_digest() -> None:
    case = await _running_case()
    harness = _harness(
        case,
        (
            _valid(_operations(case, "noop")),
            _valid(_selection(remind_created_memory=False)),
        ),
    )
    outcome = await harness.executor.execute(harness.request)
    assert type(outcome) is TwoPhaseCycleResult
    forged_final = _resigned_receipt(
        outcome.call_receipts[-1],
        grounding_state_digest="0" * 64,
    )
    values = outcome.model_dump(mode="python", exclude={"result_digest"})
    values["call_receipts"] = (*outcome.call_receipts[:-1], forged_final)

    with pytest.raises(ValidationError, match="components do not match"):
        TwoPhaseCycleResult.model_validate(values)
