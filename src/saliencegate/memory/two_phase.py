from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, cast

from saliencegate.domain import EvidenceSource, InterventionAction, canonical_json
from saliencegate.intervention import (
    GROUNDING_RECEIPT_VERSION,
    GroundingContext,
    GroundingPipeline,
    GroundingReceipt,
    GroundingState,
    ProposalParseStatus,
    verify_grounded_intervention,
)
from saliencegate.memory.materialize import (
    MATERIALIZATION_REQUEST_SCHEMA_VERSION,
    MaterializationFailureReason,
    MaterializedBankOperations,
    MemoryOperationMaterializationError,
    OperationMaterializationRequest,
    materialize_bank_operations,
    validated_materialized_bank_operations_for_request,
)
from saliencegate.memory.proposals import (
    BankOperationsProposal,
    DeleteMemory,
    InterventionSelectionOutput,
)
from saliencegate.ports.model_calls import (
    STRUCTURED_CALL_REQUEST_SCHEMA_VERSION,
    StructuredCallClient,
    StructuredCallParseStatus,
    StructuredCallPhase,
    StructuredCallRequest,
    StructuredCallResult,
    StructuredCallStatus,
    StructuredResponseSchemaVersion,
    validated_result_for_request,
)
from saliencegate.ports.repository import RunRepository
from saliencegate.ports.two_phase import (
    CallReceipt,
    OperationMaterializer,
    PhaseOneCycleOutcome,
    PhaseOneCycleResult,
    TwoPhaseBoundaryError,
    TwoPhaseCallPolicy,
    TwoPhaseCycleFailure,
    TwoPhaseCycleOutcome,
    TwoPhaseCycleRequest,
    TwoPhaseCycleResult,
    TwoPhaseFailureReason,
    TwoPhaseModelProfile,
    TwoPhaseUsage,
    call_policy_accepts_receipts,
    validated_two_phase_cycle_request,
)
from saliencegate.prompts import (
    PAPER_TWO_PHASE_FORCED_REMINDER_V1,
    PAPER_TWO_PHASE_V1,
    ActiveBankPromptView,
    BankViewKind,
    BuiltPrompt,
    build_active_bank_prompt_view,
)
from saliencegate.prompts.paper_two_phase_v1 import PaperTwoPhasePromptBundle


class TwoPhaseExecutionError(ValueError):
    """A value-free failure before or outside a known-cost cycle outcome."""

    def __init__(self) -> None:
        super().__init__("two-phase cycle execution failed validation")


class TwoPhaseExecutionCancelled(asyncio.CancelledError):
    """A value-free cancellation with the exact receipts known before interruption."""

    __slots__ = ("call_receipts", "cost_certainty", "usage")

    def __init__(
        self,
        call_receipts: tuple[CallReceipt, ...],
        *,
        cost_certainty: Literal["known", "unknown"],
    ) -> None:
        super().__init__()
        self.call_receipts = call_receipts
        self.cost_certainty = cost_certainty
        self.usage = TwoPhaseUsage.from_receipts(call_receipts)


class _ModelCallCancelled(asyncio.CancelledError):
    """Private marker for a dispatch whose terminal accounting is unknown."""


class RepositoryOperationMaterializer:
    """Bind authoritative operation materialization to one repository capability."""

    __slots__ = ("_repository",)

    def __init__(self, repository: RunRepository) -> None:
        self._repository = repository

    async def materialize(
        self,
        request: OperationMaterializationRequest,
    ) -> MaterializedBankOperations:
        return await materialize_bank_operations(request, repository=self._repository)


def _claims_have_candidate_memory_shape(output: InterventionSelectionOutput) -> bool:
    for claim in output.claims:
        reference = claim.evidence
        if (
            reference.source is not EvidenceSource.MEMORY
            or reference.field_path != "/content"
            or reference.span is not None
        ):
            return False
    return True


def _write_count(proposal: BankOperationsProposal) -> int:
    return sum(type(operation) is not DeleteMemory for operation in proposal.operations)


def _structured_request(
    *,
    cycle: TwoPhaseCycleRequest,
    prompt: BuiltPrompt,
    model_id: str,
    model_call_index: int,
    attempt: int,
) -> StructuredCallRequest:
    return StructuredCallRequest(
        schema_version=STRUCTURED_CALL_REQUEST_SCHEMA_VERSION,
        run_id=cycle.cycle_receipt.cycle.run_id,
        cycle_id=cycle.cycle_receipt.cycle.cycle_id,
        model_call_index=model_call_index,
        phase=prompt.identity.phase,
        attempt=attempt,
        model_id=model_id,
        prompt_template_id=prompt.identity.template_id,
        prompt_template_digest=prompt.identity.template_digest,
        response_schema_version=cast(
            StructuredResponseSchemaVersion,
            prompt.identity.response_schema_version,
        ),
        payload=prompt.request_payload.as_json_object(),
    )


def _transport_failure_reason(result: StructuredCallResult) -> TwoPhaseFailureReason | None:
    if result.status is StructuredCallStatus.MODEL_ERROR:
        return TwoPhaseFailureReason.MODEL_ERROR
    if result.status is StructuredCallStatus.MODEL_TIMEOUT:
        return TwoPhaseFailureReason.MODEL_TIMEOUT
    return None


def _is_repairable(result: StructuredCallResult) -> bool:
    return result.status is StructuredCallStatus.COMPLETED and result.parse_status in (
        StructuredCallParseStatus.SCHEMA_INVALID,
        StructuredCallParseStatus.EMPTY_REMINDER,
        StructuredCallParseStatus.CLAIM_OVER_LIMIT,
    )


@dataclass(frozen=True, slots=True)
class _PhaseRun:
    result: StructuredCallResult
    receipts: tuple[CallReceipt, ...]
    policy_accepted: bool


@dataclass(frozen=True, slots=True)
class _PhaseOneExecution:
    request: TwoPhaseCycleRequest
    proposal: BankOperationsProposal
    materialization: MaterializedBankOperations
    candidate_bank: ActiveBankPromptView
    candidate_state: GroundingState
    receipts: tuple[CallReceipt, ...]


_PROPOSAL_PARSE_STATUS = {
    StructuredCallParseStatus.SCHEMA_INVALID: ProposalParseStatus.SCHEMA_INVALID,
    StructuredCallParseStatus.EMPTY_REMINDER: ProposalParseStatus.EMPTY_REMINDER,
    StructuredCallParseStatus.CLAIM_OVER_LIMIT: ProposalParseStatus.CLAIM_OVER_LIMIT,
}


def _canonical_rejection_receipt(
    parse_status: ProposalParseStatus,
    context: GroundingContext,
) -> GroundingReceipt:
    return GroundingReceipt(
        receipt_version=GROUNDING_RECEIPT_VERSION,
        parse_status=parse_status,
        proposal_action=(
            None
            if parse_status is ProposalParseStatus.SCHEMA_INVALID
            else InterventionAction.REMIND
        ),
        claims=(),
        confidence=1.0,
        requested_delivery_target=context.requested_delivery_target,
        model_call_index=context.model_call_index,
        model_call_digest=context.model_call_digest,
    )


class PaperTwoPhaseCycleExecutor:
    """Execute the reviewed two-phase cycle without committing or delivering."""

    __slots__ = (
        "_call_policy",
        "_client",
        "_grounding_pipeline",
        "_materializer",
        "_model_profile",
        "_prompt_bundle",
    )

    def __init__(
        self,
        *,
        materializer: OperationMaterializer,
        client: StructuredCallClient,
        prompt_bundle: PaperTwoPhasePromptBundle,
        grounding_pipeline: GroundingPipeline,
        model_profile: TwoPhaseModelProfile,
        call_policy: TwoPhaseCallPolicy,
    ) -> None:
        try:
            exact_profile = TwoPhaseModelProfile.model_validate_json(
                model_profile.model_dump_json(warnings=False)
            )
            exact_policy = TwoPhaseCallPolicy.model_validate_json(
                call_policy.model_dump_json(warnings=False)
            )
            if (
                not isinstance(materializer, OperationMaterializer)
                or not isinstance(client, StructuredCallClient)
                or type(prompt_bundle) is not PaperTwoPhasePromptBundle
                or prompt_bundle not in (PAPER_TWO_PHASE_V1, PAPER_TWO_PHASE_FORCED_REMINDER_V1)
                or type(grounding_pipeline) is not GroundingPipeline
                or exact_profile.prompt_bundle_id != prompt_bundle.identity.bundle_id
                or exact_profile.prompt_bundle_digest != prompt_bundle.identity.bundle_digest
            ):
                raise ValueError
        except Exception:
            raise TwoPhaseExecutionError() from None
        self._materializer = materializer
        self._client = client
        self._prompt_bundle = prompt_bundle
        self._grounding_pipeline = grounding_pipeline
        self._model_profile = exact_profile
        self._call_policy = exact_policy

    def _preflight(self, request: TwoPhaseCycleRequest) -> None:
        cycle = request.cycle_receipt.cycle
        resolved = self._grounding_pipeline.resolved_configuration
        if (
            cycle.grounding_version != resolved.pipeline_version
            or cycle.grounding_configuration_digest != resolved.configuration_digest
            or canonical_json(cycle.grounding_configuration)
            != canonical_json(resolved.configuration)
        ):
            raise TwoPhaseExecutionError()

    def _policy_accepts(self, receipts: tuple[CallReceipt, ...]) -> bool:
        return call_policy_accepts_receipts(self._call_policy, receipts)

    def _failure(
        self,
        request: TwoPhaseCycleRequest,
        *,
        failed_phase: StructuredCallPhase,
        reason: TwoPhaseFailureReason,
        receipts: tuple[CallReceipt, ...],
        memory_edit_output: BankOperationsProposal | None = None,
        intervention_output: InterventionSelectionOutput | None = None,
        materialization_failure_reason: MaterializationFailureReason | None = None,
    ) -> TwoPhaseCycleFailure:
        cycle = request.cycle_receipt.cycle
        return TwoPhaseCycleFailure(
            request_digest=request.request_digest,
            run_id=cycle.run_id,
            cycle_id=cycle.cycle_id,
            window_digest=request.window.window_digest,
            prompt_bundle_digest=self._prompt_bundle.identity.bundle_digest,
            model_id=self._model_profile.model_id,
            model_profile_digest=self._model_profile.profile_digest,
            call_policy_digest=self._call_policy.policy_digest,
            call_policy=self._call_policy,
            failed_phase=failed_phase,
            reason=reason,
            assigned_memory_id_capacity=len(request.assigned_memory_ids),
            memory_edit_output=memory_edit_output,
            intervention_output=intervention_output,
            materialization_failure_reason=materialization_failure_reason,
            call_receipts=receipts,
            usage=TwoPhaseUsage.from_receipts(receipts),
        )

    async def _call(
        self,
        request: TwoPhaseCycleRequest,
        prompt: BuiltPrompt,
        *,
        model_call_index: int,
        attempt: int,
        grounding_state: GroundingState | None,
    ) -> tuple[StructuredCallResult, CallReceipt]:
        cancelled = False
        try:
            call_request = _structured_request(
                cycle=request,
                prompt=prompt,
                model_id=self._model_profile.model_id,
                model_call_index=model_call_index,
                attempt=attempt,
            )
            raw_result = await self._client.generate(call_request)
            result = validated_result_for_request(call_request, raw_result)
            return result, CallReceipt.from_call(
                prompt,
                call_request,
                result,
                grounding_state=grounding_state,
            )
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            pass
        if cancelled:
            raise _ModelCallCancelled()
        raise TwoPhaseExecutionError()

    async def _run_phase(
        self,
        request: TwoPhaseCycleRequest,
        prompt: BuiltPrompt,
        *,
        prior_receipts: tuple[CallReceipt, ...],
        grounding_state: GroundingState | None,
    ) -> _PhaseRun:
        receipts = prior_receipts
        attempt = 0
        while True:
            call_cancelled = False
            try:
                result, receipt = await self._call(
                    request,
                    prompt,
                    model_call_index=len(receipts),
                    attempt=attempt,
                    grounding_state=grounding_state,
                )
            except _ModelCallCancelled:
                call_cancelled = True
            if call_cancelled:
                raise TwoPhaseExecutionCancelled(
                    receipts,
                    cost_certainty="unknown",
                )
            receipts += (receipt,)
            try:
                usage = TwoPhaseUsage.from_receipts(receipts)
                policy_accepted = self._policy_accepts(receipts)
            except Exception:
                return _PhaseRun(
                    result=result,
                    receipts=receipts,
                    policy_accepted=False,
                )
            if (
                not policy_accepted
                or not _is_repairable(result)
                or usage.schema_repairs >= self._call_policy.max_schema_repairs
            ):
                return _PhaseRun(
                    result=result,
                    receipts=receipts,
                    policy_accepted=policy_accepted,
                )
            attempt += 1

    async def execute(self, request: TwoPhaseCycleRequest) -> TwoPhaseCycleOutcome:
        known_cancellation: TwoPhaseExecutionCancelled | None = None
        cancelled = False
        boundary_failed = False
        try:
            return await self._execute(request)
        except TwoPhaseExecutionCancelled as error:
            known_cancellation = error
        except asyncio.CancelledError:
            cancelled = True
        except TwoPhaseBoundaryError:
            boundary_failed = True
        except Exception:
            pass
        if known_cancellation is not None:
            raise known_cancellation
        if cancelled:
            raise TwoPhaseExecutionCancelled((), cost_certainty="unknown")
        if boundary_failed:
            raise TwoPhaseBoundaryError("cycle request")
        raise TwoPhaseExecutionError()

    async def execute_phase_one(self, request: TwoPhaseCycleRequest) -> PhaseOneCycleOutcome:
        known_cancellation: TwoPhaseExecutionCancelled | None = None
        cancelled = False
        boundary_failed = False
        try:
            return await self._execute_phase_one(request)
        except TwoPhaseExecutionCancelled as error:
            known_cancellation = error
        except asyncio.CancelledError:
            cancelled = True
        except TwoPhaseBoundaryError:
            boundary_failed = True
        except Exception:
            pass
        if known_cancellation is not None:
            raise known_cancellation
        if cancelled:
            raise TwoPhaseExecutionCancelled((), cost_certainty="unknown")
        if boundary_failed:
            raise TwoPhaseBoundaryError("cycle request")
        raise TwoPhaseExecutionError()

    async def _execute_phase_one(self, request: TwoPhaseCycleRequest) -> PhaseOneCycleOutcome:
        outcome = await self._run_phase_one(request)
        if type(outcome) is TwoPhaseCycleFailure:
            return outcome
        phase_one = cast(_PhaseOneExecution, outcome)
        checked = phase_one.request
        cycle = checked.cycle_receipt.cycle
        return PhaseOneCycleResult(
            request_digest=checked.request_digest,
            run_id=cycle.run_id,
            cycle_id=cycle.cycle_id,
            window_digest=checked.window.window_digest,
            current_bank_view_digest=checked.current_bank.view_digest,
            current_bank_source_projection_digest=(checked.current_bank.source_projection_digest),
            candidate_bank_view_digest=phase_one.candidate_bank.view_digest,
            prompt_bundle_digest=self._prompt_bundle.identity.bundle_digest,
            model_id=self._model_profile.model_id,
            model_profile_digest=self._model_profile.profile_digest,
            model_profile=self._model_profile,
            call_policy_digest=self._call_policy.policy_digest,
            call_policy=self._call_policy,
            materialization=phase_one.materialization,
            memory_edit_output=phase_one.proposal,
            call_receipts=phase_one.receipts,
            usage=TwoPhaseUsage.from_receipts(phase_one.receipts),
        )

    async def _run_phase_one(
        self,
        request: TwoPhaseCycleRequest,
    ) -> _PhaseOneExecution | TwoPhaseCycleFailure:
        try:
            checked = validated_two_phase_cycle_request(request)
            self._preflight(checked)
            phase_one_prompt = self._prompt_bundle.build_memory_edit(
                window=checked.window,
                bank=checked.current_bank,
            )
        except (TwoPhaseBoundaryError, TwoPhaseExecutionError):
            raise
        except Exception:
            raise TwoPhaseExecutionError() from None

        phase_one_run = await self._run_phase(
            checked,
            phase_one_prompt,
            prior_receipts=(),
            grounding_state=None,
        )
        phase_one = phase_one_run.result
        phase_one_output = (
            phase_one.output if type(phase_one.output) is BankOperationsProposal else None
        )
        receipts = phase_one_run.receipts
        failure_reason = _transport_failure_reason(phase_one)
        if failure_reason is not None:
            return self._failure(
                checked,
                failed_phase=StructuredCallPhase.MEMORY_EDIT,
                reason=failure_reason,
                receipts=receipts,
                memory_edit_output=phase_one_output,
            )
        if not phase_one_run.policy_accepted:
            return self._failure(
                checked,
                failed_phase=StructuredCallPhase.MEMORY_EDIT,
                reason=TwoPhaseFailureReason.CALL_POLICY_EXCEEDED,
                receipts=receipts,
                memory_edit_output=phase_one_output,
            )
        if phase_one.parse_status is not StructuredCallParseStatus.VALID:
            return self._failure(
                checked,
                failed_phase=StructuredCallPhase.MEMORY_EDIT,
                reason=(
                    TwoPhaseFailureReason.REPAIR_EXHAUSTED
                    if receipts[-1].attempt > 0
                    else TwoPhaseFailureReason.SCHEMA_INVALID
                ),
                receipts=receipts,
                memory_edit_output=phase_one_output,
            )
        if phase_one_output is None:
            raise TwoPhaseExecutionError()

        proposal = phase_one_output
        write_count = _write_count(proposal)
        if write_count > len(checked.assigned_memory_ids):
            return self._failure(
                checked,
                failed_phase=StructuredCallPhase.MEMORY_EDIT,
                reason=TwoPhaseFailureReason.OPERATION_OVERFLOW,
                receipts=receipts,
                memory_edit_output=proposal,
            )
        materialization: MaterializedBankOperations | None = None
        materialization_cancelled = False
        try:
            materialization_request = OperationMaterializationRequest(
                schema_version=MATERIALIZATION_REQUEST_SCHEMA_VERSION,
                cycle_receipt=checked.cycle_receipt,
                proposal=proposal,
                delta_id=checked.delta_id,
                created_at=checked.created_at,
                assigned_memory_ids=checked.assigned_memory_ids[:write_count],
            )
            raw_materialization = await self._materializer.materialize(materialization_request)
            materialization = validated_materialized_bank_operations_for_request(
                materialization_request,
                raw_materialization,
            )
        except asyncio.CancelledError:
            materialization_cancelled = True
        except MemoryOperationMaterializationError as error:
            reason = (
                TwoPhaseFailureReason.INVALID_OPERATION
                if error.reason
                in {
                    MaterializationFailureReason.INVALID_INPUT,
                    MaterializationFailureReason.REFERENCE_MISSING,
                    MaterializationFailureReason.REFERENCE_STALE,
                    MaterializationFailureReason.REFERENCE_INACTIVE,
                    MaterializationFailureReason.REFERENCE_EXPIRED,
                    MaterializationFailureReason.REFERENCE_FUTURE,
                    MaterializationFailureReason.REFERENCE_INVALID,
                    MaterializationFailureReason.OPERATION_CONFLICT,
                }
                else TwoPhaseFailureReason.MATERIALIZATION_REJECTED
            )
            return self._failure(
                checked,
                failed_phase=StructuredCallPhase.MEMORY_EDIT,
                reason=reason,
                receipts=receipts,
                memory_edit_output=proposal,
                materialization_failure_reason=error.reason,
            )
        except Exception:
            return self._failure(
                checked,
                failed_phase=StructuredCallPhase.MEMORY_EDIT,
                reason=TwoPhaseFailureReason.MATERIALIZATION_REJECTED,
                receipts=receipts,
                memory_edit_output=proposal,
            )
        if materialization_cancelled:
            raise TwoPhaseExecutionCancelled(
                receipts,
                cost_certainty="known",
            )
        if materialization is None:
            return self._failure(
                checked,
                failed_phase=StructuredCallPhase.MEMORY_EDIT,
                reason=TwoPhaseFailureReason.MATERIALIZATION_REJECTED,
                receipts=receipts,
                memory_edit_output=proposal,
                materialization_failure_reason=MaterializationFailureReason.RESULT_MISMATCH,
            )
        if (
            materialization.source_projection_digest
            != checked.current_bank.source_projection_digest
        ):
            return self._failure(
                checked,
                failed_phase=StructuredCallPhase.MEMORY_EDIT,
                reason=TwoPhaseFailureReason.MATERIALIZATION_REJECTED,
                receipts=receipts,
                memory_edit_output=proposal,
                materialization_failure_reason=MaterializationFailureReason.RESULT_MISMATCH,
            )

        try:
            candidate_bank = build_active_bank_prompt_view(
                kind=BankViewKind.CANDIDATE_POST_DELTA,
                run_id=materialization.run_id,
                as_of=materialization.delta.created_at,
                source_projection_digest=materialization.preview_projection_digest,
                records=materialization.active_bank,
            )
            candidate_state = GroundingState(
                schema_version=checked.grounding_state.schema_version,
                events=checked.grounding_state.events,
                memories=materialization.active_bank,
                reminder_history=checked.grounding_state.reminder_history,
            )
        except Exception:
            return self._failure(
                checked,
                failed_phase=StructuredCallPhase.MEMORY_EDIT,
                reason=TwoPhaseFailureReason.MATERIALIZATION_REJECTED,
                receipts=receipts,
                memory_edit_output=proposal,
                materialization_failure_reason=MaterializationFailureReason.PREVIEW_REJECTED,
            )

        return _PhaseOneExecution(
            request=checked,
            proposal=proposal,
            materialization=materialization,
            candidate_bank=candidate_bank,
            candidate_state=candidate_state,
            receipts=receipts,
        )

    async def _execute(self, request: TwoPhaseCycleRequest) -> TwoPhaseCycleOutcome:
        phase_one_outcome = await self._run_phase_one(request)
        if type(phase_one_outcome) is TwoPhaseCycleFailure:
            return phase_one_outcome
        phase_one = cast(_PhaseOneExecution, phase_one_outcome)
        checked = phase_one.request
        proposal = phase_one.proposal
        materialization = phase_one.materialization
        candidate_bank = phase_one.candidate_bank
        candidate_state = phase_one.candidate_state
        receipts = phase_one.receipts
        try:
            phase_two_prompt = self._prompt_bundle.build_intervention(
                window=checked.window,
                bank=candidate_bank,
            )
        except Exception:
            return self._failure(
                checked,
                failed_phase=StructuredCallPhase.MEMORY_EDIT,
                reason=TwoPhaseFailureReason.MATERIALIZATION_REJECTED,
                receipts=receipts,
                memory_edit_output=proposal,
                materialization_failure_reason=MaterializationFailureReason.PREVIEW_REJECTED,
            )

        phase_two_run = await self._run_phase(
            checked,
            phase_two_prompt,
            prior_receipts=receipts,
            grounding_state=candidate_state,
        )
        phase_two = phase_two_run.result
        phase_two_output = (
            phase_two.output if type(phase_two.output) is InterventionSelectionOutput else None
        )
        receipts = phase_two_run.receipts
        phase_two_receipt = receipts[-1]
        failure_reason = _transport_failure_reason(phase_two)
        if failure_reason is not None:
            return self._failure(
                checked,
                failed_phase=StructuredCallPhase.INTERVENTION,
                reason=failure_reason,
                receipts=receipts,
                memory_edit_output=proposal,
                intervention_output=phase_two_output,
            )
        if not phase_two_run.policy_accepted:
            return self._failure(
                checked,
                failed_phase=StructuredCallPhase.INTERVENTION,
                reason=TwoPhaseFailureReason.CALL_POLICY_EXCEEDED,
                receipts=receipts,
                memory_edit_output=proposal,
                intervention_output=phase_two_output,
            )

        try:
            grounding_context = GroundingContext(
                schema_version="1.0",
                intervention_id=checked.intervention_id,
                run_id=materialization.run_id,
                cycle_id=checked.cycle_receipt.cycle.cycle_id,
                current_event_sequence=checked.cycle_receipt.cycle.last_event_sequence,
                created_at=checked.created_at,
                requested_delivery_target=checked.cycle_receipt.cycle.requested_delivery_target,
                model_call_index=phase_two_receipt.model_call_index,
                model_call_digest=phase_two_receipt.call_digest,
            )
            if phase_two.parse_status is StructuredCallParseStatus.VALID:
                if type(phase_two.output) is not InterventionSelectionOutput:
                    raise ValueError
                if _claims_have_candidate_memory_shape(phase_two.output):
                    intervention = self._grounding_pipeline.ground(
                        phase_two.output.to_grounding_proposal(),
                        context=grounding_context,
                        state=candidate_state,
                    )
                else:
                    rejection = _canonical_rejection_receipt(
                        ProposalParseStatus.SCHEMA_INVALID,
                        grounding_context,
                    )
                    intervention = self._grounding_pipeline.replay_receipt(
                        rejection,
                        context=grounding_context,
                        state=candidate_state,
                    )
            else:
                proposal_parse_status = _PROPOSAL_PARSE_STATUS.get(phase_two.parse_status)
                if proposal_parse_status is None:
                    raise ValueError
                rejection = _canonical_rejection_receipt(
                    proposal_parse_status,
                    grounding_context,
                )
                intervention = self._grounding_pipeline.replay_receipt(
                    rejection,
                    context=grounding_context,
                    state=candidate_state,
                )
            verify_grounded_intervention(
                intervention,
                context=grounding_context,
                state=candidate_state,
                expected_configuration=self._grounding_pipeline.resolved_configuration,
            )
            grounding_receipt = GroundingReceipt.model_validate_json(
                canonical_json(intervention.grounding_receipt)
            )
            return TwoPhaseCycleResult(
                request_digest=checked.request_digest,
                run_id=materialization.run_id,
                cycle_id=checked.cycle_receipt.cycle.cycle_id,
                window_digest=checked.window.window_digest,
                current_bank_view_digest=checked.current_bank.view_digest,
                candidate_bank_view_digest=candidate_bank.view_digest,
                prompt_bundle_digest=self._prompt_bundle.identity.bundle_digest,
                model_id=self._model_profile.model_id,
                model_profile_digest=self._model_profile.profile_digest,
                call_policy_digest=self._call_policy.policy_digest,
                call_policy=self._call_policy,
                materialization=materialization,
                memory_edit_output=proposal,
                intervention_output=phase_two_output,
                intervention=intervention,
                grounding_receipt=grounding_receipt,
                grounding_state=candidate_state,
                call_receipts=receipts,
                usage=TwoPhaseUsage.from_receipts(receipts),
            )
        except Exception:
            return self._failure(
                checked,
                failed_phase=StructuredCallPhase.INTERVENTION,
                reason=TwoPhaseFailureReason.CALL_CONTRACT_INVALID,
                receipts=receipts,
                memory_edit_output=proposal,
                intervention_output=phase_two_output,
            )


__all__ = [
    "PaperTwoPhaseCycleExecutor",
    "RepositoryOperationMaterializer",
    "TwoPhaseExecutionCancelled",
    "TwoPhaseExecutionError",
]
