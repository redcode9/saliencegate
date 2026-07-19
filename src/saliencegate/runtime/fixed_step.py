from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Any, Literal, TypeVar, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.domain import (
    MAX_MEMORY_DELTA_ITEMS,
    BudgetAmounts,
    BudgetSnapshot,
    ConstraintStatus,
    CycleRecord,
    CycleState,
    DeliveryRecord,
    DeliveryTarget,
    EventType,
    InterventionAction,
    InterventionOutcome,
    InvocationDecision,
    MemoryRecord,
    NormalizedTraceEventDraft,
    OutcomeEvidenceMode,
    ReasonCode,
    RepeatedErrorStatus,
    TraceEvent,
    ValidityState,
    canonical_json,
    normalized_trace_event_draft_is_bounded,
)
from saliencegate.domain.records import UUID4, ComponentIdentifier
from saliencegate.intervention import (
    GroundingPipeline,
    GroundingState,
    ReminderHistory,
    claim_fingerprint,
)
from saliencegate.memory.materialize import (
    MaterializationFailureReason,
    MemoryOperationMaterializationError,
    OperationMaterializationRequest,
    materialize_bank_operations,
)
from saliencegate.ports.adapters import (
    AdapterCapabilities,
    DeliveryAdapter,
    DeliveryEnvelope,
    DeliveryReceipt,
    enqueue_delivery_binding,
    validated_capabilities,
)
from saliencegate.ports.model_calls import CanonicalUsageProvenance
from saliencegate.ports.repository import (
    CycleReceipt,
    EnqueueDelivery,
    PreviewConflictError,
    RevisionConflictError,
    RunNotFoundError,
    RunRepository,
)
from saliencegate.ports.trajectory import (
    ActionStepBinding,
    EventTextSelector,
    LogicalMessageBinding,
)
from saliencegate.ports.two_phase import (
    CallReceipt,
    TwoPhaseCycleExecutor,
    TwoPhaseCycleFailure,
    TwoPhaseCycleOutcome,
    TwoPhaseCycleRequest,
    TwoPhaseCycleResult,
    TwoPhaseFailureReason,
    call_policy_accepts_receipts,
)
from saliencegate.prompts import BankViewKind, build_active_bank_prompt_view
from saliencegate.runtime.algorithm_result import (
    AlgorithmConfigurationAttestation,
    AlgorithmRunResult,
    FixedStepRecoveryResult,
    _algorithm_runtime_uuid,
    _model_token_usage_attestation,
    _semantic_projection_digests,
)
from saliencegate.runtime.budget import BudgetGovernor, BudgetReservationDeniedError
from saliencegate.runtime.cycles import CycleCoordinator
from saliencegate.runtime.delivery import DeliveryWorker
from saliencegate.runtime.fixed_step_core import (
    FixedStepTraceBoundary,
    FixedStepTraceDriver,
    FixedStepTraceInput,
    FixedStepTraceInputError,
    FixedStepTraceInvariantError,
    record_reconciled_invocation_decision,
)
from saliencegate.runtime.message_window import MessageWindow
from saliencegate.runtime.scheduling import (
    FixedStepDecision,
    FixedStepReason,
)

FIXED_STEP_POLICY_VERSION: Literal["paper-fixed-step/v1"] = "paper-fixed-step/v1"
FIXED_STEP_EVENT_INPUT_SCHEMA_VERSION: Literal["fixed-step-event-input/v1"] = (
    "fixed-step-event-input/v1"
)

_MAX_EVENTS = 10_000


class FixedStepRuntimeError(RuntimeError):
    """A value-free failure at the fixed-step orchestration boundary."""


class FixedStepInputError(FixedStepRuntimeError):
    def __init__(self) -> None:
        super().__init__("fixed-step runtime input failed validation")


class FixedStepExecutionError(FixedStepRuntimeError):
    def __init__(self) -> None:
        super().__init__("fixed-step cycle execution failed")


class FixedStepInvariantError(FixedStepRuntimeError):
    def __init__(self) -> None:
        super().__init__("fixed-step authoritative state diverged")


class _FixedStepMemoryConflictError(Exception):
    pass


class _FixedStepModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class FixedStepEventInput(_FixedStepModel):
    """One normalized event plus selectors resolved only after persistence."""

    schema_version: Literal["fixed-step-event-input/v1"] = FIXED_STEP_EVENT_INPUT_SCHEMA_VERSION
    draft: NormalizedTraceEventDraft = Field(repr=False)
    expected_event_id: UUID4
    task_description: EventTextSelector | None = None
    logical_messages: tuple[LogicalMessageBinding, ...] = Field(default=(), max_length=64)
    action_step: ActionStepBinding | None = None
    target_request_id: ComponentIdentifier | None = None

    @model_validator(mode="after")
    def draft_and_selectors_are_exact(self) -> FixedStepEventInput:
        try:
            copied = NormalizedTraceEventDraft.model_validate_json(
                self.draft.model_dump_json(warnings=False)
            )
            if copied != self.draft or not normalized_trace_event_draft_is_bounded(copied):
                raise ValueError
        except Exception:
            raise ValueError("fixed-step normalized draft failed validation") from None
        return self


@dataclass(frozen=True, slots=True)
class _FrozenDeliveryAdapter:
    adapter: DeliveryAdapter
    declared_capabilities: AdapterCapabilities

    def capabilities(self) -> AdapterCapabilities:
        return self.declared_capabilities

    async def deliver(self, delivery: DeliveryEnvelope) -> DeliveryReceipt:
        return await self.adapter.deliver(delivery)


@dataclass(slots=True)
class _CycleGuard:
    running: CycleReceipt | None = None


@dataclass(frozen=True, slots=True)
class _CycleCompletion:
    cycle: CycleRecord
    request: TwoPhaseCycleRequest
    execution: TwoPhaseCycleOutcome
    calls: tuple[CallReceipt, ...]
    delivery: DeliveryRecord | None
    outcome: InterventionOutcome | None


@dataclass(frozen=True, slots=True)
class _FixedStepBoundaryProjection:
    decision: InvocationDecision
    completion: _CycleCompletion | None


def _utc(value: object) -> datetime:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise FixedStepInputError()
    return value.astimezone(UTC)


def _draft_pointer(item: FixedStepEventInput, field_path: str) -> object:
    value: object = {"payload": item.draft.payload}
    try:
        for encoded_segment in field_path.split("/")[1:]:
            segment = encoded_segment.replace("~1", "/").replace("~0", "~")
            if isinstance(value, Mapping):
                value = value[segment]
            elif type(value) in (list, tuple):
                if not segment.isdigit() or (len(segment) > 1 and segment.startswith("0")):
                    raise KeyError
                sequence = cast(list[object] | tuple[object, ...], value)
                value = sequence[int(segment)]
            else:
                raise KeyError
        return value
    except Exception:
        raise FixedStepInputError() from None


def _preflight_selectors(items: tuple[FixedStepEventInput, ...]) -> None:
    last_step: int | None = None
    for item in items:
        selectors = []
        selector_paths: list[str] = []
        if item.task_description is not None:
            selectors.append(item.task_description)
            selector_paths.append(item.task_description.field_path)
        selectors.extend(binding.selector for binding in item.logical_messages)
        selector_paths.extend(binding.selector.field_path for binding in item.logical_messages)
        if item.action_step is not None:
            selector_paths.append(item.action_step.field_path)
        if len(set(selector_paths)) != len(selector_paths):
            raise FixedStepInputError()
        for selector in selectors:
            selected = _draft_pointer(item, selector.field_path)
            if type(selected) is not str or not selected:
                raise FixedStepInputError()
            encoded = selected.encode("utf-8", errors="strict")
            if selector.span is not None:
                try:
                    if selector.span.end_byte > len(encoded):
                        raise ValueError
                    selected = encoded[selector.span.start_byte : selector.span.end_byte].decode(
                        "utf-8", errors="strict"
                    )
                except Exception:
                    raise FixedStepInputError() from None
                if not selected:
                    raise FixedStepInputError()
        step_binding = item.action_step
        if step_binding is None:
            continue
        step = _draft_pointer(item, step_binding.field_path)
        if type(step) is not int or not 1 <= step <= (1 << 63) - 1:
            raise FixedStepInputError()
        if last_step is not None and step < last_step:
            raise FixedStepInputError()
        last_step = step


def _exact_event_inputs(value: object) -> tuple[FixedStepEventInput, ...]:
    if type(value) is not tuple or not value or len(value) > _MAX_EVENTS:
        raise FixedStepInputError()
    copied: list[FixedStepEventInput] = []
    try:
        for item in cast(tuple[object, ...], value):
            if type(item) is not FixedStepEventInput:
                raise ValueError
            exact = FixedStepEventInput.model_validate_json(item.model_dump_json(warnings=False))
            if exact != item:
                raise ValueError
            copied.append(exact)
    except Exception:
        raise FixedStepInputError() from None
    run_id = copied[0].draft.run_id
    expected_order = {
        item.expected_event_id: ordinal for ordinal, item in enumerate(copied, start=1)
    }
    if (
        copied[0].draft.event_type is not EventType.RUN_START
        or copied[0].task_description is None
        or any(item.draft.run_id != run_id for item in copied)
        or any(item.task_description is not None for item in copied[1:])
        or any(
            later.draft.timestamp < earlier.draft.timestamp for earlier, later in pairwise(copied)
        )
        or len({item.expected_event_id for item in copied}) != len(copied)
        or len({item.draft.source_event_id for item in copied}) != len(copied)
        or any(
            parent_id not in expected_order or expected_order[parent_id] >= ordinal
            for ordinal, item in enumerate(copied, start=1)
            for parent_id in item.draft.parent_ids
        )
    ):
        raise FixedStepInputError()
    result = tuple(copied)
    _preflight_selectors(result)
    return result


def _decision_reason(scheduled: FixedStepDecision, *, budget_available: bool) -> ReasonCode:
    if scheduled.invoke and not budget_available:
        return ReasonCode.BUDGET_EXHAUSTED
    if scheduled.reason is FixedStepReason.BOOTSTRAP:
        return ReasonCode.BOOTSTRAP
    if scheduled.invoke:
        return ReasonCode.SCRIPTED_INVOKE
    return ReasonCode.SCRIPTED_SILENCE


def _failure_reason(value: TwoPhaseFailureReason) -> ReasonCode:
    if value is TwoPhaseFailureReason.MODEL_TIMEOUT:
        return ReasonCode.MODEL_TIMEOUT
    if value is TwoPhaseFailureReason.MODEL_ERROR:
        return ReasonCode.MODEL_ERROR
    return ReasonCode.INVALID_STRUCTURED_OUTPUT


def _usage_settlement(
    outcome: TwoPhaseCycleOutcome,
    reservation: BudgetAmounts,
    *,
    interventions: int,
) -> BudgetAmounts:
    usage = outcome.usage
    provider_known = (
        usage.provider_input_tokens is not None and usage.provider_output_tokens is not None
    )
    return BudgetAmounts(
        model_calls=usage.model_calls,
        input_tokens=(
            cast(int, usage.provider_input_tokens) if provider_known else reservation.input_tokens
        ),
        output_tokens=(
            cast(int, usage.provider_output_tokens) if provider_known else reservation.output_tokens
        ),
        canonical_token_equivalents=(
            usage.canonical_token_equivalents
            if usage.canonical_token_equivalents is not None
            else reservation.canonical_token_equivalents
        ),
        latency_us=usage.latency_us,
        interventions=interventions,
        schema_repairs=usage.schema_repairs,
    )


def _canonical_usage_matches_configuration(
    receipts: tuple[CallReceipt, ...],
    configuration: AlgorithmConfigurationAttestation,
) -> bool:
    configured = configuration.model_token_counter
    for receipt in receipts:
        usage = receipt.usage
        if usage.canonical_usage_provenance is CanonicalUsageProvenance.UNAVAILABLE:
            continue
        if configured is None or (
            usage.local_counter_id != configured.counter_id
            or usage.local_counter_version != configured.counter_version
            or usage.local_counter_configuration_digest != configured.configuration_digest
            or usage.local_counter_model_id != configured.model_id
        ):
            return False
    return True


def _delivery_id_factory(trace_digest: str, delivery_id: UUID) -> Callable[[], UUID]:
    ordinal = 0

    def next_identifier() -> UUID:
        nonlocal ordinal
        ordinal += 1
        return _algorithm_runtime_uuid(trace_digest, "delivery-worker", delivery_id, ordinal)

    return next_identifier


def _recovery_id_factory(
    configuration_digest: str,
    run_id: UUID,
    recovered_at: datetime,
    ledger_head_tag: str,
) -> Callable[[], UUID]:
    ordinal = 0

    def next_identifier() -> UUID:
        nonlocal ordinal
        ordinal += 1
        return _algorithm_runtime_uuid(
            configuration_digest,
            "delivery-recovery",
            run_id,
            recovered_at.isoformat().replace("+00:00", "Z"),
            ledger_head_tag,
            ordinal,
        )

    return next_identifier


_BoundaryResult = TypeVar("_BoundaryResult")


async def _complete_boundary(
    operation: Coroutine[Any, Any, _BoundaryResult],
) -> tuple[_BoundaryResult, bool]:
    """Finish one repository transition even if its caller is cancelled."""

    task = asyncio.create_task(operation)
    cancellation_requested = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.cancelled():
                cancellation_requested = True
        except Exception:
            break
    try:
        result = task.result()
    except asyncio.CancelledError:
        raise
    except Exception:
        if cancellation_requested:
            raise asyncio.CancelledError() from None
        raise
    return result, cancellation_requested


async def _drain_cleanup(task: asyncio.Task[None]) -> None:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    task.result()


class FixedStepRunner:
    """Execute the paper-style schedule over the authoritative repository."""

    __slots__ = (
        "_configuration",
        "_delivery_adapter",
        "_executor",
        "_governor",
        "_grounding",
        "_repository",
    )

    def __init__(
        self,
        *,
        repository: RunRepository,
        executor: TwoPhaseCycleExecutor,
        grounding_pipeline: GroundingPipeline,
        configuration: AlgorithmConfigurationAttestation,
        delivery_adapter: DeliveryAdapter | None = None,
    ) -> None:
        try:
            exact = AlgorithmConfigurationAttestation.model_validate_json(
                configuration.model_dump_json(warnings=False)
            )
            resolved = grounding_pipeline.resolved_configuration
            if (
                not isinstance(executor, TwoPhaseCycleExecutor)
                or type(grounding_pipeline) is not GroundingPipeline
                or exact.grounding_configuration != resolved
                or exact.policy_version != FIXED_STEP_POLICY_VERSION
                or (
                    delivery_adapter is not None
                    and not isinstance(delivery_adapter, DeliveryAdapter)
                )
            ):
                raise ValueError
        except Exception:
            raise FixedStepInputError() from None
        self._repository = repository
        self._executor = executor
        self._grounding = grounding_pipeline
        self._configuration = exact
        self._delivery_adapter = delivery_adapter
        self._governor = BudgetGovernor()

    def _capabilities(self) -> AdapterCapabilities | None:
        target = self._configuration.requested_delivery_target
        adapter = self._delivery_adapter
        if target is None or adapter is None:
            return None
        try:
            capabilities = validated_capabilities(adapter.capabilities())
            if target is DeliveryTarget.PRE_ACTION_REPLAN and not (
                capabilities.pre_action_interception
            ):
                return None
            return capabilities
        except Exception:
            return None

    def _routing(
        self,
        target_request_id: ComponentIdentifier | None,
        capabilities: AdapterCapabilities | None,
    ) -> tuple[DeliveryTarget | None, EnqueueDelivery | None]:
        target = self._configuration.requested_delivery_target
        if target is None:
            return None, None
        if capabilities is None or target_request_id is None:
            return target, None
        try:
            binding = enqueue_delivery_binding(
                target_request_id=target_request_id,
                capabilities=capabilities,
            )
        except Exception:
            return target, None
        return target, binding

    async def _budget(self, run_id: UUID, *, first_decision: bool) -> BudgetSnapshot:
        if first_decision:
            return BudgetSnapshot(
                limits=self._configuration.budget_limits,
                reserved=BudgetAmounts(),
                consumed=BudgetAmounts(),
            )
        snapshot = await self._repository.budget_snapshot(run_id)
        if snapshot.limits != self._configuration.budget_limits:
            raise FixedStepInvariantError()
        return snapshot

    async def _grounding_state(
        self,
        run_id: UUID,
        *,
        current_sequence: int,
        memories: tuple[MemoryRecord, ...],
    ) -> GroundingState:
        entries = await self._repository.ledger(run_id)
        events = tuple(
            entry.record
            for entry in entries
            if type(entry.record) is TraceEvent and entry.record.sequence <= current_sequence
        )
        latest_cycles: dict[str, CycleRecord] = {}
        for entry in entries:
            if type(entry.record) is CycleRecord:
                cycle = entry.record
                latest_cycles[cycle.cycle_id] = cycle
        configuration = self._grounding.configuration
        history_window = max(
            configuration.duplicate_window_events,
            configuration.cooldown_events,
        )
        first_sequence = max(1, current_sequence - history_window)
        history: list[ReminderHistory] = []
        for cycle in sorted(
            latest_cycles.values(),
            key=lambda value: (value.last_event_sequence, value.cycle_id),
        ):
            intervention = cycle.intervention
            if (
                cycle.state is CycleState.COMMITTED
                and intervention is not None
                and intervention.action is InterventionAction.REMIND
                and first_sequence <= cycle.last_event_sequence < current_sequence
            ):
                history.append(
                    ReminderHistory(
                        schema_version="1.0",
                        intervention_id=intervention.intervention_id,
                        run_id=run_id,
                        event_sequence=cycle.last_event_sequence,
                        claim_digests=tuple(
                            claim_fingerprint(claim) for claim in intervention.claims
                        ),
                    )
                )
        try:
            return GroundingState(
                schema_version="1.0",
                events=events,
                memories=memories,
                reminder_history=tuple(history),
            )
        except Exception:
            raise FixedStepInvariantError() from None

    async def _unknown_failure(
        self,
        coordinator: CycleCoordinator,
        guard: _CycleGuard,
        *,
        updated_at: datetime,
    ) -> None:
        if guard.running is None:
            return
        state = guard.running.cycle.state
        if state is CycleState.PENDING:
            failed, cancelled = await _complete_boundary(
                coordinator.fail(
                    guard.running,
                    reason=ReasonCode.MODEL_ERROR,
                    updated_at=updated_at,
                )
            )
        elif state is CycleState.RESERVED:
            failed, cancelled = await _complete_boundary(
                coordinator.fail(
                    guard.running,
                    reason=ReasonCode.MODEL_ERROR,
                    settlement=BudgetAmounts(),
                    updated_at=updated_at,
                )
            )
        elif state is CycleState.RUNNING:
            failed, cancelled = await _complete_boundary(
                coordinator.fail(
                    guard.running,
                    reason=ReasonCode.FAILED_UNKNOWN_COST,
                    settlement=self._configuration.cycle_reservation,
                    updated_at=updated_at,
                )
            )
        else:  # pragma: no cover - guarded by the active lifecycle states above
            raise FixedStepInvariantError()
        del failed
        guard.running = None
        if cancelled:
            raise asyncio.CancelledError()

    async def _terminalize_unknown(
        self,
        coordinator: CycleCoordinator,
        guard: _CycleGuard,
        *,
        updated_at: datetime,
    ) -> None:
        cleanup = asyncio.create_task(
            self._unknown_failure(coordinator, guard, updated_at=updated_at)
        )
        try:
            await _drain_cleanup(cleanup)
        except BaseException:
            raise FixedStepInvariantError() from None

    async def _validate_outcome_for_request(
        self,
        outcome: TwoPhaseCycleOutcome,
        request: TwoPhaseCycleRequest,
    ) -> None:
        templates = {
            (item.phase, item.template_id, item.template_digest)
            for item in self._configuration.prompt_bundle.templates
        }
        receipts = outcome.call_receipts
        policy_accepted = call_policy_accepts_receipts(
            self._configuration.call_policy,
            receipts,
        )
        if (
            outcome.run_id != request.cycle_receipt.cycle.run_id
            or outcome.cycle_id != request.cycle_receipt.cycle.cycle_id
            or outcome.request_digest != request.request_digest
            or outcome.window_digest != request.window.window_digest
            or outcome.model_id != self._configuration.model_profile.model_id
            or outcome.model_profile_digest != self._configuration.model_profile.profile_digest
            or outcome.call_policy_digest != self._configuration.call_policy.policy_digest
            or outcome.call_policy != self._configuration.call_policy
            or outcome.prompt_bundle_digest != self._configuration.prompt_bundle.bundle_digest
            or any(
                receipt.run_id != outcome.run_id
                or receipt.cycle_id != outcome.cycle_id
                or receipt.model_id != outcome.model_id
                or receipt.window_digest != outcome.window_digest
                or (
                    receipt.phase,
                    receipt.prompt_template_id,
                    receipt.prompt_template_digest,
                )
                not in templates
                for receipt in receipts
            )
            or (
                type(outcome) is TwoPhaseCycleFailure
                and (outcome.reason is TwoPhaseFailureReason.CALL_POLICY_EXCEEDED)
                is policy_accepted
            )
            or (type(outcome) is TwoPhaseCycleResult and not policy_accepted)
        ):
            raise FixedStepExecutionError()
        if type(outcome) is not TwoPhaseCycleResult:
            return
        completed = outcome
        if (
            completed.current_bank_view_digest != request.current_bank.view_digest
            or completed.grounding_state.events != request.grounding_state.events
            or completed.grounding_state.reminder_history
            != request.grounding_state.reminder_history
            or completed.materialization.source_projection_digest
            != request.current_bank.source_projection_digest
        ):
            raise FixedStepExecutionError()
        try:
            materialization_request = OperationMaterializationRequest(
                schema_version="operation-materialization-request/v1",
                cycle_receipt=request.cycle_receipt,
                proposal=completed.memory_edit_output,
                delta_id=request.delta_id,
                created_at=request.created_at,
                assigned_memory_ids=request.assigned_memory_ids[
                    : len(completed.memory_id_assignments)
                ],
            )
            trusted = await materialize_bank_operations(
                materialization_request,
                repository=self._repository,
            )
        except MemoryOperationMaterializationError as error:
            if error.reason in {
                MaterializationFailureReason.SOURCE_CONFLICT,
                MaterializationFailureReason.IDENTITY_CONFLICT,
                MaterializationFailureReason.ASSIGNMENT_CONFLICT,
                MaterializationFailureReason.OPERATION_CONFLICT,
                MaterializationFailureReason.REFERENCE_STALE,
            }:
                raise _FixedStepMemoryConflictError() from None
            raise FixedStepExecutionError() from None
        except (PreviewConflictError, RevisionConflictError):
            raise _FixedStepMemoryConflictError() from None
        except Exception:
            raise FixedStepExecutionError() from None
        if trusted != completed.materialization:
            raise FixedStepExecutionError()

    async def _known_execution_failure(
        self,
        *,
        coordinator: CycleCoordinator,
        running: CycleReceipt,
        guard: _CycleGuard,
        request: TwoPhaseCycleRequest,
        execution: TwoPhaseCycleOutcome,
        reason: ReasonCode,
        updated_at: datetime,
    ) -> _CycleCompletion:
        calls = execution.call_receipts
        call_digests = tuple(receipt.call_digest for receipt in calls)
        latencies = tuple(receipt.usage.latency_us for receipt in calls)
        try:
            settlement = _usage_settlement(
                execution,
                self._configuration.cycle_reservation,
                interventions=0,
            )
            held = await self._repository.budget_snapshot(running.cycle.run_id)
            self._governor.settle(
                held,
                self._configuration.cycle_reservation,
                settlement,
                model_call_latencies_us=latencies,
            )
            failed, cancelled = await _complete_boundary(
                coordinator.fail(
                    running,
                    reason=reason,
                    settlement=settlement,
                    model_call_digests=call_digests,
                    model_call_latencies_us=latencies,
                    updated_at=updated_at,
                )
            )
            guard.running = None
            if cancelled:
                raise asyncio.CancelledError()
        except asyncio.CancelledError:
            await self._terminalize_unknown(coordinator, guard, updated_at=updated_at)
            raise
        except Exception:
            await self._terminalize_unknown(coordinator, guard, updated_at=updated_at)
            raise FixedStepInvariantError() from None
        return _CycleCompletion(
            cycle=failed.cycle,
            request=request,
            execution=execution,
            calls=calls,
            delivery=None,
            outcome=None,
        )

    async def _reconcile_commit(
        self,
        *,
        running: CycleReceipt,
        result: TwoPhaseCycleResult,
        settlement: BudgetAmounts,
        delivery_expected: bool,
        call_digests: tuple[str, ...],
        latencies: tuple[int, ...],
        updated_at: datetime,
    ) -> CycleReceipt | None:
        ledger = await self._repository.ledger(running.cycle.run_id)
        cycle_entries = tuple(
            entry
            for entry in ledger
            if type(entry.record) is CycleRecord and entry.record.cycle_id == running.cycle.cycle_id
        )
        if not cycle_entries:
            raise FixedStepInvariantError()
        entry = cycle_entries[-1]
        cycle = cast(CycleRecord, entry.record)
        if cycle.state is CycleState.RUNNING and cycle.revision == running.cycle.revision:
            return None
        expected_values = running.cycle.model_dump(mode="python", warnings=False)
        expected_values.update(
            revision=running.cycle.revision + 1,
            state=CycleState.COMMITTED,
            budget_settlement=settlement,
            model_call_digests=call_digests,
            model_call_latencies_us=latencies,
            validated_delta=result.validated_delta,
            memory_id_assignments=result.memory_id_assignments,
            intervention=result.intervention,
            updated_at=updated_at,
        )
        try:
            expected = CycleRecord.model_validate(expected_values)
        except Exception:
            raise FixedStepInvariantError() from None
        if cycle != expected:
            raise FixedStepInvariantError()
        pending_deliveries = tuple(
            record
            for record in (item.record for item in ledger)
            if type(record) is DeliveryRecord
            and record.cycle_id == cycle.cycle_id
            and record.revision == 1
        )
        if len(pending_deliveries) != int(delivery_expected):
            raise FixedStepInvariantError()
        return CycleReceipt(
            appended=False,
            cycle=cycle,
            record_tag=entry.record_tag,
            ledger_position=entry.position,
            chain_tag=entry.chain_tag,
            budget_snapshot=await self._repository.budget_snapshot(cycle.run_id),
            delivery=pending_deliveries[0] if pending_deliveries else None,
        )

    async def _outcome_is_durable(self, outcome: InterventionOutcome) -> bool:
        ledger = await self._repository.ledger(outcome.run_id)
        matches = tuple(
            record
            for record in (entry.record for entry in ledger)
            if type(record) is InterventionOutcome and record.outcome_id == outcome.outcome_id
        )
        if len(matches) > 1:
            raise FixedStepInvariantError()
        return matches == (outcome,)

    async def _record_outcome_idempotently(self, outcome: InterventionOutcome) -> None:
        try:
            receipt, cancelled = await _complete_boundary(self._repository.record_outcome(outcome))
        except asyncio.CancelledError:
            if not await self._outcome_is_durable(outcome):
                raise
            raise
        except Exception:
            if not await self._outcome_is_durable(outcome):
                raise FixedStepInvariantError() from None
            return
        if not receipt.appended and not await self._outcome_is_durable(outcome):
            raise FixedStepInvariantError()
        if cancelled:
            raise asyncio.CancelledError()

    async def _execute_cycle(
        self,
        *,
        coordinator: CycleCoordinator,
        decision: InvocationDecision,
        window: MessageWindow,
        target: DeliveryTarget | None,
        delivery_binding: EnqueueDelivery | None,
        capabilities: AdapterCapabilities | None,
        trace_digest: str,
    ) -> _CycleCompletion:
        timestamp = _utc(decision.created_at)
        pin = self._grounding.pin(target)
        guard = _CycleGuard()
        try:
            pending, cancelled = await _complete_boundary(
                coordinator.begin(decision, grounding=pin, created_at=timestamp)
            )
            guard.running = pending
            if cancelled:
                raise asyncio.CancelledError()
            reserved, cancelled = await _complete_boundary(
                coordinator.reserve(
                    pending,
                    reservation=self._configuration.cycle_reservation,
                    updated_at=timestamp,
                )
            )
            guard.running = reserved
            if cancelled:
                raise asyncio.CancelledError()
            running, cancelled = await _complete_boundary(
                coordinator.start(
                    reserved,
                    batch_digest=window.window_digest,
                    updated_at=timestamp,
                )
            )
            guard.running = running
            if cancelled:
                raise asyncio.CancelledError()
        except asyncio.CancelledError:
            await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
            raise
        except Exception:
            await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
            raise FixedStepExecutionError() from None
        try:
            snapshot = await self._repository.snapshot(decision.run_id)
        except asyncio.CancelledError:
            await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
            raise
        except Exception:
            await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
            raise FixedStepExecutionError() from None
        cycle = running.cycle
        if (
            snapshot.ledger_position != running.ledger_position
            or snapshot.ingestion_cursor != cycle.last_event_sequence
            or snapshot.memory_cursor != cycle.first_event_sequence - 1
        ):
            await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
            raise FixedStepInvariantError()
        records = tuple(
            sorted(
                (
                    record
                    for record in snapshot.records
                    if record.validity is ValidityState.ACTIVE
                    and (record.expires_at is None or record.expires_at > timestamp)
                ),
                key=lambda record: (record.kind.value, str(record.memory_id)),
            )
        )
        try:
            bank = build_active_bank_prompt_view(
                kind=BankViewKind.CURRENT,
                run_id=decision.run_id,
                as_of=timestamp,
                source_projection_digest=snapshot.projection_digest,
                records=records,
            )
            grounding_state = await self._grounding_state(
                decision.run_id,
                current_sequence=cycle.last_event_sequence,
                memories=records,
            )
            request = TwoPhaseCycleRequest(
                schema_version="two-phase-cycle-request/v1",
                cycle_receipt=running,
                window=window,
                current_bank=bank,
                grounding_state=grounding_state,
                delta_id=_algorithm_runtime_uuid(trace_digest, "delta", cycle.cycle_id),
                assigned_memory_ids=tuple(
                    _algorithm_runtime_uuid(trace_digest, "memory", cycle.cycle_id, ordinal)
                    for ordinal in range(1, MAX_MEMORY_DELTA_ITEMS + 1)
                ),
                intervention_id=_algorithm_runtime_uuid(
                    trace_digest,
                    "intervention",
                    cycle.cycle_id,
                ),
                created_at=timestamp,
            )
        except asyncio.CancelledError:
            await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
            raise
        except Exception:
            await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
            raise FixedStepInvariantError() from None

        try:
            raw_outcome = await self._executor.execute(request)
        except asyncio.CancelledError:
            await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
            raise
        except Exception:
            await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
            raise FixedStepExecutionError() from None

        outcome: TwoPhaseCycleOutcome
        try:
            if type(raw_outcome) is TwoPhaseCycleResult:
                outcome = TwoPhaseCycleResult.model_validate_json(
                    raw_outcome.model_dump_json(warnings=False)
                )
            elif type(raw_outcome) is TwoPhaseCycleFailure:
                outcome = TwoPhaseCycleFailure.model_validate_json(
                    raw_outcome.model_dump_json(warnings=False)
                )
            else:
                raise ValueError
            if (
                outcome.run_id != decision.run_id
                or outcome.cycle_id != cycle.cycle_id
                or outcome.request_digest != request.request_digest
                or outcome.model_profile_digest != self._configuration.model_profile.profile_digest
                or outcome.call_policy_digest != self._configuration.call_policy.policy_digest
                or outcome.prompt_bundle_digest != self._configuration.prompt_bundle.bundle_digest
                or not _canonical_usage_matches_configuration(
                    outcome.call_receipts,
                    self._configuration,
                )
            ):
                raise ValueError
            await self._validate_outcome_for_request(outcome, request)
        except _FixedStepMemoryConflictError:
            return await self._known_execution_failure(
                coordinator=coordinator,
                running=running,
                guard=guard,
                request=request,
                execution=outcome,
                reason=ReasonCode.MEMORY_CONFLICT,
                updated_at=timestamp,
            )
        except asyncio.CancelledError:
            await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
            raise
        except Exception:
            await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
            raise FixedStepExecutionError() from None

        calls = outcome.call_receipts
        call_digests = tuple(receipt.call_digest for receipt in calls)
        latencies = tuple(receipt.usage.latency_us for receipt in calls)
        if type(outcome) is TwoPhaseCycleFailure:
            failure = outcome
            try:
                settlement = _usage_settlement(
                    failure,
                    self._configuration.cycle_reservation,
                    interventions=0,
                )
                held = await self._repository.budget_snapshot(decision.run_id)
                self._governor.settle(
                    held,
                    self._configuration.cycle_reservation,
                    settlement,
                    model_call_latencies_us=latencies,
                )
            except asyncio.CancelledError:
                await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
                raise
            except Exception:
                await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
                raise FixedStepExecutionError() from None
            try:
                failed, cancelled = await _complete_boundary(
                    coordinator.fail(
                        running,
                        reason=_failure_reason(failure.reason),
                        settlement=settlement,
                        model_call_digests=call_digests,
                        model_call_latencies_us=latencies,
                        updated_at=timestamp,
                    )
                )
                guard.running = None
                if cancelled:
                    raise asyncio.CancelledError()
            except asyncio.CancelledError:
                await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
                raise
            except Exception:
                await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
                raise FixedStepInvariantError() from None
            return _CycleCompletion(
                cycle=failed.cycle,
                request=request,
                execution=failure,
                calls=calls,
                delivery=None,
                outcome=None,
            )

        result = cast(TwoPhaseCycleResult, outcome)
        intervention_count = int(result.intervention.action is InterventionAction.REMIND)
        try:
            settlement = _usage_settlement(
                result,
                self._configuration.cycle_reservation,
                interventions=intervention_count,
            )
            held = await self._repository.budget_snapshot(decision.run_id)
            self._governor.settle(
                held,
                self._configuration.cycle_reservation,
                settlement,
                model_call_latencies_us=latencies,
            )
        except asyncio.CancelledError:
            await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
            raise
        except Exception:
            await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
            raise FixedStepExecutionError() from None
        enqueue = delivery_binding if intervention_count else None
        if intervention_count and enqueue is None:
            try:
                failed_settlement = settlement.model_copy(update={"interventions": 0})
                self._governor.settle(
                    held,
                    self._configuration.cycle_reservation,
                    failed_settlement,
                    model_call_latencies_us=latencies,
                )
                failed, cancelled = await _complete_boundary(
                    coordinator.fail(
                        running,
                        reason=ReasonCode.TARGET_UNAVAILABLE,
                        settlement=failed_settlement,
                        model_call_digests=call_digests,
                        model_call_latencies_us=latencies,
                        updated_at=timestamp,
                    )
                )
                guard.running = None
                if cancelled:
                    raise asyncio.CancelledError()
            except asyncio.CancelledError:
                await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
                raise
            except Exception:
                await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
                raise FixedStepInvariantError() from None
            return _CycleCompletion(
                cycle=failed.cycle,
                request=request,
                execution=result,
                calls=calls,
                delivery=None,
                outcome=None,
            )
        try:
            committed, cancelled = await _complete_boundary(
                coordinator.commit(
                    running,
                    settlement=settlement,
                    validated_delta=result.validated_delta,
                    memory_id_assignments=result.memory_id_assignments,
                    intervention=result.intervention,
                    delivery=enqueue,
                    updated_at=timestamp,
                    model_call_digests=call_digests,
                    model_call_latencies_us=latencies,
                )
            )
            guard.running = None
            if cancelled:
                raise asyncio.CancelledError()
        except asyncio.CancelledError:
            if guard.running is not None:
                reconciled = await self._reconcile_commit(
                    running=running,
                    result=result,
                    settlement=settlement,
                    delivery_expected=enqueue is not None,
                    call_digests=call_digests,
                    latencies=latencies,
                    updated_at=timestamp,
                )
                if reconciled is None:
                    await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
                else:
                    guard.running = None
            raise
        except (PreviewConflictError, RevisionConflictError):
            reconciled = await self._reconcile_commit(
                running=running,
                result=result,
                settlement=settlement,
                delivery_expected=enqueue is not None,
                call_digests=call_digests,
                latencies=latencies,
                updated_at=timestamp,
            )
            if reconciled is None:
                return await self._known_execution_failure(
                    coordinator=coordinator,
                    running=running,
                    guard=guard,
                    request=request,
                    execution=result,
                    reason=ReasonCode.MEMORY_CONFLICT,
                    updated_at=timestamp,
                )
            committed = reconciled
            guard.running = None
        except Exception:
            try:
                reconciled = await self._reconcile_commit(
                    running=running,
                    result=result,
                    settlement=settlement,
                    delivery_expected=enqueue is not None,
                    call_digests=call_digests,
                    latencies=latencies,
                    updated_at=timestamp,
                )
            except Exception:
                await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
                raise FixedStepInvariantError() from None
            if reconciled is None:
                await self._terminalize_unknown(coordinator, guard, updated_at=timestamp)
                raise FixedStepExecutionError() from None
            committed = reconciled
            guard.running = None
        final_delivery: DeliveryRecord | None = None
        if committed.delivery is not None:
            if self._delivery_adapter is None or capabilities is None:
                raise FixedStepInvariantError()
            worker = DeliveryWorker(
                repository=self._repository,
                adapter=_FrozenDeliveryAdapter(
                    adapter=self._delivery_adapter,
                    declared_capabilities=capabilities,
                ),
                id_factory=_delivery_id_factory(
                    trace_digest,
                    committed.delivery.delivery_id,
                ),
            )
            delivered = await worker.deliver(
                decision.run_id,
                committed.delivery.delivery_id,
                now=timestamp,
            )
            final_delivery = delivered.delivery

        neutral = InterventionOutcome(
            outcome_id=_algorithm_runtime_uuid(
                trace_digest,
                "outcome",
                result.intervention.intervention_id,
            ),
            run_id=decision.run_id,
            intervention_id=result.intervention.intervention_id,
            repeated_error_status=RepeatedErrorStatus.UNKNOWN,
            constraint_status=ConstraintStatus.UNKNOWN,
            evidence_mode=OutcomeEvidenceMode.POLICY_REPLAY,
            created_at=timestamp,
        )
        await self._record_outcome_idempotently(neutral)
        return _CycleCompletion(
            cycle=committed.cycle,
            request=request,
            execution=result,
            calls=calls,
            delivery=final_delivery,
            outcome=neutral,
        )

    async def recover(
        self,
        run_id: UUID,
        *,
        recovered_at: datetime,
    ) -> FixedStepRecoveryResult:
        """Recover only durable cycle and delivery state; never call the model executor."""

        if type(run_id) is not UUID or run_id.version != 4:
            raise FixedStepInputError()
        timestamp = _utc(recovered_at)
        identifier = UUID(int=run_id.int)
        try:
            existing = await self._repository.ledger(identifier)
            decisions = tuple(
                entry.record for entry in existing if type(entry.record) is InvocationDecision
            )
            cycles = tuple(entry.record for entry in existing if type(entry.record) is CycleRecord)
            grounding = self._configuration.grounding_configuration
            if (
                not existing
                or not decisions
                or any(
                    decision.policy_version != self._configuration.policy_version
                    or decision.configuration_digest != self._configuration.configuration_digest
                    or decision.budget_snapshot.limits != self._configuration.budget_limits
                    for decision in decisions
                )
                or any(
                    cycle.policy_version != self._configuration.policy_version
                    or cycle.configuration_digest != self._configuration.configuration_digest
                    or cycle.grounding_version != grounding.pipeline_version
                    or cycle.grounding_configuration_digest != grounding.configuration_digest
                    or canonical_json(cycle.grounding_configuration)
                    != canonical_json(grounding.configuration)
                    or cycle.requested_delivery_target
                    is not self._configuration.requested_delivery_target
                    or (
                        cycle.budget_reservation is not None
                        and cycle.budget_reservation != self._configuration.cycle_reservation
                    )
                    for cycle in cycles
                )
            ):
                raise ValueError
            pre_recovery_head = await self._repository.ledger_head(identifier)
            coordinator = CycleCoordinator(self._repository)
            cycle_recovery = await coordinator.recover(identifier, recovered_at=timestamp)
            deliveries: tuple[DeliveryRecord, ...]
            if self._delivery_adapter is None:
                delivery_recovery = await self._repository.recover_deliveries(
                    identifier,
                    recovered_at=timestamp,
                )
                if (
                    delivery_recovery.resumable_pending
                    or delivery_recovery.resumable_claimed
                    or delivery_recovery.retryable_unknown
                ):
                    raise ValueError
                deliveries = delivery_recovery.non_retryable_unknown
            else:
                capabilities = self._capabilities()
                if capabilities is None:
                    raise ValueError
                seed_head = await self._repository.ledger_head(identifier)
                worker = DeliveryWorker(
                    repository=self._repository,
                    adapter=_FrozenDeliveryAdapter(
                        adapter=self._delivery_adapter,
                        declared_capabilities=capabilities,
                    ),
                    id_factory=_recovery_id_factory(
                        self._configuration.configuration_digest,
                        identifier,
                        timestamp,
                        seed_head.head_tag.value,
                    ),
                )
                deliveries = tuple(
                    result.delivery
                    for result in await worker.recover(identifier, recovered_at=timestamp)
                )
            rebuild = await self._repository.rebuild(identifier)
            if not rebuild.equivalent:
                raise ValueError
            ledger = await self._repository.ledger(identifier)
            ledger_head = await self._repository.ledger_head(identifier)
            budget = await self._repository.budget_snapshot(identifier)
            snapshot = await self._repository.snapshot(identifier)
            return FixedStepRecoveryResult(
                schema_version="fixed-step-recovery-result/v1",
                run_id=identifier,
                recovered_at=timestamp,
                configuration=self._configuration,
                cycle_recovery=cycle_recovery,
                deliveries=deliveries,
                budget_snapshot=budget,
                memory_snapshot=snapshot,
                semantic_projection_digests=_semantic_projection_digests(
                    identifier,
                    ledger,
                ),
                repository_projection_digests=rebuild.after,
                pre_recovery_ledger_head=pre_recovery_head,
                ledger=ledger,
                ledger_entry_count=len(ledger),
                ledger_head=ledger_head,
                rebuild_equivalent=rebuild.equivalent,
            )
        except RunNotFoundError:
            raise FixedStepInputError() from None
        except asyncio.CancelledError:
            raise
        except Exception:
            raise FixedStepInvariantError() from None

    async def run(self, events: tuple[FixedStepEventInput, ...]) -> AlgorithmRunResult:
        items = _exact_event_inputs(events)
        capabilities = self._capabilities()
        coordinator = CycleCoordinator(self._repository)
        trace_inputs = tuple(
            FixedStepTraceInput(
                draft=item.draft,
                expected_event_id=item.expected_event_id,
                task_description=item.task_description,
                logical_messages=item.logical_messages,
                action_step=item.action_step,
                target_request_id=item.target_request_id,
            )
            for item in items
        )

        async def project_boundary(
            boundary: FixedStepTraceBoundary,
        ) -> _FixedStepBoundaryProjection:
            event = boundary.event
            scheduled = boundary.scheduled
            window = boundary.window
            budget = await self._budget(
                event.run_id,
                first_decision=boundary.ordinal == 1,
            )
            budget_available = True
            if scheduled.invoke:
                try:
                    self._governor.reserve(
                        budget,
                        self._configuration.cycle_reservation,
                    )
                except BudgetReservationDeniedError:
                    budget_available = False
            invoke = scheduled.invoke and budget_available
            decision = InvocationDecision(
                decision_id=_algorithm_runtime_uuid(
                    boundary.trace_digest,
                    "decision",
                    event.sequence,
                ),
                run_id=event.run_id,
                event_sequence=event.sequence,
                invoke=invoke,
                risk_score=None,
                reason_codes=(_decision_reason(scheduled, budget_available=budget_available),),
                policy_version=self._configuration.policy_version,
                configuration_digest=self._configuration.configuration_digest,
                budget_snapshot=budget,
                cooldown_active=False,
                created_at=event.timestamp,
            )
            _, decision_cancelled = await record_reconciled_invocation_decision(
                self._repository,
                decision,
            )

            async def finish_boundary() -> _FixedStepBoundaryProjection:
                completion: _CycleCompletion | None = None
                if invoke:
                    if window is None:  # pragma: no cover - schedule invariant
                        raise FixedStepInvariantError()
                    target, delivery_binding = self._routing(
                        boundary.trace_input.target_request_id,
                        capabilities,
                    )
                    completion = await self._execute_cycle(
                        coordinator=coordinator,
                        decision=decision,
                        window=window,
                        target=target,
                        delivery_binding=delivery_binding,
                        capabilities=capabilities,
                        trace_digest=boundary.trace_digest,
                    )
                return _FixedStepBoundaryProjection(
                    decision=decision,
                    completion=completion,
                )

            if decision_cancelled:
                await finish_boundary()
                raise asyncio.CancelledError
            return await finish_boundary()

        try:
            trace = await FixedStepTraceDriver(self._repository).run(
                trace_inputs,
                project_boundary,
            )
        except FixedStepTraceInputError:
            raise FixedStepInputError() from None
        except FixedStepTraceInvariantError:
            raise FixedStepInvariantError() from None

        spine = trace.spine
        projections = trace.boundary_projections
        completions = tuple(
            projection.completion for projection in projections if projection.completion is not None
        )
        calls = tuple(call for completion in completions for call in completion.calls)
        deliveries = tuple(
            completion.delivery for completion in completions if completion.delivery is not None
        )
        outcomes = tuple(
            completion.outcome for completion in completions if completion.outcome is not None
        )
        try:
            return AlgorithmRunResult(
                schema_version="algorithm-run-result/v1",
                run_id=spine.run_id,
                trace_digest=spine.trace_digest,
                trace_event_count=len(items),
                normalized_draft_digests=spine.normalized_draft_digests,
                persisted_event_draft_digests=spine.persisted_event_draft_digests,
                trajectory_prefix=spine.trajectory_prefix,
                schedule=spine.schedule,
                windows=spine.windows,
                configuration=self._configuration,
                decisions=tuple(projection.decision for projection in projections),
                cycles=tuple(completion.cycle for completion in completions),
                cycle_requests=tuple(completion.request for completion in completions),
                executions=tuple(completion.execution for completion in completions),
                call_receipts=calls,
                deliveries=deliveries,
                outcomes=outcomes,
                model_token_usage=_model_token_usage_attestation(
                    self._configuration,
                    calls,
                ),
                projection_digests=trace.projection_digests,
                ledger_entry_count=len(trace.ledger),
                ledger_head=trace.ledger_head,
                rebuild_equivalent=trace.rebuild_equivalent,
            )
        except Exception:
            raise FixedStepInvariantError() from None


__all__ = [
    "FIXED_STEP_EVENT_INPUT_SCHEMA_VERSION",
    "FIXED_STEP_POLICY_VERSION",
    "FixedStepEventInput",
    "FixedStepExecutionError",
    "FixedStepInputError",
    "FixedStepInvariantError",
    "FixedStepRecoveryResult",
    "FixedStepRunner",
    "FixedStepRuntimeError",
]
