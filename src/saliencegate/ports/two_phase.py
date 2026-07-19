from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Protocol, Self, TypeAlias, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.domain import (
    MAX_MEMORY_DELTA_ITEMS,
    ClaimKind,
    CycleState,
    EvidenceSource,
    InterventionAction,
    InterventionDecision,
    MemoryDelta,
    MemoryIdAssignment,
    MemoryKind,
    PayloadDigest,
    ReasonCode,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.domain.records import (
    UUID4,
    ComponentIdentifier,
    Sha256Digest,
    UtcDatetime,
)
from saliencegate.intervention import (
    GroundingConfig,
    GroundingContext,
    GroundingReceipt,
    GroundingState,
    ProposalParseStatus,
    materialize_claim,
    resolve_grounding_configuration,
    verify_grounded_intervention,
)
from saliencegate.memory.materialize import (
    MaterializationFailureReason,
    MaterializedBankOperations,
    OperationMaterializationRequest,
    source_operations_digest,
)
from saliencegate.memory.proposals import (
    INTERVENTION_OUTPUT_SCHEMA_VERSION,
    MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
    BankOperationsProposal,
    DeleteMemory,
    InterventionSelectionOutput,
)
from saliencegate.ports.model_calls import (
    MAX_STRUCTURED_CALL_OUTPUT_BYTES,
    STRUCTURED_CALL_RESULT_SCHEMA_VERSION,
    StructuredCallBoundaryError,
    StructuredCallParseStatus,
    StructuredCallPhase,
    StructuredCallRequest,
    StructuredCallResult,
    StructuredCallStatus,
    StructuredCallUsage,
    StructuredPhaseOutput,
    validated_result_for_request,
    validated_structured_call_request,
)
from saliencegate.ports.repository import CycleReceipt
from saliencegate.prompts import (
    ActiveBankPromptView,
    BankViewKind,
    BuiltPrompt,
    build_active_bank_prompt_view,
)
from saliencegate.runtime.message_window import MessageWindow

TWO_PHASE_CYCLE_REQUEST_SCHEMA_VERSION: Literal["two-phase-cycle-request/v1"] = (
    "two-phase-cycle-request/v1"
)
TWO_PHASE_MODEL_PROFILE_SCHEMA_VERSION: Literal["two-phase-model-profile/v1"] = (
    "two-phase-model-profile/v1"
)
TWO_PHASE_CALL_POLICY_SCHEMA_VERSION: Literal["two-phase-call-policy/v1"] = (
    "two-phase-call-policy/v1"
)
CALL_RECEIPT_SCHEMA_VERSION: Literal["two-phase-call-receipt/v1"] = "two-phase-call-receipt/v1"
TWO_PHASE_USAGE_SCHEMA_VERSION: Literal["two-phase-usage/v1"] = "two-phase-usage/v1"
PHASE_ONE_CYCLE_RESULT_SCHEMA_VERSION: Literal["phase-one-cycle-result/v1"] = (
    "phase-one-cycle-result/v1"
)
TWO_PHASE_CYCLE_RESULT_SCHEMA_VERSION: Literal["two-phase-cycle-result/v1"] = (
    "two-phase-cycle-result/v1"
)
TWO_PHASE_CYCLE_FAILURE_SCHEMA_VERSION: Literal["two-phase-cycle-failure/v1"] = (
    "two-phase-cycle-failure/v1"
)

_MODEL_PROFILE_DIGEST_DOMAIN = "saliencegate:two-phase:model-profile:v1"
_CALL_POLICY_DIGEST_DOMAIN = "saliencegate:two-phase:call-policy:v1"
_CYCLE_REQUEST_DIGEST_DOMAIN = "saliencegate:two-phase:cycle-request:v1"
_CALL_RECEIPT_DIGEST_DOMAIN = "saliencegate:two-phase:call-receipt:v1"
_GROUNDING_STATE_DIGEST_DOMAIN = "saliencegate:two-phase:grounding-state:v1"
_PHASE_ONE_RESULT_DIGEST_DOMAIN = "saliencegate:two-phase:phase-one-cycle-result:v1"
_CYCLE_RESULT_DIGEST_DOMAIN = "saliencegate:two-phase:cycle-result:v1"
_CYCLE_FAILURE_DIGEST_DOMAIN = "saliencegate:two-phase:cycle-failure:v1"
_MAX_SIGNED_64 = (1 << 63) - 1
_MAX_VISIBLE_CALLS = 34

_CLAIM_KINDS_BY_MEMORY_KIND = {
    MemoryKind.KNOWLEDGE: frozenset((ClaimKind.REQUIREMENT, ClaimKind.ENVIRONMENT_FACT)),
    MemoryKind.PROCEDURAL: frozenset((ClaimKind.FAILED_ATTEMPT, ClaimKind.DIAGNOSIS)),
    MemoryKind.PRIVATE_STATUS: frozenset((ClaimKind.OPEN_SUBGOAL,)),
}
_INVALID_OPERATION_FAILURES = frozenset(
    (
        MaterializationFailureReason.INVALID_INPUT,
        MaterializationFailureReason.REFERENCE_MISSING,
        MaterializationFailureReason.REFERENCE_STALE,
        MaterializationFailureReason.REFERENCE_INACTIVE,
        MaterializationFailureReason.REFERENCE_EXPIRED,
        MaterializationFailureReason.REFERENCE_FUTURE,
        MaterializationFailureReason.REFERENCE_INVALID,
        MaterializationFailureReason.OPERATION_CONFLICT,
    )
)

NonNegativeSigned64 = Annotated[int, Field(ge=0, le=_MAX_SIGNED_64)]
AggregateNonNegative = Annotated[
    int,
    Field(ge=0, le=_MAX_SIGNED_64 * _MAX_VISIBLE_CALLS),
]


def _proposal_write_count(proposal: BankOperationsProposal) -> int:
    return sum(type(operation) is not DeleteMemory for operation in proposal.operations)


class TwoPhaseBoundaryError(ValueError):
    """A value-free rejection at the two-phase orchestration boundary."""

    def __init__(self, operation: str) -> None:
        super().__init__(f"{operation} failed two-phase boundary validation")


class _TwoPhaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def _profile_digest(values: Mapping[str, object]) -> str:
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": values["schema_version"],
                "profile_id": values["profile_id"],
                "model_id": values["model_id"],
                "prompt_bundle_id": values["prompt_bundle_id"],
                "prompt_bundle_digest": values["prompt_bundle_digest"],
            }
        ),
        domain=_MODEL_PROFILE_DIGEST_DOMAIN,
    )


class TwoPhaseModelProfile(_TwoPhaseModel):
    """The minimal model and reviewed-prompt identity used by the executor."""

    schema_version: Literal["two-phase-model-profile/v1"] = TWO_PHASE_MODEL_PROFILE_SCHEMA_VERSION
    profile_id: ComponentIdentifier
    model_id: ComponentIdentifier
    prompt_bundle_id: ComponentIdentifier
    prompt_bundle_digest: Sha256Digest
    profile_digest: Sha256Digest = Field(default_factory=_profile_digest)

    @model_validator(mode="after")
    def identity_digest_matches(self) -> Self:
        values = self.model_dump(mode="json", exclude={"profile_digest"})
        if self.profile_digest != _profile_digest(values):
            raise ValueError("two-phase model profile digest does not match")
        return self


def _policy_digest(values: Mapping[str, object]) -> str:
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": values["schema_version"],
                "max_model_calls": values["max_model_calls"],
                "max_schema_repairs": values["max_schema_repairs"],
                "client_retries": values["client_retries"],
                "max_provider_input_tokens": values["max_provider_input_tokens"],
                "max_provider_output_tokens": values["max_provider_output_tokens"],
                "max_total_latency_us": values["max_total_latency_us"],
                "max_call_latency_us": values["max_call_latency_us"],
            }
        ),
        domain=_CALL_POLICY_DIGEST_DOMAIN,
    )


class TwoPhaseCallPolicy(_TwoPhaseModel):
    """A finite visible-call envelope; clients may never retry invisibly."""

    schema_version: Literal["two-phase-call-policy/v1"] = TWO_PHASE_CALL_POLICY_SCHEMA_VERSION
    max_model_calls: Annotated[int, Field(ge=2, le=_MAX_VISIBLE_CALLS)]
    max_schema_repairs: Annotated[int, Field(ge=0, le=_MAX_VISIBLE_CALLS - 2)]
    client_retries: Literal[0]
    max_provider_input_tokens: NonNegativeSigned64
    max_provider_output_tokens: NonNegativeSigned64
    max_total_latency_us: NonNegativeSigned64
    max_call_latency_us: NonNegativeSigned64
    policy_digest: Sha256Digest = Field(default_factory=_policy_digest)

    @model_validator(mode="after")
    def bounds_and_digest_match(self) -> Self:
        if self.max_model_calls != 2 + self.max_schema_repairs:
            raise ValueError("two-phase call count must expose every repair")
        if self.max_call_latency_us > self.max_total_latency_us:
            raise ValueError("two-phase per-call latency exceeds total latency")
        values = self.model_dump(mode="json", exclude={"policy_digest"})
        if self.policy_digest != _policy_digest(values):
            raise ValueError("two-phase call policy digest does not match")
        return self


def _cycle_request_digest(values: Mapping[str, object]) -> str:
    created_at = values["created_at"]
    if isinstance(created_at, datetime):
        created_at = created_at.isoformat().replace("+00:00", "Z")
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": values["schema_version"],
                "cycle_receipt": values["cycle_receipt"],
                "window": values["window"],
                "current_bank": values["current_bank"],
                "grounding_state": values["grounding_state"],
                "delta_id": str(values["delta_id"]),
                "assigned_memory_ids": tuple(
                    str(item) for item in cast(tuple[object, ...], values["assigned_memory_ids"])
                ),
                "intervention_id": str(values["intervention_id"]),
                "created_at": created_at,
            }
        ),
        domain=_CYCLE_REQUEST_DIGEST_DOMAIN,
    )


class TwoPhaseCycleRequest(_TwoPhaseModel):
    """Repository-attested inputs and runtime-owned identities for one cycle.

    The caller is responsible for obtaining ``window``, ``current_bank``, events, and
    reminder history through verified repository projections. This repository-agnostic
    port revalidates their exact relationship but does not authenticate their origin.
    """

    schema_version: Literal["two-phase-cycle-request/v1"] = TWO_PHASE_CYCLE_REQUEST_SCHEMA_VERSION
    cycle_receipt: CycleReceipt = Field(repr=False)
    window: MessageWindow = Field(repr=False)
    current_bank: ActiveBankPromptView = Field(repr=False)
    grounding_state: GroundingState = Field(repr=False)
    delta_id: UUID4
    assigned_memory_ids: Annotated[
        tuple[UUID4, ...],
        Field(max_length=MAX_MEMORY_DELTA_ITEMS, repr=False),
    ]
    intervention_id: UUID4
    created_at: UtcDatetime
    request_digest: Sha256Digest = Field(default_factory=_cycle_request_digest)

    @model_validator(mode="after")
    def authoritative_views_and_identity_match(self) -> Self:
        cycle = self.cycle_receipt.cycle
        try:
            bank_records = tuple(item.to_memory_record() for item in self.current_bank.records)
        except Exception:
            raise ValueError("two-phase current bank failed exact validation") from None
        current_events = tuple(
            item
            for item in self.grounding_state.events
            if item.sequence == cycle.last_event_sequence
        )
        identity_values = (
            self.delta_id,
            self.intervention_id,
            *self.assigned_memory_ids,
        )
        invalid = (
            cycle.state is not CycleState.RUNNING
            or self.created_at < cycle.updated_at
            or self.window.run_id != cycle.run_id
            or self.window.boundary_event_sequence != cycle.last_event_sequence
            or self.window.boundary_ledger_position >= self.cycle_receipt.ledger_position
            or self.current_bank.kind is not BankViewKind.CURRENT
            or self.current_bank.run_id != cycle.run_id
            or self.current_bank.as_of != self.created_at
            or self.grounding_state.memories != bank_records
            or len(current_events) != 1
            or current_events[0].event_id != self.window.boundary_event_id
            or any(
                item.run_id != cycle.run_id
                or item.sequence > cycle.last_event_sequence
                or item.timestamp > self.created_at
                for item in self.grounding_state.events
            )
            or any(
                item.run_id != cycle.run_id or item.event_sequence >= cycle.last_event_sequence
                for item in self.grounding_state.reminder_history
            )
            or len(set(identity_values)) != len(identity_values)
        )
        if invalid:
            raise ValueError("two-phase cycle request views do not match")
        values = self.model_dump(mode="json", exclude={"request_digest"})
        if self.request_digest != _cycle_request_digest(values):
            raise ValueError("two-phase cycle request digest does not match")
        return self


def _call_receipt_digest(values: Mapping[str, object]) -> str:
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": values["schema_version"],
                "run_id": str(values["run_id"]),
                "cycle_id": values["cycle_id"],
                "model_call_index": values["model_call_index"],
                "phase": values["phase"],
                "attempt": values["attempt"],
                "model_id": values["model_id"],
                "prompt_template_id": values["prompt_template_id"],
                "prompt_template_digest": values["prompt_template_digest"],
                "prompt_digest": values["prompt_digest"],
                "request_payload_digest": values["request_payload_digest"],
                "window_digest": values["window_digest"],
                "bank_view_digest": values["bank_view_digest"],
                "grounding_state_digest": values["grounding_state_digest"],
                "request_digest": values["request_digest"],
                "status": values["status"],
                "parse_status": values["parse_status"],
                "completion_digest": values["completion_digest"],
                "completion_byte_count": values["completion_byte_count"],
                "usage": values["usage"],
                "call_digest": values["call_digest"],
            }
        ),
        domain=_CALL_RECEIPT_DIGEST_DOMAIN,
    )


def _grounding_state_digest(state: GroundingState) -> str:
    checked = GroundingState.model_validate_json(state.model_dump_json(warnings=False))
    return length_prefixed_sha256(
        canonical_json(checked),
        domain=_GROUNDING_STATE_DIGEST_DOMAIN,
    )


class CallReceipt(_TwoPhaseModel):
    """A content-bound call projection that excludes prompt and completion bodies."""

    schema_version: Literal["two-phase-call-receipt/v1"] = CALL_RECEIPT_SCHEMA_VERSION
    run_id: UUID4
    cycle_id: Sha256Digest
    model_call_index: NonNegativeSigned64
    phase: StructuredCallPhase
    attempt: NonNegativeSigned64
    model_id: ComponentIdentifier
    prompt_template_id: ComponentIdentifier
    prompt_template_digest: Sha256Digest
    prompt_digest: Sha256Digest
    request_payload_digest: Sha256Digest
    window_digest: Sha256Digest
    bank_view_digest: Sha256Digest
    grounding_state_digest: Sha256Digest | None
    request_digest: Sha256Digest
    status: StructuredCallStatus
    parse_status: StructuredCallParseStatus
    completion_digest: PayloadDigest | None = Field(repr=False)
    completion_byte_count: Annotated[
        int | None,
        Field(ge=0, le=MAX_STRUCTURED_CALL_OUTPUT_BYTES),
    ]
    usage: StructuredCallUsage
    call_digest: Sha256Digest
    receipt_digest: Sha256Digest = Field(default_factory=_call_receipt_digest)

    @classmethod
    def from_call(
        cls,
        prompt: BuiltPrompt,
        request: StructuredCallRequest,
        result: StructuredCallResult,
        *,
        grounding_state: GroundingState | None = None,
    ) -> CallReceipt:
        try:
            if type(prompt) is not BuiltPrompt:
                raise ValueError
            checked_prompt = BuiltPrompt.model_validate_json(prompt.model_dump_json(warnings=False))
            checked_request = validated_structured_call_request(request)
            checked_result = validated_result_for_request(checked_request, result)
            if (
                checked_request.phase is not checked_prompt.identity.phase
                or checked_request.prompt_template_id != checked_prompt.identity.template_id
                or checked_request.prompt_template_digest != checked_prompt.identity.template_digest
                or checked_request.response_schema_version
                != checked_prompt.identity.response_schema_version
                or canonical_json(checked_request.payload)
                != canonical_json(checked_prompt.request_payload.as_json_object())
                or (
                    checked_request.phase is StructuredCallPhase.MEMORY_EDIT
                    and grounding_state is not None
                )
                or (
                    checked_request.phase is StructuredCallPhase.INTERVENTION
                    and grounding_state is None
                )
            ):
                raise ValueError
            return cls(
                run_id=checked_request.run_id,
                cycle_id=checked_request.cycle_id,
                model_call_index=checked_request.model_call_index,
                phase=checked_request.phase,
                attempt=checked_request.attempt,
                model_id=checked_request.model_id,
                prompt_template_id=checked_request.prompt_template_id,
                prompt_template_digest=checked_request.prompt_template_digest,
                prompt_digest=checked_prompt.prompt_digest,
                request_payload_digest=checked_prompt.request_payload_digest,
                window_digest=checked_prompt.window_digest,
                bank_view_digest=checked_prompt.bank_view_digest,
                grounding_state_digest=(
                    _grounding_state_digest(grounding_state)
                    if grounding_state is not None
                    else None
                ),
                request_digest=checked_request.request_digest,
                status=checked_result.status,
                parse_status=checked_result.parse_status,
                completion_digest=checked_result.completion_digest,
                completion_byte_count=checked_result.completion_byte_count,
                usage=checked_result.usage,
                call_digest=checked_result.call_digest,
            )
        except (StructuredCallBoundaryError, ValueError):
            raise TwoPhaseBoundaryError("call receipt") from None
        except Exception:
            raise TwoPhaseBoundaryError("call receipt") from None

    @model_validator(mode="after")
    def call_state_and_digest_match(self) -> Self:
        has_completion = self.completion_digest is not None
        if self.attempt > self.model_call_index:
            raise ValueError("two-phase call receipt attempt exceeds call index")
        if self.phase is StructuredCallPhase.INTERVENTION and self.model_call_index == 0:
            raise ValueError("two-phase intervention receipt requires a positive call index")
        if (self.phase is StructuredCallPhase.INTERVENTION) != (
            self.grounding_state_digest is not None
        ):
            raise ValueError("two-phase grounding state digest does not match phase")
        if (self.completion_digest is None) != (self.completion_byte_count is None):
            raise ValueError("two-phase call receipt completion fields do not match")
        if (self.status is StructuredCallStatus.COMPLETED) != has_completion:
            raise ValueError("two-phase call status does not match completion fields")
        if (
            self.status is StructuredCallStatus.COMPLETED
            and self.parse_status is StructuredCallParseStatus.NOT_ATTEMPTED
        ) or (
            self.status is not StructuredCallStatus.COMPLETED
            and self.parse_status is not StructuredCallParseStatus.NOT_ATTEMPTED
        ):
            raise ValueError("two-phase call status does not match parse status")
        values = self.model_dump(mode="json", exclude={"receipt_digest"})
        if self.receipt_digest != _call_receipt_digest(values):
            raise ValueError("two-phase call receipt digest does not match")
        return self


class TwoPhaseUsage(_TwoPhaseModel):
    """Exact provider and canonical totals without crossing provenance domains."""

    schema_version: Literal["two-phase-usage/v1"] = TWO_PHASE_USAGE_SCHEMA_VERSION
    model_calls: Annotated[int, Field(ge=0, le=_MAX_VISIBLE_CALLS)]
    provider_input_tokens: AggregateNonNegative | None
    provider_output_tokens: AggregateNonNegative | None
    canonical_token_equivalents: AggregateNonNegative | None = None
    latency_us: AggregateNonNegative
    schema_repairs: Annotated[int, Field(ge=0, le=_MAX_VISIBLE_CALLS - 2)]
    canonical_input_tokens: AggregateNonNegative | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    canonical_output_tokens: AggregateNonNegative | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @classmethod
    def from_receipts(cls, receipts: tuple[CallReceipt, ...]) -> TwoPhaseUsage:
        if (
            type(receipts) is not tuple
            or len(receipts) > _MAX_VISIBLE_CALLS
            or any(type(item) is not CallReceipt for item in receipts)
        ):
            raise TwoPhaseBoundaryError("call usage")
        try:
            checked = tuple(
                CallReceipt.model_validate_json(item.model_dump_json(warnings=False))
                for item in receipts
            )
            counts_known = all(item.usage.provider_input_tokens is not None for item in checked)
            provider_input_tokens = (
                sum(cast(int, item.usage.provider_input_tokens) for item in checked)
                if counts_known
                else None
            )
            provider_output_tokens = (
                sum(cast(int, item.usage.provider_output_tokens) for item in checked)
                if counts_known
                else None
            )
            canonical_input_known = all(
                item.usage.canonical_input_tokens is not None for item in checked
            )
            canonical_output_known = all(
                item.usage.canonical_output_tokens is not None for item in checked
            )
            canonical_input_tokens = (
                sum(cast(int, item.usage.canonical_input_tokens) for item in checked)
                if canonical_input_known
                else None
            )
            canonical_output_tokens = (
                sum(cast(int, item.usage.canonical_output_tokens) for item in checked)
                if canonical_output_known
                else None
            )
            return cls(
                model_calls=len(checked),
                provider_input_tokens=provider_input_tokens,
                provider_output_tokens=provider_output_tokens,
                canonical_token_equivalents=(
                    canonical_input_tokens + canonical_output_tokens
                    if canonical_input_tokens is not None and canonical_output_tokens is not None
                    else None
                ),
                latency_us=sum(item.usage.latency_us for item in checked),
                schema_repairs=sum(item.attempt > 0 for item in checked),
                canonical_input_tokens=canonical_input_tokens,
                canonical_output_tokens=canonical_output_tokens,
            )
        except Exception:
            raise TwoPhaseBoundaryError("call usage") from None

    @model_validator(mode="after")
    def optional_provider_totals_match(self) -> Self:
        if (self.provider_input_tokens is None) != (self.provider_output_tokens is None):
            raise ValueError("two-phase provider totals must be both known or both unavailable")
        canonical_complete = (
            self.canonical_input_tokens is not None and self.canonical_output_tokens is not None
        )
        if canonical_complete != (self.canonical_token_equivalents is not None):
            raise ValueError("two-phase canonical total does not match component availability")
        if (
            self.canonical_token_equivalents is not None
            and self.canonical_input_tokens is not None
            and self.canonical_output_tokens is not None
            and self.canonical_token_equivalents
            != self.canonical_input_tokens + self.canonical_output_tokens
        ):
            raise ValueError("two-phase canonical total does not match its components")
        if self.schema_repairs > self.model_calls:
            raise ValueError("two-phase repair count exceeds call count")
        return self


def call_policy_accepts_receipts(
    policy: TwoPhaseCallPolicy,
    receipts: tuple[CallReceipt, ...],
) -> bool:
    """Check exact totals and every known token lower bound against one policy."""

    if type(policy) is not TwoPhaseCallPolicy:
        raise TwoPhaseBoundaryError("call policy accounting")
    try:
        checked_policy = TwoPhaseCallPolicy.model_validate_json(
            policy.model_dump_json(warnings=False)
        )
        usage = TwoPhaseUsage.from_receipts(receipts)
        known_input_tokens = sum(item.usage.provider_input_tokens or 0 for item in receipts)
        known_output_tokens = sum(item.usage.provider_output_tokens or 0 for item in receipts)
        known_canonical_tokens = sum(
            (item.usage.canonical_input_tokens or 0) + (item.usage.canonical_output_tokens or 0)
            for item in receipts
        )
        return not (
            usage.model_calls > checked_policy.max_model_calls
            or usage.schema_repairs > checked_policy.max_schema_repairs
            or usage.latency_us > checked_policy.max_total_latency_us
            or known_input_tokens > checked_policy.max_provider_input_tokens
            or known_output_tokens > checked_policy.max_provider_output_tokens
            or known_canonical_tokens
            > checked_policy.max_provider_input_tokens + checked_policy.max_provider_output_tokens
            or any(item.usage.latency_us > checked_policy.max_call_latency_us for item in receipts)
        )
    except TwoPhaseBoundaryError:
        raise
    except Exception:
        raise TwoPhaseBoundaryError("call policy accounting") from None


class TwoPhaseFailureReason(StrEnum):
    MODEL_ERROR = "model_error"
    MODEL_TIMEOUT = "model_timeout"
    SCHEMA_INVALID = "schema_invalid"
    INVALID_OPERATION = "invalid_operation"
    OPERATION_OVERFLOW = "operation_overflow"
    MATERIALIZATION_REJECTED = "materialization_rejected"
    REPAIR_EXHAUSTED = "repair_exhausted"
    CALL_POLICY_EXCEEDED = "call_policy_exceeded"
    CALL_CONTRACT_INVALID = "call_contract_invalid"


_GROUNDING_PARSE_STATUS = {
    StructuredCallParseStatus.SCHEMA_INVALID: ProposalParseStatus.SCHEMA_INVALID,
    StructuredCallParseStatus.EMPTY_REMINDER: ProposalParseStatus.EMPTY_REMINDER,
    StructuredCallParseStatus.CLAIM_OVER_LIMIT: ProposalParseStatus.CLAIM_OVER_LIMIT,
}
_GROUNDING_REJECTION_REASON = {
    ProposalParseStatus.SCHEMA_INVALID: ReasonCode.SCHEMA_INVALID,
    ProposalParseStatus.EMPTY_REMINDER: ReasonCode.NO_GROUNDED_CLAIMS,
    ProposalParseStatus.CLAIM_OVER_LIMIT: ReasonCode.CLAIM_OVER_LIMIT,
}


def _receipts_are_ordered(receipts: tuple[CallReceipt, ...]) -> bool:
    if tuple(item.model_call_index for item in receipts) != tuple(range(len(receipts))):
        return False
    if any(
        len({getattr(item, field_name) for item in receipts}) != len(receipts)
        for field_name in ("request_digest", "call_digest", "receipt_digest")
    ):
        return False
    expected_attempt = {
        StructuredCallPhase.MEMORY_EDIT: 0,
        StructuredCallPhase.INTERVENTION: 0,
    }
    intervention_started = False
    phase_bindings: dict[
        StructuredCallPhase,
        tuple[str, str, str, str, str, str, str | None],
    ] = {}
    for item in receipts:
        if item.phase is StructuredCallPhase.INTERVENTION:
            intervention_started = True
        elif intervention_started:
            return False
        if item.attempt != expected_attempt[item.phase]:
            return False
        expected_attempt[item.phase] += 1
        binding = (
            item.prompt_template_id,
            item.prompt_template_digest,
            item.prompt_digest,
            item.request_payload_digest,
            item.window_digest,
            item.bank_view_digest,
            item.grounding_state_digest,
        )
        prior = phase_bindings.setdefault(item.phase, binding)
        if binding != prior:
            return False
    return True


def _is_completed_rejection(receipt: CallReceipt) -> bool:
    allowed = (
        (StructuredCallParseStatus.SCHEMA_INVALID,)
        if receipt.phase is StructuredCallPhase.MEMORY_EDIT
        else (
            StructuredCallParseStatus.SCHEMA_INVALID,
            StructuredCallParseStatus.EMPTY_REMINDER,
            StructuredCallParseStatus.CLAIM_OVER_LIMIT,
        )
    )
    return receipt.status is StructuredCallStatus.COMPLETED and receipt.parse_status in allowed


def _call_output_matches_receipt(
    output: StructuredPhaseOutput | None,
    receipt: CallReceipt,
) -> bool:
    try:
        checked = StructuredCallResult(
            schema_version=STRUCTURED_CALL_RESULT_SCHEMA_VERSION,
            request_digest=receipt.request_digest,
            model_call_index=receipt.model_call_index,
            phase=receipt.phase,
            attempt=receipt.attempt,
            response_schema_version=(
                MEMORY_EDIT_OUTPUT_SCHEMA_VERSION
                if receipt.phase is StructuredCallPhase.MEMORY_EDIT
                else INTERVENTION_OUTPUT_SCHEMA_VERSION
            ),
            status=receipt.status,
            parse_status=receipt.parse_status,
            output=output,
            completion_digest=receipt.completion_digest,
            completion_byte_count=receipt.completion_byte_count,
            usage=receipt.usage,
            call_digest=receipt.call_digest,
        )
    except Exception:
        return False
    return checked.call_digest == receipt.call_digest


def _result_digest(values: Mapping[str, object]) -> str:
    material = {key: value for key, value in values.items() if key != "result_digest"}
    material["run_id"] = str(values["run_id"])
    return length_prefixed_sha256(
        canonical_json(material),
        domain=_CYCLE_RESULT_DIGEST_DOMAIN,
    )


def _phase_one_result_digest(values: Mapping[str, object]) -> str:
    material = {key: value for key, value in values.items() if key != "result_digest"}
    material["run_id"] = str(values["run_id"])
    return length_prefixed_sha256(
        canonical_json(material),
        domain=_PHASE_ONE_RESULT_DIGEST_DOMAIN,
    )


class PhaseOneCycleResult(_TwoPhaseModel):
    """A detached Phase 1 bank update with exact visible-call accounting."""

    outcome: Literal["phase_one_completed"] = "phase_one_completed"
    schema_version: Literal["phase-one-cycle-result/v1"] = PHASE_ONE_CYCLE_RESULT_SCHEMA_VERSION
    request_digest: Sha256Digest
    run_id: UUID4
    cycle_id: Sha256Digest
    window_digest: Sha256Digest
    current_bank_view_digest: Sha256Digest
    current_bank_source_projection_digest: PayloadDigest
    candidate_bank_view_digest: Sha256Digest
    prompt_bundle_digest: Sha256Digest
    model_id: ComponentIdentifier
    model_profile_digest: Sha256Digest
    model_profile: TwoPhaseModelProfile = Field(repr=False)
    call_policy_digest: Sha256Digest
    call_policy: TwoPhaseCallPolicy = Field(repr=False)
    materialization: MaterializedBankOperations = Field(repr=False)
    memory_edit_output: BankOperationsProposal = Field(repr=False)
    call_receipts: Annotated[
        tuple[CallReceipt, ...],
        Field(min_length=1, max_length=_MAX_VISIBLE_CALLS),
    ]
    usage: TwoPhaseUsage
    result_digest: Sha256Digest = Field(default_factory=_phase_one_result_digest)

    @property
    def validated_delta(self) -> MemoryDelta:
        return MemoryDelta.model_validate_json(
            self.materialization.delta.model_dump_json(warnings=False)
        )

    @property
    def memory_id_assignments(self) -> tuple[MemoryIdAssignment, ...]:
        return tuple(
            MemoryIdAssignment.model_validate_json(item.model_dump_json(warnings=False))
            for item in self.materialization.memory_id_assignments
        )

    @model_validator(mode="after")
    def successful_phase_is_exact(self) -> Self:
        receipts = self.call_receipts
        try:
            materialization = MaterializedBankOperations.model_validate_json(
                self.materialization.model_dump_json(warnings=False)
            )
            candidate = build_active_bank_prompt_view(
                kind=BankViewKind.CANDIDATE_POST_DELTA,
                run_id=self.run_id,
                as_of=materialization.delta.created_at,
                source_projection_digest=materialization.preview_projection_digest,
                records=materialization.active_bank,
            )
            usage = TwoPhaseUsage.from_receipts(receipts)
            policy_accepted = call_policy_accepts_receipts(self.call_policy, receipts)
        except Exception:
            raise ValueError("phase-one cycle result components do not match") from None
        final = receipts[-1]
        invalid = (
            not _receipts_are_ordered(receipts)
            or any(item.phase is not StructuredCallPhase.MEMORY_EDIT for item in receipts)
            or final.status is not StructuredCallStatus.COMPLETED
            or final.parse_status is not StructuredCallParseStatus.VALID
            or any(not _is_completed_rejection(item) for item in receipts[:-1])
            or any(
                item.run_id != self.run_id
                or item.cycle_id != self.cycle_id
                or item.model_id != self.model_id
                or item.window_digest != self.window_digest
                or item.bank_view_digest != self.current_bank_view_digest
                or item.grounding_state_digest is not None
                for item in receipts
            )
            or not _call_output_matches_receipt(self.memory_edit_output, final)
            or materialization.source_operations_digest
            != source_operations_digest(self.memory_edit_output)
            or materialization.source_projection_digest
            != self.current_bank_source_projection_digest
            or materialization.run_id != self.run_id
            or materialization.source_cycle_id != self.cycle_id
            or candidate.view_digest != self.candidate_bank_view_digest
            or self.model_profile.model_id != self.model_id
            or self.model_profile.profile_digest != self.model_profile_digest
            or self.model_profile.prompt_bundle_digest != self.prompt_bundle_digest
            or self.call_policy.policy_digest != self.call_policy_digest
            or not policy_accepted
            or usage != self.usage
        )
        if invalid:
            raise ValueError("phase-one cycle result components do not match")
        values = self.model_dump(mode="json", exclude={"result_digest"})
        if self.result_digest != _phase_one_result_digest(values):
            raise ValueError("phase-one cycle result digest does not match")
        return self


class TwoPhaseCycleResult(_TwoPhaseModel):
    """A committable visible-call result; the executor itself performs no write."""

    outcome: Literal["completed"] = "completed"
    schema_version: Literal["two-phase-cycle-result/v1"] = TWO_PHASE_CYCLE_RESULT_SCHEMA_VERSION
    request_digest: Sha256Digest
    run_id: UUID4
    cycle_id: Sha256Digest
    window_digest: Sha256Digest
    current_bank_view_digest: Sha256Digest
    candidate_bank_view_digest: Sha256Digest
    prompt_bundle_digest: Sha256Digest
    model_id: ComponentIdentifier
    model_profile_digest: Sha256Digest
    call_policy_digest: Sha256Digest
    call_policy: TwoPhaseCallPolicy = Field(repr=False)
    materialization: MaterializedBankOperations = Field(repr=False)
    memory_edit_output: BankOperationsProposal = Field(repr=False)
    intervention_output: InterventionSelectionOutput | None = Field(repr=False)
    intervention: InterventionDecision = Field(repr=False)
    grounding_receipt: GroundingReceipt = Field(repr=False)
    grounding_state: GroundingState = Field(repr=False)
    call_receipts: Annotated[
        tuple[CallReceipt, ...],
        Field(min_length=2, max_length=_MAX_VISIBLE_CALLS),
    ]
    usage: TwoPhaseUsage
    result_digest: Sha256Digest = Field(default_factory=_result_digest)

    @property
    def validated_delta(self) -> MemoryDelta:
        return MemoryDelta.model_validate_json(
            self.materialization.delta.model_dump_json(warnings=False)
        )

    @property
    def memory_id_assignments(self) -> tuple[MemoryIdAssignment, ...]:
        return tuple(
            MemoryIdAssignment.model_validate_json(item.model_dump_json(warnings=False))
            for item in self.materialization.memory_id_assignments
        )

    @model_validator(mode="after")
    def successful_cycle_is_exact(self) -> Self:
        receipts = self.call_receipts
        memory_receipts = tuple(
            item for item in receipts if item.phase is StructuredCallPhase.MEMORY_EDIT
        )
        intervention_receipts = tuple(
            item for item in receipts if item.phase is StructuredCallPhase.INTERVENTION
        )
        try:
            memory_output = self.memory_edit_output
            intervention_output = self.intervention_output
            candidate = build_active_bank_prompt_view(
                kind=BankViewKind.CANDIDATE_POST_DELTA,
                run_id=self.run_id,
                as_of=self.materialization.delta.created_at,
                source_projection_digest=self.materialization.preview_projection_digest,
                records=self.materialization.active_bank,
            )
            decision_receipt = GroundingReceipt.model_validate_json(
                canonical_json(self.intervention.grounding_receipt)
            )
            final_intervention_receipt = intervention_receipts[-1]
            grounding_context = GroundingContext(
                schema_version="1.0",
                intervention_id=self.intervention.intervention_id,
                run_id=self.run_id,
                cycle_id=self.cycle_id,
                current_event_sequence=self.materialization.source_last_event_sequence,
                created_at=self.materialization.delta.created_at,
                requested_delivery_target=self.grounding_receipt.requested_delivery_target,
                model_call_index=final_intervention_receipt.model_call_index,
                model_call_digest=final_intervention_receipt.call_digest,
            )
            grounding_config = GroundingConfig.model_validate_json(
                canonical_json(self.intervention.grounding_configuration)
            )
            resolved_grounding = resolve_grounding_configuration(grounding_config)
            verify_grounded_intervention(
                self.intervention,
                context=grounding_context,
                state=self.grounding_state,
                expected_configuration=resolved_grounding,
            )
            exact_grounding_state_digest = _grounding_state_digest(self.grounding_state)
            records = {item.memory_id: item for item in self.materialization.active_bank}
            grounded_claims = []
            claims_have_candidate_shape = True
            claims_are_exact = True
            candidate_failure: ReasonCode | None = None
            for proposed in self.grounding_receipt.claims:
                reference = proposed.evidence
                record = records.get(reference.source_id)
                if (
                    reference.source is not EvidenceSource.MEMORY
                    or reference.field_path != "/content"
                    or reference.span is not None
                ):
                    claims_have_candidate_shape = False
                    claims_are_exact = False
                    continue
                if record is None:
                    candidate_failure = ReasonCode.CITATION_MISSING
                    claims_are_exact = False
                    continue
                if (
                    reference.revision is None
                    or reference.revision != record.revision
                    or proposed.kind not in _CLAIM_KINDS_BY_MEMORY_KIND[record.kind]
                ):
                    if candidate_failure is None:
                        candidate_failure = ReasonCode.INVALID_PROVENANCE
                    claims_are_exact = False
                    continue
                grounded_claims.append(materialize_claim(proposed, source_text=record.content))
            usage = TwoPhaseUsage.from_receipts(receipts)
            policy_accepted = call_policy_accepts_receipts(
                self.call_policy,
                receipts,
            )
        except Exception:
            raise ValueError("two-phase cycle result components do not match") from None
        final_parse = intervention_receipts[-1].parse_status if intervention_receipts else None
        selection_has_candidate_shape = type(
            intervention_output
        ) is InterventionSelectionOutput and all(
            claim.evidence.source is EvidenceSource.MEMORY
            and claim.evidence.field_path == "/content"
            and claim.evidence.span is None
            for claim in intervention_output.claims
        )
        expected_grounding_parse = (
            (
                ProposalParseStatus.VALID
                if selection_has_candidate_shape
                else ProposalParseStatus.SCHEMA_INVALID
            )
            if final_parse is StructuredCallParseStatus.VALID
            else _GROUNDING_PARSE_STATUS.get(final_parse)
            if final_parse is not None
            else None
        )
        grounding_parse_matches = (
            expected_grounding_parse is not None
            and self.grounding_receipt.parse_status is expected_grounding_parse
            and (
                expected_grounding_parse is not ProposalParseStatus.VALID
                or (
                    type(intervention_output) is InterventionSelectionOutput
                    and self.grounding_receipt.proposal_action is intervention_output.action
                    and self.grounding_receipt.claims == intervention_output.claims
                    and self.grounding_receipt.confidence == intervention_output.confidence
                )
            )
        )
        invalid = (
            not memory_receipts
            or not intervention_receipts
            or not _receipts_are_ordered(receipts)
            or memory_receipts[-1].parse_status is not StructuredCallParseStatus.VALID
            or any(not _is_completed_rejection(item) for item in memory_receipts[:-1])
            or any(not _is_completed_rejection(item) for item in intervention_receipts[:-1])
            or any(
                item.status is not StructuredCallStatus.COMPLETED
                or item.run_id != self.run_id
                or item.cycle_id != self.cycle_id
                or item.model_id != self.model_id
                or item.window_digest != self.window_digest
                for item in receipts
            )
            or any(
                item.bank_view_digest != self.current_bank_view_digest for item in memory_receipts
            )
            or any(
                item.bank_view_digest != self.candidate_bank_view_digest
                for item in intervention_receipts
            )
            or self.grounding_state.memories != self.materialization.active_bank
            or any(
                item.grounding_state_digest != exact_grounding_state_digest
                for item in intervention_receipts
            )
            or not _call_output_matches_receipt(
                memory_output,
                memory_receipts[-1],
            )
            or not _call_output_matches_receipt(
                intervention_output,
                intervention_receipts[-1],
            )
            or self.materialization.source_operations_digest
            != source_operations_digest(memory_output)
            or self.call_policy.policy_digest != self.call_policy_digest
            or not policy_accepted
            or (
                final_parse is not StructuredCallParseStatus.VALID
                and usage.schema_repairs != self.call_policy.max_schema_repairs
            )
            or candidate.view_digest != self.candidate_bank_view_digest
            or self.materialization.run_id != self.run_id
            or self.materialization.source_cycle_id != self.cycle_id
            or self.intervention.run_id != self.run_id
            or self.intervention.cycle_id != self.cycle_id
            or self.intervention.created_at != self.materialization.delta.created_at
            or self.grounding_receipt != decision_receipt
            or not grounding_parse_matches
            or self.grounding_receipt.model_call_index != receipts[-1].model_call_index
            or self.grounding_receipt.model_call_digest != receipts[-1].call_digest
            or (
                self.grounding_receipt.parse_status is ProposalParseStatus.VALID
                and not claims_have_candidate_shape
            )
            or (
                self.grounding_receipt.parse_status is ProposalParseStatus.VALID
                and self.grounding_receipt.proposal_action is InterventionAction.SILENCE
                and self.intervention.reason_code is not ReasonCode.SILENCE_SELECTED
            )
            or (
                self.grounding_receipt.parse_status is ProposalParseStatus.VALID
                and self.grounding_receipt.proposal_action is InterventionAction.REMIND
                and not claims_are_exact
                and (
                    candidate_failure is None
                    or self.intervention.action is not InterventionAction.SILENCE
                    or self.intervention.reason_code is not candidate_failure
                )
            )
            or (
                self.grounding_receipt.parse_status is not ProposalParseStatus.VALID
                and (
                    self.intervention.action is not InterventionAction.SILENCE
                    or self.intervention.reason_code
                    is not _GROUNDING_REJECTION_REASON[self.grounding_receipt.parse_status]
                )
            )
            or (
                self.intervention.action is InterventionAction.REMIND
                and (
                    self.grounding_receipt.parse_status is not ProposalParseStatus.VALID
                    or self.grounding_receipt.proposal_action is not InterventionAction.REMIND
                    or not claims_are_exact
                    or self.intervention.claims != tuple(grounded_claims)
                    or self.intervention.cited_event_ids
                    or self.intervention.delivery_target
                    is not self.grounding_receipt.requested_delivery_target
                )
            )
            or usage != self.usage
        )
        if invalid:
            raise ValueError("two-phase cycle result components do not match")
        values = self.model_dump(mode="json", exclude={"result_digest"})
        if self.result_digest != _result_digest(values):
            raise ValueError("two-phase cycle result digest does not match")
        return self


def _failure_digest(values: Mapping[str, object]) -> str:
    material = {key: value for key, value in values.items() if key != "failure_digest"}
    material["run_id"] = str(values["run_id"])
    return length_prefixed_sha256(
        canonical_json(material),
        domain=_CYCLE_FAILURE_DIGEST_DOMAIN,
    )


class TwoPhaseCycleFailure(_TwoPhaseModel):
    """A known-cost, non-committable failure with no delta or intervention."""

    outcome: Literal["failed"] = "failed"
    schema_version: Literal["two-phase-cycle-failure/v1"] = TWO_PHASE_CYCLE_FAILURE_SCHEMA_VERSION
    request_digest: Sha256Digest
    run_id: UUID4
    cycle_id: Sha256Digest
    window_digest: Sha256Digest
    prompt_bundle_digest: Sha256Digest
    model_id: ComponentIdentifier
    model_profile_digest: Sha256Digest
    call_policy_digest: Sha256Digest
    call_policy: TwoPhaseCallPolicy = Field(repr=False)
    failed_phase: StructuredCallPhase
    reason: TwoPhaseFailureReason
    assigned_memory_id_capacity: Annotated[
        int,
        Field(ge=0, le=MAX_MEMORY_DELTA_ITEMS),
    ]
    memory_edit_output: BankOperationsProposal | None = Field(repr=False)
    intervention_output: InterventionSelectionOutput | None = Field(repr=False)
    materialization_failure_reason: MaterializationFailureReason | None
    call_receipts: Annotated[
        tuple[CallReceipt, ...],
        Field(min_length=1, max_length=_MAX_VISIBLE_CALLS),
    ]
    usage: TwoPhaseUsage
    cost_certainty: Literal["known"] = "known"
    failure_digest: Sha256Digest = Field(default_factory=_failure_digest)

    @model_validator(mode="after")
    def known_receipts_and_digest_match(self) -> Self:
        receipts = self.call_receipts
        memory_receipts = tuple(
            item for item in receipts if item.phase is StructuredCallPhase.MEMORY_EDIT
        )
        intervention_receipts = tuple(
            item for item in receipts if item.phase is StructuredCallPhase.INTERVENTION
        )
        try:
            usage = TwoPhaseUsage.from_receipts(receipts)
            policy_accepted = call_policy_accepts_receipts(
                self.call_policy,
                receipts,
            )
        except TwoPhaseBoundaryError:
            raise ValueError("two-phase failure usage failed validation") from None
        final = receipts[-1]
        memory_output_matches = bool(memory_receipts) and _call_output_matches_receipt(
            self.memory_edit_output,
            memory_receipts[-1],
        )
        intervention_output_matches = (
            _call_output_matches_receipt(
                self.intervention_output,
                intervention_receipts[-1],
            )
            if intervention_receipts
            else self.intervention_output is None
        )
        write_count = (
            _proposal_write_count(self.memory_edit_output)
            if self.memory_edit_output is not None
            else None
        )
        materialization_detail = self.materialization_failure_reason
        reason_matches = (
            (
                self.reason is TwoPhaseFailureReason.MODEL_ERROR
                and final.status is StructuredCallStatus.MODEL_ERROR
                and materialization_detail is None
            )
            or (
                self.reason is TwoPhaseFailureReason.MODEL_TIMEOUT
                and final.status is StructuredCallStatus.MODEL_TIMEOUT
                and materialization_detail is None
            )
            or (
                self.reason is TwoPhaseFailureReason.SCHEMA_INVALID
                and self.failed_phase is StructuredCallPhase.MEMORY_EDIT
                and final.attempt == 0
                and _is_completed_rejection(final)
                and policy_accepted
                and self.call_policy.max_schema_repairs == 0
                and materialization_detail is None
            )
            or (
                self.reason is TwoPhaseFailureReason.REPAIR_EXHAUSTED
                and self.failed_phase is StructuredCallPhase.MEMORY_EDIT
                and final.attempt > 0
                and _is_completed_rejection(final)
                and policy_accepted
                and usage.schema_repairs == self.call_policy.max_schema_repairs
                and materialization_detail is None
            )
            or (
                self.reason is TwoPhaseFailureReason.INVALID_OPERATION
                and self.failed_phase is StructuredCallPhase.MEMORY_EDIT
                and final.status is StructuredCallStatus.COMPLETED
                and final.parse_status is StructuredCallParseStatus.VALID
                and policy_accepted
                and write_count is not None
                and write_count <= self.assigned_memory_id_capacity
                and materialization_detail in _INVALID_OPERATION_FAILURES
            )
            or (
                self.reason is TwoPhaseFailureReason.OPERATION_OVERFLOW
                and self.failed_phase is StructuredCallPhase.MEMORY_EDIT
                and final.status is StructuredCallStatus.COMPLETED
                and final.parse_status is StructuredCallParseStatus.VALID
                and policy_accepted
                and write_count is not None
                and write_count > self.assigned_memory_id_capacity
                and materialization_detail is None
            )
            or (
                self.reason is TwoPhaseFailureReason.MATERIALIZATION_REJECTED
                and self.failed_phase is StructuredCallPhase.MEMORY_EDIT
                and final.status is StructuredCallStatus.COMPLETED
                and final.parse_status is StructuredCallParseStatus.VALID
                and policy_accepted
                and write_count is not None
                and write_count <= self.assigned_memory_id_capacity
                and materialization_detail not in _INVALID_OPERATION_FAILURES
            )
            or (
                self.reason is TwoPhaseFailureReason.CALL_POLICY_EXCEEDED
                and final.status is StructuredCallStatus.COMPLETED
                and not policy_accepted
                and materialization_detail is None
            )
            or (
                self.reason is TwoPhaseFailureReason.CALL_CONTRACT_INVALID
                and self.failed_phase is StructuredCallPhase.INTERVENTION
                and final.status is StructuredCallStatus.COMPLETED
                and final.parse_status is StructuredCallParseStatus.VALID
                and policy_accepted
                and materialization_detail is None
            )
        )
        if (
            not memory_receipts
            or not _receipts_are_ordered(receipts)
            or any(
                item.run_id != self.run_id
                or item.cycle_id != self.cycle_id
                or item.model_id != self.model_id
                or item.window_digest != self.window_digest
                for item in receipts
            )
            or final.phase is not self.failed_phase
            or not reason_matches
            or not memory_output_matches
            or not intervention_output_matches
            or self.call_policy.policy_digest != self.call_policy_digest
            or any(not _is_completed_rejection(item) for item in memory_receipts[:-1])
            or (
                intervention_receipts
                and (
                    memory_receipts[-1].status is not StructuredCallStatus.COMPLETED
                    or memory_receipts[-1].parse_status is not StructuredCallParseStatus.VALID
                    or any(not _is_completed_rejection(item) for item in intervention_receipts[:-1])
                )
            )
            or (
                self.failed_phase is StructuredCallPhase.MEMORY_EDIT and bool(intervention_receipts)
            )
            or (self.failed_phase is StructuredCallPhase.INTERVENTION and not intervention_receipts)
            or usage != self.usage
        ):
            raise ValueError("two-phase failure receipts do not match")
        values = self.model_dump(mode="json", exclude={"failure_digest"})
        if self.failure_digest != _failure_digest(values):
            raise ValueError("two-phase cycle failure digest does not match")
        return self


PhaseOneCycleOutcome: TypeAlias = Annotated[
    PhaseOneCycleResult | TwoPhaseCycleFailure,
    Field(discriminator="outcome"),
]


TwoPhaseCycleOutcome: TypeAlias = Annotated[
    TwoPhaseCycleResult | TwoPhaseCycleFailure,
    Field(discriminator="outcome"),
]


@runtime_checkable
class OperationMaterializer(Protocol):
    async def materialize(
        self,
        request: OperationMaterializationRequest,
    ) -> MaterializedBankOperations: ...


@runtime_checkable
class PhaseOneCycleExecutor(Protocol):
    async def execute_phase_one(
        self,
        request: TwoPhaseCycleRequest,
    ) -> PhaseOneCycleOutcome: ...


@runtime_checkable
class TwoPhaseCycleExecutor(Protocol):
    async def execute(self, request: TwoPhaseCycleRequest) -> TwoPhaseCycleOutcome: ...


def validated_two_phase_cycle_request(value: object) -> TwoPhaseCycleRequest:
    if type(value) is not TwoPhaseCycleRequest:
        raise TwoPhaseBoundaryError("cycle request")
    try:
        return TwoPhaseCycleRequest.model_validate_json(value.model_dump_json(warnings=False))
    except Exception:
        raise TwoPhaseBoundaryError("cycle request") from None


__all__ = [
    "CALL_RECEIPT_SCHEMA_VERSION",
    "PHASE_ONE_CYCLE_RESULT_SCHEMA_VERSION",
    "TWO_PHASE_CALL_POLICY_SCHEMA_VERSION",
    "TWO_PHASE_CYCLE_FAILURE_SCHEMA_VERSION",
    "TWO_PHASE_CYCLE_REQUEST_SCHEMA_VERSION",
    "TWO_PHASE_CYCLE_RESULT_SCHEMA_VERSION",
    "TWO_PHASE_MODEL_PROFILE_SCHEMA_VERSION",
    "TWO_PHASE_USAGE_SCHEMA_VERSION",
    "CallReceipt",
    "OperationMaterializer",
    "PhaseOneCycleExecutor",
    "PhaseOneCycleOutcome",
    "PhaseOneCycleResult",
    "TwoPhaseBoundaryError",
    "TwoPhaseCallPolicy",
    "TwoPhaseCycleExecutor",
    "TwoPhaseCycleFailure",
    "TwoPhaseCycleOutcome",
    "TwoPhaseCycleRequest",
    "TwoPhaseCycleResult",
    "TwoPhaseFailureReason",
    "TwoPhaseModelProfile",
    "TwoPhaseUsage",
    "call_policy_accepts_receipts",
    "validated_two_phase_cycle_request",
]
