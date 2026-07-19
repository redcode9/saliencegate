from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.domain import (
    MAX_MEMORY_DELTA_ITEMS,
    BudgetAmounts,
    BudgetLimits,
    BudgetSnapshot,
    ConstraintStatus,
    CycleRecord,
    CycleState,
    DeliveryRecord,
    DeliveryState,
    DeliveryTarget,
    InterventionAction,
    InterventionOutcome,
    InvocationDecision,
    NormalizedTraceEventDraft,
    OutcomeEvidenceMode,
    ReasonCode,
    RepeatedErrorStatus,
    TraceEvent,
    canonical_digest,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.domain.records import UUID4, ComponentIdentifier, Sha256Digest, UtcDatetime
from saliencegate.intervention import (
    GroundingConfig,
    GroundingReceipt,
    ReminderHistory,
    ResolvedGroundingConfiguration,
    claim_fingerprint,
)
from saliencegate.intervention.grounding import resolve_grounding_configuration
from saliencegate.ports.model_calls import (
    CanonicalUsageProvenance,
    ProviderUsageProvenance,
    StructuredCallPhase,
)
from saliencegate.ports.repository import (
    CycleRecoveryReceipt,
    LedgerEntry,
    LedgerHead,
    MemorySnapshot,
    ProjectionDigests,
)
from saliencegate.ports.trajectory import (
    ATTESTED_TRAJECTORY_PREFIX_SCHEMA_VERSION,
    MAX_TRAJECTORY_EVENTS,
    TRAJECTORY_PREFIX_REQUEST_SCHEMA_VERSION,
    AttestedTrajectoryPrefix,
    TrajectoryPrefixRequest,
)
from saliencegate.ports.two_phase import (
    CallReceipt,
    TwoPhaseCallPolicy,
    TwoPhaseCycleFailure,
    TwoPhaseCycleOutcome,
    TwoPhaseCycleRequest,
    TwoPhaseCycleResult,
    TwoPhaseFailureReason,
    TwoPhaseModelProfile,
    call_policy_accepts_receipts,
)
from saliencegate.prompts.contracts import PromptBundleIdentity
from saliencegate.repository.integrity import IntegrityContext
from saliencegate.repository.projector import (
    apply_entry,
    empty_projection,
    projection_digests,
)
from saliencegate.repository.projector import (
    budget_snapshot as projected_budget_snapshot,
)
from saliencegate.runtime.message_window import MessageWindow, _project_verified_message_window
from saliencegate.runtime.model_token_counting import ModelTokenCounterIdentity
from saliencegate.runtime.scheduling import (
    FixedStepSchedule,
    _project_verified_fixed_step_schedule,
)

ALGORITHM_CONFIGURATION_SCHEMA_VERSION: Literal["algorithm-configuration-attestation/v1"] = (
    "algorithm-configuration-attestation/v1"
)
ALGORITHM_RUN_RESULT_SCHEMA_VERSION: Literal["algorithm-run-result/v1"] = "algorithm-run-result/v1"
FIXED_STEP_RECOVERY_RESULT_SCHEMA_VERSION: Literal["fixed-step-recovery-result/v1"] = (
    "fixed-step-recovery-result/v1"
)

_ALGORITHM_CONFIGURATION_DIGEST_DOMAIN = (
    "saliencegate:runtime:algorithm-configuration-attestation:v1"
)
_ALGORITHM_TRACE_DIGEST_DOMAIN = "saliencegate:runtime:algorithm-trace:v1"
_ALGORITHM_RESULT_DIGEST_DOMAIN = "saliencegate:runtime:algorithm-run-result:v1"
_FIXED_STEP_RECOVERY_DIGEST_DOMAIN = "saliencegate:runtime:fixed-step-recovery-result:v1"
_ALGORITHM_IDENTITY_DOMAIN = "saliencegate:algorithm-runtime:fixed-step-identity:v1"
_MAX_CALLS_PER_INVOCATION = 34
_MAX_RECOVERY_LEDGER_ENTRIES = MAX_TRAJECTORY_EVENTS * 10
_MAX_TOKEN_TOTAL = ((1 << 63) - 1) * MAX_TRAJECTORY_EVENTS * _MAX_CALLS_PER_INVOCATION

TokenAggregate = Annotated[int, Field(ge=0, le=_MAX_TOKEN_TOTAL)]


class _AlgorithmResultModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def derive_cycle_reservation(call_policy: TwoPhaseCallPolicy) -> BudgetAmounts:
    """Derive the complete per-cycle envelope represented by a call policy."""

    if type(call_policy) is not TwoPhaseCallPolicy:
        raise ValueError("algorithm call policy failed exact validation")
    try:
        checked = TwoPhaseCallPolicy.model_validate_json(
            call_policy.model_dump_json(warnings=False)
        )
    except Exception:
        raise ValueError("algorithm call policy failed exact validation") from None
    return BudgetAmounts(
        model_calls=checked.max_model_calls,
        input_tokens=checked.max_provider_input_tokens,
        output_tokens=checked.max_provider_output_tokens,
        canonical_token_equivalents=(
            checked.max_provider_input_tokens + checked.max_provider_output_tokens
        ),
        latency_us=checked.max_total_latency_us,
        interventions=1,
        schema_repairs=checked.max_schema_repairs,
    )


def _configuration_digest(values: Mapping[str, object]) -> str:
    material = {
        key: value
        for key, value in values.items()
        if key != "configuration_digest" and not (key == "model_token_counter" and value is None)
    }
    return length_prefixed_sha256(
        canonical_json(material),
        domain=_ALGORITHM_CONFIGURATION_DIGEST_DOMAIN,
    )


class AlgorithmConfigurationAttestation(_AlgorithmResultModel):
    """Closed identity and maximum resource envelope for one algorithm run."""

    schema_version: Literal["algorithm-configuration-attestation/v1"] = (
        ALGORITHM_CONFIGURATION_SCHEMA_VERSION
    )
    policy_version: ComponentIdentifier
    budget_limits: BudgetLimits
    cycle_reservation: BudgetAmounts
    prompt_bundle: PromptBundleIdentity = Field(repr=False)
    model_profile: TwoPhaseModelProfile
    call_policy: TwoPhaseCallPolicy
    grounding_configuration: ResolvedGroundingConfiguration = Field(repr=False)
    model_token_counter: ModelTokenCounterIdentity | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    requested_delivery_target: DeliveryTarget | None = None
    configuration_digest: Sha256Digest = Field(default_factory=_configuration_digest)

    @model_validator(mode="after")
    def identities_envelope_and_digest_match(self) -> Self:
        expected_reservation = derive_cycle_reservation(self.call_policy)
        budget_fields = (
            "model_calls",
            "input_tokens",
            "output_tokens",
            "canonical_token_equivalents",
            "latency_us",
            "interventions",
            "schema_repairs",
        )
        if self.cycle_reservation != expected_reservation or any(
            getattr(self.cycle_reservation, field_name) > getattr(self.budget_limits, field_name)
            for field_name in budget_fields
        ):
            raise ValueError("algorithm cycle reservation does not fit its run budget")
        if self.call_policy.max_call_latency_us > self.budget_limits.max_call_latency_us:
            raise ValueError("algorithm call latency exceeds its run budget")
        if (
            self.model_profile.prompt_bundle_id != self.prompt_bundle.bundle_id
            or self.model_profile.prompt_bundle_digest != self.prompt_bundle.bundle_digest
        ):
            raise ValueError("algorithm model profile does not identify its prompt bundle")
        if (
            self.model_token_counter is not None
            and self.model_token_counter.model_id != self.model_profile.model_id
        ):
            raise ValueError("algorithm model token counter does not match the model profile")
        try:
            grounding = GroundingConfig.model_validate_json(
                canonical_json(self.grounding_configuration.configuration)
            )
            resolved = resolve_grounding_configuration(grounding)
        except Exception:
            raise ValueError("algorithm grounding configuration failed validation") from None
        if resolved != self.grounding_configuration or (
            self.requested_delivery_target is not None
            and self.requested_delivery_target not in grounding.allowed_delivery_targets
        ):
            raise ValueError("algorithm delivery target is not grounded by its configuration")
        values = self.model_dump(mode="json", exclude={"configuration_digest"}, warnings=False)
        if self.configuration_digest != _configuration_digest(values):
            raise ValueError("algorithm configuration digest does not match")
        return self


class ModelTokenUsageSource(StrEnum):
    LOCAL_COUNTER = "local_counter"
    PROVIDER_REPORTED = "provider_reported"
    REPLAY_ATTESTED = "replay_attested"
    UNAVAILABLE = "unavailable"


class ModelTokenUsageAttestation(_AlgorithmResultModel):
    """Run-level token evidence that preserves provider/local disagreement."""

    schema_version: Literal["model-token-usage-attestation/v1"] = "model-token-usage-attestation/v1"
    configured_counter: ModelTokenCounterIdentity | None
    usage_sources: tuple[ModelTokenUsageSource, ...]
    provider_input_tokens: TokenAggregate | None
    provider_output_tokens: TokenAggregate | None
    canonical_input_tokens: TokenAggregate | None
    canonical_output_tokens: TokenAggregate | None
    canonical_token_equivalents: TokenAggregate | None
    provider_canonical_disagreement: bool | None

    @model_validator(mode="after")
    def totals_sources_and_disagreement_match(self) -> Self:
        canonical_complete = (
            self.canonical_input_tokens is not None and self.canonical_output_tokens is not None
        )
        expected_disagreement = (
            self.provider_input_tokens + self.provider_output_tokens
            != self.canonical_token_equivalents
            if self.provider_input_tokens is not None
            and self.provider_output_tokens is not None
            and self.canonical_token_equivalents is not None
            else None
        )
        if (
            not self.usage_sources
            or tuple(sorted(set(self.usage_sources), key=lambda source: source.value))
            != self.usage_sources
            or (self.provider_input_tokens is None) != (self.provider_output_tokens is None)
            or canonical_complete != (self.canonical_token_equivalents is not None)
            or (
                canonical_complete
                and self.canonical_token_equivalents
                != cast(int, self.canonical_input_tokens) + cast(int, self.canonical_output_tokens)
            )
            or self.provider_canonical_disagreement is not expected_disagreement
            or (
                self.configured_counter is None
                and (
                    self.canonical_input_tokens is not None
                    or self.canonical_output_tokens is not None
                    or ModelTokenUsageSource.LOCAL_COUNTER in self.usage_sources
                )
            )
            or (
                ModelTokenUsageSource.UNAVAILABLE in self.usage_sources
                and self.usage_sources != (ModelTokenUsageSource.UNAVAILABLE,)
            )
        ):
            raise ValueError("model token usage attestation is inconsistent")
        return self


def _model_token_usage_attestation(
    configuration: AlgorithmConfigurationAttestation,
    receipts: tuple[CallReceipt, ...],
) -> ModelTokenUsageAttestation:
    sources: set[ModelTokenUsageSource] = set()
    for receipt in receipts:
        usage = receipt.usage
        if usage.provider_usage_provenance is ProviderUsageProvenance.PROVIDER_REPORTED:
            sources.add(ModelTokenUsageSource.PROVIDER_REPORTED)
        elif usage.provider_usage_provenance is ProviderUsageProvenance.REPLAY_ATTESTED:
            sources.add(ModelTokenUsageSource.REPLAY_ATTESTED)
        if usage.canonical_usage_provenance is CanonicalUsageProvenance.LOCAL_COUNTER:
            sources.add(ModelTokenUsageSource.LOCAL_COUNTER)
        elif usage.canonical_usage_provenance is CanonicalUsageProvenance.REPLAY_ATTESTED:
            sources.add(ModelTokenUsageSource.REPLAY_ATTESTED)
        if usage.canonical_usage_provenance is not CanonicalUsageProvenance.UNAVAILABLE:
            identity = ModelTokenCounterIdentity(
                counter_id=cast(str, usage.local_counter_id),
                counter_version=cast(str, usage.local_counter_version),
                configuration_digest=cast(
                    str,
                    usage.local_counter_configuration_digest,
                ),
                model_id=cast(str, usage.local_counter_model_id),
            )
            if (
                configuration.model_token_counter is None
                or identity != configuration.model_token_counter
            ):
                raise ValueError("model token usage counter differs from configuration")
    if not sources:
        sources.add(ModelTokenUsageSource.UNAVAILABLE)
    provider_complete = bool(receipts) and all(
        receipt.usage.provider_input_tokens is not None
        and receipt.usage.provider_output_tokens is not None
        for receipt in receipts
    )
    canonical_input_complete = bool(receipts) and all(
        receipt.usage.canonical_input_tokens is not None for receipt in receipts
    )
    canonical_output_complete = bool(receipts) and all(
        receipt.usage.canonical_output_tokens is not None for receipt in receipts
    )
    provider_input = (
        sum(cast(int, receipt.usage.provider_input_tokens) for receipt in receipts)
        if provider_complete
        else None
    )
    provider_output = (
        sum(cast(int, receipt.usage.provider_output_tokens) for receipt in receipts)
        if provider_complete
        else None
    )
    canonical_input = (
        sum(cast(int, receipt.usage.canonical_input_tokens) for receipt in receipts)
        if canonical_input_complete
        else None
    )
    canonical_output = (
        sum(cast(int, receipt.usage.canonical_output_tokens) for receipt in receipts)
        if canonical_output_complete
        else None
    )
    canonical_total = (
        canonical_input + canonical_output
        if canonical_input is not None and canonical_output is not None
        else None
    )
    disagreement = (
        provider_input + provider_output != canonical_total
        if provider_input is not None
        and provider_output is not None
        and canonical_total is not None
        else None
    )
    return ModelTokenUsageAttestation(
        configured_counter=configuration.model_token_counter,
        usage_sources=tuple(sorted(sources, key=lambda source: source.value)),
        provider_input_tokens=provider_input,
        provider_output_tokens=provider_output,
        canonical_input_tokens=canonical_input,
        canonical_output_tokens=canonical_output,
        canonical_token_equivalents=canonical_total,
        provider_canonical_disagreement=disagreement,
    )


def model_token_usage_attestation(
    configuration: AlgorithmConfigurationAttestation,
    receipts: tuple[CallReceipt, ...],
) -> ModelTokenUsageAttestation:
    """Aggregate visible call receipts without collapsing token provenance."""

    return _model_token_usage_attestation(configuration, receipts)


def algorithm_trace_digest(draft_digests: tuple[str, ...]) -> str:
    """Bind the ordered normalized inputs without persisting their content."""

    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": "algorithm-normalized-trace/v1",
                "draft_digests": draft_digests,
            }
        ),
        domain=_ALGORITHM_TRACE_DIGEST_DOMAIN,
    )


def algorithm_result_digest(values: Mapping[str, object]) -> str:
    """Compute the domain-separated digest of an algorithm result payload."""

    material = {key: value for key, value in values.items() if key != "result_digest"}
    run_id = material.get("run_id")
    if isinstance(run_id, UUID):
        material["run_id"] = str(run_id)
    return length_prefixed_sha256(
        canonical_json(material),
        domain=_ALGORITHM_RESULT_DIGEST_DOMAIN,
    )


def _algorithm_runtime_uuid(trace_digest: str, label: str, *parts: object) -> UUID:
    digest = length_prefixed_sha256(
        trace_digest,
        label,
        *(str(part) for part in parts),
        domain=_ALGORITHM_IDENTITY_DOMAIN,
    )
    raw = bytearray(bytes.fromhex(digest)[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(raw))


def algorithm_runtime_uuid(trace_digest: str, label: str, *parts: object) -> UUID:
    """Derive a deterministic runtime UUID from an attested trace identity."""

    return _algorithm_runtime_uuid(trace_digest, label, *parts)


def fixed_step_recovery_digest(values: Mapping[str, object]) -> str:
    """Compute the domain-separated digest of a durable recovery result."""

    material = {key: value for key, value in values.items() if key != "result_digest"}
    run_id = material.get("run_id")
    recovered_at = material.get("recovered_at")
    if isinstance(run_id, UUID):
        material["run_id"] = str(run_id)
    if isinstance(recovered_at, datetime):
        material["recovered_at"] = recovered_at.isoformat().replace("+00:00", "Z")
    return length_prefixed_sha256(
        canonical_json(material),
        domain=_FIXED_STEP_RECOVERY_DIGEST_DOMAIN,
    )


def _semantic_projection_digests(
    run_id: UUID,
    ledger: tuple[LedgerEntry, ...],
) -> ProjectionDigests:
    projection = empty_projection(run_id)
    for entry in ledger:
        projection = apply_entry(projection, entry)
    return projection_digests(
        projection,
        IntegrityContext(key=None, synthetic_benchmark=True),
        ledger_position=len(ledger),
    )


def semantic_projection_digests(
    run_id: UUID,
    ledger: tuple[LedgerEntry, ...],
) -> ProjectionDigests:
    """Rebuild semantic projection digests from an exact ordered ledger."""

    return _semantic_projection_digests(run_id, ledger)


def _persisted_event_draft_digest(event: TraceEvent) -> str:
    return canonical_digest(
        NormalizedTraceEventDraft(
            run_id=event.run_id,
            source_event_id=event.source_event_id,
            timestamp=event.timestamp,
            event_type=event.event_type,
            phase=event.phase,
            payload=event.payload,
            parent_ids=event.parent_ids,
            source_adapter=event.source_adapter,
            trust_label=event.trust_label,
        )
    )


def _prefix_through(
    prefix: AttestedTrajectoryPrefix,
    boundary_event_sequence: int,
) -> AttestedTrajectoryPrefix:
    items = prefix.items[:boundary_event_sequence]
    request = TrajectoryPrefixRequest(
        schema_version=TRAJECTORY_PREFIX_REQUEST_SCHEMA_VERSION,
        run_id=prefix.run_id,
        boundary_event_sequence=boundary_event_sequence,
        bindings=tuple(item.binding for item in items),
    )
    return AttestedTrajectoryPrefix(
        schema_version=ATTESTED_TRAJECTORY_PREFIX_SCHEMA_VERSION,
        run_id=prefix.run_id,
        boundary_event_sequence=boundary_event_sequence,
        request_digest=request.request_digest,
        items=items,
    )


def _reservation_fits_snapshot(
    reservation: BudgetAmounts,
    decision: InvocationDecision,
) -> bool:
    snapshot = decision.budget_snapshot
    fields = (
        "model_calls",
        "input_tokens",
        "output_tokens",
        "canonical_token_equivalents",
        "latency_us",
        "interventions",
        "schema_repairs",
    )
    return all(
        getattr(snapshot.reserved, field_name)
        + getattr(snapshot.consumed, field_name)
        + getattr(reservation, field_name)
        <= getattr(snapshot.limits, field_name)
        for field_name in fields
    )


class AlgorithmRunResult(_AlgorithmResultModel):
    """Replay-safe, repository-bound result of the fixed-step algorithm."""

    schema_version: Literal["algorithm-run-result/v1"] = ALGORITHM_RUN_RESULT_SCHEMA_VERSION
    run_id: UUID4
    trace_digest: Sha256Digest
    trace_event_count: Annotated[int, Field(ge=1, le=MAX_TRAJECTORY_EVENTS)]
    normalized_draft_digests: Annotated[
        tuple[Sha256Digest, ...],
        Field(min_length=1, max_length=MAX_TRAJECTORY_EVENTS),
    ]
    persisted_event_draft_digests: Annotated[
        tuple[Sha256Digest, ...],
        Field(min_length=1, max_length=MAX_TRAJECTORY_EVENTS),
    ]
    trajectory_prefix: AttestedTrajectoryPrefix = Field(repr=False)
    schedule: FixedStepSchedule
    windows: Annotated[
        tuple[MessageWindow, ...],
        Field(min_length=1, max_length=MAX_TRAJECTORY_EVENTS, repr=False),
    ]
    configuration: AlgorithmConfigurationAttestation = Field(repr=False)
    decisions: Annotated[
        tuple[InvocationDecision, ...],
        Field(min_length=1, max_length=MAX_TRAJECTORY_EVENTS),
    ]
    cycles: Annotated[
        tuple[CycleRecord, ...],
        Field(max_length=MAX_TRAJECTORY_EVENTS),
    ] = ()
    cycle_requests: Annotated[
        tuple[TwoPhaseCycleRequest, ...],
        Field(max_length=MAX_TRAJECTORY_EVENTS, repr=False),
    ] = ()
    executions: Annotated[
        tuple[TwoPhaseCycleOutcome, ...],
        Field(max_length=MAX_TRAJECTORY_EVENTS, repr=False),
    ] = ()
    call_receipts: Annotated[
        tuple[CallReceipt, ...],
        Field(max_length=MAX_TRAJECTORY_EVENTS * _MAX_CALLS_PER_INVOCATION, repr=False),
    ] = ()
    deliveries: Annotated[
        tuple[DeliveryRecord, ...],
        Field(max_length=MAX_TRAJECTORY_EVENTS),
    ] = ()
    outcomes: Annotated[
        tuple[InterventionOutcome, ...],
        Field(max_length=MAX_TRAJECTORY_EVENTS),
    ] = ()
    model_token_usage: ModelTokenUsageAttestation
    projection_digests: ProjectionDigests
    ledger_entry_count: Annotated[int, Field(ge=1)]
    ledger_head: LedgerHead
    rebuild_equivalent: bool
    result_digest: Sha256Digest = Field(default_factory=algorithm_result_digest)

    @model_validator(mode="after")
    def result_attests_one_complete_execution(self) -> Self:
        values = self.model_dump(mode="json", exclude={"result_digest"}, warnings=False)
        if self.result_digest != algorithm_result_digest(values):
            raise ValueError("algorithm result digest does not match")
        if self.model_token_usage != _model_token_usage_attestation(
            self.configuration,
            self.call_receipts,
        ):
            raise ValueError("algorithm model token usage differs from call receipts")
        self._validate_trace_schedule_and_windows()
        self._validate_decisions_and_cycles()
        self._validate_calls_deliveries_and_outcomes()
        self._validate_ledger()
        return self

    def _validate_trace_schedule_and_windows(self) -> None:
        events = tuple(item.event for item in self.trajectory_prefix.items)
        event_ids = tuple(event.event_id for event in events)
        try:
            request = TrajectoryPrefixRequest(
                schema_version=TRAJECTORY_PREFIX_REQUEST_SCHEMA_VERSION,
                run_id=self.run_id,
                boundary_event_sequence=self.trace_event_count,
                bindings=tuple(item.binding for item in self.trajectory_prefix.items),
            )
        except Exception:
            raise ValueError("algorithm trajectory request failed reconstruction") from None
        if (
            self.trajectory_prefix.run_id != self.run_id
            or self.trajectory_prefix.boundary_event_sequence != self.trace_event_count
            or self.trajectory_prefix.request_digest != request.request_digest
            or len(self.normalized_draft_digests) != self.trace_event_count
            or len(self.persisted_event_draft_digests) != self.trace_event_count
            or self.trace_digest != algorithm_trace_digest(self.normalized_draft_digests)
            or self.persisted_event_draft_digests
            != tuple(_persisted_event_draft_digest(event) for event in events)
            or len(set(event_ids)) != len(event_ids)
        ):
            raise ValueError("algorithm trace attestation does not match persisted events")
        event_order = {event_id: ordinal for ordinal, event_id in enumerate(event_ids)}
        if any(
            event.parent_ids != tuple(sorted(event.parent_ids, key=str))
            or any(
                parent_id not in event_order
                or event_order[parent_id] >= event_order[event.event_id]
                for parent_id in event.parent_ids
            )
            for event in events
        ):
            raise ValueError("algorithm trace parent graph is not causal")
        try:
            expected_schedule = _project_verified_fixed_step_schedule(self.trajectory_prefix)
        except Exception:
            raise ValueError("algorithm fixed-step schedule failed reprojection") from None
        if canonical_json(self.schedule) != canonical_json(expected_schedule):
            raise ValueError("algorithm fixed-step schedule does not match its trajectory")
        scheduled = tuple(item for item in self.schedule.decisions if item.invoke)
        if len(self.windows) != len(scheduled):
            raise ValueError("algorithm windows do not cover every scheduled invocation")
        for scheduled_decision, window in zip(scheduled, self.windows, strict=True):
            try:
                expected_window = _project_verified_message_window(
                    _prefix_through(self.trajectory_prefix, scheduled_decision.event_sequence)
                )
            except Exception:
                raise ValueError("algorithm message window failed reprojection") from None
            if canonical_json(window) != canonical_json(expected_window):
                raise ValueError("algorithm message window does not match its trajectory boundary")

    def _validate_decisions_and_cycles(self) -> None:
        events = tuple(item.event for item in self.trajectory_prefix.items)
        if len(self.decisions) != self.trace_event_count:
            raise ValueError("algorithm must contain one invocation decision per event")
        decision_ids = tuple(decision.decision_id for decision in self.decisions)
        for event, scheduled, decision in zip(
            events,
            self.schedule.decisions,
            self.decisions,
            strict=True,
        ):
            if (
                decision.run_id != self.run_id
                or decision.event_sequence != event.sequence
                or decision.created_at != event.timestamp
                or decision.policy_version != self.configuration.policy_version
                or decision.configuration_digest != self.configuration.configuration_digest
                or decision.budget_snapshot.limits != self.configuration.budget_limits
            ):
                raise ValueError("algorithm invocation decision is not configuration-bound")
            reservation_fits = _reservation_fits_snapshot(
                self.configuration.cycle_reservation,
                decision,
            )
            if decision.risk_score is not None or decision.cooldown_active:
                raise ValueError("fixed-step decisions cannot claim risk or cooldown state")
            if scheduled.invoke:
                if decision.invoke and not reservation_fits:
                    raise ValueError("algorithm invoked without enough reserved capacity")
                if not decision.invoke and (
                    decision.reason_codes != (ReasonCode.BUDGET_EXHAUSTED,) or reservation_fits
                ):
                    raise ValueError("only budget exhaustion may demote a scheduled invocation")
                if decision.invoke:
                    expected_reason = (
                        ReasonCode.BOOTSTRAP
                        if scheduled.event_sequence == 1
                        else ReasonCode.SCRIPTED_INVOKE
                    )
                    if decision.reason_codes != (expected_reason,):
                        raise ValueError("algorithm invocation reason is not fixed-step exact")
            elif decision.invoke or decision.reason_codes != (ReasonCode.SCRIPTED_SILENCE,):
                raise ValueError("algorithm invoked or reasoned outside the fixed-step schedule")
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("algorithm invocation decisions are not unique")

        invoked = tuple(decision for decision in self.decisions if decision.invoke)
        if (
            len(self.cycles) != len(invoked)
            or len(self.cycle_requests) != len(self.cycles)
            or len(self.executions) != len(self.cycles)
        ):
            raise ValueError("algorithm cycles do not cover every executed invocation")
        try:
            grounding = GroundingConfig.model_validate_json(
                canonical_json(self.configuration.grounding_configuration.configuration)
            )
        except Exception:
            raise ValueError("algorithm grounding history configuration is invalid") from None
        history_window = max(
            grounding.duplicate_window_events,
            grounding.cooldown_events,
        )
        memory_cursor = 0
        prior_cycles: list[CycleRecord] = []
        for decision, cycle, request, execution in zip(
            invoked,
            self.cycles,
            self.cycle_requests,
            self.executions,
            strict=True,
        ):
            running_values = cycle.model_dump(mode="python", warnings=False)
            running_values.update(
                revision=cycle.revision - 1,
                state=CycleState.RUNNING,
                budget_settlement=None,
                model_call_digests=(),
                model_call_latencies_us=(),
                validated_delta=None,
                memory_id_assignments=(),
                intervention=None,
                failure_reason=None,
            )
            try:
                expected_running = CycleRecord.model_validate(running_values)
            except Exception:
                raise ValueError("algorithm running cycle failed reconstruction") from None
            expected_events = tuple(
                item.event
                for item in self.trajectory_prefix.items
                if item.event.sequence <= cycle.last_event_sequence
            )
            first_history_sequence = max(
                1,
                cycle.last_event_sequence - history_window,
            )
            try:
                expected_history = tuple(
                    ReminderHistory(
                        schema_version="1.0",
                        intervention_id=prior.intervention.intervention_id,
                        run_id=self.run_id,
                        event_sequence=prior.last_event_sequence,
                        claim_digests=tuple(
                            claim_fingerprint(claim) for claim in prior.intervention.claims
                        ),
                    )
                    for prior in sorted(
                        prior_cycles,
                        key=lambda value: (value.last_event_sequence, value.cycle_id),
                    )
                    if prior.state is CycleState.COMMITTED
                    and prior.intervention is not None
                    and prior.intervention.action is InterventionAction.REMIND
                    and first_history_sequence
                    <= prior.last_event_sequence
                    < cycle.last_event_sequence
                )
            except Exception:
                raise ValueError("algorithm reminder history failed reconstruction") from None
            expected_assigned_ids = tuple(
                _algorithm_runtime_uuid(
                    self.trace_digest,
                    "memory",
                    cycle.cycle_id,
                    ordinal,
                )
                for ordinal in range(1, MAX_MEMORY_DELTA_ITEMS + 1)
            )
            if (
                cycle.run_id != self.run_id
                or cycle.invocation_decision_id != decision.decision_id
                or cycle.last_event_sequence != decision.event_sequence
                or cycle.first_event_sequence != memory_cursor + 1
                or cycle.state not in (CycleState.COMMITTED, CycleState.FAILED)
                or cycle.revision != 4
                or cycle.policy_version != self.configuration.policy_version
                or cycle.configuration_digest != self.configuration.configuration_digest
                or cycle.grounding_version
                != self.configuration.grounding_configuration.pipeline_version
                or cycle.grounding_configuration_digest
                != self.configuration.grounding_configuration.configuration_digest
                or canonical_json(cycle.grounding_configuration)
                != canonical_json(self.configuration.grounding_configuration.configuration)
                or cycle.requested_delivery_target
                is not self.configuration.requested_delivery_target
                or cycle.budget_reservation != self.configuration.cycle_reservation
                or request.cycle_receipt.cycle != expected_running
                or request.created_at != decision.created_at
                or request.delta_id
                != _algorithm_runtime_uuid(self.trace_digest, "delta", cycle.cycle_id)
                or request.assigned_memory_ids != expected_assigned_ids
                or request.intervention_id
                != _algorithm_runtime_uuid(
                    self.trace_digest,
                    "intervention",
                    cycle.cycle_id,
                )
                or request.grounding_state.events != expected_events
                or request.grounding_state.reminder_history != expected_history
                or request.request_digest != execution.request_digest
                or execution.run_id != cycle.run_id
                or execution.cycle_id != cycle.cycle_id
                or execution.model_id != self.configuration.model_profile.model_id
                or execution.model_profile_digest != self.configuration.model_profile.profile_digest
                or execution.call_policy_digest != self.configuration.call_policy.policy_digest
                or execution.call_policy != self.configuration.call_policy
                or execution.prompt_bundle_digest != self.configuration.prompt_bundle.bundle_digest
            ):
                raise ValueError("algorithm cycle is not configuration- or cursor-bound")
            if cycle.state is CycleState.COMMITTED:
                memory_cursor = cycle.last_event_sequence
            prior_cycles.append(cycle)

    def _validate_calls_deliveries_and_outcomes(self) -> None:
        cycle_ids = tuple(cycle.cycle_id for cycle in self.cycles)
        if len(set(cycle_ids)) != len(cycle_ids):
            raise ValueError("algorithm cycles are not unique")
        window_by_boundary = {
            scheduled.event_sequence: window
            for scheduled, window in zip(
                (item for item in self.schedule.decisions if item.invoke),
                self.windows,
                strict=True,
            )
        }
        templates = {
            (item.phase, item.template_id, item.template_digest)
            for item in self.configuration.prompt_bundle.templates
        }
        ordered_receipts: list[CallReceipt] = []
        for cycle, request, execution in zip(
            self.cycles,
            self.cycle_requests,
            self.executions,
            strict=True,
        ):
            receipts = tuple(item for item in self.call_receipts if item.cycle_id == cycle.cycle_id)
            ordered_receipts.extend(receipts)
            window = window_by_boundary[cycle.last_event_sequence]
            policy_accepted = call_policy_accepts_receipts(
                self.configuration.call_policy,
                receipts,
            )
            if (
                tuple(item.model_call_index for item in receipts) != tuple(range(len(receipts)))
                or any(
                    item.run_id != self.run_id
                    or item.model_id != self.configuration.model_profile.model_id
                    or item.window_digest != window.window_digest
                    or (
                        item.phase,
                        item.prompt_template_id,
                        item.prompt_template_digest,
                    )
                    not in templates
                    for item in receipts
                )
                or tuple(item.call_digest for item in receipts) != cycle.model_call_digests
                or tuple(item.usage.latency_us for item in receipts)
                != cycle.model_call_latencies_us
                or cycle.batch_digest != window.window_digest
                or canonical_json(request.window) != canonical_json(window)
                or not request.cycle_receipt.appended
                or request.cycle_receipt.delivery is not None
                or request.cycle_receipt.ledger_position > self.ledger_entry_count
                or execution.window_digest != window.window_digest
                or execution.request_digest != request.request_digest
                or execution.call_receipts != receipts
                or receipts[0].bank_view_digest != request.current_bank.view_digest
                or (
                    type(execution) is TwoPhaseCycleFailure
                    and (execution.reason is TwoPhaseFailureReason.CALL_POLICY_EXCEEDED)
                    is policy_accepted
                )
                or (type(execution) is TwoPhaseCycleResult and not policy_accepted)
            ):
                raise ValueError("algorithm call receipts do not match their cycle")
            if type(execution) is TwoPhaseCycleResult and (
                request.delta_id != execution.validated_delta.delta_id
                or request.intervention_id != execution.intervention.intervention_id
                or request.grounding_state.events != execution.grounding_state.events
                or request.grounding_state.reminder_history
                != execution.grounding_state.reminder_history
                or request.current_bank.view_digest != execution.current_bank_view_digest
                or tuple(item.memory_id for item in execution.memory_id_assignments)
                != request.assigned_memory_ids[: len(execution.memory_id_assignments)]
            ):
                raise ValueError("algorithm request differs from its successful execution")
            if cycle.state is CycleState.COMMITTED:
                if type(execution) is not TwoPhaseCycleResult:
                    raise ValueError("committed cycle lacks a successful execution attestation")
                completed = execution
                if self.configuration.call_policy.max_schema_repairs == 0 and (
                    len(receipts) != 2
                    or tuple(item.phase for item in receipts)
                    != (
                        StructuredCallPhase.MEMORY_EDIT,
                        StructuredCallPhase.INTERVENTION,
                    )
                    or any(item.attempt != 0 for item in receipts)
                ):
                    raise ValueError("baseline committed cycle is not exactly two phase calls")
                intervention = cycle.intervention
                if intervention is None or not receipts:
                    raise ValueError("committed cycle lacks its intervention call")
                if (
                    cycle.validated_delta != completed.validated_delta
                    or cycle.memory_id_assignments != completed.memory_id_assignments
                    or intervention != completed.intervention
                ):
                    raise ValueError("committed outputs differ from their execution attestation")
                try:
                    grounding_receipt = GroundingReceipt.model_validate_json(
                        canonical_json(intervention.grounding_receipt)
                    )
                except Exception:
                    raise ValueError("committed grounding receipt failed validation") from None
                final_call = receipts[-1]
                if (
                    final_call.phase is not StructuredCallPhase.INTERVENTION
                    or grounding_receipt.model_call_index != final_call.model_call_index
                    or grounding_receipt.model_call_digest != final_call.call_digest
                ):
                    raise ValueError("committed grounding receipt does not name its final call")
            elif type(execution) is TwoPhaseCycleFailure:
                expected_reason = {
                    TwoPhaseFailureReason.MODEL_ERROR: ReasonCode.MODEL_ERROR,
                    TwoPhaseFailureReason.MODEL_TIMEOUT: ReasonCode.MODEL_TIMEOUT,
                }.get(execution.reason, ReasonCode.INVALID_STRUCTURED_OUTPUT)
                if (
                    cycle.failure_reason is not expected_reason
                    or len(request.assigned_memory_ids) != execution.assigned_memory_id_capacity
                ):
                    raise ValueError("failed cycle reason differs from its execution failure")
            elif cycle.failure_reason not in {
                ReasonCode.MEMORY_CONFLICT,
                ReasonCode.TARGET_UNAVAILABLE,
            }:
                raise ValueError("successful execution cannot explain the failed cycle")
            self._validate_cycle_settlement(cycle, receipts)
        if tuple(ordered_receipts) != self.call_receipts:
            raise ValueError("algorithm call receipts are not grouped in cycle order")
        call_digests = tuple(item.call_digest for item in self.call_receipts)
        request_digests = tuple(item.request_digest for item in self.call_receipts)
        if len(set(call_digests)) != len(call_digests) or len(set(request_digests)) != len(
            request_digests
        ):
            raise ValueError("algorithm call identities are not unique")

        committed = tuple(cycle for cycle in self.cycles if cycle.state is CycleState.COMMITTED)
        reminders = tuple(
            cycle
            for cycle in committed
            if cycle.intervention is not None
            and cycle.intervention.action is InterventionAction.REMIND
        )
        if len(self.deliveries) != len(reminders):
            raise ValueError("algorithm deliveries do not cover committed reminders")
        for cycle, delivery in zip(reminders, self.deliveries, strict=True):
            assert cycle.intervention is not None  # established above
            if (
                delivery.run_id != self.run_id
                or delivery.cycle_id != cycle.cycle_id
                or delivery.intervention_id != cycle.intervention.intervention_id
                or delivery.target is not cycle.requested_delivery_target
                or cycle.intervention.rendered_text is None
                or delivery.rendered_text_digest
                != canonical_digest(cycle.intervention.rendered_text)
                or delivery.state
                in (DeliveryState.PENDING, DeliveryState.CLAIMED, DeliveryState.ATTEMPTING)
            ):
                raise ValueError("algorithm delivery does not match its committed reminder")
        delivery_ids = tuple(item.delivery_id for item in self.deliveries)
        if len(set(delivery_ids)) != len(delivery_ids):
            raise ValueError("algorithm delivery identities are not unique")

        interventions = tuple(
            cycle.intervention for cycle in committed if cycle.intervention is not None
        )
        if len(interventions) != len(committed) or len(self.outcomes) != len(interventions):
            raise ValueError("algorithm outcomes do not cover committed interventions")
        for intervention, outcome in zip(interventions, self.outcomes, strict=True):
            if (
                outcome.run_id != self.run_id
                or outcome.intervention_id != intervention.intervention_id
                or outcome.created_at != intervention.created_at
                or outcome.evidence_mode is not OutcomeEvidenceMode.POLICY_REPLAY
                or outcome.next_action_fingerprint is not None
                or outcome.repeated_error_status is not RepeatedErrorStatus.UNKNOWN
                or outcome.constraint_status is not ConstraintStatus.UNKNOWN
                or outcome.utility is not None
                or outcome.action_changed is not None
                or outcome.task_reward is not None
                or outcome.task_passed is not None
                or outcome.steps != 0
                or outcome.tool_calls != 0
                or outcome.memory_calls != 0
                or outcome.input_tokens != 0
                or outcome.output_tokens != 0
                or outcome.canonical_token_equivalents != 0
                or outcome.latency_us != 0
            ):
                raise ValueError("algorithm result contains an unsupported outcome claim")
        outcome_ids = tuple(item.outcome_id for item in self.outcomes)
        if len(set(outcome_ids)) != len(outcome_ids):
            raise ValueError("algorithm outcome identities are not unique")

    def _validate_cycle_settlement(
        self,
        cycle: CycleRecord,
        receipts: tuple[CallReceipt, ...],
    ) -> None:
        settlement = cycle.budget_settlement
        if settlement is None:
            if receipts:
                raise ValueError("algorithm unaccounted call receipts are not allowed")
            return
        if cycle.failure_reason is ReasonCode.FAILED_UNKNOWN_COST:
            return
        if settlement.model_calls != len(receipts):
            raise ValueError("algorithm model-call settlement does not reconcile")
        known_input = sum(item.usage.provider_input_tokens or 0 for item in receipts)
        known_output = sum(item.usage.provider_output_tokens or 0 for item in receipts)
        all_tokens_known = all(
            item.usage.provider_input_tokens is not None
            and item.usage.provider_output_tokens is not None
            for item in receipts
        )
        all_canonical_tokens_known = all(
            item.usage.canonical_input_tokens is not None
            and item.usage.canonical_output_tokens is not None
            for item in receipts
        )
        known_canonical_tokens = sum(
            (item.usage.canonical_input_tokens or 0) + (item.usage.canonical_output_tokens or 0)
            for item in receipts
        )
        if (
            settlement.latency_us != sum(item.usage.latency_us for item in receipts)
            or settlement.schema_repairs != sum(item.attempt > 0 for item in receipts)
            or settlement.canonical_token_equivalents
            != (
                known_canonical_tokens
                if all_canonical_tokens_known
                else self.configuration.cycle_reservation.canonical_token_equivalents
            )
            or settlement.input_tokens < known_input
            or settlement.output_tokens < known_output
            or (
                all_tokens_known
                and (
                    settlement.input_tokens != known_input
                    or settlement.output_tokens != known_output
                )
            )
            or (
                not all_tokens_known
                and (
                    settlement.input_tokens != self.configuration.cycle_reservation.input_tokens
                    or settlement.output_tokens
                    != self.configuration.cycle_reservation.output_tokens
                )
            )
        ):
            raise ValueError("algorithm cycle settlement does not reconcile with call receipts")

    def _validate_ledger(self) -> None:
        represented_entries = (
            2 * self.trace_event_count
            + sum(cycle.revision for cycle in self.cycles)
            + sum(delivery.revision for delivery in self.deliveries)
            + len(self.outcomes)
        )
        if (
            not self.rebuild_equivalent
            or self.ledger_entry_count != represented_entries
            or self.ledger_head.run_id != self.run_id
            or self.ledger_head.entry_count != self.ledger_entry_count
            or self.ledger_head.head_tag.algorithm is not self.projection_digests.overall.algorithm
        ):
            raise ValueError("algorithm ledger attestation is inconsistent")


class FixedStepRecoveryResult(_AlgorithmResultModel):
    """Durable recovery attestation that never invents lost model-call witnesses."""

    schema_version: Literal["fixed-step-recovery-result/v1"] = (
        FIXED_STEP_RECOVERY_RESULT_SCHEMA_VERSION
    )
    run_id: UUID4
    recovered_at: UtcDatetime
    configuration: AlgorithmConfigurationAttestation = Field(repr=False)
    cycle_recovery: CycleRecoveryReceipt
    deliveries: Annotated[
        tuple[DeliveryRecord, ...],
        Field(max_length=MAX_TRAJECTORY_EVENTS),
    ] = ()
    budget_snapshot: BudgetSnapshot
    memory_snapshot: MemorySnapshot = Field(repr=False)
    semantic_projection_digests: ProjectionDigests = Field(
        description="Portable synthetic digests recomputed from the included ledger.",
    )
    repository_projection_digests: ProjectionDigests = Field(
        repr=False,
        description=("Opaque keyed repository digests; verify them with the owning repository."),
    )
    pre_recovery_ledger_head: LedgerHead
    ledger: Annotated[
        tuple[LedgerEntry, ...],
        Field(min_length=1, max_length=_MAX_RECOVERY_LEDGER_ENTRIES, repr=False),
    ]
    ledger_entry_count: Annotated[int, Field(ge=1)]
    ledger_head: LedgerHead
    rebuild_equivalent: bool
    result_digest: Sha256Digest = Field(default_factory=fixed_step_recovery_digest)

    @model_validator(mode="after")
    def recovery_is_durable_and_configuration_bound(self) -> Self:
        values = self.model_dump(mode="json", exclude={"result_digest"}, warnings=False)
        if self.result_digest != fixed_step_recovery_digest(values):
            raise ValueError("fixed-step recovery digest does not match")
        if (
            len(self.ledger) != self.ledger_entry_count
            or tuple(entry.position for entry in self.ledger)
            != tuple(range(1, self.ledger_entry_count + 1))
            or any(entry.run_id != self.run_id for entry in self.ledger)
            or any(
                later.previous_chain_tag != earlier.chain_tag
                for earlier, later in zip(self.ledger, self.ledger[1:], strict=False)
            )
            or self.ledger[-1].chain_tag != self.ledger_head.chain_tag
            or self.pre_recovery_ledger_head.run_id != self.run_id
            or self.pre_recovery_ledger_head.entry_count > self.ledger_entry_count
            or self.ledger[self.pre_recovery_ledger_head.entry_count - 1].chain_tag
            != self.pre_recovery_ledger_head.chain_tag
        ):
            raise ValueError("fixed-step recovery ledger chain is inconsistent")
        try:
            projection = empty_projection(self.run_id)
            for entry in self.ledger:
                projection = apply_entry(projection, entry)
            projected_budget = projected_budget_snapshot(projection)
        except Exception:
            raise ValueError("fixed-step recovery ledger failed semantic replay") from None
        replayed_projection_digests = projection_digests(
            projection,
            IntegrityContext(key=None, synthetic_benchmark=True),
            ledger_position=self.ledger_entry_count,
        )
        if self.semantic_projection_digests != replayed_projection_digests:
            raise ValueError("fixed-step recovery projection attestation is inconsistent")
        grounding_configuration = self.configuration.grounding_configuration
        if (
            not projection.decisions
            or any(
                decision.policy_version != self.configuration.policy_version
                or decision.configuration_digest != self.configuration.configuration_digest
                or decision.budget_snapshot.limits != self.configuration.budget_limits
                for decision in projection.decisions.values()
            )
            or any(
                cycle.policy_version != self.configuration.policy_version
                or cycle.configuration_digest != self.configuration.configuration_digest
                or cycle.grounding_version != grounding_configuration.pipeline_version
                or cycle.grounding_configuration_digest
                != grounding_configuration.configuration_digest
                or canonical_json(cycle.grounding_configuration)
                != canonical_json(grounding_configuration.configuration)
                or cycle.requested_delivery_target
                is not self.configuration.requested_delivery_target
                or (
                    cycle.budget_reservation is not None
                    and cycle.budget_reservation != self.configuration.cycle_reservation
                )
                for cycle in projection.cycles.values()
            )
        ):
            raise ValueError("fixed-step recovery ledger is not configuration-bound")
        if self.cycle_recovery.run_id != self.run_id:
            raise ValueError("fixed-step cycle recovery belongs to another run")
        pending = self.cycle_recovery.resumable_pending
        reserved = self.cycle_recovery.resumable_reserved
        failed = tuple(receipt.cycle for receipt in self.cycle_recovery.failed_unknown_cost)
        cycles = (*pending, *reserved, *failed)
        if len({cycle.cycle_id for cycle in cycles}) != len(cycles):
            raise ValueError("fixed-step recovery cycles are not unique")
        if any(
            cycle.run_id != self.run_id
            or cycle.policy_version != self.configuration.policy_version
            or cycle.configuration_digest != self.configuration.configuration_digest
            or cycle.updated_at > self.recovered_at
            for cycle in cycles
        ):
            raise ValueError("fixed-step recovery cycle is not configuration-bound")
        if any(
            cycle.state is not CycleState.PENDING
            or cycle.budget_reservation is not None
            or cycle.budget_settlement is not None
            for cycle in pending
        ):
            raise ValueError("fixed-step pending recovery classification is invalid")
        if any(
            cycle.state is not CycleState.RESERVED
            or cycle.budget_reservation != self.configuration.cycle_reservation
            or cycle.budget_settlement is not None
            for cycle in reserved
        ):
            raise ValueError("fixed-step reserved recovery classification is invalid")
        if any(
            not receipt.appended
            or receipt.cycle.state is not CycleState.FAILED
            or receipt.cycle.failure_reason is not ReasonCode.FAILED_UNKNOWN_COST
            or receipt.cycle.budget_reservation != self.configuration.cycle_reservation
            or receipt.cycle.budget_settlement != self.configuration.cycle_reservation
            or receipt.delivery is not None
            for receipt in self.cycle_recovery.failed_unknown_cost
        ):
            raise ValueError("fixed-step unknown-cost recovery classification is invalid")
        recovered_unknown_cycles = tuple(
            entry.record
            for entry in self.ledger[self.pre_recovery_ledger_head.entry_count :]
            if type(entry.record) is CycleRecord
            and entry.record.state is CycleState.FAILED
            and entry.record.failure_reason is ReasonCode.FAILED_UNKNOWN_COST
        )
        if failed != recovered_unknown_cycles or any(
            receipt.ledger_position <= self.pre_recovery_ledger_head.entry_count
            for receipt in self.cycle_recovery.failed_unknown_cost
        ):
            raise ValueError("fixed-step recovered cycles do not match the ledger suffix")
        projection_pending = tuple(
            sorted(
                (
                    cycle
                    for cycle in projection.cycles.values()
                    if cycle.state is CycleState.PENDING
                ),
                key=lambda cycle: cycle.cycle_id,
            )
        )
        projection_reserved = tuple(
            sorted(
                (
                    cycle
                    for cycle in projection.cycles.values()
                    if cycle.state is CycleState.RESERVED
                ),
                key=lambda cycle: cycle.cycle_id,
            )
        )
        if (
            tuple(sorted(pending, key=lambda cycle: cycle.cycle_id)) != projection_pending
            or tuple(sorted(reserved, key=lambda cycle: cycle.cycle_id)) != projection_reserved
            or any(cycle.state is CycleState.RUNNING for cycle in projection.cycles.values())
            or any(projection.cycles.get(cycle.cycle_id) != cycle for cycle in failed)
        ):
            raise ValueError("fixed-step recovery cycles differ from ledger replay")
        for receipt in self.cycle_recovery.failed_unknown_cost:
            entry = self.ledger[receipt.ledger_position - 1]
            if (
                entry.record != receipt.cycle
                or entry.record_tag != receipt.record_tag
                or entry.chain_tag != receipt.chain_tag
            ):
                raise ValueError("fixed-step recovery receipt differs from its ledger entry")
        terminal_delivery_states = {
            DeliveryState.DELIVERED,
            DeliveryState.FAILED,
            DeliveryState.REJECTED,
            DeliveryState.UNKNOWN,
        }
        if len({delivery.delivery_id for delivery in self.deliveries}) != len(
            self.deliveries
        ) or any(
            delivery.run_id != self.run_id or delivery.state not in terminal_delivery_states
            for delivery in self.deliveries
        ):
            raise ValueError("fixed-step recovered deliveries are not terminal")
        pre_recovery_projection = empty_projection(self.run_id)
        for entry in self.ledger[: self.pre_recovery_ledger_head.entry_count]:
            pre_recovery_projection = apply_entry(pre_recovery_projection, entry)
        recoverable_delivery_states = {
            DeliveryState.PENDING,
            DeliveryState.CLAIMED,
            DeliveryState.ATTEMPTING,
            DeliveryState.UNKNOWN,
        }
        expected_recovered_deliveries = tuple(
            sorted(
                (
                    projection.deliveries[delivery_id]
                    for delivery_id, delivery in pre_recovery_projection.deliveries.items()
                    if delivery.state in recoverable_delivery_states
                ),
                key=lambda delivery: delivery.delivery_id,
            )
        )
        if (
            tuple(sorted(self.deliveries, key=lambda delivery: delivery.delivery_id))
            != expected_recovered_deliveries
        ):
            raise ValueError("fixed-step recovered deliveries differ from ledger replay")
        projected_records = tuple(
            sorted(
                projection.memories.values(),
                key=lambda record: (record.kind.value, str(record.memory_id)),
            )
        )
        if (
            self.budget_snapshot != projected_budget
            or self.budget_snapshot.limits != self.configuration.budget_limits
            or self.memory_snapshot.run_id != self.run_id
            or self.memory_snapshot.ledger_position != self.ledger_entry_count
            or self.memory_snapshot.ingestion_cursor != projection.ingestion_cursor
            or self.memory_snapshot.memory_cursor != projection.memory_cursor
            or self.memory_snapshot.records != projected_records
            or self.memory_snapshot.projection_digest != self.repository_projection_digests.overall
            or not self.rebuild_equivalent
            or self.ledger_head.run_id != self.run_id
            or self.ledger_head.entry_count != self.ledger_entry_count
            or self.ledger_head.head_tag.algorithm
            is not self.repository_projection_digests.overall.algorithm
        ):
            raise ValueError("fixed-step recovery repository attestation is inconsistent")
        return self


__all__ = [
    "ALGORITHM_CONFIGURATION_SCHEMA_VERSION",
    "ALGORITHM_RUN_RESULT_SCHEMA_VERSION",
    "FIXED_STEP_RECOVERY_RESULT_SCHEMA_VERSION",
    "AlgorithmConfigurationAttestation",
    "AlgorithmRunResult",
    "FixedStepRecoveryResult",
    "ModelTokenUsageAttestation",
    "ModelTokenUsageSource",
    "algorithm_result_digest",
    "algorithm_runtime_uuid",
    "algorithm_trace_digest",
    "derive_cycle_reservation",
    "fixed_step_recovery_digest",
    "model_token_usage_attestation",
    "semantic_projection_digests",
]
