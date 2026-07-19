from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from typing import Literal, Protocol, Self, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import to_jsonable_python

from saliencegate.domain import (
    MAX_SIGNAL_EVIDENCE_EVENTS,
    BudgetAmounts,
    BudgetLimits,
    BudgetSnapshot,
    ClaimKind,
    ConstraintStatus,
    CycleRecord,
    CycleState,
    DeliveryRecord,
    DeliveryState,
    DeliveryTarget,
    EvidenceSource,
    InterventionAction,
    InterventionOutcome,
    InvocationDecision,
    MemoryIdAssignment,
    MemoryKind,
    MemoryRecord,
    NormalizedTraceEventDraft,
    OutcomeEvidenceMode,
    ReasonCode,
    RepeatedErrorStatus,
    Signal,
    SignalType,
    TraceEvent,
    ValidityState,
    canonical_digest,
    canonical_json,
    cycle_id,
    evidence_reference_is_bounded,
    length_prefixed_sha256,
    memory_delta_is_bounded,
    normalized_trace_event_draft_is_bounded,
)
from saliencegate.domain import (
    delivery_id as derive_delivery_id,
)
from saliencegate.domain.records import ComponentIdentifier, Sha256Digest
from saliencegate.intervention import (
    GROUNDING_RECEIPT_VERSION,
    GroundingContext,
    GroundingPipeline,
    GroundingReceipt,
    GroundingState,
    ProposalParseStatus,
    ProposedClaim,
    ReminderHistory,
    claim_fingerprint,
)
from saliencegate.policy.config import RunState
from saliencegate.ports.adapters import (
    AdapterCapabilities,
    DeliveryAdapter,
    DeliveryEnvelope,
    DeliveryReceipt,
    adapter_capabilities_digest,
    enqueue_delivery_binding,
    validated_capabilities,
)
from saliencegate.ports.memory import GroundingObservation, MemoryCycleOutput
from saliencegate.ports.models import (
    MAX_MODEL_REQUEST_PAYLOAD_BYTES,
    ModelCallStatus,
    ModelRequest,
    ModelResult,
    ModelUsage,
    StructuredModel,
    validated_model_result,
)
from saliencegate.ports.outcomes import OutcomeRecorder, OutcomeRecordingError
from saliencegate.ports.repository import (
    AppendDisposition,
    EnqueueDelivery,
    GroundingPin,
    LedgerEntry,
    LedgerHead,
    ProjectionDigests,
    RepositoryError,
    RunNotFoundError,
    RunRepository,
)
from saliencegate.repository.projector import (
    Projection,
    apply_entry,
    empty_projection,
    preview_memory_delta,
    validate_complete_projection,
)
from saliencegate.runtime.batching import (
    BatchConfig,
    BatchManifest,
    BatchRequest,
    BatchStatus,
    DeterministicBatcher,
)
from saliencegate.runtime.budget import (
    BudgetGovernor,
    BudgetReservationDeniedError,
    BudgetSettlementError,
)
from saliencegate.runtime.cycles import CycleCoordinator
from saliencegate.runtime.delivery import DeliveryWorker, DeliveryWorkerResult
from saliencegate.signals import DetectionContext

_MAX_REPLAY_EVENTS = 1_000
_MAX_DETECTION_EVENTS = 10_000
_MAX_SIGNALS_PER_EVENT = 64
_MAX_CANDIDATE_MEMORIES = 32
_MAX_NORMALIZED_TRACE_BYTES = 8 * 1024 * 1024
_MAX_SIGNAL_BYTES_PER_EVENT = 256 * 1024
_MAX_SIGNALS_PER_RUN = 4_096
_MAX_SIGNAL_BYTES_PER_RUN = 8 * 1024 * 1024
_MAX_MODEL_OUTPUT_BYTES = 1024 * 1024
_MAX_MODEL_OUTPUT_BYTES_PER_RUN = 16 * 1024 * 1024


class ReplayEngineError(RuntimeError):
    """Base error whose message never embeds trace, model, or adapter content."""


class ReplayEngineInputError(ReplayEngineError):
    def __init__(self) -> None:
        super().__init__("replay engine input failed validation")


class ReplayEngineModelError(ReplayEngineError):
    def __init__(self) -> None:
        super().__init__("replay model boundary failed")


class ReplayEngineInvariantError(ReplayEngineError):
    def __init__(self) -> None:
        super().__init__("replay engine authoritative state diverged")


async def _drain_cleanup(task: asyncio.Task[None]) -> None:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    with suppress(asyncio.CancelledError, Exception):
        task.result()


class ReplayTraceAdapter(Protocol):
    def normalize(self, native_event: object) -> NormalizedTraceEventDraft: ...

    def resolve_event_id(self, native_event: object, ordinal: int) -> UUID | None: ...

    def resolve_target_request_id(
        self,
        native_event: object,
        target: DeliveryTarget,
    ) -> str | None: ...


class _AttestedReplayTrace(Protocol):
    @property
    def events(self) -> tuple[BaseModel, ...]: ...

    @property
    def expected_event_ids(self) -> tuple[UUID, ...]: ...


class ReplaySignalExtractor(Protocol):
    def extract(self, context: DetectionContext) -> tuple[Signal, ...]: ...


class ReplayTriggerPolicy(Protocol):
    def decide(
        self,
        signals: list[Signal],
        state: RunState,
        budget: BudgetSnapshot,
    ) -> InvocationDecision: ...


class _EngineModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class ReplayEngineConfig(_EngineModel):
    schema_version: Literal["1.0"] = "1.0"
    model_id: ComponentIdentifier
    prompt_template_digest: Sha256Digest
    budget_limits: BudgetLimits
    reservation: BudgetAmounts
    batch: BatchConfig
    requested_delivery_target: DeliveryTarget | None = None

    @model_validator(mode="after")
    def reservation_fits_the_run_budget(self) -> Self:
        fields = (
            "model_calls",
            "input_tokens",
            "output_tokens",
            "canonical_token_equivalents",
            "latency_us",
            "interventions",
            "schema_repairs",
        )
        if self.reservation.model_calls < 1 or any(
            getattr(self.reservation, field_name) > getattr(self.budget_limits, field_name)
            for field_name in fields
        ):
            raise ValueError("memory-cycle reservation exceeds the run budget")
        return self


class _ReplayFixtureState(_EngineModel):
    replay_id: ComponentIdentifier
    fixture_digest: Sha256Digest
    response_count: int = Field(ge=0)
    remaining_count: int = Field(ge=0)

    @model_validator(mode="after")
    def remaining_responses_fit_fixture(self) -> Self:
        if self.remaining_count > self.response_count:
            raise ValueError("remaining replay responses exceed the fixture")
        return self


class ReplayModelPayload(_EngineModel):
    schema_version: Literal["replay-engine-model-payload/v1"] = "replay-engine-model-payload/v1"
    trace_digest: Sha256Digest
    batch: BatchManifest = Field(repr=False)
    candidate_view_digest: Sha256Digest
    candidate_memories: tuple[MemoryRecord, ...] = Field(repr=False)
    grounding_version: ComponentIdentifier
    grounding_configuration_digest: Sha256Digest
    requested_delivery_target: DeliveryTarget | None


class ReplayRoutingBinding(_EngineModel):
    ordinal: int = Field(ge=1, le=_MAX_REPLAY_EVENTS)
    target: DeliveryTarget | None = None
    target_request_id_digest: Sha256Digest | None = None
    adapter_id: ComponentIdentifier | None = None
    adapter_capabilities_digest: Sha256Digest | None = None

    @model_validator(mode="after")
    def binding_is_complete_or_unavailable(self) -> Self:
        values = (
            self.target,
            self.target_request_id_digest,
            self.adapter_id,
            self.adapter_capabilities_digest,
        )
        if any(value is None for value in values) and any(value is not None for value in values):
            raise ValueError("replay routing binding is partial")
        return self


@dataclass(frozen=True, slots=True)
class _FrozenDeliveryAdapter:
    adapter: DeliveryAdapter
    declared_capabilities: AdapterCapabilities

    def capabilities(self) -> AdapterCapabilities:
        return self.declared_capabilities

    async def deliver(self, delivery: DeliveryEnvelope) -> DeliveryReceipt:
        return await self.adapter.deliver(delivery)


def _result_digest(values: object) -> str:
    if not isinstance(values, dict):
        raise ValueError("replay result digest input is not a mapping")
    payload = {key: value for key, value in values.items() if key != "result_digest"}
    return length_prefixed_sha256(
        canonical_json(to_jsonable_python(payload)),
        domain="saliencegate:replay-engine:result:v1",
    )


def _normalized_trace_digest_from_digests(digests: tuple[str, ...]) -> str:
    return canonical_digest(
        {
            "schema_version": "engine-normalized-trace/v1",
            "draft_digests": digests,
        }
    )


def _engine_normalized_trace_digest(
    normalized_digest: str,
    routing_digest: str,
    expected_event_ids: tuple[UUID, ...],
) -> str:
    return canonical_digest(
        {
            "schema_version": "engine-normalized-execution/v1",
            "normalized_trace_digest": normalized_digest,
            "routing_digest": routing_digest,
            "expected_event_ids": tuple(str(event_id) for event_id in expected_event_ids),
        }
    )


def _normalized_event_draft_digest(event: TraceEvent) -> str:
    draft = NormalizedTraceEventDraft(
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
    return canonical_digest(draft)


class ReplayEventResult(_EngineModel):
    event: TraceEvent
    signals: tuple[Signal, ...]
    decision: InvocationDecision
    cycle: CycleRecord | None = None
    delivery: DeliveryRecord | None = None
    model_request_digest: Sha256Digest | None = None

    @model_validator(mode="after")
    def records_belong_to_the_event_run(self) -> Self:
        if (
            self.decision.run_id != self.event.run_id
            or self.decision.event_sequence != self.event.sequence
            or self.decision.created_at != self.event.timestamp
            or any(signal.run_id != self.event.run_id for signal in self.signals)
            or any(self.event.event_id not in signal.evidence_event_ids for signal in self.signals)
            or any(signal.created_at != self.event.timestamp for signal in self.signals)
            or self.decision.invoke is not (self.cycle is not None)
        ):
            raise ValueError("replay event records belong to different runs")
        if self.cycle is not None and (
            self.cycle.run_id != self.event.run_id
            or self.cycle.invocation_decision_id != self.decision.decision_id
            or self.cycle.last_event_sequence != self.event.sequence
            or self.cycle.state not in (CycleState.COMMITTED, CycleState.FAILED)
            or self.cycle.created_at != self.event.timestamp
            or self.cycle.updated_at != self.event.timestamp
        ):
            raise ValueError("replay cycle belongs to a different run")
        if self.cycle is not None and (
            (self.cycle.state is CycleState.COMMITTED and self.cycle.revision != 4)
            or (self.cycle.state is CycleState.FAILED and not 2 <= self.cycle.revision <= 4)
        ):
            raise ValueError("replay cycle has an impossible terminal revision")
        reached_model_start = self.cycle is not None and self.cycle.batch_digest is not None
        if (self.model_request_digest is not None) is not reached_model_start:
            raise ValueError("replay model request digest does not match cycle execution")
        reminder_committed = (
            self.cycle is not None
            and self.cycle.state is CycleState.COMMITTED
            and self.cycle.intervention is not None
            and self.cycle.intervention.action is InterventionAction.REMIND
        )
        if reminder_committed is not (self.delivery is not None):
            raise ValueError("only a committed reminder requires a replay delivery")
        if self.delivery is not None and (
            self.cycle is None
            or self.delivery.run_id != self.event.run_id
            or self.delivery.cycle_id != self.cycle.cycle_id
            or self.cycle.intervention is None
            or self.delivery.intervention_id != self.cycle.intervention.intervention_id
            or self.cycle.intervention.action is not InterventionAction.REMIND
            or self.delivery.created_at != self.event.timestamp
            or self.delivery.updated_at != self.event.timestamp
            or self.delivery.state
            in (DeliveryState.PENDING, DeliveryState.CLAIMED, DeliveryState.ATTEMPTING)
        ):
            raise ValueError("replay delivery belongs to a different run")
        return self


class ReplayRunResult(_EngineModel):
    schema_version: Literal["replay-run-result/v1"] = "replay-run-result/v1"
    run_id: UUID
    trace_digest: Sha256Digest
    trace_attestation_mode: Literal["adapter_manifest", "engine_normalized"]
    trace_event_count: int = Field(ge=1)
    normalized_trace_digest: Sha256Digest
    normalized_draft_digests: tuple[Sha256Digest, ...]
    persisted_event_draft_digests: tuple[Sha256Digest, ...]
    routing_bindings: tuple[ReplayRoutingBinding, ...] = Field(repr=False)
    routing_digest: Sha256Digest
    trace_record_digests: tuple[Sha256Digest, ...] = ()
    trace_expected_event_ids: tuple[UUID, ...] = ()
    events_digest: Sha256Digest
    model_id: ComponentIdentifier
    prompt_template_digest: Sha256Digest
    engine_configuration: ReplayEngineConfig = Field(repr=False)
    engine_configuration_digest: Sha256Digest
    model_execution_mode: Literal["structured_model", "frozen_replay"]
    replay_id: ComponentIdentifier | None = None
    fixture_digest: Sha256Digest | None = None
    fixture_response_count: int | None = Field(default=None, ge=0)
    fixture_consumed_count: int | None = Field(default=None, ge=0)
    events: tuple[ReplayEventResult, ...]
    decisions_json: str = Field(repr=False)
    decisions_digest: Sha256Digest
    projection_digests: ProjectionDigests
    ledger_entry_count: int = Field(ge=1)
    ledger_head: LedgerHead
    outcomes: tuple[InterventionOutcome, ...]
    rebuild_equivalent: bool
    result_digest: Sha256Digest

    @model_validator(mode="after")
    def result_is_replay_safe(self) -> Self:
        expected_result_digest = _result_digest(
            self.model_dump(mode="python", exclude={"result_digest"}, warnings=False)
        )
        if self.result_digest != expected_result_digest:
            raise ValueError("replay result digest does not match")
        decisions = tuple(item.decision.model_dump(mode="json") for item in self.events)
        expected_json = canonical_json(decisions).decode("utf-8")
        if self.decisions_json != expected_json or self.decisions_digest != canonical_digest(
            decisions
        ):
            raise ValueError("replay decision export does not match event results")
        if any(item.event.run_id != self.run_id for item in self.events):
            raise ValueError("replay event result belongs to a different run")
        if self.events_digest != canonical_digest(
            tuple(item.event.model_dump(mode="json") for item in self.events)
        ):
            raise ValueError("replay event digest does not match event results")
        if (
            len(self.normalized_draft_digests) != self.trace_event_count
            or self.normalized_trace_digest
            != _normalized_trace_digest_from_digests(self.normalized_draft_digests)
            or len(self.persisted_event_draft_digests) != self.trace_event_count
            or self.persisted_event_draft_digests
            != tuple(_normalized_event_draft_digest(item.event) for item in self.events)
            or len(self.routing_bindings) != self.trace_event_count
            or tuple(binding.ordinal for binding in self.routing_bindings)
            != tuple(range(1, self.trace_event_count + 1))
            or self.routing_digest
            != canonical_digest(
                tuple(binding.model_dump(mode="json") for binding in self.routing_bindings)
            )
        ):
            raise ValueError("normalized trace attestation does not match")
        if self.trace_attestation_mode == "adapter_manifest":
            expected_manifest_digest = canonical_digest(
                {
                    "schema_version": "1.0",
                    "run_id": str(self.run_id),
                    "record_digests": self.trace_record_digests,
                }
            )
            if (
                len(self.trace_record_digests) != self.trace_event_count
                or len(self.trace_expected_event_ids) != self.trace_event_count
                or self.trace_expected_event_ids
                != tuple(item.event.event_id for item in self.events)
                or self.trace_digest != expected_manifest_digest
            ):
                raise ValueError("adapter trace manifest does not match replay events")
        elif (
            self.trace_record_digests
            or len(self.trace_expected_event_ids) != self.trace_event_count
            or self.trace_expected_event_ids != tuple(item.event.event_id for item in self.events)
            or self.trace_digest
            != _engine_normalized_trace_digest(
                self.normalized_trace_digest,
                self.routing_digest,
                self.trace_expected_event_ids,
            )
        ):
            raise ValueError("engine-normalized trace attestation does not match")
        if (
            self.model_id != self.engine_configuration.model_id
            or self.prompt_template_digest != self.engine_configuration.prompt_template_digest
            or self.engine_configuration_digest
            != canonical_digest(self.engine_configuration.model_dump(mode="json", warnings=False))
        ):
            raise ValueError("replay engine configuration attestation does not match")
        fixture_values = (
            self.replay_id,
            self.fixture_digest,
            self.fixture_response_count,
            self.fixture_consumed_count,
        )
        model_request_digests = tuple(
            item.model_request_digest
            for item in self.events
            if item.model_request_digest is not None
        )
        if len(set(model_request_digests)) != len(model_request_digests):
            raise ValueError("replay model request identities are not unique")
        if self.model_execution_mode == "structured_model":
            if any(value is not None for value in fixture_values):
                raise ValueError("structured model result cannot claim a replay fixture")
        elif (
            any(value is None for value in fixture_values)
            or self.fixture_consumed_count != self.fixture_response_count
            or self.fixture_response_count != len(model_request_digests)
        ):
            raise ValueError("frozen replay fixture must be completely consumed and attested")
        sequences = tuple(item.event.sequence for item in self.events)
        event_ids = tuple(item.event.event_id for item in self.events)
        decision_ids = tuple(item.decision.decision_id for item in self.events)
        signal_ids = tuple(signal.signal_id for item in self.events for signal in item.signals)
        cycle_ids = tuple(item.cycle.cycle_id for item in self.events if item.cycle is not None)
        delivery_ids = tuple(
            item.delivery.delivery_id for item in self.events if item.delivery is not None
        )
        if (
            len(sequences) != self.trace_event_count
            or sequences != tuple(range(1, len(sequences) + 1))
            or len(set(event_ids)) != len(event_ids)
            or len(set(decision_ids)) != len(decision_ids)
            or len(set(signal_ids)) != len(signal_ids)
            or len(set(cycle_ids)) != len(cycle_ids)
            or len(set(delivery_ids)) != len(delivery_ids)
        ):
            raise ValueError("replay events and decisions are not one unique ordered trace")
        event_order = {event_id: ordinal for ordinal, event_id in enumerate(event_ids)}
        memory_cursor = 0
        for item in self.events:
            if item.event.parent_ids != tuple(sorted(item.event.parent_ids, key=str)) or any(
                parent_id not in event_order
                or event_order[parent_id] >= event_order[item.event.event_id]
                for parent_id in item.event.parent_ids
            ):
                raise ValueError("replay event parent graph is not causal")
            if item.decision.decision_id != _semantic_uuid(
                self.trace_digest,
                "decision",
                item.event.sequence,
            ):
                raise ValueError("replay decision identity is not deterministic")
            if item.signals != tuple(
                sorted(
                    item.signals,
                    key=lambda signal: (
                        signal.signal_type.value,
                        signal.detector_version,
                        str(signal.signal_id),
                    ),
                )
            ):
                raise ValueError("replay signal order is not canonical")
            for signal in item.signals:
                if (
                    not set(signal.evidence_event_ids).issubset(event_order)
                    or signal.evidence_event_ids
                    != tuple(sorted(signal.evidence_event_ids, key=event_order.__getitem__))
                    or any(
                        event_order[evidence_id] > event_order[item.event.event_id]
                        for evidence_id in signal.evidence_event_ids
                    )
                ):
                    raise ValueError("replay signal evidence is not canonical")
            cycle = item.cycle
            if cycle is None:
                continue
            if (
                cycle.first_event_sequence != memory_cursor + 1
                or cycle.last_event_sequence != item.event.sequence
                or cycle.policy_version != item.decision.policy_version
                or cycle.configuration_digest != item.decision.configuration_digest
                or cycle.cycle_id
                != cycle_id(
                    self.run_id,
                    cycle.first_event_sequence,
                    cycle.last_event_sequence,
                    cycle.policy_version,
                    cycle.configuration_digest,
                    cycle.grounding_version,
                    cycle.grounding_configuration_digest,
                    cycle.requested_delivery_target,
                )
            ):
                raise ValueError("replay cycle identity or range is inconsistent")
            if cycle.state is CycleState.COMMITTED:
                memory_cursor = cycle.last_event_sequence
            if cycle.intervention is not None and (
                cycle.intervention.intervention_id
                != _semantic_uuid(self.trace_digest, "intervention", cycle.cycle_id)
                or cycle.intervention.created_at != item.event.timestamp
            ):
                raise ValueError("replay intervention identity is not deterministic")
            if any(
                assignment.memory_id
                != _semantic_uuid(
                    self.trace_digest,
                    "memory",
                    cycle.cycle_id,
                    assignment.handle,
                )
                for assignment in cycle.memory_id_assignments
            ):
                raise ValueError("replay memory identity is not deterministic")
        for item, binding in zip(self.events, self.routing_bindings, strict=True):
            if (
                item.cycle is not None
                and item.cycle.requested_delivery_target is not binding.target
            ):
                raise ValueError("routing binding does not match replay records")
            if item.delivery is not None and (
                binding.target is None
                or item.delivery.target is not binding.target
                or binding.target_request_id_digest
                != canonical_digest(item.delivery.target_request_id)
                or binding.adapter_id != item.delivery.adapter_id
                or binding.adapter_capabilities_digest != item.delivery.adapter_capabilities_digest
            ):
                raise ValueError("routing binding does not match replay records")
            delivery = item.delivery
            if delivery is not None and (
                delivery.delivery_id
                != derive_delivery_id(
                    delivery.run_id,
                    delivery.cycle_id,
                    delivery.intervention_id,
                    delivery.target_request_id,
                    delivery.target,
                    delivery.adapter_id,
                    delivery.adapter_capabilities_digest,
                    delivery.rendered_text_digest,
                )
                or (delivery.state is DeliveryState.REJECTED and delivery.revision != 2)
                or (delivery.state is not DeliveryState.REJECTED and delivery.revision != 4)
                or (
                    delivery.claim_id is not None
                    and delivery.claim_id
                    != _semantic_uuid(
                        self.trace_digest,
                        "delivery-claim",
                        delivery.delivery_id,
                    )
                )
                or (
                    delivery.attempt_id is not None
                    and delivery.attempt_id
                    != _semantic_uuid(
                        self.trace_digest,
                        "delivery-attempt",
                        delivery.delivery_id,
                    )
                )
            ):
                raise ValueError("replay delivery identity is not deterministic")
        if any(
            outcome.run_id != self.run_id
            or outcome.evidence_mode is not OutcomeEvidenceMode.POLICY_REPLAY
            or outcome.next_action_fingerprint is not None
            or outcome.repeated_error_status is not RepeatedErrorStatus.UNKNOWN
            or outcome.constraint_status is not ConstraintStatus.UNKNOWN
            or outcome.utility is not None
            or outcome.action_changed is not None
            or outcome.task_reward is not None
            or outcome.task_passed is not None
            for outcome in self.outcomes
        ):
            raise ValueError("replay result contains a causal outcome claim")
        if not self.rebuild_equivalent:
            raise ValueError("replay result requires an equivalent rebuild")
        committed_interventions = {
            item.cycle.intervention.intervention_id
            for item in self.events
            if item.cycle is not None
            and item.cycle.state is CycleState.COMMITTED
            and item.cycle.intervention is not None
        }
        outcome_ids = tuple(outcome.outcome_id for outcome in self.outcomes)
        ordered_interventions = tuple(
            item.cycle.intervention.intervention_id
            for item in self.events
            if item.cycle is not None
            and item.cycle.state is CycleState.COMMITTED
            and item.cycle.intervention is not None
        )
        ordered_intervention_times = tuple(
            item.event.timestamp
            for item in self.events
            if item.cycle is not None
            and item.cycle.state is CycleState.COMMITTED
            and item.cycle.intervention is not None
        )
        if (
            len(set(outcome_ids)) != len(outcome_ids)
            or {outcome.intervention_id for outcome in self.outcomes} != committed_interventions
            or tuple(outcome.intervention_id for outcome in self.outcomes) != ordered_interventions
            or tuple(outcome.created_at for outcome in self.outcomes) != ordered_intervention_times
            or any(
                outcome.outcome_id
                != _semantic_uuid(
                    self.trace_digest,
                    "outcome",
                    outcome.intervention_id,
                )
                for outcome in self.outcomes
            )
        ):
            raise ValueError("replay outcomes do not exactly attest committed interventions")
        represented_ledger_entries = sum(
            2
            + len(item.signals)
            + (0 if item.cycle is None else item.cycle.revision)
            + (0 if item.delivery is None else item.delivery.revision)
            for item in self.events
        ) + len(self.outcomes)
        if (
            self.ledger_entry_count != represented_ledger_entries
            or self.ledger_head.run_id != self.run_id
            or self.ledger_head.entry_count != self.ledger_entry_count
            or self.ledger_head.head_tag.algorithm is not self.projection_digests.overall.algorithm
        ):
            raise ValueError("replay ledger attestation is inconsistent")
        return self


@dataclass(slots=True)
class _RunGuard:
    run_id: UUID | None = None
    cycle_id: str | None = None
    updated_at: datetime | None = None


def _semantic_uuid(trace_digest: str, label: str, *parts: object) -> UUID:
    digest = length_prefixed_sha256(
        trace_digest,
        label,
        *(str(part) for part in parts),
        domain="saliencegate:replay-engine:identity:v1",
    )
    raw = bytearray(bytes.fromhex(digest)[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(raw))


def normalized_trace_digest(drafts: tuple[NormalizedTraceEventDraft, ...]) -> str:
    """Derive an engine-owned identity for one bounded normalized trace."""

    if type(drafts) is not tuple or not drafts or len(drafts) > _MAX_REPLAY_EVENTS:
        raise ReplayEngineInputError()
    total_bytes = 0
    draft_digests: list[str] = []
    try:
        for draft in drafts:
            if not normalized_trace_event_draft_is_bounded(draft):
                raise ValueError
            item = canonical_json(draft)
            total_bytes += len(item)
            if total_bytes > _MAX_NORMALIZED_TRACE_BYTES:
                raise ValueError
            draft_digests.append(canonical_digest(draft))
    except Exception:
        raise ReplayEngineInputError() from None
    return _normalized_trace_digest_from_digests(tuple(draft_digests))


def _projection(run_id: UUID, entries: tuple[LedgerEntry, ...]) -> Projection:
    projected = empty_projection(run_id)
    try:
        for entry in entries:
            projected = apply_entry(projected, entry)
        validate_complete_projection(projected)
    except Exception:
        raise ReplayEngineInvariantError() from None
    return projected


def _memory_assignments(
    trace_digest: str,
    cycle_id: str,
    output: ModelResult,
) -> tuple[MemoryIdAssignment, ...]:
    if output.output is None:  # pragma: no cover - caller status invariant
        raise ReplayEngineInvariantError()
    delta = output.output.delta
    handles = tuple(item.handle for item in delta.creates)
    if delta.private_status_replacement is not None:
        handles += (delta.private_status_replacement.replacement.handle,)
    return tuple(
        MemoryIdAssignment(
            handle=handle,
            memory_id=_semantic_uuid(trace_digest, "memory", cycle_id, handle),
        )
        for handle in handles
    )


def _usage_amounts(
    usage: ModelUsage,
    *,
    interventions: int,
    canonical_token_floor: int = 0,
) -> BudgetAmounts:
    return BudgetAmounts(
        model_calls=1,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        canonical_token_equivalents=max(
            usage.canonical_token_equivalents,
            canonical_token_floor,
        ),
        latency_us=usage.latency_us,
        interventions=interventions,
        schema_repairs=usage.schema_repairs,
    )


def _validated_decision(
    value: object,
    state: RunState,
    budget: BudgetSnapshot,
) -> InvocationDecision:
    decision: InvocationDecision | None = None
    try:
        if type(value) is InvocationDecision:
            decision = InvocationDecision.model_validate_json(value.model_dump_json(warnings=False))
    except Exception:
        pass
    if decision is None or (
        decision.decision_id != state.decision_id
        or decision.run_id != state.run_id
        or decision.event_sequence != state.event_sequence
        or decision.budget_snapshot != budget
        or decision.created_at != state.created_at
    ):
        raise ReplayEngineInputError()
    return decision


def _validated_signals(
    value: object,
    context: DetectionContext,
) -> tuple[Signal, ...]:
    if type(value) is not tuple or len(value) > _MAX_SIGNALS_PER_EVENT:
        raise ReplayEngineInputError()
    known_event_ids = {event.event_id for event in context.events}
    event_order = {event.event_id: ordinal for ordinal, event in enumerate(context.events)}
    current = context.current
    validated: list[Signal] = []
    total_bytes = 0
    try:
        for item in value:
            if (
                type(item) is not Signal
                or type(item.signal_id) is not UUID
                or item.signal_id.version != 4
                or type(item.run_id) is not UUID
                or item.run_id.version != 4
                or type(item.signal_type) is not SignalType
                or type(item.evidence_event_ids) is not tuple
                or not 1 <= len(item.evidence_event_ids) <= MAX_SIGNAL_EVIDENCE_EVENTS
                or any(
                    type(event_id) is not UUID or event_id.version != 4
                    for event_id in item.evidence_event_ids
                )
                or type(item.detector_version) is not str
                or not 1 <= len(item.detector_version) <= 256
                or type(item.reason_code) is not ReasonCode
            ):
                raise ValueError
            encoded = item.model_dump_json(warnings=False)
            total_bytes += len(encoded.encode("utf-8", errors="strict"))
            if total_bytes > _MAX_SIGNAL_BYTES_PER_EVENT:
                raise ValueError
            signal = Signal.model_validate_json(encoded)
            evidence = set(signal.evidence_event_ids)
            if (
                signal.run_id != context.run_id
                or current.event_id not in evidence
                or not evidence.issubset(known_event_ids)
                or signal.created_at != current.timestamp
                or signal.evidence_event_ids
                != tuple(sorted(signal.evidence_event_ids, key=event_order.__getitem__))
            ):
                raise ValueError
            validated.append(signal)
    except Exception:
        raise ReplayEngineInputError() from None
    signal_ids = tuple(signal.signal_id for signal in validated)
    if len(set(signal_ids)) != len(signal_ids):
        raise ReplayEngineInputError()
    return tuple(
        sorted(
            validated,
            key=lambda signal: (
                signal.signal_type.value,
                signal.detector_version,
                str(signal.signal_id),
            ),
        )
    )


def _model_fixture_state(value: object) -> _ReplayFixtureState | None:
    names = (
        "replay_id",
        "fixture_digest",
        "total_responses",
        "remaining_responses",
    )
    try:
        attributes = tuple(getattr(value, name, None) for name in names)
        if all(attribute is None for attribute in attributes):
            return None
        if any(attribute is None for attribute in attributes):
            raise ValueError
        return _ReplayFixtureState.model_validate(
            {
                "replay_id": attributes[0],
                "fixture_digest": attributes[1],
                "response_count": attributes[2],
                "remaining_count": attributes[3],
            }
        )
    except Exception:
        raise ReplayEngineModelError() from None


def _bounded_model_result_output_bytes(
    value: object,
    *,
    byte_limit: int,
    usage_limits: BudgetAmounts,
) -> int:
    try:
        if (
            type(byte_limit) is not int
            or byte_limit < 0
            or type(usage_limits) is not BudgetAmounts
            or type(value) is not ModelResult
            or value.schema_version != "model-result/v1"
            or type(value.status) is not ModelCallStatus
            or type(value.request_digest) is not str
            or len(value.request_digest) != 64
            or any(character not in "0123456789abcdef" for character in value.request_digest)
            or type(value.call_digest) is not str
            or len(value.call_digest) != 64
            or any(character not in "0123456789abcdef" for character in value.call_digest)
            or type(value.usage) is not ModelUsage
        ):
            raise ValueError
        usage_fields = (
            "input_tokens",
            "output_tokens",
            "canonical_token_equivalents",
            "latency_us",
            "schema_repairs",
        )
        if any(
            type(getattr(value.usage, field_name)) is not int
            or not 0 <= getattr(value.usage, field_name) <= getattr(usage_limits, field_name)
            for field_name in usage_fields
        ):
            raise ValueError
        output = value.output
        if output is None:
            if value.status is ModelCallStatus.COMPLETED:
                raise ValueError
            return 0
        if not (
            value.status is ModelCallStatus.COMPLETED
            and type(output) is MemoryCycleOutput
            and output.schema_version == "memory-cycle-output/v1"
            and memory_delta_is_bounded(output.delta)
            and type(output.observation) is GroundingObservation
            and output.observation.schema_version == "grounding-observation/v1"
            and type(output.observation.parse_status) is ProposalParseStatus
            and (
                output.observation.proposal_action is None
                or type(output.observation.proposal_action) is InterventionAction
            )
            and type(output.observation.claims) is tuple
            and len(output.observation.claims) <= 2
            and type(output.observation.confidence) is float
            and 0.0 <= output.observation.confidence <= 1.0
            and all(
                type(claim) is ProposedClaim
                and type(claim.kind) is ClaimKind
                and evidence_reference_is_bounded(claim.evidence)
                for claim in output.observation.claims
            )
        ):
            raise ValueError
        encoded_bytes = len(canonical_json(output))
        if encoded_bytes > byte_limit:
            raise ValueError
        return encoded_bytes
    except Exception:
        raise ReplayEngineModelError() from None


def _advance_ledger_position(
    expected: int,
    observed: object,
    *,
    appended_records: int = 1,
) -> int:
    if (
        type(observed) is not int
        or observed != expected + 1
        or type(appended_records) is not int
        or appended_records < 1
    ):
        raise ReplayEngineInvariantError()
    return expected + appended_records


def _selected_grounding_state(
    projection: Projection,
    receipt: GroundingReceipt,
    *,
    current_sequence: int,
    grounding: GroundingPipeline,
) -> GroundingState:
    current = projection.events_by_sequence.get(current_sequence)
    if current is None:
        raise ReplayEngineInvariantError()
    events = {current.event_id: current}
    memories: dict[UUID, MemoryRecord] = {}
    for claim in receipt.claims:
        reference = claim.evidence
        if reference.source is EvidenceSource.EVENT:
            cited_event = projection.events_by_id.get(reference.source_id)
            if cited_event is not None:
                events[cited_event.event_id] = cited_event
        else:
            cited_memory = projection.memories.get(reference.source_id)
            if cited_memory is not None:
                memories[cited_memory.memory_id] = cited_memory

    configuration = grounding.configuration
    history_window = max(
        configuration.duplicate_window_events,
        configuration.cooldown_events,
    )
    first_sequence = max(1, current_sequence - history_window)
    history: list[ReminderHistory] = []
    for sequence in range(first_sequence, current_sequence):
        prior = projection.reminder_interventions_by_sequence.get(sequence)
        if prior is not None:
            history.append(
                ReminderHistory(
                    schema_version="1.0",
                    intervention_id=prior.intervention_id,
                    run_id=prior.run_id,
                    event_sequence=sequence,
                    claim_digests=tuple(claim_fingerprint(claim) for claim in prior.claims),
                )
            )
    return GroundingState(
        schema_version="1.0",
        events=tuple(sorted(events.values(), key=lambda item: (item.sequence, str(item.event_id)))),
        memories=tuple(sorted(memories.values(), key=lambda item: str(item.memory_id))),
        reminder_history=tuple(history),
    )


class ReplayEngine:
    """A narrow orchestrator over the authoritative deterministic components."""

    __slots__ = (
        "_adapter",
        "_batcher",
        "_config",
        "_delivery_adapter",
        "_extractor",
        "_governor",
        "_grounding",
        "_model",
        "_outcome_recorder",
        "_policy",
        "_repository",
    )

    def __init__(
        self,
        *,
        repository: RunRepository,
        adapter: ReplayTraceAdapter,
        extractor: ReplaySignalExtractor,
        policy: ReplayTriggerPolicy,
        model: StructuredModel,
        grounding: GroundingPipeline,
        config: ReplayEngineConfig,
        delivery_adapter: DeliveryAdapter | None = None,
        batcher: DeterministicBatcher | None = None,
        outcome_recorder: OutcomeRecorder | None = None,
    ) -> None:
        self._repository = repository
        self._adapter = adapter
        self._extractor = extractor
        self._policy = policy
        self._model = model
        self._grounding = grounding
        self._config = ReplayEngineConfig.model_validate_json(config.model_dump_json())
        self._delivery_adapter = delivery_adapter
        self._batcher = DeterministicBatcher() if batcher is None else batcher
        self._outcome_recorder = outcome_recorder
        self._governor = BudgetGovernor()

    async def _state(self, run_id: UUID) -> tuple[Projection, tuple[LedgerEntry, ...]]:
        entries = await self._repository.ledger(run_id)
        return _projection(run_id, entries), entries

    async def _preflight(
        self,
        native_events: tuple[object, ...],
        *,
        trace_digest: str,
    ) -> tuple[
        tuple[NormalizedTraceEventDraft, ...],
        tuple[UUID, ...],
        str,
        str,
        Literal["adapter_manifest", "engine_normalized"],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        try:
            attested_trace_digest = getattr(self._adapter, "trace_digest", None)
        except Exception:
            raise ReplayEngineInputError() from None
        if attested_trace_digest is not None and (
            type(attested_trace_digest) is not str or attested_trace_digest != trace_digest
        ):
            raise ReplayEngineInputError()

        expected_ids: tuple[UUID, ...]
        optional_expected_ids: list[UUID | None] = []
        event_id_resolver: Callable[[object, int], object] | None = None
        trace_record_digests: tuple[str, ...] = ()
        if attested_trace_digest is not None:
            try:
                attested_adapter = cast(_AttestedReplayTrace, self._adapter)
                attested_events = attested_adapter.events
                attested_ids = attested_adapter.expected_event_ids
                if (
                    type(attested_events) is not tuple
                    or type(attested_ids) is not tuple
                    or len(attested_events) != len(native_events)
                    or len(attested_ids) != len(native_events)
                ):
                    raise ValueError
                for supplied, attested in zip(native_events, attested_events, strict=True):
                    if (
                        type(supplied) is not type(attested)
                        or not isinstance(supplied, BaseModel)
                        or supplied.model_dump_json(warnings=False)
                        != attested.model_dump_json(warnings=False)
                    ):
                        raise ValueError
                if any(type(value) is not UUID or value.version != 4 for value in attested_ids):
                    raise ValueError
                record_digests = tuple(
                    getattr(event, "record_digest", None) for event in attested_events
                )
                if any(
                    type(digest) is not str
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                    for digest in record_digests
                ):
                    raise ValueError
                expected_ids = tuple(attested_ids)
                trace_record_digests = cast(tuple[str, ...], record_digests)
            except Exception:
                raise ReplayEngineInputError() from None
        else:
            try:
                candidate_resolver = getattr(self._adapter, "resolve_event_id", None)
                if candidate_resolver is not None and not callable(candidate_resolver):
                    raise ValueError
                event_id_resolver = cast(
                    Callable[[object, int], object] | None,
                    candidate_resolver,
                )
            except Exception:
                raise ReplayEngineInputError() from None

        drafts: list[NormalizedTraceEventDraft] = []
        draft_digests: list[str] = []
        total_bytes = 0
        try:
            for ordinal, native_event in enumerate(native_events, start=1):
                value = self._adapter.normalize(native_event)
                if not normalized_trace_event_draft_is_bounded(value):
                    raise ValueError
                encoded = value.model_dump_json(warnings=False)
                total_bytes += len(encoded.encode("utf-8", errors="strict"))
                if total_bytes > _MAX_NORMALIZED_TRACE_BYTES:
                    raise ValueError
                copied = NormalizedTraceEventDraft.model_validate_json(encoded)
                drafts.append(copied)
                draft_digests.append(canonical_digest(copied))
                if attested_trace_digest is None:
                    expected = (
                        None
                        if event_id_resolver is None
                        else event_id_resolver(native_event, ordinal)
                    )
                    if expected is not None and (
                        type(expected) is not UUID or expected.version != 4
                    ):
                        raise ValueError
                    optional_expected_ids.append(expected)
        except Exception:
            raise ReplayEngineInputError() from None
        first = drafts[0]
        if any(draft.run_id != first.run_id for draft in drafts) or any(
            right.timestamp < left.timestamp for left, right in pairwise(drafts)
        ):
            raise ReplayEngineInputError()
        try:
            await self._repository.ledger(first.run_id)
        except RunNotFoundError:
            pass
        except Exception:
            raise ReplayEngineInvariantError() from None
        else:
            raise ReplayEngineInputError()
        copied_drafts = tuple(drafts)
        copied_digests = tuple(draft_digests)
        normalized_digest = _normalized_trace_digest_from_digests(copied_digests)
        if attested_trace_digest is None:
            expected_ids = tuple(
                supplied
                if supplied is not None
                else _semantic_uuid(normalized_digest, "event", ordinal)
                for ordinal, supplied in enumerate(optional_expected_ids, start=1)
            )
        if len(set(expected_ids)) != len(expected_ids):
            raise ReplayEngineInputError()
        preceding_ids: set[UUID] = set()
        for draft, expected_event_id in zip(copied_drafts, expected_ids, strict=True):
            if any(parent_id not in preceding_ids for parent_id in draft.parent_ids):
                raise ReplayEngineInputError()
            preceding_ids.add(expected_event_id)
        if attested_trace_digest is None:
            return (
                copied_drafts,
                expected_ids,
                normalized_digest,
                normalized_digest,
                "engine_normalized",
                copied_digests,
                (),
            )
        return (
            copied_drafts,
            expected_ids,
            attested_trace_digest,
            normalized_digest,
            "adapter_manifest",
            copied_digests,
            trace_record_digests,
        )

    @staticmethod
    def _assert_owned_prefix(
        projection: Projection,
        processed: tuple[TraceEvent, ...],
    ) -> None:
        if projection.ingestion_cursor != len(processed):
            raise ReplayEngineInvariantError()
        for sequence, expected in enumerate(processed, start=1):
            actual = projection.events_by_sequence.get(sequence)
            if actual is None or actual.event_id != expected.event_id:
                raise ReplayEngineInvariantError()

    @staticmethod
    def _assert_owned_ledger(
        entries: tuple[LedgerEntry, ...],
        expected_position: int,
    ) -> None:
        if len(entries) != expected_position or any(
            entry.position != position for position, entry in enumerate(entries, start=1)
        ):
            raise ReplayEngineInvariantError()

    async def _terminalize_guard(
        self,
        coordinator: CycleCoordinator,
        guard: _RunGuard,
    ) -> None:
        if guard.run_id is None or guard.cycle_id is None or guard.updated_at is None:
            return
        try:
            projection, _entries = await self._state(guard.run_id)
            cycle = projection.cycles.get(guard.cycle_id)
            if cycle is None or cycle.state in (CycleState.COMMITTED, CycleState.FAILED):
                return
            updated_at = max(cycle.updated_at, guard.updated_at)
            if cycle.state is CycleState.PENDING:
                await coordinator.fail(
                    cycle,
                    reason=ReasonCode.MODEL_ERROR,
                    updated_at=updated_at,
                )
            elif cycle.state is CycleState.RESERVED:
                await coordinator.fail(
                    cycle,
                    reason=ReasonCode.MODEL_ERROR,
                    settlement=BudgetAmounts(),
                    updated_at=updated_at,
                )
            elif cycle.state is CycleState.RUNNING:
                await coordinator.fail(
                    cycle,
                    reason=ReasonCode.FAILED_UNKNOWN_COST,
                    settlement=cycle.budget_reservation,
                    updated_at=updated_at,
                )
            else:  # pragma: no cover - closed enum invariant
                raise ReplayEngineInvariantError()
        except Exception:
            raise ReplayEngineInvariantError() from None

    def _model_request(
        self,
        *,
        run_id: UUID,
        cycle_id: str,
        trace_digest: str,
        manifest: BatchManifest,
        projection: Projection,
        pin: GroundingPin,
        requested_target: DeliveryTarget | None,
    ) -> ModelRequest | None:
        byte_limit = min(
            MAX_MODEL_REQUEST_PAYLOAD_BYTES,
            self._config.reservation.input_tokens * 4,
            self._config.reservation.canonical_token_equivalents * 4,
        )
        candidates = tuple(
            sorted(
                (
                    memory
                    for memory in projection.memories.values()
                    if memory.validity is ValidityState.ACTIVE
                    and memory.kind is MemoryKind.PROCEDURAL
                ),
                key=lambda memory: (str(memory.memory_id), memory.revision),
            )
        )[:_MAX_CANDIDATE_MEMORIES]

        def request_for(selected: tuple[MemoryRecord, ...]) -> ModelRequest | None:
            payload = ReplayModelPayload(
                trace_digest=trace_digest,
                batch=manifest,
                candidate_view_digest=canonical_digest(
                    tuple(memory.model_dump(mode="json") for memory in selected)
                ),
                candidate_memories=selected,
                grounding_version=pin.grounding_version,
                grounding_configuration_digest=pin.grounding_configuration_digest,
                requested_delivery_target=requested_target,
            )
            try:
                request = ModelRequest(
                    run_id=run_id,
                    cycle_id=cycle_id,
                    model_call_index=0,
                    model_id=self._config.model_id,
                    prompt_template_digest=self._config.prompt_template_digest,
                    payload=payload.model_dump(mode="json", warnings=False),
                )
            except Exception:
                raise ReplayEngineInvariantError() from None
            return request if len(canonical_json(request)) <= byte_limit else None

        selected: tuple[MemoryRecord, ...] = ()
        request = request_for(selected)
        if request is None:
            return None
        for candidate in candidates:
            proposed = request_for((*selected, candidate))
            if proposed is not None:
                selected = (*selected, candidate)
                request = proposed
        return request

    async def _budget(self, projection: Projection) -> BudgetSnapshot:
        if projection.budget_limits is None:
            return BudgetSnapshot(
                limits=self._config.budget_limits,
                reserved=BudgetAmounts(),
                consumed=BudgetAmounts(),
            )
        snapshot = await self._repository.budget_snapshot(projection.run_id)
        if snapshot.limits != self._config.budget_limits:
            raise ReplayEngineInvariantError()
        return snapshot

    def _freeze_delivery_bindings(
        self,
        native_events: tuple[object, ...],
    ) -> tuple[
        tuple[
            tuple[DeliveryTarget | None, EnqueueDelivery | None, AdapterCapabilities | None],
            ...,
        ],
        tuple[ReplayRoutingBinding, ...],
        str,
    ]:
        target = self._config.requested_delivery_target
        adapter = self._delivery_adapter
        unavailable = tuple((None, None, None) for _native_event in native_events)
        unavailable_attestation = tuple(
            ReplayRoutingBinding(ordinal=ordinal) for ordinal in range(1, len(native_events) + 1)
        )
        if target is None or adapter is None:
            return (
                unavailable,
                unavailable_attestation,
                canonical_digest(
                    tuple(item.model_dump(mode="json") for item in unavailable_attestation)
                ),
            )
        try:
            capabilities = validated_capabilities(adapter.capabilities())
            if target is DeliveryTarget.PRE_ACTION_REPLAN and not (
                capabilities.pre_action_interception
            ):
                raise ValueError
            capabilities_digest = adapter_capabilities_digest(capabilities)
        except Exception:
            return (
                unavailable,
                unavailable_attestation,
                canonical_digest(
                    tuple(item.model_dump(mode="json") for item in unavailable_attestation)
                ),
            )

        frozen: list[
            tuple[DeliveryTarget | None, EnqueueDelivery | None, AdapterCapabilities | None]
        ] = []
        attestations: list[ReplayRoutingBinding] = []
        for ordinal, native_event in enumerate(native_events, start=1):
            try:
                target_request_id = self._adapter.resolve_target_request_id(native_event, target)
                if target_request_id is None:
                    raise ValueError
                binding = enqueue_delivery_binding(
                    target_request_id=target_request_id,
                    capabilities=capabilities,
                )
                frozen.append((target, binding, capabilities))
                attestations.append(
                    ReplayRoutingBinding(
                        ordinal=ordinal,
                        target=target,
                        target_request_id_digest=canonical_digest(target_request_id),
                        adapter_id=capabilities.adapter_id,
                        adapter_capabilities_digest=capabilities_digest,
                    )
                )
            except Exception:
                frozen.append((None, None, None))
                attestations.append(ReplayRoutingBinding(ordinal=ordinal))
        copied_attestations = tuple(attestations)
        return (
            tuple(frozen),
            copied_attestations,
            canonical_digest(tuple(item.model_dump(mode="json") for item in copied_attestations)),
        )

    def _batch(
        self,
        projection: Projection,
        *,
        current_sequence: int,
    ) -> tuple[BatchStatus, BatchManifest | None]:
        events = tuple(
            projection.events_by_sequence[sequence]
            for sequence in range(projection.memory_cursor + 1, current_sequence + 1)
        )
        event_ids = {event.event_id for event in events}
        known_event_ids = set(projection.events_by_id)
        signals = tuple(
            sorted(
                (
                    signal
                    for signal in projection.signals.values()
                    if set(signal.evidence_event_ids).issubset(known_event_ids)
                    and not set(signal.evidence_event_ids).isdisjoint(event_ids)
                ),
                key=lambda item: (item.created_at, str(item.signal_id)),
            )
        )
        active = tuple(
            memory
            for memory in projection.memories.values()
            if memory.validity is ValidityState.ACTIVE
        )
        request = BatchRequest(
            run_id=projection.run_id,
            memory_cursor=projection.memory_cursor,
            events=events,
            historical_event_ids=tuple(
                sorted(
                    {
                        evidence_id
                        for signal in signals
                        for evidence_id in signal.evidence_event_ids
                        if evidence_id not in event_ids
                    },
                    key=str,
                )
            ),
            signals=signals,
            task_requirements=tuple(
                memory for memory in active if memory.kind is MemoryKind.KNOWLEDGE
            ),
            unresolved_state=tuple(
                memory for memory in active if memory.kind is MemoryKind.PRIVATE_STATUS
            ),
        )
        result = self._batcher.build(request, self._config.batch)
        return result.status, result.manifest

    async def _fail_unknown_cost(
        self,
        coordinator: CycleCoordinator,
        cycle: CycleRecord,
        *,
        updated_at: datetime,
    ) -> None:
        await coordinator.fail(
            cycle,
            reason=ReasonCode.FAILED_UNKNOWN_COST,
            settlement=self._config.reservation,
            updated_at=updated_at,
        )

    async def _record_neutral_outcome(
        self,
        intervention_id: UUID,
        run_id: UUID,
        *,
        trace_digest: str,
        created_at: datetime,
    ) -> tuple[InterventionOutcome, int]:
        outcome = InterventionOutcome(
            outcome_id=_semantic_uuid(trace_digest, "outcome", intervention_id),
            run_id=run_id,
            intervention_id=intervention_id,
            repeated_error_status=RepeatedErrorStatus.UNKNOWN,
            constraint_status=ConstraintStatus.UNKNOWN,
            evidence_mode=OutcomeEvidenceMode.POLICY_REPLAY,
            utility=None,
            action_changed=None,
            task_reward=None,
            task_passed=None,
            created_at=created_at,
        )
        receipt = await self._repository.record_outcome(outcome)
        if self._outcome_recorder is not None:
            try:
                await self._outcome_recorder.record(outcome)
            except Exception:
                raise OutcomeRecordingError() from None
        return outcome, receipt.ledger_position

    async def run(
        self,
        native_events: tuple[object, ...],
        *,
        trace_digest: str,
    ) -> ReplayRunResult:
        coordinator = CycleCoordinator(self._repository)
        guard = _RunGuard()
        try:
            return await self._run(
                native_events,
                trace_digest=trace_digest,
                coordinator=coordinator,
                guard=guard,
            )
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(self._terminalize_guard(coordinator, guard))
            await _drain_cleanup(cleanup)
            raise
        except Exception as error:
            await self._terminalize_guard(coordinator, guard)
            if isinstance(error, (ReplayEngineError, OutcomeRecordingError)):
                raise
            raise ReplayEngineInvariantError() from None

    async def _run(
        self,
        native_events: tuple[object, ...],
        *,
        trace_digest: str,
        coordinator: CycleCoordinator,
        guard: _RunGuard,
    ) -> ReplayRunResult:
        if (
            type(native_events) is not tuple
            or not native_events
            or len(native_events) > _MAX_REPLAY_EVENTS
            or type(trace_digest) is not str
            or len(trace_digest) != 64
            or any(character not in "0123456789abcdef" for character in trace_digest)
        ):
            raise ReplayEngineInputError()
        initial_fixture = _model_fixture_state(self._model)
        if (
            initial_fixture is not None
            and initial_fixture.remaining_count != initial_fixture.response_count
        ):
            raise ReplayEngineModelError()
        (
            drafts,
            expected_event_ids,
            effective_trace_digest,
            normalized_digest,
            trace_attestation_mode,
            normalized_draft_digests,
            trace_record_digests,
        ) = await self._preflight(
            native_events,
            trace_digest=trace_digest,
        )
        frozen_delivery_bindings, routing_bindings, routing_digest = self._freeze_delivery_bindings(
            native_events
        )
        if trace_attestation_mode == "engine_normalized":
            effective_trace_digest = _engine_normalized_trace_digest(
                normalized_digest,
                routing_digest,
                expected_event_ids,
            )
        trace_digest = effective_trace_digest

        event_results: list[ReplayEventResult] = []
        run_id: UUID | None = None
        outcomes: list[InterventionOutcome] = []
        processed_events: list[TraceEvent] = []
        expected_ledger_position = 0
        run_signal_count = 0
        run_signal_bytes = 0
        run_model_output_bytes = 0

        for draft, expected_event_id, frozen_delivery_binding in zip(
            drafts,
            expected_event_ids,
            frozen_delivery_bindings,
            strict=True,
        ):
            try:
                append = await self._repository.append(draft, event_id=expected_event_id)
            except Exception:
                raise ReplayEngineInputError() from None
            if append.disposition is not AppendDisposition.APPENDED:
                raise ReplayEngineInputError()
            expected_ledger_position = _advance_ledger_position(
                expected_ledger_position,
                append.ledger_position,
            )
            if append.event.event_id != expected_event_id:
                raise ReplayEngineInvariantError()
            event = append.event
            if run_id is None:
                run_id = event.run_id
            elif event.run_id != run_id:
                raise ReplayEngineInputError()
            processed_events.append(event)

            projection, _entries = await self._state(run_id)
            self._assert_owned_prefix(projection, tuple(processed_events))
            self._assert_owned_ledger(_entries, expected_ledger_position)
            detection_events = tuple(
                projection.events_by_sequence[sequence]
                for sequence in sorted(projection.events_by_sequence)
                if sequence <= event.sequence
            )[-_MAX_DETECTION_EVENTS:]
            context = DetectionContext(run_id=run_id, events=detection_events)
            try:
                signals = _validated_signals(self._extractor.extract(context), context)
            except Exception:
                raise ReplayEngineInputError() from None
            run_signal_count += len(signals)
            run_signal_bytes += len(canonical_json(signals))
            if (
                run_signal_count > _MAX_SIGNALS_PER_RUN
                or run_signal_bytes > _MAX_SIGNAL_BYTES_PER_RUN
            ):
                raise ReplayEngineInputError()
            for signal in signals:
                signal_receipt = await self._repository.record_signal(signal)
                expected_ledger_position = _advance_ledger_position(
                    expected_ledger_position,
                    signal_receipt.ledger_position,
                )

            projection, _entries = await self._state(run_id)
            self._assert_owned_prefix(projection, tuple(processed_events))
            self._assert_owned_ledger(_entries, expected_ledger_position)
            budget = await self._budget(projection)
            decision_state = RunState(
                schema_version="1.0",
                decision_id=_semantic_uuid(trace_digest, "decision", event.sequence),
                run_id=run_id,
                current_event_id=event.event_id,
                event_sequence=event.sequence,
                decision_ordinal=len(projection.decisions) + 1,
                created_at=event.timestamp,
            )
            try:
                decision = _validated_decision(
                    self._policy.decide(list(signals), decision_state, budget),
                    decision_state,
                    budget,
                )
            except Exception:
                raise ReplayEngineInputError() from None
            decision_receipt = await self._repository.record_invocation_decision(decision)
            expected_ledger_position = _advance_ledger_position(
                expected_ledger_position,
                decision_receipt.ledger_position,
            )
            if not decision.invoke:
                event_results.append(
                    ReplayEventResult(event=event, signals=signals, decision=decision)
                )
                continue

            try:
                self._governor.reserve(budget, self._config.reservation)
            except BudgetReservationDeniedError:
                raise ReplayEngineInputError() from None
            requested_target, delivery_binding, _capabilities = frozen_delivery_binding
            pin = self._grounding.pin(requested_target)
            guard.run_id = run_id
            guard.cycle_id = cycle_id(
                run_id,
                projection.memory_cursor + 1,
                event.sequence,
                decision.policy_version,
                decision.configuration_digest,
                pin.grounding_version,
                pin.grounding_configuration_digest,
                pin.requested_delivery_target,
            )
            guard.updated_at = event.timestamp
            pending = await coordinator.begin(decision, grounding=pin, created_at=event.timestamp)
            expected_ledger_position = _advance_ledger_position(
                expected_ledger_position,
                pending.ledger_position,
            )
            if pending.cycle.cycle_id != guard.cycle_id:
                raise ReplayEngineInvariantError()
            reserved = await coordinator.reserve(
                pending,
                reservation=self._config.reservation,
                updated_at=event.timestamp,
            )
            expected_ledger_position = _advance_ledger_position(
                expected_ledger_position,
                reserved.ledger_position,
            )

            projection, _entries = await self._state(run_id)
            self._assert_owned_prefix(projection, tuple(processed_events))
            self._assert_owned_ledger(_entries, expected_ledger_position)
            status, manifest = self._batch(projection, current_sequence=event.sequence)
            if status is BatchStatus.MANDATORY_INPUT_OVERFLOW or manifest is None:
                held = await self._repository.budget_snapshot(run_id)
                self._governor.settle(
                    held,
                    self._config.reservation,
                    BudgetAmounts(),
                    model_call_latencies_us=(),
                )
                failed = await coordinator.fail(
                    reserved,
                    reason=ReasonCode.MANDATORY_INPUT_OVERFLOW,
                    settlement=BudgetAmounts(),
                    updated_at=event.timestamp,
                )
                expected_ledger_position = _advance_ledger_position(
                    expected_ledger_position,
                    failed.ledger_position,
                )
                event_results.append(
                    ReplayEventResult(
                        event=event,
                        signals=signals,
                        decision=decision,
                        cycle=failed.cycle,
                    )
                )
                continue

            request = self._model_request(
                run_id=run_id,
                cycle_id=reserved.cycle.cycle_id,
                trace_digest=trace_digest,
                manifest=manifest,
                projection=projection,
                pin=pin,
                requested_target=requested_target,
            )
            if request is None:
                held = await self._repository.budget_snapshot(run_id)
                self._governor.settle(
                    held,
                    self._config.reservation,
                    BudgetAmounts(),
                    model_call_latencies_us=(),
                )
                failed = await coordinator.fail(
                    reserved,
                    reason=ReasonCode.MANDATORY_INPUT_OVERFLOW,
                    settlement=BudgetAmounts(),
                    updated_at=event.timestamp,
                )
                expected_ledger_position = _advance_ledger_position(
                    expected_ledger_position,
                    failed.ledger_position,
                )
                event_results.append(
                    ReplayEventResult(
                        event=event,
                        signals=signals,
                        decision=decision,
                        cycle=failed.cycle,
                    )
                )
                continue

            running = await coordinator.start(
                reserved,
                batch_digest=manifest.batch_digest,
                updated_at=event.timestamp,
            )
            expected_ledger_position = _advance_ledger_position(
                expected_ledger_position,
                running.ledger_position,
            )
            try:
                request_bytes = len(canonical_json(request))
                raw_model_result = await self._model.generate(request)
                output_bytes = _bounded_model_result_output_bytes(
                    raw_model_result,
                    byte_limit=min(
                        _MAX_MODEL_OUTPUT_BYTES,
                        self._config.reservation.output_tokens * 4,
                        self._config.reservation.canonical_token_equivalents * 4 - request_bytes,
                    ),
                    usage_limits=self._config.reservation,
                )
                run_model_output_bytes += output_bytes
                if run_model_output_bytes > min(
                    _MAX_MODEL_OUTPUT_BYTES_PER_RUN,
                    self._config.budget_limits.output_tokens * 4,
                    self._config.budget_limits.canonical_token_equivalents * 4,
                ):
                    raise ReplayEngineModelError()
                model_result = validated_model_result(raw_model_result)
                if model_result.request_digest != request.request_digest:
                    raise ReplayEngineModelError()
                canonical_token_floor = (request_bytes + output_bytes + 3) // 4
                if canonical_token_floor > self._config.reservation.canonical_token_equivalents:
                    raise ReplayEngineModelError()
            except Exception:
                await self._fail_unknown_cost(
                    coordinator,
                    running.cycle,
                    updated_at=event.timestamp,
                )
                raise ReplayEngineModelError() from None

            actual_without_intervention = _usage_amounts(
                model_result.usage,
                interventions=0,
                canonical_token_floor=canonical_token_floor,
            )
            held = await self._repository.budget_snapshot(run_id)
            try:
                self._governor.settle(
                    held,
                    self._config.reservation,
                    actual_without_intervention,
                    model_call_latencies_us=(model_result.usage.latency_us,),
                )
            except BudgetSettlementError:
                await self._fail_unknown_cost(
                    coordinator,
                    running.cycle,
                    updated_at=event.timestamp,
                )
                raise ReplayEngineModelError() from None

            if model_result.status is not ModelCallStatus.COMPLETED:
                reason = (
                    ReasonCode.MODEL_TIMEOUT
                    if model_result.status is ModelCallStatus.MODEL_TIMEOUT
                    else ReasonCode.MODEL_ERROR
                )
                failed = await coordinator.fail(
                    running,
                    reason=reason,
                    settlement=actual_without_intervention,
                    model_call_digests=(model_result.call_digest,),
                    model_call_latencies_us=(model_result.usage.latency_us,),
                    updated_at=event.timestamp,
                )
                expected_ledger_position = _advance_ledger_position(
                    expected_ledger_position,
                    failed.ledger_position,
                )
                event_results.append(
                    ReplayEventResult(
                        event=event,
                        signals=signals,
                        decision=decision,
                        cycle=failed.cycle,
                        model_request_digest=request.request_digest,
                    )
                )
                continue

            output = model_result.output
            if output is None:  # pragma: no cover - result invariant
                raise ReplayEngineInvariantError()
            assignments = _memory_assignments(
                trace_digest,
                running.cycle.cycle_id,
                model_result,
            )
            if output.delta.run_id != run_id or output.delta.created_at != event.timestamp:
                failed = await coordinator.fail(
                    running,
                    reason=ReasonCode.INVALID_STRUCTURED_OUTPUT,
                    settlement=actual_without_intervention,
                    model_call_digests=(model_result.call_digest,),
                    model_call_latencies_us=(model_result.usage.latency_us,),
                    updated_at=event.timestamp,
                )
                expected_ledger_position = _advance_ledger_position(
                    expected_ledger_position,
                    failed.ledger_position,
                )
                event_results.append(
                    ReplayEventResult(
                        event=event,
                        signals=signals,
                        decision=decision,
                        cycle=failed.cycle,
                        model_request_digest=request.request_digest,
                    )
                )
                continue
            projection, _entries = await self._state(run_id)
            self._assert_owned_prefix(projection, tuple(processed_events))
            self._assert_owned_ledger(_entries, expected_ledger_position)
            try:
                preview = preview_memory_delta(
                    projection,
                    output.delta,
                    assignments,
                    last_event_sequence=event.sequence,
                )
            except RepositoryError:
                failed = await coordinator.fail(
                    running,
                    reason=ReasonCode.MEMORY_CONFLICT,
                    settlement=actual_without_intervention,
                    model_call_digests=(model_result.call_digest,),
                    model_call_latencies_us=(model_result.usage.latency_us,),
                    updated_at=event.timestamp,
                )
                expected_ledger_position = _advance_ledger_position(
                    expected_ledger_position,
                    failed.ledger_position,
                )
                event_results.append(
                    ReplayEventResult(
                        event=event,
                        signals=signals,
                        decision=decision,
                        cycle=failed.cycle,
                        model_request_digest=request.request_digest,
                    )
                )
                continue

            observation = output.observation
            receipt = GroundingReceipt(
                receipt_version=GROUNDING_RECEIPT_VERSION,
                parse_status=observation.parse_status,
                proposal_action=observation.proposal_action,
                claims=observation.claims,
                confidence=observation.confidence,
                requested_delivery_target=requested_target,
                model_call_index=0,
                model_call_digest=model_result.call_digest,
            )
            intervention_time = event.timestamp
            grounding_context = GroundingContext(
                schema_version="1.0",
                intervention_id=_semantic_uuid(
                    trace_digest,
                    "intervention",
                    running.cycle.cycle_id,
                ),
                run_id=run_id,
                cycle_id=running.cycle.cycle_id,
                current_event_sequence=event.sequence,
                created_at=intervention_time,
                requested_delivery_target=requested_target,
                model_call_index=0,
                model_call_digest=model_result.call_digest,
            )
            try:
                grounding_state = _selected_grounding_state(
                    preview,
                    receipt,
                    current_sequence=event.sequence,
                    grounding=self._grounding,
                )
                intervention = self._grounding.replay_receipt(
                    receipt,
                    context=grounding_context,
                    state=grounding_state,
                )
            except Exception:
                failed = await coordinator.fail(
                    running,
                    reason=ReasonCode.INVALID_STRUCTURED_OUTPUT,
                    settlement=actual_without_intervention,
                    model_call_digests=(model_result.call_digest,),
                    model_call_latencies_us=(model_result.usage.latency_us,),
                    updated_at=intervention_time,
                )
                expected_ledger_position = _advance_ledger_position(
                    expected_ledger_position,
                    failed.ledger_position,
                )
                event_results.append(
                    ReplayEventResult(
                        event=event,
                        signals=signals,
                        decision=decision,
                        cycle=failed.cycle,
                        model_request_digest=request.request_digest,
                    )
                )
                continue

            intervention_count = int(intervention.action is InterventionAction.REMIND)
            actual = _usage_amounts(
                model_result.usage,
                interventions=intervention_count,
                canonical_token_floor=canonical_token_floor,
            )
            try:
                self._governor.settle(
                    held,
                    self._config.reservation,
                    actual,
                    model_call_latencies_us=(model_result.usage.latency_us,),
                )
            except BudgetSettlementError:
                await self._fail_unknown_cost(
                    coordinator,
                    running.cycle,
                    updated_at=intervention_time,
                )
                raise ReplayEngineModelError() from None
            enqueue = delivery_binding if intervention_count else None
            if intervention_count and enqueue is None:
                raise ReplayEngineInvariantError()
            try:
                committed = await coordinator.commit(
                    running,
                    settlement=actual,
                    validated_delta=output.delta,
                    memory_id_assignments=assignments,
                    intervention=intervention,
                    delivery=enqueue,
                    model_call_digests=(model_result.call_digest,),
                    model_call_latencies_us=(model_result.usage.latency_us,),
                    updated_at=intervention_time,
                )
            except Exception:
                try:
                    failed = await coordinator.fail(
                        running,
                        reason=ReasonCode.INVALID_STRUCTURED_OUTPUT,
                        settlement=actual_without_intervention,
                        model_call_digests=(model_result.call_digest,),
                        model_call_latencies_us=(model_result.usage.latency_us,),
                        updated_at=intervention_time,
                    )
                    expected_ledger_position = _advance_ledger_position(
                        expected_ledger_position,
                        failed.ledger_position,
                    )
                except Exception:
                    raise ReplayEngineInvariantError() from None
                event_results.append(
                    ReplayEventResult(
                        event=event,
                        signals=signals,
                        decision=decision,
                        cycle=failed.cycle,
                        model_request_digest=request.request_digest,
                    )
                )
                continue

            expected_ledger_position = _advance_ledger_position(
                expected_ledger_position,
                committed.ledger_position,
                appended_records=(2 if committed.delivery is not None else 1),
            )
            delivery_result: DeliveryWorkerResult | None = None
            if committed.delivery is not None:
                if (
                    self._delivery_adapter is None or _capabilities is None
                ):  # pragma: no cover - binding invariant
                    raise ReplayEngineInvariantError()
                delivery_ids = iter(
                    (
                        _semantic_uuid(
                            trace_digest,
                            "delivery-claim",
                            committed.delivery.delivery_id,
                        ),
                        _semantic_uuid(
                            trace_digest,
                            "delivery-attempt",
                            committed.delivery.delivery_id,
                        ),
                    )
                )
                delivery_result = await DeliveryWorker(
                    repository=self._repository,
                    adapter=_FrozenDeliveryAdapter(
                        adapter=self._delivery_adapter,
                        declared_capabilities=_capabilities,
                    ),
                    id_factory=delivery_ids.__next__,
                ).deliver(
                    run_id,
                    committed.delivery.delivery_id,
                    now=intervention_time,
                )
                expected_ledger_position += delivery_result.delivery.revision - 1
            outcome, outcome_position = await self._record_neutral_outcome(
                intervention.intervention_id,
                run_id,
                trace_digest=trace_digest,
                created_at=intervention_time,
            )
            expected_ledger_position = _advance_ledger_position(
                expected_ledger_position,
                outcome_position,
            )
            outcomes.append(outcome)
            event_results.append(
                ReplayEventResult(
                    event=event,
                    signals=signals,
                    decision=decision,
                    cycle=committed.cycle,
                    delivery=(None if delivery_result is None else delivery_result.delivery),
                    model_request_digest=request.request_digest,
                )
            )

        if run_id is None:  # pragma: no cover - non-empty input invariant
            raise ReplayEngineInputError()
        completed_fixture = _model_fixture_state(self._model)
        if (initial_fixture is None) is not (completed_fixture is None):
            raise ReplayEngineModelError()
        if initial_fixture is not None and (
            completed_fixture is None
            or completed_fixture.replay_id != initial_fixture.replay_id
            or completed_fixture.fixture_digest != initial_fixture.fixture_digest
            or completed_fixture.response_count != initial_fixture.response_count
            or completed_fixture.remaining_count != 0
        ):
            raise ReplayEngineModelError()
        rebuild = await self._repository.rebuild(run_id)
        if not rebuild.equivalent or rebuild.entries_replayed != expected_ledger_position:
            raise ReplayEngineInvariantError()
        final_projection, _entries = await self._state(run_id)
        self._assert_owned_prefix(final_projection, tuple(processed_events))
        self._assert_owned_ledger(_entries, expected_ledger_position)
        try:
            ledger_head = await self._repository.ledger_head(run_id)
        except Exception:
            raise ReplayEngineInvariantError() from None
        if (
            ledger_head.run_id != run_id
            or ledger_head.entry_count != len(_entries)
            or ledger_head.chain_tag != _entries[-1].chain_tag
        ):
            raise ReplayEngineInvariantError()
        decisions = tuple(item.decision.model_dump(mode="json") for item in event_results)
        decisions_json = canonical_json(decisions).decode("utf-8")
        result_values: dict[str, object] = {
            "schema_version": "replay-run-result/v1",
            "run_id": run_id,
            "trace_digest": trace_digest,
            "trace_attestation_mode": trace_attestation_mode,
            "trace_event_count": len(drafts),
            "normalized_trace_digest": normalized_digest,
            "normalized_draft_digests": normalized_draft_digests,
            "persisted_event_draft_digests": tuple(
                _normalized_event_draft_digest(item.event) for item in event_results
            ),
            "routing_bindings": routing_bindings,
            "routing_digest": routing_digest,
            "trace_record_digests": trace_record_digests,
            "trace_expected_event_ids": expected_event_ids,
            "events_digest": canonical_digest(
                tuple(item.event.model_dump(mode="json") for item in event_results)
            ),
            "model_id": self._config.model_id,
            "prompt_template_digest": self._config.prompt_template_digest,
            "engine_configuration": self._config,
            "engine_configuration_digest": canonical_digest(
                self._config.model_dump(mode="json", warnings=False)
            ),
            "model_execution_mode": (
                "structured_model" if completed_fixture is None else "frozen_replay"
            ),
            "replay_id": None if completed_fixture is None else completed_fixture.replay_id,
            "fixture_digest": (
                None if completed_fixture is None else completed_fixture.fixture_digest
            ),
            "fixture_response_count": (
                None if completed_fixture is None else completed_fixture.response_count
            ),
            "fixture_consumed_count": (
                None if completed_fixture is None else completed_fixture.response_count
            ),
            "events": tuple(event_results),
            "decisions_json": decisions_json,
            "decisions_digest": canonical_digest(decisions),
            "projection_digests": rebuild.after,
            "ledger_entry_count": len(_entries),
            "ledger_head": ledger_head,
            "outcomes": tuple(outcomes),
            "rebuild_equivalent": rebuild.equivalent,
        }
        result_values["result_digest"] = _result_digest(result_values)
        return ReplayRunResult.model_validate(result_values)


__all__ = [
    "ReplayEngine",
    "ReplayEngineConfig",
    "ReplayEngineError",
    "ReplayEngineInputError",
    "ReplayEngineInvariantError",
    "ReplayEngineModelError",
    "ReplayEventResult",
    "ReplayModelPayload",
    "ReplayRoutingBinding",
    "ReplayRunResult",
    "ReplaySignalExtractor",
    "ReplayTraceAdapter",
    "ReplayTriggerPolicy",
    "normalized_trace_digest",
]
