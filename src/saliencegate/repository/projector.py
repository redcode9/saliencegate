from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from types import MappingProxyType
from uuid import UUID

from pydantic import ValidationError

from saliencegate.domain import (
    BudgetAmounts,
    BudgetLimits,
    BudgetSnapshot,
    CycleRecord,
    CycleState,
    DeliveryRecord,
    DeliveryState,
    EvidenceReference,
    EvidenceSource,
    ExpirationAction,
    InterventionAction,
    InterventionDecision,
    InterventionOutcome,
    InvocationDecision,
    MemoryCreate,
    MemoryDelta,
    MemoryIdAssignment,
    MemoryKind,
    MemoryRecord,
    ReasonCode,
    Signal,
    TraceEvent,
    TrustLabel,
    ValidityState,
    canonical_digest,
    canonical_json,
)
from saliencegate.intervention.claims import (
    GROUNDING_RECEIPT_SELECTOR_VERSION,
    GROUNDING_RECEIPT_VERSION,
    DeterministicSelectorProvenance,
    GroundingReceipt,
    claim_fingerprint,
)
from saliencegate.intervention.grounding import (
    GroundingConfig,
    GroundingContext,
    GroundingState,
    ReminderHistory,
    ResolvedGroundingConfiguration,
    resolve_grounding_configuration,
    verify_grounded_intervention,
)
from saliencegate.ports.repository import (
    CrossRunReferenceError,
    LedgerEntry,
    MemoryHit,
    MemoryQuery,
    MemorySnapshot,
    ProjectionDigests,
    ProjectionInvariantError,
    RevisionConflictError,
)
from saliencegate.repository.integrity import IntegrityContext

_MAPPING_PROXY_TYPE: type[object] = type(MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class Projection:
    run_id: UUID
    events_by_id: Mapping[UUID, TraceEvent] = field(default_factory=dict)
    events_by_sequence: Mapping[int, TraceEvent] = field(default_factory=dict)
    events_by_source: Mapping[str, TraceEvent] = field(default_factory=dict)
    event_positions: Mapping[UUID, int] = field(default_factory=dict)
    signals: Mapping[UUID, Signal] = field(default_factory=dict)
    decisions: Mapping[UUID, InvocationDecision] = field(default_factory=dict)
    decisions_by_event_sequence: Mapping[int, InvocationDecision] = field(
        init=False,
        repr=False,
        compare=False,
    )
    cycle_history: Mapping[tuple[str, int], CycleRecord] = field(default_factory=dict)
    cycles: Mapping[str, CycleRecord] = field(default_factory=dict)
    memory_history: Mapping[tuple[UUID, int], MemoryRecord] = field(default_factory=dict)
    memories: Mapping[UUID, MemoryRecord] = field(default_factory=dict)
    interventions: Mapping[UUID, InterventionDecision] = field(default_factory=dict)
    reminder_interventions_by_sequence: Mapping[int, InterventionDecision] = field(
        default_factory=dict,
        repr=False,
    )
    outcomes: Mapping[UUID, InterventionOutcome] = field(default_factory=dict)
    delivery_history: Mapping[tuple[UUID, int], DeliveryRecord] = field(default_factory=dict)
    deliveries: Mapping[UUID, DeliveryRecord] = field(default_factory=dict)
    current_private_status_id: UUID | None = None
    budget_limits: BudgetLimits | None = None
    ingestion_cursor: int = 0
    memory_cursor: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "events_by_id",
            "events_by_sequence",
            "events_by_source",
            "event_positions",
            "signals",
            "decisions",
            "cycle_history",
            "cycles",
            "memory_history",
            "memories",
            "interventions",
            "reminder_interventions_by_sequence",
            "outcomes",
            "delivery_history",
            "deliveries",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, _MAPPING_PROXY_TYPE):
                object.__setattr__(self, field_name, MappingProxyType(dict(value)))
        decisions_by_event_sequence: dict[int, InvocationDecision] = {}
        for decision in self.decisions.values():
            if decision.event_sequence in decisions_by_event_sequence:
                raise ProjectionInvariantError("invocation event already has a decision")
            decisions_by_event_sequence[decision.event_sequence] = decision
        object.__setattr__(
            self,
            "decisions_by_event_sequence",
            MappingProxyType(decisions_by_event_sequence),
        )


def empty_projection(run_id: UUID) -> Projection:
    return Projection(run_id=run_id)


def _record_values(record: MemoryRecord, **updates: object) -> MemoryRecord:
    values = record.model_dump(mode="python")
    values.update(updates)
    try:
        return MemoryRecord.model_validate(values)
    except ValidationError:
        pass
    raise ProjectionInvariantError("projected memory record is invalid")


def _validate_evidence(
    projection: Projection,
    evidence: tuple[EvidenceReference, ...],
    *,
    max_event_sequence: int | None = None,
) -> None:
    for reference in evidence:
        if reference.source is EvidenceSource.EVENT:
            event = projection.events_by_id.get(reference.source_id)
            if event is None:
                raise CrossRunReferenceError("event evidence")
            if max_event_sequence is not None and event.sequence > max_event_sequence:
                raise ProjectionInvariantError("event evidence falls after the cycle range")
        elif (reference.source_id, reference.revision) not in projection.memory_history:
            raise CrossRunReferenceError("memory evidence")


def _resolved_cycle_grounding(
    cycle: CycleRecord,
) -> tuple[GroundingConfig, ResolvedGroundingConfiguration]:
    try:
        configuration = GroundingConfig.model_validate_json(
            canonical_json(cycle.grounding_configuration)
        )
        resolved = resolve_grounding_configuration(configuration)
        if (
            cycle.grounding_version != resolved.pipeline_version
            or cycle.grounding_configuration_digest != resolved.configuration_digest
            or canonical_json(cycle.grounding_configuration)
            != canonical_json(resolved.configuration)
        ):
            raise ValueError("grounding pin mismatch")
        return configuration, resolved
    except Exception:
        raise ProjectionInvariantError("cycle grounding pin failed validation") from None


def _authoritatively_verify_intervention(
    projection: Projection,
    cycle: CycleRecord,
    intervention: InterventionDecision,
) -> None:
    """Rebuild a verdict from a pre-model pin and bounded source-owned state."""

    try:
        configuration, resolved_configuration = _resolved_cycle_grounding(cycle)
        receipt = GroundingReceipt.model_validate_json(
            canonical_json(intervention.grounding_receipt)
        )
        selector_provenance = (
            None
            if cycle.selector_provenance is None
            else DeterministicSelectorProvenance.model_validate_json(
                canonical_json(cycle.selector_provenance)
            )
        )
        if selector_provenance is not None and canonical_json(
            selector_provenance
        ) != canonical_json(cycle.selector_provenance):
            raise ValueError("cycle selector provenance is not exact")
        model_provenance_invalid = receipt.receipt_version == GROUNDING_RECEIPT_VERSION and (
            selector_provenance is not None
            or receipt.model_call_index is None
            or receipt.model_call_digest is None
            or receipt.model_call_index >= len(cycle.model_call_digests)
            or cycle.model_call_digests[receipt.model_call_index] != receipt.model_call_digest
        )
        selector_provenance_invalid = (
            receipt.receipt_version == GROUNDING_RECEIPT_SELECTOR_VERSION
            and (selector_provenance is None or receipt.selector_provenance != selector_provenance)
        )
        if (
            intervention.grounding_version != cycle.grounding_version
            or intervention.grounding_configuration_digest != cycle.grounding_configuration_digest
            or canonical_json(intervention.grounding_configuration)
            != canonical_json(cycle.grounding_configuration)
            or receipt.requested_delivery_target is not cycle.requested_delivery_target
            or model_provenance_invalid
            or selector_provenance_invalid
        ):
            raise ValueError("grounding receipt does not match its cycle pin")

        current_event = projection.events_by_sequence.get(cycle.last_event_sequence)
        if current_event is None:
            raise ValueError("current grounding event is unavailable")
        selected_events: dict[UUID, TraceEvent] = {current_event.event_id: current_event}
        selected_memories: dict[UUID, MemoryRecord] = {}
        for claim in receipt.claims:
            reference = claim.evidence
            if reference.source is EvidenceSource.EVENT:
                event = projection.events_by_id.get(reference.source_id)
                if event is not None:
                    selected_events[event.event_id] = event
            else:
                memory = projection.memories.get(reference.source_id)
                if memory is not None:
                    selected_memories[memory.memory_id] = memory

        history_window = max(
            configuration.duplicate_window_events,
            configuration.cooldown_events,
        )
        first_history_sequence = max(1, cycle.last_event_sequence - history_window)
        reminder_history: list[ReminderHistory] = []
        for sequence in range(first_history_sequence, cycle.last_event_sequence):
            prior = projection.reminder_interventions_by_sequence.get(sequence)
            if prior is None:
                continue
            reminder_history.append(
                ReminderHistory(
                    schema_version="1.0",
                    intervention_id=prior.intervention_id,
                    run_id=prior.run_id,
                    event_sequence=sequence,
                    claim_digests=tuple(claim_fingerprint(claim) for claim in prior.claims),
                )
            )

        context = GroundingContext(
            schema_version=(
                "1.0" if receipt.receipt_version == GROUNDING_RECEIPT_VERSION else "2.0"
            ),
            intervention_id=intervention.intervention_id,
            run_id=intervention.run_id,
            cycle_id=cycle.cycle_id,
            current_event_sequence=cycle.last_event_sequence,
            created_at=intervention.created_at,
            requested_delivery_target=cycle.requested_delivery_target,
            model_call_index=receipt.model_call_index,
            model_call_digest=receipt.model_call_digest,
            selector_provenance=receipt.selector_provenance,
        )
        state = GroundingState(
            schema_version="1.0",
            events=tuple(
                sorted(
                    selected_events.values(),
                    key=lambda item: (item.sequence, str(item.event_id)),
                )
            ),
            memories=tuple(
                sorted(selected_memories.values(), key=lambda item: str(item.memory_id))
            ),
            reminder_history=tuple(reminder_history),
        )
        verify_grounded_intervention(
            intervention,
            context=context,
            state=state,
            expected_configuration=resolved_configuration,
        )
    except Exception:
        raise ProjectionInvariantError(
            "grounded intervention failed authoritative verification"
        ) from None


def _validate_memory_target(
    projection: Projection,
    memory_id: UUID,
    expected_revision: int,
) -> MemoryRecord:
    existing = projection.memories.get(memory_id)
    if existing is None:
        raise RevisionConflictError(memory_id, expected_revision, None)
    if existing.revision != expected_revision:
        raise RevisionConflictError(memory_id, expected_revision, existing.revision)
    if existing.validity is not ValidityState.ACTIVE:
        raise ProjectionInvariantError("memory mutations require an active target")
    return existing


def _store_memory(
    history: dict[tuple[UUID, int], MemoryRecord],
    latest: dict[UUID, MemoryRecord],
    record: MemoryRecord,
) -> None:
    key = (record.memory_id, record.revision)
    if key in history:
        raise ProjectionInvariantError("memory revision already exists")
    history[key] = record
    latest[record.memory_id] = record


def _new_memory(
    *,
    run_id: UUID,
    memory_id: UUID,
    create: MemoryCreate,
    created_at: datetime,
) -> MemoryRecord:
    try:
        return MemoryRecord(
            memory_id=memory_id,
            run_id=run_id,
            kind=create.kind,
            content=create.content,
            provenance=create.provenance,
            confidence=create.confidence,
            validity=ValidityState.ACTIVE,
            revision=1,
            created_at=created_at,
            updated_at=created_at,
            expires_at=create.expires_at,
            trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
        )
    except ValidationError:
        pass
    raise ProjectionInvariantError("projected memory record is invalid")


def preview_memory_delta(
    projection: Projection,
    delta: MemoryDelta,
    memory_id_assignments: tuple[MemoryIdAssignment, ...],
    *,
    last_event_sequence: int,
) -> Projection:
    """Project a validated memory delta without advancing any ledger-owned state."""

    if delta.run_id != projection.run_id:
        raise CrossRunReferenceError("memory delta")
    histories = dict(projection.memory_history)
    memories = dict(projection.memories)
    current_private = projection.current_private_status_id
    assignment_handles = tuple(item.handle for item in memory_id_assignments)
    assignment_ids = tuple(item.memory_id for item in memory_id_assignments)
    expected_handles = tuple(item.handle for item in delta.creates)
    if delta.private_status_replacement is not None:
        expected_handles += (delta.private_status_replacement.replacement.handle,)
    if len(set(assignment_handles)) != len(assignment_handles):
        raise ProjectionInvariantError("memory ID assignment handles must be unique")
    if len(set(assignment_ids)) != len(assignment_ids):
        raise ProjectionInvariantError("assigned memory IDs must be unique")
    if assignment_handles != expected_handles:
        raise ProjectionInvariantError("memory ID assignments must exactly match created handles")
    assignments = dict(zip(assignment_handles, assignment_ids, strict=True))

    replacement = delta.private_status_replacement
    replacement_target = replacement.expected_memory_id if replacement is not None else None
    touched_ids = {item.memory_id for item in delta.updates} | {
        item.memory_id for item in delta.invalidations
    }
    if replacement_target is not None and replacement_target in touched_ids:
        raise ProjectionInvariantError("private-status replacement cannot overlap another mutation")

    for create in delta.creates:
        memory_id = assignments[create.handle]
        if memory_id in memories:
            raise ProjectionInvariantError("assigned memory ID already exists")
        _validate_evidence(
            projection,
            create.provenance,
            max_event_sequence=last_event_sequence,
        )
        _store_memory(
            histories,
            memories,
            _new_memory(
                run_id=projection.run_id,
                memory_id=memory_id,
                create=create,
                created_at=delta.created_at,
            ),
        )

    for update in delta.updates:
        existing = _validate_memory_target(
            replace(projection, memories=memories),
            update.memory_id,
            update.expected_revision,
        )
        if delta.created_at < existing.updated_at:
            raise ProjectionInvariantError("memory update timestamp moved backwards")
        provenance = existing.provenance if update.provenance is None else update.provenance
        _validate_evidence(
            replace(projection, memory_history=histories),
            provenance,
            max_event_sequence=last_event_sequence,
        )
        if update.expiration.action is ExpirationAction.KEEP:
            expires_at = existing.expires_at
        elif update.expiration.action is ExpirationAction.CLEAR:
            expires_at = None
        else:
            expires_at = update.expiration.value
        updated = _record_values(
            existing,
            content=existing.content if update.content is None else update.content,
            provenance=provenance,
            confidence=existing.confidence if update.confidence is None else update.confidence,
            expires_at=expires_at,
            trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
            revision=existing.revision + 1,
            updated_at=delta.created_at,
        )
        _store_memory(histories, memories, updated)

    for invalidation in delta.invalidations:
        existing = _validate_memory_target(
            replace(projection, memories=memories),
            invalidation.memory_id,
            invalidation.expected_revision,
        )
        if delta.created_at < existing.updated_at:
            raise ProjectionInvariantError("memory invalidation timestamp moved backwards")
        invalidated = _record_values(
            existing,
            validity=ValidityState.INVALIDATED,
            invalidated_at=delta.created_at,
            revision=existing.revision + 1,
            updated_at=delta.created_at,
        )
        _store_memory(histories, memories, invalidated)
        if current_private == existing.memory_id:
            current_private = None

    if replacement is not None:
        if replacement.expected_memory_id is None:
            if current_private is not None:
                raise ProjectionInvariantError("an active private status already exists")
        else:
            existing = _validate_memory_target(
                replace(projection, memories=memories),
                replacement.expected_memory_id,
                replacement.expected_revision or 0,
            )
            if (
                existing.kind is not MemoryKind.PRIVATE_STATUS
                or current_private != existing.memory_id
            ):
                raise ProjectionInvariantError("private-status replacement target is not current")
            if delta.created_at < existing.updated_at:
                raise ProjectionInvariantError("private-status timestamp moved backwards")
            superseded = _record_values(
                existing,
                validity=ValidityState.SUPERSEDED,
                revision=existing.revision + 1,
                updated_at=delta.created_at,
            )
            _store_memory(histories, memories, superseded)

        create = replacement.replacement
        memory_id = assignments[create.handle]
        if memory_id in memories:
            raise ProjectionInvariantError("assigned private-status ID already exists")
        _validate_evidence(
            replace(projection, memory_history=histories),
            create.provenance,
            max_event_sequence=last_event_sequence,
        )
        created = _new_memory(
            run_id=projection.run_id,
            memory_id=memory_id,
            create=create,
            created_at=delta.created_at,
        )
        _store_memory(histories, memories, created)
        current_private = memory_id

    return replace(
        projection,
        memory_history=histories,
        memories=memories,
        current_private_status_id=current_private,
    )


def _apply_delta(projection: Projection, delta: MemoryDelta, cycle: CycleRecord) -> Projection:
    return preview_memory_delta(
        projection,
        delta,
        cycle.memory_id_assignments,
        last_event_sequence=cycle.last_event_sequence,
    )


def _apply_event(projection: Projection, entry: LedgerEntry, event: TraceEvent) -> Projection:
    if event.sequence != projection.ingestion_cursor + 1:
        raise ProjectionInvariantError("event sequence is not contiguous")
    if event.sequence in projection.events_by_sequence:
        raise ProjectionInvariantError("event sequence already exists")
    if event.event_id in projection.events_by_id:
        raise ProjectionInvariantError("event ID already exists")
    if event.source_event_id in projection.events_by_source:
        raise ProjectionInvariantError("source event ID already exists")
    if any(parent_id not in projection.events_by_id for parent_id in event.parent_ids):
        raise CrossRunReferenceError("event parent")
    by_id = dict(projection.events_by_id)
    by_sequence = dict(projection.events_by_sequence)
    by_source = dict(projection.events_by_source)
    positions = dict(projection.event_positions)
    by_id[event.event_id] = event
    by_sequence[event.sequence] = event
    by_source[event.source_event_id] = event
    positions[event.event_id] = entry.position
    return replace(
        projection,
        events_by_id=by_id,
        events_by_sequence=by_sequence,
        events_by_source=by_source,
        event_positions=positions,
        ingestion_cursor=event.sequence,
    )


_CYCLE_TRANSITIONS = {
    CycleState.PENDING: frozenset((CycleState.RESERVED, CycleState.FAILED)),
    CycleState.RESERVED: frozenset((CycleState.RUNNING, CycleState.FAILED)),
    CycleState.RUNNING: frozenset((CycleState.COMMITTED, CycleState.FAILED)),
}

_DELIVERY_TRANSITIONS = {
    DeliveryState.PENDING: frozenset((DeliveryState.CLAIMED, DeliveryState.REJECTED)),
    DeliveryState.CLAIMED: frozenset((DeliveryState.ATTEMPTING, DeliveryState.REJECTED)),
    DeliveryState.ATTEMPTING: frozenset(
        (DeliveryState.DELIVERED, DeliveryState.UNKNOWN, DeliveryState.FAILED)
    ),
    DeliveryState.UNKNOWN: frozenset(
        (DeliveryState.CLAIMED, DeliveryState.DELIVERED, DeliveryState.FAILED)
    ),
}

_BUDGET_FIELDS = (
    "model_calls",
    "input_tokens",
    "output_tokens",
    "canonical_token_equivalents",
    "latency_us",
    "interventions",
    "schema_repairs",
)


def _add_amounts(*values: BudgetAmounts) -> BudgetAmounts:
    return BudgetAmounts(
        **{
            field_name: sum(getattr(value, field_name) for value in values)
            for field_name in _BUDGET_FIELDS
        }
    )


def budget_snapshot(
    projection: Projection,
    *,
    limits: BudgetLimits | None = None,
) -> BudgetSnapshot:
    resolved_limits = projection.budget_limits if limits is None else limits
    if resolved_limits is None:
        raise ProjectionInvariantError("run budget limits are unavailable")
    active_reservations = tuple(
        cycle.budget_reservation
        for cycle in projection.cycles.values()
        if cycle.state in (CycleState.RESERVED, CycleState.RUNNING)
        and cycle.budget_reservation is not None
    )
    settlements = tuple(
        cycle.budget_settlement
        for cycle in projection.cycles.values()
        if cycle.state in (CycleState.COMMITTED, CycleState.FAILED)
        and cycle.budget_settlement is not None
    )
    return BudgetSnapshot(
        limits=resolved_limits,
        reserved=_add_amounts(*active_reservations),
        consumed=_add_amounts(*settlements),
    )


def _validate_cycle_budget(
    cycle: CycleRecord,
    decision: InvocationDecision,
    projection: Projection,
    previous: CycleRecord | None,
) -> None:
    reservation = cycle.budget_reservation
    if reservation is not None and (previous is None or previous.budget_reservation is None):
        if reservation.model_calls < 1:
            raise ProjectionInvariantError("cycle reservation requires a model call")
        snapshot = budget_snapshot(projection)
        if any(
            getattr(reservation, field_name)
            > getattr(snapshot.limits, field_name)
            - getattr(snapshot.reserved, field_name)
            - getattr(snapshot.consumed, field_name)
            for field_name in _BUDGET_FIELDS
        ):
            raise ProjectionInvariantError("cycle reservation exceeds the decision budget")
    settlement = cycle.budget_settlement
    if settlement is not None and (
        reservation is None
        or any(
            getattr(settlement, field_name) > getattr(reservation, field_name)
            for field_name in _BUDGET_FIELDS
        )
    ):
        raise ProjectionInvariantError("cycle settlement exceeds its reservation")
    if len(cycle.model_call_digests) != len(cycle.model_call_latencies_us):
        raise ProjectionInvariantError("cycle model-call receipts are incomplete")
    if settlement is not None and sum(cycle.model_call_latencies_us) > settlement.latency_us:
        raise ProjectionInvariantError("model-call latency exceeds settled cycle latency")
    if any(
        latency_us > decision.budget_snapshot.limits.max_call_latency_us
        for latency_us in cycle.model_call_latencies_us
    ):
        raise ProjectionInvariantError("model call exceeds the per-call latency limit")
    if cycle.state is CycleState.COMMITTED:
        if settlement is None:  # pragma: no cover - record invariant
            raise ProjectionInvariantError("committed cycle is missing its settlement")
        if settlement.model_calls < 1:
            raise ProjectionInvariantError("committed cycle must settle a model call")
        if len(cycle.model_call_digests) != settlement.model_calls:
            raise ProjectionInvariantError(
                "committed cycle call digests do not match settled calls"
            )
        if cycle.intervention is None:  # pragma: no cover - record invariant
            raise ProjectionInvariantError("committed cycle is missing its intervention")
        expected_interventions = int(cycle.intervention.action is InterventionAction.REMIND)
        if settlement.interventions != expected_interventions:
            raise ProjectionInvariantError(
                "committed cycle intervention usage does not match its verdict"
            )
    elif cycle.state is CycleState.FAILED and settlement is not None:
        if (
            cycle.failure_reason is not ReasonCode.FAILED_UNKNOWN_COST
            and settlement.interventions != 0
        ):
            raise ProjectionInvariantError("failed cycle cannot consume an intervention")
        if cycle.failure_reason is not ReasonCode.FAILED_UNKNOWN_COST and (
            len(cycle.model_call_digests) != settlement.model_calls
        ):
            raise ProjectionInvariantError("failed cycle call digests do not match settled calls")
        if (
            cycle.failure_reason is not ReasonCode.FAILED_UNKNOWN_COST
            and previous is not None
            and previous.state is CycleState.RESERVED
            and (
                settlement.model_calls != 0
                or settlement.input_tokens != 0
                or settlement.output_tokens != 0
                or settlement.canonical_token_equivalents != 0
                or settlement.schema_repairs != 0
            )
        ):
            raise ProjectionInvariantError("failure before running cannot consume model usage")
        if (
            cycle.failure_reason is not ReasonCode.FAILED_UNKNOWN_COST
            and previous is not None
            and previous.state is CycleState.RUNNING
            and settlement.model_calls < 1
        ):
            raise ProjectionInvariantError("running failure must settle a model call")


def _apply_cycle(projection: Projection, cycle: CycleRecord) -> Projection:
    decision = projection.decisions.get(cycle.invocation_decision_id)
    if decision is None:
        raise CrossRunReferenceError("cycle invocation decision")
    if not decision.invoke:
        raise ProjectionInvariantError("cycle requires an invoking decision")
    if (
        cycle.policy_version != decision.policy_version
        or cycle.configuration_digest != decision.configuration_digest
    ):
        raise ProjectionInvariantError("cycle policy does not match its invocation decision")
    _resolved_cycle_grounding(cycle)
    if cycle.created_at < decision.created_at:
        raise ProjectionInvariantError("cycle cannot precede its invocation decision")
    if any(
        sequence not in projection.events_by_sequence
        for sequence in range(cycle.first_event_sequence, cycle.last_event_sequence + 1)
    ):
        raise CrossRunReferenceError("cycle event range")
    if decision.event_sequence != cycle.last_event_sequence:
        raise ProjectionInvariantError("cycle must end at its invocation event")
    history = dict(projection.cycle_history)
    cycles = dict(projection.cycles)
    previous = cycles.get(cycle.cycle_id)
    if previous is None:
        if cycle.revision != 1 or cycle.state is not CycleState.PENDING:
            raise ProjectionInvariantError("a cycle must begin pending at revision 1")
        if cycle.first_event_sequence != projection.memory_cursor + 1:
            raise ProjectionInvariantError("cycle does not continue the memory cursor")
        if any(
            existing.invocation_decision_id == cycle.invocation_decision_id
            for existing in projection.cycles.values()
        ):
            raise ProjectionInvariantError("invocation decision already has a cycle")
        if any(
            existing.state in (CycleState.PENDING, CycleState.RESERVED, CycleState.RUNNING)
            for existing in projection.cycles.values()
        ):
            raise ProjectionInvariantError("run already has an active cycle")
    else:
        if cycle.revision != previous.revision + 1:
            raise ProjectionInvariantError("cycle revision is not contiguous")
        immutable_fields = (
            "run_id",
            "invocation_decision_id",
            "policy_version",
            "configuration_digest",
            "grounding_version",
            "grounding_configuration_digest",
            "requested_delivery_target",
            "first_event_sequence",
            "last_event_sequence",
            "created_at",
        )
        if any(getattr(cycle, name) != getattr(previous, name) for name in immutable_fields):
            raise ProjectionInvariantError("cycle identity fields changed across revisions")
        if canonical_json(cycle.grounding_configuration) != canonical_json(
            previous.grounding_configuration
        ):
            raise ProjectionInvariantError("cycle identity fields changed across revisions")
        if cycle.updated_at < previous.updated_at:
            raise ProjectionInvariantError("cycle update timestamp moved backwards")
        if (
            previous.budget_reservation is not None
            and cycle.budget_reservation != previous.budget_reservation
        ):
            raise ProjectionInvariantError("cycle budget reservation changed")
        if previous.batch_digest is not None and cycle.batch_digest != previous.batch_digest:
            raise ProjectionInvariantError("cycle batch digest changed")
        if cycle.state not in _CYCLE_TRANSITIONS.get(previous.state, frozenset()):
            raise ProjectionInvariantError("invalid cycle state transition")
        if cycle.state is CycleState.FAILED:
            if previous.state is CycleState.PENDING and (
                cycle.budget_reservation is not None
                or cycle.budget_settlement is not None
                or cycle.batch_digest is not None
                or cycle.model_call_digests
                or cycle.model_call_latencies_us
            ):
                raise ProjectionInvariantError(
                    "failure before reservation cannot introduce budget or batch state"
                )
            if previous.state is CycleState.RESERVED and (
                cycle.batch_digest is not None
                or cycle.model_call_digests
                or cycle.model_call_latencies_us
            ):
                raise ProjectionInvariantError(
                    "failure before running cannot introduce batch or model-call state"
                )
            if (
                cycle.failure_reason is ReasonCode.FAILED_UNKNOWN_COST
                and previous.state is not CycleState.RUNNING
            ):
                raise ProjectionInvariantError(
                    "failed_unknown_cost requires a previously running cycle"
                )
    _validate_cycle_budget(cycle, decision, projection, previous)
    key = (cycle.cycle_id, cycle.revision)
    if key in history:
        raise ProjectionInvariantError("cycle revision already exists")

    updated = projection
    if cycle.state is CycleState.COMMITTED:
        if cycle.first_event_sequence != projection.memory_cursor + 1:
            raise ProjectionInvariantError("committed cycle does not continue the memory cursor")
        if cycle.validated_delta is None:  # pragma: no cover - record invariant
            raise ProjectionInvariantError("committed cycle is missing its memory delta")
        if previous is None:  # pragma: no cover - transition invariant
            raise ProjectionInvariantError("committed cycle is missing its running revision")
        if not previous.updated_at <= cycle.validated_delta.created_at <= cycle.updated_at:
            raise ProjectionInvariantError("memory delta timestamp falls outside its cycle")
        updated = _apply_delta(projection, cycle.validated_delta, cycle)
        if cycle.intervention is None:  # pragma: no cover - record invariant
            raise ProjectionInvariantError("committed cycle is missing its intervention")
        if not previous.updated_at <= cycle.intervention.created_at <= cycle.updated_at:
            raise ProjectionInvariantError("intervention timestamp falls outside its cycle")
        if cycle.intervention.created_at < cycle.validated_delta.created_at:
            raise ProjectionInvariantError("intervention cannot precede its memory delta")
        _authoritatively_verify_intervention(updated, cycle, cycle.intervention)
        interventions = dict(updated.interventions)
        if cycle.intervention.intervention_id in interventions:
            raise ProjectionInvariantError("intervention ID already exists")
        interventions[cycle.intervention.intervention_id] = cycle.intervention
        reminder_interventions = dict(updated.reminder_interventions_by_sequence)
        if cycle.intervention.action is InterventionAction.REMIND:
            if cycle.last_event_sequence in reminder_interventions:
                raise ProjectionInvariantError("event sequence already has a reminder")
            reminder_interventions[cycle.last_event_sequence] = cycle.intervention
        updated = replace(
            updated,
            interventions=interventions,
            reminder_interventions_by_sequence=reminder_interventions,
            memory_cursor=cycle.last_event_sequence,
        )

    history[key] = cycle
    cycles[cycle.cycle_id] = cycle
    return replace(updated, cycle_history=history, cycles=cycles)


def apply_entry(projection: Projection, entry: LedgerEntry) -> Projection:
    record = entry.record
    if record.run_id != projection.run_id:
        raise CrossRunReferenceError("ledger record")
    if isinstance(record, TraceEvent):
        return _apply_event(projection, entry, record)
    if isinstance(record, Signal):
        if record.signal_id in projection.signals:
            raise ProjectionInvariantError("signal ID already exists")
        if any(event_id not in projection.events_by_id for event_id in record.evidence_event_ids):
            raise CrossRunReferenceError("signal evidence")
        signals = dict(projection.signals)
        signals[record.signal_id] = record
        return replace(projection, signals=signals)
    if isinstance(record, InvocationDecision):
        if record.decision_id in projection.decisions:
            raise ProjectionInvariantError("invocation decision ID already exists")
        event = projection.events_by_sequence.get(record.event_sequence)
        if event is None:
            raise CrossRunReferenceError("invocation event")
        if record.created_at < event.timestamp:
            raise ProjectionInvariantError("invocation decision precedes its event")
        if record.event_sequence in projection.decisions_by_event_sequence:
            raise ProjectionInvariantError("invocation event already has a decision")
        limits = projection.budget_limits
        if limits is None:
            limits = record.budget_snapshot.limits
        elif limits != record.budget_snapshot.limits:
            raise ProjectionInvariantError("run budget limits changed")
        if record.budget_snapshot != budget_snapshot(projection, limits=limits):
            raise ProjectionInvariantError("invocation decision budget snapshot is stale")
        decisions = dict(projection.decisions)
        decisions[record.decision_id] = record
        return replace(projection, decisions=decisions, budget_limits=limits)
    if isinstance(record, CycleRecord):
        return _apply_cycle(projection, record)
    if isinstance(record, InterventionOutcome):
        if record.outcome_id in projection.outcomes:
            raise ProjectionInvariantError("outcome ID already exists")
        if record.intervention_id not in projection.interventions:
            raise CrossRunReferenceError("outcome intervention")
        outcomes = dict(projection.outcomes)
        outcomes[record.outcome_id] = record
        return replace(projection, outcomes=outcomes)
    if isinstance(record, DeliveryRecord):
        cycle = projection.cycles.get(record.cycle_id)
        if cycle is None:
            raise CrossRunReferenceError("delivery cycle")
        intervention = projection.interventions.get(record.intervention_id)
        if intervention is None:
            raise CrossRunReferenceError("delivery intervention")
        if (
            cycle.state is not CycleState.COMMITTED
            or cycle.intervention is None
            or cycle.intervention != intervention
        ):
            raise ProjectionInvariantError("delivery does not match its committed cycle")
        if (
            intervention.action is not InterventionAction.REMIND
            or intervention.delivery_target is None
            or record.target is not intervention.delivery_target
        ):
            raise ProjectionInvariantError("delivery is not authorized by its intervention")
        if intervention.rendered_text is None or record.rendered_text_digest != canonical_digest(
            intervention.rendered_text
        ):
            raise ProjectionInvariantError("delivery reminder digest is not authoritative")
        history = dict(projection.delivery_history)
        deliveries = dict(projection.deliveries)
        previous = deliveries.get(record.delivery_id)
        expected_revision = 1 if previous is None else previous.revision + 1
        if record.revision != expected_revision:
            raise ProjectionInvariantError("delivery revision is not contiguous")
        if previous is None:
            if any(
                existing.intervention_id == record.intervention_id
                for existing in deliveries.values()
            ):
                raise ProjectionInvariantError("intervention already has a delivery")
            if record.state is not DeliveryState.PENDING:
                raise ProjectionInvariantError("a delivery must begin pending at revision 1")
            if record.created_at != cycle.updated_at or record.updated_at != record.created_at:
                raise ProjectionInvariantError("delivery was not enqueued with its cycle commit")
        else:
            immutable_fields = (
                "run_id",
                "cycle_id",
                "intervention_id",
                "rendered_text_digest",
                "target_request_id",
                "target",
                "adapter_id",
                "adapter_deduplicates",
                "adapter_deduplication_guarantee",
                "adapter_supports_pre_action",
                "adapter_contract_version",
                "adapter_capabilities_digest",
                "created_at",
            )
            if any(getattr(record, name) != getattr(previous, name) for name in immutable_fields):
                raise ProjectionInvariantError("delivery identity fields changed across revisions")
            if record.updated_at < previous.updated_at:
                raise ProjectionInvariantError("delivery update timestamp moved backwards")
            if record.state not in _DELIVERY_TRANSITIONS.get(previous.state, frozenset()):
                raise ProjectionInvariantError("invalid delivery state transition")
            if record.state is DeliveryState.CLAIMED:
                if any(
                    existing.claim_id == record.claim_id
                    for existing in history.values()
                    if existing.claim_id is not None
                ):
                    raise ProjectionInvariantError("delivery claim owner token was reused")
                if previous.state is DeliveryState.UNKNOWN:
                    if not record.adapter_deduplicates:
                        raise ProjectionInvariantError(
                            "non-deduplicating delivery cannot retry an unknown attempt"
                        )
                    if record.claim_id == previous.claim_id:
                        raise ProjectionInvariantError(
                            "unknown delivery retry requires a new claim owner"
                        )
                if record.claim_id is None or record.attempt_id is not None:
                    raise ProjectionInvariantError("invalid delivery claim ownership")
            elif record.state is DeliveryState.ATTEMPTING:
                if any(
                    existing.attempt_id == record.attempt_id
                    for existing in history.values()
                    if existing.attempt_id is not None
                ):
                    raise ProjectionInvariantError("delivery attempt token was reused")
                if record.attempt_count != previous.attempt_count + 1:
                    raise ProjectionInvariantError("delivery attempt count is not contiguous")
                if (
                    record.claim_id != previous.claim_id
                    or record.attempt_id is None
                    or record.attempt_id == previous.attempt_id
                ):
                    raise ProjectionInvariantError("invalid delivery attempt ownership")
            elif record.attempt_count != previous.attempt_count:
                raise ProjectionInvariantError("delivery attempt count changed outside an attempt")
            if record.state in (
                DeliveryState.DELIVERED,
                DeliveryState.FAILED,
                DeliveryState.UNKNOWN,
            ) and (
                record.claim_id != previous.claim_id or record.attempt_id != previous.attempt_id
            ):
                raise ProjectionInvariantError("delivery completion ownership changed")
            if record.state is DeliveryState.REJECTED:
                if previous.state is DeliveryState.PENDING and record.claim_id is not None:
                    raise ProjectionInvariantError("unclaimed delivery rejection gained an owner")
                if previous.state is DeliveryState.CLAIMED and (
                    record.claim_id != previous.claim_id
                ):
                    raise ProjectionInvariantError("claimed delivery rejection changed owner")
        history[(record.delivery_id, record.revision)] = record
        deliveries[record.delivery_id] = record
        return replace(
            projection,
            delivery_history=history,
            deliveries=deliveries,
        )
    raise ProjectionInvariantError("unsupported ledger record")


def validate_complete_projection(projection: Projection) -> None:
    """Reject a settled projection that is missing either half of a reminder outbox commit."""

    pending_by_cycle: dict[str, DeliveryRecord] = {}
    for (delivery_id, revision), delivery in projection.delivery_history.items():
        if revision != 1:
            continue
        if delivery.delivery_id != delivery_id or delivery.cycle_id in pending_by_cycle:
            raise ProjectionInvariantError("cycle has an ambiguous initial delivery")
        pending_by_cycle[delivery.cycle_id] = delivery

    for cycle in projection.cycles.values():
        if cycle.state is not CycleState.COMMITTED or cycle.intervention is None:
            continue
        initial = pending_by_cycle.get(cycle.cycle_id)
        if cycle.intervention.action is InterventionAction.REMIND:
            if initial is None or initial.intervention_id != cycle.intervention.intervention_id:
                raise ProjectionInvariantError("committed reminder is missing its delivery outbox")
        elif initial is not None:
            raise ProjectionInvariantError("silent cycle cannot own a delivery outbox")

    committed_cycle_ids = {
        cycle.cycle_id
        for cycle in projection.cycles.values()
        if cycle.state is CycleState.COMMITTED
        and cycle.intervention is not None
        and cycle.intervention.action is InterventionAction.REMIND
    }
    if set(pending_by_cycle) != committed_cycle_ids:
        raise ProjectionInvariantError("delivery outbox does not match committed reminders")


def _projection_key(value: object) -> object:
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [_projection_key(item) for item in value],
        }
    if isinstance(value, UUID):
        return {"type": "uuid", "value": str(value)}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    raise TypeError("unsupported projection index key")


def _models(values: object) -> object:
    if isinstance(values, Mapping):
        return [
            {
                "key": _projection_key(key),
                "record": value.model_dump(mode="json"),
            }
            for key, value in sorted(values.items(), key=lambda item: str(item[0]))
        ]
    raise TypeError("projection component must be a dictionary")


def projection_digests(
    projection: Projection,
    integrity: IntegrityContext,
    *,
    ledger_position: int,
) -> ProjectionDigests:
    events = integrity.tag(
        {
            "records": [
                projection.events_by_sequence[index].model_dump(mode="json")
                for index in sorted(projection.events_by_sequence)
            ],
            "id_index": {
                str(event_id): event.model_dump(mode="json")
                for event_id, event in sorted(
                    projection.events_by_id.items(),
                    key=lambda item: str(item[0]),
                )
            },
            "source_index": {
                source_id: event.model_dump(mode="json")
                for source_id, event in sorted(projection.events_by_source.items())
            },
            "positions": {
                str(event_id): position
                for event_id, position in sorted(
                    projection.event_positions.items(),
                    key=lambda item: str(item[0]),
                )
            },
        },
        domain="saliencegate:projection:events:v1",
    )
    memory = integrity.tag(
        {
            "history": _models(projection.memory_history),
            "latest": {
                str(memory_id): record.model_dump(mode="json")
                for memory_id, record in sorted(
                    projection.memories.items(),
                    key=lambda item: str(item[0]),
                )
            },
            "current_private_status_id": (
                None
                if projection.current_private_status_id is None
                else str(projection.current_private_status_id)
            ),
        },
        domain="saliencegate:projection:memory:v1",
    )
    signals = integrity.tag(
        _models(projection.signals),
        domain="saliencegate:projection:signals:v1",
    )
    decisions = integrity.tag(
        _models(projection.decisions),
        domain="saliencegate:projection:decisions:v1",
    )
    cycles = integrity.tag(
        {
            "history": _models(projection.cycle_history),
            "latest": {
                cycle_id: record.model_dump(mode="json")
                for cycle_id, record in sorted(projection.cycles.items())
            },
        },
        domain="saliencegate:projection:cycles:v1",
    )
    interventions = integrity.tag(
        _models(projection.interventions),
        domain="saliencegate:projection:interventions:v1",
    )
    budgets = integrity.tag(
        {
            "snapshot": (
                None
                if projection.budget_limits is None
                else budget_snapshot(projection).model_dump(mode="json")
            ),
            "cycles": [
                {
                    "cycle_id": cycle.cycle_id,
                    "revision": cycle.revision,
                    "reservation": cycle.budget_reservation,
                    "settlement": cycle.budget_settlement,
                    "model_call_digests": cycle.model_call_digests,
                    "model_call_latencies_us": cycle.model_call_latencies_us,
                }
                for _, cycle in sorted(projection.cycles.items())
            ],
        },
        domain="saliencegate:projection:budgets:v1",
    )
    cursors = integrity.tag(
        {
            "ledger_position": ledger_position,
            "ingestion_cursor": projection.ingestion_cursor,
            "memory_cursor": projection.memory_cursor,
        },
        domain="saliencegate:projection:cursors:v1",
    )
    deliveries = integrity.tag(
        {
            "history": _models(projection.delivery_history),
            "latest": {
                str(delivery_id): record.model_dump(mode="json")
                for delivery_id, record in sorted(
                    projection.deliveries.items(),
                    key=lambda item: str(item[0]),
                )
            },
        },
        domain="saliencegate:projection:deliveries:v1",
    )
    outcomes = integrity.tag(
        _models(projection.outcomes),
        domain="saliencegate:projection:outcomes:v1",
    )
    components = {
        "events": events,
        "memory": memory,
        "signals": signals,
        "decisions": decisions,
        "cycles": cycles,
        "interventions": interventions,
        "budgets": budgets,
        "cursors": cursors,
        "deliveries": deliveries,
        "outcomes": outcomes,
    }
    overall = integrity.tag(
        {key: value.model_dump(mode="json") for key, value in components.items()},
        domain="saliencegate:projection:overall:v1",
    )
    return ProjectionDigests(**components, overall=overall)


def snapshot(
    projection: Projection,
    integrity: IntegrityContext,
    *,
    ledger_position: int,
) -> MemorySnapshot:
    records = tuple(
        sorted(
            projection.memories.values(),
            key=lambda record: (record.kind.value, str(record.memory_id)),
        )
    )
    digests = projection_digests(
        projection,
        integrity,
        ledger_position=ledger_position,
    )
    return MemorySnapshot(
        run_id=projection.run_id,
        ledger_position=ledger_position,
        ingestion_cursor=projection.ingestion_cursor,
        memory_cursor=projection.memory_cursor,
        records=records,
        projection_digest=digests.overall,
    )


def search(projection: Projection, query: MemoryQuery) -> tuple[MemoryHit, ...]:
    terms = tuple(dict.fromkeys(re.findall(r"\w+", query.text.casefold()))) if query.text else ()
    candidates: list[MemoryHit] = []
    for memory in projection.memories.values():
        if query.kinds and memory.kind not in query.kinds:
            continue
        if memory.validity not in query.validity:
            continue
        if query.trust_labels and memory.trust_label not in query.trust_labels:
            continue
        content = memory.content.casefold()
        matched = tuple(term for term in terms if term in content)
        if terms and not matched:
            continue
        score = 1.0 if not terms else len(matched) / len(terms)
        candidates.append(
            MemoryHit(
                memory=memory,
                score=score,
                matched_terms=matched,
            )
        )
    candidates.sort(key=lambda hit: (-hit.score, -hit.memory.revision, str(hit.memory.memory_id)))
    return tuple(candidates[: query.limit])
