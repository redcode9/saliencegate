from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from saliencegate.domain import (
    BudgetAmounts,
    BudgetLimits,
    BudgetSnapshot,
    ClaimKind,
    ConstraintStatus,
    CycleRecord,
    CycleState,
    DeduplicationGuarantee,
    DeliveryOutcome,
    DeliveryRecord,
    DeliveryState,
    DeliveryTarget,
    EventPhase,
    EventType,
    EvidenceReference,
    EvidenceSource,
    InterventionAction,
    InterventionClaim,
    InterventionDecision,
    InterventionOutcome,
    InvocationDecision,
    MemoryCreate,
    MemoryDelta,
    MemoryIdAssignment,
    MemoryInvalidation,
    MemoryKind,
    MemoryRecord,
    MemoryUpdate,
    OutcomeEvidenceMode,
    PayloadDigest,
    PayloadDigestAlgorithm,
    ReasonCode,
    RepeatedErrorStatus,
    Signal,
    SignalType,
    TraceEvent,
    TrustLabel,
    ValidityState,
    canonical_digest,
    cycle_id,
)
from saliencegate.intervention.claims import (
    GROUNDING_RECEIPT_VERSION,
    GroundingReceipt,
    InterventionProposal,
    ProposalParseStatus,
    ProposedClaim,
)
from saliencegate.intervention.grounding import (
    GroundingConfig,
    GroundingContext,
    GroundingPipeline,
    GroundingState,
    resolve_grounding_configuration,
)
from saliencegate.intervention.rendering import RenderingConfig
from saliencegate.ports.repository import (
    CrossRunReferenceError,
    LedgerEntry,
    MemoryQuery,
    ProjectionInvariantError,
    RevisionConflictError,
)
from saliencegate.repository.integrity import IntegrityContext
from saliencegate.repository.projector import (
    Projection,
    _models,
    _new_memory,
    _record_values,
    _store_memory,
    _validate_evidence,
    _validate_memory_target,
    apply_entry,
    empty_projection,
    preview_memory_delta,
    projection_digests,
    search,
    snapshot,
    validate_complete_projection,
)
from saliencegate.repository.projector import (
    budget_snapshot as projected_budget_snapshot,
)

RUN_ID = UUID("00000000-0000-4000-8000-000000000201")
EVENT_1_ID = UUID("00000000-0000-4000-8000-000000000202")
EVENT_2_ID = UUID("00000000-0000-4000-8000-000000000203")
DECISION_1_ID = UUID("00000000-0000-4000-8000-000000000204")
DECISION_2_ID = UUID("00000000-0000-4000-8000-000000000205")
DELTA_1_ID = UUID("00000000-0000-4000-8000-000000000206")
DELTA_2_ID = UUID("00000000-0000-4000-8000-000000000207")
MEMORY_1_ID = UUID("00000000-0000-4000-8000-000000000208")
MEMORY_2_ID = UUID("00000000-0000-4000-8000-000000000209")
MEMORY_3_ID = UUID("00000000-0000-4000-8000-000000000218")
MEMORY_4_ID = UUID("00000000-0000-4000-8000-000000000219")
MEMORY_5_ID = UUID("00000000-0000-4000-8000-00000000021a")
MISSING_ID = UUID("00000000-0000-4000-8000-00000000020a")
OTHER_RUN_ID = UUID("00000000-0000-4000-8000-000000000210")
OTHER_RUN_EVENT_ID = UUID("00000000-0000-4000-8000-000000000211")
INTERVENTION_1_ID = UUID("00000000-0000-4000-8000-00000000020b")
INTERVENTION_2_ID = UUID("00000000-0000-4000-8000-00000000020c")
SIGNAL_ID = UUID("00000000-0000-4000-8000-00000000020d")
OUTCOME_ID = UUID("00000000-0000-4000-8000-00000000020e")
DELIVERY_ID = UUID("00000000-0000-4000-8000-00000000020f")
CLAIM_1_ID = UUID("00000000-0000-4000-8000-000000000215")
CLAIM_2_ID = UUID("00000000-0000-4000-8000-000000000216")
ATTEMPT_1_ID = UUID("00000000-0000-4000-8000-000000000217")
CONFIGURATION_DIGEST = "f" * 64
NOW = datetime(2026, 7, 11, 12, 30, tzinfo=UTC)


def grounding_config(
    *,
    duplicate_window_events: int = 0,
    cooldown_events: int = 0,
) -> GroundingConfig:
    return GroundingConfig(
        schema_version="1.0",
        pipeline_version="grounding-pipeline/v1",
        claim_schema_version="citation-only-claims/v1",
        max_claims=2,
        max_evidence_per_claim=1,
        max_pointer_segments=32,
        max_pointer_utf8_bytes=1_024,
        duplicate_window_events=duplicate_window_events,
        cooldown_events=cooldown_events,
        ttl_steps=1,
        allowed_delivery_targets=(
            DeliveryTarget.NEXT_MODEL_CALL,
            DeliveryTarget.PRE_ACTION_REPLAN,
        ),
        rendering=RenderingConfig(
            schema_version="1.0",
            renderer_version="fixed-ascii/v1",
            token_counter_version="utf8-bytes-ceil-div-4-v1",
            max_claims=2,
            max_evidence_bytes=1_024,
            max_output_bytes=4_096,
            max_token_equivalents=1_024,
            include_provenance=False,
        ),
    )


def tag(character: str = "a") -> PayloadDigest:
    return PayloadDigest(
        algorithm=PayloadDigestAlgorithm.SYNTHETIC_SHA256,
        value=character * 64,
    )


def entry(position: int, record: object) -> LedgerEntry:
    previous = None if position == 1 else tag("b")
    return LedgerEntry(
        run_id=RUN_ID,
        position=position,
        record_key=f"fixture:{position}",
        record_tag=tag("c"),
        previous_chain_tag=previous,
        chain_tag=tag("d"),
        record=record,
    )


def trace_event(sequence: int, event_id: UUID) -> TraceEvent:
    return TraceEvent(
        event_id=event_id,
        run_id=RUN_ID,
        sequence=sequence,
        source_event_id=f"source-{sequence}",
        timestamp=NOW + timedelta(seconds=sequence),
        event_type=EventType.OBSERVATION,
        phase=EventPhase.POST_ACTION,
        payload={"message": f"event {sequence}"},
        payload_digest=tag("e"),
        source_adapter="fixture",
        trust_label=TrustLabel.SYNTHETIC_FIXTURE,
    )


def invocation(
    sequence: int,
    decision_id: UUID,
    *,
    budget: BudgetSnapshot | None = None,
) -> InvocationDecision:
    limits = BudgetLimits(
        model_calls=10,
        input_tokens=10_000,
        output_tokens=10_000,
        canonical_token_equivalents=20_000,
        latency_us=10_000_000,
        max_call_latency_us=1_000_000,
        interventions=10,
        schema_repairs=2,
    )
    return InvocationDecision(
        decision_id=decision_id,
        run_id=RUN_ID,
        event_sequence=sequence,
        invoke=True,
        risk_score=0.8,
        reason_codes=(ReasonCode.TOOL_ERROR,),
        policy_version="fixture/1",
        configuration_digest=CONFIGURATION_DIGEST,
        budget_snapshot=(
            BudgetSnapshot(
                limits=limits,
                reserved=BudgetAmounts(),
                consumed=BudgetAmounts(),
            )
            if budget is None
            else budget
        ),
        cooldown_active=False,
        created_at=NOW + timedelta(seconds=sequence),
    )


def test_budget_projection_freezes_limits_and_rejects_stale_decisions() -> None:
    with pytest.raises(ProjectionInvariantError, match="limits are unavailable"):
        projected_budget_snapshot(empty_projection(RUN_ID))

    projection = apply_records(
        (
            trace_event(1, EVENT_1_ID),
            invocation(1, DECISION_1_ID),
            trace_event(2, EVENT_2_ID),
        )
    )
    current = projected_budget_snapshot(projection)
    changed_limits = current.limits.model_copy(update={"model_calls": 11})
    changed = invocation(
        2,
        DECISION_2_ID,
        budget=BudgetSnapshot(
            limits=changed_limits,
            reserved=current.reserved,
            consumed=current.consumed,
        ),
    )
    with pytest.raises(ProjectionInvariantError, match="limits changed"):
        apply_entry(projection, entry(4, changed))

    stale = invocation(
        2,
        DECISION_2_ID,
        budget=BudgetSnapshot(
            limits=current.limits,
            reserved=BudgetAmounts(model_calls=1),
            consumed=current.consumed,
        ),
    )
    with pytest.raises(ProjectionInvariantError, match="snapshot is stale"):
        apply_entry(projection, entry(4, stale))


def event_reference(event_id: UUID) -> EvidenceReference:
    return EvidenceReference(
        source=EvidenceSource.EVENT,
        source_id=event_id,
        field_path="/payload/message",
    )


def silence(
    cycle: str,
    intervention_id: UUID,
    created_at: datetime,
    *,
    configuration: GroundingConfig | None = None,
) -> InterventionDecision:
    configuration = grounding_config() if configuration is None else configuration
    resolved = resolve_grounding_configuration(configuration)
    receipt = GroundingReceipt(
        receipt_version=GROUNDING_RECEIPT_VERSION,
        parse_status=ProposalParseStatus.VALID,
        proposal_action=InterventionAction.SILENCE,
        claims=(),
        confidence=1.0,
        requested_delivery_target=DeliveryTarget.NEXT_MODEL_CALL,
        model_call_index=0,
        model_call_digest="b" * 64,
    )
    return InterventionDecision(
        intervention_id=intervention_id,
        run_id=RUN_ID,
        cycle_id=cycle,
        grounding_version=resolved.pipeline_version,
        grounding_configuration=configuration.model_dump(mode="json"),
        grounding_configuration_digest=resolved.configuration_digest,
        grounding_receipt=receipt.model_dump(mode="json"),
        action=InterventionAction.SILENCE,
        confidence=1.0,
        reason_code=ReasonCode.SILENCE_SELECTED,
        created_at=created_at,
    )


def grounded_reminder(
    *,
    cycle: str,
    intervention_id: UUID,
    current_event: TraceEvent,
    evidence: EvidenceReference,
    created_at: datetime,
    events: tuple[TraceEvent, ...] | None = None,
    memories: tuple[MemoryRecord, ...] = (),
    configuration: GroundingConfig | None = None,
) -> InterventionDecision:
    return GroundingPipeline(grounding_config() if configuration is None else configuration).ground(
        InterventionProposal(
            action=InterventionAction.REMIND,
            claims=(
                ProposedClaim(
                    kind=ClaimKind.ENVIRONMENT_FACT,
                    evidence=evidence,
                ),
            ),
            confidence=0.9,
            model_free_text="ignored and never rendered",
        ),
        context=GroundingContext(
            schema_version="1.0",
            intervention_id=intervention_id,
            run_id=RUN_ID,
            cycle_id=cycle,
            current_event_sequence=current_event.sequence,
            created_at=created_at,
            requested_delivery_target=DeliveryTarget.NEXT_MODEL_CALL,
            model_call_index=0,
            model_call_digest="b" * 64,
        ),
        state=GroundingState(
            schema_version="1.0",
            events=(current_event,) if events is None else events,
            memories=memories,
            reminder_history=(),
        ),
    )


def cycle_revisions(
    *,
    first_sequence: int,
    decision_id: UUID,
    delta: MemoryDelta,
    assignments: tuple[MemoryIdAssignment, ...],
    intervention_id: UUID,
    configuration: GroundingConfig | None = None,
) -> tuple[CycleRecord, ...]:
    configuration = grounding_config() if configuration is None else configuration
    resolved = resolve_grounding_configuration(configuration)
    identifier = cycle_id(
        RUN_ID,
        first_sequence,
        first_sequence,
        "fixture/1",
        CONFIGURATION_DIGEST,
        resolved.pipeline_version,
        resolved.configuration_digest,
        DeliveryTarget.NEXT_MODEL_CALL,
    )
    created_at = NOW + timedelta(seconds=first_sequence * 10)
    reservation = BudgetAmounts(
        model_calls=1,
        input_tokens=100,
        output_tokens=100,
        canonical_token_equivalents=200,
        latency_us=10_000,
        interventions=1,
    )
    common = {
        "cycle_id": identifier,
        "run_id": RUN_ID,
        "invocation_decision_id": decision_id,
        "policy_version": "fixture/1",
        "configuration_digest": CONFIGURATION_DIGEST,
        "grounding_version": resolved.pipeline_version,
        "grounding_configuration": resolved.configuration,
        "grounding_configuration_digest": resolved.configuration_digest,
        "requested_delivery_target": DeliveryTarget.NEXT_MODEL_CALL,
        "first_event_sequence": first_sequence,
        "last_event_sequence": first_sequence,
        "created_at": created_at,
    }
    return (
        CycleRecord(
            **common,
            revision=1,
            state=CycleState.PENDING,
            updated_at=created_at,
        ),
        CycleRecord(
            **common,
            revision=2,
            state=CycleState.RESERVED,
            budget_reservation=reservation,
            updated_at=created_at + timedelta(seconds=1),
        ),
        CycleRecord(
            **common,
            revision=3,
            state=CycleState.RUNNING,
            budget_reservation=reservation,
            batch_digest="a" * 64,
            updated_at=created_at + timedelta(seconds=2),
        ),
        CycleRecord(
            **common,
            revision=4,
            state=CycleState.COMMITTED,
            budget_reservation=reservation,
            budget_settlement=BudgetAmounts(
                model_calls=1,
                input_tokens=50,
                output_tokens=10,
                canonical_token_equivalents=60,
                latency_us=1_000,
            ),
            batch_digest="a" * 64,
            model_call_digests=("b" * 64,),
            model_call_latencies_us=(1_000,),
            validated_delta=delta,
            memory_id_assignments=assignments,
            intervention=silence(
                identifier,
                intervention_id,
                created_at + timedelta(seconds=3),
                configuration=configuration,
            ),
            updated_at=created_at + timedelta(seconds=3),
        ),
    )


def committed_with_intervention(
    committed: CycleRecord,
    intervention: InterventionDecision,
) -> CycleRecord:
    settlement = committed.budget_settlement
    assert settlement is not None
    return CycleRecord.model_validate(
        {
            **committed.model_dump(mode="python"),
            "budget_settlement": settlement.model_copy(update={"interventions": 1}),
            "intervention": intervention,
        }
    )


def apply_records(records: tuple[object, ...]) -> Projection:
    projection = empty_projection(RUN_ID)
    for position, record in enumerate(records, start=1):
        projection = apply_entry(projection, entry(position, record))
    return projection


def first_committed_projection(
    *,
    kind: MemoryKind = MemoryKind.KNOWLEDGE,
    expires_at: datetime | None = None,
) -> Projection:
    event = trace_event(1, EVENT_1_ID)
    decision = invocation(1, DECISION_1_ID)
    delta = MemoryDelta(
        delta_id=DELTA_1_ID,
        run_id=RUN_ID,
        creates=(
            MemoryCreate(
                handle="memory-one",
                kind=kind,
                content="Run tests from the working directory.",
                provenance=(event_reference(EVENT_1_ID),),
                confidence=0.9,
                trust_label=TrustLabel.TRUSTED_CONTROLLER,
                expires_at=expires_at,
            ),
        )
        if kind is not MemoryKind.PRIVATE_STATUS
        else (),
        private_status_replacement=(
            {
                "replacement": {
                    "handle": "memory-one",
                    "kind": MemoryKind.PRIVATE_STATUS,
                    "content": "Tests are currently failing.",
                    "provenance": (event_reference(EVENT_1_ID),),
                    "confidence": 0.9,
                    "trust_label": TrustLabel.TRUSTED_CONTROLLER,
                }
            }
            if kind is MemoryKind.PRIVATE_STATUS
            else None
        ),
        created_at=NOW + timedelta(seconds=13),
    )
    cycles = cycle_revisions(
        first_sequence=1,
        decision_id=DECISION_1_ID,
        delta=delta,
        assignments=(MemoryIdAssignment(handle="memory-one", memory_id=MEMORY_1_ID),),
        intervention_id=INTERVENTION_1_ID,
    )
    return apply_records((event, decision, *cycles))


def first_reminder_projection() -> Projection:
    event = trace_event(1, EVENT_1_ID)
    decision = invocation(1, DECISION_1_ID)
    delta = MemoryDelta(
        delta_id=DELTA_1_ID,
        run_id=RUN_ID,
        created_at=NOW + timedelta(seconds=13),
    )
    cycles = list(
        cycle_revisions(
            first_sequence=1,
            decision_id=DECISION_1_ID,
            delta=delta,
            assignments=(),
            intervention_id=INTERVENTION_1_ID,
        )
    )
    committed = cycles[-1]
    context = GroundingContext(
        schema_version="1.0",
        intervention_id=INTERVENTION_1_ID,
        run_id=RUN_ID,
        cycle_id=committed.cycle_id,
        current_event_sequence=1,
        created_at=committed.updated_at,
        requested_delivery_target=DeliveryTarget.NEXT_MODEL_CALL,
        model_call_index=0,
        model_call_digest="b" * 64,
    )
    reminder = GroundingPipeline(grounding_config()).ground(
        InterventionProposal(
            action=InterventionAction.REMIND,
            claims=(
                ProposedClaim(
                    kind=ClaimKind.ENVIRONMENT_FACT,
                    evidence=event_reference(EVENT_1_ID),
                ),
            ),
            confidence=0.9,
            model_free_text="ignored and never rendered",
        ),
        context=context,
        state=GroundingState(
            schema_version="1.0",
            events=(event,),
            memories=(),
            reminder_history=(),
        ),
    )
    cycles[-1] = committed_with_intervention(committed, reminder)
    return apply_records((event, decision, *cycles))


def test_authoritative_grounding_rejects_cross_run_event_reference() -> None:
    event = trace_event(1, EVENT_1_ID)
    decision = invocation(1, DECISION_1_ID)
    delta = MemoryDelta(
        delta_id=DELTA_1_ID,
        run_id=RUN_ID,
        created_at=NOW + timedelta(seconds=13),
    )
    cycles = list(
        cycle_revisions(
            first_sequence=1,
            decision_id=DECISION_1_ID,
            delta=delta,
            assignments=(),
            intervention_id=INTERVENTION_1_ID,
        )
    )
    committed = cycles[-1]
    reminder = grounded_reminder(
        cycle=committed.cycle_id,
        intervention_id=INTERVENTION_1_ID,
        current_event=event,
        evidence=event_reference(EVENT_1_ID),
        created_at=committed.updated_at,
    )
    claim = reminder.claims[0]
    cross_run_reference = claim.evidence[0].model_copy(update={"source_id": OTHER_RUN_EVENT_ID})
    forged_claim = InterventionClaim.model_validate(
        {
            **claim.model_dump(mode="python"),
            "evidence": (cross_run_reference,),
        }
    )
    forged = InterventionDecision.model_validate(
        {
            **reminder.model_dump(mode="python"),
            "claims": (forged_claim,),
            "cited_event_ids": (OTHER_RUN_EVENT_ID,),
        }
    )
    cycles[-1] = committed_with_intervention(committed, forged)
    projection = apply_records((event, decision, *cycles[:-1]))

    with pytest.raises(
        ProjectionInvariantError,
        match="grounded intervention failed authoritative verification",
    ):
        apply_entry(projection, entry(6, cycles[-1]))

    other_run_event = event.model_copy(
        update={"run_id": OTHER_RUN_ID, "event_id": OTHER_RUN_EVENT_ID}
    )
    assert other_run_event.run_id != projection.run_id
    assert OTHER_RUN_EVENT_ID not in projection.events_by_id


def test_authoritative_grounding_rejects_stale_memory_revision_after_delta() -> None:
    projection = first_committed_projection()
    stale_memory = projection.memories[MEMORY_1_ID]
    first_event = projection.events_by_sequence[1]
    second_event = trace_event(2, EVENT_2_ID)
    projection = apply_entry(projection, entry(7, second_event))
    projection = apply_entry(
        projection,
        entry(
            8,
            invocation(
                2,
                DECISION_2_ID,
                budget=projected_budget_snapshot(projection),
            ),
        ),
    )
    delta = MemoryDelta(
        delta_id=DELTA_2_ID,
        run_id=RUN_ID,
        updates=(
            MemoryUpdate(
                memory_id=MEMORY_1_ID,
                expected_revision=1,
                content="The authoritative current revision is now different.",
            ),
        ),
        created_at=NOW + timedelta(seconds=23),
    )
    cycles = list(
        cycle_revisions(
            first_sequence=2,
            decision_id=DECISION_2_ID,
            delta=delta,
            assignments=(),
            intervention_id=INTERVENTION_2_ID,
        )
    )
    committed = cycles[-1]
    reminder = grounded_reminder(
        cycle=committed.cycle_id,
        intervention_id=INTERVENTION_2_ID,
        current_event=second_event,
        evidence=EvidenceReference(
            source=EvidenceSource.MEMORY,
            source_id=MEMORY_1_ID,
            revision=1,
            field_path="/content",
        ),
        created_at=committed.updated_at,
        events=(first_event, second_event),
        memories=(stale_memory,),
    )
    cycles[-1] = committed_with_intervention(committed, reminder)
    for offset, cycle in enumerate(cycles[:-1], start=9):
        projection = apply_entry(projection, entry(offset, cycle))

    with pytest.raises(
        ProjectionInvariantError,
        match="grounded intervention failed authoritative verification",
    ):
        apply_entry(projection, entry(12, cycles[-1]))

    assert projection.memories[MEMORY_1_ID].revision == 1
    assert projection.memory_cursor == 1


def test_authoritative_grounding_rejects_expired_memory_reference() -> None:
    projection = first_committed_projection(expires_at=NOW + timedelta(seconds=14))
    expired_memory = projection.memories[MEMORY_1_ID]
    first_event = projection.events_by_sequence[1]
    second_event = trace_event(2, EVENT_2_ID)
    projection = apply_entry(projection, entry(7, second_event))
    projection = apply_entry(
        projection,
        entry(
            8,
            invocation(
                2,
                DECISION_2_ID,
                budget=projected_budget_snapshot(projection),
            ),
        ),
    )
    delta = MemoryDelta(
        delta_id=DELTA_2_ID,
        run_id=RUN_ID,
        created_at=NOW + timedelta(seconds=23),
    )
    cycles = list(
        cycle_revisions(
            first_sequence=2,
            decision_id=DECISION_2_ID,
            delta=delta,
            assignments=(),
            intervention_id=INTERVENTION_2_ID,
        )
    )
    committed = cycles[-1]
    reminder = grounded_reminder(
        cycle=committed.cycle_id,
        intervention_id=INTERVENTION_2_ID,
        current_event=second_event,
        evidence=EvidenceReference(
            source=EvidenceSource.MEMORY,
            source_id=MEMORY_1_ID,
            revision=1,
            field_path="/content",
        ),
        created_at=committed.updated_at,
        events=(first_event, second_event),
        memories=(expired_memory.model_copy(update={"expires_at": None}),),
    )
    cycles[-1] = committed_with_intervention(committed, reminder)
    for offset, cycle in enumerate(cycles[:-1], start=9):
        projection = apply_entry(projection, entry(offset, cycle))

    with pytest.raises(
        ProjectionInvariantError,
        match="grounded intervention failed authoritative verification",
    ):
        apply_entry(projection, entry(12, cycles[-1]))

    assert expired_memory.expires_at is not None
    assert expired_memory.expires_at < committed.updated_at


def test_authoritative_grounding_rebuilds_prior_reminder_history() -> None:
    configuration = grounding_config(duplicate_window_events=5)
    first_event = trace_event(1, EVENT_1_ID)
    first_decision = invocation(1, DECISION_1_ID)
    first_delta = MemoryDelta(
        delta_id=DELTA_1_ID,
        run_id=RUN_ID,
        created_at=NOW + timedelta(seconds=13),
    )
    first_cycles = list(
        cycle_revisions(
            first_sequence=1,
            decision_id=DECISION_1_ID,
            delta=first_delta,
            assignments=(),
            intervention_id=INTERVENTION_1_ID,
            configuration=configuration,
        )
    )
    first_committed = first_cycles[-1]
    first_reminder = grounded_reminder(
        cycle=first_committed.cycle_id,
        intervention_id=INTERVENTION_1_ID,
        current_event=first_event,
        evidence=event_reference(EVENT_1_ID),
        created_at=first_committed.updated_at,
        configuration=configuration,
    )
    first_cycles[-1] = committed_with_intervention(first_committed, first_reminder)
    projection = apply_records((first_event, first_decision, *first_cycles))

    second_event = trace_event(2, EVENT_2_ID)
    projection = apply_entry(projection, entry(7, second_event))
    projection = apply_entry(
        projection,
        entry(
            8,
            invocation(
                2,
                DECISION_2_ID,
                budget=projected_budget_snapshot(projection),
            ),
        ),
    )
    second_delta = MemoryDelta(
        delta_id=DELTA_2_ID,
        run_id=RUN_ID,
        created_at=NOW + timedelta(seconds=23),
    )
    second_cycles = list(
        cycle_revisions(
            first_sequence=2,
            decision_id=DECISION_2_ID,
            delta=second_delta,
            assignments=(),
            intervention_id=INTERVENTION_2_ID,
            configuration=configuration,
        )
    )
    second_committed = second_cycles[-1]
    duplicate = grounded_reminder(
        cycle=second_committed.cycle_id,
        intervention_id=INTERVENTION_2_ID,
        current_event=second_event,
        evidence=event_reference(EVENT_1_ID),
        created_at=second_committed.updated_at,
        events=(first_event, second_event),
        configuration=configuration,
    )
    second_cycles[-1] = committed_with_intervention(second_committed, duplicate)
    for offset, cycle in enumerate(second_cycles[:-1], start=9):
        projection = apply_entry(projection, entry(offset, cycle))

    with pytest.raises(
        ProjectionInvariantError,
        match="grounded intervention failed authoritative verification",
    ):
        apply_entry(projection, entry(12, second_cycles[-1]))

    assert projection.interventions == {INTERVENTION_1_ID: first_reminder}


def test_committed_cycle_creates_revision_history_and_advances_memory_cursor() -> None:
    projection = first_committed_projection()
    record = projection.memories[MEMORY_1_ID]

    assert record.revision == 1
    assert record.validity is ValidityState.ACTIVE
    assert projection.memory_history[(MEMORY_1_ID, 1)] == record
    assert projection.memory_cursor == 1

    integrity = IntegrityContext(key=None, synthetic_benchmark=True)
    view = snapshot(projection, integrity, ledger_position=6)
    hits = search(
        projection,
        MemoryQuery(run_id=RUN_ID, text="working directory"),
    )
    assert view.records == (record,)
    assert view.projection_digest.algorithm is PayloadDigestAlgorithm.SYNTHETIC_SHA256
    assert hits[0].memory == record
    assert hits[0].matched_terms == ("working", "directory")


def test_projection_digests_bind_index_keys_and_full_latest_values() -> None:
    projection = first_committed_projection()
    integrity = IntegrityContext(key=None, synthetic_benchmark=True)
    baseline = projection_digests(projection, integrity, ledger_position=6)
    memory = projection.memories[MEMORY_1_ID]

    poisoned = replace(
        projection,
        memories={
            MEMORY_1_ID: memory.model_copy(update={"content": "poisoned latest value"}),
        },
    )
    assert projection_digests(poisoned, integrity, ledger_position=6).memory != baseline.memory

    rekeyed = replace(
        projection,
        memory_history={(MISSING_ID, memory.revision): memory},
    )
    assert projection_digests(rekeyed, integrity, ledger_position=6).memory != baseline.memory

    with pytest.raises(TypeError):
        projection.memories[MEMORY_1_ID] = memory


def test_invocation_event_index_is_canonical_and_cannot_diverge_from_decisions() -> None:
    projection = first_committed_projection()
    decision = projection.decisions[DECISION_1_ID]

    assert projection.decisions_by_event_sequence == {decision.event_sequence: decision}
    with pytest.raises(ValueError, match="init=False"):
        replace(projection, decisions_by_event_sequence={})
    with pytest.raises(ProjectionInvariantError, match="already has a decision"):
        replace(
            projection,
            decisions={
                **projection.decisions,
                DECISION_2_ID: decision.model_copy(update={"decision_id": DECISION_2_ID}),
            },
        )
    with pytest.raises(TypeError):
        projection.decisions_by_event_sequence[decision.event_sequence] = decision


def delta_preview_fixture() -> tuple[
    Projection,
    MemoryDelta,
    tuple[MemoryIdAssignment, ...],
    CycleRecord,
]:
    first_event = trace_event(1, EVENT_1_ID)
    initial_delta = MemoryDelta(
        delta_id=DELTA_1_ID,
        run_id=RUN_ID,
        creates=(
            MemoryCreate(
                handle="knowledge-before",
                kind=MemoryKind.KNOWLEDGE,
                content="The repository has a deterministic test suite.",
                provenance=(event_reference(EVENT_1_ID),),
                confidence=0.9,
                trust_label=TrustLabel.TRUSTED_CONTROLLER,
            ),
            MemoryCreate(
                handle="procedure-before",
                kind=MemoryKind.PROCEDURAL,
                content="Run the deterministic test suite before delivery.",
                provenance=(event_reference(EVENT_1_ID),),
                confidence=0.9,
                trust_label=TrustLabel.TRUSTED_CONTROLLER,
            ),
        ),
        private_status_replacement={
            "replacement": {
                "handle": "private-before",
                "kind": MemoryKind.PRIVATE_STATUS,
                "content": "The preview has not run yet.",
                "provenance": (event_reference(EVENT_1_ID),),
                "confidence": 0.9,
                "trust_label": TrustLabel.TRUSTED_CONTROLLER,
            },
        },
        created_at=NOW + timedelta(seconds=13),
    )
    initial_cycles = cycle_revisions(
        first_sequence=1,
        decision_id=DECISION_1_ID,
        delta=initial_delta,
        assignments=(
            MemoryIdAssignment(handle="knowledge-before", memory_id=MEMORY_1_ID),
            MemoryIdAssignment(handle="procedure-before", memory_id=MEMORY_2_ID),
            MemoryIdAssignment(handle="private-before", memory_id=MEMORY_3_ID),
        ),
        intervention_id=INTERVENTION_1_ID,
    )
    projection = apply_records((first_event, invocation(1, DECISION_1_ID), *initial_cycles))

    projection = apply_entry(projection, entry(7, trace_event(2, EVENT_2_ID)))
    projection = apply_entry(
        projection,
        entry(
            8,
            invocation(
                2,
                DECISION_2_ID,
                budget=projected_budget_snapshot(projection),
            ),
        ),
    )
    delta = MemoryDelta(
        delta_id=DELTA_2_ID,
        run_id=RUN_ID,
        creates=(
            MemoryCreate(
                handle="knowledge-after",
                kind=MemoryKind.KNOWLEDGE,
                content="The pure preview matches committed memory state.",
                provenance=(event_reference(EVENT_2_ID),),
                confidence=1.0,
                trust_label=TrustLabel.TRUSTED_CONTROLLER,
            ),
        ),
        updates=(
            MemoryUpdate(
                memory_id=MEMORY_1_ID,
                expected_revision=1,
                content="The repository has a verified deterministic test suite.",
            ),
        ),
        invalidations=(
            MemoryInvalidation(
                memory_id=MEMORY_2_ID,
                expected_revision=1,
                reason_code=ReasonCode.CONFLICT,
            ),
        ),
        private_status_replacement={
            "expected_memory_id": MEMORY_3_ID,
            "expected_revision": 1,
            "replacement": {
                "handle": "private-after",
                "kind": MemoryKind.PRIVATE_STATUS,
                "content": "The preview has completed.",
                "provenance": (event_reference(EVENT_2_ID),),
                "confidence": 1.0,
                "trust_label": TrustLabel.TRUSTED_CONTROLLER,
            },
        },
        created_at=NOW + timedelta(seconds=23),
    )
    assignments = (
        MemoryIdAssignment(handle="knowledge-after", memory_id=MEMORY_4_ID),
        MemoryIdAssignment(handle="private-after", memory_id=MEMORY_5_ID),
    )
    cycles = cycle_revisions(
        first_sequence=2,
        decision_id=DECISION_2_ID,
        delta=delta,
        assignments=assignments,
        intervention_id=INTERVENTION_2_ID,
    )
    for offset, cycle in enumerate(cycles[:-1], start=9):
        projection = apply_entry(projection, entry(offset, cycle))
    return projection, delta, assignments, cycles[-1]


def test_memory_delta_preview_matches_committed_memory_without_advancing_state() -> None:
    projection, delta, assignments, committed = delta_preview_fixture()
    original_memories = dict(projection.memories)
    original_history = dict(projection.memory_history)
    original_delta = delta.model_dump(mode="python")
    original_assignments = tuple(assignments)

    preview = preview_memory_delta(
        projection,
        delta,
        assignments,
        last_event_sequence=committed.last_event_sequence,
    )
    applied = apply_entry(projection, entry(12, committed))

    assert preview.memories == applied.memories
    assert preview.memory_history == applied.memory_history
    assert preview.current_private_status_id == applied.current_private_status_id
    assert preview == replace(
        projection,
        memories=applied.memories,
        memory_history=applied.memory_history,
        current_private_status_id=applied.current_private_status_id,
    )
    assert preview.memory_cursor == projection.memory_cursor
    assert preview.cycles == projection.cycles
    assert preview.interventions == projection.interventions
    assert preview.memories[MEMORY_1_ID].revision == 2
    assert preview.memories[MEMORY_2_ID].validity is ValidityState.INVALIDATED
    assert preview.memories[MEMORY_3_ID].validity is ValidityState.SUPERSEDED
    assert preview.memories[MEMORY_4_ID].revision == 1
    assert preview.current_private_status_id == MEMORY_5_ID
    assert {memory.trust_label for memory in preview.memories.values()} == {
        TrustLabel.UNTRUSTED_MODEL_OUTPUT
    }
    assert {memory.trust_label for memory in applied.memories.values()} == {
        TrustLabel.UNTRUSTED_MODEL_OUTPUT
    }
    assert all(create.trust_label is TrustLabel.TRUSTED_CONTROLLER for create in delta.creates)
    assert delta.private_status_replacement is not None
    assert delta.private_status_replacement.replacement.trust_label is TrustLabel.TRUSTED_CONTROLLER

    assert dict(projection.memories) == original_memories
    assert dict(projection.memory_history) == original_history
    assert delta.model_dump(mode="python") == original_delta
    assert assignments == original_assignments


def test_memory_update_downgrades_even_a_preexisting_trusted_projection() -> None:
    projection = first_committed_projection()
    current = projection.memories[MEMORY_1_ID]
    trusted = current.model_copy(update={"trust_label": TrustLabel.TRUSTED_CONTROLLER})
    trusted_history = dict(projection.memory_history)
    trusted_history[(trusted.memory_id, trusted.revision)] = trusted
    trusted_projection = replace(
        projection,
        memories={trusted.memory_id: trusted},
        memory_history=trusted_history,
    )
    delta = MemoryDelta(
        delta_id=DELTA_2_ID,
        run_id=RUN_ID,
        updates=(
            MemoryUpdate(
                memory_id=MEMORY_1_ID,
                expected_revision=1,
                content="Model updates cannot retain an authoritative trust label.",
            ),
        ),
        created_at=NOW + timedelta(seconds=23),
    )

    preview = preview_memory_delta(
        trusted_projection,
        delta,
        (),
        last_event_sequence=trusted_projection.memory_cursor,
    )

    assert trusted_projection.memories[MEMORY_1_ID].trust_label is TrustLabel.TRUSTED_CONTROLLER
    assert preview.memories[MEMORY_1_ID].trust_label is TrustLabel.UNTRUSTED_MODEL_OUTPUT
    assert preview.memory_history[(MEMORY_1_ID, 1)].trust_label is TrustLabel.TRUSTED_CONTROLLER
    assert preview.memory_history[(MEMORY_1_ID, 2)].trust_label is TrustLabel.UNTRUSTED_MODEL_OUTPUT


def test_model_claimed_trust_is_downgraded_on_commit_and_fresh_ledger_replay() -> None:
    first_event = trace_event(1, EVENT_1_ID)
    delta = MemoryDelta(
        delta_id=DELTA_1_ID,
        run_id=RUN_ID,
        creates=(
            MemoryCreate(
                handle="claimed-trusted",
                kind=MemoryKind.KNOWLEDGE,
                content="Model-authored content is evidence, not authority.",
                provenance=(event_reference(EVENT_1_ID),),
                confidence=1.0,
                trust_label=TrustLabel.TRUSTED_CONTROLLER,
            ),
        ),
        created_at=NOW + timedelta(seconds=13),
    )
    cycles = cycle_revisions(
        first_sequence=1,
        decision_id=DECISION_1_ID,
        delta=delta,
        assignments=(MemoryIdAssignment(handle="claimed-trusted", memory_id=MEMORY_1_ID),),
        intervention_id=INTERVENTION_1_ID,
    )
    records = (first_event, invocation(1, DECISION_1_ID), *cycles)
    before_commit = apply_records(records[:-1])

    committed = apply_entry(before_commit, entry(len(records), records[-1]))
    rebuilt = apply_records(records)

    assert committed == rebuilt
    assert committed.memories[MEMORY_1_ID].trust_label is TrustLabel.UNTRUSTED_MODEL_OUTPUT
    assert rebuilt.memories[MEMORY_1_ID].trust_label is TrustLabel.UNTRUSTED_MODEL_OUTPUT
    assert delta.creates[0].trust_label is TrustLabel.TRUSTED_CONTROLLER
    rebuilt_cycle = rebuilt.cycles[cycles[-1].cycle_id]
    assert rebuilt_cycle.validated_delta is not None
    assert rebuilt_cycle.validated_delta.creates[0].trust_label is TrustLabel.TRUSTED_CONTROLLER
    validate_complete_projection(rebuilt)


def test_memory_delta_preview_rejects_invalid_assignments_without_mutation() -> None:
    projection, delta, assignments, committed = delta_preview_fixture()
    original = projection

    with pytest.raises(ProjectionInvariantError, match="exactly match created handles"):
        preview_memory_delta(
            projection,
            delta,
            assignments[:-1],
            last_event_sequence=committed.last_event_sequence,
        )

    duplicate_handle_assignments = (
        assignments[0],
        assignments[1].model_copy(update={"handle": assignments[0].handle}),
    )
    with pytest.raises(ProjectionInvariantError, match="handles must be unique"):
        preview_memory_delta(
            projection,
            delta,
            duplicate_handle_assignments,
            last_event_sequence=committed.last_event_sequence,
        )

    duplicate_id_assignments = (
        assignments[0],
        assignments[1].model_copy(update={"memory_id": assignments[0].memory_id}),
    )
    with pytest.raises(ProjectionInvariantError, match="IDs must be unique"):
        preview_memory_delta(
            projection,
            delta,
            duplicate_id_assignments,
            last_event_sequence=committed.last_event_sequence,
        )

    assert projection == original


def test_memory_delta_preview_rejects_cross_run_and_invalid_provenance() -> None:
    projection, delta, assignments, committed = delta_preview_fixture()

    with pytest.raises(CrossRunReferenceError, match="memory delta"):
        preview_memory_delta(
            projection,
            delta.model_copy(update={"run_id": OTHER_RUN_ID}),
            assignments,
            last_event_sequence=committed.last_event_sequence,
        )

    missing_provenance_delta = MemoryDelta(
        delta_id=DELTA_2_ID,
        run_id=RUN_ID,
        creates=(
            MemoryCreate(
                handle="missing-provenance",
                kind=MemoryKind.KNOWLEDGE,
                content="This memory cites an unavailable event.",
                provenance=(event_reference(MISSING_ID),),
                confidence=0.5,
                trust_label=TrustLabel.TRUSTED_CONTROLLER,
            ),
        ),
        created_at=delta.created_at,
    )
    with pytest.raises(CrossRunReferenceError, match="event evidence"):
        preview_memory_delta(
            projection,
            missing_provenance_delta,
            (MemoryIdAssignment(handle="missing-provenance", memory_id=MEMORY_4_ID),),
            last_event_sequence=committed.last_event_sequence,
        )

    with pytest.raises(ProjectionInvariantError, match="falls after the cycle range"):
        preview_memory_delta(
            projection,
            MemoryDelta(
                delta_id=DELTA_2_ID,
                run_id=RUN_ID,
                creates=(
                    MemoryCreate(
                        handle="future-provenance",
                        kind=MemoryKind.KNOWLEDGE,
                        content="This memory cites a future event.",
                        provenance=(event_reference(EVENT_2_ID),),
                        confidence=0.5,
                        trust_label=TrustLabel.TRUSTED_CONTROLLER,
                    ),
                ),
                created_at=delta.created_at,
            ),
            (MemoryIdAssignment(handle="future-provenance", memory_id=MEMORY_4_ID),),
            last_event_sequence=1,
        )


def test_memory_delta_preview_rejects_backward_memory_timestamp() -> None:
    projection = first_committed_projection()
    delta = MemoryDelta(
        delta_id=DELTA_2_ID,
        run_id=RUN_ID,
        updates=(
            MemoryUpdate(
                memory_id=MEMORY_1_ID,
                expected_revision=1,
                content="This update predates the current memory revision.",
            ),
        ),
        created_at=projection.memories[MEMORY_1_ID].updated_at - timedelta(microseconds=1),
    )

    with pytest.raises(ProjectionInvariantError, match="timestamp moved backwards"):
        preview_memory_delta(
            projection,
            delta,
            (),
            last_event_sequence=projection.memory_cursor,
        )


def test_later_committed_cycle_updates_exact_revision_and_preserves_history() -> None:
    projection = first_committed_projection()
    second_event = trace_event(2, EVENT_2_ID)
    second_decision = invocation(
        2,
        DECISION_2_ID,
        budget=projected_budget_snapshot(projection),
    )
    projection = apply_entry(projection, entry(7, second_event))
    projection = apply_entry(projection, entry(8, second_decision))
    delta = MemoryDelta(
        delta_id=DELTA_2_ID,
        run_id=RUN_ID,
        updates=(
            MemoryUpdate(
                memory_id=MEMORY_1_ID,
                expected_revision=1,
                content="Run tests from the repository root.",
            ),
        ),
        created_at=NOW + timedelta(seconds=23),
    )
    cycles = cycle_revisions(
        first_sequence=2,
        decision_id=DECISION_2_ID,
        delta=delta,
        assignments=(),
        intervention_id=INTERVENTION_2_ID,
    )
    for offset, cycle in enumerate(cycles, start=9):
        projection = apply_entry(projection, entry(offset, cycle))

    assert projection.memories[MEMORY_1_ID].revision == 2
    assert projection.memories[MEMORY_1_ID].content.endswith("repository root.")
    assert projection.memory_history[(MEMORY_1_ID, 1)].content.endswith("working directory.")
    assert projection.memory_cursor == 2


def test_stale_multi_update_is_atomic() -> None:
    projection = first_committed_projection()
    projection = apply_entry(projection, entry(7, trace_event(2, EVENT_2_ID)))
    projection = apply_entry(
        projection,
        entry(
            8,
            invocation(
                2,
                DECISION_2_ID,
                budget=projected_budget_snapshot(projection),
            ),
        ),
    )
    delta = MemoryDelta(
        delta_id=DELTA_2_ID,
        run_id=RUN_ID,
        updates=(
            MemoryUpdate(
                memory_id=MEMORY_1_ID,
                expected_revision=1,
                content="This valid-looking update must roll back.",
            ),
            MemoryUpdate(
                memory_id=MISSING_ID,
                expected_revision=1,
                content="Missing target.",
            ),
        ),
        created_at=NOW + timedelta(seconds=23),
    )
    cycles = cycle_revisions(
        first_sequence=2,
        decision_id=DECISION_2_ID,
        delta=delta,
        assignments=(),
        intervention_id=INTERVENTION_2_ID,
    )
    for offset, cycle in enumerate(cycles[:-1], start=9):
        projection = apply_entry(projection, entry(offset, cycle))
    before = projection

    with pytest.raises(RevisionConflictError):
        apply_entry(projection, entry(12, cycles[-1]))

    assert projection == before
    assert projection.memories[MEMORY_1_ID].revision == 1
    assert projection.cycles[cycles[-1].cycle_id].state is CycleState.RUNNING
    assert projection.memory_cursor == 1


def test_private_status_replacement_supersedes_the_previous_revision() -> None:
    projection = first_committed_projection(kind=MemoryKind.PRIVATE_STATUS)
    projection = apply_entry(projection, entry(7, trace_event(2, EVENT_2_ID)))
    projection = apply_entry(
        projection,
        entry(
            8,
            invocation(
                2,
                DECISION_2_ID,
                budget=projected_budget_snapshot(projection),
            ),
        ),
    )
    delta = MemoryDelta(
        delta_id=DELTA_2_ID,
        run_id=RUN_ID,
        private_status_replacement={
            "expected_memory_id": MEMORY_1_ID,
            "expected_revision": 1,
            "replacement": {
                "handle": "memory-two",
                "kind": MemoryKind.PRIVATE_STATUS,
                "content": "Tests now pass.",
                "provenance": (event_reference(EVENT_2_ID),),
                "confidence": 1.0,
                "trust_label": TrustLabel.TRUSTED_CONTROLLER,
            },
        },
        created_at=NOW + timedelta(seconds=23),
    )
    cycles = cycle_revisions(
        first_sequence=2,
        decision_id=DECISION_2_ID,
        delta=delta,
        assignments=(MemoryIdAssignment(handle="memory-two", memory_id=MEMORY_2_ID),),
        intervention_id=INTERVENTION_2_ID,
    )
    for offset, cycle in enumerate(cycles, start=9):
        projection = apply_entry(projection, entry(offset, cycle))

    assert projection.current_private_status_id == MEMORY_2_ID
    assert projection.memories[MEMORY_1_ID].validity is ValidityState.SUPERSEDED
    assert projection.memories[MEMORY_1_ID].revision == 2
    assert projection.memories[MEMORY_2_ID].validity is ValidityState.ACTIVE
    assert projection.memories[MEMORY_2_ID].revision == 1


def test_invalidation_creates_a_revision_and_default_search_excludes_it() -> None:
    projection = first_committed_projection()
    projection = apply_entry(projection, entry(7, trace_event(2, EVENT_2_ID)))
    projection = apply_entry(
        projection,
        entry(
            8,
            invocation(
                2,
                DECISION_2_ID,
                budget=projected_budget_snapshot(projection),
            ),
        ),
    )
    delta = MemoryDelta(
        delta_id=DELTA_2_ID,
        run_id=RUN_ID,
        invalidations=(
            MemoryInvalidation(
                memory_id=MEMORY_1_ID,
                expected_revision=1,
                reason_code=ReasonCode.CONFLICT,
            ),
        ),
        created_at=NOW + timedelta(seconds=23),
    )
    cycles = cycle_revisions(
        first_sequence=2,
        decision_id=DECISION_2_ID,
        delta=delta,
        assignments=(),
        intervention_id=INTERVENTION_2_ID,
    )
    for offset, cycle in enumerate(cycles, start=9):
        projection = apply_entry(projection, entry(offset, cycle))

    invalidated = projection.memories[MEMORY_1_ID]
    assert invalidated.revision == 2
    assert invalidated.validity is ValidityState.INVALIDATED
    assert search(projection, MemoryQuery(run_id=RUN_ID, text="tests")) == ()
    hits = search(
        projection,
        MemoryQuery(
            run_id=RUN_ID,
            text="tests",
            validity=(ValidityState.INVALIDATED,),
        ),
    )
    assert hits[0].memory == invalidated
    assert hits[0].memory.access_count == 0


def test_cycle_cannot_skip_a_state_transition() -> None:
    event = trace_event(1, EVENT_1_ID)
    decision = invocation(1, DECISION_1_ID)
    projection = apply_records((event, decision))
    delta = MemoryDelta(
        delta_id=DELTA_1_ID,
        run_id=RUN_ID,
        created_at=NOW + timedelta(seconds=13),
    )
    pending, _reserved, running, _committed = cycle_revisions(
        first_sequence=1,
        decision_id=DECISION_1_ID,
        delta=delta,
        assignments=(),
        intervention_id=INTERVENTION_1_ID,
    )
    projection = apply_entry(projection, entry(3, pending))
    values = running.model_dump(mode="python")
    values["revision"] = 2
    running_at_revision_two = CycleRecord.model_validate(values)

    with pytest.raises(ProjectionInvariantError, match="transition"):
        apply_entry(projection, entry(4, running_at_revision_two))


def test_cycle_must_match_an_invoking_decision() -> None:
    event = trace_event(1, EVENT_1_ID)
    decision = invocation(1, DECISION_1_ID)
    delta = MemoryDelta(
        delta_id=DELTA_1_ID,
        run_id=RUN_ID,
        created_at=NOW + timedelta(seconds=13),
    )
    pending = cycle_revisions(
        first_sequence=1,
        decision_id=DECISION_1_ID,
        delta=delta,
        assignments=(),
        intervention_id=INTERVENTION_1_ID,
    )[0]

    non_invoking = decision.model_copy(update={"invoke": False})
    projection = apply_records((event, non_invoking))
    with pytest.raises(ProjectionInvariantError, match="invoking decision"):
        apply_entry(projection, entry(3, pending))

    projection = apply_records((event, decision))
    other_policy = "other/1"
    mismatched = pending.model_copy(
        update={
            "policy_version": other_policy,
            "cycle_id": cycle_id(
                RUN_ID,
                1,
                1,
                other_policy,
                CONFIGURATION_DIGEST,
                pending.grounding_version,
                pending.grounding_configuration_digest,
                pending.requested_delivery_target,
            ),
        }
    )
    with pytest.raises(ProjectionInvariantError, match="policy"):
        apply_entry(projection, entry(3, mismatched))

    preceding = pending.model_copy(
        update={
            "created_at": decision.created_at - timedelta(seconds=1),
            "updated_at": decision.created_at - timedelta(seconds=1),
        }
    )
    with pytest.raises(ProjectionInvariantError, match="precede"):
        apply_entry(projection, entry(3, preceding))


def test_cycle_budget_state_must_follow_the_transition_that_created_it() -> None:
    event = trace_event(1, EVENT_1_ID)
    decision = invocation(1, DECISION_1_ID)
    delta = MemoryDelta(
        delta_id=DELTA_1_ID,
        run_id=RUN_ID,
        created_at=NOW + timedelta(seconds=13),
    )
    pending, reserved, running, committed = cycle_revisions(
        first_sequence=1,
        decision_id=DECISION_1_ID,
        delta=delta,
        assignments=(),
        intervention_id=INTERVENTION_1_ID,
    )
    projection = apply_records((event, decision, pending))
    introduced_values = pending.model_dump(mode="python")
    introduced_values.update(
        revision=2,
        state=CycleState.FAILED,
        budget_reservation=reserved.budget_reservation,
        budget_settlement=reserved.budget_reservation,
        failure_reason=ReasonCode.MODEL_ERROR,
        updated_at=pending.updated_at + timedelta(seconds=1),
    )
    introduced = pending.model_copy(update=introduced_values)
    with pytest.raises(ProjectionInvariantError, match="before reservation"):
        apply_entry(
            projection,
            entry(4, pending).model_copy(update={"record": introduced}),
        )

    oversized = reserved.model_copy(update={"budget_reservation": BudgetAmounts(model_calls=11)})
    with pytest.raises(ProjectionInvariantError, match="decision budget"):
        apply_entry(projection, entry(4, oversized))

    projection = apply_entry(projection, entry(4, reserved))
    failed_before_running = reserved.model_copy(
        update={
            "revision": 3,
            "state": CycleState.FAILED,
            "budget_settlement": reserved.budget_reservation,
            "batch_digest": "a" * 64,
            "failure_reason": ReasonCode.MODEL_ERROR,
            "updated_at": reserved.updated_at + timedelta(seconds=1),
        }
    )
    with pytest.raises(ProjectionInvariantError, match="before running"):
        apply_entry(
            projection,
            entry(5, reserved).model_copy(update={"record": failed_before_running}),
        )
    unknown_before_running = failed_before_running.model_copy(update={"batch_digest": None})
    unknown_before_running = unknown_before_running.model_copy(
        update={"failure_reason": ReasonCode.FAILED_UNKNOWN_COST}
    )
    with pytest.raises(ProjectionInvariantError, match="previously running"):
        apply_entry(
            projection,
            entry(5, reserved).model_copy(update={"record": unknown_before_running}),
        )

    projection = apply_entry(projection, entry(5, running))
    over_settled = committed.model_copy(update={"budget_settlement": BudgetAmounts(model_calls=2)})
    with pytest.raises(ProjectionInvariantError, match="settlement"):
        apply_entry(
            projection,
            entry(6, committed).model_copy(update={"record": over_settled}),
        )


def test_committed_cycle_cannot_hide_model_calls_or_interventions() -> None:
    event = trace_event(1, EVENT_1_ID)
    decision = invocation(1, DECISION_1_ID)
    delta = MemoryDelta(
        delta_id=DELTA_1_ID,
        run_id=RUN_ID,
        created_at=NOW + timedelta(seconds=13),
    )
    pending, reserved, running, committed = cycle_revisions(
        first_sequence=1,
        decision_id=DECISION_1_ID,
        delta=delta,
        assignments=(),
        intervention_id=INTERVENTION_1_ID,
    )
    projection = apply_records((event, decision, pending, reserved, running))

    hidden_call = committed.model_copy(
        update={
            "budget_settlement": BudgetAmounts(),
            "model_call_digests": (),
            "model_call_latencies_us": (),
        }
    )
    with pytest.raises(ProjectionInvariantError, match="settle a model call"):
        apply_entry(
            projection,
            entry(6, committed).model_copy(update={"record": hidden_call}),
        )

    missing_digest = committed.model_copy(
        update={"model_call_digests": (), "model_call_latencies_us": ()}
    )
    with pytest.raises(ProjectionInvariantError, match="call digests"):
        apply_entry(
            projection,
            entry(6, committed).model_copy(update={"record": missing_digest}),
        )

    reminder = first_reminder_projection().cycles[committed.cycle_id].intervention
    assert reminder is not None
    unmetered_reminder = committed.model_copy(update={"intervention": reminder})
    with pytest.raises(ProjectionInvariantError, match="intervention usage"):
        apply_entry(
            projection,
            entry(6, committed).model_copy(update={"record": unmetered_reminder}),
        )


def test_known_running_failure_must_settle_its_attempted_calls() -> None:
    event = trace_event(1, EVENT_1_ID)
    decision = invocation(1, DECISION_1_ID)
    delta = MemoryDelta(
        delta_id=DELTA_1_ID,
        run_id=RUN_ID,
        created_at=NOW + timedelta(seconds=13),
    )
    pending, reserved, running, _committed = cycle_revisions(
        first_sequence=1,
        decision_id=DECISION_1_ID,
        delta=delta,
        assignments=(),
        intervention_id=INTERVENTION_1_ID,
    )
    projection = apply_records((event, decision, pending, reserved, running))
    failed = running.model_copy(
        update={
            "revision": 4,
            "state": CycleState.FAILED,
            "budget_settlement": BudgetAmounts(),
            "failure_reason": ReasonCode.MODEL_ERROR,
            "updated_at": running.updated_at + timedelta(seconds=1),
        }
    )

    with pytest.raises(ProjectionInvariantError, match="settle a model call"):
        apply_entry(
            projection,
            entry(6, running).model_copy(update={"record": failed}),
        )


def test_cycle_cannot_commit_evidence_from_a_future_event() -> None:
    first_event = trace_event(1, EVENT_1_ID)
    decision = invocation(1, DECISION_1_ID)
    future_event = trace_event(2, EVENT_2_ID)
    delta = MemoryDelta(
        delta_id=DELTA_1_ID,
        run_id=RUN_ID,
        creates=(
            MemoryCreate(
                handle="future-memory",
                kind=MemoryKind.KNOWLEDGE,
                content="Future evidence must not enter the earlier cycle.",
                provenance=(event_reference(EVENT_2_ID),),
                confidence=0.8,
                trust_label=TrustLabel.TRUSTED_CONTROLLER,
            ),
        ),
        created_at=NOW + timedelta(seconds=13),
    )
    cycles = cycle_revisions(
        first_sequence=1,
        decision_id=DECISION_1_ID,
        delta=delta,
        assignments=(MemoryIdAssignment(handle="future-memory", memory_id=MEMORY_1_ID),),
        intervention_id=INTERVENTION_1_ID,
    )
    projection = apply_records((first_event, decision, *cycles[:3], future_event))

    with pytest.raises(ProjectionInvariantError, match="after the cycle range"):
        apply_entry(projection, entry(7, cycles[-1]))


def test_event_signal_and_decision_indexes_reject_duplicates_and_missing_links() -> None:
    first_event = trace_event(1, EVENT_1_ID)
    projection = apply_records((first_event,))

    with pytest.raises(ProjectionInvariantError, match="sequence"):
        apply_entry(projection, entry(2, trace_event(3, EVENT_2_ID)))
    with pytest.raises(ProjectionInvariantError, match="event ID"):
        apply_entry(projection, entry(2, trace_event(2, EVENT_1_ID)))
    duplicate_source = trace_event(2, EVENT_2_ID).model_copy(
        update={"source_event_id": first_event.source_event_id}
    )
    with pytest.raises(ProjectionInvariantError, match="source event"):
        apply_entry(projection, entry(2, duplicate_source))
    missing_parent = trace_event(2, EVENT_2_ID).model_copy(update={"parent_ids": (MISSING_ID,)})
    with pytest.raises(CrossRunReferenceError, match="parent"):
        apply_entry(projection, entry(2, missing_parent))

    signal = Signal(
        signal_id=SIGNAL_ID,
        run_id=RUN_ID,
        created_at=NOW,
        signal_type=SignalType.TOOL_ERROR,
        strength=1.0,
        evidence_event_ids=(EVENT_1_ID,),
        detector_version="fixture/1",
        reason_code=ReasonCode.TOOL_ERROR,
    )
    projection = apply_entry(projection, entry(2, signal))
    with pytest.raises(ProjectionInvariantError, match="signal ID"):
        apply_entry(projection, entry(3, signal))
    missing_signal = signal.model_copy(
        update={
            "signal_id": UUID("00000000-0000-4000-8000-000000000210"),
            "evidence_event_ids": (EVENT_2_ID,),
        }
    )
    with pytest.raises(CrossRunReferenceError, match="evidence"):
        apply_entry(projection, entry(3, missing_signal))

    decision = invocation(1, DECISION_1_ID)
    projection = apply_entry(projection, entry(3, decision))
    with pytest.raises(ProjectionInvariantError, match="decision ID"):
        apply_entry(projection, entry(4, decision))
    missing_decision = invocation(2, DECISION_2_ID)
    with pytest.raises(CrossRunReferenceError, match="invocation event"):
        apply_entry(projection, entry(4, missing_decision))


def test_outcome_and_delivery_projection_validate_references_and_revisions() -> None:
    projection = first_reminder_projection()
    with pytest.raises(ProjectionInvariantError, match="missing its delivery outbox"):
        validate_complete_projection(projection)
    outcome = InterventionOutcome(
        outcome_id=OUTCOME_ID,
        run_id=RUN_ID,
        intervention_id=INTERVENTION_1_ID,
        repeated_error_status=RepeatedErrorStatus.NOT_OBSERVED,
        constraint_status=ConstraintStatus.NOT_OBSERVED,
        evidence_mode=OutcomeEvidenceMode.POLICY_REPLAY,
        created_at=NOW + timedelta(seconds=20),
    )
    projection = apply_entry(projection, entry(7, outcome))
    with pytest.raises(ProjectionInvariantError, match="outcome ID"):
        apply_entry(projection, entry(8, outcome))
    missing_outcome = outcome.model_copy(
        update={
            "outcome_id": UUID("00000000-0000-4000-8000-000000000211"),
            "intervention_id": INTERVENTION_2_ID,
        }
    )
    with pytest.raises(CrossRunReferenceError, match="intervention"):
        apply_entry(projection, entry(8, missing_outcome))

    cycle = next(iter(projection.cycles.values()))
    delivery = DeliveryRecord(
        delivery_id=DELIVERY_ID,
        run_id=RUN_ID,
        revision=1,
        cycle_id=cycle.cycle_id,
        intervention_id=INTERVENTION_1_ID,
        rendered_text_digest=canonical_digest(
            projection.interventions[INTERVENTION_1_ID].rendered_text
        ),
        target_request_id="request-1",
        target=DeliveryTarget.NEXT_MODEL_CALL,
        state=DeliveryState.PENDING,
        attempt_count=0,
        adapter_id="fixture",
        adapter_deduplicates=True,
        adapter_deduplication_guarantee=DeduplicationGuarantee.DURABLE_DELIVERY_ID,
        adapter_supports_pre_action=True,
        adapter_contract_version="adapter-contract/v1",
        adapter_capabilities_digest="8" * 64,
        created_at=cycle.updated_at,
        updated_at=cycle.updated_at,
    )
    silence_projection = first_committed_projection()
    validate_complete_projection(silence_projection)
    with pytest.raises(ProjectionInvariantError, match="authorized"):
        apply_entry(silence_projection, entry(7, delivery))
    wrong_target = delivery.model_copy(update={"target": DeliveryTarget.PRE_ACTION_REPLAN})
    with pytest.raises(ProjectionInvariantError, match="authorized"):
        apply_entry(projection, entry(8, wrong_target))
    wrong_reminder = delivery.model_copy(update={"rendered_text_digest": "0" * 64})
    with pytest.raises(ProjectionInvariantError, match="reminder digest"):
        apply_entry(projection, entry(8, wrong_reminder))
    future_pending = delivery.model_copy(
        update={"updated_at": delivery.updated_at + timedelta(seconds=1)}
    )
    with pytest.raises(ProjectionInvariantError, match="enqueued with its cycle"):
        apply_entry(projection, entry(8, future_pending))
    claimed_first = delivery.model_copy(
        update={"state": DeliveryState.CLAIMED, "claim_id": CLAIM_1_ID}
    )
    with pytest.raises(ProjectionInvariantError, match="begin pending"):
        apply_entry(projection, entry(8, claimed_first))
    projection = apply_entry(projection, entry(8, delivery))
    validate_complete_projection(projection)
    duplicate_delivery = delivery.model_copy(
        update={
            "delivery_id": UUID("00000000-0000-4000-8000-000000000214"),
        }
    )
    with pytest.raises(ProjectionInvariantError, match="already has a delivery"):
        apply_entry(projection, entry(9, duplicate_delivery))
    claimed_values = delivery.model_dump(mode="python")
    claimed_values.update(
        revision=2,
        state=DeliveryState.CLAIMED,
        claim_id=CLAIM_1_ID,
        updated_at=NOW + timedelta(seconds=22),
    )
    claimed = DeliveryRecord.model_validate(claimed_values)
    changed_target = claimed.model_copy(update={"target_request_id": "different-request"})
    with pytest.raises(ProjectionInvariantError, match="identity fields"):
        apply_entry(projection, entry(9, changed_target))
    projection = apply_entry(projection, entry(9, claimed))
    assert projection.deliveries[DELIVERY_ID] == claimed

    delivered_values = claimed.model_dump(mode="python")
    delivered_values.update(
        revision=3,
        state=DeliveryState.DELIVERED,
        attempt_count=1,
        attempt_id=ATTEMPT_1_ID,
        receipt={"provider_receipt_id": "accepted"},
        outcome=DeliveryOutcome.DELIVERED,
        reason_code=ReasonCode.DELIVERY_SUCCEEDED,
        updated_at=NOW + timedelta(seconds=23),
    )
    delivered_without_attempt = DeliveryRecord.model_validate(delivered_values)
    with pytest.raises(ProjectionInvariantError, match="state transition"):
        apply_entry(projection, entry(10, delivered_without_attempt))

    attempting_values = claimed.model_dump(mode="python")
    attempting_values.update(
        revision=3,
        state=DeliveryState.ATTEMPTING,
        attempt_count=1,
        attempt_id=ATTEMPT_1_ID,
        updated_at=NOW + timedelta(seconds=23),
    )
    attempting = DeliveryRecord.model_validate(attempting_values)
    backwards = attempting.model_copy(update={"updated_at": delivery.created_at})
    with pytest.raises(ProjectionInvariantError, match="timestamp"):
        apply_entry(projection, entry(10, backwards))
    skipped_attempt = attempting.model_copy(update={"attempt_count": 2})
    with pytest.raises(ProjectionInvariantError, match="attempt count"):
        apply_entry(projection, entry(10, skipped_attempt))
    projection = apply_entry(projection, entry(10, attempting))

    delivered_values["revision"] = 4
    delivered = DeliveryRecord.model_validate(delivered_values)
    skipped_count_values = dict(delivered_values)
    skipped_count_values["attempt_count"] = 2
    skipped_count = DeliveryRecord.model_validate(skipped_count_values)
    with pytest.raises(ProjectionInvariantError, match="outside an attempt"):
        apply_entry(projection, entry(11, skipped_count))
    projection = apply_entry(projection, entry(11, delivered))
    terminal_revision = delivered.model_copy(
        update={"revision": 5, "updated_at": NOW + timedelta(seconds=24)}
    )
    with pytest.raises(ProjectionInvariantError, match="state transition"):
        apply_entry(projection, entry(12, terminal_revision))

    skipped = claimed.model_copy(update={"revision": 4})
    with pytest.raises(ProjectionInvariantError, match="revision"):
        apply_entry(projection, entry(12, skipped))

    missing_cycle = delivery.model_copy(
        update={
            "delivery_id": UUID("00000000-0000-4000-8000-000000000212"),
            "cycle_id": "0" * 64,
        }
    )
    with pytest.raises(CrossRunReferenceError, match="cycle"):
        apply_entry(projection, entry(12, missing_cycle))
    missing_intervention = delivery.model_copy(
        update={
            "delivery_id": UUID("00000000-0000-4000-8000-000000000213"),
            "intervention_id": INTERVENTION_2_ID,
        }
    )
    with pytest.raises(CrossRunReferenceError, match="intervention"):
        apply_entry(projection, entry(12, missing_intervention))


def test_non_deduplicating_delivery_cannot_retry_unknown_side_effect() -> None:
    projection = first_reminder_projection()
    cycle = next(iter(projection.cycles.values()))
    pending = DeliveryRecord(
        delivery_id=DELIVERY_ID,
        run_id=RUN_ID,
        revision=1,
        cycle_id=cycle.cycle_id,
        intervention_id=INTERVENTION_1_ID,
        rendered_text_digest=canonical_digest(
            projection.interventions[INTERVENTION_1_ID].rendered_text
        ),
        target_request_id="request-1",
        target=DeliveryTarget.NEXT_MODEL_CALL,
        state=DeliveryState.PENDING,
        attempt_count=0,
        adapter_id="fixture",
        adapter_deduplicates=False,
        adapter_deduplication_guarantee=DeduplicationGuarantee.AT_MOST_ONCE_ATTEMPT,
        adapter_supports_pre_action=True,
        adapter_contract_version="adapter-contract/v1",
        adapter_capabilities_digest="8" * 64,
        created_at=cycle.updated_at,
        updated_at=cycle.updated_at,
    )
    claimed = pending.model_copy(
        update={
            "revision": 2,
            "state": DeliveryState.CLAIMED,
            "claim_id": CLAIM_1_ID,
            "updated_at": NOW + timedelta(seconds=22),
        }
    )
    attempting = claimed.model_copy(
        update={
            "revision": 3,
            "state": DeliveryState.ATTEMPTING,
            "attempt_count": 1,
            "attempt_id": ATTEMPT_1_ID,
            "updated_at": NOW + timedelta(seconds=23),
        }
    )
    unknown = attempting.model_copy(
        update={
            "revision": 4,
            "state": DeliveryState.UNKNOWN,
            "outcome": DeliveryOutcome.UNKNOWN,
            "reason_code": ReasonCode.DELIVERY_UNKNOWN,
            "updated_at": NOW + timedelta(seconds=24),
        }
    )
    for position, record in enumerate((pending, claimed, attempting, unknown), start=7):
        projection = apply_entry(projection, entry(position, record))
    retry = unknown.model_copy(
        update={
            "revision": 5,
            "state": DeliveryState.CLAIMED,
            "claim_id": CLAIM_2_ID,
            "attempt_id": None,
            "outcome": None,
            "reason_code": None,
            "updated_at": NOW + timedelta(seconds=25),
        }
    )

    with pytest.raises(ProjectionInvariantError, match="cannot retry"):
        forged_entry = entry(11, unknown).model_copy(update={"record": retry})
        apply_entry(projection, forged_entry)


def test_projector_defensive_guards_reject_stale_cross_run_and_untyped_state() -> None:
    projection = first_committed_projection()
    memory = projection.memories[MEMORY_1_ID]

    with pytest.raises(RevisionConflictError):
        _validate_memory_target(projection, MEMORY_1_ID, 2)
    inactive = memory.model_copy(update={"validity": ValidityState.INVALIDATED})
    with pytest.raises(ProjectionInvariantError, match="active"):
        _validate_memory_target(
            replace(projection, memories={MEMORY_1_ID: inactive}),
            MEMORY_1_ID,
            1,
        )
    missing_event = event_reference(EVENT_2_ID)
    with pytest.raises(CrossRunReferenceError, match="event evidence"):
        _validate_evidence(projection, (missing_event,))
    missing_memory = EvidenceReference(
        source=EvidenceSource.MEMORY,
        source_id=MISSING_ID,
        revision=1,
        field_path="/content",
    )
    with pytest.raises(CrossRunReferenceError, match="memory evidence"):
        _validate_evidence(projection, (missing_memory,))
    with pytest.raises(ProjectionInvariantError, match="invalid"):
        _record_values(memory, updated_at=memory.created_at - timedelta(seconds=1))
    invalid_create = MemoryCreate(
        handle="invalid-expiration",
        kind=MemoryKind.KNOWLEDGE,
        content="Expiration precedes projected creation.",
        provenance=(event_reference(EVENT_1_ID),),
        confidence=0.5,
        trust_label=TrustLabel.TRUSTED_CONTROLLER,
        expires_at=NOW,
    )
    with pytest.raises(ProjectionInvariantError, match="invalid"):
        _new_memory(
            run_id=RUN_ID,
            memory_id=MISSING_ID,
            create=invalid_create,
            created_at=NOW + timedelta(seconds=1),
        )
    with pytest.raises(ProjectionInvariantError, match="already exists"):
        _store_memory(
            dict(projection.memory_history),
            dict(projection.memories),
            memory,
        )
    with pytest.raises(TypeError, match="dictionary"):
        _models(())
    with pytest.raises(TypeError, match="index key"):
        _models({object(): memory})

    other_run_event = trace_event(2, EVENT_2_ID).model_copy(
        update={"run_id": UUID("00000000-0000-4000-8000-000000000299")}
    )
    cross_run_entry = entry(7, trace_event(2, EVENT_2_ID)).model_copy(
        update={"record": other_run_event}
    )
    with pytest.raises(CrossRunReferenceError, match="ledger record"):
        apply_entry(projection, cross_run_entry)
    unsupported = entry(7, trace_event(2, EVENT_2_ID)).model_copy(update={"record": memory})
    with pytest.raises(ProjectionInvariantError, match="unsupported"):
        apply_entry(projection, unsupported)


def test_cycle_requires_existing_decision_range_and_pending_initial_state() -> None:
    event = trace_event(1, EVENT_1_ID)
    delta = MemoryDelta(
        delta_id=DELTA_1_ID,
        run_id=RUN_ID,
        created_at=NOW + timedelta(seconds=13),
    )
    pending, reserved, running, _committed = cycle_revisions(
        first_sequence=1,
        decision_id=DECISION_1_ID,
        delta=delta,
        assignments=(),
        intervention_id=INTERVENTION_1_ID,
    )
    projection = apply_records((event,))
    with pytest.raises(CrossRunReferenceError, match="decision"):
        apply_entry(projection, entry(2, pending))

    projection = apply_entry(projection, entry(2, invocation(1, DECISION_1_ID)))
    missing_range = cycle_revisions(
        first_sequence=2,
        decision_id=DECISION_1_ID,
        delta=delta,
        assignments=(),
        intervention_id=INTERVENTION_1_ID,
    )[0]
    with pytest.raises(CrossRunReferenceError, match="range"):
        apply_entry(projection, entry(3, missing_range))

    running_values = running.model_dump(mode="python")
    running_values["revision"] = 1
    running_first = CycleRecord.model_validate(running_values)
    with pytest.raises(ProjectionInvariantError, match="begin pending"):
        apply_entry(projection, entry(3, running_first))

    projection = apply_entry(projection, entry(3, pending))
    reserved_values = reserved.model_dump(mode="python")
    reserved_values["created_at"] = reserved.created_at + timedelta(microseconds=1)
    changed_reserved = CycleRecord.model_validate(reserved_values)
    with pytest.raises(ProjectionInvariantError, match="identity"):
        apply_entry(projection, entry(4, changed_reserved))


def test_search_filters_kind_trust_and_nonmatching_text_without_side_effects() -> None:
    projection = first_committed_projection()
    memory = projection.memories[MEMORY_1_ID]

    assert (
        search(
            projection,
            MemoryQuery(run_id=RUN_ID, kinds=(MemoryKind.PROCEDURAL,)),
        )
        == ()
    )
    untrusted = search(
        projection,
        MemoryQuery(
            run_id=RUN_ID,
            trust_labels=(TrustLabel.UNTRUSTED_MODEL_OUTPUT,),
        ),
    )
    assert tuple(hit.memory for hit in untrusted) == (memory,)
    assert (
        search(
            projection,
            MemoryQuery(
                run_id=RUN_ID,
                trust_labels=(TrustLabel.TRUSTED_CONTROLLER,),
            ),
        )
        == ()
    )
    assert (
        search(
            projection,
            MemoryQuery(run_id=RUN_ID, text="not-present"),
        )
        == ()
    )
    assert projection.memories[MEMORY_1_ID] == memory
