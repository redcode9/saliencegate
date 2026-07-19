from __future__ import annotations

from datetime import UTC, datetime
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
    RuntimeRecord,
    Signal,
    SignalType,
    TextSpan,
    TraceEvent,
    TrustLabel,
    UtilityLabel,
    ValidityState,
    cycle_id,
)

RUN_ID = UUID("00000000-0000-4000-8000-000000000001")
EVENT_ID = UUID("00000000-0000-4000-8000-000000000002")
MEMORY_ID = UUID("00000000-0000-4000-8000-000000000003")
INTERVENTION_ID = UUID("00000000-0000-4000-8000-000000000004")
SIGNAL_ID = UUID("00000000-0000-4000-8000-000000000006")
INVOCATION_ID = UUID("00000000-0000-4000-8000-000000000007")
DELTA_ID = UUID("00000000-0000-4000-8000-000000000008")
OUTCOME_ID = UUID("00000000-0000-4000-8000-000000000009")
DELIVERY_ID = UUID("00000000-0000-4000-8000-00000000000a")
CREATED_MEMORY_ID = UUID("00000000-0000-4000-8000-00000000000b")
CONFIGURATION_DIGEST = "f" * 64
GROUNDING_CONFIGURATION_DIGEST = "9" * 64
GROUNDING_VERSION = "fixture-grounding/1"
GROUNDING_CONFIGURATION = {"max_claims": 2, "renderer": "fixed"}
REQUESTED_DELIVERY_TARGET = DeliveryTarget.NEXT_MODEL_CALL
CYCLE_ID = cycle_id(
    RUN_ID,
    1,
    1,
    "scripted/1",
    CONFIGURATION_DIGEST,
    GROUNDING_VERSION,
    GROUNDING_CONFIGURATION_DIGEST,
    REQUESTED_DELIVERY_TARGET,
)
NOW = datetime(2026, 7, 11, 12, 30, tzinfo=UTC)


@pytest.fixture
def trace_event() -> TraceEvent:
    return TraceEvent(
        event_id=EVENT_ID,
        run_id=RUN_ID,
        sequence=1,
        source_event_id="adapter-event-1",
        timestamp=NOW,
        event_type=EventType.TOOL_COMPLETION,
        phase=EventPhase.POST_ACTION,
        payload={"message": "café", "nested": {"b": 2, "a": 1}},
        payload_digest=PayloadDigest(
            algorithm=PayloadDigestAlgorithm.SYNTHETIC_SHA256,
            value="a" * 64,
        ),
        parent_ids=(),
        source_adapter="fixture",
        trust_label=TrustLabel.UNTRUSTED_TOOL_OUTPUT,
    )


@pytest.fixture
def sample_records(trace_event: TraceEvent) -> tuple[RuntimeRecord, ...]:
    event_ref = EvidenceReference(
        source=EvidenceSource.EVENT,
        source_id=EVENT_ID,
        field_path="/payload/message",
        span=TextSpan(start_byte=0, end_byte=5),
    )
    memory_ref = EvidenceReference(
        source=EvidenceSource.MEMORY,
        source_id=MEMORY_ID,
        revision=1,
        field_path="/content",
    )
    budget_limits = BudgetLimits(
        model_calls=10,
        input_tokens=10_000,
        output_tokens=2_000,
        canonical_token_equivalents=12_000,
        latency_us=30_000_000,
        max_call_latency_us=5_000_000,
        interventions=5,
        schema_repairs=2,
    )
    budget_reserved = BudgetAmounts(
        model_calls=1,
        input_tokens=1_000,
        output_tokens=100,
        canonical_token_equivalents=600,
        latency_us=250_000,
        interventions=1,
    )
    budget = BudgetSnapshot(
        limits=budget_limits,
        reserved=budget_reserved,
        consumed=BudgetAmounts(),
    )
    signal = Signal(
        signal_id=SIGNAL_ID,
        run_id=RUN_ID,
        created_at=NOW,
        signal_type=SignalType.TOOL_ERROR,
        strength=0.9,
        evidence_event_ids=(EVENT_ID,),
        detector_version="tool-errors/1",
        reason_code=ReasonCode.TOOL_ERROR,
    )
    memory = MemoryRecord(
        memory_id=MEMORY_ID,
        run_id=RUN_ID,
        kind=MemoryKind.KNOWLEDGE,
        content="The test command exits with status 1.",
        provenance=(event_ref,),
        confidence=0.9,
        validity=ValidityState.ACTIVE,
        revision=1,
        created_at=NOW,
        updated_at=NOW,
        access_count=0,
        trust_label=TrustLabel.TRUSTED_CONTROLLER,
    )
    invocation = InvocationDecision(
        decision_id=INVOCATION_ID,
        run_id=RUN_ID,
        event_sequence=1,
        invoke=True,
        risk_score=0.9,
        reason_codes=(ReasonCode.TOOL_ERROR,),
        policy_version="scripted/1",
        configuration_digest=CONFIGURATION_DIGEST,
        budget_snapshot=budget,
        cooldown_active=False,
        created_at=NOW,
    )
    delta = MemoryDelta(
        delta_id=DELTA_ID,
        run_id=RUN_ID,
        creates=(
            MemoryCreate(
                handle="new-diagnosis",
                kind=MemoryKind.PROCEDURAL,
                content="The failing command needs a different working directory.",
                provenance=(event_ref,),
                confidence=0.8,
                trust_label=TrustLabel.TRUSTED_CONTROLLER,
            ),
        ),
        updates=(
            MemoryUpdate(
                memory_id=MEMORY_ID,
                expected_revision=1,
                confidence=0.95,
            ),
        ),
        invalidations=(
            MemoryInvalidation(
                memory_id=UUID("00000000-0000-4000-8000-000000000005"),
                expected_revision=2,
                reason_code=ReasonCode.CONFLICT,
            ),
        ),
        created_at=NOW,
    )
    claim = InterventionClaim(
        kind=ClaimKind.ENVIRONMENT_FACT,
        fields={"fact": "Run tests from the repository root."},
        evidence=(memory_ref,),
    )
    intervention = InterventionDecision(
        intervention_id=INTERVENTION_ID,
        run_id=RUN_ID,
        cycle_id=CYCLE_ID,
        grounding_version=GROUNDING_VERSION,
        grounding_configuration=GROUNDING_CONFIGURATION,
        grounding_configuration_digest=GROUNDING_CONFIGURATION_DIGEST,
        grounding_receipt={"status": "fixture-verified"},
        action=InterventionAction.REMIND,
        delivery_target=DeliveryTarget.NEXT_MODEL_CALL,
        claims=(claim,),
        rendered_text="Relevant evidence: Run tests from the repository root.",
        cited_memory_ids=(MEMORY_ID,),
        cited_event_ids=(),
        confidence=0.9,
        reason_code=ReasonCode.GROUNDED_REMINDER,
        ttl_steps=1,
        created_at=NOW,
    )
    outcome = InterventionOutcome(
        outcome_id=OUTCOME_ID,
        run_id=RUN_ID,
        intervention_id=INTERVENTION_ID,
        next_action_fingerprint="b" * 64,
        repeated_error_status=RepeatedErrorStatus.AVOIDED,
        constraint_status=ConstraintStatus.RESPECTED,
        evidence_mode=OutcomeEvidenceMode.DETERMINISTIC_ORACLE,
        utility=UtilityLabel.HELPFUL,
        task_reward=1.0,
        memory_calls=1,
        input_tokens=500,
        output_tokens=100,
        canonical_token_equivalents=600,
        latency_us=250_000,
        created_at=NOW,
    )
    cycle = CycleRecord(
        cycle_id=CYCLE_ID,
        run_id=RUN_ID,
        revision=1,
        invocation_decision_id=INVOCATION_ID,
        policy_version="scripted/1",
        configuration_digest=CONFIGURATION_DIGEST,
        grounding_version=GROUNDING_VERSION,
        grounding_configuration=GROUNDING_CONFIGURATION,
        grounding_configuration_digest=GROUNDING_CONFIGURATION_DIGEST,
        requested_delivery_target=REQUESTED_DELIVERY_TARGET,
        first_event_sequence=1,
        last_event_sequence=1,
        state=CycleState.COMMITTED,
        budget_reservation=budget_reserved,
        budget_settlement=BudgetAmounts(
            model_calls=1,
            input_tokens=500,
            output_tokens=100,
            canonical_token_equivalents=600,
            latency_us=250_000,
            interventions=1,
        ),
        batch_digest="d" * 64,
        model_call_digests=("e" * 64,),
        model_call_latencies_us=(250_000,),
        validated_delta=delta,
        memory_id_assignments=(
            MemoryIdAssignment(handle="new-diagnosis", memory_id=CREATED_MEMORY_ID),
        ),
        intervention=intervention,
        created_at=NOW,
        updated_at=NOW,
    )
    delivery = DeliveryRecord(
        delivery_id=DELIVERY_ID,
        run_id=RUN_ID,
        revision=1,
        cycle_id=CYCLE_ID,
        intervention_id=INTERVENTION_ID,
        rendered_text_digest="7" * 64,
        target_request_id="request-2",
        target=DeliveryTarget.NEXT_MODEL_CALL,
        state=DeliveryState.DELIVERED,
        attempt_count=1,
        adapter_id="fixture-adapter",
        adapter_deduplicates=True,
        adapter_deduplication_guarantee=DeduplicationGuarantee.DURABLE_DELIVERY_ID,
        adapter_supports_pre_action=True,
        adapter_contract_version="adapter-contract/v1",
        adapter_capabilities_digest="8" * 64,
        claim_id=UUID("00000000-0000-4000-8000-00000000000c"),
        attempt_id=UUID("00000000-0000-4000-8000-00000000000d"),
        receipt={"provider_receipt_id": "receipt-1"},
        outcome=DeliveryOutcome.DELIVERED,
        reason_code=ReasonCode.DELIVERY_SUCCEEDED,
        created_at=NOW,
        updated_at=NOW,
    )
    return (
        trace_event,
        signal,
        memory,
        invocation,
        delta,
        intervention,
        outcome,
        cycle,
        delivery,
    )
