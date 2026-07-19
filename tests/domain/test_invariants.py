from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from saliencegate.domain import (
    MAX_MEMORY_CONTENT_BYTES,
    MAX_MEMORY_DELTA_ITEMS,
    MAX_TRACE_EVENT_PAYLOAD_BYTES,
    MAX_TRACE_EVENT_PAYLOAD_DEPTH,
    BudgetAmounts,
    BudgetLimits,
    BudgetSnapshot,
    CycleRecord,
    CycleState,
    DeduplicationGuarantee,
    DeliveryOutcome,
    DeliveryRecord,
    DeliveryState,
    DeliveryTarget,
    EvidenceReference,
    EvidenceSource,
    ExpirationAction,
    ExpirationPatch,
    InterventionAction,
    InterventionClaim,
    InterventionDecision,
    InterventionOutcome,
    InvocationDecision,
    LedgerRecord,
    MemoryCreate,
    MemoryDelta,
    MemoryIdAssignment,
    MemoryKind,
    MemoryRecord,
    MemoryUpdate,
    OutcomeEvidenceMode,
    PrivateStatusReplacement,
    ReasonCode,
    SignalType,
    TextSpan,
    TraceEvent,
    TrustLabel,
    UtilityLabel,
    ValidityState,
    cycle_id,
)

OTHER_RUN_ID = UUID("10000000-0000-4000-8000-000000000001")

FORCED_INVOKE_REASONS = (
    ReasonCode.POLICY_ALWAYS,
    ReasonCode.SCRIPTED_INVOKE,
    ReasonCode.BOOTSTRAP,
    ReasonCode.WATCHDOG,
    ReasonCode.HARD_SIGNAL,
    ReasonCode.RISK_THRESHOLD_MET,
)
FORCED_SILENCE_REASONS = (
    ReasonCode.POLICY_NEVER,
    ReasonCode.SCRIPTED_SILENCE,
    ReasonCode.SCRIPT_EXHAUSTED,
    ReasonCode.BUDGET_EXHAUSTED,
    ReasonCode.COOLDOWN_ACTIVE,
    ReasonCode.RISK_BELOW_THRESHOLD,
)
SIGNAL_REASONS = tuple(ReasonCode(signal_type.value) for signal_type in SignalType)
INVOCATION_REASONS = frozenset(FORCED_INVOKE_REASONS + FORCED_SILENCE_REASONS + SIGNAL_REASONS)
INTERVENTION_SILENCE_REASONS = (
    ReasonCode.SILENCE_SELECTED,
    ReasonCode.NO_GROUNDED_CLAIMS,
    ReasonCode.SCHEMA_INVALID,
    ReasonCode.CLAIM_OVER_LIMIT,
    ReasonCode.CITATION_MISSING,
    ReasonCode.CITATION_CROSS_RUN,
    ReasonCode.CITATION_EXPIRED,
    ReasonCode.CITATION_INVALIDATED,
    ReasonCode.INVALID_PROVENANCE,
    ReasonCode.UNGROUNDED,
    ReasonCode.DUPLICATE_REMINDER,
    ReasonCode.COOLDOWN_BLOCKED,
    ReasonCode.UNSUPPORTED_DELIVERY_TARGET,
)


def _record(records: tuple[LedgerRecord, ...], record_type: type[object]) -> object:
    return next(item for item in records if isinstance(item, record_type))


def test_json_payload_accepts_finite_floats_and_thaws_arrays(trace_event: TraceEvent) -> None:
    values = trace_event.model_dump(mode="python")
    values["payload"] = {"values": [1.5, True, None]}

    restored = TraceEvent.model_validate(values)
    assert restored.model_dump(mode="python")["payload"] == {"values": [1.5, True, None]}


def test_json_payload_rejects_non_json_objects(trace_event: TraceEvent) -> None:
    values = trace_event.model_dump(mode="python")
    values["payload"] = {"bad": object()}

    with pytest.raises(ValidationError, match="unsupported JSON"):
        TraceEvent.model_validate(values)


def test_trace_payload_bytes_and_depth_are_bounded(trace_event: TraceEvent) -> None:
    values = trace_event.model_dump(mode="python")
    values["payload"] = {"text": "x" * (MAX_TRACE_EVENT_PAYLOAD_BYTES + 1)}
    with pytest.raises(ValidationError, match="structural bound"):
        TraceEvent.model_validate(values)

    nested: dict[str, object] = {}
    root = nested
    for _ in range(MAX_TRACE_EVENT_PAYLOAD_DEPTH + 1):
        child: dict[str, object] = {}
        nested["child"] = child
        nested = child
    values["payload"] = root
    with pytest.raises(ValidationError, match="structural bound"):
        TraceEvent.model_validate(values)


def test_memory_delta_cardinality_and_content_are_bounded() -> None:
    evidence = (
        EvidenceReference(
            source=EvidenceSource.EVENT,
            source_id=UUID("00000000-0000-4000-8000-000000000002"),
            field_path="/payload/message",
        ),
    )
    with pytest.raises(ValidationError, match="UTF-8 byte bound"):
        MemoryCreate(
            handle="oversized",
            kind=MemoryKind.PROCEDURAL,
            content="x" * (MAX_MEMORY_CONTENT_BYTES + 1),
            provenance=evidence,
            confidence=1.0,
            trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
        )

    create = MemoryCreate(
        handle="bounded",
        kind=MemoryKind.PROCEDURAL,
        content="bounded",
        provenance=evidence,
        confidence=1.0,
        trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
    )
    with pytest.raises(ValidationError):
        MemoryDelta(
            delta_id=UUID("00000000-0000-4000-8000-000000000003"),
            run_id=UUID("00000000-0000-4000-8000-000000000001"),
            creates=(create,) * (MAX_MEMORY_DELTA_ITEMS + 1),
            created_at=datetime(2026, 7, 11, tzinfo=UTC),
        )


def test_text_span_requires_a_nonempty_half_open_range() -> None:
    assert TextSpan(start_byte=0, end_byte=1).end_byte == 1
    maximum_offset = (1 << 63) - 1
    assert (
        TextSpan(start_byte=maximum_offset - 1, end_byte=maximum_offset).end_byte == maximum_offset
    )
    with pytest.raises(ValidationError, match="greater than start"):
        TextSpan(start_byte=1, end_byte=1)
    with pytest.raises(ValidationError):
        TextSpan(start_byte=maximum_offset + 1, end_byte=maximum_offset + 2)
    with pytest.raises(ValidationError):
        TextSpan(start_byte=0, end_byte=maximum_offset + 1)


@pytest.mark.parametrize(
    "values",
    [
        {"source": EvidenceSource.MEMORY, "revision": None, "field_path": "/content"},
        {"source": EvidenceSource.EVENT, "revision": 1, "field_path": "/payload"},
        {"source": EvidenceSource.EVENT, "revision": None, "field_path": None},
        {"source": EvidenceSource.EVENT, "revision": None},
    ],
)
def test_evidence_references_identify_a_valid_source_and_selector(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        EvidenceReference(
            source_id=UUID("00000000-0000-4000-8000-000000000002"),
            **values,
        )


def test_evidence_reference_accepts_an_optional_span_within_a_pointed_to_field() -> None:
    reference = EvidenceReference(
        source=EvidenceSource.EVENT,
        source_id=UUID("00000000-0000-4000-8000-000000000002"),
        field_path="/payload/message",
        span=TextSpan(start_byte=0, end_byte=4),
    )
    assert reference.field_path == "/payload/message"
    assert reference.span == TextSpan(start_byte=0, end_byte=4)


def test_evidence_reference_requires_a_bounded_stable_json_pointer() -> None:
    source_id = UUID("00000000-0000-4000-8000-000000000002")
    with pytest.raises(ValidationError):
        EvidenceReference(
            source=EvidenceSource.EVENT,
            source_id=source_id,
            field_path="payload.message",
        )

    maximum_pointer = "/" + "a" * 1023
    assert (
        EvidenceReference(
            source=EvidenceSource.EVENT,
            source_id=source_id,
            field_path=maximum_pointer,
        ).field_path
        == maximum_pointer
    )
    with pytest.raises(ValidationError):
        EvidenceReference(
            source=EvidenceSource.EVENT,
            source_id=source_id,
            field_path=maximum_pointer + "a",
        )


def test_budget_snapshot_rejects_allocations_above_a_limit() -> None:
    with pytest.raises(ValidationError, match="model_calls exceed"):
        BudgetSnapshot(
            limits=BudgetLimits(model_calls=1, max_call_latency_us=1_000),
            reserved=BudgetAmounts(model_calls=1),
            consumed=BudgetAmounts(model_calls=1),
        )


@pytest.mark.parametrize(
    ("reason", "invoke"),
    tuple((reason, True) for reason in FORCED_INVOKE_REASONS)
    + tuple((reason, False) for reason in FORCED_SILENCE_REASONS)
    + tuple((reason, True) for reason in SIGNAL_REASONS),
)
def test_invocation_decision_accepts_consistent_trigger_reasons(
    sample_records: tuple[LedgerRecord, ...],
    reason: ReasonCode,
    invoke: bool,
) -> None:
    decision = _record(sample_records, InvocationDecision)
    assert isinstance(decision, InvocationDecision)
    values = decision.model_dump(mode="python")
    values.update(
        invoke=invoke,
        risk_score=0.5
        if reason
        in {
            ReasonCode.RISK_THRESHOLD_MET,
            ReasonCode.RISK_BELOW_THRESHOLD,
        }
        else None,
        reason_codes=(reason,),
        cooldown_active=reason is ReasonCode.COOLDOWN_ACTIVE,
    )

    restored = InvocationDecision.model_validate(values)
    assert restored.reason_codes == (reason,)


@pytest.mark.parametrize(
    "reason",
    tuple(reason for reason in ReasonCode if reason not in INVOCATION_REASONS),
)
def test_invocation_decision_rejects_non_trigger_reasons(
    sample_records: tuple[LedgerRecord, ...],
    reason: ReasonCode,
) -> None:
    decision = _record(sample_records, InvocationDecision)
    assert isinstance(decision, InvocationDecision)
    values = decision.model_dump(mode="python")
    values["reason_codes"] = (reason,)

    with pytest.raises(ValidationError, match="non-invocation reason"):
        InvocationDecision.model_validate(values)


@pytest.mark.parametrize(
    ("reason", "invoke", "message"),
    tuple((reason, False, "forced-invocation") for reason in FORCED_INVOKE_REASONS)
    + tuple((reason, True, "forced-silence") for reason in FORCED_SILENCE_REASONS),
)
def test_invocation_decision_rejects_reasons_that_contradict_the_decision(
    sample_records: tuple[LedgerRecord, ...],
    reason: ReasonCode,
    invoke: bool,
    message: str,
) -> None:
    decision = _record(sample_records, InvocationDecision)
    assert isinstance(decision, InvocationDecision)
    values = decision.model_dump(mode="python")
    values.update(
        invoke=invoke,
        reason_codes=(reason,),
        cooldown_active=reason is ReasonCode.COOLDOWN_ACTIVE,
    )

    with pytest.raises(ValidationError, match=message):
        InvocationDecision.model_validate(values)


@pytest.mark.parametrize(
    "reason",
    (ReasonCode.RISK_THRESHOLD_MET, ReasonCode.RISK_BELOW_THRESHOLD),
)
def test_invocation_threshold_reasons_require_a_risk_score(
    sample_records: tuple[LedgerRecord, ...],
    reason: ReasonCode,
) -> None:
    decision = _record(sample_records, InvocationDecision)
    assert isinstance(decision, InvocationDecision)
    values = decision.model_dump(mode="python")
    values.update(
        invoke=reason is ReasonCode.RISK_THRESHOLD_MET,
        risk_score=None,
        reason_codes=(reason,),
    )

    with pytest.raises(ValidationError, match="requires risk_score"):
        InvocationDecision.model_validate(values)


def test_invocation_cooldown_reason_requires_active_state(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    decision = _record(sample_records, InvocationDecision)
    assert isinstance(decision, InvocationDecision)
    values = decision.model_dump(mode="python")
    values.update(
        invoke=False,
        reason_codes=(ReasonCode.COOLDOWN_ACTIVE,),
        cooldown_active=False,
    )

    with pytest.raises(ValidationError, match="requires cooldown_active=True"):
        InvocationDecision.model_validate(values)


def test_invocation_reasons_are_unique_and_bounded(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    decision = _record(sample_records, InvocationDecision)
    assert isinstance(decision, InvocationDecision)
    values = decision.model_dump(mode="python")
    values["reason_codes"] = (ReasonCode.TOOL_ERROR,) * 2
    with pytest.raises(ValidationError, match="must be unique"):
        InvocationDecision.model_validate(values)

    values["reason_codes"] = (ReasonCode.TOOL_ERROR,) * 22
    with pytest.raises(ValidationError, match="at most 21"):
        InvocationDecision.model_validate(values)


@pytest.mark.parametrize("invoke", [False, True])
def test_tool_error_is_neutral_and_cooldown_state_alone_does_not_force_silence(
    sample_records: tuple[LedgerRecord, ...],
    invoke: bool,
) -> None:
    decision = _record(sample_records, InvocationDecision)
    assert isinstance(decision, InvocationDecision)
    values = decision.model_dump(mode="python")
    values.update(
        invoke=invoke,
        risk_score=None,
        reason_codes=(ReasonCode.TOOL_ERROR,),
        cooldown_active=True,
    )

    restored = InvocationDecision.model_validate(values)
    assert restored.invoke is invoke


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"updated_at_delta": -1}, "updated_at"),
        ({"last_accessed_at_delta": -1}, "last_accessed_at"),
        ({"expires_at_delta": 0}, "expires_at"),
        ({"validity": ValidityState.INVALIDATED}, "requires invalidated_at"),
        ({"invalidated_at_delta": 1}, "only invalidated memory"),
        (
            {"validity": ValidityState.INVALIDATED, "invalidated_at_delta": -1},
            "invalidated_at cannot precede",
        ),
    ],
)
def test_memory_timestamp_and_validity_invariants(
    sample_records: tuple[LedgerRecord, ...],
    changes: dict[str, object],
    message: str,
) -> None:
    memory = _record(sample_records, MemoryRecord)
    assert isinstance(memory, MemoryRecord)
    values = memory.model_dump(mode="python")
    created_at = values["created_at"]
    for key, value in changes.items():
        if key.endswith("_delta"):
            values[key.removesuffix("_delta")] = created_at + timedelta(seconds=int(value))
        else:
            values[key] = value

    with pytest.raises(ValidationError, match=message):
        MemoryRecord.model_validate(values)


def test_memory_update_requires_a_patch() -> None:
    with pytest.raises(ValidationError, match="at least one changed field"):
        MemoryUpdate(
            memory_id=UUID("00000000-0000-4000-8000-000000000003"),
            expected_revision=1,
        )


def test_expiration_patch_distinguishes_keep_set_and_clear() -> None:
    expires_at = datetime(2026, 7, 11, tzinfo=UTC)
    assert ExpirationPatch(action=ExpirationAction.CLEAR).value is None
    assert ExpirationPatch(action=ExpirationAction.SET, value=expires_at).value == expires_at
    with pytest.raises(ValidationError, match="requires a timestamp"):
        ExpirationPatch(action=ExpirationAction.SET)
    with pytest.raises(ValidationError, match="only an expiration set"):
        ExpirationPatch(
            action=ExpirationAction.CLEAR,
            value=expires_at,
        )


def test_memory_update_rejects_empty_provenance() -> None:
    with pytest.raises(ValidationError):
        MemoryUpdate(
            memory_id=UUID("00000000-0000-4000-8000-000000000003"),
            expected_revision=1,
            provenance=(),
        )


def test_private_status_replacement_is_typed_and_revision_checked(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    delta = _record(sample_records, MemoryDelta)
    assert isinstance(delta, MemoryDelta)
    source_create = delta.creates[0]
    private_create = MemoryCreate(
        handle="private-status",
        kind=MemoryKind.PRIVATE_STATUS,
        content="Testing remains in progress.",
        provenance=source_create.provenance,
        confidence=1.0,
        trust_label=TrustLabel.TRUSTED_CONTROLLER,
    )
    replacement = PrivateStatusReplacement(replacement=private_create)
    values = delta.model_dump(mode="python")
    values["private_status_replacement"] = replacement
    assert MemoryDelta.model_validate(values).private_status_replacement == replacement

    with pytest.raises(ValidationError, match="supplied together"):
        PrivateStatusReplacement(
            expected_memory_id=UUID("00000000-0000-4000-8000-000000000003"),
            replacement=private_create,
        )
    with pytest.raises(ValidationError, match="kind private_status"):
        PrivateStatusReplacement(replacement=source_create)


def test_memory_delta_rejects_private_status_in_general_creates(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    delta = _record(sample_records, MemoryDelta)
    assert isinstance(delta, MemoryDelta)
    values = delta.model_dump(mode="python")
    values["creates"][0]["kind"] = MemoryKind.PRIVATE_STATUS

    with pytest.raises(ValidationError, match="replacement"):
        MemoryDelta.model_validate(values)


def test_memory_delta_rejects_duplicate_update_and_invalidation_targets(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    delta = _record(sample_records, MemoryDelta)
    assert isinstance(delta, MemoryDelta)
    values = delta.model_dump(mode="python")
    values["updates"] = (values["updates"][0], values["updates"][0])
    with pytest.raises(ValidationError, match="update the same memory"):
        MemoryDelta.model_validate(values)

    values = delta.model_dump(mode="python")
    values["invalidations"] = (
        values["invalidations"][0],
        values["invalidations"][0],
    )
    with pytest.raises(ValidationError, match="invalidate the same memory"):
        MemoryDelta.model_validate(values)


def test_claim_rejects_duplicate_evidence(sample_records: tuple[LedgerRecord, ...]) -> None:
    decision = _record(sample_records, InterventionDecision)
    assert isinstance(decision, InterventionDecision)
    claim = decision.claims[0]

    with pytest.raises(ValidationError, match="duplicates"):
        InterventionClaim(kind=claim.kind, fields=claim.fields, evidence=claim.evidence * 2)


def test_claim_accepts_at_most_eight_evidence_references(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    decision = _record(sample_records, InterventionDecision)
    assert isinstance(decision, InterventionDecision)
    claim = decision.claims[0]
    references = tuple(
        EvidenceReference(
            source=EvidenceSource.EVENT,
            source_id=UUID(f"00000000-0000-4000-8000-{index:012x}"),
            field_path="/payload/message",
        )
        for index in range(1, 10)
    )

    accepted = InterventionClaim(
        kind=claim.kind,
        fields=claim.fields,
        evidence=references[:8],
    )
    assert len(accepted.evidence) == 8
    with pytest.raises(ValidationError):
        InterventionClaim(
            kind=claim.kind,
            fields=claim.fields,
            evidence=references,
        )


def test_intervention_sensitive_render_inputs_are_hidden_from_repr(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    decision = _record(sample_records, InterventionDecision)
    assert isinstance(decision, InterventionDecision)

    assert "Run tests from the repository root" not in repr(decision.claims[0])
    assert "max_claims" not in repr(decision)
    assert "Relevant evidence" not in repr(decision)


@pytest.mark.parametrize(
    "field_name",
    (
        "grounding_version",
        "grounding_configuration",
        "grounding_configuration_digest",
    ),
)
def test_intervention_requires_complete_grounding_identity(
    sample_records: tuple[LedgerRecord, ...],
    field_name: str,
) -> None:
    decision = _record(sample_records, InterventionDecision)
    assert isinstance(decision, InterventionDecision)
    values = decision.model_dump(mode="python")
    values.pop(field_name)

    with pytest.raises(ValidationError):
        InterventionDecision.model_validate(values)


def test_policy_replay_cannot_claim_causal_utility(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    outcome = _record(sample_records, InterventionOutcome)
    assert isinstance(outcome, InterventionOutcome)
    values = outcome.model_dump(mode="python")
    values.update(
        evidence_mode=OutcomeEvidenceMode.POLICY_REPLAY,
        utility=UtilityLabel.HELPFUL,
    )
    with pytest.raises(ValidationError, match="cannot assign a causal"):
        InterventionOutcome.model_validate(values)

    values["utility"] = None
    assert InterventionOutcome.model_validate(values).utility is None


def test_event_citation_index_must_match_claim_evidence(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    decision = _record(sample_records, InterventionDecision)
    delta = _record(sample_records, MemoryDelta)
    assert isinstance(decision, InterventionDecision)
    assert isinstance(delta, MemoryDelta)
    values = decision.model_dump(mode="python")
    values["claims"][0]["evidence"] += delta.creates[0].provenance

    with pytest.raises(ValidationError, match="cited_event_ids"):
        InterventionDecision.model_validate(values)


@pytest.mark.parametrize("reason_code", INTERVENTION_SILENCE_REASONS)
def test_valid_silence_has_no_claims_and_an_allowlisted_reason(
    sample_records: tuple[LedgerRecord, ...],
    reason_code: ReasonCode,
) -> None:
    decision = _record(sample_records, InterventionDecision)
    assert isinstance(decision, InterventionDecision)
    values = decision.model_dump(mode="python")
    values.update(
        action=InterventionAction.SILENCE,
        delivery_target=None,
        claims=(),
        rendered_text=None,
        cited_memory_ids=(),
        cited_event_ids=(),
        reason_code=reason_code,
        ttl_steps=0,
    )

    silence = InterventionDecision.model_validate(values)
    assert silence.action is InterventionAction.SILENCE
    assert not silence.claims
    assert silence.reason_code is reason_code


def test_silence_rejects_claims_and_non_grounding_reasons(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    decision = _record(sample_records, InterventionDecision)
    assert isinstance(decision, InterventionDecision)
    values = decision.model_dump(mode="python")
    values.update(
        action=InterventionAction.SILENCE,
        delivery_target=None,
        rendered_text=None,
        reason_code=ReasonCode.SILENCE_SELECTED,
        ttl_steps=0,
    )
    with pytest.raises(ValidationError, match="cannot carry claims"):
        InterventionDecision.model_validate(values)

    values.update(claims=(), cited_memory_ids=(), cited_event_ids=())
    values["reason_code"] = ReasonCode.NO_INTERVENTION
    with pytest.raises(ValidationError, match="allowlisted grounding reason"):
        InterventionDecision.model_validate(values)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"action": InterventionAction.SILENCE}, "silence cannot carry"),
        ({"delivery_target": None}, "requires a delivery target"),
        ({"claims": (), "cited_memory_ids": ()}, "requires at least one"),
        ({"rendered_text": None}, "requires deterministically rendered"),
        ({"reason_code": ReasonCode.SILENCE_SELECTED}, "grounded_reminder"),
        ({"ttl_steps": 0}, "one-step time-to-live"),
        ({"ttl_steps": 2}, "one-step time-to-live"),
    ],
)
def test_intervention_action_invariants(
    sample_records: tuple[LedgerRecord, ...],
    changes: dict[str, object],
    message: str,
) -> None:
    decision = _record(sample_records, InterventionDecision)
    assert isinstance(decision, InterventionDecision)
    values = decision.model_dump(mode="python")
    values.update(changes)

    with pytest.raises(ValidationError, match=message):
        InterventionDecision.model_validate(values)


def test_silence_requires_zero_ttl(sample_records: tuple[LedgerRecord, ...]) -> None:
    decision = _record(sample_records, InterventionDecision)
    assert isinstance(decision, InterventionDecision)
    values = decision.model_dump(mode="python")
    values.update(
        action=InterventionAction.SILENCE,
        delivery_target=None,
        claims=(),
        rendered_text=None,
        cited_memory_ids=(),
        cited_event_ids=(),
        reason_code=ReasonCode.SILENCE_SELECTED,
        ttl_steps=1,
    )

    with pytest.raises(ValidationError, match="zero-step"):
        InterventionDecision.model_validate(values)


def _cycle_values(sample_records: tuple[LedgerRecord, ...]) -> dict[str, object]:
    cycle = _record(sample_records, CycleRecord)
    assert isinstance(cycle, CycleRecord)
    return cycle.model_dump(mode="python")


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value.update(first_event_sequence=2), "cannot precede"),
        (
            lambda value: value.update(updated_at=value["created_at"] - timedelta(seconds=1)),
            "updated_at",
        ),
        (
            lambda value: value["validated_delta"].update(run_id=OTHER_RUN_ID),
            "delta belongs",
        ),
        (
            lambda value: value["intervention"].update(run_id=OTHER_RUN_ID),
            "intervention belongs to a different run",
        ),
        (
            lambda value: value["intervention"].update(cycle_id="0" * 64),
            "different cycle",
        ),
        (
            lambda value: value["intervention"].update(grounding_version="different-grounding/1"),
            "grounding pin",
        ),
        (
            lambda value: value["intervention"].update(grounding_configuration={"different": True}),
            "grounding pin",
        ),
        (
            lambda value: value["intervention"].update(grounding_configuration_digest="8" * 64),
            "grounding pin",
        ),
        (
            lambda value: value.update(configuration_digest="0" * 64),
            "cycle_id does not match",
        ),
        (
            lambda value: value.update(grounding_configuration_digest="0" * 64),
            "cycle_id does not match",
        ),
    ],
)
def test_cycle_identity_and_time_invariants(
    sample_records: tuple[LedgerRecord, ...],
    mutator: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    values = _cycle_values(sample_records)
    mutator(values)
    with pytest.raises(ValidationError, match=message):
        CycleRecord.model_validate(values)


def test_cycle_requested_target_matches_a_reminder(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    values = _cycle_values(sample_records)
    values["requested_delivery_target"] = DeliveryTarget.PRE_ACTION_REPLAN
    values["cycle_id"] = cycle_id(
        values["run_id"],
        values["first_event_sequence"],
        values["last_event_sequence"],
        values["policy_version"],
        values["configuration_digest"],
        values["grounding_version"],
        values["grounding_configuration_digest"],
        values["requested_delivery_target"],
    )
    values["intervention"]["cycle_id"] = values["cycle_id"]

    with pytest.raises(ValidationError, match="delivery target"):
        CycleRecord.model_validate(values)


def test_intervention_requires_an_opaque_grounding_receipt(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    cycle = _record(sample_records, CycleRecord)
    assert isinstance(cycle, CycleRecord)
    intervention = cycle.intervention
    assert intervention is not None
    values = intervention.model_dump(mode="python")
    values.pop("grounding_receipt")

    with pytest.raises(ValidationError, match="Field required"):
        InterventionDecision.model_validate(values)


def test_existing_v1_cycle_serialization_omits_selector_provenance(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    cycle = _record(sample_records, CycleRecord)
    assert isinstance(cycle, CycleRecord)

    assert cycle.selector_provenance is None
    assert "selector_provenance" not in cycle.model_dump(mode="json", warnings=False)
    assert '"selector_provenance"' not in cycle.model_dump_json(warnings=False)


def test_pending_reserved_and_running_cycle_shapes(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    values = _cycle_values(sample_records)
    values.update(
        state=CycleState.PENDING,
        batch_digest=None,
        budget_reservation=None,
        budget_settlement=None,
        validated_delta=None,
        memory_id_assignments=(),
        intervention=None,
        model_call_digests=(),
        model_call_latencies_us=(),
    )
    assert CycleRecord.model_validate(values).state is CycleState.PENDING

    values.update(
        state=CycleState.RESERVED,
        budget_reservation=BudgetAmounts(model_calls=1),
    )
    assert CycleRecord.model_validate(values).state is CycleState.RESERVED
    values.update(state=CycleState.RUNNING, batch_digest="d" * 64)
    assert CycleRecord.model_validate(values).state is CycleState.RUNNING


def test_committed_cycle_resolves_every_created_memory_id(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    values = _cycle_values(sample_records)
    values["memory_id_assignments"] = ()
    with pytest.raises(ValidationError, match="exactly match created handles"):
        CycleRecord.model_validate(values)

    assignment = values["validated_delta"]["creates"][0]["handle"]
    memory_id = UUID("00000000-0000-4000-8000-00000000000b")
    values["memory_id_assignments"] = (
        MemoryIdAssignment(handle=assignment, memory_id=memory_id),
        MemoryIdAssignment(handle=assignment, memory_id=memory_id),
    )
    with pytest.raises(ValidationError, match="unique"):
        CycleRecord.model_validate(values)

    values = _cycle_values(sample_records)
    second_create = dict(values["validated_delta"]["creates"][0])
    second_create["handle"] = "second-memory"
    values["validated_delta"]["creates"] += (second_create,)
    values["memory_id_assignments"] = (
        MemoryIdAssignment(handle=assignment, memory_id=memory_id),
        MemoryIdAssignment(handle="second-memory", memory_id=memory_id),
    )
    with pytest.raises(ValidationError, match="memory IDs must be unique"):
        CycleRecord.model_validate(values)


@pytest.mark.parametrize(
    ("state", "changes", "message"),
    [
        (
            CycleState.PENDING,
            {"failure_reason": ReasonCode.MODEL_ERROR},
            "pending cycle cannot carry a failure",
        ),
        (CycleState.RESERVED, {"budget_reservation": None}, "requires a budget reservation"),
        (
            CycleState.RESERVED,
            {"budget_settlement": BudgetAmounts()},
            "cannot carry committed outputs",
        ),
        (
            CycleState.RUNNING,
            {"failure_reason": ReasonCode.MODEL_ERROR},
            "cannot carry a failure reason",
        ),
        (
            CycleState.COMMITTED,
            {"budget_settlement": None},
            "committed cycle requires",
        ),
        (
            CycleState.COMMITTED,
            {"failure_reason": ReasonCode.MODEL_ERROR},
            "committed cycle cannot carry",
        ),
        (CycleState.FAILED, {"failure_reason": None}, "failed cycle requires"),
        (
            CycleState.FAILED,
            {"failure_reason": ReasonCode.MODEL_ERROR},
            "failed cycle cannot carry committed",
        ),
        (
            CycleState.FAILED,
            {
                "failure_reason": ReasonCode.MODEL_ERROR,
                "validated_delta": None,
                "memory_id_assignments": (),
                "intervention": None,
                "budget_settlement": None,
            },
            "requires a budget settlement",
        ),
    ],
)
def test_cycle_lifecycle_shapes(
    sample_records: tuple[LedgerRecord, ...],
    state: CycleState,
    changes: dict[str, object],
    message: str,
) -> None:
    values = _cycle_values(sample_records)
    if state in (CycleState.PENDING, CycleState.RESERVED, CycleState.RUNNING):
        values.update(
            budget_settlement=None,
            validated_delta=None,
            memory_id_assignments=(),
            intervention=None,
            model_call_digests=(),
            model_call_latencies_us=(),
        )
    if state is CycleState.PENDING:
        values.update(batch_digest=None, budget_reservation=None)
    values["state"] = state
    values.update(changes)

    with pytest.raises(ValidationError, match=message):
        CycleRecord.model_validate(values)


def test_failed_cycle_always_reconciles_a_persisted_reservation(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    values = _cycle_values(sample_records)
    values.update(
        state=CycleState.FAILED,
        validated_delta=None,
        memory_id_assignments=(),
        intervention=None,
        failure_reason=ReasonCode.MODEL_ERROR,
        budget_settlement=None,
    )
    with pytest.raises(ValidationError, match="requires a budget settlement"):
        CycleRecord.model_validate(values)

    values.update(
        budget_reservation=None,
        budget_settlement=BudgetAmounts(),
        batch_digest=None,
    )
    with pytest.raises(ValidationError, match="before reservation"):
        CycleRecord.model_validate(values)


def test_failed_unknown_cost_consumes_the_full_reservation(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    values = _cycle_values(sample_records)
    values.update(
        state=CycleState.FAILED,
        validated_delta=None,
        memory_id_assignments=(),
        intervention=None,
        failure_reason=ReasonCode.FAILED_UNKNOWN_COST,
    )
    with pytest.raises(ValidationError, match="full reservation"):
        CycleRecord.model_validate(values)


def test_cycle_budget_accounting_is_self_consistent(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    values = _cycle_values(sample_records)
    values["budget_reservation"] = BudgetAmounts()
    with pytest.raises(ValidationError, match="reservation requires a model call"):
        CycleRecord.model_validate(values)

    values = _cycle_values(sample_records)
    settlement = values["budget_settlement"]
    assert isinstance(settlement, dict)
    settlement["input_tokens"] = values["budget_reservation"]["input_tokens"] + 1
    with pytest.raises(ValidationError, match="cannot exceed"):
        CycleRecord.model_validate(values)

    values = _cycle_values(sample_records)
    values["model_call_digests"] = ()
    values["model_call_latencies_us"] = ()
    with pytest.raises(ValidationError, match="digests must match"):
        CycleRecord.model_validate(values)

    values = _cycle_values(sample_records)
    values["model_call_latencies_us"] = ()
    with pytest.raises(ValidationError, match="equal length"):
        CycleRecord.model_validate(values)

    values = _cycle_values(sample_records)
    values["model_call_latencies_us"] = (values["budget_settlement"]["latency_us"] + 1,)
    with pytest.raises(ValidationError, match="exceeds settled"):
        CycleRecord.model_validate(values)

    values = _cycle_values(sample_records)
    values["budget_settlement"]["interventions"] = 0
    with pytest.raises(ValidationError, match="intervention usage"):
        CycleRecord.model_validate(values)


def test_failed_cycle_accounting_matches_its_last_started_stage(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    values = _cycle_values(sample_records)
    values.update(
        state=CycleState.FAILED,
        validated_delta=None,
        memory_id_assignments=(),
        intervention=None,
        failure_reason=ReasonCode.MODEL_ERROR,
        budget_settlement=BudgetAmounts(),
        model_call_digests=(),
        model_call_latencies_us=(),
    )
    with pytest.raises(ValidationError, match="running failure"):
        CycleRecord.model_validate(values)

    values.update(
        batch_digest=None,
        budget_settlement=BudgetAmounts(model_calls=1),
        model_call_digests=("f" * 64,),
        model_call_latencies_us=(0,),
    )
    with pytest.raises(ValidationError, match="before running"):
        CycleRecord.model_validate(values)

    values.update(
        failure_reason=ReasonCode.FAILED_UNKNOWN_COST,
        budget_settlement=values["budget_reservation"],
        model_call_digests=(),
        model_call_latencies_us=(),
    )
    with pytest.raises(ValidationError, match="requires a running cycle"):
        CycleRecord.model_validate(values)

    values = _cycle_values(sample_records)
    values.update(
        state=CycleState.FAILED,
        validated_delta=None,
        memory_id_assignments=(),
        intervention=None,
        failure_reason=ReasonCode.FAILED_UNKNOWN_COST,
        budget_settlement=values["budget_reservation"],
        model_call_digests=("f" * 64, "e" * 64),
        model_call_latencies_us=(0, 0),
    )
    with pytest.raises(ValidationError, match="exceed the reservation"):
        CycleRecord.model_validate(values)


def test_cycle_stage_rejects_out_of_stage_budget_artifacts(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    pending = _cycle_values(sample_records)
    pending.update(
        state=CycleState.PENDING,
        batch_digest=None,
        budget_reservation=None,
        budget_settlement=None,
        validated_delta=None,
        memory_id_assignments=(),
        intervention=None,
        failure_reason=None,
        model_call_digests=(),
        model_call_latencies_us=(),
    )
    with pytest.raises(ValidationError, match="reservations or committed outputs"):
        CycleRecord.model_validate({**pending, "budget_reservation": BudgetAmounts(model_calls=1)})
    with pytest.raises(ValidationError, match=r"pending cycle.*receipts"):
        CycleRecord.model_validate(
            {
                **pending,
                "model_call_digests": ("f" * 64,),
                "model_call_latencies_us": (0,),
            }
        )

    reserved = {
        **pending,
        "state": CycleState.RESERVED,
        "budget_reservation": BudgetAmounts(model_calls=1),
    }
    with pytest.raises(ValidationError, match="running cycle requires"):
        CycleRecord.model_validate({**reserved, "state": CycleState.RUNNING})
    with pytest.raises(ValidationError, match=r"uncommitted cycle.*receipts"):
        CycleRecord.model_validate(
            {
                **reserved,
                "model_call_digests": ("f" * 64,),
                "model_call_latencies_us": (0,),
            }
        )

    committed = _cycle_values(sample_records)
    with pytest.raises(ValidationError, match="requires a batch digest"):
        CycleRecord.model_validate({**committed, "batch_digest": None})
    zero_call = dict(committed)
    zero_call["budget_settlement"] = BudgetAmounts(interventions=1)
    zero_call["model_call_digests"] = ()
    zero_call["model_call_latencies_us"] = ()
    with pytest.raises(ValidationError, match="settle a model call"):
        CycleRecord.model_validate(zero_call)

    failed = _cycle_values(sample_records)
    failed.update(
        state=CycleState.FAILED,
        validated_delta=None,
        memory_id_assignments=(),
        intervention=None,
        failure_reason=ReasonCode.MODEL_ERROR,
    )
    with pytest.raises(ValidationError, match="cannot consume an intervention"):
        CycleRecord.model_validate(failed)
    failed_settlement = dict(failed["budget_settlement"])
    failed_settlement["interventions"] = 0
    with pytest.raises(ValidationError, match="digests must match"):
        CycleRecord.model_validate(
            {
                **failed,
                "budget_settlement": failed_settlement,
                "model_call_digests": (),
                "model_call_latencies_us": (),
            }
        )

    failed_before_reservation = {
        **pending,
        "state": CycleState.FAILED,
        "failure_reason": ReasonCode.MODEL_ERROR,
        "model_call_digests": ("f" * 64,),
        "model_call_latencies_us": (0,),
    }
    with pytest.raises(ValidationError, match=r"before reservation.*receipts"):
        CycleRecord.model_validate(failed_before_reservation)


def _delivery_values(sample_records: tuple[LedgerRecord, ...]) -> dict[str, object]:
    delivery = _record(sample_records, DeliveryRecord)
    assert isinstance(delivery, DeliveryRecord)
    return delivery.model_dump(mode="python")


def test_delivery_timestamp_and_pending_shape(sample_records: tuple[LedgerRecord, ...]) -> None:
    values = _delivery_values(sample_records)
    values["updated_at"] = values["created_at"] - timedelta(seconds=1)
    with pytest.raises(ValidationError, match="updated_at"):
        DeliveryRecord.model_validate(values)

    values = _delivery_values(sample_records)
    values.update(
        state=DeliveryState.PENDING,
        attempt_count=0,
        claim_id=None,
        attempt_id=None,
        receipt=None,
        outcome=None,
        reason_code=None,
    )
    assert DeliveryRecord.model_validate(values).state is DeliveryState.PENDING
    values["attempt_count"] = 1
    with pytest.raises(ValidationError, match="pending delivery"):
        DeliveryRecord.model_validate(values)


@pytest.mark.parametrize(
    ("state", "changes", "message"),
    [
        (DeliveryState.DELIVERED, {"receipt": None}, "bounded successful receipt"),
        (
            DeliveryState.DELIVERED,
            {"outcome": DeliveryOutcome.FAILED},
            "bounded successful receipt",
        ),
        (
            DeliveryState.UNKNOWN,
            {"outcome": DeliveryOutcome.FAILED},
            "requires an attempt and unknown outcome",
        ),
        (
            DeliveryState.FAILED,
            {"outcome": DeliveryOutcome.DELIVERED},
            "requires an attempt and failed outcome",
        ),
    ],
)
def test_delivery_terminal_state_invariants(
    sample_records: tuple[LedgerRecord, ...],
    state: DeliveryState,
    changes: dict[str, object],
    message: str,
) -> None:
    values = _delivery_values(sample_records)
    values["state"] = state
    values.update(changes)
    with pytest.raises(ValidationError, match=message):
        DeliveryRecord.model_validate(values)


def test_unknown_and_failed_delivery_shapes(sample_records: tuple[LedgerRecord, ...]) -> None:
    values = _delivery_values(sample_records)
    values.update(
        state=DeliveryState.UNKNOWN,
        receipt=None,
        outcome=DeliveryOutcome.UNKNOWN,
        reason_code=ReasonCode.DELIVERY_UNKNOWN,
    )
    assert DeliveryRecord.model_validate(values).state is DeliveryState.UNKNOWN

    values.update(
        state=DeliveryState.FAILED,
        outcome=DeliveryOutcome.FAILED,
        reason_code=ReasonCode.DELIVERY_FAILED,
    )
    assert DeliveryRecord.model_validate(values).state is DeliveryState.FAILED


def test_claimed_attempting_and_rejected_delivery_shapes(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    values = _delivery_values(sample_records)
    values.update(
        state=DeliveryState.CLAIMED,
        attempt_count=0,
        attempt_id=None,
        receipt=None,
        outcome=None,
        reason_code=None,
    )
    assert DeliveryRecord.model_validate(values).state is DeliveryState.CLAIMED

    values.update(
        state=DeliveryState.ATTEMPTING,
        attempt_count=1,
        attempt_id=UUID("00000000-0000-4000-8000-00000000000d"),
    )
    assert DeliveryRecord.model_validate(values).state is DeliveryState.ATTEMPTING
    values["attempt_count"] = 0
    with pytest.raises(ValidationError, match="attempting delivery"):
        DeliveryRecord.model_validate(values)

    values.update(
        state=DeliveryState.REJECTED,
        attempt_count=0,
        attempt_id=None,
        receipt=None,
        outcome=DeliveryOutcome.REFUSED,
        reason_code=ReasonCode.UNSAFE_ROLE_MAPPING,
    )
    assert DeliveryRecord.model_validate(values).state is DeliveryState.REJECTED
    values["attempt_id"] = UUID("00000000-0000-4000-8000-00000000000d")
    with pytest.raises(ValidationError, match="rejected state"):
        DeliveryRecord.model_validate(values)


def test_delivery_adapter_guarantees_are_structurally_consistent(
    sample_records: tuple[LedgerRecord, ...],
) -> None:
    values = _delivery_values(sample_records)
    values["adapter_deduplication_guarantee"] = DeduplicationGuarantee.AT_MOST_ONCE_ATTEMPT
    with pytest.raises(ValidationError, match="deduplication declaration"):
        DeliveryRecord.model_validate(values)

    values = _delivery_values(sample_records)
    values.update(
        target=DeliveryTarget.PRE_ACTION_REPLAN,
        adapter_supports_pre_action=False,
    )
    with pytest.raises(ValidationError, match="interception capability"):
        DeliveryRecord.model_validate(values)
