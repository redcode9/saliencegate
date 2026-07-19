from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from saliencegate.domain import (
    CycleRecord,
    CycleState,
    DeliveryRecord,
    InterventionDecision,
    InterventionOutcome,
    InvocationDecision,
    MemoryDelta,
    MemoryRecord,
    NormalizedTraceEventDraft,
    ReasonCode,
    RedactedTraceEventDraft,
    RuntimeRecord,
    Signal,
    SignalType,
    TraceEvent,
)


def test_every_record_is_frozen_and_forbids_extra_fields(
    sample_records: tuple[RuntimeRecord, ...],
) -> None:
    for record in sample_records:
        with pytest.raises(ValidationError, match="frozen"):
            record.schema_version = "9.0"

        values = record.model_dump(mode="python")
        values["unexpected"] = True
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            type(record).model_validate(values)


def test_json_payload_is_deeply_immutable_and_detached() -> None:
    source = {"items": [{"value": 1}]}
    event = TraceEvent.model_validate(
        {
            "event_id": "00000000-0000-4000-8000-000000000002",
            "run_id": "00000000-0000-4000-8000-000000000001",
            "sequence": 1,
            "source_event_id": "source-1",
            "timestamp": datetime(2026, 7, 11, tzinfo=UTC),
            "event_type": "observation",
            "phase": "post_action",
            "payload": source,
            "payload_digest": {
                "algorithm": "synthetic_sha256",
                "value": "a" * 64,
            },
            "source_adapter": "fixture",
            "trust_label": "untrusted_tool_output",
        },
        strict=False,
    )

    source["items"][0]["value"] = 2
    assert event.payload["items"][0]["value"] == 1

    with pytest.raises(TypeError):
        event.payload["items"][0]["value"] = 3


def test_mutating_a_dump_does_not_mutate_the_record(trace_event: TraceEvent) -> None:
    dumped = trace_event.model_dump(mode="python")
    dumped["payload"]["nested"]["a"] = 99

    assert trace_event.payload["nested"]["a"] == 1


@pytest.mark.parametrize(
    "timestamp",
    [
        datetime(2026, 7, 11, 12, 30),
        datetime(2026, 7, 11, 13, 30, tzinfo=timezone(timedelta(hours=1))),
    ],
)
def test_trace_events_require_utc_not_merely_an_aware_timestamp(
    trace_event: TraceEvent,
    timestamp: datetime,
) -> None:
    values = trace_event.model_dump(mode="python")
    values["timestamp"] = timestamp

    with pytest.raises(ValidationError, match="UTC"):
        TraceEvent.model_validate(values)


def test_record_timestamps_serialize_in_one_utc_form(trace_event: TraceEvent) -> None:
    assert trace_event.model_dump(mode="json")["timestamp"] == "2026-07-11T12:30:00Z"


@pytest.mark.parametrize(
    ("record_type", "field"),
    [
        (Signal, "strength"),
        (MemoryRecord, "confidence"),
        (InterventionDecision, "confidence"),
        (InterventionOutcome, "task_reward"),
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_numeric_fields_reject_non_finite_values(
    sample_records: tuple[RuntimeRecord, ...],
    record_type: type[RuntimeRecord],
    field: str,
    value: float,
) -> None:
    record = next(item for item in sample_records if isinstance(item, record_type))
    values = record.model_dump(mode="python")
    values[field] = value

    with pytest.raises(ValidationError):
        record_type.model_validate(values)


def test_json_payload_rejects_non_finite_numbers(trace_event: TraceEvent) -> None:
    values = trace_event.model_dump(mode="python")
    values["payload"]["score"] = float("nan")

    with pytest.raises(ValidationError, match="finite"):
        TraceEvent.model_validate(values)


def test_trace_event_draft_is_distinct_from_a_canonically_sequenced_event(
    trace_event: TraceEvent,
) -> None:
    values = trace_event.model_dump(mode="python")
    values.pop("event_id")
    values.pop("sequence")
    payload_digest = values.pop("payload_digest")
    values["record_type"] = "normalized_trace_event_draft"

    normalized = NormalizedTraceEventDraft.model_validate(values)
    assert normalized.source_event_id == trace_event.source_event_id

    values["record_type"] = "redacted_trace_event_draft"
    values["payload_digest"] = payload_digest
    redacted = RedactedTraceEventDraft.model_validate(values)
    assert redacted.payload_digest == trace_event.payload_digest

    persisted = trace_event.model_dump(mode="python")
    persisted.pop("sequence")
    with pytest.raises(ValidationError, match="Field required"):
        TraceEvent.model_validate(persisted)


def test_repository_owned_ids_must_be_supplied_for_replay(
    sample_records: tuple[RuntimeRecord, ...],
) -> None:
    id_fields = {
        "trace_event": "event_id",
        "signal": "signal_id",
        "memory_record": "memory_id",
        "invocation_decision": "decision_id",
        "memory_delta": "delta_id",
        "intervention_decision": "intervention_id",
        "intervention_outcome": "outcome_id",
        "delivery_record": "delivery_id",
    }
    for record in sample_records:
        id_field = id_fields.get(record.record_type)
        if id_field is None:
            continue
        values = record.model_dump(mode="python")
        values.pop(id_field)
        with pytest.raises(ValidationError, match="Field required"):
            type(record).model_validate(values)


def test_signal_reason_code_must_match_signal_type(
    sample_records: tuple[RuntimeRecord, ...],
) -> None:
    signal = next(item for item in sample_records if isinstance(item, Signal))
    values = signal.model_dump(mode="python")
    values["signal_type"] = SignalType.CONFLICT

    with pytest.raises(ValidationError, match="reason_code must match signal_type"):
        Signal.model_validate(values)


def test_reference_policy_decision_can_leave_risk_unestimated(
    sample_records: tuple[RuntimeRecord, ...],
) -> None:
    invocation = next(item for item in sample_records if isinstance(item, InvocationDecision))
    values = invocation.model_dump(mode="python")
    values["risk_score"] = None

    restored = InvocationDecision.model_validate(values)
    assert restored.risk_score is None
    assert restored.configuration_digest == "f" * 64


def test_cycle_state_cannot_claim_committed_outputs_while_pending(
    sample_records: tuple[RuntimeRecord, ...],
) -> None:
    cycle = next(item for item in sample_records if isinstance(item, CycleRecord))
    values = cycle.model_dump(mode="python")
    values["state"] = CycleState.PENDING

    with pytest.raises(ValidationError, match="pending cycle"):
        CycleRecord.model_validate(values)


def test_non_deduplicating_delivery_has_at_most_one_attempt(
    sample_records: tuple[RuntimeRecord, ...],
) -> None:
    delivery = next(item for item in sample_records if isinstance(item, DeliveryRecord))
    values = delivery.model_dump(mode="python")
    values["adapter_deduplicates"] = False
    values["attempt_count"] = 2

    with pytest.raises(ValidationError, match="at most once"):
        DeliveryRecord.model_validate(values)


def test_intervention_citation_indexes_must_match_claim_evidence(
    sample_records: tuple[RuntimeRecord, ...],
) -> None:
    decision = next(item for item in sample_records if isinstance(item, InterventionDecision))
    values = decision.model_dump(mode="python")
    values["cited_memory_ids"] = ()

    with pytest.raises(ValidationError, match="cited_memory_ids"):
        InterventionDecision.model_validate(values)


def test_memory_delta_rejects_duplicate_create_handles(
    sample_records: tuple[RuntimeRecord, ...],
) -> None:
    delta = next(item for item in sample_records if isinstance(item, MemoryDelta))
    values = delta.model_dump(mode="python")
    values["creates"] = (values["creates"][0], values["creates"][0])

    with pytest.raises(ValidationError, match="create handles"):
        MemoryDelta.model_validate(values)


def test_memory_delta_rejects_update_and_invalidation_of_the_same_revision(
    sample_records: tuple[RuntimeRecord, ...],
) -> None:
    delta = next(item for item in sample_records if isinstance(item, MemoryDelta))
    values = delta.model_dump(mode="python")
    values["invalidations"][0]["memory_id"] = values["updates"][0]["memory_id"]
    values["invalidations"][0]["expected_revision"] = values["updates"][0]["expected_revision"]

    with pytest.raises(ValidationError, match="both update and invalidate"):
        MemoryDelta.model_validate(values)


def test_failed_unknown_cost_is_a_failure_reason_not_a_lifecycle_state(
    sample_records: tuple[RuntimeRecord, ...],
) -> None:
    cycle = next(item for item in sample_records if isinstance(item, CycleRecord))
    values = cycle.model_dump(mode="python")
    values.update(
        state=CycleState.FAILED,
        validated_delta=None,
        memory_id_assignments=(),
        intervention=None,
        failure_reason=ReasonCode.FAILED_UNKNOWN_COST,
    )
    values["budget_settlement"] = values["budget_reservation"]

    failed = CycleRecord.model_validate(values)
    assert failed.state is CycleState.FAILED
    assert failed.failure_reason is ReasonCode.FAILED_UNKNOWN_COST


def test_stage_one_reason_codes_have_stable_wire_values() -> None:
    required_values = {
        "always_invoke",
        "budget_exhausted",
        "citation_cross_run",
        "delivery_unknown",
        "failed_unknown_cost",
        "invalid_structured_output",
        "mandatory_input_overflow",
        "never_invoke",
        "script_exhausted",
        "source_event_collision",
        "unsafe_role_mapping",
        "ungrounded",
    }
    assert required_values <= {reason.value for reason in ReasonCode}
