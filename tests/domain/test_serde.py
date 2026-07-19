from __future__ import annotations

import json
from typing import get_args

import pytest
from pydantic import ValidationError

from saliencegate.domain import (
    CanonicalJSONError,
    InterventionDecision,
    InvalidSchemaVersionError,
    InvocationDecision,
    LedgerRecord,
    MemoryDelta,
    MemoryRecord,
    ReasonCode,
    RuntimeRecord,
    Signal,
    TraceEvent,
    UnknownRecordTypeError,
    UnsupportedSchemaVersionError,
    canonical_digest,
    canonical_json,
    load_record,
)


def test_canonical_json_has_exact_stable_bytes(trace_event: TraceEvent) -> None:
    assert canonical_json(trace_event) == (
        b'{"event_id":"00000000-0000-4000-8000-000000000002",'
        b'"event_type":"tool_completion","parent_ids":[],"payload":{"message":"caf\xc3\xa9",'
        b'"nested":{"a":1,"b":2}},"payload_digest":{"algorithm":"synthetic_sha256",'
        b'"value":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},'
        b'"phase":"post_action","record_type":"trace_event",'
        b'"run_id":"00000000-0000-4000-8000-000000000001","schema_version":"1.0",'
        b'"sequence":1,"source_adapter":"fixture","source_event_id":"adapter-event-1",'
        b'"timestamp":"2026-07-11T12:30:00Z","trust_label":"untrusted_tool_output"}'
    )


def test_mapping_order_does_not_change_canonical_bytes(trace_event: TraceEvent) -> None:
    first = trace_event.model_dump(mode="python")
    second = dict(reversed(first.items()))
    second["payload"] = dict(reversed(second["payload"].items()))

    assert canonical_json(TraceEvent.model_validate(first)) == canonical_json(
        TraceEvent.model_validate(second)
    )


def test_every_record_round_trips_with_the_same_type_and_bytes(
    sample_records: tuple[RuntimeRecord, ...],
) -> None:
    for record in sample_records:
        encoded = canonical_json(record)
        restored = load_record(encoded)

        assert type(restored) is type(record)
        assert canonical_json(restored) == encoded
        assert restored.schema_version == "1.0"


def test_signal_reason_code_invariant_survives_deserialization_boundaries(
    sample_records: tuple[RuntimeRecord, ...],
) -> None:
    signal = next(item for item in sample_records if isinstance(item, Signal))
    payload = signal.model_dump(mode="json")
    payload["reason_code"] = ReasonCode.CONFLICT.value

    with pytest.raises(ValidationError, match="reason_code must match signal_type"):
        load_record(canonical_json(payload))


def test_invocation_reason_invariants_survive_deserialization_boundaries(
    sample_records: tuple[RuntimeRecord, ...],
) -> None:
    decision = next(item for item in sample_records if isinstance(item, InvocationDecision))
    payload = decision.model_dump(mode="json")
    payload.update(
        invoke=False,
        risk_score=None,
        reason_codes=[ReasonCode.BUDGET_EXHAUSTED.value],
    )

    restored = load_record(canonical_json(payload))
    assert isinstance(restored, InvocationDecision)
    assert restored.reason_codes == (ReasonCode.BUDGET_EXHAUSTED,)

    payload["invoke"] = True
    with pytest.raises(ValidationError, match="forced-silence"):
        load_record(canonical_json(payload))

    payload.update(
        invoke=False,
        reason_codes=[ReasonCode.DELIVERY_SUCCEEDED.value],
    )
    with pytest.raises(ValidationError, match="non-invocation reason"):
        load_record(canonical_json(payload))


def test_intervention_grounding_identity_survives_serialization(
    sample_records: tuple[RuntimeRecord, ...],
) -> None:
    decision = next(item for item in sample_records if isinstance(item, InterventionDecision))

    restored = load_record(canonical_json(decision))
    assert isinstance(restored, InterventionDecision)
    assert restored.grounding_version == "fixture-grounding/1"
    assert restored.grounding_configuration == {
        "max_claims": 2,
        "renderer": "fixed",
    }
    assert restored.grounding_configuration_digest == "9" * 64
    assert restored.grounding_receipt == {"status": "fixture-verified"}
    assert canonical_json(restored) == canonical_json(decision)


def test_authoritative_ledger_excludes_proposals_and_memory_projections() -> None:
    ledger_types = set(get_args(LedgerRecord))
    assert MemoryRecord not in ledger_types
    assert MemoryDelta not in ledger_types
    assert InterventionDecision not in ledger_types


def test_canonical_digest_is_stable(trace_event: TraceEvent) -> None:
    restored = load_record(canonical_json(trace_event))
    assert canonical_digest(trace_event) == canonical_digest(restored)


def test_canonical_json_normalizes_models_nested_in_containers(
    trace_event: TraceEvent,
) -> None:
    encoded = canonical_json({"records": (trace_event,)})

    assert b'"records":[{"event_id"' in encoded
    assert trace_event.source_event_id.encode() in encoded


def test_unsupported_major_version_fails_before_record_dispatch() -> None:
    payload = {"schema_version": "2.0", "record_type": "not_a_record"}

    with pytest.raises(UnsupportedSchemaVersionError) as error:
        load_record(json.dumps(payload).encode())

    assert error.value.version == "2.0"


@pytest.mark.parametrize("version", [None, 1, "one", "1", "1.0.0"])
def test_missing_or_malformed_schema_versions_are_typed(version: object) -> None:
    payload = {"record_type": "trace_event"}
    if version is not None:
        payload["schema_version"] = version

    with pytest.raises(InvalidSchemaVersionError):
        load_record(json.dumps(payload).encode())


def test_unknown_record_type_is_typed_after_version_validation() -> None:
    with pytest.raises(UnknownRecordTypeError):
        load_record(b'{"record_type":"unknown","schema_version":"1.0"}')


def test_duplicate_json_keys_are_rejected() -> None:
    with pytest.raises(CanonicalJSONError, match="duplicate"):
        load_record(b'{"schema_version":"1.0","schema_version":"1.0"}')


@pytest.mark.parametrize("token", [b"NaN", b"Infinity", b"-Infinity"])
def test_non_finite_raw_json_is_rejected(token: bytes) -> None:
    payload = b'{"record_type":"trace_event","schema_version":"1.0","value":' + token + b"}"

    with pytest.raises(CanonicalJSONError, match="finite"):
        load_record(payload)


def test_canonical_json_rejects_non_finite_input() -> None:
    with pytest.raises(CanonicalJSONError, match="finite"):
        canonical_json({"value": float("nan")})


def test_canonical_json_rejects_non_string_keys_and_unsupported_values() -> None:
    with pytest.raises(CanonicalJSONError, match="keys must be strings"):
        canonical_json({1: "value"})
    with pytest.raises(CanonicalJSONError, match="unsupported JSON"):
        canonical_json({"value": object()})


def test_canonical_json_wraps_invalid_unicode() -> None:
    with pytest.raises(CanonicalJSONError):
        canonical_json({"value": "\ud800"})


@pytest.mark.parametrize("payload", [b"\xff", b"{", b"[]"])
def test_record_loader_wraps_invalid_or_non_object_json(payload: bytes) -> None:
    with pytest.raises(CanonicalJSONError):
        load_record(payload)
