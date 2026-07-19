from __future__ import annotations

import json
from datetime import UTC, datetime
from types import MappingProxyType
from typing import get_args

import pytest
from pydantic import BaseModel, ValidationError

from saliencegate.capture.capabilities import (
    CaptureProfile,
    capture_capability_digest,
    capture_profile,
)
from saliencegate.capture.schema import (
    CAPTURE_NATIVE_JSON_LIMITS,
    MAX_CAPTURE_EVENT_BYTES,
    MAX_CAPTURE_JSON_DEPTH,
    MAX_CAPTURE_JSON_ITEMS,
    MAX_CAPTURE_JSON_STRING_BYTES,
    MAX_CAPTURE_NATIVE_BYTES,
    CaptureActionFinishedIntake,
    CaptureActionStartedIntake,
    CaptureControllerFailedIntake,
    CaptureEvent,
    CaptureJSONLimits,
    CapturePermissionDeniedIntake,
    CaptureSchemaError,
    CaptureSessionFinishedIntake,
    CaptureSessionStartedIntake,
    CaptureSubagentFinishedIntake,
    CaptureSubagentStartedIntake,
    CaptureTurnFinishedIntake,
    canonical_capture_event,
    canonical_capture_intake,
    load_capture_event,
    load_capture_intake,
    read_bounded_json,
    validate_capture_intake,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64
DIGEST_E = "e" * 64
DIGEST_F = "f" * 64
OCCURRED_AT = datetime(2026, 7, 19, 10, 30, tzinfo=UTC)
CAPABILITY_DIGEST = capture_capability_digest(capture_profile(CaptureProfile.CODEX_HOOKS_V1))


def _common(kind: str) -> dict[str, object]:
    return {
        "schema_version": "capture-intake/v1",
        "kind": kind,
        "adapter_profile": "codex-hooks/v1",
        "capability_manifest_digest": CAPABILITY_DIGEST,
        "connection_id": "connection-1",
        "session_id": DIGEST_B,
        "producer_event_digest": DIGEST_C,
        "intake_tag": DIGEST_D,
        "occurred_at": OCCURRED_AT,
        "timestamp_authority": "local_observation",
        "producer_sequence": None,
        "sequence_authority": "unavailable",
        "capture_disposition": "captured",
    }


def _intake_payloads() -> tuple[tuple[dict[str, object], type[BaseModel]], ...]:
    return (
        (_common("session_started"), CaptureSessionStartedIntake),
        (
            {
                **_common("action_started"),
                "call_ref": DIGEST_E,
                "action_digest": DIGEST_F,
                "workspace_digest": "1" * 64,
                "environment_digest": "2" * 64,
                "tool_class": "shell",
                "identity_authority": "exact",
            },
            CaptureActionStartedIntake,
        ),
        (
            {
                **_common("action_finished"),
                "call_ref": DIGEST_E,
                "outcome_status": "succeeded",
                "outcome_authority": "producer_claimed_structured",
                "exit_status": 0,
                "error_code": None,
                "failure_signature": None,
            },
            CaptureActionFinishedIntake,
        ),
        (
            {**_common("permission_denied"), "call_ref": DIGEST_E},
            CapturePermissionDeniedIntake,
        ),
        (
            {**_common("subagent_started"), "subagent_id": DIGEST_E},
            CaptureSubagentStartedIntake,
        ),
        (
            {**_common("subagent_finished"), "subagent_id": DIGEST_E},
            CaptureSubagentFinishedIntake,
        ),
        ({**_common("turn_finished"), "turn_id": DIGEST_E}, CaptureTurnFinishedIntake),
        (
            {**_common("controller_failed"), "error_code": "provider_callback_failed"},
            CaptureControllerFailedIntake,
        ),
        (_common("session_finished"), CaptureSessionFinishedIntake),
    )


@pytest.mark.parametrize(("payload", "expected_type"), _intake_payloads())
def test_capture_intake_union_is_strict_discriminated_and_canonical(
    payload: dict[str, object],
    expected_type: type[BaseModel],
) -> None:
    intake = validate_capture_intake(payload)

    assert type(intake) is expected_type
    assert intake.schema_version == "capture-intake/v1"
    assert intake.kind == payload["kind"]
    restored = load_capture_intake(canonical_capture_intake(intake))
    assert type(restored) is expected_type
    assert canonical_capture_intake(restored) == canonical_capture_intake(intake)

    with pytest.raises(ValidationError):
        intake.kind = "session_finished"  # type: ignore[misc]
    with pytest.raises(CaptureSchemaError):
        validate_capture_intake({**payload, "provider_native_payload": "forbidden"})


def test_discriminator_never_falls_back_to_a_compatible_shape() -> None:
    action = dict(_intake_payloads()[1][0])

    for invalid_kind in ("unknown", "session_started", 1, None):
        with pytest.raises(CaptureSchemaError):
            validate_capture_intake({**action, "kind": invalid_kind})

    with pytest.raises(CaptureSchemaError):
        validate_capture_intake({**action, "schema_version": "capture-intake/v2"})
    with pytest.raises(CaptureSchemaError):
        validate_capture_intake({**action, "receipt_ordinal": 1})


def test_capture_event_wraps_one_exact_intake_and_only_admission_metadata() -> None:
    intake = validate_capture_intake(_intake_payloads()[1][0])
    event = CaptureEvent(
        receipt_ordinal=1,
        previous_event_tag=None,
        event_tag="3" * 64,
        intake=intake,
    )

    assert event.schema_version == "capture-event/v1"
    assert event.receipt_ordinal == 1
    assert event.intake is not intake
    assert type(event.intake) is CaptureActionStartedIntake
    encoded = canonical_capture_event(event)
    assert load_capture_event(encoded) == event
    assert canonical_capture_event(load_capture_event(encoded)) == encoded

    invalid = event.model_dump(mode="python")
    invalid["receipt_ordinal"] = "1"
    with pytest.raises(CaptureSchemaError):
        load_capture_event(json.dumps(invalid, default=str).encode())
    with pytest.raises(ValidationError):
        CaptureEvent.model_validate({**event.model_dump(), "chain_head": "4" * 64})


def _annotation_contains_bytes(annotation: object) -> bool:
    return annotation is bytes or any(
        _annotation_contains_bytes(item) for item in get_args(annotation)
    )


def _value_contains_bytes(value: object) -> bool:
    if isinstance(value, bytes):
        return True
    if isinstance(value, BaseModel):
        return _value_contains_bytes(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return any(
            _value_contains_bytes(key) or _value_contains_bytes(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_value_contains_bytes(item) for item in value)
    return False


def test_persistable_models_cannot_contain_raw_bytes_or_native_content_fields() -> None:
    model_types = {expected for _, expected in _intake_payloads()} | {CaptureEvent}
    forbidden_fields = {
        "argv",
        "command",
        "content",
        "cwd",
        "payload",
        "raw",
        "tool_input",
        "working_directory",
    }

    for model_type in model_types:
        assert forbidden_fields.isdisjoint(model_type.model_fields)
        assert all(
            not _annotation_contains_bytes(field.annotation)
            for field in model_type.model_fields.values()
        )

    action = validate_capture_intake(_intake_payloads()[1][0])
    event = CaptureEvent(
        receipt_ordinal=1,
        previous_event_tag=None,
        event_tag="3" * 64,
        intake=action,
    )
    assert not _value_contains_bytes(event)
    with pytest.raises(CaptureSchemaError):
        validate_capture_intake({**_intake_payloads()[1][0], "call_ref": b"native-call-id"})


def test_authority_pairs_and_digest_shapes_fail_closed() -> None:
    base = _common("session_started")
    invalid = (
        {**base, "capability_manifest_digest": "A" * 64},
        {**base, "session_id": "a" * 63},
        {**base, "producer_sequence": 1, "sequence_authority": "unavailable"},
        {**base, "producer_sequence": None, "sequence_authority": "producer_exact"},
        {**base, "occurred_at": None, "timestamp_authority": "local_observation"},
        {**base, "occurred_at": OCCURRED_AT, "timestamp_authority": "unavailable"},
        {**base, "capability_manifest_digest": DIGEST_A},
        {**base, "adapter_profile": "unlisted-provider/v1"},
        {**base, "capture_disposition": "provider_claimed_complete"},
    )

    for payload in invalid:
        with pytest.raises(CaptureSchemaError):
            validate_capture_intake(payload)

    for disposition in ("captured", "coverage_boundary", "degraded"):
        assert (
            validate_capture_intake(
                {**base, "capture_disposition": disposition}
            ).capture_disposition
            == disposition
        )


def test_action_and_outcome_authority_are_closed_structured_contracts() -> None:
    action = _intake_payloads()[1][0]
    finished = _intake_payloads()[2][0]

    for authority in ("exact", "coarse", "unavailable"):
        assert validate_capture_intake({**action, "identity_authority": authority})
    with pytest.raises(CaptureSchemaError):
        validate_capture_intake({**action, "identity_authority": "inferred_from_text"})
    with pytest.raises(CaptureSchemaError):
        validate_capture_intake({**action, "tool_class": "arbitrary-provider-tool"})
    with pytest.raises(CaptureSchemaError):
        validate_capture_intake({**finished, "outcome_status": "task_completed"})
    with pytest.raises(CaptureSchemaError):
        validate_capture_intake(
            {
                **finished,
                "outcome_status": None,
                "outcome_authority": "producer_claimed_structured",
            }
        )


def _limits(
    *,
    max_bytes: int = 4_096,
    max_depth: int = 8,
    max_items: int = 64,
    max_string_bytes: int = 1_024,
) -> CaptureJSONLimits:
    return CaptureJSONLimits(
        max_bytes=max_bytes,
        max_depth=max_depth,
        max_items=max_items,
        max_string_bytes=max_string_bytes,
    )


def test_capture_limits_are_the_locked_v1_bounds() -> None:
    assert MAX_CAPTURE_EVENT_BYTES == 64 * 1_024
    assert MAX_CAPTURE_NATIVE_BYTES == 2 * 1_024 * 1_024
    assert CAPTURE_NATIVE_JSON_LIMITS.max_bytes == MAX_CAPTURE_NATIVE_BYTES
    assert MAX_CAPTURE_JSON_DEPTH == 32
    assert MAX_CAPTURE_JSON_ITEMS == 10_000
    assert MAX_CAPTURE_JSON_STRING_BYTES == 1 * 1_024 * 1_024


@pytest.mark.parametrize(
    "document",
    (
        b'{"key":1,"key":2}',
        b'{"key":1,"\\u006bey":2}',
        b'{"value":"\xff"}',
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b"{} trailing",
    ),
)
def test_bounded_reader_rejects_duplicate_keys_invalid_utf8_and_non_json(
    document: bytes,
) -> None:
    with pytest.raises(CaptureSchemaError) as raised:
        read_bounded_json(document, limits=_limits())

    assert str(raised.value) == "capture schema is invalid"
    assert not raised.value.args or raised.value.args == ("capture schema is invalid",)


def test_bounded_reader_enforces_bytes_depth_items_and_transitive_string_budget() -> None:
    cases = (
        (b'{"value":"0123456789"}', _limits(max_bytes=16)),
        (b'{"a":{"b":{"c":1}}}', _limits(max_depth=2)),
        (b'{"items":[1,2,3,4]}', _limits(max_items=4)),
        (b'{"a":"12345678","b":"abcdefgh"}', _limits(max_string_bytes=15)),
    )

    for document, limits in cases:
        with pytest.raises(CaptureSchemaError):
            read_bounded_json(document, limits=limits)


def test_bounded_reader_is_immutable_and_persisted_codec_requires_canonical_bytes() -> None:
    value = read_bounded_json(b'{ "z": [true, null], "a": {"n": 1} }', limits=_limits())

    assert isinstance(value, MappingProxyType)
    assert value == {"z": (True, None), "a": {"n": 1}}
    with pytest.raises(TypeError):
        value["z"] = ()  # type: ignore[index]

    intake = validate_capture_intake(_common("session_started"))
    canonical = canonical_capture_intake(intake)
    assert b"\n" not in canonical
    assert canonical == canonical_capture_intake(load_capture_intake(canonical))

    noncanonical = json.dumps(json.loads(canonical), indent=2).encode()
    with pytest.raises(CaptureSchemaError):
        load_capture_intake(noncanonical)


def test_public_schema_errors_and_representations_are_content_free() -> None:
    marker = "provider-native-secret-marker"
    with pytest.raises(CaptureSchemaError) as raised:
        load_capture_intake(
            json.dumps(
                {**_common("session_started"), "session_id": marker},
                default=str,
            ).encode()
        )

    assert marker not in str(raised.value)
    assert marker not in repr(raised.value)
    assert raised.value.__cause__ is None

    intake = validate_capture_intake(_intake_payloads()[1][0])
    representation = repr(intake)
    for secret in (DIGEST_B, DIGEST_C, DIGEST_D, DIGEST_E, DIGEST_F):
        assert secret not in representation
    assert "<redacted>" in representation

    with pytest.raises(TypeError):
        CaptureSchemaError(marker)  # type: ignore[call-arg]
