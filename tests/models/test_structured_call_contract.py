from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from uuid import UUID

import pytest
from pydantic import ValidationError

import saliencegate.ports as public_ports
import saliencegate.ports.model_calls as call_ports
from saliencegate.domain import (
    EvidenceReference,
    EvidenceSource,
    InterventionAction,
    PayloadDigest,
    PayloadDigestAlgorithm,
    canonical_json,
)
from saliencegate.memory import (
    INTERVENTION_OUTPUT_SCHEMA_VERSION,
    MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
    BankOperationsProposal,
    InterventionSelectionOutput,
    SaveKnowledge,
)
from saliencegate.ports.model_calls import (
    MAX_STRUCTURED_CALL_OUTPUT_BYTES,
    CanonicalUsageProvenance,
    ProviderUsageProvenance,
    StructuredCallBoundaryError,
    StructuredCallClient,
    StructuredCallParseStatus,
    StructuredCallPhase,
    StructuredCallRequest,
    StructuredCallResult,
    StructuredCallStatus,
    StructuredCallUsage,
    validated_result_for_request,
    validated_structured_call_request,
    validated_structured_call_result,
)
from saliencegate.ports.models import ModelRequest

RUN_ID = UUID("00000000-0000-4000-8000-00000000d001")
OTHER_RUN_ID = UUID("00000000-0000-4000-8000-00000000d002")
EVENT_ID = UUID("00000000-0000-4000-8000-00000000d003")
REQUEST_SCHEMA_VERSION = "structured-call-request/v1"
RESULT_SCHEMA_VERSION = "structured-call-result/v1"
USAGE_SCHEMA_VERSION = "structured-call-usage/v1"


def _bank_output() -> BankOperationsProposal:
    return BankOperationsProposal(
        schema_version=MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
        operations=(),
    )


def _bank_output_with_content(content: str) -> BankOperationsProposal:
    return BankOperationsProposal(
        schema_version=MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
        operations=(
            SaveKnowledge(
                operation="save_knowledge",
                content=content,
                evidence=(
                    EvidenceReference(
                        source=EvidenceSource.EVENT,
                        source_id=EVENT_ID,
                        field_path="/payload/message",
                    ),
                ),
                confidence=1.0,
            ),
        ),
    )


def _intervention_output() -> InterventionSelectionOutput:
    return InterventionSelectionOutput(
        schema_version=INTERVENTION_OUTPUT_SCHEMA_VERSION,
        action=InterventionAction.SILENCE,
        claims=(),
        confidence=1.0,
    )


def _completion_digest(seed: str = "c") -> PayloadDigest:
    return PayloadDigest(
        algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
        value=seed * 64,
    )


def _request(
    *,
    phase: StructuredCallPhase = StructuredCallPhase.MEMORY_EDIT,
    run_id: UUID = RUN_ID,
    cycle_id: str = "a" * 64,
    model_call_index: int | None = None,
    attempt: int = 0,
    model_id: str = "gpt-oss:20b",
    prompt_template_id: str | None = None,
    prompt_template_digest: str = "b" * 64,
    payload: dict[str, object] | None = None,
) -> StructuredCallRequest:
    if model_call_index is None:
        model_call_index = 0 if phase is StructuredCallPhase.MEMORY_EDIT else 1
    response_schema_version = (
        MEMORY_EDIT_OUTPUT_SCHEMA_VERSION
        if phase is StructuredCallPhase.MEMORY_EDIT
        else INTERVENTION_OUTPUT_SCHEMA_VERSION
    )
    if prompt_template_id is None:
        prompt_template_id = (
            "paper-two-phase/memory-edit-v1"
            if phase is StructuredCallPhase.MEMORY_EDIT
            else "paper-two-phase/intervention-v1"
        )
    return StructuredCallRequest(
        schema_version=REQUEST_SCHEMA_VERSION,
        run_id=run_id,
        cycle_id=cycle_id,
        model_call_index=model_call_index,
        phase=phase,
        attempt=attempt,
        model_id=model_id,
        prompt_template_id=prompt_template_id,
        prompt_template_digest=prompt_template_digest,
        response_schema_version=response_schema_version,
        payload={"window": {"first": 1, "last": 2}} if payload is None else payload,
    )


def _usage(
    *,
    provenance: ProviderUsageProvenance = ProviderUsageProvenance.PROVIDER_REPORTED,
    input_tokens: int | None = 10,
    output_tokens: int | None = 3,
    latency_us: int = 50,
) -> StructuredCallUsage:
    return StructuredCallUsage(
        schema_version=USAGE_SCHEMA_VERSION,
        provider_input_tokens=input_tokens,
        provider_output_tokens=output_tokens,
        provider_usage_provenance=provenance,
        latency_us=latency_us,
    )


def _result_values(request: StructuredCallRequest) -> dict[str, object]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "request_digest": request.request_digest,
        "model_call_index": request.model_call_index,
        "phase": request.phase,
        "attempt": request.attempt,
        "response_schema_version": request.response_schema_version,
        "usage": _usage(),
    }


def _valid_result(request: StructuredCallRequest) -> StructuredCallResult:
    output = (
        _bank_output()
        if request.phase is StructuredCallPhase.MEMORY_EDIT
        else _intervention_output()
    )
    return StructuredCallResult(
        **_result_values(request),
        status=StructuredCallStatus.COMPLETED,
        parse_status=StructuredCallParseStatus.VALID,
        output=output,
        completion_digest=_completion_digest(),
        completion_byte_count=128,
    )


def _schema_invalid_result(request: StructuredCallRequest) -> StructuredCallResult:
    return StructuredCallResult(
        **_result_values(request),
        status=StructuredCallStatus.COMPLETED,
        parse_status=StructuredCallParseStatus.SCHEMA_INVALID,
        output=None,
        completion_digest=_completion_digest("d"),
        completion_byte_count=0,
    )


def test_request_digest_is_canonical_version_separated_and_sensitive() -> None:
    first = _request(payload={"z": 1, "a": {"y": 2, "x": 3}})
    reordered = _request(payload={"a": {"x": 3, "y": 2}, "z": 1})

    assert first.request_digest == reordered.request_digest
    assert StructuredCallRequest.model_validate_json(first.model_dump_json()) == first
    assert first.request_digest != "" and len(first.request_digest) == 64
    assert _request().request_digest == (
        "9148bca06436737aefe8004c645be44dbe9f8c45b9afb53ed035d46f9a57f59a"
    )
    legacy = ModelRequest(
        run_id=RUN_ID,
        cycle_id="a" * 64,
        model_call_index=0,
        model_id="gpt-oss:20b",
        prompt_template_digest="b" * 64,
        payload={"window": {"first": 1, "last": 2}},
    )
    assert legacy.request_digest != _request().request_digest

    changes = (
        _request(run_id=OTHER_RUN_ID),
        _request(cycle_id="c" * 64),
        _request(model_call_index=1),
        _request(phase=StructuredCallPhase.INTERVENTION),
        _request(model_call_index=1, attempt=1),
        _request(model_id="gpt-oss:120b"),
        _request(prompt_template_id="paper-two-phase/memory-edit-v2"),
        _request(prompt_template_digest="e" * 64),
        _request(payload={"window": {"first": 1, "last": 3}}),
    )
    for changed in changes:
        assert changed.request_digest != _request().request_digest

    tampered = first.model_dump(mode="python")
    tampered["request_digest"] = "f" * 64
    with pytest.raises(ValidationError, match="request digest does not match"):
        StructuredCallRequest.model_validate(tampered)


@pytest.mark.parametrize(
    ("phase", "response_schema_version"),
    (
        (StructuredCallPhase.MEMORY_EDIT, INTERVENTION_OUTPUT_SCHEMA_VERSION),
        (StructuredCallPhase.INTERVENTION, MEMORY_EDIT_OUTPUT_SCHEMA_VERSION),
    ),
)
def test_request_phase_requires_its_exact_response_schema(
    phase: StructuredCallPhase,
    response_schema_version: str,
) -> None:
    values = _request(phase=phase).model_dump(
        mode="python",
        exclude={"request_digest", "response_schema_version"},
    )
    with pytest.raises(ValidationError, match="phase response schema does not match"):
        StructuredCallRequest(
            **values,
            response_schema_version=response_schema_version,
        )


def test_request_rejects_impossible_phase_attempt_identities() -> None:
    with pytest.raises(ValidationError, match="attempt cannot exceed call index"):
        _request(model_call_index=0, attempt=1)
    with pytest.raises(ValidationError, match="intervention call index must be positive"):
        _request(
            phase=StructuredCallPhase.INTERVENTION,
            model_call_index=0,
        )


def test_request_is_strict_frozen_bounded_and_secret_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "credential-like-prompt-value"
    request = _request(payload={"secret": secret})

    assert secret not in repr(request)
    with pytest.raises(ValidationError):
        request.model_call_index = 1  # type: ignore[misc]
    with pytest.raises(ValidationError):
        StructuredCallRequest.model_validate(
            {**request.model_dump(mode="json"), "provider_body": secret}
        )

    for field, value in (
        ("model_call_index", True),
        ("model_call_index", -1),
        ("model_call_index", 1 << 63),
        ("attempt", True),
        ("attempt", -1),
        ("attempt", 1 << 63),
        ("run_id", str(RUN_ID)),
        ("cycle_id", "A" * 64),
        ("model_id", "contains spaces"),
    ):
        values = request.model_dump(mode="python", exclude={"request_digest"})
        values[field] = value
        with pytest.raises(ValidationError):
            StructuredCallRequest.model_validate(values)

    multibyte_payload = {"value": "é"}
    exact_bytes = len(canonical_json(multibyte_payload))
    monkeypatch.setattr(call_ports, "MAX_STRUCTURED_CALL_PAYLOAD_BYTES", exact_bytes)
    assert _request(payload=multibyte_payload).payload["value"] == "é"
    monkeypatch.setattr(call_ports, "MAX_STRUCTURED_CALL_PAYLOAD_BYTES", exact_bytes - 1)
    with pytest.raises(ValidationError, match="bound"):
        _request(payload=multibyte_payload)


def test_request_structural_bound_precedes_recursive_freezing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(call_ports, "MAX_STRUCTURED_CALL_PAYLOAD_DEPTH", 4)
    deep: dict[str, object] = {}
    cursor = deep
    for _ in range(5):
        nested: dict[str, object] = {}
        cursor["next"] = nested
        cursor = nested
    with pytest.raises(ValidationError, match="structural bound"):
        _request(payload=deep)

    monkeypatch.setattr(call_ports, "MAX_STRUCTURED_CALL_PAYLOAD_NODES", 4)
    recursive: dict[str, object] = {}
    recursive["self"] = recursive
    with pytest.raises(ValidationError, match="structural bound"):
        _request(payload=recursive)

    invalid_payloads = (
        ["not", "an", "object"],
        {"unsupported": object()},
        {"not_finite": float("nan")},
        {1: "non-string-key"},
        {"\ud800": "invalid-utf8-key"},
    )
    for payload in invalid_payloads:
        with pytest.raises(ValidationError, match="structural bound"):
            _request(payload=payload)  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="structural bound"):
        _request(payload={"values": [1, 2, 3, 4]})


def test_usage_preserves_provider_provenance_without_inventing_zero() -> None:
    reported_zero = _usage(input_tokens=0, output_tokens=0)
    unavailable = _usage(
        provenance=ProviderUsageProvenance.UNAVAILABLE,
        input_tokens=None,
        output_tokens=None,
    )
    replay = _usage(provenance=ProviderUsageProvenance.REPLAY_ATTESTED)

    assert reported_zero.provider_tokens == 0
    assert unavailable.provider_tokens is None
    assert replay.provider_tokens == 13
    assert reported_zero.model_dump(mode="json") != unavailable.model_dump(mode="json")

    invalid_values = (
        {
            "provider_input_tokens": 1,
            "provider_output_tokens": None,
            "provider_usage_provenance": ProviderUsageProvenance.PROVIDER_REPORTED,
        },
        {
            "provider_input_tokens": None,
            "provider_output_tokens": None,
            "provider_usage_provenance": ProviderUsageProvenance.PROVIDER_REPORTED,
        },
        {
            "provider_input_tokens": 1,
            "provider_output_tokens": 1,
            "provider_usage_provenance": ProviderUsageProvenance.UNAVAILABLE,
        },
        {
            "provider_input_tokens": True,
            "provider_output_tokens": 1,
            "provider_usage_provenance": ProviderUsageProvenance.PROVIDER_REPORTED,
        },
        {
            "provider_input_tokens": -1,
            "provider_output_tokens": 1,
            "provider_usage_provenance": ProviderUsageProvenance.PROVIDER_REPORTED,
        },
        {
            "provider_input_tokens": 1 << 63,
            "provider_output_tokens": 1,
            "provider_usage_provenance": ProviderUsageProvenance.PROVIDER_REPORTED,
        },
        {
            "provider_input_tokens": None,
            "provider_output_tokens": None,
            "provider_usage_provenance": ProviderUsageProvenance.REPLAY_ATTESTED,
        },
        {
            "provider_input_tokens": (1 << 63) - 1,
            "provider_output_tokens": 1,
            "provider_usage_provenance": ProviderUsageProvenance.PROVIDER_REPORTED,
        },
    )
    for values in invalid_values:
        with pytest.raises(ValidationError):
            StructuredCallUsage(
                schema_version=USAGE_SCHEMA_VERSION,
                latency_us=1,
                **values,
            )

    for latency_us in (True, -1, 1 << 63):
        with pytest.raises(ValidationError):
            StructuredCallUsage(
                schema_version=USAGE_SCHEMA_VERSION,
                provider_input_tokens=1,
                provider_output_tokens=1,
                provider_usage_provenance=ProviderUsageProvenance.PROVIDER_REPORTED,
                latency_us=latency_us,
            )


def test_usage_keeps_local_canonical_counts_distinct_and_model_bound() -> None:
    identity = {
        "canonical_usage_provenance": CanonicalUsageProvenance.LOCAL_COUNTER,
        "local_counter_id": "deterministic-model-token-counter",
        "local_counter_version": "scripted-counts-v1",
        "local_counter_configuration_digest": "d" * 64,
        "local_counter_model_id": "gpt-oss:20b",
    }
    counted = StructuredCallUsage(
        schema_version=USAGE_SCHEMA_VERSION,
        provider_input_tokens=10,
        provider_output_tokens=3,
        provider_usage_provenance=ProviderUsageProvenance.PROVIDER_REPORTED,
        latency_us=50,
        canonical_input_tokens=11,
        canonical_output_tokens=5,
        **identity,
    )
    partial = StructuredCallUsage(
        schema_version=USAGE_SCHEMA_VERSION,
        provider_input_tokens=None,
        provider_output_tokens=None,
        provider_usage_provenance=ProviderUsageProvenance.UNAVAILABLE,
        latency_us=50,
        canonical_input_tokens=11,
        canonical_output_tokens=None,
        **identity,
    )

    assert counted.provider_tokens == 13
    assert counted.canonical_tokens == 16
    assert counted.provider_tokens != counted.canonical_tokens
    assert partial.canonical_tokens is None
    assert partial.canonical_input_tokens == 11
    assert set(_usage().model_dump(mode="json")) == {
        "schema_version",
        "provider_input_tokens",
        "provider_output_tokens",
        "provider_usage_provenance",
        "latency_us",
    }

    invalid = (
        {"canonical_input_tokens": 1},
        {
            **identity,
            "canonical_input_tokens": None,
            "canonical_output_tokens": None,
        },
        {
            **identity,
            "local_counter_version": None,
            "canonical_input_tokens": 1,
        },
        {
            **identity,
            "canonical_input_tokens": (1 << 63) - 1,
            "canonical_output_tokens": 1,
        },
    )
    for local_values in invalid:
        with pytest.raises(ValidationError):
            StructuredCallUsage(
                schema_version=USAGE_SCHEMA_VERSION,
                provider_input_tokens=None,
                provider_output_tokens=None,
                provider_usage_provenance=ProviderUsageProvenance.UNAVAILABLE,
                latency_us=1,
                **local_values,
            )

    request = _request()
    mismatched = StructuredCallResult(
        **{
            **_result_values(request),
            "usage": counted.model_copy(update={"local_counter_model_id": "gpt-oss:120b"}),
        },
        status=StructuredCallStatus.COMPLETED,
        parse_status=StructuredCallParseStatus.VALID,
        output=_bank_output(),
        completion_digest=_completion_digest(),
        completion_byte_count=128,
    )
    with pytest.raises(StructuredCallBoundaryError):
        validated_result_for_request(request, mismatched)


@pytest.mark.parametrize(
    ("status", "parse_status", "has_output", "has_attestation"),
    (
        (StructuredCallStatus.COMPLETED, StructuredCallParseStatus.VALID, True, True),
        (
            StructuredCallStatus.COMPLETED,
            StructuredCallParseStatus.SCHEMA_INVALID,
            False,
            True,
        ),
        (StructuredCallStatus.MODEL_ERROR, StructuredCallParseStatus.NOT_ATTEMPTED, False, False),
        (
            StructuredCallStatus.MODEL_TIMEOUT,
            StructuredCallParseStatus.NOT_ATTEMPTED,
            False,
            False,
        ),
    ),
)
def test_result_state_matrix_accepts_only_four_shapes(
    status: StructuredCallStatus,
    parse_status: StructuredCallParseStatus,
    has_output: bool,
    has_attestation: bool,
) -> None:
    request = _request()
    usage = (
        _usage()
        if status is StructuredCallStatus.COMPLETED
        else _usage(
            provenance=ProviderUsageProvenance.UNAVAILABLE,
            input_tokens=None,
            output_tokens=None,
        )
    )
    result = StructuredCallResult(
        **{**_result_values(request), "usage": usage},
        status=status,
        parse_status=parse_status,
        output=_bank_output() if has_output else None,
        completion_digest=_completion_digest() if has_attestation else None,
        completion_byte_count=1 if has_attestation else None,
    )

    assert (result.output is not None) is has_output
    assert (result.completion_digest is not None) is has_attestation
    assert result.usage.provider_tokens == (13 if has_attestation else None)


@pytest.mark.parametrize(
    "parse_status",
    (
        StructuredCallParseStatus.EMPTY_REMINDER,
        StructuredCallParseStatus.CLAIM_OVER_LIMIT,
    ),
)
def test_intervention_parse_rejection_reason_survives_raw_completion_discard(
    parse_status: StructuredCallParseStatus,
) -> None:
    request = _request(phase=StructuredCallPhase.INTERVENTION)
    result = StructuredCallResult(
        **_result_values(request),
        status=StructuredCallStatus.COMPLETED,
        parse_status=parse_status,
        output=None,
        completion_digest=_completion_digest(),
        completion_byte_count=1,
    )
    assert result.parse_status is parse_status

    with pytest.raises(ValidationError, match="parse status does not match phase"):
        StructuredCallResult(
            **_result_values(_request()),
            status=StructuredCallStatus.COMPLETED,
            parse_status=parse_status,
            output=None,
            completion_digest=_completion_digest(),
            completion_byte_count=1,
        )


@pytest.mark.parametrize(
    ("status", "parse_status", "output", "completion_digest", "completion_byte_count"),
    (
        (
            StructuredCallStatus.COMPLETED,
            StructuredCallParseStatus.VALID,
            None,
            _completion_digest(),
            1,
        ),
        (
            StructuredCallStatus.COMPLETED,
            StructuredCallParseStatus.SCHEMA_INVALID,
            _bank_output(),
            _completion_digest(),
            1,
        ),
        (
            StructuredCallStatus.COMPLETED,
            StructuredCallParseStatus.NOT_ATTEMPTED,
            None,
            None,
            None,
        ),
        (
            StructuredCallStatus.MODEL_ERROR,
            StructuredCallParseStatus.VALID,
            _bank_output(),
            _completion_digest(),
            1,
        ),
        (
            StructuredCallStatus.MODEL_TIMEOUT,
            StructuredCallParseStatus.SCHEMA_INVALID,
            None,
            _completion_digest(),
            1,
        ),
        (
            StructuredCallStatus.MODEL_ERROR,
            StructuredCallParseStatus.NOT_ATTEMPTED,
            None,
            _completion_digest(),
            1,
        ),
        (
            StructuredCallStatus.COMPLETED,
            StructuredCallParseStatus.SCHEMA_INVALID,
            None,
            None,
            None,
        ),
        (
            StructuredCallStatus.COMPLETED,
            StructuredCallParseStatus.VALID,
            _bank_output(),
            _completion_digest(),
            None,
        ),
        (
            StructuredCallStatus.COMPLETED,
            StructuredCallParseStatus.VALID,
            _bank_output(),
            _completion_digest(),
            0,
        ),
    ),
)
def test_result_state_matrix_rejects_every_inverse_shape(
    status: StructuredCallStatus,
    parse_status: StructuredCallParseStatus,
    output: object,
    completion_digest: PayloadDigest | None,
    completion_byte_count: int | None,
) -> None:
    with pytest.raises(ValidationError, match="result state is inconsistent"):
        StructuredCallResult(
            **_result_values(_request()),
            status=status,
            parse_status=parse_status,
            output=output,
            completion_digest=completion_digest,
            completion_byte_count=completion_byte_count,
        )


def test_result_rejects_cross_phase_output_and_phase_schema_mismatch() -> None:
    memory_request = _request()
    intervention_request = _request(phase=StructuredCallPhase.INTERVENTION)

    with pytest.raises(ValidationError, match="output does not match phase"):
        StructuredCallResult(
            **_result_values(memory_request),
            status=StructuredCallStatus.COMPLETED,
            parse_status=StructuredCallParseStatus.VALID,
            output=_intervention_output(),
            completion_digest=_completion_digest(),
            completion_byte_count=1,
        )
    with pytest.raises(ValidationError, match="output does not match phase"):
        StructuredCallResult(
            **_result_values(intervention_request),
            status=StructuredCallStatus.COMPLETED,
            parse_status=StructuredCallParseStatus.VALID,
            output=_bank_output(),
            completion_digest=_completion_digest(),
            completion_byte_count=1,
        )

    values = _result_values(memory_request)
    values["response_schema_version"] = INTERVENTION_OUTPUT_SCHEMA_VERSION
    with pytest.raises(ValidationError, match="phase response schema does not match"):
        StructuredCallResult(
            **values,
            status=StructuredCallStatus.COMPLETED,
            parse_status=StructuredCallParseStatus.SCHEMA_INVALID,
            output=None,
            completion_digest=_completion_digest(),
            completion_byte_count=1,
        )


def test_result_output_discriminator_error_never_echoes_provider_tag() -> None:
    encoded = _valid_result(_request()).model_dump(mode="json")
    secret_tag = "unknown-provider-secret-tag"
    assert isinstance(encoded["output"], dict)
    encoded["output"]["schema_version"] = secret_tag

    with pytest.raises(ValidationError) as captured:
        StructuredCallResult.model_validate_json(json.dumps(encoded))
    assert secret_tag not in str(captured.value)
    assert secret_tag not in repr(captured.value)

    encoded = _valid_result(_request()).model_dump(mode="json")
    encoded["output"] = "not-a-json-object"
    with pytest.raises(ValidationError, match="output failed validation"):
        StructuredCallResult.model_validate_json(json.dumps(encoded))

    encoded = _valid_result(_request()).model_dump(mode="json")
    assert isinstance(encoded["output"], dict)
    encoded["output"].pop("operations")
    encoded["output"]["provider_secret"] = secret_tag
    with pytest.raises(ValidationError) as captured:
        StructuredCallResult.model_validate_json(json.dumps(encoded))
    assert secret_tag not in str(captured.value)


def test_result_rejects_impossible_attempt_identity() -> None:
    values = _result_values(_request())
    values["attempt"] = 1
    with pytest.raises(ValidationError, match="attempt cannot exceed call index"):
        StructuredCallResult(
            **values,
            status=StructuredCallStatus.COMPLETED,
            parse_status=StructuredCallParseStatus.SCHEMA_INVALID,
            output=None,
            completion_digest=_completion_digest(),
            completion_byte_count=1,
        )

    intervention_values = _result_values(_request(phase=StructuredCallPhase.INTERVENTION))
    intervention_values["model_call_index"] = 0
    with pytest.raises(ValidationError, match="intervention call index must be positive"):
        StructuredCallResult(
            **intervention_values,
            status=StructuredCallStatus.COMPLETED,
            parse_status=StructuredCallParseStatus.SCHEMA_INVALID,
            output=None,
            completion_digest=_completion_digest(),
            completion_byte_count=1,
        )


def test_schema_invalid_completion_is_bounded_attested_and_raw_free() -> None:
    request = _request()
    empty = _schema_invalid_result(request)
    different = StructuredCallResult(
        **_result_values(request),
        status=StructuredCallStatus.COMPLETED,
        parse_status=StructuredCallParseStatus.SCHEMA_INVALID,
        output=None,
        completion_digest=_completion_digest("e"),
        completion_byte_count=0,
    )

    assert empty.call_digest != different.call_digest
    assert empty.completion_byte_count == 0
    assert "completion=" not in repr(empty)
    assert empty.completion_digest == _completion_digest("d")
    assert empty.model_dump(mode="json")["completion_digest"] == {
        "algorithm": "hmac_sha256",
        "value": "d" * 64,
    }
    assert set(empty.model_dump(mode="json")) == {
        "schema_version",
        "request_digest",
        "model_call_index",
        "phase",
        "attempt",
        "response_schema_version",
        "status",
        "parse_status",
        "output",
        "completion_digest",
        "completion_byte_count",
        "usage",
        "call_digest",
    }

    for forbidden in ("raw_completion", "provider_body", "reasoning", "error_text"):
        with pytest.raises(ValidationError):
            StructuredCallResult.model_validate(
                {**empty.model_dump(mode="python"), forbidden: "provider-secret"}
            )

    with pytest.raises(ValidationError):
        StructuredCallResult(
            **_result_values(request),
            status=StructuredCallStatus.COMPLETED,
            parse_status=StructuredCallParseStatus.SCHEMA_INVALID,
            output=None,
            completion_digest=_completion_digest(),
            completion_byte_count=MAX_STRUCTURED_CALL_OUTPUT_BYTES + 1,
        )

    with pytest.raises(ValidationError):
        StructuredCallResult(
            **_result_values(request),
            status=StructuredCallStatus.COMPLETED,
            parse_status=StructuredCallParseStatus.SCHEMA_INVALID,
            output=None,
            completion_digest="c" * 64,
            completion_byte_count=1,
        )

    forged_digest = PayloadDigest.model_construct(
        algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
        value="not-a-digest",
    )
    with pytest.raises(ValidationError, match="completion digest failed validation"):
        StructuredCallResult(
            **_result_values(request),
            status=StructuredCallStatus.COMPLETED,
            parse_status=StructuredCallParseStatus.SCHEMA_INVALID,
            output=None,
            completion_digest=forged_digest,
            completion_byte_count=1,
        )

    synthetic_digest = PayloadDigest(
        algorithm=PayloadDigestAlgorithm.SYNTHETIC_SHA256,
        value="a" * 64,
    )
    synthetic = StructuredCallResult(
        **_result_values(request),
        status=StructuredCallStatus.COMPLETED,
        parse_status=StructuredCallParseStatus.SCHEMA_INVALID,
        output=None,
        completion_digest=synthetic_digest,
        completion_byte_count=1,
    )
    assert synthetic.completion_digest == synthetic_digest


def test_result_digest_binds_every_persisted_fact_and_is_verified() -> None:
    request = _request()
    base = _valid_result(request)
    assert base.call_digest == ("f3530945dedebd936fcd9ac5d20586806f23f98d0c981a8a0cf75fd539d1690d")
    indexed = _valid_result(_request(model_call_index=1))
    repaired = _valid_result(_request(model_call_index=1, attempt=1))
    assert indexed.call_digest != repaired.call_digest
    variants = (
        _valid_result(_request(cycle_id="e" * 64)),
        indexed,
        repaired,
        _valid_result(_request(phase=StructuredCallPhase.INTERVENTION)),
        StructuredCallResult(
            **_result_values(request),
            status=StructuredCallStatus.COMPLETED,
            parse_status=StructuredCallParseStatus.VALID,
            output=_bank_output(),
            completion_digest=_completion_digest("d"),
            completion_byte_count=128,
        ),
        StructuredCallResult(
            **_result_values(request),
            status=StructuredCallStatus.COMPLETED,
            parse_status=StructuredCallParseStatus.VALID,
            output=_bank_output_with_content("different valid memory"),
            completion_digest=_completion_digest(),
            completion_byte_count=128,
        ),
        StructuredCallResult(
            **_result_values(request),
            status=StructuredCallStatus.COMPLETED,
            parse_status=StructuredCallParseStatus.VALID,
            output=_bank_output(),
            completion_digest=_completion_digest(),
            completion_byte_count=129,
        ),
        StructuredCallResult(
            **{**_result_values(request), "usage": _usage(input_tokens=11)},
            status=StructuredCallStatus.COMPLETED,
            parse_status=StructuredCallParseStatus.VALID,
            output=_bank_output(),
            completion_digest=_completion_digest(),
            completion_byte_count=128,
        ),
        StructuredCallResult(
            **{**_result_values(request), "usage": _usage(latency_us=51)},
            status=StructuredCallStatus.COMPLETED,
            parse_status=StructuredCallParseStatus.VALID,
            output=_bank_output(),
            completion_digest=_completion_digest(),
            completion_byte_count=128,
        ),
        StructuredCallResult(
            **{
                **_result_values(request),
                "usage": _usage(provenance=ProviderUsageProvenance.REPLAY_ATTESTED),
            },
            status=StructuredCallStatus.COMPLETED,
            parse_status=StructuredCallParseStatus.VALID,
            output=_bank_output(),
            completion_digest=_completion_digest(),
            completion_byte_count=128,
        ),
        _schema_invalid_result(request),
    )
    for variant in variants:
        assert variant.call_digest != base.call_digest

    assert StructuredCallResult.model_validate_json(base.model_dump_json()) == base
    tampered = base.model_dump(mode="python")
    tampered["call_digest"] = "f" * 64
    with pytest.raises(ValidationError, match="call digest does not match"):
        StructuredCallResult.model_validate(tampered)


def test_valid_output_has_a_canonical_byte_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    output = _bank_output_with_content("é")
    exact_bytes = len(canonical_json(output))
    monkeypatch.setattr(call_ports, "MAX_STRUCTURED_CALL_OUTPUT_BYTES", exact_bytes)
    request = _request()
    accepted = StructuredCallResult(
        **_result_values(request),
        status=StructuredCallStatus.COMPLETED,
        parse_status=StructuredCallParseStatus.VALID,
        output=output,
        completion_digest=_completion_digest(),
        completion_byte_count=exact_bytes,
    )
    assert accepted.output == output

    monkeypatch.setattr(call_ports, "MAX_STRUCTURED_CALL_OUTPUT_BYTES", exact_bytes - 1)
    with pytest.raises(ValidationError, match="output exceeds its canonical byte limit"):
        StructuredCallResult(
            **_result_values(request),
            status=StructuredCallStatus.COMPLETED,
            parse_status=StructuredCallParseStatus.VALID,
            output=output,
            completion_digest=_completion_digest(),
            completion_byte_count=exact_bytes - 1,
        )


def test_boundaries_require_exact_types_revalidate_and_sanitize() -> None:
    secret = "provider-secret-must-not-leak"
    request = _request(payload={"secret": secret})
    result = _valid_result(request)

    copied_request = validated_structured_call_request(request)
    copied_result = validated_structured_call_result(result)
    assert copied_request == request and copied_request is not request
    assert copied_result == result and copied_result is not result

    class RequestSubclass(StructuredCallRequest):
        pass

    class ResultSubclass(StructuredCallResult):
        pass

    invalid_requests = (
        request.model_dump(mode="python"),
        RequestSubclass.model_validate_json(request.model_dump_json()),
        request.model_copy(update={"request_digest": "f" * 64}),
    )
    for invalid in invalid_requests:
        with pytest.raises(StructuredCallBoundaryError) as captured:
            validated_structured_call_request(invalid)
        assert str(captured.value) == (
            "structured call request failed structured-call boundary validation"
        )
        assert secret not in str(captured.value)
        assert captured.value.__cause__ is None

    forged_usage = result.usage.model_copy(update={"provider_input_tokens": -1})
    invalid_results = (
        result.model_dump(mode="python"),
        ResultSubclass.model_validate_json(result.model_dump_json()),
        result.model_copy(update={"usage": forged_usage}),
    )
    for invalid in invalid_results:
        with pytest.raises(StructuredCallBoundaryError) as captured:
            validated_structured_call_result(invalid)
        assert str(captured.value) == (
            "structured call result failed structured-call boundary validation"
        )
        assert secret not in str(captured.value)
        assert captured.value.__cause__ is None

    secret_result = StructuredCallResult(
        **_result_values(request),
        status=StructuredCallStatus.COMPLETED,
        parse_status=StructuredCallParseStatus.VALID,
        output=_bank_output_with_content(secret),
        completion_digest=_completion_digest(),
        completion_byte_count=128,
    )
    assert secret not in repr(secret_result)
    with pytest.raises(StructuredCallBoundaryError) as captured:
        validated_structured_call_result(secret_result.model_copy(update={"call_digest": "f" * 64}))
    assert secret not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__ is True


def test_result_for_request_rejects_every_identity_mismatch() -> None:
    request = _request()
    assert validated_result_for_request(request, _valid_result(request)) == _valid_result(request)

    intervention_request = _request(phase=StructuredCallPhase.INTERVENTION)
    cross_phase_values = _result_values(intervention_request)
    cross_phase_values["request_digest"] = request.request_digest
    mismatches = (
        _valid_result(_request(cycle_id="e" * 64)),
        StructuredCallResult(
            **cross_phase_values,
            status=StructuredCallStatus.COMPLETED,
            parse_status=StructuredCallParseStatus.VALID,
            output=_intervention_output(),
            completion_digest=_completion_digest(),
            completion_byte_count=128,
        ),
        StructuredCallResult(
            **{**_result_values(request), "model_call_index": 1},
            status=StructuredCallStatus.COMPLETED,
            parse_status=StructuredCallParseStatus.VALID,
            output=_bank_output(),
            completion_digest=_completion_digest(),
            completion_byte_count=128,
        ),
    )
    for mismatch in mismatches:
        with pytest.raises(StructuredCallBoundaryError) as captured:
            validated_result_for_request(request, mismatch)
        assert str(captured.value) == (
            "structured call result for request failed structured-call boundary validation"
        )
        assert captured.value.__cause__ is None

    indexed_request = _request(model_call_index=1)
    attempt_mismatch = StructuredCallResult(
        **{**_result_values(indexed_request), "attempt": 1},
        status=StructuredCallStatus.COMPLETED,
        parse_status=StructuredCallParseStatus.VALID,
        output=_bank_output(),
        completion_digest=_completion_digest(),
        completion_byte_count=128,
    )
    with pytest.raises(StructuredCallBoundaryError):
        validated_result_for_request(indexed_request, attempt_mismatch)


def test_structured_call_client_protocol_is_narrow() -> None:
    request = _request()
    expected = _valid_result(request)

    class FakeClient:
        async def generate(self, value: StructuredCallRequest) -> StructuredCallResult:
            assert value == request
            return expected

    client = FakeClient()
    assert isinstance(client, StructuredCallClient)
    assert asyncio.run(client.generate(request)) == expected
    assert not isinstance(object(), StructuredCallClient)


def test_structured_call_contract_is_exported_without_replacing_stage_one() -> None:
    assert public_ports.COMPLETION_DIGEST_SCOPE == "assistant-message-content-utf8/v1"
    assert public_ports.StructuredCallRequest is StructuredCallRequest
    assert public_ports.StructuredCallResult is StructuredCallResult
    assert public_ports.StructuredCallClient is StructuredCallClient
    assert public_ports.ModelRequest.__name__ == "ModelRequest"
    assert public_ports.ModelResult.__name__ == "ModelResult"


def test_memory_package_keeps_leaf_schemas_lazy_for_future_executor_imports() -> None:
    probe = """
import sys
import saliencegate.memory as memory
assert "saliencegate.memory.proposals" not in sys.modules
assert memory.BankOperationsProposal.__name__ == "BankOperationsProposal"
assert "saliencegate.memory.proposals" in sys.modules
from saliencegate.ports.model_calls import StructuredCallRequest
assert StructuredCallRequest.__name__ == "StructuredCallRequest"
"""
    completed = subprocess.run(
        (sys.executable, "-I", "-c", probe),
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == completed.stderr == ""
