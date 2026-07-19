from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest
from pydantic import ValidationError

import saliencegate.models.replay_two_phase as replay_module
from saliencegate.domain import (
    ClaimKind,
    EvidenceReference,
    EvidenceSource,
    InterventionAction,
    PayloadDigest,
    PayloadDigestAlgorithm,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.intervention import ProposedClaim
from saliencegate.memory.proposals import (
    INTERVENTION_OUTPUT_SCHEMA_VERSION,
    MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
    BankOperationsProposal,
    InterventionSelectionOutput,
    SaveKnowledge,
)
from saliencegate.models import (
    TWO_PHASE_REPLAY_RECORD_SCHEMA_VERSION,
    TWO_PHASE_REPLAY_VERSION,
    two_phase_receipts_are_replay_native,
    two_phase_replay_fixture_digest_from_receipts,
)
from saliencegate.models.replay import (
    ReplayFixtureError,
    ReplayIntegrityError,
    ReplayMissingResponseError,
    ReplayResponseReusedError,
)
from saliencegate.models.replay_two_phase import (
    TwoPhaseReplayClient,
    TwoPhaseReplayRecord,
)
from saliencegate.ports.model_calls import (
    CanonicalUsageProvenance,
    ProviderUsageProvenance,
    StructuredCallClient,
    StructuredCallParseStatus,
    StructuredCallPhase,
    StructuredCallRequest,
    StructuredCallResult,
    StructuredCallStatus,
    StructuredCallUsage,
)
from saliencegate.ports.two_phase import CallReceipt
from saliencegate.prompts import PAPER_TWO_PHASE_V1

RUN_ID = UUID("00000000-0000-4000-8000-000000004001")
EVENT_ID = UUID("00000000-0000-4000-8000-000000004002")
MEMORY_ID = UUID("00000000-0000-4000-8000-000000004003")
REPLAY_ID = "paper-two-phase-contract-responses/v1"
FIXTURE = Path(__file__).parents[1] / "fixtures" / "models" / "two_phase_contract_responses.jsonl"
_FIXTURE_DIGEST_DOMAIN = "saliencegate:model:two-phase-replay-fixture:v1"
_COMPLETION_DIGEST_DOMAIN = "saliencegate:test:two-phase-replay-completion:v1"
_SCENARIO_CYCLES = {
    "valid_edit_remind": "1" * 64,
    "valid_edit_silence": "2" * 64,
    "schema_invalid_second": "3" * 64,
    "model_error": "4" * 64,
    "timeout": "5" * 64,
}


def _request(scenario: str, phase: StructuredCallPhase) -> StructuredCallRequest:
    template = (
        PAPER_TWO_PHASE_V1.memory_edit_template
        if phase is StructuredCallPhase.MEMORY_EDIT
        else PAPER_TWO_PHASE_V1.intervention_template
    )
    return StructuredCallRequest(
        schema_version="structured-call-request/v1",
        run_id=RUN_ID,
        cycle_id=_SCENARIO_CYCLES[scenario],
        model_call_index=0 if phase is StructuredCallPhase.MEMORY_EDIT else 1,
        phase=phase,
        attempt=0,
        model_id="openai-compatible-replay/v1",
        prompt_template_id=template.template_id,
        prompt_template_digest=template.template_digest,
        response_schema_version=template.response_schema_version,  # type: ignore[arg-type]
        payload={
            "contract_case": scenario,
            "phase": phase.value,
            "candidate_bank_revision": (0 if phase is StructuredCallPhase.MEMORY_EDIT else 1),
        },
    )


def _usage(*, available: bool = True, latency_us: int = 50) -> StructuredCallUsage:
    return StructuredCallUsage(
        schema_version="structured-call-usage/v1",
        provider_input_tokens=20 if available else None,
        provider_output_tokens=5 if available else None,
        provider_usage_provenance=(
            ProviderUsageProvenance.REPLAY_ATTESTED
            if available
            else ProviderUsageProvenance.UNAVAILABLE
        ),
        latency_us=latency_us,
    )


def _memory_output(*, save: bool) -> BankOperationsProposal:
    return BankOperationsProposal(
        schema_version=MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
        operations=(
            (
                SaveKnowledge(
                    operation="save_knowledge",
                    content="Keep the verified deployment constraint.",
                    evidence=(
                        EvidenceReference(
                            source=EvidenceSource.EVENT,
                            source_id=EVENT_ID,
                            field_path="/payload/message",
                        ),
                    ),
                    confidence=1.0,
                ),
            )
            if save
            else ()
        ),
    )


def _intervention_output(*, remind: bool) -> InterventionSelectionOutput:
    return InterventionSelectionOutput(
        schema_version=INTERVENTION_OUTPUT_SCHEMA_VERSION,
        action=InterventionAction.REMIND if remind else InterventionAction.SILENCE,
        claims=(
            (
                ProposedClaim(
                    kind=ClaimKind.REQUIREMENT,
                    evidence=EvidenceReference(
                        source=EvidenceSource.MEMORY,
                        source_id=MEMORY_ID,
                        revision=1,
                        field_path="/content",
                    ),
                ),
            )
            if remind
            else ()
        ),
        confidence=1.0,
    )


def _completion_digest(content: bytes) -> PayloadDigest:
    return PayloadDigest(
        algorithm=PayloadDigestAlgorithm.SYNTHETIC_SHA256,
        value=length_prefixed_sha256(content, domain=_COMPLETION_DIGEST_DOMAIN),
    )


def _completed_result(
    request: StructuredCallRequest,
    output: BankOperationsProposal | InterventionSelectionOutput,
) -> StructuredCallResult:
    completion = canonical_json(output)
    return StructuredCallResult(
        schema_version="structured-call-result/v1",
        request_digest=request.request_digest,
        model_call_index=request.model_call_index,
        phase=request.phase,
        attempt=request.attempt,
        response_schema_version=request.response_schema_version,
        status=StructuredCallStatus.COMPLETED,
        parse_status=StructuredCallParseStatus.VALID,
        output=output,
        completion_digest=_completion_digest(completion),
        completion_byte_count=len(completion),
        usage=_usage(),
    )


def _schema_invalid_result(request: StructuredCallRequest) -> StructuredCallResult:
    completion = b'{"invalid":'
    return StructuredCallResult(
        schema_version="structured-call-result/v1",
        request_digest=request.request_digest,
        model_call_index=request.model_call_index,
        phase=request.phase,
        attempt=request.attempt,
        response_schema_version=request.response_schema_version,
        status=StructuredCallStatus.COMPLETED,
        parse_status=StructuredCallParseStatus.SCHEMA_INVALID,
        output=None,
        completion_digest=_completion_digest(completion),
        completion_byte_count=len(completion),
        usage=_usage(),
    )


def _failed_result(
    request: StructuredCallRequest,
    status: StructuredCallStatus,
) -> StructuredCallResult:
    return StructuredCallResult(
        schema_version="structured-call-result/v1",
        request_digest=request.request_digest,
        model_call_index=request.model_call_index,
        phase=request.phase,
        attempt=request.attempt,
        response_schema_version=request.response_schema_version,
        status=status,
        parse_status=StructuredCallParseStatus.NOT_ATTEMPTED,
        output=None,
        completion_digest=None,
        completion_byte_count=None,
        usage=_usage(available=False),
    )


def _changed_result(
    result: StructuredCallResult,
    **changes: object,
) -> StructuredCallResult:
    values = result.model_dump(mode="json", exclude={"call_digest"})
    values.update(
        {
            key: value.model_dump(mode="json") if isinstance(value, StructuredCallUsage) else value
            for key, value in changes.items()
        }
    )
    return StructuredCallResult.model_validate_json(canonical_json(values))


def _reviewed_requests_and_results() -> tuple[
    tuple[StructuredCallRequest, StructuredCallResult], ...
]:
    remind_edit = _request("valid_edit_remind", StructuredCallPhase.MEMORY_EDIT)
    remind = _request("valid_edit_remind", StructuredCallPhase.INTERVENTION)
    silence_edit = _request("valid_edit_silence", StructuredCallPhase.MEMORY_EDIT)
    silence = _request("valid_edit_silence", StructuredCallPhase.INTERVENTION)
    invalid_edit = _request("schema_invalid_second", StructuredCallPhase.MEMORY_EDIT)
    invalid = _request("schema_invalid_second", StructuredCallPhase.INTERVENTION)
    model_error = _request("model_error", StructuredCallPhase.MEMORY_EDIT)
    timeout = _request("timeout", StructuredCallPhase.INTERVENTION)
    return (
        (remind_edit, _completed_result(remind_edit, _memory_output(save=True))),
        (remind, _completed_result(remind, _intervention_output(remind=True))),
        (silence_edit, _completed_result(silence_edit, _memory_output(save=False))),
        (silence, _completed_result(silence, _intervention_output(remind=False))),
        (invalid_edit, _completed_result(invalid_edit, _memory_output(save=True))),
        (invalid, _schema_invalid_result(invalid)),
        (model_error, _failed_result(model_error, StructuredCallStatus.MODEL_ERROR)),
        (timeout, _failed_result(timeout, StructuredCallStatus.MODEL_TIMEOUT)),
    )


def _fixture_digest(results: tuple[StructuredCallResult, ...]) -> str:
    material = tuple(
        {
            "schema_version": "two-phase-replay-record/v1",
            "record_type": "two_phase_replay_response",
            "replay_version": "two-phase-replay/v1",
            "ordinal": ordinal,
            "request_digest": result.request_digest,
            "model_call_index": result.model_call_index,
            "phase": result.phase.value,
            "attempt": result.attempt,
            "response_schema_version": result.response_schema_version,
            "call_digest": result.call_digest,
        }
        for ordinal, result in enumerate(results, start=1)
    )
    return length_prefixed_sha256(
        REPLAY_ID,
        canonical_json(material),
        domain=_FIXTURE_DIGEST_DOMAIN,
    )


def _sealed_records(
    pairs: tuple[tuple[StructuredCallRequest, StructuredCallResult], ...],
) -> tuple[TwoPhaseReplayRecord, ...]:
    results = tuple(result for _request_value, result in pairs)
    fixture_digest = _fixture_digest(results)
    return tuple(
        TwoPhaseReplayRecord(
            replay_id=REPLAY_ID,
            fixture_digest=fixture_digest,
            ordinal=ordinal,
            request_digest=result.request_digest,
            model_call_index=result.model_call_index,
            phase=result.phase,
            attempt=result.attempt,
            response_schema_version=result.response_schema_version,
            result=result,
        )
        for ordinal, result in enumerate(results, start=1)
    )


def _reviewed_records() -> tuple[TwoPhaseReplayRecord, ...]:
    return _sealed_records(_reviewed_requests_and_results())


def _reviewed_receipts() -> tuple[CallReceipt, ...]:
    return tuple(
        CallReceipt(
            run_id=request.run_id,
            cycle_id=request.cycle_id,
            model_call_index=request.model_call_index,
            phase=request.phase,
            attempt=request.attempt,
            model_id=request.model_id,
            prompt_template_id=request.prompt_template_id,
            prompt_template_digest=request.prompt_template_digest,
            prompt_digest="6" * 64,
            request_payload_digest="7" * 64,
            window_digest="8" * 64,
            bank_view_digest="9" * 64,
            grounding_state_digest=(
                None if request.phase is StructuredCallPhase.MEMORY_EDIT else "a" * 64
            ),
            request_digest=result.request_digest,
            status=result.status,
            parse_status=result.parse_status,
            completion_digest=result.completion_digest,
            completion_byte_count=result.completion_byte_count,
            usage=result.usage,
            call_digest=result.call_digest,
        )
        for request, result in _reviewed_requests_and_results()
    )


def _receipt_with_result(
    receipt: CallReceipt,
    result: StructuredCallResult,
) -> CallReceipt:
    assert (
        receipt.request_digest,
        receipt.model_call_index,
        receipt.phase,
        receipt.attempt,
    ) == (
        result.request_digest,
        result.model_call_index,
        result.phase,
        result.attempt,
    )
    values = receipt.model_dump(mode="python", exclude={"receipt_digest"})
    values.update(
        {
            "status": result.status,
            "parse_status": result.parse_status,
            "completion_digest": result.completion_digest,
            "completion_byte_count": result.completion_byte_count,
            "usage": result.usage,
            "call_digest": result.call_digest,
        }
    )
    return CallReceipt.model_validate(values)


def _fixture_bytes(records: tuple[TwoPhaseReplayRecord, ...]) -> bytes:
    return b"".join(canonical_json(record) + b"\n" for record in records)


def test_committed_fixture_is_canonical_independent_and_digest_pinned() -> None:
    pairs = _reviewed_requests_and_results()
    records = _reviewed_records()
    committed = FIXTURE.read_bytes()

    assert committed == _fixture_bytes(records)
    assert all(
        request.request_digest == record.request_digest
        for (request, _result), record in zip(pairs, records, strict=True)
    )
    assert (
        records[0].fixture_digest
        == "68e2412dfd5c90af7a4d2916f291b8b0868f985ed95c43e2f15df9252a720434"
    )
    assert tuple(request.request_digest for request, _result in pairs) == (
        "a043dd96073da094f444f6cf26aab40765117ecd1e394946e952d515d80926f6",
        "4bd9d7962cdf507dd7f827de72cbf4ae57f367013043cc6fdd774f050cfbd1b2",
        "91f00bd3486ba2efbaad35e98944f81d9304bafcfcaa5fd5d00af768d803a662",
        "5072e435beacbb16212f8feaa40729d4e5e97961139cefb61e1ecc399220eac7",
        "7d60a707adf9a036aed128b2651ab249e1716341f6952e62810522ea0c9beddb",
        "cf21d0bc1b550689d771613493a58ba8e017d3ffcd2c4e748e29679e29e1e452",
        "e40448e2c5e33a12ad5f58c34706e3b68dd50fb74c5d71fb269cdf2c48d3f3b2",
        "4790dbab1ba8910949ae69b19f6be78c1d63c8d632355b530fb68f5856b252b3",
    )
    assert committed.endswith(b"\n")
    assert b"\r" not in committed
    assert all(canonical_json(json.loads(line)) == line for line in committed.splitlines())


def test_fixture_digest_is_reconstructed_exactly_from_real_call_receipts() -> None:
    records = _reviewed_records()
    receipts = _reviewed_receipts()

    reconstructed = two_phase_replay_fixture_digest_from_receipts(
        receipts,
        replay_id=REPLAY_ID,
    )

    assert reconstructed == records[0].fixture_digest
    assert reconstructed == TwoPhaseReplayClient(records, replay_id=REPLAY_ID).fixture_digest
    assert (
        two_phase_replay_fixture_digest_from_receipts(
            tuple(reversed(receipts)),
            replay_id=REPLAY_ID,
        )
        != reconstructed
    )


def test_fixture_digest_from_receipts_rejects_invalid_and_tampered_inputs() -> None:
    receipts = _reviewed_receipts()
    duplicate_call_values = receipts[1].model_dump(mode="python", exclude={"receipt_digest"})
    duplicate_call_values["call_digest"] = receipts[0].call_digest
    duplicate_call = CallReceipt.model_validate(duplicate_call_values)
    malformed: tuple[object, ...] = (
        list(receipts),
        (cast(CallReceipt, object()),),
        (receipts[0], receipts[0]),
        (receipts[0], duplicate_call),
        (receipts[0].model_copy(update={"call_digest": "0" * 64}),),
    )

    for value in malformed:
        with pytest.raises(ReplayFixtureError):
            two_phase_replay_fixture_digest_from_receipts(
                cast(tuple[CallReceipt, ...], value),
                replay_id=REPLAY_ID,
            )

    secret = "invalid replay identity with secret"
    with pytest.raises(ReplayFixtureError) as raised:
        two_phase_replay_fixture_digest_from_receipts(receipts, replay_id=secret)
    assert secret not in str(raised.value)


def test_replay_native_receipt_predicate_is_exact_and_fail_closed() -> None:
    receipts = _reviewed_receipts()
    completed = _reviewed_requests_and_results()[0][1]
    live = _changed_result(
        completed,
        completion_digest=PayloadDigest(
            algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
            value="a" * 64,
        ),
        usage=StructuredCallUsage(
            schema_version="structured-call-usage/v1",
            provider_input_tokens=20,
            provider_output_tokens=5,
            provider_usage_provenance=ProviderUsageProvenance.PROVIDER_REPORTED,
            latency_us=50,
        ),
    )
    completed_without_attestation = _changed_result(completed, usage=_usage(available=False))
    failed_with_canonical_attestation_values = receipts[-2].model_dump(
        mode="python",
        exclude={"receipt_digest"},
    )
    failed_with_canonical_attestation_values["usage"] = StructuredCallUsage(
        schema_version="structured-call-usage/v1",
        provider_input_tokens=None,
        provider_output_tokens=None,
        provider_usage_provenance=ProviderUsageProvenance.UNAVAILABLE,
        latency_us=50,
        canonical_input_tokens=22,
        canonical_output_tokens=0,
        canonical_usage_provenance=CanonicalUsageProvenance.REPLAY_ATTESTED,
        local_counter_id="deterministic-model-token-counter",
        local_counter_version="fixed-count-fixture/v1",
        local_counter_configuration_digest="d" * 64,
        local_counter_model_id="openai-compatible-replay/v1",
    )
    failed_with_canonical_attestation = CallReceipt.model_validate(
        failed_with_canonical_attestation_values
    )
    failed_with_partial_canonical_values = failed_with_canonical_attestation_values | {
        "usage": StructuredCallUsage(
            schema_version="structured-call-usage/v1",
            provider_input_tokens=None,
            provider_output_tokens=None,
            provider_usage_provenance=ProviderUsageProvenance.UNAVAILABLE,
            latency_us=50,
            canonical_input_tokens=22,
            canonical_output_tokens=None,
            canonical_usage_provenance=CanonicalUsageProvenance.REPLAY_ATTESTED,
            local_counter_id="deterministic-model-token-counter",
            local_counter_version="fixed-count-fixture/v1",
            local_counter_configuration_digest="d" * 64,
            local_counter_model_id="openai-compatible-replay/v1",
        )
    }
    failed_with_partial_canonical = CallReceipt.model_validate(failed_with_partial_canonical_values)

    assert two_phase_receipts_are_replay_native(receipts)
    assert two_phase_receipts_are_replay_native(receipts[-2:])
    assert not two_phase_receipts_are_replay_native((_receipt_with_result(receipts[0], live),))
    assert not two_phase_receipts_are_replay_native(
        (_receipt_with_result(receipts[0], completed_without_attestation),)
    )
    assert not two_phase_receipts_are_replay_native((failed_with_canonical_attestation,))
    assert two_phase_receipts_are_replay_native((failed_with_partial_canonical,))
    assert not two_phase_receipts_are_replay_native(())
    assert not two_phase_receipts_are_replay_native(cast(tuple[CallReceipt, ...], [receipts[0]]))
    assert not two_phase_receipts_are_replay_native(
        (receipts[0].model_copy(update={"receipt_digest": "0" * 64}),)
    )


def test_fixture_covers_required_states_and_no_repair_call_order() -> None:
    records = _reviewed_records()

    assert (records[0].model_call_index, records[0].phase, records[0].attempt) == (
        0,
        StructuredCallPhase.MEMORY_EDIT,
        0,
    )
    assert (records[1].model_call_index, records[1].phase, records[1].attempt) == (
        1,
        StructuredCallPhase.INTERVENTION,
        0,
    )
    assert tuple(record.result.status for record in records) == (
        StructuredCallStatus.COMPLETED,
        StructuredCallStatus.COMPLETED,
        StructuredCallStatus.COMPLETED,
        StructuredCallStatus.COMPLETED,
        StructuredCallStatus.COMPLETED,
        StructuredCallStatus.COMPLETED,
        StructuredCallStatus.MODEL_ERROR,
        StructuredCallStatus.MODEL_TIMEOUT,
    )
    assert records[5].result.parse_status is StructuredCallParseStatus.SCHEMA_INVALID
    assert isinstance(records[1].result.output, InterventionSelectionOutput)
    assert records[1].result.output.action is InterventionAction.REMIND
    assert isinstance(records[3].result.output, InterventionSelectionOutput)
    assert records[3].result.output.action is InterventionAction.SILENCE


@pytest.mark.asyncio
async def test_client_returns_revalidated_results_once_and_implements_protocol() -> None:
    pairs = _reviewed_requests_and_results()[:2]
    records = _sealed_records(pairs)
    client = TwoPhaseReplayClient(records, replay_id=REPLAY_ID)

    first = await client.generate(pairs[0][0])
    second = await client.generate(pairs[1][0])

    assert isinstance(client, StructuredCallClient)
    assert first == pairs[0][1] and first is not pairs[0][1]
    assert second == pairs[1][1] and second is not pairs[1][1]
    assert client.total_responses == 2
    assert client.remaining_responses == 0
    with pytest.raises(ReplayResponseReusedError):
        await client.generate(pairs[0][0])


@pytest.mark.asyncio
async def test_missing_and_reused_errors_are_distinct_and_value_free() -> None:
    pairs = _reviewed_requests_and_results()
    records = _sealed_records((pairs[0],))
    client = TwoPhaseReplayClient(records, replay_id=REPLAY_ID)

    with pytest.raises(ReplayMissingResponseError) as missing:
        await client.generate(pairs[2][0])
    assert pairs[2][0].request_digest not in str(missing.value)
    assert client.remaining_responses == 1

    await client.generate(pairs[0][0])
    with pytest.raises(ReplayResponseReusedError) as reused:
        await client.generate(pairs[0][0])
    assert pairs[0][0].request_digest not in str(reused.value)


@pytest.mark.asyncio
async def test_concurrent_duplicate_consumers_have_one_winner() -> None:
    request, _result = _reviewed_requests_and_results()[0]
    records = _sealed_records((_reviewed_requests_and_results()[0],))
    client = TwoPhaseReplayClient(records, replay_id=REPLAY_ID)

    received = await asyncio.gather(
        client.generate(request),
        client.generate(request),
        return_exceptions=True,
    )

    assert sum(type(item) is StructuredCallResult for item in received) == 1
    assert sum(type(item) is ReplayResponseReusedError for item in received) == 1


def test_record_and_constructor_bind_every_duplicate_identity_field() -> None:
    first = _reviewed_records()[0]
    changes: tuple[dict[str, object], ...] = (
        {"request_digest": "0" * 64},
        {"model_call_index": 1},
        {"phase": StructuredCallPhase.INTERVENTION},
        {"attempt": 1},
        {"response_schema_version": INTERVENTION_OUTPUT_SCHEMA_VERSION},
        {"record_digest": "0" * 64},
    )
    for change in changes:
        with pytest.raises(ValidationError):
            TwoPhaseReplayRecord.model_validate(first.model_dump(mode="python") | change)

    malformed_sets: tuple[object, ...] = (
        [first],
        (first.model_copy(update={"ordinal": 2}),),
        (first, first.model_copy(update={"ordinal": 2})),
        (first.model_copy(update={"fixture_digest": "0" * 64}),),
        (first.model_copy(update={"replay_id": "other/v1"}),),
        (cast(TwoPhaseReplayRecord, object()),),
    )
    for records in malformed_sets:
        with pytest.raises(ReplayFixtureError):
            TwoPhaseReplayClient(
                cast(tuple[TwoPhaseReplayRecord, ...], records),
                replay_id=REPLAY_ID,
            )


def test_record_rejects_forged_result_and_non_replay_provenance() -> None:
    valid = _reviewed_records()[0]
    forged = valid.model_copy(update={"result": cast(StructuredCallResult, object())})
    with pytest.raises(ValueError, match="result failed validation"):
        forged.result_metadata_and_digests_match()
    mismatched = valid.model_copy(update={"model_call_index": valid.model_call_index + 1})
    with pytest.raises(ValueError, match="metadata does not match"):
        mismatched.result_metadata_and_digests_match()

    completed = _reviewed_requests_and_results()[0][1]
    provider_reported = _changed_result(
        completed,
        usage=StructuredCallUsage(
            schema_version="structured-call-usage/v1",
            provider_input_tokens=20,
            provider_output_tokens=5,
            provider_usage_provenance=ProviderUsageProvenance.PROVIDER_REPORTED,
            latency_us=50,
        ),
    )
    unavailable = _changed_result(completed, usage=_usage(available=False))
    canonical_replay = _changed_result(
        completed,
        usage=StructuredCallUsage(
            schema_version="structured-call-usage/v1",
            provider_input_tokens=None,
            provider_output_tokens=None,
            provider_usage_provenance=ProviderUsageProvenance.UNAVAILABLE,
            latency_us=50,
            canonical_input_tokens=22,
            canonical_output_tokens=8,
            canonical_usage_provenance=CanonicalUsageProvenance.REPLAY_ATTESTED,
            local_counter_id="deterministic-model-token-counter",
            local_counter_version="fixed-count-fixture/v1",
            local_counter_configuration_digest="d" * 64,
            local_counter_model_id="openai-compatible-replay/v1",
        ),
    )
    canonical_local = _changed_result(
        canonical_replay,
        usage=canonical_replay.usage.model_copy(
            update={"canonical_usage_provenance": CanonicalUsageProvenance.LOCAL_COUNTER}
        ),
    )
    hmac_completion = _changed_result(
        completed,
        completion_digest=PayloadDigest(
            algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
            value="a" * 64,
        ),
    )
    accepted = TwoPhaseReplayRecord(
        replay_id=REPLAY_ID,
        fixture_digest="a" * 64,
        ordinal=1,
        request_digest=canonical_replay.request_digest,
        model_call_index=canonical_replay.model_call_index,
        phase=canonical_replay.phase,
        attempt=canonical_replay.attempt,
        response_schema_version=canonical_replay.response_schema_version,
        result=canonical_replay,
    )
    assert accepted.result.usage.canonical_tokens == 30

    for result in (provider_reported, unavailable, canonical_local, hmac_completion):
        with pytest.raises(ValidationError):
            TwoPhaseReplayRecord(
                replay_id=REPLAY_ID,
                fixture_digest="a" * 64,
                ordinal=1,
                request_digest=result.request_digest,
                model_call_index=result.model_call_index,
                phase=result.phase,
                attempt=result.attempt,
                response_schema_version=result.response_schema_version,
                result=result,
            )


def test_fixture_digest_binds_order_replay_identity_and_external_expectation() -> None:
    records = _reviewed_records()
    client = TwoPhaseReplayClient(
        records,
        replay_id=REPLAY_ID,
        expected_fixture_digest=records[0].fixture_digest,
    )

    assert client.fixture_digest == records[0].fixture_digest
    with pytest.raises(ReplayFixtureError):
        TwoPhaseReplayClient(
            records,
            replay_id=REPLAY_ID,
            expected_fixture_digest="0" * 64,
        )
    with pytest.raises(ReplayFixtureError):
        TwoPhaseReplayClient(tuple(reversed(records)), replay_id=REPLAY_ID)


def test_from_path_loads_exact_fixture_and_generate_performs_no_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _sealed_records(_reviewed_requests_and_results()[:2])
    path = tmp_path / "responses.jsonl"
    path.write_bytes(_fixture_bytes(records))
    client = TwoPhaseReplayClient.from_path(path, replay_id=REPLAY_ID)
    path.unlink()
    monkeypatch.setattr(replay_module.os, "open", lambda *_args, **_kwargs: pytest.fail("I/O"))

    received = asyncio.run(client.generate(_reviewed_requests_and_results()[0][0]))

    assert received == records[0].result


def test_committed_fixture_loads_with_external_digest_anchor() -> None:
    records = _reviewed_records()
    client = TwoPhaseReplayClient.from_path(
        FIXTURE,
        replay_id=REPLAY_ID,
        expected_fixture_digest=records[0].fixture_digest,
    )

    assert client.replay_id == REPLAY_ID
    assert client.fixture_digest == records[0].fixture_digest
    assert client.total_responses == 8
    assert client.remaining_responses == 8


@pytest.mark.parametrize(
    "contents",
    (
        b"\n",
        b"[]\n",
        b'{"ordinal":1,"ordinal":1}\n',
        b'{"value":NaN}\n',
        b"\xff\n",
        b" {}\n",
        b"{}\r\n",
        b"{}",
    ),
)
def test_from_path_rejects_noncanonical_or_invalid_jsonl(
    tmp_path: Path,
    contents: bytes,
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_bytes(contents)

    with pytest.raises(ReplayFixtureError):
        TwoPhaseReplayClient.from_path(path, replay_id=REPLAY_ID)


def test_from_path_rejects_reordered_tampered_or_incomplete_records(tmp_path: Path) -> None:
    records = _sealed_records(_reviewed_requests_and_results()[:2])
    path = tmp_path / "changed.jsonl"
    first = records[0].model_dump(mode="json")
    del first["record_digest"]
    missing_call_digest = records[0].model_dump(mode="json")
    missing_call_result = cast(dict[str, object], missing_call_digest["result"])
    del missing_call_result["call_digest"]
    missing_usage_field = records[0].model_dump(mode="json")
    usage_result = cast(dict[str, object], missing_usage_field["result"])
    usage = cast(dict[str, object], usage_result["usage"])
    del usage["latency_us"]
    variants = (
        canonical_json(first) + b"\n",
        canonical_json(missing_call_digest) + b"\n",
        canonical_json(missing_usage_field) + b"\n",
        _fixture_bytes(tuple(reversed(records))),
        canonical_json(records[0].model_dump(mode="json") | {"record_digest": "0" * 64}) + b"\n",
        json.dumps(records[0].model_dump(mode="json"), sort_keys=True).encode() + b"\n",
    )
    for contents in variants:
        path.write_bytes(contents)
        with pytest.raises(ReplayFixtureError):
            TwoPhaseReplayClient.from_path(path, replay_id=REPLAY_ID)


def test_from_path_applies_count_line_total_and_structural_bounds_before_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _sealed_records(_reviewed_requests_and_results()[:2])
    path = tmp_path / "bounded.jsonl"
    path.write_bytes(_fixture_bytes(records))

    monkeypatch.setattr(replay_module, "_MAX_RECORDS", 1)
    with pytest.raises(ReplayFixtureError):
        TwoPhaseReplayClient.from_path(path, replay_id=REPLAY_ID)
    monkeypatch.setattr(replay_module, "_MAX_RECORDS", 100_000)
    monkeypatch.setattr(replay_module, "_MAX_LINE_BYTES", 8)
    with pytest.raises(ReplayFixtureError):
        TwoPhaseReplayClient.from_path(path, replay_id=REPLAY_ID)
    monkeypatch.setattr(replay_module, "_MAX_LINE_BYTES", 16 * 1024 * 1024)
    monkeypatch.setattr(replay_module, "_MAX_FIXTURE_BYTES", path.stat().st_size - 1)
    with pytest.raises(ReplayFixtureError):
        TwoPhaseReplayClient.from_path(path, replay_id=REPLAY_ID)
    monkeypatch.setattr(replay_module, "_MAX_FIXTURE_BYTES", 64 * 1024 * 1024)
    monkeypatch.setattr(replay_module, "_MAX_JSON_NODES", 1)
    with pytest.raises(ReplayFixtureError):
        TwoPhaseReplayClient.from_path(path, replay_id=REPLAY_ID)


def test_from_path_requires_regular_single_link_stable_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "responses.jsonl"
    single = _sealed_records((_reviewed_requests_and_results()[0],))
    source.write_bytes(_fixture_bytes(single))
    symbolic = tmp_path / "symbolic.jsonl"
    symbolic.symlink_to(source)
    hard = tmp_path / "hard.jsonl"
    hard.hardlink_to(source)

    for path in (tmp_path, symbolic, hard):
        with pytest.raises(ReplayFixtureError):
            TwoPhaseReplayClient.from_path(path, replay_id=REPLAY_ID)
    hard.unlink()

    original_stat = replay_module.os.stat

    def changed_stat(path: object, *, follow_symlinks: bool = True) -> SimpleNamespace:
        value = original_stat(path, follow_symlinks=follow_symlinks)
        return SimpleNamespace(
            st_mode=value.st_mode,
            st_nlink=value.st_nlink,
            st_dev=value.st_dev,
            st_ino=value.st_ino,
            st_size=value.st_size,
            st_mtime_ns=value.st_mtime_ns + 1,
            st_ctime_ns=value.st_ctime_ns,
        )

    monkeypatch.setattr(replay_module.os, "stat", changed_stat)
    with pytest.raises(ReplayFixtureError):
        TwoPhaseReplayClient.from_path(source, replay_id=REPLAY_ID)


def test_from_path_rejects_wrong_missing_and_non_text_path_values(tmp_path: Path) -> None:
    with pytest.raises(ReplayFixtureError):
        TwoPhaseReplayClient.from_path(cast(Path, object()), replay_id=REPLAY_ID)
    with pytest.raises(ReplayFixtureError):
        TwoPhaseReplayClient.from_path(tmp_path / "missing.jsonl", replay_id=REPLAY_ID)

    class BytesPath:
        def __fspath__(self) -> bytes:
            return b"secret-path"

    with pytest.raises(ReplayFixtureError) as raised:
        TwoPhaseReplayClient.from_path(BytesPath(), replay_id=REPLAY_ID)
    assert "secret-path" not in str(raised.value)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires POSIX")
def test_from_path_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "responses.fifo"
    os.mkfifo(fifo)

    with pytest.raises(ReplayFixtureError):
        TwoPhaseReplayClient.from_path(fifo, replay_id=REPLAY_ID)


@pytest.mark.asyncio
async def test_forged_request_or_stored_result_fails_without_consuming() -> None:
    request, _result = _reviewed_requests_and_results()[0]
    records = _sealed_records((_reviewed_requests_and_results()[0],))
    client = TwoPhaseReplayClient(records, replay_id=REPLAY_ID)
    forged_request = request.model_copy(update={"phase": StructuredCallPhase.INTERVENTION})

    with pytest.raises(ReplayIntegrityError):
        await client.generate(forged_request)
    assert client.remaining_responses == 1

    stored = cast(dict[str, TwoPhaseReplayRecord], client._records)[request.request_digest]
    stored.result.__dict__["call_digest"] = "0" * 64
    with pytest.raises(ReplayIntegrityError):
        await client.generate(request)
    assert client.remaining_responses == 1

    replacement_result = _changed_result(records[0].result, usage=_usage(latency_us=51))
    replacement = TwoPhaseReplayRecord(
        replay_id=REPLAY_ID,
        fixture_digest=records[0].fixture_digest,
        ordinal=records[0].ordinal,
        request_digest=records[0].request_digest,
        model_call_index=records[0].model_call_index,
        phase=records[0].phase,
        attempt=records[0].attempt,
        response_schema_version=records[0].response_schema_version,
        result=replacement_result,
    )
    assert replacement.record_digest != records[0].record_digest
    fresh = TwoPhaseReplayClient(records, replay_id=REPLAY_ID)
    with pytest.raises(TypeError):
        cast(dict[str, TwoPhaseReplayRecord], fresh._records)[request.request_digest] = replacement
    object.__setattr__(fresh, "_records", {request.request_digest: replacement})
    with pytest.raises(ReplayIntegrityError):
        await fresh.generate(request)
    assert fresh.remaining_responses == 1


def test_public_versions_and_record_are_strict_frozen_and_value_safe() -> None:
    record = _reviewed_records()[0]

    assert TWO_PHASE_REPLAY_VERSION == "two-phase-replay/v1"
    assert TWO_PHASE_REPLAY_RECORD_SCHEMA_VERSION == "two-phase-replay-record/v1"
    assert "output=" not in repr(record)
    with pytest.raises(ValidationError):
        record.ordinal = 2  # type: ignore[misc]
    with pytest.raises(ReplayFixtureError) as raised:
        TwoPhaseReplayClient((), replay_id="contains spaces")
    assert "contains spaces" not in str(raised.value)
