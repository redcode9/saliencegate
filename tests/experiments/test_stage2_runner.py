from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from saliencegate.domain import (
    ClaimKind,
    CycleRecord,
    CycleState,
    EvidenceReference,
    EvidenceSource,
    InterventionAction,
    MemoryKind,
    PayloadDigest,
    PayloadDigestAlgorithm,
    ReasonCode,
    canonical_digest,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.experiments import (
    Stage2ConditionId,
    Stage2ExperimentError,
    Stage2ExperimentMetrics,
    Stage2ExperimentRunner,
    Stage2ExperimentRunResult,
    build_stage2_trajectory,
    load_stage2_trajectory,
    replay_stage2_fixture_twice,
)
from saliencegate.experiments import runner as runner_module
from saliencegate.intervention import DeterministicSelectorProvenance, ProposedClaim
from saliencegate.memory import (
    BankOperation,
    BankOperationsProposal,
    DeleteMemory,
    InterventionSelectionOutput,
    SaveKnowledge,
    SaveProcedural,
    UpdatePrivateStatus,
)
from saliencegate.models.replay_two_phase import (
    TWO_PHASE_REPLAY_VERSION,
    TwoPhaseReplayClient,
    TwoPhaseReplayRecord,
    two_phase_replay_fixture_digest_from_receipts,
)
from saliencegate.ports.adapters import enqueue_delivery_binding
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
    StructuredPhaseOutput,
)
from saliencegate.ports.repository import LedgerEntry, ProjectionInvariantError
from saliencegate.ports.two_phase import TwoPhaseFailureReason
from saliencegate.prompts.contracts import (
    PromptDataSectionName,
    StructuredPromptPayload,
    parse_untrusted_prompt_data,
)
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.repository.projector import Projection, apply_entry, empty_projection
from saliencegate.runtime.algorithm_result import (
    algorithm_runtime_uuid,
    algorithm_trace_digest,
)

ROOT = Path(__file__).resolve().parents[2]
TRAJECTORY_FIXTURE = ROOT / "tests/fixtures/runs/paper_two_phase_basic.jsonl"
FIXED_RESPONSES = ROOT / "tests/fixtures/models/paper_two_phase_fixed_step_responses.jsonl"
RETRIEVAL_RESPONSES = ROOT / "tests/fixtures/models/paper_two_phase_retrieval_responses.jsonl"
ALWAYS_RESPONSES = ROOT / "tests/fixtures/models/paper_two_phase_always_inject_responses.jsonl"

TRAJECTORY_DIGEST = "751489f55ac9d5ea56408ca6f5036b55e895be0fa130c36f19e624ee094d1266"
RUN_ID = UUID("00000000-0000-4000-8000-000000009000")
EVENT_1 = UUID("00000000-0000-4000-8000-000000009001")
EVENT_6 = UUID("00000000-0000-4000-8000-000000009006")
_FIXTURE_DIGEST_DOMAIN = "saliencegate:model:two-phase-replay-fixture:v1"
_COMPLETION_DIGEST_DOMAIN = "saliencegate:test:stage2-runner-completion:v1"
_COUNTER_CONFIGURATION_DIGEST = "4" * 64


def _repository(trace_digest: str) -> MemoryRunRepository:
    ordinal = 0

    def next_identifier() -> UUID:
        nonlocal ordinal
        ordinal += 1
        return algorithm_runtime_uuid(trace_digest, "stage2-repository", ordinal)

    return MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=next_identifier,
    )


def _prompt_bank(request: StructuredCallRequest) -> tuple[Mapping[str, object], ...]:
    payload = StructuredPromptPayload.model_validate_json(canonical_json(request.payload))
    envelope = parse_untrusted_prompt_data(payload.messages[1].content)
    bank = envelope.section(PromptDataSectionName.MEMORY_BANK)
    records = bank["records"]
    assert isinstance(records, (tuple, list))
    assert all(isinstance(item, Mapping) for item in records)
    return cast(tuple[Mapping[str, object], ...], tuple(records))


def _memory_reference(
    request: StructuredCallRequest,
    kind: MemoryKind,
) -> EvidenceReference:
    record = next(item for item in _prompt_bank(request) if item["kind"] == kind.value)
    return EvidenceReference(
        source=EvidenceSource.MEMORY,
        source_id=UUID(cast(str, record["memory_id"])),
        revision=cast(int, record["revision"]),
        field_path="/content",
    )


def _usage(
    request: StructuredCallRequest,
    completion: bytes,
    *,
    live_evidence: bool = False,
) -> StructuredCallUsage:
    input_tokens = (len(canonical_json(request.payload)) + 3) // 4
    output_tokens = (len(completion) + 3) // 4
    return StructuredCallUsage(
        schema_version="structured-call-usage/v1",
        provider_input_tokens=input_tokens if live_evidence else None,
        provider_output_tokens=output_tokens if live_evidence else None,
        provider_usage_provenance=(
            ProviderUsageProvenance.PROVIDER_REPORTED
            if live_evidence
            else ProviderUsageProvenance.UNAVAILABLE
        ),
        latency_us=1_000 + request.model_call_index,
        canonical_input_tokens=input_tokens,
        canonical_output_tokens=output_tokens,
        canonical_usage_provenance=(
            CanonicalUsageProvenance.LOCAL_COUNTER
            if live_evidence
            else CanonicalUsageProvenance.REPLAY_ATTESTED
        ),
        local_counter_id="stage2-reviewed-utf8-counter/v1",
        local_counter_version="utf8-bytes-ceil-div-4/v1",
        local_counter_configuration_digest=_COUNTER_CONFIGURATION_DIGEST,
        local_counter_model_id=request.model_id,
    )


def _completed_result(
    request: StructuredCallRequest,
    output: StructuredPhaseOutput,
    *,
    live_evidence: bool = False,
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
        completion_digest=PayloadDigest(
            algorithm=(
                PayloadDigestAlgorithm.HMAC_SHA256
                if live_evidence
                else PayloadDigestAlgorithm.SYNTHETIC_SHA256
            ),
            value=length_prefixed_sha256(
                completion,
                domain=_COMPLETION_DIGEST_DOMAIN,
            ),
        ),
        completion_byte_count=len(completion),
        usage=_usage(request, completion, live_evidence=live_evidence),
    )


def _schema_invalid_result(
    request: StructuredCallRequest,
    *,
    live_evidence: bool = False,
) -> StructuredCallResult:
    completion = b"{}"
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
        completion_digest=PayloadDigest(
            algorithm=(
                PayloadDigestAlgorithm.HMAC_SHA256
                if live_evidence
                else PayloadDigestAlgorithm.SYNTHETIC_SHA256
            ),
            value=length_prefixed_sha256(
                completion,
                domain=_COMPLETION_DIGEST_DOMAIN,
            ),
        ),
        completion_byte_count=len(completion),
        usage=_usage(request, completion, live_evidence=live_evidence),
    )


class _ReviewedClient:
    def __init__(
        self,
        condition: Stage2ConditionId,
        *,
        live_evidence: bool = False,
    ) -> None:
        self.condition = condition
        self.live_evidence = live_evidence
        self.cycle_ordinal = 0
        self.pairs: list[tuple[StructuredCallRequest, StructuredCallResult]] = []

    def _memory_output(self, request: StructuredCallRequest) -> BankOperationsProposal:
        operations: tuple[BankOperation, ...]
        if self.cycle_ordinal == 1:
            operations = ()
        elif self.cycle_ordinal == 2:
            operations = (
                UpdatePrivateStatus(
                    operation="update_private_status",
                    content=(
                        "Complete final verification before reporting the release workflow "
                        "finished."
                    ),
                    evidence=(
                        EvidenceReference(
                            source=EvidenceSource.EVENT,
                            source_id=EVENT_1,
                            field_path="/payload/task",
                        ),
                    ),
                    confidence=1.0,
                ),
                SaveKnowledge(
                    operation="save_knowledge",
                    content=(
                        "The release must remain offline and preserve the verified SHA-256 "
                        "manifest checksums."
                    ),
                    evidence=(
                        EvidenceReference(
                            source=EvidenceSource.EVENT,
                            source_id=EVENT_6,
                            field_path="/payload/constraint",
                        ),
                    ),
                    confidence=1.0,
                ),
                SaveProcedural(
                    operation="save_procedural",
                    content=(
                        "If the local build lock times out, clear only the stale lock before "
                        "retrying."
                    ),
                    evidence=(
                        EvidenceReference(
                            source=EvidenceSource.EVENT,
                            source_id=EVENT_6,
                            field_path="/payload/failure_summary",
                        ),
                    ),
                    confidence=1.0,
                ),
            )
        else:
            procedural = next(
                item
                for item in _prompt_bank(request)
                if item["kind"] == MemoryKind.PROCEDURAL.value
            )
            operations = (
                DeleteMemory(
                    operation="delete_memory",
                    memory_id=UUID(cast(str, procedural["memory_id"])),
                    expected_revision=cast(int, procedural["revision"]),
                ),
            )
        return BankOperationsProposal(
            schema_version="memory-edit-output/v1",
            operations=operations,
        )

    def _intervention_output(
        self,
        request: StructuredCallRequest,
    ) -> InterventionSelectionOutput | None:
        if self.condition is Stage2ConditionId.ALWAYS_INJECT and self.cycle_ordinal == 1:
            return None
        remind = self.cycle_ordinal == 2 or (
            self.condition is Stage2ConditionId.ALWAYS_INJECT and self.cycle_ordinal == 3
        )
        return InterventionSelectionOutput(
            schema_version="intervention-output/v1",
            action=(InterventionAction.REMIND if remind else InterventionAction.SILENCE),
            claims=(
                (
                    ProposedClaim(
                        kind=ClaimKind.REQUIREMENT,
                        evidence=_memory_reference(request, MemoryKind.KNOWLEDGE),
                    ),
                )
                if remind
                else ()
            ),
            confidence=1.0,
        )

    async def generate(self, request: StructuredCallRequest) -> StructuredCallResult:
        if request.phase is StructuredCallPhase.MEMORY_EDIT:
            self.cycle_ordinal += 1
            result = _completed_result(
                request,
                self._memory_output(request),
                live_evidence=self.live_evidence,
            )
        else:
            output = self._intervention_output(request)
            result = (
                _schema_invalid_result(request, live_evidence=self.live_evidence)
                if output is None
                else _completed_result(
                    request,
                    output,
                    live_evidence=self.live_evidence,
                )
            )
        self.pairs.append((request, result))
        return result


class _SchemaInvalidClient:
    def __init__(self) -> None:
        self.pairs: list[tuple[StructuredCallRequest, StructuredCallResult]] = []

    async def generate(self, request: StructuredCallRequest) -> StructuredCallResult:
        result = _schema_invalid_result(request)
        self.pairs.append((request, result))
        return result


def _fixture_digest(results: tuple[StructuredCallResult, ...]) -> str:
    material = tuple(
        {
            "schema_version": "two-phase-replay-record/v1",
            "record_type": "two_phase_replay_response",
            "replay_version": TWO_PHASE_REPLAY_VERSION,
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
        TWO_PHASE_REPLAY_VERSION,
        canonical_json(material),
        domain=_FIXTURE_DIGEST_DOMAIN,
    )


def _sealed_records(
    pairs: tuple[tuple[StructuredCallRequest, StructuredCallResult], ...],
) -> tuple[TwoPhaseReplayRecord, ...]:
    results = tuple(result for _request, result in pairs)
    fixture_digest = _fixture_digest(results)
    return tuple(
        TwoPhaseReplayRecord(
            replay_id=TWO_PHASE_REPLAY_VERSION,
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


async def _reviewed_run(
    condition: Stage2ConditionId,
) -> tuple[Stage2ExperimentRunResult, tuple[TwoPhaseReplayRecord, ...]]:
    trajectory = load_stage2_trajectory(
        TRAJECTORY_FIXTURE,
        expected_fixture_digest=TRAJECTORY_DIGEST,
    )
    trace_digest = algorithm_trace_digest(
        tuple(canonical_digest(item.draft) for item in trajectory.inputs)
    )
    client = _ReviewedClient(condition)
    result = await Stage2ExperimentRunner(
        repository=_repository(trace_digest),
        condition=condition,
        client=client,
    ).run(trajectory)
    return result, _sealed_records(tuple(client.pairs))


def _fixture_bytes(records: tuple[TwoPhaseReplayRecord, ...]) -> bytes:
    return b"".join(canonical_json(record) + b"\n" for record in records)


ACTIVE_CASES = (
    (
        Stage2ConditionId.FIXED_STEP,
        FIXED_RESPONSES,
        "aed71f320f03f4783fc25089f8c0a638323c2381e498b0349df2def6de5179ae",
        "02fb2c09309ce11f0f32ca1c50b1921a51b7d1e7b5d811292d9144eb1d670d21",
        "4097e78486eb70b11ca9b0626d245521f3300012a4c709cefc2c1107d3e217bb",
        6,
        1,
    ),
    (
        Stage2ConditionId.RETRIEVAL_ALWAYS,
        RETRIEVAL_RESPONSES,
        "50bce1255d7e313547ead4e73395b0f9b1e9a6e08c67f42b18f1c69552b24ca7",
        "42bf1ca1d1aba573d06d1eee5723941fcf0a211f0feaceab1d5eee3a8f110397",
        "2267f43967067f159398e626c926007dacfd601a6c141a5562cae94a18e0f63a",
        3,
        2,
    ),
    (
        Stage2ConditionId.ALWAYS_INJECT,
        ALWAYS_RESPONSES,
        "8b9c69c0bfd68c6d953cf8a002665f4029192088ffb6e8a06fa993136a55637e",
        "097cd6886137a25e959aaa622654c40939b95b307f08e41b4f0208ab315561c6",
        "6c24ffd44208e97af2112750630b761110b77d9471813963c30956fa5c195456",
        6,
        2,
    ),
)

ALL_CASES = (
    (
        Stage2ConditionId.NO_MEMORY,
        None,
        None,
        "2dd8ff174cec4d2c868126eda780136e4f43162cb969a42e2b190fb24bc9bcc5",
        "2dd8ff174cec4d2c868126eda780136e4f43162cb969a42e2b190fb24bc9bcc5",
        0,
        0,
    ),
    *ACTIVE_CASES,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "condition",
        "path",
        "fixture_digest",
        "reviewed_result_digest",
        "_replay_result_digest",
        "calls",
        "interventions",
    ),
    ACTIVE_CASES,
    ids=lambda value: value.value if isinstance(value, Stage2ConditionId) else None,
)
async def test_committed_response_fixtures_are_exact_reviewed_outputs(
    condition: Stage2ConditionId,
    path: Path,
    fixture_digest: str,
    reviewed_result_digest: str,
    _replay_result_digest: str,
    calls: int,
    interventions: int,
) -> None:
    reviewed, records = await _reviewed_run(condition)
    client = TwoPhaseReplayClient.from_path(
        path,
        expected_fixture_digest=fixture_digest,
    )

    assert path.read_bytes() == _fixture_bytes(records)
    assert client.total_responses == calls
    assert client.remaining_responses == calls
    assert records[0].fixture_digest == fixture_digest
    assert reviewed.response_fixture is None
    assert reviewed.result_digest == reviewed_result_digest
    assert reviewed.metrics.intervention_count == interventions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "condition",
        "path",
        "fixture_digest",
        "_reviewed_result_digest",
        "result_digest",
        "calls",
        "interventions",
    ),
    ALL_CASES,
    ids=lambda value: value.value if isinstance(value, Stage2ConditionId) else None,
)
async def test_every_condition_replays_twice_through_one_closed_path(
    condition: Stage2ConditionId,
    path: Path | None,
    fixture_digest: str | None,
    _reviewed_result_digest: str,
    result_digest: str,
    calls: int,
    interventions: int,
) -> None:
    result = await replay_stage2_fixture_twice(
        TRAJECTORY_FIXTURE,
        condition=condition,
        responses_path=path,
        expected_trajectory_fixture_digest=TRAJECTORY_DIGEST,
        expected_response_fixture_digest=fixture_digest,
    )

    assert result.result_digest == result_digest
    assert result.run_id == RUN_ID
    assert result.rebuild_equivalent
    assert len(result.decisions) == 11
    assert tuple(item.boundary_event.sequence for item in result.boundaries) == (1, 6, 11)
    assert result.metrics.model_call_count == calls
    assert result.metrics.intervention_count == interventions
    assert result.metrics.memory_mutation_count == (0 if calls == 0 else 4)
    assert result.semantic_projection_digests == result.repository_projection_digests
    assert result.ledger_head.entry_count == result.ledger_entry_count
    if calls:
        assert result.response_fixture is not None
        assert result.response_fixture.replay_id == TWO_PHASE_REPLAY_VERSION
        assert result.response_fixture.fixture_digest == fixture_digest
        assert result.response_fixture.response_count == calls
        assert result.metrics.provider_input_tokens is None
        assert result.metrics.provider_output_tokens is None
        assert result.metrics.canonical_token_equivalents is not None
        assert all(
            item.usage.provider_usage_provenance is ProviderUsageProvenance.UNAVAILABLE
            and item.usage.provider_input_tokens is None
            and item.usage.provider_output_tokens is None
            and item.usage.canonical_usage_provenance is CanonicalUsageProvenance.REPLAY_ATTESTED
            and item.usage.local_counter_id == "stage2-reviewed-utf8-counter/v1"
            and item.usage.local_counter_version == "utf8-bytes-ceil-div-4/v1"
            and item.usage.local_counter_configuration_digest == _COUNTER_CONFIGURATION_DIGEST
            and item.usage.local_counter_model_id == item.model_id
            for item in result.call_receipts
        )
        assert len(result.final_memory_snapshot.records) == 3
        active_records = tuple(
            item for item in result.final_memory_snapshot.records if item.validity.value == "active"
        )
        assert {item.kind for item in active_records} == {
            MemoryKind.KNOWLEDGE,
            MemoryKind.PRIVATE_STATUS,
        }
        invalid_records = tuple(
            item
            for item in result.final_memory_snapshot.records
            if item.validity.value == "invalidated"
        )
        assert len(invalid_records) == 1
        assert invalid_records[0].kind is MemoryKind.PROCEDURAL
    else:
        assert result.response_fixture is None
        assert result.metrics.provider_input_tokens == 0
        assert result.metrics.provider_output_tokens == 0
        assert result.metrics.canonical_token_equivalents == 0
        assert result.final_memory_snapshot.records == ()
    if condition is Stage2ConditionId.ALWAYS_INJECT:
        assert result.metrics.grounding_rejection_count == 1
        assert result.metrics.condition_violation_count == 1
    else:
        assert result.metrics.condition_violation_count == 0


@pytest.mark.asyncio
async def test_no_memory_never_constructs_a_replay_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> TwoPhaseReplayClient:
        raise AssertionError("no_memory attempted to construct a model client")

    monkeypatch.setattr(TwoPhaseReplayClient, "from_path", forbidden)
    result = await replay_stage2_fixture_twice(
        TRAJECTORY_FIXTURE,
        condition=Stage2ConditionId.NO_MEMORY,
        expected_trajectory_fixture_digest=TRAJECTORY_DIGEST,
    )

    assert result.call_receipts == ()
    assert result.response_fixture is None
    assert all(not item.invoke for item in result.decisions)
    with pytest.raises(Stage2ExperimentError):
        await replay_stage2_fixture_twice(
            TRAJECTORY_FIXTURE,
            condition=Stage2ConditionId.NO_MEMORY,
            responses_path=FIXED_RESPONSES,
        )
    with pytest.raises(Stage2ExperimentError):
        await replay_stage2_fixture_twice(
            TRAJECTORY_FIXTURE,
            condition=Stage2ConditionId.FIXED_STEP,
        )


@pytest.mark.asyncio
async def test_missing_delivery_route_preserves_known_call_cost() -> None:
    trajectory = load_stage2_trajectory(TRAJECTORY_FIXTURE)
    inputs = list(trajectory.inputs)
    inputs[5] = inputs[5].model_copy(update={"target_request_id": None})
    changed = build_stage2_trajectory(tuple(inputs))
    trace_digest = algorithm_trace_digest(
        tuple(canonical_digest(item.draft) for item in changed.inputs)
    )
    repository = _repository(trace_digest)
    client = _ReviewedClient(Stage2ConditionId.FIXED_STEP)
    runner = Stage2ExperimentRunner(
        repository=repository,
        condition=Stage2ConditionId.FIXED_STEP,
        client=client,
    )

    with pytest.raises(Stage2ExperimentError):
        await runner.run(changed)

    ledger = await repository.ledger(RUN_ID)
    cycles: dict[str, CycleRecord] = {}
    for entry in ledger:
        if type(entry.record) is CycleRecord:
            cycles[entry.record.cycle_id] = entry.record
    failed = max(cycles.values(), key=lambda item: item.last_event_sequence)
    assert failed.state is CycleState.FAILED
    assert failed.failure_reason is ReasonCode.TARGET_UNAVAILABLE
    assert failed.budget_settlement is not None
    assert failed.budget_settlement.model_calls == 2
    assert failed.model_call_digests == tuple(
        result.call_digest for _request, result in client.pairs[-2:]
    )


@pytest.mark.asyncio
async def test_tampered_trace_response_and_binding_fail_closed(tmp_path: Path) -> None:
    tampered_trace = tmp_path / "trace.jsonl"
    trace_bytes = TRAJECTORY_FIXTURE.read_bytes()
    assert b"offline" in trace_bytes
    tampered_trace.write_bytes(trace_bytes.replace(b"offline", b"online!", 1))
    with pytest.raises(Stage2ExperimentError):
        await replay_stage2_fixture_twice(
            tampered_trace,
            condition=Stage2ConditionId.NO_MEMORY,
        )

    tampered_responses = tmp_path / "responses.jsonl"
    response_bytes = FIXED_RESPONSES.read_bytes()
    assert b'"latency_us":1000' in response_bytes
    tampered_responses.write_bytes(
        response_bytes.replace(b'"latency_us":1000', b'"latency_us":1002', 1)
    )
    with pytest.raises(Stage2ExperimentError):
        await replay_stage2_fixture_twice(
            TRAJECTORY_FIXTURE,
            condition=Stage2ConditionId.FIXED_STEP,
            responses_path=tampered_responses,
        )

    trajectory = load_stage2_trajectory(TRAJECTORY_FIXTURE)
    inputs = list(trajectory.inputs)
    source = inputs[3]
    logical_messages = list(source.logical_messages)
    first = logical_messages[0]
    second = logical_messages[1]
    logical_messages[0] = first.model_copy(update={"selector": second.selector})
    logical_messages[1] = second.model_copy(update={"selector": first.selector})
    inputs[3] = source.model_copy(update={"logical_messages": tuple(logical_messages)})
    changed = build_stage2_trajectory(tuple(inputs))
    tampered_binding = tmp_path / "binding.jsonl"
    tampered_binding.write_bytes(changed.canonical_bytes)
    with pytest.raises(Stage2ExperimentError):
        await replay_stage2_fixture_twice(
            tampered_binding,
            condition=Stage2ConditionId.FIXED_STEP,
            responses_path=FIXED_RESPONSES,
        )


@pytest.mark.asyncio
async def test_result_rejects_duplicate_boundary_evidence() -> None:
    result = await replay_stage2_fixture_twice(
        TRAJECTORY_FIXTURE,
        condition=Stage2ConditionId.NO_MEMORY,
        expected_trajectory_fixture_digest=TRAJECTORY_DIGEST,
    )
    values = result.model_dump(mode="python", exclude={"result_digest"})
    values["boundaries"] = (
        result.boundaries[0],
        result.boundaries[0],
        result.boundaries[2],
    )

    with pytest.raises(ValidationError, match="do not reconcile"):
        Stage2ExperimentRunResult.model_validate(values)

    tampered_digest = result.model_dump(mode="python")
    tampered_digest["result_digest"] = "f" * 64
    with pytest.raises(ValidationError, match="result digest does not match"):
        Stage2ExperimentRunResult.model_validate(tampered_digest)


@pytest.mark.asyncio
async def test_result_rejects_resealed_response_fixture_identity_tampering() -> None:
    result = await replay_stage2_fixture_twice(
        TRAJECTORY_FIXTURE,
        condition=Stage2ConditionId.FIXED_STEP,
        responses_path=FIXED_RESPONSES,
        expected_trajectory_fixture_digest=TRAJECTORY_DIGEST,
    )
    assert result.response_fixture is not None
    for changed in ({"fixture_digest": "f" * 64}, {"response_count": 5}):
        values = result.model_dump(mode="python", exclude={"result_digest"})
        fixture = values["response_fixture"]
        assert isinstance(fixture, dict)
        fixture.update(changed)
        with pytest.raises(ValidationError, match="sources failed exact validation"):
            Stage2ExperimentRunResult.model_validate(values)

    no_memory = await replay_stage2_fixture_twice(
        TRAJECTORY_FIXTURE,
        condition=Stage2ConditionId.NO_MEMORY,
        expected_trajectory_fixture_digest=TRAJECTORY_DIGEST,
    )
    no_memory_values = no_memory.model_dump(mode="python", exclude={"result_digest"})
    no_memory_values["response_fixture"] = {
        "replay_id": TWO_PHASE_REPLAY_VERSION,
        "fixture_digest": "f" * 64,
        "response_count": 1,
    }
    with pytest.raises(ValidationError, match="sources failed exact validation"):
        Stage2ExperimentRunResult.model_validate(no_memory_values)


@pytest.mark.asyncio
async def test_result_rejects_live_receipts_resealed_as_a_replay_fixture() -> None:
    trajectory = load_stage2_trajectory(
        TRAJECTORY_FIXTURE,
        expected_fixture_digest=TRAJECTORY_DIGEST,
    )
    trace_digest = algorithm_trace_digest(
        tuple(canonical_digest(item.draft) for item in trajectory.inputs)
    )
    result = await Stage2ExperimentRunner(
        repository=_repository(trace_digest),
        condition=Stage2ConditionId.FIXED_STEP,
        client=_ReviewedClient(Stage2ConditionId.FIXED_STEP, live_evidence=True),
    ).run(trajectory)
    assert result.response_fixture is None
    assert all(
        receipt.completion_digest is not None
        and receipt.completion_digest.algorithm is PayloadDigestAlgorithm.HMAC_SHA256
        and receipt.usage.provider_usage_provenance is ProviderUsageProvenance.PROVIDER_REPORTED
        for receipt in result.call_receipts
    )

    values = result.model_dump(mode="python", exclude={"result_digest"})
    values["response_fixture"] = {
        "replay_id": TWO_PHASE_REPLAY_VERSION,
        "fixture_digest": two_phase_replay_fixture_digest_from_receipts(
            result.call_receipts,
            replay_id=TWO_PHASE_REPLAY_VERSION,
        ),
        "response_count": len(result.call_receipts),
    }

    with pytest.raises(ValidationError, match="sources failed exact validation"):
        Stage2ExperimentRunResult.model_validate(values)


@pytest.mark.parametrize(
    "changed",
    (
        {"provider_output_tokens": None},
        {"canonical_output_tokens": None},
        {"canonical_token_equivalents": None},
        {"canonical_token_equivalents": 3},
    ),
)
def test_metrics_reject_partial_or_inconsistent_token_totals(
    changed: dict[str, int | None],
) -> None:
    values: dict[str, int | None] = {
        "model_call_count": 1,
        "provider_input_tokens": 1,
        "provider_output_tokens": 1,
        "canonical_input_tokens": 1,
        "canonical_output_tokens": 1,
        "canonical_token_equivalents": 2,
        "memory_call_latency_us": 1,
        "intervention_count": 0,
        "grounding_rejection_count": 0,
        "provenance_validated_boundary_count": 0,
        "memory_mutation_count": 0,
        "condition_violation_count": 0,
    }
    values.update(changed)

    with pytest.raises(ValidationError, match="token totals are inconsistent"):
        Stage2ExperimentMetrics.model_validate(values)


def test_runner_rejects_condition_client_mismatches() -> None:
    repository = _repository("0" * 64)

    with pytest.raises(Stage2ExperimentError):
        Stage2ExperimentRunner(
            repository=repository,
            condition=Stage2ConditionId.FIXED_STEP,
            client=None,
        )
    with pytest.raises(Stage2ExperimentError):
        Stage2ExperimentRunner(
            repository=repository,
            condition=Stage2ConditionId.NO_MEMORY,
            client=_ReviewedClient(Stage2ConditionId.FIXED_STEP),
        )
    with pytest.raises(Stage2ExperimentError):
        Stage2ExperimentRunner(
            repository=repository,
            condition=Stage2ConditionId.FIXED_STEP,
            client=cast(StructuredCallClient, object()),
        )


def test_runner_failure_mapping_and_identifier_factories_are_closed() -> None:
    assert (
        runner_module._failure_reason(TwoPhaseFailureReason.MODEL_TIMEOUT)
        is ReasonCode.MODEL_TIMEOUT
    )
    assert (
        runner_module._failure_reason(TwoPhaseFailureReason.MODEL_ERROR) is ReasonCode.MODEL_ERROR
    )
    assert (
        runner_module._failure_reason(TwoPhaseFailureReason.SCHEMA_INVALID)
        is ReasonCode.INVALID_STRUCTURED_OUTPUT
    )
    repository_ids = runner_module._repository_id_factory("1" * 64)
    assert repository_ids() != repository_ids()
    delivery_ids = runner_module._delivery_id_factory("2" * 64, UUID(int=7))
    assert delivery_ids() != delivery_ids()
    with pytest.raises(ValueError):
        runner_module._ledger_record_key(object())


@pytest.mark.asyncio
async def test_schema_invalid_call_is_terminalized_with_its_known_cost() -> None:
    trajectory = load_stage2_trajectory(
        TRAJECTORY_FIXTURE,
        expected_fixture_digest=TRAJECTORY_DIGEST,
    )
    trace_digest = algorithm_trace_digest(
        tuple(canonical_digest(item.draft) for item in trajectory.inputs)
    )
    repository = _repository(trace_digest)
    client = _SchemaInvalidClient()
    runner = Stage2ExperimentRunner(
        repository=repository,
        condition=Stage2ConditionId.FIXED_STEP,
        client=client,
    )

    with pytest.raises(Stage2ExperimentError):
        await runner.run(trajectory)

    ledger = await repository.ledger(RUN_ID)
    cycles: dict[str, CycleRecord] = {}
    for entry in ledger:
        if type(entry.record) is CycleRecord:
            cycles[entry.record.cycle_id] = entry.record
    failed = max(cycles.values(), key=lambda item: item.revision)
    assert failed.state is CycleState.FAILED
    assert failed.failure_reason is ReasonCode.INVALID_STRUCTURED_OUTPUT
    assert failed.budget_settlement is not None
    assert failed.budget_settlement.model_calls == 1
    assert failed.model_call_digests == (client.pairs[0][1].call_digest,)


@pytest.mark.asyncio
async def test_projector_requires_exact_selector_attestation_xor_model_provenance() -> None:
    retrieval, _records = await _reviewed_run(Stage2ConditionId.RETRIEVAL_ALWAYS)
    fixed, _records = await _reviewed_run(Stage2ConditionId.FIXED_STEP)

    def first_committed(
        result: Stage2ExperimentRunResult,
    ) -> tuple[Projection, LedgerEntry, CycleRecord]:
        projection = empty_projection(result.run_id)
        for ledger_entry in result.ledger:
            record = ledger_entry.record
            if type(record) is CycleRecord and record.state is CycleState.COMMITTED:
                return projection, ledger_entry, record
            projection = apply_entry(projection, ledger_entry)
        raise AssertionError("reviewed run lacks a committed cycle")

    retrieval_projection, retrieval_entry, retrieval_cycle = first_committed(retrieval)
    assert retrieval_cycle.selector_provenance is not None
    selector = DeterministicSelectorProvenance.model_validate_json(
        canonical_json(retrieval_cycle.selector_provenance)
    )
    apply_entry(retrieval_projection, retrieval_entry)

    omitted = retrieval_cycle.model_copy(update={"selector_provenance": None})
    with pytest.raises(ProjectionInvariantError, match="authoritative verification"):
        apply_entry(
            retrieval_projection,
            retrieval_entry.model_copy(update={"record": omitted}),
        )

    other_selector = DeterministicSelectorProvenance(
        selector_id=selector.selector_id,
        configuration_digest=selector.configuration_digest,
        request_digest=selector.request_digest,
        result_digest="f" * 64,
    )
    changed = retrieval_cycle.model_copy(
        update={"selector_provenance": other_selector.model_dump(mode="json", warnings=False)}
    )
    with pytest.raises(ProjectionInvariantError, match="authoritative verification"):
        apply_entry(
            retrieval_projection,
            retrieval_entry.model_copy(update={"record": changed}),
        )

    fixed_projection, fixed_entry, fixed_cycle = first_committed(fixed)
    assert fixed_cycle.selector_provenance is None
    fixed_boundary = fixed.boundaries[0]
    assert fixed_boundary.two_phase_result is not None
    with pytest.raises(Stage2ExperimentError):
        runner_module._pending_delivery(
            fixed_cycle,
            fixed_boundary.two_phase_result.intervention,
            enqueue_delivery_binding(
                target_request_id="stage2-invalid-silence-delivery/1",
                capabilities=runner_module._OfflineDeliveryAdapter().capabilities(),
            ),
            updated_at=fixed_cycle.updated_at,
        )
    injected = fixed_cycle.model_copy(
        update={"selector_provenance": selector.model_dump(mode="json", warnings=False)}
    )
    with pytest.raises(ProjectionInvariantError, match="authoritative verification"):
        apply_entry(
            fixed_projection,
            fixed_entry.model_copy(update={"record": injected}),
        )
