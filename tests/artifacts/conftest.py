from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from uuid import UUID

import pytest
from tests.runtime.test_engine import _run_frozen_replay

from saliencegate.artifacts.algorithm_manifest import (
    AlgorithmCheckpointAttestation,
    AlgorithmEndpointClassification,
    AlgorithmExecutionAttestation,
    AlgorithmHardwareAttestation,
    AlgorithmResponseFixtureAttestation,
    AlgorithmSamplingAttestation,
    AlgorithmSamplingMode,
    AlgorithmTokenizerAttestation,
    AlgorithmTokenizerStatus,
    AlgorithmWarmupPolicy,
)
from saliencegate.artifacts.manifest import RevisionEvidence, RevisionSource
from saliencegate.domain import (
    PayloadDigest,
    PayloadDigestAlgorithm,
    canonical_digest,
    canonical_json,
)
from saliencegate.experiments import (
    Stage2ConditionId,
    Stage2ExperimentRunner,
    Stage2ExperimentRunResult,
    load_stage2_trajectory,
    replay_stage2_fixture_twice,
)
from saliencegate.models.replay_two_phase import TwoPhaseReplayClient, TwoPhaseReplayRecord
from saliencegate.ports.model_calls import (
    COMPLETION_DIGEST_SCOPE,
    CanonicalUsageProvenance,
    ProviderUsageProvenance,
    StructuredCallRequest,
    StructuredCallResult,
    StructuredCallUsage,
    validated_result_for_request,
)
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.runtime.algorithm_result import algorithm_runtime_uuid, algorithm_trace_digest
from saliencegate.runtime.engine import ReplayRunResult
from saliencegate.security.keys import InstallationKey

ROOT = Path(__file__).resolve().parents[2]
TRAJECTORY_FIXTURE = ROOT / "tests/fixtures/runs/paper_two_phase_basic.jsonl"
FIXED_RESPONSES = ROOT / "tests/fixtures/models/paper_two_phase_fixed_step_responses.jsonl"
RETRIEVAL_RESPONSES = ROOT / "tests/fixtures/models/paper_two_phase_retrieval_responses.jsonl"
ALWAYS_RESPONSES = ROOT / "tests/fixtures/models/paper_two_phase_always_inject_responses.jsonl"
TRAJECTORY_DIGEST = "751489f55ac9d5ea56408ca6f5036b55e895be0fa130c36f19e624ee094d1266"
FIXED_RESPONSE_DIGEST = "aed71f320f03f4783fc25089f8c0a638323c2381e498b0349df2def6de5179ae"
RETRIEVAL_RESPONSE_DIGEST = "50bce1255d7e313547ead4e73395b0f9b1e9a6e08c67f42b18f1c69552b24ca7"
ALWAYS_RESPONSE_DIGEST = "8b9c69c0bfd68c6d953cf8a002665f4029192088ffb6e8a06fa993136a55637e"
_LIVE_COMPLETION_HMAC_DOMAIN = f"saliencegate:model:{COMPLETION_DIGEST_SCOPE}".encode("ascii")
_LIVE_INSTALLATION_KEY = InstallationKey(b"l" * 32)


class _ReplayDelegatingStructuredCallClient:
    """Replay delegate whose concrete type deliberately hides fixture identity."""

    def __init__(self) -> None:
        self._delegate = TwoPhaseReplayClient.from_path(
            FIXED_RESPONSES,
            expected_fixture_digest=FIXED_RESPONSE_DIGEST,
        )

    async def generate(self, request: StructuredCallRequest) -> StructuredCallResult:
        return await self._delegate.generate(request)


class _DeterministicLiveStructuredCallClient:
    """Rebuild live-native responses from reviewed fixture output content.

    The fixture supplies only the deterministic structured outputs and expected request
    sequence. Every returned result is newly sealed with live HMAC completion evidence,
    provider-reported usage, and the attested local tokenizer counter.
    """

    def __init__(self) -> None:
        self._source_results = tuple(
            TwoPhaseReplayRecord.model_validate_json(line).result
            for line in FIXED_RESPONSES.read_bytes().splitlines()
        )
        self._next_result = 0

    @property
    def remaining_responses(self) -> int:
        return len(self._source_results) - self._next_result

    async def generate(self, request: StructuredCallRequest) -> StructuredCallResult:
        if self._next_result >= len(self._source_results):
            raise AssertionError("deterministic live response sequence exhausted")
        source = self._source_results[self._next_result]
        if (
            source.request_digest != request.request_digest
            or source.model_call_index != request.model_call_index
            or source.phase is not request.phase
            or source.attempt != request.attempt
            or source.response_schema_version != request.response_schema_version
            or source.output is None
        ):
            raise AssertionError("deterministic live request sequence diverged")

        completion = canonical_json(source.output)
        canonical_input_tokens = (len(canonical_json(request.payload)) + 3) // 4
        canonical_output_tokens = (len(completion) + 3) // 4
        usage = StructuredCallUsage(
            schema_version="structured-call-usage/v1",
            provider_input_tokens=canonical_input_tokens,
            provider_output_tokens=canonical_output_tokens,
            provider_usage_provenance=ProviderUsageProvenance.PROVIDER_REPORTED,
            latency_us=1_000 + request.model_call_index,
            canonical_input_tokens=canonical_input_tokens,
            canonical_output_tokens=canonical_output_tokens,
            canonical_usage_provenance=CanonicalUsageProvenance.LOCAL_COUNTER,
            local_counter_id="stage2-reviewed-utf8-counter/v1",
            local_counter_version="utf8-bytes-ceil-div-4/v1",
            local_counter_configuration_digest="4" * 64,
            local_counter_model_id=request.model_id,
        )
        result = StructuredCallResult(
            schema_version="structured-call-result/v1",
            request_digest=request.request_digest,
            model_call_index=request.model_call_index,
            phase=request.phase,
            attempt=request.attempt,
            response_schema_version=request.response_schema_version,
            status=source.status,
            parse_status=source.parse_status,
            output=source.output,
            completion_digest=PayloadDigest(
                algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
                value=_LIVE_INSTALLATION_KEY._hmac_sha256(
                    completion,
                    domain=_LIVE_COMPLETION_HMAC_DOMAIN,
                ),
            ),
            completion_byte_count=len(completion),
            usage=usage,
        )
        checked_result = validated_result_for_request(request, result)
        self._next_result += 1
        return checked_result


def _repository_id_factory(trace_digest: str) -> Callable[[], UUID]:
    ordinal = 0

    def next_identifier() -> UUID:
        nonlocal ordinal
        ordinal += 1
        return algorithm_runtime_uuid(trace_digest, "stage2-repository", ordinal)

    return next_identifier


@pytest.fixture
async def replay_result() -> ReplayRunResult:
    return await _run_frozen_replay()


@pytest.fixture(scope="session")
async def fixed_stage2_result() -> Stage2ExperimentRunResult:
    return await replay_stage2_fixture_twice(
        TRAJECTORY_FIXTURE,
        condition=Stage2ConditionId.FIXED_STEP,
        responses_path=FIXED_RESPONSES,
        expected_trajectory_fixture_digest=TRAJECTORY_DIGEST,
        expected_response_fixture_digest=FIXED_RESPONSE_DIGEST,
    )


@pytest.fixture(scope="session")
async def replay_wrapped_stage2_result() -> Stage2ExperimentRunResult:
    trajectory = load_stage2_trajectory(
        TRAJECTORY_FIXTURE,
        expected_fixture_digest=TRAJECTORY_DIGEST,
    )
    trace_digest = algorithm_trace_digest(
        tuple(canonical_digest(item.draft) for item in trajectory.inputs)
    )
    runner = Stage2ExperimentRunner(
        repository=MemoryRunRepository(
            synthetic_benchmark=True,
            id_factory=_repository_id_factory(trace_digest),
        ),
        condition=Stage2ConditionId.FIXED_STEP,
        client=_ReplayDelegatingStructuredCallClient(),
    )
    result = await runner.run(trajectory)
    assert result.response_fixture is None
    return result


@pytest.fixture(scope="session")
async def live_stage2_result() -> Stage2ExperimentRunResult:
    trajectory = load_stage2_trajectory(
        TRAJECTORY_FIXTURE,
        expected_fixture_digest=TRAJECTORY_DIGEST,
    )
    trace_digest = algorithm_trace_digest(
        tuple(canonical_digest(item.draft) for item in trajectory.inputs)
    )
    client = _DeterministicLiveStructuredCallClient()
    runner = Stage2ExperimentRunner(
        repository=MemoryRunRepository(
            synthetic_benchmark=True,
            id_factory=_repository_id_factory(trace_digest),
        ),
        condition=Stage2ConditionId.FIXED_STEP,
        client=client,
    )

    result = await runner.run(trajectory)

    assert result.response_fixture is None
    assert client.remaining_responses == 0
    assert all(
        call.completion_digest is not None
        and call.completion_digest.algorithm is PayloadDigestAlgorithm.HMAC_SHA256
        and call.usage.provider_usage_provenance is ProviderUsageProvenance.PROVIDER_REPORTED
        and call.usage.canonical_usage_provenance is CanonicalUsageProvenance.LOCAL_COUNTER
        for call in result.call_receipts
    )
    return result


@pytest.fixture(scope="session")
async def no_memory_stage2_result() -> Stage2ExperimentRunResult:
    return await replay_stage2_fixture_twice(
        TRAJECTORY_FIXTURE,
        condition=Stage2ConditionId.NO_MEMORY,
        expected_trajectory_fixture_digest=TRAJECTORY_DIGEST,
    )


@pytest.fixture(scope="session")
async def retrieval_stage2_result() -> Stage2ExperimentRunResult:
    return await replay_stage2_fixture_twice(
        TRAJECTORY_FIXTURE,
        condition=Stage2ConditionId.RETRIEVAL_ALWAYS,
        responses_path=RETRIEVAL_RESPONSES,
        expected_trajectory_fixture_digest=TRAJECTORY_DIGEST,
        expected_response_fixture_digest=RETRIEVAL_RESPONSE_DIGEST,
    )


@pytest.fixture(scope="session")
async def always_stage2_result() -> Stage2ExperimentRunResult:
    return await replay_stage2_fixture_twice(
        TRAJECTORY_FIXTURE,
        condition=Stage2ConditionId.ALWAYS_INJECT,
        responses_path=ALWAYS_RESPONSES,
        expected_trajectory_fixture_digest=TRAJECTORY_DIGEST,
        expected_response_fixture_digest=ALWAYS_RESPONSE_DIGEST,
    )


@pytest.fixture
def algorithm_execution() -> AlgorithmExecutionAttestation:
    return AlgorithmExecutionAttestation.create(
        endpoint_classification=AlgorithmEndpointClassification.OFFLINE_REPLAY,
        runtime_id="saliencegate-two-phase-replay",
        runtime_version="1.0.0",
        checkpoint=AlgorithmCheckpointAttestation(
            model_id="gpt-oss:20b",
            model_tag="gpt-oss:20b-fixture/v1",
            checkpoint_digest=None,
            quantization="not-applicable-replay",
        ),
        sampling=AlgorithmSamplingAttestation(
            mode=AlgorithmSamplingMode.FROZEN_REPLAY,
            temperature=None,
            seed=None,
            reasoning_effort=None,
        ),
        tokenizer=AlgorithmTokenizerAttestation(
            status=AlgorithmTokenizerStatus.ATTESTED,
            tokenizer_id="stage2-reviewed-utf8-counter/v1",
            tokenizer_version="utf8-bytes-ceil-div-4/v1",
            configuration_digest="4" * 64,
            model_id="gpt-oss:20b",
        ),
        hardware=AlgorithmHardwareAttestation(
            model="synthetic-test-host",
            architecture="test-arch",
            logical_core_count=8,
            memory_capacity_bytes=16 * 1024**3,
            operating_system="test-os",
            operating_system_version="1.0",
        ),
        warmup_policy=AlgorithmWarmupPolicy.NOT_APPLICABLE,
        response_fixture=AlgorithmResponseFixtureAttestation(
            replay_id="two-phase-replay/v1",
            fixture_digest=FIXED_RESPONSE_DIGEST,
            response_count=6,
            consumed_count=6,
        ),
    )


@pytest.fixture
def clean_revision() -> RevisionEvidence:
    return RevisionEvidence(
        source=RevisionSource.GIT,
        package_version="0.1.0",
        commit="a" * 40,
        dirty_worktree=False,
    )
