from __future__ import annotations

import math
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Literal, Self, TypeAlias, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import to_jsonable_python

from saliencegate.artifacts.manifest import (
    ArtifactClassification,
    ArtifactEvidenceLevel,
    DeliveryAttestation,
    RevisionEvidence,
)
from saliencegate.domain import (
    BudgetAmounts,
    BudgetLimits,
    BudgetSnapshot,
    CycleState,
    EventPhase,
    EventType,
    EvidenceReference,
    EvidenceSource,
    InterventionAction,
    InterventionOutcome,
    InvocationDecision,
    PayloadDigest,
    PayloadDigestAlgorithm,
    ReasonCode,
    TrustLabel,
    canonical_digest,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.domain.records import (
    UUID4,
    ComponentIdentifier,
    Sha256Digest,
    UtcDatetime,
)
from saliencegate.experiments.conditions import (
    ResolvedStage2Condition,
    Stage2ConditionId,
    Stage2ConditionObservation,
    resolve_stage2_condition,
)
from saliencegate.experiments.runner import (
    STAGE2_EXPERIMENT_POLICY_VERSION,
    Stage2ExperimentMetrics,
    Stage2ExperimentRunResult,
)
from saliencegate.intervention import DeterministicSelectorProvenance
from saliencegate.models.replay_two_phase import two_phase_receipts_are_replay_native
from saliencegate.ports.model_calls import (
    CanonicalUsageProvenance,
    ProviderUsageProvenance,
)
from saliencegate.ports.repository import LedgerHead, ProjectionDigests
from saliencegate.ports.trajectory import TrajectoryBinding
from saliencegate.ports.two_phase import (
    CallReceipt,
    TwoPhaseCallPolicy,
    TwoPhaseModelProfile,
)
from saliencegate.prompts.contracts import PromptBundleIdentity
from saliencegate.prompts.paper_two_phase_v1 import (
    PAPER_TWO_PHASE_FORCED_REMINDER_V1,
    PAPER_TWO_PHASE_V1,
)
from saliencegate.runtime.algorithm_result import algorithm_trace_digest
from saliencegate.runtime.message_window import (
    MAX_MESSAGE_WINDOW_CANONICAL_BYTES,
    MAX_MESSAGE_WINDOW_ITEMS,
    MESSAGE_WINDOW_VERSION,
    TrajectoryTextSource,
)
from saliencegate.runtime.scheduling import FixedStepSchedule

ALGORITHM_ARTIFACT_SCHEMA_VERSION: Literal["algorithm-artifact/v1"] = "algorithm-artifact/v1"
AlgorithmArtifactClassification: TypeAlias = Literal[
    ArtifactClassification.SYNTHETIC_DIGEST_ONLY,
    ArtifactClassification.SYNTHETIC_RAW,
]
MAX_ALGORITHM_COMPONENT_BYTES = 128 * 1024 * 1024
MAX_ALGORITHM_RECORDS = 100_000
MAX_ALGORITHM_LEDGER_ENTRIES = 160_000
MAX_HARDWARE_TEXT_UTF8_BYTES = 256
MAX_SIGNED_64 = (1 << 63) - 1

_COMPONENT_DIGEST_DOMAIN = "saliencegate:algorithm-artifact:component:v1"
_CONTENT_SET_DIGEST_DOMAIN = "saliencegate:algorithm-artifact:content-set:v1"
_MANIFEST_DIGEST_DOMAIN = "saliencegate:algorithm-artifact:manifest:v1"
_CONFIGURATION_DIGEST_DOMAIN = "saliencegate:algorithm-artifact:configuration:v1"
_CHECKPOINT_DIGEST_DOMAIN = "saliencegate:algorithm-artifact:checkpoint:v1"
_SAMPLING_DIGEST_DOMAIN = "saliencegate:algorithm-artifact:sampling:v1"
_TOKENIZER_DIGEST_DOMAIN = "saliencegate:algorithm-artifact:tokenizer:v1"
_HARDWARE_DIGEST_DOMAIN = "saliencegate:algorithm-artifact:hardware:v1"
_EXECUTION_DIGEST_DOMAIN = "saliencegate:algorithm-artifact:execution:v1"
_WINDOW_SET_DIGEST_DOMAIN = "saliencegate:algorithm-artifact:window-set:v1"
_STAGE2_FIXTURE_DIGEST_DOMAIN = "saliencegate:experiment:stage2-trajectory-fixture:v1"
_PREFIX_REQUEST_DIGEST_DOMAIN = "saliencegate:trajectory:prefix-request:v1"
_PREFIX_DIGEST_DOMAIN = "saliencegate:trajectory:attested-prefix:v1"
_RUN_COMPONENT_DIGEST_DOMAIN = "saliencegate:algorithm-artifact:run:v1"
_TRAJECTORY_COMPONENT_DIGEST_DOMAIN = "saliencegate:algorithm-artifact:trajectory:v1"
_CALLS_COMPONENT_DIGEST_DOMAIN = "saliencegate:algorithm-artifact:calls:v1"
_DECISIONS_COMPONENT_DIGEST_DOMAIN = "saliencegate:algorithm-artifact:decisions:v1"
_CYCLE_ATTESTATION_DIGEST_DOMAIN = "saliencegate:algorithm-artifact:cycle:v1"
_CYCLES_COMPONENT_DIGEST_DOMAIN = "saliencegate:algorithm-artifact:cycles:v1"
_DELIVERIES_COMPONENT_DIGEST_DOMAIN = "saliencegate:algorithm-artifact:deliveries:v1"
_OUTCOMES_COMPONENT_DIGEST_DOMAIN = "saliencegate:algorithm-artifact:outcomes:v1"
_METRICS_COMPONENT_DIGEST_DOMAIN = "saliencegate:algorithm-artifact:metrics:v1"
_BOUNDARY_ATTESTATION_DIGEST_DOMAIN = "saliencegate:algorithm-artifact:boundary:v1"
_ATTESTATIONS_COMPONENT_DIGEST_DOMAIN = "saliencegate:algorithm-artifact:attestations:v1"

_HARDWARE_FORBIDDEN = re.compile(r"[@\\/\x00-\x1f\x7f]|(?:^[a-z][a-z0-9+.-]*://)", re.I)
_ModelT = TypeVar("_ModelT", bound=BaseModel)
_STAGE2_MODEL_ID = "gpt-oss:20b"
_STAGE2_MODEL_PROFILE_ID = "stage2-openai-compatible-replay/v1"
_STAGE2_MAX_PROVIDER_INPUT_TOKENS = 262_144
_STAGE2_MAX_PROVIDER_OUTPUT_TOKENS = 65_536
_STAGE2_MAX_CALL_LATENCY_US = 600_000_000
_SILENCE_REASON_CODES = frozenset(
    {
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
    }
)


class AlgorithmCycleMode(StrEnum):
    DISABLED = "disabled"
    TWO_PHASE = "two_phase"
    PHASE_ONE_RETRIEVAL = "phase_one_retrieval"


class AlgorithmExecutionMode(StrEnum):
    FROZEN_REPLAY = "frozen_replay"
    OPENAI_COMPATIBLE = "openai_compatible"


def algorithm_call_evidence_matches_execution_mode(
    execution_mode: AlgorithmExecutionMode,
    calls: tuple[CallReceipt, ...],
) -> bool:
    """Return whether every receipt carries evidence native to its execution mode."""

    if (
        type(execution_mode) is not AlgorithmExecutionMode
        or type(calls) is not tuple
        or any(type(call) is not CallReceipt for call in calls)
    ):
        return False
    if execution_mode is AlgorithmExecutionMode.FROZEN_REPLAY:
        return not calls or two_phase_receipts_are_replay_native(calls)
    if execution_mode is not AlgorithmExecutionMode.OPENAI_COMPATIBLE:
        return False
    for call in calls:
        usage = call.usage
        completion = call.completion_digest
        if (
            usage.provider_usage_provenance is ProviderUsageProvenance.REPLAY_ATTESTED
            or usage.canonical_usage_provenance is CanonicalUsageProvenance.REPLAY_ATTESTED
            or (
                completion is not None
                and completion.algorithm is not PayloadDigestAlgorithm.HMAC_SHA256
            )
        ):
            return False
    return True


class AlgorithmEndpointClassification(StrEnum):
    OFFLINE_REPLAY = "offline_replay"
    LOOPBACK_OPENAI_COMPATIBLE = "loopback_openai_compatible"
    REMOTE_OPENAI_COMPATIBLE = "remote_openai_compatible"


class AlgorithmSamplingMode(StrEnum):
    FROZEN_REPLAY = "frozen_replay"
    OPENAI_COMPATIBLE = "openai_compatible"


class AlgorithmTokenizerStatus(StrEnum):
    ATTESTED = "attested"
    UNAVAILABLE = "unavailable"


class AlgorithmWarmupPolicy(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    COLD = "cold"
    WARM = "warm"
    MIXED = "mixed"


class AlgorithmArtifactComponentName(StrEnum):
    ATTESTATIONS = "attestations"
    CALLS = "calls"
    CYCLES = "cycles"
    DECISIONS = "decisions"
    DELIVERIES = "deliveries"
    METRICS = "metrics"
    OUTCOMES = "outcomes"
    RUN = "run"
    TRAJECTORY = "trajectory"


_COMPONENT_PATHS: dict[AlgorithmArtifactComponentName, str] = {
    name: f"{name.value}.json" for name in AlgorithmArtifactComponentName
}


class _AlgorithmModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def _digest_without(values: Mapping[str, object], field: str, *, domain: str) -> str:
    material = {key: value for key, value in values.items() if key != field}
    return length_prefixed_sha256(
        canonical_json(to_jsonable_python(material)),
        domain=domain,
    )


def _hardware_text(value: str) -> str:
    if (
        value != value.strip()
        or _HARDWARE_FORBIDDEN.search(value) is not None
        or len(value.encode("utf-8", errors="strict")) > MAX_HARDWARE_TEXT_UTF8_BYTES
    ):
        raise ValueError("hardware text is not a bounded benchmark identifier")
    return value


class AlgorithmCheckpointAttestation(_AlgorithmModel):
    schema_version: Literal["algorithm-checkpoint/v1"] = "algorithm-checkpoint/v1"
    model_id: ComponentIdentifier
    model_tag: ComponentIdentifier | None
    checkpoint_digest: Sha256Digest | None
    quantization: ComponentIdentifier
    checkpoint_attestation_digest: Sha256Digest | None = None

    @model_validator(mode="after")
    def one_checkpoint_identity_and_digest(self) -> Self:
        if (self.model_tag is None) == (self.checkpoint_digest is None):
            raise ValueError("exactly one checkpoint or model tag is required")
        values = self.model_dump(
            mode="json",
            exclude={"checkpoint_attestation_digest"},
            warnings=False,
        )
        expected = _digest_without(
            values,
            "checkpoint_attestation_digest",
            domain=_CHECKPOINT_DIGEST_DOMAIN,
        )
        if self.checkpoint_attestation_digest is not None and (
            self.checkpoint_attestation_digest != expected
        ):
            raise ValueError("checkpoint attestation digest does not match")
        object.__setattr__(self, "checkpoint_attestation_digest", expected)
        return self


class AlgorithmSamplingAttestation(_AlgorithmModel):
    schema_version: Literal["algorithm-sampling/v1"] = "algorithm-sampling/v1"
    mode: AlgorithmSamplingMode
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] | None
    seed: Annotated[int, Field(ge=-(1 << 63), le=MAX_SIGNED_64)] | None
    reasoning_effort: ComponentIdentifier | None
    stream: Literal[False] = False
    sampling_digest: Sha256Digest | None = None

    @model_validator(mode="after")
    def mode_is_explicit_and_digest_matches(self) -> Self:
        if self.temperature is not None and not math.isfinite(self.temperature):
            raise ValueError("sampling temperature must be finite")
        if self.mode is AlgorithmSamplingMode.FROZEN_REPLAY:
            if any(
                value is not None for value in (self.temperature, self.seed, self.reasoning_effort)
            ):
                raise ValueError("frozen replay cannot claim provider sampling controls")
        elif self.temperature != 0.0 or self.seed is None or self.reasoning_effort is None:
            raise ValueError("OpenAI-compatible sampling controls are incomplete")
        values = self.model_dump(mode="json", exclude={"sampling_digest"}, warnings=False)
        expected = _digest_without(values, "sampling_digest", domain=_SAMPLING_DIGEST_DOMAIN)
        if self.sampling_digest is not None and self.sampling_digest != expected:
            raise ValueError("sampling attestation digest does not match")
        object.__setattr__(self, "sampling_digest", expected)
        return self


class AlgorithmTokenizerAttestation(_AlgorithmModel):
    schema_version: Literal["algorithm-tokenizer/v1"] = "algorithm-tokenizer/v1"
    status: AlgorithmTokenizerStatus
    tokenizer_id: ComponentIdentifier | None
    tokenizer_version: ComponentIdentifier | None
    configuration_digest: Sha256Digest | None
    model_id: ComponentIdentifier | None
    tokenizer_digest: Sha256Digest | None = None

    @model_validator(mode="after")
    def status_has_exact_identity_and_digest(self) -> Self:
        identity = (
            self.tokenizer_id,
            self.tokenizer_version,
            self.configuration_digest,
            self.model_id,
        )
        if self.status is AlgorithmTokenizerStatus.ATTESTED:
            if any(value is None for value in identity):
                raise ValueError("attested tokenizer identity is incomplete")
        elif any(value is not None for value in identity):
            raise ValueError("unavailable tokenizer cannot carry an identity")
        values = self.model_dump(mode="json", exclude={"tokenizer_digest"}, warnings=False)
        expected = _digest_without(values, "tokenizer_digest", domain=_TOKENIZER_DIGEST_DOMAIN)
        if self.tokenizer_digest is not None and self.tokenizer_digest != expected:
            raise ValueError("tokenizer attestation digest does not match")
        object.__setattr__(self, "tokenizer_digest", expected)
        return self


def algorithm_call_evidence_matches_tokenizer(
    tokenizer: AlgorithmTokenizerAttestation,
    calls: tuple[CallReceipt, ...],
) -> bool:
    """Return whether every canonical count is bound to the attested tokenizer."""

    if tokenizer.status is AlgorithmTokenizerStatus.UNAVAILABLE:
        return not any(
            call.usage.canonical_input_tokens is not None
            or call.usage.canonical_output_tokens is not None
            for call in calls
        )
    identity = (
        tokenizer.tokenizer_id,
        tokenizer.tokenizer_version,
        tokenizer.configuration_digest,
        tokenizer.model_id,
    )
    return all(
        (call.usage.canonical_input_tokens is None and call.usage.canonical_output_tokens is None)
        or identity
        == (
            call.usage.local_counter_id,
            call.usage.local_counter_version,
            call.usage.local_counter_configuration_digest,
            call.usage.local_counter_model_id,
        )
        for call in calls
    )


class AlgorithmHardwareAttestation(_AlgorithmModel):
    """Producer-attested coarse hardware metadata that callers must de-identify.

    Dedicated identity fields and obvious emails, paths, or URLs are rejected. Free-form
    benchmark labels cannot prove anonymity, so callers must generalize host-specific values.
    """

    schema_version: Literal["algorithm-hardware/v1"] = "algorithm-hardware/v1"
    model: Annotated[str, Field(min_length=1, max_length=256)]
    architecture: Annotated[str, Field(min_length=1, max_length=256)]
    logical_core_count: Annotated[int, Field(ge=1, le=65_536)]
    memory_capacity_bytes: Annotated[int, Field(ge=1, le=MAX_SIGNED_64)]
    operating_system: Annotated[str, Field(min_length=1, max_length=256)]
    operating_system_version: Annotated[str, Field(min_length=1, max_length=256)]
    hardware_digest: Sha256Digest | None = None

    @field_validator(
        "model",
        "architecture",
        "operating_system",
        "operating_system_version",
    )
    @classmethod
    def benchmark_text_only(cls, value: str) -> str:
        return _hardware_text(value)

    @model_validator(mode="after")
    def digest_matches(self) -> Self:
        values = self.model_dump(mode="json", exclude={"hardware_digest"}, warnings=False)
        expected = _digest_without(values, "hardware_digest", domain=_HARDWARE_DIGEST_DOMAIN)
        if self.hardware_digest is not None and self.hardware_digest != expected:
            raise ValueError("hardware attestation digest does not match")
        object.__setattr__(self, "hardware_digest", expected)
        return self


class AlgorithmResponseFixtureAttestation(_AlgorithmModel):
    schema_version: Literal["algorithm-response-fixture/v1"] = "algorithm-response-fixture/v1"
    replay_id: ComponentIdentifier
    fixture_digest: Sha256Digest
    response_count: Annotated[int, Field(ge=1, le=MAX_ALGORITHM_RECORDS)]
    consumed_count: Annotated[int, Field(ge=1, le=MAX_ALGORITHM_RECORDS)]

    @model_validator(mode="after")
    def fixture_is_fully_consumed(self) -> Self:
        if self.response_count != self.consumed_count:
            raise ValueError("response fixture must be fully consumed")
        return self


class AlgorithmExecutionAttestation(_AlgorithmModel):
    schema_version: Literal["algorithm-execution/v1"] = "algorithm-execution/v1"
    execution_mode: AlgorithmExecutionMode
    endpoint_classification: AlgorithmEndpointClassification
    runtime_id: ComponentIdentifier
    runtime_version: ComponentIdentifier
    checkpoint: AlgorithmCheckpointAttestation
    sampling: AlgorithmSamplingAttestation
    tokenizer: AlgorithmTokenizerAttestation
    hardware: AlgorithmHardwareAttestation
    warmup_policy: AlgorithmWarmupPolicy
    response_fixture: AlgorithmResponseFixtureAttestation | None = None
    execution_digest: Sha256Digest

    @model_validator(mode="after")
    def execution_mode_and_digest_match(self) -> Self:
        if self.execution_mode is AlgorithmExecutionMode.FROZEN_REPLAY:
            if (
                self.endpoint_classification is not AlgorithmEndpointClassification.OFFLINE_REPLAY
                or self.sampling.mode is not AlgorithmSamplingMode.FROZEN_REPLAY
                or self.warmup_policy is not AlgorithmWarmupPolicy.NOT_APPLICABLE
            ):
                raise ValueError("frozen replay execution metadata is inconsistent")
        elif (
            self.endpoint_classification is AlgorithmEndpointClassification.OFFLINE_REPLAY
            or self.sampling.mode is not AlgorithmSamplingMode.OPENAI_COMPATIBLE
            or self.warmup_policy is AlgorithmWarmupPolicy.NOT_APPLICABLE
            or self.response_fixture is not None
        ):
            raise ValueError("live execution metadata is inconsistent")
        if (
            self.tokenizer.status is AlgorithmTokenizerStatus.ATTESTED
            and self.tokenizer.model_id != self.checkpoint.model_id
        ):
            raise ValueError("tokenizer and checkpoint model identities differ")
        values = self.model_dump(mode="json", exclude={"execution_digest"}, warnings=False)
        expected = _digest_without(values, "execution_digest", domain=_EXECUTION_DIGEST_DOMAIN)
        if self.execution_digest != expected:
            raise ValueError("execution attestation digest does not match")
        return self

    @classmethod
    def create(
        cls,
        *,
        endpoint_classification: AlgorithmEndpointClassification,
        runtime_id: ComponentIdentifier,
        runtime_version: ComponentIdentifier,
        checkpoint: AlgorithmCheckpointAttestation,
        sampling: AlgorithmSamplingAttestation,
        tokenizer: AlgorithmTokenizerAttestation,
        hardware: AlgorithmHardwareAttestation,
        warmup_policy: AlgorithmWarmupPolicy,
        execution_mode: AlgorithmExecutionMode = AlgorithmExecutionMode.FROZEN_REPLAY,
        response_fixture: AlgorithmResponseFixtureAttestation | None = None,
    ) -> AlgorithmExecutionAttestation:
        values: dict[str, object] = {
            "schema_version": "algorithm-execution/v1",
            "execution_mode": execution_mode,
            "endpoint_classification": endpoint_classification,
            "runtime_id": runtime_id,
            "runtime_version": runtime_version,
            "checkpoint": checkpoint,
            "sampling": sampling,
            "tokenizer": tokenizer,
            "hardware": hardware,
            "warmup_policy": warmup_policy,
            "response_fixture": response_fixture,
        }
        values["execution_digest"] = _digest_without(
            values,
            "execution_digest",
            domain=_EXECUTION_DIGEST_DOMAIN,
        )
        return cls.model_validate(values)


class AlgorithmArtifactComponent(_AlgorithmModel):
    schema_version: Literal["algorithm-component/v1"] = "algorithm-component/v1"
    name: AlgorithmArtifactComponentName
    path: Annotated[str, Field(min_length=1, max_length=64)]
    byte_count: Annotated[int, Field(ge=2, le=MAX_ALGORITHM_COMPONENT_BYTES)]
    record_count: Annotated[int, Field(ge=0, le=MAX_ALGORITHM_RECORDS)]
    content_digest: Sha256Digest

    @model_validator(mode="after")
    def path_is_fixed(self) -> Self:
        if self.path != _COMPONENT_PATHS[self.name]:
            raise ValueError("algorithm component path is not the fixed v1 path")
        return self


class AlgorithmArtifactCounters(_AlgorithmModel):
    schema_version: Literal["algorithm-counters/v1"] = "algorithm-counters/v1"
    events: Annotated[int, Field(ge=1, le=MAX_ALGORITHM_RECORDS)]
    scheduled_invocations: Annotated[int, Field(ge=1, le=MAX_ALGORITHM_RECORDS)]
    decisions: Annotated[int, Field(ge=1, le=MAX_ALGORITHM_RECORDS)]
    cycles: Annotated[int, Field(ge=0, le=MAX_ALGORITHM_RECORDS)]
    requests: Annotated[int, Field(ge=0, le=MAX_ALGORITHM_RECORDS)]
    model_calls: Annotated[int, Field(ge=0, le=MAX_ALGORITHM_RECORDS)]
    deliveries: Annotated[int, Field(ge=0, le=MAX_ALGORITHM_RECORDS)]
    outcomes: Annotated[int, Field(ge=0, le=MAX_ALGORITHM_RECORDS)]
    ledger_entries: Annotated[int, Field(ge=1, le=MAX_ALGORITHM_LEDGER_ENTRIES)]

    @model_validator(mode="after")
    def totals_are_possible(self) -> Self:
        if (
            self.decisions != self.events
            or self.scheduled_invocations > self.events
            or self.cycles != self.requests
            or self.cycles > self.scheduled_invocations
            or self.outcomes != self.cycles
            or self.deliveries > self.cycles
            or self.model_calls < self.requests
        ):
            raise ValueError("algorithm artifact counters are inconsistent")
        return self


def _component_model_digest(
    model: BaseModel,
    field: str,
    *,
    domain: str,
) -> str:
    return _digest_without(
        model.model_dump(mode="json", exclude={field}, warnings=False),
        field,
        domain=domain,
    )


def _redacted_fixture_digest(
    fixture_id: str,
    records: tuple[AlgorithmTrajectoryRecordAttestation, ...],
) -> str:
    material = tuple(
        {
            "schema_version": "stage2-trajectory-record/v1",
            "record_type": "stage2_trajectory_input",
            "trajectory_version": "stage2-trajectory/v1",
            "ordinal": item.ordinal,
            "input_digest": item.input_digest,
        }
        for item in records
    )
    return length_prefixed_sha256(
        fixture_id,
        canonical_json(material),
        domain=_STAGE2_FIXTURE_DIGEST_DOMAIN,
    )


def _redacted_prefix_request_digest(
    run_id: UUID4,
    records: tuple[AlgorithmTrajectoryRecordAttestation, ...],
) -> str:
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": "trajectory-prefix-request/v1",
                "run_id": str(run_id),
                "boundary_event_sequence": len(records),
                "binding_digests": [item.binding.binding_digest for item in records],
            }
        ),
        domain=_PREFIX_REQUEST_DIGEST_DOMAIN,
    )


def _redacted_prefix_digest(
    run_id: UUID4,
    records: tuple[AlgorithmTrajectoryRecordAttestation, ...],
    request_digest: str,
) -> str:
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": "attested-trajectory-prefix/v1",
                "run_id": str(run_id),
                "boundary_event_sequence": len(records),
                "request_digest": request_digest,
                "event_attestations": [
                    {
                        "event_id": str(item.event_id),
                        "event_sequence": item.event_sequence,
                        "binding_digest": item.binding.binding_digest,
                    }
                    for item in records
                ],
            }
        ),
        domain=_PREFIX_DIGEST_DOMAIN,
    )


def _redacted_window_source_digests(
    records: tuple[AlgorithmTrajectoryRecordAttestation, ...],
) -> tuple[str, ...]:
    selected: list[str] = []
    for item in records:
        for message in item.binding.logical_messages:
            source = TrajectoryTextSource(
                evidence=EvidenceReference(
                    source=EvidenceSource.EVENT,
                    source_id=item.event_id,
                    field_path=message.selector.field_path,
                    span=message.selector.span,
                ),
                event_sequence=item.event_sequence,
                ledger_position=item.binding.ledger_position,
                trust_label=item.trust_label,
                payload_digest=item.payload_digest,
                record_tag=item.binding.record_tag,
                chain_tag=item.binding.chain_tag,
                binding_digest=item.binding.binding_digest,
            )
            selected.append(canonical_digest(source))
    return tuple(selected[-MAX_MESSAGE_WINDOW_ITEMS:])


def _redacted_stage2_trajectory_is_exact(
    run_id: UUID4,
    records: tuple[AlgorithmTrajectoryRecordAttestation, ...],
) -> bool:
    seen_event_ids: set[UUID4] = set()
    previous_timestamp: UtcDatetime | None = None
    previous_step: int | None = None
    logical_message_count = 0
    saw_action_step = False
    for index, item in enumerate(records):
        if (
            item.trust_label is not TrustLabel.SYNTHETIC_FIXTURE
            or (index == 0) != (item.event_type is EventType.RUN_START)
            or (index == 0) != (item.binding.task_description is not None)
            or item.parent_ids != tuple(sorted(item.parent_ids, key=str))
            or len(set(item.parent_ids)) != len(item.parent_ids)
            or any(parent_id not in seen_event_ids for parent_id in item.parent_ids)
            or (item.action_step_ordinal is None) != (item.binding.action_step is None)
            or (previous_timestamp is not None and item.event_timestamp < previous_timestamp)
        ):
            return False
        logical_message_count += len(item.binding.logical_messages)
        if item.action_step_ordinal is not None:
            if previous_step is not None and item.action_step_ordinal < previous_step:
                return False
            previous_step = item.action_step_ordinal
            saw_action_step = True
        seen_event_ids.add(item.event_id)
        previous_timestamp = item.event_timestamp
    return run_id not in seen_event_ids and logical_message_count > 0 and saw_action_step


def _expected_cycle_reservation(policy: TwoPhaseCallPolicy) -> BudgetAmounts:
    return BudgetAmounts(
        model_calls=policy.max_model_calls,
        input_tokens=policy.max_provider_input_tokens,
        output_tokens=policy.max_provider_output_tokens,
        canonical_token_equivalents=(
            policy.max_provider_input_tokens + policy.max_provider_output_tokens
        ),
        latency_us=policy.max_total_latency_us,
        interventions=1,
        schema_repairs=policy.max_schema_repairs,
    )


def _expected_stage2_prompt_bundle(
    condition_id: Stage2ConditionId,
) -> PromptBundleIdentity:
    bundle = (
        PAPER_TWO_PHASE_FORCED_REMINDER_V1
        if condition_id is Stage2ConditionId.ALWAYS_INJECT
        else PAPER_TWO_PHASE_V1
    )
    return bundle.identity


def _expected_stage2_model_profile(
    condition_id: Stage2ConditionId,
) -> TwoPhaseModelProfile:
    bundle = _expected_stage2_prompt_bundle(condition_id)
    return TwoPhaseModelProfile(
        schema_version="two-phase-model-profile/v1",
        profile_id=_STAGE2_MODEL_PROFILE_ID,
        model_id=_STAGE2_MODEL_ID,
        prompt_bundle_id=bundle.bundle_id,
        prompt_bundle_digest=bundle.bundle_digest,
    )


def _expected_stage2_call_policy() -> TwoPhaseCallPolicy:
    return TwoPhaseCallPolicy(
        schema_version="two-phase-call-policy/v1",
        max_model_calls=2,
        max_schema_repairs=0,
        client_retries=0,
        max_provider_input_tokens=_STAGE2_MAX_PROVIDER_INPUT_TOKENS,
        max_provider_output_tokens=_STAGE2_MAX_PROVIDER_OUTPUT_TOKENS,
        max_total_latency_us=_STAGE2_MAX_CALL_LATENCY_US * 2,
        max_call_latency_us=_STAGE2_MAX_CALL_LATENCY_US,
    )


def _sum_budget_amounts(values: tuple[BudgetAmounts, ...]) -> BudgetAmounts:
    names = (
        "model_calls",
        "input_tokens",
        "output_tokens",
        "canonical_token_equivalents",
        "latency_us",
        "interventions",
        "schema_repairs",
    )
    return BudgetAmounts(**{name: sum(getattr(value, name) for value in values) for name in names})


class AlgorithmRunComponent(_AlgorithmModel):
    schema_version: Literal["algorithm-run/v1"] = "algorithm-run/v1"
    run_id: UUID4
    condition: ResolvedStage2Condition
    policy_version: ComponentIdentifier
    cycle_mode: AlgorithmCycleMode
    prompt_bundle: PromptBundleIdentity
    model_profile: TwoPhaseModelProfile
    call_policy: TwoPhaseCallPolicy
    cycle_reservation: BudgetAmounts
    budget_limits: BudgetLimits
    execution: AlgorithmExecutionAttestation
    source_result_digest: Sha256Digest
    run_component_digest: Sha256Digest

    @model_validator(mode="after")
    def configuration_is_closed(self) -> Self:
        expected_condition = resolve_stage2_condition(self.condition.condition_id)
        if (
            self.condition != expected_condition
            or self.policy_version != STAGE2_EXPERIMENT_POLICY_VERSION
            or self.cycle_mode
            is not algorithm_cycle_mode_for_condition(self.condition.condition_id)
            or self.prompt_bundle != _expected_stage2_prompt_bundle(self.condition.condition_id)
            or self.model_profile != _expected_stage2_model_profile(self.condition.condition_id)
            or self.call_policy != _expected_stage2_call_policy()
            or self.model_profile.model_id != self.execution.checkpoint.model_id
            or self.cycle_reservation != _expected_cycle_reservation(self.call_policy)
        ):
            raise ValueError("algorithm run configuration is inconsistent")
        if self.run_component_digest != _component_model_digest(
            self,
            "run_component_digest",
            domain=_RUN_COMPONENT_DIGEST_DOMAIN,
        ):
            raise ValueError("algorithm run component digest does not match")
        return self


class AlgorithmTrajectoryRecordAttestation(_AlgorithmModel):
    ordinal: Annotated[int, Field(ge=1, le=MAX_ALGORITHM_RECORDS)]
    fixture_record_digest: Sha256Digest
    input_digest: Sha256Digest
    event_id: UUID4
    event_sequence: Annotated[int, Field(ge=1, le=MAX_ALGORITHM_RECORDS)]
    event_timestamp: UtcDatetime
    event_type: EventType
    event_phase: EventPhase
    parent_ids: tuple[UUID4, ...]
    trust_label: TrustLabel
    payload_digest: PayloadDigest
    normalized_draft_digest: Sha256Digest
    persisted_event_draft_digest: Sha256Digest
    event_digest: Sha256Digest
    binding: TrajectoryBinding
    action_step_ordinal: Annotated[int, Field(ge=1, le=MAX_SIGNED_64)] | None

    @model_validator(mode="after")
    def binding_names_the_redacted_record(self) -> Self:
        if (
            self.ordinal != self.event_sequence
            or self.binding.event_id != self.event_id
            or self.binding.event_sequence != self.event_sequence
            or self.binding.payload_digest != self.payload_digest
        ):
            raise ValueError("trajectory record attestation is inconsistent")
        return self


class AlgorithmWindowAttestation(_AlgorithmModel):
    invocation_ordinal: Annotated[int, Field(ge=1, le=MAX_ALGORITHM_RECORDS)]
    version: ComponentIdentifier
    boundary_event_id: UUID4
    boundary_event_sequence: Annotated[int, Field(ge=1, le=MAX_ALGORITHM_RECORDS)]
    boundary_ledger_position: Annotated[int, Field(ge=1, le=MAX_SIGNED_64)]
    boundary_chain_tag: PayloadDigest
    trajectory_prefix_digest: Sha256Digest
    task_digest: Sha256Digest
    message_count: Annotated[int, Field(ge=0, le=8)]
    payload_canonical_utf8_bytes: Annotated[
        int,
        Field(ge=0, le=MAX_MESSAGE_WINDOW_CANONICAL_BYTES),
    ]
    source_attestation_digests: tuple[Sha256Digest, ...]
    window_digest: Sha256Digest

    @model_validator(mode="after")
    def message_sources_are_exact(self) -> Self:
        if self.message_count != len(self.source_attestation_digests) or len(
            set(self.source_attestation_digests)
        ) != len(self.source_attestation_digests):
            raise ValueError("window source attestations are inconsistent")
        return self


class AlgorithmTrajectoryComponent(_AlgorithmModel):
    schema_version: Literal["algorithm-trajectory/v1"] = "algorithm-trajectory/v1"
    run_id: UUID4
    fixture_id: ComponentIdentifier
    fixture_digest: Sha256Digest
    trace_digest: Sha256Digest
    trajectory_prefix_request_digest: Sha256Digest
    trajectory_prefix_digest: Sha256Digest
    records: Annotated[
        tuple[AlgorithmTrajectoryRecordAttestation, ...],
        Field(min_length=1, max_length=MAX_ALGORITHM_RECORDS),
    ]
    schedule: FixedStepSchedule
    windows: Annotated[
        tuple[AlgorithmWindowAttestation, ...],
        Field(min_length=1, max_length=MAX_ALGORITHM_RECORDS),
    ]
    window_set_digest: Sha256Digest
    trajectory_component_digest: Sha256Digest

    @model_validator(mode="after")
    def records_schedule_and_windows_are_exact(self) -> Self:
        ordinals = tuple(item.ordinal for item in self.records)
        event_ids = tuple(item.event_id for item in self.records)
        invoked = tuple(item for item in self.schedule.decisions if item.invoke)
        expected_request_digest = _redacted_prefix_request_digest(self.run_id, self.records)
        window_provenance_is_exact = True
        for window in self.windows:
            prefix = self.records[: window.boundary_event_sequence]
            prefix_request_digest = _redacted_prefix_request_digest(self.run_id, prefix)
            source_digests = _redacted_window_source_digests(prefix)
            if (
                window.version != MESSAGE_WINDOW_VERSION
                or window.trajectory_prefix_digest
                != _redacted_prefix_digest(self.run_id, prefix, prefix_request_digest)
                or window.source_attestation_digests != source_digests
                or window.message_count != len(source_digests)
            ):
                window_provenance_is_exact = False
                break
        if (
            ordinals != tuple(range(1, len(self.records) + 1))
            or len(set(event_ids)) != len(event_ids)
            or not _redacted_stage2_trajectory_is_exact(self.run_id, self.records)
            or any(item.binding.run_id != self.run_id for item in self.records)
            or self.fixture_digest != _redacted_fixture_digest(self.fixture_id, self.records)
            or any(
                item.normalized_draft_digest != item.persisted_event_draft_digest
                for item in self.records
            )
            or self.trace_digest
            != algorithm_trace_digest(tuple(item.normalized_draft_digest for item in self.records))
            or self.trajectory_prefix_request_digest != expected_request_digest
            or self.trajectory_prefix_digest
            != _redacted_prefix_digest(self.run_id, self.records, expected_request_digest)
            or self.schedule.run_id != self.run_id
            or self.schedule.boundary_event_sequence != len(self.records)
            or self.schedule.trajectory_prefix_digest != self.trajectory_prefix_digest
            or tuple(item.event_id for item in self.schedule.decisions) != event_ids
            or tuple(item.event_sequence for item in self.schedule.decisions) != ordinals
            or tuple(item.action_step_ordinal for item in self.schedule.decisions)
            != tuple(item.action_step_ordinal for item in self.records)
            or len(invoked) != len(self.windows)
            or tuple(item.invocation_ordinal for item in self.windows)
            != tuple(range(1, len(self.windows) + 1))
            or tuple(item.boundary_event_id for item in self.windows)
            != tuple(item.event_id for item in invoked)
            or tuple(item.boundary_event_sequence for item in self.windows)
            != tuple(item.event_sequence for item in invoked)
            or len({item.task_digest for item in self.windows}) != 1
            or not window_provenance_is_exact
            or self.window_set_digest
            != algorithm_window_set_digest(tuple(item.window_digest for item in self.windows))
        ):
            raise ValueError("algorithm trajectory cardinality is inconsistent")
        if self.trajectory_component_digest != _component_model_digest(
            self,
            "trajectory_component_digest",
            domain=_TRAJECTORY_COMPONENT_DIGEST_DOMAIN,
        ):
            raise ValueError("algorithm trajectory component digest does not match")
        return self


class AlgorithmCallGroup(_AlgorithmModel):
    invocation_ordinal: Annotated[int, Field(ge=1, le=MAX_ALGORITHM_RECORDS)]
    boundary_event_sequence: Annotated[int, Field(ge=1, le=MAX_ALGORITHM_RECORDS)]
    cycle_id: Sha256Digest
    cycle_request_digest: Sha256Digest
    call_receipt_digests: Annotated[
        tuple[Sha256Digest, ...],
        Field(min_length=1, max_length=2),
    ]
    grounding_call_index: Annotated[int, Field(ge=0, le=1)] | None
    grounding_state_digest: Sha256Digest | None

    @model_validator(mode="after")
    def grounding_fields_are_paired(self) -> Self:
        if (self.grounding_call_index is None) != (self.grounding_state_digest is None):
            raise ValueError("call group grounding fields are incomplete")
        return self


class AlgorithmCallsComponent(_AlgorithmModel):
    schema_version: Literal["algorithm-calls/v1"] = "algorithm-calls/v1"
    run_id: UUID4
    ordered_request_digests: tuple[Sha256Digest, ...]
    groups: tuple[AlgorithmCallGroup, ...]
    calls: tuple[CallReceipt, ...]
    calls_component_digest: Sha256Digest

    @model_validator(mode="after")
    def requests_and_receipts_are_ordered(self) -> Self:
        receipt_digests = tuple(item.receipt_digest for item in self.calls)
        grouped_receipts = tuple(
            digest for group in self.groups for digest in group.call_receipt_digests
        )
        group_ordinals = tuple(group.invocation_ordinal for group in self.groups)
        if (
            self.ordered_request_digests != tuple(item.request_digest for item in self.calls)
            or len(set(self.ordered_request_digests)) != len(self.ordered_request_digests)
            or len({item.call_digest for item in self.calls}) != len(self.calls)
            or len(set(receipt_digests)) != len(receipt_digests)
            or grouped_receipts != receipt_digests
            or group_ordinals != tuple(range(1, len(self.groups) + 1))
            or any(item.run_id != self.run_id for item in self.calls)
        ):
            raise ValueError("algorithm calls are not one canonical ordered set")
        offset = 0
        for group in self.groups:
            grouped = self.calls[offset : offset + len(group.call_receipt_digests)]
            offset += len(grouped)
            if (
                not grouped
                or any(item.cycle_id != group.cycle_id for item in grouped)
                or tuple(item.model_call_index for item in grouped) != tuple(range(len(grouped)))
                or any(item.attempt != 0 for item in grouped)
            ):
                raise ValueError("algorithm call group ordering is inconsistent")
            grounding = tuple(item for item in grouped if item.grounding_state_digest is not None)
            if grounding:
                if (
                    len(grounding) != 1
                    or group.grounding_call_index != grounding[0].model_call_index
                    or group.grounding_state_digest != grounding[0].grounding_state_digest
                ):
                    raise ValueError("algorithm call grounding binding is inconsistent")
            elif group.grounding_call_index is not None:
                raise ValueError("algorithm call group has an unexpected grounding binding")
        if offset != len(self.calls):
            raise ValueError("algorithm call groups do not cover every receipt")
        if self.calls_component_digest != _component_model_digest(
            self,
            "calls_component_digest",
            domain=_CALLS_COMPONENT_DIGEST_DOMAIN,
        ):
            raise ValueError("algorithm calls component digest does not match")
        return self


class AlgorithmDecisionsComponent(_AlgorithmModel):
    schema_version: Literal["algorithm-decisions/v1"] = "algorithm-decisions/v1"
    run_id: UUID4
    decisions: Annotated[
        tuple[InvocationDecision, ...],
        Field(min_length=1, max_length=MAX_ALGORITHM_RECORDS),
    ]
    decisions_component_digest: Sha256Digest

    @model_validator(mode="after")
    def decisions_are_one_ordered_run(self) -> Self:
        if (
            any(item.run_id != self.run_id for item in self.decisions)
            or tuple(item.event_sequence for item in self.decisions)
            != tuple(range(1, len(self.decisions) + 1))
            or len({item.decision_id for item in self.decisions}) != len(self.decisions)
        ):
            raise ValueError("algorithm decisions are not one ordered run")
        if self.decisions_component_digest != _component_model_digest(
            self,
            "decisions_component_digest",
            domain=_DECISIONS_COMPONENT_DIGEST_DOMAIN,
        ):
            raise ValueError("algorithm decisions component digest does not match")
        return self


class AlgorithmCycleAttestation(_AlgorithmModel):
    invocation_ordinal: Annotated[int, Field(ge=1, le=MAX_ALGORITHM_RECORDS)]
    cycle_id: Sha256Digest
    run_id: UUID4
    revision: Annotated[int, Field(ge=1, le=MAX_SIGNED_64)]
    invocation_decision_id: UUID4
    boundary_event_sequence: Annotated[int, Field(ge=1, le=MAX_ALGORITHM_RECORDS)]
    window_digest: Sha256Digest
    policy_version: ComponentIdentifier
    configuration_digest: Sha256Digest
    grounding_version: ComponentIdentifier
    grounding_configuration_digest: Sha256Digest
    state: CycleState
    source_cycle_digest: Sha256Digest
    cycle_request_digest: Sha256Digest
    execution_result_digest: Sha256Digest
    observation_digest: Sha256Digest
    budget_reservation: BudgetAmounts
    budget_settlement: BudgetAmounts
    batch_digest: Sha256Digest
    model_call_digests: Annotated[tuple[Sha256Digest, ...], Field(min_length=1, max_length=2)]
    model_call_latencies_us: Annotated[
        tuple[Annotated[int, Field(ge=0, le=MAX_SIGNED_64)], ...],
        Field(min_length=1, max_length=2),
    ]
    call_receipt_digests: Annotated[tuple[Sha256Digest, ...], Field(min_length=1, max_length=2)]
    memory_create_count: Annotated[int, Field(ge=0)]
    memory_update_count: Annotated[int, Field(ge=0)]
    memory_invalidation_count: Annotated[int, Field(ge=0)]
    private_status_replaced: bool
    intervention_id: UUID4
    intervention_action: InterventionAction
    intervention_digest: Sha256Digest
    grounding_receipt_digest: Sha256Digest
    grounding_model_call_index: Annotated[int, Field(ge=0, le=1)] | None
    grounding_model_call_digest: Sha256Digest | None
    selector_provenance: DeterministicSelectorProvenance | None
    rendered_text_digest: Sha256Digest | None
    reason_code: ReasonCode
    delivery_source_digest: Sha256Digest | None
    boundary_evidence_digest: Sha256Digest
    cycle_attestation_digest: Sha256Digest

    @model_validator(mode="after")
    def terminal_cycle_evidence_is_complete(self) -> Self:
        budget_fields = (
            "model_calls",
            "input_tokens",
            "output_tokens",
            "canonical_token_equivalents",
            "latency_us",
            "interventions",
            "schema_repairs",
        )
        if (
            self.state is not CycleState.COMMITTED
            or self.batch_digest != self.window_digest
            or len(self.model_call_digests) != len(self.model_call_latencies_us)
            or len(self.model_call_digests) != len(self.call_receipt_digests)
            or self.budget_settlement.model_calls != len(self.model_call_digests)
            or self.budget_settlement.latency_us != sum(self.model_call_latencies_us)
            or any(
                getattr(self.budget_settlement, field) > getattr(self.budget_reservation, field)
                for field in budget_fields
            )
            or (
                self.intervention_action is InterventionAction.SILENCE
                and (
                    self.rendered_text_digest is not None or self.delivery_source_digest is not None
                )
            )
            or (
                self.intervention_action is InterventionAction.REMIND
                and (self.rendered_text_digest is None or self.delivery_source_digest is None)
            )
            or (
                self.intervention_action is InterventionAction.REMIND
                and self.reason_code is not ReasonCode.GROUNDED_REMINDER
            )
            or (
                self.intervention_action is InterventionAction.SILENCE
                and self.reason_code not in _SILENCE_REASON_CODES
            )
            or (self.grounding_model_call_index is None)
            != (self.grounding_model_call_digest is None)
            or (
                (self.grounding_model_call_index is not None)
                == (self.selector_provenance is not None)
            )
            or (
                self.grounding_model_call_index is not None
                and (
                    self.grounding_model_call_index != len(self.model_call_digests) - 1
                    or self.grounding_model_call_digest
                    != self.model_call_digests[self.grounding_model_call_index]
                )
            )
        ):
            raise ValueError("algorithm cycle terminal evidence is inconsistent")
        if self.cycle_attestation_digest != _component_model_digest(
            self,
            "cycle_attestation_digest",
            domain=_CYCLE_ATTESTATION_DIGEST_DOMAIN,
        ):
            raise ValueError("algorithm cycle attestation digest does not match")
        return self


class AlgorithmCyclesComponent(_AlgorithmModel):
    schema_version: Literal["algorithm-cycles/v1"] = "algorithm-cycles/v1"
    run_id: UUID4
    cycles: tuple[AlgorithmCycleAttestation, ...]
    cycles_component_digest: Sha256Digest

    @model_validator(mode="after")
    def cycles_are_unique_and_ordered(self) -> Self:
        if (
            tuple(item.invocation_ordinal for item in self.cycles)
            != tuple(range(1, len(self.cycles) + 1))
            or len({item.cycle_id for item in self.cycles}) != len(self.cycles)
            or any(item.run_id != self.run_id for item in self.cycles)
        ):
            raise ValueError("algorithm cycles are not one ordered run")
        if self.cycles_component_digest != _component_model_digest(
            self,
            "cycles_component_digest",
            domain=_CYCLES_COMPONENT_DIGEST_DOMAIN,
        ):
            raise ValueError("algorithm cycles component digest does not match")
        return self


class AlgorithmDeliveriesComponent(_AlgorithmModel):
    schema_version: Literal["algorithm-deliveries/v1"] = "algorithm-deliveries/v1"
    run_id: UUID4
    deliveries: tuple[DeliveryAttestation, ...]
    deliveries_component_digest: Sha256Digest

    @model_validator(mode="after")
    def deliveries_are_unique_and_ordered(self) -> Self:
        if tuple(item.event_sequence for item in self.deliveries) != tuple(
            sorted(item.event_sequence for item in self.deliveries)
        ) or len({item.delivery_id for item in self.deliveries}) != len(self.deliveries):
            raise ValueError("algorithm deliveries are not unique and ordered")
        if self.deliveries_component_digest != _component_model_digest(
            self,
            "deliveries_component_digest",
            domain=_DELIVERIES_COMPONENT_DIGEST_DOMAIN,
        ):
            raise ValueError("algorithm deliveries component digest does not match")
        return self


class AlgorithmOutcomesComponent(_AlgorithmModel):
    schema_version: Literal["algorithm-outcomes/v1"] = "algorithm-outcomes/v1"
    run_id: UUID4
    outcomes: tuple[InterventionOutcome, ...]
    outcomes_component_digest: Sha256Digest

    @model_validator(mode="after")
    def outcomes_are_one_unique_run(self) -> Self:
        if any(item.run_id != self.run_id for item in self.outcomes) or len(
            {item.outcome_id for item in self.outcomes}
        ) != len(self.outcomes):
            raise ValueError("algorithm outcomes are not one unique run")
        if self.outcomes_component_digest != _component_model_digest(
            self,
            "outcomes_component_digest",
            domain=_OUTCOMES_COMPONENT_DIGEST_DOMAIN,
        ):
            raise ValueError("algorithm outcomes component digest does not match")
        return self


class AlgorithmFinalMemoryAttestation(_AlgorithmModel):
    run_id: UUID4
    ledger_position: Annotated[int, Field(ge=0, le=MAX_SIGNED_64)]
    ingestion_cursor: Annotated[int, Field(ge=0, le=MAX_SIGNED_64)]
    memory_cursor: Annotated[int, Field(ge=0, le=MAX_SIGNED_64)]
    record_count: Annotated[int, Field(ge=0, le=MAX_ALGORITHM_RECORDS)]
    record_digests: tuple[Sha256Digest, ...]
    projection_digest: PayloadDigest
    source_snapshot_digest: Sha256Digest

    @model_validator(mode="after")
    def memory_cardinality_is_exact(self) -> Self:
        if len(self.record_digests) != self.record_count or len(set(self.record_digests)) != len(
            self.record_digests
        ):
            raise ValueError("final memory attestation cardinality is inconsistent")
        return self


class AlgorithmMetricsComponent(_AlgorithmModel):
    schema_version: Literal["algorithm-metrics/v1"] = "algorithm-metrics/v1"
    run_id: UUID4
    metrics: Stage2ExperimentMetrics
    final_budget_snapshot: BudgetSnapshot
    final_memory: AlgorithmFinalMemoryAttestation
    metrics_component_digest: Sha256Digest

    @model_validator(mode="after")
    def metrics_component_is_bound(self) -> Self:
        if self.final_memory.run_id != self.run_id:
            raise ValueError("algorithm metrics memory run differs")
        if self.metrics_component_digest != _component_model_digest(
            self,
            "metrics_component_digest",
            domain=_METRICS_COMPONENT_DIGEST_DOMAIN,
        ):
            raise ValueError("algorithm metrics component digest does not match")
        return self


class AlgorithmBoundaryAttestation(_AlgorithmModel):
    invocation_ordinal: Annotated[int, Field(ge=1, le=MAX_ALGORITHM_RECORDS)]
    boundary_event_id: UUID4
    boundary_event_sequence: Annotated[int, Field(ge=1, le=MAX_ALGORITHM_RECORDS)]
    window_digest: Sha256Digest
    invocation_decision_id: UUID4
    cycle_id: Sha256Digest | None
    source_evidence_digest: Sha256Digest
    observation: Stage2ConditionObservation
    boundary_attestation_digest: Sha256Digest

    @model_validator(mode="after")
    def observation_is_the_named_boundary(self) -> Self:
        observed = self.observation.observed
        if (
            observed.invocation_ordinal != self.invocation_ordinal
            or observed.boundary_event_id != self.boundary_event_id
            or observed.boundary_event_sequence != self.boundary_event_sequence
            or observed.window_digest != self.window_digest
            or observed.invocation_decision_id != self.invocation_decision_id
            or observed.cycle_id != self.cycle_id
        ):
            raise ValueError("algorithm boundary observation is inconsistent")
        if self.boundary_attestation_digest != _component_model_digest(
            self,
            "boundary_attestation_digest",
            domain=_BOUNDARY_ATTESTATION_DIGEST_DOMAIN,
        ):
            raise ValueError("algorithm boundary attestation digest does not match")
        return self


class AlgorithmLedgerEntryAttestation(_AlgorithmModel):
    position: Annotated[int, Field(ge=1, le=MAX_ALGORITHM_LEDGER_ENTRIES)]
    record_key: Annotated[str, Field(min_length=1, max_length=200)]
    record_type: ComponentIdentifier
    record_revision: Annotated[int, Field(ge=1, le=MAX_SIGNED_64)] | None
    record_state: ComponentIdentifier | None
    source_record_digest: Sha256Digest
    record_tag: PayloadDigest
    previous_chain_tag: PayloadDigest | None
    chain_tag: PayloadDigest


class AlgorithmAttestationsComponent(_AlgorithmModel):
    schema_version: Literal["algorithm-attestations/v1"] = "algorithm-attestations/v1"
    run_id: UUID4
    boundaries: Annotated[
        tuple[AlgorithmBoundaryAttestation, ...],
        Field(min_length=1, max_length=MAX_ALGORITHM_RECORDS),
    ]
    semantic_projection_digests: ProjectionDigests
    repository_projection_digests: ProjectionDigests
    ledger_entries: Annotated[
        tuple[AlgorithmLedgerEntryAttestation, ...],
        Field(min_length=1, max_length=MAX_ALGORITHM_LEDGER_ENTRIES),
    ]
    ledger_entry_count: Annotated[int, Field(ge=1, le=MAX_ALGORITHM_LEDGER_ENTRIES)]
    ledger_head: LedgerHead
    rebuild_equivalent: Literal[True]
    source_result_digest: Sha256Digest
    raw_synthetic_result: Stage2ExperimentRunResult | None = Field(default=None, repr=False)
    attestations_component_digest: Sha256Digest

    @model_validator(mode="after")
    def evidence_chain_and_digest_are_exact(self) -> Self:
        raw = self.raw_synthetic_result
        if (
            tuple(item.invocation_ordinal for item in self.boundaries)
            != tuple(range(1, len(self.boundaries) + 1))
            or self.semantic_projection_digests != self.repository_projection_digests
            or self.repository_projection_digests.overall.algorithm
            is not PayloadDigestAlgorithm.SYNTHETIC_SHA256
            or len(self.ledger_entries) != self.ledger_entry_count
            or tuple(item.position for item in self.ledger_entries)
            != tuple(range(1, self.ledger_entry_count + 1))
            or self.ledger_head.run_id != self.run_id
            or self.ledger_head.entry_count != self.ledger_entry_count
            or self.ledger_entries[0].previous_chain_tag is not None
            or self.ledger_entries[-1].chain_tag != self.ledger_head.chain_tag
            or any(
                later.previous_chain_tag != earlier.chain_tag
                for earlier, later in zip(
                    self.ledger_entries,
                    self.ledger_entries[1:],
                    strict=False,
                )
            )
            or (
                raw is not None
                and (
                    raw.run_id != self.run_id
                    or raw.result_digest != self.source_result_digest
                    or raw.semantic_projection_digests != self.semantic_projection_digests
                    or raw.repository_projection_digests != self.repository_projection_digests
                    or raw.ledger_entry_count != self.ledger_entry_count
                    or raw.ledger_head != self.ledger_head
                    or not raw.rebuild_equivalent
                )
            )
        ):
            raise ValueError("algorithm attestations are not one complete evidence chain")
        if self.attestations_component_digest != _component_model_digest(
            self,
            "attestations_component_digest",
            domain=_ATTESTATIONS_COMPONENT_DIGEST_DOMAIN,
        ):
            raise ValueError("algorithm attestations component digest does not match")
        return self


_SEALED_ALGORITHM_MODELS: dict[type[BaseModel], tuple[str, str]] = {
    AlgorithmRunComponent: ("run_component_digest", _RUN_COMPONENT_DIGEST_DOMAIN),
    AlgorithmTrajectoryComponent: (
        "trajectory_component_digest",
        _TRAJECTORY_COMPONENT_DIGEST_DOMAIN,
    ),
    AlgorithmCallsComponent: ("calls_component_digest", _CALLS_COMPONENT_DIGEST_DOMAIN),
    AlgorithmDecisionsComponent: (
        "decisions_component_digest",
        _DECISIONS_COMPONENT_DIGEST_DOMAIN,
    ),
    AlgorithmCycleAttestation: (
        "cycle_attestation_digest",
        _CYCLE_ATTESTATION_DIGEST_DOMAIN,
    ),
    AlgorithmCyclesComponent: ("cycles_component_digest", _CYCLES_COMPONENT_DIGEST_DOMAIN),
    AlgorithmDeliveriesComponent: (
        "deliveries_component_digest",
        _DELIVERIES_COMPONENT_DIGEST_DOMAIN,
    ),
    AlgorithmOutcomesComponent: (
        "outcomes_component_digest",
        _OUTCOMES_COMPONENT_DIGEST_DOMAIN,
    ),
    AlgorithmMetricsComponent: ("metrics_component_digest", _METRICS_COMPONENT_DIGEST_DOMAIN),
    AlgorithmBoundaryAttestation: (
        "boundary_attestation_digest",
        _BOUNDARY_ATTESTATION_DIGEST_DOMAIN,
    ),
    AlgorithmAttestationsComponent: (
        "attestations_component_digest",
        _ATTESTATIONS_COMPONENT_DIGEST_DOMAIN,
    ),
}


def _seal_algorithm_model(
    model_type: type[_ModelT],
    values: Mapping[str, object],
) -> _ModelT:
    try:
        field, domain = _SEALED_ALGORITHM_MODELS[model_type]
    except KeyError:
        raise TypeError("algorithm model does not have a sealing contract") from None
    sealed = dict(values)
    draft = model_type.model_construct(**cast(dict[str, Any], sealed))
    material = draft.model_dump(mode="json", exclude={field}, warnings=False)
    sealed[field] = _digest_without(material, field, domain=domain)
    return model_type.model_validate(sealed)


def algorithm_cycle_mode_for_condition(value: Stage2ConditionId | str) -> AlgorithmCycleMode:
    condition = value if type(value) is Stage2ConditionId else Stage2ConditionId(value)
    if condition is Stage2ConditionId.NO_MEMORY:
        return AlgorithmCycleMode.DISABLED
    if condition is Stage2ConditionId.RETRIEVAL_ALWAYS:
        return AlgorithmCycleMode.PHASE_ONE_RETRIEVAL
    return AlgorithmCycleMode.TWO_PHASE


def algorithm_component_content_digest(
    name: AlgorithmArtifactComponentName,
    data: bytes,
) -> str:
    if type(name) is not AlgorithmArtifactComponentName or type(data) is not bytes:
        raise TypeError("algorithm component digest requires an exact name and bytes")
    return length_prefixed_sha256(name.value, data, domain=_COMPONENT_DIGEST_DOMAIN)


def algorithm_overall_content_digest(
    components: tuple[AlgorithmArtifactComponent, ...],
) -> str:
    ordered = tuple(sorted(components, key=lambda component: component.name.value))
    descriptors = tuple(
        {
            "name": component.name.value,
            "path": component.path,
            "byte_count": component.byte_count,
            "record_count": component.record_count,
            "content_digest": component.content_digest,
        }
        for component in ordered
    )
    return length_prefixed_sha256(
        canonical_json(descriptors),
        domain=_CONTENT_SET_DIGEST_DOMAIN,
    )


def algorithm_window_set_digest(values: tuple[str, ...]) -> str:
    return length_prefixed_sha256(canonical_json(values), domain=_WINDOW_SET_DIGEST_DOMAIN)


def algorithm_configuration_digest(values: Mapping[str, object]) -> str:
    return length_prefixed_sha256(
        canonical_json(to_jsonable_python(values)),
        domain=_CONFIGURATION_DIGEST_DOMAIN,
    )


def algorithm_artifact_manifest_digest(values: Mapping[str, object]) -> str:
    material = {key: value for key, value in values.items() if key != "manifest_digest"}
    return length_prefixed_sha256(
        canonical_json(to_jsonable_python(material)),
        domain=_MANIFEST_DIGEST_DOMAIN,
    )


class AlgorithmArtifactManifest(_AlgorithmModel):
    record_type: Literal["algorithm_artifact_manifest"] = "algorithm_artifact_manifest"
    schema_version: Literal["algorithm-artifact/v1"] = ALGORITHM_ARTIFACT_SCHEMA_VERSION
    artifact_kind: Literal["algorithm_run"] = "algorithm_run"
    classification: AlgorithmArtifactClassification
    evidence_level: Literal[ArtifactEvidenceLevel.EXPLORATORY]
    run_id: UUID4
    revision: RevisionEvidence
    confirmatory_eligible: Literal[False]
    condition_id: Stage2ConditionId
    condition_digest: Sha256Digest
    cycle_mode: AlgorithmCycleMode
    trace_digest: Sha256Digest
    schedule_digest: Sha256Digest
    window_digests: Annotated[
        tuple[Sha256Digest, ...],
        Field(min_length=1, max_length=MAX_ALGORITHM_RECORDS),
    ]
    window_set_digest: Sha256Digest
    prompt_bundle_digest: Sha256Digest
    model_id: ComponentIdentifier
    model_profile_digest: Sha256Digest
    execution: AlgorithmExecutionAttestation
    execution_digest: Sha256Digest
    configuration_digest: Sha256Digest
    result_digest: Sha256Digest
    components: Annotated[
        tuple[AlgorithmArtifactComponent, ...],
        Field(
            min_length=len(AlgorithmArtifactComponentName),
            max_length=len(AlgorithmArtifactComponentName),
        ),
    ]
    counters: AlgorithmArtifactCounters
    overall_content_digest: Sha256Digest
    manifest_digest: Sha256Digest

    @property
    def confirmatory(self) -> Literal[False]:
        return False

    @model_validator(mode="after")
    def manifest_is_closed_and_self_attesting(self) -> Self:
        expected_mode = algorithm_cycle_mode_for_condition(self.condition_id)
        expected_condition = resolve_stage2_condition(self.condition_id)
        names = tuple(component.name for component in self.components)
        paths = tuple(component.path for component in self.components)
        all_names = set(AlgorithmArtifactComponentName)
        if (
            self.cycle_mode is not expected_mode
            or self.condition_digest != expected_condition.condition_digest
            or len(set(names)) != len(names)
            or len(set(paths)) != len(paths)
            or set(names) != all_names
            or names != tuple(sorted(names, key=lambda item: item.value))
            or self.confirmatory_eligible
            or self.execution_digest != self.execution.execution_digest
            or self.model_id != self.execution.checkpoint.model_id
            or self.window_set_digest != algorithm_window_set_digest(self.window_digests)
            or len(self.window_digests) != self.counters.scheduled_invocations
        ):
            raise ValueError("algorithm manifest anchors are inconsistent")
        if self.cycle_mode is AlgorithmCycleMode.DISABLED:
            if any(
                value != 0
                for value in (
                    self.counters.cycles,
                    self.counters.requests,
                    self.counters.model_calls,
                    self.counters.deliveries,
                    self.counters.outcomes,
                )
            ):
                raise ValueError("disabled algorithm condition cannot claim active cycles")
        elif (
            self.counters.cycles != self.counters.scheduled_invocations
            or (
                self.cycle_mode is AlgorithmCycleMode.TWO_PHASE
                and self.counters.model_calls != self.counters.requests * 2
            )
            or (
                self.cycle_mode is AlgorithmCycleMode.PHASE_ONE_RETRIEVAL
                and self.counters.model_calls != self.counters.requests
            )
        ):
            raise ValueError("algorithm condition cardinality does not match cycle mode")
        record_counts = {component.name: component.record_count for component in self.components}
        expected_counts = {
            AlgorithmArtifactComponentName.ATTESTATIONS: 1,
            AlgorithmArtifactComponentName.CALLS: self.counters.model_calls,
            AlgorithmArtifactComponentName.CYCLES: self.counters.cycles,
            AlgorithmArtifactComponentName.DECISIONS: self.counters.decisions,
            AlgorithmArtifactComponentName.DELIVERIES: self.counters.deliveries,
            AlgorithmArtifactComponentName.METRICS: 1,
            AlgorithmArtifactComponentName.OUTCOMES: self.counters.outcomes,
            AlgorithmArtifactComponentName.RUN: 1,
            AlgorithmArtifactComponentName.TRAJECTORY: self.counters.events,
        }
        if record_counts != expected_counts:
            raise ValueError("algorithm component record counts do not match counters")
        expected_configuration = algorithm_configuration_digest(
            {
                "condition_id": self.condition_id,
                "condition_digest": self.condition_digest,
                "cycle_mode": self.cycle_mode,
                "trace_digest": self.trace_digest,
                "schedule_digest": self.schedule_digest,
                "window_set_digest": self.window_set_digest,
                "prompt_bundle_digest": self.prompt_bundle_digest,
                "model_id": self.model_id,
                "model_profile_digest": self.model_profile_digest,
                "execution_digest": self.execution_digest,
            }
        )
        if self.configuration_digest != expected_configuration:
            raise ValueError("algorithm manifest configuration digest does not match")
        if self.overall_content_digest != algorithm_overall_content_digest(self.components):
            raise ValueError("algorithm overall content digest does not match")
        expected_manifest = algorithm_artifact_manifest_digest(
            self.model_dump(mode="json", exclude={"manifest_digest"}, warnings=False)
        )
        if self.manifest_digest != expected_manifest:
            raise ValueError("algorithm manifest digest does not match")
        return self

    @classmethod
    def create(
        cls,
        *,
        classification: AlgorithmArtifactClassification,
        run_id: UUID4,
        revision: RevisionEvidence,
        condition_id: Stage2ConditionId | str,
        condition_digest: Sha256Digest,
        cycle_mode: AlgorithmCycleMode,
        trace_digest: Sha256Digest,
        schedule_digest: Sha256Digest,
        window_digests: tuple[Sha256Digest, ...],
        prompt_bundle_digest: Sha256Digest,
        model_profile_digest: Sha256Digest,
        execution: AlgorithmExecutionAttestation,
        result_digest: Sha256Digest,
        components: tuple[AlgorithmArtifactComponent, ...],
        counters: AlgorithmArtifactCounters,
    ) -> AlgorithmArtifactManifest:
        checked_condition = (
            condition_id
            if type(condition_id) is Stage2ConditionId
            else Stage2ConditionId(condition_id)
        )
        ordered = tuple(sorted(components, key=lambda component: component.name.value))
        window_digest = algorithm_window_set_digest(window_digests)
        configuration = algorithm_configuration_digest(
            {
                "condition_id": checked_condition,
                "condition_digest": condition_digest,
                "cycle_mode": cycle_mode,
                "trace_digest": trace_digest,
                "schedule_digest": schedule_digest,
                "window_set_digest": window_digest,
                "prompt_bundle_digest": prompt_bundle_digest,
                "model_id": execution.checkpoint.model_id,
                "model_profile_digest": model_profile_digest,
                "execution_digest": execution.execution_digest,
            }
        )
        values: dict[str, object] = {
            "record_type": "algorithm_artifact_manifest",
            "schema_version": ALGORITHM_ARTIFACT_SCHEMA_VERSION,
            "artifact_kind": "algorithm_run",
            "classification": classification,
            "evidence_level": ArtifactEvidenceLevel.EXPLORATORY,
            "run_id": run_id,
            "revision": revision,
            "confirmatory_eligible": False,
            "condition_id": checked_condition,
            "condition_digest": condition_digest,
            "cycle_mode": cycle_mode,
            "trace_digest": trace_digest,
            "schedule_digest": schedule_digest,
            "window_digests": window_digests,
            "window_set_digest": window_digest,
            "prompt_bundle_digest": prompt_bundle_digest,
            "model_id": execution.checkpoint.model_id,
            "model_profile_digest": model_profile_digest,
            "execution": execution,
            "execution_digest": execution.execution_digest,
            "configuration_digest": configuration,
            "result_digest": result_digest,
            "components": ordered,
            "counters": counters,
            "overall_content_digest": algorithm_overall_content_digest(ordered),
        }
        values["manifest_digest"] = algorithm_artifact_manifest_digest(values)
        return cls.model_validate(values)


def expected_algorithm_component_path(name: AlgorithmArtifactComponentName) -> str:
    return _COMPONENT_PATHS[name]


__all__ = [
    "ALGORITHM_ARTIFACT_SCHEMA_VERSION",
    "MAX_ALGORITHM_COMPONENT_BYTES",
    "AlgorithmArtifactClassification",
    "AlgorithmArtifactComponent",
    "AlgorithmArtifactComponentName",
    "AlgorithmArtifactCounters",
    "AlgorithmArtifactManifest",
    "AlgorithmCheckpointAttestation",
    "AlgorithmCycleMode",
    "AlgorithmEndpointClassification",
    "AlgorithmExecutionAttestation",
    "AlgorithmExecutionMode",
    "AlgorithmHardwareAttestation",
    "AlgorithmResponseFixtureAttestation",
    "AlgorithmSamplingAttestation",
    "AlgorithmSamplingMode",
    "AlgorithmTokenizerAttestation",
    "AlgorithmTokenizerStatus",
    "AlgorithmWarmupPolicy",
    "algorithm_artifact_manifest_digest",
    "algorithm_component_content_digest",
    "algorithm_configuration_digest",
    "algorithm_cycle_mode_for_condition",
    "algorithm_overall_content_digest",
    "algorithm_window_set_digest",
    "expected_algorithm_component_path",
]
