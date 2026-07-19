from __future__ import annotations

import json
import os
import platform
import re
import stat
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal, Self, cast
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.artifacts import (
    AlgorithmCheckpointAttestation,
    AlgorithmEndpointClassification,
    AlgorithmExecutionAttestation,
    AlgorithmExecutionMode,
    AlgorithmHardwareAttestation,
    AlgorithmSamplingAttestation,
    AlgorithmSamplingMode,
    AlgorithmTokenizerAttestation,
    AlgorithmTokenizerStatus,
    AlgorithmWarmupPolicy,
    ArtifactClassification,
    ArtifactDestinationError,
    ArtifactExportError,
    ArtifactValidationError,
    RevisionEvidence,
    discover_revision,
    export_algorithm_artifact,
    load_validated_algorithm_artifact,
)
from saliencegate.commands.doctor import PilotEndpointError, validated_pilot_endpoint
from saliencegate.domain import (
    BudgetAmounts,
    EventPhase,
    EventType,
    InterventionAction,
    NormalizedTraceEventDraft,
    ReasonCode,
    TrustLabel,
    canonical_digest,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.domain.records import Sha256Digest
from saliencegate.experiments import (
    Stage2ConditionId,
    Stage2ExperimentError,
    Stage2ExperimentRunner,
    Stage2Trajectory,
    build_stage2_trajectory,
)
from saliencegate.models.openai_compatible import (
    OpenAICompatibleClient,
    OpenAICompatibleConfig,
    OpenAICompatibleError,
)
from saliencegate.ports.model_calls import (
    StructuredCallParseStatus,
    StructuredCallPhase,
    StructuredCallStatus,
)
from saliencegate.ports.trajectory import (
    ActionStepBinding,
    EventTextSelector,
    LogicalMessageBinding,
    LogicalMessageRole,
)
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.runtime.algorithm_result import algorithm_runtime_uuid, algorithm_trace_digest
from saliencegate.runtime.fixed_step import FixedStepEventInput
from saliencegate.runtime.model_token_counting import (
    HarmonyTokenCounter,
    ModelTokenCounter,
    ModelTokenCounterUnavailableError,
)
from saliencegate.security import InstallationKey, generate_installation_key

CLI_PILOT_SCHEMA_VERSION: Literal["cli-paper-two-phase-pilot-report/v1"] = (
    "cli-paper-two-phase-pilot-report/v1"
)
PILOT_SUITE_ID: Literal["paper-two-phase-local-diagnostic/v1"] = (
    "paper-two-phase-local-diagnostic/v1"
)
PILOT_TRAJECTORY_ID: Literal["paper-two-phase-basic/v1"] = "paper-two-phase-basic/v1"
PILOT_TRAJECTORY_DIGEST: Literal[
    "751489f55ac9d5ea56408ca6f5036b55e895be0fa130c36f19e624ee094d1266"
] = "751489f55ac9d5ea56408ca6f5036b55e895be0fa130c36f19e624ee094d1266"

_MODEL_ID: Literal["gpt-oss:20b"] = "gpt-oss:20b"
_RUNTIME_ID: Literal["ollama"] = "ollama"
_MAX_RUNTIME_BODY_BYTES = 1024 * 1024
_MAX_RUNTIME_JSON_NODES = 20_000
_MAX_RUNTIME_JSON_DEPTH = 32
_COMPONENT_ID = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._:/+\-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMAND_ERROR = "pilot input or output is invalid"
_UNAVAILABLE_ERROR = "pilot model runtime is unavailable"
_CONFIGURATION_ERROR = "pilot runtime configuration is invalid"
_EVIDENCE_ERROR = "pilot evidence requirements failed"
_SUITE_DIGEST_DOMAIN = "saliencegate:pilot:stage2-suite:v1"
_PROBE_CANARY = "sg-strict-7f4c2a91"


class PilotCommandError(ValueError):
    def __init__(self) -> None:
        super().__init__(_COMMAND_ERROR)


class PilotRuntimeUnavailableError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(_UNAVAILABLE_ERROR)


class PilotRuntimeConfigurationError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(_CONFIGURATION_ERROR)


class PilotEvidenceError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(_EVIDENCE_ERROR)


class _PilotModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class PaperTwoPhasePilotReport(_PilotModel):
    """Value-minimized summary constructed only from a validated live artifact."""

    schema_version: Literal["cli-paper-two-phase-pilot-report/v1"] = CLI_PILOT_SCHEMA_VERSION
    status: Literal["ok"] = "ok"
    pilot: Literal["paper-two-phase"] = "paper-two-phase"
    suite_id: Literal["paper-two-phase-local-diagnostic/v1"] = PILOT_SUITE_ID
    suite_digest: Sha256Digest
    condition: Literal[Stage2ConditionId.FIXED_STEP]
    run_id: str
    run_digest: Sha256Digest
    result_digest: Sha256Digest
    manifest_digest: Sha256Digest
    overall_content_digest: Sha256Digest
    execution_digest: Sha256Digest
    runtime_id: Literal["ollama"] = _RUNTIME_ID
    runtime_version: str
    model_id: Literal["gpt-oss:20b"] = _MODEL_ID
    model_tag: Literal["gpt-oss:20b"] = _MODEL_ID
    checkpoint_digest: Sha256Digest
    quantization: str
    warmup_policy: Literal[AlgorithmWarmupPolicy.WARM, AlgorithmWarmupPolicy.COLD]
    hardware_digest: Sha256Digest
    prompt_bundle_digest: Sha256Digest
    configuration_digest: Sha256Digest
    probe_request_digest: Sha256Digest
    probe_calls: Literal[1] = 1
    probe_provider_input_tokens: Annotated[int, Field(ge=0)]
    probe_provider_output_tokens: Annotated[int, Field(ge=0)]
    probe_latency_us: Annotated[int, Field(ge=0)]
    control_latency_us: Annotated[int, Field(ge=0)]
    postflight_latency_us: Annotated[int, Field(ge=0)]
    cases: Literal[2] = 2
    cycles: Literal[3] = 3
    calls: Literal[6] = 6
    total_calls: Literal[7] = 7
    provider_input_tokens: Annotated[int, Field(ge=0)]
    provider_output_tokens: Annotated[int, Field(ge=0)]
    total_provider_input_tokens: Annotated[int, Field(ge=0)]
    total_provider_output_tokens: Annotated[int, Field(ge=0)]
    canonical_input_tokens: Annotated[int, Field(ge=0)]
    canonical_output_tokens: Annotated[int, Field(ge=0)]
    canonical_token_equivalents: Annotated[int, Field(ge=0)]
    canonical_token_scope: Literal["diagnostic_calls_only"] = "diagnostic_calls_only"
    model_latency_us: Annotated[int, Field(ge=0)]
    memory_mutations: Annotated[int, Field(ge=1)]
    grounded_reminders: Annotated[int, Field(ge=1)]
    valid_silences: Annotated[int, Field(ge=1)]
    schema_repairs: Literal[0] = 0
    schema_invalid_outputs: Literal[0] = 0
    condition_violations: Literal[0] = 0
    classification: Literal[ArtifactClassification.SYNTHETIC_DIGEST_ONLY]
    budget_reconciled: Literal[True] = True
    rebuild_equivalent: Literal[True] = True
    artifact_validated: Literal[True] = True
    confirmatory: Literal[False] = False

    @model_validator(mode="after")
    def attestation_summary_is_consistent(self) -> Self:
        if (
            self.total_provider_input_tokens
            != self.probe_provider_input_tokens + self.provider_input_tokens
            or self.total_provider_output_tokens
            != self.probe_provider_output_tokens + self.provider_output_tokens
            or self.canonical_token_equivalents
            != self.canonical_input_tokens + self.canonical_output_tokens
            or self.suite_digest != paper_two_phase_pilot_suite_digest()
            or self.run_id != str(_RUN_ID)
        ):
            raise ValueError("pilot report attestation summary is inconsistent")
        return self


@dataclass(frozen=True, slots=True)
class _PilotDiagnosticSummary:
    provider_input_tokens: int
    provider_output_tokens: int
    canonical_input_tokens: int
    canonical_output_tokens: int
    canonical_token_equivalents: int
    model_latency_us: int
    memory_mutations: int
    grounded_reminders: int
    valid_silences: int


@dataclass(frozen=True, slots=True)
class _TrajectorySpec:
    source_event_id: str
    event_type: EventType
    phase: EventPhase
    payload: dict[str, object]
    parent_ordinals: tuple[int, ...]
    logical_messages: tuple[tuple[LogicalMessageRole, str], ...] = ()
    target_request_id: str | None = None
    task_path: str | None = None


_TRAJECTORY_SPECS = (
    _TrajectorySpec(
        "paper-two-phase-basic-01-run-start",
        EventType.RUN_START,
        EventPhase.INITIALIZATION,
        {
            "step": 1,
            "task": (
                "Complete the local package release workflow and report when final "
                "verification is finished."
            ),
        },
        (),
        target_request_id="paper-two-phase-basic-next-call-01",
        task_path="/payload/task",
    ),
    _TrajectorySpec(
        "paper-two-phase-basic-02-release-constraint",
        EventType.MODEL_OUTPUT,
        EventPhase.POST_ACTION,
        {
            "message": (
                "The release must remain offline and preserve the verified SHA-256 checksum "
                "recorded in the release manifest."
            ),
            "step": 1,
        },
        (1,),
        ((LogicalMessageRole.USER, "/payload/message"),),
    ),
    _TrajectorySpec(
        "paper-two-phase-basic-03-build-start",
        EventType.TOOL_START,
        EventPhase.ACTION_EXECUTION,
        {"command": "uv build --offline", "step": 1, "tool": "uv"},
        (2,),
    ),
    _TrajectorySpec(
        "paper-two-phase-basic-04-build-plan",
        EventType.MODEL_OUTPUT,
        EventPhase.POST_ACTION,
        {
            "assistant_message": "I will validate both artifacts without network access.",
            "step": 1,
            "user_message": "Build both the wheel and source archive deterministically.",
        },
        (3,),
        (
            (LogicalMessageRole.USER, "/payload/user_message"),
            (LogicalMessageRole.ASSISTANT, "/payload/assistant_message"),
        ),
    ),
    _TrajectorySpec(
        "paper-two-phase-basic-05-lock-timeout",
        EventType.TOOL_COMPLETION,
        EventPhase.POST_ACTION,
        {"error": "The local build lock timed out.", "exit_code": 1, "step": 1, "tool": "uv"},
        (3, 4),
    ),
    _TrajectorySpec(
        "paper-two-phase-basic-06-retry-boundary",
        EventType.OBSERVATION,
        EventPhase.POST_ACTION,
        {
            "constraint": "Keep the offline and checksum requirements active during the retry.",
            "failure_summary": (
                "The first deterministic build failed because a stale local lock timed out; "
                "clear only that lock before retrying."
            ),
            "step": 2,
        },
        (5,),
        (
            (LogicalMessageRole.TOOL, "/payload/failure_summary"),
            (LogicalMessageRole.CONTROLLER, "/payload/constraint"),
        ),
        target_request_id="paper-two-phase-basic-next-call-02",
    ),
    _TrajectorySpec(
        "paper-two-phase-basic-07-retry-progress",
        EventType.MODEL_OUTPUT,
        EventPhase.POST_ACTION,
        {
            "assistant_message": (
                "I cleared the stale local lock and restarted the deterministic build."
            ),
            "step": 2,
            "tool_message": "The retry produced the wheel and source archive locally.",
        },
        (6,),
        (
            (LogicalMessageRole.ASSISTANT, "/payload/assistant_message"),
            (LogicalMessageRole.TOOL, "/payload/tool_message"),
        ),
    ),
    _TrajectorySpec(
        "paper-two-phase-basic-08-build-complete",
        EventType.TOOL_COMPLETION,
        EventPhase.POST_ACTION,
        {"artifact_count": 2, "exit_code": 0, "step": 2, "tool": "uv"},
        (7,),
    ),
    _TrajectorySpec(
        "paper-two-phase-basic-09-artifact-validation",
        EventType.OBSERVATION,
        EventPhase.POST_ACTION,
        {
            "assistant_message": "Package validation passed without network access.",
            "step": 2,
            "tool_message": "Both artifact checksums match the verified release manifest.",
        },
        (7, 8),
        (
            (LogicalMessageRole.TOOL, "/payload/tool_message"),
            (LogicalMessageRole.ASSISTANT, "/payload/assistant_message"),
        ),
    ),
    _TrajectorySpec(
        "paper-two-phase-basic-10-subgoal-complete",
        EventType.MODEL_OUTPUT,
        EventPhase.POST_ACTION,
        {
            "controller_message": "Retain the release constraint for final verification.",
            "step": 2,
            "user_message": "The lock-timeout retry is complete; do not keep that subgoal active.",
        },
        (9,),
        (
            (LogicalMessageRole.USER, "/payload/user_message"),
            (LogicalMessageRole.CONTROLLER, "/payload/controller_message"),
        ),
    ),
    _TrajectorySpec(
        "paper-two-phase-basic-11-final-boundary",
        EventType.ACTION_PROPOSAL,
        EventPhase.PRE_ACTION,
        {
            "assistant_message": (
                "Proceeding to final verification with the retained release constraint."
            ),
            "step": 3,
            "user_message": "Do not remind me about the completed lock retry.",
        },
        (10,),
        (
            (LogicalMessageRole.ASSISTANT, "/payload/assistant_message"),
            (LogicalMessageRole.USER, "/payload/user_message"),
        ),
        target_request_id="paper-two-phase-basic-next-call-03",
    ),
)

_RUN_ID = UUID("00000000-0000-4000-8000-000000009000")
_EVENT_IDS = tuple(
    UUID(f"00000000-0000-4000-8000-{9000 + ordinal:012d}")
    for ordinal in range(1, len(_TRAJECTORY_SPECS) + 1)
)


def build_paper_two_phase_pilot_trajectory() -> Stage2Trajectory:
    """Build the reviewed diagnostic trajectory without consulting repository fixtures."""

    started = datetime(2026, 7, 13, 9, 0, tzinfo=UTC)
    inputs = tuple(
        FixedStepEventInput(
            draft=NormalizedTraceEventDraft(
                schema_version="1.0",
                run_id=_RUN_ID,
                source_event_id=spec.source_event_id,
                timestamp=started + timedelta(seconds=ordinal - 1),
                event_type=spec.event_type,
                phase=spec.phase,
                payload=spec.payload,
                parent_ids=tuple(_EVENT_IDS[parent - 1] for parent in spec.parent_ordinals),
                source_adapter=PILOT_TRAJECTORY_ID,
                trust_label=TrustLabel.SYNTHETIC_FIXTURE,
            ),
            expected_event_id=_EVENT_IDS[ordinal - 1],
            task_description=(
                None if spec.task_path is None else EventTextSelector(field_path=spec.task_path)
            ),
            logical_messages=tuple(
                LogicalMessageBinding(
                    role=role,
                    selector=EventTextSelector(field_path=path),
                )
                for role, path in spec.logical_messages
            ),
            action_step=ActionStepBinding(field_path="/payload/step"),
            target_request_id=spec.target_request_id,
        )
        for ordinal, spec in enumerate(_TRAJECTORY_SPECS, start=1)
    )
    trajectory = build_stage2_trajectory(inputs, fixture_id=PILOT_TRAJECTORY_ID)
    if trajectory.fixture_digest != PILOT_TRAJECTORY_DIGEST:
        raise PilotEvidenceError()
    return trajectory


_SUITE_CASES: tuple[Mapping[str, object], ...] = (
    MappingProxyType(
        {
            "case_id": "mutation-grounded-reminder",
            "invocation_ordinal": 2,
            "boundary_event_sequence": 6,
            "expected_action": "remind",
            "expected_reason_code": "grounded_reminder",
            "minimum_memory_mutations": 1,
        }
    ),
    MappingProxyType(
        {
            "case_id": "valid-silence",
            "invocation_ordinal": 3,
            "boundary_event_sequence": 11,
            "expected_action": "silence",
            "expected_reason_code": "silence_selected",
            "minimum_memory_mutations": 0,
        }
    ),
)


def paper_two_phase_pilot_suite_digest() -> str:
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": "stage2-pilot-suite/v1",
                "suite_id": PILOT_SUITE_ID,
                "condition": Stage2ConditionId.FIXED_STEP,
                "trajectory_fixture_id": PILOT_TRAJECTORY_ID,
                "trajectory_fixture_digest": PILOT_TRAJECTORY_DIGEST,
                "invocation_boundary_sequences": (1, 6, 11),
                "cases": _SUITE_CASES,
            }
        ),
        domain=_SUITE_DIGEST_DOMAIN,
    )


def _runtime_extra_available() -> bool:
    try:
        return bool(metadata.version("httpx")) and bool(metadata.version("openai-harmony"))
    except Exception:
        return False


def _token_counter(model: str) -> ModelTokenCounter:
    return HarmonyTokenCounter(model_id=model)


def _hardware() -> AlgorithmHardwareAttestation:
    try:
        logical_cores = os.cpu_count()
        page_size = os.sysconf("SC_PAGE_SIZE")
        physical_pages = os.sysconf("SC_PHYS_PAGES")
        if (
            type(logical_cores) is not int
            or logical_cores < 1
            or type(page_size) is not int
            or page_size < 1
            or type(physical_pages) is not int
            or physical_pages < 1
        ):
            raise ValueError
        memory_bytes = page_size * physical_pages
        if memory_bytes > (1 << 63) - 1:
            raise ValueError
        return AlgorithmHardwareAttestation(
            model="local-runtime",
            architecture=platform.machine() or "unknown-architecture",
            logical_core_count=logical_cores,
            memory_capacity_bytes=memory_bytes,
            operating_system=platform.system() or "unknown-system",
            operating_system_version=platform.release() or "unknown-release",
        )
    except Exception:
        raise PilotRuntimeConfigurationError() from None


@dataclass(frozen=True, slots=True)
class PilotRuntimeDependencies:
    transport_factory: Callable[[], httpx.AsyncBaseTransport] | None = None
    hardware_provider: Callable[[], AlgorithmHardwareAttestation] = _hardware
    runtime_extra_available: Callable[[], bool] = _runtime_extra_available
    model_token_counter_factory: Callable[[str], ModelTokenCounter] = _token_counter
    installation_key_factory: Callable[[], InstallationKey] = generate_installation_key
    monotonic_ns: Callable[[], int] = time.perf_counter_ns
    revision_provider: Callable[[], RevisionEvidence] = discover_revision


@dataclass(frozen=True, slots=True)
class _ProbeEvidence:
    request_digest: str
    provider_input_tokens: int
    provider_output_tokens: int
    latency_us: int


@dataclass(frozen=True, slots=True)
class _RuntimeProfile:
    version: str
    checkpoint_digest: str
    quantization: str
    probe: _ProbeEvidence
    control_latency_us: int


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError


def _bounded_json(value: object) -> bool:
    stack: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_RUNTIME_JSON_NODES or depth > _MAX_RUNTIME_JSON_DEPTH:
            return False
        if type(current) is dict:
            stack.extend((item, depth + 1) for item in cast(dict[str, object], current).values())
        elif type(current) is list:
            stack.extend((item, depth + 1) for item in cast(list[object], current))
    return True


def _parse_runtime_json(data: bytes) -> dict[str, object]:
    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if type(value) is not dict or not _bounded_json(value):
            raise ValueError
        return cast(dict[str, object], value)
    except Exception:
        raise PilotRuntimeConfigurationError() from None


def _elapsed_us(clock: Callable[[], int], started: int) -> int:
    try:
        finished = clock()
        if type(started) is not int or type(finished) is not int or started < 0 or finished < 0:
            raise ValueError
        return min(max(0, finished - started) // 1_000, (1 << 63) - 1)
    except Exception:
        raise PilotRuntimeConfigurationError() from None


async def _runtime_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    body: bytes | None,
    clock: Callable[[], int],
) -> tuple[dict[str, object], int]:
    try:
        started = clock()
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        async with client.stream(method, url, content=body, headers=headers) as response:
            if not 200 <= response.status_code < 300:
                raise ValueError
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
            if content_type.lower() != "application/json":
                raise ValueError
            if response.headers.get("content-encoding", "identity").strip().lower() != "identity":
                raise ValueError
            declared = response.headers.get("content-length")
            if declared is not None and (
                not declared.isascii()
                or not declared.isdigit()
                or int(declared) > _MAX_RUNTIME_BODY_BYTES
            ):
                raise ValueError
            chunks: list[bytes] = []
            observed = 0
            if response.is_stream_consumed:
                observed = len(response.content)
                if observed > _MAX_RUNTIME_BODY_BYTES:
                    raise ValueError
                chunks.append(response.content)
            else:
                async for chunk in response.aiter_raw():
                    observed += len(chunk)
                    if observed > _MAX_RUNTIME_BODY_BYTES:
                        raise ValueError
                    chunks.append(chunk)
        return _parse_runtime_json(b"".join(chunks)), _elapsed_us(clock, started)
    except PilotRuntimeConfigurationError:
        raise
    except Exception:
        raise PilotRuntimeConfigurationError() from None


def _exact_component(value: object) -> str:
    if type(value) is not str or _COMPONENT_ID.fullmatch(value) is None:
        raise PilotRuntimeConfigurationError()
    return value


def _exact_digest(value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise PilotRuntimeConfigurationError()
    return value


def _visible_model(payload: Mapping[str, object], model: str) -> tuple[str, str]:
    try:
        raw_models = payload.get("models")
        if type(raw_models) is not list:
            raise ValueError
        matches: list[tuple[str, str]] = []
        for raw in raw_models:
            if type(raw) is not dict:
                raise ValueError
            item = cast(dict[str, object], raw)
            name = item.get("name")
            model_alias = item.get("model")
            if name != model and model_alias != model:
                continue
            if name is not None and model_alias is not None and name != model_alias:
                raise ValueError
            digest = _exact_digest(item.get("digest"))
            size = item.get("size")
            details = item.get("details")
            if type(size) is not int or size <= 0 or type(details) is not dict:
                raise ValueError
            detail_values = cast(dict[str, object], details)
            if detail_values.get("format") != "gguf":
                raise ValueError
            quantization = _exact_component(detail_values.get("quantization_level"))
            matches.append((digest, quantization))
        if len(matches) != 1:
            raise ValueError
        return matches[0]
    except PilotRuntimeConfigurationError:
        raise
    except Exception:
        raise PilotRuntimeConfigurationError() from None


def _validate_show(payload: Mapping[str, object], *, quantization: str) -> None:
    try:
        capabilities = payload.get("capabilities")
        details = payload.get("details")
        if (
            type(capabilities) is not list
            or "completion" not in capabilities
            or any(type(item) is not str for item in capabilities)
            or type(details) is not dict
            or cast(dict[str, object], details).get("format") != "gguf"
            or cast(dict[str, object], details).get("quantization_level") != quantization
        ):
            raise ValueError
    except Exception:
        raise PilotRuntimeConfigurationError() from None


def _probe_evidence(
    payload: Mapping[str, object],
    *,
    model: str,
    request_body: bytes,
    latency_us: int,
) -> _ProbeEvidence:
    try:
        if payload.get("model") != model:
            raise ValueError
        choices = payload.get("choices")
        if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
            raise ValueError
        message = cast(dict[str, object], choices[0]).get("message")
        if type(message) is not dict:
            raise ValueError
        content = cast(dict[str, object], message).get("content")
        if type(content) is not str:
            raise ValueError
        decoded = json.loads(
            content,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if (
            type(decoded) is not dict
            or set(decoded) != {"canary", "ready"}
            or type(decoded.get("canary")) is not str
            or decoded.get("canary") != _PROBE_CANARY
            or type(decoded.get("ready")) is not bool
            or decoded.get("ready") is not True
        ):
            raise ValueError
        usage = payload.get("usage")
        if type(usage) is not dict:
            raise ValueError
        provider_input = cast(dict[str, object], usage).get("prompt_tokens")
        provider_output = cast(dict[str, object], usage).get("completion_tokens")
        provider_total = cast(dict[str, object], usage).get("total_tokens")
        if (
            type(provider_input) is not int
            or provider_input < 0
            or provider_input > (1 << 63) - 1
            or type(provider_output) is not int
            or provider_output < 0
            or provider_output > (1 << 63) - 1
            or provider_input + provider_output > (1 << 63) - 1
            or (
                provider_total is not None
                and (
                    type(provider_total) is not int
                    or provider_total != provider_input + provider_output
                )
            )
        ):
            raise ValueError
        return _ProbeEvidence(
            request_digest=canonical_digest(_parse_runtime_json(request_body)),
            provider_input_tokens=provider_input,
            provider_output_tokens=provider_output,
            latency_us=latency_us,
        )
    except PilotRuntimeConfigurationError:
        raise
    except Exception:
        raise PilotRuntimeConfigurationError() from None


def _model_is_running(payload: Mapping[str, object], checkpoint_digest: str) -> bool:
    try:
        models = payload.get("models")
        if type(models) is not list:
            raise ValueError
        for raw in models:
            if type(raw) is not dict:
                raise ValueError
            digest = cast(dict[str, object], raw).get("digest")
            if type(digest) is not str or _SHA256.fullmatch(digest) is None:
                raise ValueError
            if digest == checkpoint_digest:
                return True
        return False
    except Exception:
        raise PilotRuntimeConfigurationError() from None


async def _probe_runtime_once(
    endpoint: str,
    model: str,
    warmup: AlgorithmWarmupPolicy,
    dependencies: PilotRuntimeDependencies,
) -> _RuntimeProfile:
    origin = endpoint.removesuffix("/v1")
    try:
        transport = (
            httpx.AsyncHTTPTransport(retries=0, trust_env=False)
            if dependencies.transport_factory is None
            else dependencies.transport_factory()
        )
        if not isinstance(transport, httpx.AsyncBaseTransport):
            raise TypeError
        client = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(120.0),
            follow_redirects=False,
            trust_env=False,
        )
    except Exception:
        raise PilotRuntimeConfigurationError() from None

    async with client:
        version_payload, _ = await _runtime_request(
            client,
            "GET",
            f"{origin}/api/version",
            body=None,
            clock=dependencies.monotonic_ns,
        )
        version = _exact_component(version_payload.get("version"))
        tags_payload, _ = await _runtime_request(
            client,
            "GET",
            f"{origin}/api/tags",
            body=None,
            clock=dependencies.monotonic_ns,
        )
        checkpoint_digest, quantization = _visible_model(tags_payload, model)
        show_payload, _ = await _runtime_request(
            client,
            "POST",
            f"{origin}/api/show",
            body=canonical_json({"model": model, "verbose": False}),
            clock=dependencies.monotonic_ns,
        )
        _validate_show(show_payload, quantization=quantization)

        probe_body = canonical_json(
            {
                "messages": (
                    {
                        "content": "Follow the visible request literally.",
                        "role": "system",
                    },
                    {
                        "content": 'Return exactly {"ready":false} and no other fields.',
                        "role": "user",
                    },
                ),
                "model": model,
                "reasoning_effort": "medium",
                "response_format": {
                    "json_schema": {
                        "name": "saliencegate_runtime_probe_v1",
                        "schema": {
                            "additionalProperties": False,
                            "properties": {
                                "canary": {"const": _PROBE_CANARY, "type": "string"},
                                "ready": {"const": True, "type": "boolean"},
                            },
                            "required": ("canary", "ready"),
                            "type": "object",
                        },
                        "strict": True,
                    },
                    "type": "json_schema",
                },
                "seed": 0,
                "stream": False,
                "temperature": 0,
            }
        )
        probe_payload, probe_latency = await _runtime_request(
            client,
            "POST",
            f"{endpoint}/chat/completions",
            body=probe_body,
            clock=dependencies.monotonic_ns,
        )
        probe = _probe_evidence(
            probe_payload,
            model=model,
            request_body=probe_body,
            latency_us=probe_latency,
        )

        control_latency = 0
        if warmup is AlgorithmWarmupPolicy.COLD:
            _, unload_latency = await _runtime_request(
                client,
                "POST",
                f"{origin}/api/generate",
                body=canonical_json({"keep_alive": 0, "model": model, "stream": False}),
                clock=dependencies.monotonic_ns,
            )
            control_latency += unload_latency
        running_payload, running_latency = await _runtime_request(
            client,
            "GET",
            f"{origin}/api/ps",
            body=None,
            clock=dependencies.monotonic_ns,
        )
        control_latency += running_latency
        running = _model_is_running(running_payload, checkpoint_digest)
        if running is not (warmup is AlgorithmWarmupPolicy.WARM):
            raise PilotRuntimeConfigurationError()
    return _RuntimeProfile(
        version=version,
        checkpoint_digest=checkpoint_digest,
        quantization=quantization,
        probe=probe,
        control_latency_us=control_latency,
    )


async def _probe_runtime(
    endpoint: str,
    model: str,
    warmup: AlgorithmWarmupPolicy,
    dependencies: PilotRuntimeDependencies,
) -> _RuntimeProfile:
    failed = False
    try:
        return await _probe_runtime_once(endpoint, model, warmup, dependencies)
    except Exception:
        failed = True
    assert failed
    raise PilotRuntimeConfigurationError()


async def _postflight_runtime_once(
    endpoint: str,
    model: str,
    expected: _RuntimeProfile,
    dependencies: PilotRuntimeDependencies,
) -> int:
    origin = endpoint.removesuffix("/v1")
    transport = (
        httpx.AsyncHTTPTransport(retries=0, trust_env=False)
        if dependencies.transport_factory is None
        else dependencies.transport_factory()
    )
    if not isinstance(transport, httpx.AsyncBaseTransport):
        raise PilotRuntimeConfigurationError()
    client = httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(120.0),
        follow_redirects=False,
        trust_env=False,
    )
    total_latency = 0
    async with client:
        version_payload, latency = await _runtime_request(
            client,
            "GET",
            f"{origin}/api/version",
            body=None,
            clock=dependencies.monotonic_ns,
        )
        total_latency += latency
        version = _exact_component(version_payload.get("version"))
        tags_payload, latency = await _runtime_request(
            client,
            "GET",
            f"{origin}/api/tags",
            body=None,
            clock=dependencies.monotonic_ns,
        )
        total_latency += latency
        checkpoint_digest, quantization = _visible_model(tags_payload, model)
        show_payload, latency = await _runtime_request(
            client,
            "POST",
            f"{origin}/api/show",
            body=canonical_json({"model": model, "verbose": False}),
            clock=dependencies.monotonic_ns,
        )
        total_latency += latency
        _validate_show(show_payload, quantization=quantization)
        running_payload, latency = await _runtime_request(
            client,
            "GET",
            f"{origin}/api/ps",
            body=None,
            clock=dependencies.monotonic_ns,
        )
        total_latency += latency
        if (
            version != expected.version
            or checkpoint_digest != expected.checkpoint_digest
            or quantization != expected.quantization
            or not _model_is_running(running_payload, expected.checkpoint_digest)
        ):
            raise PilotRuntimeConfigurationError()
    return total_latency


async def _postflight_runtime(
    endpoint: str,
    model: str,
    expected: _RuntimeProfile,
    dependencies: PilotRuntimeDependencies,
) -> int:
    failed = False
    try:
        return await _postflight_runtime_once(endpoint, model, expected, dependencies)
    except Exception:
        failed = True
    assert failed
    raise PilotRuntimeConfigurationError()


def _output_path(value: str | os.PathLike[str]) -> Path:
    try:
        raw = os.fspath(value)
        if type(raw) is not str or not raw:
            raise ValueError
        path = Path(raw)
        if path.name in ("", ".", ".."):
            raise ValueError
        try:
            path.lstat()
        except FileNotFoundError:
            pass
        else:
            raise ValueError
        parent = path.parent
        inspected = parent
        while True:
            try:
                parent_stat = inspected.lstat()
                break
            except FileNotFoundError:
                ancestor = inspected.parent
                if ancestor == inspected:
                    raise
                inspected = ancestor
        if (
            stat.S_ISLNK(parent_stat.st_mode)
            or not stat.S_ISDIR(parent_stat.st_mode)
            or not os.access(inspected, os.W_OK | os.X_OK)
            or (
                os.name == "posix"
                and (
                    stat.S_IMODE(parent_stat.st_mode) & 0o022
                    or (hasattr(os, "getuid") and parent_stat.st_uid != os.getuid())
                )
            )
        ):
            raise ValueError
        return path
    except Exception:
        raise PilotCommandError() from None


def _dependencies(value: PilotRuntimeDependencies | None) -> PilotRuntimeDependencies:
    try:
        checked = PilotRuntimeDependencies() if value is None else value
        if type(checked) is not PilotRuntimeDependencies:
            raise TypeError
        if not all(
            callable(item)
            for item in (
                checked.hardware_provider,
                checked.runtime_extra_available,
                checked.model_token_counter_factory,
                checked.installation_key_factory,
                checked.monotonic_ns,
                checked.revision_provider,
            )
        ) or (checked.transport_factory is not None and not callable(checked.transport_factory)):
            raise TypeError
        return checked
    except Exception:
        raise PilotCommandError() from None


def _repository_id_factory(trace_digest: str) -> Callable[[], UUID]:
    ordinal = 0

    def next_identifier() -> UUID:
        nonlocal ordinal
        ordinal += 1
        return algorithm_runtime_uuid(trace_digest, "stage2-repository", ordinal)

    return next_identifier


def _sum_budget(values: tuple[BudgetAmounts, ...]) -> BudgetAmounts:
    fields = (
        "model_calls",
        "input_tokens",
        "output_tokens",
        "canonical_token_equivalents",
        "latency_us",
        "interventions",
        "schema_repairs",
    )
    return BudgetAmounts(
        **{field: sum(getattr(value, field) for value in values) for field in fields}
    )


def _gate_result(result: object) -> _PilotDiagnosticSummary:
    try:
        from saliencegate.experiments import Stage2ExperimentRunResult

        checked = Stage2ExperimentRunResult.model_validate(result)
        active_boundaries = tuple(item for item in checked.boundaries if item.cycle is not None)
        active_cycles = tuple(item.cycle for item in active_boundaries if item.cycle is not None)
        memory_mutations = sum(
            len(delta.creates)
            + len(delta.updates)
            + len(delta.invalidations)
            + int(delta.private_status_replacement is not None)
            for cycle in active_cycles
            if (delta := cycle.validated_delta) is not None
        )
        grounded_reminders = sum(
            cycle.intervention is not None
            and cycle.intervention.action is InterventionAction.REMIND
            and cycle.intervention.reason_code is ReasonCode.GROUNDED_REMINDER
            for cycle in active_cycles
        )
        valid_silences = sum(
            cycle.intervention is not None
            and cycle.intervention.action is InterventionAction.SILENCE
            and cycle.intervention.reason_code is ReasonCode.SILENCE_SELECTED
            for cycle in active_cycles
        )
        schema_repairs = sum(call.attempt > 0 for call in checked.call_receipts)
        invalid_outputs = sum(
            call.parse_status is not StructuredCallParseStatus.VALID
            for call in checked.call_receipts
        )
        if (
            checked.condition.condition_id is not Stage2ConditionId.FIXED_STEP
            or checked.trajectory.fixture_id != PILOT_TRAJECTORY_ID
            or checked.trajectory.fixture_digest != PILOT_TRAJECTORY_DIGEST
            or tuple(item.boundary_event.sequence for item in active_boundaries) != (1, 6, 11)
            or len(active_boundaries) != 3
            or len(active_cycles) != 3
            or len(checked.call_receipts) != 6
            or checked.call_policy.max_model_calls != 2
            or checked.call_policy.max_schema_repairs != 0
            or checked.call_policy.client_retries != 0
            or any(
                call.status is not StructuredCallStatus.COMPLETED
                or call.parse_status is not StructuredCallParseStatus.VALID
                or call.attempt != 0
                for call in checked.call_receipts
            )
            or any(
                tuple(call.phase for call in boundary.call_receipts)
                != (StructuredCallPhase.MEMORY_EDIT, StructuredCallPhase.INTERVENTION)
                for boundary in active_boundaries
            )
            or schema_repairs != 0
            or invalid_outputs != 0
            or checked.metrics.model_call_count != 6
            or checked.metrics.provider_input_tokens is None
            or checked.metrics.provider_output_tokens is None
            or checked.metrics.canonical_input_tokens is None
            or checked.metrics.canonical_output_tokens is None
            or checked.metrics.canonical_token_equivalents is None
            or memory_mutations < 1
            or grounded_reminders < 1
            or valid_silences < 1
            or checked.metrics.memory_mutation_count != memory_mutations
            or checked.metrics.intervention_count != grounded_reminders
            or checked.metrics.provenance_validated_boundary_count != 3
            or checked.metrics.grounding_rejection_count != 0
            or checked.metrics.condition_violation_count != 0
            or checked.final_budget_snapshot.reserved != BudgetAmounts()
            or checked.final_budget_snapshot.consumed
            != _sum_budget(
                tuple(
                    cast(BudgetAmounts, boundary.cycle.budget_settlement)
                    for boundary in active_boundaries
                    if boundary.cycle is not None
                )
            )
            or checked.semantic_projection_digests != checked.repository_projection_digests
            or checked.rebuild_equivalent is not True
        ):
            raise ValueError
        reminder = active_boundaries[1].cycle
        silence = active_boundaries[2].cycle
        if reminder is None or silence is None:
            raise ValueError
        reminder_mutations = reminder.validated_delta
        if (
            reminder_mutations is None
            or (
                len(reminder_mutations.creates)
                + len(reminder_mutations.updates)
                + len(reminder_mutations.invalidations)
                + int(reminder_mutations.private_status_replacement is not None)
                < 1
            )
            or reminder.intervention is None
            or reminder.intervention.action is not InterventionAction.REMIND
            or reminder.intervention.reason_code is not ReasonCode.GROUNDED_REMINDER
            or silence.intervention is None
            or silence.intervention.action is not InterventionAction.SILENCE
            or silence.intervention.reason_code is not ReasonCode.SILENCE_SELECTED
        ):
            raise ValueError
        return _PilotDiagnosticSummary(
            provider_input_tokens=checked.metrics.provider_input_tokens,
            provider_output_tokens=checked.metrics.provider_output_tokens,
            canonical_input_tokens=checked.metrics.canonical_input_tokens,
            canonical_output_tokens=checked.metrics.canonical_output_tokens,
            canonical_token_equivalents=checked.metrics.canonical_token_equivalents,
            model_latency_us=sum(call.usage.latency_us for call in checked.call_receipts),
            memory_mutations=memory_mutations,
            grounded_reminders=grounded_reminders,
            valid_silences=valid_silences,
        )
    except PilotEvidenceError:
        raise
    except Exception:
        raise PilotEvidenceError() from None


def _execution(
    result: object,
    *,
    runtime: _RuntimeProfile,
    hardware: AlgorithmHardwareAttestation,
    warmup: AlgorithmWarmupPolicy,
) -> AlgorithmExecutionAttestation:
    try:
        from saliencegate.experiments import Stage2ExperimentRunResult

        checked = Stage2ExperimentRunResult.model_validate(result)
        calls = checked.call_receipts
        identities = {
            (
                call.usage.local_counter_id,
                call.usage.local_counter_version,
                call.usage.local_counter_configuration_digest,
                call.usage.local_counter_model_id,
            )
            for call in calls
        }
        if len(identities) != 1:
            raise ValueError
        tokenizer_id, tokenizer_version, configuration_digest, tokenizer_model = identities.pop()
        if any(
            value is None
            for value in (
                tokenizer_id,
                tokenizer_version,
                configuration_digest,
                tokenizer_model,
            )
        ):
            raise ValueError
        return AlgorithmExecutionAttestation.create(
            execution_mode=AlgorithmExecutionMode.OPENAI_COMPATIBLE,
            endpoint_classification=AlgorithmEndpointClassification.LOOPBACK_OPENAI_COMPATIBLE,
            runtime_id=_RUNTIME_ID,
            runtime_version=runtime.version,
            checkpoint=AlgorithmCheckpointAttestation(
                model_id=_MODEL_ID,
                model_tag=None,
                checkpoint_digest=runtime.checkpoint_digest,
                quantization=runtime.quantization,
            ),
            sampling=AlgorithmSamplingAttestation(
                mode=AlgorithmSamplingMode.OPENAI_COMPATIBLE,
                temperature=0.0,
                seed=0,
                reasoning_effort="medium",
            ),
            tokenizer=AlgorithmTokenizerAttestation(
                status=AlgorithmTokenizerStatus.ATTESTED,
                tokenizer_id=cast(str, tokenizer_id),
                tokenizer_version=cast(str, tokenizer_version),
                configuration_digest=cast(str, configuration_digest),
                model_id=cast(str, tokenizer_model),
            ),
            hardware=hardware,
            warmup_policy=warmup,
            response_fixture=None,
        )
    except Exception:
        raise PilotEvidenceError() from None


async def _run_paper_two_phase_pilot(
    *,
    endpoint: str,
    model: str,
    output_path: str | os.PathLike[str],
    warmup: Literal["warm", "cold"] | AlgorithmWarmupPolicy = "warm",
    dependencies: PilotRuntimeDependencies | None = None,
) -> PaperTwoPhasePilotReport:
    """Run one guarded, non-confirmatory local-model diagnostic."""

    try:
        checked_endpoint = validated_pilot_endpoint(endpoint)
    except PilotEndpointError:
        raise PilotCommandError() from None
    if type(model) is not str or model != _MODEL_ID:
        raise PilotCommandError()
    output = _output_path(output_path)
    deps = _dependencies(dependencies)
    try:
        checked_warmup = AlgorithmWarmupPolicy(warmup)
    except (TypeError, ValueError):
        raise PilotCommandError() from None
    if checked_warmup not in (AlgorithmWarmupPolicy.WARM, AlgorithmWarmupPolicy.COLD):
        raise PilotCommandError()
    try:
        extras_available = deps.runtime_extra_available()
    except Exception:
        extras_available = False
    if extras_available is not True:
        raise PilotRuntimeUnavailableError()

    try:
        token_counter = deps.model_token_counter_factory(model)
        installation_key = deps.installation_key_factory()
        hardware = deps.hardware_provider()
        if type(hardware) is not AlgorithmHardwareAttestation:
            raise TypeError
        hardware = AlgorithmHardwareAttestation.model_validate_json(
            hardware.model_dump_json(warnings=False)
        )
    except ModelTokenCounterUnavailableError:
        raise PilotRuntimeUnavailableError() from None
    except PilotRuntimeConfigurationError:
        raise
    except Exception:
        raise PilotRuntimeConfigurationError() from None

    runtime = await _probe_runtime(
        checked_endpoint,
        model,
        checked_warmup,
        deps,
    )
    trajectory = build_paper_two_phase_pilot_trajectory()
    trace_digest = algorithm_trace_digest(
        tuple(canonical_digest(item.draft) for item in trajectory.inputs)
    )
    try:
        client = OpenAICompatibleClient(
            OpenAICompatibleConfig(
                base_url=checked_endpoint,
                model=model,
                timeout_seconds=120.0,
                seed=0,
                reasoning_effort="medium",
                credential_env=None,
                allow_remote=False,
            ),
            installation_key=installation_key,
            model_token_counter=token_counter,
            transport_factory=deps.transport_factory,
            credential_lookup=lambda _name: None,
            monotonic_ns=deps.monotonic_ns,
        )
    except OpenAICompatibleError:
        raise PilotRuntimeConfigurationError() from None
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=_repository_id_factory(trace_digest),
    )
    try:
        async with client:
            result = await Stage2ExperimentRunner(
                repository=repository,
                condition=Stage2ConditionId.FIXED_STEP,
                client=client,
            ).run(trajectory)
    except Stage2ExperimentError:
        raise PilotEvidenceError() from None
    except OpenAICompatibleError:
        raise PilotEvidenceError() from None
    except Exception:
        raise PilotEvidenceError() from None
    postflight_latency = await _postflight_runtime(
        checked_endpoint,
        model,
        runtime,
        deps,
    )
    diagnostic = _gate_result(result)
    execution = _execution(
        result,
        runtime=runtime,
        hardware=hardware,
        warmup=checked_warmup,
    )
    try:
        revision = deps.revision_provider()
        manifest = export_algorithm_artifact(
            result,
            output,
            execution=execution,
            classification=ArtifactClassification.SYNTHETIC_DIGEST_ONLY,
            revision=revision,
            replace=False,
        )
        loaded = load_validated_algorithm_artifact(
            output / "manifest.json",
            expected_manifest_digest=manifest.manifest_digest,
        )
    except ArtifactValidationError:
        raise
    except ArtifactDestinationError:
        raise PilotCommandError() from None
    except ArtifactExportError:
        raise PilotEvidenceError() from None
    except Exception:
        raise PilotEvidenceError() from None

    if (
        loaded.manifest.execution != execution
        or loaded.manifest.classification is not ArtifactClassification.SYNTHETIC_DIGEST_ONLY
        or loaded.manifest.confirmatory is not False
        or loaded.report.confirmatory is not False
    ):
        raise PilotEvidenceError()

    return PaperTwoPhasePilotReport(
        suite_digest=paper_two_phase_pilot_suite_digest(),
        condition=Stage2ConditionId.FIXED_STEP,
        run_id=str(loaded.manifest.run_id),
        run_digest=loaded.run.run_component_digest,
        result_digest=loaded.manifest.result_digest,
        manifest_digest=loaded.manifest.manifest_digest,
        overall_content_digest=loaded.manifest.overall_content_digest,
        execution_digest=execution.execution_digest,
        runtime_version=runtime.version,
        checkpoint_digest=runtime.checkpoint_digest,
        quantization=runtime.quantization,
        warmup_policy=checked_warmup,
        hardware_digest=cast(str, execution.hardware.hardware_digest),
        prompt_bundle_digest=loaded.run.prompt_bundle.bundle_digest,
        configuration_digest=loaded.manifest.configuration_digest,
        probe_request_digest=runtime.probe.request_digest,
        probe_provider_input_tokens=runtime.probe.provider_input_tokens,
        probe_provider_output_tokens=runtime.probe.provider_output_tokens,
        probe_latency_us=runtime.probe.latency_us,
        control_latency_us=runtime.control_latency_us,
        postflight_latency_us=postflight_latency,
        provider_input_tokens=diagnostic.provider_input_tokens,
        provider_output_tokens=diagnostic.provider_output_tokens,
        total_provider_input_tokens=(
            runtime.probe.provider_input_tokens + diagnostic.provider_input_tokens
        ),
        total_provider_output_tokens=(
            runtime.probe.provider_output_tokens + diagnostic.provider_output_tokens
        ),
        canonical_input_tokens=diagnostic.canonical_input_tokens,
        canonical_output_tokens=diagnostic.canonical_output_tokens,
        canonical_token_equivalents=diagnostic.canonical_token_equivalents,
        model_latency_us=diagnostic.model_latency_us,
        memory_mutations=diagnostic.memory_mutations,
        grounded_reminders=diagnostic.grounded_reminders,
        valid_silences=diagnostic.valid_silences,
        classification=loaded.manifest.classification,
    )


async def run_paper_two_phase_pilot(
    *,
    endpoint: str,
    model: str,
    output_path: str | os.PathLike[str],
    warmup: Literal["warm", "cold"] | AlgorithmWarmupPolicy = "warm",
    dependencies: PilotRuntimeDependencies | None = None,
) -> PaperTwoPhasePilotReport:
    """Run one pilot while exposing only stable, value-free failure classes."""

    failure: type[Exception] | None = None
    validation_code = None
    try:
        return await _run_paper_two_phase_pilot(
            endpoint=endpoint,
            model=model,
            output_path=output_path,
            warmup=warmup,
            dependencies=dependencies,
        )
    except PilotCommandError:
        failure = PilotCommandError
    except PilotRuntimeUnavailableError:
        failure = PilotRuntimeUnavailableError
    except PilotRuntimeConfigurationError:
        failure = PilotRuntimeConfigurationError
    except PilotEvidenceError:
        failure = PilotEvidenceError
    except ArtifactValidationError as error:
        validation_code = error.code
    except Exception:
        failure = PilotEvidenceError
    if validation_code is not None:
        raise ArtifactValidationError(validation_code)
    assert failure is not None
    raise failure()


def _validated_report(value: object) -> PaperTwoPhasePilotReport:
    try:
        if type(value) is not PaperTwoPhasePilotReport:
            raise TypeError
        checked = PaperTwoPhasePilotReport.model_validate_json(
            value.model_dump_json(warnings=False)
        )
        if checked != value:
            raise ValueError
        return checked
    except Exception:
        raise PilotCommandError() from None


def render_pilot_json(report: PaperTwoPhasePilotReport) -> str:
    checked = _validated_report(report)
    return canonical_json(checked.model_dump(mode="json", warnings=False)).decode("utf-8") + "\n"


def render_pilot_human(report: PaperTwoPhasePilotReport) -> str:
    checked = _validated_report(report)
    lines = (
        "Local paper-two-phase pilot complete",
        f"condition: {checked.condition.value}",
        f"diagnostic cases: {checked.cases}",
        f"diagnostic calls: {checked.calls}",
        f"memory mutations: {checked.memory_mutations}",
        f"grounded reminders: {checked.grounded_reminders}",
        f"valid silences: {checked.valid_silences}",
        f"canonical token equivalents: {checked.canonical_token_equivalents}",
        f"artifact digest: {checked.manifest_digest}",
        "classification: exploratory diagnostic (never confirmatory)",
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "CLI_PILOT_SCHEMA_VERSION",
    "PILOT_SUITE_ID",
    "PILOT_TRAJECTORY_DIGEST",
    "PILOT_TRAJECTORY_ID",
    "PaperTwoPhasePilotReport",
    "PilotCommandError",
    "PilotEvidenceError",
    "PilotRuntimeConfigurationError",
    "PilotRuntimeDependencies",
    "PilotRuntimeUnavailableError",
    "build_paper_two_phase_pilot_trajectory",
    "paper_two_phase_pilot_suite_digest",
    "render_pilot_human",
    "render_pilot_json",
    "run_paper_two_phase_pilot",
]
