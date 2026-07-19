from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from saliencegate.adapters import JSONLReplayAdapter, JsonlReplayError
from saliencegate.artifacts import (
    ArtifactClassification,
    ArtifactCounters,
    ArtifactDestinationError,
    ArtifactEvidenceLevel,
    ArtifactValidationError,
    export_replay_artifact,
    validate_artifact,
)
from saliencegate.domain import (
    BudgetAmounts,
    BudgetLimits,
    DeliveryTarget,
    TrustLabel,
    canonical_json,
)
from saliencegate.intervention import GroundingConfig, GroundingPipeline, RenderingConfig
from saliencegate.models import ReplayError, ReplayModel
from saliencegate.policy import ScriptedPolicy, ScriptedPolicyConfig
from saliencegate.ports.adapters import (
    AdapterCapabilities,
    AdapterDeliveryFailedError,
    DeduplicationGuarantee,
    DeliveryChannel,
    DeliveryEnvelope,
    DeliveryReceipt,
    DeliveryRole,
    InjectionMapping,
)
from saliencegate.repository import MemoryRunRepository
from saliencegate.runtime import BatchConfig
from saliencegate.runtime.engine import (
    ReplayEngine,
    ReplayEngineConfig,
    ReplayEngineInputError,
    ReplayEngineModelError,
)
from saliencegate.security import (
    InsecureKeyFileError,
    InsecureKeyPathError,
    InvalidInstallationKeyError,
)
from saliencegate.signals import DeterministicSignalExtractor

CLI_REPLAY_SCHEMA_VERSION: Literal["cli-replay-report/v1"] = "cli-replay-report/v1"
REPLAY_PROMPT_TEMPLATE_DIGEST = "a" * 64
REPLAY_MODEL_ID = "replay-fixture/1"
REPLAY_ADAPTER_ID = "engine-fixture/1"


class ReplayCommandError(ValueError):
    """A value-free invalid replay input or output error."""

    def __init__(self) -> None:
        super().__init__("replay input or output is invalid")


class ReplayConfigurationError(RuntimeError):
    """A value-free invalid local replay configuration error."""

    def __init__(self) -> None:
        super().__init__("replay configuration is invalid")


class ReplayCommandReport(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )

    schema_version: Literal["cli-replay-report/v1"] = CLI_REPLAY_SCHEMA_VERSION
    status: Literal["ok"] = "ok"
    run_id: str
    trace_digest: str
    result_digest: str
    classification: ArtifactClassification
    confirmatory: bool
    manifest_digest: str
    overall_content_digest: str
    counters: ArtifactCounters


class _LocalFailingDeliveryAdapter:
    """A deterministic sink that proves delivery behavior without external side effects."""

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            schema_version="1.0",
            adapter_id=REPLAY_ADAPTER_ID,
            pre_action_interception=False,
            deduplicates_delivery_id=True,
            deduplication_guarantee=DeduplicationGuarantee.DURABLE_DELIVERY_ID,
            injection_mappings=(
                InjectionMapping(
                    channel=DeliveryChannel.PROVIDER_DATA,
                    role=DeliveryRole.DATA,
                    provider_channel="context",
                ),
            ),
        )

    async def deliver(self, delivery: DeliveryEnvelope) -> DeliveryReceipt:
        del delivery
        raise AdapterDeliveryFailedError()


def _grounding_pipeline() -> GroundingPipeline:
    rendering = RenderingConfig(
        schema_version="1.0",
        renderer_version="fixed-ascii/v1",
        token_counter_version="utf8-bytes-ceil-div-4-v1",
        max_claims=2,
        max_evidence_bytes=1_024,
        max_output_bytes=4_096,
        max_token_equivalents=1_024,
        include_provenance=False,
    )
    return GroundingPipeline(
        GroundingConfig(
            schema_version="1.0",
            pipeline_version="grounding-pipeline/v1",
            claim_schema_version="citation-only-claims/v1",
            max_claims=2,
            max_evidence_per_claim=1,
            max_pointer_segments=32,
            max_pointer_utf8_bytes=1_024,
            duplicate_window_events=0,
            cooldown_events=0,
            ttl_steps=1,
            allowed_delivery_targets=(DeliveryTarget.NEXT_MODEL_CALL,),
            rendering=rendering,
        )
    )


def _engine_config() -> ReplayEngineConfig:
    return ReplayEngineConfig(
        model_id=REPLAY_MODEL_ID,
        prompt_template_digest=REPLAY_PROMPT_TEMPLATE_DIGEST,
        budget_limits=BudgetLimits(
            model_calls=10,
            input_tokens=10_000,
            output_tokens=10_000,
            canonical_token_equivalents=20_000,
            latency_us=1_000_000,
            interventions=10,
            schema_repairs=2,
            max_call_latency_us=100_000,
        ),
        reservation=BudgetAmounts(
            model_calls=1,
            input_tokens=1_000,
            output_tokens=1_000,
            canonical_token_equivalents=2_000,
            latency_us=100_000,
            interventions=1,
            schema_repairs=1,
        ),
        batch=BatchConfig(
            max_utf8_bytes=32_000,
            max_approximate_tokens=8_000,
            recent_event_count=4,
            max_controller_errors=4,
            max_action_proposals=4,
            max_tool_errors=4,
            max_test_failures=4,
            max_conflicts=4,
        ),
        requested_delivery_target=DeliveryTarget.NEXT_MODEL_CALL,
    )


def _path(value: str | os.PathLike[str]) -> Path:
    try:
        raw = os.fspath(value)
        if type(raw) is not str or not raw:
            raise TypeError
        return Path(raw)
    except (OSError, TypeError, ValueError):
        raise ReplayCommandError() from None


def _response_fixture(trace_path: Path, supplied: str | os.PathLike[str] | None) -> Path:
    if supplied is not None:
        return _path(supplied)
    local = trace_path.with_name(f"{trace_path.stem}_responses.jsonl")
    candidates = [local]
    if trace_path.parent.name == "runs":
        candidates.append(trace_path.parent.parent / "models" / local.name)
    existing: list[Path] = []
    for candidate in candidates:
        try:
            candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            raise ReplayCommandError() from None
        existing.append(candidate)
    if len(existing) != 1:
        raise ReplayCommandError()
    return existing[0]


async def run_replay(
    trace_path: str | os.PathLike[str],
    *,
    output_path: str | os.PathLike[str],
    responses_path: str | os.PathLike[str] | None = None,
    replace: bool = False,
) -> ReplayCommandReport:
    """Run the frozen deterministic profile without network or live-model access."""

    try:
        trace_source = _path(trace_path)
        output = _path(output_path)
        responses = _response_fixture(trace_source, responses_path)
        trace = JSONLReplayAdapter.from_path(trace_source)
        model = ReplayModel.from_path(responses)
        event_count = len(trace.events)
        response_count = model.total_responses
        if response_count > event_count:
            raise ReplayCommandError()
        decisions = (False,) * (event_count - response_count) + (True,) * response_count
        synthetic = all(
            event.draft.trust_label is TrustLabel.SYNTHETIC_FIXTURE for event in trace.events
        )
        try:
            repository = MemoryRunRepository(synthetic_benchmark=synthetic)
        except (
            InsecureKeyFileError,
            InsecureKeyPathError,
            InvalidInstallationKeyError,
            OSError,
        ):
            raise ReplayConfigurationError() from None
        engine = ReplayEngine(
            repository=repository,
            adapter=trace,
            extractor=DeterministicSignalExtractor(()),
            policy=ScriptedPolicy(
                ScriptedPolicyConfig(
                    schema_version="1.0",
                    policy_kind="scripted",
                    decisions=decisions,
                    on_exhaustion="silence",
                )
            ),
            model=model,
            grounding=_grounding_pipeline(),
            config=_engine_config(),
            delivery_adapter=_LocalFailingDeliveryAdapter(),
        )
        result = await engine.run(trace.events, trace_digest=trace.trace_digest)
        classification = (
            ArtifactClassification.SYNTHETIC_DIGEST_ONLY
            if synthetic
            else ArtifactClassification.USER_REDACTED
        )
        manifest = export_replay_artifact(
            result,
            output,
            classification=classification,
            evidence_level=ArtifactEvidenceLevel.EXPLORATORY,
            replace=replace,
        )
        validation = validate_artifact(
            output / "manifest.json",
            expected_manifest_digest=manifest.manifest_digest,
        )
        return ReplayCommandReport(
            run_id=str(result.run_id),
            trace_digest=result.trace_digest,
            result_digest=result.result_digest,
            classification=manifest.classification,
            confirmatory=validation.confirmatory,
            manifest_digest=manifest.manifest_digest,
            overall_content_digest=manifest.overall_content_digest,
            counters=manifest.counters,
        )
    except ReplayCommandError:
        raise
    except ReplayConfigurationError:
        raise
    except ArtifactValidationError:
        raise
    except (
        ArtifactDestinationError,
        JsonlReplayError,
        ReplayError,
        ReplayEngineInputError,
        ReplayEngineModelError,
    ):
        raise ReplayCommandError() from None


def render_replay_json(report: ReplayCommandReport) -> str:
    validated = ReplayCommandReport.model_validate(report)
    return canonical_json(validated.model_dump(mode="json", warnings=False)).decode("utf-8") + "\n"


def render_replay_human(report: ReplayCommandReport) -> str:
    validated = ReplayCommandReport.model_validate(report)
    return (
        "Replay complete\n"
        f"run: {validated.run_id}\n"
        f"events: {validated.counters.events}\n"
        f"cycles: {validated.counters.cycles}\n"
        f"manifest digest: {validated.manifest_digest}\n"
    )


__all__ = [
    "CLI_REPLAY_SCHEMA_VERSION",
    "ReplayCommandError",
    "ReplayCommandReport",
    "ReplayConfigurationError",
    "render_replay_human",
    "render_replay_json",
    "run_replay",
]
