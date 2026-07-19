from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from saliencegate import __version__
from saliencegate.artifacts import (
    AlgorithmCheckpointAttestation,
    AlgorithmEndpointClassification,
    AlgorithmExecutionAttestation,
    AlgorithmExecutionMode,
    AlgorithmHardwareAttestation,
    AlgorithmResponseFixtureAttestation,
    AlgorithmSamplingAttestation,
    AlgorithmSamplingMode,
    AlgorithmTokenizerAttestation,
    AlgorithmTokenizerStatus,
    AlgorithmWarmupPolicy,
    ArtifactClassification,
    ArtifactExportError,
    ArtifactValidationError,
    RevisionEvidence,
    RevisionSource,
    discover_revision,
    export_algorithm_artifact,
    load_validated_algorithm_artifact,
)
from saliencegate.domain import canonical_json
from saliencegate.domain.records import Sha256Digest
from saliencegate.experiments import (
    Stage2ConditionId,
    Stage2ExperimentError,
    Stage2ExperimentRunResult,
    replay_stage2_fixture_twice,
)

CLI_ALGORITHM_REPLAY_SCHEMA_VERSION: Literal["cli-algorithm-replay-report/v1"] = (
    "cli-algorithm-replay-report/v1"
)
_COMMAND_ERROR = "algorithm replay input or output is invalid"
_NOT_APPLICABLE_REPLAY = "not-applicable-replay"


class AlgorithmReplayCommandError(ValueError):
    """A value-free algorithm replay command failure."""

    def __init__(self) -> None:
        super().__init__(_COMMAND_ERROR)


class AlgorithmReplayCommandReport(BaseModel):
    """Stable, value-minimized summary of a validated algorithm replay artifact."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    schema_version: Literal["cli-algorithm-replay-report/v1"] = CLI_ALGORITHM_REPLAY_SCHEMA_VERSION
    status: Literal["ok"] = "ok"
    condition: Stage2ConditionId
    run_id: str
    run_digest: Sha256Digest
    result_digest: Sha256Digest
    manifest_digest: Sha256Digest
    overall_content_digest: Sha256Digest
    calls: Annotated[int, Field(ge=0)]
    canonical_input_tokens: Annotated[int, Field(ge=0)] | None
    canonical_output_tokens: Annotated[int, Field(ge=0)] | None
    canonical_token_equivalents: Annotated[int, Field(ge=0)] | None
    interventions: Annotated[int, Field(ge=0)]
    grounding_rejections: Annotated[int, Field(ge=0)]
    classification: Literal[ArtifactClassification.SYNTHETIC_DIGEST_ONLY]
    confirmatory: Literal[False] = False


def _path(value: str | os.PathLike[str]) -> Path:
    try:
        raw = os.fspath(value)
        if type(raw) is not str or not raw:
            raise TypeError
        return Path(raw)
    except (OSError, TypeError, ValueError):
        raise AlgorithmReplayCommandError() from None


def _tokenizer_attestation(
    result: Stage2ExperimentRunResult,
) -> AlgorithmTokenizerAttestation:
    identities: set[tuple[str, str, str, str]] = set()
    for call in result.call_receipts:
        usage = call.usage
        has_counts = (
            usage.canonical_input_tokens is not None or usage.canonical_output_tokens is not None
        )
        if not has_counts:
            continue
        identity = (
            usage.local_counter_id,
            usage.local_counter_version,
            usage.local_counter_configuration_digest,
            usage.local_counter_model_id,
        )
        if any(value is None for value in identity):
            raise AlgorithmReplayCommandError()
        identities.add(cast(tuple[str, str, str, str], identity))

    if not identities:
        return AlgorithmTokenizerAttestation(
            status=AlgorithmTokenizerStatus.UNAVAILABLE,
            tokenizer_id=None,
            tokenizer_version=None,
            configuration_digest=None,
            model_id=None,
        )
    if len(identities) != 1:
        raise AlgorithmReplayCommandError()
    tokenizer_id, tokenizer_version, configuration_digest, model_id = identities.pop()
    return AlgorithmTokenizerAttestation(
        status=AlgorithmTokenizerStatus.ATTESTED,
        tokenizer_id=tokenizer_id,
        tokenizer_version=tokenizer_version,
        configuration_digest=configuration_digest,
        model_id=model_id,
    )


def _execution_attestation(
    result: Stage2ExperimentRunResult,
) -> AlgorithmExecutionAttestation:
    response_fixture = result.response_fixture
    fixture_attestation = (
        None
        if response_fixture is None
        else AlgorithmResponseFixtureAttestation(
            replay_id=response_fixture.replay_id,
            fixture_digest=response_fixture.fixture_digest,
            response_count=response_fixture.response_count,
            consumed_count=response_fixture.response_count,
        )
    )
    return AlgorithmExecutionAttestation.create(
        execution_mode=AlgorithmExecutionMode.FROZEN_REPLAY,
        endpoint_classification=AlgorithmEndpointClassification.OFFLINE_REPLAY,
        runtime_id="saliencegate-two-phase-replay",
        runtime_version=__version__,
        checkpoint=AlgorithmCheckpointAttestation(
            model_id=result.model_profile.model_id,
            model_tag="gpt-oss:20b-fixture/v1",
            checkpoint_digest=None,
            quantization=_NOT_APPLICABLE_REPLAY,
        ),
        sampling=AlgorithmSamplingAttestation(
            mode=AlgorithmSamplingMode.FROZEN_REPLAY,
            temperature=None,
            seed=None,
            reasoning_effort=None,
        ),
        tokenizer=_tokenizer_attestation(result),
        hardware=AlgorithmHardwareAttestation(
            model=_NOT_APPLICABLE_REPLAY,
            architecture=_NOT_APPLICABLE_REPLAY,
            logical_core_count=1,
            memory_capacity_bytes=1,
            operating_system=_NOT_APPLICABLE_REPLAY,
            operating_system_version=_NOT_APPLICABLE_REPLAY,
        ),
        warmup_policy=AlgorithmWarmupPolicy.NOT_APPLICABLE,
        response_fixture=fixture_attestation,
    )


def _replay_revision() -> RevisionEvidence:
    """Keep CLI replay deterministic without claiming a clean Git worktree."""

    revision = discover_revision()
    if revision.source is not RevisionSource.GIT:
        return revision
    return RevisionEvidence(
        source=RevisionSource.GIT,
        package_version=revision.package_version,
        commit=revision.commit,
        dirty_worktree=True,
    )


async def run_algorithm_replay(
    trajectory_path: str | os.PathLike[str],
    *,
    condition: Stage2ConditionId | str,
    output_path: str | os.PathLike[str],
    responses_path: str | os.PathLike[str] | None = None,
    replace: bool = False,
) -> AlgorithmReplayCommandReport:
    """Replay twice, publish once, and report only the validated artifact view."""

    command_failed = False
    validation_failure = None
    try:
        if type(replace) is not bool:
            raise AlgorithmReplayCommandError()
        trajectory = _path(trajectory_path)
        output = _path(output_path)
        responses = None if responses_path is None else _path(responses_path)
        result = await replay_stage2_fixture_twice(
            trajectory,
            condition=condition,
            responses_path=responses,
        )
        manifest = export_algorithm_artifact(
            result,
            output,
            execution=_execution_attestation(result),
            classification=ArtifactClassification.SYNTHETIC_DIGEST_ONLY,
            revision=_replay_revision(),
            replace=replace,
        )
        loaded = load_validated_algorithm_artifact(
            output / "manifest.json",
            expected_manifest_digest=manifest.manifest_digest,
        )
        if loaded.manifest.classification is not ArtifactClassification.SYNTHETIC_DIGEST_ONLY:
            raise AlgorithmReplayCommandError()
        metrics = loaded.metrics.metrics
        return AlgorithmReplayCommandReport(
            condition=loaded.manifest.condition_id,
            run_id=str(loaded.manifest.run_id),
            run_digest=loaded.run.run_component_digest,
            result_digest=loaded.manifest.result_digest,
            manifest_digest=loaded.manifest.manifest_digest,
            overall_content_digest=loaded.manifest.overall_content_digest,
            calls=metrics.model_call_count,
            canonical_input_tokens=metrics.canonical_input_tokens,
            canonical_output_tokens=metrics.canonical_output_tokens,
            canonical_token_equivalents=metrics.canonical_token_equivalents,
            interventions=metrics.intervention_count,
            grounding_rejections=metrics.grounding_rejection_count,
            classification=loaded.manifest.classification,
            confirmatory=loaded.report.confirmatory,
        )
    except AlgorithmReplayCommandError:
        command_failed = True
    except ArtifactValidationError as error:
        validation_failure = error.code
    except (Stage2ExperimentError, ArtifactExportError):
        command_failed = True
    if validation_failure is not None:
        raise ArtifactValidationError(validation_failure)
    assert command_failed
    raise AlgorithmReplayCommandError()


def _validated_report(value: object) -> AlgorithmReplayCommandReport:
    try:
        if type(value) is AlgorithmReplayCommandReport:
            checked = AlgorithmReplayCommandReport.model_validate_json(
                value.model_dump_json(warnings=False)
            )
            if checked == value:
                return checked
    except Exception:
        pass
    raise AlgorithmReplayCommandError()


def render_algorithm_replay_json(report: AlgorithmReplayCommandReport) -> str:
    """Render one canonical JSON line."""

    return canonical_json(_validated_report(report)).decode("utf-8") + "\n"


def _token_count(value: int | None) -> str:
    return "unavailable" if value is None else str(value)


def render_algorithm_replay_human(report: AlgorithmReplayCommandReport) -> str:
    """Render a stable human-readable replay summary."""

    checked = _validated_report(report)
    return (
        "Algorithm replay complete\n"
        f"condition: {checked.condition.value}\n"
        f"run: {checked.run_id}\n"
        f"calls: {checked.calls}\n"
        "canonical tokens: "
        f"{_token_count(checked.canonical_input_tokens)} input, "
        f"{_token_count(checked.canonical_output_tokens)} output, "
        f"{_token_count(checked.canonical_token_equivalents)} total\n"
        f"interventions: {checked.interventions}\n"
        f"grounding rejections: {checked.grounding_rejections}\n"
        f"classification: {checked.classification.value}\n"
        "confirmatory: no\n"
        f"run digest: {checked.run_digest}\n"
        f"result digest: {checked.result_digest}\n"
        f"manifest digest: {checked.manifest_digest}\n"
        f"content digest: {checked.overall_content_digest}\n"
    )


__all__ = [
    "CLI_ALGORITHM_REPLAY_SCHEMA_VERSION",
    "AlgorithmReplayCommandError",
    "AlgorithmReplayCommandReport",
    "render_algorithm_replay_human",
    "render_algorithm_replay_json",
    "run_algorithm_replay",
]
