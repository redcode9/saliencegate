from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel

from saliencegate.artifacts import algorithm_manifest as _algorithm
from saliencegate.artifacts import algorithm_projection as _projection
from saliencegate.artifacts.export import discover_revision
from saliencegate.artifacts.manifest import ArtifactClassification, RevisionEvidence
from saliencegate.artifacts.tree import (
    ArtifactDestinationError,
    ArtifactExistsError,
    ArtifactExportError,
    ClosedTreeDescriptor,
    ClosedTreeFileSpec,
    publish_closed_tree,
)
from saliencegate.domain import TrustLabel, canonical_json
from saliencegate.experiments.runner import Stage2ExperimentRunResult
from saliencegate.security import RedactionPolicy, Redactor

_MAX_MANIFEST_BYTES = 1024 * 1024
_DEFAULT_REDACTION_POLICY = RedactionPolicy()


def _validated_result(value: object) -> Stage2ExperimentRunResult:
    try:
        if type(value) is Stage2ExperimentRunResult:
            checked = Stage2ExperimentRunResult.model_validate_json(
                value.model_dump_json(warnings=False)
            )
            if checked == value:
                return checked
    except Exception:
        pass
    raise ArtifactExportError("algorithm result failed artifact-boundary validation")


def _validated_execution(value: object) -> _algorithm.AlgorithmExecutionAttestation:
    try:
        if type(value) is _algorithm.AlgorithmExecutionAttestation:
            checked = _algorithm.AlgorithmExecutionAttestation.model_validate_json(
                value.model_dump_json(warnings=False)
            )
            if checked == value:
                return checked
    except Exception:
        pass
    raise ArtifactExportError("algorithm execution failed artifact-boundary validation")


def _validated_revision(value: object) -> RevisionEvidence:
    try:
        if type(value) is RevisionEvidence:
            checked = RevisionEvidence.model_validate_json(value.model_dump_json(warnings=False))
            if checked == value:
                return checked
    except Exception:
        pass
    raise ArtifactExportError("revision evidence failed artifact-boundary validation")


def _all_inputs_are_synthetic(result: Stage2ExperimentRunResult) -> bool:
    return all(
        item.event_input.draft.trust_label is TrustLabel.SYNTHETIC_FIXTURE
        for item in result.trajectory.records
    )


def _assert_redacted(
    components: Mapping[_algorithm.AlgorithmArtifactComponentName, BaseModel],
    policy: RedactionPolicy,
) -> None:
    redactor = Redactor(
        literal_secrets=policy.literal_secrets,
        structured_field_names=policy.structured_field_names,
    )
    try:
        for component in components.values():
            payload = component.model_dump(mode="json", warnings=False)
            redacted = redactor.redact_payload(payload)
            if canonical_json(redacted.payload.root) != canonical_json(payload):
                raise ArtifactExportError("algorithm artifact contains non-redacted data")
    except ArtifactExportError:
        raise
    except Exception:
        raise ArtifactExportError("algorithm artifact redaction verification failed") from None


def _encode_components(
    components: Mapping[_algorithm.AlgorithmArtifactComponentName, BaseModel],
    counters: _algorithm.AlgorithmArtifactCounters,
) -> tuple[dict[str, bytes], tuple[_algorithm.AlgorithmArtifactComponent, ...]]:
    counts = {
        _algorithm.AlgorithmArtifactComponentName.ATTESTATIONS: 1,
        _algorithm.AlgorithmArtifactComponentName.CALLS: counters.model_calls,
        _algorithm.AlgorithmArtifactComponentName.CYCLES: counters.cycles,
        _algorithm.AlgorithmArtifactComponentName.DECISIONS: counters.decisions,
        _algorithm.AlgorithmArtifactComponentName.DELIVERIES: counters.deliveries,
        _algorithm.AlgorithmArtifactComponentName.METRICS: 1,
        _algorithm.AlgorithmArtifactComponentName.OUTCOMES: counters.outcomes,
        _algorithm.AlgorithmArtifactComponentName.RUN: 1,
        _algorithm.AlgorithmArtifactComponentName.TRAJECTORY: counters.events,
    }
    encoded: dict[str, bytes] = {}
    descriptors: list[_algorithm.AlgorithmArtifactComponent] = []
    for name in sorted(components, key=lambda item: item.value):
        data = canonical_json(components[name])
        if len(data) > _algorithm.MAX_ALGORITHM_COMPONENT_BYTES:
            raise ArtifactExportError("algorithm component exceeds its byte limit")
        path = _algorithm.expected_algorithm_component_path(name)
        encoded[path] = data
        descriptors.append(
            _algorithm.AlgorithmArtifactComponent(
                name=name,
                path=path,
                byte_count=len(data),
                record_count=counts[name],
                content_digest=_algorithm.algorithm_component_content_digest(name, data),
            )
        )
    return encoded, tuple(descriptors)


def _tree_descriptor(
    manifest: _algorithm.AlgorithmArtifactManifest,
) -> ClosedTreeDescriptor[
    _algorithm.AlgorithmArtifactManifest,
    _algorithm.AlgorithmArtifactComponentName,
]:
    files = tuple(
        ClosedTreeFileSpec(
            key=component.name,
            name=component.path,
            maximum_bytes=_algorithm.MAX_ALGORITHM_COMPONENT_BYTES,
            expected_bytes=component.byte_count,
        )
        for component in sorted(manifest.components, key=lambda item: item.path)
    )
    return ClosedTreeDescriptor(
        manifest=manifest,
        manifest_name="manifest.json",
        manifest_digest=manifest.manifest_digest,
        replacement_key=(f"algorithm_run:{manifest.run_id}:{manifest.condition_id.value}"),
        files=files,
    )


def _parse_manifest(
    data: bytes,
) -> ClosedTreeDescriptor[
    _algorithm.AlgorithmArtifactManifest,
    _algorithm.AlgorithmArtifactComponentName,
]:
    return _tree_descriptor(_algorithm.AlgorithmArtifactManifest.model_validate_json(data))


def _validate_tree(
    path: Path,
    expected_digest: str | None,
) -> ClosedTreeDescriptor[
    _algorithm.AlgorithmArtifactManifest,
    _algorithm.AlgorithmArtifactComponentName,
]:
    from saliencegate.artifacts.algorithm_validate import load_validated_algorithm_artifact

    loaded = load_validated_algorithm_artifact(
        path / "manifest.json",
        expected_manifest_digest=expected_digest,
    )
    return _tree_descriptor(loaded.manifest)


def _publish(
    destination: os.PathLike[str] | str,
    files: Mapping[str, bytes],
    *,
    replace: bool,
) -> None:
    publish_closed_tree(
        destination,
        files,
        manifest_name="manifest.json",
        maximum_manifest_bytes=_MAX_MANIFEST_BYTES,
        parse_manifest=_parse_manifest,
        validate_tree=_validate_tree,
        replace=replace,
    )


def export_algorithm_artifact(
    result: Stage2ExperimentRunResult,
    output: os.PathLike[str] | str,
    *,
    execution: _algorithm.AlgorithmExecutionAttestation,
    classification: _algorithm.AlgorithmArtifactClassification = (
        ArtifactClassification.SYNTHETIC_DIGEST_ONLY
    ),
    revision: RevisionEvidence | None = None,
    redaction_policy: RedactionPolicy = _DEFAULT_REDACTION_POLICY,
    replace: bool = False,
    source_dir: os.PathLike[str] | str | None = None,
) -> _algorithm.AlgorithmArtifactManifest:
    """Export one closed experiment result without exposing source-derived content by default."""

    checked_result = _validated_result(result)
    checked_execution = _validated_execution(execution)
    if type(classification) is not ArtifactClassification:
        raise ArtifactExportError("algorithm classification failed validation")
    if type(redaction_policy) is not RedactionPolicy:
        raise ArtifactExportError("algorithm redaction policy failed validation")
    if type(replace) is not bool:
        raise ArtifactExportError("algorithm replace flag failed validation")
    if not _all_inputs_are_synthetic(checked_result):
        raise ArtifactExportError("algorithm result classification failed validation")
    if classification is ArtifactClassification.USER_REDACTED:
        raise ArtifactExportError("synthetic algorithm result cannot be user classified")
    _projection._validate_source_execution_binding(checked_result, checked_execution)
    provenance = (
        discover_revision(source_dir) if revision is None else _validated_revision(revision)
    )
    try:
        component_models, outcomes = _projection._project_algorithm_components(
            checked_result,
            checked_execution,
            classification,
        )
        if classification is not ArtifactClassification.SYNTHETIC_RAW:
            _assert_redacted(component_models, redaction_policy)
        counters = _algorithm.AlgorithmArtifactCounters(
            events=len(checked_result.trajectory.records),
            scheduled_invocations=checked_result.schedule.invocation_count,
            decisions=len(checked_result.decisions),
            cycles=sum(boundary.cycle is not None for boundary in checked_result.boundaries),
            requests=sum(boundary.request is not None for boundary in checked_result.boundaries),
            model_calls=len(checked_result.call_receipts),
            deliveries=sum(
                boundary.delivery_record is not None for boundary in checked_result.boundaries
            ),
            outcomes=len(outcomes),
            ledger_entries=len(checked_result.ledger),
        )
        encoded, descriptors = _encode_components(component_models, counters)
        manifest = _algorithm.AlgorithmArtifactManifest.create(
            classification=classification,
            run_id=checked_result.run_id,
            revision=provenance,
            condition_id=checked_result.condition.condition_id,
            condition_digest=checked_result.condition.condition_digest,
            cycle_mode=_algorithm.algorithm_cycle_mode_for_condition(
                checked_result.condition.condition_id
            ),
            trace_digest=checked_result.trace_digest,
            schedule_digest=checked_result.schedule.schedule_digest,
            window_digests=tuple(window.window_digest for window in checked_result.windows),
            prompt_bundle_digest=checked_result.prompt_bundle.bundle_digest,
            model_profile_digest=checked_result.model_profile.profile_digest,
            execution=checked_execution,
            result_digest=checked_result.result_digest,
            components=descriptors,
            counters=counters,
        )
        if classification is not ArtifactClassification.SYNTHETIC_RAW:
            _assert_redacted(
                {_algorithm.AlgorithmArtifactComponentName.RUN: manifest},
                redaction_policy,
            )
        encoded["manifest.json"] = canonical_json(manifest)
    except ArtifactExportError:
        raise
    except Exception:
        raise ArtifactExportError("algorithm artifact construction failed") from None
    _publish(output, encoded, replace=replace)
    return manifest


__all__ = [
    "ArtifactDestinationError",
    "ArtifactExistsError",
    "ArtifactExportError",
    "export_algorithm_artifact",
]
