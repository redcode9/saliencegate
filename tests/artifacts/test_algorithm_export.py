from __future__ import annotations

import stat
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from saliencegate.artifacts import (
    algorithm_export as exporter,
)
from saliencegate.artifacts import (
    algorithm_manifest,
    export_algorithm_artifact,
    load_validated_algorithm_artifact,
    validate_algorithm_artifact,
)
from saliencegate.artifacts import (
    algorithm_projection as projection,
)
from saliencegate.artifacts.algorithm_manifest import (
    AlgorithmCheckpointAttestation,
    AlgorithmEndpointClassification,
    AlgorithmExecutionAttestation,
    AlgorithmExecutionMode,
    AlgorithmResponseFixtureAttestation,
    AlgorithmSamplingAttestation,
    AlgorithmSamplingMode,
    AlgorithmTokenizerAttestation,
    AlgorithmTokenizerStatus,
    AlgorithmWarmupPolicy,
)
from saliencegate.artifacts.manifest import (
    ArtifactClassification,
    RevisionEvidence,
)
from saliencegate.artifacts.tree import (
    ArtifactDestinationError,
    ArtifactExistsError,
    ArtifactExportError,
)
from saliencegate.domain import (
    DeliveryState,
    PayloadDigest,
    PayloadDigestAlgorithm,
    canonical_json,
)
from saliencegate.experiments import Stage2BoundaryEvidence, Stage2ExperimentRunResult
from saliencegate.ports.model_calls import (
    CanonicalUsageProvenance,
    ProviderUsageProvenance,
    StructuredCallParseStatus,
    StructuredCallStatus,
    StructuredCallUsage,
)
from saliencegate.ports.two_phase import CallReceipt
from saliencegate.security import RedactionPolicy

RAW_SENTINEL = b"Complete final verification before reporting the release workflow finished."
_UNCHANGED = object()


def _tree_bytes(path: Path) -> dict[str, bytes]:
    return {item.name: item.read_bytes() for item in sorted(path.iterdir())}


def _frozen_execution(
    base: AlgorithmExecutionAttestation,
    *,
    checkpoint: AlgorithmCheckpointAttestation | None = None,
    tokenizer: AlgorithmTokenizerAttestation | None = None,
    response_fixture: AlgorithmResponseFixtureAttestation | None,
) -> AlgorithmExecutionAttestation:
    return AlgorithmExecutionAttestation.create(
        endpoint_classification=AlgorithmEndpointClassification.OFFLINE_REPLAY,
        runtime_id=base.runtime_id,
        runtime_version=base.runtime_version,
        checkpoint=base.checkpoint if checkpoint is None else checkpoint,
        sampling=base.sampling,
        tokenizer=base.tokenizer if tokenizer is None else tokenizer,
        hardware=base.hardware,
        warmup_policy=AlgorithmWarmupPolicy.NOT_APPLICABLE,
        execution_mode=AlgorithmExecutionMode.FROZEN_REPLAY,
        response_fixture=response_fixture,
    )


def _live_execution(
    base: AlgorithmExecutionAttestation,
    *,
    tokenizer: AlgorithmTokenizerAttestation | None = None,
) -> AlgorithmExecutionAttestation:
    return AlgorithmExecutionAttestation.create(
        endpoint_classification=AlgorithmEndpointClassification.LOOPBACK_OPENAI_COMPATIBLE,
        runtime_id="saliencegate-openai-compatible",
        runtime_version="1.0.0",
        checkpoint=base.checkpoint,
        sampling=AlgorithmSamplingAttestation(
            mode=AlgorithmSamplingMode.OPENAI_COMPATIBLE,
            temperature=0.0,
            seed=7,
            reasoning_effort="medium",
        ),
        tokenizer=base.tokenizer if tokenizer is None else tokenizer,
        hardware=base.hardware,
        warmup_policy=AlgorithmWarmupPolicy.COLD,
        execution_mode=AlgorithmExecutionMode.OPENAI_COMPATIBLE,
        response_fixture=None,
    )


def _live_usage(
    result: Stage2ExperimentRunResult,
    *,
    canonical_output_tokens: object = _UNCHANGED,
    configuration_digest: str | None = None,
) -> StructuredCallUsage:
    values = result.call_receipts[0].usage.model_dump(mode="python", warnings=False)
    values["canonical_usage_provenance"] = CanonicalUsageProvenance.LOCAL_COUNTER
    if canonical_output_tokens is not _UNCHANGED:
        values["canonical_output_tokens"] = canonical_output_tokens
    if configuration_digest is not None:
        values["local_counter_configuration_digest"] = configuration_digest
    return StructuredCallUsage.model_validate(values)


def _live_binding_result(
    result: Stage2ExperimentRunResult,
    usage: StructuredCallUsage,
    *,
    status: StructuredCallStatus = StructuredCallStatus.COMPLETED,
    completion_algorithm: PayloadDigestAlgorithm | None = PayloadDigestAlgorithm.HMAC_SHA256,
) -> Stage2ExperimentRunResult:
    completion = (
        None
        if completion_algorithm is None
        else PayloadDigest(algorithm=completion_algorithm, value="a" * 64)
    )
    receipt_values = result.call_receipts[0].model_dump(
        mode="python",
        exclude={"receipt_digest"},
    )
    receipt_values.update(
        usage=usage,
        completion_digest=completion,
        completion_byte_count=(
            receipt_values["completion_byte_count"] if completion is not None else None
        ),
        status=status,
        parse_status=(
            receipt_values["parse_status"]
            if status is StructuredCallStatus.COMPLETED
            else StructuredCallParseStatus.NOT_ATTEMPTED
        ),
    )
    value = SimpleNamespace(
        call_receipts=(CallReceipt.model_validate(receipt_values),),
        model_profile=result.model_profile,
        response_fixture=None,
    )
    return cast(Stage2ExperimentRunResult, value)


def _first_active_boundary(result: Stage2ExperimentRunResult) -> Stage2BoundaryEvidence:
    return next(boundary for boundary in result.boundaries if boundary.cycle is not None)


def _delivery_boundary(result: Stage2ExperimentRunResult) -> Stage2BoundaryEvidence:
    return next(boundary for boundary in result.boundaries if boundary.delivery_record is not None)


def test_export_is_deterministic_canonical_and_owner_only(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = export_algorithm_artifact(
        fixed_stage2_result,
        first,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    second_manifest = export_algorithm_artifact(
        fixed_stage2_result,
        second,
        execution=algorithm_execution,
        revision=clean_revision,
    )

    assert first_manifest == second_manifest
    assert _tree_bytes(first) == _tree_bytes(second)
    assert set(_tree_bytes(first)) == {
        "attestations.json",
        "calls.json",
        "cycles.json",
        "decisions.json",
        "deliveries.json",
        "manifest.json",
        "metrics.json",
        "outcomes.json",
        "run.json",
        "trajectory.json",
    }
    assert stat.S_IMODE(first.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(item.stat().st_mode) == 0o600 for item in first.iterdir())
    assert first.joinpath("manifest.json").read_bytes() == canonical_json(first_manifest)
    assert all(not data.endswith(b"\n") for data in _tree_bytes(first).values())


def test_digest_only_export_contains_no_source_prompt_or_completion_text(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    destination = tmp_path / "redacted"

    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
        classification=ArtifactClassification.SYNTHETIC_DIGEST_ONLY,
    )

    payload = b"\n".join(_tree_bytes(destination).values())
    assert RAW_SENTINEL not in payload
    assert b'"payload"' not in payload
    assert b'"rendered_text"' not in payload
    assert b'"receipt"' not in payload
    assert b'"reasoning"' not in payload
    assert b'"endpoint_url"' not in payload
    assert b'"credential"' not in payload


def test_synthetic_raw_is_explicit_and_never_confirmatory(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    destination = tmp_path / "raw"

    manifest = export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
        classification=ArtifactClassification.SYNTHETIC_RAW,
    )

    assert not manifest.confirmatory
    assert not manifest.confirmatory_eligible
    assert RAW_SENTINEL in destination.joinpath("attestations.json").read_bytes()


def test_synthetic_result_cannot_be_promoted_to_user_classification(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    with pytest.raises(ArtifactExportError, match="synthetic"):
        export_algorithm_artifact(
            fixed_stage2_result,
            tmp_path / "misclassified",
            execution=algorithm_execution,
            revision=clean_revision,
            classification=ArtifactClassification.USER_REDACTED,
        )


def test_export_refuses_accidental_overwrite_and_replaces_only_same_run(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    destination = tmp_path / "artifact"
    manifest = export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )

    with pytest.raises(ArtifactExistsError):
        export_algorithm_artifact(
            fixed_stage2_result,
            destination,
            execution=algorithm_execution,
            revision=clean_revision,
        )

    replaced = export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
        replace=True,
    )
    assert replaced == manifest


def test_export_requires_complete_response_fixture_attestation_for_replay_calls(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    values = algorithm_execution.model_dump(
        mode="python",
        exclude={"schema_version", "execution_digest", "response_fixture"},
    )
    execution = AlgorithmExecutionAttestation.create(**values, response_fixture=None)

    with pytest.raises(ArtifactExportError, match="execution"):
        export_algorithm_artifact(
            fixed_stage2_result,
            tmp_path / "artifact",
            execution=execution,
            revision=clean_revision,
        )


def test_export_boundary_revalidates_exact_models(
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    with pytest.raises(ArtifactExportError, match="result"):
        exporter._validated_result(object())
    with pytest.raises(ArtifactExportError, match="execution"):
        exporter._validated_execution(object())
    with pytest.raises(ArtifactExportError, match="revision"):
        exporter._validated_revision(object())

    malformed_result = fixed_stage2_result.model_copy(update={"run_id": object()})
    malformed_execution = algorithm_execution.model_copy(update={"runtime_id": object()})
    malformed_revision = clean_revision.model_copy(update={"package_version": object()})
    with pytest.raises(ArtifactExportError, match="result"):
        exporter._validated_result(malformed_result)
    with pytest.raises(ArtifactExportError, match="execution"):
        exporter._validated_execution(malformed_execution)
    with pytest.raises(ArtifactExportError, match="revision"):
        exporter._validated_revision(malformed_revision)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("classification", "synthetic_digest_only", "classification"),
        ("redaction_policy", None, "redaction policy"),
        ("replace", 1, "replace flag"),
    ),
)
def test_export_rejects_coerced_control_values(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    keyword: str,
    value: object,
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "execution": algorithm_execution,
        "revision": clean_revision,
        keyword: value,
    }

    with pytest.raises(ArtifactExportError, match=message):
        export_algorithm_artifact(
            fixed_stage2_result,
            tmp_path / keyword,
            **arguments,  # type: ignore[arg-type]
        )


def test_export_rejects_non_synthetic_sources_before_construction(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(exporter, "_all_inputs_are_synthetic", lambda _result: False)

    with pytest.raises(ArtifactExportError, match="classification"):
        export_algorithm_artifact(
            fixed_stage2_result,
            tmp_path / "not-synthetic",
            execution=algorithm_execution,
            revision=clean_revision,
        )
    assert not (tmp_path / "not-synthetic").exists()


def test_export_rejects_execution_model_and_fixture_count_mismatches(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    other_checkpoint = AlgorithmCheckpointAttestation(
        model_id="gpt-oss:other",
        model_tag="gpt-oss:other-fixture/v1",
        checkpoint_digest=None,
        quantization="not-applicable-replay",
    )
    other_tokenizer = AlgorithmTokenizerAttestation(
        status=AlgorithmTokenizerStatus.ATTESTED,
        tokenizer_id="stage2-reviewed-utf8-counter/v1",
        tokenizer_version="utf8-bytes-ceil-div-4/v1",
        configuration_digest="4" * 64,
        model_id="gpt-oss:other",
    )
    wrong_model = _frozen_execution(
        algorithm_execution,
        checkpoint=other_checkpoint,
        tokenizer=other_tokenizer,
        response_fixture=algorithm_execution.response_fixture,
    )
    short_fixture = AlgorithmResponseFixtureAttestation(
        replay_id="short-replay/v1",
        fixture_digest="5" * 64,
        response_count=1,
        consumed_count=1,
    )
    wrong_count = _frozen_execution(
        algorithm_execution,
        response_fixture=short_fixture,
    )
    fixture = algorithm_execution.response_fixture
    assert fixture is not None
    wrong_digest = _frozen_execution(
        algorithm_execution,
        response_fixture=AlgorithmResponseFixtureAttestation(
            replay_id=fixture.replay_id,
            fixture_digest="f" * 64,
            response_count=fixture.response_count,
            consumed_count=fixture.consumed_count,
        ),
    )

    with pytest.raises(ArtifactExportError, match="model binding"):
        export_algorithm_artifact(
            fixed_stage2_result,
            tmp_path / "wrong-model",
            execution=wrong_model,
            revision=clean_revision,
        )
    with pytest.raises(ArtifactExportError, match="fixture binding"):
        export_algorithm_artifact(
            fixed_stage2_result,
            tmp_path / "wrong-count",
            execution=wrong_count,
            revision=clean_revision,
        )
    with pytest.raises(ArtifactExportError, match="fixture binding"):
        export_algorithm_artifact(
            fixed_stage2_result,
            tmp_path / "wrong-digest",
            execution=wrong_digest,
            revision=clean_revision,
        )


def test_no_memory_export_requires_no_response_fixture(
    tmp_path: Path,
    no_memory_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    with pytest.raises(ArtifactExportError, match="fixture binding"):
        export_algorithm_artifact(
            no_memory_stage2_result,
            tmp_path / "unexpected-fixture",
            execution=algorithm_execution,
            revision=clean_revision,
        )

    execution = _frozen_execution(
        algorithm_execution,
        response_fixture=None,
    )
    manifest = export_algorithm_artifact(
        no_memory_stage2_result,
        tmp_path / "no-memory",
        execution=execution,
        revision=clean_revision,
    )

    assert manifest.counters.cycles == 0
    assert manifest.counters.model_calls == 0
    assert manifest.counters.deliveries == 0
    assert manifest.counters.outcomes == 0


def test_openai_compatible_execution_rejects_hidden_replay_evidence(
    tmp_path: Path,
    replay_wrapped_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    execution = _live_execution(algorithm_execution)

    with pytest.raises(ArtifactExportError, match="call evidence"):
        export_algorithm_artifact(
            replay_wrapped_stage2_result,
            tmp_path / "replay-as-live",
            execution=execution,
            revision=clean_revision,
        )
    assert not (tmp_path / "replay-as-live").exists()


def test_openai_compatible_public_artifact_round_trip_uses_native_live_evidence(
    tmp_path: Path,
    live_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    destination = tmp_path / "live-artifact"
    execution = _live_execution(algorithm_execution)

    manifest = export_algorithm_artifact(
        live_stage2_result,
        destination,
        execution=execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(
        destination / "manifest.json",
        expected_manifest_digest=manifest.manifest_digest,
    )
    report = validate_algorithm_artifact(
        destination / "manifest.json",
        expected_manifest_digest=manifest.manifest_digest,
    )

    assert live_stage2_result.response_fixture is None
    assert manifest.execution == execution
    assert manifest.execution.execution_mode is AlgorithmExecutionMode.OPENAI_COMPATIBLE
    assert loaded.manifest == manifest
    assert loaded.run.execution == execution
    assert loaded.calls.calls == live_stage2_result.call_receipts
    assert all(
        call.completion_digest is not None
        and call.completion_digest.algorithm is PayloadDigestAlgorithm.HMAC_SHA256
        and call.usage.provider_usage_provenance is ProviderUsageProvenance.PROVIDER_REPORTED
        and call.usage.canonical_usage_provenance is CanonicalUsageProvenance.LOCAL_COUNTER
        for call in loaded.calls.calls
    )
    assert report.valid
    assert report.expected_digest_matched


def test_openai_compatible_execution_accepts_native_call_evidence(
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
) -> None:
    result = _live_binding_result(
        fixed_stage2_result,
        _live_usage(fixed_stage2_result),
    )

    projection._validate_source_execution_binding(result, _live_execution(algorithm_execution))


def test_export_binds_canonical_usage_to_the_attested_tokenizer(
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
) -> None:
    mismatched = AlgorithmTokenizerAttestation(
        status=AlgorithmTokenizerStatus.ATTESTED,
        tokenizer_id=algorithm_execution.tokenizer.tokenizer_id,
        tokenizer_version=algorithm_execution.tokenizer.tokenizer_version,
        configuration_digest="9" * 64,
        model_id=algorithm_execution.tokenizer.model_id,
    )
    unavailable = AlgorithmTokenizerAttestation(
        status=AlgorithmTokenizerStatus.UNAVAILABLE,
        tokenizer_id=None,
        tokenizer_version=None,
        configuration_digest=None,
        model_id=None,
    )

    result = _live_binding_result(
        fixed_stage2_result,
        _live_usage(fixed_stage2_result),
    )
    for tokenizer in (mismatched, unavailable):
        with pytest.raises(ArtifactExportError, match="tokenizer binding"):
            projection._validate_source_execution_binding(
                result,
                _live_execution(algorithm_execution, tokenizer=tokenizer),
            )


def test_tokenizer_binding_allows_calls_without_canonical_usage(
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
) -> None:
    call = fixed_stage2_result.call_receipts[0]
    usage_values = call.usage.model_dump(mode="python", warnings=False)
    usage_values.update(
        {
            "canonical_input_tokens": None,
            "canonical_output_tokens": None,
            "canonical_usage_provenance": CanonicalUsageProvenance.UNAVAILABLE,
            "local_counter_id": None,
            "local_counter_version": None,
            "local_counter_configuration_digest": None,
            "local_counter_model_id": None,
        }
    )
    usage_without_canonical_counts = StructuredCallUsage.model_validate(usage_values)
    result = _live_binding_result(
        fixed_stage2_result,
        usage_without_canonical_counts,
    )

    projection._validate_source_execution_binding(
        result,
        _live_execution(algorithm_execution),
    )
    unavailable = AlgorithmTokenizerAttestation(
        status=AlgorithmTokenizerStatus.UNAVAILABLE,
        tokenizer_id=None,
        tokenizer_version=None,
        configuration_digest=None,
        model_id=None,
    )
    projection._validate_source_execution_binding(
        result,
        _live_execution(algorithm_execution, tokenizer=unavailable),
    )


def test_partial_canonical_usage_is_bound_to_its_tokenizer_identity(
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
) -> None:
    matching = _live_binding_result(
        fixed_stage2_result,
        _live_usage(fixed_stage2_result, canonical_output_tokens=None),
        status=StructuredCallStatus.MODEL_ERROR,
        completion_algorithm=None,
    )
    mismatched = _live_binding_result(
        fixed_stage2_result,
        _live_usage(
            fixed_stage2_result,
            canonical_output_tokens=None,
            configuration_digest="9" * 64,
        ),
        status=StructuredCallStatus.MODEL_ERROR,
        completion_algorithm=None,
    )
    execution = _live_execution(algorithm_execution)

    projection._validate_source_execution_binding(matching, execution)
    with pytest.raises(ArtifactExportError, match="tokenizer binding"):
        projection._validate_source_execution_binding(mismatched, execution)


def test_openai_compatible_execution_requires_hmac_completion_digests(
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
) -> None:
    result = _live_binding_result(
        fixed_stage2_result,
        _live_usage(fixed_stage2_result),
        completion_algorithm=PayloadDigestAlgorithm.SYNTHETIC_SHA256,
    )

    with pytest.raises(ArtifactExportError, match="call evidence"):
        projection._validate_source_execution_binding(result, _live_execution(algorithm_execution))


@pytest.mark.parametrize("provenance", ("provider", "canonical"))
def test_openai_compatible_execution_rejects_each_replay_provenance(
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    provenance: str,
) -> None:
    values = _live_usage(fixed_stage2_result).model_dump(mode="python", warnings=False)
    if provenance == "provider":
        values.update(
            {
                "provider_input_tokens": 1,
                "provider_output_tokens": 1,
                "provider_usage_provenance": ProviderUsageProvenance.REPLAY_ATTESTED,
            }
        )
    else:
        values["canonical_usage_provenance"] = CanonicalUsageProvenance.REPLAY_ATTESTED
    result = _live_binding_result(
        fixed_stage2_result,
        StructuredCallUsage.model_validate(values),
    )

    with pytest.raises(ArtifactExportError, match="call evidence"):
        projection._validate_source_execution_binding(result, _live_execution(algorithm_execution))


def test_call_projection_rejects_incomplete_groups_and_global_reordering(
    fixed_stage2_result: Stage2ExperimentRunResult,
) -> None:
    boundary = _first_active_boundary(fixed_stage2_result)
    incomplete = boundary.model_copy(update={"request": None})
    incomplete_result = fixed_stage2_result.model_copy(
        update={
            "boundaries": tuple(
                incomplete if item is boundary else item for item in fixed_stage2_result.boundaries
            )
        }
    )
    reordered = fixed_stage2_result.model_copy(
        update={"call_receipts": tuple(reversed(fixed_stage2_result.call_receipts))}
    )

    with pytest.raises(ArtifactExportError, match="call group"):
        projection._calls_component(incomplete_result)
    with pytest.raises(ArtifactExportError, match="call ordering"):
        projection._calls_component(reordered)


def test_call_projection_rejects_multiple_grounding_receipts(
    fixed_stage2_result: Stage2ExperimentRunResult,
) -> None:
    boundary = _first_active_boundary(fixed_stage2_result)
    assert len(boundary.call_receipts) == 2
    grounded_digest = next(
        call.grounding_state_digest
        for call in boundary.call_receipts
        if call.grounding_state_digest is not None
    )
    calls = tuple(
        call.model_copy(update={"grounding_state_digest": grounded_digest})
        for call in boundary.call_receipts
    )
    duplicated = boundary.model_copy(update={"call_receipts": calls})
    result = fixed_stage2_result.model_copy(
        update={
            "boundaries": tuple(
                duplicated if item is boundary else item for item in fixed_stage2_result.boundaries
            )
        }
    )

    with pytest.raises(ArtifactExportError, match="call grounding"):
        projection._calls_component(result)


def test_cycle_projection_rejects_incomplete_sources_and_invalid_grounding(
    fixed_stage2_result: Stage2ExperimentRunResult,
) -> None:
    boundary = _first_active_boundary(fixed_stage2_result)

    with pytest.raises(ArtifactExportError, match="cycle"):
        projection._cycle_attestation(boundary.model_copy(update={"request": None}))

    assert boundary.cycle is not None
    assert boundary.cycle.intervention is not None
    intervention = boundary.cycle.intervention.model_copy(update={"grounding_receipt": {}})
    cycle = boundary.cycle.model_copy(update={"intervention": intervention})
    with pytest.raises(ArtifactExportError, match="grounding"):
        projection._cycle_attestation(boundary.model_copy(update={"cycle": cycle}))


def test_delivery_projection_rejects_missing_or_invalid_delivery_sources(
    fixed_stage2_result: Stage2ExperimentRunResult,
) -> None:
    boundary = _delivery_boundary(fixed_stage2_result)

    with pytest.raises(ArtifactExportError, match="delivery"):
        projection._delivery_attestation(boundary.model_copy(update={"delivery_record": None}))

    assert boundary.delivery_record is not None
    invalid_delivery = boundary.delivery_record.model_copy(update={"state": DeliveryState.PENDING})
    with pytest.raises(ArtifactExportError, match="delivery"):
        projection._delivery_attestation(
            boundary.model_copy(update={"delivery_record": invalid_delivery})
        )


def test_ledger_projection_rejects_non_entries_and_untyped_records(
    fixed_stage2_result: Stage2ExperimentRunResult,
) -> None:
    with pytest.raises(ArtifactExportError, match="ledger"):
        projection._ledger_attestation(object())

    entry = fixed_stage2_result.ledger[0]
    invalid_record = SimpleNamespace(record_type=1, revision=None, state=None)
    with pytest.raises(ArtifactExportError, match="ledger"):
        projection._ledger_attestation(entry.model_copy(update={"record": invalid_record}))


def test_redaction_policy_detects_literal_leaks_before_publication(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    destination = tmp_path / "secret"

    with pytest.raises(ArtifactExportError, match="non-redacted"):
        export_algorithm_artifact(
            fixed_stage2_result,
            destination,
            execution=algorithm_execution,
            revision=clean_revision,
            redaction_policy=RedactionPolicy(literal_secrets=("gpt-oss:20b",)),
        )
    assert not destination.exists()


def test_redaction_verification_fails_closed_on_scanner_errors(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenRedactor:
        def __init__(self, **_values: object) -> None:
            pass

        def redact_payload(self, _payload: object) -> None:
            raise ValueError

    monkeypatch.setattr(exporter, "Redactor", BrokenRedactor)
    destination = tmp_path / "scanner-error"

    with pytest.raises(ArtifactExportError, match="redaction verification"):
        export_algorithm_artifact(
            fixed_stage2_result,
            destination,
            execution=algorithm_execution,
            revision=clean_revision,
        )
    assert not destination.exists()


def test_component_byte_limit_is_enforced_before_publication(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(algorithm_manifest, "MAX_ALGORITHM_COMPONENT_BYTES", 1)
    destination = tmp_path / "oversized"

    with pytest.raises(ArtifactExportError, match="byte limit"):
        export_algorithm_artifact(
            fixed_stage2_result,
            destination,
            execution=algorithm_execution,
            revision=clean_revision,
        )
    assert not destination.exists()


def test_unexpected_construction_errors_are_value_free_and_non_publishing(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_construction(*_values: object) -> None:
        raise RuntimeError("sensitive internal detail")

    monkeypatch.setattr(projection, "_project_algorithm_components", fail_construction)
    destination = tmp_path / "construction-error"

    with pytest.raises(ArtifactExportError) as failed:
        export_algorithm_artifact(
            fixed_stage2_result,
            destination,
            execution=algorithm_execution,
            revision=clean_revision,
        )
    assert str(failed.value) == "algorithm artifact construction failed"
    assert "sensitive" not in str(failed.value)
    assert not destination.exists()


def test_failed_cross_condition_replace_preserves_the_existing_tree(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    no_memory_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    destination = tmp_path / "artifact"
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    before = _tree_bytes(destination)

    with pytest.raises(ArtifactExistsError):
        export_algorithm_artifact(
            no_memory_stage2_result,
            destination,
            execution=_frozen_execution(algorithm_execution, response_fixture=None),
            revision=clean_revision,
            replace=True,
        )
    assert _tree_bytes(destination) == before


def test_invalid_destination_parent_creates_no_partial_tree(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    blocking_parent = tmp_path / "regular-file"
    blocking_parent.write_text("occupied", encoding="utf-8")
    destination = blocking_parent / "artifact"

    with pytest.raises(ArtifactDestinationError):
        export_algorithm_artifact(
            fixed_stage2_result,
            destination,
            execution=algorithm_execution,
            revision=clean_revision,
        )
    assert blocking_parent.read_text(encoding="utf-8") == "occupied"
