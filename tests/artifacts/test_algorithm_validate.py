from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import TypeVar
from uuid import UUID

import pytest
from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from saliencegate.artifacts import algorithm_manifest as algorithm_models
from saliencegate.artifacts import algorithm_projection as algorithm_projection
from saliencegate.artifacts import algorithm_validate as algorithm_validator
from saliencegate.artifacts.algorithm_export import export_algorithm_artifact
from saliencegate.artifacts.algorithm_manifest import (
    AlgorithmArtifactComponent,
    AlgorithmArtifactComponentName,
    AlgorithmEndpointClassification,
    AlgorithmExecutionAttestation,
    AlgorithmExecutionMode,
    AlgorithmResponseFixtureAttestation,
    AlgorithmSamplingAttestation,
    AlgorithmSamplingMode,
    AlgorithmWarmupPolicy,
    algorithm_component_content_digest,
)
from saliencegate.artifacts.algorithm_validate import (
    AlgorithmSourceResultAssurance,
    ValidatedAlgorithmArtifact,
    load_validated_algorithm_artifact,
    validate_algorithm_artifact,
)
from saliencegate.artifacts.manifest import (
    ArtifactClassification,
    RevisionEvidence,
    delivery_binding_digest,
)
from saliencegate.artifacts.validate import (
    ArtifactValidationCode,
    ArtifactValidationError,
)
from saliencegate.domain import (
    BudgetAmounts,
    BudgetSnapshot,
    DeliveryOutcome,
    DeliveryState,
    DeliveryTarget,
    EventType,
    PayloadDigest,
    ReasonCode,
    TrustLabel,
    canonical_digest,
    canonical_json,
)
from saliencegate.experiments import Stage2ExperimentRunResult
from saliencegate.ports.model_calls import (
    StructuredCallParseStatus,
    StructuredCallStatus,
    StructuredCallUsage,
)
from saliencegate.ports.two_phase import CallReceipt
from saliencegate.repository.integrity import IntegrityContext
from saliencegate.runtime.scheduling import FixedStepSchedule

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _assert_code(
    error: pytest.ExceptionInfo[ArtifactValidationError],
    code: ArtifactValidationCode,
) -> None:
    assert error.value.code is code
    assert "00000000-0000-4000-8000-000000009000" not in str(error.value)
    assert "00000000-0000-4000-8000-000000009000" not in repr(error.value)


def _execution_with_fixture(
    execution: AlgorithmExecutionAttestation,
    *,
    digest: str,
    calls: int,
) -> AlgorithmExecutionAttestation:
    values = execution.model_dump(
        mode="python",
        exclude={"schema_version", "execution_digest", "response_fixture"},
    )
    return AlgorithmExecutionAttestation.create(
        **values,
        response_fixture=AlgorithmResponseFixtureAttestation(
            replay_id="two-phase-replay/v1",
            fixture_digest=digest,
            response_count=calls,
            consumed_count=calls,
        ),
    )


def _changed_execution(
    execution: AlgorithmExecutionAttestation,
    field: str,
) -> AlgorithmExecutionAttestation:
    checkpoint_values = execution.checkpoint.model_dump(
        mode="python",
        exclude={"checkpoint_attestation_digest"},
    )
    tokenizer_values = execution.tokenizer.model_dump(
        mode="python",
        exclude={"tokenizer_digest"},
    )
    values = execution.model_dump(
        mode="python",
        exclude={
            "schema_version",
            "execution_digest",
            "checkpoint",
            "tokenizer",
        },
    )
    if field == "runtime":
        values["runtime_version"] = "1.0.1"
    elif field == "checkpoint":
        checkpoint_values["model_tag"] = "gpt-oss:20b-fixture/v2"
    elif field == "quantization":
        checkpoint_values["quantization"] = "replay-int8"
    elif field == "tokenizer":
        tokenizer_values["tokenizer_version"] = "utf8-bytes-ceil-div-4/v2"
    else:  # pragma: no cover - closed test helper
        raise AssertionError
    values["checkpoint"] = algorithm_models.AlgorithmCheckpointAttestation.model_validate(
        checkpoint_values
    )
    values["tokenizer"] = algorithm_models.AlgorithmTokenizerAttestation.model_validate(
        tokenizer_values
    )
    return AlgorithmExecutionAttestation.create(**values)


def _live_execution(execution: AlgorithmExecutionAttestation) -> AlgorithmExecutionAttestation:
    return AlgorithmExecutionAttestation.create(
        endpoint_classification=AlgorithmEndpointClassification.LOOPBACK_OPENAI_COMPATIBLE,
        runtime_id="saliencegate-openai-compatible",
        runtime_version="1.0.0",
        checkpoint=execution.checkpoint,
        sampling=AlgorithmSamplingAttestation(
            mode=AlgorithmSamplingMode.OPENAI_COMPATIBLE,
            temperature=0.0,
            seed=7,
            reasoning_effort="medium",
        ),
        tokenizer=execution.tokenizer,
        hardware=execution.hardware,
        warmup_policy=AlgorithmWarmupPolicy.COLD,
        execution_mode=AlgorithmExecutionMode.OPENAI_COMPATIBLE,
        response_fixture=None,
    )


def _reseal_components(
    destination: Path,
    loaded: ValidatedAlgorithmArtifact,
    replacements: Mapping[AlgorithmArtifactComponentName, object],
    *,
    manifest_execution: AlgorithmExecutionAttestation | None = None,
    manifest_schedule_digest: str | None = None,
    manifest_window_digests: tuple[str, ...] | None = None,
    manifest_prompt_bundle_digest: str | None = None,
    manifest_model_profile_digest: str | None = None,
    manifest_counters: algorithm_models.AlgorithmArtifactCounters | None = None,
) -> algorithm_models.AlgorithmArtifactManifest:
    encoded = {
        name: model if type(model) is bytes else canonical_json(to_jsonable_python(model))
        for name, model in replacements.items()
    }
    descriptors = tuple(
        AlgorithmArtifactComponent(
            name=item.name,
            path=item.path,
            byte_count=(len(encoded[item.name]) if item.name in encoded else item.byte_count),
            record_count=item.record_count,
            content_digest=(
                algorithm_component_content_digest(item.name, encoded[item.name])
                if item.name in encoded
                else item.content_digest
            ),
        )
        for item in loaded.manifest.components
    )
    original = loaded.manifest
    manifest = algorithm_models.AlgorithmArtifactManifest.create(
        classification=original.classification,
        run_id=original.run_id,
        revision=original.revision,
        condition_id=original.condition_id,
        condition_digest=original.condition_digest,
        cycle_mode=original.cycle_mode,
        trace_digest=original.trace_digest,
        schedule_digest=manifest_schedule_digest or original.schedule_digest,
        window_digests=manifest_window_digests or original.window_digests,
        prompt_bundle_digest=(manifest_prompt_bundle_digest or original.prompt_bundle_digest),
        model_profile_digest=(manifest_model_profile_digest or original.model_profile_digest),
        execution=manifest_execution or original.execution,
        result_digest=original.result_digest,
        components=descriptors,
        counters=manifest_counters or original.counters,
    )
    for name, data in encoded.items():
        destination.joinpath(f"{name.value}.json").write_bytes(data)
    destination.joinpath("manifest.json").write_bytes(canonical_json(manifest))
    return manifest


def _reseal_component(
    destination: Path,
    loaded: ValidatedAlgorithmArtifact,
    name: AlgorithmArtifactComponentName,
    model: object,
    *,
    manifest_execution: AlgorithmExecutionAttestation | None = None,
    manifest_schedule_digest: str | None = None,
    manifest_window_digests: tuple[str, ...] | None = None,
    manifest_prompt_bundle_digest: str | None = None,
    manifest_model_profile_digest: str | None = None,
    manifest_counters: algorithm_models.AlgorithmArtifactCounters | None = None,
) -> algorithm_models.AlgorithmArtifactManifest:
    return _reseal_components(
        destination,
        loaded,
        {name: model},
        manifest_execution=manifest_execution,
        manifest_schedule_digest=manifest_schedule_digest,
        manifest_window_digests=manifest_window_digests,
        manifest_prompt_bundle_digest=manifest_prompt_bundle_digest,
        manifest_model_profile_digest=manifest_model_profile_digest,
        manifest_counters=manifest_counters,
    )


def _raw_reseal_payload(
    model: BaseModel,
    changes: dict[str, object],
) -> dict[str, object]:
    model_type = type(model)
    field, domain = algorithm_models._SEALED_ALGORITHM_MODELS[model_type]
    values = model.model_dump(mode="python", exclude={field})
    values.update(changes)
    values[field] = algorithm_models._digest_without(values, field, domain=domain)
    return values


def _resealed_model(model: _ModelT, changes: dict[str, object]) -> _ModelT:
    return type(model).model_validate(_raw_reseal_payload(model, changes))


def _calls_with_replaced_first(
    calls: algorithm_models.AlgorithmCallsComponent,
    replacement: CallReceipt,
) -> algorithm_models.AlgorithmCallsComponent:
    first_group = calls.groups[0]
    changed_group = algorithm_models.AlgorithmCallGroup.model_validate(
        {
            **first_group.model_dump(mode="python"),
            "call_receipt_digests": (
                replacement.receipt_digest,
                *first_group.call_receipt_digests[1:],
            ),
        }
    )
    return _resealed_model(
        calls,
        {
            "groups": (changed_group, *calls.groups[1:]),
            "calls": (replacement, *calls.calls[1:]),
        },
    )


def _reseal_ledger_chain(
    attestations: algorithm_models.AlgorithmAttestationsComponent,
    entries: tuple[algorithm_models.AlgorithmLedgerEntryAttestation, ...],
) -> algorithm_models.AlgorithmAttestationsComponent:
    integrity = IntegrityContext(key=None, synthetic_benchmark=True)
    previous_chain_tag = None
    resealed_entries = []
    for entry in entries:
        chain_tag = integrity.tag(
            {
                "run_id": str(attestations.run_id),
                "position": entry.position,
                "record_key": entry.record_key,
                "record_tag": entry.record_tag,
                "previous_chain_tag": previous_chain_tag,
            },
            domain="saliencegate:ledger-chain:v1",
        )
        resealed_entry = type(entry).model_validate(
            {
                **entry.model_dump(mode="python"),
                "previous_chain_tag": previous_chain_tag,
                "chain_tag": chain_tag,
            }
        )
        resealed_entries.append(resealed_entry)
        previous_chain_tag = chain_tag
    assert previous_chain_tag is not None
    head = attestations.ledger_head
    head_tag = integrity.tag(
        {
            "run_id": str(head.run_id),
            "entry_count": head.entry_count,
            "chain_tag": previous_chain_tag,
            "projection_tag": head.projection_tag,
        },
        domain="saliencegate:ledger-head:v1",
    )
    changed_head = type(head).model_validate(
        {
            **head.model_dump(mode="python"),
            "chain_tag": previous_chain_tag,
            "head_tag": head_tag,
        }
    )
    return _resealed_model(
        attestations,
        {
            "ledger_entries": tuple(resealed_entries),
            "ledger_head": changed_head,
        },
    )


def test_round_trip_reports_self_consistency_without_overclaiming_authenticity(
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

    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    report = validate_algorithm_artifact(
        destination / "manifest.json",
        expected_manifest_digest=manifest.manifest_digest,
    )

    assert isinstance(loaded, ValidatedAlgorithmArtifact)
    assert loaded.manifest == manifest
    assert loaded.run.source_result_digest == fixed_stage2_result.result_digest
    assert report.valid
    assert report.structurally_valid
    assert report.self_consistent
    assert report.expected_digest_matched
    assert report.source_result_assurance is AlgorithmSourceResultAssurance.PRODUCER_ATTESTED
    assert report.component_count == 9


def test_validator_applies_the_per_receipt_call_policy(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "call-policy"
    manifest = export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    monkeypatch.setattr(
        algorithm_validator,
        "call_policy_accepts_receipts",
        lambda *_args, **_kwargs: False,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=manifest.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def test_committed_cycle_requires_a_valid_memory_edit_receipt(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    destination = tmp_path / "invalid-memory-edit"
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    first_receipt = loaded.calls.calls[0]
    assert first_receipt.parse_status is StructuredCallParseStatus.VALID
    changed_receipt = type(first_receipt).model_validate(
        first_receipt.model_dump(mode="python", exclude={"receipt_digest"})
        | {"parse_status": StructuredCallParseStatus.SCHEMA_INVALID}
    )
    changed_calls = _calls_with_replaced_first(loaded.calls, changed_receipt)

    attestations = algorithm_models.AlgorithmAttestationsComponent.model_validate_json(
        destination.joinpath("attestations.json").read_bytes()
    )
    first_boundary = attestations.boundaries[0]
    observed_values = first_boundary.observation.observed.model_dump(mode="python")
    observed_values["call_receipt_digests"] = (
        changed_receipt.receipt_digest,
        *first_boundary.observation.observed.call_receipt_digests[1:],
    )
    changed_observed = type(first_boundary.observation.observed).model_validate(observed_values)
    observation_values = first_boundary.observation.model_dump(
        mode="python",
        exclude={"observation_digest"},
    )
    observation_values["observed"] = changed_observed
    changed_observation = type(first_boundary.observation).model_validate(observation_values)
    changed_boundary = _resealed_model(
        first_boundary,
        {"observation": changed_observation},
    )
    changed_attestations = _resealed_model(
        attestations,
        {"boundaries": (changed_boundary, *attestations.boundaries[1:])},
    )

    first_cycle = loaded.cycles.cycles[0]
    changed_cycle = _resealed_model(
        first_cycle,
        {
            "call_receipt_digests": (
                changed_receipt.receipt_digest,
                *first_cycle.call_receipt_digests[1:],
            ),
            "observation_digest": changed_observation.observation_digest,
        },
    )
    changed_cycles = _resealed_model(
        loaded.cycles,
        {"cycles": (changed_cycle, *loaded.cycles.cycles[1:])},
    )
    forged = _reseal_components(
        destination,
        loaded,
        {
            AlgorithmArtifactComponentName.CALLS: changed_calls,
            AlgorithmArtifactComponentName.CYCLES: changed_cycles,
            AlgorithmArtifactComponentName.ATTESTATIONS: changed_attestations,
        },
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


@pytest.mark.parametrize("identity", ("cycle", "intervention"))
def test_validator_recomputes_deterministic_cycle_identities(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    monkeypatch: pytest.MonkeyPatch,
    identity: str,
) -> None:
    destination = tmp_path / f"identity-{identity}"
    manifest = export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    if identity == "cycle":
        monkeypatch.setattr(
            algorithm_validator,
            "derive_cycle_id",
            lambda *_args, **_kwargs: "f" * 64,
        )
    else:
        runtime_uuid = algorithm_validator.algorithm_runtime_uuid

        def changed_runtime_uuid(trace_digest: str, namespace: str, *parts: object) -> UUID:
            if namespace == "stage2-intervention":
                return UUID("00000000-0000-4000-8000-000000009001")
            return runtime_uuid(trace_digest, namespace, *parts)

        monkeypatch.setattr(algorithm_validator, "algorithm_runtime_uuid", changed_runtime_uuid)

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=manifest.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def test_synthetic_raw_is_recomputed_but_not_exposed_in_safe_view(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    destination = tmp_path / "raw"
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
        classification=ArtifactClassification.SYNTHETIC_RAW,
    )

    loaded = load_validated_algorithm_artifact(destination / "manifest.json")

    assert (
        loaded.report.source_result_assurance is AlgorithmSourceResultAssurance.RECOMPUTED_FROM_RAW
    )
    assert not hasattr(loaded.attestations, "raw_synthetic_result")


def test_no_memory_round_trip_has_zero_active_algorithm_records(
    tmp_path: Path,
    no_memory_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    values = algorithm_execution.model_dump(
        mode="python",
        exclude={"schema_version", "execution_digest", "response_fixture"},
    )
    execution = AlgorithmExecutionAttestation.create(**values, response_fixture=None)
    destination = tmp_path / "no-memory"
    manifest = export_algorithm_artifact(
        no_memory_stage2_result,
        destination,
        execution=execution,
        revision=clean_revision,
    )

    loaded = load_validated_algorithm_artifact(
        destination / "manifest.json",
        expected_manifest_digest=manifest.manifest_digest,
    )

    assert loaded.manifest.counters.cycles == 0
    assert loaded.manifest.counters.requests == 0
    assert loaded.manifest.counters.model_calls == 0
    assert loaded.cycles.cycles == ()
    assert loaded.calls.calls == ()
    assert loaded.outcomes.outcomes == ()


def test_no_memory_final_memory_cannot_be_resealed_as_nonempty(
    tmp_path: Path,
    no_memory_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    destination = tmp_path / "no-memory-record"
    execution_values = algorithm_execution.model_dump(
        mode="python",
        exclude={"schema_version", "execution_digest", "response_fixture"},
    )
    export_algorithm_artifact(
        no_memory_stage2_result,
        destination,
        execution=AlgorithmExecutionAttestation.create(
            **execution_values,
            response_fixture=None,
        ),
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    changed_memory = type(loaded.metrics.final_memory).model_validate(
        loaded.metrics.final_memory.model_dump(mode="python")
        | {"record_count": 1, "record_digests": ("f" * 64,)}
    )
    changed_metrics = _resealed_model(
        loaded.metrics,
        {"final_memory": changed_memory},
    )
    forged = _reseal_component(
        destination,
        loaded,
        AlgorithmArtifactComponentName.METRICS,
        changed_metrics,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def test_cycle_mutation_counts_cannot_target_missing_memory(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    destination = tmp_path / "missing-memory-target"
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    first_cycle = loaded.cycles.cycles[0]
    assert first_cycle.memory_create_count == 0
    changed_cycle = _resealed_model(
        first_cycle,
        {"memory_update_count": first_cycle.memory_update_count + 1},
    )
    changed_cycles = _resealed_model(
        loaded.cycles,
        {"cycles": (changed_cycle, *loaded.cycles.cycles[1:])},
    )

    attestations = algorithm_models.AlgorithmAttestationsComponent.model_validate_json(
        destination.joinpath("attestations.json").read_bytes()
    )
    first_boundary = attestations.boundaries[0]
    observed_values = first_boundary.observation.observed.model_dump(mode="python")
    observed_values["memory_mutation_count"] += 1
    changed_observed = type(first_boundary.observation.observed).model_validate(observed_values)
    observation_values = first_boundary.observation.model_dump(
        mode="python",
        exclude={"observation_digest"},
    )
    observation_values["observed"] = changed_observed
    changed_observation = type(first_boundary.observation).model_validate(observation_values)
    changed_boundary = _resealed_model(
        first_boundary,
        {"observation": changed_observation},
    )
    changed_attestations = _resealed_model(
        attestations,
        {"boundaries": (changed_boundary, *attestations.boundaries[1:])},
    )

    metric_values = loaded.metrics.metrics.model_dump(mode="python")
    metric_values["memory_mutation_count"] += 1
    changed_metrics = _resealed_model(
        loaded.metrics,
        {"metrics": type(loaded.metrics.metrics).model_validate(metric_values)},
    )
    forged = _reseal_components(
        destination,
        loaded,
        {
            AlgorithmArtifactComponentName.CYCLES: changed_cycles,
            AlgorithmArtifactComponentName.ATTESTATIONS: changed_attestations,
            AlgorithmArtifactComponentName.METRICS: changed_metrics,
        },
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def test_memory_count_replay_accounts_for_private_status_supersession(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    destination = tmp_path / "private-status-capacity"
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    first, second, third = loaded.cycles.cycles
    assert algorithm_validator._replayed_memory_record_count(loaded.cycles) == 3
    changed_cycles = _resealed_model(
        loaded.cycles,
        {
            "cycles": (
                _resealed_model(first, {"private_status_replaced": True}),
                second,
                _resealed_model(
                    third,
                    {
                        "memory_update_count": 4,
                        "memory_invalidation_count": 0,
                    },
                ),
            )
        },
    )

    assert algorithm_validator._replayed_memory_record_count(changed_cycles) is None


def test_memory_count_replay_rejects_impossible_stage2_mutations(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    destination = tmp_path / "impossible-stage2-mutations"
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    first, second, *_ = loaded.cycles.cycles
    establishes_private = _resealed_model(
        first,
        {
            "memory_create_count": 0,
            "memory_update_count": 0,
            "memory_invalidation_count": 0,
            "private_status_replaced": True,
        },
    )
    updates_existing = _resealed_model(
        second,
        {
            "memory_create_count": 0,
            "memory_update_count": 1,
            "memory_invalidation_count": 0,
            "private_status_replaced": False,
        },
    )
    replaces_and_invalidates_private = _resealed_model(
        second,
        {
            "memory_create_count": 0,
            "memory_update_count": 0,
            "memory_invalidation_count": 1,
            "private_status_replaced": True,
        },
    )

    for impossible in (updates_existing, replaces_and_invalidates_private):
        changed_cycles = _resealed_model(
            loaded.cycles,
            {"cycles": (establishes_private, impossible)},
        )
        assert algorithm_validator._replayed_memory_record_count(changed_cycles) is None


def test_retrieval_and_forced_conditions_close_distinct_grounding_paths(
    tmp_path: Path,
    retrieval_stage2_result: Stage2ExperimentRunResult,
    always_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    cases = (
        (
            retrieval_stage2_result,
            "50bce1255d7e313547ead4e73395b0f9b1e9a6e08c67f42b18f1c69552b24ca7",
            3,
            "retrieval",
        ),
        (
            always_stage2_result,
            "8b9c69c0bfd68c6d953cf8a002665f4029192088ffb6e8a06fa993136a55637e",
            6,
            "forced",
        ),
    )
    for result, digest, calls, name in cases:
        execution = _execution_with_fixture(
            algorithm_execution,
            digest=digest,
            calls=calls,
        )
        destination = tmp_path / name
        manifest = export_algorithm_artifact(
            result,
            destination,
            execution=execution,
            revision=clean_revision,
        )

        loaded = load_validated_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=manifest.manifest_digest,
        )

        assert len(loaded.calls.calls) == calls
        assert len(loaded.cycles.cycles) == 3
        if name == "retrieval":
            assert all(item.selector_provenance is not None for item in loaded.cycles.cycles)
            assert all(item.grounding_model_call_index is None for item in loaded.cycles.cycles)
        else:
            assert all(item.selector_provenance is None for item in loaded.cycles.cycles)
            assert all(item.grounding_model_call_index == 1 for item in loaded.cycles.cycles)


@pytest.mark.parametrize("field", ("runtime", "checkpoint", "quantization", "tokenizer"))
def test_resealed_execution_divergence_is_rejected_cross_component(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    field: str,
) -> None:
    destination = tmp_path / field
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    changed_execution = _changed_execution(loaded.run.execution, field)
    changed_run = _resealed_model(loaded.run, {"execution": changed_execution})
    forged = _reseal_component(
        destination,
        loaded,
        AlgorithmArtifactComponentName.RUN,
        changed_run,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def test_coherent_live_mode_relabel_cannot_reclassify_replay_call_evidence(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    destination = tmp_path / "replay-as-live"
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    live_execution = _live_execution(loaded.run.execution)
    changed_run = _resealed_model(loaded.run, {"execution": live_execution})
    forged = _reseal_component(
        destination,
        loaded,
        AlgorithmArtifactComponentName.RUN,
        changed_run,
        manifest_execution=live_execution,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def test_live_artifact_rejects_a_resealed_duplicate_call_digest(
    tmp_path: Path,
    live_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    destination = tmp_path / "live-duplicate-call"
    export_algorithm_artifact(
        live_stage2_result,
        destination,
        execution=_live_execution(algorithm_execution),
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    first, second = loaded.calls.calls[:2]
    changed_second = type(second).model_validate(
        second.model_dump(mode="python", exclude={"receipt_digest"})
        | {"call_digest": first.call_digest}
    )
    first_group = loaded.calls.groups[0]
    changed_group = algorithm_models.AlgorithmCallGroup.model_validate(
        first_group.model_dump(mode="python")
        | {
            "call_receipt_digests": (
                first.receipt_digest,
                changed_second.receipt_digest,
            )
        }
    )
    changed_calls = _raw_reseal_payload(
        loaded.calls,
        {
            "groups": (changed_group, *loaded.calls.groups[1:]),
            "calls": (first, changed_second, *loaded.calls.calls[2:]),
        },
    )
    forged = _reseal_component(
        destination,
        loaded,
        AlgorithmArtifactComponentName.CALLS,
        changed_calls,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.INVALID_COMPONENT)


def test_validator_checks_tokenizer_identity_for_partial_canonical_usage(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    destination = tmp_path / "partial-canonical"
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    original = loaded.calls.calls[0]
    usage_values = original.usage.model_dump(mode="python", warnings=False)
    usage_values.update(
        {
            "canonical_output_tokens": None,
            "local_counter_configuration_digest": "9" * 64,
        }
    )
    call_values = original.model_dump(mode="python", exclude={"receipt_digest"})
    call_values.update(
        {
            "status": StructuredCallStatus.MODEL_ERROR,
            "parse_status": StructuredCallParseStatus.NOT_ATTEMPTED,
            "completion_digest": None,
            "completion_byte_count": None,
            "usage": StructuredCallUsage.model_validate(usage_values),
        }
    )
    changed_calls = _calls_with_replaced_first(
        loaded.calls,
        CallReceipt.model_validate(call_values),
    )

    with pytest.raises(ArtifactValidationError) as error:
        algorithm_validator._validate_execution_binding(loaded.run, changed_calls)
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def test_coherent_runtime_reseal_requires_an_external_digest_anchor(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    destination = tmp_path / "artifact"
    original = export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    changed_execution = _changed_execution(loaded.run.execution, "runtime")
    changed_run = _resealed_model(loaded.run, {"execution": changed_execution})
    forged = _reseal_component(
        destination,
        loaded,
        AlgorithmArtifactComponentName.RUN,
        changed_run,
        manifest_execution=changed_execution,
    )

    unanchored = validate_algorithm_artifact(destination / "manifest.json")
    anchored = validate_algorithm_artifact(
        destination / "manifest.json",
        expected_manifest_digest=forged.manifest_digest,
    )
    assert unanchored.expected_digest_matched is None
    assert anchored.expected_digest_matched

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=original.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.EXPECTED_DIGEST_MISMATCH)


def test_coherent_tokenizer_and_fixture_count_tampering_are_rejected(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    fixture = algorithm_execution.response_fixture
    assert fixture is not None
    for name, changed_execution in (
        ("tokenizer", _changed_execution(algorithm_execution, "tokenizer")),
        (
            "fixture-count",
            _execution_with_fixture(
                algorithm_execution,
                digest=fixture.fixture_digest,
                calls=5,
            ),
        ),
        (
            "fixture-digest",
            _execution_with_fixture(
                algorithm_execution,
                digest="f" * 64,
                calls=fixture.response_count,
            ),
        ),
    ):
        destination = tmp_path / name
        export_algorithm_artifact(
            fixed_stage2_result,
            destination,
            execution=algorithm_execution,
            revision=clean_revision,
        )
        loaded = load_validated_algorithm_artifact(destination / "manifest.json")
        changed_run = _resealed_model(loaded.run, {"execution": changed_execution})
        forged = _reseal_component(
            destination,
            loaded,
            AlgorithmArtifactComponentName.RUN,
            changed_run,
            manifest_execution=changed_execution,
        )

        with pytest.raises(ArtifactValidationError) as error:
            validate_algorithm_artifact(
                destination / "manifest.json",
                expected_manifest_digest=forged.manifest_digest,
            )
        _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def test_resealed_schedule_prefix_and_window_version_tampering_are_rejected(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    schedule_root = tmp_path / "schedule"
    export_algorithm_artifact(
        fixed_stage2_result,
        schedule_root,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(schedule_root / "manifest.json")
    schedule_values = loaded.trajectory.schedule.model_dump(
        mode="python",
        exclude={"schedule_digest"},
    )
    schedule_values["trajectory_prefix_digest"] = "f" * 64
    changed_schedule = FixedStepSchedule.model_validate(schedule_values)
    changed_trajectory = _raw_reseal_payload(
        loaded.trajectory,
        {"schedule": changed_schedule},
    )
    forged = _reseal_component(
        schedule_root,
        loaded,
        AlgorithmArtifactComponentName.TRAJECTORY,
        changed_trajectory,
        manifest_schedule_digest=changed_schedule.schedule_digest,
    )
    with pytest.raises(ArtifactValidationError) as schedule_error:
        validate_algorithm_artifact(
            schedule_root / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(schedule_error, ArtifactValidationCode.INVALID_COMPONENT)

    window_root = tmp_path / "window"
    export_algorithm_artifact(
        fixed_stage2_result,
        window_root,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(window_root / "manifest.json")
    first = loaded.trajectory.windows[0]
    changed_window = algorithm_models.AlgorithmWindowAttestation.model_validate(
        {**first.model_dump(mode="python"), "version": "forged-window/v9"}
    )
    changed_trajectory_model = _raw_reseal_payload(
        loaded.trajectory,
        {"windows": (changed_window, *loaded.trajectory.windows[1:])},
    )
    forged = _reseal_component(
        window_root,
        loaded,
        AlgorithmArtifactComponentName.TRAJECTORY,
        changed_trajectory_model,
    )
    with pytest.raises(ArtifactValidationError) as window_error:
        validate_algorithm_artifact(
            window_root / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(window_error, ArtifactValidationCode.INVALID_COMPONENT)


def test_resealed_ledger_tag_breaks_the_recomputed_chain(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
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
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    attestations = algorithm_models.AlgorithmAttestationsComponent.model_validate_json(
        destination.joinpath("attestations.json").read_bytes()
    )
    first = attestations.ledger_entries[0]
    changed_entry = algorithm_models.AlgorithmLedgerEntryAttestation.model_validate(
        {
            **first.model_dump(mode="python"),
            "record_tag": PayloadDigest(
                algorithm=first.record_tag.algorithm,
                value="f" * 64,
            ),
        }
    )
    changed_attestations = _resealed_model(
        attestations,
        {"ledger_entries": (changed_entry, *attestations.ledger_entries[1:])},
    )
    forged = _reseal_component(
        destination,
        loaded,
        AlgorithmArtifactComponentName.ATTESTATIONS,
        changed_attestations,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


@pytest.mark.parametrize("budget_field", ("reserved", "consumed"))
def test_resealed_intermediate_decision_budget_must_match_prior_settlements(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    budget_field: str,
) -> None:
    destination = tmp_path / "decision-budget"
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    original = loaded.decisions.decisions[-1]
    assert any(
        cycle.boundary_event_sequence < original.event_sequence for cycle in loaded.cycles.cycles
    )
    reserved_values = original.budget_snapshot.reserved.model_dump(mode="python")
    consumed_values = original.budget_snapshot.consumed.model_dump(mode="python")
    changed_values = reserved_values if budget_field == "reserved" else consumed_values
    changed_values["model_calls"] += 1
    changed_snapshot = BudgetSnapshot(
        limits=original.budget_snapshot.limits,
        reserved=BudgetAmounts.model_validate(reserved_values),
        consumed=BudgetAmounts.model_validate(consumed_values),
    )
    changed_decision = type(original).model_validate(
        {
            **original.model_dump(mode="python"),
            "budget_snapshot": changed_snapshot,
        }
    )
    changed_decisions = _resealed_model(
        loaded.decisions,
        {
            "decisions": (
                *loaded.decisions.decisions[:-1],
                changed_decision,
            )
        },
    )

    attestations = algorithm_models.AlgorithmAttestationsComponent.model_validate_json(
        destination.joinpath("attestations.json").read_bytes()
    )
    record_key = f"invocation_decision:{changed_decision.decision_id}"
    index, original_entry = next(
        (index, entry)
        for index, entry in enumerate(attestations.ledger_entries)
        if entry.record_key == record_key
    )
    assert all(
        entry.record_type != "trace_event" for entry in attestations.ledger_entries[index + 1 :]
    )
    integrity = IntegrityContext(key=None, synthetic_benchmark=True)
    changed_entry = type(original_entry).model_validate(
        {
            **original_entry.model_dump(mode="python"),
            "source_record_digest": canonical_digest(changed_decision),
            "record_tag": integrity.tag(
                changed_decision,
                domain="saliencegate:ledger-record:v1",
            ),
        }
    )
    entries = list(attestations.ledger_entries)
    entries[index] = changed_entry
    changed_attestations = _reseal_ledger_chain(attestations, tuple(entries))
    forged = _reseal_components(
        destination,
        loaded,
        {
            AlgorithmArtifactComponentName.DECISIONS: changed_decisions,
            AlgorithmArtifactComponentName.ATTESTATIONS: changed_attestations,
        },
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def test_resealed_outcome_source_digest_cannot_keep_a_stale_record_tag(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    destination = tmp_path / "outcome-record-tag"
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    original = loaded.outcomes.outcomes[-1]
    changed_outcome = type(original).model_validate(
        {
            **original.model_dump(mode="python"),
            "steps": original.steps + 1,
        }
    )
    changed_outcomes = _resealed_model(
        loaded.outcomes,
        {
            "outcomes": (
                *loaded.outcomes.outcomes[:-1],
                changed_outcome,
            )
        },
    )

    attestations = algorithm_models.AlgorithmAttestationsComponent.model_validate_json(
        destination.joinpath("attestations.json").read_bytes()
    )
    record_key = f"intervention_outcome:{changed_outcome.outcome_id}"
    index, original_entry = next(
        (index, entry)
        for index, entry in enumerate(attestations.ledger_entries)
        if entry.record_key == record_key
    )
    assert index == len(attestations.ledger_entries) - 1
    changed_entry = type(original_entry).model_validate(
        {
            **original_entry.model_dump(mode="python"),
            "source_record_digest": canonical_digest(changed_outcome),
        }
    )
    entries = list(attestations.ledger_entries)
    entries[index] = changed_entry
    changed_attestations = _resealed_model(
        attestations,
        {"ledger_entries": tuple(entries)},
    )
    forged = _reseal_components(
        destination,
        loaded,
        {
            AlgorithmArtifactComponentName.OUTCOMES: changed_outcomes,
            AlgorithmArtifactComponentName.ATTESTATIONS: changed_attestations,
        },
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


@pytest.mark.parametrize("field", ("steps", "created_at"))
def test_coherently_resealed_outcome_preserves_the_stage2_workload(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    field: str,
) -> None:
    destination = tmp_path / f"outcome-{field}"
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    original = loaded.outcomes.outcomes[-1]
    changed_value: object = (
        original.steps + 1 if field == "steps" else original.created_at + timedelta(seconds=1)
    )
    changed_outcome = type(original).model_validate(
        original.model_dump(mode="python") | {field: changed_value}
    )
    changed_outcomes = _resealed_model(
        loaded.outcomes,
        {"outcomes": (*loaded.outcomes.outcomes[:-1], changed_outcome)},
    )

    attestations = algorithm_models.AlgorithmAttestationsComponent.model_validate_json(
        destination.joinpath("attestations.json").read_bytes()
    )
    record_key = f"intervention_outcome:{changed_outcome.outcome_id}"
    index, original_entry = next(
        (index, entry)
        for index, entry in enumerate(attestations.ledger_entries)
        if entry.record_key == record_key
    )
    integrity = IntegrityContext(key=None, synthetic_benchmark=True)
    changed_entry = type(original_entry).model_validate(
        original_entry.model_dump(mode="python")
        | {
            "source_record_digest": canonical_digest(changed_outcome),
            "record_tag": integrity.tag(
                changed_outcome,
                domain="saliencegate:ledger-record:v1",
            ),
        }
    )
    entries = list(attestations.ledger_entries)
    entries[index] = changed_entry
    changed_attestations = _reseal_ledger_chain(attestations, tuple(entries))
    forged = _reseal_components(
        destination,
        loaded,
        {
            AlgorithmArtifactComponentName.OUTCOMES: changed_outcomes,
            AlgorithmArtifactComponentName.ATTESTATIONS: changed_attestations,
        },
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


@pytest.mark.parametrize("record_type", ("invocation_decision", "intervention_outcome"))
def test_coherently_resealed_chain_rejects_wrong_tag_for_complete_record(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    record_type: str,
) -> None:
    destination = tmp_path / record_type
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    record_key = (
        f"invocation_decision:{loaded.decisions.decisions[-1].decision_id}"
        if record_type == "invocation_decision"
        else f"intervention_outcome:{loaded.outcomes.outcomes[-1].outcome_id}"
    )
    attestations = algorithm_models.AlgorithmAttestationsComponent.model_validate_json(
        destination.joinpath("attestations.json").read_bytes()
    )
    index, original = next(
        (index, entry)
        for index, entry in enumerate(attestations.ledger_entries)
        if entry.record_key == record_key
    )
    assert all(
        entry.record_type != "trace_event" for entry in attestations.ledger_entries[index + 1 :]
    )
    changed_value = "f" * 64 if original.record_tag.value != "f" * 64 else "e" * 64
    changed = type(original).model_validate(
        {
            **original.model_dump(mode="python"),
            "record_tag": PayloadDigest(
                algorithm=original.record_tag.algorithm,
                value=changed_value,
            ),
        }
    )
    entries = list(attestations.ledger_entries)
    entries[index] = changed
    changed_attestations = _reseal_ledger_chain(attestations, tuple(entries))
    forged = _reseal_component(
        destination,
        loaded,
        AlgorithmArtifactComponentName.ATTESTATIONS,
        changed_attestations,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def test_resealed_budget_settlement_must_reconcile_with_call_usage(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
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
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    first = loaded.cycles.cycles[0]
    settlement_values = first.budget_settlement.model_dump(mode="python")
    settlement_values["input_tokens"] = first.budget_settlement.input_tokens - 1
    changed_settlement = BudgetAmounts.model_validate(settlement_values)
    changed_cycle = _resealed_model(
        first,
        {"budget_settlement": changed_settlement},
    )
    changed_cycles = _resealed_model(
        loaded.cycles,
        {"cycles": (changed_cycle, *loaded.cycles.cycles[1:])},
    )

    consumed_values = loaded.metrics.final_budget_snapshot.consumed.model_dump(mode="python")
    consumed_values["input_tokens"] -= 1
    changed_budget = BudgetSnapshot(
        limits=loaded.metrics.final_budget_snapshot.limits,
        reserved=loaded.metrics.final_budget_snapshot.reserved,
        consumed=BudgetAmounts.model_validate(consumed_values),
    )
    changed_metrics = _resealed_model(
        loaded.metrics,
        {"final_budget_snapshot": changed_budget},
    )
    forged = _reseal_components(
        destination,
        loaded,
        {
            AlgorithmArtifactComponentName.CYCLES: changed_cycles,
            AlgorithmArtifactComponentName.METRICS: changed_metrics,
        },
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def test_final_memory_projection_is_bound_to_repository_projection(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
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
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    final_memory = loaded.metrics.final_memory
    changed_memory = algorithm_models.AlgorithmFinalMemoryAttestation.model_validate(
        {
            **final_memory.model_dump(mode="python"),
            "projection_digest": PayloadDigest(
                algorithm=final_memory.projection_digest.algorithm,
                value="f" * 64,
            ),
        }
    )
    changed_metrics = _resealed_model(
        loaded.metrics,
        {"final_memory": changed_memory},
    )
    forged = _reseal_component(
        destination,
        loaded,
        AlgorithmArtifactComponentName.METRICS,
        changed_metrics,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def test_cycle_action_reason_pair_is_closed_inside_the_component(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
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
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    reminder_index, reminder = next(
        (index, item)
        for index, item in enumerate(loaded.cycles.cycles)
        if item.reason_code is ReasonCode.GROUNDED_REMINDER
    )
    changed_cycle = _raw_reseal_payload(
        reminder,
        {"reason_code": ReasonCode.SILENCE_SELECTED},
    )
    changed_values: list[object] = list(loaded.cycles.cycles)
    changed_values[reminder_index] = changed_cycle
    changed_cycles = _raw_reseal_payload(
        loaded.cycles,
        {"cycles": tuple(changed_values)},
    )
    forged = _reseal_component(
        destination,
        loaded,
        AlgorithmArtifactComponentName.CYCLES,
        changed_cycles,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.INVALID_COMPONENT)


def test_raw_assurance_recomputes_the_complete_window_projection(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    destination = tmp_path / "raw"
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
        classification=ArtifactClassification.SYNTHETIC_RAW,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    changed_windows = tuple(
        algorithm_models.AlgorithmWindowAttestation.model_validate(
            {**window.model_dump(mode="python"), "task_digest": "f" * 64}
        )
        for window in loaded.trajectory.windows
    )
    changed_trajectory = _resealed_model(
        loaded.trajectory,
        {"windows": changed_windows},
    )
    forged = _reseal_component(
        destination,
        loaded,
        AlgorithmArtifactComponentName.TRAJECTORY,
        changed_trajectory,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("prompt_template_id", "forged-template/v9"),
        ("prompt_template_digest", "f" * 64),
    ),
)
def test_fully_resealed_call_still_has_to_match_the_frozen_prompt_bundle(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    field: str,
    value: str,
) -> None:
    destination = tmp_path / field
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    original_call = loaded.calls.calls[0]
    call_values = original_call.model_dump(
        mode="python",
        exclude={"receipt_digest"},
    )
    call_values[field] = value
    changed_call = CallReceipt.model_validate(call_values)

    original_group = loaded.calls.groups[0]
    changed_receipt_digests = (
        changed_call.receipt_digest,
        *original_group.call_receipt_digests[1:],
    )
    changed_group = algorithm_models.AlgorithmCallGroup.model_validate(
        {
            **original_group.model_dump(mode="python"),
            "call_receipt_digests": changed_receipt_digests,
        }
    )
    changed_calls = _resealed_model(
        loaded.calls,
        {
            "groups": (changed_group, *loaded.calls.groups[1:]),
            "calls": (changed_call, *loaded.calls.calls[1:]),
        },
    )

    original_boundary = loaded.attestations.boundaries[0]
    observed = original_boundary.observation.observed
    changed_observed = type(observed).model_validate(
        {
            **observed.model_dump(mode="python"),
            "call_receipt_digests": changed_receipt_digests,
        }
    )
    observation_values = original_boundary.observation.model_dump(
        mode="python",
        exclude={"observation_digest"},
    )
    observation_values["observed"] = changed_observed
    changed_observation = type(original_boundary.observation).model_validate(observation_values)
    changed_boundary = _resealed_model(
        original_boundary,
        {"observation": changed_observation},
    )
    private_attestations = algorithm_models.AlgorithmAttestationsComponent.model_validate_json(
        destination.joinpath("attestations.json").read_bytes()
    )
    changed_attestations = _resealed_model(
        private_attestations,
        {"boundaries": (changed_boundary, *private_attestations.boundaries[1:])},
    )

    original_cycle = loaded.cycles.cycles[0]
    changed_cycle = _resealed_model(
        original_cycle,
        {
            "call_receipt_digests": changed_receipt_digests,
            "observation_digest": changed_observation.observation_digest,
        },
    )
    changed_cycles = _resealed_model(
        loaded.cycles,
        {"cycles": (changed_cycle, *loaded.cycles.cycles[1:])},
    )
    forged = _reseal_components(
        destination,
        loaded,
        {
            AlgorithmArtifactComponentName.ATTESTATIONS: changed_attestations,
            AlgorithmArtifactComponentName.CALLS: changed_calls,
            AlgorithmArtifactComponentName.CYCLES: changed_cycles,
        },
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def test_call_order_and_grounding_index_are_closed_by_component_schemas(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    calls_root = tmp_path / "calls"
    export_algorithm_artifact(
        fixed_stage2_result,
        calls_root,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(calls_root / "manifest.json")
    changed_calls = _raw_reseal_payload(
        loaded.calls,
        {"ordered_request_digests": tuple(reversed(loaded.calls.ordered_request_digests))},
    )
    forged = _reseal_component(
        calls_root,
        loaded,
        AlgorithmArtifactComponentName.CALLS,
        changed_calls,
    )
    with pytest.raises(ArtifactValidationError) as calls_error:
        validate_algorithm_artifact(
            calls_root / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(calls_error, ArtifactValidationCode.INVALID_COMPONENT)

    grounding_root = tmp_path / "grounding"
    export_algorithm_artifact(
        fixed_stage2_result,
        grounding_root,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(grounding_root / "manifest.json")
    first = loaded.cycles.cycles[0]
    changed_cycle = _raw_reseal_payload(
        first,
        {
            "grounding_model_call_index": 0,
            "grounding_model_call_digest": first.model_call_digests[0],
        },
    )
    changed_cycles = _raw_reseal_payload(
        loaded.cycles,
        {"cycles": (changed_cycle, *loaded.cycles.cycles[1:])},
    )
    forged = _reseal_component(
        grounding_root,
        loaded,
        AlgorithmArtifactComponentName.CYCLES,
        changed_cycles,
    )
    with pytest.raises(ArtifactValidationError) as grounding_error:
        validate_algorithm_artifact(
            grounding_root / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(grounding_error, ArtifactValidationCode.INVALID_COMPONENT)


def test_wrong_out_of_band_digest_is_rejected(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
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

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest="f" * 64,
        )
    _assert_code(error, ArtifactValidationCode.EXPECTED_DIGEST_MISMATCH)


def test_component_byte_tamper_is_rejected_before_semantic_validation(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
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
    path = destination / "run.json"
    path.write_bytes(path.read_bytes() + b" ")
    os.chmod(path, 0o600)

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(destination / "manifest.json")
    _assert_code(error, ArtifactValidationCode.CONTENT_MISMATCH)


def test_manifest_digest_and_raw_call_order_tampering_fail_closed(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    manifest_root = tmp_path / "manifest"
    manifest = export_algorithm_artifact(
        fixed_stage2_result,
        manifest_root,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    manifest_values = manifest.model_dump(mode="python")
    manifest_values["manifest_digest"] = "f" * 64
    manifest_root.joinpath("manifest.json").write_bytes(
        canonical_json(to_jsonable_python(manifest_values))
    )
    with pytest.raises(ArtifactValidationError) as manifest_error:
        validate_algorithm_artifact(manifest_root / "manifest.json")
    _assert_code(manifest_error, ArtifactValidationCode.INVALID_MANIFEST)

    raw_root = tmp_path / "raw"
    export_algorithm_artifact(
        fixed_stage2_result,
        raw_root,
        execution=algorithm_execution,
        revision=clean_revision,
        classification=ArtifactClassification.SYNTHETIC_RAW,
    )
    loaded = load_validated_algorithm_artifact(raw_root / "manifest.json")
    attestations = algorithm_models.AlgorithmAttestationsComponent.model_validate_json(
        raw_root.joinpath("attestations.json").read_bytes()
    )
    assert attestations.raw_synthetic_result is not None
    raw_values = attestations.raw_synthetic_result.model_dump(mode="python")
    raw_values["call_receipts"] = tuple(reversed(raw_values["call_receipts"]))
    changed_attestations = _raw_reseal_payload(
        attestations,
        {"raw_synthetic_result": raw_values},
    )
    forged = _reseal_component(
        raw_root,
        loaded,
        AlgorithmArtifactComponentName.ATTESTATIONS,
        changed_attestations,
    )
    with pytest.raises(ArtifactValidationError) as raw_error:
        validate_algorithm_artifact(
            raw_root / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(raw_error, ArtifactValidationCode.INVALID_COMPONENT)


@pytest.mark.skipif(os.name != "posix", reason="POSIX link semantics")
def test_symbolic_and_hardlinked_algorithm_files_are_rejected(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    symbolic_root = tmp_path / "symbolic"
    export_algorithm_artifact(
        fixed_stage2_result,
        symbolic_root,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    calls = symbolic_root / "calls.json"
    outside = tmp_path / "outside.json"
    calls.rename(outside)
    calls.symlink_to(outside)
    with pytest.raises(ArtifactValidationError) as symbolic_error:
        validate_algorithm_artifact(symbolic_root / "manifest.json")
    _assert_code(symbolic_error, ArtifactValidationCode.UNSAFE_COMPONENT)

    hard_root = tmp_path / "hard"
    export_algorithm_artifact(
        fixed_stage2_result,
        hard_root,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    os.link(hard_root / "cycles.json", tmp_path / "linked-cycles.json")
    with pytest.raises(ArtifactValidationError) as hard_error:
        validate_algorithm_artifact(hard_root / "manifest.json")
    _assert_code(hard_error, ArtifactValidationCode.UNSAFE_COMPONENT)

    manifest_root = tmp_path / "manifest-link"
    export_algorithm_artifact(
        fixed_stage2_result,
        manifest_root,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    real_manifest = tmp_path / "real-manifest.json"
    manifest_root.joinpath("manifest.json").rename(real_manifest)
    manifest_root.joinpath("manifest.json").symlink_to(real_manifest)
    with pytest.raises(ArtifactValidationError) as manifest_error:
        validate_algorithm_artifact(manifest_root / "manifest.json")
    _assert_code(manifest_error, ArtifactValidationCode.UNSAFE_COMPONENT)


def test_missing_and_extra_files_are_rejected(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    missing = tmp_path / "missing"
    export_algorithm_artifact(
        fixed_stage2_result,
        missing,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    missing.joinpath("calls.json").unlink()
    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(missing / "manifest.json")
    _assert_code(error, ArtifactValidationCode.MISSING_COMPONENT)

    extra = tmp_path / "extra"
    export_algorithm_artifact(
        fixed_stage2_result,
        extra,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    extra.joinpath("extra.json").write_text("{}", encoding="utf-8")
    os.chmod(extra / "extra.json", 0o600)
    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(extra / "manifest.json")
    _assert_code(error, ArtifactValidationCode.UNSAFE_COMPONENT)


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        ("version-type", ArtifactValidationCode.INVALID_MANIFEST),
        ("future-version", ArtifactValidationCode.UNSUPPORTED_VERSION),
        ("foreign-version", ArtifactValidationCode.INVALID_MANIFEST),
        ("artifact-kind", ArtifactValidationCode.INVALID_MANIFEST),
        ("components-type", ArtifactValidationCode.INVALID_MANIFEST),
        ("component-type", ArtifactValidationCode.INVALID_MANIFEST),
        ("component-name-type", ArtifactValidationCode.INVALID_MANIFEST),
        ("component-name", ArtifactValidationCode.INVALID_MANIFEST),
        ("component-path", ArtifactValidationCode.UNSAFE_PATH),
    ),
)
def test_manifest_preflight_rejects_hostile_shapes_before_tree_reads(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    case: str,
    expected: ArtifactValidationCode,
) -> None:
    destination = tmp_path / case
    manifest = export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    values = manifest.model_dump(mode="json")
    if case == "version-type":
        values["schema_version"] = 1
    elif case == "future-version":
        values["schema_version"] = "algorithm-artifact/v2"
    elif case == "foreign-version":
        values["schema_version"] = "foreign-artifact/v1"
    elif case == "artifact-kind":
        values["artifact_kind"] = "fixture-secret"
    elif case == "components-type":
        values["components"] = None
    elif case == "component-type":
        values["components"] = [1]
    else:
        components = values["components"]
        assert isinstance(components, list)
        component = components[0]
        assert isinstance(component, dict)
        if case == "component-name-type":
            component["name"] = 1
        elif case == "component-name":
            component["name"] = "fixture-secret"
        else:
            component["path"] = "../fixture-secret.json"
    destination.joinpath("manifest.json").write_bytes(canonical_json(values))

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(destination / "manifest.json")
    _assert_code(error, expected)


@pytest.mark.parametrize(
    "payload",
    (
        b'[{"schema_version":"algorithm-artifact/v1"}]',
        b'{"schema_version":"algorithm-artifact/v1","schema_version":"algorithm-artifact/v1"}',
        b'{"schema_version":"algorithm-artifact/v1","value":NaN}',
        b'{"schema_version": "algorithm-artifact/v1"}',
    ),
)
def test_noncanonical_manifests_are_value_free_failures(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    payload: bytes,
) -> None:
    destination = tmp_path / str(len(payload))
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    destination.joinpath("manifest.json").write_bytes(payload)

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(destination / "manifest.json")
    _assert_code(error, ArtifactValidationCode.INVALID_MANIFEST)


@pytest.mark.parametrize(
    "payload",
    (
        b"[]",
        b'{"schema_version":"algorithm-run/v1","schema_version":"algorithm-run/v1"}',
        b'{"schema_version":"algorithm-run/v1","value":NaN}',
        b'{"schema_version": "algorithm-run/v1"}',
        b'{"schema_version":"algorithm-run/v1"}',
    ),
)
def test_resealed_noncanonical_components_are_invalid_not_content_mismatches(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    payload: bytes,
) -> None:
    destination = tmp_path / str(len(payload))
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    forged = _reseal_component(
        destination,
        loaded,
        AlgorithmArtifactComponentName.RUN,
        payload,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.INVALID_COMPONENT)


def test_public_validator_rejects_invalid_paths_and_digest_options(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
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
    manifest_path = destination / "manifest.json"

    cases: tuple[tuple[object, object, ArtifactValidationCode], ...] = (
        (b"manifest.json", None, ArtifactValidationCode.UNSAFE_PATH),
        (destination / "run.json", None, ArtifactValidationCode.UNSAFE_PATH),
        (manifest_path, 7, ArtifactValidationCode.EXPECTED_DIGEST_MISMATCH),
        (manifest_path, "not-a-digest", ArtifactValidationCode.EXPECTED_DIGEST_MISMATCH),
    )
    for path, expected_digest, expected_code in cases:
        with pytest.raises(ArtifactValidationError) as error:
            load_validated_algorithm_artifact(
                path,  # type: ignore[arg-type]
                expected_manifest_digest=expected_digest,  # type: ignore[arg-type]
            )
        _assert_code(error, expected_code)


def test_resealed_counters_and_component_run_ids_are_reconciled(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    counters_root = tmp_path / "counters"
    export_algorithm_artifact(
        fixed_stage2_result,
        counters_root,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(counters_root / "manifest.json")
    changed_counters = algorithm_models.AlgorithmArtifactCounters.model_validate(
        {
            **loaded.manifest.counters.model_dump(mode="python"),
            "ledger_entries": loaded.manifest.counters.ledger_entries + 1,
        }
    )
    forged = _reseal_components(
        counters_root,
        loaded,
        {},
        manifest_counters=changed_counters,
    )
    with pytest.raises(ArtifactValidationError) as counters_error:
        validate_algorithm_artifact(
            counters_root / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(counters_error, ArtifactValidationCode.INCONSISTENT_COUNTERS)

    run_root = tmp_path / "run"
    export_algorithm_artifact(
        fixed_stage2_result,
        run_root,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(run_root / "manifest.json")
    changed_deliveries = _resealed_model(
        loaded.deliveries,
        {"run_id": UUID("00000000-0000-4000-8000-000000009999")},
    )
    forged = _reseal_component(
        run_root,
        loaded,
        AlgorithmArtifactComponentName.DELIVERIES,
        changed_deliveries,
    )
    with pytest.raises(ArtifactValidationError) as run_error:
        validate_algorithm_artifact(
            run_root / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(run_error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


@pytest.mark.parametrize("projection", ("cycle", "delivery", "memory", "ledger"))
def test_raw_assurance_recomputes_every_projected_record(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    projection: str,
) -> None:
    destination = tmp_path / projection
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
        classification=ArtifactClassification.SYNTHETIC_RAW,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    replacements: dict[AlgorithmArtifactComponentName, object]
    if projection == "cycle":
        cycle = loaded.cycles.cycles[0]
        changed_cycle = _resealed_model(
            cycle,
            {"boundary_evidence_digest": "f" * 64},
        )
        replacements = {
            AlgorithmArtifactComponentName.CYCLES: _resealed_model(
                loaded.cycles,
                {"cycles": (changed_cycle, *loaded.cycles.cycles[1:])},
            )
        }
    elif projection == "delivery":
        delivery = loaded.deliveries.deliveries[0]
        values = delivery.model_dump(mode="python")
        values["adapter_id_digest"] = "f" * 64
        values["binding_digest"] = delivery_binding_digest(values)
        changed_delivery = type(delivery).model_validate(values)
        replacements = {
            AlgorithmArtifactComponentName.DELIVERIES: _resealed_model(
                loaded.deliveries,
                {"deliveries": (changed_delivery, *loaded.deliveries.deliveries[1:])},
            )
        }
    elif projection == "memory":
        memory = loaded.metrics.final_memory
        changed_digest = "f" * 64
        if changed_digest in memory.record_digests:
            changed_digest = "e" * 64
        changed_memory = type(memory).model_validate(
            {
                **memory.model_dump(mode="python"),
                "record_digests": (changed_digest, *memory.record_digests[1:]),
            }
        )
        replacements = {
            AlgorithmArtifactComponentName.METRICS: _resealed_model(
                loaded.metrics,
                {"final_memory": changed_memory},
            )
        }
    else:
        attestations = algorithm_models.AlgorithmAttestationsComponent.model_validate_json(
            destination.joinpath("attestations.json").read_bytes()
        )
        index, ledger_entry = next(
            (index, item)
            for index, item in enumerate(attestations.ledger_entries)
            if item.record_type == "cycle_record" and item.record_revision == 1
        )
        changed_entry = type(ledger_entry).model_validate(
            {
                **ledger_entry.model_dump(mode="python"),
                "source_record_digest": "f" * 64,
            }
        )
        entries = list(attestations.ledger_entries)
        entries[index] = changed_entry
        replacements = {
            AlgorithmArtifactComponentName.ATTESTATIONS: _resealed_model(
                attestations,
                {"ledger_entries": tuple(entries)},
            )
        }
    forged = _reseal_components(destination, loaded, replacements)

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(
        error,
        (
            ArtifactValidationCode.UNGROUNDED_DELIVERY
            if projection == "delivery"
            else ArtifactValidationCode.CROSS_COMPONENT_INVARIANT
        ),
    )


def test_raw_reprojection_failure_is_value_free_and_fail_closed(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "raw-reprojection"
    manifest = export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
        classification=ArtifactClassification.SYNTHETIC_RAW,
    )

    def fail_projection(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("private sentinel")

    monkeypatch.setattr(
        algorithm_projection,
        "_project_algorithm_components",
        fail_projection,
    )
    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=manifest.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    assert "private sentinel" not in str(error.value)


def test_raw_reprojection_revalidates_execution_binding_before_projection(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "raw-binding"
    manifest = export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
        classification=ArtifactClassification.SYNTHETIC_RAW,
    )
    calls: list[str] = []

    def fail_binding(*_args: object, **_kwargs: object) -> None:
        calls.append("binding")
        raise RuntimeError("private binding sentinel")

    def record_projection(*_args: object, **_kwargs: object) -> None:
        calls.append("projection")

    monkeypatch.setattr(
        algorithm_projection,
        "_validate_source_execution_binding",
        fail_binding,
    )
    monkeypatch.setattr(
        algorithm_projection,
        "_project_algorithm_components",
        record_projection,
    )
    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=manifest.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)
    assert calls == ["binding"]
    assert "private binding sentinel" not in str(error.value)
    assert "private binding sentinel" not in repr(error.value)


def test_deterministic_decision_and_terminal_cycle_revision_survive_resealing(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    decision_root = tmp_path / "decision"
    export_algorithm_artifact(
        fixed_stage2_result,
        decision_root,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(decision_root / "manifest.json")
    index = next(
        index for index, item in enumerate(loaded.trajectory.schedule.decisions) if not item.invoke
    )
    original_decision = loaded.decisions.decisions[index]
    changed_decision = type(original_decision).model_validate(
        {
            **original_decision.model_dump(mode="python"),
            "decision_id": UUID("00000000-0000-4000-8000-000000009999"),
        }
    )
    changed_decisions = list(loaded.decisions.decisions)
    changed_decisions[index] = changed_decision
    changed_component = _resealed_model(
        loaded.decisions,
        {"decisions": tuple(changed_decisions)},
    )
    forged = _reseal_component(
        decision_root,
        loaded,
        AlgorithmArtifactComponentName.DECISIONS,
        changed_component,
    )
    with pytest.raises(ArtifactValidationError) as decision_error:
        validate_algorithm_artifact(
            decision_root / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(decision_error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)

    cycle_root = tmp_path / "cycle"
    export_algorithm_artifact(
        fixed_stage2_result,
        cycle_root,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(cycle_root / "manifest.json")
    first_cycle = loaded.cycles.cycles[0]
    changed_cycle = _resealed_model(first_cycle, {"revision": 5})
    changed_cycles = _resealed_model(
        loaded.cycles,
        {"cycles": (changed_cycle, *loaded.cycles.cycles[1:])},
    )
    forged = _reseal_component(
        cycle_root,
        loaded,
        AlgorithmArtifactComponentName.CYCLES,
        changed_cycles,
    )
    with pytest.raises(ArtifactValidationError) as cycle_error:
        validate_algorithm_artifact(
            cycle_root / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(cycle_error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


@pytest.mark.parametrize(
    ("record_type", "revision", "changed_state"),
    (
        ("trace_event", None, "pending"),
        ("invocation_decision", None, "pending"),
        ("cycle_record", 2, "running"),
        ("delivery_record", 2, "attempting"),
        ("intervention_outcome", None, "pending"),
    ),
)
def test_digest_only_ledger_metadata_has_one_canonical_layout(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    record_type: str,
    revision: int | None,
    changed_state: str,
) -> None:
    destination = tmp_path / record_type
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    attestations = algorithm_models.AlgorithmAttestationsComponent.model_validate_json(
        destination.joinpath("attestations.json").read_bytes()
    )
    index, original = next(
        (index, item)
        for index, item in enumerate(attestations.ledger_entries)
        if item.record_type == record_type and item.record_revision == revision
    )
    changed = type(original).model_validate(
        {**original.model_dump(mode="python"), "record_state": changed_state}
    )
    entries = list(attestations.ledger_entries)
    entries[index] = changed
    changed_attestations = _resealed_model(
        attestations,
        {"ledger_entries": tuple(entries)},
    )
    forged = _reseal_component(
        destination,
        loaded,
        AlgorithmArtifactComponentName.ATTESTATIONS,
        changed_attestations,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def test_trajectory_binding_cannot_diverge_from_the_ledger_attestation(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    destination = tmp_path / "binding"
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    first = loaded.trajectory.records[0]
    binding_values = first.binding.model_dump(mode="python", exclude={"binding_digest"})
    binding_values["record_tag"] = PayloadDigest(
        algorithm=first.binding.record_tag.algorithm,
        value="f" * 64,
    )
    changed_binding = type(first.binding).model_validate(binding_values)
    changed_record = type(first).model_validate(
        {**first.model_dump(mode="python"), "binding": changed_binding}
    )
    records = (changed_record, *loaded.trajectory.records[1:])
    request_digest = algorithm_models._redacted_prefix_request_digest(
        loaded.trajectory.run_id,
        records,
    )
    prefix_digest = algorithm_models._redacted_prefix_digest(
        loaded.trajectory.run_id,
        records,
        request_digest,
    )
    schedule_values = loaded.trajectory.schedule.model_dump(
        mode="python",
        exclude={"schedule_digest"},
    )
    schedule_values["trajectory_prefix_digest"] = prefix_digest
    schedule = FixedStepSchedule.model_validate(schedule_values)
    changed_windows = []
    for window in loaded.trajectory.windows:
        window_records = records[: window.boundary_event_sequence]
        window_request_digest = algorithm_models._redacted_prefix_request_digest(
            loaded.trajectory.run_id,
            window_records,
        )
        changed_windows.append(
            type(window).model_validate(
                {
                    **window.model_dump(mode="python"),
                    "trajectory_prefix_digest": algorithm_models._redacted_prefix_digest(
                        loaded.trajectory.run_id,
                        window_records,
                        window_request_digest,
                    ),
                    "source_attestation_digests": (
                        algorithm_models._redacted_window_source_digests(window_records)
                    ),
                }
            )
        )
    changed_trajectory = _resealed_model(
        loaded.trajectory,
        {
            "records": records,
            "trajectory_prefix_request_digest": request_digest,
            "trajectory_prefix_digest": prefix_digest,
            "schedule": schedule,
            "windows": tuple(changed_windows),
        },
    )
    forged = _reseal_component(
        destination,
        loaded,
        AlgorithmArtifactComponentName.TRAJECTORY,
        changed_trajectory,
        manifest_schedule_digest=schedule.schedule_digest,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def test_window_boundary_cannot_diverge_from_the_ledger_head(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    destination = tmp_path / "window-ledger"
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    first = loaded.trajectory.windows[0]
    changed_window = type(first).model_validate(
        {
            **first.model_dump(mode="python"),
            "boundary_chain_tag": PayloadDigest(
                algorithm=first.boundary_chain_tag.algorithm,
                value="f" * 64,
            ),
        }
    )
    changed_trajectory = _resealed_model(
        loaded.trajectory,
        {"windows": (changed_window, *loaded.trajectory.windows[1:])},
    )
    forged = _reseal_component(
        destination,
        loaded,
        AlgorithmArtifactComponentName.TRAJECTORY,
        changed_trajectory,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


@pytest.mark.parametrize("projection", ("prefix", "sources", "version"))
def test_window_provenance_is_rebuilt_from_the_redacted_prefix(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    projection: str,
) -> None:
    destination = tmp_path / f"window-{projection}"
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    windows = [item.model_dump(mode="python") for item in loaded.trajectory.windows]
    target = (
        next(item for item in windows if item["source_attestation_digests"])
        if projection == "sources"
        else windows[0]
    )
    if projection == "prefix":
        target["trajectory_prefix_digest"] = "f" * 64
    elif projection == "sources":
        sources = list(target["source_attestation_digests"])
        sources[0] = "f" * 64 if sources[0] != "f" * 64 else "e" * 64
        target["source_attestation_digests"] = tuple(sources)
    else:
        target["version"] = "latest-eight-logical-messages/v2"
    forged_trajectory = _raw_reseal_payload(
        loaded.trajectory,
        {"windows": tuple(windows)},
    )
    forged = _reseal_component(
        destination,
        loaded,
        AlgorithmArtifactComponentName.TRAJECTORY,
        forged_trajectory,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.INVALID_COMPONENT)


@pytest.mark.parametrize("projection", ("trust", "run_start", "timestamp", "parents"))
def test_digest_only_trajectory_preserves_the_synthetic_fixture_shape(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    projection: str,
) -> None:
    destination = tmp_path / f"trajectory-{projection}"
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    records = [item.model_dump(mode="python") for item in loaded.trajectory.records]
    if projection == "trust":
        records[0]["trust_label"] = TrustLabel.UNTRUSTED_TASK_INPUT
    elif projection == "run_start":
        records[0]["event_type"] = EventType.MODEL_OUTPUT
    elif projection == "timestamp":
        records[1]["event_timestamp"] = records[0]["event_timestamp"] - timedelta(seconds=1)
    else:
        parents = records[4]["parent_ids"]
        assert len(parents) == 2
        records[4]["parent_ids"] = tuple(reversed(parents))
    forged_trajectory = _raw_reseal_payload(
        loaded.trajectory,
        {"records": tuple(records)},
    )
    forged = _reseal_component(
        destination,
        loaded,
        AlgorithmArtifactComponentName.TRAJECTORY,
        forged_trajectory,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.INVALID_COMPONENT)


@pytest.mark.parametrize(
    "projection",
    ("evidence", "decision", "schedule", "current_bank", "candidate_bank", "delivery"),
)
def test_boundary_projection_reconciles_its_source_bindings(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    projection: str,
) -> None:
    destination = tmp_path / f"boundary-{projection}"
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    if projection == "evidence":
        cycle = loaded.cycles.cycles[0]
        changed_cycle = _resealed_model(cycle, {"boundary_evidence_digest": "f" * 64})
        replacements = {
            AlgorithmArtifactComponentName.CYCLES: _resealed_model(
                loaded.cycles,
                {"cycles": (changed_cycle, *loaded.cycles.cycles[1:])},
            )
        }
    else:
        attestations = algorithm_models.AlgorithmAttestationsComponent.model_validate_json(
            destination.joinpath("attestations.json").read_bytes()
        )
        boundary = next(
            item
            for item in attestations.boundaries
            if item.observation.observed.delivery_record_digests
        )
        observed_values = boundary.observation.observed.model_dump(mode="python")
        if projection == "decision":
            observed_values["invocation_decision_digest"] = "f" * 64
        elif projection == "schedule":
            final_schedule_digest = loaded.trajectory.schedule.schedule_digest
            assert observed_values["schedule_digest"] != final_schedule_digest
            observed_values["schedule_digest"] = final_schedule_digest
        elif projection == "current_bank":
            observed_values["current_bank_view_digest"] = "f" * 64
        elif projection == "candidate_bank":
            observed_values["candidate_bank_view_digest"] = "f" * 64
        else:
            observed_values["delivery_record_digests"] = ("f" * 64,)
        changed_observed = type(boundary.observation.observed).model_validate(observed_values)
        observation_values = boundary.observation.model_dump(
            mode="python",
            exclude={"observation_digest"},
        )
        observation_values["observed"] = changed_observed
        changed_observation = type(boundary.observation).model_validate(observation_values)
        changed_boundary = _resealed_model(
            boundary,
            {"observation": changed_observation},
        )
        replacements = {
            AlgorithmArtifactComponentName.ATTESTATIONS: _resealed_model(
                attestations,
                {
                    "boundaries": tuple(
                        changed_boundary if item is boundary else item
                        for item in attestations.boundaries
                    )
                },
            )
        }
    forged = _reseal_components(destination, loaded, replacements)

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def test_no_memory_boundary_names_the_canonical_decision(
    tmp_path: Path,
    no_memory_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    execution_values = algorithm_execution.model_dump(
        mode="python",
        exclude={"schema_version", "execution_digest", "response_fixture"},
    )
    execution = AlgorithmExecutionAttestation.create(
        **execution_values,
        response_fixture=None,
    )
    destination = tmp_path / "no-memory-boundary"
    export_algorithm_artifact(
        no_memory_stage2_result,
        destination,
        execution=execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    attestations = algorithm_models.AlgorithmAttestationsComponent.model_validate_json(
        destination.joinpath("attestations.json").read_bytes()
    )
    first = attestations.boundaries[0]
    observed = first.observation.observed
    changed_id = UUID("00000000-0000-4000-8000-000000009999")
    changed_observed = type(observed).model_validate(
        {**observed.model_dump(mode="python"), "invocation_decision_id": changed_id}
    )
    observation_values = first.observation.model_dump(
        mode="python",
        exclude={"observation_digest"},
    )
    observation_values["observed"] = changed_observed
    changed_observation = type(first.observation).model_validate(observation_values)
    changed_boundary = _resealed_model(
        first,
        {
            "invocation_decision_id": changed_id,
            "observation": changed_observation,
        },
    )
    changed_attestations = _resealed_model(
        attestations,
        {"boundaries": (changed_boundary, *attestations.boundaries[1:])},
    )
    forged = _reseal_component(
        destination,
        loaded,
        AlgorithmArtifactComponentName.ATTESTATIONS,
        changed_attestations,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


def test_digest_only_delivery_is_the_completed_stage2_delivery(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> None:
    destination = tmp_path / "failed-delivery"
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    first = loaded.deliveries.deliveries[0]
    values = first.model_dump(mode="python")
    values.update(
        state=DeliveryState.FAILED,
        receipt_digest=None,
        outcome=DeliveryOutcome.FAILED,
        reason_code=ReasonCode.DELIVERY_FAILED,
    )
    changed = type(first).model_validate(values)
    changed_deliveries = _resealed_model(
        loaded.deliveries,
        {"deliveries": (changed, *loaded.deliveries.deliveries[1:])},
    )
    forged = _reseal_component(
        destination,
        loaded,
        AlgorithmArtifactComponentName.DELIVERIES,
        changed_deliveries,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.UNGROUNDED_DELIVERY)


@pytest.mark.parametrize(
    "projection",
    (
        "attempt",
        "target",
        "adapter",
        "capabilities",
        "contract",
        "worker",
        "receipt",
        "timestamp",
    ),
)
def test_digest_only_delivery_preserves_the_exact_stage2_adapter_path(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    projection: str,
) -> None:
    destination = tmp_path / f"delivery-{projection}"
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    loaded = load_validated_algorithm_artifact(destination / "manifest.json")
    first = loaded.deliveries.deliveries[0]
    values = first.model_dump(mode="python", exclude={"binding_digest"})
    if projection == "attempt":
        values["attempt_count"] = 2
    elif projection == "target":
        values["target"] = DeliveryTarget.PRE_ACTION_REPLAN
        values["adapter_supports_pre_action"] = True
    elif projection == "adapter":
        values["adapter_id_digest"] = "f" * 64
    elif projection == "capabilities":
        values["adapter_capabilities_digest"] = "f" * 64
    elif projection == "contract":
        values["adapter_contract_version"] = "adapter-contract/v2"
    elif projection == "worker":
        values["claim_id"] = UUID("00000000-0000-4000-8000-000000009999")
    elif projection == "receipt":
        values["receipt_digest"] = "f" * 64
    else:
        values["created_at"] = first.created_at - timedelta(microseconds=1)
    values["binding_digest"] = delivery_binding_digest(values)
    changed = type(first).model_validate(values)
    changed_deliveries = _resealed_model(
        loaded.deliveries,
        {"deliveries": (changed, *loaded.deliveries.deliveries[1:])},
    )
    forged = _reseal_component(
        destination,
        loaded,
        AlgorithmArtifactComponentName.DELIVERIES,
        changed_deliveries,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=forged.manifest_digest,
        )
    _assert_code(error, ArtifactValidationCode.UNGROUNDED_DELIVERY)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("classification", ArtifactClassification.USER_REDACTED),
        ("evidence_level", "confirmatory"),
        ("confirmatory_eligible", True),
    ),
)
def test_v1_manifest_rejects_resealed_nonsynthetic_claims(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
    field: str,
    value: object,
) -> None:
    destination = tmp_path / field
    manifest = export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    values = manifest.model_dump(mode="json", exclude={"manifest_digest"})
    values[field] = value
    digest = algorithm_models.algorithm_artifact_manifest_digest(values)
    values["manifest_digest"] = digest
    destination.joinpath("manifest.json").write_bytes(canonical_json(values))

    with pytest.raises(ArtifactValidationError) as error:
        validate_algorithm_artifact(
            destination / "manifest.json",
            expected_manifest_digest=digest,
        )
    _assert_code(error, ArtifactValidationCode.INVALID_MANIFEST)
