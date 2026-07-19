from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from pydantic import BaseModel, ValidationError

from saliencegate.artifacts import algorithm_manifest as algorithm_models
from saliencegate.artifacts.algorithm_export import export_algorithm_artifact
from saliencegate.artifacts.algorithm_manifest import (
    ALGORITHM_ARTIFACT_SCHEMA_VERSION,
    AlgorithmArtifactComponent,
    AlgorithmArtifactComponentName,
    AlgorithmArtifactCounters,
    AlgorithmArtifactManifest,
    AlgorithmCheckpointAttestation,
    AlgorithmCycleMode,
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
    algorithm_artifact_manifest_digest,
    algorithm_component_content_digest,
    algorithm_cycle_mode_for_condition,
    expected_algorithm_component_path,
)
from saliencegate.artifacts.algorithm_validate import (
    ValidatedAlgorithmArtifact,
    load_validated_algorithm_artifact,
)
from saliencegate.artifacts.manifest import (
    ArtifactClassification,
    ArtifactEvidenceLevel,
    RevisionEvidence,
    RevisionSource,
    component_content_digest,
)
from saliencegate.domain import CycleState, PayloadDigest, PayloadDigestAlgorithm
from saliencegate.experiments import (
    Stage2ConditionId,
    Stage2ExperimentRunResult,
    resolve_stage2_condition,
)

RUN_ID = UUID("00000000-0000-4000-8000-000000009000")
DIGEST = resolve_stage2_condition(Stage2ConditionId.FIXED_STEP).condition_digest
OTHER_DIGEST = "2" * 64
OTHER_RUN_ID = UUID("00000000-0000-4000-8000-000000009001")


def _revision(*, clean: bool = True) -> RevisionEvidence:
    return RevisionEvidence(
        source=RevisionSource.GIT,
        package_version="0.1.0",
        commit="a" * 40,
        dirty_worktree=not clean,
    )


def _environment() -> AlgorithmExecutionAttestation:
    return AlgorithmExecutionAttestation.create(
        endpoint_classification=AlgorithmEndpointClassification.OFFLINE_REPLAY,
        runtime_id="saliencegate-two-phase-replay",
        runtime_version="1.0.0",
        checkpoint=AlgorithmCheckpointAttestation(
            model_id="paper-two-phase-fixture",
            model_tag="paper-two-phase-fixture/v1",
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
            model_id="paper-two-phase-fixture",
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
    )


def _counters() -> AlgorithmArtifactCounters:
    return AlgorithmArtifactCounters(
        events=11,
        scheduled_invocations=3,
        decisions=11,
        cycles=3,
        requests=3,
        model_calls=6,
        deliveries=1,
        outcomes=3,
        ledger_entries=41,
    )


def _components(counters: AlgorithmArtifactCounters) -> tuple[AlgorithmArtifactComponent, ...]:
    counts = {
        AlgorithmArtifactComponentName.ATTESTATIONS: 1,
        AlgorithmArtifactComponentName.CALLS: counters.model_calls,
        AlgorithmArtifactComponentName.CYCLES: counters.cycles,
        AlgorithmArtifactComponentName.DECISIONS: counters.decisions,
        AlgorithmArtifactComponentName.DELIVERIES: counters.deliveries,
        AlgorithmArtifactComponentName.METRICS: 1,
        AlgorithmArtifactComponentName.OUTCOMES: counters.outcomes,
        AlgorithmArtifactComponentName.RUN: 1,
        AlgorithmArtifactComponentName.TRAJECTORY: counters.events,
    }
    return tuple(
        AlgorithmArtifactComponent(
            name=name,
            path=expected_algorithm_component_path(name),
            byte_count=2,
            record_count=counts[name],
            content_digest=algorithm_component_content_digest(name, b"{}"),
        )
        for name in sorted(AlgorithmArtifactComponentName, key=lambda item: item.value)
    )


def _manifest(
    *,
    classification: ArtifactClassification = ArtifactClassification.SYNTHETIC_DIGEST_ONLY,
    revision: RevisionEvidence | None = None,
    condition_id: Stage2ConditionId = Stage2ConditionId.FIXED_STEP,
    cycle_mode: AlgorithmCycleMode = AlgorithmCycleMode.TWO_PHASE,
    counters: AlgorithmArtifactCounters | None = None,
    components: tuple[AlgorithmArtifactComponent, ...] | None = None,
) -> AlgorithmArtifactManifest:
    checked_counters = _counters() if counters is None else counters
    environment = _environment()
    condition = resolve_stage2_condition(condition_id)
    return AlgorithmArtifactManifest.create(
        classification=classification,
        run_id=RUN_ID,
        revision=_revision() if revision is None else revision,
        condition_id=condition_id,
        condition_digest=condition.condition_digest,
        cycle_mode=cycle_mode,
        trace_digest="9" * 64,
        schedule_digest=OTHER_DIGEST,
        window_digests=("3" * 64, "4" * 64, "5" * 64),
        prompt_bundle_digest="6" * 64,
        model_profile_digest="7" * 64,
        execution=environment,
        result_digest="8" * 64,
        components=_components(checked_counters) if components is None else components,
        counters=checked_counters,
    )


@pytest.fixture
def exported_algorithm(
    tmp_path: Path,
    fixed_stage2_result: Stage2ExperimentRunResult,
    algorithm_execution: AlgorithmExecutionAttestation,
    clean_revision: RevisionEvidence,
) -> tuple[ValidatedAlgorithmArtifact, Path]:
    destination = tmp_path / "artifact"
    export_algorithm_artifact(
        fixed_stage2_result,
        destination,
        execution=algorithm_execution,
        revision=clean_revision,
    )
    return load_validated_algorithm_artifact(destination / "manifest.json"), destination


def test_algorithm_manifest_is_closed_canonical_and_self_attesting() -> None:
    manifest = _manifest()

    assert manifest.schema_version == ALGORITHM_ARTIFACT_SCHEMA_VERSION
    assert manifest.artifact_kind == "algorithm_run"
    assert manifest.record_type == "algorithm_artifact_manifest"
    assert tuple(item.name for item in manifest.components) == tuple(
        sorted(AlgorithmArtifactComponentName, key=lambda item: item.value)
    )
    assert tuple(item.path for item in manifest.components) == (
        "attestations.json",
        "calls.json",
        "cycles.json",
        "decisions.json",
        "deliveries.json",
        "metrics.json",
        "outcomes.json",
        "run.json",
        "trajectory.json",
    )
    assert manifest.execution_digest == manifest.execution.execution_digest
    assert not manifest.confirmatory_eligible
    assert not manifest.confirmatory
    assert manifest.evidence_level is ArtifactEvidenceLevel.EXPLORATORY
    assert algorithm_component_content_digest(
        AlgorithmArtifactComponentName.RUN, b"{}"
    ) != component_content_digest(b"{}")

    with pytest.raises(ValidationError):
        manifest.condition_digest = OTHER_DIGEST  # type: ignore[misc]


@pytest.mark.parametrize(
    "field",
    (
        "condition_digest",
        "schedule_digest",
        "prompt_bundle_digest",
        "model_profile_digest",
        "execution_digest",
        "result_digest",
    ),
)
def test_manifest_digest_binds_every_algorithm_configuration_anchor(field: str) -> None:
    manifest = _manifest()
    values = manifest.model_dump(mode="python", exclude={"manifest_digest"})
    values[field] = "f" * 64

    with pytest.raises(ValidationError):
        AlgorithmArtifactManifest.model_validate(
            values | {"manifest_digest": manifest.manifest_digest}
        )


def test_component_set_and_paths_are_exact() -> None:
    manifest = _manifest()
    values = manifest.model_dump(mode="python", exclude={"manifest_digest"})
    values["components"] = values["components"][:-1]

    with pytest.raises(ValidationError):
        AlgorithmArtifactManifest.model_validate(
            values | {"manifest_digest": manifest.manifest_digest}
        )

    component = manifest.components[0]
    with pytest.raises(ValidationError):
        AlgorithmArtifactComponent.model_validate(
            component.model_dump(mode="python") | {"path": "../attestations.json"}
        )


@pytest.mark.parametrize(
    "forbidden",
    ("hostname", "serial_number", "username", "home", "device_id", "account", "organization"),
)
def test_hardware_schema_has_no_dedicated_identity_fields(forbidden: str) -> None:
    values = _environment().hardware.model_dump(mode="python") | {forbidden: "sentinel"}

    with pytest.raises(ValidationError):
        AlgorithmHardwareAttestation.model_validate(values)


def test_checkpoint_requires_exactly_one_stable_checkpoint_identity() -> None:
    base = _environment().checkpoint.model_dump(mode="python")

    with pytest.raises(ValidationError):
        AlgorithmCheckpointAttestation.model_validate(
            base | {"model_tag": None, "checkpoint_digest": None}
        )
    with pytest.raises(ValidationError):
        AlgorithmCheckpointAttestation.model_validate(base | {"checkpoint_digest": "b" * 64})

    with pytest.raises(ValidationError, match="digest"):
        AlgorithmCheckpointAttestation.model_validate(
            base | {"checkpoint_attestation_digest": "f" * 64}
        )


def test_sampling_modes_are_closed_and_self_attesting() -> None:
    replay = _environment().sampling
    provider = AlgorithmSamplingAttestation(
        mode=AlgorithmSamplingMode.OPENAI_COMPATIBLE,
        temperature=0.0,
        seed=7,
        reasoning_effort="medium",
    )

    assert replay.sampling_digest is not None
    assert provider.sampling_digest is not None
    assert provider.sampling_digest != replay.sampling_digest

    with pytest.raises(ValidationError, match="frozen replay"):
        AlgorithmSamplingAttestation(
            mode=AlgorithmSamplingMode.FROZEN_REPLAY,
            temperature=0.0,
            seed=None,
            reasoning_effort=None,
        )
    with pytest.raises(ValidationError, match="controls are incomplete"):
        AlgorithmSamplingAttestation(
            mode=AlgorithmSamplingMode.OPENAI_COMPATIBLE,
            temperature=0.5,
            seed=7,
            reasoning_effort="medium",
        )
    with pytest.raises(ValidationError, match="digest"):
        AlgorithmSamplingAttestation.model_validate(
            replay.model_dump(mode="python") | {"sampling_digest": "f" * 64}
        )


def test_tokenizer_status_has_one_exact_identity_shape() -> None:
    attested = _environment().tokenizer
    unavailable = AlgorithmTokenizerAttestation(
        status=AlgorithmTokenizerStatus.UNAVAILABLE,
        tokenizer_id=None,
        tokenizer_version=None,
        configuration_digest=None,
        model_id=None,
    )

    assert unavailable.tokenizer_digest is not None
    with pytest.raises(ValidationError, match="identity is incomplete"):
        AlgorithmTokenizerAttestation.model_validate(
            attested.model_dump(mode="python") | {"tokenizer_id": None}
        )
    with pytest.raises(ValidationError, match="cannot carry an identity"):
        AlgorithmTokenizerAttestation(
            status=AlgorithmTokenizerStatus.UNAVAILABLE,
            tokenizer_id="unexpected/v1",
            tokenizer_version=None,
            configuration_digest=None,
            model_id=None,
        )
    with pytest.raises(ValidationError, match="digest"):
        AlgorithmTokenizerAttestation.model_validate(
            attested.model_dump(mode="python") | {"tokenizer_digest": "f" * 64}
        )


@pytest.mark.parametrize(
    "value",
    (
        " leading-space",
        "trailing-space ",
        "host/name",
        "https://benchmark-host",
        "alice@example.com",
        "control\ncharacter",
        "x" * 257,
    ),
)
def test_hardware_labels_reject_obvious_identifier_shapes(value: str) -> None:
    values = _environment().hardware.model_dump(mode="python")

    with pytest.raises(ValidationError):
        AlgorithmHardwareAttestation.model_validate(values | {"model": value})


def test_hardware_labels_are_caller_deidentified_not_anonymity_proof() -> None:
    values = _environment().hardware.model_dump(
        mode="python",
        exclude={"hardware_digest"},
    )
    values["model"] = "benchmark-host-01"

    attestation = AlgorithmHardwareAttestation.model_validate(values)

    assert attestation.model == "benchmark-host-01"


def test_hardware_digest_and_response_fixture_consumption_are_exact() -> None:
    hardware = _environment().hardware
    with pytest.raises(ValidationError, match="digest"):
        AlgorithmHardwareAttestation.model_validate(
            hardware.model_dump(mode="python") | {"hardware_digest": "f" * 64}
        )

    with pytest.raises(ValidationError, match="fully consumed"):
        AlgorithmResponseFixtureAttestation(
            replay_id="stage2-replay/v1",
            fixture_digest="f" * 64,
            response_count=2,
            consumed_count=1,
        )


def test_execution_modes_bind_endpoint_sampling_warmup_and_model() -> None:
    replay = _environment()
    provider_sampling = AlgorithmSamplingAttestation(
        mode=AlgorithmSamplingMode.OPENAI_COMPATIBLE,
        temperature=0.0,
        seed=7,
        reasoning_effort="medium",
    )
    live = AlgorithmExecutionAttestation.create(
        execution_mode=AlgorithmExecutionMode.OPENAI_COMPATIBLE,
        endpoint_classification=AlgorithmEndpointClassification.LOOPBACK_OPENAI_COMPATIBLE,
        runtime_id=replay.runtime_id,
        runtime_version=replay.runtime_version,
        checkpoint=replay.checkpoint,
        sampling=provider_sampling,
        tokenizer=replay.tokenizer,
        hardware=replay.hardware,
        warmup_policy=AlgorithmWarmupPolicy.COLD,
    )
    assert live.execution_digest != replay.execution_digest

    with pytest.raises(ValidationError, match="frozen replay execution"):
        AlgorithmExecutionAttestation.create(
            endpoint_classification=(AlgorithmEndpointClassification.LOOPBACK_OPENAI_COMPATIBLE),
            runtime_id=replay.runtime_id,
            runtime_version=replay.runtime_version,
            checkpoint=replay.checkpoint,
            sampling=replay.sampling,
            tokenizer=replay.tokenizer,
            hardware=replay.hardware,
            warmup_policy=AlgorithmWarmupPolicy.NOT_APPLICABLE,
        )
    with pytest.raises(ValidationError, match="live execution"):
        AlgorithmExecutionAttestation.create(
            execution_mode=AlgorithmExecutionMode.OPENAI_COMPATIBLE,
            endpoint_classification=AlgorithmEndpointClassification.OFFLINE_REPLAY,
            runtime_id=replay.runtime_id,
            runtime_version=replay.runtime_version,
            checkpoint=replay.checkpoint,
            sampling=provider_sampling,
            tokenizer=replay.tokenizer,
            hardware=replay.hardware,
            warmup_policy=AlgorithmWarmupPolicy.COLD,
        )

    mismatched_tokenizer = AlgorithmTokenizerAttestation(
        status=AlgorithmTokenizerStatus.ATTESTED,
        tokenizer_id="stage2-reviewed-utf8-counter/v1",
        tokenizer_version="utf8-bytes-ceil-div-4/v1",
        configuration_digest="4" * 64,
        model_id="different-model",
    )
    with pytest.raises(ValidationError, match="model identities differ"):
        AlgorithmExecutionAttestation.create(
            endpoint_classification=AlgorithmEndpointClassification.OFFLINE_REPLAY,
            runtime_id=replay.runtime_id,
            runtime_version=replay.runtime_version,
            checkpoint=replay.checkpoint,
            sampling=replay.sampling,
            tokenizer=mismatched_tokenizer,
            hardware=replay.hardware,
            warmup_policy=AlgorithmWarmupPolicy.NOT_APPLICABLE,
        )
    with pytest.raises(ValidationError, match="execution attestation digest"):
        AlgorithmExecutionAttestation.model_validate(
            replay.model_dump(mode="python") | {"execution_digest": "f" * 64}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("decisions", 10),
        ("scheduled_invocations", 12),
        ("cycles", 2),
        ("outcomes", 2),
        ("deliveries", 4),
        ("model_calls", 2),
    ),
)
def test_counters_reject_each_impossible_total(field: str, value: int) -> None:
    values = _counters().model_dump(mode="python")

    with pytest.raises(ValidationError, match="counters are inconsistent"):
        AlgorithmArtifactCounters.model_validate(values | {field: value})


def test_component_constructors_reject_stale_self_digests(
    exported_algorithm: tuple[ValidatedAlgorithmArtifact, Path],
) -> None:
    artifact, destination = exported_algorithm
    private_attestations = algorithm_models.AlgorithmAttestationsComponent.model_validate_json(
        destination.joinpath("attestations.json").read_bytes()
    )
    sealed_models: tuple[tuple[type[BaseModel], BaseModel, str], ...] = (
        (algorithm_models.AlgorithmRunComponent, artifact.run, "run_component_digest"),
        (
            algorithm_models.AlgorithmTrajectoryComponent,
            artifact.trajectory,
            "trajectory_component_digest",
        ),
        (algorithm_models.AlgorithmCallsComponent, artifact.calls, "calls_component_digest"),
        (
            algorithm_models.AlgorithmDecisionsComponent,
            artifact.decisions,
            "decisions_component_digest",
        ),
        (
            algorithm_models.AlgorithmCycleAttestation,
            artifact.cycles.cycles[0],
            "cycle_attestation_digest",
        ),
        (
            algorithm_models.AlgorithmCyclesComponent,
            artifact.cycles,
            "cycles_component_digest",
        ),
        (
            algorithm_models.AlgorithmDeliveriesComponent,
            artifact.deliveries,
            "deliveries_component_digest",
        ),
        (
            algorithm_models.AlgorithmOutcomesComponent,
            artifact.outcomes,
            "outcomes_component_digest",
        ),
        (
            algorithm_models.AlgorithmMetricsComponent,
            artifact.metrics,
            "metrics_component_digest",
        ),
        (
            algorithm_models.AlgorithmBoundaryAttestation,
            artifact.attestations.boundaries[0],
            "boundary_attestation_digest",
        ),
        (
            algorithm_models.AlgorithmAttestationsComponent,
            private_attestations,
            "attestations_component_digest",
        ),
    )

    for model_type, model, digest_field in sealed_models:
        with pytest.raises(ValidationError, match="digest does not match"):
            model_type.model_validate(model.model_dump(mode="python") | {digest_field: "f" * 64})


def test_redacted_record_window_and_call_group_anchors_are_closed(
    exported_algorithm: tuple[ValidatedAlgorithmArtifact, Path],
) -> None:
    artifact, _ = exported_algorithm
    record = artifact.trajectory.records[0]
    with pytest.raises(ValidationError, match="trajectory record attestation"):
        algorithm_models.AlgorithmTrajectoryRecordAttestation.model_validate(
            record.model_dump(mode="python") | {"ordinal": 2}
        )

    window = artifact.trajectory.windows[0]
    with pytest.raises(ValidationError, match="window source attestations"):
        algorithm_models.AlgorithmWindowAttestation.model_validate(
            window.model_dump(mode="python")
            | {
                "message_count": 2,
                "source_attestation_digests": (OTHER_DIGEST, OTHER_DIGEST),
            }
        )
    with pytest.raises(ValidationError):
        algorithm_models.AlgorithmWindowAttestation.model_validate(
            window.model_dump(mode="python") | {"payload_canonical_utf8_bytes": 32_001}
        )
    changed_window = algorithm_models.AlgorithmWindowAttestation.model_validate(
        window.model_dump(mode="python") | {"task_digest": OTHER_DIGEST}
    )
    with pytest.raises(ValidationError, match="trajectory cardinality"):
        algorithm_models.AlgorithmTrajectoryComponent.model_validate(
            artifact.trajectory.model_dump(mode="python")
            | {"windows": (changed_window, *artifact.trajectory.windows[1:])}
        )

    group = artifact.calls.groups[0]
    assert group.grounding_call_index is not None
    assert group.grounding_state_digest is not None
    with pytest.raises(ValidationError, match="grounding fields are incomplete"):
        algorithm_models.AlgorithmCallGroup.model_validate(
            group.model_dump(mode="python") | {"grounding_call_index": None}
        )

    memory = artifact.metrics.final_memory
    with pytest.raises(ValidationError, match="memory attestation cardinality"):
        algorithm_models.AlgorithmFinalMemoryAttestation.model_validate(
            memory.model_dump(mode="python") | {"record_count": memory.record_count + 1}
        )


def test_component_collections_reject_reordered_or_duplicate_evidence(
    exported_algorithm: tuple[ValidatedAlgorithmArtifact, Path],
) -> None:
    artifact, _ = exported_algorithm

    with pytest.raises(ValidationError, match="run configuration"):
        algorithm_models.AlgorithmRunComponent.model_validate(
            artifact.run.model_dump(mode="python") | {"policy_version": "forged-policy/v9"}
        )

    with pytest.raises(ValidationError, match="canonical ordered set"):
        algorithm_models.AlgorithmCallsComponent.model_validate(
            artifact.calls.model_dump(mode="python")
            | {"ordered_request_digests": tuple(reversed(artifact.calls.ordered_request_digests))}
        )

    first_call, second_call = artifact.calls.calls[:2]
    repeated_call_digest = type(second_call).model_validate(
        second_call.model_dump(mode="python", exclude={"receipt_digest"})
        | {"call_digest": first_call.call_digest}
    )
    first_group = artifact.calls.groups[0]
    repeated_call_group = algorithm_models.AlgorithmCallGroup.model_validate(
        first_group.model_dump(mode="python")
        | {
            "call_receipt_digests": (
                first_call.receipt_digest,
                repeated_call_digest.receipt_digest,
            )
        }
    )
    with pytest.raises(ValidationError, match="canonical ordered set"):
        algorithm_models.AlgorithmCallsComponent.model_validate(
            artifact.calls.model_dump(mode="python")
            | {
                "groups": (repeated_call_group, *artifact.calls.groups[1:]),
                "calls": (
                    first_call,
                    repeated_call_digest,
                    *artifact.calls.calls[2:],
                ),
            }
        )

    group = artifact.calls.groups[0]
    wrong_cycle = algorithm_models.AlgorithmCallGroup.model_validate(
        group.model_dump(mode="python") | {"cycle_id": OTHER_DIGEST}
    )
    with pytest.raises(ValidationError, match="group ordering"):
        algorithm_models.AlgorithmCallsComponent.model_validate(
            artifact.calls.model_dump(mode="python")
            | {"groups": (wrong_cycle, *artifact.calls.groups[1:])}
        )

    assert group.grounding_call_index is not None
    wrong_index = 1 - group.grounding_call_index
    wrong_grounding = algorithm_models.AlgorithmCallGroup.model_validate(
        group.model_dump(mode="python") | {"grounding_call_index": wrong_index}
    )
    with pytest.raises(ValidationError, match="grounding binding"):
        algorithm_models.AlgorithmCallsComponent.model_validate(
            artifact.calls.model_dump(mode="python")
            | {"groups": (wrong_grounding, *artifact.calls.groups[1:])}
        )

    first_call = artifact.calls.calls[0]
    false_grounding = algorithm_models.AlgorithmCallGroup.model_validate(
        group.model_dump(mode="python")
        | {
            "call_receipt_digests": (first_call.receipt_digest,),
            "grounding_call_index": 0,
            "grounding_state_digest": OTHER_DIGEST,
        }
    )
    with pytest.raises(ValidationError, match="unexpected grounding binding"):
        algorithm_models.AlgorithmCallsComponent.model_validate(
            artifact.calls.model_dump(mode="python")
            | {
                "ordered_request_digests": (first_call.request_digest,),
                "groups": (false_grounding,),
                "calls": (first_call,),
            }
        )

    decision = artifact.decisions.decisions[0]
    with pytest.raises(ValidationError, match="decisions are not one ordered run"):
        algorithm_models.AlgorithmDecisionsComponent.model_validate(
            artifact.decisions.model_dump(mode="python")
            | {"decisions": (decision, decision, *artifact.decisions.decisions[1:])}
        )

    cycle = artifact.cycles.cycles[0]
    with pytest.raises(ValidationError, match="terminal evidence"):
        algorithm_models.AlgorithmCycleAttestation.model_validate(
            cycle.model_dump(mode="python") | {"batch_digest": OTHER_DIGEST}
        )
    with pytest.raises(ValidationError, match="cycles are not one ordered run"):
        algorithm_models.AlgorithmCyclesComponent.model_validate(
            artifact.cycles.model_dump(mode="python")
            | {"cycles": (cycle, cycle, *artifact.cycles.cycles[1:])}
        )

    delivery = artifact.deliveries.deliveries[0]
    with pytest.raises(ValidationError, match="deliveries are not unique"):
        algorithm_models.AlgorithmDeliveriesComponent.model_validate(
            artifact.deliveries.model_dump(mode="python") | {"deliveries": (delivery, delivery)}
        )

    outcome = artifact.outcomes.outcomes[0]
    with pytest.raises(ValidationError, match="outcomes are not one unique run"):
        algorithm_models.AlgorithmOutcomesComponent.model_validate(
            artifact.outcomes.model_dump(mode="python") | {"outcomes": (outcome, outcome)}
        )


def test_metrics_boundaries_and_ledger_reject_foreign_or_incomplete_evidence(
    exported_algorithm: tuple[ValidatedAlgorithmArtifact, Path],
) -> None:
    artifact, destination = exported_algorithm
    foreign_memory = algorithm_models.AlgorithmFinalMemoryAttestation.model_validate(
        artifact.metrics.final_memory.model_dump(mode="python") | {"run_id": OTHER_RUN_ID}
    )
    with pytest.raises(ValidationError, match="metrics memory run differs"):
        algorithm_models.AlgorithmMetricsComponent.model_validate(
            artifact.metrics.model_dump(mode="python") | {"final_memory": foreign_memory}
        )

    boundary = artifact.attestations.boundaries[0]
    with pytest.raises(ValidationError, match="boundary observation"):
        algorithm_models.AlgorithmBoundaryAttestation.model_validate(
            boundary.model_dump(mode="python")
            | {"invocation_ordinal": boundary.invocation_ordinal + 1}
        )

    attestations = algorithm_models.AlgorithmAttestationsComponent.model_validate_json(
        destination.joinpath("attestations.json").read_bytes()
    )
    projection = attestations.repository_projection_digests
    hmac_projection = type(projection).model_validate(
        {
            field_name: PayloadDigest(
                algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
                value=getattr(projection, field_name).value,
            )
            for field_name in type(projection).model_fields
        }
    )
    with pytest.raises(ValidationError, match="complete evidence chain"):
        algorithm_models.AlgorithmAttestationsComponent.model_validate(
            attestations.model_dump(mode="python")
            | {
                "semantic_projection_digests": hmac_projection,
                "repository_projection_digests": hmac_projection,
            }
        )
    with pytest.raises(ValidationError, match="complete evidence chain"):
        algorithm_models.AlgorithmAttestationsComponent.model_validate(
            attestations.model_dump(mode="python")
            | {"ledger_entry_count": attestations.ledger_entry_count + 1}
        )


def test_cycle_requires_a_committed_terminal_state(
    exported_algorithm: tuple[ValidatedAlgorithmArtifact, Path],
) -> None:
    artifact, _ = exported_algorithm
    cycle = artifact.cycles.cycles[0]

    with pytest.raises(ValidationError, match="terminal evidence"):
        algorithm_models.AlgorithmCycleAttestation.model_validate(
            cycle.model_dump(mode="python") | {"state": CycleState.FAILED}
        )


def test_trajectory_component_rejects_a_foreign_window_set_digest(
    exported_algorithm: tuple[ValidatedAlgorithmArtifact, Path],
) -> None:
    artifact, _ = exported_algorithm

    with pytest.raises(ValidationError, match="trajectory cardinality"):
        algorithm_models.AlgorithmTrajectoryComponent.model_validate(
            artifact.trajectory.model_dump(mode="python") | {"window_set_digest": OTHER_DIGEST}
        )


def test_v1_is_strictly_synthetic_and_exploratory() -> None:
    clean = _manifest(revision=_revision(clean=True))
    dirty = _manifest(revision=_revision(clean=False))

    assert clean.revision.confirmatory_eligible
    assert not clean.confirmatory_eligible
    assert not dirty.confirmatory_eligible
    with pytest.raises(ValueError):
        _manifest(classification=ArtifactClassification.USER_REDACTED)

    values = clean.model_dump(mode="python", exclude={"manifest_digest"})
    values["evidence_level"] = ArtifactEvidenceLevel.CONFIRMATORY
    values["manifest_digest"] = algorithm_artifact_manifest_digest(values)
    with pytest.raises(ValidationError):
        AlgorithmArtifactManifest.model_validate(values)


def test_manifest_rejects_wrong_cycle_mode_for_condition() -> None:
    manifest = _manifest()
    values = manifest.model_dump(mode="python", exclude={"manifest_digest"})
    values["cycle_mode"] = AlgorithmCycleMode.DISABLED

    with pytest.raises(ValidationError):
        AlgorithmArtifactManifest.model_validate(
            values | {"manifest_digest": manifest.manifest_digest}
        )


def test_cycle_mode_normalizes_closed_condition_ids() -> None:
    assert algorithm_cycle_mode_for_condition("no_memory") is AlgorithmCycleMode.DISABLED
    assert (
        algorithm_cycle_mode_for_condition("retrieval_always")
        is AlgorithmCycleMode.PHASE_ONE_RETRIEVAL
    )

    with pytest.raises(ValueError):
        algorithm_cycle_mode_for_condition("unknown")


def test_manifest_rejects_self_consistent_but_unregistered_condition_digest() -> None:
    manifest = _manifest()
    values = manifest.model_dump(mode="python", exclude={"manifest_digest"})
    values["condition_digest"] = "f" * 64
    values["configuration_digest"] = "e" * 64

    with pytest.raises(ValidationError, match="anchors"):
        AlgorithmArtifactManifest.model_validate(
            values | {"manifest_digest": manifest.manifest_digest}
        )


def test_disabled_condition_cannot_claim_algorithm_cycles() -> None:
    with pytest.raises(ValueError, match="cannot claim active cycles"):
        _manifest(
            condition_id=Stage2ConditionId.NO_MEMORY,
            cycle_mode=AlgorithmCycleMode.DISABLED,
        )


def test_disabled_condition_preserves_scheduled_boundaries_without_cycles() -> None:
    counters = AlgorithmArtifactCounters(
        events=11,
        scheduled_invocations=3,
        decisions=11,
        cycles=0,
        requests=0,
        model_calls=0,
        deliveries=0,
        outcomes=0,
        ledger_entries=20,
    )

    manifest = _manifest(
        condition_id=Stage2ConditionId.NO_MEMORY,
        cycle_mode=AlgorithmCycleMode.DISABLED,
        counters=counters,
    )

    assert manifest.counters.scheduled_invocations == 3
    assert manifest.counters.cycles == 0


def test_active_condition_call_cardinality_is_exact() -> None:
    counters = _counters().model_copy(update={"model_calls": 5})

    with pytest.raises(ValueError, match="cardinality does not match"):
        _manifest(counters=counters)


def test_manifest_component_counts_are_bound_to_counters() -> None:
    counters = _counters()
    components = list(_components(counters))
    original = components[0]
    components[0] = AlgorithmArtifactComponent.model_validate(
        original.model_dump(mode="python") | {"record_count": 2}
    )

    with pytest.raises(ValueError, match="record counts"):
        _manifest(counters=counters, components=tuple(components))


def test_overall_and_manifest_digests_are_independently_checked() -> None:
    manifest = _manifest()
    values = manifest.model_dump(mode="python")

    with pytest.raises(ValidationError, match="overall content digest"):
        AlgorithmArtifactManifest.model_validate(values | {"overall_content_digest": "f" * 64})
    with pytest.raises(ValidationError, match="manifest digest"):
        AlgorithmArtifactManifest.model_validate(values | {"manifest_digest": "f" * 64})


def test_component_digest_requires_exact_public_input_types() -> None:
    with pytest.raises(TypeError, match="exact name and bytes"):
        algorithm_component_content_digest(
            "run",  # type: ignore[arg-type]
            b"{}",
        )
    with pytest.raises(TypeError, match="exact name and bytes"):
        algorithm_component_content_digest(
            AlgorithmArtifactComponentName.RUN,
            bytearray(b"{}"),  # type: ignore[arg-type]
        )
