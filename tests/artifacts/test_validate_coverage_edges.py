"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.artifacts.test_validate import _export

import saliencegate.artifacts.validate as validate_module
from saliencegate.artifacts.manifest import (
    ArtifactComponentName,
    ArtifactSyntheticComponent,
)
from saliencegate.artifacts.validate import (
    ArtifactValidationCode,
    ArtifactValidationError,
    ValidatedArtifact,
    load_validated_artifact,
)
from saliencegate.domain import CycleState, DeliveryState, DeliveryTarget, canonical_json
from saliencegate.runtime.engine import ReplayRunResult


def _parsed(loaded: ValidatedArtifact) -> dict[ArtifactComponentName, object]:
    return {
        ArtifactComponentName.RUN: loaded.run,
        ArtifactComponentName.DECISIONS: loaded.decisions,
        ArtifactComponentName.BUDGETS: loaded.budgets,
        ArtifactComponentName.DELIVERIES: loaded.deliveries,
        ArtifactComponentName.OUTCOMES: loaded.outcomes,
        ArtifactComponentName.ATTESTATIONS: loaded.attestations,
    }


def _assert_invariant_error(
    manifest: object,
    parsed: dict[ArtifactComponentName, object],
    code: ArtifactValidationCode = ArtifactValidationCode.CROSS_COMPONENT_INVARIANT,
) -> None:
    with pytest.raises(ArtifactValidationError) as error:
        validate_module._validate_cross_component_invariants(  # type: ignore[arg-type]
            manifest,
            parsed,
        )
    assert error.value.code is code


async def test_component_parser_rejects_an_unknown_component_major_version(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root, _manifest = _export(tmp_path, replay_result)
    payload = validate_module.json.loads((root / "run.json").read_bytes())
    payload["schema_version"] = "artifact-run/v2"
    with pytest.raises(ArtifactValidationError) as error:
        validate_module._parse_component(ArtifactComponentName.RUN, canonical_json(payload))
    assert error.value.code is ArtifactValidationCode.UNSUPPORTED_VERSION


async def test_cross_component_same_run_and_trace_mode_defenses(
    tmp_path: Path,
    replay_result: ReplayRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest = _export(tmp_path, replay_result)
    loaded = load_validated_artifact(root / "manifest.json")

    monkeypatch.setattr(validate_module, "_same_run", lambda *_args: False)
    _assert_invariant_error(manifest, _parsed(loaded))
    monkeypatch.undo()

    parsed = _parsed(loaded)
    parsed[ArtifactComponentName.RUN] = loaded.run.model_copy(
        update={"trace_attestation_mode": "normalized_execution"}
    )
    _assert_invariant_error(manifest, parsed)

    monkeypatch.setattr(validate_module, "canonical_digest", lambda _value: "f" * 64)
    _assert_invariant_error(manifest, _parsed(loaded))


async def test_cycle_intervention_branches_reject_each_incoherent_state(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root, manifest = _export(tmp_path, replay_result)
    loaded = load_validated_artifact(root / "manifest.json")
    cycles = list(loaded.budgets.cycles)
    index = next(index for index, cycle in enumerate(cycles) if cycle.intervention is not None)
    selected = cycles[index]

    cycles[index] = selected.model_copy(update={"state": CycleState.FAILED})
    parsed = _parsed(loaded)
    parsed[ArtifactComponentName.BUDGETS] = loaded.budgets.model_copy(
        update={"cycles": tuple(cycles)}
    )
    _assert_invariant_error(manifest, parsed)

    cycles[index] = selected.model_copy(update={"intervention": None})
    parsed = _parsed(loaded)
    parsed[ArtifactComponentName.BUDGETS] = loaded.budgets.model_copy(
        update={"cycles": tuple(cycles)}
    )
    parsed[ArtifactComponentName.DELIVERIES] = loaded.deliveries.model_copy(
        update={"deliveries": ()}
    )
    _assert_invariant_error(manifest, parsed)


async def test_delivery_mapping_and_grounding_defenses_cover_duplicate_and_delivered_edges(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root, manifest = _export(tmp_path, replay_result)
    loaded = load_validated_artifact(root / "manifest.json")
    delivery = loaded.deliveries.deliveries[-1]

    parsed = _parsed(loaded)
    parsed[ArtifactComponentName.DELIVERIES] = loaded.deliveries.model_copy(
        update={"deliveries": (*loaded.deliveries.deliveries, delivery)}
    )
    _assert_invariant_error(manifest, parsed)

    forged = delivery.model_copy(
        update={
            "rendered_text_digest": "f" * 64,
            "state": DeliveryState.DELIVERED,
        }
    )
    parsed = _parsed(loaded)
    parsed[ArtifactComponentName.DELIVERIES] = loaded.deliveries.model_copy(
        update={"deliveries": (*loaded.deliveries.deliveries[:-1], forged)}
    )
    _assert_invariant_error(manifest, parsed, ArtifactValidationCode.UNGROUNDED_DELIVERY)


async def test_routing_outcome_and_synthetic_projection_defenses(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root, manifest = _export(tmp_path, replay_result)
    loaded = load_validated_artifact(root / "manifest.json")
    cycle = loaded.budgets.cycles[-1]
    binding_index = next(
        index
        for index, binding in enumerate(loaded.attestations.routing_bindings)
        if binding.ordinal == cycle.last_event_sequence
    )
    bindings = list(loaded.attestations.routing_bindings)
    replacement_target = (
        DeliveryTarget.PRE_ACTION_REPLAN
        if cycle.requested_delivery_target is DeliveryTarget.NEXT_MODEL_CALL
        else DeliveryTarget.NEXT_MODEL_CALL
    )
    bindings[binding_index] = bindings[binding_index].model_copy(
        update={"target": replacement_target}
    )
    parsed = _parsed(loaded)
    parsed[ArtifactComponentName.ATTESTATIONS] = loaded.attestations.model_copy(
        update={"routing_bindings": tuple(bindings)}
    )
    _assert_invariant_error(manifest, parsed)

    parsed = _parsed(loaded)
    parsed[ArtifactComponentName.OUTCOMES] = loaded.outcomes.model_copy(update={"outcomes": ()})
    _assert_invariant_error(manifest, parsed)

    synthetic = ArtifactSyntheticComponent.model_construct(
        run_id=manifest.run_id,
        trace_digest="f" * 64,
        prompt_template_digest=manifest.prompt_template_digest,
        model_request_digests=loaded.attestations.model_request_digests,
        model_call_digests=tuple(
            digest for item in loaded.budgets.cycles for digest in item.model_call_digests
        ),
    )
    parsed = _parsed(loaded)
    parsed[ArtifactComponentName.SYNTHETIC] = synthetic
    _assert_invariant_error(manifest, parsed)


def test_tree_callback_and_public_wrapper_map_defensive_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invoke_file_before_manifest(
        _path: object,
        *,
        parse_file: object,
        **_kwargs: object,
    ) -> object:
        parse_file(ArtifactComponentName.RUN, b"{}")  # type: ignore[operator]
        raise AssertionError

    monkeypatch.setattr(
        validate_module.artifact_tree,
        "read_closed_tree",
        invoke_file_before_manifest,
    )
    with pytest.raises(ArtifactValidationError) as missing_descriptor:
        validate_module._validate_artifact(tmp_path / "manifest.json")
    assert missing_descriptor.value.code is ArtifactValidationCode.INVALID_MANIFEST

    monkeypatch.setattr(
        validate_module,
        "_validate_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError()),
    )
    with pytest.raises(ArtifactValidationError) as sanitized:
        load_validated_artifact(tmp_path / "manifest.json")
    assert sanitized.value.code is ArtifactValidationCode.INVALID_MANIFEST
