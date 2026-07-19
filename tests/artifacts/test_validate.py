from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from saliencegate.artifacts.export import SyntheticArtifactContent, export_replay_artifact
from saliencegate.artifacts.manifest import (
    ArtifactClassification,
    ArtifactComponent,
    ArtifactComponentName,
    ArtifactCounters,
    ArtifactManifest,
    RevisionEvidence,
    RevisionSource,
    component_content_digest,
)
from saliencegate.artifacts.validate import (
    ArtifactValidationCode,
    ArtifactValidationError,
    load_validated_artifact,
    validate_artifact,
)
from saliencegate.domain import canonical_digest, canonical_json
from saliencegate.runtime.engine import ReplayRunResult


def _revision() -> RevisionEvidence:
    return RevisionEvidence(
        source=RevisionSource.GIT,
        package_version="0.1.0",
        commit="2" * 40,
        dirty_worktree=False,
        distribution_digest=None,
    )


def _export(tmp_path: Path, result: ReplayRunResult) -> tuple[Path, ArtifactManifest]:
    root = tmp_path / "artifact"
    manifest = export_replay_artifact(
        result,
        root,
        classification=ArtifactClassification.USER_REDACTED,
        revision=_revision(),
    )
    return root, manifest


def _reseal(
    manifest: ArtifactManifest,
    *,
    components: tuple[ArtifactComponent, ...] | None = None,
    counters: ArtifactCounters | None = None,
) -> ArtifactManifest:
    return ArtifactManifest.create(
        classification=manifest.classification,
        evidence_level=manifest.evidence_level,
        run_id=manifest.run_id,
        revision=manifest.revision,
        engine_configuration_digest=manifest.engine_configuration_digest,
        trace_digest=manifest.trace_digest,
        model_id=manifest.model_id,
        replay_id=manifest.replay_id,
        prompt_template_digest=manifest.prompt_template_digest,
        result_digest=manifest.result_digest,
        components=components or manifest.components,
        counters=counters or manifest.counters,
    )


def _rewrite_component(
    root: Path,
    manifest: ArtifactManifest,
    name: ArtifactComponentName,
    mutate: Callable[[dict[str, object]], None],
) -> ArtifactManifest:
    descriptor = next(component for component in manifest.components if component.name is name)
    path = root / descriptor.path
    payload = json.loads(path.read_bytes())
    assert isinstance(payload, dict)
    mutate(payload)
    encoded = canonical_json(payload)
    path.write_bytes(encoded)
    replacement = ArtifactComponent(
        name=descriptor.name,
        path=descriptor.path,
        byte_count=len(encoded),
        record_count=descriptor.record_count,
        content_digest=component_content_digest(encoded),
    )
    components = tuple(
        replacement if component.name is name else component for component in manifest.components
    )
    updated = _reseal(manifest, components=components)
    (root / "manifest.json").write_bytes(canonical_json(updated))
    return updated


def _assert_code(
    error: pytest.ExceptionInfo[ArtifactValidationError],
    code: ArtifactValidationCode,
) -> None:
    assert error.value.code is code
    assert "fixture-secret" not in str(error.value)
    assert "fixture-secret" not in repr(error.value)


async def test_rejects_missing_altered_extra_and_symlink_components(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root, manifest = _export(tmp_path, replay_result)
    decisions = root / "decisions.json"
    original = decisions.read_bytes()

    decisions.unlink()
    with pytest.raises(ArtifactValidationError) as missing:
        validate_artifact(root / "manifest.json")
    _assert_code(missing, ArtifactValidationCode.MISSING_COMPONENT)

    decisions.write_bytes(original.replace(b"scripted/v1", b"scripted/v2", 1))
    decisions.chmod(0o600)
    with pytest.raises(ArtifactValidationError) as altered:
        validate_artifact(root / "manifest.json")
    _assert_code(altered, ArtifactValidationCode.CONTENT_MISMATCH)

    decisions.write_bytes(original)
    (root / "unexpected.json").write_bytes(b"{}")
    with pytest.raises(ArtifactValidationError) as extra:
        validate_artifact(root / "manifest.json")
    _assert_code(extra, ArtifactValidationCode.UNSAFE_COMPONENT)
    (root / "unexpected.json").unlink()

    outside = tmp_path / "outside.json"
    outside.write_bytes(original)
    decisions.unlink()
    decisions.symlink_to(outside)
    with pytest.raises(ArtifactValidationError) as symlink:
        validate_artifact(
            root / "manifest.json",
            expected_manifest_digest=manifest.manifest_digest,
        )
    _assert_code(symlink, ArtifactValidationCode.UNSAFE_COMPONENT)


async def test_rejects_manifest_symlink_and_expected_digest_mismatch(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root, _ = _export(tmp_path, replay_result)

    with pytest.raises(ArtifactValidationError) as mismatch:
        validate_artifact(root / "manifest.json", expected_manifest_digest="0" * 64)
    _assert_code(mismatch, ArtifactValidationCode.EXPECTED_DIGEST_MISMATCH)

    real = root / "real-manifest.json"
    (root / "manifest.json").rename(real)
    (root / "manifest.json").symlink_to(real)
    with pytest.raises(ArtifactValidationError) as symlink:
        validate_artifact(root / "manifest.json")
    _assert_code(symlink, ArtifactValidationCode.UNSAFE_COMPONENT)


def test_non_directory_artifact_root_preserves_missing_component_code(
    tmp_path: Path,
) -> None:
    root = tmp_path / "not-a-directory"
    root.write_bytes(b"fixture-secret")

    with pytest.raises(ArtifactValidationError) as error:
        validate_artifact(root / "manifest.json")

    _assert_code(error, ArtifactValidationCode.MISSING_COMPONENT)


async def test_rejects_unknown_major_and_hostile_component_paths_before_digest_checks(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root, manifest = _export(tmp_path, replay_result)
    values = manifest.model_dump(mode="json")
    values["schema_version"] = "2.0"
    (root / "manifest.json").write_bytes(canonical_json(values))

    with pytest.raises(ArtifactValidationError) as version:
        validate_artifact(root / "manifest.json")
    _assert_code(version, ArtifactValidationCode.UNSUPPORTED_VERSION)

    values = manifest.model_dump(mode="json")
    components = values["components"]
    assert isinstance(components, list)
    first = components[0]
    assert isinstance(first, dict)
    first["path"] = "../fixture-secret.json"
    (root / "manifest.json").write_bytes(canonical_json(values))

    with pytest.raises(ArtifactValidationError) as path:
        validate_artifact(root / "manifest.json")
    _assert_code(path, ArtifactValidationCode.UNSAFE_PATH)


async def test_rejects_resealed_inconsistent_counters(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root, manifest = _export(tmp_path, replay_result)
    counters = manifest.counters.model_copy(update={"model_calls": 2})
    forged = _reseal(manifest, counters=counters)
    (root / "manifest.json").write_bytes(canonical_json(forged))

    with pytest.raises(ArtifactValidationError) as error:
        validate_artifact(root / "manifest.json")
    _assert_code(error, ArtifactValidationCode.INCONSISTENT_COUNTERS)


async def test_rejects_resealed_delivered_reminder_without_intervention_attestation(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root, manifest = _export(tmp_path, replay_result)

    def remove_intervention(payload: dict[str, object]) -> None:
        cycles = payload["cycles"]
        assert isinstance(cycles, list)
        delivered_cycle = cycles[-1]
        assert isinstance(delivered_cycle, dict)
        delivered_cycle["intervention"] = None

    manifest = _rewrite_component(
        root,
        manifest,
        ArtifactComponentName.BUDGETS,
        remove_intervention,
    )

    def claim_delivery(payload: dict[str, object]) -> None:
        deliveries = payload["deliveries"]
        assert isinstance(deliveries, list)
        delivery = deliveries[-1]
        assert isinstance(delivery, dict)
        delivery["state"] = "delivered"
        delivery["outcome"] = "delivered"
        delivery["reason_code"] = "delivery_succeeded"
        delivery["receipt_digest"] = "9" * 64

    manifest = _rewrite_component(
        root,
        manifest,
        ArtifactComponentName.DELIVERIES,
        claim_delivery,
    )
    forged_counters = manifest.counters.model_copy(update={"delivered": 1})
    forged = _reseal(manifest, counters=forged_counters)
    (root / "manifest.json").write_bytes(canonical_json(forged))

    with pytest.raises(ArtifactValidationError) as error:
        validate_artifact(root / "manifest.json")
    _assert_code(error, ArtifactValidationCode.UNGROUNDED_DELIVERY)


@pytest.mark.skipif(os.name != "posix", reason="POSIX hard-link semantics")
async def test_rejects_hardlinked_component(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root, _ = _export(tmp_path, replay_result)
    decisions = root / "decisions.json"
    linked = tmp_path / "linked.json"
    os.link(decisions, linked)

    with pytest.raises(ArtifactValidationError) as error:
        validate_artifact(root / "manifest.json")
    _assert_code(error, ArtifactValidationCode.UNSAFE_COMPONENT)


async def test_rejects_resealed_routing_binding_divergence(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root, manifest = _export(tmp_path, replay_result)
    forged_routing_digest = ""

    def forge_binding(payload: dict[str, object]) -> None:
        nonlocal forged_routing_digest
        bindings = payload["routing_bindings"]
        assert isinstance(bindings, list)
        binding = bindings[-1]
        assert isinstance(binding, dict)
        binding["adapter_id"] = "forged-adapter/1"
        forged_routing_digest = canonical_digest(tuple(bindings))
        payload["routing_digest"] = forged_routing_digest

    manifest = _rewrite_component(
        root,
        manifest,
        ArtifactComponentName.ATTESTATIONS,
        forge_binding,
    )

    def forge_run(payload: dict[str, object]) -> None:
        payload["routing_digest"] = forged_routing_digest

    _rewrite_component(root, manifest, ArtifactComponentName.RUN, forge_run)

    with pytest.raises(ArtifactValidationError) as error:
        validate_artifact(root / "manifest.json")
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


async def test_rejects_resealed_incoherent_terminal_delivery(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root, manifest = _export(tmp_path, replay_result)

    def forge_delivery(payload: dict[str, object]) -> None:
        deliveries = payload["deliveries"]
        assert isinstance(deliveries, list)
        delivery = deliveries[-1]
        assert isinstance(delivery, dict)
        delivery.update(
            state="failed",
            outcome="delivered",
            reason_code="delivery_succeeded",
            receipt_digest="8" * 64,
            attempt_count=0,
            claim_id=None,
            attempt_id=None,
        )

    _rewrite_component(
        root,
        manifest,
        ArtifactComponentName.DELIVERIES,
        forge_delivery,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_artifact(root / "manifest.json")
    _assert_code(error, ArtifactValidationCode.INVALID_COMPONENT)


async def test_rejects_resealed_causal_claim_in_policy_replay_outcome(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root, manifest = _export(tmp_path, replay_result)

    def forge_outcome(payload: dict[str, object]) -> None:
        outcomes = payload["outcomes"]
        assert isinstance(outcomes, list)
        outcome = outcomes[0]
        assert isinstance(outcome, dict)
        outcome.update(
            evidence_mode="deterministic_oracle",
            utility="helpful",
            task_passed=True,
            task_reward=1.0,
        )

    _rewrite_component(
        root,
        manifest,
        ArtifactComponentName.OUTCOMES,
        forge_outcome,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_artifact(root / "manifest.json")
    _assert_code(error, ArtifactValidationCode.INVALID_COMPONENT)


async def test_load_validated_artifact_returns_same_pass_safe_components_only(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root = tmp_path / "artifact"
    secret = "fixture-secret-raw-synthetic-content"
    manifest = export_replay_artifact(
        replay_result,
        root,
        classification=ArtifactClassification.SYNTHETIC_RAW,
        revision=_revision(),
        synthetic_content=SyntheticArtifactContent(
            prompt={"instruction": secret},
            responses=({"answer": secret},),
        ),
    )

    loaded = load_validated_artifact(
        root / "manifest.json",
        expected_manifest_digest=manifest.manifest_digest,
    )

    assert loaded.report.valid
    assert loaded.manifest == manifest
    assert loaded.run.run_id == manifest.run_id
    assert loaded.decisions.run_id == manifest.run_id
    assert loaded.budgets.run_id == manifest.run_id
    assert loaded.deliveries.run_id == manifest.run_id
    assert loaded.outcomes.run_id == manifest.run_id
    assert loaded.attestations.run_id == manifest.run_id
    assert not hasattr(loaded, "synthetic")
    encoded = loaded.model_dump_json()
    for forbidden in (
        secret,
        "verified event 1",
        "Run the verified test suite before delivery.",
        "engine-request-4",
        str(root),
    ):
        assert forbidden not in encoded


async def test_load_validated_artifact_does_not_reopen_after_validation(
    tmp_path: Path,
    replay_result: ReplayRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _export(tmp_path, replay_result)
    original_open = os.open
    opens: list[str] = []

    def track_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        opens.append(os.fspath(path))
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr("saliencegate.artifacts.validate.os.open", track_open)

    loaded = load_validated_artifact(root / "manifest.json")

    assert loaded.report.valid
    assert opens.count("manifest.json") == 1
    assert len(opens) == loaded.report.component_count + 2


async def test_rejects_resealed_normalized_trace_digest_input(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root, manifest = _export(tmp_path, replay_result)

    def forge_trace(payload: dict[str, object]) -> None:
        digests = payload["normalized_draft_digests"]
        assert isinstance(digests, list)
        digests[0] = "f" * 64

    _rewrite_component(
        root,
        manifest,
        ArtifactComponentName.ATTESTATIONS,
        forge_trace,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_artifact(root / "manifest.json")
    _assert_code(error, ArtifactValidationCode.INVALID_COMPONENT)


async def test_rejects_outcome_identity_or_time_not_bound_to_intervention(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root, manifest = _export(tmp_path, replay_result)

    def forge_outcome(payload: dict[str, object]) -> None:
        outcomes = payload["outcomes"]
        assert isinstance(outcomes, list)
        outcome = outcomes[0]
        assert isinstance(outcome, dict)
        outcome["outcome_id"] = "00000000-0000-4000-8000-00000000f001"
        outcome["created_at"] = "2030-01-01T00:00:00Z"

    _rewrite_component(
        root,
        manifest,
        ArtifactComponentName.OUTCOMES,
        forge_outcome,
    )

    with pytest.raises(ArtifactValidationError) as error:
        validate_artifact(root / "manifest.json")
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


async def test_delivery_event_sequence_must_equal_its_cycle_terminal_sequence(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root, manifest = _export(tmp_path, replay_result)

    def move_delivery(payload: dict[str, object]) -> None:
        deliveries = payload["deliveries"]
        assert isinstance(deliveries, list)
        delivery = deliveries[-1]
        assert isinstance(delivery, dict)
        delivery["event_sequence"] = 1

    manifest = _rewrite_component(
        root,
        manifest,
        ArtifactComponentName.DELIVERIES,
        move_delivery,
    )
    forged_routing_digest = ""

    def copy_route(payload: dict[str, object]) -> None:
        nonlocal forged_routing_digest
        bindings = payload["routing_bindings"]
        assert isinstance(bindings, list)
        first = bindings[0]
        last = bindings[-1]
        assert isinstance(first, dict)
        assert isinstance(last, dict)
        for key in (
            "target",
            "target_request_id_digest",
            "adapter_id",
            "adapter_capabilities_digest",
        ):
            first[key] = last[key]
        forged_routing_digest = canonical_digest(tuple(bindings))
        payload["routing_digest"] = forged_routing_digest

    manifest = _rewrite_component(
        root,
        manifest,
        ArtifactComponentName.ATTESTATIONS,
        copy_route,
    )

    def update_run(payload: dict[str, object]) -> None:
        payload["routing_digest"] = forged_routing_digest

    _rewrite_component(root, manifest, ArtifactComponentName.RUN, update_run)

    with pytest.raises(ArtifactValidationError) as error:
        validate_artifact(root / "manifest.json")
    _assert_code(error, ArtifactValidationCode.CROSS_COMPONENT_INVARIANT)


async def test_pathological_major_versions_keep_the_stable_validation_error_contract(
    tmp_path: Path,
    replay_result: ReplayRunResult,
) -> None:
    root, manifest = _export(tmp_path, replay_result)
    values = manifest.model_dump(mode="json")
    values["schema_version"] = f"{'9' * 5_000}.0"
    (root / "manifest.json").write_bytes(canonical_json(values))

    with pytest.raises(ArtifactValidationError) as manifest_error:
        validate_artifact(root / "manifest.json")
    _assert_code(manifest_error, ArtifactValidationCode.UNSUPPORTED_VERSION)
