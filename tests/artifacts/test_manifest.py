from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from saliencegate.artifacts.manifest import (
    ArtifactClassification,
    ArtifactComponent,
    ArtifactComponentName,
    ArtifactCounters,
    ArtifactEvidenceLevel,
    ArtifactManifest,
    RevisionEvidence,
    RevisionSource,
    component_content_digest,
)

RUN_ID = UUID("00000000-0000-4000-8000-00000000a001")


def _revision(
    *,
    source: RevisionSource = RevisionSource.GIT,
    dirty: bool | None = False,
    distribution_digest: str | None = None,
) -> RevisionEvidence:
    return RevisionEvidence(
        source=source,
        package_version="0.1.0",
        commit="a" * 40 if source is RevisionSource.GIT else None,
        dirty_worktree=dirty,
        distribution_digest=distribution_digest,
    )


def _components() -> tuple[ArtifactComponent, ...]:
    payloads = {
        ArtifactComponentName.RUN: b'{"schema_version":"artifact-run/v1"}',
        ArtifactComponentName.DECISIONS: b'{"schema_version":"artifact-decisions/v1"}',
        ArtifactComponentName.BUDGETS: b'{"schema_version":"artifact-budgets/v1"}',
        ArtifactComponentName.DELIVERIES: b'{"schema_version":"artifact-deliveries/v1"}',
        ArtifactComponentName.OUTCOMES: b'{"schema_version":"artifact-outcomes/v1"}',
        ArtifactComponentName.ATTESTATIONS: b'{"schema_version":"artifact-attestations/v1"}',
    }
    record_counts = {
        ArtifactComponentName.RUN: 1,
        ArtifactComponentName.DECISIONS: 4,
        ArtifactComponentName.BUDGETS: 3,
        ArtifactComponentName.DELIVERIES: 1,
        ArtifactComponentName.OUTCOMES: 3,
        ArtifactComponentName.ATTESTATIONS: 1,
    }
    return tuple(
        ArtifactComponent(
            name=name,
            path=f"{name.value}.json",
            byte_count=len(payload),
            record_count=record_counts[name],
            content_digest=component_content_digest(payload),
        )
        for name, payload in payloads.items()
    )


def _manifest(
    revision: RevisionEvidence | None = None,
    *,
    evidence_level: ArtifactEvidenceLevel = ArtifactEvidenceLevel.CONFIRMATORY,
) -> ArtifactManifest:
    return ArtifactManifest.create(
        classification=ArtifactClassification.USER_REDACTED,
        evidence_level=evidence_level,
        run_id=RUN_ID,
        revision=revision or _revision(),
        engine_configuration_digest="b" * 64,
        trace_digest="c" * 64,
        model_id="replay-model/v1",
        replay_id="frozen-fixture/v1",
        prompt_template_digest="d" * 64,
        result_digest="e" * 64,
        components=_components(),
        counters=ArtifactCounters(
            events=4,
            decisions=4,
            invoked=3,
            cycles=3,
            model_calls=3,
            deliveries=1,
            delivered=0,
            outcomes=3,
        ),
    )


def test_manifest_is_self_attesting_and_component_order_is_canonical() -> None:
    manifest = _manifest()

    assert manifest.confirmatory
    assert tuple(component.name for component in manifest.components) == tuple(
        sorted(
            (item for item in ArtifactComponentName if item is not ArtifactComponentName.SYNTHETIC),
            key=lambda item: item.value,
        )
    )
    assert len(manifest.overall_content_digest) == 64
    assert len(manifest.manifest_digest) == 64
    assert ArtifactManifest.model_validate_json(manifest.model_dump_json()) == manifest

    altered = manifest.model_dump(mode="python")
    altered["overall_content_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="overall content digest"):
        ArtifactManifest.model_validate(altered)

    altered = manifest.model_dump(mode="python")
    altered["manifest_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="manifest digest"):
        ArtifactManifest.model_validate(altered)


def test_confirmatory_requires_clean_git_or_distribution_digest() -> None:
    dirty = _manifest(
        _revision(dirty=True),
        evidence_level=ArtifactEvidenceLevel.EXPLORATORY,
    )
    unattested = _manifest(
        _revision(
            source=RevisionSource.UNATTESTED,
            dirty=None,
        ),
        evidence_level=ArtifactEvidenceLevel.EXPLORATORY,
    )
    distribution = _manifest(
        _revision(
            source=RevisionSource.DISTRIBUTION,
            dirty=None,
            distribution_digest="f" * 64,
        )
    )

    assert not dirty.confirmatory
    assert not unattested.confirmatory
    assert distribution.confirmatory

    with pytest.raises(ValueError, match="confirmatory"):
        _manifest(_revision(dirty=True))


def test_manifest_rejects_unknown_major_duplicate_or_unsafe_components() -> None:
    manifest = _manifest()
    values = manifest.model_dump(mode="python")
    values["schema_version"] = "2.0"
    with pytest.raises(ValidationError, match="unsupported artifact schema major"):
        ArtifactManifest.model_validate(values)

    duplicate = list(manifest.components)
    duplicate[-1] = duplicate[0]
    values = manifest.model_dump(mode="python")
    values["components"] = tuple(duplicate)
    with pytest.raises(ValidationError, match="component set"):
        ArtifactManifest.model_validate(values)

    component = manifest.components[0].model_dump(mode="python")
    component["path"] = "../run.json"
    with pytest.raises(ValidationError, match="component path"):
        ArtifactComponent.model_validate(component)


def test_counters_require_one_decision_per_event_and_consistent_totals() -> None:
    with pytest.raises(ValidationError, match="decision count"):
        ArtifactCounters(
            events=4,
            decisions=3,
            invoked=3,
            cycles=3,
            model_calls=3,
            deliveries=1,
            delivered=0,
            outcomes=3,
        )

    with pytest.raises(ValidationError, match="artifact counters"):
        ArtifactCounters(
            events=4,
            decisions=4,
            invoked=2,
            cycles=3,
            model_calls=3,
            deliveries=1,
            delivered=2,
            outcomes=3,
        )
