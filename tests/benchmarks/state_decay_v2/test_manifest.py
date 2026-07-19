from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from typing import cast

import pytest
from pydantic import ValidationError

from saliencegate.benchmarks.state_decay_v2.config import (
    PUBLIC_GENERATION_SEED,
    SeedPurpose,
    derive_proposal_fixture_seed,
    derive_seed,
    proposal_fixture_seed_commitment,
)
from saliencegate.benchmarks.state_decay_v2.manifest import (
    EvidenceBoundary,
    ProposalFixtureCommitment,
    PublicBundleChildDescriptor,
    RoleManifestDescriptor,
    StateDecayV2PublicBundleManifest,
    StateDecayV2RoleManifest,
    StateDecayV2SuiteLock,
    SuiteValidationCommitment,
    ValidationAudit,
    ValidationStatus,
    build_public_bundle_manifest,
    build_role_manifest,
    build_suite_lock,
    expected_public_bundle_paths,
)
from saliencegate.benchmarks.state_decay_v2.protocol import validation_protocol_digests
from saliencegate.benchmarks.state_decay_v2.schema import (
    ArtifactRole,
    BenchmarkSplit,
)
from saliencegate.domain import canonical_json

_SPLIT_COUNTS = {
    BenchmarkSplit.TRAIN: 600,
    BenchmarkSplit.DEVELOPMENT: 300,
    BenchmarkSplit.LOCKED: 300,
    BenchmarkSplit.DIAGNOSTIC: 1_200,
}


def _digest(value: str) -> str:
    return sha256(value.encode("ascii")).hexdigest()


def _role_manifests() -> tuple[StateDecayV2RoleManifest, ...]:
    manifests: list[StateDecayV2RoleManifest] = []
    for split in BenchmarkSplit:
        for role in ArtifactRole:
            manifests.append(
                build_role_manifest(
                    split=split,
                    role=role,
                    canonical_byte_count=_SPLIT_COUNTS[split] * 200,
                    content_digest=_digest(f"content:{split.value}:{role.value}"),
                    ordered_scenario_set_digest=_digest(f"scenarios:{split.value}"),
                    generator_version="state-decay-v2-generator-v1",
                    generator_configuration_digest=_digest("generator-configuration"),
                    template_registry_digest=_digest(f"templates:{split.value}"),
                )
            )
    return tuple(manifests)


def _fixtures(
    manifests: tuple[StateDecayV2RoleManifest, ...],
    *,
    generation_commitment_digest: str | None = None,
) -> tuple[ProposalFixtureCommitment, ...]:
    policy_digests = {
        manifest.split: manifest.content_digest
        for manifest in manifests
        if manifest.role is ArtifactRole.POLICY_VIEW
    }
    generation_commitment_digest = generation_commitment_digest or _digest("generation-commitment")
    public_proposal_leaf = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.PROPOSAL)
    return tuple(
        ProposalFixtureCommitment(
            split=split,
            record_count=_SPLIT_COUNTS[split],
            generator_code_digest=_digest("proposal-generator"),
            public_profile_digest=_digest("public-response-profile"),
            seed_derivation_coordinates=("proposal_leaf", "split"),
            seed_commitment_digest=(
                proposal_fixture_seed_commitment(
                    derive_proposal_fixture_seed(public_proposal_leaf, split)
                )
                if split in (BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT)
                else _digest(f"custody-proposal-seed:{split.value}")
            ),
            generation_commitment_digest=generation_commitment_digest,
            input_policy_view_digest=policy_digests[split],
            output_fixture_digest=_digest(f"proposal-fixtures:{split.value}"),
        )
        for split in BenchmarkSplit
    )


def _validations() -> tuple[SuiteValidationCommitment, ...]:
    protocol_digests = _expected_protocol_digests()
    return tuple(
        SuiteValidationCommitment(
            audit=audit,
            status=ValidationStatus.PASSED,
            protocol_digest=protocol_digests[audit],
            report_digest=_digest(f"report:{audit.value}"),
        )
        for audit in ValidationAudit
    )


def _expected_protocol_digests() -> dict[ValidationAudit, str]:
    return validation_protocol_digests()


def _suite_lock() -> StateDecayV2SuiteLock:
    manifests = _role_manifests()
    return build_suite_lock(
        role_manifests=manifests,
        proposal_fixtures=_fixtures(manifests),
        validations=_validations(),
        expected_protocol_digests=_expected_protocol_digests(),
        generation_commitment_digest=_digest("generation-commitment"),
    )


def test_role_manifests_are_self_attesting_role_bound_and_non_circular() -> None:
    manifests = _role_manifests()
    assert len(manifests) == 12
    assert len({manifest.manifest_digest for manifest in manifests}) == 12
    assert "suite_lock_digest" not in StateDecayV2RoleManifest.model_fields
    assert "suite_lock_digest" not in RoleManifestDescriptor.model_fields

    for manifest in manifests:
        assert StateDecayV2RoleManifest.model_validate_json(canonical_json(manifest)) == manifest
        assert manifest.data_path == f"{manifest.split.value}-{manifest.role.value}.jsonl"
        assert manifest.record_count == _SPLIT_COUNTS[manifest.split]

        tampered = manifest.model_dump(mode="python")
        tampered["canonical_byte_count"] += 1
        with pytest.raises(ValidationError):
            StateDecayV2RoleManifest.model_validate(tampered)


def test_role_manifest_rejects_wrong_schema_path_geometry_and_digest() -> None:
    manifest = _role_manifests()[0]

    for field, value in (
        ("record_schema_version", "state-decay-oracle-vault-entry/v2"),
        ("data_path", "renamed.jsonl"),
        ("record_count", 599),
        ("manifest_digest", "0" * 64),
    ):
        payload = manifest.model_dump(mode="python")
        payload[field] = value
        with pytest.raises(ValidationError):
            StateDecayV2RoleManifest.model_validate(payload)

    descriptor = RoleManifestDescriptor.from_manifest(manifest).model_dump(mode="python")
    descriptor["role_manifest_digest"] = "0" * 64
    with pytest.raises(ValidationError):
        RoleManifestDescriptor.model_validate(descriptor)


def test_suite_lock_binds_every_role_fixture_and_sanitized_validation() -> None:
    lock = _suite_lock()

    assert len(lock.roles) == 12
    assert len(lock.proposal_fixtures) == 4
    assert len(lock.validations) == 5
    assert all(
        item.seed_hash_primitive == "length-prefixed-sha256/v1"
        and item.seed_derivation_coordinates == ("proposal_leaf", "split")
        and item.generation_commitment_digest == lock.generation_commitment_digest
        for item in lock.proposal_fixtures
    )
    assert tuple((item.split, item.role) for item in lock.roles) == tuple(
        (split, role) for split in BenchmarkSplit for role in ArtifactRole
    )
    assert StateDecayV2SuiteLock.model_validate_json(canonical_json(lock)) == lock
    for split in BenchmarkSplit:
        roles = tuple(item for item in lock.roles if item.split is split)
        assert len({item.record_count for item in roles}) == 1
        assert len({item.ordered_scenario_set_digest for item in roles}) == 1


def test_suite_lock_rejects_role_substitution_reordering_and_scenario_mismatch() -> None:
    lock = _suite_lock()

    substituted = lock.model_dump(mode="python")
    substituted_role = dict(substituted["roles"][0])
    substituted_role["role"] = ArtifactRole.ORACLE_VAULT
    substituted["roles"] = (substituted_role, *substituted["roles"][1:])
    with pytest.raises(ValidationError):
        StateDecayV2SuiteLock.model_validate(substituted)

    reordered = lock.model_dump(mode="python")
    reordered["roles"] = tuple(reversed(reordered["roles"]))
    with pytest.raises(ValidationError):
        StateDecayV2SuiteLock.model_validate(reordered)

    mismatched = lock.model_dump(mode="python")
    target = dict(mismatched["roles"][1])
    target["ordered_scenario_set_digest"] = _digest("mismatched-scenarios")
    mismatched["roles"] = (
        mismatched["roles"][0],
        target,
        *mismatched["roles"][2:],
    )
    with pytest.raises(ValidationError):
        StateDecayV2SuiteLock.model_validate(mismatched)

    manifests = list(_role_manifests())
    target_manifest = manifests[1]
    manifests[1] = build_role_manifest(
        split=target_manifest.split,
        role=target_manifest.role,
        canonical_byte_count=target_manifest.canonical_byte_count,
        content_digest=target_manifest.content_digest,
        ordered_scenario_set_digest=target_manifest.ordered_scenario_set_digest,
        generator_version=target_manifest.generator_version,
        generator_configuration_digest=target_manifest.generator_configuration_digest,
        template_registry_digest=_digest("substituted-template-registry"),
    )
    with pytest.raises(
        ValidationError,
        match="same scenario set and template registry",
    ):
        build_suite_lock(
            role_manifests=manifests,
            proposal_fixtures=_fixtures(tuple(manifests)),
            validations=_validations(),
            expected_protocol_digests=_expected_protocol_digests(),
            generation_commitment_digest=_digest("generation-commitment"),
        )


def test_suite_lock_rejects_fixture_validation_and_self_digest_tampering() -> None:
    lock = _suite_lock()

    wrong_public_seed = lock.proposal_fixtures[0].model_dump(mode="python")
    wrong_public_seed["seed_commitment_digest"] = _digest("wrong-public-proposal-seed")
    with pytest.raises(ValidationError, match="public proposal fixture seed commitment"):
        ProposalFixtureCommitment.model_validate(wrong_public_seed)

    mismatched_fixture = lock.model_dump(mode="python")
    fixture = dict(mismatched_fixture["proposal_fixtures"][0])
    fixture["input_policy_view_digest"] = _digest("wrong-policy-view")
    mismatched_fixture["proposal_fixtures"] = (
        fixture,
        *mismatched_fixture["proposal_fixtures"][1:],
    )
    with pytest.raises(ValidationError):
        StateDecayV2SuiteLock.model_validate(mismatched_fixture)

    mismatched_generation = lock.model_dump(mode="python")
    custody_fixture = dict(mismatched_generation["proposal_fixtures"][2])
    custody_fixture["generation_commitment_digest"] = _digest("other-generation-commitment")
    mismatched_generation["proposal_fixtures"] = (
        *mismatched_generation["proposal_fixtures"][:2],
        custody_fixture,
        mismatched_generation["proposal_fixtures"][3],
    )
    with pytest.raises(ValidationError, match="suite generation commitment"):
        StateDecayV2SuiteLock.model_validate(mismatched_generation)

    failed_validation = lock.model_dump(mode="python")
    validation = dict(failed_validation["validations"][0])
    validation["status"] = ValidationStatus.FAILED
    failed_validation["validations"] = (
        validation,
        *failed_validation["validations"][1:],
    )
    with pytest.raises(ValidationError):
        StateDecayV2SuiteLock.model_validate(failed_validation)

    wrong_digest = lock.model_dump(mode="python")
    wrong_digest["suite_lock_digest"] = "0" * 64
    with pytest.raises(ValidationError):
        StateDecayV2SuiteLock.model_validate(wrong_digest)

    manifests = _role_manifests()
    fixtures = list(_fixtures(manifests))
    fixtures[1] = ProposalFixtureCommitment.model_validate(
        {
            **fixtures[1].model_dump(mode="python"),
            "public_profile_digest": _digest("substituted-profile"),
        }
    )
    with pytest.raises(ValidationError):
        build_suite_lock(
            role_manifests=manifests,
            proposal_fixtures=fixtures,
            validations=_validations(),
            expected_protocol_digests=_expected_protocol_digests(),
            generation_commitment_digest=_digest("generation-commitment"),
        )


def test_suite_lock_rejects_substituted_validation_protocol_digest() -> None:
    manifests = _role_manifests()
    validations = list(_validations())
    target = validations[0]
    validations[0] = SuiteValidationCommitment.model_validate(
        {
            **target.model_dump(mode="python"),
            "protocol_digest": _expected_protocol_digests()[ValidationAudit.LINEAGE_REVIEW],
        }
    )

    with pytest.raises(ValueError, match="validation protocol digest does not match"):
        build_suite_lock(
            role_manifests=manifests,
            proposal_fixtures=_fixtures(manifests),
            validations=validations,
            expected_protocol_digests=_expected_protocol_digests(),
            generation_commitment_digest=_digest("generation-commitment"),
        )


def test_suite_lock_rejects_generation_commitment_substituting_a_protocol() -> None:
    manifests = _role_manifests()
    substituted = _expected_protocol_digests()[ValidationAudit.GEOMETRY]
    with pytest.raises(ValueError, match="generation commitment cannot substitute"):
        build_suite_lock(
            role_manifests=manifests,
            proposal_fixtures=_fixtures(
                manifests,
                generation_commitment_digest=substituted,
            ),
            validations=_validations(),
            expected_protocol_digests=_expected_protocol_digests(),
            generation_commitment_digest=substituted,
        )


def test_suite_lock_requires_complete_expected_protocol_digests() -> None:
    manifests = _role_manifests()
    missing = _expected_protocol_digests()
    missing.pop(ValidationAudit.GEOMETRY)
    extra_values: dict[object, object] = {
        audit: digest for audit, digest in _expected_protocol_digests().items()
    }
    extra_values["unexpected"] = _digest("unexpected-protocol")
    extra = cast("Mapping[ValidationAudit, str]", extra_values)

    for expected in (missing, extra):
        with pytest.raises(ValueError, match="expected validation protocols are invalid"):
            build_suite_lock(
                role_manifests=manifests,
                proposal_fixtures=_fixtures(manifests),
                validations=_validations(),
                expected_protocol_digests=expected,
                generation_commitment_digest=_digest("generation-commitment"),
            )


@pytest.mark.parametrize(
    "expected",
    (
        cast(
            "Mapping[ValidationAudit, str]",
            {
                **{
                    audit: digest
                    for audit, digest in _expected_protocol_digests().items()
                    if audit is not ValidationAudit.GEOMETRY
                },
                ValidationAudit.GEOMETRY.value: _digest("protocol:geometry"),
            },
        ),
        {
            **_expected_protocol_digests(),
            ValidationAudit.GEOMETRY: "A" * 64,
        },
        cast(
            "Mapping[ValidationAudit, str]",
            {
                **_expected_protocol_digests(),
                ValidationAudit.GEOMETRY: 1,
            },
        ),
    ),
    ids=("string-key", "non-lowercase-sha", "non-string-value"),
)
def test_suite_lock_rejects_ill_typed_expected_protocol_digests(
    expected: Mapping[ValidationAudit, str],
) -> None:
    manifests = _role_manifests()

    with pytest.raises(ValueError, match="expected validation protocols are invalid"):
        build_suite_lock(
            role_manifests=manifests,
            proposal_fixtures=_fixtures(manifests),
            validations=_validations(),
            expected_protocol_digests=expected,
            generation_commitment_digest=_digest("generation-commitment"),
        )


def _public_role_manifests(
    manifests: tuple[StateDecayV2RoleManifest, ...],
) -> tuple[StateDecayV2RoleManifest, ...]:
    return tuple(
        manifest
        for manifest in manifests
        if manifest.split in {BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT}
    )


def test_public_bundle_manifest_closes_the_exact_fourteen_file_tree() -> None:
    manifests = _role_manifests()
    lock = _suite_lock()
    root = build_public_bundle_manifest(
        suite_lock=lock,
        public_role_manifests=_public_role_manifests(manifests),
    )

    assert root.evidence_boundary is EvidenceBoundary.PUBLIC_SYNTHETIC_DEVELOPMENT
    assert root.confirmatory is False
    assert root.locked_evidence == "not_measured"
    assert root.external_claims_supported is False
    assert len(root.children) == 12
    assert expected_public_bundle_paths() == {
        "manifest.json",
        "suite-lock.json",
        *(child.path for child in root.children),
    }
    assert len(expected_public_bundle_paths()) == 14
    assert StateDecayV2PublicBundleManifest.model_validate_json(canonical_json(root)) == root


def test_public_bundle_builder_rejects_role_manifests_not_bound_to_the_lock() -> None:
    manifests = _role_manifests()
    public_manifests = list(_public_role_manifests(manifests))
    target = public_manifests[0]
    public_manifests[0] = build_role_manifest(
        split=target.split,
        role=target.role,
        canonical_byte_count=target.canonical_byte_count,
        content_digest=_digest("substituted-public-role"),
        ordered_scenario_set_digest=target.ordered_scenario_set_digest,
        generator_version=target.generator_version,
        generator_configuration_digest=target.generator_configuration_digest,
        template_registry_digest=target.template_registry_digest,
    )

    with pytest.raises(ValueError, match="does not match the suite lock"):
        build_public_bundle_manifest(
            suite_lock=_suite_lock(),
            public_role_manifests=public_manifests,
        )


def test_public_bundle_rejects_missing_extra_reordered_and_tampered_children() -> None:
    manifests = _role_manifests()
    root = build_public_bundle_manifest(
        suite_lock=_suite_lock(),
        public_role_manifests=_public_role_manifests(manifests),
    )

    for children in (
        root.children[:-1],
        (
            *root.children,
            PublicBundleChildDescriptor(
                path="unexpected.json",
                canonical_byte_count=1,
                content_digest=_digest("unexpected"),
            ),
        ),
        tuple(reversed(root.children)),
    ):
        payload = root.model_dump(mode="python")
        payload["children"] = children
        with pytest.raises(ValidationError):
            StateDecayV2PublicBundleManifest.model_validate(payload)

    payload = root.model_dump(mode="python")
    payload["manifest_digest"] = "0" * 64
    with pytest.raises(ValidationError):
        StateDecayV2PublicBundleManifest.model_validate(payload)
