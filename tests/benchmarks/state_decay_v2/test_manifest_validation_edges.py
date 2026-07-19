from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from typing import NoReturn, cast

import pytest
from pydantic import ValidationError

import saliencegate.benchmarks.state_decay_v2.manifest as manifest_module
from saliencegate.benchmarks.state_decay_v2.config import (
    PUBLIC_GENERATION_SEED,
    SeedPurpose,
    derive_proposal_fixture_seed,
    derive_seed,
    proposal_fixture_seed_commitment,
)
from saliencegate.benchmarks.state_decay_v2.manifest import (
    ProposalFixtureCommitment,
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
    role_data_path,
    role_manifest_digest,
    role_manifest_path,
)
from saliencegate.benchmarks.state_decay_v2.protocol import validation_protocol_digests
from saliencegate.benchmarks.state_decay_v2.schema import ArtifactRole, BenchmarkSplit

_SPLIT_COUNTS = {
    BenchmarkSplit.TRAIN: 600,
    BenchmarkSplit.DEVELOPMENT: 300,
    BenchmarkSplit.LOCKED: 300,
    BenchmarkSplit.DIAGNOSTIC: 1_200,
}


def _digest(value: str) -> str:
    return sha256(value.encode("ascii")).hexdigest()


def _role_manifests() -> tuple[StateDecayV2RoleManifest, ...]:
    return tuple(
        build_role_manifest(
            split=split,
            role=role,
            canonical_byte_count=_SPLIT_COUNTS[split] * 200,
            content_digest=_digest(f"edge-content:{split.value}:{role.value}"),
            ordered_scenario_set_digest=_digest(f"edge-scenarios:{split.value}"),
            generator_version="state-decay-v2-generator-edge-v1",
            generator_configuration_digest=_digest("edge-generator-configuration"),
            template_registry_digest=_digest(f"edge-templates:{split.value}"),
        )
        for split in BenchmarkSplit
        for role in ArtifactRole
    )


def _replace_role_manifest(
    manifest: StateDecayV2RoleManifest,
    **changes: object,
) -> StateDecayV2RoleManifest:
    values: dict[str, object] = {
        "split": manifest.split,
        "role": manifest.role,
        "canonical_byte_count": manifest.canonical_byte_count,
        "content_digest": manifest.content_digest,
        "ordered_scenario_set_digest": manifest.ordered_scenario_set_digest,
        "generator_version": manifest.generator_version,
        "generator_configuration_digest": manifest.generator_configuration_digest,
        "template_registry_digest": manifest.template_registry_digest,
    }
    values.update(changes)
    return build_role_manifest(
        split=cast(BenchmarkSplit, values["split"]),
        role=cast(ArtifactRole, values["role"]),
        canonical_byte_count=cast(int, values["canonical_byte_count"]),
        content_digest=cast(str, values["content_digest"]),
        ordered_scenario_set_digest=cast(str, values["ordered_scenario_set_digest"]),
        generator_version=cast(str, values["generator_version"]),
        generator_configuration_digest=cast(
            str,
            values["generator_configuration_digest"],
        ),
        template_registry_digest=cast(str, values["template_registry_digest"]),
    )


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
    generation_digest = generation_commitment_digest or _digest("edge-generation-commitment")
    public_proposal_leaf = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.PROPOSAL)
    return tuple(
        ProposalFixtureCommitment(
            split=split,
            record_count=_SPLIT_COUNTS[split],
            generator_code_digest=_digest("edge-proposal-generator"),
            public_profile_digest=_digest("edge-public-response-profile"),
            seed_derivation_coordinates=("proposal_leaf", "split"),
            seed_commitment_digest=(
                proposal_fixture_seed_commitment(
                    derive_proposal_fixture_seed(public_proposal_leaf, split)
                )
                if split in (BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT)
                else _digest(f"edge-custody-proposal-seed:{split.value}")
            ),
            generation_commitment_digest=generation_digest,
            input_policy_view_digest=policy_digests[split],
            output_fixture_digest=_digest(f"edge-proposal-fixtures:{split.value}"),
        )
        for split in BenchmarkSplit
    )


def _replace_fixture(
    fixture: ProposalFixtureCommitment,
    **changes: object,
) -> ProposalFixtureCommitment:
    payload = fixture.model_dump(mode="python")
    payload.update(changes)
    return ProposalFixtureCommitment.model_validate(payload)


def _expected_protocol_digests() -> dict[ValidationAudit, str]:
    return validation_protocol_digests()


def _validations() -> tuple[SuiteValidationCommitment, ...]:
    protocol_digests = _expected_protocol_digests()
    return tuple(
        SuiteValidationCommitment(
            audit=audit,
            status=ValidationStatus.PASSED,
            protocol_digest=protocol_digests[audit],
            report_digest=_digest(f"edge-report:{audit.value}"),
        )
        for audit in ValidationAudit
    )


def _suite_lock() -> StateDecayV2SuiteLock:
    manifests = _role_manifests()
    return build_suite_lock(
        role_manifests=manifests,
        proposal_fixtures=_fixtures(manifests),
        validations=_validations(),
        expected_protocol_digests=_expected_protocol_digests(),
        generation_commitment_digest=_digest("edge-generation-commitment"),
    )


def _public_role_manifests(
    manifests: tuple[StateDecayV2RoleManifest, ...],
) -> tuple[StateDecayV2RoleManifest, ...]:
    return tuple(
        manifest
        for manifest in manifests
        if manifest.split in (BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT)
    )


def _public_bundle() -> StateDecayV2PublicBundleManifest:
    manifests = _role_manifests()
    return build_public_bundle_manifest(
        suite_lock=_suite_lock(),
        public_role_manifests=_public_role_manifests(manifests),
    )


class _ExplodingProtocolMapping(Mapping[ValidationAudit, str]):
    def __getitem__(self, key: ValidationAudit) -> str:
        raise RuntimeError("private mapping detail")

    def __iter__(self):  # type: ignore[no-untyped-def]
        raise RuntimeError("private mapping detail")

    def __len__(self) -> int:
        raise RuntimeError("private mapping detail")

    def items(self) -> NoReturn:
        raise RuntimeError("private mapping detail")


@pytest.mark.parametrize("value", (".", "..", "nested/file", "nested\\file"))
def test_exact_basename_defense_rejects_non_plain_paths(value: str) -> None:
    with pytest.raises(ValueError, match="plain basename"):
        manifest_module._exact_basename(value)


@pytest.mark.parametrize(
    ("split", "role"),
    (
        ("train", ArtifactRole.POLICY_VIEW),
        (BenchmarkSplit.TRAIN, "policy-view"),
        (1, ArtifactRole.POLICY_VIEW),
    ),
)
@pytest.mark.parametrize("path_builder", (role_data_path, role_manifest_path))
def test_role_path_builders_reject_ill_typed_identity(
    split: object,
    role: object,
    path_builder: object,
) -> None:
    with pytest.raises(ValueError, match="role path identity is invalid"):
        cast(object, path_builder)(split, role)  # type: ignore[operator]


def test_role_manifest_digest_accepts_json_identity_and_is_role_bound() -> None:
    manifest = _role_manifests()[0]
    payload = manifest.model_dump(mode="json")

    assert role_manifest_digest(payload) == manifest.manifest_digest
    other_role = dict(payload)
    other_role["role"] = ArtifactRole.ORACLE_VAULT.value
    assert role_manifest_digest(other_role) != manifest.manifest_digest


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"split": BenchmarkSplit.TRAIN},
        {"split": "unknown", "role": ArtifactRole.POLICY_VIEW.value},
        {"split": 1, "role": ArtifactRole.POLICY_VIEW},
        {"split": BenchmarkSplit.TRAIN.value, "role": "unknown"},
        {"split": BenchmarkSplit.TRAIN, "role": 1},
    ),
)
def test_role_manifest_digest_normalizes_invalid_identity(
    payload: Mapping[str, object],
) -> None:
    with pytest.raises(ValueError, match="role manifest identity is invalid"):
        role_manifest_digest(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema_version", "wrong-role-manifest/v1", None),
        ("record_schema_version", "wrong-record-schema/v1", "record schema"),
        ("data_path", "renamed.jsonl", "data path"),
        ("record_count", 599, "split geometry"),
        ("canonical_byte_count", 1_799, "canonical role bytes"),
        ("manifest_digest", "0" * 64, "manifest digest"),
    ),
)
def test_role_manifest_rejects_schema_path_geometry_and_digest_edges(
    field: str,
    value: object,
    message: str | None,
) -> None:
    payload = _role_manifests()[0].model_dump(mode="python")
    payload[field] = value

    with pytest.raises(ValidationError, match=message):
        StateDecayV2RoleManifest.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema_version", "wrong-role-descriptor/v1", None),
        ("record_schema_version", "wrong-record-schema/v1", "descriptor identity"),
        ("data_path", "renamed.jsonl", "descriptor identity"),
        ("manifest_path", "renamed.manifest.json", "descriptor identity"),
        ("record_count", 599, "descriptor identity"),
        ("canonical_byte_count", 1_799, "descriptor identity"),
        ("role_manifest_digest", "0" * 64, "does not bind"),
    ),
)
def test_role_descriptor_rejects_schema_path_geometry_and_digest_edges(
    field: str,
    value: object,
    message: str | None,
) -> None:
    descriptor = RoleManifestDescriptor.from_manifest(_role_manifests()[0])
    payload = descriptor.model_dump(mode="python")
    payload[field] = value

    with pytest.raises(ValidationError, match=message):
        RoleManifestDescriptor.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema_version", "wrong-proposal-fixture/v1", None),
        ("record_count", 599, "fixture count"),
        ("seed_purpose", SeedPurpose.ID, None),
        ("seed_derivation_domain", "wrong-seed-domain", None),
        ("seed_commitment_domain", "wrong-commitment-domain", None),
        ("seed_hash_primitive", "sha256/v0", None),
        (
            "seed_derivation_coordinates",
            ("split", "proposal_leaf"),
            "seed coordinates",
        ),
        ("seed_commitment_digest", _digest("wrong-public-seed"), "public proposal"),
    ),
)
def test_proposal_fixture_rejects_schema_geometry_seed_and_coordinate_edges(
    field: str,
    value: object,
    message: str | None,
) -> None:
    payload = _fixtures(_role_manifests())[0].model_dump(mode="python")
    payload[field] = value

    with pytest.raises(ValidationError, match=message):
        ProposalFixtureCommitment.model_validate(payload)


def test_suite_lock_duplicate_path_defense_is_reachable_only_after_unchecked_copy() -> None:
    lock = _suite_lock()
    roles = list(lock.roles)
    roles[1] = roles[1].model_copy(update={"data_path": roles[0].data_path})
    unchecked = lock.model_copy(update={"roles": tuple(roles)})

    with pytest.raises(ValueError, match="role paths must be unique"):
        unchecked.bindings_are_complete_non_circular_and_self_attesting()


def test_suite_lock_rejects_duplicate_role_content_digest() -> None:
    manifests = list(_role_manifests())
    manifests[1] = _replace_role_manifest(
        manifests[1],
        content_digest=manifests[0].content_digest,
    )
    checked_manifests = tuple(manifests)

    with pytest.raises(ValidationError, match="role digests must be role-bound and unique"):
        build_suite_lock(
            role_manifests=checked_manifests,
            proposal_fixtures=_fixtures(checked_manifests),
            validations=_validations(),
            expected_protocol_digests=_expected_protocol_digests(),
            generation_commitment_digest=_digest("edge-generation-commitment"),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("generator_version", "substituted-generator-v1"),
        ("generator_configuration_digest", _digest("substituted-generator-config")),
    ),
)
def test_suite_lock_rejects_role_generator_identity_substitution(
    field: str,
    value: object,
) -> None:
    manifests = list(_role_manifests())
    manifests[1] = _replace_role_manifest(manifests[1], **{field: value})
    checked_manifests = tuple(manifests)

    with pytest.raises(ValidationError, match="role generator identity does not match"):
        build_suite_lock(
            role_manifests=checked_manifests,
            proposal_fixtures=_fixtures(checked_manifests),
            validations=_validations(),
            expected_protocol_digests=_expected_protocol_digests(),
            generation_commitment_digest=_digest("edge-generation-commitment"),
        )


def test_suite_lock_rejects_scenario_digest_reused_across_splits() -> None:
    manifests = tuple(
        _replace_role_manifest(
            manifest,
            ordered_scenario_set_digest=_digest("edge-scenarios:train"),
        )
        if manifest.split is BenchmarkSplit.DEVELOPMENT
        else manifest
        for manifest in _role_manifests()
    )

    with pytest.raises(ValidationError, match="split scenario sets must be disjoint"):
        build_suite_lock(
            role_manifests=manifests,
            proposal_fixtures=_fixtures(manifests),
            validations=_validations(),
            expected_protocol_digests=_expected_protocol_digests(),
            generation_commitment_digest=_digest("edge-generation-commitment"),
        )


@pytest.mark.parametrize(
    ("collection", "message"),
    (
        ("proposal_fixtures", "proposal fixtures must be complete"),
        ("validations", "suite validations must be complete"),
    ),
)
def test_suite_lock_model_rejects_reordered_nested_inventory(
    collection: str,
    message: str,
) -> None:
    payload = _suite_lock().model_dump(mode="python")
    payload[collection] = tuple(reversed(payload[collection]))

    with pytest.raises(ValidationError, match=message):
        StateDecayV2SuiteLock.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    (
        "generator_code_digest",
        "public_profile_digest",
        "seed_commitment_digest",
        "output_fixture_digest",
    ),
)
def test_suite_lock_rejects_inconsistent_or_duplicate_fixture_inventory(field: str) -> None:
    manifests = _role_manifests()
    fixtures = list(_fixtures(manifests))
    if field in {"seed_commitment_digest", "output_fixture_digest"}:
        fixtures[3] = _replace_fixture(
            fixtures[3],
            **{field: getattr(fixtures[2], field)},
        )
    else:
        fixtures[2] = _replace_fixture(
            fixtures[2],
            **{field: _digest(f"substituted-{field}")},
        )

    with pytest.raises(ValidationError, match="fixture commitments are inconsistent"):
        build_suite_lock(
            role_manifests=manifests,
            proposal_fixtures=fixtures,
            validations=_validations(),
            expected_protocol_digests=_expected_protocol_digests(),
            generation_commitment_digest=_digest("edge-generation-commitment"),
        )


def test_suite_lock_rejects_duplicate_validation_report_digest() -> None:
    manifests = _role_manifests()
    validations = list(_validations())
    validations[1] = SuiteValidationCommitment.model_validate(
        {
            **validations[1].model_dump(mode="python"),
            "report_digest": validations[0].report_digest,
        }
    )

    with pytest.raises(ValidationError, match="validation reports must be unique"):
        build_suite_lock(
            role_manifests=manifests,
            proposal_fixtures=_fixtures(manifests),
            validations=validations,
            expected_protocol_digests=_expected_protocol_digests(),
            generation_commitment_digest=_digest("edge-generation-commitment"),
        )


@pytest.mark.parametrize(
    "digest_source",
    (
        "generator_configuration",
        "role_content",
        "role_manifest",
        "scenario_set",
        "template_registry",
        "proposal_generator",
        "public_profile",
        "proposal_seed",
        "policy_input",
        "fixture_output",
        "validation_protocol",
        "validation_report",
    ),
)
def test_generation_commitment_cannot_substitute_any_bound_digest(
    digest_source: str,
) -> None:
    manifests = _role_manifests()
    base_fixtures = _fixtures(manifests)
    validations = _validations()
    target_by_source = {
        "generator_configuration": manifests[0].generator_configuration_digest,
        "role_content": manifests[0].content_digest,
        "role_manifest": manifests[0].manifest_digest,
        "scenario_set": manifests[0].ordered_scenario_set_digest,
        "template_registry": manifests[0].template_registry_digest,
        "proposal_generator": base_fixtures[0].generator_code_digest,
        "public_profile": base_fixtures[0].public_profile_digest,
        "proposal_seed": base_fixtures[0].seed_commitment_digest,
        "policy_input": base_fixtures[0].input_policy_view_digest,
        "fixture_output": base_fixtures[0].output_fixture_digest,
        "validation_protocol": validations[0].protocol_digest,
        "validation_report": validations[0].report_digest,
    }
    generation_digest = target_by_source[digest_source]

    with pytest.raises(ValidationError, match="generation commitment cannot substitute"):
        build_suite_lock(
            role_manifests=manifests,
            proposal_fixtures=_fixtures(
                manifests,
                generation_commitment_digest=generation_digest,
            ),
            validations=validations,
            expected_protocol_digests=_expected_protocol_digests(),
            generation_commitment_digest=generation_digest,
        )


def test_suite_lock_builder_canonicalizes_reordered_sequences() -> None:
    manifests = _role_manifests()
    fixtures = _fixtures(manifests)
    validations = _validations()
    lock = build_suite_lock(
        role_manifests=tuple(reversed(manifests)),
        proposal_fixtures=tuple(reversed(fixtures)),
        validations=tuple(reversed(validations)),
        expected_protocol_digests=dict(reversed(tuple(_expected_protocol_digests().items()))),
        generation_commitment_digest=_digest("edge-generation-commitment"),
    )

    assert tuple((item.split, item.role) for item in lock.roles) == tuple(
        (split, role) for split in BenchmarkSplit for role in ArtifactRole
    )
    assert tuple(item.split for item in lock.proposal_fixtures) == tuple(BenchmarkSplit)
    assert tuple(item.audit for item in lock.validations) == tuple(ValidationAudit)


def test_suite_lock_builder_rejects_empty_role_sequence() -> None:
    with pytest.raises(ValueError, match="suite lock requires role manifests"):
        build_suite_lock(
            role_manifests=(),
            proposal_fixtures=(),
            validations=(),
            expected_protocol_digests=_expected_protocol_digests(),
            generation_commitment_digest=_digest("edge-generation-commitment"),
        )


@pytest.mark.parametrize(
    ("collection", "message"),
    (
        ("roles", "suite roles must be complete"),
        ("fixtures", "proposal fixtures must be complete"),
        ("validations", "suite validations must be complete"),
    ),
)
def test_suite_lock_builder_rejects_duplicate_sequence_inventory(
    collection: str,
    message: str,
) -> None:
    manifests = _role_manifests()
    fixtures = _fixtures(manifests)
    validations = _validations()
    role_input = (*manifests[:-1], manifests[0]) if collection == "roles" else manifests
    fixture_input = (*fixtures[:-1], fixtures[0]) if collection == "fixtures" else fixtures
    validation_input = (
        (*validations[:-1], validations[0]) if collection == "validations" else validations
    )

    with pytest.raises(ValidationError, match=message):
        build_suite_lock(
            role_manifests=role_input,
            proposal_fixtures=fixture_input,
            validations=validation_input,
            expected_protocol_digests=_expected_protocol_digests(),
            generation_commitment_digest=_digest("edge-generation-commitment"),
        )


@pytest.mark.parametrize("invalid", ((), [], "not-a-mapping"))
def test_suite_lock_builder_rejects_non_mapping_protocol_inventory(invalid: object) -> None:
    manifests = _role_manifests()

    with pytest.raises(ValueError, match="expected validation protocols are invalid"):
        build_suite_lock(
            role_manifests=manifests,
            proposal_fixtures=_fixtures(manifests),
            validations=_validations(),
            expected_protocol_digests=cast(Mapping[ValidationAudit, str], invalid),
            generation_commitment_digest=_digest("edge-generation-commitment"),
        )


def test_suite_lock_builder_rejects_duplicate_protocol_digests() -> None:
    manifests = _role_manifests()
    expected = _expected_protocol_digests()
    expected[ValidationAudit.LEAKAGE] = expected[ValidationAudit.GEOMETRY]

    with pytest.raises(ValueError, match="expected validation protocols are invalid"):
        build_suite_lock(
            role_manifests=manifests,
            proposal_fixtures=_fixtures(manifests),
            validations=_validations(),
            expected_protocol_digests=expected,
            generation_commitment_digest=_digest("edge-generation-commitment"),
        )


def test_suite_lock_builder_normalizes_hostile_protocol_mapping() -> None:
    manifests = _role_manifests()

    with pytest.raises(ValueError, match="expected validation protocols are invalid") as captured:
        build_suite_lock(
            role_manifests=manifests,
            proposal_fixtures=_fixtures(manifests),
            validations=_validations(),
            expected_protocol_digests=_ExplodingProtocolMapping(),
            generation_commitment_digest=_digest("edge-generation-commitment"),
        )
    assert "private mapping detail" not in str(captured.value)


@pytest.mark.parametrize("substitution", ("duplicate-child", "suite-lock-as-child"))
def test_public_bundle_rejects_child_digest_substitution(substitution: str) -> None:
    root = _public_bundle()
    payload = root.model_dump(mode="python")
    children = list(payload["children"])
    target = dict(children[1])
    target["content_digest"] = (
        children[0]["content_digest"]
        if substitution == "duplicate-child"
        else root.suite_lock_digest
    )
    children[1] = target
    payload["children"] = tuple(children)
    message = "child digests must be unique|suite lock cannot substitute"

    with pytest.raises(ValidationError, match=message):
        StateDecayV2PublicBundleManifest.model_validate(payload)


@pytest.mark.parametrize(
    "inventory",
    ("missing", "duplicate", "hidden-replacement", "extra-hidden"),
)
def test_public_bundle_builder_rejects_noncanonical_manifest_inventory(inventory: str) -> None:
    manifests = _role_manifests()
    public = _public_role_manifests(manifests)
    hidden = next(manifest for manifest in manifests if manifest.split is BenchmarkSplit.LOCKED)
    by_case = {
        "missing": public[:-1],
        "duplicate": (*public[:-1], public[0]),
        "hidden-replacement": (*public[:-1], hidden),
        "extra-hidden": (*public, hidden),
    }

    with pytest.raises(ValueError, match="public role manifests must be complete"):
        build_public_bundle_manifest(
            suite_lock=_suite_lock(),
            public_role_manifests=by_case[inventory],
        )
