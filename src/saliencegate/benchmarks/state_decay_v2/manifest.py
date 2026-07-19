from __future__ import annotations

import hmac
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from saliencegate.benchmarks.state_decay_v2.authority import (
    ANALYSIS_CLUSTER_ENTRY_SCHEMA_VERSION,
    ORACLE_VAULT_ENTRY_SCHEMA_VERSION,
)
from saliencegate.benchmarks.state_decay_v2.config import (
    PROPOSAL_FIXTURE_SEED_COMMITMENT_DOMAIN,
    PROPOSAL_FIXTURE_SEED_DOMAIN,
    PUBLIC_GENERATION_SEED,
    SeedPurpose,
    derive_proposal_fixture_seed,
    derive_seed,
    proposal_fixture_seed_commitment,
)
from saliencegate.benchmarks.state_decay_v2.schema import (
    POLICY_VIEW_SCHEMA_VERSION,
    SUITE_ID,
    SUITE_VERSION,
    ArtifactRole,
    BenchmarkSplit,
)
from saliencegate.domain import canonical_json, length_prefixed_sha256
from saliencegate.domain.records import ComponentIdentifier, Sha256Digest

ROLE_MANIFEST_SCHEMA_VERSION: Literal["state-decay-v2-role-manifest/v1"] = (
    "state-decay-v2-role-manifest/v1"
)
SUITE_LOCK_SCHEMA_VERSION: Literal["state-decay-v2-suite-lock/v1"] = "state-decay-v2-suite-lock/v1"
PUBLIC_BUNDLE_MANIFEST_SCHEMA_VERSION: Literal["state-decay-v2-public-bundle/v1"] = (
    "state-decay-v2-public-bundle/v1"
)

_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
_ROLE_MANIFEST_DIGEST_DOMAIN = "saliencegate:state-decay-v2:role-manifest:v1"
_SUITE_LOCK_DIGEST_DOMAIN = "saliencegate:state-decay-v2:suite-lock:v1"
_PUBLIC_BUNDLE_DIGEST_DOMAIN = "saliencegate:state-decay-v2:public-bundle:v1"

_SPLIT_COUNTS = {
    BenchmarkSplit.TRAIN: 600,
    BenchmarkSplit.DEVELOPMENT: 300,
    BenchmarkSplit.LOCKED: 300,
    BenchmarkSplit.DIAGNOSTIC: 1_200,
}
_ROLE_RECORD_SCHEMAS = {
    ArtifactRole.POLICY_VIEW: POLICY_VIEW_SCHEMA_VERSION,
    ArtifactRole.ORACLE_VAULT: ORACLE_VAULT_ENTRY_SCHEMA_VERSION,
    ArtifactRole.ANALYSIS_CLUSTER_MAP: ANALYSIS_CLUSTER_ENTRY_SCHEMA_VERSION,
}
_ROLE_KEYS = tuple((split, role) for split in BenchmarkSplit for role in ArtifactRole)
_PUBLIC_ROLE_KEYS = tuple(
    (split, role)
    for split in (BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT)
    for role in ArtifactRole
)


def _exact_basename(value: str) -> str:
    if type(value) is not str or "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError("artifact path must be a plain basename")
    return value


ArtifactBasename = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9][a-z0-9.-]*$",
    ),
    AfterValidator(_exact_basename),
]
CanonicalByteCount = Annotated[int, Field(ge=1, le=_MAX_ARTIFACT_BYTES)]
RecordCount = Annotated[int, Field(ge=1, le=1_200)]


class ValidationAudit(StrEnum):
    GEOMETRY = "geometry"
    LINEAGE_REVIEW = "lineage-review"
    TREATMENT_COVERAGE = "treatment-coverage"
    LEAKAGE = "leakage"
    FINITE_SAMPLE = "finite-sample"


def _checked_validation_protocol_digests(
    value: Mapping[ValidationAudit, str],
) -> dict[ValidationAudit, str]:
    try:
        if not isinstance(value, Mapping):
            raise TypeError
        items = tuple(value.items())
        keys = tuple(item[0] for item in items)
        digests = tuple(item[1] for item in items)
        if (
            len(items) != len(ValidationAudit)
            or any(type(key) is not ValidationAudit for key in keys)
            or set(keys) != set(ValidationAudit)
            or any(
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in digests
            )
            or len(set(digests)) != len(digests)
        ):
            raise ValueError
        by_audit = dict(items)
        return {audit: by_audit[audit] for audit in ValidationAudit}
    except Exception:
        raise ValueError("expected validation protocols are invalid") from None


class ValidationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class EvidenceBoundary(StrEnum):
    PUBLIC_SYNTHETIC_DEVELOPMENT = "public_synthetic_development"


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def role_data_path(split: BenchmarkSplit, role: ArtifactRole) -> str:
    if type(split) is not BenchmarkSplit or type(role) is not ArtifactRole:
        raise ValueError("role path identity is invalid")
    return f"{split.value}-{role.value}.jsonl"


def role_manifest_path(split: BenchmarkSplit, role: ArtifactRole) -> str:
    return f"{role_data_path(split, role)[:-6]}.manifest.json"


def _digest_payload(
    value: BaseModel | Mapping[str, object],
    *,
    digest_field: str,
    domain: str,
) -> str:
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", exclude={digest_field}, warnings=False)
    else:
        payload = {key: item for key, item in value.items() if key != digest_field}
    return length_prefixed_sha256(canonical_json(payload), domain=domain)


def role_manifest_digest(value: BaseModel | Mapping[str, object]) -> str:
    payload = value.model_dump(mode="python") if isinstance(value, BaseModel) else value
    try:
        raw_split = payload["split"]
        raw_role = payload["role"]
        if type(raw_split) is BenchmarkSplit:
            split = raw_split
        elif type(raw_split) is str:
            split = BenchmarkSplit(raw_split)
        else:
            raise ValueError("role manifest identity type is invalid")
        if type(raw_role) is ArtifactRole:
            role = raw_role
        elif type(raw_role) is str:
            role = ArtifactRole(raw_role)
        else:
            raise ValueError("role manifest identity type is invalid")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("role manifest identity is invalid") from error
    return _digest_payload(
        value,
        digest_field="manifest_digest",
        domain=f"{_ROLE_MANIFEST_DIGEST_DOMAIN}:{split.value}:{role.value}",
    )


def suite_lock_digest(value: BaseModel | Mapping[str, object]) -> str:
    return _digest_payload(
        value,
        digest_field="suite_lock_digest",
        domain=_SUITE_LOCK_DIGEST_DOMAIN,
    )


def public_bundle_manifest_digest(value: BaseModel | Mapping[str, object]) -> str:
    return _digest_payload(
        value,
        digest_field="manifest_digest",
        domain=_PUBLIC_BUNDLE_DIGEST_DOMAIN,
    )


class StateDecayV2RoleManifest(_StrictModel):
    schema_version: Literal["state-decay-v2-role-manifest/v1"] = ROLE_MANIFEST_SCHEMA_VERSION
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    split: BenchmarkSplit
    role: ArtifactRole
    record_schema_version: ComponentIdentifier
    data_path: ArtifactBasename
    record_count: RecordCount
    canonical_byte_count: CanonicalByteCount
    content_digest: Sha256Digest
    ordered_scenario_set_digest: Sha256Digest
    generator_version: ComponentIdentifier
    generator_configuration_digest: Sha256Digest
    template_registry_digest: Sha256Digest
    manifest_digest: Sha256Digest

    @model_validator(mode="after")
    def identity_geometry_and_digest_are_exact(self) -> Self:
        if self.record_schema_version != _ROLE_RECORD_SCHEMAS[self.role]:
            raise ValueError("role record schema does not match its authority")
        if self.data_path != role_data_path(self.split, self.role):
            raise ValueError("role data path does not match its authority")
        if self.record_count != _SPLIT_COUNTS[self.split]:
            raise ValueError("role record count does not match split geometry")
        if self.canonical_byte_count < self.record_count * 3:
            raise ValueError("canonical role bytes cannot contain the declared rows")
        if not hmac.compare_digest(self.manifest_digest, role_manifest_digest(self)):
            raise ValueError("role manifest digest does not match")
        return self


def build_role_manifest(
    *,
    split: BenchmarkSplit,
    role: ArtifactRole,
    canonical_byte_count: int,
    content_digest: str,
    ordered_scenario_set_digest: str,
    generator_version: str,
    generator_configuration_digest: str,
    template_registry_digest: str,
) -> StateDecayV2RoleManifest:
    values: dict[str, object] = {
        "schema_version": ROLE_MANIFEST_SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "split": split,
        "role": role,
        "record_schema_version": _ROLE_RECORD_SCHEMAS[role],
        "data_path": role_data_path(split, role),
        "record_count": _SPLIT_COUNTS[split],
        "canonical_byte_count": canonical_byte_count,
        "content_digest": content_digest,
        "ordered_scenario_set_digest": ordered_scenario_set_digest,
        "generator_version": generator_version,
        "generator_configuration_digest": generator_configuration_digest,
        "template_registry_digest": template_registry_digest,
    }
    values["manifest_digest"] = role_manifest_digest(values)
    return StateDecayV2RoleManifest.model_validate(values)


class RoleManifestDescriptor(_StrictModel):
    schema_version: Literal["state-decay-v2-role-descriptor/v1"] = (
        "state-decay-v2-role-descriptor/v1"
    )
    split: BenchmarkSplit
    role: ArtifactRole
    record_schema_version: ComponentIdentifier
    data_path: ArtifactBasename
    manifest_path: ArtifactBasename
    record_count: RecordCount
    canonical_byte_count: CanonicalByteCount
    content_digest: Sha256Digest
    ordered_scenario_set_digest: Sha256Digest
    generator_version: ComponentIdentifier
    generator_configuration_digest: Sha256Digest
    template_registry_digest: Sha256Digest
    role_manifest_digest: Sha256Digest

    @model_validator(mode="after")
    def descriptor_matches_its_declared_role(self) -> Self:
        if (
            self.record_schema_version != _ROLE_RECORD_SCHEMAS[self.role]
            or self.data_path != role_data_path(self.split, self.role)
            or self.manifest_path != role_manifest_path(self.split, self.role)
            or self.record_count != _SPLIT_COUNTS[self.split]
            or self.canonical_byte_count < self.record_count * 3
        ):
            raise ValueError("role descriptor identity does not match")
        manifest_payload = {
            "schema_version": ROLE_MANIFEST_SCHEMA_VERSION,
            "suite_id": SUITE_ID,
            "suite_version": SUITE_VERSION,
            "split": self.split,
            "role": self.role,
            "record_schema_version": self.record_schema_version,
            "data_path": self.data_path,
            "record_count": self.record_count,
            "canonical_byte_count": self.canonical_byte_count,
            "content_digest": self.content_digest,
            "ordered_scenario_set_digest": self.ordered_scenario_set_digest,
            "generator_version": self.generator_version,
            "generator_configuration_digest": self.generator_configuration_digest,
            "template_registry_digest": self.template_registry_digest,
        }
        if not hmac.compare_digest(
            self.role_manifest_digest,
            role_manifest_digest(manifest_payload),
        ):
            raise ValueError("role descriptor does not bind its manifest fields")
        return self

    @classmethod
    def from_manifest(cls, manifest: StateDecayV2RoleManifest) -> Self:
        checked = StateDecayV2RoleManifest.model_validate_json(canonical_json(manifest))
        return cls(
            split=checked.split,
            role=checked.role,
            record_schema_version=checked.record_schema_version,
            data_path=checked.data_path,
            manifest_path=role_manifest_path(checked.split, checked.role),
            record_count=checked.record_count,
            canonical_byte_count=checked.canonical_byte_count,
            content_digest=checked.content_digest,
            ordered_scenario_set_digest=checked.ordered_scenario_set_digest,
            generator_version=checked.generator_version,
            generator_configuration_digest=checked.generator_configuration_digest,
            template_registry_digest=checked.template_registry_digest,
            role_manifest_digest=checked.manifest_digest,
        )


class ProposalFixtureCommitment(_StrictModel):
    schema_version: Literal["proposal-fixture-commitment/v1"] = "proposal-fixture-commitment/v1"
    split: BenchmarkSplit
    record_count: RecordCount
    generator_code_digest: Sha256Digest
    public_profile_digest: Sha256Digest
    seed_purpose: Literal[SeedPurpose.PROPOSAL] = SeedPurpose.PROPOSAL
    seed_derivation_domain: Literal["saliencegate:state-decay-v2:proposal-fixture-seed:v1"] = (
        PROPOSAL_FIXTURE_SEED_DOMAIN
    )
    seed_commitment_domain: Literal[
        "saliencegate:state-decay-v2:proposal-fixture-seed-commitment:v1"
    ] = PROPOSAL_FIXTURE_SEED_COMMITMENT_DOMAIN
    seed_hash_primitive: Literal["length-prefixed-sha256/v1"] = "length-prefixed-sha256/v1"
    seed_derivation_coordinates: Annotated[tuple[str, ...], Field(min_length=2, max_length=2)]
    seed_commitment_digest: Sha256Digest
    generation_commitment_digest: Sha256Digest
    input_policy_view_digest: Sha256Digest
    output_fixture_digest: Sha256Digest

    @model_validator(mode="after")
    def fixture_geometry_matches_its_split(self) -> Self:
        if self.record_count != _SPLIT_COUNTS[self.split]:
            raise ValueError("proposal fixture count does not match split geometry")
        if self.seed_derivation_coordinates != ("proposal_leaf", "split"):
            raise ValueError("proposal fixture seed coordinates are not canonical")
        if self.split in (BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT):
            proposal_leaf = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.PROPOSAL)
            expected_commitment = proposal_fixture_seed_commitment(
                derive_proposal_fixture_seed(proposal_leaf, self.split)
            )
            if not hmac.compare_digest(self.seed_commitment_digest, expected_commitment):
                raise ValueError("public proposal fixture seed commitment does not match")
        return self


class SuiteValidationCommitment(_StrictModel):
    schema_version: Literal["suite-validation-commitment/v1"] = "suite-validation-commitment/v1"
    audit: ValidationAudit
    status: ValidationStatus
    protocol_digest: Sha256Digest
    report_digest: Sha256Digest


class StateDecayV2SuiteLock(_StrictModel):
    schema_version: Literal["state-decay-v2-suite-lock/v1"] = SUITE_LOCK_SCHEMA_VERSION
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    generator_version: ComponentIdentifier
    generator_configuration_digest: Sha256Digest
    generation_commitment_digest: Sha256Digest
    roles: Annotated[tuple[RoleManifestDescriptor, ...], Field(min_length=12, max_length=12)]
    proposal_fixtures: Annotated[
        tuple[ProposalFixtureCommitment, ...],
        Field(min_length=4, max_length=4),
    ]
    validations: Annotated[
        tuple[SuiteValidationCommitment, ...],
        Field(min_length=5, max_length=5),
    ]
    suite_lock_digest: Sha256Digest

    @model_validator(mode="after")
    def bindings_are_complete_non_circular_and_self_attesting(self) -> Self:
        role_keys = tuple((item.split, item.role) for item in self.roles)
        if role_keys != _ROLE_KEYS:
            raise ValueError("suite roles must be complete and canonically ordered")
        all_paths = tuple(
            path for item in self.roles for path in (item.data_path, item.manifest_path)
        )
        if len(set(all_paths)) != len(all_paths):
            raise ValueError("suite role paths must be unique")
        if len({item.content_digest for item in self.roles}) != len(self.roles) or len(
            {item.role_manifest_digest for item in self.roles}
        ) != len(self.roles):
            raise ValueError("suite role digests must be role-bound and unique")
        if any(
            item.generator_version != self.generator_version
            or item.generator_configuration_digest != self.generator_configuration_digest
            for item in self.roles
        ):
            raise ValueError("suite role generator identity does not match")

        scenario_digests: set[str] = set()
        policy_digests: dict[BenchmarkSplit, str] = {}
        for split in BenchmarkSplit:
            split_roles = tuple(item for item in self.roles if item.split is split)
            if (
                len({item.record_count for item in split_roles}) != 1
                or len({item.ordered_scenario_set_digest for item in split_roles}) != 1
                or len({item.template_registry_digest for item in split_roles}) != 1
            ):
                raise ValueError(
                    "suite roles do not bind the same scenario set and template registry"
                )
            scenario_digest = split_roles[0].ordered_scenario_set_digest
            if scenario_digest in scenario_digests:
                raise ValueError("suite split scenario sets must be disjoint")
            scenario_digests.add(scenario_digest)
            policy_digests[split] = next(
                item.content_digest for item in split_roles if item.role is ArtifactRole.POLICY_VIEW
            )

        if tuple(item.split for item in self.proposal_fixtures) != tuple(BenchmarkSplit):
            raise ValueError("proposal fixtures must be complete and canonically ordered")
        if any(
            item.input_policy_view_digest != policy_digests[item.split]
            for item in self.proposal_fixtures
        ):
            raise ValueError("proposal fixture input does not match its policy view")
        if any(
            item.generation_commitment_digest != self.generation_commitment_digest
            for item in self.proposal_fixtures
        ):
            raise ValueError("proposal fixture seed does not bind the suite generation commitment")
        if (
            len({item.generator_code_digest for item in self.proposal_fixtures}) != 1
            or len({item.public_profile_digest for item in self.proposal_fixtures}) != 1
            or len({item.seed_commitment_digest for item in self.proposal_fixtures})
            != len(self.proposal_fixtures)
            or len({item.output_fixture_digest for item in self.proposal_fixtures})
            != len(self.proposal_fixtures)
        ):
            raise ValueError("proposal fixture commitments are inconsistent or substituted")

        if tuple(item.audit for item in self.validations) != tuple(ValidationAudit):
            raise ValueError("suite validations must be complete and canonically ordered")
        if any(item.status is not ValidationStatus.PASSED for item in self.validations):
            raise ValueError("suite validation did not pass")
        if len({item.report_digest for item in self.validations}) != len(self.validations):
            raise ValueError("suite validation reports must be unique")
        bound_digests = {
            self.generator_configuration_digest,
            *(item.content_digest for item in self.roles),
            *(item.role_manifest_digest for item in self.roles),
            *(item.ordered_scenario_set_digest for item in self.roles),
            *(item.template_registry_digest for item in self.roles),
            *(item.generator_code_digest for item in self.proposal_fixtures),
            *(item.public_profile_digest for item in self.proposal_fixtures),
            *(item.seed_commitment_digest for item in self.proposal_fixtures),
            *(item.input_policy_view_digest for item in self.proposal_fixtures),
            *(item.output_fixture_digest for item in self.proposal_fixtures),
            *(item.protocol_digest for item in self.validations),
            *(item.report_digest for item in self.validations),
        }
        if self.generation_commitment_digest in bound_digests:
            raise ValueError("generation commitment cannot substitute for another suite role")
        if not hmac.compare_digest(self.suite_lock_digest, suite_lock_digest(self)):
            raise ValueError("suite lock digest does not match")
        return self


def build_suite_lock(
    *,
    role_manifests: Sequence[StateDecayV2RoleManifest],
    proposal_fixtures: Sequence[ProposalFixtureCommitment],
    validations: Sequence[SuiteValidationCommitment],
    expected_protocol_digests: Mapping[ValidationAudit, str],
    generation_commitment_digest: str,
) -> StateDecayV2SuiteLock:
    checked_protocol_digests = _checked_validation_protocol_digests(expected_protocol_digests)
    role_order = {key: index for index, key in enumerate(_ROLE_KEYS)}
    checked_roles = tuple(
        sorted(
            (RoleManifestDescriptor.from_manifest(item) for item in role_manifests),
            key=lambda item: role_order[(item.split, item.role)],
        )
    )
    if not checked_roles:
        raise ValueError("suite lock requires role manifests")
    split_order = {split: index for index, split in enumerate(BenchmarkSplit)}
    audit_order = {audit: index for index, audit in enumerate(ValidationAudit)}
    checked_fixtures = tuple(
        sorted(
            (
                ProposalFixtureCommitment.model_validate_json(canonical_json(item))
                for item in proposal_fixtures
            ),
            key=lambda item: split_order[item.split],
        )
    )
    checked_validations = tuple(
        sorted(
            (
                SuiteValidationCommitment.model_validate_json(canonical_json(item))
                for item in validations
            ),
            key=lambda item: audit_order[item.audit],
        )
    )
    if any(
        not hmac.compare_digest(
            item.protocol_digest,
            checked_protocol_digests[item.audit],
        )
        for item in checked_validations
    ):
        raise ValueError("validation protocol digest does not match")
    values: dict[str, object] = {
        "schema_version": SUITE_LOCK_SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "generator_version": checked_roles[0].generator_version,
        "generator_configuration_digest": checked_roles[0].generator_configuration_digest,
        "generation_commitment_digest": generation_commitment_digest,
        "roles": checked_roles,
        "proposal_fixtures": checked_fixtures,
        "validations": checked_validations,
    }
    values["suite_lock_digest"] = suite_lock_digest(values)
    return StateDecayV2SuiteLock.model_validate(values)


class PublicBundleChildDescriptor(_StrictModel):
    schema_version: Literal["public-bundle-child/v1"] = "public-bundle-child/v1"
    path: ArtifactBasename
    canonical_byte_count: CanonicalByteCount
    content_digest: Sha256Digest


def _expected_public_child_paths() -> tuple[str, ...]:
    return tuple(
        path
        for split, role in _PUBLIC_ROLE_KEYS
        for path in (role_data_path(split, role), role_manifest_path(split, role))
    )


def expected_public_bundle_paths() -> frozenset[str]:
    return frozenset({"manifest.json", "suite-lock.json", *_expected_public_child_paths()})


class StateDecayV2PublicBundleManifest(_StrictModel):
    schema_version: Literal["state-decay-v2-public-bundle/v1"] = (
        PUBLIC_BUNDLE_MANIFEST_SCHEMA_VERSION
    )
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    evidence_boundary: Literal[EvidenceBoundary.PUBLIC_SYNTHETIC_DEVELOPMENT] = (
        EvidenceBoundary.PUBLIC_SYNTHETIC_DEVELOPMENT
    )
    confirmatory: Literal[False] = False
    locked_evidence: Literal["not_measured"] = "not_measured"
    external_claims_supported: Literal[False] = False
    suite_lock_digest: Sha256Digest
    children: Annotated[
        tuple[PublicBundleChildDescriptor, ...],
        Field(min_length=12, max_length=12),
    ]
    manifest_digest: Sha256Digest

    @model_validator(mode="after")
    def child_set_and_self_digest_are_exact(self) -> Self:
        if tuple(item.path for item in self.children) != _expected_public_child_paths():
            raise ValueError("public bundle children must be exact and canonically ordered")
        if len({item.content_digest for item in self.children}) != len(self.children):
            raise ValueError("public bundle child digests must be unique")
        if self.suite_lock_digest in {item.content_digest for item in self.children}:
            raise ValueError("suite lock cannot substitute for a public role child")
        if not hmac.compare_digest(
            self.manifest_digest,
            public_bundle_manifest_digest(self),
        ):
            raise ValueError("public bundle manifest digest does not match")
        return self


def build_public_bundle_manifest(
    *,
    suite_lock: StateDecayV2SuiteLock,
    public_role_manifests: Sequence[StateDecayV2RoleManifest],
) -> StateDecayV2PublicBundleManifest:
    checked_lock = StateDecayV2SuiteLock.model_validate_json(canonical_json(suite_lock))
    role_order = {key: index for index, key in enumerate(_PUBLIC_ROLE_KEYS)}
    checked_manifests = tuple(
        sorted(
            (
                StateDecayV2RoleManifest.model_validate_json(canonical_json(item))
                for item in public_role_manifests
            ),
            key=lambda item: role_order.get((item.split, item.role), len(role_order)),
        )
    )
    if tuple((item.split, item.role) for item in checked_manifests) != _PUBLIC_ROLE_KEYS:
        raise ValueError("public role manifests must be complete and canonically identifiable")

    locked_roles = {(item.split, item.role): item for item in checked_lock.roles}
    checked_children: list[PublicBundleChildDescriptor] = []
    for manifest in checked_manifests:
        key = (manifest.split, manifest.role)
        descriptor = RoleManifestDescriptor.from_manifest(manifest)
        if descriptor != locked_roles[key]:
            raise ValueError("public role manifest does not match the suite lock")
        checked_children.extend(
            (
                PublicBundleChildDescriptor(
                    path=manifest.data_path,
                    canonical_byte_count=manifest.canonical_byte_count,
                    content_digest=manifest.content_digest,
                ),
                PublicBundleChildDescriptor(
                    path=role_manifest_path(manifest.split, manifest.role),
                    canonical_byte_count=len(canonical_json(manifest)),
                    content_digest=manifest.manifest_digest,
                ),
            )
        )

    values: dict[str, object] = {
        "schema_version": PUBLIC_BUNDLE_MANIFEST_SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "evidence_boundary": EvidenceBoundary.PUBLIC_SYNTHETIC_DEVELOPMENT,
        "confirmatory": False,
        "locked_evidence": "not_measured",
        "external_claims_supported": False,
        "suite_lock_digest": checked_lock.suite_lock_digest,
        "children": tuple(checked_children),
    }
    values["manifest_digest"] = public_bundle_manifest_digest(values)
    return StateDecayV2PublicBundleManifest.model_validate(values)


__all__ = [
    "PUBLIC_BUNDLE_MANIFEST_SCHEMA_VERSION",
    "ROLE_MANIFEST_SCHEMA_VERSION",
    "SUITE_LOCK_SCHEMA_VERSION",
    "ArtifactBasename",
    "CanonicalByteCount",
    "EvidenceBoundary",
    "ProposalFixtureCommitment",
    "PublicBundleChildDescriptor",
    "RecordCount",
    "RoleManifestDescriptor",
    "StateDecayV2PublicBundleManifest",
    "StateDecayV2RoleManifest",
    "StateDecayV2SuiteLock",
    "SuiteValidationCommitment",
    "ValidationAudit",
    "ValidationStatus",
    "build_public_bundle_manifest",
    "build_role_manifest",
    "build_suite_lock",
    "expected_public_bundle_paths",
    "public_bundle_manifest_digest",
    "role_data_path",
    "role_manifest_digest",
    "role_manifest_path",
    "suite_lock_digest",
]
