from __future__ import annotations

import hmac
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.benchmarks.state_decay_v2.config import GENERATION_CONTRACT
from saliencegate.benchmarks.state_decay_v2.protocol import (
    LINEAGE_REVIEW_PROTOCOL,
    LineageReviewRecord,
    ReviewBoundary,
    ReviewDecision,
)
from saliencegate.benchmarks.state_decay_v2.public_contract import (
    CausalSemanticDelta,
    OutcomeFreeTaskTemplate,
    PublicEvidenceTopology,
    PublicFailureMechanism,
    PublicLineageKey,
    PublicSemanticSignature,
    PublicTransitionGraph,
    ReviewSafeText,
    parse_public_lineage_key,
    validate_review_safe_text,
)
from saliencegate.benchmarks.state_decay_v2.schema import (
    SUITE_ID,
    SUITE_VERSION,
    BenchmarkSplit,
    ScenarioFamily,
)
from saliencegate.domain import canonical_json, length_prefixed_sha256
from saliencegate.domain.records import ComponentIdentifier, Sha256Digest

REVIEW_CHECKLIST_DIGEST_DOMAIN: Literal[
    "saliencegate:state-decay-v2:public-review:checklist:v1"
] = "saliencegate:state-decay-v2:public-review:checklist:v1"
REVIEW_DRAFT_DIGEST_DOMAIN: Literal["saliencegate:state-decay-v2:public-review:draft:v1"] = (
    "saliencegate:state-decay-v2:public-review:draft:v1"
)
FAMILY_COMPARISON_DIGEST_DOMAIN: Literal[
    "saliencegate:state-decay-v2:public-review:family-comparison:v1"
] = "saliencegate:state-decay-v2:public-review:family-comparison:v1"
REVIEW_SUBMISSION_DIGEST_DOMAIN: Literal[
    "saliencegate:state-decay-v2:public-review:submission:v1"
] = "saliencegate:state-decay-v2:public-review:submission:v1"
REVIEW_ENVELOPE_DIGEST_DOMAIN: Literal["saliencegate:state-decay-v2:public-review:envelope:v1"] = (
    "saliencegate:state-decay-v2:public-review:envelope:v1"
)
ACCEPTED_ENVELOPE_REGISTRY_DIGEST_DOMAIN: Literal[
    "saliencegate:state-decay-v2:public-review:envelope-registry:v1"
] = "saliencegate:state-decay-v2:public-review:envelope-registry:v1"
SUBREPORT_DIGEST_DOMAIN: Literal["saliencegate:state-decay-v2:public-review:subreport:v1"] = (
    "saliencegate:state-decay-v2:public-review:subreport:v1"
)
PACK_CHILD_DIGEST_DOMAIN: Literal["saliencegate:state-decay-v2:public-review:pack-child:v1"] = (
    "saliencegate:state-decay-v2:public-review:pack-child:v1"
)
PACK_MANIFEST_DIGEST_DOMAIN: Literal[
    "saliencegate:state-decay-v2:public-review:pack-manifest:v1"
] = "saliencegate:state-decay-v2:public-review:pack-manifest:v1"

ACCEPTED_ENVELOPE_REGISTRY_BASENAME: Literal["state_decay_v2-public-review-envelopes.jsonl"] = (
    "state_decay_v2-public-review-envelopes.jsonl"
)
MAX_REVIEW_SUBMISSION_FILE_BYTES: Final = 8 * 1024
MAX_REVIEW_ENVELOPE_CANONICAL_BYTES: Final = 320 * 1024
MAX_ACCEPTED_ENVELOPE_REGISTRY_BYTES: Final = 64 * 1024 * 1024
MAX_REVIEW_PACK_LARGE_CHILD_BYTES: Final = 16 * 1024 * 1024
MAX_REVIEW_PACK_SMALL_CHILD_BYTES: Final = 256 * 1024
MAX_REVIEW_PACK_MANIFEST_FILE_BYTES: Final = 1024 * 1024
MAX_REVIEW_PACK_TOTAL_BYTES: Final = 64 * 1024 * 1024
MAX_REVIEW_CHAIN_LENGTH: Final = 32
PUBLIC_REVIEW_RECORD_COUNT: Final = 180

assert (
    PUBLIC_REVIEW_RECORD_COUNT * (MAX_REVIEW_ENVELOPE_CANONICAL_BYTES + 1)
    < MAX_ACCEPTED_ENVELOPE_REGISTRY_BYTES
)

_PUBLIC_TRAIN_FAMILIES = frozenset(
    {
        ScenarioFamily.FORGOTTEN_REQUIREMENT,
        ScenarioFamily.FAILED_PRIOR_ATTEMPT,
        ScenarioFamily.NEGLECTED_SUBGOAL,
        ScenarioFamily.STALE_MEMORY,
    }
)
_PUBLIC_DEVELOPMENT_FAMILIES = frozenset(
    {
        ScenarioFamily.STABLE_ENVIRONMENT_FACT,
        ScenarioFamily.RETAINED_DIAGNOSIS,
    }
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def _self_digest(
    value: BaseModel | Mapping[str, object],
    *,
    self_field: str,
    domain: str,
) -> str:
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", exclude={self_field}, warnings=False)
    else:
        try:
            payload = {key: item for key, item in value.items() if key != self_field}
        except Exception:
            raise ValueError("review digest payload is invalid") from None
    return length_prefixed_sha256(canonical_json(payload), domain=domain)


def public_review_checklist_digest(value: BaseModel | Mapping[str, object]) -> str:
    return _self_digest(
        value,
        self_field="checklist_digest",
        domain=REVIEW_CHECKLIST_DIGEST_DOMAIN,
    )


def review_draft_digest(value: BaseModel | Mapping[str, object]) -> str:
    return _self_digest(value, self_field="draft_digest", domain=REVIEW_DRAFT_DIGEST_DOMAIN)


def family_comparison_digest(value: BaseModel | Mapping[str, object]) -> str:
    return _self_digest(
        value,
        self_field="family_comparison_digest",
        domain=FAMILY_COMPARISON_DIGEST_DOMAIN,
    )


def review_submission_digest(value: BaseModel | Mapping[str, object]) -> str:
    return _self_digest(
        value,
        self_field="submission_digest",
        domain=REVIEW_SUBMISSION_DIGEST_DOMAIN,
    )


def review_envelope_digest(value: BaseModel | Mapping[str, object]) -> str:
    return _self_digest(
        value,
        self_field="envelope_digest",
        domain=REVIEW_ENVELOPE_DIGEST_DOMAIN,
    )


def subreport_digest(value: BaseModel | Mapping[str, object]) -> str:
    return _self_digest(value, self_field="subreport_digest", domain=SUBREPORT_DIGEST_DOMAIN)


def pack_manifest_digest(value: BaseModel | Mapping[str, object]) -> str:
    return _self_digest(
        value,
        self_field="manifest_digest",
        domain=PACK_MANIFEST_DIGEST_DOMAIN,
    )


def _expected_public_split(family: ScenarioFamily) -> BenchmarkSplit | None:
    if family in _PUBLIC_TRAIN_FAMILIES:
        return BenchmarkSplit.TRAIN
    if family in _PUBLIC_DEVELOPMENT_FAMILIES:
        return BenchmarkSplit.DEVELOPMENT
    return None


def _require_public_coordinates(
    *,
    split: BenchmarkSplit,
    family: ScenarioFamily,
    lineage_registry_key: str,
    subject: str,
) -> None:
    parsed_family, _ = parse_public_lineage_key(lineage_registry_key)
    if _expected_public_split(family) is not split or parsed_family is not family:
        raise ValueError(f"{subject} public lineage coordinates do not agree")


class PublicReviewChecklistItemId(StrEnum):
    CAUSAL_CLARITY = "causal_clarity"
    EVIDENCE_CONSISTENCY = "evidence_consistency"
    NO_OUTCOME_HINTS = "no_outcome_hints"
    EMPTY_PARENTAGE = "empty_parentage"
    PREVIEW_FIDELITY = "preview_fidelity"
    SIBLING_DISTINCTION = "sibling_distinction"
    REVIEWER_ALLOCATION_NONCONSULTATION = "reviewer_allocation_nonconsultation"


class PublicReviewAnswer(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


_CHECKLIST_TEXTS: tuple[str, ...] = (
    "The candidate defines one clear lineage-level causal distinction that can be evaluated "
    "without an assigned outcome.",
    "The task template, executable transition graph, evidence topology, raw detector fixture "
    "and expected reference projection, failure mechanism, semantic signature, and four "
    "candidate-owned causal deltas are mutually consistent.",
    "The candidate and all five previews contain no assigned outcome, allocation rank, scenario "
    "ID, oracle branch, or equivalent outcome hint.",
    "The candidate declares an empty derivation-parent tuple.",
    "All five previews faithfully materialize the candidate task template, raw detector fixture, "
    "and four causal deltas and vary by slot only through the frozen global profile catalog.",
    "The candidate is meaningfully distinct from all 29 siblings shown in the current family "
    "comparison.",
    "I attest that I did not consult or compute an allocation, allocation rank, or assigned "
    "outcome while reviewing this candidate.",
)


class PublicReviewChecklistItem(_StrictModel):
    item_id: PublicReviewChecklistItemId
    text: ReviewSafeText


class PublicReviewChecklistAnswer(_StrictModel):
    item_id: PublicReviewChecklistItemId
    answer: PublicReviewAnswer


class PublicReviewChecklist(_StrictModel):
    schema_version: Literal["state-decay-v2-public-review-checklist/v1"] = (
        "state-decay-v2-public-review-checklist/v1"
    )
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    items: Annotated[
        tuple[PublicReviewChecklistItem, ...],
        Field(min_length=7, max_length=7),
    ]
    checklist_digest: Sha256Digest

    @model_validator(mode="after")
    def checklist_is_exact_and_self_attesting(self) -> Self:
        expected = tuple(zip(tuple(PublicReviewChecklistItemId), _CHECKLIST_TEXTS, strict=True))
        actual = tuple((item.item_id, item.text) for item in self.items)
        if actual != expected:
            raise ValueError("public review checklist items are not canonical")
        if not hmac.compare_digest(
            self.checklist_digest,
            public_review_checklist_digest(self),
        ):
            raise ValueError("public review checklist digest does not match")
        return self


def _build_public_review_checklist() -> PublicReviewChecklist:
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-public-review-checklist/v1",
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "items": tuple(
            PublicReviewChecklistItem(item_id=item_id, text=text)
            for item_id, text in zip(PublicReviewChecklistItemId, _CHECKLIST_TEXTS, strict=True)
        ),
    }
    values["checklist_digest"] = public_review_checklist_digest(values)
    return PublicReviewChecklist.model_validate(values)


PUBLIC_REVIEW_CHECKLIST = _build_public_review_checklist()


class PublicCandidateSemanticProjection(_StrictModel):
    task_template: OutcomeFreeTaskTemplate
    transition_graph: PublicTransitionGraph
    evidence_topology: PublicEvidenceTopology
    failure_mechanism: PublicFailureMechanism
    semantic_signature: PublicSemanticSignature
    causal_deltas: Annotated[
        tuple[CausalSemanticDelta, ...],
        Field(min_length=4, max_length=4),
    ]

    @model_validator(mode="after")
    def causal_deltas_are_canonical(self) -> Self:
        if tuple(delta.delta_index for delta in self.causal_deltas) != tuple(range(4)):
            raise ValueError("semantic projection causal deltas are not canonical")
        return self


class PublicFamilyComparisonEntry(_StrictModel):
    lineage_registry_key: PublicLineageKey
    candidate_packet_digest: Sha256Digest
    semantic_rationale: ReviewSafeText
    semantic_projection: PublicCandidateSemanticProjection
    transition_graph_digest: Sha256Digest
    evidence_topology_digest: Sha256Digest
    failure_mechanism_id: ComponentIdentifier
    semantic_signature_digest: Sha256Digest
    preview_digests: Annotated[tuple[Sha256Digest, ...], Field(min_length=5, max_length=5)]

    @model_validator(mode="after")
    def redundant_projection_bindings_agree(self) -> Self:
        if (
            self.transition_graph_digest
            != self.semantic_projection.transition_graph.transition_graph_digest
            or self.evidence_topology_digest
            != self.semantic_projection.evidence_topology.evidence_topology_digest
            or self.failure_mechanism_id
            != self.semantic_projection.failure_mechanism.failure_mechanism_id
            or self.semantic_signature_digest
            != self.semantic_projection.semantic_signature.semantic_signature_digest
        ):
            raise ValueError("family comparison semantic projection bindings do not agree")
        if len(set(self.preview_digests)) != 5:
            raise ValueError("family comparison preview digests must be unique")
        return self


class PublicFamilyComparison(_StrictModel):
    schema_version: Literal["state-decay-v2-public-review-family-comparison/v1"] = (
        "state-decay-v2-public-review-family-comparison/v1"
    )
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    split: BenchmarkSplit
    family: ScenarioFamily
    entries: Annotated[
        tuple[PublicFamilyComparisonEntry, ...],
        Field(min_length=30, max_length=30),
    ]
    family_comparison_digest: Sha256Digest

    @model_validator(mode="after")
    def comparison_is_complete_ordered_and_self_attesting(self) -> Self:
        if _expected_public_split(self.family) is not self.split:
            raise ValueError("family comparison split and family do not agree")
        for index, entry in enumerate(self.entries):
            family, parsed_index = parse_public_lineage_key(entry.lineage_registry_key)
            if family is not self.family or parsed_index != index:
                raise ValueError("family comparison entries are not canonical and complete")
            if any(
                delta.family is not self.family
                or delta.lineage_registry_key != entry.lineage_registry_key
                for delta in entry.semantic_projection.causal_deltas
            ):
                raise ValueError("family comparison causal delta coordinates do not agree")
        candidate_digests = tuple(entry.candidate_packet_digest for entry in self.entries)
        if len(set(candidate_digests)) != 30:
            raise ValueError("family comparison candidate packets must be unique")
        if not hmac.compare_digest(
            self.family_comparison_digest,
            family_comparison_digest(self),
        ):
            raise ValueError("family comparison digest does not match")
        return self


class PublicReviewDraft(_StrictModel):
    schema_version: Literal["state-decay-v2-public-review-draft/v1"] = (
        "state-decay-v2-public-review-draft/v1"
    )
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    split: BenchmarkSplit
    family: ScenarioFamily
    lineage_registry_key: PublicLineageKey
    candidate_packet_digest: Sha256Digest
    checklist_digest: Sha256Digest
    preview_digests: Annotated[tuple[Sha256Digest, ...], Field(min_length=5, max_length=5)]
    family_comparison_digest: Sha256Digest
    profile_catalog_digest: Sha256Digest
    generator_configuration_digest: Sha256Digest
    generator_algorithm_digest: Sha256Digest
    draft_digest: Sha256Digest

    @model_validator(mode="after")
    def draft_is_public_bound_and_self_attesting(self) -> Self:
        _require_public_coordinates(
            split=self.split,
            family=self.family,
            lineage_registry_key=self.lineage_registry_key,
            subject="review draft",
        )
        if self.checklist_digest != PUBLIC_REVIEW_CHECKLIST.checklist_digest:
            raise ValueError("review draft checklist does not match")
        if len(set(self.preview_digests)) != 5:
            raise ValueError("review draft preview digests must be unique")
        if not hmac.compare_digest(self.draft_digest, review_draft_digest(self)):
            raise ValueError("review draft digest does not match")
        return self


class PublicReviewSubmission(_StrictModel):
    schema_version: Literal["state-decay-v2-public-review-submission/v1"] = (
        "state-decay-v2-public-review-submission/v1"
    )
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    split: BenchmarkSplit
    family: ScenarioFamily
    lineage_registry_key: PublicLineageKey
    candidate_packet_digest: Sha256Digest
    draft_digest: Sha256Digest
    checklist_digest: Sha256Digest
    family_comparison_digest: Sha256Digest
    profile_catalog_digest: Sha256Digest
    generator_configuration_digest: Sha256Digest
    generator_algorithm_digest: Sha256Digest
    reviewer_id: ComponentIdentifier
    review_rationale: ReviewSafeText
    checklist_answers: Annotated[
        tuple[PublicReviewChecklistAnswer, ...],
        Field(min_length=7, max_length=7),
    ]
    decision: ReviewDecision
    supersedes_submission_digest: Sha256Digest | None = None
    submission_digest: Sha256Digest

    @model_validator(mode="after")
    def submission_is_consistent_bounded_and_self_attesting(self) -> Self:
        _require_public_coordinates(
            split=self.split,
            family=self.family,
            lineage_registry_key=self.lineage_registry_key,
            subject="review submission",
        )
        if self.checklist_digest != PUBLIC_REVIEW_CHECKLIST.checklist_digest:
            raise ValueError("review submission checklist does not match")
        if tuple(answer.item_id for answer in self.checklist_answers) != tuple(
            PublicReviewChecklistItemId
        ):
            raise ValueError("review submission checklist answers are not canonical")
        all_passed = all(
            answer.answer is PublicReviewAnswer.PASSED for answer in self.checklist_answers
        )
        if (self.decision is ReviewDecision.ACCEPTED and not all_passed) or (
            self.decision is ReviewDecision.REJECTED and all_passed
        ):
            raise ValueError("review submission decision is inconsistent with its answers")
        if self.supersedes_submission_digest == self.submission_digest:
            raise ValueError("review submission cannot supersede itself")
        if not hmac.compare_digest(
            self.submission_digest,
            review_submission_digest(self),
        ):
            raise ValueError("review submission digest does not match")
        if len(canonical_json(self)) + 1 > MAX_REVIEW_SUBMISSION_FILE_BYTES:
            raise ValueError("review submission exceeds its canonical file bound")
        return self


class PublicReviewEnvelope(_StrictModel):
    schema_version: Literal["state-decay-v2-public-review-envelope/v1"] = (
        "state-decay-v2-public-review-envelope/v1"
    )
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    split: BenchmarkSplit
    family: ScenarioFamily
    lineage_registry_key: PublicLineageKey
    candidate_packet_digest: Sha256Digest
    draft_digest: Sha256Digest
    checklist_digest: Sha256Digest
    family_comparison_digest: Sha256Digest
    profile_catalog_digest: Sha256Digest
    generator_configuration_digest: Sha256Digest
    generator_algorithm_digest: Sha256Digest
    submissions: Annotated[
        tuple[PublicReviewSubmission, ...],
        Field(min_length=1, max_length=MAX_REVIEW_CHAIN_LENGTH),
    ]
    review_record: LineageReviewRecord
    envelope_digest: Sha256Digest

    @model_validator(mode="after")
    def envelope_is_linear_projected_bounded_and_self_attesting(self) -> Self:
        _require_public_coordinates(
            split=self.split,
            family=self.family,
            lineage_registry_key=self.lineage_registry_key,
            subject="review envelope",
        )
        if self.checklist_digest != PUBLIC_REVIEW_CHECKLIST.checklist_digest:
            raise ValueError("review envelope checklist does not match")
        submission_digests = tuple(item.submission_digest for item in self.submissions)
        if len(set(submission_digests)) != len(submission_digests):
            raise ValueError("review envelope submission chain contains a cycle")
        if self.submissions[0].supersedes_submission_digest is not None:
            raise ValueError("review envelope submission chain must begin with a null predecessor")
        for previous, current in zip(self.submissions, self.submissions[1:], strict=False):
            if current.supersedes_submission_digest != previous.submission_digest:
                raise ValueError("review envelope submission chain is not contiguous")
        for submission in self.submissions:
            if (
                submission.split is not self.split
                or submission.family is not self.family
                or submission.lineage_registry_key != self.lineage_registry_key
                or submission.candidate_packet_digest != self.candidate_packet_digest
                or submission.checklist_digest != self.checklist_digest
                or submission.profile_catalog_digest != self.profile_catalog_digest
                or submission.generator_configuration_digest != self.generator_configuration_digest
                or submission.generator_algorithm_digest != self.generator_algorithm_digest
            ):
                raise ValueError("review envelope submission chain bindings do not agree")
        head = self.submissions[-1]
        if (
            head.draft_digest != self.draft_digest
            or head.family_comparison_digest != self.family_comparison_digest
        ):
            raise ValueError("review envelope head does not match current pack bindings")
        record = self.review_record
        if (
            record.split is not self.split
            or record.family is not self.family
            or record.boundary is not ReviewBoundary.PUBLIC
            or record.lineage_registry_key != self.lineage_registry_key
            or record.candidate_packet_digest != self.candidate_packet_digest
            or record.derivation_parent_keys
            or record.reviewer_id != head.reviewer_id
            or record.review_rationale != head.review_rationale
            or record.decision is not head.decision
        ):
            raise ValueError("review envelope record projection does not agree")
        try:
            validate_review_safe_text(record.semantic_rationale)
        except ValueError:
            raise ValueError("review envelope semantic rationale is not review-safe") from None
        if not hmac.compare_digest(self.envelope_digest, review_envelope_digest(self)):
            raise ValueError("review envelope digest does not match")
        if len(canonical_json(self)) > MAX_REVIEW_ENVELOPE_CANONICAL_BYTES:
            raise ValueError("review envelope exceeds its canonical object bound")
        return self


class PublicReviewPackBasename(StrEnum):
    CANDIDATES = "candidates.jsonl"
    DRAFTS = "drafts.jsonl"
    FAMILY_COMPARISONS = "family-comparisons.jsonl"
    CHECKLIST = "checklist.json"
    REVIEW_GUIDE = "review-guide.md"


def pack_child_digest(basename: PublicReviewPackBasename, content: bytes) -> str:
    if type(basename) is not PublicReviewPackBasename or type(content) is not bytes:
        raise ValueError("review pack child digest input is invalid")
    return length_prefixed_sha256(
        basename.value,
        content,
        domain=PACK_CHILD_DIGEST_DOMAIN,
    )


def accepted_envelope_registry_digest(content: bytes) -> str:
    if type(content) is not bytes:
        raise ValueError("accepted envelope registry digest input is invalid")
    return length_prefixed_sha256(
        ACCEPTED_ENVELOPE_REGISTRY_BASENAME,
        content,
        domain=ACCEPTED_ENVELOPE_REGISTRY_DIGEST_DOMAIN,
    )


class PublicReviewPackChild(_StrictModel):
    schema_version: Literal["state-decay-v2-public-review-pack-child/v1"] = (
        "state-decay-v2-public-review-pack-child/v1"
    )
    basename: PublicReviewPackBasename
    canonical_byte_count: Annotated[int, Field(ge=1, le=MAX_REVIEW_PACK_LARGE_CHILD_BYTES)]
    content_digest: Sha256Digest

    @model_validator(mode="after")
    def child_size_matches_its_role_limit(self) -> Self:
        maximum = (
            MAX_REVIEW_PACK_SMALL_CHILD_BYTES
            if self.basename
            in (PublicReviewPackBasename.CHECKLIST, PublicReviewPackBasename.REVIEW_GUIDE)
            else MAX_REVIEW_PACK_LARGE_CHILD_BYTES
        )
        if self.canonical_byte_count > maximum:
            raise ValueError("review pack child exceeds its role limit")
        return self


class PublicReviewPackManifest(_StrictModel):
    schema_version: Literal["state-decay-v2-public-review-pack/v1"] = (
        "state-decay-v2-public-review-pack/v1"
    )
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    generation_contract_digest: Sha256Digest
    lineage_review_protocol_digest: Sha256Digest
    generator_configuration_digest: Sha256Digest
    generator_algorithm_digest: Sha256Digest
    profile_catalog_digest: Sha256Digest
    candidate_registry_digest: Sha256Digest
    checklist_digest: Sha256Digest
    children: Annotated[
        tuple[PublicReviewPackChild, ...],
        Field(min_length=5, max_length=5),
    ]
    manifest_digest: Sha256Digest

    @model_validator(mode="after")
    def manifest_is_complete_bounded_and_self_attesting(self) -> Self:
        if self.generation_contract_digest != GENERATION_CONTRACT.contract_digest:
            raise ValueError("review pack generation contract does not match")
        if self.lineage_review_protocol_digest != LINEAGE_REVIEW_PROTOCOL.protocol_digest:
            raise ValueError("review pack lineage review protocol does not match")
        if self.checklist_digest != PUBLIC_REVIEW_CHECKLIST.checklist_digest:
            raise ValueError("review pack checklist does not match")
        if tuple(child.basename for child in self.children) != tuple(PublicReviewPackBasename):
            raise ValueError("review pack child order is not canonical")
        if not hmac.compare_digest(self.manifest_digest, pack_manifest_digest(self)):
            raise ValueError("review pack manifest digest does not match")
        manifest_file_bytes = len(canonical_json(self)) + 1
        if manifest_file_bytes > MAX_REVIEW_PACK_MANIFEST_FILE_BYTES:
            raise ValueError("review pack manifest exceeds its canonical file bound")
        if (
            manifest_file_bytes + sum(child.canonical_byte_count for child in self.children)
            > MAX_REVIEW_PACK_TOTAL_BYTES
        ):
            raise ValueError("review pack exceeds its total byte bound")
        return self


class PublicLineageReviewSubreport(_StrictModel):
    schema_version: Literal["state-decay-v2-public-lineage-review-subreport/v1"] = (
        "state-decay-v2-public-lineage-review-subreport/v1"
    )
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    scope: tuple[BenchmarkSplit, BenchmarkSplit]
    status: Literal["passed"] = "passed"
    record_count: Literal[180] = PUBLIC_REVIEW_RECORD_COUNT
    generation_contract_digest: Sha256Digest
    lineage_review_protocol_digest: Sha256Digest
    candidate_registry_digest: Sha256Digest
    accepted_envelope_registry_digest: Sha256Digest
    review_record_digests: Annotated[
        tuple[Sha256Digest, ...],
        Field(min_length=PUBLIC_REVIEW_RECORD_COUNT, max_length=PUBLIC_REVIEW_RECORD_COUNT),
    ]
    subreport_digest: Sha256Digest

    @model_validator(mode="after")
    def subreport_is_complete_unique_and_self_attesting(self) -> Self:
        if self.scope != (BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT):
            raise ValueError("public lineage review subreport scope is not canonical")
        if self.generation_contract_digest != GENERATION_CONTRACT.contract_digest:
            raise ValueError("public lineage review subreport generation contract does not match")
        if self.lineage_review_protocol_digest != LINEAGE_REVIEW_PROTOCOL.protocol_digest:
            raise ValueError("public lineage review subreport protocol does not match")
        if len(set(self.review_record_digests)) != PUBLIC_REVIEW_RECORD_COUNT:
            raise ValueError("public lineage review subreport record digests must be unique")
        if not hmac.compare_digest(self.subreport_digest, subreport_digest(self)):
            raise ValueError("public lineage review subreport digest does not match")
        return self


__all__ = [
    "ACCEPTED_ENVELOPE_REGISTRY_BASENAME",
    "ACCEPTED_ENVELOPE_REGISTRY_DIGEST_DOMAIN",
    "FAMILY_COMPARISON_DIGEST_DOMAIN",
    "MAX_ACCEPTED_ENVELOPE_REGISTRY_BYTES",
    "MAX_REVIEW_CHAIN_LENGTH",
    "MAX_REVIEW_ENVELOPE_CANONICAL_BYTES",
    "MAX_REVIEW_PACK_LARGE_CHILD_BYTES",
    "MAX_REVIEW_PACK_MANIFEST_FILE_BYTES",
    "MAX_REVIEW_PACK_SMALL_CHILD_BYTES",
    "MAX_REVIEW_PACK_TOTAL_BYTES",
    "MAX_REVIEW_SUBMISSION_FILE_BYTES",
    "PACK_CHILD_DIGEST_DOMAIN",
    "PACK_MANIFEST_DIGEST_DOMAIN",
    "PUBLIC_REVIEW_CHECKLIST",
    "PUBLIC_REVIEW_RECORD_COUNT",
    "REVIEW_CHECKLIST_DIGEST_DOMAIN",
    "REVIEW_DRAFT_DIGEST_DOMAIN",
    "REVIEW_ENVELOPE_DIGEST_DOMAIN",
    "REVIEW_SUBMISSION_DIGEST_DOMAIN",
    "SUBREPORT_DIGEST_DOMAIN",
    "PublicCandidateSemanticProjection",
    "PublicFamilyComparison",
    "PublicFamilyComparisonEntry",
    "PublicLineageReviewSubreport",
    "PublicReviewAnswer",
    "PublicReviewChecklist",
    "PublicReviewChecklistAnswer",
    "PublicReviewChecklistItem",
    "PublicReviewChecklistItemId",
    "PublicReviewDraft",
    "PublicReviewEnvelope",
    "PublicReviewPackBasename",
    "PublicReviewPackChild",
    "PublicReviewPackManifest",
    "PublicReviewSubmission",
    "accepted_envelope_registry_digest",
    "family_comparison_digest",
    "pack_child_digest",
    "pack_manifest_digest",
    "public_review_checklist_digest",
    "review_draft_digest",
    "review_envelope_digest",
    "review_submission_digest",
    "subreport_digest",
]
