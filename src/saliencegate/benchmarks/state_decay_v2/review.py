from __future__ import annotations

import weakref
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from itertools import islice
from typing import Annotated, Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from saliencegate.benchmarks.state_decay_v2.protocol import (
    LineageReviewRecord,
    ReviewBoundary,
    ReviewDecision,
    lineage_review_record_digest,
)
from saliencegate.benchmarks.state_decay_v2.public_contract import (
    PublicLineageCandidate,
    PublicLineageKey,
    PublicLineageRegistry,
    parse_public_lineage_key,
)
from saliencegate.benchmarks.state_decay_v2.review_contract import (
    MAX_REVIEW_CHAIN_LENGTH,
    PUBLIC_REVIEW_CHECKLIST,
    PUBLIC_REVIEW_RECORD_COUNT,
    PublicCandidateSemanticProjection,
    PublicFamilyComparison,
    PublicFamilyComparisonEntry,
    PublicReviewChecklistAnswer,
    PublicReviewDraft,
    PublicReviewEnvelope,
    PublicReviewSubmission,
    family_comparison_digest,
    review_draft_digest,
    review_envelope_digest,
    review_submission_digest,
)
from saliencegate.benchmarks.state_decay_v2.schema import (
    SUITE_ID,
    SUITE_VERSION,
    BenchmarkSplit,
    ScenarioFamily,
)
from saliencegate.domain import canonical_json
from saliencegate.domain.records import Sha256Digest


class PublicReviewErrorCode(StrEnum):
    MALFORMED_INPUT = "malformed-input"
    INCOMPLETE_MATERIALS = "incomplete-materials"
    BINDING_MISMATCH = "binding-mismatch"
    DIGEST_MISMATCH = "digest-mismatch"
    UNSAFE_TEXT = "unsafe-text"
    INCONSISTENT_REVIEW = "inconsistent-review"
    MISSING_PREDECESSOR = "missing-predecessor"
    FORK = "fork"
    CYCLE = "cycle"
    DUPLICATE = "duplicate"
    MULTIPLE_HEAD = "multiple-head"


_ERROR_MESSAGES = {
    PublicReviewErrorCode.MALFORMED_INPUT: "public review input is malformed",
    PublicReviewErrorCode.INCOMPLETE_MATERIALS: "public review materials are incomplete",
    PublicReviewErrorCode.BINDING_MISMATCH: "public review bindings do not agree",
    PublicReviewErrorCode.DIGEST_MISMATCH: "public review digest does not match",
    PublicReviewErrorCode.UNSAFE_TEXT: "public review text is unsafe",
    PublicReviewErrorCode.INCONSISTENT_REVIEW: "public review declaration is inconsistent",
    PublicReviewErrorCode.MISSING_PREDECESSOR: "public review predecessor is missing",
    PublicReviewErrorCode.FORK: "public review chain contains a fork",
    PublicReviewErrorCode.CYCLE: "public review chain contains a cycle",
    PublicReviewErrorCode.DUPLICATE: "public review input contains a duplicate",
    PublicReviewErrorCode.MULTIPLE_HEAD: "public review chain has multiple heads",
}


class PublicReviewError(ValueError):
    """A stable, value-free failure at the pure public-review boundary."""

    def __init__(self, code: PublicReviewErrorCode) -> None:
        self.code = code
        super().__init__(_ERROR_MESSAGES[code])


class PublicReviewProgressState(StrEnum):
    AMBIGUOUS = "ambiguous"
    STALE_COMPARISON = "stale-comparison"
    REJECTED = "rejected"
    MISSING = "missing"
    ACCEPTED = "accepted"


_PUBLIC_FAMILY_ORDER = tuple(ScenarioFamily)[:6]
_PUBLIC_FAMILY_SPLITS = {
    ScenarioFamily.FORGOTTEN_REQUIREMENT: BenchmarkSplit.TRAIN,
    ScenarioFamily.FAILED_PRIOR_ATTEMPT: BenchmarkSplit.TRAIN,
    ScenarioFamily.NEGLECTED_SUBGOAL: BenchmarkSplit.TRAIN,
    ScenarioFamily.STALE_MEMORY: BenchmarkSplit.TRAIN,
    ScenarioFamily.STABLE_ENVIRONMENT_FACT: BenchmarkSplit.DEVELOPMENT,
    ScenarioFamily.RETAINED_DIAGNOSIS: BenchmarkSplit.DEVELOPMENT,
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class PublicReviewCandidateProgress(_StrictModel):
    schema_version: Literal["state-decay-v2-public-review-candidate-progress/v1"] = (
        "state-decay-v2-public-review-candidate-progress/v1"
    )
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    split: BenchmarkSplit
    family: ScenarioFamily
    lineage_registry_key: PublicLineageKey
    candidate_packet_digest: Sha256Digest
    state: PublicReviewProgressState
    head_submission_digest: Sha256Digest | None
    envelope_digest: Sha256Digest | None

    @model_validator(mode="after")
    def progress_is_coordinate_bound(self) -> Self:
        family, _ = parse_public_lineage_key(self.lineage_registry_key)
        if family is not self.family or _PUBLIC_FAMILY_SPLITS.get(self.family) is not self.split:
            raise ValueError("public review progress coordinates do not agree")
        has_current_head = self.state in {
            PublicReviewProgressState.STALE_COMPARISON,
            PublicReviewProgressState.REJECTED,
            PublicReviewProgressState.ACCEPTED,
        }
        if has_current_head != (
            self.head_submission_digest is not None and self.envelope_digest is not None
        ):
            raise ValueError("public review progress head bindings do not agree")
        return self


class PublicReviewGateReport(_StrictModel):
    """Human progress only; this is deliberately not a readiness authority."""

    schema_version: Literal["state-decay-v2-public-review-gate-report/v1"] = (
        "state-decay-v2-public-review-gate-report/v1"
    )
    suite_id: Literal["state-decay-v2"] = SUITE_ID
    suite_version: Literal["v2"] = SUITE_VERSION
    candidate_registry_digest: Sha256Digest
    checklist_digest: Sha256Digest
    candidates: Annotated[
        tuple[PublicReviewCandidateProgress, ...],
        Field(min_length=180, max_length=180),
    ]
    ambiguous_count: Annotated[int, Field(ge=0, le=180)]
    stale_comparison_count: Annotated[int, Field(ge=0, le=180)]
    rejected_count: Annotated[int, Field(ge=0, le=180)]
    missing_count: Annotated[int, Field(ge=0, le=180)]
    accepted_count: Annotated[int, Field(ge=0, le=180)]
    progress_complete: bool

    @model_validator(mode="after")
    def report_is_complete_and_consistent(self) -> Self:
        keys = tuple(item.lineage_registry_key for item in self.candidates)
        coordinates = tuple(
            (item.family, parse_public_lineage_key(item.lineage_registry_key)[1])
            for item in self.candidates
        )
        expected_coordinates = tuple(
            (family, index) for family in _PUBLIC_FAMILY_ORDER for index in range(30)
        )
        if len(set(keys)) != 180 or coordinates != expected_coordinates:
            raise ValueError("public review progress candidates are not canonical")
        counts = Counter(item.state for item in self.candidates)
        expected_counts = (
            counts[PublicReviewProgressState.AMBIGUOUS],
            counts[PublicReviewProgressState.STALE_COMPARISON],
            counts[PublicReviewProgressState.REJECTED],
            counts[PublicReviewProgressState.MISSING],
            counts[PublicReviewProgressState.ACCEPTED],
        )
        if expected_counts != (
            self.ambiguous_count,
            self.stale_comparison_count,
            self.rejected_count,
            self.missing_count,
            self.accepted_count,
        ):
            raise ValueError("public review progress counts do not agree")
        if self.progress_complete != (self.accepted_count == 180):
            raise ValueError("public review progress completion does not agree")
        return self


_ModelT = TypeVar("_ModelT", bound=BaseModel)
_KeyT = TypeVar("_KeyT")
_ValueT = TypeVar("_ValueT")

_REGISTRY_CACHE: dict[
    int,
    tuple[
        weakref.ReferenceType[PublicLineageRegistry],
        bytes,
        PublicLineageRegistry,
    ],
] = {}
_COMPARISON_CACHE: dict[str, tuple[PublicFamilyComparison, ...]] = {}
_DRAFT_CACHE: dict[
    tuple[str, tuple[str, ...]],
    tuple[PublicReviewDraft, ...],
] = {}
_MAX_MATERIAL_CACHE_ENTRIES = 8
_MAX_LOCAL_REVIEW_ENVELOPES = PUBLIC_REVIEW_RECORD_COUNT * 64


def _validation_error_code(error: ValidationError) -> PublicReviewErrorCode:
    details = error.errors(include_url=False, include_input=False)
    messages = tuple(str(item.get("msg", "")).casefold() for item in details)
    if any("digest" in message for message in messages):
        return PublicReviewErrorCode.DIGEST_MISMATCH
    if any("review-safe" in message for message in messages):
        return PublicReviewErrorCode.UNSAFE_TEXT
    if any(
        "checklist answers" in message or "decision is inconsistent" in message
        for message in messages
    ):
        return PublicReviewErrorCode.INCONSISTENT_REVIEW
    return PublicReviewErrorCode.MALFORMED_INPUT


def _revalidate(model_type: type[_ModelT], value: object) -> _ModelT:
    try:
        serializable = (
            value.model_dump(mode="json", warnings="error")
            if isinstance(value, BaseModel)
            else value
        )
        return model_type.model_validate_json(canonical_json(serializable))
    except ValidationError as error:
        raise PublicReviewError(_validation_error_code(error)) from None
    except Exception:
        raise PublicReviewError(PublicReviewErrorCode.MALFORMED_INPUT) from None


def _sequence(
    value: object,
    *,
    minimum_length: int,
    maximum_length: int,
) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise PublicReviewError(PublicReviewErrorCode.MALFORMED_INPUT)
    try:
        declared_length = len(value)
        if not minimum_length <= declared_length <= maximum_length:
            raise PublicReviewError(PublicReviewErrorCode.INCOMPLETE_MATERIALS)
        copied = tuple(islice(iter(value), maximum_length + 1))
        if len(copied) != declared_length:
            raise PublicReviewError(PublicReviewErrorCode.MALFORMED_INPUT)
        return copied
    except PublicReviewError:
        raise
    except Exception:
        raise PublicReviewError(PublicReviewErrorCode.MALFORMED_INPUT) from None


def _same(left: object, right: object) -> bool:
    try:
        return canonical_json(left) == canonical_json(right)
    except Exception:
        raise PublicReviewError(PublicReviewErrorCode.MALFORMED_INPUT) from None


def _registry_snapshot(registry: PublicLineageRegistry) -> bytes:
    try:
        return canonical_json(registry.model_dump(mode="json", warnings="error"))
    except Exception:
        raise PublicReviewError(PublicReviewErrorCode.MALFORMED_INPUT) from None


def _cache_registry(
    source: PublicLineageRegistry,
    checked: PublicLineageRegistry,
    snapshot: bytes,
) -> None:
    identity = id(source)

    def discard(reference: weakref.ReferenceType[PublicLineageRegistry]) -> None:
        cached = _REGISTRY_CACHE.get(identity)
        if cached is not None and cached[0] is reference:
            _REGISTRY_CACHE.pop(identity, None)

    reference = weakref.ref(source, discard)
    _REGISTRY_CACHE[identity] = (reference, snapshot, checked)


def _checked_registry(value: object) -> PublicLineageRegistry:
    if type(value) is PublicLineageRegistry:
        snapshot = _registry_snapshot(value)
        cached = _REGISTRY_CACHE.get(id(value))
        if cached is not None and cached[0]() is value and cached[1] == snapshot:
            return cached[2]
        try:
            checked = PublicLineageRegistry.model_validate_json(snapshot)
        except ValidationError as error:
            raise PublicReviewError(_validation_error_code(error)) from None
        except Exception:
            raise PublicReviewError(PublicReviewErrorCode.MALFORMED_INPUT) from None
        _cache_registry(value, checked, snapshot)
        return checked
    checked = _revalidate(PublicLineageRegistry, value)
    return checked


def _candidate_map(
    registry: PublicLineageRegistry,
) -> dict[str, PublicLineageCandidate]:
    return {candidate.lineage_registry_key: candidate for candidate in registry.candidates}


def _build_family_comparison(
    candidates: tuple[PublicLineageCandidate, ...],
) -> PublicFamilyComparison:
    family = candidates[0].family
    split = candidates[0].split
    entries = tuple(
        PublicFamilyComparisonEntry(
            lineage_registry_key=candidate.lineage_registry_key,
            candidate_packet_digest=candidate.candidate_packet_digest,
            semantic_rationale=candidate.semantic_rationale,
            semantic_projection=PublicCandidateSemanticProjection(
                task_template=candidate.task_template,
                transition_graph=candidate.transition_graph,
                evidence_topology=candidate.evidence_topology,
                failure_mechanism=candidate.failure_mechanism,
                semantic_signature=candidate.semantic_signature,
                causal_deltas=candidate.causal_deltas,
            ),
            transition_graph_digest=candidate.transition_graph.transition_graph_digest,
            evidence_topology_digest=candidate.evidence_topology.evidence_topology_digest,
            failure_mechanism_id=candidate.failure_mechanism.failure_mechanism_id,
            semantic_signature_digest=candidate.semantic_signature.semantic_signature_digest,
            preview_digests=tuple(preview.preview_digest for preview in candidate.previews),
        )
        for candidate in candidates
    )
    payload: dict[str, object] = {
        "schema_version": "state-decay-v2-public-review-family-comparison/v1",
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "split": split,
        "family": family,
        "entries": entries,
    }
    payload["family_comparison_digest"] = family_comparison_digest(payload)
    return PublicFamilyComparison.model_validate(payload)


def _remember_materials(
    cache: dict[_KeyT, _ValueT],
    key: _KeyT,
    value: _ValueT,
) -> None:
    if key not in cache and len(cache) >= _MAX_MATERIAL_CACHE_ENTRIES:
        cache.pop(next(iter(cache)))
    cache[key] = value


def _build_family_comparisons_checked(
    registry: PublicLineageRegistry,
) -> tuple[PublicFamilyComparison, ...]:
    cached = _COMPARISON_CACHE.get(registry.registry_digest)
    if cached is not None:
        return cached
    families = tuple(dict.fromkeys(candidate.family for candidate in registry.candidates))
    built = tuple(
        _build_family_comparison(
            tuple(candidate for candidate in registry.candidates if candidate.family is family)
        )
        for family in families
    )
    _remember_materials(_COMPARISON_CACHE, registry.registry_digest, built)
    return built


def build_public_family_comparisons(
    *,
    registry: PublicLineageRegistry,
) -> tuple[PublicFamilyComparison, ...]:
    checked = _checked_registry(registry)
    try:
        return tuple(
            _revalidate(PublicFamilyComparison, item)
            for item in _build_family_comparisons_checked(checked)
        )
    except PublicReviewError:
        raise
    except Exception:
        raise PublicReviewError(PublicReviewErrorCode.MALFORMED_INPUT) from None


def _checked_comparisons(
    registry: PublicLineageRegistry,
    comparisons: object,
) -> tuple[PublicFamilyComparison, ...]:
    expected = _build_family_comparisons_checked(registry)
    raw = _sequence(
        comparisons,
        minimum_length=len(expected),
        maximum_length=len(expected),
    )
    supplied = tuple(_revalidate(PublicFamilyComparison, item) for item in raw)
    if not _same(supplied, expected):
        raise PublicReviewError(PublicReviewErrorCode.BINDING_MISMATCH)
    return supplied


def _build_draft(
    *,
    registry: PublicLineageRegistry,
    candidate: PublicLineageCandidate,
    comparison: PublicFamilyComparison,
) -> PublicReviewDraft:
    payload: dict[str, object] = {
        "schema_version": "state-decay-v2-public-review-draft/v1",
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "split": candidate.split,
        "family": candidate.family,
        "lineage_registry_key": candidate.lineage_registry_key,
        "candidate_packet_digest": candidate.candidate_packet_digest,
        "checklist_digest": PUBLIC_REVIEW_CHECKLIST.checklist_digest,
        "preview_digests": tuple(preview.preview_digest for preview in candidate.previews),
        "family_comparison_digest": comparison.family_comparison_digest,
        "profile_catalog_digest": registry.profile_catalog.catalog_digest,
        "generator_configuration_digest": registry.generator_configuration_digest,
        "generator_algorithm_digest": registry.generator_algorithm_digest,
    }
    payload["draft_digest"] = review_draft_digest(payload)
    return PublicReviewDraft.model_validate(payload)


def _build_drafts_checked(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
) -> tuple[PublicReviewDraft, ...]:
    key = (
        registry.registry_digest,
        tuple(item.family_comparison_digest for item in comparisons),
    )
    cached = _DRAFT_CACHE.get(key)
    if cached is not None:
        return cached
    by_family = {comparison.family: comparison for comparison in comparisons}
    built = tuple(
        _build_draft(
            registry=registry,
            candidate=candidate,
            comparison=by_family[candidate.family],
        )
        for candidate in registry.candidates
    )
    _remember_materials(_DRAFT_CACHE, key, built)
    return built


def build_public_review_drafts(
    *,
    registry: PublicLineageRegistry,
    comparisons: Sequence[PublicFamilyComparison],
) -> tuple[PublicReviewDraft, ...]:
    checked_registry = _checked_registry(registry)
    checked_comparisons = _checked_comparisons(checked_registry, comparisons)
    try:
        return tuple(
            _revalidate(PublicReviewDraft, item)
            for item in _build_drafts_checked(checked_registry, checked_comparisons)
        )
    except PublicReviewError:
        raise
    except Exception:
        raise PublicReviewError(PublicReviewErrorCode.MALFORMED_INPUT) from None


def _checked_drafts(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: object,
) -> tuple[PublicReviewDraft, ...]:
    expected = _build_drafts_checked(registry, comparisons)
    raw = _sequence(
        drafts,
        minimum_length=len(expected),
        maximum_length=len(expected),
    )
    supplied = tuple(_revalidate(PublicReviewDraft, item) for item in raw)
    if not _same(supplied, expected):
        raise PublicReviewError(PublicReviewErrorCode.BINDING_MISMATCH)
    return supplied


def build_public_review_submission(
    *,
    draft: PublicReviewDraft,
    reviewer_id: str,
    review_rationale: str,
    checklist_answers: Sequence[PublicReviewChecklistAnswer],
    decision: ReviewDecision,
    supersedes_submission_digest: str | None,
) -> PublicReviewSubmission:
    checked_draft = _revalidate(PublicReviewDraft, draft)
    raw_answers = _sequence(
        checklist_answers,
        minimum_length=len(PUBLIC_REVIEW_CHECKLIST.items),
        maximum_length=len(PUBLIC_REVIEW_CHECKLIST.items),
    )
    answers = tuple(_revalidate(PublicReviewChecklistAnswer, answer) for answer in raw_answers)
    payload: dict[str, object] = {
        "schema_version": "state-decay-v2-public-review-submission/v1",
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "split": checked_draft.split,
        "family": checked_draft.family,
        "lineage_registry_key": checked_draft.lineage_registry_key,
        "candidate_packet_digest": checked_draft.candidate_packet_digest,
        "draft_digest": checked_draft.draft_digest,
        "checklist_digest": checked_draft.checklist_digest,
        "family_comparison_digest": checked_draft.family_comparison_digest,
        "profile_catalog_digest": checked_draft.profile_catalog_digest,
        "generator_configuration_digest": checked_draft.generator_configuration_digest,
        "generator_algorithm_digest": checked_draft.generator_algorithm_digest,
        "reviewer_id": reviewer_id,
        "review_rationale": review_rationale,
        "checklist_answers": answers,
        "decision": decision,
        "supersedes_submission_digest": supersedes_submission_digest,
    }
    payload["submission_digest"] = review_submission_digest(payload)
    return _revalidate(PublicReviewSubmission, payload)


def _submission_bindings(submission: PublicReviewSubmission) -> tuple[object, ...]:
    return (
        submission.split,
        submission.family,
        submission.lineage_registry_key,
        submission.candidate_packet_digest,
        submission.checklist_digest,
        submission.profile_catalog_digest,
        submission.generator_configuration_digest,
        submission.generator_algorithm_digest,
    )


def _order_submission_chain(submissions: object) -> tuple[PublicReviewSubmission, ...]:
    raw = _sequence(
        submissions,
        minimum_length=1,
        maximum_length=MAX_REVIEW_CHAIN_LENGTH,
    )
    checked = tuple(_revalidate(PublicReviewSubmission, item) for item in raw)
    digests = tuple(item.submission_digest for item in checked)
    if len(set(digests)) != len(digests):
        raise PublicReviewError(PublicReviewErrorCode.DUPLICATE)
    root_bindings = _submission_bindings(checked[0])
    if any(_submission_bindings(item) != root_bindings for item in checked[1:]):
        raise PublicReviewError(PublicReviewErrorCode.BINDING_MISMATCH)

    by_digest = {item.submission_digest: item for item in checked}
    children: dict[str, PublicReviewSubmission] = {}
    for item in checked:
        predecessor = item.supersedes_submission_digest
        if predecessor is None:
            continue
        if predecessor not in by_digest:
            raise PublicReviewError(PublicReviewErrorCode.MISSING_PREDECESSOR)
        if predecessor in children:
            raise PublicReviewError(PublicReviewErrorCode.FORK)
        children[predecessor] = item

    roots = tuple(item for item in checked if item.supersedes_submission_digest is None)
    if not roots:
        raise PublicReviewError(PublicReviewErrorCode.CYCLE)
    if len(roots) != 1:
        raise PublicReviewError(PublicReviewErrorCode.MULTIPLE_HEAD)

    ordered: list[PublicReviewSubmission] = []
    seen: set[str] = set()
    current = roots[0]
    while True:
        if current.submission_digest in seen:
            raise PublicReviewError(PublicReviewErrorCode.CYCLE)
        seen.add(current.submission_digest)
        ordered.append(current)
        successor = children.get(current.submission_digest)
        if successor is None:
            break
        current = successor
    if len(ordered) != len(checked):
        raise PublicReviewError(PublicReviewErrorCode.CYCLE)
    return tuple(ordered)


def validate_public_review_submission_chain(
    *,
    submissions: Sequence[PublicReviewSubmission],
) -> tuple[PublicReviewSubmission, ...]:
    """Return one revalidated oldest-to-head immutable review chain."""

    return _order_submission_chain(submissions)


def _candidate_for_draft(
    registry: PublicLineageRegistry,
    draft: PublicReviewDraft,
) -> PublicLineageCandidate:
    candidate = _candidate_map(registry).get(draft.lineage_registry_key)
    if candidate is None or (
        candidate.split is not draft.split
        or candidate.family is not draft.family
        or candidate.candidate_packet_digest != draft.candidate_packet_digest
    ):
        raise PublicReviewError(PublicReviewErrorCode.BINDING_MISMATCH)
    return candidate


def _build_review_record(
    *,
    candidate: PublicLineageCandidate,
    head: PublicReviewSubmission,
) -> LineageReviewRecord:
    payload: dict[str, object] = {
        "schema_version": "state-decay-v2-lineage-review-record/v1",
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "split": candidate.split,
        "family": candidate.family,
        "boundary": ReviewBoundary.PUBLIC,
        "lineage_registry_key": candidate.lineage_registry_key,
        "candidate_packet_digest": candidate.candidate_packet_digest,
        "independent_seed_commitment_digest": candidate.independent_seed_commitment_digest,
        "transition_graph_digest": candidate.transition_graph.transition_graph_digest,
        "evidence_topology_digest": candidate.evidence_topology.evidence_topology_digest,
        "failure_mechanism_id": candidate.failure_mechanism.failure_mechanism_id,
        "semantic_signature_digest": candidate.semantic_signature.semantic_signature_digest,
        "derivation_parent_keys": candidate.derivation_parent_keys,
        "semantic_rationale": candidate.semantic_rationale,
        "reviewer_id": head.reviewer_id,
        "review_rationale": head.review_rationale,
        "decision": head.decision,
    }
    payload["review_digest"] = lineage_review_record_digest(payload)
    return LineageReviewRecord.model_validate(payload)


def build_public_review_head_envelope(
    *,
    registry: PublicLineageRegistry,
    submissions: Sequence[PublicReviewSubmission],
) -> PublicReviewEnvelope:
    """Project a current-candidate head, including a stale comparison, into an envelope."""

    checked_registry = _checked_registry(registry)
    chain = _order_submission_chain(submissions)
    head = chain[-1]
    candidate = _candidate_map(checked_registry).get(head.lineage_registry_key)
    if candidate is None or (
        candidate.split is not head.split
        or candidate.family is not head.family
        or candidate.candidate_packet_digest != head.candidate_packet_digest
        or _submission_bindings(head)
        != (
            candidate.split,
            candidate.family,
            candidate.lineage_registry_key,
            candidate.candidate_packet_digest,
            PUBLIC_REVIEW_CHECKLIST.checklist_digest,
            checked_registry.profile_catalog.catalog_digest,
            checked_registry.generator_configuration_digest,
            checked_registry.generator_algorithm_digest,
        )
    ):
        raise PublicReviewError(PublicReviewErrorCode.BINDING_MISMATCH)

    record = _build_review_record(candidate=candidate, head=head)
    payload: dict[str, object] = {
        "schema_version": "state-decay-v2-public-review-envelope/v1",
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "split": candidate.split,
        "family": candidate.family,
        "lineage_registry_key": candidate.lineage_registry_key,
        "candidate_packet_digest": candidate.candidate_packet_digest,
        "draft_digest": head.draft_digest,
        "checklist_digest": head.checklist_digest,
        "family_comparison_digest": head.family_comparison_digest,
        "profile_catalog_digest": head.profile_catalog_digest,
        "generator_configuration_digest": head.generator_configuration_digest,
        "generator_algorithm_digest": head.generator_algorithm_digest,
        "submissions": chain,
        "review_record": record,
    }
    payload["envelope_digest"] = review_envelope_digest(payload)
    return _revalidate(PublicReviewEnvelope, payload)


def _expected_comparison_for_family(
    registry: PublicLineageRegistry,
    comparison: object,
) -> PublicFamilyComparison:
    checked = _revalidate(PublicFamilyComparison, comparison)
    expected = next(
        (
            item
            for item in _build_family_comparisons_checked(registry)
            if item.family is checked.family
        ),
        None,
    )
    if expected is None or not _same(checked, expected):
        raise PublicReviewError(PublicReviewErrorCode.BINDING_MISMATCH)
    return checked


def _expected_draft_for_candidate(
    *,
    registry: PublicLineageRegistry,
    candidate: PublicLineageCandidate,
    comparison: PublicFamilyComparison,
    draft: object,
) -> PublicReviewDraft:
    checked = _revalidate(PublicReviewDraft, draft)
    expected = _build_draft(
        registry=registry,
        candidate=candidate,
        comparison=comparison,
    )
    if not _same(checked, expected):
        raise PublicReviewError(PublicReviewErrorCode.BINDING_MISMATCH)
    return checked


def build_public_review_envelope(
    *,
    registry: PublicLineageRegistry,
    draft: PublicReviewDraft,
    family_comparison: PublicFamilyComparison,
    submissions: Sequence[PublicReviewSubmission],
) -> PublicReviewEnvelope:
    checked_registry = _checked_registry(registry)
    candidate = _candidate_for_draft(
        checked_registry,
        _revalidate(PublicReviewDraft, draft),
    )
    comparison = _expected_comparison_for_family(checked_registry, family_comparison)
    if comparison.family is not candidate.family:
        raise PublicReviewError(PublicReviewErrorCode.BINDING_MISMATCH)
    checked_draft = _expected_draft_for_candidate(
        registry=checked_registry,
        candidate=candidate,
        comparison=comparison,
        draft=draft,
    )
    chain = _order_submission_chain(submissions)
    head = chain[-1]
    if (
        _submission_bindings(head)
        != (
            candidate.split,
            candidate.family,
            candidate.lineage_registry_key,
            candidate.candidate_packet_digest,
            PUBLIC_REVIEW_CHECKLIST.checklist_digest,
            checked_registry.profile_catalog.catalog_digest,
            checked_registry.generator_configuration_digest,
            checked_registry.generator_algorithm_digest,
        )
        or head.draft_digest != checked_draft.draft_digest
        or head.family_comparison_digest != comparison.family_comparison_digest
    ):
        raise PublicReviewError(PublicReviewErrorCode.BINDING_MISMATCH)

    record = _build_review_record(candidate=candidate, head=head)
    payload: dict[str, object] = {
        "schema_version": "state-decay-v2-public-review-envelope/v1",
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "split": candidate.split,
        "family": candidate.family,
        "lineage_registry_key": candidate.lineage_registry_key,
        "candidate_packet_digest": candidate.candidate_packet_digest,
        "draft_digest": checked_draft.draft_digest,
        "checklist_digest": PUBLIC_REVIEW_CHECKLIST.checklist_digest,
        "family_comparison_digest": comparison.family_comparison_digest,
        "profile_catalog_digest": checked_registry.profile_catalog.catalog_digest,
        "generator_configuration_digest": checked_registry.generator_configuration_digest,
        "generator_algorithm_digest": checked_registry.generator_algorithm_digest,
        "submissions": chain,
        "review_record": record,
    }
    payload["envelope_digest"] = review_envelope_digest(payload)
    return _revalidate(PublicReviewEnvelope, payload)


def _record_matches_candidate(
    envelope: PublicReviewEnvelope,
    candidate: PublicLineageCandidate,
) -> bool:
    record = envelope.review_record
    return (
        record.split is candidate.split
        and record.family is candidate.family
        and record.boundary is ReviewBoundary.PUBLIC
        and record.lineage_registry_key == candidate.lineage_registry_key
        and record.candidate_packet_digest == candidate.candidate_packet_digest
        and record.independent_seed_commitment_digest
        == candidate.independent_seed_commitment_digest
        and record.transition_graph_digest == candidate.transition_graph.transition_graph_digest
        and record.evidence_topology_digest == candidate.evidence_topology.evidence_topology_digest
        and record.failure_mechanism_id == candidate.failure_mechanism.failure_mechanism_id
        and record.semantic_signature_digest
        == candidate.semantic_signature.semantic_signature_digest
        and record.derivation_parent_keys == candidate.derivation_parent_keys
        and record.semantic_rationale == candidate.semantic_rationale
    )


def _maximal_current_envelope(
    envelopes: tuple[PublicReviewEnvelope, ...],
) -> PublicReviewEnvelope | None:
    if len(envelopes) == 1:
        return envelopes[0]
    chains = tuple(
        tuple(item.submission_digest for item in envelope.submissions) for envelope in envelopes
    )
    maximum_length = max(len(chain) for chain in chains)
    maximal_indexes = tuple(
        index for index, chain in enumerate(chains) if len(chain) == maximum_length
    )
    if len(maximal_indexes) != 1:
        return None
    maximal = chains[maximal_indexes[0]]
    if any(chain != maximal[: len(chain)] for chain in chains):
        return None
    return envelopes[maximal_indexes[0]]


def evaluate_public_review_gate(
    *,
    registry: PublicLineageRegistry,
    comparisons: Sequence[PublicFamilyComparison],
    drafts: Sequence[PublicReviewDraft],
    envelopes: Sequence[PublicReviewEnvelope],
) -> PublicReviewGateReport:
    checked_registry = _checked_registry(registry)
    checked_comparisons = _checked_comparisons(checked_registry, comparisons)
    checked_drafts = _checked_drafts(
        checked_registry,
        checked_comparisons,
        drafts,
    )
    raw_envelopes = _sequence(
        envelopes,
        minimum_length=0,
        maximum_length=_MAX_LOCAL_REVIEW_ENVELOPES,
    )
    checked_envelopes = tuple(_revalidate(PublicReviewEnvelope, item) for item in raw_envelopes)
    envelope_digests = tuple(item.envelope_digest for item in checked_envelopes)
    if len(set(envelope_digests)) != len(envelope_digests):
        raise PublicReviewError(PublicReviewErrorCode.DUPLICATE)

    candidate_by_key = _candidate_map(checked_registry)
    draft_by_key = {item.lineage_registry_key: item for item in checked_drafts}
    comparison_by_family = {item.family: item for item in checked_comparisons}
    envelope_groups: dict[str, dict[str, list[PublicReviewEnvelope]]] = {
        key: {} for key in candidate_by_key
    }
    for envelope in checked_envelopes:
        candidate = candidate_by_key.get(envelope.lineage_registry_key)
        if candidate is None:
            raise PublicReviewError(PublicReviewErrorCode.BINDING_MISMATCH)
        if (
            envelope.profile_catalog_digest != checked_registry.profile_catalog.catalog_digest
            or envelope.generator_configuration_digest
            != checked_registry.generator_configuration_digest
            or envelope.generator_algorithm_digest != checked_registry.generator_algorithm_digest
            or envelope.checklist_digest != PUBLIC_REVIEW_CHECKLIST.checklist_digest
        ):
            raise PublicReviewError(PublicReviewErrorCode.BINDING_MISMATCH)
        envelope_groups[candidate.lineage_registry_key].setdefault(
            envelope.candidate_packet_digest,
            [],
        ).append(envelope)
        if (
            envelope.candidate_packet_digest == candidate.candidate_packet_digest
            and not _record_matches_candidate(envelope, candidate)
        ):
            raise PublicReviewError(PublicReviewErrorCode.BINDING_MISMATCH)

    ambiguous_keys = {
        lineage_key
        for lineage_key, revisions in envelope_groups.items()
        if any(
            _maximal_current_envelope(tuple(revision_envelopes)) is None
            for revision_envelopes in revisions.values()
        )
    }

    progress: list[PublicReviewCandidateProgress] = []
    for candidate in checked_registry.candidates:
        candidates = tuple(
            envelope_groups[candidate.lineage_registry_key].get(
                candidate.candidate_packet_digest,
                (),
            )
        )
        current_envelope = _maximal_current_envelope(candidates) if candidates else None
        if candidate.lineage_registry_key in ambiguous_keys:
            state = PublicReviewProgressState.AMBIGUOUS
            head_digest = None
            envelope_digest_value = None
        elif current_envelope is None:
            state = PublicReviewProgressState.MISSING
            head_digest = None
            envelope_digest_value = None
        else:
            current_draft = draft_by_key[candidate.lineage_registry_key]
            current_comparison = comparison_by_family[candidate.family]
            if (
                current_envelope.draft_digest != current_draft.draft_digest
                or current_envelope.family_comparison_digest
                != current_comparison.family_comparison_digest
            ):
                state = PublicReviewProgressState.STALE_COMPARISON
            elif current_envelope.submissions[-1].decision is ReviewDecision.REJECTED:
                state = PublicReviewProgressState.REJECTED
            else:
                state = PublicReviewProgressState.ACCEPTED
            head_digest = current_envelope.submissions[-1].submission_digest
            envelope_digest_value = current_envelope.envelope_digest
        progress.append(
            PublicReviewCandidateProgress(
                split=candidate.split,
                family=candidate.family,
                lineage_registry_key=candidate.lineage_registry_key,
                candidate_packet_digest=candidate.candidate_packet_digest,
                state=state,
                head_submission_digest=head_digest,
                envelope_digest=envelope_digest_value,
            )
        )

    counts = Counter(item.state for item in progress)
    return PublicReviewGateReport(
        candidate_registry_digest=checked_registry.registry_digest,
        checklist_digest=PUBLIC_REVIEW_CHECKLIST.checklist_digest,
        candidates=tuple(progress),
        ambiguous_count=counts[PublicReviewProgressState.AMBIGUOUS],
        stale_comparison_count=counts[PublicReviewProgressState.STALE_COMPARISON],
        rejected_count=counts[PublicReviewProgressState.REJECTED],
        missing_count=counts[PublicReviewProgressState.MISSING],
        accepted_count=counts[PublicReviewProgressState.ACCEPTED],
        progress_complete=counts[PublicReviewProgressState.ACCEPTED] == 180,
    )


__all__ = [
    "PublicReviewCandidateProgress",
    "PublicReviewError",
    "PublicReviewErrorCode",
    "PublicReviewGateReport",
    "PublicReviewProgressState",
    "build_public_family_comparisons",
    "build_public_review_drafts",
    "build_public_review_envelope",
    "build_public_review_head_envelope",
    "build_public_review_submission",
    "evaluate_public_review_gate",
    "validate_public_review_submission_chain",
]
