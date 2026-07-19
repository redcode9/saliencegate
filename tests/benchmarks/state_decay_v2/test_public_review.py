from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Final, cast

import pytest
from pydantic import ValidationError

import saliencegate.benchmarks.state_decay_v2.config as config_module
import saliencegate.benchmarks.state_decay_v2.review as review_module
from saliencegate.benchmarks.state_decay_v2.generation_authority import (
    PublicGenerationAuthorityError,
    require_public_generation_authority,
)
from saliencegate.benchmarks.state_decay_v2.generator import generate_public_scenarios
from saliencegate.benchmarks.state_decay_v2.protocol import (
    LineageReviewRecord,
    ReviewBoundary,
    ReviewDecision,
    lineage_review_record_digest,
)
from saliencegate.benchmarks.state_decay_v2.public_contract import (
    PublicLineageCandidate,
    PublicLineageRegistry,
    candidate_packet_digest,
    candidate_registry_digest,
)
from saliencegate.benchmarks.state_decay_v2.review import (
    PublicReviewCandidateProgress,
    PublicReviewError,
    PublicReviewErrorCode,
    PublicReviewGateReport,
    PublicReviewProgressState,
    build_public_family_comparisons,
    build_public_review_drafts,
    build_public_review_envelope,
    build_public_review_submission,
    evaluate_public_review_gate,
)
from saliencegate.benchmarks.state_decay_v2.review_contract import (
    PUBLIC_REVIEW_CHECKLIST,
    PublicFamilyComparison,
    PublicLineageReviewSubreport,
    PublicReviewAnswer,
    PublicReviewChecklistAnswer,
    PublicReviewChecklistItemId,
    PublicReviewDraft,
    PublicReviewEnvelope,
    PublicReviewSubmission,
    review_envelope_digest,
    review_submission_digest,
)
from saliencegate.benchmarks.state_decay_v2.schema import ScenarioFamily
from saliencegate.benchmarks.state_decay_v2.templates import PUBLIC_LINEAGE_REGISTRY
from saliencegate.domain import canonical_json

_REVIEWER_ID: Final = "synthetic-public-reviewer"
_REVIEW_RATIONALE: Final = "Synthetic explicit rationale for public review workflow tests."


@pytest.fixture(scope="module")
def registry() -> PublicLineageRegistry:
    return PUBLIC_LINEAGE_REGISTRY


@pytest.fixture(scope="module")
def comparisons(
    registry: PublicLineageRegistry,
) -> tuple[PublicFamilyComparison, ...]:
    return build_public_family_comparisons(registry=registry)


@pytest.fixture(scope="module")
def drafts(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
) -> tuple[PublicReviewDraft, ...]:
    return build_public_review_drafts(registry=registry, comparisons=comparisons)


def _answers(*failed_indexes: int) -> tuple[PublicReviewChecklistAnswer, ...]:
    failed = frozenset(failed_indexes)
    return tuple(
        PublicReviewChecklistAnswer(
            item_id=item_id,
            answer=(PublicReviewAnswer.FAILED if index in failed else PublicReviewAnswer.PASSED),
        )
        for index, item_id in enumerate(PublicReviewChecklistItemId)
    )


def _draft_by_key(
    drafts: tuple[PublicReviewDraft, ...],
) -> dict[str, PublicReviewDraft]:
    return {draft.lineage_registry_key: draft for draft in drafts}


def _comparison_by_family(
    comparisons: tuple[PublicFamilyComparison, ...],
) -> dict[ScenarioFamily, PublicFamilyComparison]:
    return {comparison.family: comparison for comparison in comparisons}


def _candidate_by_key(registry: PublicLineageRegistry) -> dict[str, PublicLineageCandidate]:
    return {candidate.lineage_registry_key: candidate for candidate in registry.candidates}


def _submission(
    draft: PublicReviewDraft,
    *,
    decision: ReviewDecision = ReviewDecision.ACCEPTED,
    failed_indexes: tuple[int, ...] = (),
    predecessor: str | None = None,
    reviewer_id: str = _REVIEWER_ID,
    rationale: str = _REVIEW_RATIONALE,
) -> PublicReviewSubmission:
    return build_public_review_submission(
        draft=draft,
        reviewer_id=reviewer_id,
        review_rationale=rationale,
        checklist_answers=_answers(*failed_indexes),
        decision=decision,
        supersedes_submission_digest=predecessor,
    )


def _envelope(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    draft: PublicReviewDraft,
    submissions: tuple[PublicReviewSubmission, ...],
) -> PublicReviewEnvelope:
    comparison = _comparison_by_family(comparisons)[draft.family]
    return build_public_review_envelope(
        registry=registry,
        draft=draft,
        family_comparison=comparison,
        submissions=submissions,
    )


def _direct_test_envelope(
    registry: PublicLineageRegistry,
    draft: PublicReviewDraft,
    submissions: tuple[PublicReviewSubmission, ...],
) -> PublicReviewEnvelope:
    """Materialize already-tested canonical fields without repeated registry reloads."""

    candidate = _candidate_by_key(registry)[draft.lineage_registry_key]
    head = submissions[-1]
    record_payload: dict[str, object] = {
        "schema_version": "state-decay-v2-lineage-review-record/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "split": candidate.split,
        "family": candidate.family,
        "boundary": ReviewBoundary.PUBLIC,
        "lineage_registry_key": candidate.lineage_registry_key,
        "candidate_packet_digest": candidate.candidate_packet_digest,
        "independent_seed_commitment_digest": (candidate.independent_seed_commitment_digest),
        "transition_graph_digest": candidate.transition_graph.transition_graph_digest,
        "evidence_topology_digest": candidate.evidence_topology.evidence_topology_digest,
        "failure_mechanism_id": candidate.failure_mechanism.failure_mechanism_id,
        "semantic_signature_digest": (candidate.semantic_signature.semantic_signature_digest),
        "derivation_parent_keys": candidate.derivation_parent_keys,
        "semantic_rationale": candidate.semantic_rationale,
        "reviewer_id": head.reviewer_id,
        "review_rationale": head.review_rationale,
        "decision": head.decision,
    }
    record_payload["review_digest"] = lineage_review_record_digest(record_payload)
    record = LineageReviewRecord.model_validate(record_payload)
    envelope_payload: dict[str, object] = {
        "schema_version": "state-decay-v2-public-review-envelope/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "split": candidate.split,
        "family": candidate.family,
        "lineage_registry_key": candidate.lineage_registry_key,
        "candidate_packet_digest": candidate.candidate_packet_digest,
        "draft_digest": draft.draft_digest,
        "checklist_digest": draft.checklist_digest,
        "family_comparison_digest": draft.family_comparison_digest,
        "profile_catalog_digest": draft.profile_catalog_digest,
        "generator_configuration_digest": draft.generator_configuration_digest,
        "generator_algorithm_digest": draft.generator_algorithm_digest,
        "submissions": submissions,
        "review_record": record,
    }
    envelope_payload["envelope_digest"] = review_envelope_digest(envelope_payload)
    return PublicReviewEnvelope.model_validate(envelope_payload)


def _progress_by_key(report: PublicReviewGateReport) -> dict[str, PublicReviewCandidateProgress]:
    return {candidate.lineage_registry_key: candidate for candidate in report.candidates}


def _gate(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
    envelopes: tuple[PublicReviewEnvelope, ...],
) -> PublicReviewGateReport:
    return evaluate_public_review_gate(
        registry=registry,
        comparisons=comparisons,
        drafts=drafts,
        envelopes=envelopes,
    )


def _revised_registry(registry: PublicLineageRegistry) -> PublicLineageRegistry:
    original = registry.candidates[0]
    candidate_payload = original.model_dump(mode="python")
    candidate_payload["semantic_rationale"] = (
        "Revised explicit semantic rationale for this public lineage candidate."
    )
    candidate_payload["candidate_packet_digest"] = candidate_packet_digest(candidate_payload)
    revised_candidate = PublicLineageCandidate.model_validate(candidate_payload)

    registry_payload = registry.model_dump(mode="python")
    registry_payload["candidates"] = (revised_candidate, *registry.candidates[1:])
    registry_payload["registry_digest"] = candidate_registry_digest(registry_payload)
    return PublicLineageRegistry.model_validate(registry_payload)


def test_review_materials_project_the_canonical_registry_exactly(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    assert len(comparisons) == 6
    assert len(drafts) == len(registry.candidates) == 180
    assert tuple(comparison.family for comparison in comparisons) == tuple(
        dict.fromkeys(candidate.family for candidate in registry.candidates)
    )
    assert tuple(draft.lineage_registry_key for draft in drafts) == tuple(
        candidate.lineage_registry_key for candidate in registry.candidates
    )

    candidates = _candidate_by_key(registry)
    comparisons_by_family = _comparison_by_family(comparisons)
    for comparison in comparisons:
        family_candidates = tuple(
            candidate for candidate in registry.candidates if candidate.family is comparison.family
        )
        assert len(comparison.entries) == len(family_candidates) == 30
        for entry, candidate in zip(
            comparison.entries,
            family_candidates,
            strict=True,
        ):
            assert entry.lineage_registry_key == candidate.lineage_registry_key
            assert entry.candidate_packet_digest == candidate.candidate_packet_digest
            assert entry.semantic_rationale == candidate.semantic_rationale
            projection = entry.semantic_projection
            assert projection.task_template == candidate.task_template
            assert projection.transition_graph == candidate.transition_graph
            assert projection.evidence_topology == candidate.evidence_topology
            assert projection.failure_mechanism == candidate.failure_mechanism
            assert projection.semantic_signature == candidate.semantic_signature
            assert projection.causal_deltas == candidate.causal_deltas
            assert entry.preview_digests == tuple(
                preview.preview_digest for preview in candidate.previews
            )

    for draft in drafts:
        candidate = candidates[draft.lineage_registry_key]
        comparison = comparisons_by_family[candidate.family]
        assert draft.split is candidate.split
        assert draft.family is candidate.family
        assert draft.candidate_packet_digest == candidate.candidate_packet_digest
        assert draft.checklist_digest == PUBLIC_REVIEW_CHECKLIST.checklist_digest
        assert draft.preview_digests == tuple(
            preview.preview_digest for preview in candidate.previews
        )
        assert draft.family_comparison_digest == comparison.family_comparison_digest
        assert draft.profile_catalog_digest == registry.profile_catalog.catalog_digest
        assert draft.generator_configuration_digest == registry.generator_configuration_digest
        assert draft.generator_algorithm_digest == registry.generator_algorithm_digest

    rebuilt_comparisons = build_public_family_comparisons(registry=registry)
    rebuilt_drafts = build_public_review_drafts(
        registry=registry,
        comparisons=rebuilt_comparisons,
    )
    assert canonical_json(rebuilt_comparisons) == canonical_json(comparisons)
    assert canonical_json(rebuilt_drafts) == canonical_json(drafts)


def test_material_cache_never_exposes_its_private_snapshots(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    exposed_comparisons = build_public_family_comparisons(registry=registry)
    exposed_drafts = build_public_review_drafts(
        registry=registry,
        comparisons=exposed_comparisons,
    )
    assert exposed_comparisons[0] is not comparisons[0]
    assert exposed_drafts[0] is not drafts[0]

    object.__setattr__(
        exposed_comparisons[0],
        "family_comparison_digest",
        "0" * 64,
    )
    object.__setattr__(exposed_drafts[0], "draft_digest", "0" * 64)

    rebuilt_comparisons = build_public_family_comparisons(registry=registry)
    rebuilt_drafts = build_public_review_drafts(
        registry=registry,
        comparisons=rebuilt_comparisons,
    )
    assert canonical_json(rebuilt_comparisons) == canonical_json(comparisons)
    assert canonical_json(rebuilt_drafts) == canonical_json(drafts)

    tampered_registry = _revised_registry(registry)
    build_public_family_comparisons(registry=tampered_registry)
    object.__setattr__(
        tampered_registry.candidates[0],
        "semantic_rationale",
        "Deep mutation after the registry entered the validation cache.",
    )
    with pytest.raises(PublicReviewError) as captured:
        build_public_family_comparisons(registry=tampered_registry)
    assert captured.value.code is PublicReviewErrorCode.DIGEST_MISMATCH


def test_material_and_gate_inputs_fail_closed_with_stable_codes(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    with pytest.raises(PublicReviewError) as incomplete_comparisons:
        build_public_review_drafts(
            registry=registry,
            comparisons=comparisons[:-1],
        )
    assert incomplete_comparisons.value.code is PublicReviewErrorCode.INCOMPLETE_MATERIALS

    with pytest.raises(PublicReviewError) as reordered_comparisons:
        build_public_review_drafts(
            registry=registry,
            comparisons=tuple(reversed(comparisons)),
        )
    assert reordered_comparisons.value.code is PublicReviewErrorCode.BINDING_MISMATCH

    with pytest.raises(PublicReviewError) as incomplete_drafts:
        evaluate_public_review_gate(
            registry=registry,
            comparisons=comparisons,
            drafts=drafts[:-1],
            envelopes=(),
        )
    assert incomplete_drafts.value.code is PublicReviewErrorCode.INCOMPLETE_MATERIALS

    with pytest.raises(PublicReviewError) as malformed_envelopes:
        evaluate_public_review_gate(
            registry=registry,
            comparisons=comparisons,
            drafts=drafts,
            envelopes=cast(tuple[PublicReviewEnvelope, ...], object()),
        )
    assert malformed_envelopes.value.code is PublicReviewErrorCode.MALFORMED_INPUT


@pytest.mark.parametrize(
    ("decision", "failed_indexes", "valid"),
    (
        (ReviewDecision.ACCEPTED, (), True),
        *((ReviewDecision.REJECTED, (index,), True) for index in range(7)),
        (ReviewDecision.ACCEPTED, (2,), False),
        (ReviewDecision.REJECTED, (), False),
    ),
)
def test_submission_truth_table_is_exact(
    drafts: tuple[PublicReviewDraft, ...],
    decision: ReviewDecision,
    failed_indexes: tuple[int, ...],
    valid: bool,
) -> None:
    def call() -> PublicReviewSubmission:
        return _submission(
            drafts[0],
            decision=decision,
            failed_indexes=failed_indexes,
        )

    if not valid:
        with pytest.raises(ValueError):
            call()
        return

    submission = call()
    assert submission.decision is decision
    assert (
        tuple(
            index
            for index, answer in enumerate(submission.checklist_answers)
            if answer.answer is PublicReviewAnswer.FAILED
        )
        == failed_indexes
    )


@pytest.mark.parametrize(
    "answers",
    (
        _answers()[:-1],
        (*_answers()[:-1], _answers()[0]),
        tuple(reversed(_answers())),
        (*_answers(), _answers()[0]),
    ),
    ids=("missing", "duplicate", "reordered", "extra"),
)
def test_submission_requires_the_complete_explicit_ordered_checklist(
    drafts: tuple[PublicReviewDraft, ...],
    answers: tuple[PublicReviewChecklistAnswer, ...],
) -> None:
    with pytest.raises(ValueError):
        build_public_review_submission(
            draft=drafts[0],
            reviewer_id=_REVIEWER_ID,
            review_rationale=_REVIEW_RATIONALE,
            checklist_answers=answers,
            decision=ReviewDecision.ACCEPTED,
            supersedes_submission_digest=None,
        )


def test_submission_builder_has_no_session_or_answer_defaults() -> None:
    parameters = inspect.signature(build_public_review_submission).parameters
    assert tuple(parameters) == (
        "draft",
        "reviewer_id",
        "review_rationale",
        "checklist_answers",
        "decision",
        "supersedes_submission_digest",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters.values()
    )
    assert all(
        parameters[name].default is inspect.Parameter.empty
        for name in (
            "reviewer_id",
            "review_rationale",
            "checklist_answers",
            "decision",
            "supersedes_submission_digest",
        )
    )


def test_submission_errors_distinguish_unsafe_text_and_inconsistent_review(
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    with pytest.raises(PublicReviewError) as unsafe_text:
        build_public_review_submission(
            draft=drafts[0],
            reviewer_id=_REVIEWER_ID,
            review_rationale="unsafe\u202etext",
            checklist_answers=_answers(),
            decision=ReviewDecision.ACCEPTED,
            supersedes_submission_digest=None,
        )
    assert unsafe_text.value.code is PublicReviewErrorCode.UNSAFE_TEXT

    with pytest.raises(PublicReviewError) as inconsistent:
        build_public_review_submission(
            draft=drafts[0],
            reviewer_id=_REVIEWER_ID,
            review_rationale=_REVIEW_RATIONALE,
            checklist_answers=_answers(0),
            decision=ReviewDecision.ACCEPTED,
            supersedes_submission_digest=None,
        )
    assert inconsistent.value.code is PublicReviewErrorCode.INCONSISTENT_REVIEW


def test_chain_root_correction_and_head_projection_round_trip(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    draft = drafts[0]
    candidate = registry.candidates[0]
    rejected = _submission(
        draft,
        decision=ReviewDecision.REJECTED,
        failed_indexes=(4,),
        reviewer_id="synthetic-rejecting-reviewer",
        rationale="Synthetic rejection before an explicit correction.",
    )
    accepted = _submission(
        draft,
        predecessor=rejected.submission_digest,
        reviewer_id="synthetic-correcting-reviewer",
        rationale="Synthetic correction with every checklist item passed.",
    )
    envelope = _envelope(registry, comparisons, draft, (rejected, accepted))
    reloaded = PublicReviewEnvelope.model_validate_json(canonical_json(envelope))

    assert reloaded == envelope
    assert reloaded.submissions == (rejected, accepted)
    assert reloaded.submissions[0].supersedes_submission_digest is None
    assert reloaded.submissions[1].supersedes_submission_digest == rejected.submission_digest
    record = reloaded.review_record
    assert record.split is candidate.split
    assert record.family is candidate.family
    assert record.boundary is ReviewBoundary.PUBLIC
    assert record.lineage_registry_key == candidate.lineage_registry_key
    assert record.candidate_packet_digest == candidate.candidate_packet_digest
    assert record.independent_seed_commitment_digest == candidate.independent_seed_commitment_digest
    assert record.transition_graph_digest == candidate.transition_graph.transition_graph_digest
    assert record.evidence_topology_digest == candidate.evidence_topology.evidence_topology_digest
    assert record.failure_mechanism_id == candidate.failure_mechanism.failure_mechanism_id
    assert (
        record.semantic_signature_digest == candidate.semantic_signature.semantic_signature_digest
    )
    assert record.derivation_parent_keys == candidate.derivation_parent_keys == ()
    assert record.semantic_rationale == candidate.semantic_rationale
    assert record.reviewer_id == accepted.reviewer_id
    assert record.review_rationale == accepted.review_rationale
    assert record.decision is ReviewDecision.ACCEPTED
    assert record.reviewer_id != rejected.reviewer_id
    assert record.review_rationale != rejected.review_rationale


def test_chain_rejects_non_null_root_and_missing_predecessor_but_canonicalizes_order(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    draft = drafts[0]
    non_null_root = _submission(draft, predecessor="f" * 64)
    with pytest.raises(ValueError):
        _envelope(registry, comparisons, draft, (non_null_root,))

    root = _submission(
        draft,
        decision=ReviewDecision.REJECTED,
        failed_indexes=(0,),
    )
    missing = _submission(draft, predecessor="e" * 64)
    with pytest.raises(ValueError):
        _envelope(registry, comparisons, draft, (root, missing))

    correction = _submission(draft, predecessor=root.submission_digest)
    reordered = _envelope(registry, comparisons, draft, (correction, root))
    assert reordered.submissions == (root, correction)


def test_chain_rejects_cross_candidate_fork_duplicate_and_multiple_heads(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    first_draft, second_draft = drafts[:2]
    root = _submission(
        first_draft,
        decision=ReviewDecision.REJECTED,
        failed_indexes=(1,),
    )
    cross_candidate = _submission(
        second_draft,
        predecessor=root.submission_digest,
    )
    with pytest.raises(ValueError):
        _envelope(
            registry,
            comparisons,
            second_draft,
            (root, cross_candidate),
        )

    left = _submission(
        first_draft,
        predecessor=root.submission_digest,
        reviewer_id="synthetic-left-reviewer",
        rationale="Synthetic left fork correction.",
    )
    right = _submission(
        first_draft,
        predecessor=root.submission_digest,
        reviewer_id="synthetic-right-reviewer",
        rationale="Synthetic right fork correction.",
    )
    independent_root = _submission(
        first_draft,
        reviewer_id="synthetic-independent-reviewer",
        rationale="Synthetic independent review root.",
    )
    for submissions in (
        (root, left, right),
        (root, root),
        (root, left, independent_root),
    ):
        with pytest.raises(ValueError):
            _envelope(registry, comparisons, first_draft, submissions)


def test_chain_failures_expose_stable_value_free_codes(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    draft = drafts[0]
    root = _submission(
        draft,
        decision=ReviewDecision.REJECTED,
        failed_indexes=(0,),
    )
    left = _submission(
        draft,
        predecessor=root.submission_digest,
        reviewer_id="synthetic-coded-left",
        rationale="Synthetic rationale that must never occur in an error message.",
    )
    right = _submission(
        draft,
        predecessor=root.submission_digest,
        reviewer_id="synthetic-coded-right",
        rationale="Synthetic second rationale that must remain private from errors.",
    )
    independent = _submission(
        draft,
        reviewer_id="synthetic-coded-independent",
        rationale="Synthetic independent root rationale.",
    )
    missing = _submission(draft, predecessor="f" * 64)

    cases = (
        ((missing,), PublicReviewErrorCode.MISSING_PREDECESSOR),
        ((root, left, right), PublicReviewErrorCode.FORK),
        ((root, root), PublicReviewErrorCode.DUPLICATE),
        ((root, independent), PublicReviewErrorCode.MULTIPLE_HEAD),
        ((), PublicReviewErrorCode.INCOMPLETE_MATERIALS),
    )
    for submissions, code in cases:
        with pytest.raises(PublicReviewError) as captured:
            _envelope(registry, comparisons, draft, submissions)
        assert captured.value.code is code
        assert "rationale" not in str(captured.value).casefold()
        assert _REVIEW_RATIONALE not in str(captured.value)


def test_chain_revalidates_forged_cycle_and_enforces_32_link_bound(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    draft = drafts[0]
    base = _submission(draft)
    left_payload = dict(base.__dict__)
    left_payload["supersedes_submission_digest"] = "b" * 64
    left_payload["submission_digest"] = "a" * 64
    right_payload = dict(base.__dict__)
    right_payload["supersedes_submission_digest"] = "a" * 64
    right_payload["submission_digest"] = "b" * 64
    forged_cycle = (
        PublicReviewSubmission.model_construct(**left_payload),
        PublicReviewSubmission.model_construct(**right_payload),
    )
    with pytest.raises(ValueError):
        _envelope(registry, comparisons, draft, forged_cycle)

    submissions: list[PublicReviewSubmission] = []
    predecessor: str | None = None
    for index in range(33):
        submission = _submission(
            draft,
            predecessor=predecessor,
            reviewer_id=f"synthetic-reviewer-{index}",
            rationale=f"Synthetic explicit review chain entry {index}.",
        )
        submissions.append(submission)
        predecessor = submission.submission_digest
    with pytest.raises(ValueError):
        _envelope(registry, comparisons, draft, tuple(submissions))


def test_candidate_revision_starts_a_new_chain_and_only_changes_its_family(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    revised_registry = _revised_registry(registry)
    revised_comparisons = build_public_family_comparisons(registry=revised_registry)
    revised_drafts = build_public_review_drafts(
        registry=revised_registry,
        comparisons=revised_comparisons,
    )
    original = registry.candidates[0]
    revised = revised_registry.candidates[0]
    assert revised.lineage_registry_key == original.lineage_registry_key
    assert revised.candidate_packet_digest != original.candidate_packet_digest

    old_comparisons = _comparison_by_family(comparisons)
    new_comparisons = _comparison_by_family(revised_comparisons)
    for family in old_comparisons:
        if family is original.family:
            assert canonical_json(old_comparisons[family]) != canonical_json(
                new_comparisons[family]
            )
        else:
            assert canonical_json(old_comparisons[family]) == canonical_json(
                new_comparisons[family]
            )

    old_drafts = _draft_by_key(drafts)
    new_drafts = _draft_by_key(revised_drafts)
    for candidate in registry.candidates:
        old_draft = old_drafts[candidate.lineage_registry_key]
        new_draft = new_drafts[candidate.lineage_registry_key]
        if candidate.family is original.family:
            assert old_draft.draft_digest != new_draft.draft_digest
        else:
            assert canonical_json(old_draft) == canonical_json(new_draft)

    historical = _submission(old_drafts[original.lineage_registry_key])
    historical_envelope = _envelope(
        registry,
        comparisons,
        old_drafts[original.lineage_registry_key],
        (historical,),
    )
    revised_draft = new_drafts[revised.lineage_registry_key]
    invalid_continuation = _submission(
        revised_draft,
        predecessor=historical.submission_digest,
    )
    with pytest.raises(ValueError):
        _envelope(
            revised_registry,
            revised_comparisons,
            revised_draft,
            (historical, invalid_continuation),
        )

    revised_progress = _progress_by_key(
        _gate(
            revised_registry,
            revised_comparisons,
            revised_drafts,
            (historical_envelope,),
        )
    )[revised.lineage_registry_key]
    assert revised_progress.state is PublicReviewProgressState.MISSING
    assert revised_progress.head_submission_digest is None

    current_root = _submission(revised_draft, predecessor=None)
    current_envelope = _envelope(
        revised_registry,
        revised_comparisons,
        revised_draft,
        (current_root,),
    )
    assert current_envelope.submissions[0].supersedes_submission_digest is None
    assert (
        current_envelope.candidate_packet_digest
        == revised.candidate_packet_digest
        != historical.candidate_packet_digest
    )


def test_family_comparison_change_stales_sibling_until_explicit_reentry(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    revised_registry = _revised_registry(registry)
    revised_comparisons = build_public_family_comparisons(registry=revised_registry)
    revised_drafts = build_public_review_drafts(
        registry=revised_registry,
        comparisons=revised_comparisons,
    )
    sibling = registry.candidates[1]
    old_draft = _draft_by_key(drafts)[sibling.lineage_registry_key]
    new_draft = _draft_by_key(revised_drafts)[sibling.lineage_registry_key]
    assert old_draft.candidate_packet_digest == new_draft.candidate_packet_digest
    assert old_draft.family_comparison_digest != new_draft.family_comparison_digest
    assert old_draft.draft_digest != new_draft.draft_digest

    previous = _submission(old_draft)
    previous_envelope = _envelope(
        registry,
        comparisons,
        old_draft,
        (previous,),
    )
    revised_candidate_draft = drafts[0]
    revised_candidate_history = _submission(revised_candidate_draft)
    revised_candidate_envelope = _direct_test_envelope(
        registry,
        revised_candidate_draft,
        (revised_candidate_history,),
    )
    stale_report = _gate(
        revised_registry,
        revised_comparisons,
        revised_drafts,
        (previous_envelope, revised_candidate_envelope),
    )
    stale = _progress_by_key(stale_report)[sibling.lineage_registry_key]
    assert stale.state is PublicReviewProgressState.STALE_COMPARISON
    assert stale.head_submission_digest == previous.submission_digest
    revised_progress = _progress_by_key(stale_report)[revised_candidate_draft.lineage_registry_key]
    assert revised_progress.state is PublicReviewProgressState.MISSING
    assert revised_progress.head_submission_digest is None

    current = _submission(
        new_draft,
        predecessor=previous.submission_digest,
        reviewer_id="synthetic-reentry-reviewer",
        rationale="Synthetic full explicit re-entry after sibling comparison changed.",
    )
    current_envelope = _envelope(
        revised_registry,
        revised_comparisons,
        new_draft,
        (previous, current),
    )
    current_report = _gate(
        revised_registry,
        revised_comparisons,
        revised_drafts,
        (current_envelope,),
    )
    progress = _progress_by_key(current_report)[sibling.lineage_registry_key]
    assert progress.state is PublicReviewProgressState.ACCEPTED
    assert progress.head_submission_digest == current.submission_digest


def test_local_gate_reports_missing_rejected_accepted_and_ambiguous_states(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    missing_draft, rejected_draft, accepted_draft, ambiguous_draft = drafts[:4]
    rejected = _submission(
        rejected_draft,
        decision=ReviewDecision.REJECTED,
        failed_indexes=(3,),
    )
    rejected_envelope = _direct_test_envelope(
        registry,
        rejected_draft,
        (rejected,),
    )
    accepted = _submission(accepted_draft)
    accepted_envelope = _direct_test_envelope(
        registry,
        accepted_draft,
        (accepted,),
    )
    other_head = _submission(
        ambiguous_draft,
        reviewer_id="synthetic-other-head-reviewer",
        rationale="Synthetic second current head for ambiguity testing.",
    )
    first_head = _submission(
        ambiguous_draft,
        reviewer_id="synthetic-first-head-reviewer",
        rationale="Synthetic first current head for ambiguity testing.",
    )
    first_envelope = _direct_test_envelope(
        registry,
        ambiguous_draft,
        (first_head,),
    )
    other_envelope = _direct_test_envelope(
        registry,
        ambiguous_draft,
        (other_head,),
    )
    report = _gate(
        registry,
        comparisons,
        drafts,
        (rejected_envelope, accepted_envelope, first_envelope, other_envelope),
    )
    progress = _progress_by_key(report)
    assert progress[missing_draft.lineage_registry_key].state is PublicReviewProgressState.MISSING
    assert progress[rejected_draft.lineage_registry_key].state is PublicReviewProgressState.REJECTED
    assert progress[accepted_draft.lineage_registry_key].state is PublicReviewProgressState.ACCEPTED
    ambiguous = progress[ambiguous_draft.lineage_registry_key]
    assert ambiguous.state is PublicReviewProgressState.AMBIGUOUS
    assert ambiguous.head_submission_digest is None
    assert ambiguous.envelope_digest is None
    assert report.ambiguous_count == 1
    assert report.rejected_count == 1
    assert report.accepted_count == 1
    assert report.missing_count == 177
    assert report.progress_complete is False


def test_gate_uses_the_unique_maximal_envelope_and_keeps_prefixes_as_history(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    draft = drafts[0]
    root = _submission(
        draft,
        decision=ReviewDecision.REJECTED,
        failed_indexes=(2,),
    )
    correction = _submission(
        draft,
        predecessor=root.submission_digest,
        reviewer_id="synthetic-maximal-head-reviewer",
        rationale="Synthetic accepted correction after a historical prefix.",
    )
    prefix = _direct_test_envelope(registry, draft, (root,))
    current = _direct_test_envelope(registry, draft, (root, correction))

    report = _gate(registry, comparisons, drafts, (current, prefix))
    progress = _progress_by_key(report)[draft.lineage_registry_key]
    assert progress.state is PublicReviewProgressState.ACCEPTED
    assert progress.head_submission_digest == correction.submission_digest
    assert progress.envelope_digest == current.envelope_digest


def test_gate_rejects_self_consistent_false_candidate_record_projections(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    submission = _submission(drafts[0])
    envelope = _direct_test_envelope(registry, drafts[0], (submission,))
    other = registry.candidates[1]
    replacements: dict[str, object] = {
        "transition_graph_digest": other.transition_graph.transition_graph_digest,
        "evidence_topology_digest": other.evidence_topology.evidence_topology_digest,
        "failure_mechanism_id": other.failure_mechanism.failure_mechanism_id,
        "semantic_signature_digest": other.semantic_signature.semantic_signature_digest,
        "semantic_rationale": other.semantic_rationale,
    }
    for field, replacement in replacements.items():
        payload = envelope.model_dump(mode="python")
        payload["review_record"][field] = replacement
        payload["review_record"]["review_digest"] = lineage_review_record_digest(
            payload["review_record"]
        )
        payload["envelope_digest"] = review_envelope_digest(payload)
        self_consistent = PublicReviewEnvelope.model_validate(payload)
        with pytest.raises(ValueError):
            _gate(registry, comparisons, drafts, (self_consistent,))


def test_gate_revalidates_forged_nested_models_and_invalid_inputs(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    submission = _submission(drafts[0])
    envelope = _direct_test_envelope(registry, drafts[0], (submission,))

    bad_submission = submission.model_copy(update={"submission_digest": "0" * 64})
    bad_nested = envelope.model_copy(update={"submissions": (bad_submission,)})
    bad_envelope_digest = envelope.model_copy(update={"envelope_digest": "0" * 64})
    extra_coordinate = envelope.model_copy(update={"lineage_registry_key": "pub-fr-99"})
    bad_binding_payload = submission.model_dump(mode="python")
    bad_binding_payload["profile_catalog_digest"] = "a" * 64
    bad_binding_payload["submission_digest"] = review_submission_digest(bad_binding_payload)
    bad_binding = PublicReviewSubmission.model_validate(bad_binding_payload)
    mismatched_nested = envelope.model_copy(update={"submissions": (bad_binding,)})
    global_mismatch_payload = envelope.model_dump(mode="python")
    global_mismatch_payload["profile_catalog_digest"] = "a" * 64
    global_mismatch_payload["submissions"] = (bad_binding,)
    global_mismatch_payload["envelope_digest"] = review_envelope_digest(global_mismatch_payload)
    global_mismatch = PublicReviewEnvelope.model_validate(global_mismatch_payload)
    malformed = cast(PublicReviewEnvelope, object())

    with pytest.raises(PublicReviewError) as digest_mismatch:
        _gate(registry, comparisons, drafts, (bad_envelope_digest,))
    assert digest_mismatch.value.code is PublicReviewErrorCode.DIGEST_MISMATCH

    with pytest.raises(PublicReviewError) as binding_mismatch:
        _gate(registry, comparisons, drafts, (global_mismatch,))
    assert binding_mismatch.value.code is PublicReviewErrorCode.BINDING_MISMATCH

    for invalid in (
        bad_nested,
        extra_coordinate,
        mismatched_nested,
        malformed,
    ):
        with pytest.raises(ValueError):
            _gate(registry, comparisons, drafts, (invalid,))

    with pytest.raises(ValueError):
        _gate(registry, comparisons, drafts, (envelope, envelope))


def test_gate_marks_a_valid_shared_root_fork_ambiguous(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    draft = drafts[0]
    root = _submission(
        draft,
        decision=ReviewDecision.REJECTED,
        failed_indexes=(6,),
    )
    left = _submission(
        draft,
        predecessor=root.submission_digest,
        reviewer_id="synthetic-fork-left",
        rationale="Synthetic accepted left branch.",
    )
    right = _submission(
        draft,
        predecessor=root.submission_digest,
        reviewer_id="synthetic-fork-right",
        rationale="Synthetic accepted right branch.",
    )
    left_envelope = _direct_test_envelope(registry, draft, (root, left))
    right_envelope = _direct_test_envelope(registry, draft, (root, right))
    report = _gate(
        registry,
        comparisons,
        drafts,
        (left_envelope, right_envelope),
    )
    progress = _progress_by_key(report)[draft.lineage_registry_key]
    assert progress.state is PublicReviewProgressState.AMBIGUOUS
    assert report.ambiguous_count == 1
    assert report.progress_complete is False


def test_historical_revision_fork_prevents_a_false_current_acceptance(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    revised_registry = _revised_registry(registry)
    revised_comparisons = build_public_family_comparisons(registry=revised_registry)
    revised_drafts = build_public_review_drafts(
        registry=revised_registry,
        comparisons=revised_comparisons,
    )
    old_draft = drafts[0]
    old_root = _submission(
        old_draft,
        decision=ReviewDecision.REJECTED,
        failed_indexes=(5,),
    )
    old_left = _submission(
        old_draft,
        predecessor=old_root.submission_digest,
        reviewer_id="synthetic-historical-left",
        rationale="Synthetic left correction in a historical revision.",
    )
    old_right = _submission(
        old_draft,
        predecessor=old_root.submission_digest,
        reviewer_id="synthetic-historical-right",
        rationale="Synthetic right correction in a historical revision.",
    )
    historical_left = _direct_test_envelope(
        registry,
        old_draft,
        (old_root, old_left),
    )
    historical_right = _direct_test_envelope(
        registry,
        old_draft,
        (old_root, old_right),
    )
    current_draft = revised_drafts[0]
    current_submission = _submission(current_draft)
    current = _direct_test_envelope(
        revised_registry,
        current_draft,
        (current_submission,),
    )

    report = _gate(
        revised_registry,
        revised_comparisons,
        revised_drafts,
        (historical_left, historical_right, current),
    )
    progress = _progress_by_key(report)[current_draft.lineage_registry_key]
    assert progress.state is PublicReviewProgressState.AMBIGUOUS
    assert progress.head_submission_digest is None
    assert report.progress_complete is False


def test_review_progress_literals_and_report_fields_are_frozen() -> None:
    assert tuple(state.value for state in PublicReviewProgressState) == (
        "ambiguous",
        "stale-comparison",
        "rejected",
        "missing",
        "accepted",
    )
    assert tuple(PublicReviewCandidateProgress.model_fields) == (
        "schema_version",
        "suite_id",
        "suite_version",
        "split",
        "family",
        "lineage_registry_key",
        "candidate_packet_digest",
        "state",
        "head_submission_digest",
        "envelope_digest",
    )
    assert tuple(PublicReviewGateReport.model_fields) == (
        "schema_version",
        "suite_id",
        "suite_version",
        "candidate_registry_digest",
        "checklist_digest",
        "candidates",
        "ambiguous_count",
        "stale_comparison_count",
        "rejected_count",
        "missing_count",
        "accepted_count",
        "progress_complete",
    )


def test_progress_report_reload_requires_canonical_coordinates(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    report = _gate(registry, comparisons, drafts, ())
    reordered = report.model_dump(mode="python")
    reordered["candidates"] = tuple(reversed(reordered["candidates"]))
    with pytest.raises(ValidationError):
        PublicReviewGateReport.model_validate(reordered)

    wrong_split = report.model_dump(mode="python")
    wrong_split["candidates"][0]["split"] = "development"
    with pytest.raises(ValidationError):
        PublicReviewGateReport.model_validate(wrong_split)


def test_complete_accepted_gate_report_is_progress_only_and_cannot_authorize(
    monkeypatch: pytest.MonkeyPatch,
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    envelopes = tuple(
        _direct_test_envelope(
            registry,
            draft,
            (
                _submission(
                    draft,
                    reviewer_id=(f"synthetic-complete-reviewer-{index}"),
                    rationale=(f"Synthetic complete explicit review {index}."),
                ),
            ),
        )
        for index, draft in enumerate(drafts)
    )
    report = _gate(registry, comparisons, drafts, envelopes)

    assert type(report) is PublicReviewGateReport
    assert PublicReviewGateReport.model_validate_json(canonical_json(report)) == report
    assert report.candidate_registry_digest == registry.registry_digest
    assert report.checklist_digest == PUBLIC_REVIEW_CHECKLIST.checklist_digest
    assert tuple(candidate.lineage_registry_key for candidate in report.candidates) == tuple(
        candidate.lineage_registry_key for candidate in registry.candidates
    )
    assert report.ambiguous_count == 0
    assert report.stale_comparison_count == 0
    assert report.rejected_count == 0
    assert report.missing_count == 0
    assert report.accepted_count == 180
    assert report.progress_complete is True
    assert all(
        candidate.state is PublicReviewProgressState.ACCEPTED for candidate in report.candidates
    )

    forbidden_fragments = (
        "accepted_envelope_registry",
        "subreport",
        "readiness",
        "authority",
        "review_record_digests",
    )
    assert not any(
        fragment in field
        for field in PublicReviewGateReport.model_fields
        for fragment in forbidden_fragments
    )
    exported = set(getattr(review_module, "__all__", ()))
    assert not any(
        fragment in name.casefold()
        for name in exported
        for fragment in ("subreport", "readiness", "authority", "issuer", "finalize")
    )

    with pytest.raises(ValidationError):
        PublicLineageReviewSubreport.model_validate(report.model_dump(mode="python"))
    with pytest.raises(PublicGenerationAuthorityError):
        require_public_generation_authority(report)

    allocation_called = False

    def forbidden_allocation(*args: object, **kwargs: object) -> object:
        nonlocal allocation_called
        del args, kwargs
        allocation_called = True
        raise AssertionError("allocation must remain unreachable from a review-gate report")

    monkeypatch.setattr(
        config_module,
        "allocate_balanced_outcomes",
        forbidden_allocation,
    )
    with pytest.raises(PublicGenerationAuthorityError):
        asyncio.run(generate_public_scenarios(report))
    assert allocation_called is False


def test_progress_models_reject_coordinate_head_count_and_completion_drift(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
) -> None:
    report = _gate(registry, comparisons, drafts, ())
    progress_payload = report.candidates[0].model_dump(mode="python")
    progress_payload["family"] = ScenarioFamily.FAILED_PRIOR_ATTEMPT
    with pytest.raises(ValidationError, match="coordinates"):
        PublicReviewCandidateProgress.model_validate(progress_payload)

    progress_payload = report.candidates[0].model_dump(mode="python")
    progress_payload["head_submission_digest"] = "1" * 64
    progress_payload["envelope_digest"] = "2" * 64
    with pytest.raises(ValidationError, match="head bindings"):
        PublicReviewCandidateProgress.model_validate(progress_payload)

    report_payload = report.model_dump(mode="python")
    report_payload["missing_count"] = 179
    report_payload["accepted_count"] = 1
    with pytest.raises(ValidationError, match="counts"):
        PublicReviewGateReport.model_validate(report_payload)

    report_payload = report.model_dump(mode="python")
    report_payload["progress_complete"] = True
    with pytest.raises(ValidationError, match="completion"):
        PublicReviewGateReport.model_validate(report_payload)


def test_review_defensive_adapters_fail_closed_and_material_cache_is_bounded(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InconsistentSequence(list[object]):
        def __len__(self) -> int:
            return 1

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(())

    class ExplodingSequence(list[object]):
        def __len__(self) -> int:
            raise RuntimeError("private-length")

    with pytest.raises(PublicReviewError) as inconsistent:
        review_module._sequence(
            InconsistentSequence((object(),)),
            minimum_length=1,
            maximum_length=1,
        )
    assert inconsistent.value.code is PublicReviewErrorCode.MALFORMED_INPUT

    with pytest.raises(PublicReviewError) as exploding:
        review_module._sequence(
            ExplodingSequence((object(),)),
            minimum_length=0,
            maximum_length=1,
        )
    assert exploding.value.code is PublicReviewErrorCode.MALFORMED_INPUT

    with pytest.raises(PublicReviewError) as unequal:
        review_module._same(object(), object())
    assert unequal.value.code is PublicReviewErrorCode.MALFORMED_INPUT
    with pytest.raises(PublicReviewError) as malformed_registry:
        review_module._registry_snapshot(object())  # type: ignore[arg-type]
    assert malformed_registry.value.code is PublicReviewErrorCode.MALFORMED_INPUT

    cache = {index: str(index) for index in range(review_module._MAX_MATERIAL_CACHE_ENTRIES)}
    review_module._remember_materials(cache, 99, "new")
    assert len(cache) == review_module._MAX_MATERIAL_CACHE_ENTRIES
    assert 0 not in cache and cache[99] == "new"

    with monkeypatch.context() as patch:
        patch.setattr(
            review_module,
            "_build_family_comparisons_checked",
            lambda checked: (_ for _ in ()).throw(RuntimeError("private-comparison")),
        )
        with pytest.raises(PublicReviewError) as comparison_error:
            build_public_family_comparisons(registry=registry)
    assert comparison_error.value.code is PublicReviewErrorCode.MALFORMED_INPUT

    with monkeypatch.context() as patch:
        patch.setattr(
            review_module,
            "_build_drafts_checked",
            lambda checked_registry, checked_comparisons: (_ for _ in ()).throw(
                RuntimeError("private-draft")
            ),
        )
        with pytest.raises(PublicReviewError) as draft_error:
            build_public_review_drafts(registry=registry, comparisons=comparisons)
    assert draft_error.value.code is PublicReviewErrorCode.MALFORMED_INPUT

    with pytest.raises(PublicReviewError) as drafts_error:
        review_module._checked_drafts(registry, comparisons, tuple(reversed(drafts)))
    assert drafts_error.value.code is PublicReviewErrorCode.BINDING_MISMATCH


def test_review_chain_and_binding_defenses_cover_disconnected_and_stale_materials(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_submission(digest: str, predecessor: str | None) -> SimpleNamespace:
        return SimpleNamespace(
            split="split",
            family="family",
            lineage_registry_key="lineage",
            candidate_packet_digest="candidate",
            checklist_digest="checklist",
            profile_catalog_digest="profile",
            generator_configuration_digest="configuration",
            generator_algorithm_digest="algorithm",
            submission_digest=digest,
            supersedes_submission_digest=predecessor,
        )

    cyclic = (
        fake_submission("1" * 64, "2" * 64),
        fake_submission("2" * 64, "1" * 64),
    )
    with monkeypatch.context() as patch:
        patch.setattr(review_module, "_revalidate", lambda model_type, value: value)
        with pytest.raises(PublicReviewError) as no_root:
            review_module._order_submission_chain(cyclic)
    assert no_root.value.code is PublicReviewErrorCode.CYCLE

    disconnected = (
        fake_submission("3" * 64, None),
        fake_submission("4" * 64, "5" * 64),
        fake_submission("5" * 64, "4" * 64),
    )
    with monkeypatch.context() as patch:
        patch.setattr(review_module, "_revalidate", lambda model_type, value: value)
        with pytest.raises(PublicReviewError) as disconnected_cycle:
            review_module._order_submission_chain(disconnected)
    assert disconnected_cycle.value.code is PublicReviewErrorCode.CYCLE

    missing_draft = SimpleNamespace(lineage_registry_key="missing")
    with pytest.raises(PublicReviewError) as missing_candidate:
        review_module._candidate_for_draft(
            registry,
            cast(PublicReviewDraft, missing_draft),
        )
    assert missing_candidate.value.code is PublicReviewErrorCode.BINDING_MISMATCH

    original_submission = _submission(drafts[0])
    revised_registry = _revised_registry(registry)
    revised_comparisons = build_public_family_comparisons(registry=revised_registry)
    with pytest.raises(PublicReviewError) as stale_head:
        review_module.build_public_review_head_envelope(
            registry=revised_registry,
            submissions=(original_submission,),
        )
    assert stale_head.value.code is PublicReviewErrorCode.BINDING_MISMATCH

    with pytest.raises(PublicReviewError) as stale_comparison:
        review_module._expected_comparison_for_family(revised_registry, comparisons[0])
    assert stale_comparison.value.code is PublicReviewErrorCode.BINDING_MISMATCH

    revised_candidate = revised_registry.candidates[0]
    revised_comparison = _comparison_by_family(revised_comparisons)[revised_candidate.family]
    with pytest.raises(PublicReviewError) as stale_draft:
        review_module._expected_draft_for_candidate(
            registry=revised_registry,
            candidate=revised_candidate,
            comparison=revised_comparison,
            draft=drafts[0],
        )
    assert stale_draft.value.code is PublicReviewErrorCode.BINDING_MISMATCH

    with pytest.raises(PublicReviewError) as cross_family:
        build_public_review_envelope(
            registry=registry,
            draft=drafts[0],
            family_comparison=comparisons[1],
            submissions=(original_submission,),
        )
    assert cross_family.value.code is PublicReviewErrorCode.BINDING_MISMATCH

    cross_candidate_submission = _submission(drafts[30])
    with pytest.raises(PublicReviewError) as cross_candidate:
        build_public_review_envelope(
            registry=registry,
            draft=drafts[0],
            family_comparison=comparisons[0],
            submissions=(cross_candidate_submission,),
        )
    assert cross_candidate.value.code is PublicReviewErrorCode.BINDING_MISMATCH


def test_gate_rejects_nonprefix_maximum_and_unmapped_envelope(
    registry: PublicLineageRegistry,
    comparisons: tuple[PublicFamilyComparison, ...],
    drafts: tuple[PublicReviewDraft, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    short = SimpleNamespace(submissions=(SimpleNamespace(submission_digest="1" * 64),))
    long = SimpleNamespace(
        submissions=(
            SimpleNamespace(submission_digest="2" * 64),
            SimpleNamespace(submission_digest="3" * 64),
        )
    )
    assert (
        review_module._maximal_current_envelope(
            cast(tuple[PublicReviewEnvelope, ...], (short, long))
        )
        is None
    )

    submission = _submission(drafts[0])
    envelope = _envelope(registry, comparisons, drafts[0], (submission,))
    with monkeypatch.context() as patch:
        patch.setattr(review_module, "_candidate_map", lambda checked: {})
        with pytest.raises(PublicReviewError) as unmapped:
            _gate(registry, comparisons, drafts, (envelope,))
    assert unmapped.value.code is PublicReviewErrorCode.BINDING_MISMATCH
