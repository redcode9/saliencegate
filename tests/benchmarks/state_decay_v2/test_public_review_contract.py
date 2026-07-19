from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest
from pydantic import ValidationError

import saliencegate.benchmarks.state_decay_v2.review_contract as review_contract_module
from saliencegate.benchmarks.state_decay_v2.config import (
    GENERATION_CONTRACT,
    PUBLIC_GENERATION_SEED,
    SeedPurpose,
    derive_seed,
)
from saliencegate.benchmarks.state_decay_v2.protocol import (
    LINEAGE_REVIEW_PROTOCOL,
    LineageReviewRecord,
    ReviewBoundary,
    ReviewDecision,
    derive_independent_lineage_seed,
    independent_lineage_seed_commitment,
    lineage_review_record_digest,
)
from saliencegate.benchmarks.state_decay_v2.public_contract import (
    CausalSemanticDelta,
    OutcomeFreeTaskTemplate,
    PublicCausalExposure,
    PublicCausalFactor,
    PublicCausalFactorValue,
    PublicEvidenceRelation,
    PublicEvidenceTopology,
    PublicFailureMechanism,
    PublicSemanticSignature,
    PublicTerminalState,
    PublicTransition,
    PublicTransitionGraph,
    PublicTransitionState,
    causal_delta_digest,
    evidence_topology_digest,
    semantic_signature_digest,
    transition_graph_digest,
)
from saliencegate.benchmarks.state_decay_v2.review_contract import (
    ACCEPTED_ENVELOPE_REGISTRY_DIGEST_DOMAIN,
    FAMILY_COMPARISON_DIGEST_DOMAIN,
    PACK_CHILD_DIGEST_DOMAIN,
    PACK_MANIFEST_DIGEST_DOMAIN,
    PUBLIC_REVIEW_CHECKLIST,
    REVIEW_CHECKLIST_DIGEST_DOMAIN,
    REVIEW_DRAFT_DIGEST_DOMAIN,
    REVIEW_ENVELOPE_DIGEST_DOMAIN,
    REVIEW_SUBMISSION_DIGEST_DOMAIN,
    SUBREPORT_DIGEST_DOMAIN,
    PublicCandidateSemanticProjection,
    PublicFamilyComparison,
    PublicFamilyComparisonEntry,
    PublicLineageReviewSubreport,
    PublicReviewAnswer,
    PublicReviewChecklist,
    PublicReviewChecklistAnswer,
    PublicReviewChecklistItemId,
    PublicReviewDraft,
    PublicReviewEnvelope,
    PublicReviewPackBasename,
    PublicReviewPackChild,
    PublicReviewPackManifest,
    PublicReviewSubmission,
    accepted_envelope_registry_digest,
    family_comparison_digest,
    pack_child_digest,
    pack_manifest_digest,
    public_review_checklist_digest,
    review_draft_digest,
    review_envelope_digest,
    review_submission_digest,
    subreport_digest,
)
from saliencegate.benchmarks.state_decay_v2.schema import (
    AdapterMetadata,
    BenchmarkSplit,
    ScenarioFamily,
)
from saliencegate.domain import canonical_json

_CAUSAL_FACTOR_IDS = ("guidance-relevant", "baseline-can-recover")
_CAUSAL_FACTOR_VECTORS = (
    (False, False),
    (False, True),
    (True, False),
    (True, True),
)
_PAIRED_TERMINALS = (
    (PublicTerminalState.GOAL_REACHED, PublicTerminalState.GOAL_NOT_REACHED),
    (PublicTerminalState.GOAL_NOT_REACHED, PublicTerminalState.GOAL_REACHED),
    (PublicTerminalState.GOAL_REACHED, PublicTerminalState.GOAL_REACHED),
    (PublicTerminalState.GOAL_NOT_REACHED, PublicTerminalState.GOAL_NOT_REACHED),
)


def _causal_factor_values(
    vector: tuple[bool, bool],
) -> tuple[PublicCausalFactorValue, ...]:
    return tuple(
        PublicCausalFactorValue(factor_id=factor_id, value=value)
        for factor_id, value in zip(_CAUSAL_FACTOR_IDS, vector, strict=True)
    )


def _causal_factor_vector_key(vector: tuple[bool, bool]) -> str:
    return "".join("1" if value else "0" for value in vector)


def _transition_graph() -> PublicTransitionGraph:
    transitions = tuple(
        PublicTransition(
            source_state_id="initial",
            target_state_id=(
                "goal-reached"
                if terminal is PublicTerminalState.GOAL_REACHED
                else "goal-not-reached"
            ),
            exposure=exposure,
            factor_values=_causal_factor_values(vector),
            action_fingerprint_id=(f"action-{exposure.value}-{_causal_factor_vector_key(vector)}"),
            failure_fingerprint_id=(
                None
                if terminal is PublicTerminalState.GOAL_REACHED
                else f"failure-{exposure.value}-{_causal_factor_vector_key(vector)}"
            ),
            trigger=f"Execute the {exposure.value} path for this factor vector.",
        )
        for vector, paired_terminals in zip(
            _CAUSAL_FACTOR_VECTORS,
            _PAIRED_TERMINALS,
            strict=True,
        )
        for exposure, terminal in zip(
            (
                PublicCausalExposure.GUIDANCE_APPLIED,
                PublicCausalExposure.BASELINE_CONTINUED,
            ),
            paired_terminals,
            strict=True,
        )
    )
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-public-transition-graph/v1",
        "initial_state_id": "initial",
        "factors": (
            PublicCausalFactor(
                factor_id=_CAUSAL_FACTOR_IDS[0],
                true_description="The guidance changes the decisive action path.",
                false_description="The decisive action path remains available.",
            ),
            PublicCausalFactor(
                factor_id=_CAUSAL_FACTOR_IDS[1],
                true_description="The uninterrupted baseline can recover the task goal.",
                false_description="The uninterrupted baseline cannot recover the task goal.",
            ),
        ),
        "states": (
            PublicTransitionState(
                state_id="initial",
                description="The action path has not terminated.",
            ),
            PublicTransitionState(
                state_id="goal-reached",
                description="The task goal is reached.",
                terminal=PublicTerminalState.GOAL_REACHED,
            ),
            PublicTransitionState(
                state_id="goal-not-reached",
                description="The task goal is not reached.",
                terminal=PublicTerminalState.GOAL_NOT_REACHED,
            ),
        ),
        "transitions": transitions,
    }
    values["transition_graph_digest"] = transition_graph_digest(values)
    return PublicTransitionGraph.model_validate(values)


def test_review_digest_domains_and_literal_payload_goldens_are_frozen() -> None:
    payload = {
        "schema_version": "digest-probe/v1",
        "value": "alpha",
        "other_digest": "1" * 64,
    }
    cases: tuple[tuple[str, str, Callable[[Mapping[str, object]], str], str], ...] = (
        (
            "checklist_digest",
            REVIEW_CHECKLIST_DIGEST_DOMAIN,
            public_review_checklist_digest,
            "62422ce647ac12e0e6029292df004ecd55b20dc300ca43dca367d807605d141c",
        ),
        (
            "draft_digest",
            REVIEW_DRAFT_DIGEST_DOMAIN,
            review_draft_digest,
            "f81380698506a7d5d0c9d97a4f8ee0692fc797856c06f2281e0566d149f44065",
        ),
        (
            "family_comparison_digest",
            FAMILY_COMPARISON_DIGEST_DOMAIN,
            family_comparison_digest,
            "e0628a5c7f484472222ba387beb434d7eaf84a0aca6ef1acc90b8ab59e6af58d",
        ),
        (
            "submission_digest",
            REVIEW_SUBMISSION_DIGEST_DOMAIN,
            review_submission_digest,
            "899ef8140b415160ea97b95c6c0465e01ec67e9b8f7a4b3f435a02dce8a4a97e",
        ),
        (
            "envelope_digest",
            REVIEW_ENVELOPE_DIGEST_DOMAIN,
            review_envelope_digest,
            "92dd3d676919d4d8e8b971fc2401c9bd0b564ce13cdb86653fb99124b5ba2781",
        ),
        (
            "subreport_digest",
            SUBREPORT_DIGEST_DOMAIN,
            subreport_digest,
            "77bdc6c03bcd35a5e4e5f0906ab0b097fd320cdd21b5b061fb0ee060a7c5bff8",
        ),
        (
            "manifest_digest",
            PACK_MANIFEST_DIGEST_DOMAIN,
            pack_manifest_digest,
            "8ddcbb76dd7f4114f2fc97b5e704da53fdcc4547c09849f3db2cdcd1aa72f6d7",
        ),
    )
    assert tuple(domain for _, domain, _, _ in cases) == (
        "saliencegate:state-decay-v2:public-review:checklist:v1",
        "saliencegate:state-decay-v2:public-review:draft:v1",
        "saliencegate:state-decay-v2:public-review:family-comparison:v1",
        "saliencegate:state-decay-v2:public-review:submission:v1",
        "saliencegate:state-decay-v2:public-review:envelope:v1",
        "saliencegate:state-decay-v2:public-review:subreport:v1",
        "saliencegate:state-decay-v2:public-review:pack-manifest:v1",
    )
    for self_field, _, digest, golden in cases:
        value = {**payload, self_field: "0" * 64}
        assert digest(value) == golden
        assert digest({**value, self_field: "f" * 64}) == golden

    content = b'{"value":"alpha"}\n'
    assert PACK_CHILD_DIGEST_DOMAIN == "saliencegate:state-decay-v2:public-review:pack-child:v1"
    assert pack_child_digest(PublicReviewPackBasename.CHECKLIST, content) == (
        "f051ac82fdbe31fe951fdda4061226974c1ad455f0431d6d35fee4450ca5b02c"
    )
    assert ACCEPTED_ENVELOPE_REGISTRY_DIGEST_DOMAIN == (
        "saliencegate:state-decay-v2:public-review:envelope-registry:v1"
    )
    assert accepted_envelope_registry_digest(content) == (
        "f91a0ddc4149fe7570aec73cdad3d2422fad4071bad5b51f5e0f04a851bf6cb2"
    )


def test_frozen_checklist_is_exact_ordered_and_self_attesting() -> None:
    checklist = PUBLIC_REVIEW_CHECKLIST

    assert PublicReviewChecklist.model_validate_json(canonical_json(checklist)) == checklist
    assert tuple(item.item_id for item in checklist.items) == tuple(PublicReviewChecklistItemId)
    assert tuple(item.text for item in checklist.items) == (
        "The candidate defines one clear lineage-level causal distinction that can be evaluated "
        "without an assigned outcome.",
        "The task template, executable transition graph, evidence topology, raw detector fixture "
        "and expected reference projection, failure mechanism, semantic signature, and four "
        "candidate-owned causal deltas are mutually consistent.",
        "The candidate and all five previews contain no assigned outcome, allocation rank, "
        "scenario ID, oracle branch, or equivalent outcome hint.",
        "The candidate declares an empty derivation-parent tuple.",
        "All five previews faithfully materialize the candidate task template, raw detector "
        "fixture, and four causal deltas and vary by slot only through the frozen global profile "
        "catalog.",
        "The candidate is meaningfully distinct from all 29 siblings shown in the current family "
        "comparison.",
        "I attest that I did not consult or compute an allocation, allocation rank, or assigned "
        "outcome while reviewing this candidate.",
    )

    payload = checklist.model_dump(mode="python")
    payload["items"] = tuple(reversed(payload["items"]))
    payload["checklist_digest"] = public_review_checklist_digest(payload)
    with pytest.raises(ValidationError, match="checklist items"):
        PublicReviewChecklist.model_validate(payload)

    payload = checklist.model_dump(mode="python")
    payload["checklist_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="checklist digest"):
        PublicReviewChecklist.model_validate(payload)


def _semantic_projection(lineage_key: str) -> PublicCandidateSemanticProjection:
    template = OutcomeFreeTaskTemplate.model_validate(
        {
            "event_pool": tuple(
                {
                    "event_id": f"event-{index}",
                    "statement": f"Repository-authored event statement number {index}.",
                }
                for index in range(8)
            ),
            "memory_pool": tuple(
                {
                    "memory_id": f"memory-{index}",
                    "statement": f"Repository-authored memory statement number {index}.",
                    "evidence_event_ids": ("event-0", "event-1", "event-2"),
                    "recorded_event_id": "event-2",
                }
                for index in range(4)
            ),
            "pivot": {
                "event_id": "pivot-1",
                "statement": "Choose exactly one next action.",
            },
            "action_pool": (
                {
                    "action_id": "action-primary",
                    "statement": "Apply the retained constraint.",
                },
                {
                    "action_id": "action-alternate",
                    "statement": "Apply the current alternative.",
                },
            ),
            "adapter": AdapterMetadata(
                adapter_id="state-decay-public-test",
                adapter_version="v1",
                response_profile_id="two-action-choice",
                response_profile_digest="1" * 64,
            ),
        }
    )
    graph = _transition_graph()

    topology_values: dict[str, object] = {
        "schema_version": "state-decay-v2-public-evidence-topology/v1",
        "nodes": (
            {"evidence_id": "retained", "statement": "The retained fact was observed."},
            {"evidence_id": "current", "statement": "The current fact was observed."},
        ),
        "edges": (
            {
                "source_evidence_id": "current",
                "target_evidence_id": "retained",
                "relation": PublicEvidenceRelation.CONTEXTUALIZES,
            },
        ),
    }
    topology_values["evidence_topology_digest"] = evidence_topology_digest(topology_values)
    topology = PublicEvidenceTopology.model_validate(topology_values)

    signature_values: dict[str, object] = {
        "schema_version": "state-decay-v2-public-semantic-signature/v1",
        "concept_ids": ("retained-fact", "current-evidence"),
        "canonical_claims": (
            "The retained fact is visible before the pivot.",
            "Current evidence is visible before the action choice.",
        ),
    }
    signature_values["semantic_signature_digest"] = semantic_signature_digest(signature_values)
    signature = PublicSemanticSignature.model_validate(signature_values)

    deltas: list[CausalSemanticDelta] = []
    for index, word in enumerate(("adjusted", "modified", "reframed", "restated")):
        delta_values: dict[str, object] = {
            "schema_version": "state-decay-v2-public-causal-semantic-delta/v1",
            "delta_index": index,
            "delta_id": f"delta-{lineage_key}-{index}",
            "family": ScenarioFamily.FORGOTTEN_REQUIREMENT,
            "lineage_registry_key": lineage_key,
            "factor_values": _causal_factor_values(_CAUSAL_FACTOR_VECTORS[index]),
            "semantic_replacements": (
                {
                    "template_pointer": "/event_pool/0/statement",
                    "replacement": f"Repository-{word} event statement number 0.",
                },
            ),
            "evidence_replacements": (),
        }
        delta_values["causal_delta_digest"] = causal_delta_digest(delta_values)
        deltas.append(CausalSemanticDelta.model_validate(delta_values))

    return PublicCandidateSemanticProjection(
        task_template=template,
        transition_graph=graph,
        evidence_topology=topology,
        failure_mechanism=PublicFailureMechanism(
            failure_mechanism_id="ignored-decisive-evidence",
            description="The wrong action ignores decisive visible evidence.",
        ),
        semantic_signature=signature,
        causal_deltas=tuple(deltas),
    )


def _family_comparison() -> PublicFamilyComparison:
    entries: list[PublicFamilyComparisonEntry] = []
    for index in range(30):
        lineage_key = f"pub-fr-{index:02d}"
        projection = _semantic_projection(lineage_key)
        entries.append(
            PublicFamilyComparisonEntry(
                lineage_registry_key=lineage_key,
                candidate_packet_digest=f"{1_000 + index:064x}",
                semantic_rationale=f"Distinct synthetic rationale number {index}.",
                semantic_projection=projection,
                transition_graph_digest=projection.transition_graph.transition_graph_digest,
                evidence_topology_digest=projection.evidence_topology.evidence_topology_digest,
                failure_mechanism_id=projection.failure_mechanism.failure_mechanism_id,
                semantic_signature_digest=(projection.semantic_signature.semantic_signature_digest),
                preview_digests=tuple(f"{2_000 + index * 5 + slot:064x}" for slot in range(5)),
            )
        )
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-public-review-family-comparison/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "split": BenchmarkSplit.TRAIN,
        "family": ScenarioFamily.FORGOTTEN_REQUIREMENT,
        "entries": tuple(entries),
    }
    values["family_comparison_digest"] = family_comparison_digest(values)
    return PublicFamilyComparison.model_validate(values)


def test_family_comparison_round_trips_all_thirty_exact_semantic_projections() -> None:
    comparison = _family_comparison()
    assert PublicFamilyComparison.model_validate_json(canonical_json(comparison)) == comparison

    payload = comparison.model_dump(mode="python")
    payload["entries"] = tuple(reversed(payload["entries"]))
    payload["family_comparison_digest"] = family_comparison_digest(payload)
    with pytest.raises(ValidationError, match="canonical and complete"):
        PublicFamilyComparison.model_validate(payload)

    payload = comparison.model_dump(mode="python")
    payload["family_comparison_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="comparison digest"):
        PublicFamilyComparison.model_validate(payload)


def test_family_comparison_rejects_projection_and_delta_coordinate_substitution() -> None:
    comparison = _family_comparison()
    payload = comparison.model_dump(mode="python")
    payload["entries"][0]["transition_graph_digest"] = "a" * 64
    payload["family_comparison_digest"] = family_comparison_digest(payload)
    with pytest.raises(ValidationError, match="projection bindings"):
        PublicFamilyComparison.model_validate(payload)

    payload = comparison.model_dump(mode="python")
    for delta in payload["entries"][0]["semantic_projection"]["causal_deltas"]:
        delta["family"] = ScenarioFamily.FAILED_PRIOR_ATTEMPT
        delta["lineage_registry_key"] = "pub-fp-00"
        delta["causal_delta_digest"] = causal_delta_digest(delta)
    payload["family_comparison_digest"] = family_comparison_digest(payload)
    with pytest.raises(ValidationError, match="causal delta coordinates"):
        PublicFamilyComparison.model_validate(payload)


def test_family_comparison_rejects_noncanonical_projection_and_duplicate_bindings() -> None:
    comparison = _family_comparison()

    payload = comparison.model_dump(mode="python")
    payload["entries"][0]["semantic_projection"]["causal_deltas"] = tuple(
        reversed(payload["entries"][0]["semantic_projection"]["causal_deltas"])
    )
    payload["family_comparison_digest"] = family_comparison_digest(payload)
    with pytest.raises(ValidationError, match="causal deltas are not canonical"):
        PublicFamilyComparison.model_validate(payload)

    payload = comparison.model_dump(mode="python")
    payload["entries"][0]["preview_digests"] = (payload["entries"][0]["preview_digests"][0],) * 5
    payload["family_comparison_digest"] = family_comparison_digest(payload)
    with pytest.raises(ValidationError, match="preview digests must be unique"):
        PublicFamilyComparison.model_validate(payload)

    payload = comparison.model_dump(mode="python")
    payload["entries"][1]["candidate_packet_digest"] = payload["entries"][0][
        "candidate_packet_digest"
    ]
    payload["family_comparison_digest"] = family_comparison_digest(payload)
    with pytest.raises(ValidationError, match="candidate packets must be unique"):
        PublicFamilyComparison.model_validate(payload)

    payload = comparison.model_dump(mode="python")
    payload["split"] = BenchmarkSplit.DEVELOPMENT
    payload["family_comparison_digest"] = family_comparison_digest(payload)
    with pytest.raises(ValidationError, match="split and family"):
        PublicFamilyComparison.model_validate(payload)


def _answers(*, failed_index: int | None = None) -> tuple[PublicReviewChecklistAnswer, ...]:
    return tuple(
        PublicReviewChecklistAnswer(
            item_id=item_id,
            answer=(
                PublicReviewAnswer.FAILED if index == failed_index else PublicReviewAnswer.PASSED
            ),
        )
        for index, item_id in enumerate(PublicReviewChecklistItemId)
    )


def _submission(
    *,
    decision: ReviewDecision = ReviewDecision.ACCEPTED,
    failed_index: int | None = None,
    predecessor: str | None = None,
    review_rationale: str = "Synthetic reviewer rationale for contract tests.",
) -> PublicReviewSubmission:
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-public-review-submission/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "split": BenchmarkSplit.TRAIN,
        "family": ScenarioFamily.FORGOTTEN_REQUIREMENT,
        "lineage_registry_key": "pub-fr-00",
        "candidate_packet_digest": "1" * 64,
        "draft_digest": "2" * 64,
        "checklist_digest": PUBLIC_REVIEW_CHECKLIST.checklist_digest,
        "family_comparison_digest": "3" * 64,
        "profile_catalog_digest": "4" * 64,
        "generator_configuration_digest": "5" * 64,
        "generator_algorithm_digest": "6" * 64,
        "reviewer_id": "synthetic-reviewer",
        "review_rationale": review_rationale,
        "checklist_answers": _answers(failed_index=failed_index),
        "decision": decision,
        "supersedes_submission_digest": predecessor,
    }
    values["submission_digest"] = review_submission_digest(values)
    return PublicReviewSubmission.model_validate(values)


def test_submission_truth_table_bounds_and_self_digest_are_exact() -> None:
    accepted = _submission()
    rejected = _submission(decision=ReviewDecision.REJECTED, failed_index=2)

    assert PublicReviewSubmission.model_validate_json(canonical_json(accepted)) == accepted
    assert PublicReviewSubmission.model_validate_json(canonical_json(rejected)) == rejected
    assert tuple(PublicReviewSubmission.model_fields) == (
        "schema_version",
        "suite_id",
        "suite_version",
        "split",
        "family",
        "lineage_registry_key",
        "candidate_packet_digest",
        "draft_digest",
        "checklist_digest",
        "family_comparison_digest",
        "profile_catalog_digest",
        "generator_configuration_digest",
        "generator_algorithm_digest",
        "reviewer_id",
        "review_rationale",
        "checklist_answers",
        "decision",
        "supersedes_submission_digest",
        "submission_digest",
    )
    assert len(canonical_json(accepted)) + 1 <= 8 * 1024

    for decision, failed_index in (
        (ReviewDecision.ACCEPTED, 2),
        (ReviewDecision.REJECTED, None),
    ):
        values = accepted.model_dump(mode="python")
        values["decision"] = decision
        values["checklist_answers"] = _answers(failed_index=failed_index)
        values["submission_digest"] = review_submission_digest(values)
        with pytest.raises(ValidationError, match="decision"):
            PublicReviewSubmission.model_validate(values)

    values = accepted.model_dump(mode="python")
    values["checklist_answers"] = tuple(reversed(values["checklist_answers"]))
    values["submission_digest"] = review_submission_digest(values)
    with pytest.raises(ValidationError, match="checklist answers"):
        PublicReviewSubmission.model_validate(values)

    values = accepted.model_dump(mode="python")
    values["submission_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="submission digest"):
        PublicReviewSubmission.model_validate(values)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("lineage_registry_key", "pub-fp-00", "coordinates"),
        ("checklist_digest", "a" * 64, "checklist"),
    ),
)
def test_submission_rejects_coordinate_and_checklist_substitution(
    field: str,
    replacement: str,
    message: str,
) -> None:
    payload = _submission().model_dump(mode="python")
    payload[field] = replacement
    payload["submission_digest"] = review_submission_digest(payload)
    with pytest.raises(ValidationError, match=message):
        PublicReviewSubmission.model_validate(payload)


def test_submission_rejects_a_direct_self_predecessor() -> None:
    payload = _submission().model_dump(mode="python")
    payload["supersedes_submission_digest"] = "a" * 64
    payload["submission_digest"] = "a" * 64
    with pytest.raises(ValidationError, match="cannot supersede itself"):
        PublicReviewSubmission.model_validate(payload)


def _draft() -> PublicReviewDraft:
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-public-review-draft/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "split": BenchmarkSplit.TRAIN,
        "family": ScenarioFamily.FORGOTTEN_REQUIREMENT,
        "lineage_registry_key": "pub-fr-00",
        "candidate_packet_digest": "1" * 64,
        "checklist_digest": PUBLIC_REVIEW_CHECKLIST.checklist_digest,
        "preview_digests": tuple(f"{index + 10:064x}" for index in range(5)),
        "family_comparison_digest": "2" * 64,
        "profile_catalog_digest": "3" * 64,
        "generator_configuration_digest": "4" * 64,
        "generator_algorithm_digest": "5" * 64,
    }
    values["draft_digest"] = review_draft_digest(values)
    return PublicReviewDraft.model_validate(values)


def test_review_draft_binds_five_previews_and_every_global_contract() -> None:
    draft = _draft()
    assert PublicReviewDraft.model_validate_json(canonical_json(draft)) == draft

    payload = draft.model_dump(mode="python")
    payload["preview_digests"] = (payload["preview_digests"][0],) * 5
    payload["draft_digest"] = review_draft_digest(payload)
    with pytest.raises(ValidationError, match="preview digests"):
        PublicReviewDraft.model_validate(payload)

    payload = draft.model_dump(mode="python")
    payload["draft_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="draft digest"):
        PublicReviewDraft.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("lineage_registry_key", "pub-fp-00", "coordinates"),
        ("checklist_digest", "a" * 64, "checklist"),
    ),
)
def test_draft_rejects_coordinate_and_checklist_substitution(
    field: str,
    replacement: str,
    message: str,
) -> None:
    payload = _draft().model_dump(mode="python")
    payload[field] = replacement
    payload["draft_digest"] = review_draft_digest(payload)
    with pytest.raises(ValidationError, match=message):
        PublicReviewDraft.model_validate(payload)


def _lineage_record(submission: PublicReviewSubmission) -> LineageReviewRecord:
    public_leaf = derive_seed(PUBLIC_GENERATION_SEED, SeedPurpose.PUBLIC)
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-lineage-review-record/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "split": submission.split,
        "family": submission.family,
        "boundary": ReviewBoundary.PUBLIC,
        "lineage_registry_key": submission.lineage_registry_key,
        "candidate_packet_digest": submission.candidate_packet_digest,
        "independent_seed_commitment_digest": independent_lineage_seed_commitment(
            derive_independent_lineage_seed(
                public_leaf,
                split=submission.split,
                family=submission.family,
                lineage_registry_key=submission.lineage_registry_key,
            )
        ),
        "transition_graph_digest": "7" * 64,
        "evidence_topology_digest": "8" * 64,
        "failure_mechanism_id": "synthetic-failure-mechanism",
        "semantic_signature_digest": "9" * 64,
        "derivation_parent_keys": (),
        "semantic_rationale": "Synthetic candidate rationale for contract tests.",
        "reviewer_id": submission.reviewer_id,
        "review_rationale": submission.review_rationale,
        "decision": submission.decision,
    }
    values["review_digest"] = lineage_review_record_digest(values)
    return LineageReviewRecord.model_validate(values)


def _envelope(submissions: tuple[PublicReviewSubmission, ...]) -> PublicReviewEnvelope:
    head = submissions[-1]
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-public-review-envelope/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "split": head.split,
        "family": head.family,
        "lineage_registry_key": head.lineage_registry_key,
        "candidate_packet_digest": head.candidate_packet_digest,
        "draft_digest": head.draft_digest,
        "checklist_digest": head.checklist_digest,
        "family_comparison_digest": head.family_comparison_digest,
        "profile_catalog_digest": head.profile_catalog_digest,
        "generator_configuration_digest": head.generator_configuration_digest,
        "generator_algorithm_digest": head.generator_algorithm_digest,
        "submissions": submissions,
        "review_record": _lineage_record(head),
    }
    values["envelope_digest"] = review_envelope_digest(values)
    return PublicReviewEnvelope.model_validate(values)


def test_envelope_revalidates_complete_linear_chain_and_record_projection() -> None:
    first = _submission(decision=ReviewDecision.REJECTED, failed_index=4)
    head = _submission(predecessor=first.submission_digest)
    envelope = _envelope((first, head))

    assert PublicReviewEnvelope.model_validate_json(canonical_json(envelope)) == envelope
    assert tuple(PublicReviewEnvelope.model_fields) == (
        "schema_version",
        "suite_id",
        "suite_version",
        "split",
        "family",
        "lineage_registry_key",
        "candidate_packet_digest",
        "draft_digest",
        "checklist_digest",
        "family_comparison_digest",
        "profile_catalog_digest",
        "generator_configuration_digest",
        "generator_algorithm_digest",
        "submissions",
        "review_record",
        "envelope_digest",
    )
    assert len(canonical_json(envelope)) <= 320 * 1024

    values = envelope.model_dump(mode="python")
    values["submissions"][1]["supersedes_submission_digest"] = None
    values["submissions"][1]["submission_digest"] = review_submission_digest(
        values["submissions"][1]
    )
    values["envelope_digest"] = review_envelope_digest(values)
    with pytest.raises(ValidationError, match="submission chain"):
        PublicReviewEnvelope.model_validate(values)

    values = envelope.model_dump(mode="python")
    values["envelope_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="envelope digest"):
        PublicReviewEnvelope.model_validate(values)

    rejected = _submission(decision=ReviewDecision.REJECTED, failed_index=0)
    assert _envelope((rejected,)).review_record.decision is ReviewDecision.REJECTED


def test_envelope_rejects_non_null_root_and_duplicate_digest_cycle() -> None:
    valid = _submission()
    envelope = _envelope((valid,))

    root_payload = valid.model_dump(mode="python")
    root_payload["supersedes_submission_digest"] = "a" * 64
    root_payload["submission_digest"] = review_submission_digest(root_payload)
    non_null_root = PublicReviewSubmission.model_validate(root_payload)
    payload = envelope.model_dump(mode="python")
    payload["submissions"] = (non_null_root.model_dump(mode="python"),)
    payload["envelope_digest"] = review_envelope_digest(payload)
    with pytest.raises(ValidationError, match="begin with a null predecessor"):
        PublicReviewEnvelope.model_validate(payload)

    payload = envelope.model_dump(mode="python")
    payload["submissions"] = (
        valid.model_dump(mode="python"),
        valid.model_dump(mode="python"),
    )
    payload["envelope_digest"] = review_envelope_digest(payload)
    with pytest.raises(ValidationError, match="contains a cycle"):
        PublicReviewEnvelope.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("lineage_registry_key", "pub-fp-00", "coordinates"),
        ("candidate_packet_digest", "a" * 64, "chain bindings"),
        ("checklist_digest", "a" * 64, "checklist"),
        ("profile_catalog_digest", "a" * 64, "chain bindings"),
        ("generator_configuration_digest", "a" * 64, "chain bindings"),
        ("generator_algorithm_digest", "a" * 64, "chain bindings"),
        ("draft_digest", "a" * 64, "head does not match"),
        ("family_comparison_digest", "a" * 64, "head does not match"),
    ),
)
def test_envelope_rejects_top_level_binding_substitution(
    field: str,
    replacement: str,
    message: str,
) -> None:
    payload = _envelope((_submission(),)).model_dump(mode="python")
    payload[field] = replacement
    payload["envelope_digest"] = review_envelope_digest(payload)
    with pytest.raises(ValidationError, match=message):
        PublicReviewEnvelope.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("candidate_packet_digest", "a" * 64),
        ("derivation_parent_keys", ("unexpected-parent",)),
        ("reviewer_id", "different-reviewer"),
        ("review_rationale", "Different safe reviewer rationale."),
        ("decision", ReviewDecision.REJECTED),
    ),
)
def test_envelope_rejects_record_projection_substitution(
    field: str,
    replacement: object,
) -> None:
    payload = _envelope((_submission(),)).model_dump(mode="python")
    payload["review_record"][field] = replacement
    payload["review_record"]["review_digest"] = lineage_review_record_digest(
        payload["review_record"]
    )
    payload["envelope_digest"] = review_envelope_digest(payload)
    with pytest.raises(ValidationError, match="record projection"):
        PublicReviewEnvelope.model_validate(payload)


@pytest.mark.parametrize(
    ("nested_path", "message"),
    (
        (("submissions", 0, "submission_digest"), "submission digest"),
        (("review_record", "review_digest"), "review record digest"),
    ),
)
def test_envelope_revalidates_nested_self_digests(
    nested_path: tuple[object, ...],
    message: str,
) -> None:
    payload = _envelope((_submission(),)).model_dump(mode="python")
    if len(nested_path) == 3:
        collection, index, field = nested_path
        payload[collection][index][field] = "0" * 64
    else:
        collection, field = nested_path
        payload[collection][field] = "0" * 64
    payload["envelope_digest"] = review_envelope_digest(payload)
    with pytest.raises(ValidationError, match=message):
        PublicReviewEnvelope.model_validate(payload)


def test_envelope_revalidates_already_constructed_nested_instances() -> None:
    submission = _submission()
    envelope = _envelope((submission,))

    bad_submission = submission.model_copy(update={"submission_digest": "0" * 64})
    payload = envelope.model_dump(mode="python")
    payload["submissions"] = (bad_submission,)
    payload["envelope_digest"] = review_envelope_digest(payload)
    with pytest.raises(ValidationError, match="submission digest"):
        PublicReviewEnvelope.model_validate(payload)

    bad_record = envelope.review_record.model_copy(update={"review_digest": "0" * 64})
    payload = envelope.model_dump(mode="python")
    payload["review_record"] = bad_record
    payload["envelope_digest"] = review_envelope_digest(payload)
    with pytest.raises(ValidationError, match="review record digest"):
        PublicReviewEnvelope.model_validate(payload)


def test_envelope_allows_historical_draft_and_comparison_bindings() -> None:
    first_payload = _submission(
        decision=ReviewDecision.REJECTED,
        failed_index=0,
    ).model_dump(mode="python")
    first_payload["draft_digest"] = "a" * 64
    first_payload["family_comparison_digest"] = "b" * 64
    first_payload["submission_digest"] = review_submission_digest(first_payload)
    first = PublicReviewSubmission.model_validate(first_payload)
    head = _submission(predecessor=first.submission_digest)

    assert _envelope((first, head)).submissions == (first, head)


@pytest.mark.parametrize(
    "unsafe_rationale",
    (
        "unsafe\nline",
        "spoof\u202e",
        "e\u0301",
        "é" * 2_049,
    ),
)
def test_envelope_rechecks_candidate_rationale_as_review_safe_text(
    unsafe_rationale: str,
) -> None:
    envelope = _envelope((_submission(),))
    payload = envelope.model_dump(mode="python")
    payload["review_record"]["semantic_rationale"] = unsafe_rationale
    payload["review_record"]["review_digest"] = lineage_review_record_digest(
        payload["review_record"]
    )
    payload["envelope_digest"] = review_envelope_digest(payload)
    with pytest.raises(ValidationError, match="semantic rationale is not review-safe"):
        PublicReviewEnvelope.model_validate(payload)


def test_maximal_submission_and_32_link_envelope_fit_frozen_bounds() -> None:
    submissions: list[PublicReviewSubmission] = []
    predecessor: str | None = None
    rationale = "r" * 4_096
    for _ in range(32):
        submission = _submission(
            predecessor=predecessor,
            review_rationale=rationale,
        )
        assert len(canonical_json(submission)) + 1 <= 8 * 1024
        submissions.append(submission)
        predecessor = submission.submission_digest

    envelope = _envelope(tuple(submissions))
    assert len(canonical_json(envelope)) <= 320 * 1024

    payload = envelope.model_dump(mode="python")
    payload["submissions"] = (*payload["submissions"], payload["submissions"][-1])
    payload["envelope_digest"] = review_envelope_digest(payload)
    with pytest.raises(ValidationError):
        PublicReviewEnvelope.model_validate(payload)


def _pack_manifest() -> PublicReviewPackManifest:
    contents = {
        PublicReviewPackBasename.CANDIDATES: b'{"candidate":1}\n',
        PublicReviewPackBasename.DRAFTS: b'{"draft":1}\n',
        PublicReviewPackBasename.FAMILY_COMPARISONS: b'{"comparison":1}\n',
        PublicReviewPackBasename.CHECKLIST: b'{"checklist":1}\n',
        PublicReviewPackBasename.REVIEW_GUIDE: b"# Review guide\n",
    }
    children = tuple(
        PublicReviewPackChild(
            basename=basename,
            canonical_byte_count=len(contents[basename]),
            content_digest=pack_child_digest(basename, contents[basename]),
        )
        for basename in PublicReviewPackBasename
    )
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-public-review-pack/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "generation_contract_digest": GENERATION_CONTRACT.contract_digest,
        "lineage_review_protocol_digest": LINEAGE_REVIEW_PROTOCOL.protocol_digest,
        "generator_configuration_digest": "1" * 64,
        "generator_algorithm_digest": "2" * 64,
        "profile_catalog_digest": "3" * 64,
        "candidate_registry_digest": "4" * 64,
        "checklist_digest": PUBLIC_REVIEW_CHECKLIST.checklist_digest,
        "children": children,
    }
    values["manifest_digest"] = pack_manifest_digest(values)
    return PublicReviewPackManifest.model_validate(values)


def test_pack_manifest_has_exact_child_order_limits_and_self_digest() -> None:
    manifest = _pack_manifest()
    assert PublicReviewPackManifest.model_validate_json(canonical_json(manifest)) == manifest

    payload = manifest.model_dump(mode="python")
    payload["children"] = tuple(reversed(payload["children"]))
    payload["manifest_digest"] = pack_manifest_digest(payload)
    with pytest.raises(ValidationError, match="child order"):
        PublicReviewPackManifest.model_validate(payload)

    payload = manifest.model_dump(mode="python")
    payload["manifest_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="manifest digest"):
        PublicReviewPackManifest.model_validate(payload)

    with pytest.raises(ValidationError, match="role limit"):
        PublicReviewPackChild(
            basename=PublicReviewPackBasename.CHECKLIST,
            canonical_byte_count=256 * 1024 + 1,
            content_digest="0" * 64,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("generation_contract_digest", "a" * 64, "generation contract"),
        ("lineage_review_protocol_digest", "a" * 64, "lineage review protocol"),
        ("checklist_digest", "a" * 64, "checklist"),
    ),
)
def test_pack_manifest_rejects_frozen_protocol_binding_substitution(
    field: str,
    replacement: str,
    message: str,
) -> None:
    payload = _pack_manifest().model_dump(mode="python")
    payload[field] = replacement
    payload["manifest_digest"] = pack_manifest_digest(payload)
    with pytest.raises(ValidationError, match=message):
        PublicReviewPackManifest.model_validate(payload)


def test_review_contract_models_are_strict_frozen_and_forbid_extra_fields() -> None:
    manifest = _pack_manifest()

    with pytest.raises(ValidationError, match="extra"):
        PublicReviewPackManifest.model_validate(
            {**manifest.model_dump(mode="python"), "unexpected": "value"}
        )
    with pytest.raises(ValidationError):
        PublicReviewPackChild(
            basename=PublicReviewPackBasename.CHECKLIST,
            canonical_byte_count="1",  # type: ignore[arg-type]
            content_digest="0" * 64,
        )
    with pytest.raises(ValidationError, match="frozen"):
        manifest.manifest_digest = "0" * 64


def _subreport() -> PublicLineageReviewSubreport:
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-public-lineage-review-subreport/v1",
        "suite_id": "state-decay-v2",
        "suite_version": "v2",
        "scope": (BenchmarkSplit.TRAIN, BenchmarkSplit.DEVELOPMENT),
        "status": "passed",
        "record_count": 180,
        "generation_contract_digest": GENERATION_CONTRACT.contract_digest,
        "lineage_review_protocol_digest": LINEAGE_REVIEW_PROTOCOL.protocol_digest,
        "candidate_registry_digest": "1" * 64,
        "accepted_envelope_registry_digest": "2" * 64,
        "review_record_digests": tuple(f"{index:064x}" for index in range(180)),
    }
    values["subreport_digest"] = subreport_digest(values)
    return PublicLineageReviewSubreport.model_validate(values)


def test_public_subreport_is_complete_ordered_unique_and_self_attesting() -> None:
    subreport = _subreport()
    assert PublicLineageReviewSubreport.model_validate_json(canonical_json(subreport)) == subreport

    payload = subreport.model_dump(mode="python")
    payload["review_record_digests"] = (payload["review_record_digests"][0],) * 180
    payload["subreport_digest"] = subreport_digest(payload)
    with pytest.raises(ValidationError, match="record digests"):
        PublicLineageReviewSubreport.model_validate(payload)

    payload = subreport.model_dump(mode="python")
    payload["subreport_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="subreport digest"):
        PublicLineageReviewSubreport.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        (
            "scope",
            (BenchmarkSplit.DEVELOPMENT, BenchmarkSplit.TRAIN),
            "scope",
        ),
        ("generation_contract_digest", "a" * 64, "generation contract"),
        ("lineage_review_protocol_digest", "a" * 64, "protocol"),
    ),
)
def test_public_subreport_rejects_scope_and_protocol_binding_substitution(
    field: str,
    replacement: object,
    message: str,
) -> None:
    payload = _subreport().model_dump(mode="python")
    payload[field] = replacement
    payload["subreport_digest"] = subreport_digest(payload)
    with pytest.raises(ValidationError, match=message):
        PublicLineageReviewSubreport.model_validate(payload)


def test_digest_helpers_and_hidden_coordinates_fail_closed() -> None:
    with pytest.raises(ValueError, match="digest payload is invalid"):
        public_review_checklist_digest(object())  # type: ignore[arg-type]

    assert (
        review_contract_module._expected_public_split(ScenarioFamily.CONFLICTING_EVIDENCE) is None
    )
    with pytest.raises(ValueError, match="child digest input is invalid"):
        pack_child_digest("checklist.json", b"value")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="child digest input is invalid"):
        pack_child_digest(PublicReviewPackBasename.CHECKLIST, bytearray(b"value"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="registry digest input is invalid"):
        accepted_envelope_registry_digest(bytearray(b"value"))  # type: ignore[arg-type]


def test_canonical_object_and_pack_size_guards_are_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission = _submission(predecessor=None)
    envelope = _envelope((submission,))
    manifest = _pack_manifest()

    monkeypatch.setattr(review_contract_module, "MAX_REVIEW_SUBMISSION_FILE_BYTES", 1)
    with pytest.raises(ValidationError, match="submission exceeds"):
        PublicReviewSubmission.model_validate(submission.model_dump(mode="python"))

    monkeypatch.setattr(
        review_contract_module,
        "MAX_REVIEW_SUBMISSION_FILE_BYTES",
        8 * 1024,
    )
    monkeypatch.setattr(review_contract_module, "MAX_REVIEW_ENVELOPE_CANONICAL_BYTES", 1)
    with pytest.raises(ValidationError, match="envelope exceeds"):
        PublicReviewEnvelope.model_validate(envelope.model_dump(mode="python"))

    monkeypatch.setattr(review_contract_module, "MAX_REVIEW_PACK_MANIFEST_FILE_BYTES", 1)
    with pytest.raises(ValidationError, match="manifest exceeds"):
        PublicReviewPackManifest.model_validate(manifest.model_dump(mode="python"))

    monkeypatch.setattr(
        review_contract_module,
        "MAX_REVIEW_PACK_MANIFEST_FILE_BYTES",
        1024 * 1024,
    )
    monkeypatch.setattr(review_contract_module, "MAX_REVIEW_PACK_TOTAL_BYTES", 1)
    with pytest.raises(ValidationError, match="pack exceeds"):
        PublicReviewPackManifest.model_validate(manifest.model_dump(mode="python"))
