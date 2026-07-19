from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import ValidationError

import saliencegate.experiments.retrieval as retrieval_module
from saliencegate.domain import (
    EvidenceReference,
    EvidenceSource,
    InterventionAction,
    MemoryDelta,
    MemoryKind,
    MemoryRecord,
    PayloadDigest,
    PayloadDigestAlgorithm,
    TrustLabel,
    ValidityState,
    canonical_json,
)
from saliencegate.experiments import (
    RetrievalConfig,
    RetrievalError,
    RetrievalHit,
    RetrievalResult,
    Stage2ConditionId,
    build_retrieval_request,
    resolve_retrieval_config,
    resolve_stage2_condition,
    retrieval_selector_provenance,
    retrieve_candidate_bank,
    validated_retrieval_result,
)
from saliencegate.memory.materialize import (
    MATERIALIZATION_RESULT_SCHEMA_VERSION,
    MaterializedBankOperations,
    _materialization_digest,
)
from saliencegate.ports.trajectory import LogicalMessageRole
from saliencegate.prompts import (
    PAPER_TWO_PHASE_V1,
    BankViewKind,
    PromptContractError,
    build_active_bank_prompt_view,
)
from saliencegate.runtime import (
    MESSAGE_WINDOW_VERSION,
    TASK_DESCRIPTION_VERSION,
    AttestedTaskDescription,
    MessageWindow,
    MessageWindowMessage,
    MessageWindowPayload,
    TrajectoryTextSource,
)

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
OTHER_RUN_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
TASK_EVENT_ID = UUID("10000000-0000-4000-8000-000000000001")
AS_OF = datetime(2026, 7, 13, 8, 0, tzinfo=UTC)
CYCLE_ID = "c" * 64


def _payload_digest(character: str = "a") -> PayloadDigest:
    return PayloadDigest(
        algorithm=PayloadDigestAlgorithm.SYNTHETIC_SHA256,
        value=character * 64,
    )


def _source(event_id: UUID, *, sequence: int = 1) -> TrajectoryTextSource:
    return TrajectoryTextSource(
        evidence=EvidenceReference(
            source=EvidenceSource.EVENT,
            source_id=event_id,
            field_path="/payload/text",
        ),
        event_sequence=sequence,
        ledger_position=sequence,
        trust_label=TrustLabel.SYNTHETIC_FIXTURE,
        payload_digest=_payload_digest("1"),
        record_tag=_payload_digest("2"),
        chain_tag=_payload_digest("3"),
        binding_digest="4" * 64,
    )


def _window(task: str, *messages: str) -> MessageWindow:
    task_source = _source(TASK_EVENT_ID)
    message_items: list[MessageWindowMessage] = []
    sources: list[TrajectoryTextSource] = []
    for index, content in enumerate(messages, start=2):
        event_id = UUID(f"10000000-0000-4000-8000-{index:012d}")
        source = _source(event_id, sequence=index)
        sources.append(source)
        message_items.append(
            MessageWindowMessage(
                role=(LogicalMessageRole.USER if index % 2 == 0 else LogicalMessageRole.ASSISTANT),
                content=content,
                evidence=source.evidence,
                trust_label=source.trust_label,
            )
        )
    payload = MessageWindowPayload(
        version=MESSAGE_WINDOW_VERSION,
        messages=tuple(message_items),
    )
    boundary_id = sources[-1].evidence.source_id if sources else TASK_EVENT_ID
    boundary_sequence = sources[-1].event_sequence if sources else 1
    return MessageWindow(
        version=MESSAGE_WINDOW_VERSION,
        run_id=RUN_ID,
        boundary_event_id=boundary_id,
        boundary_event_sequence=boundary_sequence,
        boundary_ledger_position=boundary_sequence,
        boundary_chain_tag=_payload_digest("3"),
        trajectory_prefix_digest="5" * 64,
        task_description=AttestedTaskDescription(
            version=TASK_DESCRIPTION_VERSION,
            content=task,
            source=task_source,
        ),
        payload=payload,
        payload_canonical_utf8_bytes=len(canonical_json(payload)),
        source_attestations=tuple(sources),
    )


def _record(
    number: int,
    content: str,
    *,
    kind: MemoryKind = MemoryKind.KNOWLEDGE,
    revision: int = 1,
    run_id: UUID = RUN_ID,
    created_at: datetime = AS_OF - timedelta(minutes=10),
    updated_at: datetime | None = None,
    validity: ValidityState = ValidityState.ACTIVE,
    expires_at: datetime | None = None,
) -> MemoryRecord:
    invalidated_at = AS_OF - timedelta(minutes=1) if validity is ValidityState.INVALIDATED else None
    return MemoryRecord(
        memory_id=UUID(f"20000000-0000-4000-8000-{number:012d}"),
        run_id=run_id,
        kind=kind,
        content=content,
        provenance=(
            EvidenceReference(
                source=EvidenceSource.EVENT,
                source_id=TASK_EVENT_ID,
                field_path="/payload/text",
            ),
        ),
        confidence=0.9,
        validity=validity,
        revision=revision,
        created_at=created_at,
        updated_at=updated_at or created_at,
        expires_at=expires_at,
        invalidated_at=invalidated_at,
        trust_label=TrustLabel.SYNTHETIC_FIXTURE,
    )


def _bank(*records: MemoryRecord):
    ordered = tuple(sorted(records, key=lambda item: (item.kind.value, str(item.memory_id))))
    return build_active_bank_prompt_view(
        kind=BankViewKind.CANDIDATE_POST_DELTA,
        run_id=RUN_ID,
        as_of=AS_OF,
        source_projection_digest=_payload_digest("6"),
        records=ordered,
    )


def _materialization(
    *records: MemoryRecord,
    source_last_event_sequence: int = 1,
    source_ledger_position: int | None = None,
    source_ingestion_cursor: int | None = None,
) -> MaterializedBankOperations:
    ordered = tuple(sorted(records, key=lambda item: (item.kind.value, str(item.memory_id))))
    ledger_position = (
        source_last_event_sequence + 1 if source_ledger_position is None else source_ledger_position
    )
    ingestion_cursor = (
        source_last_event_sequence if source_ingestion_cursor is None else source_ingestion_cursor
    )
    values: dict[str, object] = {
        "schema_version": MATERIALIZATION_RESULT_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "source_cycle_id": CYCLE_ID,
        "source_last_event_sequence": source_last_event_sequence,
        "source_ledger_position": ledger_position,
        "source_ingestion_cursor": ingestion_cursor,
        "source_memory_cursor": 0,
        "source_record_tag": _payload_digest("1"),
        "source_chain_tag": _payload_digest("2"),
        "source_projection_digest": _payload_digest("5"),
        "source_operations_digest": "7" * 64,
        "delta": MemoryDelta(
            delta_id=UUID("30000000-0000-4000-8000-000000000001"),
            run_id=RUN_ID,
            created_at=AS_OF,
        ),
        "memory_id_assignments": (),
        "active_bank": ordered,
        "preview_projection_digest": _payload_digest("6"),
    }
    return MaterializedBankOperations(
        **values,
        materialization_digest=_materialization_digest(values),
    )


def _request(window: MessageWindow, *records: MemoryRecord):
    return build_retrieval_request(
        condition=resolve_stage2_condition(Stage2ConditionId.RETRIEVAL_ALWAYS),
        window=window,
        materialization=_materialization(
            *records,
            source_last_event_sequence=window.boundary_event_sequence,
            source_ledger_position=window.boundary_ledger_position + 1,
            source_ingestion_cursor=window.boundary_event_sequence,
        ),
    )


def test_configuration_is_closed_detached_and_content_addressed() -> None:
    first = resolve_retrieval_config()
    second = resolve_retrieval_config()

    assert first == second
    assert first is not second
    assert first.retrieval_version == "candidate-bank-ascii-token-top-k/v1"
    assert first.query_version == "task-latest-eight-ascii-tokens/v1"
    assert first.ranker_version == "ascii-token-overlap/v1"
    assert first.top_k == 2
    assert first.max_query_terms == 4_096
    assert first.configuration_digest == (
        "ecd22fb340646cba2c45f6759a81d1f976fcba81e47a4486713408f5918e8f98"
    )

    payload = json.loads(first.model_dump_json(warnings=False))
    assert type(payload) is dict
    payload.pop("configuration_digest")
    for top_k in (0, 1, 3):
        changed = dict(payload, top_k=top_k)
        with pytest.raises(ValidationError):
            RetrievalConfig.model_validate_json(json.dumps(changed))


def test_query_uses_task_then_logical_messages_casefolds_and_deduplicates_terms() -> None:
    request = _request(
        _window("ALPHA alpha,", "beta! ALPHA"),
        _record(1, "Beta and alpha are both present"),
    )
    result = retrieve_candidate_bank(request)

    assert result.selection.action is InterventionAction.REMIND
    assert result.hits[0].matched_terms == ("alpha", "beta")
    assert result.hits[0].score == 1.0
    assert result.query_digest == "d6ff77dbb91fedb365c66d8915c492a3b37461627e472a910ee2c676d45b4937"


def test_verified_result_has_content_addressed_non_model_provenance() -> None:
    request = _request(_window("alpha"), _record(1, "alpha"))
    result = retrieve_candidate_bank(request)

    provenance = retrieval_selector_provenance(request, result)

    assert provenance.selector_id == request.configuration.retrieval_version
    assert provenance.configuration_digest == request.configuration.configuration_digest
    assert provenance.request_digest == request.request_digest
    assert provenance.result_digest == result.result_digest
    assert provenance.provenance_digest == (
        "058c1c85021e003dca47be0b478684bf5257af400dcd955c402e27150a31961e"
    )

    forged = result.model_copy(update={"result_digest": "0" * 64})
    with pytest.raises(RetrievalError):
        retrieval_selector_provenance(request, forged)


def test_rank_is_score_then_revision_then_uuid_and_top_k_is_exact() -> None:
    records = (
        _record(4, "alpha beta", revision=1),
        _record(3, "alpha", revision=9),
        _record(2, "beta", revision=9),
        _record(1, "alpha", revision=9),
    )
    result = retrieve_candidate_bank(_request(_window("alpha beta"), *records))

    assert len(result.hits) == 2
    assert tuple(hit.memory_id for hit in result.hits) == (
        records[0].memory_id,
        records[3].memory_id,
    )
    assert tuple(hit.rank for hit in result.hits) == (1, 2)
    assert tuple(hit.score for hit in result.hits) == (1.0, 0.5)


def test_uuid_breaks_an_exact_score_and_revision_tie() -> None:
    high_uuid = _record(9, "alpha", revision=3)
    low_uuid = _record(1, "alpha", revision=3)
    result = retrieve_candidate_bank(_request(_window("alpha"), high_uuid, low_uuid))

    assert tuple(hit.memory_id for hit in result.hits) == (
        low_uuid.memory_id,
        high_uuid.memory_id,
    )


@pytest.mark.parametrize("task", ["---", "nothing-matches"])
def test_empty_terms_or_no_candidate_match_produces_canonical_silence(task: str) -> None:
    result = retrieve_candidate_bank(_request(_window(task), _record(1, "alpha")))

    assert result.hits == ()
    assert result.selection.action is InterventionAction.SILENCE
    assert result.selection.claims == ()
    assert result.selection.confidence == 1.0


def test_empty_candidate_bank_produces_silence() -> None:
    result = retrieve_candidate_bank(_request(_window("alpha")))
    assert result.hits == ()
    assert result.selection.action is InterventionAction.SILENCE


def test_ascii_tokenizer_is_runtime_independent_and_does_not_use_substring_matches() -> None:
    substring = retrieve_candidate_bank(_request(_window("alpha"), _record(1, "alphabet")))
    unicode_only = retrieve_candidate_bank(
        _request(_window("\U00011f02"), _record(1, "\U00011f02"))
    )

    assert substring.selection.action is InterventionAction.SILENCE
    assert unicode_only.selection.action is InterventionAction.SILENCE


def test_query_term_ceiling_rejects_cpu_amplification_before_ranking() -> None:
    oversized_terms = " ".join(f"t{index}" for index in range(4_097))
    with pytest.raises(RetrievalError):
        _request(_window(oversized_terms), _record(1, "t1"))


def test_request_builds_the_fixed_step_intervention_within_the_shared_payload_budget() -> None:
    request = _request(_window("alpha", "beta"), _record(1, "alpha beta"))
    prompt = PAPER_TWO_PHASE_V1.build_intervention(
        window=request.window,
        bank=request.candidate_bank,
    )
    condition = resolve_stage2_condition(Stage2ConditionId.RETRIEVAL_ALWAYS)

    assert (
        len(canonical_json(prompt.request_payload))
        <= condition.shared_controls.prompt_context_budget_utf8_bytes
    )


def test_request_rejects_an_intervention_payload_above_the_fixed_step_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = resolve_stage2_condition(
        Stage2ConditionId.RETRIEVAL_ALWAYS
    ).shared_controls.prompt_context_budget_utf8_bytes

    class OversizedPromptBundle:
        @staticmethod
        def build_intervention(**_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(request_payload="x" * (budget + 1))

    monkeypatch.setattr(retrieval_module, "PAPER_TWO_PHASE_V1", OversizedPromptBundle())

    with pytest.raises(RetrievalError):
        _request(_window("alpha"), _record(1, "alpha"))


@pytest.mark.parametrize(
    ("kind", "expected_claim_kind"),
    [
        (MemoryKind.KNOWLEDGE, "environment_fact"),
        (MemoryKind.PROCEDURAL, "diagnosis"),
        (MemoryKind.PRIVATE_STATUS, "open_subgoal"),
    ],
)
def test_hit_to_claim_mapping_is_mechanical_and_cites_exact_current_revision(
    kind: MemoryKind, expected_claim_kind: str
) -> None:
    record = _record(1, "alpha", kind=kind, revision=7)
    result = retrieve_candidate_bank(_request(_window("alpha"), record))
    claim = result.selection.claims[0]

    assert claim.kind.value == expected_claim_kind
    assert claim.evidence.source is EvidenceSource.MEMORY
    assert claim.evidence.source_id == record.memory_id
    assert claim.evidence.revision == 7
    assert claim.evidence.field_path == "/content"
    assert claim.evidence.span is None


def test_candidate_bank_can_include_a_memory_created_by_phase_one_in_the_same_cycle() -> None:
    same_cycle = _record(
        1,
        "new diagnosis alpha",
        kind=MemoryKind.PROCEDURAL,
        created_at=AS_OF,
        updated_at=AS_OF,
    )
    request = _request(_window("alpha"), same_cycle)
    result = retrieve_candidate_bank(request)

    assert result.hits[0].memory_id == same_cycle.memory_id
    assert result.candidate_bank_view_digest == request.candidate_bank.view_digest


@pytest.mark.parametrize(
    "record",
    [
        _record(1, "alpha", validity=ValidityState.INVALIDATED),
        _record(1, "alpha", validity=ValidityState.SUPERSEDED),
        _record(1, "alpha", validity=ValidityState.EXPIRED),
        _record(1, "alpha", expires_at=AS_OF),
        _record(1, "alpha", run_id=OTHER_RUN_ID),
    ],
)
def test_inactive_expired_or_cross_run_memory_cannot_enter_the_candidate_bank(
    record: MemoryRecord,
) -> None:
    with pytest.raises(PromptContractError):
        _bank(record)


def test_future_updated_memory_is_rejected_even_if_the_upstream_view_accepts_it() -> None:
    future = _record(1, "alpha", updated_at=AS_OF + timedelta(seconds=1))
    with pytest.raises(ValidationError):
        _materialization(future)


def test_request_binds_window_sequence_ledger_and_ingestion_to_the_materialization() -> None:
    window = _window("alpha")
    record = _record(1, "alpha")
    valid = _request(window, record)

    assert valid.window.boundary_event_sequence == valid.materialization.source_last_event_sequence
    assert valid.window.boundary_ledger_position < valid.materialization.source_ledger_position
    assert (
        valid.materialization.source_ingestion_cursor
        == valid.materialization.source_last_event_sequence
    )

    mismatches = (
        _materialization(
            record,
            source_last_event_sequence=2,
            source_ledger_position=3,
            source_ingestion_cursor=2,
        ),
        _materialization(
            record,
            source_last_event_sequence=1,
            source_ledger_position=1,
            source_ingestion_cursor=1,
        ),
        _materialization(
            record,
            source_last_event_sequence=1,
            source_ledger_position=2,
            source_ingestion_cursor=2,
        ),
    )
    condition = resolve_stage2_condition(Stage2ConditionId.RETRIEVAL_ALWAYS)
    for materialization in mismatches:
        with pytest.raises(RetrievalError):
            build_retrieval_request(
                condition=condition,
                window=window,
                materialization=materialization,
            )


def test_duplicate_memory_identity_is_rejected_instead_of_selecting_a_stale_revision() -> None:
    current = _record(1, "alpha", revision=2)
    stale = _record(1, "alpha stale", revision=1)
    with pytest.raises(PromptContractError):
        _bank(current, stale)


def test_request_requires_retrieval_condition_and_candidate_post_delta_view() -> None:
    materialization = _materialization(_record(1, "alpha"))
    current_bank = build_active_bank_prompt_view(
        kind=BankViewKind.CURRENT,
        run_id=RUN_ID,
        as_of=AS_OF,
        source_projection_digest=_payload_digest("6"),
        records=(_record(1, "alpha"),),
    )
    with pytest.raises(RetrievalError):
        build_retrieval_request(
            condition=resolve_stage2_condition(Stage2ConditionId.FIXED_STEP),
            window=_window("alpha"),
            materialization=materialization,
        )
    valid = _request(_window("alpha"), _record(1, "alpha"))
    relabelled = valid.model_copy(update={"candidate_bank": current_bank})
    with pytest.raises(RetrievalError):
        retrieve_candidate_bank(relabelled)


def test_request_rejects_a_self_consistent_subset_of_the_phase_one_materialization() -> None:
    first = _record(1, "alpha")
    second = _record(2, "alpha")
    request = _request(_window("alpha"), first, second)
    subset = _bank(first)

    forged = request.model_copy(update={"candidate_bank": subset})
    with pytest.raises(RetrievalError):
        retrieve_candidate_bank(forged)


def test_repeated_execution_and_authoritative_verification_are_byte_identical() -> None:
    request = _request(_window("alpha beta", "beta"), _record(1, "alpha beta"))
    first = retrieve_candidate_bank(request)
    second = retrieve_candidate_bank(request)

    assert canonical_json(first) == canonical_json(second)
    assert validated_retrieval_result(request, first) == first


def test_verifier_recomputes_ranking_instead_of_trusting_self_consistent_tampering() -> None:
    request = _request(_window("alpha beta"), _record(1, "alpha beta"))
    result = retrieve_candidate_bank(request)
    payload = json.loads(result.model_dump_json(warnings=False))
    assert type(payload) is dict
    hits = payload["hits"]
    assert type(hits) is list
    first_hit = hits[0]
    assert type(first_hit) is dict
    first_hit["score"] = 0.5
    first_hit.pop("hit_digest")
    payload.pop("result_digest")
    self_consistent = RetrievalResult.model_validate_json(json.dumps(payload))

    with pytest.raises(RetrievalError):
        validated_retrieval_result(request, self_consistent)


@pytest.mark.parametrize(
    "field",
    [
        "request_digest",
        "window_digest",
        "candidate_bank_view_digest",
        "materialization_digest",
        "query_digest",
        "result_digest",
    ],
)
def test_result_digest_or_source_binding_tampering_is_rejected(field: str) -> None:
    request = _request(_window("alpha"), _record(1, "alpha"))
    result = retrieve_candidate_bank(request)
    payload = json.loads(result.model_dump_json(warnings=False))
    assert type(payload) is dict
    payload[field] = "0" * 64

    with pytest.raises((RetrievalError, ValidationError)):
        candidate = RetrievalResult.model_validate_json(json.dumps(payload))
        validated_retrieval_result(request, candidate)


def test_request_rejects_a_tampered_condition_digest_without_echoing_data() -> None:
    request = _request(_window("alpha"), _record(1, "alpha"))
    forged = request.model_copy(update={"condition_digest": "0" * 64})
    with pytest.raises(RetrievalError) as error:
        retrieve_candidate_bank(forged)
    assert "alpha" not in str(error.value)
    assert error.value.__cause__ is None


def test_request_rejects_a_tampered_request_digest() -> None:
    request = _request(_window("alpha"), _record(1, "alpha"))
    forged = request.model_copy(update={"request_digest": "0" * 64})

    with pytest.raises(RetrievalError):
        retrieve_candidate_bank(forged)


def test_request_digest_fails_closed_for_an_impossible_materialization_shape() -> None:
    with pytest.raises(TypeError, match="materialization digest source is invalid"):
        retrieval_module._request_digest(
            {
                "run_id": RUN_ID,
                "as_of": AS_OF,
                "materialization": object(),
            }
        )


def test_public_retrieval_boundaries_reject_untyped_objects() -> None:
    condition = resolve_stage2_condition(Stage2ConditionId.RETRIEVAL_ALWAYS)
    window = _window("alpha")
    materialization = _materialization(_record(1, "alpha"))
    request = _request(window, _record(1, "alpha"))

    with pytest.raises(RetrievalError):
        build_retrieval_request(
            condition=object(),
            window=window,
            materialization=materialization,
        )
    with pytest.raises(RetrievalError):
        build_retrieval_request(
            condition=condition,
            window=window,
            materialization=object(),
        )
    with pytest.raises(RetrievalError):
        retrieve_candidate_bank(object())
    with pytest.raises(RetrievalError):
        validated_retrieval_result(request, object())


@pytest.mark.parametrize("tampering", ["duplicate_terms", "hit_digest"])
def test_hit_rejects_noncanonical_terms_or_digest(tampering: str) -> None:
    result = retrieve_candidate_bank(_request(_window("alpha"), _record(1, "alpha")))
    hit = json.loads(result.hits[0].model_dump_json(warnings=False))
    assert type(hit) is dict
    if tampering == "duplicate_terms":
        hit["matched_terms"] = ["alpha", "alpha"]
        hit.pop("hit_digest")
    else:
        hit["hit_digest"] = "0" * 64

    with pytest.raises(ValidationError):
        RetrievalHit.model_validate_json(json.dumps(hit))


@pytest.mark.parametrize(
    "matched_terms",
    [
        ("Alpha",),
        ("alpha-beta",),
        ("café",),
        ("raw memory secret\nsecond line",),
        tuple(f"t{index}" for index in range(4_097)),
        ("a" * (resolve_retrieval_config().max_query_utf8_bytes + 1),),
    ],
)
def test_hit_rejects_non_ascii_unbounded_or_non_token_matched_terms(
    matched_terms: tuple[str, ...],
) -> None:
    result = retrieve_candidate_bank(_request(_window("alpha"), _record(1, "alpha")))
    hit = json.loads(result.hits[0].model_dump_json(warnings=False))
    assert type(hit) is dict
    hit["matched_terms"] = matched_terms
    hit.pop("hit_digest")

    with pytest.raises(ValidationError):
        RetrievalHit.model_validate_json(json.dumps(hit))


def test_result_rejects_noncontiguous_ranks_and_duplicate_memory_ids() -> None:
    request = _request(
        _window("alpha"),
        _record(1, "alpha"),
        _record(2, "alpha"),
    )
    result = retrieve_candidate_bank(request)

    noncontiguous = json.loads(result.model_dump_json(warnings=False))
    assert type(noncontiguous) is dict
    noncontiguous["hits"][0]["rank"] = 2
    noncontiguous["hits"][0].pop("hit_digest")
    noncontiguous.pop("result_digest")
    with pytest.raises(ValidationError):
        RetrievalResult.model_validate_json(json.dumps(noncontiguous))

    duplicate = json.loads(result.model_dump_json(warnings=False))
    assert type(duplicate) is dict
    duplicate["hits"][1]["memory_id"] = duplicate["hits"][0]["memory_id"]
    duplicate["hits"][1].pop("hit_digest")
    duplicate.pop("result_digest")
    with pytest.raises(ValidationError):
        RetrievalResult.model_validate_json(json.dumps(duplicate))


def test_result_rejects_selection_cardinality_and_exact_citation_tampering() -> None:
    request = _request(_window("alpha"), _record(1, "alpha"))
    result = retrieve_candidate_bank(request)
    silence = retrieve_candidate_bank(_request(_window("nothing"), _record(1, "alpha")))

    empty_with_reminder = json.loads(result.model_dump_json(warnings=False))
    assert type(empty_with_reminder) is dict
    empty_with_reminder["hits"] = []
    empty_with_reminder.pop("result_digest")
    with pytest.raises(ValidationError):
        RetrievalResult.model_validate_json(json.dumps(empty_with_reminder))

    hit_with_silence = json.loads(result.model_dump_json(warnings=False))
    assert type(hit_with_silence) is dict
    hit_with_silence["selection"] = json.loads(silence.selection.model_dump_json(warnings=False))
    hit_with_silence.pop("result_digest")
    with pytest.raises(ValidationError):
        RetrievalResult.model_validate_json(json.dumps(hit_with_silence))

    wrong_citation = json.loads(result.model_dump_json(warnings=False))
    assert type(wrong_citation) is dict
    wrong_citation["selection"]["claims"][0]["evidence"]["revision"] = 2
    wrong_citation.pop("result_digest")
    with pytest.raises(ValidationError):
        RetrievalResult.model_validate_json(json.dumps(wrong_citation))
