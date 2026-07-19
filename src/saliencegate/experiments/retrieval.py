from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.domain import (
    EvidenceReference,
    EvidenceSource,
    InterventionAction,
    MemoryKind,
    MemoryRecord,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.domain.records import (
    UUID4,
    PositiveSigned64Offset,
    Sha256Digest,
    UnitInterval,
    UtcDatetime,
)
from saliencegate.intervention import DeterministicSelectorProvenance, ProposedClaim
from saliencegate.memory.materialize import MaterializedBankOperations
from saliencegate.memory.proposals import (
    INTERVENTION_OUTPUT_SCHEMA_VERSION,
    InterventionSelectionOutput,
)
from saliencegate.prompts import (
    PAPER_TWO_PHASE_V1,
    ActiveBankPromptView,
    BankViewKind,
    build_active_bank_prompt_view,
)
from saliencegate.prompts.contracts import MAX_PROMPT_PAYLOAD_BYTES
from saliencegate.runtime.message_window import MessageWindow

from .conditions import (
    ResolvedStage2Condition,
    Stage2ConditionId,
    Stage2RetrievalControls,
    resolve_stage2_condition,
)

RETRIEVAL_CONFIG_SCHEMA_VERSION: Literal["stage2-retrieval-controls/v1"] = (
    "stage2-retrieval-controls/v1"
)
RETRIEVAL_REQUEST_SCHEMA_VERSION: Literal["retrieval-request/v1"] = "retrieval-request/v1"
RETRIEVAL_HIT_SCHEMA_VERSION: Literal["retrieval-hit/v1"] = "retrieval-hit/v1"
RETRIEVAL_RESULT_SCHEMA_VERSION: Literal["retrieval-result/v1"] = "retrieval-result/v1"
RETRIEVAL_VERSION: Literal["candidate-bank-ascii-token-top-k/v1"] = (
    "candidate-bank-ascii-token-top-k/v1"
)
RETRIEVAL_QUERY_VERSION: Literal["task-latest-eight-ascii-tokens/v1"] = (
    "task-latest-eight-ascii-tokens/v1"
)

_REQUEST_DIGEST_DOMAIN = "saliencegate:experiments:retrieval-request:v1"
_QUERY_DIGEST_DOMAIN = "saliencegate:experiments:retrieval-query:ascii-token-v1"
_HIT_DIGEST_DOMAIN = "saliencegate:experiments:retrieval-hit:v1"
_RESULT_DIGEST_DOMAIN = "saliencegate:experiments:retrieval-result:v1"
_ASCII_LOWER_TRANSLATION = bytes.maketrans(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    b"abcdefghijklmnopqrstuvwxyz",
)
_ASCII_WORD = re.compile(rb"[a-z0-9_]+")


class RetrievalError(ValueError):
    """A value-free failure at the deterministic retrieval boundary."""

    def __init__(self) -> None:
        super().__init__("offline experiment retrieval failed validation")


class _RetrievalModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


RetrievalConfig = Stage2RetrievalControls


def resolve_retrieval_config() -> RetrievalConfig:
    condition = resolve_stage2_condition(Stage2ConditionId.RETRIEVAL_ALWAYS)
    return RetrievalConfig.model_validate_json(
        condition.shared_controls.retrieval.model_dump_json(warnings=False)
    )


_RETRIEVAL_CONFIG = resolve_retrieval_config()
_CLAIM_KIND_BY_MEMORY_KIND = {
    item.memory_kind: item.claim_kind for item in _RETRIEVAL_CONFIG.claim_kind_mapping
}


def _query_material(window: MessageWindow) -> tuple[dict[str, object], str]:
    contents = (
        window.task_description.content,
        *(message.content for message in window.payload.messages),
    )
    return (
        {
            "query_version": RETRIEVAL_QUERY_VERSION,
            "task_description": contents[0],
            "logical_messages": contents[1:],
        },
        "\n".join(contents),
    )


def _ascii_terms(value: str) -> tuple[str, ...]:
    encoded = value.encode("utf-8", errors="strict").translate(_ASCII_LOWER_TRANSLATION)
    return tuple(dict.fromkeys(match.decode("ascii") for match in _ASCII_WORD.findall(encoded)))


def _query_digest(window: MessageWindow) -> str:
    material, _query = _query_material(window)
    return length_prefixed_sha256(canonical_json(material), domain=_QUERY_DIGEST_DOMAIN)


def _request_digest(values: Mapping[str, object]) -> str:
    run_id = values["run_id"]
    as_of = values["as_of"]
    material = {
        key: value
        for key, value in values.items()
        if key not in {"request_digest", "run_id", "as_of", "materialization"}
    }
    material["run_id"] = str(run_id)
    material["as_of"] = (
        as_of.isoformat().replace("+00:00", "Z") if isinstance(as_of, datetime) else as_of
    )
    source = values["materialization"]
    if isinstance(source, MaterializedBankOperations):
        material["materialization_digest"] = source.materialization_digest
    elif isinstance(source, Mapping):
        material["materialization_digest"] = source["materialization_digest"]
    else:
        raise TypeError("retrieval request materialization digest source is invalid")
    return length_prefixed_sha256(canonical_json(material), domain=_REQUEST_DIGEST_DOMAIN)


class RetrievalRequest(_RetrievalModel):
    """Exact post-Phase-1 bank and attested window consumed by retrieval."""

    schema_version: Literal["retrieval-request/v1"] = RETRIEVAL_REQUEST_SCHEMA_VERSION
    condition_id: Literal[Stage2ConditionId.RETRIEVAL_ALWAYS]
    condition_digest: Sha256Digest
    run_id: UUID4
    cycle_id: Sha256Digest
    as_of: UtcDatetime
    window: MessageWindow = Field(repr=False)
    materialization: MaterializedBankOperations = Field(repr=False)
    candidate_bank: ActiveBankPromptView = Field(repr=False)
    configuration: RetrievalConfig
    request_digest: Sha256Digest = Field(default_factory=_request_digest)

    @model_validator(mode="after")
    def inputs_are_exact_and_post_delta(self) -> Self:
        condition = resolve_stage2_condition(Stage2ConditionId.RETRIEVAL_ALWAYS)
        try:
            exact_window = MessageWindow.model_validate_json(
                self.window.model_dump_json(warnings=False)
            )
            exact_bank = ActiveBankPromptView.model_validate_json(
                self.candidate_bank.model_dump_json(warnings=False)
            )
            exact_config = RetrievalConfig.model_validate_json(
                self.configuration.model_dump_json(warnings=False)
            )
            exact_materialization = MaterializedBankOperations.model_validate_json(
                self.materialization.model_dump_json(warnings=False)
            )
            records = tuple(item.to_memory_record() for item in exact_bank.records)
            _material, query = _query_material(exact_window)
            query_size = len(query.encode("utf-8", errors="strict"))
            query_terms = _ascii_terms(query)
            intervention_prompt = PAPER_TWO_PHASE_V1.build_intervention(
                window=exact_window,
                bank=exact_bank,
            )
            intervention_payload_size = len(canonical_json(intervention_prompt.request_payload))
        except Exception:
            raise ValueError("retrieval request inputs failed exact validation") from None
        if (
            self.condition_digest != condition.condition_digest
            or exact_window != self.window
            or exact_bank != self.candidate_bank
            or exact_config != self.configuration
            or exact_config != _RETRIEVAL_CONFIG
            or exact_materialization != self.materialization
            or self.window.run_id != self.run_id
            or self.window.boundary_event_sequence
            != self.materialization.source_last_event_sequence
            or self.window.boundary_ledger_position >= self.materialization.source_ledger_position
            or self.candidate_bank.run_id != self.run_id
            or self.candidate_bank.kind is not BankViewKind.CANDIDATE_POST_DELTA
            or self.candidate_bank.as_of != self.as_of
            or query_size > self.configuration.max_query_utf8_bytes
            or len(query_terms) > self.configuration.max_query_terms
            or intervention_payload_size
            > condition.shared_controls.prompt_context_budget_utf8_bytes
            or len(canonical_json(exact_bank)) > MAX_PROMPT_PAYLOAD_BYTES
            or any(record.updated_at > self.as_of for record in records)
            or self.materialization.run_id != self.run_id
            or self.materialization.source_cycle_id != self.cycle_id
            or self.materialization.source_ingestion_cursor
            != self.materialization.source_last_event_sequence
            or self.materialization.delta.created_at != self.as_of
            or self.materialization.preview_projection_digest
            != self.candidate_bank.source_projection_digest
            or self.materialization.active_bank != records
        ):
            raise ValueError("retrieval request is not bound to the candidate post-delta view")
        actual = self.model_dump(mode="json", exclude={"request_digest"}, warnings=False)
        if self.request_digest != _request_digest(actual):
            raise ValueError("retrieval request digest does not match")
        return self


def build_retrieval_request(
    *,
    condition: object,
    window: MessageWindow,
    materialization: object,
) -> RetrievalRequest:
    try:
        if type(condition) is not ResolvedStage2Condition:
            raise TypeError
        exact = ResolvedStage2Condition.model_validate_json(
            condition.model_dump_json(warnings=False)
        )
        if exact != condition or exact.condition_id is not Stage2ConditionId.RETRIEVAL_ALWAYS:
            raise ValueError
        if type(materialization) is not MaterializedBankOperations:
            raise TypeError
        exact_materialization = MaterializedBankOperations.model_validate_json(
            materialization.model_dump_json(warnings=False)
        )
        candidate_bank = build_active_bank_prompt_view(
            kind=BankViewKind.CANDIDATE_POST_DELTA,
            run_id=exact_materialization.run_id,
            as_of=exact_materialization.delta.created_at,
            source_projection_digest=exact_materialization.preview_projection_digest,
            records=exact_materialization.active_bank,
        )
        return RetrievalRequest(
            condition_id=Stage2ConditionId.RETRIEVAL_ALWAYS,
            condition_digest=exact.condition_digest,
            run_id=exact_materialization.run_id,
            cycle_id=exact_materialization.source_cycle_id,
            as_of=exact_materialization.delta.created_at,
            window=window,
            materialization=exact_materialization,
            candidate_bank=candidate_bank,
            configuration=resolve_retrieval_config(),
        )
    except Exception:
        raise RetrievalError() from None


def _hit_digest(values: Mapping[str, object]) -> str:
    material = {key: value for key, value in values.items() if key != "hit_digest"}
    material["memory_id"] = str(values["memory_id"])
    return length_prefixed_sha256(canonical_json(material), domain=_HIT_DIGEST_DOMAIN)


class RetrievalHit(_RetrievalModel):
    schema_version: Literal["retrieval-hit/v1"] = RETRIEVAL_HIT_SCHEMA_VERSION
    memory_id: UUID4
    revision: PositiveSigned64Offset
    memory_kind: MemoryKind
    score: UnitInterval
    matched_terms: Annotated[
        tuple[str, ...],
        Field(min_length=1, max_length=_RETRIEVAL_CONFIG.max_query_terms, repr=False),
    ]
    rank: Annotated[int, Field(ge=1, le=2)]
    hit_digest: Sha256Digest = Field(default_factory=_hit_digest)

    @model_validator(mode="after")
    def terms_rank_and_digest_are_canonical(self) -> Self:
        if any(type(term) is not str for term in self.matched_terms):
            raise ValueError("retrieval hit terms are not exact and unique")
        try:
            encoded_terms = tuple(
                term.encode("ascii", errors="strict") for term in self.matched_terms
            )
        except UnicodeEncodeError:
            raise ValueError("retrieval hit terms are not exact and unique") from None
        if (
            len(set(self.matched_terms)) != len(self.matched_terms)
            or any(_ASCII_WORD.fullmatch(term) is None for term in encoded_terms)
            or sum(len(term) for term in encoded_terms) > _RETRIEVAL_CONFIG.max_query_utf8_bytes
        ):
            raise ValueError("retrieval hit terms are not exact and unique")
        actual = self.model_dump(mode="json", exclude={"hit_digest"}, warnings=False)
        if self.hit_digest != _hit_digest(actual):
            raise ValueError("retrieval hit digest does not match")
        return self


def _result_digest(values: Mapping[str, object]) -> str:
    material = {key: value for key, value in values.items() if key != "result_digest"}
    return length_prefixed_sha256(canonical_json(material), domain=_RESULT_DIGEST_DOMAIN)


class RetrievalResult(_RetrievalModel):
    schema_version: Literal["retrieval-result/v1"] = RETRIEVAL_RESULT_SCHEMA_VERSION
    request_digest: Sha256Digest
    window_digest: Sha256Digest
    candidate_bank_view_digest: Sha256Digest
    materialization_digest: Sha256Digest
    query_digest: Sha256Digest
    hits: Annotated[tuple[RetrievalHit, ...], Field(max_length=2)]
    selection: InterventionSelectionOutput
    result_digest: Sha256Digest = Field(default_factory=_result_digest)

    @model_validator(mode="after")
    def ordered_hits_and_selection_match(self) -> Self:
        if tuple(hit.rank for hit in self.hits) != tuple(range(1, len(self.hits) + 1)):
            raise ValueError("retrieval hit ranks are not contiguous")
        if len({hit.memory_id for hit in self.hits}) != len(self.hits):
            raise ValueError("retrieval hits are not unique")
        if not self.hits:
            if (
                self.selection.action is not InterventionAction.SILENCE
                or self.selection.claims
                or self.selection.confidence != 1.0
            ):
                raise ValueError("empty retrieval must select canonical silence")
        else:
            if (
                self.selection.action is not InterventionAction.REMIND
                or len(self.selection.claims) != len(self.hits)
                or self.selection.confidence != 1.0
            ):
                raise ValueError("retrieval hits must select one claim per hit")
            for hit, claim in zip(self.hits, self.selection.claims, strict=True):
                evidence = claim.evidence
                if (
                    claim.kind is not _CLAIM_KIND_BY_MEMORY_KIND[hit.memory_kind]
                    or evidence.source is not EvidenceSource.MEMORY
                    or evidence.source_id != hit.memory_id
                    or evidence.revision != hit.revision
                    or evidence.field_path != "/content"
                    or evidence.span is not None
                ):
                    raise ValueError("retrieval claim does not cite its exact hit")
        actual = self.model_dump(mode="json", exclude={"result_digest"}, warnings=False)
        if self.result_digest != _result_digest(actual):
            raise ValueError("retrieval result digest does not match")
        return self


def _validated_request(value: object) -> RetrievalRequest:
    if type(value) is not RetrievalRequest:
        raise RetrievalError() from None
    try:
        exact = RetrievalRequest.model_validate_json(value.model_dump_json(warnings=False))
        return exact
    except Exception:
        raise RetrievalError() from None


def _query_terms(window: MessageWindow) -> tuple[str, ...]:
    _material, query = _query_material(window)
    return _ascii_terms(query)


def retrieve_candidate_bank(request: object) -> RetrievalResult:
    """Select deterministic top-k claims from the exact post-Phase-1 active bank."""

    checked = _validated_request(request)
    terms = _query_terms(checked.window)
    ranked: list[tuple[int, int, str, MemoryRecord, tuple[str, ...]]] = []
    if terms:
        for quoted in checked.candidate_bank.records:
            record = quoted.to_memory_record()
            content_terms = set(_ascii_terms(record.content))
            matched = tuple(term for term in terms if term in content_terms)
            if matched:
                ranked.append(
                    (-len(matched), -record.revision, str(record.memory_id), record, matched)
                )
    ranked.sort(key=lambda item: item[:3])
    selected = ranked[: checked.configuration.top_k]
    hits: list[RetrievalHit] = []
    claims: list[ProposedClaim] = []
    for rank, (_negative_matches, _negative_revision, _memory_id, value, matched) in enumerate(
        selected, start=1
    ):
        record = value
        hit = RetrievalHit(
            memory_id=record.memory_id,
            revision=record.revision,
            memory_kind=record.kind,
            score=len(matched) / len(terms),
            matched_terms=matched,
            rank=rank,
        )
        hits.append(hit)
        claims.append(
            ProposedClaim(
                kind=_CLAIM_KIND_BY_MEMORY_KIND[record.kind],
                evidence=EvidenceReference(
                    source=EvidenceSource.MEMORY,
                    source_id=record.memory_id,
                    revision=record.revision,
                    field_path="/content",
                ),
            )
        )
    selection = InterventionSelectionOutput(
        schema_version=INTERVENTION_OUTPUT_SCHEMA_VERSION,
        action=InterventionAction.REMIND if claims else InterventionAction.SILENCE,
        claims=tuple(claims),
        confidence=1.0,
    )
    return RetrievalResult(
        request_digest=checked.request_digest,
        window_digest=checked.window.window_digest,
        candidate_bank_view_digest=checked.candidate_bank.view_digest,
        materialization_digest=checked.materialization.materialization_digest,
        query_digest=_query_digest(checked.window),
        hits=tuple(hits),
        selection=selection,
    )


def validated_retrieval_result(
    request: object,
    result: object,
) -> RetrievalResult:
    checked = _validated_request(request)
    try:
        if type(result) is not RetrievalResult:
            raise TypeError
        exact = RetrievalResult.model_validate_json(result.model_dump_json(warnings=False))
        expected = retrieve_candidate_bank(checked)
        if exact != result or canonical_json(exact) != canonical_json(expected):
            raise ValueError
        return exact
    except Exception:
        raise RetrievalError() from None


def retrieval_selector_provenance(
    request: object,
    result: object,
) -> DeterministicSelectorProvenance:
    """Bind a verified deterministic selection without inventing a model call."""

    checked = _validated_request(request)
    exact = validated_retrieval_result(checked, result)
    try:
        return DeterministicSelectorProvenance(
            selector_id=checked.configuration.retrieval_version,
            configuration_digest=checked.configuration.configuration_digest,
            request_digest=checked.request_digest,
            result_digest=exact.result_digest,
        )
    except Exception:
        raise RetrievalError() from None


__all__ = [
    "RETRIEVAL_CONFIG_SCHEMA_VERSION",
    "RETRIEVAL_HIT_SCHEMA_VERSION",
    "RETRIEVAL_QUERY_VERSION",
    "RETRIEVAL_REQUEST_SCHEMA_VERSION",
    "RETRIEVAL_RESULT_SCHEMA_VERSION",
    "RETRIEVAL_VERSION",
    "RetrievalConfig",
    "RetrievalError",
    "RetrievalHit",
    "RetrievalRequest",
    "RetrievalResult",
    "build_retrieval_request",
    "resolve_retrieval_config",
    "retrieval_selector_provenance",
    "retrieve_candidate_bank",
    "validated_retrieval_result",
]
