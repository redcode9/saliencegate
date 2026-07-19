from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Literal, Never, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from saliencegate.domain import (
    ClaimKind,
    DeliveryTarget,
    EvidenceReference,
    InterventionAction,
    InterventionClaim,
    TextSpan,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.domain.records import ComponentIdentifier, Sha256Digest

CLAIM_SCHEMA_VERSION: Literal["citation-only-claims/v1"] = "citation-only-claims/v1"
GROUNDING_RECEIPT_VERSION: Literal["grounding-receipt/v1"] = "grounding-receipt/v1"
DETERMINISTIC_SELECTOR_PROVENANCE_SCHEMA_VERSION: Literal[
    "deterministic-selector-provenance/v1"
] = "deterministic-selector-provenance/v1"
GROUNDING_RECEIPT_SELECTOR_VERSION: Literal["grounding-receipt/v2"] = "grounding-receipt/v2"

_CLAIM_FINGERPRINT_DOMAIN = "saliencegate:intervention:claim-fingerprint:v1"
_SELECTOR_PROVENANCE_DIGEST_DOMAIN = (
    "saliencegate:intervention:deterministic-selector-provenance:v1"
)
_MAX_MODEL_FREE_TEXT_BYTES = 4_096
_FIELD_BY_KIND: dict[ClaimKind, str] = {
    ClaimKind.REQUIREMENT: "requirement",
    ClaimKind.ENVIRONMENT_FACT: "fact",
    ClaimKind.FAILED_ATTEMPT: "attempt",
    ClaimKind.DIAGNOSIS: "diagnosis",
    ClaimKind.OPEN_SUBGOAL: "subgoal",
}


class ClaimInputError(ValueError):
    """A sanitized failure at a claim-processing boundary."""

    def __init__(self) -> None:
        super().__init__("claim input failed validation")


def _raise_claim_input_error() -> Never:
    raise ClaimInputError() from None


def _exact_utf8_text(value: object, *, max_bytes: int | None = None) -> str | None:
    if type(value) is not str:
        return None
    assert isinstance(value, str)
    encoded: bytes | None = None
    with suppress(UnicodeEncodeError):
        encoded = value.encode("utf-8", errors="strict")
    if encoded is None:
        return None
    if max_bytes is not None and len(encoded) > max_bytes:
        return None
    return value


def _validated_span(value: object) -> tuple[bool, TextSpan | None]:
    if value is None:
        return True, None
    if type(value) is not TextSpan:
        return False, None
    assert isinstance(value, TextSpan)
    validated: TextSpan | None = None
    with suppress(Exception):
        validated = TextSpan.model_validate(
            {
                "start_byte": value.start_byte,
                "end_byte": value.end_byte,
            }
        )
    return validated is not None, validated


def _validated_evidence(value: object) -> EvidenceReference | None:
    if type(value) is not EvidenceReference:
        return None
    assert isinstance(value, EvidenceReference)
    validated: EvidenceReference | None = None
    with suppress(Exception):
        span_is_valid, span = _validated_span(value.span)
        if span_is_valid:
            validated = EvidenceReference.model_validate(
                {
                    "source": value.source,
                    "source_id": value.source_id,
                    "revision": value.revision,
                    "field_path": value.field_path,
                    "span": span,
                }
            )
    return validated


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class ProposedClaim(_FrozenModel):
    """A model-selected claim kind and one citation, without model-authored claim text."""

    kind: ClaimKind
    evidence: EvidenceReference

    @property
    def claim_schema_version(self) -> Literal["citation-only-claims/v1"]:
        return CLAIM_SCHEMA_VERSION


class InterventionProposal(_FrozenModel):
    """A bounded structured intervention proposal.

    ``model_free_text`` exists only to ingest paper-style replay fixtures. Automatic
    materialization and rendering never read it.
    """

    action: InterventionAction
    claims: Annotated[tuple[ProposedClaim, ...], Field(max_length=2)]
    confidence: Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
    model_free_text: str | None = Field(repr=False)

    @field_validator("model_free_text", mode="before")
    @classmethod
    def bound_replay_only_text(cls, value: object) -> object:
        if value is None:
            return None
        validated = _exact_utf8_text(value, max_bytes=_MAX_MODEL_FREE_TEXT_BYTES)
        if validated is None:
            raise ValueError("model free text must be bounded UTF-8 text") from None
        return validated

    @model_validator(mode="after")
    def action_matches_claims(self) -> Self:
        if self.action is InterventionAction.REMIND and not self.claims:
            raise ValueError("a remind proposal requires at least one claim")
        if self.action is InterventionAction.SILENCE and self.claims:
            raise ValueError("a silence proposal cannot carry claims")
        return self


class ProposalParseStatus(StrEnum):
    """Sanitized parser outcome retained for authoritative grounding replay."""

    VALID = "valid"
    EMPTY_REMINDER = "empty_reminder"
    CLAIM_OVER_LIMIT = "claim_over_limit"
    SCHEMA_INVALID = "schema_invalid"


def _selector_provenance_digest(values: Mapping[str, object]) -> str:
    material = {key: value for key, value in values.items() if key != "provenance_digest"}
    return length_prefixed_sha256(
        canonical_json(material),
        domain=_SELECTOR_PROVENANCE_DIGEST_DOMAIN,
    )


class DeterministicSelectorProvenance(_FrozenModel):
    """Content address for a non-model selector that produced a proposal."""

    schema_version: Literal["deterministic-selector-provenance/v1"] = (
        DETERMINISTIC_SELECTOR_PROVENANCE_SCHEMA_VERSION
    )
    selector_id: ComponentIdentifier
    configuration_digest: Sha256Digest
    request_digest: Sha256Digest
    result_digest: Sha256Digest
    provenance_digest: Sha256Digest = Field(default_factory=_selector_provenance_digest)

    @model_validator(mode="after")
    def digest_matches(self) -> Self:
        values = self.model_dump(mode="json", exclude={"provenance_digest"}, warnings=False)
        if self.provenance_digest != _selector_provenance_digest(values):
            raise ValueError("deterministic selector provenance digest does not match")
        return self


class GroundingReceipt(_FrozenModel):
    """Citation-only replay receipt that excludes model-authored free text."""

    receipt_version: Literal["grounding-receipt/v1", "grounding-receipt/v2"]
    parse_status: ProposalParseStatus
    proposal_action: InterventionAction | None
    claims: Annotated[tuple[ProposedClaim, ...], Field(max_length=2)]
    confidence: Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
    requested_delivery_target: DeliveryTarget | None
    model_call_index: Annotated[int, Field(ge=0, le=(1 << 63) - 1)] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    model_call_digest: Sha256Digest | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    selector_provenance: DeterministicSelectorProvenance | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def status_matches_sanitized_proposal(self) -> Self:
        model_provenance = self.model_call_index is not None and self.model_call_digest is not None
        selector_provenance = self.selector_provenance is not None
        if (
            (self.model_call_index is None) != (self.model_call_digest is None)
            or model_provenance == selector_provenance
            or (
                self.receipt_version == GROUNDING_RECEIPT_VERSION
                and (not model_provenance or selector_provenance)
            )
            or (
                self.receipt_version == GROUNDING_RECEIPT_SELECTOR_VERSION
                and (model_provenance or not selector_provenance)
            )
        ):
            raise ValueError("grounding receipt provenance does not match its version")
        if self.parse_status is ProposalParseStatus.VALID:
            if self.proposal_action is InterventionAction.REMIND and not self.claims:
                raise ValueError("a valid remind receipt requires claims")
            if self.proposal_action is InterventionAction.SILENCE and self.claims:
                raise ValueError("a valid silence receipt cannot carry claims")
            if self.proposal_action is None:
                raise ValueError("a valid receipt requires a proposal action")
            return self
        if self.parse_status is ProposalParseStatus.EMPTY_REMINDER:
            if self.proposal_action is not InterventionAction.REMIND or self.claims:
                raise ValueError("an empty-reminder receipt requires an empty remind proposal")
            return self
        if self.parse_status is ProposalParseStatus.CLAIM_OVER_LIMIT:
            if self.proposal_action is not InterventionAction.REMIND or self.claims:
                raise ValueError("an over-limit receipt stores no unbounded claims")
            return self
        if self.proposal_action is not None or self.claims or self.confidence != 1.0:
            raise ValueError("a schema-invalid receipt uses the canonical rejection sentinel")
        return self


def _validated_proposed_claim(value: object) -> ProposedClaim | None:
    if type(value) is not ProposedClaim:
        return None
    assert isinstance(value, ProposedClaim)
    validated: ProposedClaim | None = None
    with suppress(Exception):
        kind = value.kind
        evidence = _validated_evidence(value.evidence)
        if type(kind) is ClaimKind and evidence is not None:
            validated = ProposedClaim(kind=kind, evidence=evidence)
    return validated


def _validated_claim_metadata(
    value: object,
) -> tuple[ClaimKind, EvidenceReference] | None:
    if type(value) is not InterventionClaim:
        return None
    assert isinstance(value, InterventionClaim)
    validated: tuple[ClaimKind, EvidenceReference] | None = None
    with suppress(Exception):
        kind = value.kind
        evidence_values = value.evidence
        if type(kind) is ClaimKind and type(evidence_values) is tuple and len(evidence_values) == 1:
            evidence = _validated_evidence(evidence_values[0])
            if evidence is not None:
                validated = kind, evidence
    return validated


def materialize_claim(
    proposed: ProposedClaim,
    *,
    source_text: str,
) -> InterventionClaim:
    """Build a domain claim whose only text is resolved from its cited source."""

    validated = _validated_proposed_claim(proposed)
    selected_text = _exact_utf8_text(source_text)
    if validated is None or selected_text is None:
        _raise_claim_input_error()
    materialized: InterventionClaim | None = None
    with suppress(Exception):
        field_name = _FIELD_BY_KIND[validated.kind]
        materialized = InterventionClaim(
            kind=validated.kind,
            fields={field_name: selected_text},
            evidence=(validated.evidence,),
        )
    if materialized is None:
        _raise_claim_input_error()
    return materialized


def grounded_claim_text(claim: InterventionClaim) -> str:
    """Return the single source-derived field from a materialized allowlisted claim."""

    metadata = _validated_claim_metadata(claim)
    if metadata is None:
        _raise_claim_input_error()
    kind, _evidence = metadata
    fields: object | None = None
    with suppress(Exception):
        fields = claim.fields
    if type(fields) is not MappingProxyType:
        _raise_claim_input_error()
    assert isinstance(fields, MappingProxyType)
    field_name = _FIELD_BY_KIND[kind]
    if len(fields) != 1 or tuple(fields) != (field_name,):
        _raise_claim_input_error()
    selected = _exact_utf8_text(fields[field_name])
    if selected is None:
        _raise_claim_input_error()
    return selected


def claim_fingerprint(claim: InterventionClaim) -> str:
    """Hash only the allowlisted kind and exact citation selector metadata."""

    validated = _validated_claim_metadata(claim)
    if validated is None:
        _raise_claim_input_error()
    kind, evidence = validated
    span = evidence.span
    metadata = {
        "claim_schema_version": CLAIM_SCHEMA_VERSION,
        "kind": kind.value,
        "evidence": {
            "source": evidence.source.value,
            "source_id": str(evidence.source_id),
            "revision": evidence.revision,
            "field_path": evidence.field_path,
            "span": None
            if span is None
            else {
                "start_byte": span.start_byte,
                "end_byte": span.end_byte,
            },
        },
    }
    fingerprint: str | None = None
    with suppress(Exception):
        fingerprint = length_prefixed_sha256(
            canonical_json(metadata),
            domain=_CLAIM_FINGERPRINT_DOMAIN,
        )
    if fingerprint is None:
        _raise_claim_input_error()
    return fingerprint


__all__ = [
    "CLAIM_SCHEMA_VERSION",
    "GROUNDING_RECEIPT_VERSION",
    "ClaimInputError",
    "GroundingReceipt",
    "InterventionProposal",
    "ProposalParseStatus",
    "ProposedClaim",
    "claim_fingerprint",
    "grounded_claim_text",
    "materialize_claim",
]
