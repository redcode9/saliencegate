from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from saliencegate.domain import (
    ClaimKind,
    DeliveryTarget,
    EvidenceReference,
    EvidenceSource,
    InterventionAction,
    InterventionClaim,
    InterventionDecision,
    JsonObject,
    MemoryKind,
    MemoryRecord,
    ReasonCode,
    TraceEvent,
    TrustLabel,
    ValidityState,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.domain.records import UUID4, ComponentIdentifier, Sha256Digest, UtcDatetime
from saliencegate.intervention.claims import (
    GROUNDING_RECEIPT_SELECTOR_VERSION,
    GROUNDING_RECEIPT_VERSION,
    DeterministicSelectorProvenance,
    GroundingReceipt,
    InterventionProposal,
    ProposalParseStatus,
    ProposedClaim,
    claim_fingerprint,
    materialize_claim,
)
from saliencegate.intervention.rendering import (
    DeterministicReminderRenderer,
    GroundedClaim,
    RenderingConfig,
    RenderingInputError,
)
from saliencegate.intervention.versions import GROUNDING_PIPELINE_VERSION
from saliencegate.ports.repository import GroundingPin

MAX_STATE_RECORDS = 10_000
MAX_REMINDER_HISTORY = 10_000
MAX_SIGNED_64 = (1 << 63) - 1
_CONFIGURATION_DIGEST_DOMAIN = "saliencegate:grounding:configuration:v1"
_MAX_MODEL_FREE_TEXT_CODE_POINTS = 16_384
_MAX_SOURCE_CODE_POINTS = 1_000_000
_POINTER_INDEX = re.compile(r"^(?:0|[1-9][0-9]*)$")
_DELIVERY_TARGET_ORDER: tuple[DeliveryTarget, ...] = (
    DeliveryTarget.NEXT_MODEL_CALL,
    DeliveryTarget.PRE_ACTION_REPLAN,
)

_CLAIM_MEMORY_KINDS: dict[ClaimKind, MemoryKind] = {
    ClaimKind.REQUIREMENT: MemoryKind.KNOWLEDGE,
    ClaimKind.ENVIRONMENT_FACT: MemoryKind.KNOWLEDGE,
    ClaimKind.FAILED_ATTEMPT: MemoryKind.PROCEDURAL,
    ClaimKind.DIAGNOSIS: MemoryKind.PROCEDURAL,
    ClaimKind.OPEN_SUBGOAL: MemoryKind.PRIVATE_STATUS,
}
_CITATION_REASON_PRECEDENCE = (
    ReasonCode.CITATION_MISSING,
    ReasonCode.CITATION_CROSS_RUN,
    ReasonCode.CITATION_EXPIRED,
    ReasonCode.CITATION_INVALIDATED,
    ReasonCode.INVALID_PROVENANCE,
    ReasonCode.CLAIM_OVER_LIMIT,
)


class GroundingError(ValueError):
    """Base class for value-free grounding failures."""


class GroundingConfigurationError(GroundingError):
    def __init__(self) -> None:
        super().__init__("grounding configuration failed validation")


class GroundingInputError(GroundingError):
    def __init__(self) -> None:
        super().__init__("grounding runtime input failed validation")


class GroundingVerificationError(GroundingError):
    def __init__(self) -> None:
        super().__init__("grounded intervention failed authoritative verification")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
        validate_default=True,
    )


class GroundingConfig(_FrozenModel):
    schema_version: Literal["1.0"]
    pipeline_version: Literal["grounding-pipeline/v1"]
    claim_schema_version: Literal["citation-only-claims/v1"]
    max_claims: Annotated[int, Field(ge=1, le=2)]
    max_evidence_per_claim: Literal[1]
    max_pointer_segments: Annotated[int, Field(ge=1, le=32)]
    max_pointer_utf8_bytes: Annotated[int, Field(ge=1, le=1_024)]
    duplicate_window_events: Annotated[int, Field(ge=0, le=MAX_REMINDER_HISTORY)]
    cooldown_events: Annotated[int, Field(ge=0, le=MAX_REMINDER_HISTORY)]
    ttl_steps: Literal[1]
    allowed_delivery_targets: Annotated[
        tuple[DeliveryTarget, ...], Field(min_length=1, max_length=2)
    ]
    rendering: RenderingConfig

    @field_validator("allowed_delivery_targets")
    @classmethod
    def exact_unique_targets(
        cls,
        value: tuple[DeliveryTarget, ...],
    ) -> tuple[DeliveryTarget, ...]:
        if len(set(value)) != len(value):
            raise ValueError("delivery targets must be an exact unique tuple")
        selected = set(value)
        return tuple(target for target in _DELIVERY_TARGET_ORDER if target in selected)

    @model_validator(mode="after")
    def rendering_accepts_grounding_claim_limit(self) -> Self:
        if self.rendering.max_claims < self.max_claims:
            raise ValueError("renderer claim limit cannot be below grounding claim limit")
        return self


class ResolvedGroundingConfiguration(_FrozenModel):
    schema_version: Literal["1.0"]
    pipeline_version: ComponentIdentifier
    configuration: JsonObject = Field(repr=False)
    configuration_digest: Sha256Digest

    @model_validator(mode="after")
    def digest_matches(self) -> Self:
        expected = _configuration_digest(self.pipeline_version, self.configuration)
        if expected is None or expected != self.configuration_digest:
            raise ValueError("grounding configuration digest does not match")
        return self


class GroundingContext(_FrozenModel):
    schema_version: Literal["1.0", "2.0"]
    intervention_id: UUID4
    run_id: UUID4
    cycle_id: Sha256Digest
    current_event_sequence: Annotated[int, Field(ge=1, le=MAX_SIGNED_64)]
    created_at: UtcDatetime
    requested_delivery_target: DeliveryTarget | None
    model_call_index: Annotated[int, Field(ge=0, le=MAX_SIGNED_64)] | None = Field(
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
    def provenance_matches_schema(self) -> Self:
        model_provenance = self.model_call_index is not None and self.model_call_digest is not None
        selector_provenance = self.selector_provenance is not None
        if (
            (self.model_call_index is None) != (self.model_call_digest is None)
            or model_provenance == selector_provenance
            or (self.schema_version == "1.0" and (not model_provenance or selector_provenance))
            or (self.schema_version == "2.0" and (model_provenance or not selector_provenance))
        ):
            raise ValueError("grounding context provenance does not match its schema")
        return self


class ReminderHistory(_FrozenModel):
    schema_version: Literal["1.0"]
    intervention_id: UUID4
    run_id: UUID4
    event_sequence: Annotated[int, Field(ge=1, le=MAX_SIGNED_64)]
    claim_digests: Annotated[tuple[Sha256Digest, ...], Field(min_length=1, max_length=2)]

    @field_validator("claim_digests")
    @classmethod
    def exact_unique_digests(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("claim digests must be an exact unique tuple")
        return value


class GroundingState(_FrozenModel):
    schema_version: Literal["1.0"]
    events: Annotated[tuple[TraceEvent, ...], Field(max_length=MAX_STATE_RECORDS, repr=False)]
    memories: Annotated[tuple[MemoryRecord, ...], Field(max_length=MAX_STATE_RECORDS, repr=False)]
    reminder_history: Annotated[tuple[ReminderHistory, ...], Field(max_length=MAX_REMINDER_HISTORY)]

    @model_validator(mode="after")
    def indexes_are_unambiguous(self) -> Self:
        event_ids = tuple(item.event_id for item in self.events)
        event_sequences = tuple((item.run_id, item.sequence) for item in self.events)
        memory_ids = tuple(item.memory_id for item in self.memories)
        history_ids = tuple(item.intervention_id for item in self.reminder_history)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("grounding events must have unique IDs")
        if len(set(event_sequences)) != len(event_sequences):
            raise ValueError("grounding events must have unique run sequences")
        if len(set(memory_ids)) != len(memory_ids):
            raise ValueError("grounding memories must have unique IDs")
        if len(set(history_ids)) != len(history_ids):
            raise ValueError("reminder history must have unique intervention IDs")
        return self


@dataclass(frozen=True, slots=True, repr=False)
class _ResolvedClaim:
    claim: InterventionClaim
    text: str
    origin: TrustLabel
    digest: str


@dataclass(frozen=True, slots=True)
class _CitationFailure:
    reason: ReasonCode


def _configuration_digest(pipeline_version: str, configuration: object) -> str | None:
    digest: str | None = None
    try:
        digest = length_prefixed_sha256(
            pipeline_version,
            canonical_json(configuration),
            domain=_CONFIGURATION_DIGEST_DOMAIN,
        )
    except Exception:
        digest = None
    return digest


def resolve_grounding_configuration(configuration: object) -> ResolvedGroundingConfiguration:
    validated: GroundingConfig | None = None
    if type(configuration) is GroundingConfig:
        try:
            candidate = GroundingConfig.model_validate_json(
                configuration.model_dump_json(warnings=False)
            )
            if candidate == configuration:
                validated = candidate
        except Exception:
            validated = None
    if validated is None:
        raise GroundingConfigurationError()
    payload = validated.model_dump(mode="json", warnings=False)
    digest = _configuration_digest(validated.pipeline_version, payload)
    if digest is None:
        raise GroundingConfigurationError()
    resolved: ResolvedGroundingConfiguration | None = None
    try:
        resolved = ResolvedGroundingConfiguration(
            schema_version="1.0",
            pipeline_version=validated.pipeline_version,
            configuration=payload,
            configuration_digest=digest,
        )
    except Exception:
        resolved = None
    if resolved is None:
        raise GroundingConfigurationError()
    return resolved


def _preflight_proposal(value: object) -> ReasonCode | None:
    try:
        if type(value) is not InterventionProposal:
            return ReasonCode.SCHEMA_INVALID
        if type(value.action) is not InterventionAction or type(value.claims) is not tuple:
            return ReasonCode.SCHEMA_INVALID
        if value.action is InterventionAction.REMIND and not value.claims:
            return ReasonCode.NO_GROUNDED_CLAIMS
        if len(value.claims) > 2:
            return ReasonCode.CLAIM_OVER_LIMIT
        if type(value.confidence) is not float:
            return ReasonCode.SCHEMA_INVALID
        if value.model_free_text is not None and (
            type(value.model_free_text) is not str
            or len(value.model_free_text) > _MAX_MODEL_FREE_TEXT_CODE_POINTS
        ):
            return ReasonCode.SCHEMA_INVALID
        if any(type(item) is not ProposedClaim for item in value.claims):
            return ReasonCode.SCHEMA_INVALID
    except Exception:
        return ReasonCode.SCHEMA_INVALID
    return None


def _validated_proposal(value: object) -> InterventionProposal | None:
    if _preflight_proposal(value) is not None:
        return None
    validated: InterventionProposal | None = None
    try:
        assert type(value) is InterventionProposal
        candidate = InterventionProposal.model_validate_json(value.model_dump_json(warnings=False))
        if candidate == value:
            validated = candidate
    except Exception:
        validated = None
    return validated


def _validated_runtime_model(value: object, model_type: type[BaseModel]) -> BaseModel | None:
    validated: BaseModel | None = None
    try:
        if type(value) is model_type:
            candidate = model_type.model_validate_json(value.model_dump_json(warnings=False))
            if candidate == value:
                validated = candidate
    except Exception:
        validated = None
    return validated


def _safe_proposal_confidence(value: object) -> float:
    candidate: object = None
    try:
        if type(value) is InterventionProposal:
            candidate = value.confidence
    except Exception:
        candidate = None
    if type(candidate) is float and math.isfinite(candidate) and 0.0 <= candidate <= 1.0:
        return candidate
    return 1.0


def _grounding_receipt(
    context: GroundingContext,
    *,
    parse_status: ProposalParseStatus,
    proposal_action: InterventionAction | None,
    claims: tuple[ProposedClaim, ...],
    confidence: float,
) -> GroundingReceipt:
    return GroundingReceipt(
        receipt_version=(
            GROUNDING_RECEIPT_VERSION
            if context.schema_version == "1.0"
            else GROUNDING_RECEIPT_SELECTOR_VERSION
        ),
        parse_status=parse_status,
        proposal_action=proposal_action,
        claims=claims,
        confidence=confidence,
        requested_delivery_target=context.requested_delivery_target,
        model_call_index=context.model_call_index,
        model_call_digest=context.model_call_digest,
        selector_provenance=context.selector_provenance,
    )


def _validated_grounding_inputs(
    context: object,
    state: object,
) -> tuple[GroundingContext, GroundingState] | None:
    validated_context = _validated_runtime_model(context, GroundingContext)
    validated_state = _validated_runtime_model(state, GroundingState)
    if (
        type(validated_context) is not GroundingContext
        or type(validated_state) is not GroundingState
    ):
        return None
    current_events = tuple(
        item
        for item in validated_state.events
        if item.run_id == validated_context.run_id
        and item.sequence == validated_context.current_event_sequence
    )
    if (
        len(current_events) != 1
        or current_events[0].timestamp > validated_context.created_at
        or any(
            item.run_id != validated_context.run_id
            or item.event_sequence >= validated_context.current_event_sequence
            for item in validated_state.reminder_history
        )
    ):
        return None
    return validated_context, validated_state


def _grounding_provenance_matches(
    context: GroundingContext,
    receipt: GroundingReceipt,
) -> bool:
    return (
        (context.schema_version == "1.0") == (receipt.receipt_version == GROUNDING_RECEIPT_VERSION)
        and receipt.model_call_index == context.model_call_index
        and receipt.model_call_digest == context.model_call_digest
        and receipt.selector_provenance == context.selector_provenance
    )


def _decode_pointer(pointer: str, *, config: GroundingConfig) -> tuple[str, ...] | None:
    try:
        if len(pointer.encode("utf-8")) > config.max_pointer_utf8_bytes:
            return None
        raw_segments = pointer.split("/")[1:]
        if not raw_segments or len(raw_segments) > config.max_pointer_segments:
            return None
        return tuple(segment.replace("~1", "/").replace("~0", "~") for segment in raw_segments)
    except Exception:
        return None


def _resolve_pointer(root: object, pointer: str, *, config: GroundingConfig) -> str | None:
    segments = _decode_pointer(pointer, config=config)
    if segments is None:
        return None
    value = root
    try:
        for segment in segments:
            if isinstance(value, Mapping):
                if segment not in value:
                    return None
                value = value[segment]
            elif isinstance(value, (list, tuple)):
                if _POINTER_INDEX.fullmatch(segment) is None:
                    return None
                index = int(segment)
                if index >= len(value):
                    return None
                value = value[index]
            else:
                return None
    except Exception:
        return None
    return value if type(value) is str else None


def _select_span(text: str, reference: EvidenceReference) -> str | None:
    if not text or len(text) > _MAX_SOURCE_CODE_POINTS:
        return None
    encoded: bytes | None = None
    try:
        encoded = text.encode("utf-8", errors="strict")
    except Exception:
        encoded = None
    if encoded is None:
        return None
    if reference.span is None:
        return text
    if reference.span.end_byte > len(encoded):
        return None
    selected: str | None = None
    try:
        selected = encoded[reference.span.start_byte : reference.span.end_byte].decode(
            "utf-8", errors="strict"
        )
    except Exception:
        selected = None
    return selected if selected else None


def _resolve_evidence(
    proposed: ProposedClaim,
    *,
    context: GroundingContext,
    state: GroundingState,
    config: GroundingConfig,
) -> _ResolvedClaim | _CitationFailure:
    reference = proposed.evidence
    if reference.source is EvidenceSource.MEMORY:
        record = next(
            (item for item in state.memories if item.memory_id == reference.source_id),
            None,
        )
        if record is None:
            return _CitationFailure(ReasonCode.CITATION_MISSING)
        if record.run_id != context.run_id:
            return _CitationFailure(ReasonCode.CITATION_CROSS_RUN)
        if record.validity is ValidityState.EXPIRED or (
            record.expires_at is not None and record.expires_at <= context.created_at
        ):
            return _CitationFailure(ReasonCode.CITATION_EXPIRED)
        if record.validity in (ValidityState.INVALIDATED, ValidityState.SUPERSEDED):
            return _CitationFailure(ReasonCode.CITATION_INVALIDATED)
        if (
            record.validity is not ValidityState.ACTIVE
            or reference.revision != record.revision
            or reference.field_path != "/content"
            or record.updated_at > context.created_at
            or record.kind is not _CLAIM_MEMORY_KINDS[proposed.kind]
        ):
            return _CitationFailure(ReasonCode.INVALID_PROVENANCE)
        selected = _select_span(record.content, reference)
        origin = record.trust_label
    else:
        event = next(
            (item for item in state.events if item.event_id == reference.source_id),
            None,
        )
        if event is None:
            return _CitationFailure(ReasonCode.CITATION_MISSING)
        if event.run_id != context.run_id:
            return _CitationFailure(ReasonCode.CITATION_CROSS_RUN)
        if (
            event.sequence > context.current_event_sequence
            or event.timestamp > context.created_at
            or not reference.field_path.startswith("/payload/")
        ):
            return _CitationFailure(ReasonCode.INVALID_PROVENANCE)
        selected = _resolve_pointer(
            {"payload": event.payload},
            reference.field_path,
            config=config,
        )
        if selected is not None:
            selected = _select_span(selected, reference)
        origin = event.trust_label
    if selected is None:
        return _CitationFailure(ReasonCode.INVALID_PROVENANCE)
    try:
        if len(selected.encode("utf-8")) > config.rendering.max_evidence_bytes:
            return _CitationFailure(ReasonCode.CLAIM_OVER_LIMIT)
        claim = materialize_claim(proposed, source_text=selected)
        digest = claim_fingerprint(claim)
    except Exception:
        return _CitationFailure(ReasonCode.INVALID_PROVENANCE)
    return _ResolvedClaim(claim=claim, text=selected, origin=origin, digest=digest)


def _cited_ids(
    claims: tuple[InterventionClaim, ...],
    source: EvidenceSource,
) -> tuple[UUID, ...]:
    ordered: dict[UUID, None] = {}
    for claim in claims:
        for evidence in claim.evidence:
            if evidence.source is source:
                ordered[evidence.source_id] = None
    return tuple(ordered)


@dataclass(frozen=True, slots=True, init=False, repr=False)
class GroundingPipeline:
    _configuration_json: str
    _resolved_json: str
    _configuration_digest: str

    def __init__(self, configuration: GroundingConfig) -> None:
        resolved = resolve_grounding_configuration(configuration)
        object.__setattr__(
            self,
            "_configuration_json",
            canonical_json(resolved.configuration).decode("utf-8"),
        )
        object.__setattr__(self, "_resolved_json", resolved.model_dump_json(warnings=False))
        object.__setattr__(self, "_configuration_digest", resolved.configuration_digest)

    @property
    def pipeline_version(self) -> str:
        return GROUNDING_PIPELINE_VERSION

    @property
    def configuration_digest(self) -> str:
        return self._configuration_digest

    @property
    def configuration(self) -> GroundingConfig:
        return GroundingConfig.model_validate_json(self._configuration_json)

    @property
    def resolved_configuration(self) -> ResolvedGroundingConfiguration:
        return ResolvedGroundingConfiguration.model_validate_json(self._resolved_json)

    def pin(
        self,
        requested_delivery_target: DeliveryTarget | None,
    ) -> GroundingPin:
        """Return the complete immutable value object required before a model call."""

        resolved = self.resolved_configuration
        return GroundingPin(
            grounding_version=resolved.pipeline_version,
            grounding_configuration=resolved.configuration,
            grounding_configuration_digest=resolved.configuration_digest,
            requested_delivery_target=requested_delivery_target,
        )

    def _silence(
        self,
        context: GroundingContext,
        reason: ReasonCode,
        *,
        confidence: float,
        receipt: GroundingReceipt,
    ) -> InterventionDecision:
        resolved_payload = self.resolved_configuration.model_dump(mode="json", warnings=False)
        configuration = resolved_payload["configuration"]
        assert isinstance(configuration, dict)
        return InterventionDecision(
            intervention_id=context.intervention_id,
            run_id=context.run_id,
            cycle_id=context.cycle_id,
            grounding_version=self.pipeline_version,
            grounding_configuration=configuration,
            grounding_configuration_digest=self.configuration_digest,
            grounding_receipt=receipt.model_dump(mode="json", warnings=False),
            action=InterventionAction.SILENCE,
            confidence=confidence,
            reason_code=reason,
            ttl_steps=0,
            created_at=context.created_at,
        )

    def ground(
        self,
        proposal: InterventionProposal,
        *,
        context: GroundingContext,
        state: GroundingState,
    ) -> InterventionDecision:
        validated_inputs = _validated_grounding_inputs(context, state)
        if validated_inputs is None:
            raise GroundingInputError()
        validated_context, validated_state = validated_inputs
        config = self.configuration

        preflight_reason = _preflight_proposal(proposal)
        confidence = _safe_proposal_confidence(proposal)
        if preflight_reason in (ReasonCode.NO_GROUNDED_CLAIMS, ReasonCode.CLAIM_OVER_LIMIT):
            parse_status = (
                ProposalParseStatus.EMPTY_REMINDER
                if preflight_reason is ReasonCode.NO_GROUNDED_CLAIMS
                else ProposalParseStatus.CLAIM_OVER_LIMIT
            )
            receipt = _grounding_receipt(
                validated_context,
                parse_status=parse_status,
                proposal_action=InterventionAction.REMIND,
                claims=(),
                confidence=confidence,
            )
            return self._silence(
                validated_context,
                preflight_reason,
                confidence=confidence,
                receipt=receipt,
            )
        validated_proposal = _validated_proposal(proposal)
        if validated_proposal is None:
            receipt = _grounding_receipt(
                validated_context,
                parse_status=ProposalParseStatus.SCHEMA_INVALID,
                proposal_action=None,
                claims=(),
                confidence=1.0,
            )
            return self._silence(
                validated_context,
                ReasonCode.SCHEMA_INVALID,
                confidence=1.0,
                receipt=receipt,
            )
        receipt = _grounding_receipt(
            validated_context,
            parse_status=ProposalParseStatus.VALID,
            proposal_action=validated_proposal.action,
            claims=validated_proposal.claims,
            confidence=validated_proposal.confidence,
        )
        if validated_proposal.action is InterventionAction.SILENCE:
            return self._silence(
                validated_context,
                ReasonCode.SILENCE_SELECTED,
                confidence=validated_proposal.confidence,
                receipt=receipt,
            )
        if (
            validated_context.requested_delivery_target is None
            or validated_context.requested_delivery_target not in config.allowed_delivery_targets
        ):
            return self._silence(
                validated_context,
                ReasonCode.UNSUPPORTED_DELIVERY_TARGET,
                confidence=validated_proposal.confidence,
                receipt=receipt,
            )
        if len(validated_proposal.claims) > config.max_claims:
            return self._silence(
                validated_context,
                ReasonCode.CLAIM_OVER_LIMIT,
                confidence=validated_proposal.confidence,
                receipt=receipt,
            )

        resolved = tuple(
            _resolve_evidence(
                item,
                context=validated_context,
                state=validated_state,
                config=config,
            )
            for item in validated_proposal.claims
        )
        failures = {item.reason for item in resolved if isinstance(item, _CitationFailure)}
        if failures:
            reason = next(item for item in _CITATION_REASON_PRECEDENCE if item in failures)
            return self._silence(
                validated_context,
                reason,
                confidence=validated_proposal.confidence,
                receipt=receipt,
            )
        grounded = tuple(item for item in resolved if isinstance(item, _ResolvedClaim))
        digests = tuple(item.digest for item in grounded)
        if len(set(digests)) != len(digests):
            return self._silence(
                validated_context,
                ReasonCode.DUPLICATE_REMINDER,
                confidence=validated_proposal.confidence,
                receipt=receipt,
            )
        if config.duplicate_window_events > 0 and any(
            validated_context.current_event_sequence - history.event_sequence
            <= config.duplicate_window_events
            and bool(set(digests).intersection(history.claim_digests))
            for history in validated_state.reminder_history
        ):
            return self._silence(
                validated_context,
                ReasonCode.DUPLICATE_REMINDER,
                confidence=validated_proposal.confidence,
                receipt=receipt,
            )
        if config.cooldown_events > 0 and any(
            validated_context.current_event_sequence - history.event_sequence
            <= config.cooldown_events
            for history in validated_state.reminder_history
        ):
            return self._silence(
                validated_context,
                ReasonCode.COOLDOWN_BLOCKED,
                confidence=validated_proposal.confidence,
                receipt=receipt,
            )

        rendered_claims = tuple(
            GroundedClaim(
                claim=item.claim,
                source_text=item.text,
                origin_trust_label=item.origin,
            )
            for item in grounded
        )
        try:
            rendered = DeterministicReminderRenderer(config.rendering).render(rendered_claims)
        except RenderingInputError:
            return self._silence(
                validated_context,
                ReasonCode.CLAIM_OVER_LIMIT,
                confidence=validated_proposal.confidence,
                receipt=receipt,
            )
        claims = tuple(item.claim for item in grounded)
        resolved_payload = self.resolved_configuration.model_dump(mode="json", warnings=False)
        configuration = resolved_payload["configuration"]
        assert isinstance(configuration, dict)
        return InterventionDecision(
            intervention_id=validated_context.intervention_id,
            run_id=validated_context.run_id,
            cycle_id=validated_context.cycle_id,
            grounding_version=self.pipeline_version,
            grounding_configuration=configuration,
            grounding_configuration_digest=self.configuration_digest,
            grounding_receipt=receipt.model_dump(mode="json", warnings=False),
            action=InterventionAction.REMIND,
            delivery_target=validated_context.requested_delivery_target,
            claims=claims,
            rendered_text=rendered,
            cited_memory_ids=_cited_ids(claims, EvidenceSource.MEMORY),
            cited_event_ids=_cited_ids(claims, EvidenceSource.EVENT),
            confidence=validated_proposal.confidence,
            reason_code=ReasonCode.GROUNDED_REMINDER,
            ttl_steps=config.ttl_steps,
            created_at=validated_context.created_at,
        )

    def replay_receipt(
        self,
        receipt: GroundingReceipt,
        *,
        context: GroundingContext,
        state: GroundingState,
    ) -> InterventionDecision:
        """Replay a sanitized parser receipt without rebuilding invalid proposal objects."""

        validated_inputs = _validated_grounding_inputs(context, state)
        validated_receipt = _validated_runtime_model(receipt, GroundingReceipt)
        if validated_inputs is None or type(validated_receipt) is not GroundingReceipt:
            raise GroundingInputError()
        validated_context, validated_state = validated_inputs
        if (
            validated_receipt.requested_delivery_target
            is not validated_context.requested_delivery_target
            or not _grounding_provenance_matches(validated_context, validated_receipt)
        ):
            raise GroundingInputError()
        if validated_receipt.parse_status is ProposalParseStatus.VALID:
            if validated_receipt.proposal_action is None:
                raise GroundingInputError()
            proposal: InterventionProposal | None = None
            try:
                proposal = InterventionProposal(
                    action=validated_receipt.proposal_action,
                    claims=validated_receipt.claims,
                    confidence=validated_receipt.confidence,
                    model_free_text=None,
                )
            except Exception:
                proposal = None
            if proposal is None:
                raise GroundingInputError()
            return self.ground(
                proposal,
                context=validated_context,
                state=validated_state,
            )
        reason = {
            ProposalParseStatus.EMPTY_REMINDER: ReasonCode.NO_GROUNDED_CLAIMS,
            ProposalParseStatus.CLAIM_OVER_LIMIT: ReasonCode.CLAIM_OVER_LIMIT,
            ProposalParseStatus.SCHEMA_INVALID: ReasonCode.SCHEMA_INVALID,
        }[validated_receipt.parse_status]
        return self._silence(
            validated_context,
            reason,
            confidence=validated_receipt.confidence,
            receipt=validated_receipt,
        )


def _config_from_decision(
    decision: InterventionDecision,
) -> tuple[GroundingConfig, ResolvedGroundingConfiguration] | None:
    validated: tuple[GroundingConfig, ResolvedGroundingConfiguration] | None = None
    try:
        encoded = canonical_json(decision.grounding_configuration)
        candidate = GroundingConfig.model_validate_json(encoded)
        resolved = resolve_grounding_configuration(candidate)
        if (
            decision.grounding_version == resolved.pipeline_version
            and decision.grounding_configuration_digest == resolved.configuration_digest
        ):
            validated = candidate, resolved
    except Exception:
        validated = None
    return validated


def _receipt_from_decision(decision: InterventionDecision) -> GroundingReceipt | None:
    receipt: GroundingReceipt | None = None
    try:
        receipt = GroundingReceipt.model_validate_json(canonical_json(decision.grounding_receipt))
    except Exception:
        receipt = None
    return receipt


def verify_grounded_intervention(
    decision: InterventionDecision,
    *,
    context: GroundingContext,
    state: GroundingState,
    expected_configuration: ResolvedGroundingConfiguration,
) -> None:
    validated_decision = _validated_runtime_model(decision, InterventionDecision)
    validated_inputs = _validated_grounding_inputs(context, state)
    if type(validated_decision) is not InterventionDecision or validated_inputs is None:
        raise GroundingVerificationError()
    validated_context, validated_state = validated_inputs
    validated_expected = _validated_runtime_model(
        expected_configuration,
        ResolvedGroundingConfiguration,
    )
    configured = _config_from_decision(validated_decision)
    receipt = _receipt_from_decision(validated_decision)
    if (
        type(validated_expected) is not ResolvedGroundingConfiguration
        or configured is None
        or receipt is None
    ):
        raise GroundingVerificationError()
    config, resolved = configured
    if (
        canonical_json(resolved) != canonical_json(validated_expected)
        or validated_decision.intervention_id != validated_context.intervention_id
        or validated_decision.run_id != validated_context.run_id
        or validated_decision.cycle_id != validated_context.cycle_id
        or validated_decision.created_at != validated_context.created_at
        or receipt.requested_delivery_target is not validated_context.requested_delivery_target
        or not _grounding_provenance_matches(validated_context, receipt)
    ):
        raise GroundingVerificationError()
    expected: InterventionDecision | None = None
    try:
        expected = GroundingPipeline(config).replay_receipt(
            receipt,
            context=validated_context,
            state=validated_state,
        )
    except Exception:
        expected = None
    if expected is None or canonical_json(expected) != canonical_json(validated_decision):
        raise GroundingVerificationError()


__all__ = [
    "GROUNDING_PIPELINE_VERSION",
    "MAX_REMINDER_HISTORY",
    "MAX_STATE_RECORDS",
    "GroundingConfig",
    "GroundingConfigurationError",
    "GroundingContext",
    "GroundingError",
    "GroundingInputError",
    "GroundingPipeline",
    "GroundingState",
    "GroundingVerificationError",
    "ReminderHistory",
    "ResolvedGroundingConfiguration",
    "resolve_grounding_configuration",
    "verify_grounded_intervention",
]
