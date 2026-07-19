from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.domain import InterventionAction, MemoryDelta
from saliencegate.intervention.claims import ProposalParseStatus, ProposedClaim
from saliencegate.ports.repository import MemoryHit, MemoryQuery

if TYPE_CHECKING:
    from saliencegate.ports.models import ModelRequest, ModelResult


class _MemoryModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class GroundingObservation(_MemoryModel):
    """The citation-only portion of a grounding receipt produced by a model call."""

    schema_version: Literal["grounding-observation/v1"] = "grounding-observation/v1"
    parse_status: ProposalParseStatus
    proposal_action: InterventionAction | None
    claims: Annotated[tuple[ProposedClaim, ...], Field(max_length=2)]
    confidence: Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]

    @model_validator(mode="after")
    def status_matches_sanitized_proposal(self) -> Self:
        if self.parse_status is ProposalParseStatus.VALID:
            if self.proposal_action is InterventionAction.REMIND and not self.claims:
                raise ValueError("a valid remind observation requires claims")
            if self.proposal_action is InterventionAction.SILENCE and self.claims:
                raise ValueError("a valid silence observation cannot carry claims")
            if self.proposal_action is None:
                raise ValueError("a valid observation requires a proposal action")
            return self
        if self.parse_status is ProposalParseStatus.EMPTY_REMINDER:
            if self.proposal_action is not InterventionAction.REMIND or self.claims:
                raise ValueError("an empty-reminder observation requires an empty remind proposal")
            return self
        if self.parse_status is ProposalParseStatus.CLAIM_OVER_LIMIT:
            if self.proposal_action is not InterventionAction.REMIND or self.claims:
                raise ValueError("an over-limit observation stores no unbounded claims")
            return self
        if self.proposal_action is not None or self.claims or self.confidence != 1.0:
            raise ValueError("a schema-invalid observation uses the canonical rejection sentinel")
        return self


class MemoryCycleOutput(_MemoryModel):
    """The only structured output accepted from the deterministic memory model."""

    schema_version: Literal["memory-cycle-output/v1"] = "memory-cycle-output/v1"
    delta: MemoryDelta
    observation: GroundingObservation


@runtime_checkable
class MemorySource(Protocol):
    async def search(self, query: MemoryQuery) -> tuple[MemoryHit, ...]: ...


@runtime_checkable
class MemoryCycle(Protocol):
    async def generate(self, request: ModelRequest) -> ModelResult: ...


__all__ = [
    "GroundingObservation",
    "MemoryCycle",
    "MemoryCycleOutput",
    "MemorySource",
]
