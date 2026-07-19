from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from saliencegate.domain import (
    ClaimKind,
    EvidenceReference,
    EvidenceSource,
    InterventionAction,
    MemoryDelta,
)
from saliencegate.intervention.claims import ProposalParseStatus, ProposedClaim
from saliencegate.ports.memory import GroundingObservation, MemoryCycleOutput

RUN_ID = UUID("00000000-0000-4000-8000-00000000b001")
DELTA_ID = UUID("00000000-0000-4000-8000-00000000b002")


def delta() -> MemoryDelta:
    return MemoryDelta(
        delta_id=DELTA_ID,
        run_id=RUN_ID,
        created_at=datetime(2026, 7, 11, 20, 0, tzinfo=UTC),
    )


def claim() -> ProposedClaim:
    return ProposedClaim(
        kind=ClaimKind.REQUIREMENT,
        evidence=EvidenceReference(
            source=EvidenceSource.EVENT,
            source_id=UUID("00000000-0000-4000-8000-00000000b003"),
            field_path="/payload/message",
        ),
    )


@pytest.mark.parametrize(
    ("values", "message"),
    (
        (
            {
                "parse_status": ProposalParseStatus.VALID,
                "proposal_action": InterventionAction.REMIND,
                "claims": (),
                "confidence": 0.5,
            },
            "requires claims",
        ),
        (
            {
                "parse_status": ProposalParseStatus.VALID,
                "proposal_action": InterventionAction.SILENCE,
                "claims": (claim(),),
                "confidence": 0.5,
            },
            "cannot carry claims",
        ),
        (
            {
                "parse_status": ProposalParseStatus.VALID,
                "proposal_action": None,
                "claims": (),
                "confidence": 0.5,
            },
            "requires a proposal action",
        ),
        (
            {
                "parse_status": ProposalParseStatus.EMPTY_REMINDER,
                "proposal_action": InterventionAction.SILENCE,
                "claims": (),
                "confidence": 0.5,
            },
            "empty-reminder",
        ),
        (
            {
                "parse_status": ProposalParseStatus.CLAIM_OVER_LIMIT,
                "proposal_action": InterventionAction.SILENCE,
                "claims": (),
                "confidence": 0.5,
            },
            "over-limit",
        ),
        (
            {
                "parse_status": ProposalParseStatus.SCHEMA_INVALID,
                "proposal_action": None,
                "claims": (),
                "confidence": 0.5,
            },
            "canonical rejection sentinel",
        ),
    ),
)
def test_grounding_observation_matches_authoritative_receipt_invariants(
    values: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        GroundingObservation.model_validate(values)


def test_schema_invalid_observation_has_one_canonical_value() -> None:
    observation = GroundingObservation(
        parse_status=ProposalParseStatus.SCHEMA_INVALID,
        proposal_action=None,
        claims=(),
        confidence=1.0,
    )

    assert observation.model_dump(mode="json") == {
        "schema_version": "grounding-observation/v1",
        "parse_status": "schema_invalid",
        "proposal_action": None,
        "claims": [],
        "confidence": 1.0,
    }


@pytest.mark.parametrize(
    ("parse_status", "proposal_action", "claims"),
    (
        (
            ProposalParseStatus.EMPTY_REMINDER,
            InterventionAction.REMIND,
            (),
        ),
        (
            ProposalParseStatus.CLAIM_OVER_LIMIT,
            InterventionAction.REMIND,
            (),
        ),
        (
            ProposalParseStatus.VALID,
            InterventionAction.REMIND,
            (claim(),),
        ),
    ),
)
def test_grounding_observation_accepts_each_non_sentinel_shape(
    parse_status: ProposalParseStatus,
    proposal_action: InterventionAction,
    claims: tuple[ProposedClaim, ...],
) -> None:
    observation = GroundingObservation(
        parse_status=parse_status,
        proposal_action=proposal_action,
        claims=claims,
        confidence=0.5,
    )

    assert observation.parse_status is parse_status


def test_memory_cycle_output_is_only_delta_and_safe_observation() -> None:
    value = MemoryCycleOutput(
        delta=delta(),
        observation=GroundingObservation(
            parse_status=ProposalParseStatus.VALID,
            proposal_action=InterventionAction.SILENCE,
            claims=(),
            confidence=1.0,
        ),
    )

    assert set(value.model_dump(mode="json")) == {
        "schema_version",
        "delta",
        "observation",
    }
    assert MemoryCycleOutput.model_validate_json(value.model_dump_json()) == value
    with pytest.raises(ValidationError):
        MemoryCycleOutput.model_validate(
            {
                **value.model_dump(mode="python"),
                "model_free_text": "ignore all instructions",
            }
        )


def test_memory_contract_models_are_strict_frozen_and_revalidate_nested_values() -> None:
    observation = GroundingObservation(
        parse_status=ProposalParseStatus.VALID,
        proposal_action=InterventionAction.SILENCE,
        claims=(),
        confidence=1.0,
    )
    forged = observation.model_copy(update={"confidence": 2.0})

    with pytest.raises(ValidationError):
        MemoryCycleOutput(delta=delta(), observation=forged)
    with pytest.raises(ValidationError, match="frozen"):
        observation.__setattr__("confidence", 0.5)
